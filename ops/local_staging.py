"""Windows-only disposable staging PostgreSQL and loopback app launcher.

Never uses the production database, Bot, Google credentials or sender. All data
and credentials stay under ignored directories in this repository copy.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
from uuid import uuid4

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_ROOT = (ROOT / ".local-tools").resolve()
PRIVATE_ROOT = (ROOT / ".release-local").resolve()
ROLE = "pa_stage_app"
DATABASE = "personal_assistant_stage"
RESERVED_PORTS = {5432, 55432, 55632}


def locations(port: int) -> tuple[Path, Path]:
    if not 1024 < port < 65536 or port in RESERVED_PORTS:
        raise RuntimeError("Choose a non-reserved high staging port")
    data = (DATA_ROOT / f"public_dev_pg18_{port}").resolve()
    private = (PRIVATE_ROOT / f"public_dev_pg18_{port}").resolve()
    if not data.is_relative_to(DATA_ROOT) or not private.is_relative_to(PRIVATE_ROOT):
        raise RuntimeError("Staging path escaped this repository")
    return data, private


def pg_binary(pg_bin: Path, name: str) -> Path:
    target = (pg_bin / (name + ".exe")).resolve()
    if not target.is_file():
        raise RuntimeError(f"PostgreSQL binary missing: {name}")
    return target


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def run(command: list[str], **kwargs) -> None:
    # A started PostgreSQL server can inherit capture pipes and keep them open.
    result = subprocess.run(command, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, **kwargs)
    if result.returncode:
        # PostgreSQL command output may contain local paths. Never print DSNs.
        raise RuntimeError(f"{Path(command[0]).name} failed with exit code {result.returncode}")


def protect(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    who = subprocess.check_output(["whoami"], text=True).strip()
    run(["icacls", str(directory), "/inheritance:r", "/grant:r", f"{who}:(OI)(CI)F"])


def read_private(port: int) -> tuple[Path, Path, Path, str, str]:
    data, private = locations(port)
    config = json.loads((private / "config.json").read_text(encoding="utf-8"))
    pg_bin = Path(config["pg_bin"]).resolve()
    if config["port"] != port or Path(config["data"]).resolve() != data or not (data / "PG_VERSION").is_file():
        raise RuntimeError("Staging configuration or data directory mismatch")
    admin = (private / "admin_password.txt").read_text(encoding="utf-8")
    app = (private / "app_password.txt").read_text(encoding="utf-8")
    return data, private, pg_bin, admin, app


def verify_server(data: Path, port: int, admin: str) -> None:
    with psycopg.connect(host="127.0.0.1", port=port, user="postgres",
                         password=admin, dbname="postgres", connect_timeout=3) as conn:
        directory, got_port, version, listen = conn.execute(
            "SELECT current_setting('data_directory'), current_setting('port'), "
            "current_setting('server_version'), current_setting('listen_addresses')").fetchone()
    if (Path(directory).resolve() != data or got_port != str(port)
            or not version.startswith("18.") or listen != "127.0.0.1"):
        raise RuntimeError("Connected server is not this loopback PostgreSQL 18 staging cluster")


def start(port: int) -> None:
    data, private, pg_bin, admin, _ = read_private(port)
    if port_open(port):
        verify_server(data, port, admin)
        print("isolated_postgres=already_running")
        return
    run([str(pg_binary(pg_bin, "pg_ctl")), "-D", str(data), "-l", str(private / "postgres.log"),
         "-o", f"-c listen_addresses=127.0.0.1 -p {port}", "-w", "start"])
    verify_server(data, port, admin)
    print("isolated_postgres=started")


def initialize(port: int, pg_bin: Path) -> None:
    data, private = locations(port)
    if data.exists() or private.exists() or port_open(port):
        raise RuntimeError("Staging path or port already exists; refusing overwrite")
    pg_bin = pg_bin.resolve()
    initdb = pg_binary(pg_bin, "initdb")
    pg_binary(pg_bin, "pg_ctl")
    version = subprocess.check_output([str(initdb), "--version"], text=True).strip()
    if "PostgreSQL) 18." not in version:
        raise RuntimeError("PostgreSQL 18 is required")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    protect(private)
    admin = secrets.token_hex(24)
    app = secrets.token_hex(24)
    (private / "admin_password.txt").write_text(admin, encoding="utf-8")
    (private / "app_password.txt").write_text(app, encoding="utf-8")
    (private / "config.json").write_text(json.dumps({"port": port, "data": str(data),
                                                     "pg_bin": str(pg_bin)}), encoding="utf-8")
    run([str(initdb), "-D", str(data), "-U", "postgres", "--pwfile",
         str(private / "admin_password.txt"), "--auth-host", "scram-sha-256",
         "--auth-local", "scram-sha-256", "--encoding", "UTF8", "--locale", "C"])
    start(port)
    with psycopg.connect(host="127.0.0.1", port=port, user="postgres",
                         password=admin, dbname="postgres", autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DATABASE,)).fetchone():
            raise RuntimeError("Database already exists in isolated cluster")
        conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
            sql.Identifier(ROLE), sql.Literal(app)))
        conn.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(
            sql.Identifier(DATABASE), sql.Identifier(ROLE)))
    env = safe_environment(port, admin, app)
    env.update({"LLM_PROVIDER": "openai", "LLM_MODEL": "placeholder-model",
                "OPENAI_BASE_URL": "http://127.0.0.1:65433/v1",
                "OPENAI_API_KEY": "placeholder-not-a-secret"})
    env["PPA_OFFLINE_ADMIN_DATABASE_URL"] = (
        f"postgresql+asyncpg://postgres:{admin}@127.0.0.1:{port}/{DATABASE}")
    run([sys.executable, "-m", "app.db.offline_initialize", "--mode", "staging",
         "--expected-data-directory", str(data)], cwd=ROOT, env=env, timeout=180)
    print(f"isolated_staging_ready=True port={port} pg=18")


def safe_environment(port: int, _admin: str, app: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "DATABASE_URL": f"postgresql+asyncpg://{ROLE}:{app}@127.0.0.1:{port}/{DATABASE}",
        "DATABASE_URL_SYNC": f"postgresql://{ROLE}:{app}@127.0.0.1:{port}/{DATABASE}",
        "TELEGRAM_BOT_TOKEN": "", "TELEGRAM_WEBHOOK_URL": "",
        "TELEGRAM_LIVE_START_PREFLIGHT_ACK": "false", "GOOGLE_CALENDAR_ENABLED": "false",
        "GMAIL_ENABLED": "false", "SCHEDULER_ENABLED": "false",
        "DAILY_RUNDOWN_ENABLED": "false", "APP_ENV": "development",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def create_web_user(port: int) -> None:
    data, private, _, admin, app = read_private(port)
    verify_server(data, port, admin)
    if (private / "web_password.txt").exists():
        raise RuntimeError("Local Web user already provisioned; refusing password overwrite")
    env = safe_environment(port, admin, app)
    env.update({"LLM_PROVIDER": "openai", "LLM_MODEL": "placeholder-model",
                "OPENAI_BASE_URL": "http://127.0.0.1:65433/v1",
                "OPENAI_API_KEY": "placeholder-not-a-secret"})
    os.environ.update(env)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def create() -> str:
        from app.db.models import User
        from app.db.session import AsyncSessionLocal
        from app.web.auth import bind_existing_user
        user_id = str(uuid4())
        async with AsyncSessionLocal() as session:
            session.add(User(id=user_id, telegram_id=f"isolated-dev-{uuid4().hex}",
                             first_name="Local Demo", calendar_time_block_sync=False))
            await session.commit()
            await bind_existing_user(session, user_id, password)
        return user_id

    password = secrets.token_urlsafe(24)
    user_id = asyncio.run(create())
    (private / "web_user_id.txt").write_text(user_id, encoding="utf-8")
    (private / "web_password.txt").write_text(password, encoding="utf-8")
    (private / "web_secret.txt").write_text(secrets.token_hex(32), encoding="utf-8")
    make_local_certificate(private)
    print(f"isolated_web_user_ready=True password_file={private / 'web_password.txt'}")


def make_local_certificate(private: Path) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=30))
            .add_extension(x509.SubjectAlternativeName([
                x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))
            ]), critical=False).sign(key, hashes.SHA256()))
    (private / "localhost-cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (private / "localhost-key.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))


def serve(port: int, web: bool) -> None:
    data, private, _, admin, app = read_private(port)
    verify_server(data, port, admin)
    env = safe_environment(port, admin, app)
    if web:
        for name in ("web_password.txt", "web_secret.txt", "localhost-cert.pem", "localhost-key.pem"):
            if not (private / name).is_file():
                raise RuntimeError(f"Local Web setup missing: {name}")
        if not (ROOT / "web" / "dist" / "index.html").is_file():
            raise RuntimeError("Build the Web workbench first: npm ci; npm run build")
        env["WEB_ENABLED"] = "true"
        env["WEB_PUBLIC_ORIGIN"] = "https://127.0.0.1:18443"
        env["SECRET_KEY"] = (private / "web_secret.txt").read_text(encoding="utf-8")
    else:
        env["WEB_ENABLED"] = "false"
    os.environ.update(env)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import uvicorn
    if web:
        print("Local self-signed HTTPS: https://127.0.0.1:18443/workbench")
        uvicorn.run("app.main:app", host="127.0.0.1", port=18443, loop="none",
                    proxy_headers=False, ssl_certfile=str(private / "localhost-cert.pem"),
                    ssl_keyfile=str(private / "localhost-key.pem"))
    else:
        print("Local API: http://127.0.0.1:8000/ ; full /health remains 503 without Telegram")
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000,
                    loop="none", proxy_headers=False)


def stop(port: int) -> None:
    data, _, pg_bin, admin, _ = read_private(port)
    if not port_open(port):
        print("isolated_postgres=already_stopped")
        return
    verify_server(data, port, admin)
    run([str(pg_binary(pg_bin, "pg_ctl")), "-D", str(data), "-m", "fast", "-w", "stop"])
    print("isolated_postgres=stopped")


def main() -> None:
    if sys.platform != "win32":
        raise RuntimeError("This isolated launch recipe has only been verified on Windows")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "start", "web-user", "serve", "stop"))
    parser.add_argument("--port", type=int, default=56520, help="isolated PostgreSQL loopback port")
    parser.add_argument("--pg-bin", type=Path, default=Path(r"C:\Program Files\PostgreSQL\18\bin"))
    parser.add_argument("--web", action="store_true", help="serve self-signed loopback HTTPS Web demo")
    args = parser.parse_args()
    if args.action == "init": initialize(args.port, args.pg_bin)
    elif args.action == "start": start(args.port)
    elif args.action == "web-user": create_web_user(args.port)
    elif args.action == "serve": serve(args.port, args.web)
    elif args.action == "stop": stop(args.port)


if __name__ == "__main__":
    main()