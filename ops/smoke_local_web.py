"""One synthetic HTTPS Web Todo round trip against local_staging.py serve --web.

Creates and deletes only its own unique Todo in the disposable staging database.
No Bot, model, Google or production database calls.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=56520,
                        help="port identifying the local staging credential directory")
    args = parser.parse_args()
    private = ROOT / ".release-local" / f"public_dev_pg18_{args.port}"
    cert = private / "localhost-cert.pem"
    password_file = private / "web_password.txt"
    if not cert.is_file() or not password_file.is_file():
        raise RuntimeError("Run local_staging.py web-user and serve --web first")
    origin = "https://127.0.0.1:18443"
    with httpx.Client(base_url=origin, verify=str(cert), trust_env=False,
                      timeout=10.0) as client:
        page = client.get("/workbench/")
        page.raise_for_status()
        if "text/html" not in page.headers.get("content-type", ""):
            raise RuntimeError("Workbench HTML was not served")
        health = client.get("/health").json()
        if not health.get("components", {}).get("web_auth"):
            raise RuntimeError("Local Web authentication is not ready")
        bootstrap = client.get("/api/v3/auth/bootstrap",
                               headers={"sec-fetch-site": "same-origin"})
        bootstrap.raise_for_status()
        login = client.post("/api/v3/auth/login",
                            headers={"origin": origin,
                                     "x-csrf-token": bootstrap.json()["csrf_token"]},
                            json={"password": password_file.read_text(encoding="utf-8")})
        if login.status_code != 204:
            raise RuntimeError(f"Synthetic login failed: HTTP {login.status_code}")
        me = client.get("/api/v3/me")
        me.raise_for_status()
        csrf = me.json()["csrf_token"]
        title = f"Isolated smoke {uuid4().hex[:12]}"
        body = {"title": title}
        headers = {"origin": origin, "x-csrf-token": csrf,
                   "idempotency-key": uuid4().hex}
        created = client.post("/api/v3/todos", headers=headers, json=body)
        created.raise_for_status()
        repeated = client.post("/api/v3/todos", headers=headers, json=body)
        repeated.raise_for_status()
        todo_id = created.json()["id"]
        if repeated.json()["id"] != todo_id:
            raise RuntimeError("Identical submission created a second Todo")
        listing = client.get("/api/v3/todos")
        listing.raise_for_status()
        if sum(item["id"] == todo_id for item in listing.json()["items"]) != 1:
            raise RuntimeError("Created Todo missing or duplicated in list")
        delete_headers = {"origin": origin, "x-csrf-token": csrf,
                          "idempotency-key": uuid4().hex,
                          "if-match": created.json()["etag"]}
        deleted = client.delete(f"/api/v3/todos/{todo_id}", headers=delete_headers)
        deleted.raise_for_status()
        if deleted.json() != {"deleted": True, "title": title}:
            raise RuntimeError("Synthetic Todo cleanup was not confirmed")
    print("local_https_web_smoke=passed login=create=deduplicated=list=delete")


if __name__ == "__main__":
    main()