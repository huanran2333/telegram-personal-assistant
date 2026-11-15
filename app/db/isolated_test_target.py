"""Fail-closed identity checks for writable PostgreSQL test targets.

This module is deliberately independent of application settings: callers must
pass both configured URLs and the exact isolated cluster directory.  A URL or
an operator-supplied expected path alone is not proof of isolation.
"""

from pathlib import Path

from sqlalchemy.engine import make_url


def validate_isolated_pg18_configuration(
    async_url: str,
    sync_url: str,
    expected_directory: str,
    *,
    repository_root: Path,
) -> None:
    """Reject unsafe configuration without even connecting to PostgreSQL."""
    allowed_root = (repository_root / ".local-tools").resolve()
    expected = Path(expected_directory).resolve()
    if (not expected.is_dir() or not expected.is_relative_to(allowed_root)
            or expected == allowed_root):
        raise ValueError("Isolated PG18 data directory identity mismatch")

    configured = (make_url(async_url), make_url(sync_url))
    if configured[0].drivername != "postgresql+asyncpg" or configured[1].drivername != "postgresql":
        raise ValueError("Isolated PG18 test requires PostgreSQL async/sync URLs")
    for url in configured:
        if (url.host not in {"127.0.0.1", "localhost"}
                or url.port is None or url.port in {5432, 55432, 55632}
                or url.database != "assistant"):
            raise ValueError("Refusing production or historical PostgreSQL endpoint")
    if (configured[0].host != configured[1].host
            or configured[0].port != configured[1].port
            or configured[0].database != configured[1].database
            or configured[0].username != configured[1].username):
        raise ValueError("Async and sync PostgreSQL targets differ")


def validate_isolated_pg18_target(
    async_url: str,
    sync_url: str,
    expected_directory: str,
    actual_directory: str,
    actual_database: str,
    actual_port: str,
    server_version: str,
    *,
    repository_root: Path,
) -> None:
    """Reject production/system/historical clusters before any test DDL/DML."""
    validate_isolated_pg18_configuration(
        async_url, sync_url, expected_directory, repository_root=repository_root)
    configured = make_url(async_url)
    if Path(actual_directory).resolve() != Path(expected_directory).resolve():
        raise ValueError("Connected PostgreSQL data directory is not the isolated target")
    if (actual_database != "assistant" or int(actual_port) != configured.port
            or not str(server_version).startswith("18.")):
        raise ValueError("Connected PostgreSQL instance is not the isolated PG18 target")
