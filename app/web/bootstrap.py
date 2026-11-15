"""Local operator-only credential binding; never exposed as an HTTP endpoint."""

import argparse
import asyncio
import getpass

from app.db.session import AsyncSessionLocal, init_db
from app.web.auth import bind_existing_user


async def main() -> None:
    parser = argparse.ArgumentParser(description="Bind Web login to an existing User.id")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--reset-password", action="store_true")
    args = parser.parse_args()
    first = getpass.getpass("Web password: ")
    second = getpass.getpass("Confirm password: ")
    if first != second:
        raise SystemExit("Passwords do not match")
    await init_db()
    async with AsyncSessionLocal() as db:
        await bind_existing_user(db, args.user_id, first, reset=args.reset_password)
    print("Web credential bound to existing user; active sessions revoked on reset.")


if __name__ == "__main__":
    asyncio.run(main())
