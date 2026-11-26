from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from integrations.moltbook.client import (
    MoltbookAPIError,
    MoltbookClient,
)
from integrations.moltbook.posts import create_post


ROOT = Path(__file__).resolve().parent
CREDENTIALS_FILE = ROOT / ".config" / "moltbook" / "credentials.json"


def _load_api_key() -> str:
    api_key = os.getenv("MOLTBOOK_API_KEY")
    if api_key:
        return api_key

    if CREDENTIALS_FILE.is_file():
        try:
            credentials: Any = json.loads(
                CREDENTIALS_FILE.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read {CREDENTIALS_FILE}."
            ) from exc

        if isinstance(credentials, dict):
            api_key = credentials.get("api_key")
            if isinstance(api_key, str) and api_key.strip():
                return api_key.strip()

    raise RuntimeError(
        "MOLTBOOK_API_KEY is missing, and no api_key was found in "
        f"{CREDENTIALS_FILE}."
    )


def main() -> int:
    try:
        client = MoltbookClient(api_key=_load_api_key())
        submolt = input("Submolt [general]: ").strip() or "general"
        title = input("Post title: ").strip()
        content = input("Post content: ").strip()

        if not title or not content:
            raise ValueError("Post title and content cannot be empty.")

        result = create_post(client, submolt, title, content)

        verification = result.get("verification")
        if verification is None:
            post = result.get("post")
            if isinstance(post, dict):
                verification = post.get("verification")

        if verification:
            if not isinstance(verification, dict):
                raise RuntimeError(
                    "Moltbook returned an invalid verification challenge."
                )

            verification_code = verification.get("verification_code")
            challenge_text = verification.get("challenge_text")
            if not isinstance(verification_code, str) or not verification_code:
                raise RuntimeError(
                    "Moltbook's verification response did not include a code."
                )

            print("Moltbook requires verification before the post is visible.")
            if challenge_text:
                print(f"Challenge: {challenge_text}")
            answer = input("Enter the numeric answer (for example, 41.00): ").strip()
            if not answer:
                raise ValueError("Verification answer cannot be empty.")

            result = client.post(
                "/verify",
                {
                    "verification_code": verification_code,
                    "answer": answer,
                },
            )
    except MoltbookAPIError as exc:
        print(f"Post failed: {exc}", file=sys.stderr)
        if exc.status_code is not None:
            print(f"HTTP status: {exc.status_code}", file=sys.stderr)
        if exc.response_data is not None:
            print(
                json.dumps(exc.response_data, indent=2, ensure_ascii=False),
                file=sys.stderr,
            )
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Post failed: {exc}", file=sys.stderr)
        return 1

    print("Moltbook response:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())