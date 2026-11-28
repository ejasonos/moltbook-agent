from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


EXCLUDED_PARTS = {
    ".git", ".config", "data", "logs", "__pycache__", ".pytest_cache",
    ".venv", "venv", "node_modules", "dist", "build", ".mypy_cache",
    ".codex", ".agents",
}
SENSITIVE_NAMES = {
    "credentials.json", "secrets.json", "id_rsa", "id_ed25519",
}
DATABASE_SUFFIXES = (".db", ".db-wal", ".db-shm")


def excluded(relative_path: Path) -> bool:
    parts = [part.casefold() for part in relative_path.parts]
    name = parts[-1] if parts else ""
    return bool(
        any(part in EXCLUDED_PARTS for part in parts)
        or any(part.startswith(".env") for part in parts)
        or name in SENSITIVE_NAMES
        or name.endswith(DATABASE_SUFFIXES)
        or name.endswith((".pyc", ".pyo"))
        or name.endswith(".zip")
    )


def validate_archive(archive_path: Path) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        forbidden = [name for name in members if excluded(Path(name))]
        required = {"requirements.txt", "README.md", "agent/loop.py"}
        missing = required.difference(members)
        if forbidden:
            raise ValueError(f"Archive contains excluded paths: {forbidden}")
        if missing:
            raise ValueError(f"Archive is missing required source files: {sorted(missing)}")
        return members


def build_archive(project_root: str | Path, output_path: str | Path) -> tuple[Path, int]:
    root = Path(project_root).resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    included = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if excluded(relative) or path.resolve() == output:
                continue
            archive.write(path, relative.as_posix())
            included += 1
    validate_archive(output)
    return output, included


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build a source-only Moltbook agent archive.")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "dist" / "moltbook-agent.zip",
        help="Archive path (defaults to dist/moltbook-agent.zip).",
    )
    args = parser.parse_args()
    archive_path, count = build_archive(root, args.output)
    print(f"Archive created: {archive_path}")
    print(f"Included source files: {count}")
    print("Sensitive files and local databases: excluded and validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
