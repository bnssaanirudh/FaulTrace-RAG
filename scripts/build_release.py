"""Build a clean source archive from allowlisted, non-generated files."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DENIED_PARTS = {
    ".git",
    ".next",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "artifacts",
    "outputs",
    "reports",
    "dist",
    "test-results",
    "playwright-report",
}
DENIED_SUFFIXES = {".db", ".pyc", ".pyo", ".parquet", ".zip", ".tsbuildinfo"}
ALLOWED_ROOT_DIRECTORIES = {
    ".github",
    "apps",
    "configs",
    "docker",
    "docs",
    "paper",
    "packages",
    "scripts",
}
ALLOWED_ROOT_FILES = {
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "alembic.ini",
    "CITATION.cff",
    "docker-compose.yml",
    "INSTALLATION.md",
    "LICENSE",
    "Makefile",
    "openapi.json",
    "pyproject.toml",
    "README.md",
    "REPRODUCIBILITY.md",
    "RELEASE_NOTES.md",
    "USER_GUIDE.md",
}


def source_files() -> list[Path]:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT
    )
    files: set[Path] = set()
    for raw in (tracked + untracked).split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        posix = PurePosixPath(relative.as_posix())
        if not (
            (len(posix.parts) == 1 and posix.as_posix() in ALLOWED_ROOT_FILES)
            or (len(posix.parts) > 1 and posix.parts[0] in ALLOWED_ROOT_DIRECTORIES)
        ):
            continue
        if any(part in DENIED_PARTS for part in posix.parts):
            continue
        if relative.suffix.lower() in DENIED_SUFFIXES or posix.parts[0] == "data":
            continue
        path = ROOT / relative
        if path.is_file():
            files.add(relative)
    return sorted(files)


def build() -> Path:
    version = "0.1.0"
    destination_dir = ROOT / "dist"
    destination_dir.mkdir(parents=True, exist_ok=True)
    archive_path = destination_dir / f"faulttrace-rag-{version}-source.zip"
    manifest = {"version": version, "created_at": datetime.now(UTC).isoformat(), "files": {}}

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in source_files():
            payload = (ROOT / relative).read_bytes()
            archive.writestr(relative.as_posix(), payload)
            manifest["files"][relative.as_posix()] = hashlib.sha256(payload).hexdigest()
        archive.writestr("release-manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return archive_path


if __name__ == "__main__":
    print(build())
