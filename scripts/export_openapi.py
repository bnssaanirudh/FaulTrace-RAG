"""Regenerate the checked-in OpenAPI schema from the live FastAPI application."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "apps" / "api"))

from faulttrace_api.main import app  # noqa: E402


def main() -> int:
    payload = json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"
    for destination in (REPOSITORY_ROOT / "openapi.json", REPOSITORY_ROOT / "docs" / "openapi.json"):
        destination.write_text(payload, encoding="utf-8")
        print(destination.relative_to(REPOSITORY_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
