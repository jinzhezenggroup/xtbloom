"""Fail wheel CI before upload when an artifact exceeds PyPI's size limit."""

from __future__ import annotations

import sys
from pathlib import Path

PYPI_FILE_LIMIT_BYTES = 100_000_000


def main() -> int:
    """Report wheel sizes and fail when any exceeds PyPI's file limit."""
    wheel_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("wheelhouse")
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"no wheels found in {wheel_dir}")
    oversized = []
    for wheel in wheels:
        size = wheel.stat().st_size
        print(  # noqa: T201 - CLI validation report
            f"{wheel.name}: {size / 1_000_000:.1f} MB"
        )
        if size > PYPI_FILE_LIMIT_BYTES:
            oversized.append((wheel, size))
    if oversized:
        details = ", ".join(
            f"{wheel.name} ({size / 1_000_000:.1f} MB)" for wheel, size in oversized
        )
        raise SystemExit(f"wheel exceeds PyPI's 100 MB per-file limit: {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
