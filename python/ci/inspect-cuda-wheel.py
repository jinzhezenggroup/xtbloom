#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract libxtbloom from wheels and run the repository CUDA ABI inspector."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def inspect_wheel(
    wheel: Path, *, checker: Path, readelf: str, temporary_root: Path
) -> None:
    """Inspect the single ELF libxtbloom payload retained by one Linux wheel."""
    with zipfile.ZipFile(wheel) as archive:
        candidates = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and PurePosixPath(info.filename).name.startswith("libxtbloom.so")
            and archive.read(info)[:4] == b"\x7fELF"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"{wheel} must contain exactly one ELF libxtbloom.so payload; "
                f"found {len(candidates)}"
            )
        payload = archive.read(candidates[0])

    extracted = temporary_root / wheel.stem / "libxtbloom.so"
    extracted.parent.mkdir(parents=True)
    extracted.write_bytes(payload)
    subprocess.run(
        [
            sys.executable,
            str(checker),
            "--readelf",
            readelf,
            "--library",
            str(extracted),
        ],
        check=True,
    )


def main() -> int:
    """Inspect each requested wheel with the configured CUDA ABI checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", required=True, type=Path)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    if not args.checker.is_file():
        raise SystemExit(f"CUDA ABI checker does not exist: {args.checker}")
    with tempfile.TemporaryDirectory(prefix="xtbloom-wheel-inspect-") as directory:
        root = Path(directory)
        for wheel in args.wheels:
            inspect_wheel(
                wheel,
                checker=args.checker,
                readelf=args.readelf,
                temporary_root=root,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
