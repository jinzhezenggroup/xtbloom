#!/usr/bin/env python3
"""Emit a canonical identity for the Python runtime used by every stage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_origin(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return (
        str(Path(spec.origin).resolve()) if spec is not None and spec.origin else None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dxtb-source", type=Path)
    parser.add_argument("--require", action="append", default=[])
    args = parser.parse_args()
    if args.dxtb_source is not None:
        sys.path.insert(0, str(args.dxtb_source.resolve() / "src"))
    executable = Path(sys.executable).resolve()
    modules = {name: module_origin(name) for name in ("numpy", "torch", "dxtb")}
    missing = sorted(name for name in args.require if modules.get(name) is None)
    if missing:
        raise RuntimeError(f"required Python modules are unavailable: {missing}")
    distributions = sorted(
        {
            (
                distribution.metadata.get("Name", "").lower(),
                distribution.version,
                str(Path(distribution.locate_file("")).resolve()),
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )
    document = {
        "schema_version": 1,
        "python": {
            "executable": str(executable),
            "executable_sha256": sha256(executable),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "modules": modules,
        "distributions": [
            {"name": name, "version": version, "location": location}
            for name, version, location in distributions
        ],
    }
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
