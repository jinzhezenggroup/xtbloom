#!/usr/bin/env python3
"""Verify the dependency-free MKL pthread bridge and its load ordering."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

EXPECTED_EXPORTS = {
    "xtbloom_mkl_pthread_tss_bridge_initialize",
    "pthread_key_create",
    "pthread_key_delete",
    "pthread_getspecific",
    "pthread_setspecific",
    "__pthread_key_create",
    "__pthread_key_delete",
    "__pthread_getspecific",
    "__pthread_setspecific",
}


def _run(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _needed(readelf: str, library: Path) -> list[str]:
    dynamic = _run([readelf, "-dW", str(library)])
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dynamic)


def _exports(nm: str, library: Path) -> set[str]:
    output = _run([nm, "-D", "--defined-only", str(library)])
    symbols: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if fields:
            symbols.add(fields[-1].split("@", 1)[0])
    return symbols


def _require_bridge(bridge: Path, *, readelf: str, nm: str) -> None:
    needed = _needed(readelf, bridge)
    if needed:
        raise SystemExit(
            f"MKL pthread bridge must have zero DT_NEEDED entries: {needed}"
        )
    exports = _exports(nm, bridge)
    if exports != EXPECTED_EXPORTS:
        missing = sorted(EXPECTED_EXPORTS - exports)
        unexpected = sorted(exports - EXPECTED_EXPORTS)
        raise SystemExit(
            "MKL pthread bridge exports differ; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _require_bridge_first(
    library: Path, bridge: Path, *, readelf: str, require_old_libdl_scope: bool
) -> None:
    needed = _needed(readelf, library)
    bridge_soname = bridge.name
    if not needed or needed[0] != bridge_soname:
        raise SystemExit(
            f"{library.name} must list {bridge_soname} first in DT_NEEDED: {needed}"
        )
    if require_old_libdl_scope:
        for required in ("libdl.so.2", "libpthread.so.0"):
            if required not in needed:
                raise SystemExit(
                    f"bootstrap fixture is missing DT_NEEDED {required}: {needed}"
                )


def _require_origin_rpath(library: Path, *, readelf: str) -> None:
    dynamic = _run([readelf, "-dW", str(library)])
    paths = re.findall(
        r"\((?:RPATH|RUNPATH)\)\s+Library (?:rpath|runpath): \[([^\]]+)\]", dynamic
    )
    if not any("$ORIGIN" in path.split(":") for path in paths):
        raise SystemExit(
            f"{library.name} does not search for its private bridge through $ORIGIN"
        )


def main() -> int:
    """Validate bridge exports and dependency ordering for built ELF images."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--nm", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--provider", type=Path)
    args = parser.parse_args()

    _require_bridge(args.bridge, readelf=args.readelf, nm=args.nm)
    if args.fixture is not None:
        _require_bridge_first(
            args.fixture,
            args.bridge,
            readelf=args.readelf,
            require_old_libdl_scope=True,
        )
        _require_origin_rpath(args.fixture, readelf=args.readelf)
    if args.provider is not None:
        _require_bridge_first(
            args.provider,
            args.bridge,
            readelf=args.readelf,
            require_old_libdl_scope=False,
        )
        _require_origin_rpath(args.provider, readelf=args.readelf)

    checked = [args.bridge.name]
    checked.extend(
        path.name for path in (args.fixture, args.provider) if path is not None
    )
    print(f"MKL pthread bridge ELF contract OK: {', '.join(checked)}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
