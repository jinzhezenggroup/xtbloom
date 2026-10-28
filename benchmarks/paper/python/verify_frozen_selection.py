#!/usr/bin/env python3
"""Revalidate every frozen bundle and selected-manifest identity before a job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manifest_freeze import find_bundle_root, sha256, verify_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--qm9-main-manifest", type=Path, required=True)
    parser.add_argument("--qm9-performance-manifest", type=Path, required=True)
    parser.add_argument("--omol25-manifest", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.selection.read_text(encoding="utf-8"))
    paths = {
        "qm9_main": args.qm9_main_manifest,
        "qm9_performance": args.qm9_performance_manifest,
        "omol25": args.omol25_manifest,
    }
    for name, path in paths.items():
        frozen = document["manifests"][name]
        if str(path.resolve()) != frozen["path"] or sha256(path) != frozen["sha256"]:
            raise RuntimeError(f"selected manifest drifted after freeze: {name}")
        current_bundle = verify_bundle(find_bundle_root(path))
        frozen_bundle = document["bundles"][name]
        if current_bundle["root"] != frozen_bundle["root"]:
            raise RuntimeError(f"bundle root drifted after freeze: {name}")
        if current_bundle["sha256sums_sha256"] != frozen_bundle["sha256sums_sha256"]:
            raise RuntimeError(f"bundle checksum manifest drifted after freeze: {name}")
        if current_bundle["listed_files"] != frozen_bundle["listed_files"]:
            raise RuntimeError(f"bundle contents drifted after freeze: {name}")
    print(f"frozen_selection=PASS selection={args.selection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
