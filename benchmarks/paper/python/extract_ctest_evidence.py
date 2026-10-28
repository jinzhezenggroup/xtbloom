#!/usr/bin/env python3
"""Extract an immutable focused test ledger from a sealed P0-A JUnit run.

Downstream evidence stages must never rerun CTest inside the already-checksummed
P0-A build tree because CTest mutates ``Testing/Temporary``.  This script proves
that the requested tests were registered and passed in that canonical run.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--regex", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    pattern = re.compile(args.regex, re.IGNORECASE)
    root = ET.parse(args.junit).getroot()
    rows = []
    for testcase in root.iter("testcase"):
        name = testcase.attrib.get("name", "")
        classname = testcase.attrib.get("classname", "")
        coordinate = f"{classname}/{name}" if classname else name
        if not pattern.search(coordinate):
            continue
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped = testcase.find("skipped")
        status = (
            "failed"
            if failure is not None or error is not None
            else "skipped"
            if skipped is not None
            else "passed"
        )
        rows.append(
            {
                "name": name,
                "classname": classname,
                "time_seconds": float(testcase.attrib.get("time", 0.0)),
                "status": status,
            }
        )
    failures = []
    if not rows:
        failures.append("no canonical CTest coordinate matched the requested regex")
    failures.extend(row["name"] for row in rows if row["status"] != "passed")
    document = {
        "schema_version": 1,
        "source_junit": str(args.junit.resolve()),
        "regex": args.regex,
        "tests": rows,
        "summary": {
            "matched": len(rows),
            "passed": sum(row["status"] == "passed" for row in rows),
            "failed": len(failures),
        },
        "formal_gate": {"status": "fail" if failures else "pass", "failures": failures},
        "provenance_note": "read-only extraction from the sealed canonical P0-A CTest run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
