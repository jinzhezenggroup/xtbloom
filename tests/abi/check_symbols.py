"""Verify that a shared gpuxtb library exports exactly the C ABI allowlist."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

EXPECTED_SYMBOLS = {
    "gpuxtb_batch_init",
    "gpuxtb_batch_result_init",
    "gpuxtb_compute",
    "gpuxtb_compute_options_init",
    "gpuxtb_context_create",
    "gpuxtb_context_destroy",
    "gpuxtb_context_get_backend",
    "gpuxtb_context_get_device_id",
    "gpuxtb_context_options_init",
    "gpuxtb_get_last_error",
    "gpuxtb_plan_compute",
    "gpuxtb_plan_create",
    "gpuxtb_plan_destroy",
    "gpuxtb_plan_query_workspace",
    "gpuxtb_status_string",
    "gpuxtb_version_string",
    "gpuxtb_workspace_query_init",
}


def main() -> int:
    """Verify one shared library against the public symbol allowlist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nm", required=True, help="nm-compatible executable")
    parser.add_argument(
        "--library", required=True, type=Path, help="shared library to inspect"
    )
    args = parser.parse_args()

    completed = subprocess.run(
        [args.nm, "-D", "--defined-only", "--format=posix", str(args.library)],
        check=True,
        capture_output=True,
        text=True,
    )
    exported: set[str] = set()
    unexpected: set[str] = set()
    for line in completed.stdout.splitlines():
        if not line:
            continue
        symbol = line.split(maxsplit=1)[0].split("@", maxsplit=1)[0]
        if symbol.startswith("GPUXTB_"):
            continue  # ELF symbol-version node, not a callable API entry.
        if symbol.startswith("gpuxtb_"):
            exported.add(symbol)
        else:
            unexpected.add(symbol)

    missing = EXPECTED_SYMBOLS - exported
    extra = exported - EXPECTED_SYMBOLS
    if missing or extra or unexpected:
        if missing:
            print(  # noqa: T201 - CLI validation report
                "missing C ABI symbols:", ", ".join(sorted(missing))
            )
        if extra:
            print(  # noqa: T201 - CLI validation report
                "unlisted gpuxtb symbols:", ", ".join(sorted(extra))
            )
        if unexpected:
            print(  # noqa: T201 - CLI validation report
                "unexpected non-C ABI exports:", ", ".join(sorted(unexpected))
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
