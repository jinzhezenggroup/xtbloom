"""Verify that a shared xtbloom library exports exactly the C ABI allowlist."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

EXPECTED_SYMBOLS = {
    "xtbloom_batch_init",
    "xtbloom_batch_result_init",
    "xtbloom_compute",
    "xtbloom_compute_enqueue",
    "xtbloom_compute_options_init",
    # Source builds deliberately export the opt-in external-energy extension
    # even though its separate research header is excluded from installations.
    # Listing the symbols here prevents accidental export drift without
    # representing them as members of the installed stable-header contract.
    "xtbloom_context_copy_external_energy_device_gradients",
    "xtbloom_context_create",
    "xtbloom_context_destroy",
    "xtbloom_context_get_backend",
    "xtbloom_context_get_device_id",
    "xtbloom_context_options_init",
    "xtbloom_context_set_external_energy_callback",
    "xtbloom_context_set_external_energy_device_model",
    "xtbloom_get_last_error",
    "xtbloom_plan_compute",
    "xtbloom_plan_compute_enqueue",
    "xtbloom_plan_create",
    "xtbloom_plan_destroy",
    "xtbloom_plan_query_workspace",
    "xtbloom_result_owner_buffer",
    "xtbloom_result_owner_create",
    "xtbloom_result_owner_export_dltensor",
    "xtbloom_result_owner_options_init",
    "xtbloom_result_owner_release",
    "xtbloom_result_owner_retain",
    "xtbloom_request_create",
    "xtbloom_request_destroy",
    "xtbloom_request_get_error",
    "xtbloom_request_info_init",
    "xtbloom_request_query",
    "xtbloom_request_wait",
    "xtbloom_status_string",
    "xtbloom_version_string",
    "xtbloom_workspace_query_init",
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
        if symbol.startswith("XTBLOOM_"):
            continue  # ELF symbol-version node, not a callable API entry.
        if symbol.startswith("xtbloom_"):
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
                "unlisted xtbloom symbols:", ", ".join(sorted(extra))
            )
        if unexpected:
            print(  # noqa: T201 - CLI validation report
                "unexpected non-C ABI exports:", ", ".join(sorted(unexpected))
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
