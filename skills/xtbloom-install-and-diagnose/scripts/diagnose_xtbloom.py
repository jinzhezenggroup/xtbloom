#!/usr/bin/env python3
"""Report xTBloom package, native-library, and backend availability as JSON.

The script performs no installation, filesystem writes, environment changes, or
scientific calculation.  With ``--backend`` it transiently creates and destroys
one native context so the requested runtime boundary is tested directly.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def _exception(error: BaseException) -> dict[str, Any]:
    """Return a stable, JSON-serializable diagnostic for one exception."""
    result: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    status = getattr(error, "status", None)
    if status is not None:
        result["status"] = int(status)
    return result


def _distribution_report() -> dict[str, Any]:
    """Inspect installed metadata without importing xTBloom."""
    try:
        distribution = importlib.metadata.distribution("xtbloom")
    except importlib.metadata.PackageNotFoundError as error:
        return {"installed": False, "error": _exception(error)}
    return {
        "installed": True,
        "version": distribution.version,
        "location": str(Path(str(distribution.locate_file(""))).resolve()),
    }


def _override_report() -> dict[str, Any]:
    """Describe only the documented library override, not the full environment."""
    value = os.environ.get("XTBLOOM_LIBRARY")
    if value is None:
        return {"set": False}
    path = Path(value).expanduser()
    return {
        "set": True,
        "value": value,
        "is_file": path.is_file(),
    }


def _probe(backend: str) -> tuple[dict[str, Any], bool]:
    """Import xTBloom and optionally create the requested native context."""
    package_report: dict[str, Any]
    native_report: dict[str, Any] = {"candidate_found": False, "loaded": False}
    backend_report: dict[str, Any] = {"requested": backend, "probed": False}

    try:
        xtbloom = importlib.import_module("xtbloom")
    # A diagnostic must preserve package-defined and platform-loader failures
    # without depending on xTBloom's exception classes before import succeeds.
    except Exception as error:  # noqa: BLE001
        package_report = {"imported": False, "error": _exception(error)}
        return {
            "package": package_report,
            "native_library": native_report,
            "backend": backend_report,
        }, False

    package_report = {
        "imported": True,
        "version": getattr(xtbloom, "__version__", None),
        "path": str(Path(xtbloom.__file__).resolve()),
    }
    library = importlib.import_module("xtbloom.library")
    try:
        candidate = library.library_path()
    except Exception as error:  # noqa: BLE001 - report package-specific errors
        native_report["error"] = _exception(error)
        return {
            "package": package_report,
            "native_library": native_report,
            "backend": backend_report,
        }, False

    native_report.update({"candidate_found": True, "candidate": str(candidate)})
    if backend == "none":
        return {
            "package": package_report,
            "native_library": native_report,
            "backend": backend_report,
        }, True

    try:
        native_report["version"] = library.get_version()
        native_report["loaded"] = True
        context_type = xtbloom.Context
        with context_type(backend=backend) as context:
            resolved_value = int(context.backend)
            resolved_names = {
                int(library.BACKEND_CPU): "cpu",
                int(library.BACKEND_CUDA): "cuda",
            }
            backend_report.update(
                {
                    "probed": True,
                    "success": True,
                    "resolved": resolved_names.get(resolved_value, str(resolved_value)),
                    "device_id": int(context.device_id),
                }
            )
    # Context creation crosses Python, the native ABI, and provider loaders;
    # report any normal exception while still allowing process-control signals.
    except Exception as error:  # noqa: BLE001
        backend_report.update(
            {"probed": True, "success": False, "error": _exception(error)}
        )
        return {
            "package": package_report,
            "native_library": native_report,
            "backend": backend_report,
        }, False

    return {
        "package": package_report,
        "native_library": native_report,
        "backend": backend_report,
    }, True


def main() -> int:
    """Run the requested read-only diagnostic and return a CI-friendly status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("none", "auto", "cpu", "cuda"),
        default="none",
        help="create one transient context for this backend (default: locate only)",
    )
    arguments = parser.parse_args()

    runtime_report, success = _probe(arguments.backend)
    report = {
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "environment": {"xtbloom_library_override": _override_report()},
        "distribution": _distribution_report(),
        **runtime_report,
    }
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
