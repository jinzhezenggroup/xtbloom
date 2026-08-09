#!/usr/bin/env python3
"""Cross-engine GFN2-xTB atom-count scaling benchmark with distinct systems.

This runner measures public single-point GFN2-xTB inference latency. Context,
calculator, and descriptor construction happen outside the timed region. For
gpuxtb FRESH calls, state initialization occurs inside ``gpuxtb_compute``;
xTB/tblite calculator rebuilds are outside timing, while dxtb's required
``Calculator.reset()`` remains inside its public inference call. The timed
boundary ends after synchronous host-visible energy and force publication:

- ``gpuxtb`` CPU and CUDA through the committed ctypes conformance adapter
  (``gpuxtb_public_api``), exactly like ``natoms_scaling.py``;
- ``xtb`` and ``tblite`` through their persistent public C API adapters;
- ``dxtb`` through the persistent in-process PyTorch adapter.

For every molecule size and batch size the batch is built from *distinct*
seeded thermal-like conformers of the same alkane stoichiometry (identical
atomic numbers, slightly different coordinates), so an engine cannot win a
batch row by reusing one identical geometry.  At batch size one the first slot
keeps the clean ideal alkane geometry. The sweep reports per-call latency
rather than process or calculator setup time. ``auto-warm`` rows are WARM
steady state after an untimed cold seed for engines that expose continuation;
dxtb remains cold. ``cold`` rows clear electronic state before every timed
inference call.

An optional MD-trajectory mode measures per-frame latency over a sequence of
nearly identical frames (positions mutated in place through the persistent
host descriptors), exercising gpuxtb's sequential geometry path while the
reference engines update their persistent structures per frame.

Artifacts are two explicit JSON/CSV paths in the same style as the other
benchmark harnesses. JSON is authoritative and retains raw timing samples,
complete final energy/force vectors, force digests for every repetition, run
identity, hardware, threads, correctness, and per-row diagnostics; CSV is the
compact row summary. Final publication evidence rejects a dirty repository and
can compare every row with a clean independent reference-engine artifact.
"""

from __future__ import annotations

import argparse
import array
import base64
import contextlib
import csv
import ctypes
import functools
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFORMANCE_TOOLS = REPOSITORY_ROOT / "tools" / "conformance"
if str(CONFORMANCE_TOOLS) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_TOOLS))
public_api = importlib.import_module("gpuxtb_public_api")

try:
    from .natoms_scaling import (
        CROSS_ENGINE_TOLERANCE_SOURCE,
        DEFAULT_CROSS_ENGINE_ENERGY_ATOL,
        DEFAULT_CROSS_ENGINE_FORCE_ATOL,
        Molecule,
        SystemSlice,
        _meson_compiler_executable_provenance,
        make_alkane,
    )
except ImportError:  # Direct ``python benchmarks/natoms_cross_engine.py`` execution.
    from natoms_scaling import (
        CROSS_ENGINE_TOLERANCE_SOURCE,
        DEFAULT_CROSS_ENGINE_ENERGY_ATOL,
        DEFAULT_CROSS_ENGINE_FORCE_ATOL,
        Molecule,
        SystemSlice,
        _meson_compiler_executable_provenance,
        make_alkane,
    )

try:
    from .xtb_adapter import XtbAdapter, XtbError
except ImportError:
    from xtb_adapter import XtbAdapter, XtbError

try:
    from .tblite_adapter import TbliteAdapter, TbliteError
except ImportError:
    from tblite_adapter import TbliteAdapter, TbliteError

try:
    from .dxtb_adapter import DxtbAdapter, DxtbError
except ImportError:
    from dxtb_adapter import DxtbAdapter, DxtbError

SCHEMA_VERSION = 2
DEFAULT_NATOMS = (5, 32, 122, 362, 602, 962)
DEFAULT_BATCH_SIZES = (1, 128)
SUPPORTED_ENGINES = (
    "gpuxtb-cpu",
    "gpuxtb-cuda",
    "xtb",
    "tblite",
    "dxtb-cpu",
    "dxtb-cuda",
)
DEFAULT_ENGINES = ("gpuxtb-cpu", "gpuxtb-cuda", "xtb", "tblite", "dxtb-cpu")
# This benchmark has a deliberately broader, owner-authorized compatibility
# gate than gpuxtb's primary conformance suite.  It decides whether one timing
# point may support the public cross-library performance figure; it is not a
# scientific acceptance tolerance and must never replace the manifest gates.
PUBLIC_BENCHMARK_ENERGY_ATOL_HARTREE = 2.0e-3
PUBLIC_BENCHMARK_FORCE_ATOL_HARTREE_PER_BOHR = 2.0e-3
CROSS_ENGINE_ENERGY_ATOL_HARTREE = PUBLIC_BENCHMARK_ENERGY_ATOL_HARTREE
CROSS_ENGINE_FORCE_ATOL_HARTREE_PER_BOHR = PUBLIC_BENCHMARK_FORCE_ATOL_HARTREE_PER_BOHR

# Record each library through its native public convergence controls. xTB and
# tblite document an accuracy factor of 1.0 as their default; gpuxtb exposes
# charge/energy thresholds directly, while dxtb exposes fixed-point/function
# residual tolerances with different stopping semantics.
PUBLIC_BENCHMARK_SCC_CHARGE_TOLERANCE = 1.0e-4
PUBLIC_BENCHMARK_SCC_ENERGY_TOLERANCE = 1.0e-6
REFERENCE_ACCURACY = 1.0
DXTB_FIXED_POINT_TOLERANCE = 1.0e-4
DXTB_MAX_NORM_TOLERANCE = 1.0e-5
DXTB_FUNCTION_TOLERANCE = 1.0e-4
DXTB_FORCE_CONVERGENCE = True
REPEATABILITY_ENERGY_ATOL_HARTREE = 1.0e-10
REPEATABILITY_FORCE_ATOL_HARTREE_PER_BOHR = 1.0e-8
PERTURB_SIGMA_BOHR = 0.02
TRAJECTORY_STEP_SIGMA_BOHR = 0.01


class BenchmarkError(RuntimeError):
    """An actionable benchmark request, adapter, or publication failure."""


def parse_csv_values(value: str) -> tuple[Any, ...]:
    """Parse one nonempty comma-separated CLI selection of integers."""
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise BenchmarkError("empty comma-separated selection")
    return tuple(parts)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse one nonempty comma-separated integer selection."""
    try:
        parsed = tuple(int(part.strip()) for part in parse_csv_values(value))
    except ValueError as exc:
        raise BenchmarkError(f"expected integers, got {value!r}") from exc
    if not parsed:
        raise BenchmarkError("empty integer selection")
    return parsed


def sha256_file(path: Path | None) -> str | None:
    """Return the lowercase SHA-256 of a file, or None when it is absent."""
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: Path) -> dict[str, Any]:
    """Return verified clean/dirty and HEAD revision of a Git checkout."""
    revision = run_text(("git", "-C", str(path), "rev-parse", "HEAD"), required=True)
    status = run_text(
        ("git", "-C", str(path), "status", "--porcelain"),
        required=True,
        allow_empty=True,
    )
    return {
        "head": revision,
        "dirty": bool(status),
    }


@functools.cache
def installed_distribution_identity(name: str) -> dict[str, Any] | None:
    """Return hash-pinned metadata for one installed Python distribution.

    Benchmark artifacts must remain reproducible when dxtb is consumed from
    an installed wheel rather than a Git checkout.  Verify every file listed
    by ``RECORD`` against its recorded hash/size, then hash a canonical
    inventory of the actual installed payload.  Only the digest and a
    credential-free source identity from a potentially sensitive
    ``direct_url.json`` are retained.
    """
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    record_text = distribution.read_text("RECORD")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url_identity = sanitize_direct_url_identity(direct_url_text)
    errors: list[str] = []
    payload: list[dict[str, Any]] = []
    verified_hashes = 0
    if record_text is None:
        errors.append("distribution has no RECORD")
    else:
        for record_path, hash_spec, size_text, *_ in csv.reader(
            record_text.splitlines()
        ):
            resolved = Path(distribution.locate_file(record_path)).resolve()
            if not resolved.is_file():
                errors.append(f"missing RECORD file: {record_path}")
                continue
            raw = resolved.read_bytes()
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            actual_size = len(raw)
            if size_text and int(size_text) != actual_size:
                errors.append(f"size mismatch: {record_path}")
            if hash_spec:
                algorithm, encoded = hash_spec.split("=", maxsplit=1)
                if algorithm != "sha256":
                    errors.append(f"unsupported RECORD hash: {record_path}")
                else:
                    expected = base64.urlsafe_b64decode(
                        encoded + "=" * (-len(encoded) % 4)
                    ).hex()
                    if expected != actual_sha256:
                        errors.append(f"hash mismatch: {record_path}")
                    else:
                        verified_hashes += 1
            payload.append(
                {
                    "path": record_path,
                    "sha256": actual_sha256,
                    "size": actual_size,
                }
            )
    payload.sort(key=lambda item: item["path"])
    payload_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "name": distribution.metadata.get("Name") or name,
        "version": distribution.version,
        "record_sha256": (
            hashlib.sha256(record_text.encode("utf-8")).hexdigest()
            if record_text is not None
            else None
        ),
        "direct_url_sha256": (
            hashlib.sha256(direct_url_text.encode("utf-8")).hexdigest()
            if direct_url_text is not None
            else None
        ),
        "direct_url_identity": direct_url_identity,
        "payload_verification": {
            "status": "verified" if not errors else "failed",
            "record_entries": len(payload),
            "verified_hashes": verified_hashes,
            "payload_sha256": payload_sha256,
            "errors": errors,
        },
    }


def sanitize_direct_url_identity(text: str | None) -> dict[str, Any] | None:
    """Return the source binding without retaining URL credentials.

    PEP 610 ``direct_url.json`` is the only reliable link between an installed
    editable dxtb payload and the clean source checkout supplied to the
    publication harness.  Local file URLs retain only their resolved path and
    editable flag; non-file URLs retain only the scheme.
    """
    if text is None:
        return None
    try:
        direct_url = json.loads(text)
        if not isinstance(direct_url, dict):
            raise TypeError("direct_url.json root is not an object")
        parsed_url = urllib.parse.urlparse(str(direct_url.get("url", "")))
        if parsed_url.scheme == "file":
            if parsed_url.netloc not in {"", "localhost"}:
                return {"scheme": "file", "parse_status": "nonlocal_authority"}
            dir_info = direct_url.get("dir_info") or {}
            if not isinstance(dir_info, dict):
                raise TypeError("direct_url.json dir_info is not an object")
            return {
                "scheme": "file",
                "local_source_path": str(
                    Path(urllib.parse.unquote(parsed_url.path)).resolve()
                ),
                "editable": bool(dir_info.get("editable", False)),
            }
        return {"scheme": parsed_url.scheme or None}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"scheme": None, "parse_status": "invalid"}


def run_text(
    command: Sequence[str], *, required: bool = False, allow_empty: bool = False
) -> str | None:
    """Run one diagnostic command without treating failed commands as evidence."""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        if required:
            raise BenchmarkError(f"cannot execute {command[0]!r}: {exc}") from exc
        return None
    if completed.returncode != 0:
        if required:
            diagnostic = completed.stderr.strip() or completed.stdout.strip()
            raise BenchmarkError(
                f"command failed ({completed.returncode}): {' '.join(command)}"
                + (f": {diagnostic}" if diagnostic else "")
            )
        return None
    output = completed.stdout.strip()
    return output if output or allow_empty else None


@functools.cache
def native_library_identity(path_text: str) -> dict[str, Any] | None:
    """Capture one shared library and every dependency resolved by ``ldd``."""
    path = Path(path_text).resolve()
    if not path.is_file():
        return None
    dynamic = run_text(("readelf", "-d", str(path)), required=True)
    ldd_output = run_text(("ldd", str(path)), required=True)
    dependencies: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for line in (ldd_output or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "not found" in stripped:
            unresolved.append(stripped)
            continue
        candidate: str | None = None
        soname = stripped.split(maxsplit=1)[0]
        if "=>" in stripped:
            right = stripped.split("=>", maxsplit=1)[1].strip()
            candidate = right.split(maxsplit=1)[0]
        elif stripped.startswith("/"):
            candidate = stripped.split(maxsplit=1)[0]
        if candidate is None or not candidate.startswith("/"):
            continue
        resolved = Path(candidate).resolve()
        if not resolved.is_file():
            continue
        dependencies.append(
            {
                "soname": soname,
                "path": str(resolved),
                "sha256": sha256_file(resolved),
            }
        )
    dependencies.sort(key=lambda item: (item["soname"], item["path"]))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "readelf_dynamic": dynamic,
        "ldd": ldd_output,
        "unresolved_dependencies": unresolved,
        "resolved_dependencies": dependencies,
    }


_MESON_SYNC_RESULTS: dict[tuple[str, str], dict[str, Any]] = {}


def meson_build_root(library: Path) -> Path | None:
    """Locate the Meson build tree containing one requested library path."""
    requested_library = library.absolute()
    return next(
        (
            directory
            for directory in (
                requested_library.parent,
                *requested_library.parents[:5],
            )
            if (directory / "meson-info" / "meson-info.json").is_file()
        ),
        None,
    )


def synchronize_meson_target(library: Path, source_root: Path) -> dict[str, Any]:
    """Update a clean Meson tree before publication timing and identity capture."""
    build_root = meson_build_root(library)
    if build_root is None:
        raise BenchmarkError("cannot locate Meson build tree for publication target")
    info_root = build_root / "meson-info"
    try:
        info = json.loads((info_root / "meson-info.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read Meson build identity: {exc}") from exc
    directories = info.get("directories") or {}
    if Path(str(directories.get("source", ""))).resolve() != source_root.resolve():
        raise BenchmarkError("Meson build tree is not bound to the requested source")
    try:
        targets = json.loads(
            (info_root / "intro-targets.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read Meson target identity: {exc}") from exc
    requested_library = library.resolve()
    selected_target = next(
        (
            target
            for target in targets
            if any(
                Path(str(filename)).resolve() == requested_library
                for filename in target.get("filename", [])
            )
        ),
        None,
    )
    if selected_target is None:
        raise BenchmarkError("selected library is not a Meson target output")
    target_type = str(selected_target.get("type", "")).replace(" ", "_")
    target_name = str(selected_target.get("name", ""))
    if not target_name or not target_type:
        raise BenchmarkError("selected Meson target has no stable compile selector")
    target_parent = requested_library.parent.relative_to(build_root.resolve())
    target_prefix = "" if target_parent == Path(".") else f"{target_parent}/"
    target_selector = f"{target_prefix}{target_name}:{target_type}"
    output = run_text(
        ("meson", "compile", "-C", str(build_root), target_selector),
        required=True,
        allow_empty=True,
    )
    resolved_library = library.resolve()
    if not resolved_library.is_file():
        raise BenchmarkError("Meson compile did not produce the selected library")
    result = {
        "command": [
            "meson",
            "compile",
            "-C",
            str(build_root.resolve()),
            target_selector,
        ],
        "target_id": selected_target.get("id"),
        "target_selector": target_selector,
        "status": "up_to_date_or_rebuilt",
        "output": output,
        "library_path": str(resolved_library),
        "library_sha256": sha256_file(resolved_library),
        "library_mtime_ns": resolved_library.stat().st_mtime_ns,
    }
    key = (str(library.absolute()), str(source_root.resolve()))
    _MESON_SYNC_RESULTS[key] = result
    meson_build_identity.cache_clear()
    native_library_identity.cache_clear()
    return result


@functools.cache
def meson_build_identity(library: Path, source_root: Path) -> dict[str, Any] | None:
    """Bind one native library to Meson source, options, and compiler bytes."""
    build_root = meson_build_root(library)
    if build_root is None:
        return None
    resolved_library = library.resolve()
    info_root = build_root / "meson-info"
    try:
        info = json.loads((info_root / "meson-info.json").read_text(encoding="utf-8"))
        project = json.loads(
            (info_root / "intro-projectinfo.json").read_text(encoding="utf-8")
        )
        targets = json.loads(
            (info_root / "intro-targets.json").read_text(encoding="utf-8")
        )
        compilers = json.loads(
            (info_root / "intro-compilers.json").read_text(encoding="utf-8")
        )
        build_options = json.loads(
            (info_root / "intro-buildoptions.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read Meson build identity: {exc}") from exc
    directories = info.get("directories") or {}
    if Path(str(directories.get("source", ""))).resolve() != source_root.resolve():
        return None
    target = next(
        (
            item
            for item in targets
            if any(
                Path(filename).resolve() == resolved_library
                for filename in item.get("filename") or []
            )
        ),
        None,
    )
    if target is None:
        return None
    compiler_identities: list[dict[str, Any]] = []
    for machine, languages in compilers.items():
        for language, compiler in languages.items():
            executable_files, unresolved_entries = (
                _meson_compiler_executable_provenance(compiler.get("exelist"))
            )
            primary = executable_files[0] if executable_files else {}
            compiler_identities.append(
                {
                    "machine": machine,
                    "language": language,
                    "id": compiler.get("id"),
                    "version": compiler.get("version"),
                    "full_version": compiler.get("full_version"),
                    "exelist": compiler.get("exelist"),
                    "executable": primary.get("path"),
                    "executable_sha256": primary.get("sha256"),
                    "executable_files": executable_files,
                    "unresolved_configure_time_entries": unresolved_entries,
                }
            )
    compiler_identities.sort(key=lambda item: (item["machine"], item["language"]))
    introspection_hashes = {
        path.name: sha256_file(path)
        for path in sorted(info_root.glob("*.json"))
        if path.is_file()
    }
    return {
        "build_system": "meson",
        "build_root": str(build_root.resolve()),
        "meson_version": (info.get("meson_version") or {}).get("full"),
        "project": project,
        "source_state": git_state(source_root),
        "target": {
            "name": target.get("name"),
            "id": target.get("id"),
            "type": target.get("type"),
            "filename": [str(Path(item).resolve()) for item in target["filename"]],
        },
        "source_target_sync": _MESON_SYNC_RESULTS.get(
            (str(library.absolute()), str(source_root.resolve()))
        ),
        "build_options": {
            str(item.get("name")): item.get("value") for item in build_options
        },
        "compilers": compiler_identities,
        "introspection_sha256": introspection_hashes,
    }


def cmake_build_identity(library: Path) -> dict[str, Any] | None:
    """Capture the selected build cache and compiler/toolkit identity."""
    resolved_library = library.resolve()
    cache = next(
        (
            directory / "CMakeCache.txt"
            for directory in (resolved_library.parent, *resolved_library.parents[:5])
            if (directory / "CMakeCache.txt").is_file()
        ),
        None,
    )
    if cache is None:
        return None
    selected_names = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_HOME_DIRECTORY",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_FLAGS",
        "CMAKE_CUDA_FLAGS_RELEASE",
        "CMAKE_SHARED_LINKER_FLAGS",
        "GPUXTB_ENABLE_CUDA",
        "GPUXTB_CPU_LINALG_LIBRARY",
        "GPUXTB_CPU_LINALG_RUNTIME",
        "GPUXTB_MKL_RT_LIBRARY",
    }
    selected: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", maxsplit=1)
        key = key_and_type.split(":", maxsplit=1)[0]
        if key in selected_names:
            selected[key] = value
    cuda_compiler = selected.get("CMAKE_CUDA_COMPILER")
    cxx_compiler = Path(selected.get("CMAKE_CXX_COMPILER", "c++")).resolve()
    source_root_text = selected.get("CMAKE_HOME_DIRECTORY")
    source_state = git_state(Path(source_root_text)) if source_root_text else None
    provider_text = (
        selected.get("GPUXTB_CPU_LINALG_LIBRARY")
        or selected.get("GPUXTB_MKL_RT_LIBRARY")
        or selected.get("GPUXTB_CPU_LINALG_RUNTIME")
    )
    provider = Path(provider_text).resolve() if provider_text else None
    return {
        "cache_path": str(cache),
        "cache_sha256": sha256_file(cache),
        "library_size": resolved_library.stat().st_size,
        "library_mtime_ns": resolved_library.stat().st_mtime_ns,
        "selected": selected,
        "source_state": source_state,
        "cxx_compiler": str(cxx_compiler),
        "cxx_compiler_sha256": sha256_file(cxx_compiler),
        "cxx_version": run_text((str(cxx_compiler), "--version"), required=True),
        "cuda_compiler": str(Path(cuda_compiler).resolve()) if cuda_compiler else None,
        "cuda_compiler_sha256": (
            sha256_file(Path(cuda_compiler).resolve()) if cuda_compiler else None
        ),
        "cuda_compiler_version": (
            run_text((cuda_compiler, "--version"), required=True)
            if cuda_compiler
            else None
        ),
        "cpu_linalg_provider": (
            native_library_identity(str(provider))
            if provider and provider.is_file()
            else None
        ),
    }


def cpu_model() -> str | None:
    """Read the CPU model name when /proc/cpuinfo is available."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def cuda_runtime_device_uuid(device_id: int) -> str | None:
    """Query the selected logical device UUID through the CUDA driver API."""
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        uuid_function = getattr(driver, "cuDeviceGetUuid_v2", None)
        if uuid_function is None:
            uuid_function = driver.cuDeviceGetUuid
        uuid_function.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
        uuid_function.restype = ctypes.c_int
        if driver.cuInit(0) != 0:
            return None
        device = ctypes.c_int()
        if driver.cuDeviceGet(ctypes.byref(device), device_id) != 0:
            return None
        raw_uuid = (ctypes.c_ubyte * 16)()
        if uuid_function(raw_uuid, device.value) != 0:
            return None
    except (AttributeError, OSError):
        return None
    hexadecimal = bytes(raw_uuid).hex()
    groups = (
        hexadecimal[:8],
        hexadecimal[8:12],
        hexadecimal[12:16],
        hexadecimal[16:20],
        hexadecimal[20:],
    )
    return "GPU-" + "-".join(groups)


@functools.cache
def selected_cuda_device_identity(device_id: int) -> dict[str, Any] | None:
    """Resolve one CUDA ordinal through CUDA_VISIBLE_DEVICES to GPU identity."""
    inventory = run_text(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    if inventory is None:
        return None
    devices: list[dict[str, str]] = []
    for line in inventory.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        devices.append(
            dict(
                zip(
                    ("physical_index", "uuid", "name", "driver", "memory_mib"),
                    fields,
                    strict=True,
                )
            )
        )
    runtime_uuid = cuda_runtime_device_uuid(device_id)
    visible_text = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = (
        [item.strip() for item in visible_text.split(",") if item.strip()]
        if visible_text
        else []
    )
    token = (
        visible[device_id] if visible and device_id < len(visible) else str(device_id)
    )
    selected = next(
        (
            item
            for item in devices
            if item["uuid"] == runtime_uuid
            or token in {item["physical_index"], item["uuid"]}
            or item["uuid"].startswith(token)
        ),
        None,
    )
    return {
        "cuda_ordinal": device_id,
        "CUDA_VISIBLE_DEVICES": visible_text,
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
        "resolved_visibility_token": token,
        "runtime_uuid": runtime_uuid,
        "device": selected,
        "inventory": devices,
    }


def current_rss_bytes() -> int | None:
    """Return current host RSS for this process, or None."""
    try:
        pages = int(Path(f"/proc/{os.getpid()}/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None


def log(message: str) -> None:
    """Emit one flushable, timestamped progress line on stdout."""
    stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
    print(  # noqa: T201 - benchmark CLI progress output
        f"[{stamp}] {message}", flush=True
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the requested nearest-rank percentile of a sample set."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def timing_summary(samples_ms: Sequence[float], batch_size: int) -> dict[str, Any]:
    """Summarize raw latencies without discarding the samples."""
    return {
        "samples_ms": list(samples_ms),
        "count": len(samples_ms),
        "min_ms": min(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "systems_per_second_at_median": 1000.0
        * batch_size
        / statistics.median(samples_ms),
    }


@dataclass(frozen=True)
class Cell:
    """One engine/natoms/batch-size benchmark coordinate."""

    engine: str
    natoms: int
    batch_size: int
    cpu_threads: int
    device_id: int


@dataclass(frozen=True)
class ReferenceArtifact:
    """One clean independent xTB/tblite artifact for output qualification."""

    path: Path
    sha256: str
    engine: str
    metadata: dict[str, Any]
    rows: dict[tuple[int, int], dict[str, Any]]


@dataclass
class BatchStorage:
    """Duck-typed storage accepted by ``gpuxtb_public_api._make_batch``."""

    atom_offsets: list[int]
    atomic_numbers: list[int]
    positions: list[float]
    molecular_charges: list[float]
    unpaired_electrons: list[int]
    spin_channels: list[int]
    point_charge_offsets: list[int]
    point_charge_positions: list[float]
    point_charge_values: list[float]
    point_charge_gammas: list[float]
    efields: list[list[float] | None]
    slices: list[SystemSlice]
    keepalive: list[Any]


def _perturbed_positions(
    base: Sequence[float],
    rng: random.Random,
    sigma_bohr: float,
) -> tuple[float, ...]:
    """Apply an independent Gaussian displacement to every coordinate."""
    return tuple(float(value) + rng.gauss(0.0, sigma_bohr) for value in base)


def build_batch(
    base_molecule: Molecule,
    batch_size: int,
    seed: int,
    perturb_sigma_bohr: float = PERTURB_SIGMA_BOHR,
) -> BatchStorage:
    """Build a ragged batch of *distinct* conformers of one alkane.

    Slot zero keeps the ideal geometry; every later slot receives a seeded
    independent thermal-like perturbation so the batch never contains two
    identical systems, which would permit an engine to reuse one geometry.
    """
    atom_offsets = [0]
    atomic_numbers: list[int] = []
    positions: list[float] = []
    slices: list[SystemSlice] = []
    rng = random.Random(seed)
    for slot in range(batch_size):
        if slot == 0:
            slot_positions = tuple(base_molecule.positions_bohr)
        else:
            slot_positions = _perturbed_positions(
                base_molecule.positions_bohr, rng, perturb_sigma_bohr
            )
        begin = len(atomic_numbers)
        atomic_numbers.extend(base_molecule.atomic_numbers)
        positions.extend(slot_positions)
        atom_offsets.append(len(atomic_numbers))
        slices.append(SystemSlice(begin, len(atomic_numbers)))
    return BatchStorage(
        atom_offsets=atom_offsets,
        atomic_numbers=atomic_numbers,
        positions=positions,
        molecular_charges=[0.0] * batch_size,
        unpaired_electrons=[0] * batch_size,
        spin_channels=[1] * batch_size,
        point_charge_offsets=[0] * (batch_size + 1),
        point_charge_positions=[],
        point_charge_values=[],
        point_charge_gammas=[],
        efields=[None] * batch_size,
        slices=slices,
        keepalive=[],
    )


def build_trajectory(
    base_molecule: Molecule,
    frames: int,
    seed: int,
    step_sigma_bohr: float = TRAJECTORY_STEP_SIGMA_BOHR,
) -> list[tuple[float, ...]]:
    """Return frames of one alkane mutated by a seeded small random walk.

    Successive frames deliberately keep their coordinates close (like an MD
    trajectory), which is exactly the sequential-geometry regime the benchmark
    wants to isolate.
    """
    rng = random.Random(seed)
    frames_out: list[tuple[float, ...]] = []
    current = list(base_molecule.positions_bohr)
    for _ in range(frames):
        current = list(_perturbed_positions(current, rng, step_sigma_bohr))
        frames_out.append(tuple(current))
    return frames_out


def configure_gpuxtb_scc(
    options: Any,  # noqa: ANN401 - ctypes compute-options mirror
    charge_tolerance: float = PUBLIC_BENCHMARK_SCC_CHARGE_TOLERANCE,
    energy_tolerance: float = PUBLIC_BENCHMARK_SCC_ENERGY_TOLERANCE,
    max_iterations: int = 500,
) -> None:
    """Pin the SCC convergence controls on a gpuxtb options object.

    Publication rows align gpuxtb's direct controls with the nominal public
    defaults used by xTB/tblite. The resulting observables must still pass the
    separately declared benchmark compatibility gate before their timings can
    support a public comparison.
    """
    options.max_scc_iterations = max_iterations
    options.charge_tolerance = charge_tolerance
    options.energy_tolerance = energy_tolerance


def cuda_synchronize(
    control: Any,  # noqa: ANN401 - ctypes CUDA runtime control mirror
) -> None:
    """Complete all CUDA work at the documented timing boundary."""
    control.runtime.cudaDeviceSynchronize.argtypes = []
    control.runtime.cudaDeviceSynchronize.restype = ctypes.c_int
    status = control.runtime.cudaDeviceSynchronize()
    control._check(status, "cudaDeviceSynchronize")


class GpuxtbRunner:
    """Persistent public-C-ABI runner with in-place position mutation."""

    def __init__(
        self,
        library_path: Path,
        storage: BatchStorage,
        backend: str,
        cpu_threads: int,
        device_id: int,
        scc_charge_tolerance: float = PUBLIC_BENCHMARK_SCC_CHARGE_TOLERANCE,
        scc_energy_tolerance: float = PUBLIC_BENCHMARK_SCC_ENERGY_TOLERANCE,
        scc_max_iterations: int = 500,
    ) -> None:
        self.library = public_api._configure_library(library_path)
        self.storage = storage
        self.backend = backend
        self.context = public_api._make_context(
            self.library, backend, device_id, cpu_threads
        )
        self.memory = public_api.DescriptorMemory("host", device_id)
        self.has_cuda = backend == "cuda"
        self.cuda_control = None
        if self.has_cuda:
            self.cuda_control = public_api.CudaRuntime(device_id)
        self.batch = public_api._make_batch(
            self.library,
            storage,
            self.memory,
            include_spin_channels=True,
        )
        self.options = public_api.ComputeOptions()
        public_api._call_ok(
            self.library,
            self.library.gpuxtb_compute_options_init(
                ctypes.byref(self.options), ctypes.sizeof(self.options)
            ),
            "gpuxtb_compute_options_init",
        )
        self.options.model = public_api.GPUXTB_MODEL_GFN2_XTB
        self.options.flags = public_api.GPUXTB_COMPUTE_ENERGY
        self.options.flags |= public_api.GPUXTB_COMPUTE_FORCES
        configure_gpuxtb_scc(
            self.options,
            charge_tolerance=scc_charge_tolerance,
            energy_tolerance=scc_energy_tolerance,
            max_iterations=scc_max_iterations,
        )
        systems = len(storage.slices)
        atoms = len(storage.atomic_numbers)
        self.energies = (ctypes.c_double * systems)()
        self.forces = (ctypes.c_double * (3 * atoms))()
        self.iterations = (ctypes.c_int32 * systems)()
        self.converged = (ctypes.c_uint8 * systems)()
        self.statuses = (ctypes.c_int32 * systems)()
        self.result = public_api.BatchResult()
        public_api._call_ok(
            self.library,
            self.library.gpuxtb_batch_result_init(
                ctypes.byref(self.result), ctypes.sizeof(self.result)
            ),
            "gpuxtb_batch_result_init",
        )
        self.result.energies = self.memory.output(self.energies, "energies")
        self.result.forces = self.memory.output(self.forces, "forces")
        self.result.scc_iterations = self.memory.output(
            self.iterations, "scc_iterations"
        )
        self.result.scc_converged = self.memory.output(self.converged, "scc_converged")
        self.result.per_system_status = self.memory.output(
            self.statuses, "per_system_status"
        )
        # Retain the caller-owned positions owner array for in-place mutation
        # so repeated calls can stream updated trajectories through the same
        # host descriptors without rebuilding them. ``input`` appended the
        # owner to ``storage.keepalive``; the position owner is the only
        # c_double array covering the full flattened coordinate set.
        self._position_owner = None
        for owner in storage.keepalive:
            if (
                isinstance(owner, ctypes.Array)
                and getattr(owner, "_type_", None) is ctypes.c_double
                and len(owner) == len(storage.positions)
            ):
                self._position_owner = owner
                break
        if self._position_owner is None:
            raise BenchmarkError("gpuxtb positions owner array is missing")
        self.closed = False

    def set_start_mode(self, mode: str) -> None:
        """Select FRESH or WARM SCC continuation without rebuilding descriptors."""
        if mode not in ("fresh", "warm"):
            raise BenchmarkError(f"unsupported gpuxtb start mode: {mode}")
        self.options.scc_start_mode = (
            public_api.GPUXTB_SCC_START_WARM
            if mode == "warm"
            else public_api.GPUXTB_SCC_START_FRESH
        )

    def set_positions(self, positions: Sequence[float]) -> None:
        """Write a new flattened position vector into the persistent owner."""
        if len(positions) != len(self.storage.positions):
            raise BenchmarkError("trajectory frame length changed")
        for index, value in enumerate(positions):
            self._position_owner[index] = value

    def invoke(self) -> None:
        """Execute one synchronous public batch inference."""
        public_api._call_ok(
            self.library,
            self.library.gpuxtb_compute(
                self.context,
                ctypes.byref(self.batch),
                ctypes.byref(self.options),
                ctypes.byref(self.result),
            ),
            f"gpuxtb {self.backend} inference",
        )
        if self.has_cuda:
            cuda_synchronize(self.cuda_control)

    def snapshot(self) -> dict[str, Any]:
        """Download outputs once and normalize observables."""
        self.memory.download_outputs()
        return {
            "energies_hartree": [float(value) for value in self.energies],
            "forces_hartree_per_bohr": [float(value) for value in self.forces],
            "scc_iterations": [int(value) for value in self.iterations],
            "scc_converged": [int(value) for value in self.converged],
            "per_system_status": [int(value) for value in self.statuses],
        }

    def close(self) -> None:
        """Release descriptors before destroying the backend context."""
        if self.closed:
            return
        self.closed = True
        try:
            if self.cuda_control is not None:
                self.cuda_control.close()
        finally:
            self.library.gpuxtb_context_destroy(self.context)


class ReferenceRunner:
    """Persistent adapter for xtb/tblite/dxtb logical batches."""

    def __init__(
        self,
        engine: str,
        library_path: Path,
        storage: BatchStorage,
        cpu_threads: int,
        device_id: int,
        dxtb_source: Path | None,
        max_iterations: int,
    ) -> None:
        if engine == "xtb":
            self.adapter = XtbAdapter(
                library_path,
                storage,
                "force",
                None,
                accuracy=REFERENCE_ACCURACY,
                max_iterations=max_iterations,
                threads=cpu_threads,
            )
            self.engine = "xtb"
        elif engine == "tblite":
            self.adapter = TbliteAdapter(
                library_path,
                storage,
                "force",
                accuracy=REFERENCE_ACCURACY,
                max_iterations=max_iterations,
                collect_atomic_charges=False,
                threads=cpu_threads,
            )
            self.engine = "tblite"
        elif engine in ("dxtb-cpu", "dxtb-cuda"):
            backend = "cpu" if engine == "dxtb-cpu" else "cuda"
            self.adapter = DxtbAdapter(
                storage,
                "force",
                backend,
                device_id=device_id,
                cpu_threads=cpu_threads,
                source_root=dxtb_source,
                accuracy=DXTB_FIXED_POINT_TOLERANCE,
                max_norm_tolerance=DXTB_MAX_NORM_TOLERANCE,
                function_tolerance=DXTB_FUNCTION_TOLERANCE,
                force_convergence=DXTB_FORCE_CONVERGENCE,
                max_iterations=max_iterations,
            )
            self.engine = "dxtb"
        else:
            raise BenchmarkError(f"unsupported reference engine: {engine}")
        self._dxtb_adapter = self.adapter if engine.startswith("dxtb") else None
        # dxtb invalidates its autograd graph and result cache inside every
        # measured invocation, so it has no public warm-continuation path
        # equivalent to gpuxtb/xTB/tblite.
        self.always_cold = self._dxtb_adapter is not None
        states = getattr(self.adapter, "states", ())
        if self._dxtb_adapter is not None or not states:
            self._position_slices = []
        else:
            self._position_slices = [
                (state, slice_.atom_begin, slice_.atom_end)
                for state, slice_ in zip(states, storage.slices, strict=True)
            ]

    def set_positions(self, positions: Sequence[float]) -> None:
        """Stream one frame into every persistent system without rebuilding."""
        if self._dxtb_adapter is not None:
            # dxtb holds one torch tensor on the selected device; write each
            # system's atom range into the persistent tensor in place.
            import torch

            flat = torch.tensor(
                list(positions),
                dtype=torch.float64,
                device=self._dxtb_adapter.device,
            )
            tensor = self._dxtb_adapter.positions
            with torch.no_grad():
                tensor.view(-1).copy_(flat)
            return
        if not self._position_slices:
            raise BenchmarkError(f"{self.engine} adapter has no position stream")
        for state, atom_begin, atom_end in self._position_slices:
            state_length = len(state.positions)
            expected = 3 * (atom_end - atom_begin)
            if state_length != expected:
                raise BenchmarkError(
                    f"{self.engine} state positions length {state_length} "
                    f"does not match slice {expected}"
                )
            for index in range(expected):
                state.positions[index] = positions[3 * atom_begin + index]

    def invoke(self) -> None:
        """Run through synchronous host-visible energy/force publication."""
        self.adapter.invoke()
        if self._dxtb_adapter is not None:
            # Include CUDA-to-host tensor publication in the same timed
            # boundary as caller-owned host buffers in the native adapters.
            # Python list normalization remains outside timing for all engines.
            self.adapter.publish_to_host()
        elif getattr(self.adapter, "backend", None) == "cuda":
            self.adapter.synchronize()

    def restart_scc(self) -> None:
        """Drop convergence state so the next sample is a genuine cold solve.

        Only engines whose persistent adapter can rebuild the SCC state support
        this; the runner no-ops otherwise (gpuxtb FRESH and dxtb reset already
        cold-start every measured call).
        """
        restart = getattr(self.adapter, "restart_scc", None)
        if restart is not None:
            restart()

    def snapshot(self) -> dict[str, Any]:
        """Normalize persistent reference results."""
        output = self.adapter.results()
        return {
            "energies_hartree": list(output["energies_hartree"]),
            "forces_hartree_per_bohr": (
                list(output["forces_hartree_per_bohr"])
                if "forces_hartree_per_bohr" in output
                else None
            ),
            "scc_iterations": None,
            "scc_converged": None,
            "per_system_status": None,
        }

    def close(self) -> None:
        """Release all persistent reference state exactly once."""
        self.adapter.close()


def _force_digest(values: Sequence[float]) -> str:
    """Hash one complete binary64 force vector in canonical little endian."""
    packed = array.array("d", values)
    if sys.byteorder != "little":
        packed.byteswap()
    return hashlib.sha256(packed.tobytes()).hexdigest()


def encode_force_vector(values: Sequence[float]) -> dict[str, Any]:
    """Encode a complete force vector as compressed portable binary64."""
    numbers = array.array("d", (float(value) for value in values))
    if sys.byteorder != "little":
        numbers.byteswap()
    raw = numbers.tobytes()
    return {
        "encoding": "zlib+base64",
        "count": len(numbers),
        "sha256_binary64_le": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(zlib.compress(raw, level=9)).decode("ascii"),
    }


def decode_force_vector(row: dict[str, Any]) -> list[float] | None:
    """Decode and validate a row's complete final force vector.

    Plain lists remain accepted for development artifacts produced before the
    compact schema-v2 representation was introduced.
    """
    plain = row.get("forces_hartree_per_bohr")
    if isinstance(plain, list):
        return [float(value) for value in plain]
    encoded = row.get("forces_binary64_le_zlib_base64")
    if not isinstance(encoded, dict) or encoded.get("encoding") != "zlib+base64":
        return None
    count = encoded.get("count")
    digest = encoded.get("sha256_binary64_le")
    data = encoded.get("data")
    if type(count) is not int or count < 0 or not isinstance(digest, str):
        raise BenchmarkError("encoded force vector has invalid metadata")
    if not isinstance(data, str):
        raise BenchmarkError("encoded force vector has no base64 payload")
    try:
        raw = zlib.decompress(base64.b64decode(data, validate=True))
    except (ValueError, zlib.error) as exc:
        raise BenchmarkError(f"encoded force vector is corrupt: {exc}") from exc
    if len(raw) != count * struct.calcsize("<d"):
        raise BenchmarkError("encoded force vector byte count is inconsistent")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise BenchmarkError("encoded force vector digest mismatch")
    values = [value[0] for value in struct.iter_unpack("<d", raw)]
    if not all(math.isfinite(value) for value in values):
        raise BenchmarkError("encoded force vector contains non-finite values")
    return values


def _max_abs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the maximum absolute difference between equal-length vectors."""
    if len(left) != len(right):
        raise BenchmarkError(
            f"vector length mismatch: observed {len(left)}, expected {len(right)}"
        )
    return max(
        (abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)),
        default=0.0,
    )


def measure_cell(
    runner: Any,  # noqa: ANN401 - gpuxtb/xtb/tblite/dxtb adapter union
    protocol: tuple[int, int],
    cell: Cell,
    start_policy: str = "auto-warm",
    repeatability_energy_atol_hartree: float = REPEATABILITY_ENERGY_ATOL_HARTREE,
    repeatability_force_atol_hartree_per_bohr: float = (
        REPEATABILITY_FORCE_ATOL_HARTREE_PER_BOHR
    ),
) -> dict[str, Any]:
    """Run warmups and measured samples and return a normalized row fragment.

    ``start_policy`` controls SCC restart semantics for every engine:

    - ``auto-warm``: one untimed cold seed establishes state even when the
      requested warmup count is zero, then every warmup and measured sample
      continues from that state (gpuxtb strict WARM; xTB/tblite persistent
      warm state). dxtb has no equivalent continuation and remains cold.
    - ``cold``: every measured call starts without reusable electronic state.
      gpuxtb selects FRESH before timing, but its state initialization occurs
      inside ``gpuxtb_compute``; xTB/tblite rebuild their calculator outside
      timing; dxtb's required reset remains inside its measured public call.
    """
    warmups, repetitions = protocol
    gpuxtb_runner = hasattr(runner, "set_start_mode")
    restart = getattr(runner, "restart_scc", None)
    always_cold = bool(getattr(runner, "always_cold", False))

    def cold_start() -> None:
        """Force a genuine cold start for whichever runner variant is active."""
        if gpuxtb_runner:
            runner.set_start_mode("fresh")
        elif restart is not None:
            restart()

    if start_policy == "auto-warm" and not always_cold:
        # The seed is a protocol step, not one of the optional warmups. Keep
        # it explicit so ``--warmups 0`` still produces genuine WARM samples.
        cold_start()
        runner.invoke()

    for _ in range(warmups):
        if start_policy == "auto-warm" and not always_cold:
            if gpuxtb_runner:
                runner.set_start_mode("warm")
        else:
            cold_start()
        runner.invoke()
    raw_samples: list[dict[str, Any]] = []
    force_samples: list[list[float]] = []
    expected_energy_count = cell.batch_size
    expected_force_count = 3 * cell.natoms * cell.batch_size
    for sample_index in range(repetitions):
        if start_policy == "auto-warm" and not always_cold:
            if gpuxtb_runner:
                runner.set_start_mode("warm")
        else:
            cold_start()
        start = time.perf_counter_ns()
        runner.invoke()
        elapsed_ms = (time.perf_counter_ns() - start) * 1.0e-6
        snapshot = runner.snapshot()
        energies = snapshot.get("energies_hartree")
        forces = snapshot.get("forces_hartree_per_bohr")
        if not isinstance(energies, list) or len(energies) != expected_energy_count:
            actual = len(energies) if isinstance(energies, list) else None
            raise BenchmarkError(
                f"inference returned {actual} energies; expected "
                f"{expected_energy_count}"
            )
        if not isinstance(forces, list) or len(forces) != expected_force_count:
            actual = len(forces) if isinstance(forces, list) else None
            raise BenchmarkError(
                f"inference returned {actual} force values; expected "
                f"{expected_force_count}"
            )
        if not all(math.isfinite(float(value)) for value in energies):
            raise BenchmarkError("inference returned non-finite energies")
        if not all(math.isfinite(float(value)) for value in forces):
            raise BenchmarkError("inference returned non-finite forces")
        normalized_forces = [float(value) for value in forces]
        force_samples.append(normalized_forces)
        raw_samples.append(
            {
                "sample_index": sample_index,
                "latency_ms": elapsed_ms,
                "energies_hartree": [float(value) for value in energies],
                "force_count": len(normalized_forces),
                "forces_sha256_binary64_le": _force_digest(normalized_forces),
                "scc_iterations": snapshot["scc_iterations"],
                "scc_converged": snapshot["scc_converged"],
                "per_system_status": snapshot["per_system_status"],
            }
        )
    status_known = all(
        sample.get("per_system_status") is not None for sample in raw_samples
    )
    status_ok = status_known and all(
        sample.get("per_system_status") is None
        or all(
            status == public_api.GPUXTB_STATUS_SUCCESS
            for status in sample["per_system_status"]
        )
        for sample in raw_samples
    )
    convergence_known = all(
        sample.get("scc_converged") is not None for sample in raw_samples
    )
    converged_ok = convergence_known and all(
        sample.get("scc_converged") is None
        or all(value == 1 for value in sample["scc_converged"])
        for sample in raw_samples
    )
    energy_reference = raw_samples[0]["energies_hartree"]
    force_reference = force_samples[0]
    energy_drift = max(
        _max_abs_delta(sample["energies_hartree"], energy_reference)
        for sample in raw_samples
    )
    force_drift = max(
        _max_abs_delta(sample, force_reference) for sample in force_samples
    )
    repeatability_ok = (
        energy_drift <= repeatability_energy_atol_hartree
        and force_drift <= repeatability_force_atol_hartree_per_bohr
    )
    # A persistent WARM call starts from the electronic state published by the
    # preceding call.  At a finite SCC tolerance, repeated calls at identical
    # geometry may therefore continue converging and are not identical-state
    # repetitions.  Preserve that drift as useful evidence, but apply the
    # strict repeatability eligibility gate only when every measured call is
    # independently cold/reset (including dxtb, whose reset is timed).
    repeatability_gate_applied = start_policy == "cold" or always_cold
    latencies = [sample["latency_ms"] for sample in raw_samples]
    iteration_min = iteration_max = None
    iterations = [
        value
        for sample in raw_samples
        if sample["scc_iterations"] is not None
        for value in sample["scc_iterations"]
    ]
    if iterations:
        iteration_min = min(iterations)
        iteration_max = max(iterations)
    fragment = {
        "effective_start_policy": "cold" if always_cold else start_policy,
        "state_preparation_timing": (
            "inside_timed_invoke"
            if always_cold
            else (
                "untimed_cold_seed_then_persistent"
                if start_policy == "auto-warm"
                else (
                    "fresh_state_initialization_inside_timed_public_call"
                    if gpuxtb_runner
                    else "untimed_calculator_rebuild_before_each_call"
                )
            )
        ),
        "raw_samples": raw_samples,
        # Cross-engine qualification happens before publication and must cover
        # every timed result, including evolving WARM samples.  Keep complete
        # force samples transiently in memory; the JSON archive retains only
        # the final compressed vector plus one digest per repetition.
        "_force_samples_hartree_per_bohr": force_samples,
        "timing": timing_summary(latencies, cell.batch_size),
        "energies_hartree": raw_samples[-1]["energies_hartree"],
        "forces_hartree_per_bohr": force_samples[-1],
        "iteration_summary": {"min": iteration_min, "max": iteration_max},
        "correctness": {
            "status": (
                "pass"
                if (
                    (status_ok or not status_known)
                    and (converged_ok or not convergence_known)
                    and (repeatability_ok or not repeatability_gate_applied)
                )
                else "fail"
            ),
            "finite_energies": True,
            "finite_forces": True,
            "force_value_count": expected_force_count,
            "scc_converged_ok": converged_ok if convergence_known else None,
            "scc_status_ok": status_ok if status_known else None,
            "repeatability": {
                "gate_applied": repeatability_gate_applied,
                "gate_reason": (
                    "independent_cold_or_reset_calls"
                    if repeatability_gate_applied
                    else "persistent_warm_state_can_continue_scc_convergence"
                ),
                "energy_atol_hartree": repeatability_energy_atol_hartree,
                "max_abs_energy_drift_hartree": energy_drift,
                "force_atol_hartree_per_bohr": (
                    repeatability_force_atol_hartree_per_bohr
                ),
                "max_abs_force_drift_hartree_per_bohr": force_drift,
            },
            "cross_engine": {"status": "not_requested"},
        },
    }
    return fragment


def base_row(cell: Cell) -> dict[str, Any]:
    """Create stable identity fields shared by every row type."""
    if cell.engine == "gpuxtb-cuda":
        input_residency = "host_descriptor_staged_by_public_api"
    elif cell.engine == "dxtb-cuda":
        input_residency = "persistent_cuda_device_tensors"
    else:
        input_residency = "host"
    return {
        "engine": cell.engine,
        "natoms": cell.natoms,
        "batch_size": cell.batch_size,
        "total_atoms_in_batch": cell.natoms * cell.batch_size,
        "cpu_threads": cell.cpu_threads,
        "device_id": cell.device_id,
        "requested_properties": ["energy", "forces"],
        "workload_seed": cell.natoms * 1000 + cell.batch_size,
        "input_residency": input_residency,
        "output_residency_at_timing_boundary": "host_visible",
    }


def unavailable_row(cell: Cell, reason: str) -> dict[str, Any]:
    """Create an explicitly unavailable row for a missing engine."""
    row = base_row(cell)
    row.update({"availability": "unavailable", "reason": reason})
    return row


def error_row(cell: Cell, error: str) -> dict[str, Any]:
    """Create an error row without inventing unavailable reasons."""
    row = base_row(cell)
    row.update({"availability": "error", "error": error})
    return row


def load_reference_artifact(
    path: Path, allow_dirty_evidence: bool = False
) -> ReferenceArtifact:
    """Load one clean panel-matched xTB/tblite output-reference artifact."""
    resolved = path.resolve()
    try:
        payload = resolved.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(
            f"cannot read reference artifact {resolved}: {exc}"
        ) from exc
    if document.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkError(
            f"reference artifact must use schema version {SCHEMA_VERSION}"
        )
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise BenchmarkError("reference artifact has no metadata object")
    designation = metadata.get("comparison_reference") or {}
    reference_engine = designation.get("engine")
    if designation.get(
        "designation"
    ) != "independent_baseline" or reference_engine not in {"xtb", "tblite"}:
        raise BenchmarkError(
            "reference artifact must designate one independent xTB/tblite baseline"
        )
    commit = metadata.get("commit") or {}
    if not commit.get("head") or (
        commit.get("dirty") is not False and not allow_dirty_evidence
    ):
        raise BenchmarkError("reference artifact must come from a verified clean HEAD")
    eligibility = metadata.get("evidence_eligibility") or {}
    if not allow_dirty_evidence and (
        eligibility.get("status") != "eligible_clean_head"
        or eligibility.get("allow_dirty_evidence") is not False
    ):
        raise BenchmarkError("reference artifact used a diagnostic evidence override")
    protocol = metadata.get("protocol") or {}
    reference_start_policy = protocol.get("start_policy")
    if reference_start_policy not in {"cold", "auto-warm"}:
        raise BenchmarkError("reference artifact has an invalid start policy")
    if (
        protocol.get("scc_charge_tolerance") != PUBLIC_BENCHMARK_SCC_CHARGE_TOLERANCE
        or protocol.get("scc_energy_tolerance") != PUBLIC_BENCHMARK_SCC_ENERGY_TOLERANCE
    ):
        raise BenchmarkError(
            "reference artifact must record the aligned public benchmark SCC contract"
        )
    convergence_contract = protocol.get("convergence_contract") or {}
    reference_contract = convergence_contract.get(reference_engine) or {}
    if reference_contract.get("public_accuracy_factor") != REFERENCE_ACCURACY:
        raise BenchmarkError(
            "reference artifact does not record the documented public accuracy factor"
        )
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for index, row in enumerate(document.get("rows") or []):
        if row.get("availability") != "available":
            continue
        if row.get("engine") != reference_engine:
            raise BenchmarkError(
                f"reference row {index} uses {row.get('engine')!r}, expected "
                f"{reference_engine!r}"
            )
        correctness = row.get("correctness") or {}
        cross_engine = correctness.get("cross_engine") or {}
        if (
            correctness.get("status") != "pass"
            or cross_engine.get("status") != "reference"
        ):
            raise BenchmarkError(f"reference row {index} is not qualified")
        if (
            row.get("start_policy") != reference_start_policy
            or row.get("effective_start_policy") != reference_start_policy
        ):
            raise BenchmarkError(
                f"reference row {index} does not match the artifact start policy"
            )
        natoms = row.get("natoms")
        batch_size = row.get("batch_size")
        if type(natoms) is not int or type(batch_size) is not int:
            raise BenchmarkError(f"reference row {index} has invalid coordinate")
        energies = row.get("energies_hartree")
        forces = decode_force_vector(row)
        if not isinstance(energies, list) or len(energies) != batch_size:
            raise BenchmarkError(f"reference row {index} has invalid energies")
        if not isinstance(forces, list) or len(forces) != 3 * natoms * batch_size:
            raise BenchmarkError(f"reference row {index} has invalid forces")
        if not all(math.isfinite(float(value)) for value in energies) or not all(
            math.isfinite(float(value)) for value in forces
        ):
            raise BenchmarkError(f"reference row {index} has non-finite observables")
        key = (natoms, batch_size)
        if key in rows:
            raise BenchmarkError(f"reference artifact duplicates coordinate {key}")
        normalized_row = dict(row)
        normalized_row["forces_hartree_per_bohr"] = forces
        rows[key] = normalized_row
    if not rows:
        raise BenchmarkError("reference artifact has no qualified available rows")
    return ReferenceArtifact(
        resolved,
        hashlib.sha256(payload).hexdigest(),
        reference_engine,
        metadata,
        rows,
    )


def apply_cross_engine_reference(
    row: dict[str, Any],
    reference: ReferenceArtifact,
    energy_atol_hartree: float,
    force_atol_hartree_per_bohr: float,
) -> bool:
    """Compare every timed result with the matching independent baseline row."""
    if row.get("availability") != "available":
        return False
    correctness = row.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("status") != "pass":
        return True
    key = (row["natoms"], row["batch_size"])
    expected = reference.rows.get(key)
    comparison: dict[str, Any] = {
        "status": "missing_reference",
        "reference_engine": reference.engine,
        "artifact": str(reference.path),
        "artifact_sha256": reference.sha256,
        "energy_atol_hartree": energy_atol_hartree,
        "force_atol_hartree_per_bohr": force_atol_hartree_per_bohr,
        "max_abs_energy_delta_hartree": None,
        "max_abs_force_delta_hartree_per_bohr": None,
    }
    failed = expected is None
    if expected is not None:
        try:
            raw_samples = row.get("raw_samples")
            energy_samples = (
                [sample["energies_hartree"] for sample in raw_samples]
                if isinstance(raw_samples, list) and raw_samples
                else [row["energies_hartree"]]
            )
            force_samples = row.get("_force_samples_hartree_per_bohr")
            if force_samples is None:
                force_samples = [row["forces_hartree_per_bohr"]]
            if not isinstance(force_samples, list) or len(force_samples) != len(
                energy_samples
            ):
                raise BenchmarkError(
                    "timed energy/force sample counts are inconsistent"
                )
            energy_deltas = [
                _max_abs_delta(sample, expected["energies_hartree"])
                for sample in energy_samples
            ]
            force_deltas = [
                _max_abs_delta(sample, expected["forces_hartree_per_bohr"])
                for sample in force_samples
            ]
            energy_delta = max(energy_deltas)
            force_delta = max(force_deltas)
        except (BenchmarkError, KeyError, TypeError) as exc:
            comparison["status"] = "fail"
            comparison["error"] = str(exc)
            failed = True
        else:
            comparison["max_abs_energy_delta_hartree"] = energy_delta
            comparison["max_abs_force_delta_hartree_per_bohr"] = force_delta
            comparison["timed_sample_count_checked"] = len(energy_samples)
            comparison["timed_sample_deltas"] = [
                {
                    "sample_index": index,
                    "max_abs_energy_delta_hartree": sample_energy_delta,
                    "max_abs_force_delta_hartree_per_bohr": sample_force_delta,
                    "status": (
                        "pass"
                        if sample_energy_delta <= energy_atol_hartree
                        and sample_force_delta <= force_atol_hartree_per_bohr
                        else "fail"
                    ),
                }
                for index, (sample_energy_delta, sample_force_delta) in enumerate(
                    zip(energy_deltas, force_deltas, strict=True)
                )
            ]
            failed = (
                energy_delta > energy_atol_hartree
                or force_delta > force_atol_hartree_per_bohr
            )
            comparison["status"] = "fail" if failed else "pass"
    correctness["cross_engine"] = comparison
    if failed:
        correctness["status"] = "fail"
    return failed


def run_cell(
    cell: Cell,
    library: Path,
    xtb_library: Path | None,
    tblite_library: Path | None,
    dxtb_source: Path | None,
    warmups: int,
    repetitions: int,
    start_policy: str = "auto-warm",
    repeatability_energy_atol_hartree: float = REPEATABILITY_ENERGY_ATOL_HARTREE,
    repeatability_force_atol_hartree_per_bohr: float = (
        REPEATABILITY_FORCE_ATOL_HARTREE_PER_BOHR
    ),
    scc_charge_tolerance: float = PUBLIC_BENCHMARK_SCC_CHARGE_TOLERANCE,
    scc_energy_tolerance: float = PUBLIC_BENCHMARK_SCC_ENERGY_TOLERANCE,
    scc_max_iterations: int = 500,
) -> dict[str, Any]:
    """Measure one cell and return a complete row."""
    base = base_row(cell)
    try:
        molecule = make_alkane(cell.natoms)
    except Exception as exc:  # noqa: BLE001 - input validation diagnostics
        row = error_row(cell, f"molecule builder failed: {exc}")
        row.update(base)
        return row
    storage = build_batch(
        molecule, cell.batch_size, seed=cell.natoms * 1000 + cell.batch_size
    )
    runner: Any = None
    try:
        if cell.engine in ("gpuxtb-cpu", "gpuxtb-cuda"):
            backend = "cpu" if cell.engine == "gpuxtb-cpu" else "cuda"
            runner = GpuxtbRunner(
                library,
                storage,
                backend,
                cell.cpu_threads,
                cell.device_id,
                scc_charge_tolerance=scc_charge_tolerance,
                scc_energy_tolerance=scc_energy_tolerance,
                scc_max_iterations=scc_max_iterations,
            )
        elif cell.engine == "xtb":
            if xtb_library is None:
                row = unavailable_row(cell, "no --xtb-library supplied")
                row.update(base)
                return row
            runner = ReferenceRunner(
                cell.engine,
                xtb_library,
                storage,
                cell.cpu_threads,
                cell.device_id,
                dxtb_source,
                scc_max_iterations,
            )
        elif cell.engine == "tblite":
            if tblite_library is None:
                row = unavailable_row(cell, "no --tblite-library supplied")
                row.update(base)
                return row
            runner = ReferenceRunner(
                cell.engine,
                tblite_library,
                storage,
                cell.cpu_threads,
                cell.device_id,
                dxtb_source,
                scc_max_iterations,
            )
        else:
            runner = ReferenceRunner(
                cell.engine,
                library,
                storage,
                cell.cpu_threads,
                cell.device_id,
                dxtb_source,
                scc_max_iterations,
            )
        fragment = measure_cell(
            runner,
            (warmups, repetitions),
            cell,
            start_policy,
            repeatability_energy_atol_hartree,
            repeatability_force_atol_hartree_per_bohr,
        )
        adapter = getattr(runner, "adapter", None)
        if adapter is not None:
            module_path_text = getattr(adapter, "module_path", None)
            module_path = Path(module_path_text) if module_path_text else None
            fragment["runtime_identity"] = {
                "api_version": getattr(adapter, "api_version", None),
                "version": getattr(adapter, "version", None),
                "torch_version": getattr(adapter, "torch_version", None),
                "module_path": module_path_text,
                "module_sha256": sha256_file(module_path),
                "convergence_settings": {
                    "public_accuracy_or_fixed_point_tolerance": getattr(
                        adapter, "accuracy", None
                    ),
                    "max_norm_tolerance": getattr(adapter, "max_norm_tolerance", None),
                    "function_tolerance": getattr(adapter, "function_tolerance", None),
                    "force_convergence": getattr(adapter, "force_convergence", None),
                    "max_iterations": getattr(adapter, "max_iterations", None),
                },
                "thread_control": getattr(adapter, "thread_control", None),
            }
        row = base_row(cell)
        row.update(fragment)
        row["availability"] = "available"
        return row
    except (BenchmarkError, XtbError, TbliteError, DxtbError, OSError) as exc:
        row = error_row(cell, str(exc))
        row.update(base)
        return row
    except Exception as exc:  # noqa: BLE001 - cell-level isolation
        row = error_row(cell, f"{type(exc).__name__}: {exc}")
        row.update(base)
        return row
    finally:
        if runner is not None:
            with contextlib.suppress(Exception):
                runner.close()


def run_trajectory(
    engine: str,
    natoms: int,
    frames: int,
    seed: int,
    library: Path,
    xtb_library: Path | None,
    tblite_library: Path | None,
    dxtb_source: Path | None,
    cpu_threads: int,
    device_id: int,
    warmups: int,
    repetitions: int,
    energy_atol_hartree: float,
    force_atol_hartree_per_bohr: float,
    scc_charge_tolerance: float = PUBLIC_BENCHMARK_SCC_CHARGE_TOLERANCE,
    scc_energy_tolerance: float = PUBLIC_BENCHMARK_SCC_ENERGY_TOLERANCE,
    scc_max_iterations: int = 500,
) -> dict[str, Any]:
    """Measure per-frame latency over one nearly identical MD-style trajectory."""
    base = {
        "engine": engine,
        "natoms": natoms,
        "batch_size": 1,
        "total_atoms_in_batch": natoms,
        "cpu_threads": cpu_threads,
        "device_id": device_id,
        "job": "trajectory",
        "frames": frames,
    }
    try:
        molecule = make_alkane(natoms)
        frames_list = build_trajectory(molecule, frames, seed)
        storage = build_batch(molecule, 1, seed)
        runner: Any = None
        try:
            if engine in ("gpuxtb-cpu", "gpuxtb-cuda"):
                backend = "cpu" if engine == "gpuxtb-cpu" else "cuda"
                runner = GpuxtbRunner(
                    library,
                    storage,
                    backend,
                    cpu_threads,
                    device_id,
                    scc_charge_tolerance=scc_charge_tolerance,
                    scc_energy_tolerance=scc_energy_tolerance,
                    scc_max_iterations=scc_max_iterations,
                )
            elif engine == "xtb":
                if xtb_library is None:
                    row = dict(base)
                    row.update(
                        {
                            "availability": "unavailable",
                            "reason": "no --xtb-library supplied",
                        }
                    )
                    return row
                runner = ReferenceRunner(
                    engine,
                    xtb_library,
                    storage,
                    cpu_threads,
                    device_id,
                    dxtb_source,
                    scc_max_iterations,
                )
            elif engine == "tblite":
                if tblite_library is None:
                    row = dict(base)
                    row.update(
                        {
                            "availability": "unavailable",
                            "reason": "no --tblite-library supplied",
                        }
                    )
                    return row
                runner = ReferenceRunner(
                    engine,
                    tblite_library,
                    storage,
                    cpu_threads,
                    device_id,
                    dxtb_source,
                    scc_max_iterations,
                )
            else:
                runner = ReferenceRunner(
                    engine,
                    library,
                    storage,
                    cpu_threads,
                    device_id,
                    dxtb_source,
                    scc_max_iterations,
                )
            is_gpuxtb = engine in ("gpuxtb-cpu", "gpuxtb-cuda")
            if is_gpuxtb:
                # MD-style workflow: seed SCC once on frame zero, then continue
                # in WARM mode for every later (nearly identical) frame.
                runner.set_start_mode("fresh")
                runner.set_positions(frames_list[0])
                runner.invoke()
            for _ in range(warmups):
                runner.invoke()
            sample_latencies: list[float] = []
            energies: list[float] = []
            force_digests: list[str] = []
            final_forces: list[float] | None = None
            frame_count = 0
            for _ in range(repetitions):
                for frame in frames_list:
                    runner.set_positions(frame)
                    if is_gpuxtb:
                        runner.set_start_mode("warm")
                    start = time.perf_counter_ns()
                    runner.invoke()
                    sample_latencies.append((time.perf_counter_ns() - start) * 1.0e-6)
                    snapshot = runner.snapshot()
                    snapshot_energies = snapshot.get("energies_hartree")
                    snapshot_forces = snapshot.get("forces_hartree_per_bohr")
                    if (
                        not isinstance(snapshot_energies, list)
                        or len(snapshot_energies) != 1
                    ):
                        raise BenchmarkError("trajectory frame returned invalid energy")
                    if (
                        not isinstance(snapshot_forces, list)
                        or len(snapshot_forces) != 3 * natoms
                    ):
                        raise BenchmarkError("trajectory frame returned invalid forces")
                    if not all(
                        math.isfinite(float(value))
                        for value in (*snapshot_energies, *snapshot_forces)
                    ):
                        raise BenchmarkError(
                            "trajectory frame returned non-finite observables"
                        )
                    energies.extend(float(value) for value in snapshot_energies)
                    final_forces = [float(value) for value in snapshot_forces]
                    force_digests.append(_force_digest(final_forces))
                    frame_count += 1
                    if frame_count % 8 == 0:
                        log(
                            f"trajectory {engine}: {frame_count}/"
                            f"{frames * repetitions} frames"
                        )
            row = dict(base)
            row.update(
                {
                    "availability": "available",
                    "timing": timing_summary(sample_latencies, 1),
                    "per_frame_samples_ms": sample_latencies,
                    "energies_hartree": [energies[-1]],
                    "forces_hartree_per_bohr": final_forces,
                    "energies_hartree_min": min(energies),
                    "energies_hartree_max": max(energies),
                    "force_sample_digests_binary64_le": force_digests,
                    "correctness": {
                        "status": "pass",
                        "finite_energies": True,
                        "finite_forces": True,
                        "force_value_count": 3 * natoms,
                        "energy_atol_hartree": energy_atol_hartree,
                        "force_atol_hartree_per_bohr": (force_atol_hartree_per_bohr),
                        "cross_engine": {"status": "not_requested"},
                    },
                }
            )
            return row
        finally:
            if runner is not None:
                with contextlib.suppress(Exception):
                    runner.close()
    except (BenchmarkError, XtbError, TbliteError, DxtbError, OSError) as exc:
        row = dict(base)
        row.update({"availability": "error", "error": str(exc)})
        return row
    except Exception as exc:  # noqa: BLE001 - cell-level isolation
        row = dict(base)
        row.update({"availability": "error", "error": f"{type(exc).__name__}: {exc}"})
        return row


def environment_metadata(
    args: argparse.Namespace, reference: ReferenceArtifact | None = None
) -> dict[str, Any]:
    """Capture the exact revisions, hardware, runtime, and thread environment."""
    repository_state = git_state(REPOSITORY_ROOT)
    dxtb_threads = args.dxtb_cpu_threads or args.cpu_threads
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "commit": repository_state,
        "evidence_eligibility": {
            "status": (
                "diagnostic_override"
                if args.allow_dirty_evidence
                else "eligible_clean_head"
            ),
            "allow_dirty_evidence": bool(args.allow_dirty_evidence),
        },
        "comparison_reference": {
            "designation": (
                "independent_baseline" if args.make_reference else "dependent_run"
            ),
            "engine": (
                args.engines[0]
                if args.make_reference
                else (reference.engine if reference is not None else None)
            ),
            "artifact": str(reference.path) if reference is not None else None,
            "artifact_sha256": reference.sha256 if reference is not None else None,
        },
        "runner": {
            "python": sys.version,
            "platform": platform.platform(),
            "gpuxtb_library": str(args.library.resolve()),
            "gpuxtb_library_sha256": sha256_file(args.library),
            "gpuxtb_native_identity": native_library_identity(
                str(args.library.resolve())
            ),
            "gpuxtb_build": cmake_build_identity(args.library),
            "xtb_library": str(args.xtb_library.resolve())
            if args.xtb_library
            else None,
            "xtb_library_sha256": sha256_file(args.xtb_library),
            "xtb_source": git_state(args.xtb_source) if args.xtb_source else None,
            "xtb_build": (
                meson_build_identity(args.xtb_library, args.xtb_source)
                if args.xtb_library and args.xtb_source
                else None
            ),
            "xtb_native_identity": (
                native_library_identity(str(args.xtb_library.resolve()))
                if args.xtb_library
                else None
            ),
            "tblite_library": str(args.tblite_library.resolve())
            if args.tblite_library
            else None,
            "tblite_library_sha256": sha256_file(args.tblite_library),
            "tblite_source": (
                git_state(args.tblite_source) if args.tblite_source else None
            ),
            "tblite_build": (
                meson_build_identity(args.tblite_library, args.tblite_source)
                if args.tblite_library and args.tblite_source
                else None
            ),
            "tblite_native_identity": (
                native_library_identity(str(args.tblite_library.resolve()))
                if args.tblite_library
                else None
            ),
            "dxtb_source": git_state(args.dxtb_source) if args.dxtb_source else None,
            "python_distributions": {
                name: installed_distribution_identity(name)
                for name in ("dxtb", "torch", "tad-libcint")
            },
            "reference_json": (
                str(args.reference_json.resolve()) if args.reference_json else None
            ),
            "reference_json_sha256": sha256_file(args.reference_json),
        },
        "hardware": {
            "hostname": platform.node(),
            "cpu_model": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "process_rss_bytes": current_rss_bytes(),
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "process_cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "nvidia_smi": run_text(("nvidia-smi", "-L")),
            "nvidia_smi_runtime": run_text(
                (
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version,memory.total,"
                    "pstate,power.limit,clocks.current.sm,clocks.current.memory",
                    "--format=csv,noheader",
                )
            ),
            "selected_cuda_device": selected_cuda_device_identity(args.device_id),
        },
        "threads": {
            "cpu_threads": args.cpu_threads,
            "dxtb_cpu_threads": dxtb_threads,
            "reference_threads": args.cpu_threads,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "MKL_INTERFACE_LAYER",
                "MKL_THREADING_LAYER",
            )
        },
        "protocol": {
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "start_policy": args.start_policy,
            "cross_engine_energy_atol_hartree": args.energy_atol,
            "cross_engine_force_atol_hartree_per_bohr": args.force_atol,
            "cross_engine_tolerance_source": {
                "scope": "owner_authorized_public_benchmark_output_compatibility",
                "not_primary_conformance": True,
                "selected_energy_atol_hartree": args.energy_atol,
                "selected_force_atol_hartree_per_bohr": args.force_atol,
                "maximum_authorized_energy_atol_hartree": (
                    PUBLIC_BENCHMARK_ENERGY_ATOL_HARTREE
                ),
                "maximum_authorized_force_atol_hartree_per_bohr": (
                    PUBLIC_BENCHMARK_FORCE_ATOL_HARTREE_PER_BOHR
                ),
                "strict_repository_gate_not_replaced": {
                    **CROSS_ENGINE_TOLERANCE_SOURCE,
                    "energy_atol_hartree": DEFAULT_CROSS_ENGINE_ENERGY_ATOL,
                    "force_atol_hartree_per_bohr": DEFAULT_CROSS_ENGINE_FORCE_ATOL,
                },
            },
            "repeatability_energy_atol_hartree": args.repeatability_energy_atol,
            "repeatability_force_atol_hartree_per_bohr": (
                args.repeatability_force_atol
            ),
            "perturb_sigma_bohr": PERTURB_SIGMA_BOHR,
            "trajectory_step_sigma_bohr": TRAJECTORY_STEP_SIGMA_BOHR,
            "scc_max_iterations": args.scc_max_iterations,
            "scc_charge_tolerance": args.scc_charge_tolerance,
            "scc_energy_tolerance": args.scc_energy_tolerance,
            "convergence_contract": {
                "gpuxtb": {
                    "charge_tolerance": args.scc_charge_tolerance,
                    "energy_tolerance": args.scc_energy_tolerance,
                },
                "xtb": {
                    "public_accuracy_factor": REFERENCE_ACCURACY,
                },
                "tblite": {
                    "public_accuracy_factor": REFERENCE_ACCURACY,
                },
                "dxtb": {
                    "x_atol": DXTB_FIXED_POINT_TOLERANCE,
                    "x_atol_max": DXTB_MAX_NORM_TOLERANCE,
                    "f_atol": DXTB_FUNCTION_TOLERANCE,
                    "force_convergence": DXTB_FORCE_CONVERGENCE,
                },
            },
        },
    }


def write_json(
    path: Path,
    document: Any,  # noqa: ANN401 - JSON-serializable benchmark document
) -> None:
    """Write compact JSON while retaining every final force value exactly.

    Decimal force arrays dominate large-batch evidence size. Encode each final
    vector as compressed little-endian binary64 with its count and SHA-256;
    :func:`load_reference_artifact` validates it before comparison.
    """
    encoded_document = dict(document)
    encoded_rows: list[dict[str, Any]] = []
    for row in document.get("rows", []):
        encoded_row = dict(row)
        forces = encoded_row.pop("forces_hartree_per_bohr", None)
        if forces is not None:
            encoded_row["forces_binary64_le_zlib_base64"] = encode_force_vector(forces)
        encoded_rows.append(encoded_row)
    encoded_document["rows"] = encoded_rows
    with path.open("w", encoding="utf-8") as handle:
        json.dump(encoded_document, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write one compact CSV row summary with LF line endings."""
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    columns = [
        "engine",
        "natoms",
        "batch_size",
        "total_atoms_in_batch",
        "cpu_threads",
        "device_id",
        "job",
        "availability",
        "median_ms",
        "mean_ms",
        "p95_ms",
        "min_ms",
        "systems_per_second_at_median",
        "correctness_status",
        "cross_engine_status",
        "max_abs_energy_delta_hartree",
        "max_abs_force_delta_hartree_per_bohr",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            timing = row.get("timing") or {}
            flat["median_ms"] = timing.get("median_ms")
            flat["mean_ms"] = timing.get("mean_ms")
            flat["p95_ms"] = timing.get("p95_ms")
            flat["min_ms"] = timing.get("min_ms")
            flat["systems_per_second_at_median"] = timing.get(
                "systems_per_second_at_median"
            )
            correctness = row.get("correctness") or {}
            cross_engine = correctness.get("cross_engine") or {}
            flat["correctness_status"] = correctness.get("status")
            flat["cross_engine_status"] = cross_engine.get("status")
            flat["max_abs_energy_delta_hartree"] = cross_engine.get(
                "max_abs_energy_delta_hartree"
            )
            flat["max_abs_force_delta_hartree_per_bohr"] = cross_engine.get(
                "max_abs_force_delta_hartree_per_bohr"
            )
            writer.writerow(flat)


def publish_artifacts(
    json_path: Path,
    csv_path: Path,
    document: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> None:
    """Publish a JSON/CSV pair atomically and refuse stale destinations."""
    if json_path.parent.resolve() != csv_path.parent.resolve():
        raise BenchmarkError("JSON and CSV outputs must share one directory")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (json_path, csv_path):
        if path.exists():
            raise BenchmarkError(f"refusing to overwrite existing artifact: {path}")
    staging: list[Path] = []
    published: list[Path] = []
    try:
        for final_path in (json_path, csv_path):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{final_path.name}.",
                suffix=".tmp",
                dir=final_path.parent,
            )
            os.close(descriptor)
            staging.append(Path(temporary_name))
        write_json(staging[0], document)
        write_csv(staging[1], rows)
        for temporary_path, final_path in zip(
            staging, (json_path, csv_path), strict=True
        ):
            os.replace(temporary_path, final_path)
            published.append(final_path)
    except BaseException:
        for path in staging:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        for path in published:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    """Configure the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Cross-engine GFN2-xTB atom-count scaling benchmark with distinct "
            "per-slot systems and an optional MD-trajectory mode."
        )
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--xtb-library", type=Path)
    parser.add_argument("--xtb-source", type=Path)
    parser.add_argument("--tblite-library", type=Path)
    parser.add_argument("--tblite-source", type=Path)
    parser.add_argument("--dxtb-source", type=Path)
    parser.add_argument(
        "--reference-json",
        type=Path,
        help=(
            "clean cold-start xTB/tblite artifact used to qualify every timed "
            "energy and force sample against the public benchmark output gate"
        ),
    )
    parser.add_argument(
        "--make-reference",
        action="store_true",
        help=(
            "designate this single-engine cold xTB/tblite run as the "
            "independent output reference"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--engines", type=parse_csv_values, default=DEFAULT_ENGINES)
    parser.add_argument("--natoms", type=parse_csv_ints, default=DEFAULT_NATOMS)
    parser.add_argument(
        "--natoms-large-batch",
        type=parse_csv_ints,
        help=(
            "molecule sizes used when batch_size > 1; defaults to --natoms. "
            "Use a smaller set for large batches so serial reference engines "
            "remain practical."
        ),
    )
    parser.add_argument(
        "--batch-sizes", type=parse_csv_ints, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument(
        "--start-policy",
        choices=("auto-warm", "cold"),
        default="auto-warm",
        help=(
            "SCC restart policy for the batch matrices. auto-warm (default): "
            "one untimed cold seed, then WARM measured samples. cold: clear "
            "electronic state before every timed inference call. gpuxtb FRESH "
            "initialization and dxtb reset are timed; xTB/tblite rebuild is "
            "excluded. --cold-samples is an alias."
        ),
    )
    parser.add_argument(
        "--cold-samples",
        dest="start_policy",
        action="store_const",
        const="cold",
        help="alias for --start-policy cold (every sample cold-start).",
    )
    parser.add_argument(
        "--trajectory-natoms",
        type=parse_csv_ints,
        default=(32, 62, 122, 242),
        help=(
            "molecule sizes measured in MD-trajectory mode; each size yields "
            "one per-frame latency row per engine so the figure can sweep "
            "atom count on its x-axis."
        ),
    )
    parser.add_argument("--trajectory-frames", type=int, default=20)
    parser.add_argument(
        "--energy-atol",
        type=float,
        default=CROSS_ENGINE_ENERGY_ATOL_HARTREE,
        help=(
            "public benchmark output-compatibility gate in Hartree; this does "
            "not replace gpuxtb conformance"
        ),
    )
    parser.add_argument(
        "--force-atol",
        type=float,
        default=CROSS_ENGINE_FORCE_ATOL_HARTREE_PER_BOHR,
        help=(
            "public benchmark output-compatibility gate in Hartree/bohr; this "
            "does not replace gpuxtb conformance"
        ),
    )
    parser.add_argument(
        "--repeatability-energy-atol",
        type=float,
        default=REPEATABILITY_ENERGY_ATOL_HARTREE,
        help="within-engine identical-input energy repeatability gate",
    )
    parser.add_argument(
        "--repeatability-force-atol",
        type=float,
        default=REPEATABILITY_FORCE_ATOL_HARTREE_PER_BOHR,
        help="within-engine identical-input force repeatability gate",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--dxtb-cpu-threads",
        type=int,
        default=None,
        help="optional dxtb override; defaults to --cpu-threads",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--allow-dirty-evidence",
        action="store_true",
        help="diagnostic only: permit a dirty repository and mark it in metadata",
    )
    parser.add_argument(
        "--scc-charge-tolerance",
        type=float,
        default=PUBLIC_BENCHMARK_SCC_CHARGE_TOLERANCE,
        help="gpuxtb SCC charge convergence tolerance (publication: 1e-4)",
    )
    parser.add_argument(
        "--scc-energy-tolerance",
        type=float,
        default=PUBLIC_BENCHMARK_SCC_ENERGY_TOLERANCE,
        help="gpuxtb SCC energy convergence tolerance (publication: 1e-6)",
    )
    parser.add_argument(
        "--scc-max-iterations",
        type=int,
        default=500,
        help="gpuxtb SCC iteration cap",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject unsupported engine sets and nonpositive controls."""
    for engine in args.engines:
        if engine not in SUPPORTED_ENGINES:
            raise BenchmarkError(f"unsupported engine: {engine}")
    if args.make_reference:
        if args.reference_json is not None:
            raise BenchmarkError("--make-reference cannot use --reference-json")
        if len(args.engines) != 1 or args.engines[0] not in {"xtb", "tblite"}:
            raise BenchmarkError(
                "--make-reference requires exactly one xTB or tblite engine"
            )
    elif args.reference_json is None and not args.allow_dirty_evidence:
        raise BenchmarkError(
            "publication evidence requires --reference-json or --make-reference"
        )
    if not args.allow_dirty_evidence and (
        args.scc_charge_tolerance != PUBLIC_BENCHMARK_SCC_CHARGE_TOLERANCE
        or args.scc_energy_tolerance != PUBLIC_BENCHMARK_SCC_ENERGY_TOLERANCE
    ):
        raise BenchmarkError(
            "publication evidence requires the aligned gpuxtb 1e-4 charge / "
            "1e-6 energy convergence contract; custom settings are diagnostic only"
        )
    if args.warmups < 0 or args.repetitions <= 0:
        raise BenchmarkError("warmups must be >= 0 and repetitions must be > 0")
    if args.cpu_threads <= 0 or (
        args.dxtb_cpu_threads is not None and args.dxtb_cpu_threads <= 0
    ):
        raise BenchmarkError("thread counts must be positive")
    if (
        not args.allow_dirty_evidence
        and args.dxtb_cpu_threads is not None
        and args.dxtb_cpu_threads != args.cpu_threads
    ):
        raise BenchmarkError(
            "publication evidence requires one equal CPU thread budget for every engine"
        )
    for name, value in (
        ("cross-engine energy tolerance", args.energy_atol),
        ("cross-engine force tolerance", args.force_atol),
        ("repeatability energy tolerance", args.repeatability_energy_atol),
        ("repeatability force tolerance", args.repeatability_force_atol),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise BenchmarkError(f"{name} must be finite and nonnegative")
    for name, value in (
        ("gpuxtb SCC charge tolerance", args.scc_charge_tolerance),
        ("gpuxtb SCC energy tolerance", args.scc_energy_tolerance),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise BenchmarkError(f"{name} must be finite and positive")
    if args.scc_max_iterations <= 0:
        raise BenchmarkError("SCC max iterations must be positive")
    if args.energy_atol > CROSS_ENGINE_ENERGY_ATOL_HARTREE:
        raise BenchmarkError(
            "--energy-atol cannot exceed the owner-authorized public benchmark gate"
        )
    if args.force_atol > CROSS_ENGINE_FORCE_ATOL_HARTREE_PER_BOHR:
        raise BenchmarkError(
            "--force-atol cannot exceed the owner-authorized public benchmark gate"
        )
    if args.device_id < 0:
        raise BenchmarkError("device id must be nonnegative")
    if args.trajectory_frames <= 0:
        raise BenchmarkError("trajectory frames must be positive")
    for natoms in args.trajectory_natoms:
        try:
            make_alkane(natoms)
        except Exception as exc:  # noqa: PERF203 - per-size validation before timing
            raise BenchmarkError(
                f"unsupported trajectory natoms {natoms}: {exc}"
            ) from exc
    if args.output_json == args.output_csv:
        raise BenchmarkError("JSON and CSV output paths must be distinct")
    if args.output_json.parent.resolve() != args.output_csv.parent.resolve():
        raise BenchmarkError("JSON and CSV outputs must share one directory")
    for path in (args.output_json, args.output_csv):
        if path.exists():
            raise BenchmarkError(f"refusing to overwrite existing artifact: {path}")
    for natoms in args.natoms:
        try:
            make_alkane(natoms)
        except Exception as exc:  # noqa: PERF203 - per-size validation before timing
            raise BenchmarkError(f"unsupported natoms {natoms}: {exc}") from exc


def validate_publication_provenance(
    args: argparse.Namespace, repository_state: dict[str, Any]
) -> None:
    """Reject final evidence whose source, build, or runtime bytes are unbound."""
    build = cmake_build_identity(args.library)
    if build is None:
        raise BenchmarkError(
            "publication evidence requires a gpuxtb library beside a CMakeCache.txt"
        )
    source_state = build.get("source_state") or {}
    if (
        source_state.get("head") != repository_state.get("head")
        or source_state.get("dirty") is not False
    ):
        raise BenchmarkError(
            "gpuxtb build source does not match the current clean benchmark HEAD"
        )
    if not build.get("cxx_compiler_sha256"):
        raise BenchmarkError("gpuxtb build compiler bytes could not be identified")
    if "gpuxtb-cpu" in args.engines and not build.get("cpu_linalg_provider"):
        raise BenchmarkError(
            "gpuxtb CPU evidence requires a hash-pinned LP64 provider identity"
        )

    native_inputs = [("gpuxtb", args.library)]
    source_inputs: list[tuple[str, Path | None]] = []
    external_builds: list[tuple[str, Path | None, Path | None]] = []
    if "xtb" in args.engines:
        native_inputs.append(("xTB", args.xtb_library))
        source_inputs.append(("xTB", args.xtb_source))
        external_builds.append(("xTB", args.xtb_library, args.xtb_source))
    if "tblite" in args.engines:
        native_inputs.append(("tblite", args.tblite_library))
        source_inputs.append(("tblite", args.tblite_source))
        external_builds.append(("tblite", args.tblite_library, args.tblite_source))
    if any(engine.startswith("dxtb") for engine in args.engines):
        source_inputs.append(("dxtb", args.dxtb_source))

    for label, source in source_inputs:
        if source is None:
            raise BenchmarkError(f"{label} evidence requires its clean source checkout")
        state = git_state(source)
        if not state.get("head") or state.get("dirty") is not False:
            raise BenchmarkError(
                f"{label} source checkout is not a verified clean HEAD"
            )

    for label, library, source in external_builds:
        if library is None or source is None:
            raise BenchmarkError(f"{label} build identity is incomplete")
        synchronize_meson_target(library, source)
        build_identity = meson_build_identity(library, source)
        if build_identity is None:
            raise BenchmarkError(
                f"{label} evidence requires a Meson target bound to its clean source"
            )
        compilers = build_identity.get("compilers") or []
        if not compilers or any(
            not item.get("executable_files")
            or item.get("unresolved_configure_time_entries")
            for item in compilers
        ):
            raise BenchmarkError(
                f"{label} build compiler bytes could not be identified"
            )
        sync = build_identity.get("source_target_sync") or {}
        if sync.get("library_sha256") != sha256_file(library.resolve()):
            raise BenchmarkError(
                f"{label} source-target synchronization is inconsistent"
            )

    for label, path in native_inputs:
        if path is None or not path.is_file():
            raise BenchmarkError(
                f"{label} evidence requires an existing shared library"
            )
        identity = native_library_identity(str(path.resolve()))
        if identity is None:
            raise BenchmarkError(f"{label} shared-library identity is unavailable")
        if identity.get("unresolved_dependencies"):
            raise BenchmarkError(f"{label} shared library has unresolved dependencies")
    if any(engine.endswith("cuda") for engine in args.engines):
        selected_cuda = selected_cuda_device_identity(args.device_id) or {}
        selected_device = selected_cuda.get("device") or {}
        if (
            not selected_device.get("uuid")
            or not selected_cuda.get("runtime_uuid")
            or selected_device.get("uuid") != selected_cuda.get("runtime_uuid")
        ):
            raise BenchmarkError(
                "CUDA publication evidence requires a runtime-verified "
                "selected GPU UUID"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested matrix and leave complete artifacts on any failure path."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    failed = False
    rows: list[dict[str, Any]] = []
    try:
        validate_arguments(args)
        repository_state = git_state(REPOSITORY_ROOT)
        if not repository_state.get("head"):
            raise BenchmarkError("cannot verify the benchmark repository revision")
        if repository_state.get("dirty") and not args.allow_dirty_evidence:
            raise BenchmarkError(
                "publication evidence requires a clean repository; use "
                "--allow-dirty-evidence only for diagnostics"
            )
        if not args.allow_dirty_evidence:
            validate_publication_provenance(args, repository_state)
        if args.dxtb_source is not None:
            dxtb_source_state = git_state(args.dxtb_source)
            if not dxtb_source_state.get("head"):
                raise BenchmarkError("--dxtb-source must be a versioned Git checkout")
            if dxtb_source_state.get("dirty") and not args.allow_dirty_evidence:
                raise BenchmarkError(
                    "publication evidence requires a clean dxtb source checkout"
                )
        if any(engine.startswith("dxtb") for engine in args.engines):
            for distribution_name in ("dxtb", "torch", "tad-libcint"):
                identity = installed_distribution_identity(distribution_name)
                verification = (
                    identity.get("payload_verification") if identity else None
                )
                if not identity or not isinstance(verification, dict):
                    raise BenchmarkError(
                        f"dxtb evidence requires installed {distribution_name} identity"
                    )
                if verification.get("status") != "verified":
                    raise BenchmarkError(
                        f"installed {distribution_name} payload failed RECORD "
                        "verification"
                    )
            dxtb_identity = installed_distribution_identity("dxtb") or {}
            direct_url = dxtb_identity.get("direct_url_identity") or {}
            if (
                args.dxtb_source is None
                or direct_url.get("scheme") != "file"
                or Path(str(direct_url.get("local_source_path", ""))).resolve()
                != args.dxtb_source.resolve()
            ):
                raise BenchmarkError(
                    "installed dxtb payload is not bound to --dxtb-source"
                )
        reference = (
            load_reference_artifact(args.reference_json, args.allow_dirty_evidence)
            if args.reference_json is not None
            else None
        )
        if reference is not None:
            reference_commit = (reference.metadata.get("commit") or {}).get("head")
            if reference_commit != repository_state["head"]:
                raise BenchmarkError(
                    "reference artifact and current runner must use the same clean HEAD"
                )
            reference_protocol = reference.metadata.get("protocol") or {}
            for name, expected in (
                ("warmups", args.warmups),
                ("repetitions", args.repetitions),
                ("start_policy", args.start_policy),
                ("cross_engine_energy_atol_hartree", args.energy_atol),
                ("cross_engine_force_atol_hartree_per_bohr", args.force_atol),
                (
                    "repeatability_energy_atol_hartree",
                    args.repeatability_energy_atol,
                ),
                (
                    "repeatability_force_atol_hartree_per_bohr",
                    args.repeatability_force_atol,
                ),
                ("scc_max_iterations", args.scc_max_iterations),
                ("scc_charge_tolerance", args.scc_charge_tolerance),
                ("scc_energy_tolerance", args.scc_energy_tolerance),
            ):
                if reference_protocol.get(name) != expected:
                    raise BenchmarkError(
                        f"reference protocol {name}={reference_protocol.get(name)!r} "
                        f"does not match requested {expected!r}"
                    )
            expected_contract = {
                "gpuxtb": {
                    "charge_tolerance": args.scc_charge_tolerance,
                    "energy_tolerance": args.scc_energy_tolerance,
                },
                "xtb": {"public_accuracy_factor": REFERENCE_ACCURACY},
                "tblite": {"public_accuracy_factor": REFERENCE_ACCURACY},
                "dxtb": {
                    "x_atol": DXTB_FIXED_POINT_TOLERANCE,
                    "x_atol_max": DXTB_MAX_NORM_TOLERANCE,
                    "f_atol": DXTB_FUNCTION_TOLERANCE,
                    "force_convergence": DXTB_FORCE_CONVERGENCE,
                },
            }
            if reference_protocol.get("convergence_contract") != expected_contract:
                raise BenchmarkError(
                    "reference artifact uses a different native convergence contract"
                )
            reference_threads = reference.metadata.get("threads") or {}
            if reference_threads.get("reference_threads") != args.cpu_threads:
                raise BenchmarkError(
                    "reference artifact uses a different thread budget"
                )
        library = args.library.resolve()
        natoms_large_batch = args.natoms_large_batch or args.natoms
        dxtb_threads = args.dxtb_cpu_threads or args.cpu_threads
        cells = [
            Cell(
                engine,
                natoms,
                batch_size,
                dxtb_threads if engine.startswith("dxtb") else args.cpu_threads,
                args.device_id,
            )
            for engine in args.engines
            for batch_size in args.batch_sizes
            for natoms in (args.natoms if batch_size == 1 else natoms_large_batch)
        ]
        total_cells = len(cells)
        log(
            f"matrix: {total_cells} cells from engines={args.engines} "
            f"batch={args.batch_sizes} natoms={args.natoms} "
            f"natoms_large_batch={natoms_large_batch}"
        )
        for cell_index, cell in enumerate(cells, start=1):
            log_start = time.perf_counter()
            row = run_cell(
                cell,
                library,
                args.xtb_library,
                args.tblite_library,
                args.dxtb_source,
                args.warmups,
                args.repetitions,
                start_policy=args.start_policy,
                repeatability_energy_atol_hartree=args.repeatability_energy_atol,
                repeatability_force_atol_hartree_per_bohr=(
                    args.repeatability_force_atol
                ),
                scc_charge_tolerance=args.scc_charge_tolerance,
                scc_energy_tolerance=args.scc_energy_tolerance,
                scc_max_iterations=args.scc_max_iterations,
            )
            # Record the SCC start policy on the cell row itself.  A
            # trajectory-invoked matrix cell (for example a batch=1 sample
            # measured with ``--start-policy auto-warm``) is then
            # unambiguously distinguishable from a genuine cold-start row by
            # downstream consumers such as the plotter.
            row["start_policy"] = args.start_policy
            if row.get("availability") == "available":
                if (row.get("correctness") or {}).get("status") != "pass":
                    failed = True
                if args.make_reference:
                    row["correctness"]["cross_engine"] = {
                        "status": "reference",
                        "role": "independent_panel_matched_output_baseline",
                        "engine": args.engines[0],
                    }
                elif reference is not None:
                    failed = (
                        apply_cross_engine_reference(
                            row,
                            reference,
                            args.energy_atol,
                            args.force_atol,
                        )
                        or failed
                    )
                # Complete per-repetition forces are required only while the
                # cross-engine gate is evaluated.  Do not inflate archived
                # JSON after each sample digest and the final vector are kept.
                row.pop("_force_samples_hartree_per_bohr", None)
            elapsed_s = time.perf_counter() - log_start
            rows.append(row)
            if row["availability"] == "available":
                log(
                    f"[{cell_index}/{total_cells}] {row['engine']} "
                    f"natoms={row['natoms']} batch={row['batch_size']} "
                    f"median={row['timing']['median_ms']:.6f} ms "
                    f"({elapsed_s:.1f} s)"
                )
            else:
                log(
                    f"[{cell_index}/{total_cells}] {row['engine']} "
                    f"natoms={row['natoms']} batch={row['batch_size']} "
                    f"UNAVAILABLE: {row.get('reason', row.get('error', 'unknown'))}"
                )
                failed = True
        if args.trajectory:
            for engine in args.engines:
                for trajectory_natoms in args.trajectory_natoms:
                    log(
                        f"trajectory engine={engine} natoms={trajectory_natoms} "
                        f"frames={args.trajectory_frames}"
                    )
                    row = run_trajectory(
                        engine,
                        trajectory_natoms,
                        args.trajectory_frames,
                        seed=trajectory_natoms * 7 + 42,
                        library=library,
                        xtb_library=args.xtb_library,
                        tblite_library=args.tblite_library,
                        dxtb_source=args.dxtb_source,
                        cpu_threads=args.cpu_threads,
                        device_id=args.device_id,
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                        energy_atol_hartree=args.energy_atol,
                        force_atol_hartree_per_bohr=args.force_atol,
                        scc_charge_tolerance=args.scc_charge_tolerance,
                        scc_energy_tolerance=args.scc_energy_tolerance,
                        scc_max_iterations=args.scc_max_iterations,
                    )
                    rows.append(row)
                    if row["availability"] == "available":
                        log(
                            f"trajectory {row['engine']} natoms={row['natoms']} "
                            f"median={row['timing']['median_ms']:.6f} ms/frame"
                        )
                    else:
                        log(
                            f"trajectory {row['engine']} UNAVAILABLE: "
                            f"{row.get('reason', row.get('error', 'unknown'))}"
                        )
                        failed = True
        document = {
            "schema_version": SCHEMA_VERSION,
            "metadata": environment_metadata(args, reference),
            "rows": rows,
        }
        publish_artifacts(args.output_json, args.output_csv, document, rows)
    except (BenchmarkError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 2
    print(f"wrote {args.output_json} and {args.output_csv}")  # noqa: T201
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
