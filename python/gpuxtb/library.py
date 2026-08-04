"""Low-level :mod:`ctypes` binding to the gpuxtb public C ABI.

This module mirrors ``tools/conformance/gpuxtb_public_api.py`` and keeps the
package self-contained inside a wheel. It knows nothing about molecules or
batches; it owns the C structure mirrors, shared-library discovery, symbol
signatures, and the thin checked wrappers used by :mod:`gpuxtb.interface`.

Units follow the C ABI: positions in bohr, energies in Hartree, forces in
Hartree/bohr, charges and external point-charge values in elementary-charge
units. Electronic temperatures in the ``gpuxtb_compute_options_t`` are
``k_B * T`` in Hartree.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .exceptions import GPUxtbRuntimeError

# --- ABI constants (kept in sync with include/gpuxtb/gpuxtb.h) ----------------

API_VERSION = 1

STATUS_SUCCESS = 0
STATUS_INVALID_ARGUMENT = 1
STATUS_BACKEND_UNAVAILABLE = 2
STATUS_NOT_SUPPORTED = 3
STATUS_ALLOCATION_FAILED = 4
STATUS_NOT_IMPLEMENTED = 5
STATUS_INTERNAL_ERROR = 6
STATUS_SCC_NOT_CONVERGED = 7
STATUS_EIGENSOLVER_FAILED = 8

BACKEND_AUTO = 0
BACKEND_CPU = 1
BACKEND_CUDA = 2
BACKEND_ROCM = 3

MEMORY_HOST = 0
MEMORY_CUDA_DEVICE = 1
MEMORY_ROCM_DEVICE = 2

MODEL_GFN1_XTB = 1
MODEL_GFN2_XTB = 2

SCC_START_FRESH = 1
SCC_START_WARM = 2

COMPUTE_ENERGY = 1 << 0
COMPUTE_FORCES = 1 << 1
COMPUTE_ATOMIC_CHARGES = 1 << 2
COMPUTE_POINT_CHARGE_FORCES = 1 << 3

KELVIN_TO_HARTREE = 3.166808578545117e-6
DEFAULT_ELECTRONIC_TEMPERATURE = 300.0 * KELVIN_TO_HARTREE

# --- ctypes mirrors of the public ABI structures -----------------------------


class ContextOptions(ctypes.Structure):
    """ctypes mirror of ``gpuxtb_context_options_t`` ABI version 1."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("backend", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        ("cpu_threads", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("stream", ctypes.c_void_p),
    ]


class ConstBuffer(ctypes.Structure):
    """ctypes mirror of ``gpuxtb_const_buffer_t`` (caller-owned input view)."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("size_bytes", ctypes.c_size_t),
        ("memory_space", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class Buffer(ctypes.Structure):
    """ctypes mirror of ``gpuxtb_buffer_t`` (caller-owned output view)."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("size_bytes", ctypes.c_size_t),
        ("memory_space", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class Batch(ctypes.Structure):
    """ctypes mirror of ``gpuxtb_batch_t`` including the ABI-v2 spin suffix."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("batch_size", ctypes.c_int64),
        ("total_atoms", ctypes.c_int64),
        ("total_point_charges", ctypes.c_int64),
        ("total_charge_response_elements", ctypes.c_int64),
        ("atom_offsets", ConstBuffer),
        ("atomic_numbers", ConstBuffer),
        ("positions", ConstBuffer),
        ("molecular_charges", ConstBuffer),
        ("unpaired_electrons", ConstBuffer),
        ("point_charge_offsets", ConstBuffer),
        ("point_charge_positions", ConstBuffer),
        ("point_charge_values", ConstBuffer),
        ("point_charge_gammas", ConstBuffer),
        ("atomic_potential_shifts", ConstBuffer),
        ("charge_response_offsets", ConstBuffer),
        ("charge_response_matrix", ConstBuffer),
        ("spin_channels", ConstBuffer),
    ]


class ComputeOptions(ctypes.Structure):
    """ctypes mirror of ``gpuxtb_compute_options_t`` through ABI version 2."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("model", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("max_scc_iterations", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("charge_tolerance", ctypes.c_double),
        ("energy_tolerance", ctypes.c_double),
        ("electronic_temperature", ctypes.c_double),
        ("scc_start_mode", ctypes.c_int32),
        ("reserved_v2", ctypes.c_uint32),
    ]


class BatchResult(ctypes.Structure):
    """ctypes mirror of ``gpuxtb_batch_result_t`` ABI version 1."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("energies", Buffer),
        ("forces", Buffer),
        ("atomic_charges", Buffer),
        ("point_charge_forces", Buffer),
        ("scc_iterations", Buffer),
        ("scc_converged", Buffer),
        ("per_system_status", Buffer),
    ]


# --- Shared-library discovery -------------------------------------------------


def _installed_package_library() -> Optional[Path]:
    """Locate ``libgpuxtb`` installed next to the Python package inside a wheel.

    With ``wheel.install-dir = "gpuxtb"`` the CMake install step drops the
    versioned shared object plus symlinks into ``gpuxtb/lib``. We glob so a
    wheel that flattened the SONAME symlink still resolves, and we accept the
    platform-specific file naming (ELF ``libgpuxtb.so*``, macOS
    ``libgpuxtb.dylib``, Windows ``gpuxtb.dll``).
    """
    package_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    # CMake installs runtime DLLs into ``bin`` on Windows and shared libraries
    # into ``lib`` on POSIX platforms.  Search both locations so the binding
    # follows the platform's standard install layout.
    for runtime_dir in (
        package_dir / "lib",
        package_dir / "lib64",
        package_dir / "bin",
    ):
        candidates.extend(sorted(runtime_dir.glob("libgpuxtb.so*")))
        candidates.extend(sorted(runtime_dir.glob("libgpuxtb.dylib*")))
        candidates.extend(sorted(runtime_dir.glob("gpuxtb.dll")))
    return candidates[0] if candidates else None


def library_path() -> Union[str, Path]:
    """Return the path to the gpuxtb shared library.

    Resolution order:

    1. ``GPUXTB_LIBRARY`` environment variable (explicit override).
    2. The ``lib`` directory shipped inside the installed package/wheel.
    3. A ``ctypes`` platform lookup of ``gpuxtb`` (system installation).

    Raises
    ------
    GPUxtbRuntimeError
        when no shared library can be found.
    """
    explicit = os.environ.get("GPUXTB_LIBRARY")
    candidates: list[Union[str, Path]] = []
    if explicit:
        candidates.append(explicit)
    bundled = _installed_package_library()
    if bundled is not None:
        candidates.append(bundled)
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path.resolve()

    # ``find_library`` normally returns a loader name such as
    # ``libgpuxtb.so.0``, not an absolute filesystem path.  Passing that name
    # directly to ctypes preserves the documented system-install fallback.
    system_library = ctypes.util.find_library("gpuxtb")
    if system_library:
        return system_library

    raise GPUxtbRuntimeError(
        "cannot locate the gpuxtb shared library; set GPUXTB_LIBRARY or build "
        "the package with scikit-build-core so libgpuxtb is bundled"
    )


def _configure_library(library: ctypes.CDLL) -> None:
    """Declare every C symbol signature the Python package calls."""
    library.gpuxtb_get_last_error.argtypes = []
    library.gpuxtb_get_last_error.restype = ctypes.c_char_p
    library.gpuxtb_version_string.argtypes = []
    library.gpuxtb_version_string.restype = ctypes.c_char_p
    library.gpuxtb_status_string.argtypes = [ctypes.c_int32]
    library.gpuxtb_status_string.restype = ctypes.c_char_p
    library.gpuxtb_context_options_init.argtypes = [
        ctypes.POINTER(ContextOptions),
        ctypes.c_size_t,
    ]
    library.gpuxtb_context_options_init.restype = ctypes.c_int32
    library.gpuxtb_batch_init.argtypes = [ctypes.POINTER(Batch), ctypes.c_size_t]
    library.gpuxtb_batch_init.restype = ctypes.c_int32
    library.gpuxtb_compute_options_init.argtypes = [
        ctypes.POINTER(ComputeOptions),
        ctypes.c_size_t,
    ]
    library.gpuxtb_compute_options_init.restype = ctypes.c_int32
    library.gpuxtb_batch_result_init.argtypes = [
        ctypes.POINTER(BatchResult),
        ctypes.c_size_t,
    ]
    library.gpuxtb_batch_result_init.restype = ctypes.c_int32
    library.gpuxtb_context_create.argtypes = [
        ctypes.POINTER(ContextOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.gpuxtb_context_create.restype = ctypes.c_int32
    library.gpuxtb_context_destroy.argtypes = [ctypes.c_void_p]
    library.gpuxtb_context_destroy.restype = None
    library.gpuxtb_context_get_backend.argtypes = [ctypes.c_void_p]
    library.gpuxtb_context_get_backend.restype = ctypes.c_int32
    library.gpuxtb_context_get_device_id.argtypes = [ctypes.c_void_p]
    library.gpuxtb_context_get_device_id.restype = ctypes.c_int32
    library.gpuxtb_compute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Batch),
        ctypes.POINTER(ComputeOptions),
        ctypes.POINTER(BatchResult),
    ]
    library.gpuxtb_compute.restype = ctypes.c_int32


_lib: Optional[ctypes.CDLL] = None
_dll_directory_handles: list = []


def _runtime_search_dirs() -> list[Path]:
    """Return directories that may contain the runtimes libgpuxtb depends on.

    libgpuxtb links the CUDA math/runtime libraries (cuBLAS, cuSOLVER, ...)
    when built with CUDA, and the CPU eigensolver dlopens the LP64 BLAS
    runtime (MKL libmkl_rt on x86_64, OpenBLAS on aarch64) at compute time.
    Those live either in the ``mkl``/``nvidia-*`` PyPI packages, an installed
    CUDA toolkit, or the ``scipy-openblas*`` PyPI wheels (the LP64
    ``scipy-openblas32`` is what numpy uses on aarch64). Collecting them lets
    the package resolve everything on import instead of forcing users to set
    ``LD_LIBRARY_PATH`` (or ``PATH`` on Windows).
    """
    dirs: list[Path] = []

    try:
        import site
    except Exception:  # pragma: no cover - defensive
        site = None
    site_packages = (
        getattr(site, "getsitepackages", lambda: [])() if site is not None else []
    )
    user_site = (
        getattr(site, "getusersitepackages", lambda: None)()
        if site is not None
        else None
    )
    if user_site and (
        bool(getattr(site, "ENABLE_USER_SITE", False)) or str(user_site) in sys.path
    ):
        site_packages = [*site_packages, user_site]

    for sp in dict.fromkeys(site_packages):
        base = Path(sp)
        mkl_lib = base / "mkl" / "lib"
        if mkl_lib.is_dir():
            dirs.append(mkl_lib)
        # The PyPI `mkl` package installs its libraries into the interpreter
        # prefix root's lib/ (site-packages/../../libmkl_rt.so.3 in a venv or
        # conda env), so add that location as well.
        for prefix_lib in (base.parent.parent, base.parent.parent.parent):
            if prefix_lib.name == "lib" and prefix_lib.is_dir():
                dirs.append(prefix_lib)
        # scipy-openblas32 (LP64) / scipy-openblas64 (ILP64) install their
        # prefixed runtime under <site-packages>/scipy_openblas{32,64}/lib.
        for openblas_style in ("scipy_openblas32", "scipy_openblas64"):
            openblas_lib = base / openblas_style / "lib"
            if openblas_lib.is_dir():
                dirs.append(openblas_lib)
        nvidia_root = base / "nvidia"
        if nvidia_root.is_dir():
            dirs.extend(
                entry / "lib"
                for entry in nvidia_root.iterdir()
                if (entry / "lib").is_dir()
            )

    for candidate in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
    ):
        path = Path(candidate)
        if path.is_dir():
            dirs.append(path)

    return dirs


# Exact dependency groups in load order.  Prefix-scanning every NVIDIA package
# can load unused CUDA stacks (and even conflicting major versions), inflating
# startup time and RSS.  ``libcuda`` is deliberately absent: the NVIDIA kernel
# driver must always come from the system loader, never a toolkit stub.
_RUNTIME_LIBRARY_GROUPS = (
    ("libnvJitLink.so.12",),
    ("libcudart.so.12",),
    ("libcublasLt.so.12",),
    ("libcublas.so.12",),
    ("libcusparse.so.12",),
    ("libcusolver.so.11",),
    # MKL changes its SONAME between releases; load exactly one runtime.
    ("libmkl_rt.so.4", "libmkl_rt.so.3", "libmkl_rt.so.2", "libmkl_rt.so"),
    # OpenBLAS is the LP64 BLAS used on platforms without Intel MKL builds
    # (the scipy-openblas32 wheel ships libscipy_openblas.so with scipy_-prefixed
    # symbols). Preload at most one instance by SONAME so the eigensolver's
    # by-name dlopen reuses it instead of loading a second, conflicting BLAS.
    (
        "libscipy_openblas.so",
        "libscipy_openblas32_.so",
        "libopenblas.so.0",
        "libopenblas.so",
        "libopenblas.so.3",
    ),
)


def _preload_runtime_libraries() -> list[str]:
    """Preload the runtimes libgpuxtb needs so users need no environment setup.

    On POSIX, loading the dependency by absolute path registers it under its
    SONAME, so a later ``dlopen`` by name (the CPU eigensolver's MKL lookup)
    and the DT_NEEDED resolution of a CUDA-enabled ``libgpuxtb`` reuse the
    already-loaded object instead of failing. On Windows the discovered
    directories are registered as DLL search locations instead.
    """
    search_dirs = _runtime_search_dirs()
    if os.name == "nt":
        for directory in search_dirs:
            try:
                # The returned object removes the directory when it is closed;
                # keep it alive for as long as the native library may be used.
                _dll_directory_handles.append(os.add_dll_directory(str(directory)))
            except OSError:
                pass
        return []

    loaded: list[str] = []
    for alternatives in _RUNTIME_LIBRARY_GROUPS:
        group_loaded = False
        for name in alternatives:
            for directory in search_dirs:
                candidate = directory / name
                if not candidate.is_file():
                    continue
                try:
                    ctypes.CDLL(str(candidate))
                    loaded.append(name)
                    group_loaded = True
                    break
                except OSError:
                    # Try the same SONAME in the next known runtime directory.
                    continue
            if group_loaded:
                break
    return loaded


def load_library() -> ctypes.CDLL:
    """Load (once) and configure the gpuxtb shared library."""
    global _lib
    if _lib is None:
        path = library_path()
        _preload_runtime_libraries()
        try:
            library = ctypes.CDLL(str(path))
        except OSError as exc:
            raise GPUxtbRuntimeError(
                f"cannot load gpuxtb shared library {path}: {exc}"
            ) from exc
        _configure_library(library)
        _lib = library
    return _lib


def get_library() -> ctypes.CDLL:
    """Return the configured :mod:`ctypes` shared-library handle."""
    return load_library()


# --- Thin checked wrappers ------------------------------------------------------


def _decode(value: Optional[Union[bytes, str]]) -> str:
    if value is None:
        return "<null>"
    return (
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
    )


def get_version() -> str:
    """Return the C library version string (``gpuxtb_version_string``)."""
    return _decode(load_library().gpuxtb_version_string())


def status_string(status: int) -> str:
    """Return the human-readable name for a C status value."""
    return _decode(load_library().gpuxtb_status_string(status))


def get_last_error() -> str:
    """Return the thread-local diagnostic of the most recent failing C call."""
    return _decode(load_library().gpuxtb_get_last_error())


def _check_init(operation: str, status: int) -> None:
    if status != STATUS_SUCCESS:
        raise GPUxtbRuntimeError(
            f"{operation} failed with {status_string(status)}: {get_last_error()}",
            status,
        )


def compute_checked(
    context: ctypes.c_void_p,
    batch: Batch,
    options: ComputeOptions,
    result: BatchResult,
) -> None:
    """Call ``gpuxtb_compute`` and raise on a non-success status.

    Per-system SCC or eigensolver failures are data-level results, not global
    failures: the C call returns SUCCESS and records them in
    ``per_system_status``. Those are surfaced by the caller inspecting the
    result diagnostics. Any other return value is a global failure.
    """
    library = load_library()
    status = library.gpuxtb_compute(
        context,
        ctypes.byref(batch),
        ctypes.byref(options),
        ctypes.byref(result),
    )
    if status != STATUS_SUCCESS:
        raise GPUxtbRuntimeError(
            f"gpuxtb_compute failed with {status_string(status)}: {get_last_error()}",
            status,
        )


# --- Host descriptor helpers ----------------------------------------------------


def host_const(
    values: Optional[Sequence[Union[int, float, bool]]], ctype, dtype
) -> ConstBuffer:
    """Build a host ``gpuxtb_const_buffer_t`` from a numpy-compatible sequence.

    The returned buffer aliases the contiguous numpy array, so the caller must
    keep that array alive (the *owner* is returned) until the compute call
    completes.
    """
    if values is None or len(values) == 0:
        return ConstBuffer(None, 0, MEMORY_HOST, 0)
    owner = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    if owner.nbytes == 0:
        return ConstBuffer(None, 0, MEMORY_HOST, 0)
    return ConstBuffer(
        ctypes.cast(owner.ctypes.data, ctypes.c_void_p),
        owner.nbytes,
        MEMORY_HOST,
        0,
    ), owner


def empty_result_shape(shape, ctype, dtype) -> "tuple[Buffer, np.ndarray]":
    """Allocate a host output buffer and return its descriptor plus owner."""
    owner = np.empty(shape, dtype=dtype)
    return (
        Buffer(
            ctypes.cast(owner.ctypes.data, ctypes.c_void_p),
            owner.nbytes,
            MEMORY_HOST,
            0,
        ),
        owner,
    )


__all__ = [
    "API_VERSION",
    "STATUS_SUCCESS",
    "STATUS_INVALID_ARGUMENT",
    "STATUS_BACKEND_UNAVAILABLE",
    "STATUS_NOT_SUPPORTED",
    "STATUS_ALLOCATION_FAILED",
    "STATUS_NOT_IMPLEMENTED",
    "STATUS_INTERNAL_ERROR",
    "STATUS_SCC_NOT_CONVERGED",
    "STATUS_EIGENSOLVER_FAILED",
    "BACKEND_AUTO",
    "BACKEND_CPU",
    "BACKEND_CUDA",
    "BACKEND_ROCM",
    "MEMORY_HOST",
    "MEMORY_CUDA_DEVICE",
    "MEMORY_ROCM_DEVICE",
    "MODEL_GFN1_XTB",
    "MODEL_GFN2_XTB",
    "SCC_START_FRESH",
    "SCC_START_WARM",
    "COMPUTE_ENERGY",
    "COMPUTE_FORCES",
    "COMPUTE_ATOMIC_CHARGES",
    "COMPUTE_POINT_CHARGE_FORCES",
    "KELVIN_TO_HARTREE",
    "DEFAULT_ELECTRONIC_TEMPERATURE",
    "ContextOptions",
    "ConstBuffer",
    "Buffer",
    "Batch",
    "ComputeOptions",
    "BatchResult",
    "library_path",
    "load_library",
    "get_library",
    "get_version",
    "status_string",
    "get_last_error",
    "compute_checked",
    "host_const",
    "empty_result_shape",
]
