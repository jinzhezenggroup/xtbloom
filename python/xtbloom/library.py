"""Low-level :mod:`ctypes` binding to the xTBloom public C ABI.

This module mirrors ``tools/conformance/xtbloom_public_api.py`` and keeps the
package self-contained inside a wheel. It knows nothing about molecules or
batches; it owns the C structure mirrors, shared-library discovery, symbol
signatures, and the thin checked wrappers used by :mod:`xtbloom.interface`.

Units follow the C ABI: positions in bohr, energies in Hartree, forces in
Hartree/bohr, charges and external point-charge values in elementary-charge
units. Electronic temperatures in the ``xtbloom_compute_options_t`` are
``k_B * T`` in Hartree.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os
import sys
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import numpy.typing as npt

from .exceptions import XTBloomRuntimeError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType

_cuda_driver_handle: ctypes.CDLL | None = None

# --- ABI constants (kept in sync with include/xtbloom/xtbloom.h) ----------------

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

REQUEST_IDLE = 0
REQUEST_PENDING = 1
REQUEST_COMPLETE = 2

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

SCC_MIXER_MODIFIED_BROYDEN = 1

DETERMINISM_DEFAULT = 0
DETERMINISM_REPRODUCIBLE = 1

DEFAULT_SCC_MIXER_HISTORY = 8
DEFAULT_SCC_MIXER_DAMPING = 0.4
MAX_SCC_MIXER_HISTORY = 64

COMPUTE_ENERGY = 1 << 0
COMPUTE_FORCES = 1 << 1
COMPUTE_ATOMIC_CHARGES = 1 << 2
COMPUTE_POINT_CHARGE_FORCES = 1 << 3
COMPUTE_DIPOLE_MOMENTS = 1 << 4

RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES = 1 << 0
RESULT_DIPOLE_MOMENTS = 1 << 4

# Interaction-type tags (mirror of xtbloom_interaction_type_t). Both released
# backends execute the uniform electric field; the remaining values stay
# reserved and return NOT_IMPLEMENTED so future interactions never renumber an
# existing tag.
INTERACTION_NONE = 0
INTERACTION_ELECTRIC_FIELD = 0x0101
INTERACTION_ELECTRIC_FIELD_GRADIENT = 0x0102
INTERACTION_POINT_CHARGES_MULTIPOLE = 0x0103
INTERACTION_ATOMIC_POTENTIAL_GRID = 0x0104
INTERACTION_ALPB_SOLVATION = 0x0201
INTERACTION_GBSA_SOLVATION = 0x0202
INTERACTION_GB_SOLVATION = 0x0203
INTERACTION_GBE_SOLVATION = 0x0204
INTERACTION_DDX_SOLVATION = 0x0205
INTERACTION_D3_DISPERSION = 0x0301
INTERACTION_D4_VARIANT_DISPERSION = 0x0302
INTERACTION_HALOGEN_BOND = 0x0401

# DLPack device/dtype codes used by the xTBloom-owned result producer
# (mirrors DLPack 1.0; see xtbloom._dlpack for the consumer-side constants).
DLPACK_DEVICE_CPU = 1
DLPACK_DEVICE_CUDA = 2
DLPACK_DTYPE_INT = 0
DLPACK_DTYPE_UINT = 1
DLPACK_DTYPE_FLOAT = 2
DLPACK_DTYPE_BFLOAT = 4
DLPACK_DTYPE_BOOL = 6
DLPACK_MAX_NDIM = 8

KELVIN_TO_HARTREE = 3.166808578545117e-6
DEFAULT_ELECTRONIC_TEMPERATURE = 300.0 * KELVIN_TO_HARTREE

# --- ctypes mirrors of the public ABI structures -----------------------------


class ContextOptions(ctypes.Structure):
    """ctypes mirror of ``xtbloom_context_options_t`` ABI version 1."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("backend", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        ("cpu_threads", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("stream", ctypes.c_void_p),
    ]


class ConstBuffer(ctypes.Structure):
    """ctypes mirror of ``xtbloom_const_buffer_t`` (caller-owned input view)."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("data", ctypes.c_void_p),
        ("size_bytes", ctypes.c_size_t),
        ("memory_space", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class Buffer(ctypes.Structure):
    """ctypes mirror of ``xtbloom_buffer_t`` (caller-owned output view)."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("data", ctypes.c_void_p),
        ("size_bytes", ctypes.c_size_t),
        ("memory_space", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class Batch(ctypes.Structure):
    """ctypes mirror of ``xtbloom_batch_t`` through the ABI-v3 interaction suffix."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
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
        ("total_interactions", ctypes.c_int64),
        ("interaction_descriptors", ConstBuffer),
        ("interaction_payload", ConstBuffer),
    ]


class Interaction(ctypes.Structure):
    """ctypes mirror of ``xtbloom_interaction_t`` (one attachment entry)."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("type", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("system_index", ctypes.c_int64),
        ("payload_offset", ctypes.c_uint64),
        ("payload_size", ctypes.c_uint64),
    ]


class ComputeOptions(ctypes.Structure):
    """ctypes mirror of ``xtbloom_compute_options_t`` through ABI version 3."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
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
        ("scc_mixer", ctypes.c_int32),
        ("scc_mixer_history", ctypes.c_int32),
        ("scc_mixer_damping", ctypes.c_double),
        ("determinism", ctypes.c_int32),
        ("reserved_v3", ctypes.c_uint32),
    ]


class BatchResult(ctypes.Structure):
    """ctypes mirror of ``xtbloom_batch_result_t`` through the ABI-v2 suffix."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
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
        ("dipole_moments", Buffer),
        ("quadrupole_moments", Buffer),
        ("wiberg_orders", Buffer),
        ("spin_populations", Buffer),
    ]


class WorkspaceQuery(ctypes.Structure):
    """ctypes mirror of ``xtbloom_workspace_query_t`` ABI version 1.

    ``compute_flags`` is an input; the four sizing fields are filled by
    ``xtbloom_plan_query_workspace``.
    """

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("compute_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("host_required_bytes", ctypes.c_uint64),
        ("host_required_alignment", ctypes.c_uint32),
        ("device_required_bytes", ctypes.c_uint64),
        ("device_required_alignment", ctypes.c_uint32),
        ("reserved_v2", ctypes.c_uint32),
    ]


class RequestInfo(ctypes.Structure):
    """ctypes mirror of ``xtbloom_request_info_t`` ABI version 1."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("state", ctypes.c_int32),
        ("completion_status", ctypes.c_int32),
        ("result_flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class ResultOwnerOptions(ctypes.Structure):
    """ctypes mirror of ``xtbloom_result_owner_options_t`` ABI version 1."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("memory_space", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        ("size_bytes", ctypes.c_uint64),
        ("reserved", ctypes.c_uint32),
    ]


class DlpackView(ctypes.Structure):
    """ctypes mirror of ``xtbloom_dlpack_view_t`` ABI version 1.

    ``shape`` is a caller-owned pointer to ``ndim`` ``int64`` extents that is
    copied by xTBloom during the export call.
    """

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("byte_offset", ctypes.c_uint64),
        ("dtype_code", ctypes.c_int32),
        ("dtype_bits", ctypes.c_int32),
        ("dtype_lanes", ctypes.c_int32),
        ("ndim", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
    ]


# --- Shared-library discovery -------------------------------------------------


def _installed_package_library() -> Path | None:
    """Locate ``libxtbloom`` installed next to the Python package inside a wheel.

    With ``wheel.install-dir = "xtbloom"`` the CMake install step drops the
    versioned shared object plus symlinks into ``xtbloom/lib``. We glob so a
    wheel that flattened the SONAME symlink still resolves, and we accept the
    platform-specific file naming (ELF ``libxtbloom.so*``, macOS
    ``libxtbloom.dylib``, Windows ``xtbloom.dll``).
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
        candidates.extend(sorted(runtime_dir.glob("libxtbloom.so*")))
        candidates.extend(sorted(runtime_dir.glob("libxtbloom.dylib*")))
        candidates.extend(sorted(runtime_dir.glob("xtbloom.dll")))
    return candidates[0] if candidates else None


def library_path() -> str | Path:
    """Return the path to the xTBloom shared library.

    Resolution order:

    1. ``XTBLOOM_LIBRARY`` environment variable (explicit override).
    2. The ``lib`` directory shipped inside the installed package/wheel.
    3. A ``ctypes`` platform lookup of ``xtbloom`` (system installation).

    Raises
    ------
    XTBloomRuntimeError
        when no shared library can be found.
    """
    explicit = os.environ.get("XTBLOOM_LIBRARY")
    candidates: list[str | Path] = []
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
    # ``libxtbloom.so.0``, not an absolute filesystem path.  Passing that name
    # directly to ctypes preserves the documented system-install fallback.
    system_library = ctypes.util.find_library("xtbloom")
    if system_library:
        return system_library

    raise XTBloomRuntimeError(
        "cannot locate the xTBloom shared library; set XTBLOOM_LIBRARY or build "
        "the package with scikit-build-core so libxtbloom is bundled"
    )


def _configure_pyodide_openblas_paths(path: str | Path) -> None:
    """Publish exact private WebAssembly provider paths to the native loader.

    Emscripten has neither ``dladdr``-based sibling discovery nor isolated
    dynamic-linker namespaces. The repaired wheel layout is authoritative: one
    adapter lives beside ``libxtbloom`` and one content-qualified provider
    lives in auditwheel's top-level ``xtbloom.libs`` directory. These internal
    environment values are overwritten from installed paths on every first
    load, so user-supplied generic OpenBLAS names cannot become a fallback.
    """
    if sys.platform != "emscripten":
        return
    library = Path(path)
    if not library.is_absolute() or not library.is_file():
        raise XTBloomRuntimeError(
            "Pyodide requires the bundled xTBloom library at an absolute path"
        )
    adapter = library.parent / "libxtbloom_pyodide_lapacke.so"
    provider_dir = Path(__file__).resolve().parent.parent / "xtbloom.libs"
    providers = sorted(provider_dir.glob("libxtbloom_openblas-*.so"))
    if not adapter.is_file():
        raise XTBloomRuntimeError(
            f"private Pyodide LAPACKE adapter is missing: {adapter}"
        )
    if len(providers) != 1 or not providers[0].is_file():
        raise XTBloomRuntimeError(
            f"private Pyodide OpenBLAS provider is missing or ambiguous: {providers}"
        )
    os.environ["XTBLOOM_PYODIDE_LAPACKE_SHIM"] = str(adapter.resolve())
    os.environ["XTBLOOM_PYODIDE_OPENBLAS"] = str(providers[0].resolve())


def _configure_library(library: ctypes.CDLL) -> None:
    """Declare every C symbol signature the Python package calls."""
    library.xtbloom_get_last_error.argtypes = []
    library.xtbloom_get_last_error.restype = ctypes.c_char_p
    library.xtbloom_version_string.argtypes = []
    library.xtbloom_version_string.restype = ctypes.c_char_p
    library.xtbloom_status_string.argtypes = [ctypes.c_int32]
    library.xtbloom_status_string.restype = ctypes.c_char_p
    library.xtbloom_context_options_init.argtypes = [
        ctypes.POINTER(ContextOptions),
        ctypes.c_size_t,
    ]
    library.xtbloom_context_options_init.restype = ctypes.c_int32
    library.xtbloom_batch_init.argtypes = [ctypes.POINTER(Batch), ctypes.c_size_t]
    library.xtbloom_batch_init.restype = ctypes.c_int32
    library.xtbloom_compute_options_init.argtypes = [
        ctypes.POINTER(ComputeOptions),
        ctypes.c_size_t,
    ]
    library.xtbloom_compute_options_init.restype = ctypes.c_int32
    library.xtbloom_batch_result_init.argtypes = [
        ctypes.POINTER(BatchResult),
        ctypes.c_size_t,
    ]
    library.xtbloom_batch_result_init.restype = ctypes.c_int32
    library.xtbloom_workspace_query_init.argtypes = [
        ctypes.POINTER(WorkspaceQuery),
        ctypes.c_size_t,
    ]
    library.xtbloom_workspace_query_init.restype = ctypes.c_int32
    library.xtbloom_result_owner_options_init.argtypes = [
        ctypes.POINTER(ResultOwnerOptions),
        ctypes.c_size_t,
    ]
    library.xtbloom_result_owner_options_init.restype = ctypes.c_int32
    library.xtbloom_result_owner_create.argtypes = [
        ctypes.POINTER(ResultOwnerOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.xtbloom_result_owner_create.restype = ctypes.c_int32
    library.xtbloom_result_owner_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Buffer),
    ]
    library.xtbloom_result_owner_buffer.restype = ctypes.c_int32
    library.xtbloom_result_owner_retain.argtypes = [ctypes.c_void_p]
    library.xtbloom_result_owner_retain.restype = None
    library.xtbloom_result_owner_release.argtypes = [ctypes.c_void_p]
    library.xtbloom_result_owner_release.restype = None
    library.xtbloom_result_owner_export_dltensor.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(DlpackView),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.xtbloom_result_owner_export_dltensor.restype = ctypes.c_int32
    library.xtbloom_plan_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Batch),
        ctypes.POINTER(ComputeOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.xtbloom_plan_create.restype = ctypes.c_int32
    library.xtbloom_plan_destroy.argtypes = [ctypes.c_void_p]
    library.xtbloom_plan_destroy.restype = None
    library.xtbloom_plan_query_workspace.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(WorkspaceQuery),
    ]
    library.xtbloom_plan_query_workspace.restype = ctypes.c_int32
    library.xtbloom_plan_compute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Batch),
        ctypes.POINTER(ComputeOptions),
        ctypes.POINTER(BatchResult),
    ]
    library.xtbloom_plan_compute.restype = ctypes.c_int32
    library.xtbloom_context_create.argtypes = [
        ctypes.POINTER(ContextOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.xtbloom_context_create.restype = ctypes.c_int32
    library.xtbloom_context_destroy.argtypes = [ctypes.c_void_p]
    library.xtbloom_context_destroy.restype = None
    library.xtbloom_context_get_backend.argtypes = [ctypes.c_void_p]
    library.xtbloom_context_get_backend.restype = ctypes.c_int32
    library.xtbloom_context_get_device_id.argtypes = [ctypes.c_void_p]
    library.xtbloom_context_get_device_id.restype = ctypes.c_int32
    library.xtbloom_compute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Batch),
        ctypes.POINTER(ComputeOptions),
        ctypes.POINTER(BatchResult),
    ]
    library.xtbloom_compute.restype = ctypes.c_int32
    _configure_request_api(library)


_REQUEST_API_SYMBOLS = (
    "xtbloom_request_info_init",
    "xtbloom_request_create",
    "xtbloom_compute_enqueue",
    "xtbloom_plan_compute_enqueue",
    "xtbloom_request_query",
    "xtbloom_request_wait",
    "xtbloom_request_get_error",
    "xtbloom_request_destroy",
)
_REQUEST_API_AVAILABILITY: weakref.WeakKeyDictionary[object, bool] = (
    weakref.WeakKeyDictionary()
)
_COMPUTE_OPTIONS_V3_AVAILABILITY: weakref.WeakKeyDictionary[object, bool] = (
    weakref.WeakKeyDictionary()
)


def _probe_compute_options_v3(library: ctypes.CDLL) -> bool:
    """Detect whether the selected core initializes the complete ABI-v3 suffix.

    Future-larger structures are accepted by older cores, so a successful init
    does not itself prove support. A sentinel-filled suffix distinguishes an
    older 56-byte initializer from the frozen V3 defaults without requiring a
    new public symbol.
    """
    options = ComputeOptions()
    ctypes.memset(ctypes.byref(options), 0xA5, ctypes.sizeof(options))
    status = library.xtbloom_compute_options_init(
        ctypes.byref(options), ctypes.sizeof(options)
    )
    if status != STATUS_SUCCESS:
        raise XTBloomRuntimeError(
            "xtbloom_compute_options_init failed while probing ABI-v3 support",
            status,
        )
    available = (
        options.scc_mixer == SCC_MIXER_MODIFIED_BROYDEN
        and options.scc_mixer_history == DEFAULT_SCC_MIXER_HISTORY
        and options.scc_mixer_damping == DEFAULT_SCC_MIXER_DAMPING
        and options.determinism == DETERMINISM_DEFAULT
        and options.reserved_v3 == 0
    )
    _COMPUTE_OPTIONS_V3_AVAILABILITY[library] = available
    return available


def compute_options_v3_available(library: ctypes.CDLL | None = None) -> bool:
    """Return whether the resolved core supports the complete options V3 suffix."""
    handle = load_library() if library is None else library
    return _COMPUTE_OPTIONS_V3_AVAILABILITY.get(handle, False)


def require_compute_options_v3(
    scc_mixer: int,
    scc_mixer_history: int,
    scc_mixer_damping: float,
    determinism: int,
    library: ctypes.CDLL | None = None,
) -> None:
    """Fail closed when an older core would ignore a nondefault V3 policy."""
    if (
        scc_mixer == SCC_MIXER_MODIFIED_BROYDEN
        and scc_mixer_history == DEFAULT_SCC_MIXER_HISTORY
        and scc_mixer_damping == DEFAULT_SCC_MIXER_DAMPING
        and determinism == DETERMINISM_DEFAULT
    ):
        return
    handle = load_library() if library is None else library
    if not compute_options_v3_available(handle):
        raise XTBloomRuntimeError(
            "the loaded xTBloom core does not support compute-options ABI v3; "
            "nondefault scc_mixer_history, scc_mixer_damping, or determinism "
            "would be ignored"
        )


def _configure_request_api(library: ctypes.CDLL) -> bool:
    """Configure the additive request ABI when the loaded library provides it.

    ``XTBLOOM_LIBRARY`` may intentionally select an older ABI-compatible core
    library. Such a library remains usable through the synchronous API when it
    exports none of the additive request symbols. A partial symbol group,
    however, identifies an incompatible or damaged installation and is rejected
    before any request operation can observe mismatched semantics.
    """
    missing = [name for name in _REQUEST_API_SYMBOLS if not hasattr(library, name)]

    if len(missing) == len(_REQUEST_API_SYMBOLS):
        _REQUEST_API_AVAILABILITY[library] = False
        return False
    if missing:
        missing_list = ", ".join(missing)
        raise XTBloomRuntimeError(
            "incompatible xtbloom shared library: the request ABI symbol group "
            f"is incomplete; missing {missing_list}"
        )

    symbols: dict[str, Any] = {
        name: getattr(library, name) for name in _REQUEST_API_SYMBOLS
    }
    symbols["xtbloom_request_info_init"].argtypes = [
        ctypes.POINTER(RequestInfo),
        ctypes.c_size_t,
    ]
    symbols["xtbloom_request_info_init"].restype = ctypes.c_int32
    symbols["xtbloom_request_create"].argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    symbols["xtbloom_request_create"].restype = ctypes.c_int32
    for name in ("xtbloom_compute_enqueue", "xtbloom_plan_compute_enqueue"):
        symbols[name].argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Batch),
            ctypes.POINTER(ComputeOptions),
            ctypes.POINTER(BatchResult),
            ctypes.c_void_p,
        ]
        symbols[name].restype = ctypes.c_int32
    for name in ("xtbloom_request_query", "xtbloom_request_wait"):
        symbols[name].argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(RequestInfo),
        ]
        symbols[name].restype = ctypes.c_int32
    symbols["xtbloom_request_get_error"].argtypes = [ctypes.c_void_p]
    symbols["xtbloom_request_get_error"].restype = ctypes.c_char_p
    symbols["xtbloom_request_destroy"].argtypes = [ctypes.c_void_p]
    symbols["xtbloom_request_destroy"].restype = None
    _REQUEST_API_AVAILABILITY[library] = True
    return True


def request_api_available(library: ctypes.CDLL | None = None) -> bool:
    """Return whether the resolved native library has the complete request ABI."""
    handle = load_library() if library is None else library
    return _REQUEST_API_AVAILABILITY.get(handle, False)


_lib: ctypes.CDLL | None = None
_dll_directory_handles: list = []


def _runtime_search_dirs() -> list[Path]:
    """Return directories that may contain xTBloom's optional host runtimes.

    The CUDA host-API shims and CPU eigensolver resolve their providers by
    SONAME. Those providers live in optional ``mkl``/``nvidia-*`` packages, an
    installed CUDA toolkit. Linux wheels carry their CPU OpenBLAS provider in a
    private auditwheel-managed dependency closure, while native installs let
    the C++ runtime discover a compatible system OpenBLAS directly. Collecting
    the remaining provider locations lets the package register their SONAMEs
    before libxtbloom is loaded without forcing users to set ``LD_LIBRARY_PATH``
    (or ``PATH`` on Windows).
    """
    dirs: list[Path] = []

    site_module: ModuleType | None = None
    try:
        import site

        site_module = site
    except Exception:  # noqa: BLE001 - defensive: probe environments where
        # `site` is unavailable without aborting library discovery
        site_module = None  # pragma: no cover - defensive
    site_packages = (
        getattr(site_module, "getsitepackages", lambda: [])()
        if site_module is not None
        else []
    )
    user_site = (
        getattr(site_module, "getusersitepackages", lambda: None)()
        if site_module is not None
        else None
    )
    if user_site and (
        bool(getattr(site_module, "ENABLE_USER_SITE", False))
        or str(user_site) in sys.path
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
        dirs.extend(
            prefix_lib
            for prefix_lib in (base.parent.parent, base.parent.parent.parent)
            if prefix_lib.name == "lib" and prefix_lib.is_dir()
        )
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


def _load_cuda_driver() -> ctypes.CDLL | None:
    """Load the process-global NVIDIA driver used to scope DLPack export.

    The driver API is independent of the CUDA runtime major used by an array
    producer.  Keeping this handle alive avoids mixing xTBloom's CUDA-12
    runtime cohort with a producer such as a CUDA-13 PyTorch build merely to
    select the producer's current device.
    """
    global _cuda_driver_handle
    if _cuda_driver_handle is not None:
        return _cuda_driver_handle
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    _cuda_driver_handle = driver
    return driver


@contextlib.contextmanager
def _cuda_device_scope(device_id: int) -> Iterator[None]:
    """Push one CUDA primary context temporarily and restore the caller state.

    Some conforming DLPack producers, notably PyTorch, require their array's
    device to be current while ``__dlpack__`` exports the capsule.  The driver
    context stack is runtime-major-neutral and thread-local; using it avoids
    importing an array backend or calling one CUDA runtime's ``cudaSetDevice``
    inside a producer built against another runtime major.  The scope performs
    no synchronization.  A CUDA-less process is a no-op so protocol fakes
    remain testable without a driver.
    """
    driver = _load_cuda_driver()
    if driver is None:
        yield
        return

    try:
        cu_init = driver.cuInit
        cu_device_get = driver.cuDeviceGet
        cu_primary_retain = driver.cuDevicePrimaryCtxRetain
        cu_primary_release = driver.cuDevicePrimaryCtxRelease
        cu_context_push = driver.cuCtxPushCurrent_v2
        cu_context_pop = driver.cuCtxPopCurrent_v2
    except AttributeError:
        yield
        return
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    cu_device_get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cu_device_get.restype = ctypes.c_int
    cu_primary_retain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    cu_primary_retain.restype = ctypes.c_int
    cu_primary_release.argtypes = [ctypes.c_int]
    cu_primary_release.restype = ctypes.c_int
    cu_context_push.argtypes = [ctypes.c_void_p]
    cu_context_push.restype = ctypes.c_int
    cu_context_pop.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cu_context_pop.restype = ctypes.c_int

    init_status = int(cu_init(0))
    if init_status != 0:
        # A loader stub or CUDA-less host cannot establish a meaningful
        # context stack.  Let the producer provide its normal diagnostic.
        yield
        return

    target = int(device_id)
    device = ctypes.c_int()
    device_status = int(cu_device_get(ctypes.byref(device), target))
    if device_status != 0:
        raise XTBloomRuntimeError(
            f"could not resolve CUDA device {target} for DLPack export "
            f"(cuDeviceGet status {device_status})"
        )
    context = ctypes.c_void_p()
    retain_status = int(cu_primary_retain(ctypes.byref(context), device.value))
    if retain_status != 0:
        raise XTBloomRuntimeError(
            f"could not retain CUDA device {target}'s primary context for "
            f"DLPack export (cuDevicePrimaryCtxRetain status {retain_status})"
        )
    push_status = int(cu_context_push(context))
    if push_status != 0:
        release_status = int(cu_primary_release(device.value))
        suffix = (
            f"; primary-context release status {release_status}"
            if release_status != 0
            else ""
        )
        raise XTBloomRuntimeError(
            f"could not make CUDA device {target} current for DLPack export "
            f"(cuCtxPushCurrent status {push_status}{suffix})"
        )

    try:
        yield
    except BaseException as export_error:
        popped = ctypes.c_void_p()
        restore_status = int(cu_context_pop(ctypes.byref(popped)))
        release_status = (
            int(cu_primary_release(device.value)) if restore_status == 0 else None
        )
        if restore_status != 0 or release_status != 0:
            release_diagnostic = (
                str(release_status)
                if release_status is not None
                else "not attempted after pop failure"
            )
            raise XTBloomRuntimeError(
                f"DLPack export failed ({export_error}); additionally could "
                "not restore the caller's CUDA context after DLPack export "
                f"(cuCtxPopCurrent status {restore_status}, "
                f"cuDevicePrimaryCtxRelease status {release_diagnostic})"
            ) from export_error
        raise
    else:
        popped = ctypes.c_void_p()
        restore_status = int(cu_context_pop(ctypes.byref(popped)))
        release_status = (
            int(cu_primary_release(device.value)) if restore_status == 0 else None
        )
        if restore_status != 0 or release_status != 0:
            release_diagnostic = (
                str(release_status)
                if release_status is not None
                else "not attempted after pop failure"
            )
            raise XTBloomRuntimeError(
                "could not restore the caller's CUDA context after DLPack export "
                f"(cuCtxPopCurrent status {restore_status}, "
                f"cuDevicePrimaryCtxRelease status {release_diagnostic})"
            )


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
)


def _preload_runtime_libraries() -> list[str]:
    """Preload optional host runtimes so users need no environment setup.

    On POSIX, loading the dependency by absolute path registers it under its
    SONAME, so the CUDA host-API shims and the CPU eigensolver's later
    by-name ``dlopen`` reuse the already-loaded object. On Windows the
    discovered directories are registered as DLL search locations instead.
    """
    search_dirs = _runtime_search_dirs()
    if os.name == "nt":
        # os.add_dll_directory exists only on Windows; this branch proves its
        # presence at runtime, but a type checker resolving for a generic
        # platform cannot see the member, hence the targeted suppression.
        add_dll_directory = os.add_dll_directory  # type: ignore[unresolved-attribute]
        for directory in search_dirs:
            # The returned object removes the directory when it is closed;
            # keep it alive for as long as the native library may be used.
            with contextlib.suppress(OSError):
                _dll_directory_handles.append(add_dll_directory(str(directory)))
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
    """Load (once) and configure the xTBloom shared library."""
    global _lib
    if _lib is None:
        path = library_path()
        _configure_pyodide_openblas_paths(path)
        _preload_runtime_libraries()
        try:
            library = ctypes.CDLL(str(path))
        except OSError as exc:
            raise XTBloomRuntimeError(
                f"cannot load xTBloom shared library {path}: {exc}"
            ) from exc
        _configure_library(library)
        _probe_compute_options_v3(library)
        _lib = library
    return _lib


def get_library() -> ctypes.CDLL:
    """Return the configured :mod:`ctypes` shared-library handle."""
    return load_library()


# --- Thin checked wrappers ------------------------------------------------------


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return "<null>"
    return (
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
    )


def get_version() -> str:
    """Return the C library version string (``xtbloom_version_string``)."""
    return _decode(load_library().xtbloom_version_string())


def status_string(status: int) -> str:
    """Return the human-readable name for a C status value."""
    return _decode(load_library().xtbloom_status_string(status))


def get_last_error() -> str:
    """Return the thread-local diagnostic of the most recent failing C call."""
    return _decode(load_library().xtbloom_get_last_error())


def _check_init(operation: str, status: int) -> None:
    if status != STATUS_SUCCESS:
        raise XTBloomRuntimeError(
            f"{operation} failed with {status_string(status)}: {get_last_error()}",
            status,
        )


def device_memory_info(device_id: int = 0) -> tuple[int, int] | None:
    """Return ``(free_bytes, total_bytes)`` for one CUDA device, or ``None``.

    Used by the auto-batch-size controller to budget a batch slice from actual
    device memory instead of only a hard-coded atom limit. The query binds the
    exact CUDA-12 runtime cohort used by xTBloom, never an arbitrary system CUDA
    major. A CUDA-less host, a loader stub without a real driver, a failed
    query, or a failed device restoration returns ``None``. The function
    attempts to restore the caller's current CUDA device on every changed-device
    exit, including query failure.
    """
    try:
        cudart = ctypes.CDLL("libcudart.so.12")
    except OSError:
        for directory in _runtime_search_dirs():
            candidate = directory / "libcudart.so.12"
            if not candidate.is_file():
                continue
            try:
                cudart = ctypes.CDLL(str(candidate))
                break
            except OSError:
                continue
        else:
            return None

    try:
        cuda_get_device = cudart.cudaGetDevice
        cuda_set_device = cudart.cudaSetDevice
        cuda_mem_get_info = cudart.cudaMemGetInfo
    except AttributeError:
        return None

    cuda_get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda_get_device.restype = ctypes.c_int
    cuda_set_device.argtypes = [ctypes.c_int]
    cuda_set_device.restype = ctypes.c_int
    cuda_mem_get_info.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    cuda_mem_get_info.restype = ctypes.c_int

    current = ctypes.c_int()
    if cuda_get_device(ctypes.byref(current)) != 0:
        return None
    changed_device = current.value != int(device_id)
    if changed_device and cuda_set_device(int(device_id)) != 0:
        return None

    free_bytes = ctypes.c_size_t()
    total_bytes = ctypes.c_size_t()
    query_ok = False
    try:
        try:
            query_ok = (
                cuda_mem_get_info(ctypes.byref(free_bytes), ctypes.byref(total_bytes))
                == 0
            )
        except (OSError, ValueError, ctypes.ArgumentError):
            query_ok = False
    finally:
        restored = not changed_device or cuda_set_device(int(current.value)) == 0

    if not query_ok or not restored:
        return None
    free = int(free_bytes.value)
    total = int(total_bytes.value)
    if total <= 0 or free < 0 or free > total:
        return None
    return free, total


def compute_checked(
    context: ctypes.c_void_p,
    batch: Batch,
    options: ComputeOptions,
    result: BatchResult,
) -> None:
    """Call ``xtbloom_compute`` and raise on a non-success status.

    Per-system SCC or eigensolver failures are data-level results, not global
    failures: the C call returns SUCCESS and records them in
    ``per_system_status``. Those are surfaced by the caller inspecting the
    result diagnostics. Any other return value is a global failure.
    """
    library = load_library()
    status = library.xtbloom_compute(
        context,
        ctypes.byref(batch),
        ctypes.byref(options),
        ctypes.byref(result),
    )
    if status != STATUS_SUCCESS:
        raise XTBloomRuntimeError(
            f"xtbloom_compute failed with {status_string(status)}: {get_last_error()}",
            status,
        )


class Plan:
    """`ctypes` ownership wrapper around a ``xtbloom_plan_t``.

    A fixed-topology plan is created from one validated batch descriptor and
    keeps its topology and backend workspace reserved so repeated
    ``plan_compute`` calls (changing geometry only) perform zero steady-state
    allocations. The plan must be destroyed before the context it was created
    from.
    """

    def __init__(
        self,
        context: ctypes.c_void_p,
        batch: Batch,
        options: ComputeOptions,
        context_owner: object | None = None,
    ) -> None:
        library = load_library()
        handle = ctypes.c_void_p()
        status = library.xtbloom_plan_create(
            context, ctypes.byref(batch), ctypes.byref(options), ctypes.byref(handle)
        )
        if status != STATUS_SUCCESS:
            raise XTBloomRuntimeError(
                "xtbloom_plan_create failed with "
                f"{status_string(status)}: {get_last_error()}",
                status,
            )
        self._handle = handle
        self._library = library
        self._closed = False
        # The native plan borrows the context lifetime. Retaining the Python
        # owner prevents garbage collection from invalidating a live plan.
        self._context_owner = context_owner

    @property
    def handle(self) -> ctypes.c_void_p:
        """:return: the opaque native handle (kept for ctypes consumers)."""
        return self._handle

    def query_workspace(self, compute_flags: int) -> WorkspaceQuery:
        """Query reserved host/device workspace for the requested properties."""
        self._ensure_open()
        query = WorkspaceQuery()
        self._check_init(query, "xtbloom_workspace_query_init")
        query.compute_flags = int(compute_flags)
        status = self._library.xtbloom_plan_query_workspace(
            self._handle, ctypes.byref(query)
        )
        if status != STATUS_SUCCESS:
            raise XTBloomRuntimeError(
                "xtbloom_plan_query_workspace failed with "
                f"{status_string(status)}: {get_last_error()}",
                status,
            )
        return query

    def compute(
        self,
        batch: Batch,
        options: ComputeOptions,
        result: BatchResult,
    ) -> None:
        """Run one fixed-topology inference; raises on a non-success status."""
        self._ensure_open()
        status = self._library.xtbloom_plan_compute(
            self._handle,
            ctypes.byref(batch),
            ctypes.byref(options),
            ctypes.byref(result),
        )
        if status != STATUS_SUCCESS:
            raise XTBloomRuntimeError(
                f"xtbloom_plan_compute failed with {status_string(status)}: "
                f"{get_last_error()}",
                status,
            )

    def destroy(self) -> None:
        """Release the native plan handle (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._library.xtbloom_plan_destroy(self._handle)
        self._handle = ctypes.c_void_p()
        self._context_owner = None

    def __del__(self) -> None:
        """Best-effort native cleanup when the wrapper is garbage-collected."""
        with contextlib.suppress(Exception):
            self.destroy()

    @staticmethod
    def _check_init(query: WorkspaceQuery, operation: str) -> None:
        status = load_library().xtbloom_workspace_query_init(
            ctypes.byref(query), ctypes.sizeof(query)
        )
        if status != STATUS_SUCCESS:
            raise XTBloomRuntimeError(
                f"{operation} failed with {status_string(status)}: {get_last_error()}",
                status,
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise XTBloomRuntimeError(
                "fixed-topology plan is closed", STATUS_INVALID_ARGUMENT
            )


# --- Host descriptor helpers ----------------------------------------------------


def host_const(
    values: Sequence[int | float | bool] | None,
    ctype: type[ctypes._SimpleCData],
    dtype: npt.DTypeLike,
) -> tuple[ConstBuffer, np.ndarray]:
    """Build a host ``xtbloom_const_buffer_t`` from a numpy-compatible sequence.

    The returned buffer aliases the contiguous numpy array, so the caller must
    keep that array alive (the *owner*, returned as the second element) until
    the compute call completes. Empty input yields a null buffer descriptor
    with a zero-sized owner.
    """
    if values is None:
        owner = np.empty(0, dtype=dtype)
        return ConstBuffer(None, 0, MEMORY_HOST, 0), owner
    owner = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    if owner.size == 0:
        return ConstBuffer(None, 0, MEMORY_HOST, 0), owner
    return (
        ConstBuffer(
            ctypes.cast(owner.ctypes.data, ctypes.c_void_p),
            owner.nbytes,
            MEMORY_HOST,
            0,
        ),
        owner,
    )


def empty_result_shape(
    shape: int | tuple[int, ...],
    ctype: type[ctypes._SimpleCData],
    dtype: npt.DTypeLike,
) -> tuple[Buffer, np.ndarray]:
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
    "BACKEND_AUTO",
    "BACKEND_CPU",
    "BACKEND_CUDA",
    "BACKEND_ROCM",
    "COMPUTE_ATOMIC_CHARGES",
    "COMPUTE_DIPOLE_MOMENTS",
    "COMPUTE_ENERGY",
    "COMPUTE_FORCES",
    "COMPUTE_POINT_CHARGE_FORCES",
    "DEFAULT_ELECTRONIC_TEMPERATURE",
    "DEFAULT_SCC_MIXER_DAMPING",
    "DEFAULT_SCC_MIXER_HISTORY",
    "DETERMINISM_DEFAULT",
    "DETERMINISM_REPRODUCIBLE",
    "DLPACK_DEVICE_CPU",
    "DLPACK_DEVICE_CUDA",
    "DLPACK_DTYPE_BFLOAT",
    "DLPACK_DTYPE_BOOL",
    "DLPACK_DTYPE_FLOAT",
    "DLPACK_DTYPE_INT",
    "DLPACK_DTYPE_UINT",
    "DLPACK_MAX_NDIM",
    "INTERACTION_ALPB_SOLVATION",
    "INTERACTION_ATOMIC_POTENTIAL_GRID",
    "INTERACTION_D3_DISPERSION",
    "INTERACTION_D4_VARIANT_DISPERSION",
    "INTERACTION_DDX_SOLVATION",
    "INTERACTION_ELECTRIC_FIELD",
    "INTERACTION_ELECTRIC_FIELD_GRADIENT",
    "INTERACTION_GBE_SOLVATION",
    "INTERACTION_GBSA_SOLVATION",
    "INTERACTION_GB_SOLVATION",
    "INTERACTION_HALOGEN_BOND",
    "INTERACTION_NONE",
    "INTERACTION_POINT_CHARGES_MULTIPOLE",
    "KELVIN_TO_HARTREE",
    "MAX_SCC_MIXER_HISTORY",
    "MEMORY_CUDA_DEVICE",
    "MEMORY_HOST",
    "MEMORY_ROCM_DEVICE",
    "MODEL_GFN1_XTB",
    "MODEL_GFN2_XTB",
    "REQUEST_COMPLETE",
    "REQUEST_IDLE",
    "REQUEST_PENDING",
    "RESULT_DIPOLE_MOMENTS",
    "RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES",
    "SCC_MIXER_MODIFIED_BROYDEN",
    "SCC_START_FRESH",
    "SCC_START_WARM",
    "STATUS_ALLOCATION_FAILED",
    "STATUS_BACKEND_UNAVAILABLE",
    "STATUS_EIGENSOLVER_FAILED",
    "STATUS_INTERNAL_ERROR",
    "STATUS_INVALID_ARGUMENT",
    "STATUS_NOT_IMPLEMENTED",
    "STATUS_NOT_SUPPORTED",
    "STATUS_SCC_NOT_CONVERGED",
    "STATUS_SUCCESS",
    "Batch",
    "BatchResult",
    "Buffer",
    "ComputeOptions",
    "ConstBuffer",
    "ContextOptions",
    "DlpackView",
    "Interaction",
    "Plan",
    "RequestInfo",
    "ResultOwnerOptions",
    "WorkspaceQuery",
    "compute_checked",
    "compute_options_v3_available",
    "device_memory_info",
    "empty_result_shape",
    "get_last_error",
    "get_library",
    "get_version",
    "host_const",
    "library_path",
    "load_library",
    "require_compute_options_v3",
    "status_string",
]
