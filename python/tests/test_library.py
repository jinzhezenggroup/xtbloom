"""Tests for the low-level ctypes binding (``xtbloom.library``)."""

from __future__ import annotations

import ctypes
import os
import re
import site
import subprocess
import sys
from importlib.metadata import version as distribution_version
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _legacy_core import build_legacy_core
from xtbloom import __version__, library
from xtbloom.exceptions import XTBloomRuntimeError, XTBloomValueError

if TYPE_CHECKING:
    from pathlib import Path


def test_pyodide_private_provider_paths_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve one repaired provider and overwrite untrusted environment paths."""
    package = tmp_path / "site-packages" / "xtbloom"
    native_dir = package / "lib"
    provider_dir = tmp_path / "site-packages" / "xtbloom.libs"
    native_dir.mkdir(parents=True)
    provider_dir.mkdir()
    native = native_dir / "libxtbloom.so"
    adapter = native_dir / "libxtbloom_pyodide_lapacke.so"
    provider = provider_dir / "libxtbloom_openblas-deadbeef.so"
    for path in (native, adapter, provider):
        path.write_bytes(b"wasm")

    monkeypatch.setattr(library.sys, "platform", "emscripten")
    monkeypatch.setattr(library, "__file__", str(package / "library.py"))
    monkeypatch.setenv("XTBLOOM_PYODIDE_LAPACKE_SHIM", "/untrusted/adapter.so")
    monkeypatch.setenv("XTBLOOM_PYODIDE_OPENBLAS", "/untrusted/provider.so")
    library._configure_pyodide_openblas_paths(native)

    assert os.environ["XTBLOOM_PYODIDE_LAPACKE_SHIM"] == str(adapter.resolve())
    assert os.environ["XTBLOOM_PYODIDE_OPENBLAS"] == str(provider.resolve())


class _FakeSymbol:
    """Minimal mutable stand-in for a configured ``ctypes`` function."""

    argtypes: object = None
    restype: object = None


class _FakeLibrary:
    """Weak-referenceable fake shared-library handle for symbol probing."""


class _FakeComputeOptionsInit:
    """Initialize either the frozen V3 image or only the legacy V2 prefix."""

    def __init__(self, supports_v3: bool) -> None:
        self.supports_v3 = supports_v3

    def __call__(self, pointer: object, struct_size: int) -> int:
        options = ctypes.cast(pointer, ctypes.POINTER(library.ComputeOptions)).contents
        ctypes.memset(pointer, 0, min(struct_size, 56))
        options.struct_size = struct_size
        options.api_version = library.API_VERSION
        if self.supports_v3:
            ctypes.memset(pointer, 0, min(struct_size, ctypes.sizeof(options)))
            options.struct_size = struct_size
            options.api_version = library.API_VERSION
            options.scc_mixer = library.SCC_MIXER_MODIFIED_BROYDEN
            options.scc_mixer_history = library.DEFAULT_SCC_MIXER_HISTORY
            options.scc_mixer_damping = library.DEFAULT_SCC_MIXER_DAMPING
            options.determinism = library.DETERMINISM_DEFAULT
        return library.STATUS_SUCCESS


def _fake_request_library(*, omit: str | None = None) -> _FakeLibrary:
    """Build a fake library with all but an optional request ABI symbol."""
    fake = _FakeLibrary()
    for name in library._REQUEST_API_SYMBOLS:
        if name != omit:
            setattr(fake, name, _FakeSymbol())
    return fake


def _expected_native_version(python_version: str) -> str:
    """Map a release or setuptools-scm development version to its base tag."""
    public = python_version.partition("+")[0]
    release = re.fullmatch(
        r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)",
        public,
    )
    if release is not None:
        return public
    development = re.fullmatch(
        r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\."
        r"(0|[1-9][0-9]*)\.post1\.dev(0|[1-9][0-9]*)",
        public,
    )
    if development is None:
        raise AssertionError(
            f"unexpected Python distribution version: {python_version}"
        )
    major, minor, patch, _distance = development.groups()
    return f"{major}.{minor}.{patch}"


def test_version_string() -> None:
    """Expose SCM identity in Python without changing the native product tag."""
    assert __version__ == distribution_version("xtbloom")
    assert library.get_version() == _expected_native_version(__version__)


def test_request_api_is_optional_as_a_complete_symbol_group() -> None:
    """An older core library remains usable when every request symbol is absent."""
    fake = _FakeLibrary()

    assert not library._configure_request_api(fake)
    assert not library.request_api_available(fake)


def test_request_api_configures_only_when_complete() -> None:
    """Configure all signatures and advertise the complete additive ABI."""
    fake = _fake_request_library()

    assert library._configure_request_api(fake)
    assert library.request_api_available(fake)
    assert fake.xtbloom_request_info_init.restype is ctypes.c_int32
    assert fake.xtbloom_plan_compute_enqueue.restype is ctypes.c_int32
    assert fake.xtbloom_request_destroy.restype is None


def test_partial_request_api_is_incompatible() -> None:
    """Reject a library that cannot provide one coherent request contract."""
    fake = _fake_request_library(omit="xtbloom_request_wait")

    with pytest.raises(XTBloomRuntimeError, match="xtbloom_request_wait"):
        library._configure_request_api(fake)


@pytest.mark.parametrize("supports_v3", [False, True])
def test_compute_options_v3_capability_probe(supports_v3: bool) -> None:
    """Distinguish a legacy future-size initializer from a complete V3 one."""
    fake = _FakeLibrary()
    fake.xtbloom_compute_options_init = _FakeComputeOptionsInit(supports_v3)

    assert library._probe_compute_options_v3(fake) is supports_v3
    assert library.compute_options_v3_available(fake) is supports_v3


def test_compute_options_v3_capability_probe_reports_initializer_failure() -> None:
    """Surface a failed native initializer instead of guessing V3 support."""
    fake = _FakeLibrary()
    fake.xtbloom_compute_options_init = lambda pointer, struct_size: (
        library.STATUS_INTERNAL_ERROR
    )

    with pytest.raises(XTBloomRuntimeError, match="probing ABI-v3 support"):
        library._probe_compute_options_v3(fake)


def test_legacy_core_accepts_defaults_but_rejects_nondefault_v3_policy() -> None:
    """Never silently ignore a user-requested mixer or determinism policy."""
    fake = _FakeLibrary()
    fake.xtbloom_compute_options_init = _FakeComputeOptionsInit(False)
    assert not library._probe_compute_options_v3(fake)

    library.require_compute_options_v3(1, 8, 0.4, 0, fake)
    with pytest.raises(XTBloomRuntimeError, match=r"does not support.*ABI v3"):
        library.require_compute_options_v3(1, 16, 0.25, 1, fake)


def test_library_override_fails_closed_for_legacy_core_policy(tmp_path: Path) -> None:
    """Exercise the real loader handshake against a V2-only override library."""
    if not sys.platform.startswith("linux"):
        pytest.skip("the legacy shared-library fixture currently targets ELF")
    legacy = build_legacy_core(tmp_path)
    script = r"""
import numpy as np
import xtbloom.library as library
from xtbloom import BatchCalculator, Calculator, Structure, compute_arrays
from xtbloom.exceptions import XTBloomRuntimeError

assert not library.compute_options_v3_available()
library.require_compute_options_v3(1, 8, 0.4, 0)
Calculator("GFN2-xTB", np.array([1]), np.zeros((1, 3)))
BatchCalculator([Structure(np.array([1]), np.zeros((1, 3)))])

for factory in (
    lambda: Calculator(
        "GFN2-xTB", np.array([1]), np.zeros((1, 3)), scc_mixer_history=16
    ),
    lambda: BatchCalculator(
        [Structure(np.array([1]), np.zeros((1, 3)))], determinism="reproducible"
    ),
):
    try:
        factory()
    except XTBloomRuntimeError as exc:
        assert "ABI v3" in str(exc)
    else:
        raise AssertionError("legacy core silently accepted nondefault V3 policy")

arrays = dict(
    atom_offsets=np.array([0, 1], dtype=np.int64),
    atomic_numbers=np.array([1], dtype=np.int32),
    positions=np.zeros((1, 3), dtype=np.float64),
    molecular_charges=np.zeros(1, dtype=np.float64),
    unpaired_electrons=np.zeros(1, dtype=np.int32),
)
compute_arrays(**arrays)
try:
    compute_arrays(**arrays, scc_mixer_damping=0.25)
except XTBloomRuntimeError as exc:
    assert "ABI v3" in str(exc)
else:
    raise AssertionError("legacy core silently accepted Array V3 policy")
"""
    env = os.environ.copy()
    env["XTBLOOM_LIBRARY"] = str(legacy)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


def test_runtime_search_includes_user_site_packages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Search CUDA runtime libraries beneath the user site-packages path."""
    user_site = tmp_path / "site-packages"
    runtime_dir = user_site / "nvidia" / "cublas" / "lib"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(site, "getsitepackages", lambda: [])
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)

    assert runtime_dir in library._runtime_search_dirs()


def test_runtime_search_does_not_load_scipy_openblas32(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep the build-only provider out of the process-global namespace."""
    user_site = tmp_path / "site-packages"
    runtime_dir = user_site / "scipy_openblas32" / "lib"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(site, "getsitepackages", list)
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)

    assert runtime_dir not in library._runtime_search_dirs()
    assert all(
        "openblas" not in name
        for alternatives in library._RUNTIME_LIBRARY_GROUPS
        for name in alternatives
    )


def test_status_strings() -> None:
    """Map public status values to stable diagnostic strings."""
    assert library.status_string(library.STATUS_SUCCESS) == "success"
    assert (
        library.status_string(library.STATUS_SCC_NOT_CONVERGED) == "SCC not converged"
    )
    assert (
        library.status_string(library.STATUS_EIGENSOLVER_FAILED) == "eigensolver failed"
    )


def test_abi_struct_sizes() -> None:
    """Keep ctypes structure sizes and suffix offsets aligned with the C ABI."""
    # The ctypes mirrors must match the C ABI field layout used by validation.
    assert ctypes.sizeof(ctypes.c_void_p) == 8  # supported wheel platforms are 64-bit
    assert ctypes.sizeof(library.ContextOptions) == 32
    assert ctypes.sizeof(library.ConstBuffer) == 24
    assert ctypes.sizeof(library.Buffer) == 24
    assert ctypes.sizeof(library.Batch) == 456
    assert library.Batch.total_interactions.offset == 352
    assert library.Batch.interaction_descriptors.offset == 360
    assert library.Batch.interaction_payload.offset == 384
    assert library.Batch.cell_matrices.offset == 408
    assert library.Batch.periodic_axes.offset == 432
    assert library.PERIODIC_AXES_NONE == 0
    assert library.PERIODIC_AXES_XYZ == 7
    assert {
        "PERIODIC_AXES_NONE",
        "PERIODIC_AXES_XYZ",
        "PERIODIC_AXIS_X",
        "PERIODIC_AXIS_Y",
        "PERIODIC_AXIS_Z",
    }.issubset(library.__all__)
    batch = library.Batch()
    assert (
        library.load_library().xtbloom_batch_init(
            ctypes.byref(batch), ctypes.sizeof(batch)
        )
        == library.STATUS_SUCCESS
    )
    assert batch.struct_size == ctypes.sizeof(batch)
    assert batch.api_version == library.API_VERSION
    assert batch.cell_matrices.data is None
    assert batch.cell_matrices.size_bytes == 0
    assert batch.cell_matrices.memory_space == library.MEMORY_HOST
    assert batch.cell_matrices.reserved == 0
    assert batch.periodic_axes.data is None
    assert batch.periodic_axes.size_bytes == 0
    assert batch.periodic_axes.memory_space == library.MEMORY_HOST
    assert batch.periodic_axes.reserved == 0
    assert ctypes.sizeof(library.ComputeOptions) == 80
    assert library.ComputeOptions.scc_start_mode.offset == 48
    assert library.ComputeOptions.reserved_v2.offset == 52
    assert library.ComputeOptions.scc_mixer.offset == 56
    assert library.ComputeOptions.scc_mixer_history.offset == 60
    assert library.ComputeOptions.scc_mixer_damping.offset == 64
    assert library.ComputeOptions.determinism.offset == 72
    assert library.ComputeOptions.reserved_v3.offset == 76
    assert ctypes.sizeof(library.BatchResult) == 280
    assert library.BatchResult.dipole_moments.offset == 184
    assert library.BatchResult.quadrupole_moments.offset == 208
    assert library.BatchResult.wiberg_orders.offset == 232
    assert library.BatchResult.spin_populations.offset == 256
    assert ctypes.sizeof(library.Interaction) == 32
    assert library.Interaction.type.offset == 0
    assert library.Interaction.flags.offset == 4
    assert library.Interaction.system_index.offset == 8
    assert library.Interaction.payload_offset.offset == 16
    assert library.Interaction.payload_size.offset == 24
    assert library.RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES == 1
    assert library.RESULT_DIPOLE_MOMENTS == 1 << 4
    assert ctypes.sizeof(library.ResultOwnerOptions) == 32
    assert library.ResultOwnerOptions.memory_space.offset == 8
    assert library.ResultOwnerOptions.size_bytes.offset == 16
    assert library.ResultOwnerOptions.reserved.offset == 24
    assert ctypes.sizeof(library.DlpackView) == 48
    assert library.DlpackView.byte_offset.offset == 8
    assert library.DlpackView.shape.offset == 40
    options = library.ComputeOptions()
    library.load_library().xtbloom_compute_options_init(
        ctypes.byref(options), ctypes.sizeof(options)
    )
    assert options.electronic_temperature == pytest.approx(
        library.DEFAULT_ELECTRONIC_TEMPERATURE
    )
    assert options.max_scc_iterations == 250
    assert options.scc_start_mode == library.SCC_START_FRESH
    assert options.reserved_v2 == 0
    assert options.scc_mixer == library.SCC_MIXER_MODIFIED_BROYDEN
    assert options.scc_mixer_history == library.DEFAULT_SCC_MIXER_HISTORY
    assert options.scc_mixer_damping == library.DEFAULT_SCC_MIXER_DAMPING
    assert options.determinism == library.DETERMINISM_DEFAULT
    assert options.reserved_v3 == 0
    assert library.SCC_START_WARM == 2
    # xtbloom_workspace_query_t: struct_size/api_version/flags/reserved (16) +
    # host bytes (8) + host alignment (4) + device bytes (8) + device alignment
    # (4) + reserved_v2 (4) = 48 bytes, with device bytes aligned to 8.
    assert ctypes.sizeof(library.WorkspaceQuery) == 48
    assert library.WorkspaceQuery.host_required_bytes.offset == 16
    assert library.WorkspaceQuery.host_required_alignment.offset == 24
    assert library.WorkspaceQuery.device_required_bytes.offset == 32
    assert library.WorkspaceQuery.device_required_alignment.offset == 40
    query = library.WorkspaceQuery()
    library.load_library().xtbloom_workspace_query_init(
        ctypes.byref(query), ctypes.sizeof(query)
    )
    assert query.compute_flags == 0
    assert query.host_required_bytes == 0
    assert query.host_required_alignment == 0
    assert query.device_required_bytes == 0
    assert query.device_required_alignment == 0
    assert ctypes.sizeof(library.RequestInfo) == 24
    assert library.RequestInfo.state.offset == 8
    assert library.RequestInfo.completion_status.offset == 12
    assert library.RequestInfo.result_flags.offset == 16
    request_info = library.RequestInfo()
    library.load_library().xtbloom_request_info_init(
        ctypes.byref(request_info), ctypes.sizeof(request_info)
    )
    assert request_info.struct_size == 24
    assert request_info.api_version == library.API_VERSION
    assert request_info.state == library.REQUEST_IDLE
    assert request_info.completion_status == library.STATUS_SUCCESS
    assert request_info.result_flags == 0
    assert request_info.reserved == 0


def test_unknown_method_rejected() -> None:
    """Distinguish unknown methods from reserved unsupported GFN1-xTB."""
    from xtbloom.exceptions import XTBloomNotSupportedError
    from xtbloom.interface import Calculator

    with pytest.raises(XTBloomValueError):
        Calculator("NoSuchMethod", np.array([1]), np.zeros((1, 3)))
    with pytest.raises(XTBloomNotSupportedError):
        Calculator("GFN1-xTB", np.array([1]), np.zeros((1, 3)))


def test_host_const_returns_consistent_buffer_and_owner() -> None:
    """Keep host descriptor pointers and owning arrays consistent."""
    buf, owner = library.host_const([1.0, 2.0, 3.0], ctypes.c_double, np.float64)
    assert isinstance(buf, library.ConstBuffer)
    assert buf.data is not None
    assert buf.size_bytes == 24
    assert buf.memory_space == library.MEMORY_HOST
    assert owner.dtype == np.float64
    assert owner.shape == (3,)
    assert np.allclose(owner, [1.0, 2.0, 3.0])
    assert int(buf.data) == owner.ctypes.data

    buf_empty, owner_empty = library.host_const(None, ctypes.c_double, np.float64)
    assert isinstance(buf_empty, library.ConstBuffer)
    assert buf_empty.data is None
    assert buf_empty.size_bytes == 0
    assert owner_empty.size == 0

    buf_empty_seq, owner_empty_seq = library.host_const([], ctypes.c_double, np.float64)
    assert isinstance(buf_empty_seq, library.ConstBuffer)
    assert buf_empty_seq.data is None
    assert buf_empty_seq.size_bytes == 0
    assert owner_empty_seq.size == 0
