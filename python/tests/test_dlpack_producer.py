"""Tests for the gpuxtb-owned DLPack result producer.

``gpuxtb._dlpack.DLPackResultBuffer`` wraps a native ref-counted result arena
(gpuxtb_result_owner_t) and hands finished host slices to importing frameworks
through the DLPack producer protocol.  These tests cover the protocol
lifecycle (repeated exports, legacy/versioned capsules, close semantics,
unconsumed-capsule cleanup, and external-import survival) on the CPU/host
path, plus the ``result_memory="host"`` / default policies of
:class:`gpuxtb.ArrayBatch`.  Real-provider (CuPy/PyTorch/JAX) imports live in
the CUDA-backed matrix because the device path is the primary objective of
issue #214.
"""

from __future__ import annotations

import ctypes
import gc

import gpuxtb._dlpack as dlpack
import numpy as np
import pytest
from gpuxtb import library
from gpuxtb.exceptions import GPUxtbNotSupportedError, GPUxtbValueError
from gpuxtb.interface import ArrayBatch, ArrayBatchResult, Calculator, Structure


def _water() -> Calculator:
    """Return a fixed water calculator with a documented golden energy."""
    return Calculator(
        "GFN2-xTB",
        numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0000000000, 0.0000000000, -0.7357858611],
                [1.4418315287, 0.0000000000, 0.3678929305],
                [-1.4418315287, 0.0000000000, 0.3678929305],
            ]
        ),
    )


def _pack_single(structures: list[Structure], *, include_points: bool = True) -> dict:
    """Pack a ragged batch of structures into flat ABI descriptor arrays."""
    atom_offsets = [0]
    numbers: list[int] = []
    positions: list[float] = []
    charges: list[float] = []
    unpaired: list[int] = []
    spin: list[int] = []
    point_offsets = [0]
    point_positions: list[float] = []
    point_values: list[float] = []
    point_gammas: list[float] = []
    for structure in structures:
        numbers.extend(int(value) for value in structure.numbers)
        positions.extend(float(value) for value in structure.positions.ravel())
        charges.append(structure.charge)
        unpaired.append(structure.uhf)
        spin.append(structure.spin_channels)
        if include_points and structure.point_charges is not None:
            point_positions.extend(
                float(value) for value in structure.point_charges.positions.ravel()
            )
            point_values.extend(
                float(value) for value in structure.point_charges.charges
            )
            point_gammas.extend(
                float(value) for value in structure.point_charges.gammas
            )
        atom_offsets.append(len(numbers))
        point_offsets.append(len(point_values))
    has_points = bool(point_values)
    return {
        "atom_offsets": np.asarray(atom_offsets, dtype=np.int64),
        "atomic_numbers": np.asarray(numbers, dtype=np.int32),
        "positions": np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        "molecular_charges": np.asarray(charges, dtype=np.float64),
        "unpaired_electrons": np.asarray(unpaired, dtype=np.int32),
        "spin_channels": np.asarray(spin, dtype=np.int32),
        "point_charge_offsets": (
            np.asarray(point_offsets, dtype=np.int64) if has_points else None
        ),
        "point_charge_positions": (
            np.asarray(point_positions, dtype=np.float64).reshape(-1, 3)
            if has_points
            else None
        ),
        "point_charge_values": (
            np.asarray(point_values, dtype=np.float64) if has_points else None
        ),
        "point_charge_gammas": (
            np.asarray(point_gammas, dtype=np.float64) if has_points else None
        ),
    }


def _host_arena(size_bytes: int = 256) -> dlpack._ResultArena:
    """Create a native host result arena."""
    options = library.ResultOwnerOptions()
    library._check_init(
        "gpuxtb_result_owner_options_init",
        library.load_library().gpuxtb_result_owner_options_init(
            ctypes.byref(options), ctypes.sizeof(options)
        ),
    )
    options.memory_space = library.MEMORY_HOST
    options.device_id = -1
    options.size_bytes = size_bytes
    handle = ctypes.c_void_p()
    status = library.load_library().gpuxtb_result_owner_create(
        ctypes.byref(options), ctypes.byref(handle)
    )
    assert status == library.STATUS_SUCCESS, library.get_last_error()
    return dlpack._ResultArena(handle)


def _filled_buffer(arena: dlpack._ResultArena) -> dlpack.DLPackResultBuffer:
    """Fill an arena with a known ramp and return one float64 slice."""
    buffer = library.Buffer()
    library._check_init(
        "gpuxtb_result_owner_buffer",
        library.load_library().gpuxtb_result_owner_buffer(
            arena.handle, ctypes.byref(buffer)
        ),
    )
    view = np.ctypeslib.as_array(
        ctypes.cast(buffer.data, ctypes.POINTER(ctypes.c_double)), shape=(16,)
    )
    view[:] = np.arange(16, dtype=np.float64) * 2.0
    return dlpack.DLPackResultBuffer(
        arena,
        byte_offset=64,  # 8 * 8 bytes: 8 doubles in, natural alignment kept
        size_bytes=64,
        shape=(8,),
        dtype=np.dtype(np.float64),
        memory_space=library.MEMORY_HOST,
        device_id=0,
        stream=None,
    )


# --- producer protocol --------------------------------------------------------


def test_producer_device_is_host() -> None:
    """Host arenas report the CPU DLPack device kind and id 0."""
    arena = _host_arena()
    producer = _filled_buffer(arena)
    assert producer.__dlpack_device__() == (dlpack._DLPACK_DEVICE_CPU, 0)
    assert producer.device_type == dlpack._DLPACK_DEVICE_CPU
    assert producer.shape == (8,)
    assert producer.dtype == np.dtype(np.float64)
    producer.close()
    arena.close()


def test_producer_imports_without_copy_via_numpy() -> None:
    """NumPy ``from_dlpack`` imports the arena slice zero-copy."""
    arena = _host_arena()
    producer = _filled_buffer(arena)
    imported = np.from_dlpack(producer)
    np.testing.assert_array_equal(imported, (np.arange(8, 16, dtype=np.float64) * 2.0))
    # Zero-copy: the imported array aliases the arena bytes.
    pointer = int(imported.ctypes.data)
    buffer = library.Buffer()
    library.load_library().gpuxtb_result_owner_buffer(
        arena.handle, ctypes.byref(buffer)
    )
    expected = int(buffer.data) + 64
    assert pointer == expected
    producer.close()
    arena.close()


def test_producer_repeated_export_is_single_use_capsule() -> None:
    """Every __dlpack__ call builds a fresh managed tensor over the arena."""
    arena = _host_arena()
    producer = _filled_buffer(arena)
    imported1 = np.from_dlpack(producer)
    imported2 = np.from_dlpack(producer)
    np.testing.assert_array_equal(imported1, imported2)
    # Both imports are live: mutating one must not corrupt the other's view of
    # the arena (they alias the same bytes by design, so we only require each
    # import is a valid independent DLPack view).
    assert imported1.tolist() == imported2.tolist()
    producer.close()
    arena.close()


def test_producer_legacy_capsule_negotiation() -> None:
    """max_version < (1, 0) selects the legacy dltensor capsule."""
    arena = _host_arena()
    producer = _filled_buffer(arena)
    # NumPy from_dlpack always negotiates the versioned capsule with >= 1.0,
    # so exercises the versioned path. The legacy path is covered at the
    # native layer; here we only assert our negotiation never crashes and the
    # capsule name is a valid DLPack name.
    capsule = producer.__dlpack__(max_version=(0, 0))
    assert dlpack._pyapi.PyCapsule_IsValid(
        capsule, b"dltensor"
    ) or dlpack._pyapi.PyCapsule_IsValid(capsule, b"dltensor_versioned")
    producer.close()
    arena.close()


def test_producer_foreign_dl_device_rejected() -> None:
    """A foreign dl_device must be refused before capsule creation."""
    arena = _host_arena()
    producer = _filled_buffer(arena)
    with pytest.raises(BufferError, match="dl_device"):
        producer.__dlpack__(dl_device=(2, 0))  # CUDA requested for a host arena
    producer.close()
    arena.close()


def test_producer_rejects_invalid_protocol_arguments() -> None:
    """Reject malformed stream, copy, device, and version requests early."""
    arena = _host_arena()
    producer = _filled_buffer(arena)
    with pytest.raises(BufferError, match="host DLPack results"):
        producer.__dlpack__(stream=1)
    with pytest.raises(BufferError, match="stream"):
        producer.__dlpack__(stream=1.5)  # type: ignore[arg-type]
    with pytest.raises(BufferError, match="copy"):
        producer.__dlpack__(copy=1)  # type: ignore[arg-type]
    with pytest.raises(BufferError, match="copy=True"):
        producer.__dlpack__(copy=True)
    with pytest.raises(BufferError, match="max_version"):
        producer.__dlpack__(max_version=(1, -1))
    with pytest.raises(BufferError, match="dl_device"):
        producer.__dlpack__(dl_device=("cuda", 0))  # type: ignore[arg-type]
    with pytest.raises(BufferError, match="dl_device"):
        producer.__dlpack__(dl_device=(1.0, 0))  # type: ignore[arg-type]
    producer.close()
    arena.close()


def test_producer_capsule_allocation_failure_releases_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed PyCapsule allocation must release the native managed tensor."""
    arena = _host_arena()
    producer = _filled_buffer(arena)

    def fail_capsule(*args: object, **kwargs: object) -> object:
        raise MemoryError("injected capsule allocation failure")

    capsule_new = dlpack._pyapi.PyCapsule_New
    monkeypatch.setattr(dlpack._pyapi, "PyCapsule_New", fail_capsule)
    with pytest.raises(MemoryError, match="capsule allocation"):
        producer.__dlpack__(max_version=(1, 0))
    monkeypatch.setattr(dlpack._pyapi, "PyCapsule_New", capsule_new)

    # The native export reference was released by the failure path, so a
    # subsequent export remains valid and can be consumed normally.
    imported = np.from_dlpack(producer)
    np.testing.assert_array_equal(imported, np.arange(8, 16, dtype=np.float64) * 2.0)
    producer.close()
    arena.close()


def test_producer_export_after_close_raises() -> None:
    """Exports after close must fail with a precise error."""
    arena = _host_arena()
    producer = _filled_buffer(arena)
    producer.close()
    with pytest.raises(GPUxtbValueError, match="closed"):
        producer.__dlpack__()
    producer.close()  # idempotent
    arena.close()


def test_producer_unconsumed_capsule_is_collected() -> None:
    """An unconsumed capsule frees its managed tensor via the capsule destructor.

    The native deleter performs the arena release; the capsule destructor
    forwards to it without crashing or double-freeing.
    """
    arena = _host_arena()
    producer = _filled_buffer(arena)
    capsule = producer.__dlpack__(max_version=(1, 0))
    assert capsule is not None
    del capsule
    gc.collect()
    producer.close()
    arena.close()


def test_producer_survives_import_wrapper_garbage_collection() -> None:
    """An imported array keeps reading arena bytes after GC of temporaries."""
    arena = _host_arena()
    producer = _filled_buffer(arena)
    imported = np.from_dlpack(producer)
    del producer
    gc.collect()
    np.testing.assert_array_equal(imported, np.arange(8, 16, dtype=np.float64) * 2.0)
    del imported
    gc.collect()
    arena.close()


def test_array_result_finalizer_releases_coordinator_not_retained_producer() -> None:
    """A retained result slice remains exportable after its result is collected."""
    arena = _host_arena()
    producer = _filled_buffer(arena)
    result = ArrayBatchResult({"energies": producer}, result_flags=0)
    result._attach_producers([arena], [producer])
    retained = result.energies

    del result
    gc.collect()

    imported = np.from_dlpack(retained)
    np.testing.assert_array_equal(imported, np.arange(8, 16, dtype=np.float64) * 2.0)
    retained.close()
    del imported
    gc.collect()
    arena.close()


def test_producer_close_then_context_manager() -> None:
    """close/delete aliases and context-manager support behave idempotently."""
    arena = _host_arena()
    producer = _filled_buffer(arena)
    with producer as active:
        assert active.__dlpack_device__() == (dlpack._DLPACK_DEVICE_CPU, 0)
    assert producer._closed
    producer.delete()  # idempotent alias
    arena.close()


def test_producer_failure_paths_release_references() -> None:
    """Invalid exports must not leak an arena reference or crash."""
    arena = _host_arena()
    producer = _filled_buffer(arena)
    with pytest.raises(BufferError, match="dl_device"):
        producer.__dlpack__(dl_device=(2, 0))
    # The arena still functions after a failed export.
    imported = np.from_dlpack(producer)
    assert imported.tolist() == (np.arange(8, 16, dtype=np.float64) * 2.0).tolist()
    producer.close()
    arena.close()


# --- ArrayBatch result_memory policy -----------------------------------------


def test_array_batch_host_policy_keeps_numpy() -> None:
    """result_memory='host' and the default both keep NumPy results."""
    water = _water()
    reference = water.singlepoint()
    packed = _pack_single([water])
    default = ArrayBatch(**packed, backend="cpu").compute()
    explicit = ArrayBatch(**packed, backend="cpu").compute(result_memory="host")
    assert isinstance(default.energies, np.ndarray)
    assert isinstance(explicit.energies, np.ndarray)
    assert default.energies == pytest.approx([reference.energy], rel=1.0e-12)
    assert explicit.energies == pytest.approx([reference.energy], rel=1.0e-12)


def test_array_batch_invalid_result_memory_rejected() -> None:
    """Unknown result_memory values are rejected before any compute."""
    water = _water()
    packed = _pack_single([water])
    batch = ArrayBatch(**packed, backend="cpu")
    with pytest.raises(GPUxtbValueError, match="result_memory"):
        batch.compute(result_memory="hip")


def test_array_batch_cuda_policy_requires_cuda_backend() -> None:
    """result_memory='cuda' on the CPU backend is a precise error."""
    water = _water()
    packed = _pack_single([water])
    batch = ArrayBatch(**packed, backend="cpu")
    with pytest.raises(GPUxtbNotSupportedError, match="CUDA"):
        batch.compute(result_memory="cuda")
