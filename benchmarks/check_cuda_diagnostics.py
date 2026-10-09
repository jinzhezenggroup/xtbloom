"""Exercise repeated public SCC capture, including native-periodic overrides."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

from benchmarks import cuda_diagnostics

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFORMANCE_TOOLS = REPOSITORY_ROOT / "tools" / "conformance"
if str(CONFORMANCE_TOOLS) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_TOOLS))
conformance = importlib.import_module("xtbloom_conformance")
public_api = importlib.import_module("xtbloom_public_api")


def check_capture(library_path: Path, memory_mode: str, device_id: int) -> None:
    """Compare captured canonical state against caller-visible results.

    A short iteration cap intentionally permits nonconvergence. This gate
    checks capture/publication semantics, not independent scientific accuracy;
    the ordinary full conformance gates retain their original tolerances.
    """
    library = public_api._configure_library(library_path)
    root = Path(__file__).resolve().parents[1]
    periodic_path = root / "data/conformance/periodic/manifest.json"
    gas_path = conformance.DEFAULT_MANIFEST
    periodic_manifest = conformance.load_json(periodic_path)
    gas_manifest = conformance.load_json(gas_path)
    periodic_case = public_api.supported_cases(
        periodic_manifest, ["periodic_neon_one_atom"], "cuda"
    )[0]
    gas_case = public_api.supported_cases(gas_manifest, ["ketene"], "cuda")[0]
    options = public_api.pinned_compute_options(
        library,
        public_api.XTBLOOM_MODEL_GFN2_XTB,
        request_forces=True,
        request_charges=True,
        request_point_forces=False,
        request_dipoles=False,
    )
    options.max_scc_iterations = 4
    context = public_api._make_context(library, "cuda", device_id, 1)
    capture_variable = "XTBLOOM_CUDA_SCC_DIAGNOSTICS_FILE"
    previous_capture = os.environ.get(capture_variable)
    expected = []
    try:
        with tempfile.TemporaryDirectory(prefix="xtbloom-scc-capture-") as directory:
            capture_path = Path(directory) / "calls.jsonl"
            os.environ[capture_variable] = str(capture_path)
            # Reuse each topology before changing it, then revisit periodic
            # input. Neither retained owner family nor old trace may set mode.
            for periodic in (True, True, False, False, True):
                manifest_path, manifest, case = (
                    (periodic_path, periodic_manifest, periodic_case)
                    if periodic
                    else (gas_path, gas_manifest, gas_case)
                )
                storage = public_api.assemble_batch(manifest_path, manifest, [case])
                atoms = len(storage.atomic_numbers)
                energies = (ctypes.c_double * 1)()
                forces = (ctypes.c_double * (3 * atoms))()
                charges = (ctypes.c_double * atoms)()
                iterations = (ctypes.c_int32 * 1)()
                converged = (ctypes.c_uint8 * 1)()
                statuses = (ctypes.c_int32 * 1)()
                with public_api.DescriptorMemory(memory_mode, device_id) as memory:
                    batch = public_api._make_batch(
                        library, storage, memory, include_spin_channels=True
                    )
                    result = public_api.BatchResult()
                    public_api._call_ok(
                        library,
                        library.xtbloom_batch_result_init(
                            ctypes.byref(result), ctypes.sizeof(result)
                        ),
                        "initialize capture result",
                    )
                    for field, values in (
                        ("energies", energies),
                        ("forces", forces),
                        ("atomic_charges", charges),
                        ("scc_iterations", iterations),
                        ("scc_converged", converged),
                        ("per_system_status", statuses),
                    ):
                        setattr(result, field, memory.output(values, field))
                    public_api._call_ok(
                        library,
                        library.xtbloom_compute(
                            context,
                            ctypes.byref(batch),
                            ctypes.byref(options),
                            ctypes.byref(result),
                        ),
                        "repeated public capture",
                    )
                    memory.download_outputs()
                for values in (energies, forces, charges):
                    if statuses[0] == public_api.XTBLOOM_STATUS_SUCCESS:
                        if not all(math.isfinite(value) for value in values):
                            raise AssertionError("successful peer has nonfinite output")
                    else:
                        bits = struct.unpack(
                            f"{len(values)}Q",
                            ctypes.string_at(values, ctypes.sizeof(values)),
                        )
                        if not all(
                            word & 0x7FF0000000000000 == 0x7FF0000000000000
                            and word & (1 << 51)
                            for word in bits
                        ):
                            raise AssertionError(
                                "failed peer is not fully quiet-NaN published"
                            )
                expected.append(
                    (periodic, int(iterations[0]), int(statuses[0]), int(converged[0]))
                )
            with capture_path.open(encoding="utf-8") as stream:
                captured = list(cuda_diagnostics.iter_jsonl(stream))
            if len(captured) != len(expected):
                raise AssertionError(
                    "public calls are missing or duplicated in capture"
                )
            gas_modes = []
            for call, (periodic, iterations, status, converged) in zip(
                captured, expected, strict=True
            ):
                actual_mode = call["execution_mode"]
                if call["start_policy"] != 1 or (
                    periodic and actual_mode != "bounded_fallback"
                ):
                    raise AssertionError(
                        "capture reports prepared, not executed, SCC identity"
                    )
                if not periodic:
                    if actual_mode not in ("device_tail_graph", "bounded_fallback"):
                        raise AssertionError("singleton selected an unsupported mode")
                    gas_modes.append(actual_mode)
                if actual_mode == "bounded_fallback" and call["fallback_reason"] == 0:
                    raise AssertionError("bounded execution lacks a fallback reason")
                if call["terminal_systems"] != [
                    {
                        "system_index": 0,
                        "iterations": iterations,
                        "status": status,
                        "converged": converged,
                    }
                ]:
                    raise AssertionError(
                        "captured terminal state differs from public publication"
                    )
                if (
                    actual_mode == "bounded_fallback"
                    and len(call["iterations"]) != options.max_scc_iterations
                ):
                    raise AssertionError(
                        "bounded inactive provider submissions were hidden"
                    )
                if (
                    actual_mode == "device_tail_graph"
                    and len(call["iterations"]) != iterations
                ):
                    raise AssertionError(
                        "device-tail replay retained a prior call's body count"
                    )
            if len(set(gas_modes)) != 1:
                raise AssertionError(
                    "same-topology replay changed execution capability"
                )
    finally:
        library.xtbloom_context_destroy(context)
        if previous_capture is None:
            os.environ.pop(capture_variable, None)
        else:
            os.environ[capture_variable] = previous_capture


def main() -> int:
    """Run the explicit CUDA capture gate or return CTest's unavailable code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--memory-mode", choices=("host", "device", "mixed"), default="host"
    )
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    try:
        check_capture(args.library, args.memory_mode, args.device_id)
    except public_api.BackendUnavailable as error:
        print(f"CUDA unavailable: {error}")  # noqa: T201 - CTest diagnostic
        return 77
    print(  # noqa: T201 - CTest diagnostic
        f"Five repeated public CUDA/{args.memory_mode} captures passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
