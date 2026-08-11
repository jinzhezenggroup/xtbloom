#!/usr/bin/env python3
"""Compare public CUDA plan workspace accounting for issue #220 binaries."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

# This script is archived four directories below the repository root.
SOURCE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT / "tools" / "conformance"))

from benchmarks import natoms_scaling  # noqa: E402

public_api = natoms_scaling.public_api


class WorkspaceQuery(ctypes.Structure):
    """ctypes mirror of ``xtbloom_workspace_query_t`` ABI version 1."""

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


def configure_plan_api(library: ctypes.CDLL) -> None:
    """Declare only the public fixed-plan calls needed by this diagnostic."""
    library.xtbloom_workspace_query_init.argtypes = [
        ctypes.POINTER(WorkspaceQuery),
        ctypes.c_size_t,
    ]
    library.xtbloom_workspace_query_init.restype = ctypes.c_int32
    library.xtbloom_plan_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(public_api.Batch),
        ctypes.POINTER(public_api.ComputeOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.xtbloom_plan_create.restype = ctypes.c_int32
    library.xtbloom_plan_query_workspace.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(WorkspaceQuery),
    ]
    library.xtbloom_plan_query_workspace.restype = ctypes.c_int32
    library.xtbloom_plan_destroy.argtypes = [ctypes.c_void_p]
    library.xtbloom_plan_destroy.restype = None


def query(library_path: Path, natoms: int, batch_size: int, device_id: int) -> dict:
    """Create one public force plan and return its retained workspace bytes."""
    library = public_api._configure_library(library_path)
    configure_plan_api(library)
    molecule = natoms_scaling.make_alkane(natoms)
    storage = natoms_scaling.make_storage(molecule, batch_size)
    context = public_api._make_context(library, "cuda", device_id, 1)
    memory = public_api.DescriptorMemory("host", device_id)
    plan = ctypes.c_void_p()
    try:
        batch = public_api._make_batch(
            library, storage, memory, include_spin_channels=True
        )
        options = public_api.ComputeOptions()
        public_api._call_ok(
            library,
            library.xtbloom_compute_options_init(
                ctypes.byref(options), ctypes.sizeof(options)
            ),
            "xtbloom_compute_options_init",
        )
        flags = public_api.XTBLOOM_COMPUTE_ENERGY | public_api.XTBLOOM_COMPUTE_FORCES
        options.model = public_api.XTBLOOM_MODEL_GFN2_XTB
        options.flags = flags
        natoms_scaling.configure_xtbloom_conformance_scc(options)
        public_api._call_ok(
            library,
            library.xtbloom_plan_create(
                context, ctypes.byref(batch), ctypes.byref(options), ctypes.byref(plan)
            ),
            "xtbloom_plan_create",
        )
        workspace = WorkspaceQuery()
        public_api._call_ok(
            library,
            library.xtbloom_workspace_query_init(
                ctypes.byref(workspace), ctypes.sizeof(workspace)
            ),
            "xtbloom_workspace_query_init",
        )
        workspace.compute_flags = flags
        public_api._call_ok(
            library,
            library.xtbloom_plan_query_workspace(plan, ctypes.byref(workspace)),
            "xtbloom_plan_query_workspace",
        )
        packed_pairs = batch_size * natoms * (natoms - 1) // 2
        return {
            "natoms_per_system": natoms,
            "batch_size": batch_size,
            "total_atoms": natoms * batch_size,
            "packed_pairs": packed_pairs,
            "host_required_bytes": int(workspace.host_required_bytes),
            "host_required_alignment": int(workspace.host_required_alignment),
            "device_required_bytes": int(workspace.device_required_bytes),
            "device_required_alignment": int(workspace.device_required_alignment),
        }
    finally:
        if plan.value:
            library.xtbloom_plan_destroy(plan)
        memory.close()
        library.xtbloom_context_destroy(context)


def main() -> int:
    """Measure both binaries in separate public contexts and write JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    cells = [(32, 1), (62, 1), (122, 1), (242, 1), (272, 1), (32, 128)]
    rows = []
    for natoms, batch_size in cells:
        baseline = query(args.baseline.resolve(), natoms, batch_size, args.device_id)
        candidate = query(args.candidate.resolve(), natoms, batch_size, args.device_id)
        dense_bytes_per_copy = (
            5 * baseline["packed_pairs"] * ctypes.sizeof(ctypes.c_double)
        )
        baseline["dense_d4_pair_cache_bytes_per_copy"] = dense_bytes_per_copy
        baseline["baseline_dense_d4_retained_copies"] = 3
        baseline["baseline_dense_d4_retained_bytes"] = 3 * dense_bytes_per_copy
        before = baseline["device_required_bytes"]
        after = candidate["device_required_bytes"]
        delta = before - after
        rows.append(
            {
                "workload": {"natoms_per_system": natoms, "batch_size": batch_size},
                "baseline": baseline,
                "candidate": candidate,
                "device_bytes_saved": delta,
                "device_percent_saved": 100.0 * delta / before,
                "saved_minus_baseline_dense_d4_retained_bytes": (
                    delta - baseline["baseline_dense_d4_retained_bytes"]
                ),
            }
        )
    document = {
        "schema_version": 1,
        "claim": "public CUDA force-plan retained workspace before/after issue #220",
        "baseline_library": str(args.baseline.resolve()),
        "candidate_library": str(args.candidate.resolve()),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
