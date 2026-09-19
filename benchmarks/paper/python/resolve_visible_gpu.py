#!/usr/bin/env python3
"""Resolve a CUDA-visible ordinal to its physical PCI bus identity."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path


def resolve_cudart(cuda_root: Path, explicit: Path | None = None) -> Path:
    """Return an explicit or toolkit-local CUDA runtime without assuming host ISA."""
    if explicit is not None:
        if not explicit.is_file():
            raise RuntimeError(f"configured libcudart is missing: {explicit}")
        return explicit.resolve()
    candidates = sorted((cuda_root / "targets").glob("*-linux/lib/libcudart.so*"))
    candidates.extend(sorted((cuda_root / "lib64").glob("libcudart.so*")))
    library_path = next((path for path in candidates if path.is_file()), None)
    if library_path is None:
        raise RuntimeError(f"libcudart.so is missing below {cuda_root}")
    return library_path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--cudart-library", type=Path)
    parser.add_argument("--device-id", type=int, required=True)
    args = parser.parse_args()
    if args.device_id < 0:
        raise RuntimeError("device ID must be a nonnegative visible CUDA ordinal")
    library = ctypes.CDLL(str(resolve_cudart(args.cuda_root, args.cudart_library)))
    library.cudaDeviceGetPCIBusId.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.cudaDeviceGetPCIBusId.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(64)
    status = library.cudaDeviceGetPCIBusId(buffer, len(buffer), args.device_id)
    if status != 0:
        raise RuntimeError(
            f"cudaDeviceGetPCIBusId failed for visible ordinal {args.device_id}: status={status}"
        )
    bus_id = buffer.value.decode("ascii").strip()
    if not bus_id:
        raise RuntimeError("CUDA runtime returned an empty PCI bus ID")
    print(bus_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
