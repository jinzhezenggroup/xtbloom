"""Exercise direct CUDA-enabled DSO loading and the no-runtime fallback."""

from __future__ import annotations

import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

GPUXTB_STATUS_SUCCESS = 0
GPUXTB_STATUS_BACKEND_UNAVAILABLE = 2
GPUXTB_BACKEND_CUDA = 2


class ContextOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("backend", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        ("cpu_threads", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("stream", ctypes.c_void_p),
    ]


def configure_api(library: ctypes.CDLL) -> None:
    library.gpuxtb_context_options_init.argtypes = [
        ctypes.POINTER(ContextOptions),
        ctypes.c_size_t,
    ]
    library.gpuxtb_context_options_init.restype = ctypes.c_int32
    library.gpuxtb_context_create.argtypes = [
        ctypes.POINTER(ContextOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.gpuxtb_context_create.restype = ctypes.c_int32
    library.gpuxtb_context_destroy.argtypes = [ctypes.c_void_p]
    library.gpuxtb_context_destroy.restype = None


def exercise_unavailable_contexts(
    library: ctypes.CDLL, thread_count: int, calls_per_thread: int
) -> None:
    """Start CUDA API calls together after the DSO fallback cohort is ready."""

    configure_api(library)
    barrier = Barrier(thread_count)

    def worker() -> None:
        barrier.wait()
        for _ in range(calls_per_thread):
            options = ContextOptions()
            status = library.gpuxtb_context_options_init(
                ctypes.byref(options), ctypes.sizeof(options)
            )
            if status != GPUXTB_STATUS_SUCCESS:
                raise RuntimeError(f"context-options init returned {status}")
            options.backend = GPUXTB_BACKEND_CUDA
            context = ctypes.c_void_p()
            status = library.gpuxtb_context_create(
                ctypes.byref(options), ctypes.byref(context)
            )
            if status != GPUXTB_STATUS_BACKEND_UNAVAILABLE or context.value is not None:
                if context.value is not None:
                    library.gpuxtb_context_destroy(context)
                raise RuntimeError(
                    "forced no-runtime CUDA context creation returned "
                    f"status={status}, context={context.value!r}"
                )

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [executor.submit(worker) for _ in range(thread_count)]
        for future in futures:
            future.result()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument(
        "--mode", required=True, choices=("load", "concurrent-unavailable")
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--calls-per-thread", type=int, default=16)
    args = parser.parse_args()

    library = ctypes.CDLL(str(args.library), mode=ctypes.RTLD_LOCAL)
    if args.mode == "concurrent-unavailable":
        exercise_unavailable_contexts(library, args.threads, args.calls_per_thread)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
