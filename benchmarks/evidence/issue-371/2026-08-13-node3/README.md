# Issue #371 CUDA context-enqueue steady-state evidence

This bundle records the narrow resource and synchronization claim required by
issue #371. After context-cache setup and one asynchronous warmup, one reusable
request executes ten `xtbloom_compute_enqueue` / `xtbloom_request_wait`
iterations without steady-state allocation, CUDA resource construction or
destruction, host progress polling, stream synchronization, or device-wide
synchronization.

This is not a latency, throughput, or release-wide performance claim. The
owner-cancelled large issue #84/#220 matrix was not restored, and Nsight
Compute counters were not required.

## Source and environment

- Source revision: `81168b42adf5b12192970811e49fc11ca9d291c5`.
- Source branch: `feat/18-context-compute-enqueue`.
- Source state before and after measurement: clean.
- Host: `node3`, AMD EPYC 7K62 48-Core Processor.
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0.
- Driver: 580.95.05.
- CUDA compiler/toolkit: 12.9.86; `CMAKE_CUDA_ARCHITECTURES=120`.
- Host compiler: GCC 11.4.0; CMake 4.2.1.
- Build: shared Release, explicit CUDA ON, LP64 SciPy OpenBLAS provider.
- Nsight Systems: 2025.1.3.140.
- Profile binary/library hashes: see `build-metadata.txt`.

## Workload and capture boundary

`xtbloom_cuda_public_api_test --context-request-profile` constructs a
four-system H2/He/LiH/CH2 ragged restricted/unrestricted batch with QM/MM and
periodic inputs, mixed input descriptors, CUDA outputs, and host diagnostics.
It prepares the context cache synchronously, performs one asynchronous warmup,
restores the result-flags canary, then places exactly ten enqueue/wait/reuse
iterations between `cudaProfilerStart()` and `cudaProfilerStop()`. Final output
is compared against the public CPU path, and the caller result descriptor must
retain its canary.

The profile command was:

```bash
srun --job-name=codex-371-clean-nsys --gres=gpu:5090:1 \
  --cpus-per-task=4 --mem=16G --time=00:20:00 \
  env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:<scipy-lp64-dir> \
  /group/software/cuda-12.9.1/bin/nsys profile \
    --trace=cuda --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-event-trace=false --force-overwrite=true \
    --output=/tmp/<issue371-profile>/context-request-profile \
    build/cuda-dev/xtbloom_cuda_public_api_test --context-request-profile
```

Derived reports were extracted with:

```bash
/group/software/cuda-12.9.1/bin/nsys stats \
  --force-export=true --force-overwrite=true \
  --report cuda_api_sum,cuda_gpu_mem_time_sum --format csv \
  --output /tmp/<issue371-profile>/nsys-derived \
  /tmp/<issue371-profile>/context-request-profile.nsys-rep
```

Only sanitized derived summaries are retained. The raw `.nsys-rep` and
exported SQLite database were hash-pinned and deleted after extraction; they
are reproducible and may contain process-environment metadata.

## Result

The ten captured iterations contain exactly ten `cudaEventSynchronize` calls
and ten `cudaGraphLaunch_v10000` calls. They contain zero calls to
`cudaDeviceSynchronize`, `cudaStreamSynchronize`, event/stream query APIs,
allocation/free APIs, or event/stream/Graph construction/destruction APIs.

Expected execution traffic remains visible: numerical device-to-device
staging, small host/device diagnostics and publication copies, stream waits,
event records, Graph launches, kernel launches, and memsets. These are not
classified as resource churn or hidden synchronization.

See `steady-state-audit.txt` for the complete count ledger. The profile found
no blocker and makes no Nsight Compute counter claim.

## Correctness and runtime qualification

On the same RTX 5090 environment, the formatted final source passed the full
shared CUDA Release CTest matrix: 133/133, including public CUDA host, device,
and mixed conformance/invariant coordinates. Focused request tests, installed
CUDA consumer execution, memcheck, racecheck, and initcheck also passed. The
clean direct/host-Graph synccheck control passed with zero errors; the known
device-Graph coordinate remains covered by the owner-approved issue #279
disposition and was not rerun solely to reproduce that upstream signature.

The node exposes one tested GPU. This bundle does not claim dual-GPU coverage;
issue #243 remains independent.
