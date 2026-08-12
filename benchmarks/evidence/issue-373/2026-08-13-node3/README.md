# Issue #373 async strict-WARM steady-state evidence

This bundle records the narrow resource and synchronization claim required by
issue #373. After one synchronous `FRESH` setup and one asynchronous `WARM`
warmup, one reusable request executes ten strict-WARM
`xtbloom_compute_enqueue` / `xtbloom_request_wait` iterations. The captured
range contains no steady-state allocation/free, CUDA resource construction or
destruction, host progress polling, stream synchronization, or device-wide
synchronization.

This is not a latency, throughput, or release-wide performance claim. The
owner-cancelled issue #84/#220 matrix was not restored, and no Nsight Compute
counter claim is made.

## Source and environment

- Source revision: `05c6d723860e24e157e7e939e773ede3f586d26d`.
- Source branch: `feat/373-async-strict-warm`.
- Source state before and after measurement: clean.
- Host: `node3`, AMD EPYC 7K62 48-Core Processor.
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0.
- Driver: 580.95.05.
- CUDA compiler/toolkit: 12.9.86; `CMAKE_CUDA_ARCHITECTURES=120`.
- Host compiler: GCC 11.4.0; CMake 4.2.1.
- Build: shared Release, explicit CUDA ON, LP64 `scipy_openblas32` provider.
- Nsight Systems: 2025.1.3.140.
- Profile binary/library hashes: see `build-metadata.txt`.

## Workload and capture

`xtbloom_cuda_public_api_test --context-request-profile` constructs a
four-system H2/He/LiH/CH2 ragged restricted/unrestricted batch with QM/MM and
periodic inputs, mixed input descriptors, CUDA outputs, and host diagnostics.
It prepares the context cache with synchronous `FRESH`, consumes and republishes
one checkpoint through asynchronous strict `WARM`, then captures ten more
strict-WARM request iterations between `cudaProfilerStart()` and
`cudaProfilerStop()`. Final output is compared against the public CPU path, and
the caller result descriptor retains its canary.

The clean-commit capture used:

```bash
PROFILE_DIR="$(mktemp -d /tmp/xtbloom-issue373-final-profile.XXXXXX)"
srun --job-name=codex-373-final-nsys --partition=main \
  --nodes=1 --ntasks=1 --gpus=5090:1 --cpus-per-task=4 \
  --mem=16G --time=00:20:00 \
  env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:\
$PWD/.venv/lib/python3.13/site-packages/scipy_openblas32/lib \
  /group/software/cuda-12.9.1/bin/nsys profile \
    --trace=cuda --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-event-trace=false --force-overwrite=true \
    --output="$PROFILE_DIR/context-warm" \
    build/cuda-dev/xtbloom_cuda_public_api_test --context-request-profile
```

Derived summaries were extracted with:

```bash
/group/software/cuda-12.9.1/bin/nsys stats \
  --force-export=true --force-overwrite=true \
  --report cuda_api_sum,cuda_gpu_mem_time_sum --format csv \
  --output "$PROFILE_DIR/nsys-derived" \
  "$PROFILE_DIR/context-warm.nsys-rep"
```

Only sanitized derived summaries are retained. The raw `.nsys-rep` and
exported SQLite database were SHA-256 recorded in `steady-state-audit.txt` and
then deleted because native profiler captures may embed process-environment
metadata.

## Result

The ten captured WARM iterations contain exactly ten
`cudaEventSynchronize` and ten `cudaGraphLaunch_v10000` calls. They contain no
`cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaEventQuery`,
`cudaStreamQuery`, allocation/free API, or event/stream/Graph
construction/destruction call. Expected WARM execution traffic remains:
stream-ordered checkpoint reset/publication kernels, memsets, numerical
staging, host/device diagnostic copies, stream waits, event records, Graph
launches, and exact request completion waits.

See `steady-state-audit.txt` for the count ledger. The profile found no
steady-state blocker.

## Correctness qualification and limits

On the same RTX 5090 environment, the implementation passed the full shared
CUDA Release CTest matrix (133/133), full shared CPU Release CTest matrix
(59/59), focused context/plan WARM request tests, installed CUDA consumer,
memcheck, racecheck, initcheck, and the clean direct/host-Graph synccheck
control. The known device-Graph coordinate remains covered by the owner-approved
issue #279 disposition and was not rerun solely to reproduce that upstream
signature.

The node exposes one tested GPU. This bundle does not claim dual-GPU coverage;
issue #243 remains independent.
