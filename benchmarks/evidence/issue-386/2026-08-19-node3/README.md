# Issue #386 GFN1 CUDA steady-state evidence

This bundle records the narrow steady-state and sanitizer claims required by
issue #386. After fixed-plan setup, one `FRESH` seed, and one strict-WARM
warmup, one reusable public request executes ten strict-WARM
`xtbloom_plan_compute_enqueue` / `xtbloom_request_wait` iterations. The
captured range contains no allocation/free, CUDA resource construction or
destruction, host query polling, stream synchronization, or device-wide
synchronization.

This is a resource and synchronization audit, not a latency, throughput, or
release-wide performance claim. Expected request completion waits, device
diagnostic copies, numerical kernels, and result publication remain present.

## Source and environment

- Source revision: `1c9c5b7264e522cde4b875d73d254470ad1e5116`.
- Source branch: `review/439-gfn1-cuda`.
- Source state before and after measurement: clean.
- Host: `node3`, Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62 48-Core Processor, 48 physical cores.
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0.
- Driver: 580.95.05.
- CUDA compiler/toolkit: 12.9.86; `CMAKE_CUDA_ARCHITECTURES=120`.
- Host compiler: GCC 11.4.0; CMake 4.2.1; Ninja 1.13.0.
- Build: shared Release, explicit CUDA ON, LP64 `scipy-openblas32` provider.
- Nsight Systems: 2025.1.3.140.
- Compute Sanitizer: 2025.2.1.0, build 35969825.
- Profile binary/library hashes: see `build-metadata.txt`.

## Workload and capture

`xtbloom_cuda_public_api_test --gfn1-profile` constructs a four-system
H2/He/LiH/CH2 GFN1 ragged batch. The CH2 triplet exercises unrestricted scalar
SCC beside restricted peers. Inputs are mixed host/device descriptors, outputs
are CUDA-device descriptors, and one fixed plan and request are reused.

CPU qualification, plan creation, the `FRESH` cache seed, the first strict-WARM
call, correctness downloads, input-identity checks, result canaries, and guard
checks all execute outside the measured range. Exactly ten further strict-WARM
enqueue/wait calls execute between `cudaProfilerStart()` and
`cudaProfilerStop()` and the test prints `gfn1_profile_iterations=10`.

The clean-commit capture used:

```bash
srun --job-name=codex-386-final-nsys --partition=main --nodelist=node3 \
  --nodes=1 --ntasks=1 --gpus=5090:1 --cpus-per-task=4 \
  --mem=16G --time=00:20:00 \
  env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:\
$PWD/.venv/lib/python3.13/site-packages/scipy_openblas32/lib \
  /group/software/cuda-12.9.1/bin/nsys profile \
    --trace=cuda --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-event-trace=false --force-overwrite=true \
    --output=/tmp/xtbloom-issue386-final-profile.wlmgW9/gfn1-warm \
    build/nox/cuda/xtbloom_cuda_public_api_test --gfn1-profile
```

Derived summaries were extracted with:

```bash
/group/software/cuda-12.9.1/bin/nsys stats \
  --force-export=true --force-overwrite=true \
  --report cuda_api_sum,cuda_gpu_mem_time_sum --format csv \
  --output /tmp/xtbloom-issue386-final-profile.wlmgW9/nsys-derived \
  /tmp/xtbloom-issue386-final-profile.wlmgW9/gfn1-warm.nsys-rep
```

Only the sanitized derived summaries are retained. The raw `.nsys-rep` and
exported SQLite database were hashed, recorded in `build-metadata.txt`, and
deleted because native profiler captures may contain process-environment data.

## Steady-state result

The ten captured WARM requests contain exactly ten `cudaEventSynchronize` and
ten `cudaGraphLaunch_v10000` calls. They contain zero
`cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaEventQuery`,
`cudaStreamQuery`, allocation/free APIs, or event/stream/Graph
construction/destruction APIs.

The twenty device-to-host copies are two fixed diagnostic/completion transfers
per public request. There are no host-to-device transfers and no host transfer
or progress poll inside an SCC iteration. This preserves the existing public
request boundary without introducing iteration-level host traffic. See
`steady-state-audit.txt` for the complete count ledger.

## Sanitizer result

The production `--gfn1-sanitizer` path was run under all four Compute
Sanitizer tools on the same clean revision. Memcheck and initcheck report zero
errors; racecheck reports zero hazards, errors, and warnings.

Synccheck exits 99 and reports 256 barrier errors, with all 100 printed reports
matching only `reduce_spin_atomic_charges_kernel+0x2b0` under device-launched
Graph execution. The GPU, driver, toolkit, tool build, kernel, program counter,
report class, and launch path exactly match the owner-approved issue #279
Blackwell disposition. The direct plus host-Graph SCC-potential control reports
zero errors. This row is `PASS (OWNER-DISPOSITIONED UPSTREAM TOOL DEFECT)`; the
tool itself did not return a clean pass. See `sanitizer-summary.txt` for exact
commands, exit codes, counts, hashes, and classification.

## Correctness qualification and limits

The same integrated source passed the real-GPU CUDA Release CTest matrix
179/179 with zero skips, including GFN1 host/device/mixed conformance,
invariants, restricted/unrestricted execution, FRESH/WARM, failure publication,
and complete released GFN2 CUDA regression coverage. The GPU-backed installed
Python adapter matrix passed 106/106. The complete CPU, Python, canonical,
package, install-consumer, sdist, wheel, licensing, and benchmark-harness gates
also passed before capture.

Independent scientific/CUDA review found no correctness, ABI, runtime, or
payload finding in the measured source tree. This bundle covers one prewarmed
four-system mixed-descriptor GFN1 request for ten repetitions. It does not make
a latency claim, a multi-GPU claim, or a claim about unsupported ROCm behavior.
