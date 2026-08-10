# Issue #279 CUDA request steady-state evidence

This bundle records the narrow runtime claim required by issue #279: after
setup and one warmup, reusing one fixed-topology CUDA plan/request slot for ten
submissions performs no steady-state allocation, CUDA resource creation or
destruction, progress-query polling, stream synchronization, or device-wide
synchronization. Each public `gpuxtb_request_wait` blocks on the request's
single completion event.

This is a resource/synchronization audit, not a latency or throughput claim.
The test still performs the expected numerical kernels, device copies, result
publication, and exact request completion waits.

## Source and environment

- Source revision: `99f235959dccb7084dd6d9db4b4f172e3b4ebcf8`.
- Source state when profiled: clean.
- Host: `node3`, Ubuntu 22.04.5 LTS, Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62 48-Core Processor.
- GPU: NVIDIA GeForce RTX 5090, UUID
  `GPU-8e9c9e1a-e183-258c-0b3a-03a5ddebb2f8`, 32,607 MiB, compute capability
  12.0.
- Driver: 580.95.05.
- CUDA compiler/toolkit: 12.9.86; `CMAKE_CUDA_ARCHITECTURES=120`.
- Host compiler: GCC 11.4.0; CMake 4.2.1; Ninja 1.13.0.
- Build: shared Release, `GPUXTB_ENABLE_CUDA=ON`, LP64 SciPy OpenBLAS at
  `/home/jzzeng/codes/gpuxtb-pr269-fix/.venv/lib/python3.13/site-packages/scipy_openblas32/lib/libscipy_openblas.so`.
- Nsight Systems: 2025.1.3.140.
- Compute Sanitizer: 2025.2.1.0, build 35969825.
- Profile binary SHA-256:
  `9af3bbb7d1d2d62673b5b09d63b52270ecb9cd4359de8bb51dc2108c1d931670`.
- Profile library SHA-256:
  `094b75dfbbad66f63722e479a9460f89040856018f65ebe1b206685877d44fe6`.

## Workload and capture boundary

`gpuxtb_cuda_public_api_test --request-profile` constructs and warms one
four-system H2/He/LiH/CH2 fixed-topology request path, including ragged
restricted/unrestricted systems, QM/MM, periodic inputs, mixed host/device
descriptors, CUDA tensor outputs, and host diagnostics. It then places exactly
ten enqueue/wait/reuse iterations between `cudaProfilerStart()` and
`cudaProfilerStop()` and prints `request_profile_iterations=10`.

The scheduler/profile command was:

```bash
srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \
  --gres=gpu:5090:1 --time=00:20:00 --job-name=codex-279-nsys-final2 \
  bash -lc '
    cd /home/jzzeng/codes/gpuxtb-pr279
    export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/home/jzzeng/codes/gpuxtb-pr269-fix/.venv/lib/python3.13/site-packages/scipy_openblas32/lib
    export MKL_INTERFACE_LAYER=LP64
    export MKL_THREADING_LAYER=SEQUENTIAL
    /group/software/cuda-12.9.1/bin/nsys profile \
      --force-overwrite=true \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda,osrt --cuda-memory-usage=true \
      --output=/tmp/gpuxtb-issue279-nsys-final2.jtjXN8/request-profile \
      build/request-cuda129/gpuxtb_cuda_public_api_test --request-profile
  '
```

The raw `.nsys-rep` and exported SQLite database remain outside the repository
under `/tmp`; they are intentionally not archived because profiler captures can
embed process environment data. Only the sanitized reports below were retained:

```bash
nsys stats --force-export=true --force-overwrite=true \
  --report cuda_api_sum --format csv \
  --output /tmp/gpuxtb-issue279-nsys-final2.jtjXN8/cuda_api_sum \
  /tmp/gpuxtb-issue279-nsys-final2.jtjXN8/request-profile.nsys-rep
nsys stats --force-export=true --force-overwrite=true \
  --report cuda_gpu_mem_time_sum --format csv \
  --output /tmp/gpuxtb-issue279-nsys-final2.jtjXN8/cuda_gpu_mem_time_sum \
  /tmp/gpuxtb-issue279-nsys-final2.jtjXN8/request-profile.nsys-rep
nsys stats --force-export=true --force-overwrite=true \
  --report cuda_gpu_mem_size_sum --format csv \
  --output /tmp/gpuxtb-issue279-nsys-final2.jtjXN8/cuda_gpu_mem_size_sum \
  /tmp/gpuxtb-issue279-nsys-final2.jtjXN8/request-profile.nsys-rep
```

## Result

The ten iterations contain exactly ten `cudaEventSynchronize` calls. The
captured API trace contains zero calls to `cudaStreamSynchronize`,
`cudaDeviceSynchronize`, `cudaEventQuery`, `cudaStreamQuery`, allocation/free
APIs, and event/stream/Graph create/destroy/instantiate APIs. The ten
`cudaStreamWaitEvent` calls are stream-ordered topology/publication dependencies,
not host polling. See `steady-state-audit.txt` and the retained CSV reports.

The final real-GPU CUDA CTest run on the same build passed 120/120. Focused
`--request-only`, `--request-sanitizer`, and `--request-profile` modes also
passed outside instrumentation.

## Compute Sanitizer disposition

The same production `--request-sanitizer` path was run under all four tools.
Memcheck, racecheck, and initcheck are clean. Synccheck reports 1,024 barrier
errors at `reduce_spin_atomic_charges_kernel+0x2b0` under device-launched Graph
replay. This is the exact stack, kernel, PC, and report class already archived
and owner-classified on issue #80 as an NVIDIA CUDA / Compute Sanitizer
device-Graph issue. The row remains `FAIL` and is not rewritten as a literal
four-tool pass; `sanitizer-summary.txt` records the exact result.

## Limitations

- The profile covers one prewarmed four-system mixed-descriptor request slot
  for ten repetitions. It proves the issue's steady-state resource boundary,
  not a release-wide performance claim.
- The hosted wheel jobs do not provide real-GPU runtime evidence; the results
  above are from the scheduler-allocated RTX 5090.
- Raw profiler and full synccheck logs are intentionally outside this bundle.
  The full matching synccheck signature and direct clean control are already
  archived under `benchmarks/evidence/issue-80/2026-08-08-node3/`.
