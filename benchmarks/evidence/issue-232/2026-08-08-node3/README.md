# Issue #232 large-AO CUDA SCC early-stop evidence

This directory records the focused fix for the B=128, 62-atom public CUDA
regression exposed by PR #231. The claim is narrow: on the recorded RTX 5090
stack, binding the setup-owned eigensolver arena as the cuBLAS workspace keeps
the 122-AO batched TRSM capture compatible with device-launch CUDA Graphs. The
production SCC loop then stops after device-observed convergence instead of
falling back to all 500 configured iterations.

## Root cause and history

The slowdown was not an inherent fixed eigensolver cost. Without a caller
workspace, large batched `cublasDtrsmBatched` capture introduced cuBLAS-managed
allocation/free Graph nodes. Those nodes cannot be instantiated in the SCC
device-tail Graph, so production used its correctness-preserving bounded
fallback and submitted every configured SCC body.

- `73348cb` first added the CUDA eigensolver without `cublasSetWorkspace`,
  creating the latent provider-capture defect.
- `d105a1d` added the bounded production SCC loop, making the large-AO defect
  execute the complete iteration cap.
- `cbd81c2` added conditional-Graph early stop for supported cases, but large
  AO capture still contained unsupported allocator nodes.
- `5afccc1` replaced the conditional Graph with device-tail replay and made the
  unsupported-node fallback explicit; it did not create the root cause.

## Environment and build

- Host: `node3`, Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62 48-Core Processor.
- GPU: NVIDIA GeForce RTX 5090, `sm_120`, 32 GiB.
- Driver: 580.95.05; CUDA toolkit: 12.9.86.
- Compiler: GCC 11.4.0; CMake 4.2.1; Release shared build.
- LP64 runtime: `/group/software/deepmd-kit-3.1.1/lib/libmkl_rt.so`.
- Clean measured source: `2d5f67fa6327798c1f5d865b1c91bb19d64b5add`.
- Library SHA-256: `12874d2a4b65c9b89b66a378ee0140b1ea9609ff35b93ae226de87fee5c6b8a8`.

The CUDA build used `GPUXTB_ENABLE_CUDA=ON`,
`CMAKE_CUDA_ARCHITECTURES=120`, `BUILD_SHARED_LIBS=ON`, and
`CMAKE_BUILD_TYPE=Release`. `cuda-b128.json` retains the complete CMake cache,
compiler, provider, library, source, and clean-worktree identities checked by
the benchmark runner.

## Performance and correctness

The clean-head run used the public C ABI, FRESH SCC, energy publication, five
warmups, and 50 measured calls. Each call contained 128 copies of the
deterministic 62-atom alkane coordinate and retained all raw energies,
iteration counts, and latencies.

| Metric | Result |
| --- | ---: |
| Median latency | 589.669684 ms |
| Mean latency | 589.737273 ms |
| Minimum latency | 589.432652 ms |
| p95 latency | 589.977304 ms |
| Median throughput | 217.070681 systems/s |
| SCC iterations | 18 for every system in every sample |
| Measured samples | 50 |
| Within-run energy drift | 0 Ha |

The issue checkpoint measured the broken main path at approximately
4704.14 ms and the CPU16 path at approximately 633.12 ms. Relative to that
reproduction, the fixed clean-head median is 7.98x faster (87.46% lower
latency) and 1.07x faster than CPU16. The historical numbers were diagnostic
runs and are not represented as additional clean-head artifact rows here.

The production device counter is reset before each launch and incremented once
inside each numerical body. The registered SCC production matrix reads that
counter and requires it to equal the maximum peer iteration count for
B=1/8/32/128, including terminal replay and failed-peer cases. It passed on
this build. Combined with the public result above, the submitted numerical
body count for this coordinate is 18, not the configured bound of 500.

## Nsight Systems

`nsys-fixed_cuda_api_sum.csv` and `nsys-fixed_cuda_gpu_kern_sum.csv` come from
one clean-head public B=128 call with node-level CUDA Graph tracing. The API
summary records one host `cudaGraphLaunch`, one `cudaGraphUpload`, and two Graph
instantiations (the body and root executables). It does not contain the former
500 host-side bounded submissions.

Nsight Systems 2025.1.3 does not expand fire-and-forget or tail device Graph
launches into per-replay child rows, so the kernel summary is not used to
claim an exact nested-body count. The exact 18-body result comes from the
production device counter described above. Raw `.nsys-rep` and derived SQLite
files are not committed because they can contain process-environment data.

## Compute Sanitizer

Compute Sanitizer 2025.2.1.0 used `--error-exitcode=99`.

| Tool | Target | Result |
| --- | --- | --- |
| memcheck | full eigensolver test, including B=128/122-AO device-launch capture | 0 errors |
| initcheck | same full test | 0 errors |
| racecheck | B=128 direct provider control, two solves | 0 hazards, errors, or warnings |
| synccheck | B=128 direct provider control, two solves | 0 errors |

On this CUDA/driver combination, `racecheck` exits the target before a
device-launch Graph test begins, while `synccheck` reports only NVIDIA provider
kernels when they execute through `cuGraphLaunch`. The same instrumentation
boundary and clean direct controls are independently archived under issue
#131. The two direct controls here exercise the changed cuBLAS configuration
on an ordinary stream; the full memcheck/initcheck rows exercise the new
122-AO Graph regression.

## Validation

- Fresh RTX 5090 CUDA CTest: 112/112 passed, including public host/device/mixed
  conformance and CUDA ABI loader checks.
- Post-format focused CUDA CTest: 6/6 passed for eigensolver, SCC production,
  runtime Graph, and loader/`DT_NEEDED` checks.
- Fresh shared CPU Release CTest with the LP64 provider: 41/41 passed.
- `prek@0.3.1`, `git diff --check`, source license check, and 20 licensing unit
  tests passed.
- `uv lock --check` reports that the unchanged `origin/main` lock file needs an
  update; neither `pyproject.toml` nor `uv.lock` differs on this branch.

## Reproduce

```bash
srun --partition=main --gres=gpu:5090:1 --ntasks=1 env \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/deepmd-kit-3.1.1/lib \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
  python3 benchmarks/natoms_scaling.py \
  --library build/cuda-fix232/libgpuxtb.so \
  --output-json cuda-b128.json --output-csv cuda-b128.csv \
  --natoms 62 --batch-sizes 128 --backend cuda --property energy \
  --start-mode fresh --warmups 5 --repetitions 50
```

The Nsight command used the same environment and workload with
`--warmups 0 --repetitions 1`, prefixed by:

```text
nsys profile --trace=cuda --sample=none --cuda-graph-trace=node \
  --cuda-memory-usage=false --output nsys-fixed
```
