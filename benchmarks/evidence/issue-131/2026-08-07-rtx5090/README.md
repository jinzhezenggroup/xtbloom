# Issue #131 CUDA active-set eigensolver compaction evidence

This directory archives the missing capacity, correctness, latency, profiler,
and sanitizer evidence for issue #131 (a sub-issue of #80). The exact-capacity
compaction implementation landed earlier in `9c2a026`. This bundle measures
that implementation with the auditable runner at clean source revision
`4f8b075f0317ebab1d5f0a8fa116ec97289abd35`.

The claim is deliberately narrow: on the recorded RTX 5090/CUDA 12.9 stack,
the compacted Graph submits exactly the active count in each synthetic AO
bucket, preserves the uncompacted solve result, avoids provider arithmetic at
the empty tier, and has a measurable capacity-dependent crossover. These are
component measurements, not end-to-end SCC or release performance claims.

## Environment and build identity

- Host: `node3`, Ubuntu 22.04.5 LTS, Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62 48-Core Processor.
- GPU: NVIDIA GeForce RTX 5090 (GB202), `sm_120`, 32 GiB.
- Driver: 580.95.05; CUDA toolkit: 12.9.86.
- Compiler: GCC 11.4.0; CMake Release build; Ninja generator.
- Nsight Systems: 2025.1.3.140.
- Compute Sanitizer: 2025.2.1.0.
- LP64 runtime: `/group/software/deepmd-kit-3.1.1/lib/libmkl_rt.so`.
- Source revision: `4f8b075f0317ebab1d5f0a8fa116ec97289abd35`,
  clean immediately before the clean rebuild and benchmark run.

`build-metadata.txt` records the exact source revision and clean bit, CMake
cache identity, compiler/tool versions, GPU/driver, and SHA-256 hashes of the
benchmark binary, test binary, linked static library, and runner source. The
benchmark binary SHA-256 is
`15b3b9694ece0ad56e498e3b22fdc687c6747d2bb437a9733a9021c5bd60b74f`.

The clean rebuild used the existing `build/cuda-sm120` cache after confirming
these effective settings:

```text
CMAKE_BUILD_TYPE=Release
CMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc
CMAKE_CUDA_ARCHITECTURES=120
XTBLOOM_ENABLE_CUDA=ON
XTBLOOM_MKL_RT_LIBRARY=/group/software/deepmd-kit-3.1.1/lib/libmkl_rt.so
```

The two measured paths use the same stream and synchronization boundary.
Error resets are enqueued before the start event and excluded from timing.
Every CUDA event operation, Graph/provider enqueue, and asynchronous
completion is checked. The runner validates system/device error words and
outputs after timing; a failed launch cannot be emitted as a timing row.

## Workloads and raw samples

The synthetic, well-conditioned restricted eigensolver workloads are:

- homogeneous: one `n=32` AO bucket;
- heterogeneous: alternating `n=16` and `n=40` systems, producing two buckets
  for batch sizes 8/32/128;
- batch 1's `heterogeneous` coordinate necessarily contains one `n=16` bucket
  only and is retained as the small-matrix endpoint, not described as a
  two-bucket workload.

For batch sizes 1/8/32/128, the runner sweeps a deterministic canonical active
ladder from full to empty. Each tier first compares compacted and uncompacted
active-peer eigenvalues and coefficients and rechecks the generalized
residual. Inactive peer outputs must retain their sentinels. It then records 5
warmups and 50 measured samples for each path with CUDA events.

`compaction-capacity-trace.jsonl` is authoritative. It is true JSON Lines: one
protocol object followed by 44 measurement objects. Every measurement retains
all 50 compacted and all 50 uncompacted samples plus mean/min/p50/max,
per-bucket active/submitted/completed telemetry, and the correctness result.
`compaction-crossover-summary.csv` is a derived compact view. Its statistics
and all 4,400 raw samples were re-parsed and checked before archival.

## Results

Every one of the 44 measured tiers reports, per bucket:

```text
active == submitted_eigensolver == submitted_backtransform == completed
```

The sampled threshold at which the compacted path first becomes faster while
descending the active ladder is hardware/workload dependent:

| batch | workload | first sampled compacted win | compacted mean (us) | uncompacted mean (us) |
| ---: | --- | ---: | ---: | ---: |
| 1 | homogeneous n=32 | empty only | 21.705 | 345.359 |
| 1 | single n=16 endpoint | active 1 (100%) | 386.186 | 403.587 |
| 8 | homogeneous n=32 | active 2 (25%) | 859.663 | 868.536 |
| 8 | heterogeneous n=16/40 | active 1 (12.5%) | 400.495 | 932.803 |
| 32 | homogeneous n=32 | active 1 (3.125%) | 865.431 | 870.973 |
| 32 | heterogeneous n=16/40 | active 4 (12.5%) | 1672.631 | 1694.945 |
| 128 | homogeneous n=32 | empty only | 31.791 | 369.669 |
| 128 | heterogeneous n=16/40 | active 4 (3.125%) | 1678.230 | 1749.076 |

At the empty tier, the compacted Graph is 8.9x to 18.2x faster across the
recorded coordinates. At high residual activity, the uncompacted path is
usually faster because the exact-capacity SWITCH Graph contains one provider
body for every capacity. The data therefore supports tail compaction on this
stack, not unconditional high-activity compaction or an interpolated threshold
between the sampled tiers.

## Nsight Systems evidence

The retained CSVs are derived with node-level CUDA Graph tracing from a
heterogeneous B=32 profile over five replays:

- `nsys-active0-kern_sum.csv`: the ten always-on per-bucket compaction kernels
  execute (5 replays x 2 buckets), but no Graph-replayed gather, eigensolver,
  TRSM, symmetrize, validation, or scatter kernel executes. One-time overlap
  setup/provider kernels remain visible because setup is intentionally inside
  the same capture.
- `nsys-active16-kern_sum.csv`: each bucket has eight active systems; the
  compacted staging kernels and provider eigensolver/TRSM kernels each appear
  ten times (5 replays x 2 buckets).

The sanitized profiler stdout is retained as `nsys-active0-profile.log` and
`nsys-active16-profile.log`. Raw `.nsys-rep` and derived SQLite databases are
not committed because they can embed process environment data; the sanitized
kernel summaries are the archival artifacts.

## Compute Sanitizer evidence

All commands used `--error-exitcode=97`. The complete tool output is retained
in the named log, with trailing whitespace normalized for repository checks;
short clean logs are short because the tool emitted only its banner, target
success line, and zero-error summary.

| Tool | Target | Process exit | Result |
| --- | --- | ---: | --- |
| memcheck | full eigensolver test | 0 | 0 errors |
| initcheck | full eigensolver test | 0 | 0 errors |
| memcheck | Graph profile B=128, active=64, 20 loops | 0 | 0 errors |
| racecheck | same Graph profile | 11 | 0 hazards displayed; process did not terminate cleanly |
| synccheck | same Graph profile | 97 | reports only NVIDIA `batch_trsm_left_kernel` via `cuGraphLaunch` |
| racecheck | exact-capacity direct control | 0 | 0 hazards/errors/warnings |
| synccheck | exact-capacity direct control | 0 | 0 errors |

The Graph profile's canonical first 64 systems contain 32 `n=16` and 32
`n=40` systems. The direct control uses a fully active B=64 batch containing
the same first 64 synthetic matrices and the same two provider batch counts of
32, but launches the ordinary path directly on the stream. This corrects the
earlier control, which kept B=128 fixed and therefore submitted 64 identity-
placeholder systems per bucket on the direct path.

The clean direct control and lack of any xTBloom kernel in the Graph report are
differential evidence for a provider/Graph instrumentation boundary. They are
not represented as a literal four-tool pass or an NVIDIA-confirmed defect.
Issue #131 and parent #80 still require the recorded owner decision on whether
the narrow #130/#140/#141 waiver applies here.

## Reproduce

Benchmark matrix:

```bash
srun --gres=gpu:1 --cpus-per-task=4 env \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/deepmd-kit-3.1.1/lib \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
  build/cuda-sm120/xtbloom_cuda_compaction_benchmark \
  > compaction-capacity-trace.jsonl
```

Nsight profiles (repeat with active count 16):

```bash
srun --gres=gpu:1 --cpus-per-task=4 env \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/deepmd-kit-3.1.1/lib \
  /group/software/cuda-12.9.1/nsight-systems-2025.1.3/target-linux-x64/nsys profile \
  --trace=cuda --cuda-graph-trace=node --cuda-memory-usage=false \
  --output nsys-active0 \
  build/cuda-sm120/xtbloom_cuda_eigensolver_test \
  --compaction-profile 32 1 0 5
```

The sanitizer command prefix was:

```bash
srun --gres=gpu:1 --cpus-per-task=4 env \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/deepmd-kit-3.1.1/lib \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
  /group/software/cuda-12.9.1/bin/compute-sanitizer \
  --tool TOOL --error-exitcode=97 \
  build/cuda-sm120/xtbloom_cuda_eigensolver_test TARGET_ARGUMENTS
```

`TOOL`/`TARGET_ARGUMENTS` were `memcheck` or `initcheck` with no target
arguments for the full-test rows, `memcheck`/`racecheck`/`synccheck` with
`--compaction-profile 128 1 64 20` for Graph rows, and
`racecheck`/`synccheck` with `--direct-solve 64 1 20` for direct-control rows.

## Limitations

- Synthetic component buckets only; no full SCC iteration or public C API.
- One GPU, toolkit, driver, and canonical active ordering.
- AO dimensions are limited to 16/32/40.
- No Nsight Compute counters; profiler permission is restricted on this host.
- The production device-tail SCC loop does not currently integrate this
  exact-capacity conditional Graph; that remains a separate design decision.
