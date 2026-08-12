# Issue #128 steady-state CUDA evidence

This bundle records the compact final-revision performance and profiler
evidence requested by issue #128. The measured source revision is the exact
`main` commit accepted by #127:
`56afcad36d05aa4ac6f3c721d78f09e39c56069b`.

The narrow conclusions are:

- all eight RTX 5090 latency/throughput coordinates passed the committed
  independent conformance golden;
- one prewarmed fixed-plan request reused the same Graph and resources for ten
  iterations without steady-state allocation, resource creation/destruction,
  host progress polling, stream synchronization, or device-wide
  synchronization;
- the captured steady state has no SCC-iteration host round trip or topology
  descriptor rebuild; the expected once-per-request numerical D2D staging and
  diagnostics/publication D2H traffic remains visible and is reported below;
- the dominant profiled GPU work is pair-list commit/preflight, followed by
  D4 weight preparation and integral-force evaluation; and
- Nsight Compute counters are `UNAVAILABLE` because the single attempted run
  returned `ERR_NVGPUCTRPERM`. No privileged retry was requested, and no NCU
  counter result is reported as a pass.

This is an xTBloom CUDA baseline on one machine, not an optimization delta or
a cross-library performance claim. The owner-cancelled large #84/#220
Cartesian sweep was not restored.

## Source and environment

- Source revision: `56afcad36d05aa4ac6f3c721d78f09e39c56069b`.
- Source state before and after measurement: clean; generated outputs were
  written outside the repository until the measurements finished.
- Host: `node3`, AMD EPYC 7K62 48-Core Processor.
- GPU: NVIDIA GeForce RTX 5090, UUID
  `GPU-8e9c9e1a-e183-258c-0b3a-03a5ddebb2f8`, 32,607 MiB, compute capability
  12.0.
- Driver: 580.95.05.
- CUDA compiler/toolkit: 12.9.86; `CMAKE_CUDA_ARCHITECTURES=120`.
- Host compiler: GCC 11.4.0; CMake 4.2.1; Ninja 1.13.0.
- Build: shared Release, explicit CUDA ON, LP64 MKL runtime at
  `/group/software/deepmd-kit-3.1.1/lib/libmkl_rt.so.2`.
- Nsight Systems: 2025.1.3.140.
- Nsight Compute: 2025.2.1.0, build 35987062.
- Compute Sanitizer: 2025.2.1.0, build 35969825.
- Library SHA-256:
  `6e1b9f3558199627eab29766ad410bf1180b9a30dbabf93ab93a77da773c9f5a`.
- Profile binary SHA-256:
  `472938cc9660762b23fd955f9b66f1ee84c679925f90e4e659eb9694e3b4125f`.

`build-metadata.txt` retains the remaining build/input hashes and the hashes
of omitted reproducible raw artifacts.

## Correctness-qualified latency and throughput

`benchmarks/run.py` measured one representative homogeneous gas-phase ketene
workload through the public CUDA API with device descriptors, FRESH SCC,
three warmups, and twenty samples per cell. Setup, the first cold call, and
steady-state calls are recorded separately. Every measured interval includes
the documented CUDA synchronization boundary; correctness downloads remain
outside timing.

| Property | Batch | Median (ms) | p95 (ms) | Systems/s at median | Correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| energy | 1 | 12.471125 | 15.999606 | 80.185 | PASS |
| energy | 8 | 16.036390 | 18.996680 | 498.865 | PASS |
| energy | 32 | 21.481499 | 24.825991 | 1489.654 | PASS |
| energy | 128 | 45.785573 | 45.820252 | 2795.640 | PASS |
| force | 1 | 14.121573 | 16.928761 | 70.814 | PASS |
| force | 8 | 16.907016 | 19.858461 | 473.176 | PASS |
| force | 32 | 23.787194 | 26.039080 | 1345.262 | PASS |
| force | 128 | 47.834845 | 47.881810 | 2675.874 | PASS |

All cells converged in 12 SCC iterations. The maximum absolute energy error
was `1.7763568394002505e-15` Hartree. Force cells had maximum absolute force
error at most `6.704153887940323e-08` Hartree/bohr, within the pinned public
conformance tolerance. `latency-throughput.csv` retains all 160 timing samples,
setup/cold timings, correctness errors, and memory snapshots.
`performance-summary.csv` is a compact derivation of the omitted runner JSON
and preserves each row's successful per-system status, convergence state, SCC
iteration range, and sample count.

The exact command was:

```bash
PROFILE_TMP=$(mktemp -d /tmp/xtbloom-issue128.XXXXXX)
env -u CUDA_VISIBLE_DEVICES \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
  python3 benchmarks/run.py \
    --library "$PWD/build/issue128-cuda/libxtbloom.so" \
    --engines xtbloom --backends cuda --cuda-memory-modes device \
    --workloads gas --batch-sizes 1,8,32,128 \
    --properties energy,force --warmups 3 --repetitions 20 \
    --cpu-threads 1 --device-id 0 \
    --cuda-root /group/software/cuda-12.9.1 \
    --fail-on-correctness \
    --output-json "$PROFILE_TMP/latency-throughput.json" \
    --output-csv "$PROFILE_TMP/latency-throughput.csv"
```

The complete JSON was omitted because it repeats the CSV samples while also
embedding local absolute paths and the process environment. Its SHA-256 and
byte count are retained in `build-metadata.txt`; omitting it did not reduce the
matrix or sample distribution.

## Fixed-plan changed-geometry coordinate

One intentionally small coordinate separates plan setup and the first original
geometry from changed-geometry warm execution without restoring the cancelled
large Cartesian sweep. A clean detached worktree at the measured revision ran
batch 4 ketene forces with device descriptors and a fixed CUDA plan. After
three changed-geometry warmups, twenty distinct position updates produced:

| Scope | Result |
| --- | ---: |
| Adapter/context/descriptor setup | 344.518457 ms |
| Plan creation | 44.023809 ms |
| First original-geometry plan compute | 17.090005 ms |
| Changed-geometry warm median | 16.421544 ms |
| Changed-geometry warm p95 | 19.337331 ms |
| Throughput at median | 243.582 systems/s |

Every sample reported `SUCCESS`, converged in 12 SCC iterations, and retained
the same device position-descriptor address. The final changed geometry was
independently recomputed through the public CPU path: maximum CUDA/CPU energy
error was `2.4868995751603507e-14` Hartree and maximum force error was
`4.288236432614667e-15` Hartree/bohr.

The synchronous H2D position update occurs before and is excluded from each
timed interval; each interval is `xtbloom_plan_compute` plus explicit
`cudaDeviceSynchronize`. `changed-geometry.json` retains the twenty raw
samples and correctness record. `changed-geometry-plan.py` is the exact
reproducer; it requires a separately supplied clean source root so running the
script from this evidence-bearing branch cannot make the measured source dirty.

The reproduction command was:

```bash
git worktree add --detach /tmp/xtbloom-issue128-clean-worktree \
  56afcad36d05aa4ac6f3c721d78f09e39c56069b

env -u CUDA_VISIBLE_DEVICES \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
  python3 benchmarks/evidence/issue-128/2026-08-12-node3/changed-geometry-plan.py \
    --source-root /tmp/xtbloom-issue128-clean-worktree \
    --library build/issue128-cuda/libxtbloom.so \
    --output-json benchmarks/evidence/issue-128/2026-08-12-node3/changed-geometry.json \
    --device-id 0
```

## Fixed-plan steady-state profile

`xtbloom_cuda_public_api_test --request-profile` constructs and warms one
four-system H2/He/LiH/CH2 fixed-topology request with ragged
restricted/unrestricted systems, QM/MM and periodic inputs, mixed descriptors,
device outputs, and host diagnostics. Exactly ten enqueue/wait/reuse
iterations occur between `cudaProfilerStart()` and `cudaProfilerStop()`.
Final numerical output is compared against the public CPU path.

The ten iterations contain exactly:

- 10 `cudaGraphLaunch` calls;
- 10 `cudaEventSynchronize` calls, one for each public request wait;
- 0 `cudaStreamSynchronize` and 0 `cudaDeviceSynchronize` calls;
- 0 `cudaEventQuery` and 0 `cudaStreamQuery` calls;
- 0 CUDA allocation/free calls;
- 0 event, stream, or Graph create/destroy/instantiate calls.

The retained memory summaries show expected steady-state publication traffic:
10 device-to-device copies totaling 12.032 MB, 20 small device-to-host copies,
and no host-to-device copy in the captured range. `steady-state-audit.txt`
contains the complete acceptance count ledger.

The exact profile and extraction commands were:

```bash
env -u CUDA_VISIBLE_DEVICES \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
  /group/software/cuda-12.9.1/bin/nsys profile \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --trace=cuda,osrt --sample=none --cuda-memory-usage=true \
    --output="$PROFILE_TMP/request-profile" \
    build/issue128-cuda/xtbloom_cuda_public_api_test --request-profile

/group/software/cuda-12.9.1/bin/nsys stats \
  --force-export=true --force-overwrite=true \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv --output "$PROFILE_TMP/nsys-derived" \
  "$PROFILE_TMP/request-profile.nsys-rep"
```

Only the reviewed derived CSV reports are retained. The raw `.nsys-rep` and
exported SQLite database were hash-pinned and deleted after extraction; they
are neither committed nor externally archived because the exact capture is
readily reproducible and can contain process-environment metadata.

The largest GPU-time rows in this four-system force profile are:

| Kernel | GPU time share | Instances | Median (us) |
| --- | ---: | ---: | ---: |
| pair-list commit | 22.0% | 10 | 112.269 |
| pair-list coordination preflight | 16.9% | 10 | 167.595 |
| D4 pair-list weight preparation | 9.5% | 40 | 43.071 |
| integral-force shell pairs | 5.4% | 10 | 96.541 |
| integral shell pairs | 4.0% | 10 | 72.094 |

The pair-list rows have high per-instance variance in this short profile, so
they identify investigation targets rather than stable optimization deltas.

## Nsight Compute availability

One representative attempt was made with Nsight Compute console CSV output:

```bash
/group/software/cuda-12.9.1/bin/ncu \
  --csv --page raw --set basic --launch-count 1 --target-processes all \
  build/issue128-cuda/xtbloom_cuda_public_api_test --request-profile
```

It exited 1 with:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

Status: `UNAVAILABLE (ERR_NVGPUCTRPERM)`. Per the owner decision, this does not
block #128, no privileged access was requested, and no NCU counter claim is
made. `ncu-status.txt` preserves the tool version, command, literal exit code,
and diagnostic without retaining a native `.ncu-rep`.

## Validation and limitations

- Exact-head fixed-plan profile mode: PASS; printed
  `request_profile_iterations=10`.
- Exact-head focused real-GPU matrix: 5/5 PASS (`public_api`, runtime Graph,
  cache diagnostics, device public conformance, and device invariants).
- The exact-head runtime reuse controls also passed 3/3 with
  `ctest --test-dir build/issue128-cuda -R
  '^xtbloom\.cuda\.gfn2_runtime_(owner|parity|graph)$' --output-on-failure`.
  These tests cover same-identity reuse across batch sizes 1/8/32/128,
  public/runtime parity, changed numerical values with the same Graph
  executable, and stable Graph-bound addresses.
- Benchmark harness and evidence-size unit tests: 23/23 PASS.
- Raw profiler captures are absent from the bundle.
- The latency matrix is one representative homogeneous gas workload with
  direct-device descriptors. It does not claim every workload, memory mode,
  molecule size, or external engine.
- Changed-geometry latency is limited to the one representative batch-4 force
  coordinate above. Changed-value correctness, stable addresses, and Graph
  reuse are additionally proven by the exact-head runtime controls; no broader
  changed-geometry performance claim is made.
- The fixed-plan profile uses the existing four-system force/QM-MM/periodic
  production request coordinate; it proves resource reuse and identifies
  hotspots, not a release-wide timing distribution.

`SHA256SUMS` covers every retained artifact after final formatting.
