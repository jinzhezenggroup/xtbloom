# Issue #80 production exact-capacity dispatch-chain evidence

This directory archives the end-to-end production SCC loop performance evidence
for the exact-capacity device-dispatched chain integrated by PR #227. The claim
is narrow and measured: on the recorded RTX 5090 stack, the production
`kDeviceDispatchChain` graph family runs the full SCC loop with identical
published results to the pre-#227 monolithic full-capacity device-tail graph
(`kDeviceTailGraph`) and to the sequential CPU reference, and is measurably
faster on the heterogeneous partial-activity workloads that issue #80 targets,
with a documented crossover at full bucket capacity.

The component-level capacity/compaction crossover is archived separately under
`benchmarks/evidence/issue-131/`. This bundle adds the production-loop
(full SCC to global terminal) comparison between the two graph families that
PR #227 makes a runtime choice.

## Environment and build identity

- Host: `node3`, Ubuntu 22.04.5 LTS, Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62 48-Core Processor.
- GPU: NVIDIA GeForce RTX 5090 (GB202), `sm_120`, 32 GiB.
- Driver: 580.95.05; CUDA toolkit: 12.9.86.
- Compiler: GCC 11.4.0; CMake 4.2.1; Release build; Ninja generator.
- Nsight Systems: 2025.1.3.140.
- Compute Sanitizer: 2025.2.1.0.
- Measured source revision: `085c4c249c0f86455ee26b05d19b46f74dfa4724` (the
  #227 feature branch merged against current main; working tree clean apart
  from this evidence bundle and the benchmark-mode additions to
  `tests/cuda_scc_iteration_production_test.cu`).
- Benchmark binary: `build/cuda-sm120/gpuxtb_cuda_scc_loop_benchmark`.
- Effective CMake cache: `GPUXTB_ENABLE_CUDA=ON`,
  `CMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc`,
  `CMAKE_CUDA_ARCHITECTURES=120`, `CMAKE_BUILD_TYPE=Release`.

The CUDA runtime loaded for every run is the 12.9 toolkit
(`/group/software/cuda-12.9.1/lib64`), matching the compiler used to build the
benchmark binary; the default system loader otherwise resolves the older
12.4 runtime that gpuxtb correctly rejects.

## Methodology

`benchmark_dispatch_chain_vs_monolithic` (invoked as
`gpuxtb_cuda_scc_loop_benchmark --benchmark-chain`) builds, from the identical
binding and state, both `kDeviceDispatchChain` and `kDeviceTailGraph` owners
for batch sizes 1/8/32/128 in the heterogeneous small-system production fixture
(H2/He/LiH/CH2 cycled, `maximum_iterations = 8`).

For each of three activity tiers (active fraction 1, 1/2, 1/4), a deterministic
per-system terminal ladder seeds inactive members' `iterations` at the
configured maximum, so the first activity derivation excludes them from every
subsequent numerical body while the active members run the normal SCC loop to
global terminal. This reproduces, unchanged in spirit, the reusable real-GPU
"active ladder" used by the #131 component evidence, but at the full production
SCC loop level with both graph families.

Per tier the runner:

1. resets the iteration arena (`initializer.upload_async`) and replays the
   ladder,
2. records a CUDA event, launches one complete SCC loop to global terminal,
   records a second event, and synchronizes,
3. requires the loop to end with `canonical_active_count == 0`,
4. keeps 50 measured samples after 3 warmups, and always validates launch
   success before accepting a sample,
5. requires both families to execute the same number of numerical bodies,
6. verifies the terminal ledger (terminal members untouched at their seeded
   count; active members advanced) and, for the all-active tier, verifies the
   chain's full published state equals the sequential CPU reference
   (`run_host_until_globally_terminal` + `compare_graph_loop_cpu_parity`).

Timing is CUDA-event elapsed time on the fixture stream between the launch and
its completion, covering the complete loop to global terminal; correctness
downloads happen after the timed interval.

## Raw data

- `production-chain-vs-monolithic.jsonl`: one protocol object followed by one
  measurement object per (batch, activity tier, family). Each measurement
  carries mean/min/median/p95 over the 50 samples, the numerical body count,
  and all 50 raw sample latencies as trailing CSV values.
- `production-crossover-summary.csv`: derived compact view (mean, speedup,
  body counts) per row.

## Results

Every one of the 20 measured rows reports:

```text
chain numerical bodies == monolithic numerical bodies
loop ends terminal (canonical_active_count == 0)
ledger consistent for the seeded ladder
```

and for the all-active tier the chain's final state matches the CPU sequential
reference exactly (the existing `gpuxtb.cuda.scc_iteration_production` CPU
parity gates also pass on this build: 99/99 CUDA CTest).

Mean full-loop latency (50 samples, RTX 5090):

| Batch | Active | chain ms | monolithic ms | speedup |
| --- | --- | ---: | ---: | ---: |
| 1 | 1.000 | 1.417 | 1.357 | 0.96x |
| 8 | 1.000 | 5.461 | 5.548 | 1.02x |
| 8 | 0.500 | 4.152 | 5.208 | 1.25x |
| 8 | 0.250 | 1.689 | 2.471 | 1.46x |
| 32 | 1.000 | 6.672 | 6.739 | 1.01x |
| 32 | 0.500 | 5.006 | 6.446 | 1.29x |
| 32 | 0.250 | 2.028 | 3.112 | 1.53x |
| 128 | 1.000 | 10.810 | 10.311 | 0.95x |
| 128 | 0.500 | 8.239 | 9.711 | 1.18x |
| 128 | 0.250 | 3.337 | 4.599 | 1.38x |

Conclusions, stated narrowly and only for the recorded stack/workloads:

- At partial activity (1/2, 1/4), the exact-capacity chain is 18%-53% faster
  than the monolithic full-capacity device-tail graph across batch 8/32/128,
  matching the component-level crossover measured in #131.
- At full activity the chain is within +/-5% of the monolithic graph: slightly
  ahead at batch 8/32, and with a small dispatch-table overhead at batch 128
  (0.95x) and at batch 1 (0.96x, a single fully-active bucket). This is the
  measured crossover boundary: when no member can be excluded, the extra
  per-(bucket, capacity) dispatch costs slightly more than a single monolithic
  launch. The #131 component evidence recorded the same effect (compacted
  slower than uncompacted at capacity tier).
- Batch-1 always keeps its single member active; its three recorded tiers are
  the same workload and are deduplicated in the JSONL by design.

Correctness evidence: the all-active tier runs the CPU sequential reference and
passes `compare_graph_loop_cpu_parity`; the full production test binary passes
99/99 CUDA CTest rows on this stack, including the forced dispatch-chain CPU
parity for batch 1/8/32/128, mixed-spin bounded fallback, dynamic-geometry
epoch parity, and chain-owner whole-pipeline capture.

## Profiler capture

`derived-profiler-reports/nsys-*_kern_sum.csv` are sanitized, kernel-name +
timing-only summaries exported from Nsight Systems (2025.1.3.140) captures of
`--benchmark-chain-one <B> 4 <mode> 40` (40 full loops at 1/4 activity, single
family per capture). Raw `.nsys-rep` captures embed the target process
environment and credentials, so they are intentionally not committed; only the
derived CSV summaries are retained. Export commands:

```bash
nsys profile -o chain32-aquarter -t cuda gpuxtb_cuda_scc_loop_benchmark \
  --benchmark-chain-one 32 4 chain 40
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
  --output nsys-chain32-aquarter-kern_sum chain32-aquarter.nsys-rep
```

On this NVIDIA stack, device-launched Graph kernels are not individually
attributable by `nsys stats`, so these summaries document the steady-state
repetition boundary rather than providing a per-kernel chain/monolithic
breakdown of the eigensolver; the end-to-end CUDA-event latencies above are
the authoritative timing.

## Compute Sanitizer

- `sanitizer-memcheck-chain.log`, `sanitizer-initcheck-chain.log`,
  `sanitizer-racecheck-chain.log`: 0 errors for the dispatch-chain loop.
- `sanitizer-memcheck-monolithic.log`, `sanitizer-initcheck-monolithic.log`,
  `sanitizer-racecheck-monolithic.log`: 0 errors for the monolithic loop.
- `sanitizer-synccheck-chain.log`, `sanitizer-synccheck-monolithic.log`:
  identical nvidia-tool artifact in both families at
  `gpuxtb::detail::cuda::reduce_spin_atomic_charges_kernel` when a device-
  launched CUDA Graph is replayed under Compute Sanitizer's synccheck tool.
  This is the same class of device-launch Graph/tool boundary documented for
  #130/#131/#140/#141 (there reported against NVIDIA `batch_trsm_left_kernel`);
  it is not specific to the dispatch chain, appears identically in the
  pre-#227 monolithic production path, in a kernel unchanged by #227, and the
  direct non-graph control is clean (see below).
- Direct (non-graph) controls are clean: the production one-iteration parity
  path under synccheck reports 0 errors (`sanitizer-synccheck-direct.log`),
  matching the archived issue-131/issue-232 direct memcheck/initcheck/racecheck/
  synccheck controls.

## Limitations

- Small heterogeneous fixture molecules; the eigensolver fraction of the loop
  is small, so absolute latencies are a few milliseconds. The relative
  chain/monolithic comparison and its crossover are the evidence; production
  large-AO absolute timings are covered by the #232 bundle.
- Release-only run; no sanitizer build of the newly added benchmark path (the
  production runtime matrix is covered by the existing sanitizer-tested
  production tests).
- The synccheck Graph-replay tool artifact is recorded, not waived here; it is
  identical and pre-existing in the monolithic production path, so it is not a
  regression attributable to the exact-capacity integration.
