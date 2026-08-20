# Issue #461 pure-FP64 CUDA Hamiltonian-assembly evidence

This bundle qualifies the issue #461 change that distributes each active
GFN1/GFN2 effective-Hamiltonian channel across multiple topology-fixed CUDA
CTAs. The implementation changes scheduling only: one thread still owns every
upper-triangular AO pair and executes the existing ordered binary64
scalar/dipole/quadrupole expression for both directed matrix elements.

The narrow conclusion on this RTX 5090 is:

- the selected 362-atom, approximately 722-AO, batch-one FRESH public
  energy-plus-force call improves from 1267.696 ms to 1194.240 ms, or 5.794%;
- the matching strict WARM call improves from 298.582 ms to 289.852 ms, or
  2.924%;
- 122-AO FRESH controls at B=8/32/128 improve by 1.425%, 0.832%, and 0.470%;
- 122-AO WARM controls improve by 7.614% at stabilized B=8 and 0.319% at B=32,
  while B=128 changes by -0.018%, far below the issue's material 2% regression
  threshold;
- therefore the issue's at-least-5% primary gate passes and no control has a
  material regression.

performance-summary.csv is the compact result table.
raw-timing-samples.json retains all 540 headline timing samples. It also retains
the order-sensitive B=8 WARM diagnostic blocks, the four 100-warmup ABBA
blocks, and the large-WARM repeat. The original runner JSON files were
1.2-35.7 MiB because they retain complete energy/force outputs for every
sample, so they are intentionally omitted under the repository evidence
budget. Exact commands, clean revisions, binary hashes, correctness
qualification, and every timing value needed for the claim remain here.

## Source and implementation

- Baseline source: clean
  2dc1619fed0a124690b8e3e13ec61ac25e667e69.
- Candidate source: clean
  450b16eacd64639195bb344e2aaa5a3072e2938e.
- Baseline CUDA library SHA-256:
  323ecd39739bfbc8560b7ca89931e13ddd2a2dedb08d7b1943945f7e1a2924a3.
- Candidate CUDA library SHA-256:
  49e60305c7988fd5aaac852b6322f4ccd426cb37f803a6fe778f5cde6acb3d25.
- Candidate CPU library SHA-256:
  9e37209c98cefe9a43511a6dbd6b74fa878ead653ed343ae214d3570709c88ba.
- Branch diff: 14 files, 417 insertions, 75 deletions.
- Public C ABI: unchanged.
- Internal SCC schema: v6 to v7, adding topology-fixed
  assembly_tiles_per_channel metadata.
- Restricted assembly grid: (system, tile, 1).
- Spin assembly grid: (system, channel, tile).
- Tile selection reuses the density setup policy, so the direct and Graph
  shapes are fixed by topology rather than SCC activity.
- Publication remains one CTA per system. The primary performance gate already
  passes, WARM also improves, and the evidence does not justify the additional
  publication-tiling complexity.

Focused tests cover zero and over-budget metadata, stale-v6 rejection, setup
propagation, GFN1/GFN2 one-tile-versus-nine-tile bitwise equality, one/two spin
channels, actual host-captured Graph grids, and a nonfinite value owned by the
final tile with healthy-peer and output-sentinel preservation.

## Environment

- Host: node3, Ubuntu 22.04.5 LTS, Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62 48-Core Processor, 48 CPUs in the exclusive allocation.
- GPU: NVIDIA GeForce RTX 5090, UUID
  GPU-8e9c9e1a-e183-258c-0b3a-03a5ddebb2f8, 32,607 MiB, compute capability
  12.0.
- Driver: 580.95.05.
- CUDA compiler/toolkit: 12.9.86; CMAKE_CUDA_ARCHITECTURES=120.
- Host compiler: GCC 11.4.0; CMake 4.2.1; Ninja 1.13.0.
- Build: shared Release, XTBLOOM_ENABLE_CUDA=ON, LP64 MKL runtime
  /home/jzzeng/miniconda3/lib/libmkl_rt.so.3.
- Thread boundary: OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
  MKL_NUM_THREADS=1, OMP_DYNAMIC=FALSE, MKL_DYNAMIC=FALSE.
- Nsight Systems: 2025.1.3.140-251335620677v0.
- Compute Sanitizer: 2025.2.1.0 build 35969825.
- Nsight Compute hardware-counter collection was unavailable because
  /proc/driver/nvidia/params reported RmProfilingAdminOnly=1. This bundle makes
  no achieved-FP64-utilization, occupancy, or hardware-counter claim.

## Correctness and validation

The identical baseline and candidate CUDA configurations were built as:

~~~bash
cmake -S . -B build/cuda-issue461-perf-<short-sha> -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/miniconda3/lib/libmkl_rt.so.3 \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda-issue461-perf-<short-sha> --parallel
ctest --test-dir build/cuda-issue461-perf-<short-sha> -N
~~~

The baseline and candidate each passed the focused eight-test real-GPU CUDA
gate:

~~~bash
srun --exclusive -N1 -n1 -c48 --mem=0 --gres=gpu:1 \
  --kill-on-bad-exit=1 \
  env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/home/jzzeng/miniconda3/lib \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  ctest --test-dir build/cuda-issue461-perf-<short-sha> \
  -R 'xtbloom\.cuda\.(hamiltonian(_force)?|density|electronic_gradient|energy_force_execution|scc_iteration_binding|scc_setup_inputs|scc_iteration_production)$' \
  --output-on-failure
~~~

Result: baseline 8/8 passed; candidate 8/8 passed.

The exact candidate CUDA build registered 187 tests. On the RTX 5090:

~~~bash
srun --exclusive -N1 -n1 -c48 --mem=0 --gres=gpu:1 \
  --kill-on-bad-exit=1 \
  env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/home/jzzeng/miniconda3/lib \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  ctest --test-dir build/cuda-issue461-perf-450b16e \
  --output-on-failure
~~~

Result: 187/187 passed in 164.80 seconds. This includes host/device/mixed public
conformance and invariants, GFN1/GFN2, restricted/unrestricted paths, ragged
failure isolation, Graphs, cache/WARM behavior, publication, ABI symbols, and
CUDA dependency checks.

The independent CPU-only shared configuration used the same LP64 MKL provider:

~~~bash
cmake -S . -B build/cpu-issue461-final-450b16e -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/miniconda3/lib/libmkl_rt.so.3 \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu-issue461-final-450b16e --parallel
ctest --test-dir build/cpu-issue461-final-450b16e -N
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  ctest --test-dir build/cpu-issue461-final-450b16e --output-on-failure
~~~

Result: 96/96 passed in 31.48 seconds.

Before the evidence-only commit, the full repository hooks passed 13 checks
with three no-file skips, canonical-PyPI uv lock --check passed, and
git diff --check passed. An independent read-only implementation review
reported LGTM with no correctness blocker. The final actual-head evidence review
is recorded in issue #461 and the pull request rather than asserted by this
pre-review bundle.

Compute Sanitizer results and the exact owner disposition are in
sanitizer-summary.txt. The Hamiltonian direct/host-Graph binary is clean under
all four tools. The focused 542-AO production device-Graph smoke is clean under
memcheck, racecheck, and initcheck; synccheck retains exit 99 with exactly 256
approved reports at reduce_spin_atomic_charges_kernel+0x2b0. The tool itself did
not return a clean pass, and no Hamiltonian signature was present.

## Performance protocol and commands

All headline rows use the public synchronous C ABI through
benchmarks/natoms_scaling.py, request energy plus analytic forces, and use the
deterministic alkane workload. FRESH performs independent SCC initialization
for every timed call. WARM performs one untimed FRESH seed, then strict WARM
warmups and timed calls; each WARM artifact references the FRESH JSON produced
by the same library and clean revision. Energy and force gates are 1e-8 Hartree
and 5e-7 Hartree/bohr.

The common scheduler and environment boundary was:

~~~bash
srun --exclusive -N1 -n1 -c48 --mem=0 --gres=gpu:1 \
  --kill-on-bad-exit=1 \
  env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/home/jzzeng/miniconda3/lib \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE bash -c '<commands below>'
~~~

The local worktree parent contains a retired repository name, so absolute
worktree paths are intentionally replaced below. With <worktree-parent>
standing for that parent:

~~~bash
benchmark_python=/home/jzzeng/miniconda3/bin/python3
stabilized_python=/usr/bin/python3
baseline_root=<worktree-parent>/issue-461-baseline
candidate_root=<worktree-parent>/issue-461-hamiltonian
baseline_library="$baseline_root/build/cuda-issue461-perf-2dc1619/libxtbloom.so"
candidate_library="$candidate_root/build/cuda-issue461-perf-450b16e/libxtbloom.so"
~~~

For each label/root/library tuple, the 122-AO controls were:

~~~bash
"$benchmark_python" "$root/benchmarks/natoms_scaling.py" \
  --engine xtbloom --library "$library" --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 62 --batch-sizes 8,32,128 \
  --warmups 10 --repetitions 30 --start-mode fresh \
  --output-json "build/benchmarks/issue-461/$label-controls-fresh.json" \
  --output-csv "build/benchmarks/issue-461/$label-controls-fresh.csv"

"$benchmark_python" "$root/benchmarks/natoms_scaling.py" \
  --engine xtbloom --library "$library" --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 62 --batch-sizes 8,32,128 \
  --warmups 10 --repetitions 30 --start-mode warm \
  --energy-reference-json "build/benchmarks/issue-461/$label-controls-fresh.json" \
  --output-json "build/benchmarks/issue-461/$label-controls-warm.json" \
  --output-csv "build/benchmarks/issue-461/$label-controls-warm.csv"
~~~

The selected large-AO commands differed only in these arguments and output
names:

~~~text
--natoms 362 --batch-sizes 1 --warmups 10 --repetitions 30
--output-json build/benchmarks/issue-461/<label>-large-<fresh-or-warm>.json
--output-csv build/benchmarks/issue-461/<label>-large-<fresh-or-warm>.csv
~~~

The WARM large-AO command used the matching
<label>-large-fresh.json as its energy reference.

The initial B=8 WARM blocks were order-sensitive and bimodal. They are retained
in raw-timing-samples.json. The stabilized sequence used 100 warmups and 30
samples per block in this ABBA order:

~~~text
baseline-a, candidate-a, candidate-b, baseline-b
~~~

Each block used the corresponding root/library, --natoms 62,
--batch-sizes 8, --start-mode warm, and the matching controls-FRESH reference.
The reported B=8 WARM median is computed over the two 30-sample blocks for each
library, not from a selected favorable block.

## Results

| Coordinate | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| 722 AO, B=1 FRESH | 1267.696 ms | 1194.240 ms | 5.794% faster |
| 722 AO, B=1 WARM | 298.582 ms | 289.852 ms | 2.924% faster |
| 122 AO, B=8 FRESH | 142.714 ms | 140.680 ms | 1.425% faster |
| 122 AO, B=32 FRESH | 195.282 ms | 193.658 ms | 0.832% faster |
| 122 AO, B=128 FRESH | 440.023 ms | 437.953 ms | 0.470% faster |
| 122 AO, B=8 WARM, stabilized ABBA | 35.604 ms | 32.893 ms | 7.614% faster |
| 122 AO, B=32 WARM | 60.399 ms | 60.207 ms | 0.319% faster |
| 122 AO, B=128 WARM | 179.317 ms | 179.350 ms | 0.018% slower |

Every row passed energy/force drift, per-system status, convergence,
SCC-iteration, and, for WARM, same-library FRESH-reference gates. Median SCC
iterations are unchanged: 17 versus 17 on the 722-AO FRESH row, 2 versus 2 on
WARM rows, and 18 versus 18 on the control FRESH rows. The few high-latency
large-WARM observations and every bimodal B=8 WARM sample are retained.

## Nsight Systems audit

Baseline and candidate were each profiled for one 722-AO B=1 FRESH coordinate
with one warmup and one measured public call:

~~~bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=<baseline-or-candidate-prefix> \
  "$benchmark_python" <matching-root>/benchmarks/natoms_scaling.py \
  --engine xtbloom --library <matching-library> --backend cuda \
  --property force --natoms 362 --batch-sizes 1 --warmups 1 \
  --repetitions 1 --start-mode fresh \
  --output-json <matching-output>.json --output-csv <matching-output>.csv

nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum \
  --format csv --force-overwrite=true --output <derived-prefix> \
  <capture>.nsys-rep

head -n 101 <derived-kernel-summary.csv> \
  > <retained-kernel-summary.csv>
~~~

The instrumented measured medians were 1269.047 ms baseline and 1195.738 ms
candidate. Across each complete setup/warmup/measured process, CUDA API counts
are identical, including 32 cudaMalloc, 38 cudaFree, 11 cudaMallocHost,
11 cudaFreeHost, 81 cudaMemcpyAsync, 328 cudaMemsetAsync, two Graph
instantiations, two Graph launches, ten event synchronizations, five stream
synchronizations, and 20 device synchronizations. CUDA memory-operation counts
and bytes are also identical: memset 282.530 MB/257, device-to-device
155.230 MB/3, host-to-device 141.344 MB/56, and device-to-host 0.028 MB/26.

Thus the scheduling change adds no allocation, transfer, Graph-resource, or
synchronization call relative to baseline. The capture includes setup and is
not represented as a zero-allocation steady-state-only trace.

The API and memory summaries are retained in full. To stay within the
repository-wide 16 MiB evidence budget, each time-sorted kernel summary retains
its header plus the first 100 entries. Those entries account for 99.5% of the
baseline and 99.4% of the candidate kernel time using Nsight's one-decimal
Time (%) field. Every omitted entry individually rounds to 0.0%; the aggregate
omitted tail is approximately 0.5%-0.6%. This deterministic truncation does not
filter timing samples or any API/memory activity used by the acceptance claim.

Raw .nsys-rep and generated SQLite files remain under ignored build paths and
are prohibited from the repository because they can contain process
environment data. Only the eight sanitized derived CSV reports are retained.

## Limitations

- Performance was measured on one consumer GPU, RTX 5090. No A100 or other
  stronger-FP64 GPU speedup is claimed.
- The latency matrix uses homogeneous deterministic alkanes. Heterogeneous
  ragged, unrestricted, failure, host/device/mixed, and QM/MM behavior is
  correctness-qualified by the 187-test real-GPU matrix but has no separate
  latency claim here.
- Hardware-counter access was unavailable, so this evidence demonstrates
  end-to-end and CUDA-runtime behavior rather than achieved FP64 utilization.
- The evidence supports assembly tiling only. It does not support publication
  tiling, mixed precision, or a public precision policy.
