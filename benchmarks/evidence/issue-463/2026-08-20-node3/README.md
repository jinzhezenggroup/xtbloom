# Issue #463 pure-FP64 CUDA integral/H0 force evidence

This bundle qualifies the issue #463 change that distributes disjoint
integral/H0 validation, seed, and publication scans across bounded
topology-fixed CUDA CTAs. It is a scheduling-only binary64 change: the H0 Pulay
contraction and integral shell-pair contraction retain their original
arithmetic and reduction order, the public C ABI is unchanged, and SCC policy,
mixed precision, tolerances, and physics equations are outside scope.

The narrow conclusion on this RTX 5090 is:

- the selected 362-atom, approximately 722-AO, batch-one FRESH public
  energy-plus-force call improves from 1198.276 ms to 1165.991 ms, or 2.694%;
- the matching strict WARM call improves from 289.660 ms to 258.415 ms, or
  10.787%;
- all six 122-AO B=8/32/128 FRESH/WARM controls improve, from 0.043% to 2.141%;
- therefore the issue's at-least-2% primary gate passes and no control has a
  material regression.

performance-summary.csv is the compact result table. raw-timing-samples.json
retains all 480 headline timing samples and the order-sensitive large-system
ABBA block medians. Original runner JSON files range from 0.6 MiB to 35.7 MiB
because they retain complete energy and force arrays for every sample, so they
are intentionally omitted under the repository evidence budget. Exact
commands, clean revisions, binary hashes, correctness qualification, and every
timing value needed for the claim remain here.

## Source and implementation

- Baseline source: clean
  eaa563725733a199ccd8ccefd47f64d6c27b68a0.
- Candidate source: clean
  e342dc7d836ba6dcf1a7e5682f075a7b2b4e9e49.
- Baseline CUDA library SHA-256:
  a38bb486f9802992eb4a58e650f4193e5e2e7b73859c26433706280c932df03e.
- Candidate CUDA library SHA-256:
  86a616982624f03dad5b0497b417e9a8731248837263a5f8b1981d61d8410b6f.
- Candidate CPU library SHA-256:
  9e37209c98cefe9a43511a6dbd6b74fa878ead653ed343ae214d3570709c88ba.
- Baseline/candidate CUDA CMake-cache SHA-256:
  2b7898ed02bfb865d9a6d896acce3579ab04aeb2a948c10b23a69ae3d86d3f23
  and ef610fdf2cb4281ba33b2533aa2f9cb01a9019e8a5abda33096dbd7714b88892.
- Candidate CPU CMake-cache SHA-256:
  22fef5d1c123daf80cc6720e2207d37cd37c8663c05ef557dcb13a70f90bde1f.
- Branch diff: 13 files, 771 insertions, 113 deletions.
- Public C ABI and generated scientific artifacts: unchanged.

The implementation reuses the density contraction topology selector. Integral
and H0 publication, H0-force seed/preflight/publication, and integral-force
numerical preflight/publication use disjoint element ownership across tiles.
Integral-force topology validation remains one CTA per system, followed by a
kernel boundary before tiled numerical scans. This preserves peer-local failure
and transactional publication semantics while eliminating cross-CTA metadata
races. Direct and CUDA Graph launch shapes depend only on fixed topology, not
runtime SCC activity.

Focused tests cover selector bounds, one-tile-versus-multi-tile equality,
invalid tile metadata before reset/enqueue, nonfinite values, inactive and
failed systems, hostile primitive offsets, healthy ragged peers, caller-output
sentinels, and host-captured Graph grids.

## Environment

- Host: node3, Ubuntu 22.04.5 LTS, Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62 48-Core Processor, 48 logical CPUs in the exclusive
  allocation.
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
  /proc/driver/nvidia/params reports RmProfilingAdminOnly=1. This bundle makes
  no achieved-FP64-utilization, occupancy, or hardware-counter claim.

## Correctness and validation

The baseline and candidate CUDA configurations were built with these exact
commands from their matching clean source revisions:

~~~bash
candidate_root="$(git rev-parse --show-toplevel)"
baseline_root="$(git -C ../issue-463-baseline-clean rev-parse --show-toplevel)"

cmake -S "$baseline_root" \
  -B "$baseline_root/build/cuda-issue463-baseline" \
  -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/miniconda3/lib/libmkl_rt.so.3 \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build \
  "$baseline_root/build/cuda-issue463-baseline" \
  --target xtbloom --parallel

cd "$candidate_root"
cmake -S . -B build/cuda-issue463-dev -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/miniconda3/lib/libmkl_rt.so.3 \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda-issue463-dev --parallel
ctest --test-dir build/cuda-issue463-dev -N
~~~

Candidate focused real-GPU gate:

~~~bash
env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/home/jzzeng/miniconda3/lib \
  CUDA_VISIBLE_DEVICES=0 \
  ctest --test-dir build/cuda-issue463-dev \
  -R 'xtbloom\.cuda\.(integrals|integral_force|h0_force|gfn2_preprocessing|electronic_gradient|energy_force_execution)$' \
  --output-on-failure
~~~

Result: 6/6 passed.

The candidate CUDA build registered and executed 187 tests on the RTX 5090:

~~~bash
env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/home/jzzeng/miniconda3/lib \
  CUDA_VISIBLE_DEVICES=0 \
  ctest --test-dir build/cuda-issue463-dev --output-on-failure
~~~

Result: 187/187 passed in 164.68 seconds. This includes public host/device/mixed
conformance and invariants, restricted/unrestricted paths, term-level force
tests, finite differences, ragged failure isolation, Graphs, cache/WARM,
publication, ABI symbols, and CUDA dependency checks.

The independent shared CPU configuration used the same LP64 MKL provider:

~~~bash
cmake -S . -B build/cpu-issue463-dev -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/miniconda3/lib/libmkl_rt.so.3 \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu-issue463-dev --parallel
ctest --test-dir build/cpu-issue463-dev -N
ctest --test-dir build/cpu-issue463-dev --output-on-failure
~~~

Result: 96/96 passed in 31.58 seconds.

The exact repository validation commands were:

~~~bash
candidate_root="$(git rev-parse --show-toplevel)"
cd "$candidate_root"
python3 -m unittest -v \
  benchmarks.test_run benchmarks.test_dxtb_adapter benchmarks.test_evidence_size
UV_DEFAULT_INDEX=https://pypi.org/simple \
  uv tool run --from prek==0.3.1 prek run \
  --show-diff-on-failure --color=always --all-files
UV_DEFAULT_INDEX=https://pypi.org/simple uv lock --check
git diff --check
cd benchmarks/evidence/issue-463/2026-08-20-node3
sha256sum --check SHA256SUMS
~~~

Benchmark harness unit tests passed 32/32. The pinned full-file repository
hooks passed after their formatting edit was included, canonical-PyPI
`uv lock --check` and `git diff --check` passed, and the evidence hashes passed
11/11. An independent read-only final implementation review reported LGTM with
no blocking finding.

Compute Sanitizer results and the exact owner disposition are in
sanitizer-summary.txt. Six focused direct/host-Graph binaries are clean under
all four tools (24/24 coordinates). The production device-Graph runtime is
clean under memcheck, racecheck, and initcheck; synccheck retains exit 99 with
256 errors. The default print limit emitted 100 reports, all matching the
approved issue #279 signature at reduce_spin_atomic_charges_kernel+0x2b0. The
tool itself did not return a clean pass, and no changed integral/H0 kernel
signature appeared.

## Performance protocol and commands

All headline rows use the public synchronous C ABI through
benchmarks/natoms_scaling.py, request energy plus analytic forces, and use the
deterministic alkane workload. FRESH performs independent SCC initialization
for every timed call. WARM performs an untimed FRESH seed and strict WARM calls;
each WARM artifact references the FRESH JSON produced by the same clean library
revision. Energy and force gates are 1e-8 Hartree and 5e-7 Hartree/bohr.

The exact common environment and path boundary was:

~~~bash
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/home/jzzeng/miniconda3/lib
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
candidate_root="$(git rev-parse --show-toplevel)"
baseline_root="$(git -C ../issue-463-baseline-clean rev-parse --show-toplevel)"
benchmark_python=/home/jzzeng/miniconda3/bin/python3
~~~

The 122-AO controls were recorded with these exact commands and output paths:

~~~bash
"$benchmark_python" "$baseline_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$baseline_root/build/cuda-issue463-baseline/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 62 --batch-sizes 8,32,128 \
  --warmups 10 --repetitions 30 --start-mode fresh \
  --output-json "$baseline_root/build/benchmarks/issue-463-final/baseline-controls-fresh.json" \
  --output-csv "$baseline_root/build/benchmarks/issue-463-final/baseline-controls-fresh.csv"

"$benchmark_python" "$candidate_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$candidate_root/build/cuda-issue463-dev/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 62 --batch-sizes 8,32,128 \
  --warmups 10 --repetitions 30 --start-mode fresh \
  --output-json "$candidate_root/build/benchmarks/issue-463-final/candidate-controls-fresh.json" \
  --output-csv "$candidate_root/build/benchmarks/issue-463-final/candidate-controls-fresh.csv"

"$benchmark_python" "$baseline_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$baseline_root/build/cuda-issue463-baseline/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 62 --batch-sizes 8,32,128 \
  --warmups 10 --repetitions 30 --start-mode warm \
  --energy-reference-json "$baseline_root/build/benchmarks/issue-463-final/baseline-controls-fresh.json" \
  --output-json "$baseline_root/build/benchmarks/issue-463-final/baseline-controls-warm.json" \
  --output-csv "$baseline_root/build/benchmarks/issue-463-final/baseline-controls-warm.csv"

"$benchmark_python" "$candidate_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$candidate_root/build/cuda-issue463-dev/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 62 --batch-sizes 8,32,128 \
  --warmups 10 --repetitions 30 --start-mode warm \
  --energy-reference-json "$candidate_root/build/benchmarks/issue-463-final/candidate-controls-fresh.json" \
  --output-json "$candidate_root/build/benchmarks/issue-463-final/candidate-controls-warm.json" \
  --output-csv "$candidate_root/build/benchmarks/issue-463-final/candidate-controls-warm.csv"
~~~

The large coordinate used four 15-sample blocks in ABBA order:
baseline-a, candidate-a, candidate-b, baseline-b. The exact FRESH sequence was:

~~~bash
"$benchmark_python" "$baseline_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$baseline_root/build/cuda-issue463-baseline/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 362 --batch-sizes 1 \
  --warmups 10 --repetitions 15 --start-mode fresh \
  --output-json "$baseline_root/build/benchmarks/issue-463-final/baseline-a-large-fresh.json" \
  --output-csv "$baseline_root/build/benchmarks/issue-463-final/baseline-a-large-fresh.csv"

"$benchmark_python" "$candidate_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$candidate_root/build/cuda-issue463-dev/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 362 --batch-sizes 1 \
  --warmups 10 --repetitions 15 --start-mode fresh \
  --output-json "$candidate_root/build/benchmarks/issue-463-final/candidate-a-large-fresh.json" \
  --output-csv "$candidate_root/build/benchmarks/issue-463-final/candidate-a-large-fresh.csv"

"$benchmark_python" "$candidate_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$candidate_root/build/cuda-issue463-dev/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 362 --batch-sizes 1 \
  --warmups 10 --repetitions 15 --start-mode fresh \
  --output-json "$candidate_root/build/benchmarks/issue-463-final/candidate-b-large-fresh.json" \
  --output-csv "$candidate_root/build/benchmarks/issue-463-final/candidate-b-large-fresh.csv"

"$benchmark_python" "$baseline_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$baseline_root/build/cuda-issue463-baseline/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 362 --batch-sizes 1 \
  --warmups 10 --repetitions 15 --start-mode fresh \
  --output-json "$baseline_root/build/benchmarks/issue-463-final/baseline-b-large-fresh.json" \
  --output-csv "$baseline_root/build/benchmarks/issue-463-final/baseline-b-large-fresh.csv"
~~~

The WARM sequence used the matching A-block FRESH artifact for each clean
library revision:

~~~bash
"$benchmark_python" "$baseline_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$baseline_root/build/cuda-issue463-baseline/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 362 --batch-sizes 1 \
  --warmups 10 --repetitions 15 --start-mode warm \
  --energy-reference-json "$baseline_root/build/benchmarks/issue-463-final/baseline-a-large-fresh.json" \
  --output-json "$baseline_root/build/benchmarks/issue-463-final/baseline-a-large-warm.json" \
  --output-csv "$baseline_root/build/benchmarks/issue-463-final/baseline-a-large-warm.csv"

"$benchmark_python" "$candidate_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$candidate_root/build/cuda-issue463-dev/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 362 --batch-sizes 1 \
  --warmups 10 --repetitions 15 --start-mode warm \
  --energy-reference-json "$candidate_root/build/benchmarks/issue-463-final/candidate-a-large-fresh.json" \
  --output-json "$candidate_root/build/benchmarks/issue-463-final/candidate-a-large-warm.json" \
  --output-csv "$candidate_root/build/benchmarks/issue-463-final/candidate-a-large-warm.csv"

"$benchmark_python" "$candidate_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$candidate_root/build/cuda-issue463-dev/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 362 --batch-sizes 1 \
  --warmups 10 --repetitions 15 --start-mode warm \
  --energy-reference-json "$candidate_root/build/benchmarks/issue-463-final/candidate-a-large-fresh.json" \
  --output-json "$candidate_root/build/benchmarks/issue-463-final/candidate-b-large-warm.json" \
  --output-csv "$candidate_root/build/benchmarks/issue-463-final/candidate-b-large-warm.csv"

"$benchmark_python" "$baseline_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$baseline_root/build/cuda-issue463-baseline/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 362 --batch-sizes 1 \
  --warmups 10 --repetitions 15 --start-mode warm \
  --energy-reference-json "$baseline_root/build/benchmarks/issue-463-final/baseline-a-large-fresh.json" \
  --output-json "$baseline_root/build/benchmarks/issue-463-final/baseline-b-large-warm.json" \
  --output-csv "$baseline_root/build/benchmarks/issue-463-final/baseline-b-large-warm.csv"
~~~

The reported large medians pool both baseline blocks and both candidate blocks
(30 samples per revision and start mode). Control rows use 30 samples per
revision. Every row passed energy/force drift, per-system status, convergence,
SCC-iteration, and, for WARM, same-library FRESH-reference gates. Median SCC
iterations are unchanged: 17 versus 17 on large FRESH, 2 versus 2 on WARM, and
18 versus 18 on control FRESH.

| Coordinate | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| 722 AO, B=1 FRESH | 1198.276 ms | 1165.991 ms | 2.694% faster |
| 722 AO, B=1 WARM | 289.660 ms | 258.415 ms | 10.787% faster |
| 122 AO, B=8 FRESH | 138.040 ms | 137.259 ms | 0.566% faster |
| 122 AO, B=32 FRESH | 193.714 ms | 193.629 ms | 0.043% faster |
| 122 AO, B=128 FRESH | 439.222 ms | 438.269 ms | 0.217% faster |
| 122 AO, B=8 WARM | 32.869 ms | 32.165 ms | 2.141% faster |
| 122 AO, B=32 WARM | 60.318 ms | 59.610 ms | 1.174% faster |
| 122 AO, B=128 WARM | 179.652 ms | 178.514 ms | 0.634% faster |

## Nsight Systems audit

Baseline and candidate were each profiled for one 722-AO B=1 FRESH coordinate
with one warmup and one measured public call. These are the exact profile and
derivation commands:

~~~bash
nsys=/group/software/cuda-12.9.1/bin/nsys
baseline_dir="$baseline_root/build/benchmarks/issue-463-final"
candidate_dir="$candidate_root/build/benchmarks/issue-463-final"

"$nsys" profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output="$baseline_dir/nsys-baseline" \
  "$benchmark_python" "$baseline_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$baseline_root/build/cuda-issue463-baseline/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 362 --batch-sizes 1 \
  --warmups 1 --repetitions 1 --start-mode fresh \
  --output-json "$baseline_dir/nsys-baseline.json" \
  --output-csv "$baseline_dir/nsys-baseline.csv"

"$nsys" profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output="$candidate_dir/nsys-candidate" \
  "$benchmark_python" "$candidate_root/benchmarks/natoms_scaling.py" \
  --engine xtbloom \
  --library "$candidate_root/build/cuda-issue463-dev/libxtbloom.so" \
  --backend cuda --device-id 0 \
  --cpu-threads 1 --property force --natoms 362 --batch-sizes 1 \
  --warmups 1 --repetitions 1 --start-mode fresh \
  --output-json "$candidate_dir/nsys-candidate.json" \
  --output-csv "$candidate_dir/nsys-candidate.csv"

"$nsys" stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum \
  --format csv --force-overwrite=true \
  --output "$baseline_dir/nsys-baseline-stats" \
  "$baseline_dir/nsys-baseline.nsys-rep"

"$nsys" stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum \
  --format csv --force-overwrite=true \
  --output "$candidate_dir/nsys-candidate-stats" \
  "$candidate_dir/nsys-candidate.nsys-rep"

rg 'integral_force|publish_integrals_kernel|Gfn2H0Force' \
  "$baseline_dir/nsys-baseline-stats_cuda_gpu_kern_sum.csv" \
  "$candidate_dir/nsys-candidate-stats_cuda_gpu_kern_sum.csv"
~~~

The instrumented samples were 1195.114 ms baseline and 1163.346 ms candidate,
a 2.658% reduction, with 17 SCC iterations and passing correctness in both.

Across each complete setup/warmup/measured process, allocation and lifetime
counts are identical: 32 cudaMalloc, 38 cudaFree, 11 cudaMallocHost, and
11 cudaFreeHost. Transfer/memset API counts are identical: one cudaMemcpy,
81 cudaMemcpyAsync, three cudaMemcpyToSymbol, one cudaMemset, and
328 cudaMemsetAsync. Graph and synchronization counts are also identical,
including two Graph instantiations, two Graph launches, one Graph upload,
two Graph-exec destructions, two Graph destructions, ten event
synchronizations, five stream synchronizations, and 20 device synchronizations.

CUDA memory-operation counts and bytes are identical: memset 282.530 MB/257,
device-to-device 155.230 MB/3, host-to-device 141.344 MB/56, and
device-to-host 0.028 MB/26. cudaLaunchKernel increases from 1557 to 1560, the
expected three launches for the separated per-system topology validation and
tiled numerical preflight boundary. This adds no allocation, transfer, Graph
resource, polling, event/stream/device synchronization, or host work.

The capture includes setup and is not represented as a zero-allocation
steady-state-only trace. API and memory summaries are retained in full.
nsys-targeted-kernel-summary.csv retains the exact rows for the changed
integral/H0 staging kernels, with the overloaded H0 publication names made
explicit. The complete time-sorted kernel reports are reproducible from the
command above but omitted because the repository-wide evidence directory was
already close to its 16 MiB cap; they are not needed for the allocation,
transfer, Graph, or synchronization acceptance claim.

Raw .nsys-rep and generated SQLite files remain under ignored build paths and
are prohibited from the repository because they can contain process
environment data. Six complete sanitized API/memory CSV reports and one
targeted kernel CSV are retained.

## Limitations

- Performance was measured on one consumer GPU, RTX 5090. No A100 or other
  stronger-FP64 GPU speedup is claimed.
- The latency matrix uses homogeneous deterministic alkanes. Heterogeneous
  ragged, unrestricted, failure, host/device/mixed, and QM/MM behavior is
  correctness-qualified by the 187-test real-GPU matrix but has no separate
  latency claim here.
- Hardware-counter access was unavailable, so the evidence demonstrates
  end-to-end and CUDA-runtime behavior rather than achieved FP64 utilization.
- This evidence supports directions 1-3 only through issue #463's completed
  integral/H0 leaf. It does not implement or justify mixed precision, SCC
  policy changes, or the explicitly excluded directions 4 and 5.
