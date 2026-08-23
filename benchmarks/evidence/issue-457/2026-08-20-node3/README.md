# Issue #457 pure-FP64 CUDA density-contraction evidence

This bundle qualifies the issue #457 change that distributes each GFN1/GFN2
density-matrix contraction across multiple CUDA CTAs without mixed precision.
The implementation changes scheduling only: one thread still owns each matrix
pair and executes the existing ordered binary64 orbital/FMA accumulation.

The narrow conclusion on this RTX 5090 is:

- large-AO, batch-one FRESH public energy+force calls improve by 1.88x to
  5.04x, and strict WARM calls improve by 1.42x to 2.94x;
- the 122-AO control improves at B=1, 8, and 32, while B=128 is essentially
  neutral; no control regresses by the issue's material 2% threshold;
- therefore the change is not B=1-only, but its value decreases as existing
  system-level batch concurrency already fills the GPU;
- every timed row passes the unchanged correctness gates, every WARM row
  passes its same-library FRESH-reference gate, and SCC iteration medians are
  unchanged.

`performance-summary.csv` is the compact result table and
`raw-timing-samples.json` retains every timing sample used below. The original
runner JSON files were 1.2-24.7 MiB because they retain every force vector for
every sample, so they are intentionally omitted under the repository evidence
budget. Exact commands, clean revisions, binary hashes, inputs, and correctness
qualification are retained here.

## Source and implementation

- Baseline source: clean
  `45af658f666af287a5f81f9122502be20d0062c2`.
- Candidate source: clean
  `e948d63b3e4523a8e9be8e879f20f1638ff57fdc`.
- Baseline library SHA-256:
  `7af2194647a4e6c44e6b61283d8abf863ba1491c356659f79b5fe5ffc2e6c4f8`.
- Candidate library SHA-256:
  `b855f0d7e01128eb54e2294039d2138a064142b725e85dbc330f584197611e77`.
- Branch diff: 11 files, 669 insertions, 18 deletions. The public C ABI and
  public binary64 data model are unchanged; the SCC setup/iteration schema is
  internal ABI v6.
- Restricted launch grid: `(system, tile, 1)`.
- Spin launch grid: `(system, channel, tile)`.
- Setup selects a stable tile count from AO pair work and batch/channel
  capacity, targeting about 512 contraction CTAs without changing Graph node
  shapes between calls. Fully restricted batches launch one channel; mixed or
  unrestricted batches reserve two.
- Actual host-captured Graph kernel-node tests assert restricted `(1,9,1)` and
  spin `(1,channels,9)` grids with 256-thread blocks. CUDA 12.9 does not permit
  the same kernel-node query on the sealed production device-launch Graph;
  the ordinary capture uses the same production density launcher, while the
  production Graph path is covered by execution, numerical, and sanitizer
  tests.

## Environment

- Host: `node3`, Ubuntu 22.04.5 LTS, Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62 48-Core Processor, 48 physical/logical CPUs available to
  the exclusive job.
- GPU: NVIDIA GeForce RTX 5090, UUID
  `GPU-8e9c9e1a-e183-258c-0b3a-03a5ddebb2f8`, 32,607 MiB, compute capability
  12.0.
- Driver: 580.95.05.
- CUDA compiler/toolkit: 12.9.86;
  `CMAKE_CUDA_ARCHITECTURES=120`.
- Host compiler: GCC 11.4.0; CMake 4.2.1; Ninja 1.13.0.
- Build: shared Release, `XTBLOOM_ENABLE_CUDA=ON`, LP64 MKL runtime
  `/home/jzzeng/miniconda3/lib/libmkl_rt.so.3`.
- Thread boundary: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE`, `MKL_DYNAMIC=FALSE`.
- Nsight Systems: 2025.1.3.140-251335620677v0.
- Compute Sanitizer: 2025.2.1.0 build 35969825.
- Nsight Compute hardware-counter collection was unavailable because
  `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`; this bundle
  makes no achieved-FP64-utilization or hardware-counter claim.

## Correctness and validation

The exact candidate CUDA build was configured and built with:

```bash
cmake -S . -B build/cuda-issue457-final-e948d63 -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/miniconda3/lib/libmkl_rt.so.3 \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda-issue457-final-e948d63 --parallel
ctest --test-dir build/cuda-issue457-final-e948d63 -N
```

The configuration registered 187 tests. On the RTX 5090:

```bash
srun --exclusive -N1 -n1 -c48 --mem=0 --gres=gpu:1 \
  --kill-on-bad-exit=1 \
  env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/home/jzzeng/miniconda3/lib \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  ctest --test-dir build/cuda-issue457-final-e948d63 --output-on-failure
```

Result: 187/187 passed. This includes host/device/mixed public conformance and
invariants, GFN1/GFN2, restricted/unrestricted paths, ragged failure isolation,
Graphs, cache/WARM behavior, publication, ABI symbols, and CUDA dependency
checks. The focused density/setup/binding/production/large-singleton selection
also passed 5/5 after the actual Graph grid assertions were added.

The independent CPU-only shared configuration used the same LP64 MKL provider:

```bash
cmake -S . -B build/cpu-issue457-final-e948d63 -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/miniconda3/lib/libmkl_rt.so.3 \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu-issue457-final-e948d63 --parallel
ctest --test-dir build/cpu-issue457-final-e948d63 -N
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  ctest --test-dir build/cpu-issue457-final-e948d63 --output-on-failure
```

Result: 96/96 passed. The canonical Nox session passed (44 parameter tests
with one upstream-checkout skip, 122 oracle/conformance tests with three
optional-jsonschema skips), as did canonical-PyPI `uv lock --check`, the full
prek hook set, and `git diff --check`. An independent read-only review reported
LGTM with no findings.

Compute Sanitizer results and the exact owner disposition are in
`sanitizer-summary.txt`. The density direct/host-Graph control is clean under
all four tools. The production 542-AO device-tail Graph is clean under
memcheck, racecheck, and initcheck; synccheck retains exit 99 with 256 copies
of the exact issue #279 approved upstream-tool signature at
`reduce_spin_atomic_charges_kernel+0x2b0`. The tool itself did not return a
clean pass.

## Performance protocol and commands

All rows use the public synchronous C ABI through
`benchmarks/natoms_scaling.py`, request energy plus analytic forces, and use
the deterministic alkane workload. FRESH performs independent SCC
initialization for every timed call. WARM performs one untimed FRESH seed, then
strict WARM warmups and timed calls; each WARM artifact references the FRESH
JSON produced by the same library and clean revision. Energy and force gates
are both `5e-7` in atomic units.

The common scheduler and environment boundary was:

```bash
srun --exclusive -N1 -n1 -c48 --mem=0 --gres=gpu:1 \
  --kill-on-bad-exit=1 \
  env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/home/jzzeng/miniconda3/lib \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE bash -c '<commands below>'
```

Inside that single exclusive allocation, the local worktree parent contained
the retired project name forbidden by repository policy, so its absolute path
is omitted. With `<worktree-parent>` standing for that parent, the roots were:

```bash
python_bin=<worktree-parent>/issue-457-density/.venv/bin/python
baseline_root=<worktree-parent>/issue-457-baseline-clean
candidate_root=<worktree-parent>/issue-457-density
output_root=<worktree-parent>/issue-457-density/build/issue457-final-performance
baseline_library="$baseline_root/build/cuda-issue457-baseline-clean/libxtbloom.so.0.2.0"
candidate_library="$candidate_root/build/cuda-issue457-final-e948d63/libxtbloom.so.0.2.0"
```

For each root/library pair, the 122-AO control commands were:

```bash
"$python_bin" "$root/benchmarks/natoms_scaling.py" \
  --engine xtbloom --library "$library" --backend cuda --property force \
  --natoms 62 --batch-sizes 1,8,32,128 --warmups 5 --repetitions 20 \
  --start-mode fresh --energy-atol 5e-7 --force-atol 5e-7 \
  --output-json "$output_root/${label}-controls-fresh.json" \
  --output-csv "$output_root/${label}-controls-fresh.csv"

"$python_bin" "$root/benchmarks/natoms_scaling.py" \
  --engine xtbloom --library "$library" --backend cuda --property force \
  --natoms 62 --batch-sizes 1,8,32,128 --warmups 5 --repetitions 20 \
  --start-mode warm --energy-atol 5e-7 --force-atol 5e-7 \
  --cross-engine-energy-atol 5e-7 --cross-engine-force-atol 5e-7 \
  --energy-reference-json "$output_root/${label}-controls-fresh.json" \
  --output-json "$output_root/${label}-controls-warm.json" \
  --output-csv "$output_root/${label}-controls-warm.csv"
```

The large-AO B=1 commands differed only in these arguments and output names:

```bash
--natoms 122,242,272,362 --batch-sizes 1 --warmups 2 --repetitions 10
--output-json "$output_root/${label}-large-${mode}.json"
--output-csv "$output_root/${label}-large-${mode}.csv"
```

Here `label/root/library` were respectively
`baseline/$baseline_root/$baseline_library` and
`candidate/$candidate_root/$candidate_library`; `mode` was `fresh` or `warm`,
and the WARM command used the matching `${label}-large-fresh.json` reference.

## Results

The 62-atom, 122-AO control shows that the optimization is useful beyond B=1,
but the gain diminishes as batch concurrency fills the GPU:

| Batch | FRESH baseline -> candidate | FRESH gain | WARM baseline -> candidate | WARM gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 152.937 -> 126.736 ms | 17.1% | 29.949 -> 26.287 ms | 12.2% |
| 8 | 166.520 -> 142.702 ms | 14.3% | 38.627 -> 33.144 ms | 14.2% |
| 32 | 214.942 -> 195.523 ms | 9.0% | 62.589 -> 60.339 ms | 3.6% |
| 128 | 445.414 -> 440.443 ms | 1.1% | 180.439 -> 179.709 ms | 0.4% |

Large-AO B=1 calls are the main beneficiary because the old launch exposed
only one contraction CTA per system/channel:

| AO | FRESH baseline -> candidate | FRESH speedup | WARM baseline -> candidate | WARM speedup |
| ---: | ---: | ---: | ---: | ---: |
| 242 | 442.268 -> 235.744 ms | 1.88x | 77.798 -> 54.817 ms | 1.42x |
| 482 | 2069.654 -> 510.229 ms | 4.06x | 313.835 -> 130.229 ms | 2.41x |
| 542 | 3055.371 -> 855.216 ms | 3.57x | 449.449 -> 188.952 ms | 2.38x |
| 722 | 6521.070 -> 1293.447 ms | 5.04x | 929.939 -> 316.382 ms | 2.94x |

All rows are correctness-qualified. The 542-AO WARM samples contain two
high-latency observations in both baseline and candidate distributions; every
sample is retained rather than filtered, and the median conclusion is
unchanged.

## Nsight Systems audit

Baseline and candidate were each profiled for one 722-AO B=1 FRESH coordinate
with one warmup and one measured public call:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=<baseline-or-candidate-prefix> \
  "$python_bin" <matching-clean-root>/benchmarks/natoms_scaling.py \
  --engine xtbloom --library <matching-library> --backend cuda \
  --property force --natoms 362 --batch-sizes 1 --warmups 1 \
  --repetitions 1 --start-mode fresh --energy-atol 5e-7 --force-atol 5e-7 \
  --output-json <matching-output>.json --output-csv <matching-output>.csv

nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum \
  --format csv --force-overwrite=true --output <derived-prefix> \
  <capture>.nsys-rep
```

The instrumented measured medians were 6517.549 ms baseline and 1268.419 ms
candidate; these are diagnostic, not substituted for the uninstrumented
headline distributions. Across each complete setup/warmup/measured process,
the key CUDA API counts were identical: 32 `cudaMalloc`, 38 `cudaFree`, 11
`cudaMallocHost`, 11 `cudaFreeHost`, 81 `cudaMemcpyAsync`, 328
`cudaMemsetAsync`, two Graph instantiations, two Graph launches, ten event
synchronizations, five stream synchronizations, and 20 device
synchronizations. CUDA memory-operation counts and bytes were also identical.
Thus the scheduling change adds no allocation, transfer, Graph-resource, or
synchronization call relative to baseline. The capture includes setup and is
not represented as a zero-allocation steady-state-only trace.

The raw `.nsys-rep` and generated SQLite files remain under ignored `build/`
paths and are prohibited from the repository because they can contain process
environment data. Only the eight sanitized derived CSV reports are retained in
this bundle.

## Limitations

- Performance was measured on one consumer GPU, RTX 5090. No stronger-FP64 GPU
  comparison is claimed.
- The latency matrix uses homogeneous deterministic alkanes. Heterogeneous
  ragged, unrestricted, failure, host/device/mixed, and QM/MM behavior is
  correctness-qualified by the 187-test real-GPU matrix but is not assigned a
  separate latency claim here.
- Hardware-counter access was unavailable, so this evidence demonstrates
  end-to-end and CUDA-runtime behavior rather than achieved FP64 utilization.
- The single adaptive launch policy covers all batch sizes; the evidence does
  not justify a separate user-visible B=1 mode. At very large batches the
  policy is intentionally close to neutral because system-level concurrency is
  already sufficient.
