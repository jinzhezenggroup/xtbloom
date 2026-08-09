# Issue #256: matched-accuracy cross-engine benchmark, CPU + CUDA
# (batch=1 cold / batch=128 auto-warm / batch=512 cold)

Corrected-protocol measurements that replace the misleading archived #231 rows.
Every engine runs with the same `--cpu-threads 16` budget on every panel,
with the **same SCC energy accuracy (1e-4)** and explicit start semantics:

- **batch=1 (cold)**: every measured sample is a genuine cold start (gpuxtb
  FRESH; xTB/tblite rebuild their calculator; dxtb resets per call). Rows are
  `final-<engine>-cold.*`.
- **batch=128 (auto-warm)**: the first call in a cell is cold, every later
  warmup and measured sample continues warm for gpuxtb (strict `WARM`), xTB
  and tblite (persistent calculator). All 128 systems per call are distinct
  conformers. Rows are `final-<engine>-b128.*`.
- **batch=512 (cold)**: a ragged cold-start batch of 512 *distinct* systems
  per call (gpuxtb FRESH each sample; references rebuilt per sample). Replaces
  the earlier MD-trajectory panel, which duplicated the batch=1 regime. Rows
  are `final-<engine>-b512.*`.

xTB 6.7.1 and tblite 0.7.0 both export `libtblite.so.0` but need different
tblite versions, so their rows are measured in separate processes
(`LD_LIBRARY_PATH=/tmp/gpuxtb-reference-env.E0KcEA/lib` for xTB,
`/tmp/tblite-pr169-6f7f1e2` for tblite). dxtb runs from its dedicated
environment with 16 Torch intra-op threads.

## Data-correction note (2026-08-09)

- The first archived `final-<engine>-traj.*` files leaked one job-less
  steady-state batch=1 row (the auto-warm `--natoms 62` matrix cell that a
  trajectory invocation also runs) into the trajectory output, which made
  the batch=1 cold panel dip at 62 atoms. The runner now records each cell
  row's `start_policy` and the plotter excludes auto-warm rows from the
  batch=1 cold panel. The MD-trajectory panel is removed entirely (it was
  too similar to batch=1) and replaced by batch=512 cold.
- SCC accuracy is now **matched at 1e-4 for every engine** (gpuxtb via
  `--scc-charge-tolerance 1e-4 --scc-energy-tolerance 1e-4`; xTB/tblite keep
  their default `accuracy=1e-4`, max 500 iterations). The earlier gpuxtb
  rows ran conformance-tight 1e-10/1e-12, which forced strictly more SCC
  iterations and made gpuxtb look 1.3-3.0x slower at batch=1; that gap was a
  measurement artifact, not runtime behavior.
- gpuxtb CPU rows were re-measured on 2026-08-09 at the PR merged head
  `b51452b` (origin/main merged into `perf/natoms-readme-benchmark`,
  including CUDA fixes #247/#252/#254/#257/#259/#261). The 302-atom batch=128
  CPU row is back inside `final-gpuxtb-b128.*` (the old separate
  `final-gpuxtb-cpu-b128-302` file is gone).
- gpuxtb **CUDA** rows were re-measured on 2026-08-09 at the next PR merged
  head `1d2838b` (origin/main merged again, now also including the CUDA
  eigensolver changes #244 and #263). All other engines keep their rows
  unchanged. The batch=1 @362 CUDA eigensolve cliff dropped from 10160.9 ms
  to 2546.97 ms (the #263 large-singleton eigensolver); every other CUDA row
  is unchanged within noise.

## Environment

- node3 (SLURM): Ubuntu 22.04.5, kernel 6.8.0-110, AMD EPYC 7K62 48-core,
  single socket; CPU rows pinned to 16 threads. Every GPU row ran under
  `srun -n 1 --gres=gpu:1 -c 16 -w node3`.
- CPU gpuxtb and CUDA gpuxtb: shared Release builds (`-O3`, MKL LP64 shim;
  CUDA `sm_120`, NVCC 12.9.1) from the PR merged heads (`b51452b` for CPU,
  `1d2838b` for the re-measured CUDA rows); CUDA runtime
  `LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib`.
- GPU: NVIDIA GeForce RTX 5090 (32 GiB), driver 580.95.05 / CUDA 13.0
  driver. CUDA rows use host-pointed descriptors with a trailing
  `cudaDeviceSynchronize`; correctness downloads happen after timing.
- xTB 6.7.1 `libxtb.so.6.7.1`; tblite 0.7.0 `libtblite.so.0.7.0`;
  dxtb 0.4.0 (PyTorch CPU 16 threads; PyTorch 2.13.0+cu130 on CUDA).
- SCC accuracy: 1e-4 for every engine (matched), up to 500 iterations.

### Build and optimization flags (CPU fairness)

No engine in this comparison was compiled with `-march=native`; the CPU rows
therefore do not give any single engine a hard ISA advantage:

- **gpuxtb CPU**: system GCC (`/usr/bin/c++`), `-O3 -DNDEBUG`, generic x86-64
  baseline (no `-march` / `-mtune`). The eigensolve uses runtime-loaded MKL
  LP64 (issue #30 isolated shim); MKL internally dispatches to AVX2/AVX512
  at run time.
- **xTB 6.7.1**: conda-forge binary, linked against conda OpenBLAS 0.3.33
  (runtime dispatch), gfortran/gcc from the conda env, generic x86-64.
- **tblite 0.7.0** (`/tmp/tblite-pr169-6f7f1e2`): Meson `-Dbuildtype=release`
  with conda gfortran, `-O3` (a few files `-O1`), linked against conda
  OpenBLAS 0.3.33, generic x86-64.
- **dxtb 0.4.0**: pure Python over PyTorch 2.13.0+cu130; the PyTorch wheel
  reports CPU capability AVX2 and dispatches AVX2/AVX512 kernels at run time,
  which (if anything) favors the reference side.

Both gpuxtb and the references thus run runtime-dispatching vendor BLAS (MKL
vs OpenBLAS) on top of generic `-O3` builds, so the latency comparison is
not distorted by a bespoke `-march` for any engine. The CUDA rows use the
same gpuxtb host build plus NVCC 12.9.1 kernels for `sm_120`.

## Commands

Per engine group (xTB/tblite in separate processes, dxtb and GPU rows in their
own processes/environments). gpuxtb rows add the matched SCC flags:

```bash
# panel 1 (cold)
python3 benchmarks/natoms_cross_engine.py --library <lib> \
  --engines <gpuxtb-cpu|xtb|tblite|dxtb-cpu> \
  --natoms 14,32,62,122,242,362 --batch-sizes 1 \
  --warmups 1 --repetitions 3 --cpu-threads 16 --start-policy cold \
  --scc-charge-tolerance 1e-4 --scc-energy-tolerance 1e-4 \
  --output-json final-<engine>-cold.json --output-csv final-<engine>-cold.csv

# panel 2 (batch=128, auto-warm)
python3 benchmarks/natoms_cross_engine.py --library <lib> \
  --engines <engine> --natoms-large-batch 14,32,62,122,242,302 --batch-sizes 128 \
  --warmups 1 --repetitions 3 --cpu-threads 16 --start-policy auto-warm \
  --scc-charge-tolerance 1e-4 --scc-energy-tolerance 1e-4 \
  --output-json final-<engine>-b128.json --output-csv final-<engine>-b128.csv

# panel 3 (batch=512, cold)
python3 benchmarks/natoms_cross_engine.py --library <lib> \
  --engines <engine> --natoms-large-batch 14,32,62,122 --batch-sizes 512 \
  --warmups 1 --repetitions 3 --cpu-threads 16 --start-policy cold \
  --scc-charge-tolerance 1e-4 --scc-energy-tolerance 1e-4 \
  --output-json final-<engine>-b512.json --output-csv final-<engine>-b512.csv

# CUDA rows (same panels; dxtb-cuda via its environment)
srun -n 1 --gres=gpu:1 -c 16 -w node3 bash -c ' \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib \
  python3 benchmarks/natoms_cross_engine.py --library <cuda-lib> \
  --engines gpuxtb-cuda --natoms 14,32,62,122,242,362 --batch-sizes 1 \
  --warmups 1 --repetitions 3 --cpu-threads 16 --start-policy cold \
  --scc-charge-tolerance 1e-4 --scc-energy-tolerance 1e-4 \
  --output-json final-gpuxtb-cuda-cold.json --output-csv final-gpuxtb-cuda-cold.csv'
```

Figure: `python3 benchmarks/plot_natoms_cross_engine.py --artifact ...-cold.json
--artifact ...-b128.json --artifact ...-b512.json
--artifact final-gpuxtb-cuda-cold.json --artifact final-gpuxtb-cuda-b128.json
--artifact final-gpuxtb-cuda-b512.json
--artifact final-dxtb-cuda-cold.json --artifact final-dxtb-cuda-b128.json
--artifact final-dxtb-cuda-b512.json --commit <sha> --output natoms_cross_engine.svg`
(also `docs/assets/natoms_cross_engine.svg`).

## Results (median energy+force latency, ms)

Panel 1 (batch=1, cold start, 16 threads, SCC 1e-4):

| natoms | gpuxtb CPU | gpuxtb CUDA | xTB | tblite | dxtb CPU\* | dxtb CUDA\* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 3.0 | 14.5 | 4.0 | 5.3 | 148.0 | 366.2 |
| 32 | 10.9 | 38.2 | 15.6 | 18.6 | 206.1 | 400.7 |
| 62 | 33.6 | 60.4 | 52.2 | 60.5 | 341.6 | 444.5 |
| 122 | 120.3 | 172.8 | 178.7 | 205.8 | 748.0 | 513.2 |
| 242 | 548.5 | 828.0 | 651.5 | 707.8 | -- | -- |
| 362 | 1448.4 | 2547.0 | 1405.8 | 1545.3 | -- | -- |

Panel 2 (batch=128, distinct conformers, first call cold then WARM, 16 threads):

| natoms | gpuxtb CPU | gpuxtb CUDA | xTB | tblite | dxtb CPU\* | dxtb CUDA\* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 18.2 | 28.2 | 205.4 | 181.8 | 1456.7 | 1162.7 |
| 32 | 53.6 | 60.7 | 677.3 | 588.3 | 3938.3 | 2363.5 |
| 62 | 177.6 | 182.2 | 2113.5 | 1634.8 | 16578.6 | 6755.7 |
| 122 | 641.1 | 637.0 | 7394.7 | 5342.5 | -- | -- |
| 242 | 2760.7 | 2578.0 | -- | -- | -- | -- |
| 302 | 4520.2 | -- | -- | -- | -- | -- |

Panel 3 (batch=512, cold start, 16 threads):

| natoms | gpuxtb CPU | gpuxtb CUDA | xTB | tblite | dxtb CPU\* | dxtb CUDA\* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 64.9 | 207.0 | 1926.0 | 2398.2 | 6291.9 | 3465.7 |
| 32 | 322.8 | 289.5 | 8216.3 | 9842.7 | 27699.0 | 15732.3 |
| 62 | 1139.0 | 1094.6 | 26470.9 | 29313.2 | 122434.0 | OOM |
| 122 | 4633.1 | 3796.7 | 91984.1 | 99209.3 | -- | -- |

\* dxtb resets per call by design (Torch autograd prevents warm continuation
of the measured public path), so its rows are cold-every-call. dxtb-cuda
@62 x 512 ran out of device memory (OOM) and is recorded as error in
`final-dxtb-cuda-b512.json`; the row is excluded from the figure.

## Key findings

- **Matched accuracy changes the story at batch=1.** With every engine at
  SCC 1e-4, gpuxtb CPU no longer carries a strict-tolerance iteration
  penalty and is 1.2-1.7x *faster* than xTB/tblite at 14-242 atoms
  (62: 33.6 ms vs 52.2/60.5; 242: 548 vs 651/708), reaching parity at 362
  (1448 vs 1406/1545). The previous 1.3-3.0x "single-molecule gap" was the
  conformance-tight 1e-10/1e-12 measurement artifact.
- gpuxtb batch=1 does not use its 16 annotated workers (1 worker active;
  the outer-batch pool is idle for a single system). At `--cpu-threads 1`
  the batch=128 per-call advantage collapses toward 1.0-1.3x vs tblite
  (62: 2792 ms vs 3508; 122: 9664 vs 10091), confirming the ragged-batch
  speedup is a 16-worker cross-system parallelism effect, and that gpuxtb
  per-system cost is otherwise comparable at equal accuracy.
- **Ragged-batch advantage is large and real at 16 threads**: batch=128
  gpuxtb CPU is ~10-12x faster than xTB/tblite per call (62: 178 vs 2113/
  1635; extending to 302 atoms), and the new **batch=512 cold panel** shows
  gpuxtb CPU ~20-30x faster (62: 1139 ms vs 26471/29313; 122: 4633 vs
  91984/99209) and gpuxtb CUDA ~10-25x faster (62: 1093 ms vs 26471/29313),
  because gpuxtb solves the whole ragged batch in one call across its worker
  pool while the reference adapters loop systems serially.
- gpuxtb CUDA at batch=1 carries a fixed per-call cost (14-62 ms up to 62
  atoms) and a single-system eigensolve cliff past ~272 atoms (the cuSOLVER
  `syevd` path degenerates into thousands of tiny serial kernels; forces add
  <1%; 362 atoms costs 2547 ms vs 1448 ms on CPU). Merging the #263
  large-singleton CUDA eigensolver into this PR's branch cut that 362-atom
  cost ~4x (10161 ms before, 2547 ms after) with unchanged SCC iterations
  (6) and identical energies. The cliff still amortizes at batch=128/512:
  @242 x 128 it is already faster than gpuxtb CPU (2576 vs
  2761 ms), and at batch=512 @62-122 it is at or below CPU latency.

## Files

- `final-<engine>-{cold,b128,b512}.json/.csv` for gpuxtb-cpu, xtb, tblite, dxtb-cpu
- `final-{gpuxtb-cuda,dxtb-cuda}-{cold,b128,b512}.json/.csv` (gpuxtb-cuda
  re-measured 2026-08-09 at merged head `1d2838b`; dxtb-cuda unchanged)
- `natoms_cross_engine.svg` (rendered figure)
- `README.md`, `SHA256SUMS`
