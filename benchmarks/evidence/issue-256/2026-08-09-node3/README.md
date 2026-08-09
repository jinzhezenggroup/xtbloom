# Issue #256: corrected cross-engine benchmark, CPU + CUDA (cold / auto-warm / WARM)

Corrected-protocol measurements that replace the misleading archived #231 rows.
Every engine runs with the same `--cpu-threads 16` budget on every panel, and
each panel has explicit SCC start semantics:

- **batch=1 (cold)**: every measured sample is a genuine cold start (gpuxtb
  FRESH; xTB/tblite rebuild their calculator; dxtb resets per call). Rows are
  `final-<engine>-cold.*`.
- **batch=128 (auto-warm)**: the first call in a cell is cold (gpuxtb FRESH;
  xTB/tblite/dxtb start cold by construction), every later warmup and measured
  sample continues warm for gpuxtb (strict `WARM`), xTB and tblite (persistent
  calculator). All 128 systems per call are distinct conformers. Rows are
  `final-<engine>-b128.*`.
- **trajectory (WARM)**: nearly identical MD frames; gpuxtb strict `WARM`,
  references persistent. Rows are `final-<engine>-traj.*`.

xTB 6.7.1 and tblite 0.7.0 both export `libtblite.so.0` but need different
tblite versions, so their rows are measured in separate processes
(`LD_LIBRARY_PATH=/tmp/gpuxtb-reference-env.E0KcEA/lib` for xTB,
`/tmp/tblite-pr169-6f7f1e2` for tblite). dxtb runs from its dedicated
environment with 16 Torch intra-op threads.

## Data-correction note (2026-08-09)

The first archived `final-<engine>-traj.*` files leaked one job-less
steady-state batch=1 row (the auto-warm `--natoms 62` matrix cell that a
trajectory invocation also runs) into the trajectory output. That warm row
was ~3.6x cheaper than a genuine cold sample and made the batch=1 cold panel
dip at 62 atoms. The runner now records each cell row's `start_policy`, the
plotter excludes auto-warm rows from the batch=1 cold panel, and the leaked
rows were removed from every archived `-traj` artifact. The earlier files
without `start_policy` in their rows are treated as cold by the plotter.

The CUDA rows in this archive were re-measured on 2026-08-09 at the PR merged
head `506da8c5` (merge of main `fa05133` into `perf/natoms-readme-benchmark`,
i.e. including CUDA fixes #247/#252/#254/#257) with the corrected runner.
The previous #231 figures dropped CUDA entirely; the current figure restores
`gpuxtb-cuda` and `dxtb-cuda` lines.

## Environment

- node3 (SLURM): Ubuntu 22.04.5, kernel 6.8.0-110, AMD EPYC 7K62 48-core,
  single socket; CPU rows pinned to 16 threads. Every GPU row ran under
  `srun -n 1 --gres=gpu:1 -c 16 -w node3`.
- CPU gpuxtb: shared Release build `build/batch1-cpu/libgpuxtb.so.0.1.0` from
  main-tree `81e3d67` (clean; dev timing instrumentation removed).
- CUDA gpuxtb: shared Release build (sm_120, CUDA toolkit 12.9.1,
  MKL LP64 shim) from the PR merged head `506da8c5`; runtime
  `LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib`.
- GPU: NVIDIA GeForce RTX 5090 (32 GiB), driver 580.95.05 / CUDA 13.0
  driver. CUDA rows use host-pointed descriptors with a trailing
  `cudaDeviceSynchronize`; correctness downloads happen after timing.
- xTB 6.7.1 `libxtb.so.6.7.1`; tblite 0.7.0 `libtblite.so.0.7.0`;
  dxtb 0.4.0 (PyTorch CPU 16 threads; PyTorch 2.13.0+cu130 on CUDA).
- gpuxtb SCC conformance-tight (charge 1e-10, energy 1e-12, up to 500 iters);
  xTB/tblite default acc 1e-4. gpuxtb timings include strictly more SCC work.

## Commands

Per engine group (xTB/tblite in separate processes, dxtb and GPU rows in their
own processes/environments):

```bash
# panel 1 (cold)
python3 benchmarks/natoms_cross_engine.py --library <lib> \
  --engines <gpuxtb-cpu|xtb|tblite|dxtb-cpu> \
  --natoms 14,32,62,122,242,362 --batch-sizes 1 \
  --warmups 1 --repetitions 3 --cpu-threads 16 --start-policy cold \
  --output-json final-<engine>-cold.json --output-csv final-<engine>-cold.csv

# panel 2 (batch=128, auto-warm)
python3 benchmarks/natoms_cross_engine.py --library <lib> \
  --engines <engine> --natoms-large-batch 14,32,62,122 --batch-sizes 128 \
  --warmups 1 --repetitions 3 --cpu-threads 16 --start-policy auto-warm \
  --output-json final-<engine>-b128.json --output-csv final-<engine>-b128.csv

# panel 3 (trajectory, WARM)
python3 benchmarks/natoms_cross_engine.py --library <lib> \
  --engines <engine> --natoms 62 --batch-sizes 1 \
  --warmups 1 --repetitions 3 --cpu-threads 16 \
  --trajectory --trajectory-natoms 32,62,122,242 --trajectory-frames 6 \
  --output-json final-<engine>-traj.json --output-csv final-<engine>-traj.csv

# CUDA rows (same panels; dxtb-cuda via its environment)
srun -n 1 --gres=gpu:1 -c 16 -w node3 bash -c ' \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib \
  python3 benchmarks/natoms_cross_engine.py --library <cuda-lib> \
  --engines gpuxtb-cuda --natoms 14,32,62,122,242,362 --batch-sizes 1 \
  --warmups 1 --repetitions 3 --cpu-threads 16 --start-policy cold \
  --output-json final-gpuxtb-cuda-cold.json --output-csv final-gpuxtb-cuda-cold.csv'
```

Figure: `python3 benchmarks/plot_natoms_cross_engine.py --artifact ...-cold.json
--artifact ...-b128.json --artifact ...-traj.json --artifact final-gpuxtb-cuda-cold.json
--artifact final-gpuxtb-cuda-b128.json --artifact final-gpuxtb-cuda-traj.json
--artifact final-dxtb-cuda-cold.json --artifact final-dxtb-cuda-b128.json
--artifact final-dxtb-cuda-traj.json --commit <sha> --output natoms_cross_engine.svg`
(also `docs/assets/natoms_cross_engine.svg`).

## Results (median energy+force latency, ms)

Panel 1 (batch=1, cold start, 16 threads):

| natoms | gpuxtb CPU | gpuxtb CUDA | xTB | tblite | dxtb CPU | dxtb CUDA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 3.3 | 26.6 | 4.0 | 5.3 | 148.0 | 366.2 |
| 32 | 19.8 | 80.2 | 15.6 | 18.6 | 206.1 | 400.7 |
| 62 | 76.0 | 191.5 | 52.2 | 60.5 | 341.6 | 444.5 |
| 122 | 339.3 | 565.6 | 178.7 | 205.8 | 748.0 | 513.2 |
| 242 | 1679.9 | 2467.2 | 651.5 | 707.8 | -- | -- |
| 362 | 4659.7 | 14628.9 | 1405.8 | 1545.3 | -- | -- |

Panel 2 (batch=128, distinct conformers, first call cold then WARM, 16 threads):

| natoms | gpuxtb CPU | gpuxtb CUDA | xTB | tblite | dxtb CPU\* | dxtb CUDA\* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 15.3 | 28.1 | 205.4 | 181.8 | 1456.7 | 1162.7 |
| 32 | 52.3 | 60.6 | 677.3 | 588.3 | 3938.3 | 2363.5 |
| 62 | 174.4 | 216.9 | 2113.5 | 1634.8 | 16578.6 | 6755.7 |
| 122 | 651.3 | 735.6 | 7394.7 | 5342.5 | -- | -- |
| 242 | 3115.2 | 2851.9 | -- | -- | -- | -- |

\* dxtb resets per call by design (Torch autograd prevents warm continuation
of the measured public path), so its rows are cold-every-call.

Panel 3 (trajectory, WARM continuation, ms/frame, 16 threads):

| natoms | gpuxtb CPU | gpuxtb CUDA | xTB | tblite | dxtb CPU\* | dxtb CUDA\* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 18.4 | 73.1 | 13.9 | 17.7 | 203.7 | 394.9 |
| 62 | 70.8 | 178.1 | 48.9 | 53.6 | 348.9 | 436.4 |
| 122 | 300.5 | 520.9 | 164.3 | 185.2 | -- | 508.9 |
| 242 | 1785.9 | 2359.8 | 619.9 | 683.8 | -- | -- |

## Key findings

- With equal 16-thread budgets and honest cold/vs/warm semantics, gpuxtb's
  real single-molecule disadvantage is 1.3-3.0x (cold) and 1.3-2.9x (WARM
  trajectory), NOT the 6.4x in the archived #231 figure (which compared a
  warm-continued 2-iteration tblite against FRESH 17-iteration gpuxtb).
- gpuxtb batch=1 does not use its 16 annotated workers: FRESH @242 is 1593 ms
  at cpu_threads=1 and 1592 ms at 16. The outer-batch pool is idle for one
  system; issuing #256 records the measured per-iteration breakdown (~92 ms:
  eigensolve 57, potentials+H 16.5, mulliken 14.4) and the dsyevd threading
  ceiling (MKL 2.6x, OpenBLAS 1.6x at n=484).
- gpuxtb's ragged-batch advantage is real and large: batch=128 from 14-122
  atoms is 9-13x faster than xTB/tblite, because it parallelizes systems
  across the worker pool while reference adapters loop serially.
- gpuxtb CUDA is a fixed per-call cost at batch=1 (26-190 ms for 14-62
  atoms; 362 atoms is dominated by a slow force stage, 14 629 ms) and
  becomes competitive at batch=128 where the fixed overhead is amortized:
  @242 x 128 systems gpuxtb CUDA (2852 ms) edges out gpuxtb CPU (3115 ms).
  The re-measured batch=128 CUDA rows are 5-7x faster than the CUDA rows
  archived under #231, which predated the #252/#254/#257 CUDA fixes.

## Files

- `final-<engine>-{cold,b128,traj}.json/.csv` for gpuxtb-cpu, xtb, tblite, dxtb-cpu
- `final-{gpuxtb-cuda,dxtb-cuda}-{cold,b128,traj}.json/.csv` (re-measured 2026-08-09)
- `natoms_cross_engine.svg` (rendered figure)
- `README.md`, `SHA256SUMS`
