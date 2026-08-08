# Issue #256: corrected 16-thread cross-engine benchmark (cold / auto-warm / WARM)

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

## Environment

- node3 (SLURM): Ubuntu 22.04.5, kernel 6.8.0-110, AMD EPYC 7K62 48-core,
  single socket; every row pinned to 16 threads.
- gpuxtb: shared Release build `build/batch1-cpu/libgpuxtb.so.0.1.0` from
  main-tree `81e3d67` (clean; dev timing instrumentation removed).
- xTB 6.7.1 `libxtb.so.6.7.1`; tblite 0.7.0 `libtblite.so.0.7.0`;
  dxtb 0.4.0 (PyTorch CPU, 16 threads).
- gpuxtb SCC conformance-tight (charge 1e-10, energy 1e-12, up to 500 iters);
  xTB/tblite default acc 1e-4. gpuxtb timings include strictly more SCC work.

## Commands

Per engine group (xTB/tblite in separate processes, dxtb with its env):

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
```

Figure: `python3 benchmarks/plot_natoms_cross_engine.py --artifact ...-cold.json
--artifact ...-b128.json --artifact ...-traj.json ... --commit <sha> --output
natoms_cross_engine.png` (also `docs/assets/natoms_cross_engine.png`).

## Results (median energy+force latency, ms)

Panel 1 (batch=1, cold start, 16 threads):

| natoms | gpuxtb | xTB | tblite | dxtb CPU |
| --- | ---: | ---: | ---: | ---: |
| 14 | 3.3 | 4.0 | 5.3 | 148.0 |
| 32 | 19.8 | 15.6 | 18.6 | 206.1 |
| 62 | 76.0 | 52.2 | 60.5 | 341.6 |
| 122 | 339.3 | 178.7 | 205.8 | 748.0 |
| 242 | 1679.9 | 651.5 | 707.8 | -- |
| 362 | 4659.7 | 1405.8 | 1545.3 | -- |

Panel 2 (batch=128, distinct conformers, first call cold then WARM, 16 threads):

| natoms | gpuxtb | xTB | tblite | dxtb CPU* |
| --- | ---: | ---: | ---: | ---: |
| 14 | 15.3 | 205.4 | 181.8 | 1456.7 |
| 32 | 52.3 | 677.3 | 588.3 | 3938.3 |
| 62 | 174.4 | 2113.5 | 1634.8 | 16578.6 |
| 122 | 651.3 | 7394.7 | 5342.5 | -- |

\* dxtb CPU resets per call by design (Torch autograd prevents warm
continuation of the measured public path), so its rows are cold-every-call.

Panel 3 (trajectory, WARM continuation, ms/frame, 16 threads):

| natoms | gpuxtb | xTB | tblite | dxtb CPU* |
| --- | ---: | ---: | ---: | ---: |
| 32 | 18.4 | 13.9 | 17.7 | 203.7 |
| 62 | 70.8 | 48.9 | 53.6 | 348.9 |
| 122 | 300.5 | 164.3 | 185.2 | -- |
| 242 | 1785.9 | 619.9 | 683.8 | -- |

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

## Files

- `final-<engine>-{cold,b128,traj}.json/.csv` for gpuxtb-cpu, xtb, tblite, dxtb-cpu
- `natoms_cross_engine.png` (rendered figure)
- `README.md`, `SHA256SUMS`
