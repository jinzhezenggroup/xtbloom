# Issue #256: honest 16-thread batch-1 cross-engine evidence (cold + trajectory)

This bundle measures the corrected protocol after the harness fix in commit
`9685775` ("bench: run all cross-engine rows with equal worker budgets and cold
samples"): every engine runs with the same `--cpu-threads 16` budget on all
three panels, and panel-1 rows use `--cold-samples` so xTB/tblite rebuild
their calculator every measured sample (cold SAD start) instead of
warm-continuing from the previous sample's converged density.

## Environment

- Host: `node3` (SLURM), Ubuntu 22.04.5, kernel 6.8.0-110.
- CPU: AMD EPYC 7K62 48-core (single socket); each row pinned to 16 workers.
- gpuxtb: shared Release build of `perf/batch1-cpu-match-tblite` at clean main
  tree `81e3d67` + local dev-only timing instrumentation removed; MPI of the
  ARC... library: `/home/jzzeng/codes/gpuxtb/build/batch1-cpu/libgpuxtb.so.0.1.0`.
- xTB: 6.7.1 `libxtb.so.6.7.1` (reference env, resolved with
  `LD_LIBRARY_PATH=/tmp/gpuxtb-reference-env.E0KcEA/lib`; its `DT_NEEDED`
  `libtblite.so.0` is tblite 0.6, so xTB and the 0.7 tblite rows are measured
  in separate processes to avoid a shared-SONAME clash).
- tblite: 0.7.0 `libtblite.so.0.7.0` (pr169 build, resolved with
  `LD_LIBRARY_PATH=/tmp/tblite-pr169-6f7f1e2`).

## Commands

Panel-1 (batch=1, cold, 16 threads), one process per engine:

```bash
# gpuxtb (FRESH is already cold; no --cold-samples needed)
python3 benchmarks/natoms_cross_engine.py \
  --library /home/jzzeng/codes/gpuxtb/build/batch1-cpu/libgpuxtb.so.0.1.0 \
  --engines gpuxtb-cpu --natoms 62,122,242,362 --batch-sizes 1 \
  --warmups 1 --repetitions 3 --cpu-threads 16 \
  --output-json gpuxtb-cold.json --output-csv gpuxtb-cold.csv

# xTB (cold)
LD_LIBRARY_PATH=/tmp/gpuxtb-reference-env.E0KcEA/lib \
python3 benchmarks/natoms_cross_engine.py \
  --library <dummy> --engines xtb \
  --xtb-library /tmp/gpuxtb-reference-env.E0KcEA/lib/libxtb.so.6.7.1 \
  --natoms 62,122,242,362 --batch-sizes 1 --warmups 1 --repetitions 3 \
  --cpu-threads 16 --cold-samples \
  --output-json xtb-cold.json --output-csv xtb-cold.csv

# tblite (cold)
LD_LIBRARY_PATH=/tmp/tblite-pr169-6f7f1e2 \
python3 benchmarks/natoms_cross_engine.py \
  --library <dummy> --engines tblite \
  --tblite-library /tmp/tblite-pr169-6f7f1e2/libtblite.so.0.7.0 \
  --natoms 62,122,242,362 --batch-sizes 1 --warmups 1 --repetitions 3 \
  --cpu-threads 16 --cold-samples \
  --output-json tblite-cold.json --output-csv tblite-cold.csv
```

Panel-3 (trajectory, WARM continuation, 16 threads, 6 frames x 3 reps):

```bash
LD_LIBRARY_PATH=/tmp/gpuxtb-reference-env.E0KcEA/lib \
python3 benchmarks/natoms_cross_engine.py \
  --library /home/jzzeng/codes/gpuxtb/build/batch1-cpu/libgpuxtb.so.0.1.0 \
  --engines gpuxtb-cpu,xtb --xtb-library <...> \
  --natoms 62 --batch-sizes 1 --warmups 1 --repetitions 3 \
  --cpu-threads 16 --trajectory --trajectory-natoms 62,122,242 \
  --trajectory-frames 6 --output-json traj-xtb.json ...

# same with --engines gpuxtb-cpu,tblite --tblite-library <...> + LD path
```

## Results (median energy+force latency, ms; energy+force, 300 K)

Panel 1 (single molecule, cold start, 16 threads):

| natoms | gpuxtb FRESH | xTB | tblite | gpuxtb/tblite |
| --- | ---: | ---: | ---: | ---: |
| 62 | 75.6 | 52.3 | 61.5 | 1.23x |
| 122 | 336.3 | 179.7 | 208.0 | 1.62x |
| 242 | 1689.0 | 650.5 | 727.1 | 2.32x |
| 362 | 4673.1 | 1385.2 | 1540.7 | 3.03x |

Panel 3 (nearly identical frames, WARM/hot start, 16 threads):

| natoms | gpuxtb WARM | xTB | tblite | gpuxtb/tblite |
| --- | ---: | ---: | ---: | ---: |
| 62 | 70.0 | 46.3 | 53.1 | 1.32x |
| 122 | 300.1 | 156.9 | 175.7 | 1.71x |
| 242 | 1787.8 | 608.8 | 660.3 | 2.71x |

## Interpretation

- The archived #231 "6.4x @242 batch=1" was an artifact of warm-continued
  xTB/tblite (2 iterations per sample, verified by instrumenting libtblite)
  against FRESH gpuxtb (17 iterations). Under the corrected equal-thread cold
  protocol the true gap is 1.2-3.0x and comes from per-SCC-iteration cost
  (~92 ms gpuxtb vs ~50 ms xTB/tblite at 242) plus gpuxtb's strictly tighter
  SCC tolerance (17-18 iterations vs ~14).
- gpuxtb batch=1 still does not use its 16 annotated workers (FRESH @242 is
  identical at cpu_threads=1 and 16, 1593 vs 1592 ms): the outer-batch pool is
  idle for a single system. Closing the per-iteration gap is scoped in issue
  #256 (thread the eigensolve block and the scalar potentials/mulliken loops
  over the idle pool at batch=1; dsyevd n=484 multithreading ceiling measured
  2.6x on MKL, 1.6x on OpenBLAS).

## Files

- `gpuxtb-cold.json/.csv`, `xtb-cold.json/.csv`, `tblite-cold.json/.csv`
- `traj-xtb.json/.csv` (gpuxtb + xTB trajectory), `traj-tblite.json/.csv`
  (gpuxtb + tblite trajectory)
- `SHA256SUMS`

gpuxtb SCC is deliberately conformance-tight (charge 1e-10, energy 1e-12, up
to 500 iterations); xTB/tblite run their default accuracy 1e-4. gpuxtb timing
therefore includes strictly more SCC work at the same iteration count.