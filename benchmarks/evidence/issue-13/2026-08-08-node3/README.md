# Issue #13 cross-engine GFN2-xTB scaling evidence (merged-main re-run)

This bundle archives the correctness-qualified public-API benchmark matrix and
figure for the merged-main re-run of PR #231. The runner is
`benchmarks/natoms_cross_engine.py` and the figure is produced by
`benchmarks/plot_natoms_cross_engine.py`; both are committed, so the exact
runner that generated these artifacts is the current checkout.

## Environment

- Host: `node3` (SLURM cluster, `main` partition), Ubuntu 22.04.5 LTS,
  Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62 48-Core Processor (single socket, 48 logical CPU).
  gpuxtb CPU was pinned to 16 workers (`--cpu-threads 16`); xTB and tblite
  adapters enforce their documented single-thread contract; dxtb used 16
  PyTorch threads.
- GPU: NVIDIA GeForce RTX 5090 (GB202), `sm_120`, 32 GiB, driver 580.95.05 /
  CUDA 13.0 driver, toolkit CUDA 12.9.1 (`/group/software/cuda-12.9.1`).
- gpuxtb: shared Release builds of the merged-main branch (CPU
  `build/bench-cpu-shared/libgpuxtb.so.0.1.0`, CUDA
  `build/bench-cuda-shared/libgpuxtb.so.0.1.0`) with
  `-DGPUXTB_ENABLE_CUDA=ON/OFF`,
  `-DGPUXTB_MKL_RT_LIBRARY=/group/software/deepmd-kit-3.1.1/lib/libmkl_rt.so`,
  CUDA `sm_120`.
- xTB: 6.7.1 `libxtb.so.6.7.1` from a pinned reference environment.
- tblite: 0.7.0 `libtblite.so.0.7.0` built with GCC/GFortran 13.4.0.
- dxtb: 0.4.0 via PyTorch 2.13.0+cu130 (CPU and CUDA) in a dedicated uv
  environment.
- Execution used `srun` for both CPU and GPU allocations
  (`srun -n 1 -c 16 ...` and `srun -n 1 --gres=gpu:1 -c 16 ...`).

The CPU artifacts record repository commit `5eaee50` and the CUDA artifacts
record `247a23f`; those commits are equivalent to current branch head
`9298bab` (the intervening squash changed only commit history, and the CPU
path is untouched by the CUDA-only fixes), so the numeric tree is identical.

## Protocol

Every row measures end-to-end GFN2-xTB **energy + analytic forces** through
public interfaces only, with persistent contexts and caller-owned buffers:

- gpuxtb calls the public C ABI through the committed `gpuxtb_public_api`
  ctypes adapter, with SCC pinned to the conformance settings (500 iterations,
  charge tolerance 1e-10, energy tolerance 1e-12, 300 K electronic
  temperature). CUDA rows use host-pointed descriptors and end with an
  explicit `cudaDeviceSynchronize`; correctness downloads happen after timing.
- xTB 6.7.1 and tblite 0.7.0 run their persistent public C API loops with
  their default accuracy 1e-4 and up to 500 SCC iterations. dxtb 0.4.0 runs a
  persistent PyTorch `Calculator` with `Calculator.reset()` inside each
  measured call.
- Each batch of size > 1 is built from **distinct** seeded thermal-like
  conformers of one alkane (identical atomic numbers, slightly different
  coordinates) so no engine can reuse one geometry.
- Batch = 1 uses molecule sizes 14, 32, 62, 122, 242, 362, 602 atoms for
  gpuxtb; xTB/tblite stop at 362 because xTB 6.7.1 segfaults on the 602-atom
  alkane. Batch = 128 uses sizes 14, 32, 62, 122 for every engine, plus
  gpuxtb-only extensions to 152, 242, 302 (CPU) and 242 (CUDA). The CUDA
  row at 302 atoms x 128 systems exceeds the 32 GiB card and is recorded as
  unavailable in the first run; the clean CUDA matrix therefore caps batch=128
  at 242 atoms (30,976 atoms per call).
- Each row uses 1 warmup and 5 measured samples; the reported value is the
  median. Energies across all engines agree to within ~1e-12 Hartree for
  every shared gpuxtb cell (CPU vs CUDA); all SCC runs converged and all
  published forces were finite.
- The trajectory job streams 12 nearly identical frames of each alkane
  (seeded random walk, sigma 0.01 bohr) at every requested molecule size
  (32, 62, 122, 242, and 362 for CPU-only gpuxtb); gpuxtb seeds one FRESH
  solve and then uses strict `WARM` continuation, while the reference engines
  run their persistent per-frame path. Per-frame latency is reported per
  (engine, natoms), so the figure's trajectory panel sweeps molecule size.

The figure intentionally annotates that gpuxtb computes with strictly tighter
convergence than the references, so its timings include more SCC work.

## Results summary

Median energy+force latency (ms), merged epsilon = conformance-tight gpuxtb
SCC (charge tol 1e-10 / energy tol 1e-12), refs default acc 1e-4.

| Engine | batch=1 @62 atm | batch=128 @62 atm | batch=128 @122 atm | traj @62 atm (ms/frame) |
| --- | ---: | ---: | ---: | ---: |
| gpuxtb-cpu | 75.65 | 627.19 | 3011.41 | 70.54 |
| gpuxtb-cuda | 191.69 | 494.61 | 1798.42 | 178.87 |
| xtb | 54.03 | 6955.45 | 20758.49 | 82.81 |
| tblite | 27.16 | 3512.76 | 10054.33 | 53.92 |
| dxtb-cpu | 342.42 | 15480.15 | — | 342.42* |
| dxtb-cuda | 438.86 | 6694.77 | — | — |

\* dxtb trajectory was measured at 122 atoms (755.67 ms/frame) and 62 atoms
(342.42 ms/frame) on different ranges; see the JSON for exact coordinates.

At batch = 128 gpuxtb CPU is ~11x faster than xTB, ~5.6x faster than tblite,
and ~25x faster than dxtb CPU at 62 atoms, and ~6.9x / ~3.3x at 122 atoms.
Unlike the pre-merge evidence, gpuxtb CUDA is now the fastest measured engine
at batch = 128 (faster than gpuxtb CPU from 62 atoms up), and its batch = 1
single-molecule path is ~11x faster at 62 atoms than the pre-merge build
(191.69 ms vs 2129.96 ms), reflecting the merged CUDA SCC/force perf work.

Extended gpuxtb ranges: batch = 1 reaches 602 atoms (gpuxtb-cpu 20131.72 ms,
gpuxtb-cuda 44349.84 ms); batch = 128 reaches 302 atoms on CPU (28109.11 ms)
and 242 atoms on CUDA (8779.61 ms, 32 GiB limited).

See `build/benchmarks/final/natoms_cross_engine.{png,svg}` (also
`docs/assets/`) for the figure.

## Files

- `cpu-gpuxtb-xtb-tblite.json` / `.csv`: CPU gpuxtb, xTB, tblite matrix and
  trajectory rows (system python3).
- `cpu-gpuxtb-b128-large.json` / `.csv`: gpuxtb-CPU-only extended rows
  (batch = 1 at 602 atoms, batch = 128 at 152/242/302 atoms, trajectory at
  362 atoms).
- `cpu-dxtb.json` / `.csv`: dxtb CPU matrix and trajectory rows (dxtb uv
  environment).
- `cuda-gpuxtb.json` / `.csv`: gpuxtb CUDA matrix and trajectory rows.
- `cuda-dxtb.json` / `.csv`: dxtb CUDA matrix and trajectory rows.
- `natoms_cross_engine.png` / `.svg`: rendered three-panel figure.
- `README.md`, `SHA256SUMS`: this document and artifact hashes.

## Reproduction commands

The exact invocations are recorded in each JSON artifact's `metadata.command`.
The workflow used five `srun` invocations (gpuxtb/xTB/tblite, gpuxtb-CPU
extended, and dxtb CPU on the CPU partition; gpuxtb and dxtb CUDA with
`--gres=gpu:1`). dxtb runs used the dedicated uv environment python; the
other engines ran the system python. The figure was then generated with:

```bash
python3 benchmarks/plot_natoms_cross_engine.py \
  --artifact build/benchmarks/final/cpu-gpuxtb-xtb-tblite.json \
  --artifact build/benchmarks/final/cpu-gpuxtb-b128-large.json \
  --artifact build/benchmarks/final/cpu-dxtb.json \
  --artifact build/benchmarks/final/cuda-gpuxtb.json \
  --artifact build/benchmarks/final/cuda-dxtb.json \
  --commit "$(git rev-parse HEAD)" --output docs/assets/natoms_cross_engine.png
```

## CUDA regression found and fixed during this re-run

The merged main (through the issue-#84 sparse pair-list coordination VJP,
commit `71c0e32`) rejected every CUDA energy+force call for molecules with
more than 40 atoms. Two defects were fixed on this branch:

1. The energy/force binding-construction smoke ran the committed sparse
   consumer before the first numerical refresh committed the pair list; the
   stale-geometry preflight then rejected the not-yet-committed consumer.
2. The strict bitwise sparse-vs-dense coordination-gradient parity gate
   rejected every >40-atom result because cross-translation-unit FMA
   contraction made the sparse recomputed pair values differ by 1 ulp.
   `gfn2_pairlist.cu` and `gfn2_geometry.cu` are now compiled with
   `-fmad=false` so the two paths agree bit-for-bit.

See commit `9298bab` ("fix: restore CUDA force execution for molecules over
40 atoms") and issue #84 for details. The CPU path is unaffected.

## Known limitations

- Single-molecule (batch = 1) latency at 242+ atoms grows steeply for both
  gpuxtb backends; the advantages remain ragged batch throughput, CUDA batch
  scaling, trajectory reuse, and the strict convergence guarantee.
- xTB 6.7.1 segfaults on the 602-atom alkane, so batch = 1 reference points
  stop at 362 atoms.
- gpuxtb CUDA batch = 128 is memory-limited on the 32 GiB RTX 5090: 242 atoms
  x 128 systems (30,976 atoms) works; 302 x 128 does not allocate.
- dxtb CUDA circles per-system padding in batch mode; its published numbers
  are the measured persistent path, not an optimized graph replay.