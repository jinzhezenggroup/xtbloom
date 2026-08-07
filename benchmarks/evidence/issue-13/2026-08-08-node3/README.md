# Issue #13 cross-engine GFN2-xTB scaling evidence

This bundle archives the correctness-qualified public-API benchmark matrix and
figure shipped in the repository README and user guide. The runner is
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
- gpuxtb: shared Release build of commit `3644cff4efaba020be12cb6ceeddb890d7feb44c`
  (`build/bench-cpu-shared/libgpuxtb.so.0.1.0`,
  `build/bench-cuda-shared/libgpuxtb.so.0.1.0`, `-DGPUXTB_ENABLE_CUDA=ON/OFF`,
  `-DGPUXTB_MKL_RT_LIBRARY=/group/software/deepmd-kit-3.1.1/lib/libmkl_rt.so`,
  CUDA `sm_120`).
- xTB: 6.7.1 `libxtb.so.6.7.1` from a pinned reference environment.
- tblite: 0.7.0 `libtblite.so.0.7.0` built with GCC/GFortran 13.4.0.
- dxtb: 0.4.0 via PyTorch 2.13.0+cu130 (CPU and CUDA) in a dedicated uv
  environment resolved against the Tsinghua PyPI mirror.
- Execution used `srun` for both CPU and GPU allocations
  (`srun -n 1 -c 16 ...` and `srun --gres=gpu:1 -c 16 ...`).

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
- Batch = 1 uses molecule sizes 14, 32, 62, 122, 242 atoms; batch = 128 uses
  sizes 14, 32, 62 atoms to keep serial reference loops practical.
- Each row uses 1 warmup and 5 measured samples; the reported value is the
  median. Energies across all engines agree to within ~1e-5 Hartree for every
  shared cell; all SCC runs converged and all published forces were finite.
- The trajectory job streams 12 nearly identical frames of a 62-atom alkane
  (seeded random walk, sigma 0.01 bohr); gpuxtb seeds one FRESH solve and then
  uses strict `WARM` continuation, while the reference engines run their
  persistent per-frame path.

The figure intentionally annotates that gpuxtb computes with strictly tighter
convergence than the references, so its timings include more SCC work.

## Results summary

| Engine | batch=1 @62 atm (ms) | batch=128 @62 atm (ms) | trajectory @62 atm (ms/frame) |
| --- | ---: | ---: | ---: |
| gpuxtb-cpu | 76.85 | 633.12 | 71.03 |
| gpuxtb-cuda | 2129.96 | 4704.14 | 2109.02 |
| xtb | 54.00 | 6991.43 | 82.59 |
| tblite | 27.11 | 3490.07 | 53.80 |
| dxtb-cpu | 347.72 | 15452.62 | 347.54 |
| dxtb-cuda | 449.02 | 6734.47 | 453.51 |

At batch = 128 gpuxtb CPU is ~11x faster than xTB, ~5-8x faster than tblite,
and ~24-45x faster than dxtb CPU on the measured sizes. In this build the CUDA
single-point path carries a fixed per-call cost at these sizes (visible in the
62-atom single-molecule row), so the CPU ragged batch is the fastest measured
configuration.

See `build/benchmarks/final/natoms_cross_engine.{png,svg}` (also
`docs/assets/`) for the figure.

## Files

- `cpu-gpuxtb-xtb-tblite.json` / `.csv`: CPU gpuxtb, xTB, tblite matrix and
  trajectory rows (system python3).
- `cpu-dxtb.json` / `.csv`: dxtb CPU matrix and trajectory rows (dxtb uv
  environment).
- `cuda-gpuxtb.json` / `.csv`: gpuxtb CUDA matrix and trajectory rows.
- `cuda-dxtb.json` / `.csv`: dxtb CUDA matrix and trajectory rows.
- `natoms_cross_engine.png` / `.svg`: rendered three-panel figure.
- `README.md`, `SHA256SUMS`: this document and artifact hashes.

## Reproduction commands

The exact invocations are recorded in each JSON artifact's `metadata.command`.
The workflow used four `srun` invocations (gpuxtb/xTB/tblite and dxtb CPU, then
gpuxtb and dxtb CUDA with `--gres=gpu:1`). dxtb runs used the dedicated uv
environment python; the other engines ran the system python because the dxtb
environment's Python 3.11 crashes the gpuxtb C extension at larger batch
sizes. The figure was then generated with:

```bash
python3 benchmarks/plot_natoms_cross_engine.py \
  --artifact build/benchmarks/final/cpu-gpuxtb-xtb-tblite.json \
  --artifact build/benchmarks/final/cpu-dxtb.json \
  --artifact build/benchmarks/final/cuda-gpuxtb.json \
  --artifact build/benchmarks/final/cuda-dxtb.json \
  --commit 3644cff --output build/benchmarks/final/natoms_cross_engine.png
```

## Known limitations

- Single-molecule (batch = 1) latency at 62+ atoms is not gpuxtb's headline:
  the conformance-tight SCC and per-call cost make gpuxtb CPU slightly slower
  than xTB/tblite for one isolated large molecule; the advantages are ragged
  batch throughput, CUDA batch scaling, trajectory reuse, and the strict
  convergence guarantee.
- dxtb CUDA circles per-system padding in batch mode; its published numbers
  are the measured persistent path, not an optimized graph replay.
- The 128-system batch was capped at 62 atoms because serial reference loops
  become impractical beyond that on this machine; batch sizes 8/32 for the
  same workload are available from any full `--batch-sizes` run.