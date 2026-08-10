# Issue #340 adaptive CPU precision evidence

This directory archives correctness-qualified public-C-ABI latency evidence for
the experimental `XTBLOOM_CPU_PRECISION=adaptive` CPU policy. Measurements were
generated from clean implementation commit
`584d8a0b880993af042ff3c36094086b9474a574` on 2026-08-11 local time.

## Environment

- Scheduler: one exclusive Slurm task on `node3`, requested with 48 CPUs and
  pinned by `taskset` to logical CPUs 0-15.
- CPU: AMD EPYC 7K62 48-Core Processor; xTBloom outer workers: 16.
- Build: CMake 4.2.1, Ninja, Release, shared library, CUDA disabled.
- Compiler: GCC 11.4.0 (`/usr/bin/x86_64-linux-gnu-g++-11`).
- xTBloom library SHA-256:
  `24036c9f0b43ee107d4b244ea37256c8f649a6d7ea1768c3b032596bbbcecf05`.
- CMake discovery entry `/home/jzzeng/miniconda3/lib/libmkl_rt.so.3`
  SHA-256:
  `b2ff0e31d7cd18c91813d8f6500f37665597d89de22649d90687aa6bf7bd2c0f`.
- Actual isolated provider closure resolved from the benchmark build:
  - `build/issue340-mkl/libxtbloom_mkl_lp64_shim.so`:
    `2fa4f18bebfe47d11d9c51d98d573966ecffd7bff294e73ceaf9ae50f43ce6ce`;
  - `/home/jzzeng/miniconda3/lib/libmkl_intel_lp64.so.3`:
    `9dbdb7d4193679f028787da6b9b2b545e0e16b87a95a64ee7318f544da8b08ac`;
  - `/home/jzzeng/miniconda3/lib/libmkl_sequential.so.3`:
    `4e7ff529bec90a1b0ce70299258a64a2ebae7a5a6f3ec031306a6ddb39b5a650`;
  - `/home/jzzeng/miniconda3/lib/libmkl_core.so.3`:
    `b4f086651ee5a8471140d53fedfdbbaba606287784763b0739f282872c31aa9b`.
- BLAS policy: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, dynamic threading disabled, MKL LP64 sequential.

Every JSON artifact records the complete argv, clean repository revision,
library hash, configure-time provider-entry hash, build cache, compiler,
process affinity, thread environment, workload identity, observables,
convergence state, and all 30 raw timing samples. The actual private shim and
fixed MKL component hashes are recorded above. `XTBLOOM_CPU_PRECISION` is
recorded explicitly so FP64 and adaptive evidence cannot be confused.

## Protocol and result

All rows request GFN2-xTB energy plus analytic forces at 300 K using 500 SCC
iterations, charge tolerance `1e-10`, and energy tolerance `1e-12`. Each row
uses 10 untimed warmups followed by 30 measured synchronous public calls. WARM
rows first perform one untimed compatible FRESH seed. Correctness gates are
`1e-8 Eh` for energy and `5e-7 Eh/bohr` for force.

| Workload | FP64 median (ms) | Adaptive median (ms) | Adaptive change | FP64 -> adaptive SCC iterations |
| --- | ---: | ---: | ---: | ---: |
| 122 atoms, B1, FRESH | 276.5656 | 265.6369 | -3.95% | 18 -> 18 |
| 242 atoms, B1, FRESH | 1306.4202 | 1246.4361 | -4.59% | 17 -> 17 |
| 62 atoms/system, B32, FRESH | 156.6672 | 146.6527 | -6.39% | 18 -> 17 |
| 122 atoms, B1, WARM | 68.9184 | 68.8420 | -0.11% | 2 -> 2 |
| 242 atoms, B1, WARM | 277.3540 | 276.6604 | -0.25% | 2 -> 2 |
| 62 atoms/system, B32, WARM | 45.3587 | 45.3945 | +0.08% | 2 -> 2 |

All eight JSON artifacts and every measured sample passed their declared
correctness gates. Across direct FP64/adaptive sample comparisons, the maximum
absolute difference was `7.9581e-13 Eh` for energy and
`3.4284e-11 Eh/bohr` for force. WARM medians differ by at most 0.25%, and all
WARM rows retain two SCC iterations, consistent with the implementation's
FP64-only WARM policy.

The narrow conclusion is that this policy produces a reproducible 3.95-4.59%
FRESH latency reduction for the two tested single-system long alkanes without
changing their SCC iteration count. The B32 row is 6.39% faster, but its SCC
count also falls from 18 to 17, so that gain cannot be attributed solely to
faster FP32 linear algebra. These homogeneous alkane rows do not establish a
release-wide speedup or close issue #340's remaining injected-failure and
per-lane switching evidence.

## Command boundary

The complete matrix ran inside one exclusive allocation:

```bash
srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=48 \
  --mem=0 --exclusive --time=00:20:00 bash -lc '<matrix below>'
```

Each matrix entry used this command shape in a separate Python process:

```bash
env OMP_DYNAMIC=FALSE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_DYNAMIC=FALSE MKL_INTERFACE_LAYER=LP64 MKL_NUM_THREADS=1 \
  MKL_THREADING_LAYER=SEQUENTIAL XTBLOOM_CPU_PRECISION=<fp64|adaptive> \
taskset -c 0-15 python3 benchmarks/natoms_scaling.py \
  --engine xtbloom \
  --library "$PWD/build/issue340-mkl/libxtbloom.so.0.0.0" \
  --backend cpu --cpu-threads 16 --property force \
  --natoms <122,242|62> --batch-sizes <1|32> \
  --warmups 10 --repetitions 30 --start-mode <fresh|warm> \
  --energy-atol 1e-8 --force-atol 5e-7 \
  <WARM: --energy-reference-json matching-fresh.json \
          --cross-engine-energy-atol 1e-8 \
          --cross-engine-force-atol 5e-7> \
  --output-json <artifact>.json --output-csv <artifact>.csv
```

The eight combinations are FP64/adaptive crossed with FRESH/WARM for
`122,242` atoms at batch 1 and 62 atoms at batch 32. The authoritative exact
argv for each combination is embedded in its JSON artifact.

## Qualification and limitations

Before measurement, the implementation commit passed the shared CPU Release
CTest suite twice: 54/54 with the default FP64 environment and 54/54 with
`XTBLOOM_CPU_PRECISION=adaptive`. The installed Python suite passed 309 tests
with 43 expected CUDA/optional-runtime skips. The final benchmark self-test
command ran 92 tests with 91 passes and one intentional optional plotting
skip. Real RTX 5090 checks also confirmed that CUDA ignores the CPU-only
variable and retained public host/device/mixed conformance.

Full `prek --all-files` could not initialize its ruff hook because fetching
the hook repository from GitHub stalled. Equivalent checks passed with ruff
0.16.2, clang-format 22.1.8, ty 0.0.69, direct v6.0.0 large-file/JSON checks,
an exact retired-name exclusion audit, and `git diff --check`.
`uv lock --check` passed against PyPI; this change does not modify dependency
metadata or `uv.lock`.
