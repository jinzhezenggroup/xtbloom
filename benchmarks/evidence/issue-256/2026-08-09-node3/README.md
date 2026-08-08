# Issue #256 Stage A batch=1 CPU phase-parallelism evidence

Archives the correctness-qualified batch=1 CPU measurements for the Stage A
resolution of #256 (idle-worker scalar phase parallelism), generated from clean
source commit `7b1c10a6961c6ffd8c59f4c48de145b25354f164` (the
`perf/batch1-idle-worker-phases-256` branch) on 2026-08-09.

## Environment

- Host: `node3`, AMD EPYC 7K62 48-Core Processor, Ubuntu 22.04, kernel
  6.8.0-110-generic, glibc 2.35.
- Build: CMake 4.2.1 with Ninja, Release, shared library, CUDA disabled, GCC
  11.4.0 (`/usr/bin/x86_64-linux-gnu-g++-11`).
- gpuxtb library: `build/batch1-cpu/libgpuxtb.so`, SHA-256
  `a04f8d8f5be5e241e0141c25da2f6af0b92e0b59d3dd4e9e09e2e47cb58a2c35`.
- MKL runtime: `/group/software/deepmd-kit-3.1.1/lib/libmkl_rt.so`, SHA-256
  `221e89c09644d546cdc6505fc1fdecdf6490a4c57f7da6ca3b48a1c96c4860bd`.
- Threads: 16 gpuxtb context workers (`--cpu-threads 16`); no OMP/OpenBLAS/MKL
  thread environment set (the harness asserts these).

## Protocol

All rows request GFN2-xTB energy and analytic force for a batch of one neutral
closed-shell alkane at the conformance SCC policy (500 maximum iterations,
charge tolerance `1e-10`, energy tolerance `1e-12`), with 5 untimed warmups and
10 recorded samples per cell. WARM rows perform one untimed FRESH seed before
warmups. Setup, result inspection, and serialization are outside the timing
interval. Every sample passed the within-engine FRESH drift gate (energy atol
`1e-8`, force atol `1e-6` Hartree/bohr).

The baseline comparator was measured from clean `main` (`fa05133`) with the
identical protocol and the same MKL runtime, using a separate build directory,
so the only difference is the Stage A change set.

## Results (median ms, batch=1, 16 threads)

| natoms | main FRESH | Stage A FRESH | speedup | Stage A iters | Stage A WARM (identical geometry) | WARM iters |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 62 | 76.36 | 69.77 | 1.09x | 18 | 20.46 | 2 |
| 122 | 342.03 | 281.21 | 1.22x | 19 | 67.69 | 2 |
| 242 | 1681.39 | 1285.53 | 1.31x | 17 | 273.74 | 2 |
| 362 | 4688.92 | 3696.87 | 1.27x | 18 | 673.75 | 2 |

For reference, the archived earlier-measured identical-geometry WARM 242-atom
median on clean `main` was 316.10 ms (2 iterations); the Stage A row above is
273.74 ms. Perturbed-geometry WARM trajectories are tolerance-driven and are
measured only by the #231 cross-engine harness, not this one.

Per-iteration phase cost at 242 atoms (SCC iteration, instrumented Release
build, issue #256 phase split): eigensolve 57.0 ms (unchanged, single-thread
BLAS), prepare/potentials+H 17.0 -> 5.4 ms (Hamiltonian assembly 14.3 -> 2.6
ms), Mulliken population 14.4 -> 2.3 ms, energy+mixer ~2 ms. Total per
iteration ~91 ms -> ~68 ms with unchanged 17-iteration convergence.

The threaded path is byte-identical to a single worker: with `--cpu-threads 1`
and `--cpu-threads 16` the FRESH 242-atom energy and all 726 force components
are bit-equal (verified against the same-commit serial run).

## Commands

The exact argv and captured environment are embedded in each JSON artifact. The
essential stage-A commands were:

```bash
cmake --build build/batch1-cpu --parallel
ctest --test-dir build/batch1-cpu --output-on-failure

python3 benchmarks/natoms_scaling.py \
  --library "$PWD/build/batch1-cpu/libgpuxtb.so" \
  --output-json build/benchmarks/issue-256-evidence-7b1c10a/gpuxtb-fresh.json \
  --output-csv  build/benchmarks/issue-256-evidence-7b1c10a/gpuxtb-fresh.csv \
  --start-mode fresh --natoms 62,122,242,362 --batch-sizes 1 \
  --backend cpu --cpu-threads 16 --property force --warmups 5 --repetitions 10

python3 benchmarks/natoms_scaling.py \
  --library "$PWD/build/batch1-cpu/libgpuxtb.so" \
  --output-json build/benchmarks/issue-256-evidence-7b1c10a/gpuxtb-warm.json \
  --output-csv  build/benchmarks/issue-256-evidence-7b1c10a/gpuxtb-warm.csv \
  --start-mode warm --natoms 62,122,242,362 --batch-sizes 1 \
  --backend cpu --cpu-threads 16 --property force --warmups 5 --repetitions 10 \
  --energy-reference-json build/benchmarks/issue-256-evidence-7b1c10a/gpuxtb-fresh.json
```

The full 43-test batch1-cpu CTest set (including the new
`gpuxtb.gfn2.mulliken` parallel bit-identity coverage) passes on this build.
