# Issue #168 CPU WARM benchmark evidence

This directory archives the correctness-qualified benchmark evidence for PR
#169. The measurements were generated from clean source commit
`255833dea27d2870034515c4ede0ee3764a0092d` on 2026-08-06 local time
(2026-08-05 UTC).

## Environment

- CPU: AMD EPYC 7K62 48-Core Processor; process affinity fixed to logical CPU 0.
- OS: Linux 6.8.0-110-generic x86_64, glibc 2.35.
- Build: CMake 4.2.1, Ninja, Release, shared library, CUDA disabled.
- Compiler: GCC 11.4.0 (`/usr/bin/x86_64-linux-gnu-g++-11`).
- gpuxtb library SHA-256: `b93b312c3b1aa9f76ed7a382e9d42d1d1ecf4d31abc240db442e5ac537994d5c`.
- MKL runtime SHA-256: `b2ff0e31d7cd18c91813d8f6500f37665597d89de22649d90687aa6bf7bd2c0f`.
- tblite 0.7.0 library SHA-256: `b7d23807eddf46ee2472e6522ceb1b73611166e7060a72d6bdd31bb4ae00db9c`.
- Threads: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, dynamic threading disabled, MKL LP64 sequential.

The JSON artifacts contain the complete compiler, build-option, dependency,
source-revision, environment, workload, and raw-sample provenance.

## Protocol

All rows request GFN2-xTB energy and analytic force at 300 K for a batch of one
neutral closed-shell alkane. gpuxtb uses the conformance SCC policy: 500 maximum
iterations, charge tolerance `1e-10`, and energy tolerance `1e-12`. Each row has
10 untimed warmups followed by 30 recorded samples. gpuxtb WARM performs one
additional untimed FRESH seed before its warmups. Setup, result inspection, and
serialization are outside the timing interval.

| atoms | gpuxtb FRESH median (ms) | FRESH iterations | gpuxtb WARM median (ms) | WARM iterations | FRESH / WARM | tblite persistent median (ms) | tblite / WARM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 18.9507 | 17 | 6.1486 | 2 | 3.08x | 9.5111 | 1.55x |
| 62 | 73.6680 | 18 | 20.6643 | 2 | 3.56x | 27.0605 | 1.31x |
| 98 | 192.1424 | 18 | 47.6397 | 2 | 4.03x | 57.2663 | 1.20x |
| 122 | 318.5151 | 18 | 72.8627 | 2 | 4.37x | 82.9152 | 1.14x |

Every measured sample was compared with the validated gpuxtb FRESH artifact.
The maximum gpuxtb WARM delta over the complete sweep was
`9.3792e-13` Hartree for energy and `1.2888e-11` Hartree/bohr for force. The
maximum tblite delta was `5.2012e-12` Hartree and `1.2916e-11` Hartree/bohr.
All rows passed their within-engine and cross-engine gates. xTB 6.7.1 is not
included because its legacy analytic gradient fails the unchanged force gate
for this long-chain corpus; no xTB speed claim is made here.

## Commands

The exact argv and environment are also embedded in every JSON file. The
essential commands were:

```bash
cmake --build build/pr169-cpu-public --clean-first --parallel
ctest --test-dir build/pr169-cpu-public --output-on-failure

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MKL_INTERFACE_LAYER=LP64 \
  MKL_THREADING_LAYER=SEQUENTIAL taskset -c 0 \
python3 benchmarks/natoms_scaling.py \
  --engine gpuxtb --library "$PWD/build/pr169-cpu-public/libgpuxtb.so.0.1.0" \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 --start-mode fresh \
  --output-json build/benchmarks/pr169-evidence-255833d/gpuxtb-fresh.json \
  --output-csv build/benchmarks/pr169-evidence-255833d/gpuxtb-fresh.csv

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MKL_INTERFACE_LAYER=LP64 \
  MKL_THREADING_LAYER=SEQUENTIAL taskset -c 0 \
python3 benchmarks/natoms_scaling.py \
  --engine gpuxtb --library "$PWD/build/pr169-cpu-public/libgpuxtb.so.0.1.0" \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 --start-mode warm \
  --energy-reference-json build/benchmarks/pr169-evidence-255833d/gpuxtb-fresh.json \
  --cross-engine-energy-atol 5e-7 --cross-engine-force-atol 5e-6 \
  --output-json build/benchmarks/pr169-evidence-255833d/gpuxtb-warm.json \
  --output-csv build/benchmarks/pr169-evidence-255833d/gpuxtb-warm.csv

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MKL_INTERFACE_LAYER=LP64 \
  MKL_THREADING_LAYER=SEQUENTIAL taskset -c 0 \
python3 benchmarks/natoms_scaling.py \
  --engine tblite --library /tmp/tblite-build/libtblite.so \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 \
  --energy-reference-json build/benchmarks/pr169-evidence-255833d/gpuxtb-fresh.json \
  --cross-engine-energy-atol 5e-7 --cross-engine-force-atol 5e-6 \
  --output-json build/benchmarks/pr169-evidence-255833d/tblite-persistent.json \
  --output-csv build/benchmarks/pr169-evidence-255833d/tblite-persistent.csv
```

The exact clean build passed 35/35 CTest tests. The final benchmark harness
passed 41/41 hardware-free tests across `test_natoms_scaling`, `test_run`, and
`test_dxtb_adapter`; repository-wide prek and `git diff --check` also passed
before evidence archival.
