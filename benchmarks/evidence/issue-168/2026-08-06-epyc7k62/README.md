# Issue #168 CPU WARM benchmark evidence

This directory archives the correctness-qualified benchmark evidence for PR
#169. The measurements were generated from clean source commit
`6f7f1e29b393cc5c05b63088d41f965ccf7d9aea` on 2026-08-06 local time
(2026-08-05 UTC).

## Environment

- CPU: AMD EPYC 7K62 48-Core Processor; process affinity fixed to logical CPU 0.
- OS: Linux 6.8.0-110-generic x86_64, glibc 2.35.
- Build: CMake 4.2.1, Ninja, Release, shared library, CUDA disabled.
- Compiler: GCC 11.4.0 (`/usr/bin/x86_64-linux-gnu-g++-11`).
- xTBloom library SHA-256: `b93b312c3b1aa9f76ed7a382e9d42d1d1ecf4d31abc240db442e5ac537994d5c`.
- MKL runtime SHA-256: `b2ff0e31d7cd18c91813d8f6500f37665597d89de22649d90687aa6bf7bd2c0f`.
- tblite 0.7.0 library SHA-256: `1c2fb4308b398851580af11ccad5eec22314ff45a34a553380277e776a44c3b5`.
- tblite C compiler: GCC 13.4.0 at
  `/tmp/lammps-qmmm-xtb-env/bin/x86_64-conda-linux-gnu-cc`, SHA-256
  `6420cb0972435d925a3248db3ccb0c1c4ff7f1a81e6aa8f37e1ae54da1538d98`.
- tblite Fortran compiler: GFortran 13.4.0 at
  `/tmp/lammps-qmmm-xtb-env/bin/x86_64-conda-linux-gnu-gfortran`, SHA-256
  `de744570ad7763b884735bc75503fc79520f01eb8859d5b010ec377a2d4d8174`.
- tblite OpenBLAS 0.3.33 provider at
  `/tmp/lammps-qmmm-xtb-env/lib/libopenblasp-r0.3.33.so`, SHA-256
  `60ddaebfbdae101e9325efa24360c3bd639abb77ef569b35becd6bbcbed6605e`.
- Threads: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, dynamic threading disabled, MKL LP64 sequential.

The complete JSON artifacts contain compiler, build-option, captured
dependency, source-revision, environment, workload, and raw-sample provenance.
Issue #348 removed the three over-1-MiB JSON files from the current tree while
retaining their compact CSV views. Their exact historical path, byte count,
SHA-256, and retrieval revision are recorded in
`benchmarks/evidence/legacy-large-artifacts.tsv`. Meson was configured with the
absolute compiler paths above, so each compiler identity is content-hashed
without consulting the later benchmark process's `PATH`.

## Protocol

All rows request GFN2-xTB energy and analytic force at 300 K for a batch of one
neutral closed-shell alkane. xTBloom uses the conformance SCC policy: 500 maximum
iterations, charge tolerance `1e-10`, and energy tolerance `1e-12`. Each row has
10 untimed warmups followed by 30 recorded samples. xTBloom WARM performs one
additional untimed FRESH seed before its warmups. Setup, result inspection, and
serialization are outside the timing interval.

| atoms | xTBloom FRESH median (ms) | FRESH iterations | xTBloom WARM median (ms) | WARM iterations | FRESH / WARM | tblite persistent median (ms) | tblite / WARM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 19.0047 | 17 | 6.1541 | 2 | 3.09x | 9.4993 | 1.54x |
| 62 | 74.1555 | 18 | 20.6437 | 2 | 3.59x | 27.1112 | 1.31x |
| 98 | 195.7811 | 18 | 47.4718 | 2 | 4.12x | 54.7487 | 1.15x |
| 122 | 344.6569 | 18 | 72.4041 | 2 | 4.76x | 78.9223 | 1.09x |

Every measured sample was compared with the validated xTBloom FRESH artifact.
The maximum xTBloom WARM delta over the complete sweep was
`9.3792e-13` Hartree for energy and `1.2888e-11` Hartree/bohr for force. The
maximum tblite delta was `5.2012e-12` Hartree and `1.2916e-11` Hartree/bohr.
All rows passed their within-engine and cross-engine gates. xTB 6.7.1 is not
included because its legacy analytic gradient fails the unchanged force gate
for this long-chain corpus; no xTB speed claim is made here.

## Commands

The exact argv and environment are embedded in every complete historical JSON
file. The essential commands are also retained below:

```bash
env PATH=/tmp/lammps-qmmm-xtb-env/bin:/usr/bin:/bin \
  CC=/tmp/lammps-qmmm-xtb-env/bin/x86_64-conda-linux-gnu-cc \
  FC=/tmp/lammps-qmmm-xtb-env/bin/x86_64-conda-linux-gnu-gfortran \
  PKG_CONFIG_PATH=/tmp/lammps-qmmm-xtb-env/lib/pkgconfig \
  LD_LIBRARY_PATH=/tmp/lammps-qmmm-xtb-env/lib \
  /tmp/lammps-qmmm-xtb-env/bin/meson setup \
  /tmp/tblite-pr169-6f7f1e2 /home/jzzeng/codes/tblite \
  --buildtype=release -Dopenmp=true -Dlapack=openblas -Dapi=true \
  -Dddx=false -Dpython=false --libdir=lib

env PATH=/tmp/lammps-qmmm-xtb-env/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/tmp/lammps-qmmm-xtb-env/lib \
  /tmp/lammps-qmmm-xtb-env/bin/meson compile \
  -C /tmp/tblite-pr169-6f7f1e2

cmake --build build/pr169-cpu-public --clean-first --parallel
ctest --test-dir build/pr169-cpu-public --output-on-failure

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MKL_INTERFACE_LAYER=LP64 \
  MKL_THREADING_LAYER=SEQUENTIAL taskset -c 0 \
python3 benchmarks/natoms_scaling.py \
  --engine xtbloom --library "$PWD/build/pr169-cpu-public/libxtbloom.so.0.1.0" \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 --start-mode fresh \
  --output-json build/benchmarks/pr169-evidence-6f7f1e2/xtbloom-fresh.json \
  --output-csv build/benchmarks/pr169-evidence-6f7f1e2/xtbloom-fresh.csv

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MKL_INTERFACE_LAYER=LP64 \
  MKL_THREADING_LAYER=SEQUENTIAL taskset -c 0 \
python3 benchmarks/natoms_scaling.py \
  --engine xtbloom --library "$PWD/build/pr169-cpu-public/libxtbloom.so.0.1.0" \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 --start-mode warm \
  --energy-reference-json build/benchmarks/pr169-evidence-6f7f1e2/xtbloom-fresh.json \
  --cross-engine-energy-atol 5e-7 --cross-engine-force-atol 5e-6 \
  --output-json build/benchmarks/pr169-evidence-6f7f1e2/xtbloom-warm.json \
  --output-csv build/benchmarks/pr169-evidence-6f7f1e2/xtbloom-warm.csv

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MKL_INTERFACE_LAYER=LP64 \
  MKL_THREADING_LAYER=SEQUENTIAL \
  LD_LIBRARY_PATH=/tmp/lammps-qmmm-xtb-env/lib taskset -c 0 \
python3 benchmarks/natoms_scaling.py \
  --engine tblite \
  --library /tmp/tblite-pr169-6f7f1e2/libtblite.so.0.7.0 \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 \
  --energy-reference-json build/benchmarks/pr169-evidence-6f7f1e2/xtbloom-fresh.json \
  --cross-engine-energy-atol 5e-7 --cross-engine-force-atol 5e-6 \
  --output-json build/benchmarks/pr169-evidence-6f7f1e2/tblite-persistent.json \
  --output-csv build/benchmarks/pr169-evidence-6f7f1e2/tblite-persistent.csv
```

The exact clean build passed 35/35 CTest tests. The final benchmark harness
passed 43/43 hardware-free tests across `test_natoms_scaling`, `test_run`, and
`test_dxtb_adapter`; repository-wide prek and `git diff --check` also passed
before evidence archival.
