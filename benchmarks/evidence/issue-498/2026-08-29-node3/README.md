# Issue #498: post-#494 xTBloom CPU publication refresh

This bundle refreshes only xTBloom CPU after issue #494 and PR #497. xTBloom
CUDA remains pinned to 2026-08-20, and xTB, tblite, and dxtb remain pinned to
2026-08-09. It supersedes the xTBloom CPU portion of issue #13; prior bytes
remain recoverable from Git history and the legacy artifact ledger.

## Result

Median public-call latency in milliseconds, with one warmup and three measured
samples per coordinate:

| Panel | Atoms | Median (ms) |
| --- | ---: | ---: |
| Batch 1, cold | 14 | 2.768 |
| Batch 1, cold | 32 | 11.834 |
| Batch 1, cold | 62 | 33.995 |
| Batch 1, cold | 122 | 129.641 |
| Batch 1, cold | 242 | 595.412 |
| Batch 1, cold | 362 | 1584.740 |
| Batch 128, auto-warm | 14 | 17.786 |
| Batch 128, auto-warm | 32 | 53.208 |
| Batch 128, auto-warm | 62 | 166.642 |
| Batch 128, auto-warm | 122 | 635.138 |
| Batch 512, cold | 14 | 64.710 |
| Batch 512, cold | 32 | 315.378 |
| Batch 512, cold | 62 | 1161.619 |
| Batch 512, cold | 122 | 5119.103 |

All 14 requested coordinates ran on CPU and passed the panel-matched energy
and force gate. The largest observed differences from tblite were
`4.5711e-7 Eh` in energy and `2.9323e-5 Eh/bohr` in one force component,
within the unchanged `2e-3` limits.

At 62 atoms, the refreshed xTBloom CPU medians are 9.3x faster than xTB and
8.3x faster than tblite for batch 128, and 9.9x / 11.8x faster for batch 512.

## Identities and protocol

- Clean xTBloom/runner revision:
  `ab3a3bc6132c2ff31e9723463924a24e355ad5d5`.
- Library: `libxtbloom.so.0.2.1`, SHA-256
  `05009153e15b45353232d64b85d4a3bd25055fab49b352ab28ed72675fb51820`.
- CMake cache SHA-256:
  `5e578b1b3009fe862e1f56101ffecf9aec463ac3b74c19568a6dd62a316a1610`.
- Build: shared Release, GCC 11.4.0, C++17, CUDA disabled, CPU ISA request
  `auto` (resolved AVX2/FMA on this host).
- LP64 provider: `/group/software/deepmd-kit-3.1.1/lib/libmkl_rt.so.2`,
  SHA-256
  `221e89c09644d546cdc6505fc1fdecdf6490a4c57f7da6ca3b48a1c96c4860bd`.
- Host: `node3`, AMD EPYC 7K62, affinity CPUs 0-15.
- Threads: 16 xTBloom workers; OpenBLAS/MKL one internal thread; MKL LP64
  sequential; dynamic OpenMP and MKL threading disabled.
- Outputs: energy and analytic forces through host descriptors.
- SCC: charge `1e-4`, energy `1e-6`, maximum 500 iterations.

The clean historical tblite references use revision
`c9c0a432947f122d25cb91d0a4624af0a3e761ad`. The runner explicitly allowed
the older revision only after validating the unchanged workload, start policy,
timing controls, SCC contract, correctness thresholds, and thread budget:

| Panel | Reference SHA-256 |
| --- | --- |
| Batch 1 | `9b9626f0481b8b72c12c5f5bfc95ed3a3d49e6d31301d403edb0dd1256e852f5` |
| Batch 128 | `b6997834c64a6d184c73a5b581894f2ed4f3a48e32ae736e09f274476f5dde4f` |
| Batch 512 | `1201442fa630ce40bcd93f3d788c1348bce61f4b140215a6aa6001e3efd17af4` |

## Commands

The exact build and focused pre-timing validation were:

```bash
cmake -S . -B build/cpu-issue498 -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/group/software/deepmd-kit-3.1.1/lib/libmkl_rt.so \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu-issue498 --parallel
ctest --test-dir build/cpu-issue498 \
  -R 'xtbloom\.(cpu\.public_inference|conformance\.(public_cpu|invariants_cpu)|gfn2\.(multipole_integrals|force))$' \
  --output-on-failure
```

The focused validation passed 5/5 tests. Each panel then used the following
runner shape, with the substitutions listed below:

```bash
env OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MKL_INTERFACE_LAYER=LP64 \
  MKL_THREADING_LAYER=SEQUENTIAL XTBLOOM_CPU_ISA=auto \
  taskset -c 0-15 .venv/bin/python3 benchmarks/natoms_cross_engine.py \
  --library build/cpu-issue498/libxtbloom.so.0.2.1 \
  --engines xtbloom-cpu --warmups 1 --repetitions 3 --cpu-threads 16 \
  --energy-atol 2e-3 --force-atol 2e-3 \
  --repeatability-energy-atol 1e-10 --repeatability-force-atol 1e-8 \
  --scc-charge-tolerance 1e-4 --scc-energy-tolerance 1e-6 \
  --scc-max-iterations 500 --allow-historical-reference \
  --natoms <panel-atoms> --batch-sizes <panel-batch> \
  --start-policy <cold-or-auto-warm> --reference-json <panel-reference.json> \
  --output-json <panel-output.json> --output-csv <panel-output.csv>
```

| Panel | `--natoms` | Batch | Start | Reference | Output stem |
| --- | --- | ---: | --- | --- | --- |
| Batch 1 | `14,32,62,122,242,362` | 1 | `cold` | `ref-tblite-cold.json` | `xtbloom-cpu-cold` |
| Batch 128 | `14,32,62,122` | 128 | `auto-warm` | `ref-tblite-b128.json` | `xtbloom-cpu-b128` |
| Batch 512 | `14,32,62,122` | 512 | `cold` | `ref-tblite-b512.json` | `xtbloom-cpu-b512` |

## Retention

The generated compact CSV files are retained here. Reproducible raw JSON was
not tracked because the batch-128 and batch-512 files exceed the per-file
evidence budget and the compact publication bundle is sufficient to reproduce
the selected rows:

| Raw artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `xtbloom-cpu-cold.json` | 61,799 | `5e91b51efc89de9667d5335985bf7604f8e0ba21ab3092c2ec5211d7e921d76d` |
| `xtbloom-cpu-b128.json` | 1,067,330 | `ef2522109453acd156128b62263dcfeeefd6d924153039a9d2b4ec331e168f83` |
| `xtbloom-cpu-b512.json` | 4,177,175 | `43d8caec5cbdb428c9f65e6ddfa325856b6e130cedc0670113252431b69191e5` |

The CSV files retain every requested coordinate, distribution summary,
throughput, availability, and correctness result. The compact
`publication-metadata.json` binds those CSVs to the clean source/runtime,
hardware, protocol, panel, and reference identities. `SHA256SUMS` covers every
tracked artifact in this bundle.
