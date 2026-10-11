# Issue #467: post-#466 RTX 5090 xTBloom CUDA refresh

This bundle refreshes only xTBloom CUDA. CPU, xTB, tblite, and dxtb remain
pinned to 2026-08-09. It supersedes #459; prior bytes remain in Git history.

## Result

Median public-call latency in milliseconds, with one warmup and three measured
samples per coordinate:

| Panel | Atoms | Median (ms) |
| --- | ---: | ---: |
| Batch 1, cold | 14 | 14.034 |
| Batch 1, cold | 32 | 32.468 |
| Batch 1, cold | 62 | 55.299 |
| Batch 1, cold | 122 | 103.080 |
| Batch 1, cold | 242 | 227.732 |
| Batch 1, cold | 362 | 557.470 |
| Batch 128, auto-warm | 14 | 31.232 |
| Batch 128, auto-warm | 32 | 61.549 |
| Batch 128, auto-warm | 62 | 180.613 |
| Batch 128, auto-warm | 122 | 624.969 |
| Batch 512, cold | 14 | 273.371 |
| Batch 512, cold | 32 | 414.806 |
| Batch 512, cold | 62 | 1158.460 |
| Batch 512, cold | 122 | 4064.811 |

All 14 requested coordinates ran on CUDA and passed the panel-matched energy
and force gate. The largest observed differences from tblite were
`4.5711e-7 Eh` in energy and `2.9323e-5 Eh/bohr` in one force component,
within the unchanged `2e-3` limits.

## Identities and protocol

- Clean xTBloom/runner revision:
  `e0a3b0d60a75fbc3efe2fc243a75cafee10f3b68`.
- Library: `libxtbloom.so.0.2.0`, SHA-256
  `406102812d5ef0207b4a69f3869b977d5bc89234b65dbd46b2bb836140831b5f`.
- Build: Release, CUDA 12.9.86, `sm_120`, GCC C++17.
- Host: `node3`, AMD EPYC 7K62, affinity CPUs 0-15.
- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, UUID
  `GPU-8e9c9e1a-e183-258c-0b3a-03a5ddebb2f8`, driver 580.95.05.
- Threads: 16 xTBloom workers; OpenBLAS/MKL one internal thread; MKL LP64
  sequential.
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

## Command

The three panels ran in one exclusive Slurm allocation:

```bash
srun --exclusive -N1 -n1 -c16 --mem=0 --gres=gpu:1 -w node3 \
  taskset -c 0-15 bash -lc '<three runner commands below>'

python3 benchmarks/natoms_cross_engine.py \
  --library build/cuda-issue467/libxtbloom.so.0.2.0 \
  --engines xtbloom-cuda --warmups 1 --repetitions 3 --cpu-threads 16 \
  --energy-atol 2e-3 --force-atol 2e-3 \
  --repeatability-energy-atol 1e-10 --repeatability-force-atol 1e-8 \
  --scc-charge-tolerance 1e-4 --scc-energy-tolerance 1e-6 \
  --scc-max-iterations 500 --allow-historical-reference \
  --natoms <panel-atoms> --batch-sizes <panel-batch> \
  --start-policy <cold-or-auto-warm> --reference-json <panel-reference.json> \
  --output-json <panel-output.json> --output-csv <panel-output.csv>
```

The exact panel substitutions were:

| Panel | `--natoms` | Batch | Start | Reference | Output stem |
| --- | --- | ---: | --- | --- | --- |
| Batch 1 | `14,32,62,122,242,362` | 1 | `cold` | `ref-tblite-cold.json` | `xtbloom-cuda-cold` |
| Batch 128 | `14,32,62,122` | 128 | `auto-warm` | `ref-tblite-b128.json` | `xtbloom-cuda-b128` |
| Batch 512 | `14,32,62,122` | 512 | `cold` | `ref-tblite-b512.json` | `xtbloom-cuda-b512` |

The environment set `OMP_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE`, `MKL_DYNAMIC=FALSE`,
`MKL_INTERFACE_LAYER=LP64`, and `MKL_THREADING_LAYER=SEQUENTIAL`, with CUDA
12.9.1 runtime paths in `PATH` and `LD_LIBRARY_PATH`.

## Retention

The generated compact CSV files are retained here. Reproducible raw JSON was
not tracked because the batch-128 and batch-512 files exceed the per-file
evidence budget:

| Raw artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `xtbloom-cuda-cold.json` | 63,849 | `ee0b37d179342d6d92e381e8bb6f7b7097a7a27f48a5c715887293a8b0de220b` |
| `xtbloom-cuda-b128.json` | 1,067,433 | `8ff55006a50d1fb7b5ddcadbddfac406de74d6a47f95b2536a86787e1917ea48` |
| `xtbloom-cuda-b512.json` | 4,171,128 | `ff17d4ca3f3f1c5f42c0634bdd0b0a30894111800b2ee0039942a51b38d8902f` |

The CSV files retain every requested coordinate, distribution summary,
throughput, availability, and correctness result. The compact
`publication-metadata.json` binds those CSVs to the clean source/runtime,
hardware, protocol, panel, and reference identities. `SHA256SUMS` covers every
tracked artifact in this bundle.
