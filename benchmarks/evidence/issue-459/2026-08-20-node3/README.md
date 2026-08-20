# Issue #459: RTX 5090 xTBloom CUDA refresh

This bundle refreshes only the xTBloom CUDA series in the public cross-engine
figure. CPU, xTB, tblite, and dxtb rows remain pinned to the 2026-08-09
evidence because neither those implementations nor their measurement contract
changed.

## Result

Median public-call latency in milliseconds, with one warmup and three measured
samples per coordinate:

| Panel | Atoms | Median (ms) |
| --- | ---: | ---: |
| Batch 1, cold | 14 | 13.826 |
| Batch 1, cold | 32 | 32.922 |
| Batch 1, cold | 62 | 56.907 |
| Batch 1, cold | 122 | 109.729 |
| Batch 1, cold | 242 | 252.979 |
| Batch 1, cold | 362 | 620.349 |
| Batch 128, auto-warm | 14 | 31.125 |
| Batch 128, auto-warm | 32 | 61.433 |
| Batch 128, auto-warm | 62 | 181.182 |
| Batch 128, auto-warm | 122 | 628.971 |
| Batch 512, cold | 14 | 272.537 |
| Batch 512, cold | 32 | 413.656 |
| Batch 512, cold | 62 | 1158.800 |
| Batch 512, cold | 122 | 4071.670 |

All 14 requested coordinates ran on CUDA and passed the panel-matched energy
and force gate. The largest observed differences from tblite were
`4.5711e-7 Eh` in energy and `2.9323e-5 Eh/bohr` in one force component,
within the unchanged `2e-3` limits.

## Identities and protocol

- Clean xTBloom/runner revision:
  `af65028c5dddc1a3aad3ffc554f0b7dba121a2fe`.
- Library: `libxtbloom.so.0.2.0`, SHA-256
  `6a0c81f844bda7814e450cbc716e5cc9f773377b6e79b4d3efd8c37ef1511b95`.
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
  --library build/cuda-issue459-9299c9b/libxtbloom.so.0.2.0 \
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
| `xtbloom-cuda-cold.json` | 64,424 | `cd916d51f6365c0fe6dd1b452ff0d2285e9d8075aa640f98d858d49235dc070d` |
| `xtbloom-cuda-b128.json` | 1,067,796 | `b247e749a00f04a372ea88df620d47103e5701f11f891e26dfd05cc215e97cb2` |
| `xtbloom-cuda-b512.json` | 4,171,465 | `572fbed7da26c6fd4facb3bb50b6375d1e7c2c68ec158fa53f66765574538a9d` |

The CSV files retain every requested coordinate, distribution summary,
throughput, availability, and correctness result. The compact
`publication-metadata.json` binds those CSVs to the clean source/runtime,
hardware, protocol, panel, and reference identities. `SHA256SUMS` covers every
tracked artifact in this bundle.
