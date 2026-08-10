# Issue #184 CUDA occupations benchmark evidence

This directory archives the correctness-qualified benchmark and Nsight
profiling evidence for issue #184 / PR #172 ("perf: parallelize the CUDA
occupations evaluation", squash commit `89a53220655e6fb4037890db487207819b598cc3`).

The parallel occupations implementation is already on `main`. This directory
records the durable before/after artifacts that the issue required: the raw
public-C-API benchmark matrices and derived Nsight Systems CUDA kernel
summaries for batch 1 and batch 128. "Baseline" is the occupations evaluation
before PR #172 (the exact source checkout is `cf0fe8dcf802211892f96436da2f276e49c9199a`).
"Repaired" is the parallel implementation at PR #172 head
`6b71c2a1392497ad8e595b8f88561db300e2b0a6`. Both builds use the same
`benchmarks/run.py` blob (`bb01571dc508621d715d6b62d901bb3745ce8361`); the
occupations change is the only source difference relevant to this comparison.

## Environment

- Host: `node3` (SLURM cluster), OS Ubuntu 22.04.5 LTS, Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62 48-Core Processor.
- GPU: NVIDIA GeForce RTX 5090 (GB202), `sm_120`, 32 GiB, CUDA visible device 0.
- CUDA runtime: `/group/software/cuda-12.9.1` (CUDA 12.9.1, driver 580.95.05 /
  CUDA 13.0 driver API).
- Nsight Systems 2025.1.3:
  `/group/software/cuda-12.9.1/nsight-systems-2025.1.3/target-linux-x64/nsys`.
- Build: Release, shared library, strict `sm_120`, CUDA 12.9.86 nvcc; both
  builds produced `libxtbloom.so.0.1.0`.
- Measured env: `LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/group/software/deepmd-kit-3.1.1/lib`,
  `CUDA_VISIBLE_DEVICES=0`, `MKL_INTERFACE_LAYER=LP64`,
  `MKL_THREADING_LAYER=SEQUENTIAL`.

The raw Nsight reports were inspected locally and the committed kernel-summary
CSVs were regenerated from them with Nsight Systems 2025.1.3. Raw
`.nsys-rep` files are intentionally not committed: Nsight captures the full
target-process environment, which may contain credentials or other private
session metadata. The minimally scoped recapture and export commands are
recorded below.

The end-to-end rows were produced by the historical runner snapshot retained as
`natoms_scaling_pr172.py.txt`. The recovered capture source has SHA-256
`0929f4930aa6538f900c5be96fbd90e9ca76ad10dfce620675064104979ef941`; the
archived copy has SHA-256 `73e725f920503f8fd71e91745e2242cec078e68e957eb064b097779a552be084`
after the required final newline was added. This snapshot is retained because
the current `benchmarks/natoms_scaling.py` emits a newer audit schema and the
current `benchmarks/run.py --workloads gas` workload is the five-atom ketene
case, not C40H82. The archived JSON/CSV schema and values therefore must be
interpreted as the historical PR #172 capture, not as output from today's
benchmark CLI.

## Protocol

All rows run GFN2-xTB analytic forces for a batch of one neutral closed-shell
C40H82 (122 atoms) alkane through xTBloom's public C ABI with host-pointed CUDA
descriptors and fresh SCC execution. The matrix uses one warmup and three
recorded samples per row; every row reports 13 SCC attempts. Setup, result
inspection, and serialization are outside the timing interval.

- End-to-end matrices (B1/8/32/128, historical `natoms_scaling_pr172.py.txt`):
  one warmup plus three recorded samples per row.
- Nsight profiles (the same historical runner, batch 1 or 128 only): one
  untimed run under `nsys profile` with CUDA tracing only; profiling overhead
  makes these single-sample timings slower than the unprofiled matrix rows and
  they are used only for the kernel-level comparison. The captured B1 run used
  one warmup and the B128 run used zero warmups before the profiled invocation.

## Results

### End-to-end public-C-API CUDA force latency (median, ms)

| Batch | Baseline | Repaired | Change |
| ---: | ---: | ---: | ---: |
| 1 | 2274.2901 | 2011.9968 | -11.53% |
| 8 | 3684.5013 | 3410.4332 | -7.44% |
| 32 | 4834.2904 | 4589.3533 | -5.07% |
| 128 | 9542.2905 | 9303.6981 | -2.50% |

Energy is bit-identical between baseline and repaired for every batch
(`-119.30484557472732` Ha for B1/B128 and the corresponding identical values
for B8/B32 in the JSON artifacts). The maximum force drift measured during PR
#172 review was `5.551115123125783e-17` Ha/bohr. SCC iteration counts and
`status_failures` are identical (13 iterations, no failures).

### Occupations kernel (`evaluate_kernel`) totals from the Nsight reports

| Batch | Instances | Baseline (ms) | Repaired (ms) | Speedup | Time share baseline | Time share repaired |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 500 | 608.808 | 72.216 | 8.43x | 13.8% | 1.9% |
| 128 | 250 | 304.875 | 36.451 | 8.36x | 3.3% | 0.4% |

Kernel instance counts are unchanged, so the speedup is pure per-launch
latency reduction. The full derived tables are the
`nsys-*_cuda_gpu_kern_sum.csv` exports.

## Files

- `xtbloom-baseline.json` / `xtbloom-repaired.json`: benchmark matrices with raw
  samples, SCC iterations, energy values, and status failures (B1/8/32/128).
- `xtbloom-baseline.csv` / `xtbloom-repaired.csv`: same data as CSV.
- `natoms_scaling_pr172.py.txt`: the historical runner snapshot used for the
  matrix and profile JSON rows; it is stored as text so repository Python hooks
  do not rewrite its canonical bytes.
- `nsys-{baseline,repaired}-b{1,128}.json`: the profiled single-sample
  `natoms_scaling` rows.
- `nsys-{baseline,repaired}-b{1,128}_cuda_gpu_kern_sum.csv`: canonical kernel
  summary exports regenerated from the local reports with
  `nsys stats --report cuda_gpu_kern_sum`.
- `README.md`, `SHA256SUMS`: this document and artifact hashes.

The captured JSON artifacts and the archived runner differ from their local
capture bytes only by an appended `0x0a`, which the repository-wide
`end-of-file-fixer` requires. The two end-to-end CSVs use LF rather than the
historical CSV writer's CRLF line terminators so `git diff --check` remains
clean; their field values are unchanged. The committed `SHA256SUMS` pins every
archived byte.

## Commands

Build both source revisions with the same Release/shared/sm_120 configuration
before running the matrix:

```bash
cmake -S . -B <build> -G 'Unix Makefiles' \
  -DXTBLOOM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DXTBLOOM_MKL_RT_LIBRARY=/group/software/deepmd-kit-3.1.1/lib/libmkl_rt.so.2
cmake --build <build> --parallel
```

For each source revision, place the snapshot at
`benchmarks/natoms_scaling.py` in that checkout, then run the historical
matrix command (the archive contains C40H82 only, so `--molecules 122` is
intentional):

```bash
srun --gres=gpu:1 env \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/group/software/deepmd-kit-3.1.1/lib \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
python3 benchmarks/natoms_scaling.py \
  --library <build>/libxtbloom.so.0.1.0 \
  --molecules 122 --batch-sizes 1,8,32,128 \
  --engines xtbloom --backends cuda --properties force \
  --cuda-root /group/software/cuda-12.9.1 \
  --warmups 1 --repetitions 3 \
  --output-json <out>.json --output-csv <out>.csv
```

Profiled runs (one per batch, CUDA tracing only):

```bash
srun --gres=gpu:1 env -i \
  PATH=/home/jzzeng/miniconda3/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/group/software/deepmd-kit-3.1.1/lib \
  CUDA_VISIBLE_DEVICES=0 \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
/group/software/cuda-12.9.1/nsight-systems-2025.1.3/target-linux-x64/nsys profile \
  --force-overwrite true -o <report-prefix> -t cuda \
  python3 benchmarks/natoms_scaling.py \
    --library <build>/libxtbloom.so.0.1.0 \
    --molecules 122 --batch-sizes <B> --engines xtbloom --backends cuda \
    --properties force --warmups <1 for B1, 0 for B128> \
    --repetitions 1 \
    --output-json <out>.json --output-csv <out>.csv
```

Kernel-summary export from a report:

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  --output <prefix> <report>.nsys-rep
```

The four captured reports recorded the exact target argv in
`META_DATA_CAPTURE`; the archived profile JSON/CSV rows and kernel summaries
were derived from those reports. Before sharing a raw report, inspect its
`META_DATA_CAPTURE` export and verify that no credentials or private session
metadata are present. This archive retains only the derived summaries.

## Correctness and acceptance evidence from PR #172

The parallel implementation changed only per-kernel parallelism and reduction
order; the public binary64 occupation policy, deterministic serial orbital-order
validation, and transactional per-system failure publication are unchanged and
were re-validated in PR #172:

- focused `^xtbloom\.cuda\.occupations$`: 1/1 PASS;
- full RTX 5090 CUDA CTest: 101/101 PASS, including public CUDA host/device/
  mixed conformance and the 63/64/65/129-orbital threshold matrix;
- restricted and mixed-spin B=1/8/32/128, finite-T/T=0/zero/full/near-capacity/
  small-target occupations, exact-degenerate 65-orbital cases, repeated-call
  determinism, Graph replay, and deterministic peer-isolated multi-fault
  failure isolation;
- strict `sm_120` four-tool Compute Sanitizer gate: memcheck 0 errors,
  racecheck 0 hazards/errors/warnings, initcheck 0 errors, synccheck 0 errors;
- repository-wide `prek@0.3.1` and `git diff --check`: PASS.

Those exact commands and pass/fail counts are recorded in this #184 archive and
the PR #172 merge checkpoint. Issue #138 was a stale historical label from the
original implementation branch and does not exist in this repository.
