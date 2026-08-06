# Issue #138 CUDA occupations benchmark evidence

This directory archives the correctness-qualified benchmark and Nsight
profiling evidence for issue #138 / PR #172 ("perf: parallelize the CUDA
occupations evaluation", squash commit `89a53220655e6fb4037890db487207819b598cc3`).

The parallel occupations implementation is already on `main`. This directory
records the durable before/after artifacts that the issue required: the raw
public-C-API benchmark matrices and the Nsight Systems reports for batch 1 and
batch 128. "Baseline" is the occupations evaluation before PR #172
(`pre-172`, i.e. `main` at `cf0fe8dcf802211892f96436da2f276e49c9199a` plus the
PR branch without the occupations change). "Repaired" is the parallel
implementation at PR #172 head `6b71c2a1392497ad8e595b8f88561db300e2b0a6`.

## Environment

- Host: `node3` (SLURM cluster), OS Ubuntu 22.04.5 LTS, Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62 48-Core Processor.
- GPU: NVIDIA GeForce RTX 5090 (GB202), `sm_120`, 32 GiB, CUDA visible device 0.
- CUDA runtime: `/group/software/cuda-12.9.1` (CUDA 12.9.1, driver 580.95.05 /
  CUDA 13.0 driver API).
- Nsight Systems 2025.1.3:
  `/group/software/cuda-12.9.1/nsight-systems-2025.1.3/target-linux-x64/nsys`.
- Build: Release, shared library, strict `sm_120`, CUDA 12.9.86 nvcc; both
  builds produced `libgpuxtb.so.0.1.0`.
- Measured env: `LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/group/software/deepmd-kit-3.1.1/lib`,
  `CUDA_VISIBLE_DEVICES=0`, `MKL_INTERFACE_LAYER=LP64`,
  `MKL_THREADING_LAYER=SEQUENTIAL`.

The exact argv and environment of every profiled run are embedded in the
`META_DATA_CAPTURE` table of each `.nsys-rep` report (import with
`nsys import`).

## Protocol

All rows run GFN2-xTB analytic forces for a batch of one neutral closed-shell
C40H82 (122 atoms) alkane through gpuxtb's public C ABI with host-pointed CUDA
descriptors, fresh 12-SCC-iteration execution, one warmup, and recorded
samples. Setup, result inspection, and serialization are outside the timing
interval.

- End-to-end matrices (B1/8/32/128, `benchmarks/run.py`): one warmup plus
  three recorded samples per row.
- Nsight profiles (`benchmarks/natoms_scaling.py`, batch 1 or 128 only): one
  untimed run under `nsys profile` with CUDA tracing only; profiling overhead
  makes these single-sample timings slower than the unprofiled matrix rows and
  are used only for the kernel-level comparison.

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
latency reduction. The per-report full tables are the
`nsys-*_cuda_gpu_kern_sum.csv` exports; the raw reports are the `.nsys-rep`
files.

## Files

- `gpuxtb-baseline.json` / `gpuxtb-repaired.json`: benchmark matrices with raw
  samples, SCC iterations, energy values, and status failures (B1/8/32/128).
- `gpuxtb-baseline.csv` / `gpuxtb-repaired.csv`: same data as CSV.
- `nsys-{baseline,repaired}-b{1,128}.json`: the profiled single-sample
  `natoms_scaling` rows.
- `nsys-{baseline,repaired}-b{1,128}.nsys-rep`: Nsight Systems 2025.1.3 CUDA
  reports (open with `nsys-ui` or analyze with `nsys stats`).
- `nsys-{baseline,repaired}-b{1,128}_cuda_gpu_kern_sum.csv`: canonical kernel
  summary exports regenerated with `nsys stats --report cuda_gpu_kern_sum`.
- `README.md`, `SHA256SUMS`: this document and artifact hashes.

The captured JSON artifacts are archived as captured except for a single
trailing newline per file, which the repository-wide `end-of-file-fixer` hook
requires and the current benchmark writers (`write_json`, `natoms_scaling`)
already emit; the byte difference is an appended `0x0a`, and the committed
`SHA256SUMS` pins the archived bytes.

The derived `.sqlite` analysis databases are not archived (about 140 MB) but
can be regenerated from the `.nsys-rep` files:

```bash
nsys export --type=sqlite -o report.sqlite nsys-baseline-b1.nsys-rep
```

## Commands

End-to-end matrix (per build), one warmup + three samples, B1/8/32/128:

```bash
srun --gres=gpu:1 env \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/group/software/deepmd-kit-3.1.1/lib \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
python3 benchmarks/run.py \
  --library <build>/libgpuxtb.so.0.1.0 \
  --engines gpuxtb --backends cuda --cuda-memory-modes host \
  --workloads gas --properties force --batch-sizes 1,8,32,128 \
  --cuda-root /group/software/cuda-12.9.1 \
  --warmups 1 --repetitions 3 \
  --output-json <out>.json --output-csv <out>.csv
```

Profiled runs (one per batch, CUDA tracing only):

```bash
srun --gres=gpu:1 env \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:/group/software/deepmd-kit-3.1.1/lib \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
/group/software/cuda-12.9.1/nsight-systems-2025.1.3/target-linux-x64/nsys profile \
  --force-overwrite true -o <report-prefix> -t cuda \
  python3 benchmarks/natoms_scaling.py \
    --library <build>/libgpuxtb.so.0.1.0 \
    --molecules 122 --batch-sizes <B> --engines gpuxtb --backends cuda \
    --properties force --warmups <0 or 1> --repetitions 1 \
    --cuda-root /group/software/cuda-12.9.1 \
    --output-json <out>.json --output-csv <out>.csv
```

Kernel-summary export from a report:

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  --output <prefix> <report>.nsys-rep
```

## Correctness and acceptance evidence from PR #172

The parallel implementation changed only per-kernel parallelism and reduction
order; the public binary64 occupation policy, deterministic serial orbital-order
validation, and transactional per-system failure publication are unchanged and
were re-validated in PR #172:

- focused `^gpuxtb\.cuda\.occupations$`: 1/1 PASS;
- full RTX 5090 CUDA CTest: 101/101 PASS, including public CUDA host/device/
  mixed conformance and the 63/64/65/129-orbital threshold matrix;
- restricted and mixed-spin B=1/8/32/128, finite-T/T=0/zero/full/near-capacity/
  small-target occupations, exact-degenerate 65-orbital cases, repeated-call
  determinism, Graph replay, and deterministic peer-isolated multi-fault
  failure isolation;
- strict `sm_120` four-tool Compute Sanitizer gate: memcheck 0 errors,
  racecheck 0 hazards/errors/warnings, initcheck 0 errors, synccheck 0 errors;
- repository-wide `prek@0.3.1` and `git diff --check`: PASS.

Those exact commands and pass/fail counts were recorded on issue #138 and in
the PR #172 merge checkpoint.
