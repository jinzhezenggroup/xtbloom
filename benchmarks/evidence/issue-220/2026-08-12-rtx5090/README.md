# Issue #220 CUDA D4 pair-list migration evidence

This bundle compares clean `main` baseline `21ec5463840092ac6b85a624fb151a8512f65fdd`
with clean candidate `a70acdb8f74c544e461070cfaa797b8bd495c2ef` on one NVIDIA
GeForce RTX 5090.  The candidate removes the production dense five-value D4
pair cache, consumes the committed 50-bohr physical pair-list superset with
independent inclusive 30/50/25-bohr CN/two-body/ATM predicates, and splits
large ATM traversal across CUDA blocks.

## Narrow conclusion

- The public CUDA force plan retains 0.94% to 1.56% fewer device bytes for the
  measured single-system 32--272 atom sweep.  The 32-atom batch-128 plan saves
  7,652,352 bytes (7.30 MiB, 1.03%).
- The deleted dense D4 main term is exactly `120 * P` bytes, where
  `P = sum_s n_s(n_s - 1) / 2`.  Baseline retained three copies of the five
  binary64 values per packed pair: the committed/public copy, numerical
  candidate copy, and numerical scratch copy.  The public plan delta is
  slightly larger because obsolete generation/sequence leaves and arena
  padding also disappear.
- Correctness-qualified public FRESH force medians change from -8.31% to
  +2.01% across the six coordinates (negative is faster).  The 242- and
  272-atom rows improve by 2.49% and 2.26%; 62/122 atoms and 32 atoms at
  batch 128 remain within about 2% of baseline.  This supports "no material
  end-to-end regression" rather than a uniform speedup claim.
- On the profiler-perturbed 272-atom public workload, D4-classified kernel
  time falls 13.57%, D4 kernel instances fall 31.91%, total kernel time falls
  1.18%, total kernel instances fall 3.43%, and `cudaLaunchKernel` calls fall
  by 39.  D4 classification requires the demangled kernel name/signature to
  contain `Gfn2D4`; it intentionally excludes unrelated cuSOLVER kernels such
  as `sytrd4_gpu`.
- Baseline and candidate energies are bitwise equal for every retained timing
  coordinate.  The largest cross-version force difference is
  `1.74e-16` Hartree/bohr, and every harness correctness row passes.

No DRAM-byte or occupancy claim is made.  Nsight Compute 2025.2.1 reported
`ERR_NVGPUCTRPERM` for device 0 because this user cannot access GPU performance
counters.  Nsight Systems kernel/API/memory-operation timing is available and
cleanly derived, but it does not replace hardware DRAM counters.

This narrow bundle does not satisfy the complete #84/#220 closure matrix: it
does not add the B=8/32, topology-class, unchanged-geometry reuse, term-isolated
CN/D4/ATM/AES2, or Nsight Compute counter coordinates requested by the broader
design ledger.  The implementation and this evidence may merge with
`Refs #220` / `Refs #84`, while both issues remain open for those residual
profiling rows.

## Hardware and toolchain

| Item | Value |
| --- | --- |
| Host | `node3`, AMD EPYC 7K62 48-Core Processor |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, compute capability 12.0 |
| Driver | 580.95.05 |
| CUDA compiler | 12.9.86, `/group/software/cuda-12.9.1/bin/nvcc` |
| Host compiler | GCC/G++ 11.4.0 |
| CMake / Ninja | 4.2.1 / 1.13.0 |
| Nsight Systems | 2025.1.3.140-251335620677v0 |
| Nsight Compute | 2025.2.1.0 build 35987062; counters unavailable |
| Compute Sanitizer | 2025.2.1.0 build 35969825 |
| LP64 runtime | `scipy_openblas32/lib/libscipy_openblas.so` |
| LP64 SHA-256 | `b2dfe24b9aa11cf1d1cec8edbca9423b50cfd186b486d59dd4efe45826261a98` |

The timing processes were pinned to logical CPU 0.  `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` were 1; `OMP_DYNAMIC` and
`MKL_DYNAMIC` were `FALSE`.  Public descriptors and result buffers were in
host memory.  The public synchronous CUDA boundary supplied the timing
synchronization.

## Binary identity

| Revision | Library | SHA-256 |
| --- | --- | --- |
| baseline `21ec5463840092ac6b85a624fb151a8512f65fdd` | `$BASELINE_WT/build/issue220-main-cuda/libxtbloom.so.0.1.1` | `1fe58dd29ead68f20bee3cbfc21de9316af9178e8f1362c9ffd7dc71da0918f8` |
| candidate `a70acdb8f74c544e461070cfaa797b8bd495c2ef` | `$CANDIDATE_WT/build/issue220-final-cuda/libxtbloom.so.0.1.1` | `c5f59173b558d393d79966d064aa3435ffd14519954fd5ad614ebc1dc2c3e07b` |

Both source worktrees and selected library sources were clean.  Machine-local
checkout roots are represented as `$BASELINE_WT` and `$CANDIDATE_WT` so the
bundle follows the repository's retired-name text policy; revisions, relative
paths, binary hashes, and build-input hashes remain exact.  The complete
CMake cache/compiler/provider/source identities and hashes are retained in
`performance-summary.json` under each run identity.

## Public workspace results

The query created a CUDA fixed-topology plan with energy and forces, then read
`xtbloom_plan_query_workspace`.  Geometry and topology match the timing
workloads.

| Atoms/system | Batch | Baseline device bytes | Candidate device bytes | Saved bytes | Saved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 6,357,624 | 6,297,848 | 59,776 | 0.94% |
| 62 | 1 | 17,693,464 | 17,466,016 | 227,448 | 1.29% |
| 122 | 1 | 60,352,184 | 59,465,328 | 886,856 | 1.47% |
| 242 | 1 | 225,627,000 | 222,125,840 | 3,501,160 | 1.55% |
| 272 | 1 | 283,609,336 | 279,184,440 | 4,424,896 | 1.56% |
| 32 | 128 | 739,852,728 | 732,200,376 | 7,652,352 | 1.03% |

Host retained bytes also fall in every coordinate: 3.55% at 32/B1, 4.91% at
62/B1, 5.41% at 122/B1, 5.52% at 242/B1, 5.52% at 272/B1, and 6.01% at
32/B128.  Alignments remain 64 bytes on host and 256 bytes on device.

The archived `workspace-query.py` uses only the stable public C ABI.  From the
candidate repository root, the exact query is reproduced with:

```bash
unset CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/cuda-12.9.1/targets/x86_64-linux/lib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
python3 benchmarks/evidence/issue-220/2026-08-12-rtx5090/workspace-query.py \
  --baseline "$BASELINE_WT/build/issue220-main-cuda/libxtbloom.so" \
  --candidate "$CANDIDATE_WT/build/issue220-final-cuda/libxtbloom.so" \
  --output build/issue220-measurements/workspace.json --device-id 0
```

The exact compact output and all fields are also embedded in
`performance-summary.json`.

## Public FRESH latency results

Each process retained one public context/descriptor/result image per cell,
performed 10 warmups, and recorded 30 independent FRESH force calls.  The JSON
summary retains all latency samples, eligibility state, SCC iterations,
convergence, within-run drift, and cross-version drift.

| Atoms/system | Batch | Baseline median ms | Candidate median ms | Change | Speed ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 79.280 | 72.692 | -8.31% | 1.091x |
| 62 | 1 | 148.957 | 151.951 | +2.01% | 0.980x |
| 122 | 1 | 438.867 | 442.499 | +0.83% | 0.992x |
| 242 | 1 | 2116.918 | 2064.248 | -2.49% | 1.026x |
| 272 | 1 | 3124.463 | 3053.864 | -2.26% | 1.023x |
| 32 | 128 | 158.886 | 160.768 | +1.18% | 0.988x |

Negative change means the candidate is faster.  The raw distributions, not
only the medians, are in `performance-summary.json`.

### Exact timing commands

For baseline, `WT` and `LIB` were:

```bash
BASELINE_WT=/absolute/path/to/issue-220-d4-baseline
WT=$BASELINE_WT
LIB=$WT/build/issue220-main-cuda/libxtbloom.so
TAG=baseline
```

For candidate:

```bash
CANDIDATE_WT=/absolute/path/to/issue-220-d4
WT=$CANDIDATE_WT
LIB=$WT/build/issue220-final-cuda/libxtbloom.so
TAG=candidate
```

The B=1 process for each revision was:

```bash
unset CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/cuda-12.9.1/targets/x86_64-linux/lib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
taskset -c 0 python3 "$WT/benchmarks/natoms_scaling.py" \
  --engine xtbloom --library "$LIB" \
  --backend cuda --device-id 0 --cpu-threads 1 \
  --property force --start-mode fresh \
  --natoms 32,62,122,242,272 --batch-sizes 1 \
  --warmups 10 --repetitions 30 \
  --output-json "$WT/build/issue220-measurements/$TAG-b1.json" \
  --output-csv "$WT/build/issue220-measurements/$TAG-b1.csv"
```

The batch-128 process changed only these arguments:

```bash
--natoms 32 --batch-sizes 128 \
--output-json "$WT/build/issue220-measurements/$TAG-b128.json" \
--output-csv "$WT/build/issue220-measurements/$TAG-b128.csv"
```

## Nsight Systems evidence

The profile is a structural comparison, not a latency headline.  One cold
call, one warmup, and one measured FRESH call caused three production
executions to appear in kernel totals.  Baseline and candidate used the same
272-atom/B1 public host-descriptor force workload.

```bash
unset CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/cuda-12.9.1/targets/x86_64-linux/lib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
RAW=$WT/build/issue220-measurements/nsys-raw/$TAG
taskset -c 0 /group/software/cuda-12.9.1/bin/nsys profile \
  --trace=cuda,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true --output "$RAW" \
  python3 "$WT/benchmarks/natoms_scaling.py" \
  --engine xtbloom --library "$LIB" \
  --backend cuda --device-id 0 --cpu-threads 1 \
  --property force --start-mode fresh \
  --natoms 272 --batch-sizes 1 --warmups 1 --repetitions 1 \
  --output-json "$RAW-profile.json" --output-csv "$RAW-profile.csv"

/group/software/cuda-12.9.1/bin/nsys stats \
  --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,cuda_kern_exec_sum \
  --format csv --force-overwrite=true \
  --output "$WT/build/issue220-measurements/nsys-derived/$TAG" \
  "$RAW.nsys-rep"
```

Only the sanitized derived CSV reports are tracked.  The native `.nsys-rep`
and generated `.sqlite` captures remain under ignored `build/` and are not
archived because native profiler captures can contain process environment
data.

Nsight Compute availability was tested with:

```bash
unset CUDA_VISIBLE_DEVICES
srun --gres=gpu:1 --ntasks=1 \
  /group/software/cuda-12.9.1/bin/ncu --devices 0 \
  --query-metrics --query-metrics-mode base
```

The command returned shell status zero but printed
`ERR_NVGPUCTRPERM`; therefore the DRAM/occupancy row is `UNAVAILABLE`, not
`PASS`.

## Correctness and safety qualification

Candidate `a70acdb8f74c544e461070cfaa797b8bd495c2ef` passed the following before
timing:

- Fresh explicit CUDA 12.9.86 Release shared build, `sm_120`, verified LP64
  OpenBLAS: 126 tests registered and 126/126 passed in 85.37 s on the RTX 5090.
  This includes restricted/unrestricted public inference, runtime owner,
  Graph/parity/cache tests, CUDA D4, host/device/mixed public conformance and
  invariants, CPU public inference, ABI, licensing, oracle, and canonical
  checks.
- Fresh Release shared CPU build with the same LP64 provider: 53 tests
  registered and 53/53 passed in 19.94 s.
- `xtbloom_cuda_d4_test` under Compute Sanitizer memcheck, racecheck,
  initcheck, and synccheck: exit 0 with zero errors/hazards/warnings for all
  four tools.
- Production `xtbloom_cuda_gfn2_runtime_owner_test` under memcheck: exit 0,
  zero errors.
- Repository `prek@0.3.1`, official-PyPI `uv lock --check`, and
  `git diff --check`: passed.
- Generated GFN2 parameters and conformance manifest checks: passed (9
  conformance cases).
- Parameter tests 14/14, conformance tests 45/45, licensing tests 109/109,
  and oracle tests 96/96: passed.
- Benchmark self-tests `benchmarks.test_natoms_scaling` plus
  `benchmarks.test_evidence_size`: 36/36 passed.

The exact sanitizer form was:

```bash
unset CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/cuda-12.9.1/targets/x86_64-linux/lib
for tool in memcheck racecheck initcheck synccheck; do
  /group/software/cuda-12.9.1/bin/compute-sanitizer \
    --tool "$tool" --error-exitcode=99 \
    build/issue220-final-cuda/xtbloom_cuda_d4_test
done
/group/software/cuda-12.9.1/bin/compute-sanitizer \
  --tool memcheck --error-exitcode=99 \
  build/issue220-final-cuda/xtbloom_cuda_gfn2_runtime_owner_test
```

No issue #279 disposition was needed: the affected D4 path's synccheck run was
itself clean.

## Artifact policy

The four authoritative harness JSON files are 2.40--13.12 MiB because they
retain every force vector for every measured sample.  They are intentionally
omitted from Git rather than truncated or hand-edited.  Their byte counts,
SHA-256 hashes, and local paths are in `artifact-manifest.json`; the compact
summary retains all latency samples, SCC/correctness results, cross-version
errors, complete run identity, and reproduction commands.

`SHA256SUMS` covers every retained artifact in this directory.  No raw
`.nsys-rep`, `.sqlite`, `.qdstrm`, or Nsight Compute native report is present.
