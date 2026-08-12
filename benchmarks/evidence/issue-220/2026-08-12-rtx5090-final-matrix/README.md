# Issue #220 revised CUDA profiling evidence

This bundle records the owner-approved reduced closure matrix for issues #220
and #84. It is measured from clean source revision
`12eef19400c2b9788064db3368ce15548eafacc4` on an NVIDIA GeForce RTX 5090.

The exhaustive public SCC Cartesian sweep was cancelled because its cost was
disproportionate: one compact public `N=256, B=32` FRESH call took about 358
seconds and `N=256, B=128` exceeded the tested 32 GiB device arena. Nsight
Compute counters are also removed from closure scope because this host returns
`ERR_NVGPUCTRPERM`. These coordinates are not passes; they remain explicitly
`NOT RUN` or `UNAVAILABLE` below.

## Narrow conclusion

- The pair-list matrix covers open and compact topologies, atom counts
  `16/32/48/64/96/128/256`, and batches `1/8/32/128`. All 56 cells passed
  structural, generation, error, pair-order, and coordination validation. The
  authoritative JSON retains 3,360 CUDA-event samples: three warmups followed
  by 20 samples for sparse build, forced dense build, and unchanged-generation
  coordination reuse in every cell.
- Open topology favors dense construction for all four `N=16` cells and sparse
  construction for all 24 `N>=32` cells. At `N=256`, dense/sparse median ratios
  range from 1.25x to 29.71x.
- Compact topology favors dense construction in 22 of 28 cells. Sparse wins
  consistently at `N=256`, but only by 1.002x--1.089x. Because production
  dispatch is topology-agnostic and the two topology classes disagree near the
  crossover, this evidence does not justify changing the existing 40-atom
  threshold. Production dispatch remains unchanged.
- D4 and AES2 each cover the same 56 coordinates and five production term
  entries, with three warmups and 20 samples per term: 280 rows and 5,600 raw
  samples per subsystem. Every source cell was published only after its
  post-timing CPU parity/error validation passed.
- D4 rows now distinguish the actual committed 50-bohr retained pair count
  from the dense full-triangle extent. For the largest open cell, the committed
  list contains 129,792 pairs versus a 4,177,920-pair dense extent. The sum of
  isolated term medians is 3.154 ms open versus 687.712 ms compact; these
  independently timed terms must not be added to claim public end-to-end
  latency.
- AES2 intentionally traverses packed all-pairs. The largest-cell summed
  isolated medians are 16.282 ms for both open and compact topology, supporting
  the topology-independent design.

This bundle complements the merged migration evidence in the sibling
`2026-08-12-rtx5090/` directory, which records the retained-memory reduction,
representative public correctness-qualified latency, and Nsight Systems work
reduction. No new public speedup, DRAM-byte, occupancy, or full-matrix claim is
made here.

## Matrix and timing boundary

| Evidence | Topologies | Atoms/system | Batch | Rows | Raw samples |
| --- | --- | --- | --- | ---: | ---: |
| Pair-list | open, compact | 16, 32, 48, 64, 96, 128, 256 | 1, 8, 32, 128 | 56 cells x 3 modes | 3,360 |
| D4 terms | open, compact | same | same | 56 cells x 5 terms | 5,600 |
| AES2 terms | open, compact | same | same | 56 cells x 5 terms | 5,600 |

Every measurement uses a non-blocking CUDA stream and CUDA events. Fixture
setup, correctness downloads, and post-timing validation are outside the
measured interval. The pair-list modes are the production sparse 50-bohr
builder, forced dense construction on the same geometry, and coordination
evaluation over the already built unchanged-generation sparse cache.

D4 rows time CN-cache update, two-body energy/potential, ATM energy, two-body
gradient, and ATM gradient through the production CUDA entry points. AES2 rows
time geometry, potential, energy, VJP, and the complete term chain. Optional
single-launch profiler ranges exist, but no profiler-counter data is claimed.

## Hardware and build identity

| Item | Value |
| --- | --- |
| Host / CPU | `node3`, AMD EPYC 7K62 48-Core Processor |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, compute capability 12.0 |
| Driver | 580.95.05 |
| CUDA compiler/runtime | CUDA 12.9.86 / runtime 12.9 |
| CUDA architecture | `sm_120` |
| Host compiler | GCC 11.4.0 (`/usr/bin/c++`) |
| CMake / Ninja | 4.2.1 / 1.13.0 |
| Build | shared Release, explicit CUDA ON |
| LP64 runtime | `scipy_openblas32/lib/libscipy_openblas.so` |
| CMakeCache SHA-256 | `05681c5e7250354e947ed4e2ed11621e9d13b028b364d5cdb87bcc6d52c35c61` |

Selected benchmark executable identities:

| Target | SHA-256 |
| --- | --- |
| `xtbloom_cuda_pairlist_benchmark` | `d1f59bbe89474f1fad431ef835bece36518a28e60158a3ddbb5c63d3860cd79a` |
| `xtbloom_cuda_d4_term_benchmark` | `fcdac0fc917235a121ed4f5df939b6c88499c7d62938bbedcc56f6fd38910a66` |
| `xtbloom_cuda_aes2_term_benchmark` | `13819c456c2f034dcced68ef5274044cecebe0e4eccbb90246ad0b3038085849` |

The executables validate the syntax of caller-supplied revision/hash fields and
label them `caller_supplied_and_archiver_verified`. `artifact-manifest.json`
records byte count and SHA-256 for every paired per-cell JSON and CSV input.
`aggregate.py` fails closed on missing coordinates, missing JSON/CSV peers,
inconsistent provenance, incorrect pair-count semantics, fewer than three
warmups or 20 samples, or incomplete distributions.

## Reproduction commands

The pair-list matrix was generated with:

```bash
env -u CUDA_VISIBLE_DEVICES \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/cuda-12.9.1/targets/x86_64-linux/lib \
  ./build/issue220-finalhead-cuda/xtbloom_cuda_pairlist_benchmark \
  --atoms 16,32,48,64,96,128,256 \
  --batch-sizes 1,8,32,128 --topology both --cutoff 50 \
  --warmups 3 --samples 20 \
  --json build/issue220-final-measurements/pairlist-matrix.json \
  --csv build/issue220-final-measurements/pairlist-matrix.csv \
  --source-revision 12eef19400c2b9788064db3368ce15548eafacc4 \
  --executable-sha256 d1f59bbe89474f1fad431ef835bece36518a28e60158a3ddbb5c63d3860cd79a \
  --build-identity-sha256 05681c5e7250354e947ed4e2ed11621e9d13b028b364d5cdb87bcc6d52c35c61
```

D4 and AES2 used the same environment and this loop shape, with `KIND`, `EXE`,
and `EXE_SHA256` set to the corresponding target:

```bash
for topology in compact open; do
  for atoms in 16 32 48 64 96 128 256; do
    for batch in 1 8 32 128; do
      "$EXE" --topology "$topology" --atoms-per-system "$atoms" \
        --batch "$batch" --warmups 3 --samples 20 \
        --json "build/issue220-final-measurements/terms/$KIND/$topology-b$batch-n$atoms.json" \
        --csv "build/issue220-final-measurements/terms/$KIND/$topology-b$batch-n$atoms.csv" \
        --source-revision 12eef19400c2b9788064db3368ce15548eafacc4 \
        --executable-sha256 "$EXE_SHA256" \
        --build-identity-sha256 05681c5e7250354e947ed4e2ed11621e9d13b028b364d5cdb87bcc6d52c35c61
    done
  done
done
```

Retained artifacts are regenerated and verified with:

```bash
python3 benchmarks/evidence/issue-220/2026-08-12-rtx5090-final-matrix/aggregate.py \
  --input-root build/issue220-final-measurements \
  --output-dir benchmarks/evidence/issue-220/2026-08-12-rtx5090-final-matrix
sha256sum -c benchmarks/evidence/issue-220/2026-08-12-rtx5090-final-matrix/SHA256SUMS
```

## Correctness qualification and exclusions

At the measured head, the focused real-GPU tests
`xtbloom.cuda.{pairlist,d4,aes2}` passed 3/3. The six benchmark unit modules
passed 108/108; one optional Matplotlib rendering test was skipped by design.
The final PR ledger records the fresh full CPU/CUDA and repository checks.

| Coordinate | Status | Reason |
| --- | --- | --- |
| Full public SCC Cartesian sweep across every N/B/topology/workload/start policy | NOT RUN; removed from closure scope | disproportionate multi-day cost |
| Compact public FRESH `N=256, B=128` | NOT RUN; removed from closure scope | numerical-refresh arena exceeded 32 GiB |
| Nsight Compute DRAM/occupancy counters | UNAVAILABLE; removed from closure scope | `ERR_NVGPUCTRPERM`, `RmProfilingAdminOnly: 1` |
| Gas/QM/MM heterogeneous public performance sweep | NOT RUN; removed from closure scope | only short correctness smoke remains required |

No raw `.nsys-rep`, `.ncu-rep`, `.sqlite`, or other native profiler capture is
retained. Each tracked file remains below 1 MiB and the complete tracked
`benchmarks/evidence/` directory remains below 16 MiB.
