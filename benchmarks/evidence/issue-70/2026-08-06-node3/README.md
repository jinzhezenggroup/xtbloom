# Issue #70 abrupt dense/sparse pair-list build-vs-reuse evidence

This directory archives development-only measurements for issue #70 from the
committed source revision
`d6de01433061e150a73a6efec16714de670b1105`. The data is
correctness-qualified (the same binary also passes the full
`xtbloom.cuda.pairlist` test) but is not a release performance claim.

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32 GiB, compute capability 12.0 (sm_120).
- Driver: 580.95.05 (NVIDIA Open Kernel Module).
- CUDA toolkit: 12.9.1 (`/group/software/cuda-12.9.1`), nvcc 12.9.86.
- Build: CMake, Unix Makefiles, Release, CUDA ON, `sm_120`, with
  `/home/jzzeng/miniconda3/lib/libmkl_rt.so.3` as the LP64 provider.
- Source revision: `d6de01433061e150a73a6efec16714de670b1105`
  (`feat/70-sparse-neighbor-lists`).
- Benchmark executable SHA-256:
  `ee3fd7a7c7d1d9f2fbdc6a923db15981a509e0710d54e6b579026d6e6d123956`.
- Build identity SHA-256 (the Release CUDA `CMakeCache.txt`):
  `33b8aa52998a4eb22a73fffb1a7f8b181f55bfca7beb5edfff6d072133534082`.
- Host: node3 (scheduler-gated `srun --gres=gpu:1`).

## Protocol

`xtbloom_cuda_pairlist_benchmark` (benchmark-only build of
`tests/cuda_pairlist_test.cu`) measures three entry points per
(batch, atoms_per_system) cell:

- `sparse_build_ms`: `update_gfn2_pairlist_cache_cuda` with
  `Gfn2PairListMode::kSparse` (bucketed).
- `dense_build_ms`: the same launcher with `Gfn2PairListMode::kDense`
  (deterministic all-pairs fallback).
- `reuse_ms`: `evaluate_gfn2_pairlist_coordination_cuda` over the previously
  built sparse cache (coordination reuse with no list rebuild).

Each cell runs 3 untimed warmups and 20 recorded samples with
`cudaEventRecord/ElapsedTime`; reported values are the median of a sorted copy
of the samples in ms. The raw JSON retains acquisition order and all 20
samples for every cell. `reuse_ms` includes the cached-pair preflight and
coordination evaluation, but excludes the one-time sparse rebuild.
Each system is 16-128 atoms on a 12-bohr cubic crystal-like lattice so only
nearby pairs lie within the 25-bohr cutoff.

Loading the benchmark requires `LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64`
on top of the normal environment.

The committed-source run was collected with:

```bash
srun --gres=gpu:1 --wait=60 env \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/home/jzzeng/miniconda3/lib \
  build/cuda-pr202-make/xtbloom_cuda_pairlist_benchmark \
  --source-revision d6de01433061e150a73a6efec16714de670b1105 \
  --executable-sha256 ee3fd7a7c7d1d9f2fbdc6a923db15981a509e0710d54e6b579026d6e6d123956 \
  --build-identity-sha256 33b8aa52998a4eb22a73fffb1a7f8b181f55bfca7beb5edfff6d072133534082 \
  --json benchmarks/evidence/issue-70/2026-08-06-node3/xtbloom-cuda-pairlist-bench.json \
  --csv benchmarks/evidence/issue-70/2026-08-06-node3/xtbloom-cuda-pairlist-bench.csv
```

`xtbloom-cuda-pairlist-bench.json` is the authoritative raw-sample artifact;
the CSV is its median view from the same invocation.

## Results summary (median, ms)

The dense fallback skips bucket construction entirely (true all-pairs path), so
small systems are cheaper dense.  Dense wins or ties up to about 32 atoms, the
crossover sits between 32 and 48 atoms, and the sparse bucketed build clearly
wins from 48 atoms per system upward. The coordination reuse path is lower than
either rebuild after the fixed launch overhead is amortized; for the smallest
batch-1/16-atom cell, dense construction is cheaper. This is the justification
for separating list-build cost from steady-state reuse cost. The dispatch
policy `gfn2_pairlist_use_sparse_for` therefore selects dense at or below 40
atoms and sparse above it.

Examples at batch 1 (median ms):

| atoms | sparse build | dense build | reuse |
| --- | --- | --- | --- |
| 16  | 0.0659 | 0.0541 | 0.0924 |
| 32  | 0.130  | 0.117  | 0.109  |
| 48  | 0.178  | 0.206  | 0.156 |
| 64  | 0.215  | 0.341  | 0.163 |
| 96  | 0.284  | 0.864  | 0.174 |
| 128 | 0.337  | 1.63   | 0.183 |

Crossovers at batch 8/32/128 show the same ordering.  These are development
measurements, not release claims: the workload is a synthetic lattice (not an
open GFN2 benchmark corpus), one GPU/one driver, and the measured build cost
includes the full bucket construction while reuse assumes an unchanged
geometry generation.
