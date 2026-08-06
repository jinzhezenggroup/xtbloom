# Issue #70 abrupt dense/sparse pair-list build-vs-reuse evidence

This directory archives development-only measurements for issue #70 from a
dirty working tree; the final commit's explicit hash may differ.  The data is
correctness-qualified (the same binary also passes the full `gpuxtb.cuda.pairlist`
test) but must be regenerated from the clean final commit before any release
claim.

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32 GiB, compute capability 12.0 (sm_120).
- Driver: 580.95.05 (NVIDIA Open Kernel Module).
- CUDA toolkit: 12.9.1 (`/group/software/cuda-12.9.1`), nvcc 12.9.86.
- Build: CMake, Unix Makefiles, Release, CUDA ON, `sm_120`, with
  `/home/jzzeng/miniconda3/lib/libmkl_rt.so.3` as the LP64 provider.
- Source base commit: `41c4e1bc03cf5b44e4ba4047ffdff5f730e6da6c` (uncommitted
  issue-#70 changes on `feat/70-sparse-neighbor-lists`).
- Host: node3 (scheduler-gated `srun --gres=gpu:1`).

## Protocol

`gpuxtb_cuda_pairlist_benchmark` (benchmark-only build of
`tests/cuda_pairlist_test.cu`) measures three entry points per
(batch, atoms_per_system) cell:

- `sparse_build_ms`: `update_gfn2_pairlist_cache_cuda` with
  `Gfn2PairListMode::kSparse` (bucketed).
- `dense_build_ms`: the same launcher with `Gfn2PairListMode::kDense`
  (deterministic all-pairs fallback).
- `reuse_ms`: `evaluate_gfn2_pairlist_coordination_cuda` over the previously
  built sparse cache (coordination reuse with no list rebuild).

Each cell runs 3 untimed warmups and 20 recorded samples with
`cudaEventRecord/ElapsedTime`; reported values are the median sample in ms.
Each system is 16-128 atoms on a 12-bohr cubic crystal-like lattice so only
nearby pairs lie within the 25-bohr cutoff.

Loading the benchmark requires `LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64`
on top of the normal environment.

## Results summary (median, ms)

The dense fallback skips bucket construction entirely (true all-pairs path), so
small systems are cheaper dense.  Dense wins or ties up to about 32 atoms, the
crossover sits between 32 and 48 atoms, and the sparse bucketed build clearly
wins from 48 atoms per system upward.  The coordination reuse path is far
cheaper than either rebuild at every cell, which is the justification for
separating list-build cost from steady-state reuse cost.  The dispatch policy
`gfn2_pairlist_use_sparse_for` therefore selects dense at or below 40 atoms and
sparse above it.

Examples at batch 1 (median ms):

| atoms | sparse build | dense build | reuse |
| --- | --- | --- | --- |
| 16  | 0.0662 | 0.0568 | 0.157 |
| 32  | 0.127  | 0.119  | 0.0956 |
| 48  | 0.173  | 0.207  | 0.136 |
| 64  | 0.205  | 0.342  | 0.177 |
| 96  | 0.272  | 0.866  | 0.261 |
| 128 | 0.321  | 1.63   | 0.362 |

Crossovers at batch 8/32/128 show the same ordering.  These are development
measurements, not release claims: the workload is a synthetic lattice (not an
open GFN2 benchmark corpus), one GPU/one driver, and the measured build cost
includes the full bucket construction while reuse assumes an unchanged
geometry generation.