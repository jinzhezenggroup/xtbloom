# Issue #41: transactional per-system SCC mixer update evidence

Measured on one AMD EPYC 7K62 node (48 logical CPUs, shared cluster node),
Ubuntu 22.04.5, `g++ 11.4.0`, `cmake 4.2.1`, CMake `Unix Makefiles` Release
configuration with `GPUXTB_ENABLE_CUDA=OFF` and no BLAS/LAPACK runtime (the
mixer layer is pure C++ and never calls the eigensolver provider).

## Claim

After the #41 change, each SCC driver iteration copies only the history of the
systems that are actually active in that iteration, instead of duplicating the
complete mixer state (all systems, all history) before and after the mixer
barrier. The per-iteration mixer cost therefore scales with **active-system
history**, not **total batch history**.

## Artifacts

- `gpuxtb-mixer-transaction.json` — authoritative raw per-row samples (200
  samples per row) and exact byte accounting.
- `gpuxtb-mixer-transaction.csv` — compact table view of the row medians.
- `gpuxtb-mixer-transaction.txt` — full stdout transcript including the
  configuration line.

## Command

Source tree HEAD `41c4e1bc03cf5b44e4ba4047ffdff5f730e6da6c` plus the uncommitted
#41 working tree (documented in `gpuxtb-mixer-transaction.json` provenance
convention); the benchmark binary is built from that exact tree:

```bash
cmake -S . -B build/cpu41 -DGPUXTB_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu41 --target gpuxtb_scc_mixer_transaction_benchmark --parallel
./build/cpu41/gpuxtb_scc_mixer_transaction_benchmark 64 6 64 200 \
  benchmarks/evidence/issue-41/2026-08-06-epyc7k62/gpuxtb-mixer-transaction.json
```

Configuration: ragged batch of `batch=64` systems, each `atoms_per_system=6`
carbon atoms (GFN2 sp basis), `history=64`.
`state_size_bytes=4,461,952` (~4.3 MiB) and total mixer history
`2,162,688` doubles (16.5 MiB) are copied twice today only in the pre-change
full-state path. `new_transaction_bytes` is the exact per-system move size
(`3*dim` vector records + `2*dim*history` Broyden history + `history` omegas +
batch-scalar entries) times the active count.

## Results (median of 200 samples per row, microseconds)

| active | old_copy_us | copy_transaction_us | mix_transaction_us | old_copy_bytes | new_transaction_bytes |
| --- | --- | --- | --- | --- | --- |
| 8  | 242.870 |  32.912 |  2436.134 | 8,923,904 |   557,744 |
| 16 | 240.766 |  65.454 |  4878.600 | 8,923,904 | 1,115,488 |
| 32 | 240.255 | 130.387 |  9759.674 | 8,923,904 | 2,230,976 |
| 64 | 245.956 | 282.956 | 19552.653 | 8,923,904 | 4,461,952 |

Interpretation:

- `old_copy_us` — the pre-change full mixer-state copy (two `state_size_bytes`
  memcpys) is **constant ~240-246 us per iteration** no matter how few systems
  are active; it always moves the whole 8,923,904-byte batch history.
- `copy_transaction_us` — the #41 per-system prepare+commit stage is linear in
  the active count (≈4.1 us/system): 32.9/65.5/130.4/283.0 us for 8/16/32/64
  active of 64, and 7.4x cheaper than the old full copy when only 8 of 64
  systems are active.
- `mix_transaction_us` — full prepare+Broyden transition+commit is likewise
  linear in the active count (≈305 us/system), confirming neither the staging
  nor the mixing itself depends on the batch's inactive history.
- `new_transaction_bytes` — the #41 path moves exactly `active/total` of the
  batch history (557,744 / 8,923,904 = 1/16 when 8 of 64 systems are active).

## Measurement notes and limitations

- A large read/write "scrubber" buffer is touched before every measured round
  so the per-system transactions and the whole-state memcpy baseline are both
  timed cold in the same cache regime; the reported per-system costs therefore
  sit above a tiny hot-loop lower bound.
- The same `staged` binding and identical raw outputs drive every row, and the
  driver (not the benchmark) still owns where prepare/commit are called; this
  microbenchmark isolates the mixer transaction stage and does not time
  eigensolves, potentials, or the publication loop of a full SCC iteration.
- Raw samples are recorded in the JSON: 200 per row. This is development
  evidence for a single node, workload shape, and toolchain; it is not a
  release throughput claim.