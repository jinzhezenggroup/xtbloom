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
  samples per row) and exact byte accounting for the default rows
  (active = 8/16/32/64 of a fixed batch of 64).
- `gpuxtb-mixer-transaction-full-range.json` — same protocol with explicit
  active counts 1/2/4/8/16/32/64.
- `gpuxtb-mixer-transaction.csv` — compact table view of the default row
  medians.
- `gpuxtb-mixer-transaction.txt` — full stdout transcript of the default run
  including the configuration line.

## Command

Source tree HEAD `41c4e1bc03cf5b44e4ba4047ffdff5f730e6da6c` plus the uncommitted
#41 working tree (the benchmark binary is built from that exact tree):

```bash
cmake -S . -B build/cpu41 -DGPUXTB_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu41 --target gpuxtb_scc_mixer_transaction_benchmark --parallel
./build/cpu41/gpuxtb_scc_mixer_transaction_benchmark 64 6 64 200 \
  benchmarks/evidence/issue-41/2026-08-06-epyc7k62/gpuxtb-mixer-transaction.json
./build/cpu41/gpuxtb_scc_mixer_transaction_benchmark 64 6 64 200 \
  benchmarks/evidence/issue-41/2026-08-06-epyc7k62/gpuxtb-mixer-transaction-full-range.json \
  "1,2,4,8,16,32,64"
```

Configuration: ragged batch of `batch=64` systems, each `atoms_per_system=6`
carbon atoms (GFN2 sp basis), `history=64`.
`state_size_bytes=4,461,952` (~4.3 MiB) and total mixer history
`2,162,688` doubles (16.5 MiB) are copied twice today only in the pre-change
full-state path. `new_transaction_bytes` is the exact per-system move size
(`3*dim` vector records + `2*dim*history` Broyden history + `history` omegas +
batch-scalar entries) times the active count. The measured loops allocate no
transient storage: raw-output scratch is hoisted and every seam returns the
checked status (a failure aborts the run).

## Results (median of 200 samples per row, microseconds)

Default rows (batch fixed at 64, active count varies):

| active | old_copy_us | copy_transaction_us | mix_transaction_us | old_copy_bytes | new_transaction_bytes |
| --- | --- | --- | --- | --- | --- |
| 8  | 309.628 |  32.954 |  2438.433 | 8,923,904 |   557,744 |
| 16 | 315.560 |  65.837 |  4887.605 | 8,923,904 | 1,115,488 |
| 32 | 341.400 | 134.229 |  9775.771 | 8,923,904 | 2,230,976 |
| 64 | 343.384 | 330.078 | 19599.073 | 8,923,904 | 4,461,952 |

Full active range 1..64 of the same fixed batch of 64:

| active | old_copy_us | copy_transaction_us | mix_transaction_us | old_copy_bytes | new_transaction_bytes |
| --- | --- | --- | --- | --- | --- |
| 1  | 280.512 |   3.897 |   303.516 | 8,923,904 |    69,718 |
| 2  | 260.664 |   7.886 |   607.234 | 8,923,904 |   139,436 |
| 4  | 257.237 |  15.850 |  1217.613 | 8,923,904 |   278,872 |
| 8  | 259.090 |  32.833 |  2437.540 | 8,923,904 |   557,744 |
| 16 | 283.338 |  65.446 |  4882.864 | 8,923,904 | 1,115,488 |
| 32 | 295.491 | 153.146 |  9782.130 | 8,923,904 | 2,230,976 |
| 64 | 271.805 | 354.856 | 19580.888 | 8,923,904 | 4,461,952 |

Interpretation:

- `old_copy_us` — the pre-change full mixer-state copy (two `state_size_bytes`
  memcpys) is **constant ~260-343 us per iteration** no matter how few systems
  are active; it always moves the whole 8,923,904-byte batch history. Small
  run-to-run variance comes from the shared cluster node.
- `copy_transaction_us` — the #41 per-system prepare+commit stage is linear in
  the active count (≈3.9-5.5 us/system): 3.9/7.9/15.9/32.8/65.4/153.1/354.9 us
  for 1/2/4/8/16/32/64 active of 64, and ~70x cheaper than the old full copy
  when only 1 of 64 systems is active.
- `mix_transaction_us` — full prepare+Broyden transition+commit is likewise
  linear in the active count (≈306 us/system), confirming neither the staging
  nor the mixing itself depends on the batch's inactive history.
- `new_transaction_bytes` — the #41 path moves exactly `active/total` of the
  batch history (69,718 / 8,923,904 = 1/128 when 1 of 64 systems is active).

## Measurement notes and limitations

- A large read/write "scrubber" buffer is touched before every measured round
  so the per-system transactions and the whole-state memcpy baseline are both
  timed cold in the same cache regime; the reported per-system costs therefore
  sit above a tiny hot-loop lower bound.
- Rows share one batch; every row keeps the same total history and changes only
  the active count. `mix` rows advance the per-system iteration counters; the
  Broyden history saturates at `history_size` transitions, and all rows start
  far beyond saturation (200 repetitions > 64), so the per-system mix cost is
  uniform within and across rows.
- The same `staged` binding and identical raw outputs drive every row, and the
  driver (not the benchmark) still owns where prepare/commit are called; this
  microbenchmark isolates the mixer transaction stage and does not time
  eigensolves, potentials, or the publication loop of a full SCC iteration.
- Raw samples are recorded in the JSON: 200 per row. This is development
  evidence for a single node, workload shape, and toolchain; it is not a
  release throughput claim.
