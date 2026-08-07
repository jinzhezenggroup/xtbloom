# Issue #41: transactional per-system SCC mixer update evidence

Measured on one AMD EPYC 7K62 node (48 logical CPUs, shared cluster node),
Ubuntu 22.04.5, `g++ 11.4.0`, `cmake 4.2.1`, CMake `Unix Makefiles` Release
configuration with `GPUXTB_ENABLE_CUDA=OFF` and no BLAS/LAPACK runtime (the
mixer layer is pure C++ and never calls the eigensolver provider). The rerun
was pinned to CPU 0 with one-thread BLAS environment variables.

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

The rerun used clean source revision
`4fc2f34651cf662037c1655b8d684746399511c0` (dirty bit false). The benchmark
binary SHA-256 is
`28f61b22a74414a0d629e65310154f92f8173a028ee9aeff76484af3d0363a7c`; the
corresponding `CMakeCache.txt` SHA-256 is
`94dc7e4c5ede6b9adfff858163aac7ccbfc355b5bd2e724d08f3076396c68a37`.

Build and run commands:

```bash
cmake -S . -B build/issue41-final -G 'Unix Makefiles' \
  -DGPUXTB_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/issue41-final --target gpuxtb_scc_mixer_transaction_benchmark --parallel
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  ./build/issue41-final/gpuxtb_scc_mixer_transaction_benchmark 64 6 64 200 \
  benchmarks/evidence/issue-41/2026-08-06-epyc7k62/gpuxtb-mixer-transaction.json
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  ./build/issue41-final/gpuxtb_scc_mixer_transaction_benchmark 64 6 64 200 \
  benchmarks/evidence/issue-41/2026-08-06-epyc7k62/gpuxtb-mixer-transaction-full-range.json \
  "1,2,4,8,16,32,64"
```

Configuration: ragged batch of `batch=64` systems, each `atoms_per_system=6`
carbon atoms (GFN2 sp basis), `history=64`.
`state_size_bytes=4,461,952` (~4.3 MiB) and total mixer history
`2,162,688` doubles (16.5 MiB) are copied twice in the pre-change full-state
path. `per_system_copy_bytes=69,718` is one prepare or commit direction;
`per_system_transaction_bytes=139,436` counts both directions. The row's
`new_transaction_bytes` is the latter multiplied by the active count, matching
the timed prepare+commit path. The measured loops allocate no transient
storage: raw-output scratch is hoisted and every seam returns the checked
status (a failure aborts the run).

## Results (median of 200 samples per row, microseconds)

Default rows (batch fixed at 64, active count varies):

| active | old_copy_us | copy_transaction_us | mix_transaction_us | old_copy_bytes | new_transaction_bytes |
| --- | --- | --- | --- | --- | --- |
| 8  | 249.537 |  32.823 |  2466.948 | 8,923,904 | 1,115,488 |
| 16 | 249.177 |  65.626 |  4944.557 | 8,923,904 | 2,230,976 |
| 32 | 249.738 | 131.431 |  9896.007 | 8,923,904 | 4,461,952 |
| 64 | 249.086 | 291.798 | 19813.144 | 8,923,904 | 8,923,904 |

Full active range 1..64 of the same fixed batch of 64:

| active | old_copy_us | copy_transaction_us | mix_transaction_us | old_copy_bytes | new_transaction_bytes |
| --- | --- | --- | --- | --- | --- |
| 1  | 245.149 |   4.138 |   307.087 | 8,923,904 |   139,436 |
| 2  | 244.247 |   8.226 |   616.790 | 8,923,904 |   278,872 |
| 4  | 253.215 |  16.331 |  1234.421 | 8,923,904 |   557,744 |
| 8  | 244.488 |  32.773 |  2466.848 | 8,923,904 | 1,115,488 |
| 16 | 244.077 |  65.295 |  4934.959 | 8,923,904 | 2,230,976 |
| 32 | 245.589 | 130.780 |  9894.093 | 8,923,904 | 4,461,952 |
| 64 | 244.528 | 274.505 | 19789.139 | 8,923,904 | 8,923,904 |

Interpretation:

- `old_copy_us` — the pre-change full mixer-state copy (two `state_size_bytes`
  memcpys) is **constant ~244-253 us per iteration** no matter how few systems
  are active; it always moves the whole 8,923,904-byte batch history. Small
  run-to-run variance comes from the shared cluster node.
- `copy_transaction_us` — the #41 per-system prepare+commit stage is linear in
  the active count (≈4.1-4.4 us/system): 4.1/8.2/16.3/32.8/65.3/130.8/274.5 us
  for 1/2/4/8/16/32/64 active of 64, and ~62x cheaper than the old full copy
  when only 1 of 64 systems is active.
- `mix_transaction_us` — full prepare+Broyden transition+commit is likewise
  linear in the active count (≈306 us/system), confirming neither the staging
  nor the mixing itself depends on the batch's inactive history.
- `new_transaction_bytes` — the #41 path moves exactly `active/total` of the
  batch history (139,436 / 8,923,904 = 1/64 when 1 of 64 systems is active).

## Measurement notes and limitations

- A read scrubber buffer is touched before every measured round to perturb the
  cache consistently for the per-system transactions and whole-state baseline;
  this is not a claim of a complete cache flush.
- Rows share one batch; every row keeps the same total history and changes only
  the active count. The benchmark warms every system through `history_size`
  transitions before timing, so all `mix` rows start in the saturated Broyden
  regime; timed rows then advance the per-system iteration counters further.
- The same `staged` binding and identical raw outputs drive every row, and the
  driver (not the benchmark) still owns where prepare/commit are called; this
  microbenchmark isolates the mixer transaction stage and does not time
  eigensolves, potentials, or the publication loop of a full SCC iteration.
- Raw samples are recorded in the JSON: 200 per row. This is development
  evidence for a single node, workload shape, and toolchain; it is not a
  release throughput claim.
