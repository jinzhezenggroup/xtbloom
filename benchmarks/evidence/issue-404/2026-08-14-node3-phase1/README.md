# Issue #404 Phase 1: adaptive SCC prototype is a research no-go

## Decision

The default-off CPU/FP64 prototype is numerically bounded and can recover the
300 K TMAC/Cl fixed point that the baseline does not reach in 1000 iterations.
It does **not** meet the ordinary-workload iteration or end-to-end latency
targets and is not ready to become a production default or a JCTC-level core
method.

- `controller` leaves ordinary SCC iteration counts unchanged while increasing
  measured median latency by 1.75% to 8.44%.
- `local-v1` reduces ketene from 12 to 11 iterations and latency by 3.52% to
  3.73%, but slows every measured QM/MM and heterogeneous coordinate. The
  heterogeneous gas batch regresses from 14 to 16 iterations and by 8.64% in
  median latency.
- On TMAC/Cl, baseline `off` fails at both 250 and 1000 iterations, while
  `controller` converges in 51 and `local-v1` in 44 iterations.
- The recovered TMAC/Cl point is locally fragile: a 0.001 bohr positive
  displacement of one coordinate fails to converge even with a 1000-iteration
  finite-difference ceiling under each experimental policy. Therefore a full
  difficult-state force finite-difference gate is **not passed**.

This is a useful negative result: generic controller safeguards can rescue one
charge-sloshing case, but the empirical local diagonal scaling is neither
uniformly faster nor robust enough to support a broad acceleration claim.

## Source, binary, and machine identity

- Measured clean source revision:
  `645422264815f4d70fe7cc9928e4290bdcb5efc1`
- Branch: `perf/404-pairs-scc`
- Library: `build/issue404-clean-cpu/libxtbloom.so.0.1.1`
- Library SHA-256:
  `6d0cf92d9420a32431293919f47a48d1e4704469f1953119c2d1253e9b30f99f`
- CMake cache SHA-256:
  `ecccf91f6abff17cbfc0078ecd1363632c709148f443a19f5a91c74e0a692469`
- LP64 SciPy OpenBLAS provider:
  `/home/jzzeng/miniconda3/lib/python3.13/site-packages/scipy_openblas32/lib/libscipy_openblas.so`
- Provider SHA-256:
  `b2dfe24b9aa11cf1d1cec8edbca9423b50cfd186b486d59dd4efe45826261a98`
- Compiler/build: GCC 11.4.0, CMake 4.2.1, Ninja 1.13, Release,
  shared library, CUDA disabled.
- Host: `node3`, AMD EPYC 7K62, 48 physical cores, one thread per core.
- Measurement affinity: logical CPU 0 only.
- Thread contract: `cpu_threads=1`, `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `GOMP_CPU_AFFINITY=0`.

The JSON metadata records the clean revision, selected-library path and hash,
hardware, Python runtime, workload identity, correctness result, raw timing
samples, SCC iteration range, and memory snapshots. The outer `taskset` and
`XTBLOOM_EXPERIMENTAL_GFN2_PAIRS_SCC` values are not captured by the generic
harness and are therefore pinned explicitly here.

The three CSV artifacts retain every generated field and sample but normalize
the writer's CRLF record terminators to LF so repository whitespace checks
remain clean; the JSON artifacts are the authoritative raw-sample records.

## Ordinary public-API protocol and result

The repository `benchmarks/run.py` public C-API adapter used one persistent CPU
context. Every timed compute retained `SCC_START_FRESH`; `warm` in the JSON and
CSV means a sample collected after ten harness warmups, not a WARM SCC start.
The requested property was energy, every coordinate used 31 measured samples,
and every retained row passed the committed independent-golden correctness
gate.

| Workload | Batch | Off median ms / iter | Controller median ms / iter | Controller delta | Local-v1 median ms / iter | Local-v1 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gas | 1 | 0.9120 / 12 | 0.9522 / 12 | +4.40% | 0.8763 / 11 | -3.92% |
| gas | 8 | 7.2249 / 12 | 7.5051 / 12 | +3.88% | 6.9217 / 11 | -4.20% |
| QM/MM | 1 | 0.6833 / 11 | 0.6987 / 11 | +2.25% | 0.7235 / 11 | +5.88% |
| QM/MM | 8 | 5.4071 / 11 | 5.5019 / 11 | +1.75% | 5.7147 / 11 | +5.69% |
| heterogeneous gas | 1 | 0.0814 / 4 | 0.0883 / 4 | +8.44% | 0.0893 / 4 | +9.62% |
| heterogeneous gas | 8 | 10.6421 / 14 | 10.8674 / 14 | +2.12% | 11.5617 / 16 | +8.64% |
| heterogeneous QM/MM | 1 | 0.2849 / 10 | 0.3054 / 10 | +7.21% | 0.3054 / 10 | +7.21% |
| heterogeneous QM/MM | 8 | 4.2332 / 11 | 4.3660 / 11 | +3.14% | 4.5102 / 11 | +6.54% |

The exact command was repeated with `<policy>` equal to `off`, `controller`,
and `local-v1`, and with the matching output names:

```bash
taskset -c 0 env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  GOMP_CPU_AFFINITY=0 \
  CUDA_VISIBLE_DEVICES= \
  XTBLOOM_EXPERIMENTAL_GFN2_PAIRS_SCC=<policy> \
  python benchmarks/run.py \
    --library build/issue404-clean-cpu/libxtbloom.so \
    --engines xtbloom \
    --backends cpu \
    --workloads gas,qmmm,heterogeneous-gas,heterogeneous-qmmm \
    --properties energy \
    --batch-sizes 1,8 \
    --warmups 10 \
    --repetitions 31 \
    --cpu-threads 1 \
    --output-json build/issue404-pairs-pinned-<policy>.json \
    --output-csv build/issue404-pairs-pinned-<policy>.csv \
    --fail-on-correctness
```

## TMAC/Cl public-ABI result

The exact 18-atom fixture is
`data/conformance/inputs/tmacl.xyz`, SHA-256
`7e5b80ecc7a8058d91d08afce9f5b12ec1413417c2a1defe819523b33e21ca74`.
It is neutral, restricted, FRESH, 300 K, with charge tolerance `1e-6` and
energy tolerance `1e-8` Hartree. The issue-scoped `tmacl_public.py` runner
uses the installed high-level wrapper over the selected public C ABI and
preserves per-system status, NaN publication, energy, atom-resolved charges,
and analytic forces. Its SHA-256 before archival was
  is pinned by `SHA256SUMS`.

| Policy | Ceiling | Status | Iterations | Total energy (Eh) | Me4N+ / Cl charge (e) |
| --- | ---: | --- | ---: | ---: | ---: |
| off | 250 | SCC not converged | 250 | NaN publication | NaN publication |
| off | 1000 | SCC not converged | 1000 | NaN publication | NaN publication |
| controller | 250 | success | 51 | -22.012346170801433 | +0.8285029051 / -0.8285029051 |
| local-v1 | 250 | success | 44 | -22.012346170813405 | +0.8284959207 / -0.8284959207 |

The two successful policies reach numerically matching states:

- energy difference: `1.1973e-11 Eh`;
- maximum atom-charge difference: `6.9844e-6 e`;
- maximum analytic-force difference: `1.0601e-6 Eh/bohr`;
- controller net-force norm: `9.61e-17 Eh/bohr`;
- local-v1 net-force norm: `1.75e-16 Eh/bohr`;
- controller torque norm: `2.11e-11 Eh`;
- local-v1 torque norm: `1.86e-12 Eh`.

The force finite-difference attempt used a central step of `0.001 bohr`.
`controller` failed at zero-based flat coordinate 12, positive displacement;
`local-v1` failed at coordinate 0, positive displacement. Both failures remain
`SCC_NOT_CONVERGED` at ceilings of 250 and 1000 iterations. Analytic-force
translation/rotation invariants are strong at the recovered undisplaced point,
but the complete finite-difference comparison is unavailable because the
displaced electronic states were not recovered.

The final retained JSON files were generated from the clean detached worktree
`/tmp/xtbloom-issue404-clean.zeGTp4`. The archived runner was copied
byte-identically to the ignored `build/tmacl_public.py` path so the measured
source remained clean. Base command, repeated over the four policy/ceiling
rows:

```bash
issue404_root=/tmp/xtbloom-issue404-clean.zeGTp4
taskset -c 0 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  GOMP_CPU_AFFINITY=0 CUDA_VISIBLE_DEVICES= \
  PYTHONPATH="$issue404_root/python" \
  /tmp/xtbloom-issue404-venv/bin/python \
    "$issue404_root/build/tmacl_public.py" \
    --repository "$issue404_root" \
    --library "$issue404_root/build/issue404-clean-cpu/libxtbloom.so" \
    --input "$issue404_root/data/conformance/inputs/tmacl.xyz" \
    --policy <policy> \
    --max-iterations <250-or-1000> \
    --output <output.json>
```

The finite-difference invocations additionally used:

```text
--finite-difference-step 0.001 --finite-difference-max-iterations <250-or-1000>
```

## Correctness and validation

The measured library was built from the exact clean revision with:

```bash
cmake -S . -B build/issue404-clean-cpu -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/miniconda3/lib/python3.13/site-packages/scipy_openblas32/lib/libscipy_openblas.so \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/issue404-clean-cpu --parallel
ctest --test-dir build/issue404-clean-cpu -N
ctest --test-dir build/issue404-clean-cpu --output-on-failure
```

Registration and execution result: 90 tests registered; 90 passed, zero
failed, zero skipped. Both experimental policies also passed the focused
ordinary public energy/force/charge and invariant matrix:

```bash
env XTBLOOM_EXPERIMENTAL_GFN2_PAIRS_SCC=<controller-or-local-v1> \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ctest --test-dir build/issue404-clean-cpu \
    -R '^(xtbloom\.cpu\.public_inference|xtbloom\.conformance\.public_cpu|xtbloom\.conformance\.invariants_cpu)$' \
    --output-on-failure
```

Result for each policy: 3/3 passed. The final source also passed the canonical
nox session, repository-wide `prek`, `uv lock --check`, and `git diff --check`.

## Acceptance decision and remaining gaps

| Target | Result | Decision |
| --- | --- | --- |
| Median ordinary FRESH iterations -25% | Best ordinary reduction is 12 to 11 on ketene | FAIL |
| Weighted end-to-end FRESH latency -15% | Best row is -4.20%; most rows regress | FAIL |
| Ordinary p95 iteration regression <=5% | Local-v1 heterogeneous gas rises 14 to 16 (+14.3%) | FAIL |
| Difficult-set success +20 percentage points | One TMAC/Cl rescue, no broad difficult holdout | UNVERIFIED |
| Difficult-state force validation | Displaced states fail even at 1000 iterations | FAIL/INCOMPLETE |

This bundle does not provide WARM-start timing, batches above eight, a broad
difficult/open-shell/periodic holdout, an independent oracle for the recovered
TMAC/Cl state, CUDA implementation/parity, or evidence for a physics-derived
pair-response Jacobian. Those remain required before a new method claim or
JCTC submission can be supported.

## Files

- `off.json` / `off.csv`: default/baseline ordinary public-API samples.
- `controller.json` / `controller.csv`: controller-only ordinary samples.
- `local-v1.json` / `local-v1.csv`: controller plus empirical local scaling.
- `issue404-tmacl-*.json`: clean public-ABI difficult-case and failed
  finite-difference attempts.
- `tmacl_public.py`: exact issue-scoped public-ABI runner.
- `SHA256SUMS`: hashes for every retained artifact except the hash file itself.
