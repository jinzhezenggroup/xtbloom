# Issue #407 Phase 2: fixed pair-response SCC candidate is a research no-go

## Decision

The CPU/FP64, internal, default-off `pair-response-v1` prototype is a bounded,
constraint-preserving implementation of the frozen issue #407 experiment. It
passes the applicable implementation, public-science, transaction, cache, and
correctness gates, but it does **not** meet the difficult-neighborhood,
ordinary-iteration, or end-to-end latency thresholds. This candidate is not
recommended for production merge or as the core numerical-method claim of a
JCTC submission.

The negative result is deliberately narrow. It rejects the frozen
`alpha = 0.08` QEq-style response scale used here; it does not reject the wider
pair-response family. The next scientifically useful experiment should
estimate the screened electronic susceptibility or SCC Jacobian actually seen
by the fixed-point map instead of trying another fixed hardness scale.

## Implemented method and preserved semantics

The experiment is selected only by:

```text
XTBLOOM_EXPERIMENTAL_GFN2_PAIRS_SCC=pair-response-v1
```

For shell `s`, it defines the frozen diagonal response scale and pair operator

```text
G_ss = gamma_s / 0.08
H = G + K_pair
```

where `gamma_s` is the canonical GFN2 element hardness times the shell Hubbard
scale. `K_pair` uses the production ES2 shell coupling only between different
atoms; same-atom blocks are zero. A Cholesky factorization and constrained
solve transform only the shell-charge residual while preserving exact total
charge. Magnetization, dipole, and quadrupole residuals remain unchanged.

This is a QEq-style numerical proxy, not a derived GFN2 electronic
susceptibility. AES2 has no charge-charge Hessian when its density-derived
dipole and quadrupole channels are held fixed, so it is not inserted into this
charge-only block. A caller-owned potential shift `b` does not change the
response Jacobian. A supplied periodic response matrix `A` does; because that
term is not implemented in the proxy, `A` deterministically disables pair
response and selects the established fallback.

The prototype does not change the public C/Python ABI, CUDA backend,
conformance goldens, GFN2 fixed point, raw binary64 residual/free-energy
convergence gates, Helmholtz free energy, force meaning, or publication
semantics. It adds no eigensolve or model-energy evaluation and no
steady-state per-call allocation. Fixed-topology cache validation, strict WARM
identity, per-system ragged isolation, Broyden/controller restart hygiene, and
resident workspace accounting are covered by focused tests.

## Source, binary, build, and machine identity

- Measured clean source revision:
  `f10549511cc16a16650e39ad56aec094aa3fff12`
- Branch: `perf/407-pair-response-scc`
- Library: `build/issue407-clean-cpu/libxtbloom.so.0.1.1`
- Library SHA-256:
  `5109d6a49e4e00ccd7020c5db98b1755e8d6e96d2f8343d41222bb12d97696be`
- CMake cache SHA-256:
  `77f8e69fe16d4d2e5c40423febb9bf0948811d68c5a7dd9b41ea4d23da0170b3`
- LP64 SciPy OpenBLAS provider:
  `/home/jzzeng/miniconda3/lib/python3.13/site-packages/scipy_openblas32/lib/libscipy_openblas.so`
- Provider SHA-256:
  `b2dfe24b9aa11cf1d1cec8edbca9423b50cfd186b486d59dd4efe45826261a98`
- Compiler/build: GCC 11.4.0, CMake 4.2.1, Ninja 1.13, Release,
  shared library, CUDA disabled.
- Installed package: `0.1.1.post1.dev38+gf10549511`.
- Host: `node3`, AMD EPYC 7K62, 48 logical CPUs.
- Performance affinity: logical CPU 0 only.
- Thread contract: `cpu_threads=1`, `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `GOMP_CPU_AFFINITY=0`.

The final generated artifacts used the clean detached worktree
`/tmp/xtbloom-issue407-clean.pIjFDO/repo`. The validated library was copied
byte-identically into that worktree before measurement; its SHA-256 remained
the value above. This keeps both the source status and the selected binary
identity auditable without modifying generated metadata after measurement.

The performance JSON records the clean source and selected-library identities,
hardware, workload, raw samples, SCC iteration range, correctness result, and
memory snapshots. The generic harness does not record the outer `taskset` or
experimental-policy environment values, so they are pinned below. The CSV
files retain all generated fields and samples but were mechanically normalized
from CRLF to LF; the JSON files are the authoritative raw-sample artifacts.

## Ordinary public-API performance

The repository `benchmarks/run.py` adapter used one persistent CPU context,
the public energy-only API, and `SCC_START_FRESH` on every timed compute. The
word `warm` in the JSON/CSV means a sample collected after ten harness warmups;
it does not mean the public WARM SCC start mode. Every coordinate used 31
measured samples and passed the committed independent-golden correctness gate.

| Workload | Batch | Off median ms / iter range | Pair median ms / iter range | Median delta |
| --- | ---: | ---: | ---: | ---: |
| gas | 1 | 0.913718 / 12 | 0.962723 / 12 | +5.36% |
| gas | 8 | 7.240262 / 12 | 7.617141 / 12 | +5.21% |
| QM/MM | 1 | 0.682872 / 11 | 0.674585 / 10 | -1.21% |
| QM/MM | 8 | 5.410511 / 11 | 5.329814 / 10 | -1.49% |
| heterogeneous gas | 1 | 0.083131 / 4 | 0.094964 / 4 | +14.23% |
| heterogeneous gas | 8 | 10.633178 / 4-14 | 11.019025 / 4-14 | +3.63% |
| heterogeneous QM/MM | 1 | 0.286805 / 10 | 0.318216 / 10 | +10.95% |
| heterogeneous QM/MM | 8 | 4.256127 / 10-11 | 4.304360 / 10 | +1.13% |

The following command was repeated with `<policy>` equal to `off` and
`pair-response-v1`, and with matching output names:

```bash
taskset -c 0 env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  GOMP_CPU_AFFINITY=0 \
  CUDA_VISIBLE_DEVICES= \
  XTBLOOM_EXPERIMENTAL_GFN2_PAIRS_SCC=<policy> \
  python benchmarks/run.py \
    --library build/issue407-clean-cpu/libxtbloom.so \
    --engines xtbloom \
    --backends cpu \
    --workloads gas,qmmm,heterogeneous-gas,heterogeneous-qmmm \
    --properties energy \
    --batch-sizes 1,8 \
    --warmups 10 \
    --repetitions 31 \
    --cpu-threads 1 \
    --output-json build/issue407-evidence/<policy>.json \
    --output-csv build/issue407-evidence/<policy>.csv \
    --fail-on-correctness
```

The acceptance thresholds are not close to being met. Median ordinary
iterations do not improve by 20%, and no declared row reaches the required 10%
median latency improvement or the 15% target. The p95 iteration-regression
bound does pass because the maximum iteration count does not increase.

## TMAC/Cl difficult-neighborhood gate

The exact 18-atom fixture is `data/conformance/inputs/tmacl.xyz`, SHA-256
`7e5b80ecc7a8058d91d08afce9f5b12ec1413417c2a1defe819523b33e21ca74`.
It is neutral, restricted, FRESH, 300 K, with charge tolerance `1e-6` and
energy tolerance `1e-8` Hartree.

| Policy | Ceiling | Result | Iterations |
| --- | ---: | --- | ---: |
| off | 250 | `SCC_NOT_CONVERGED` | 250 |
| off | 1000 | `SCC_NOT_CONVERGED` | 1000 |
| pair-response-v1 | 250 | `SCC_NOT_CONVERGED` | 250 |
| pair-response-v1 | 1000 | `SCC_NOT_CONVERGED` | 1000 |

All failed-system energy, charge, and force slices are correctly published as
quiet NaNs. Because the frozen policy does not recover even the undisplaced
reference state, the required +/-0.001 bohr neighborhood and complete
difficult-state force finite differences cannot proceed. This is the direct
scientific no-go gate for the candidate.

The four retained results can be reproduced with:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python \
    benchmarks/evidence/issue-407/2026-08-14-node3-phase2/tmacl_probe.py \
    --repository "$PWD" \
    --library "$PWD/build/issue407-clean-cpu/libxtbloom.so" \
    --input "$PWD/data/conformance/inputs/tmacl.xyz" \
    --policy <off-or-pair-response-v1> \
    --max-iterations <250-or-1000> \
    --output "$PWD/build/issue407-evidence/tmacl-<policy>-<ceiling>.json"
```

## Descriptive susceptibility diagnostic

`susceptibility_probe.py` finite-differences the fully reconverged public
atomic charges with respect to zero-sum caller-owned atomic potential shifts
`b`, using central steps `1e-3`, `3e-4`, `1e-4`, and `3e-5` Hartree. All five
case/direction rows converge with finite values at every step. At the smallest
step, the norm of the actual public `dq/db` is 9.1003 to 21.5854 times the bare
diagonal QEq proxy norm; the tangent directions are collinear for these small
two- and three-atom probes.

This ratio is descriptive only. The public reconverged derivative includes
band-structure, ES2, AES2, and other self-consistent screening. It is not the
bare proxy and the script also does not evaluate the implemented screened pair
operator. Therefore neither the scale mismatch nor the direction cosine is
used to reject this candidate or the broader response family. The decisive
no-go evidence is the frozen-policy TMAC/Cl and declared performance matrix.

Reproduction command:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python \
    benchmarks/evidence/issue-407/2026-08-14-node3-phase2/susceptibility_probe.py \
    --repository "$PWD" \
    --library "$PWD/build/issue407-clean-cpu/libxtbloom.so" \
    --output "$PWD/build/issue407-evidence/susceptibility.json" \
    --steps 1e-3 3e-4 1e-4 3e-5
```

## Correctness and validation

The measured library was built from the exact clean revision with:

```bash
cmake -S . -B build/issue407-clean-cpu -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/miniconda3/lib/python3.13/site-packages/scipy_openblas32/lib/libscipy_openblas.so \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/issue407-clean-cpu --parallel
ctest --test-dir build/issue407-clean-cpu -N
ctest --test-dir build/issue407-clean-cpu --output-on-failure
```

Registration and execution result: 90 tests registered; 90 passed, zero
failed, zero skipped. The experimental policy also passed the ordinary public
inference, conformance, and invariant matrix:

```bash
env XTBLOOM_EXPERIMENTAL_GFN2_PAIRS_SCC=pair-response-v1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ctest --test-dir build/issue407-clean-cpu \
    -R '^(xtbloom\.cpu\.public_inference|xtbloom\.conformance\.public_cpu|xtbloom\.conformance\.invariants_cpu)$' \
    --output-on-failure
```

Result: 3/3 passed. The same clean head also passed:

- canonical nox: 122 tests passed and three documented optional checks skipped;
- repository-wide `prek@0.3.1`;
- `uv lock --check`;
- `git diff --check`;
- benchmark tests: 120 passed with one optional plotting skip in the locked
  non-editable environment, plus 9/9 dxtb adapter tests in the available
  PyTorch environment;
- independent final review: no blocker, high, or medium findings.

## Acceptance ledger and remaining evidence

| Criterion | Result | Status |
| --- | --- | --- |
| Production-term derivation, units, signs, and exclusions documented | Direct ES2/AES2 audit and term-level tests | PASS |
| Charge constraint, permutation covariance, and omitted-channel identity | Focused preconditioner and SCC tests | PASS |
| Bounded solve and deterministic fallback | Conditioning, non-finite, one-atom, disconnected, and periodic-`A` tests | PASS |
| No extra eigensolve/evaluation/allocation | Instrumentation and exact workspace accounting | PASS |
| Ragged transaction, cache, Broyden, and WARM semantics | Public/focused adversarial tests | PASS |
| Ordinary public science and independent goldens | Public inference/conformance/invariants and harness correctness | PASS |
| TMAC/Cl reference and +/-0.001 bohr neighborhood | Reference remains nonconvergent at 1000 iterations | FAIL |
| Median ordinary iterations improve at least 20% | Best rows improve by one iteration; most are unchanged | FAIL |
| p95 iteration regression no more than 5% | Maximum iteration ranges do not increase | PASS |
| Median public FRESH latency improves at least 10%, target 15% | Best row is -1.49%; most rows regress | FAIL |
| Wider difficult/open-shell/QM/MM/periodic/ragged holdout | Ordinary coverage only; difficult and larger-batch matrix incomplete | UNVERIFIED |
| Honest go/no-go with archived distributions | Raw compact evidence and narrow conclusion retained | PASS |

This bundle does not provide WARM-start timing, batches above eight, a broad
difficult/open-shell/periodic holdout, CUDA implementation/parity, or a
complete screened electronic response Jacobian. These gaps are not converted
to passes. They remain prerequisites for any later acceleration or JCTC-level
method claim.

## Why the next direction is worth pursuing

The implementation evidence shows that a constrained non-diagonal response
map can be integrated safely without changing public scientific semantics or
adding an eigensolve. The failure is instead in the physical scale of the
frozen proxy: `gamma/0.08` is a hardness-based ansatz, whereas the SCC map sees
the screened derivative of Mulliken charges and multipoles after electronic
reoccupation and all self-consistent terms. Tuning another constant would
likely move the same mismatch rather than explain it.

A stronger next experiment is therefore to estimate a low-rank or block-sparse
screened Jacobian from existing SCC iterates, constrained to exact charge and
combined with the known ES2 pair topology. It is worth doing because it targets
the missing quantity identified by this no-go while retaining the successful
engineering properties of the prototype: per-system isolation, deterministic
fallback, no public ABI change, and no extra model evaluation. It should be
attempted only with a predeclared difficult/open-shell/QM/MM/periodic holdout
and the same correctness-qualified latency gate.

## Files

- `off.json` / `off.csv`: baseline ordinary public-API raw samples.
- `pair-response-v1.json` / `pair-response-v1.csv`: candidate ordinary
  public-API raw samples.
- `tmacl-*.json`: four frozen-policy TMAC/Cl public-ABI results.
- `susceptibility.json`: descriptive public `dq/db` finite differences.
- `tmacl_probe.py` and `susceptibility_probe.py`: exact issue-scoped runners.
- `SHA256SUMS`: hashes for every retained artifact except the hash file.
