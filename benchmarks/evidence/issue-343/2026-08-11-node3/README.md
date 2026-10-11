# Issue #343 corrected Phase-1 SCC-reuse evidence

This bundle supersedes the scientifically invalid evidence previously stored
at this path. It was generated on node3 from clean committed revision
`5e406d0ac391c3ae9a4017e1cdaa4babf6c9e1fb`; the capture executable has
SHA-256
`9d486b96852e11aac499b00336ef7380c9225e9c10421e76b856f6c0c22b952f`.
The complete build and provider identity is in `build-metadata.txt`.

The corrected evidence changes four material parts of the earlier result:

- dodecane is a validated linear C12H26 molecule (38 atoms, 74 AOs), not the
  malformed C12H14 structure;
- physical trajectory angles use the actual cross-geometry AO overlap
  `S12 = <chi(R1)|chi(R2)>`;
- the algorithmic trajectory residual uses a co-moving AO-label transport,
  target-metric reorthonormalization, and the target terminal effective SCC
  Hamiltonian;
- the warm-start iteration comparison uses WARM and FRESH runs of the exact
  same target geometry and SCC policy.

Unrestricted inputs are rejected by this Phase-1 tool. No unrestricted metric
or CUDA performance conclusion is included.

## Correctness qualification

Before capture, the exact committed revision passed a fresh shared CPU Release
build using the host-isolated LP64 MKL provider:

```text
ctest --test-dir build/issue343-evidence-cpu --output-on-failure
100% tests passed, 0 tests failed out of 55
```

The registered set included `xtbloom.gfn2.scc_reuse_smoke`, public CPU
inference, ABI symbols, CPU conformance and invariants, licensing, and the SCC
oracle corpus. Across all retained captures, the maximum observed
`|C^T S C - I|` was `3.997e-15`; the maximum normalized elementwise
generalized-eigenpair residual was `7.263e-16`.

`prepare_cases.py` reproduced every committed repository-local case byte for
byte. Its alkane validator requires exactly 12 carbon and 26 hydrogen atoms,
one linear carbon chain, and exactly one carbon neighbor for every hydrogen.
The smoke test additionally checks that an identical-geometry cross overlap
reproduces `S`, that the WARM/FRESH target states agree, that SCC-policy changes
are rejected, and that unrestricted inputs are rejected.

## Corrected single-geometry result

`first` columns give the SCC iteration number at which the threshold is first
observed. The eigensolve share is one development capture, retained only to
show where work occurs; it is not a stable latency distribution or speedup
claim.

| Case | AOs | Iterations | Converged | first angle < 1 deg | first capture >= 0.9999 | first RR error < 1e-6 Eh | eigensolve/step |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| h3_plus | 3 | 3 | yes | 2 | 2 | 2 | 0.097 |
| ketene | 14 | 13 | yes | 6 | 6 | 9 | 0.395 |
| nenacl | 22 | 14 | yes | 6 | 6 | 10 | 0.544 |
| water dimer, 6 point charges | 12 | 9 | yes | 5 | 5 | 6 | 0.324 |
| water, one point charge | 6 | 9 | yes | 5 | 5 | 6 | 0.216 |
| benzene | 30 | 7 | yes | 2 | 2 | 5 | 0.449 |
| pyridine | 29 | 10 | yes | 6 | 5 | 8 | 0.502 |
| dodecane, C12H26 | 74 | 8 | yes | 2 | 3 | 5 | 0.560 |
| TMAC/Cl sloshing case | 41 | 100 | no | 2 | 4 | 4 (intermittent) | 0.629 |

The corrected C12H26 case does not reproduce the old 17-iteration behavior:
it converges in 8 iterations. Its minimum consecutive-iteration occupied
capture is `0.999892`, maximum angle is `0.966 deg`, and maximum occupied-space
Rayleigh--Ritz eigenvalue error is `1.404e-4 Eh`.

TMAC/Cl remains the difficult control. It does not converge in 100 iterations,
its minimum capture is `0.952363` (approximately one of 21 occupied orbitals),
its maximum angle is `2.404 deg`, and its maximum RR eigenvalue error is
`6.672e-2 Eh`. Repeated frontier changes therefore remain a mandatory
full-solve fallback signal for any Phase-2 prototype.

## Corrected trajectory result

The C12H26 source and perturbed target endpoints give:

| Metric | Value |
| --- | ---: |
| physical occupied-subspace capture using `S12` | 0.9997361934 |
| physical maximum principal angle | 2.013685 deg |
| relative density-operator change using `S1/S2/S12/S21` | 0.02296983 |
| target-metric RR occupied residual | 1.734479e-3 |
| target-metric RR maximum eigenvalue error | 1.615820e-5 Eh |
| target-metric RR capture of target occupied space | 0.9999877511 |
| WARM target iterations | 6 |
| same-target FRESH iterations | 8 |
| final WARM/FRESH relative density difference | 4.178091e-6 |
| final WARM/FRESH maximum occupied-space angle | 0.0002224 deg |

This one controlled trajectory supports only the narrow statement that WARM
used two fewer SCC iterations than FRESH for this exact target. It is not an
MD-wide or optimization-wide iteration claim.

## Conclusion

Phase 1 still supports prototyping a guarded restricted recycled eigensolver:
the converged cases acquire strong occupied-subspace capture before final SCC
convergence, while the nonconvergent TMAC/Cl case exposes a clear
frontier-switch failure mode. It does not establish an end-to-end speedup.
Phase 2 still requires a robust full-solve fallback and correctness-qualified
CUDA timing on representative workloads.

## Reproduction

The environment and exact input/raw/report hashes are recorded in
`build-metadata.txt` and `CAPTURES.tsv`. The build used:

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple \
  uv sync --locked --no-install-project --no-default-groups --group wheel-build

cmake -S . -B build/issue343-evidence-cpu -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/miniconda3/lib/libmkl_rt.so.3 \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/issue343-evidence-cpu --parallel
ctest --test-dir build/issue343-evidence-cpu --output-on-failure
```

Repository-local inputs were regenerated into a temporary directory and
required to match the committed files:

```bash
.venv/bin/python tools/scc_reuse/prepare_cases.py \
  --out build/issue343-evidence-run-5e406d0/generated-cases
diff -ru tools/scc_reuse/cases \
  build/issue343-evidence-run-5e406d0/generated-cases
```

Each single case used this pattern:

```bash
build/issue343-evidence-cpu/xtbloom_scc_reuse_capture \
  single <input.spec> build/issue343-evidence-run-5e406d0/raw/<case>.diag
.venv/bin/python tools/scc_reuse/scc_reuse_analyze.py \
  build/issue343-evidence-run-5e406d0/raw/<case>.diag \
  --report build/issue343-evidence-run-5e406d0/reports/<case>.json --quiet
```

The trajectory command was:

```bash
build/issue343-evidence-cpu/xtbloom_scc_reuse_capture traj \
  tools/scc_reuse/cases/dodecane_traj1.spec \
  tools/scc_reuse/cases/dodecane_traj2.spec \
  build/issue343-evidence-run-5e406d0/raw/dodecane_traj.diag
.venv/bin/python tools/scc_reuse/scc_reuse_analyze.py \
  build/issue343-evidence-run-5e406d0/raw/dodecane_traj.diag \
  --report build/issue343-evidence-run-5e406d0/reports/dodecane_traj.json \
  --quiet
```

The large line-oriented matrix captures are reproducible intermediates and are
intentionally omitted under `benchmarks/evidence/README.md`; their byte counts
and SHA-256 digests remain in `CAPTURES.tsv`, and every compact report embeds
the digest of the raw stream it analyzed. The reports retain all per-iteration
metrics and the one timing observation from every SCC step. No external
artifact is required for the claims above.

`SHA256SUMS` covers every retained file except itself and was verified with
`sha256sum -c SHA256SUMS` after the bundle was finalized.
