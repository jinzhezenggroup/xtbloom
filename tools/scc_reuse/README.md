# SCC subspace-reuse diagnostics (issue #343, Phase 1)

Experimental repository-only tooling for
[issue #343](https://github.com/jinzhezenggroup/xtbloom/issues/343). It
measures whether an occupied eigenspace from one SCC iteration or geometry can
seed the next generalized eigensolve. Nothing in this directory is public ABI,
an installation payload, a conformance golden, or part of the versioned
`xtbloom-scc-trace-v1` contract.

The current Phase-1 capture is deliberately **restricted-SCC only**. Inputs
with nonzero unpaired electrons are rejected before execution. Extending the
tool to unrestricted SCC requires spin-major Hamiltonian and density capture;
silently analyzing both spin channels against one matrix is invalid.

## Components

- `scc_reuse_capture.cpp` builds the internal
  `xtbloom_scc_reuse_capture` executable. It drives the production CPU GFN2
  SCC driver and records the effective Hamiltonian, generalized eigenpairs,
  overlap, density, and isolated eigensolve timing at every completed
  iteration.
- `scc_reuse_analyze.py` validates the eigenpairs and produces compact JSON
  metrics.
- `prepare_cases.py` generates analytic benzene/pyridine inputs, a validated
  all-trans C12H26 geometry, a deterministic perturbed C12H26 target, and the
  committed TMAC/Cl diagnostic input. The ring and alkane coordinates are
  constructed directly; no external molecule database is copied.
- `smoke_test.py` covers single and trajectory capture, cross-AO overlap,
  same-target WARM/FRESH control, C12H26 formula/connectivity, SCC-policy
  mismatch rejection, and restricted-only rejection.

## Capture protocol

```bash
# One FRESH geometry.
build/issue343-cpu/xtbloom_scc_reuse_capture \
  single <case.spec> out.diag

# Source FRESH, target WARM, then the exact same target FRESH control.
build/issue343-cpu/xtbloom_scc_reuse_capture \
  traj <source.spec> <target.spec> out.diag

python3 tools/scc_reuse/scc_reuse_analyze.py \
  out.diag --report out.json
```

Trajectory specifications must have identical atoms, charge, spin,
temperature, mixer memory/damping, and maximum iterations. Only coordinates
and geometry-dependent point-charge values may change. This matches the
strict-WARM requirement that the compute policy remain fixed.

The `xtbloom-scc-reuse-v2` trajectory contains three explicit roles:

1. `source` / `fresh`;
2. `target_warm` / `warm`;
3. `target_fresh` / `fresh`.

Iteration reduction is reported only between roles 2 and 3, which use the
same target geometry and policy.

## Same-geometry metrics

For consecutive SCC iterations, the AO basis and overlap are unchanged:

- `rel_dH` and `rel_dP` are relative Frobenius changes;
- `subspace_capture_fraction` and `subspace_max_angle_deg` come from the SVD
  of `C_prev,occ^T S C_cur,occ`;
- `rel_residual_occupied` applies the previous occupied eigenpairs to the new
  effective Hamiltonian;
- `rr_eigenvalue_max_err` rediagonalizes that Hamiltonian in the previous
  occupied subspace;
- `step_micros` and `eigensolve_micros` are development CPU timings, not a
  CUDA performance claim.

## Cross-geometry metrics

Atom-centered basis functions move with their atoms, so `C1^T S1 C2` is not a
cross-geometry overlap. The capture evaluates the physical cross-AO matrix

```text
S12(mu,nu) = <chi_mu(R1) | chi_nu(R2)>
```

by constructing a diagnostic doubled `[source atoms, target atoms]` molecule
and extracting the off-diagonal block from the production overlap evaluator.
Physical principal angles use `C1,occ^T S12 C2,occ`. Density-operator change
uses `S1`, `S2`, `S12`, and `S21`; raw `P2-P1` is not interpreted across
different AO bases.

The algorithmic reuse residual is separate from that physical angle. Source
occupied coefficients retain their atom/AO labels on the target geometry,
are reorthonormalized in `S2`, Rayleigh--Ritz rotated with the **target terminal
effective SCC Hamiltonian**, and tested against that target generalized
eigenproblem. The analyzer rejects ill-conditioned transports, changed
occupied dimensions, missing cross overlaps, and nonconverged endpoints.

## Evidence discipline

Final evidence must be generated from a clean committed tooling revision.
Record the full source revision, complete capture-binary SHA-256, build/runtime
identity, exact commands, and all retained input/report hashes. Reproducible raw
diagnostics that exceed the repository's 1 MiB per-file or 16 MiB aggregate
budget stay untracked; compact reports and reproduction metadata remain in
Git. `SHA256SUMS` hashes every retained artifact except itself.
