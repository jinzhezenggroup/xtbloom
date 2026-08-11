# SCC subspace-reuse diagnostics (issue #343, Phase 1)

Experimental investigation tooling for
[issue #343](https://github.com/anomalyco/gpuxtb4/issues/343): measuring
whether the occupied eigenspace of one SCC iteration (or one converged
geometry) can be reused by the next, before committing to a recycled or
inexact eigensolver.

Nothing here is part of the public C ABI, the versioned
`xtbloom-scc-trace-v1` contract, or any conformance/acceptance gate.

## Components

- `scc_reuse_capture.cpp` — native executable `xtbloom_scc_reuse_capture`
  (built from this directory and registered in `CMakeLists.txt`). It drives
  the production CPU GFN2 SCC driver through a corpus-style `case.spec` and
  streams, per completed iteration: the effective Hamiltonian and density, the
  generalized eigenpairs `(C, eps)` with `C^T S C = I`, the overlap `S`, the
  core Hamiltonian `H0`, driver-step wall time, and an isolated
  eigensolve-only wall time obtained by re-running the production
  `solve_eigensystem_cpu` on the same `H_k`.
- `scc_reuse_analyze.py` — reads the stream and computes the Phase-1 metrics
  below; emits a JSON report and a console summary.
- `prepare_cases.py` — regenerates the diagnostic `case.spec` inputs in
  `cases/` (ASE g2 molecules, a deterministic trans-planar alkane, and the
  committed `tmacl.xyz` fixture). The five conformance corpus specs in
  `data/conformance/scc-traces/specs/` are consumed directly.
- `cases/` — generated `.spec` inputs (no goldens; exact geometry is
  informational only).

## Usage

```bash
# single geometry, stream to stdout but also save
build/issue343-cpu/xtbloom_scc_reuse_capture single <case.spec> out.diag
# warm-start trajectory: first geometry to convergence, then advance to a
# second geometry keeping the converged wavefunction as the SCC seed
build/issue343-cpu/xtbloom_scc_reuse_capture traj <a.spec> <b.spec> out.diag

python3 tools/scc_reuse/scc_reuse_analyze.py out.diag [--report out.json]
```

The capture requires a configured CPU LP64 eigensolver runtime (same build
gate as `xtbloom_scc_trace_capture`).

## Metrics

For consecutive SCC iterations (previous `k-1` → current `k`):

- `rel_dH`, `rel_dP` — relative Frobenius change of the effective Hamiltonian
  and the density matrix;
- `subspace_capture_fraction` — `sum_i cos^2(theta_i) / n_occ,new`, the
  fraction of the new occupied subspace reproduced by the previous one
  (principal angles in the S metric via the SVD of
  `C_prev_occ^T S C_cur_occ`);
- `subspace_max_angle_deg` — the largest principal angle between the two
  occupied subspaces;
- `rel_residual_occupied` — `||H_k C_prev - S C_prev diag(eps_prev)||_F` over
  the occupied columns, relative to `||H_k||_F`. This mixes subspace capture
  with eigenvalue drift;
- `rr_eigenvalue_max_err` — the max absolute error of the Rayleigh-Ritz
  eigenvalues `eigvals(C_prev_occ^T H_k C_prev_occ)` against the new occupied
  eigenvalues. This is the decisive number: when the recycled subspace is
  captured, it is ~0 even if individual eigenvalues drifted, because the cheap
  RR rediagonalization inside the recycled subspace cures the drift;
- `step_micros` / `eigensolve_micros` — driver-step and isolated-eigensolve
  wall time (CPU, whole-step proxy; not a CUDA sytrd microbenchmark).

For trajectory documents, the same quantities (plus `rel_dH0`) are computed
between the converged states of consecutive geometries.

## Reading the results

Bundles of captures and analysis live under
`benchmarks/evidence/issue-343/`. The headline Phase-1 finding so far: in
converged restricted GFN2 SCC, the occupied subspace is already ≥0.999
captured by iteration ~4–6 of 9–17, the Rayleigh-Ritz eigenvalue error drops
below 1e-6 well before energy convergence, and the first few iterations
(including the zero-charge-seeded first solve) are the only ones requiring a
full diagonalization. A non-converging sloshing case (tmacl at 300 K) keeps a
0.95–1.0 capture with small principal angles but shows periodic one-orbital
frontier switches — the exact situation a recycled solver must detect and
fall back from. Consecutive-geometry warm starts converge faster and the
converged occupied subspace transfers across geometries nearly unchanged
(>0.99 capture for a small MD-style step).
