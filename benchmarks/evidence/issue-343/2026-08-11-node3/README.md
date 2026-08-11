# Issue #343 Phase 1: SCC subspace-reuse diagnostics evidence

This directory archives the first defensible Phase-1 measurement for issue
#343: how much of the previous SCC iteration (or previous converged geometry)
occupied eigenspace can be reused by the next iteration. All numbers come from
the committed `tools/scc_reuse` capture tool driving the **production CPU GFN2
SCC driver** on the five pinned conformance corpus cases plus generated
diagnostic molecules (benzene, pyridine, dodecane, tmacl) and a warm-start
two-geometry dodecane trajectory.

Experimental tooling and evidence only. No public ABI, parameter, golden, or
acceptance gate changed; the versioned `xtbloom-scc-trace-v1` contract is
untouched.

## Method

Per completed SCC iteration the tool records the effective Hamiltonian, the
generalized eigenpairs `(C_k, eps_k)` with `C_k^T S C_k = I`, the density, the
overlap `S`, and two timings (whole driver step and isolated eigensolve via
the production `solve_eigensystem_cpu`). The analyzer computes:

- `rel_dH` / `rel_dP` — relative Frobenius change of H and P;
- `subspace_capture_fraction` — fraction of the new occupied subspace
  reproduced by the previous one (S-metric principal angles via SVD of
  `C_prev_occ^T S C_cur_occ`);
- `subspace_max_angle_deg` — the largest principal angle;
- `rr_eigenvalue_max_err` — max |Rayleigh-Ritz eigenvalue in the previous
  occupied subspace vs the new occupied eigenvalue|. The decisive number:
  captures subspace error cleanly, since even a perfectly captured subspace
  shows a large raw residual when eigenvalues drift, and the cheap RR
  rediagonalization cures that drift;
- `step_micros` / `eigensolve_micros` — CPU wall time (whole-step and
  eigensolve-only).

The SCC policy matches the conformance corpus (zero-charge seed, D4 two-body
potential, tblite energy 1e-6 / RMS 2e-5 tolerances). Per-iteration
`validation_ctsc_max` confirms `C^T S C = I` to roundoff in every snapshot.

## Environment and build identity

- Host: `node3`, Ubuntu 22.04.5 LTS, Linux 6.8.0-110-generic, 48 threads.
- CPU: AMD EPYC 7K62 48-Core Processor.
- Compiler: GCC 11.4.0; CMake Release, Ninja; BLAS one thread.
- LP64 runtime: `/home/jzzeng/miniconda3/lib/libmkl_rt.so.3`
  (SHA-256 prefix `b2ff0e31d7cd18c9`), host-isolated MKL shim.
- Source revision: `cbdf755f27ab02b548783bce3573ecb4385ed167` (perf/343 branch
  base; the tooling itself is the branch diff).
- Analyzer: Python 3.13.9 + numpy 2.5.1 (project venv), numpy only.

Raw capture streams are archived gzipped as `diagnostics/*.diag.gz`; the
parsed metric reports are `diagnostics/*.json`. Regenerate both with
`tools/scc_reuse/README.md` instructions.

## Numerical validation

Every snapshot passes `validation_ctsc_max <= 1e-9` (C^T S C = I to roundoff)
and `validation_he_max <= 1e-9` (eigenpair residuals), and every converged case
reproduces the pinned golden trajectory shape (same iteration count as the
`data/conformance/scc-traces` goldens: ketene 13, nenacl 14, water dimers 9,
h3_plus 3).

## Findings (restricted GFN2 SCC, 300 K default policy)

| case | nao | iters | converged | max early ang (deg) | first ang < 1 deg at iter | capture >= 0.9999 at iter | RR eig err < 1e-6 at iter | eigensolve share of step |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| h3_plus | 3 | 3 | yes | 0 | 2 | 2 | 2 | 0.07 |
| ketene | 14 | 13 | yes | 11.3 | 6 | 6 | 9 | 0.44 |
| nenacl | 22 | 14 | yes | 30.7 | 6 | 6 | 10 | 0.52 |
| water dimer 6PC | 12 | 9 | yes | 3.2 | 5 | 5 | 6 | 0.32 |
| water 1PC | 6 | 9 | yes | 3.1 | 5 | 5 | 6 | 0.21 |
| benzene | 30 | 7 | yes | 0.7 | 2 | 2 | 5 | 0.43 |
| pyridine | 29 | 10 | yes | 9.3 | 6 | 5 | 7 | 0.49 |
| dodecane | 62 | 17 | yes | 87.7 | 9 | 7* | 10 | 0.69 |
| tmacl (sloshing) | 41 | 100 | no | 2.4 | 2 | 4 | 4 (intermittent) | 0.63 |

Milestone iterations are the 1-based `k` of the pair (`k-1` -> `k`); values
are the first time the condition holds, and may regress afterwards (marked
`*` for dodecane, where iteration 9 returns to a 0.9687 capture from a single
frontier-orbital swap before settling at 1.0).

Headline results:

1. **Converged restricted SCC reuses its occupied subspace almost completely
   once the initial transient is over.** In every converged case the previous
   iteration's occupied subspace captures ≥0.9999 of the next one and the max
   principal angle drops below 1° within roughly the first half of the
   iterations, with the Rayleigh-Ritz eigenvalue error below 1e-6 well before
   energy convergence.
2. **The early iterations dominate the full-solve work.** The first several
   iterations (including the zero-charge-seeded first solve) are the only ones
   with large principal angles; dodecane shows worst-case 87° in iteration 2
   and angles above 1° through iteration 8, collapsing below 1° at iteration 9
   of 17. This matches the issue's proposed pattern
   `full solve -> several subspace updates -> full correction -> ...`.
3. **The raw residual overstates the reuse failure.** `rel_residual_occupied`
   stays large (1e-2..1e-3) even when capture is 1.0, because it includes
   eigenvalue drift; `rr_eigenvalue_max_err` shows the drift is cured by the
   cheap RR rediagonalization inside the recycled subspace.
4. **Difficult non-converging SCC (tmacl at 300 K, charge sloshing) keeps a
   0.95–1.0 capture and angles <2.4°**, but shows periodic one-orbital
   frontier switches (capture dips to 0.9524 = exactly 1/21 occupied orbitals)
   with growing RR error. A recycled solver must detect frontier switching /
   near-degeneracies and fall back to a full solve — the issue's robustness
   requirement is real.
5. **Warm-start geometry trajectories transfer the subspace nearly
   unchanged.** For a small MD-style step of dodecane (each atom perturbed σ =
   0.015 bohr, rel ||dH0|| = 1.5e-2), the converged states of the two
   geometries share 0.997 subspace capture with a 7.2° worst angle, and the
   warm-started solve converges in 13 vs 17 iterations with RR error < 1e-6 by
   iteration 9.
6. **The eigensolve is the dominant cost and grows with size** (0.44–0.69 CPU
   share of the driver step for nao > 12), consistent with the CUDA profiling
   motivation in the issue. Reducing full solves therefore has real end-to-end
   headroom if the fallback/expansion cost stays small.

## Verdict

Phase 1 is positive: subspace reuse in GFN2 SCC is quantitatively strong for
converged restricted single points and warm-start geometry trajectories, so a
recycled eigensolver is worth prototyping (Phase 2) — with the caveat that the
first 2–5 SCC iterations and all frontier-switching events must keep the full
full-solve path. This evidence does NOT justify closing #343 as unpromising.

## Caveats

- This is CPU-evidence under the corpus policy (tblite single-point
  tolerances, mixer history 2 for corpus cases and 8 for generated cases,
  300 K). CUDA uses the same equations; per-iteration eigensolve timing here is
  a CPU whole-step/eigensolve proxy, not the CUDA sytrd microbenchmark the
  issue's motivation targets.
- Sample set is modest (5 pinned + 4 generated single points + 1 warm-start
  trajectory, restricted, nao <= 62). Larger delocalized/conjugated systems,
  unrestricted SCC, and true MD/optimization chains remain unmeasured.
- The `rr_eigenvalue_max_err` metric uses a hard occupation cutoff (>0.5);
  finite-temperature fractional occupations are not folded in.
