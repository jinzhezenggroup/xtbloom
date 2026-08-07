# Issue #217 temperature-continuation evidence: tmacl ion pair

Archived per-iteration CPU GFN2 SCC diagnostics for the separated
`Me4N+ / Cl-` ion pair (18 atoms). The fixture is
`data/conformance/inputs/tmacl.xyz`, reproduced verbatim from
grimme-lab/xtb issue #678 (Angstrom coordinates, converted to bohr at
1 / 0.529177210903).

These traces were produced by the internal CPU GFN2 SCC driver
(`iterate_scc_driver_batch_cpu`) on gpuxtb main `9fd7d4d` plus this issue's
branch, against the same sealed basis, integrals, H0, ES2/ES3/AES2, D4, and
eigensolver plans used by the public CPU execution path. The committed CTest
gate `gpuxtb.gfn2.scc_temperature_continuation`
(`tests/scc_temperature_continuation_test.cpp`) regenerates and asserts the
same baseline status matrix, the funnel, the charge-sloshing signature, and
the mixer-policy result directly from the fixture.

Each row reports: iteration, internal SCC free energy (Eh), free-energy
change, mixer residual RMS, mixer residual maximum, the summed `Me4N+`
fragment charge, the `Cl-` fragment charge, mixer restart count, driver
status (0 = success, 7 = `GPUXTB_STATUS_SCC_NOT_CONVERGED`), and converged
flag. Default policy unless stated is Johnson modified-Broyden history 8,
damping 0.4, charge tolerance 1e-6, energy tolerance 1e-8.

## Files

- `tmacl_trace_300K.txt` — fresh SAD solve at 300 K, ceiling 250. Ends
  `SCC_NOT_CONVERGED`; free energy oscillates by ~1 Eh and fragment charge
  contrast flips sign repeatedly (charge sloshing). Residual RMS stays near
  1e-2..1e-1 (gate is 1e-6).
- `tmacl_trace_450K.txt` — fresh SAD solve at 450 K. Converges in 39
  iterations to the localized stationary state (internal SCC free energy
  -22.272478576591 Eh).
- `tmacl_trace_500K.txt` — fresh SAD solve at 500 K. Converges in 29
  iterations (-22.272701614316 Eh).
- `tmacl_trace_1000K.txt` — fresh SAD solve at 1000 K. Converges in 25
  iterations (-22.275035844965 Eh).
- `tmacl_continuation_funnel.txt` — bounded temperature continuation
  `1000 -> 850 -> 700 -> 550 -> 400 -> 300 K`. The first stage is a fresh SAD
  solve; each later stage reuses only the converged electronic state
  (wavefunction multipoles) of the previous stage as its initial guess. Every
  stage converges, including the final requested 300 K stage (8 iterations,
  internal SCC free energy -22.271821505179 Eh,
  q(`Me4N+`) = +0.8285, q(`Cl`) = -0.8285).
- `tmacl_mixer_sweep.txt` — fresh SAD solves at 300 K and 450 K over the
  bounded deterministic mixer-policy grid history x damping =
  {2,4,8,16} x {0.2,0.4,0.6}. At 300 K several policies (for example
  history 2 damping 0.2, history 8 damping 0.2, history 16 damping 0.4)
  converge fresh to the same localized state the funnel reaches, while the
  default policy does not. All successful policies agree on the final
  energy and fragment charges to ~1e-9 (energy) and ~2e-5 (charge).

## Decision (issue #217 investigation gate)

A reproducible 300 K localized stationary state exists — internal SCC free
energy -22.271821505 Eh, q(`Me4N+`) = +0.8285, q(`Cl`) = -0.8285 (public
total energy adds a fixed repulsion + D4-ATM offset of about +0.25947534 Eh,
projecting to about -22.01235 Eh). It is reachable without changing the
final electronic temperature, occupations, tolerances, or Hamiltonian:

1. by bounded temperature continuation (funnel) from an elevated
   temperature, reusing only the electronic state as the initial guess, and
2. by bounded deterministic mixer policies from the SAD guess.

The default fixed Johnson modified-Broyden policy (history 8, damping 0.4)
seeded from the SAD guess at 300 K does not escape charge sloshing within
250 iterations, which is what upstream xTB also reports. Reaching the state
is therefore a solver-policy property rather than an absent stationary
state. Path sensitivity exists: coarse continuation jumps (1000/800 K ->
300 K) stay in a non-converged basin, while bounded stages (<= ~150 K and
<= ~250 K steps both reproduce the state) converge.

The public contract decision is recorded in issue #217. These traces are
input to, but not a replacement for, the CPU/CUDA-identical public contract
needed for production rollout.