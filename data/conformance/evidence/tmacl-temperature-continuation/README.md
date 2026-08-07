# Issue #217 temperature-continuation evidence: tmacl ion pair

Generated per-iteration CPU GFN2 SCC diagnostics for the separated
`Me4N+ / Cl-` ion pair (18 atoms). The fixture is
`data/conformance/inputs/tmacl.xyz`, reproduced verbatim from
grimme-lab/xtb issue #678 (Angstrom coordinates, converted to bohr at
1 / 0.529177210903).

The internal CPU GFN2 SCC driver (`iterate_scc_driver_batch_cpu`) produces
these files against the same sealed basis, integrals, H0, ES2/ES3/AES2, D4,
and eigensolver plans used by the public CPU execution path. The committed
executable `gpuxtb_scc_temperature_continuation_test` serves two distinct
purposes: normal CTest execution asserts the semantic matrix without writing
the source tree, while the explicit `--write-evidence` mode regenerates every
text artifact. `manifest.json` pins the fixture, generator source, generated
files, upstream issue metadata, and the MKL/scipy-OpenBLAS providers used for
generation and semantic cross-checking. It also records that this test-only
material is included in source distributions but excluded from native installs
and wheels.

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
- `tmacl_trace_400K.txt` — fresh SAD solve at 400 K, ceiling 250. Ends
  `SCC_NOT_CONVERGED` with the same large-amplitude oscillatory behavior.
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
  default policy does not. Every cell is executed; reviewed 300 K policies
  are required to converge, while marginal cells may cross the iteration
  ceiling across compatible BLAS reduction orders. All successful 300 K
  policies agree on the final
  energy and fragment charges to ~1e-9 (energy) and ~2e-5 (charge).
- `tmacl_path_sensitivity.txt` — terminal status/count matrix for two bounded
  funnels and the direct `1000 -> 300 K` and `800 -> 300 K` jumps. Both
  bounded paths converge; both direct jumps fail at the 250-iteration ceiling.

Regenerate and then refresh the reviewed hashes with:

```bash
build/cpu-t217/gpuxtb_scc_temperature_continuation_test \
  --write-evidence data/conformance/evidence/tmacl-temperature-continuation
python3 tools/conformance/tmacl_evidence.py --update
```

The first command must use the manifest-recorded LP64/sequential provider when
refreshing committed bytes. A second build against the recorded
scipy-OpenBLAS runtime must pass the same executable assertions. Normal review
and CI use `python3 tools/conformance/tmacl_evidence.py` in check-only mode.

## Decision (issue #217 investigation gate)

A reproducible 300 K charge-localized numerical fixed point exists: internal
SCC free energy -22.271821505 Eh, q(`Me4N+`) = +0.8285, and q(`Cl`) =
-0.8285. It is reachable without changing the final electronic temperature,
occupations, tolerances, or Hamiltonian:

1. by bounded temperature continuation (funnel) from an elevated
   temperature, reusing only the electronic state as the initial guess, and
2. by bounded deterministic mixer policies from the SAD guess.

The executable compares the complete atom-resolved q/d/Q arrays and density
at the terminal 300 K state. The 150 K-step funnel, 100 K-step funnel, and
every successful fresh 300 K mixer cell agree within 5e-5. It also pins the
exact default-policy baseline counts (250/1000/250/250/39/29/25 for the issue's
ordered rows), requires at least 20 fragment-charge sign changes and an energy
step above 0.5 Eh at 300 K, and rejects any unexpected driver-level error
instead of reclassifying it as SCC nonconvergence.

The default fixed Johnson modified-Broyden policy (history 8, damping 0.4)
seeded from the SAD guess at 300 K does not escape charge sloshing within
250 iterations, which is what upstream xTB also reports. Reaching the state
is therefore a solver-policy property rather than evidence that no numerical
fixed point exists. Path sensitivity is executable: direct 1000/800 K jumps
to 300 K do not converge, while the tested 150 K and 100 K bounded paths do.

## Limits

This remains internal CPU solver evidence, not a conformance golden or an
independent physical oracle. The text traces retain scalar residual and
fragment-charge diagnostics; complete q/d/Q and density are compared in
memory at terminal successful states rather than serialized per iteration.
Analytic forces at the recovered state, atom-resolved independent review,
ordinary-corpus regression, public C ABI behavior, CUDA parity, unrestricted
systems, ragged failure isolation, WARM/Graph semantics, and steady-state
allocation are unverified. The evidence also does not select temperature
continuation over a deterministic mixer-policy change. No public start-mode
contract should be treated as accepted until those open #217 gates pass.
