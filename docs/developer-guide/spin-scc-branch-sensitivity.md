# Spin SCC branch sensitivity: draft evidence for #509

This document records diagnostic evidence for issue #509. The evidence
supports numerical sensitivity of this unrestricted SCC case; it does not
establish a physics correction or resolve the reported public CPU/CUDA
mismatch. Issue #509 remains open.

## Public reproducer

The input is `CO_6444267115601304224457962` from
`gs2-problem-data-threeway-20260930.tar.gz` (SHA256
`8716d1dde0c4f9106dcdb295387c044c826769090ed8d546a4d5def43d2554eb`): 60
atoms, YClO10C12H36, charge +3, one unpaired electron, and explicitly two
spin channels. Coordinates are in bohr. The public cold-start comparison uses main commit
`2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`, 300 K, modified Broyden, a
500-iteration limit, energy tolerance `1e-10 Eh`, charge tolerance `1e-8`,
and one CPU/BLAS thread.

| Backend | Energy (Eh) | SCC iterations | Outcome |
| --- | ---: | ---: | --- |
| CPU | -88.55751920411979 | 365 | converged |
| CUDA | -88.55856605408331 | 316 | converged |

The absolute CPU/CUDA differences are `1.0468499635294393e-3 Eh` in energy,
`8.560817198010031e-4 Eh/bohr` in maximum force, and
`7.994193958632767e-3 e` in maximum atomic charge. This is the unresolved
public mismatch under the reported settings.

## Native trajectory evidence

The native fixture uses the same superposition-of-atomic-densities (SAD)
initial state on CPU/MKL and CUDA. Both begin with zero initial magnetization
and share the same spin targets. At the first iteration, the maximum
differences are `4.44e-16` in the Hamiltonian, `1.11e-15 Eh` in eigenvalues,
and `2e-14` in raw shell charges. Independent SCC trajectories nevertheless
reach different results. The native `HostSccCase` uses baseline scalar
Mulliken kernels, while public CPU calculations use ISA-dispatched kernels;
even with the same SAD and production LP64 runtime, the native trajectory is
not an exact reproduction of the public CPU trajectory.

| Trajectory | Native SCC-only free energy (Eh) | Iterations | Outcome |
| --- | ---: | ---: | --- |
| CPU/MKL | -89.235303886317482 | 287 | converged |
| CUDA | -89.233701774909392 | 281 | converged |

The final independent `--spin-branch-probe` reports CPU `287, converged=1`
and CUDA `281, converged=1`.

These native free energies omit geometry-only repulsion and D4 ATM terms.
They are diagnostic SCC quantities and must not be compared directly with the
public total energies above.

The complete-state frozen replay passed 287 steps. At each step it clones the
complete CPU mixer and driver pre-state, including history, inputs, and scalar
trace/activity masks, then compares one CPU/CUDA transition; it does not freeze
only the mixed fields.

| Term | Maximum absolute difference | Absolute threshold |
| --- | ---: | ---: |
| Hamiltonian | 4.4408920985006262e-16 | 1e-12 |
| Eigenvalues | 3.0461744238152733e-15 | 1e-12 |
| Occupations | 8.7631291112444387e-13 | 1e-10 |
| Raw shell charges | 1.3939960297193466e-12 | 1e-10 |
| Mixed state | 7.9103390504542404e-12 | 3e-9 |
| Energy (Eh) | 4.0216718844021671e-12 | 1e-10 |
| Spin energy (Eh) | 8.3678723672431232e-15 | 1e-10 |
| Raw dipoles | 3.8535841184739184e-13 | 1e-10 |
| Raw quadrupoles | 1.0409451078885468e-12 | 1e-10 |
| Density | 5.3712589931365073e-13 | 1e-10 |
| Energy-weighted density | 3.872457909892546e-13 | 1e-10 |
| Raw atomic charges | 1.3950507415927405e-12 | 1e-10 |

CPU-only perturbations to a paired spin-shell seed preserve total
magnetization. The resulting native SCC-only outcomes are:

| Paired seed perturbation | Free energy (Eh) | Outcome |
| ---: | ---: | --- |
| 0 | -89.235303886317482 | converged, 287 iterations |
| +1e-14 | -89.233887562368466 | converged, 344 iterations |
| -1e-14 | -89.23282347065313 | not converged at 500 iterations |
| +1e-12 | -89.235499218473507 | converged, 203 iterations |
| -1e-12 | -89.234758622282314 | converged, 242 iterations |

Together, the CPU seed perturbations and complete-state frozen replay most
strongly support branch selection amplified by small numerical differences as
the explanation for the divergent endpoints. They do not prove that an
endpoint is a global minimum or guarantee that no backend defect exists
outside the audited scope. The public mismatch remains unresolved.

## Probe behavior and run environment

The diagnostic entry points are:

- `xtbloom_cuda_scc_iteration_production_test --spin-branch-probe` reports
  independent CPU and CUDA trajectories. It records their endpoints and does
  not require endpoint parity as a test assertion.
- `xtbloom_cuda_scc_iteration_production_test --spin-cpu-sensitivity` runs the
  CPU-only paired spin-shell seed sensitivity cases. Main dispatch reaches
  this mode before any `cudaGetDeviceCount` call, so it requires no real GPU.
- `xtbloom_cuda_scc_iteration_production_test --spin-frozen-replay` compares
  one transition from cloned complete CPU state using the absolute thresholds
  above. CTest registers this case as
  `xtbloom.cuda.scc_spin_frozen_replay` only when a production LP64 provider is
  available.

In a no-driver configuration, the frozen-replay CTest exits 77 as a skip; this
is not a pass. CUDA 12.9 Compute Sanitizer `memcheck` completed the full
287-step replay under Slurm with zero errors. The final targeted CTest passes
all twelve comparisons (1 passed, 0 failed).

The full 199-test run records **198 passed and 1 subprocess terminated**:
the unchanged `xtbloom.cuda.public_api` executable stopped making observable
progress and was terminated after more than eight minutes. This is not a
green full-suite result, and its cause has not been established. The passing
tests include public unrestricted cases, host/device/mixed conformance,
forces/invariants, graph/cache paths, and peer-failure isolation. The full
run uses Slurm `--cpus-per-task=32`; the GPU probes remain single-threaded.

An earlier full run failed five provider-dependent tests because a trailing
empty entry in `LD_LIBRARY_PATH` made the loader search the current directory
and prevented discovery of adjacent MKL shims. Use
`CUDA_RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}` when prefixing the
runtime path; do not add an empty search-path entry. The corrected run passes
the provider-dependent CPU and runtime-parity tests but retains the API-test
termination described above.

For example, run each real-GPU probe through Slurm with an explicit finite
time limit, and preserve the device visibility assigned by Slurm:

```sh
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:20:00 bash -lc \
  './xtbloom_cuda_scc_iteration_production_test --spin-branch-probe'

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:20:00 bash -lc \
  './xtbloom_cuda_scc_iteration_production_test --spin-frozen-replay'
```

The CPU sensitivity probe is CPU-only. Evidence uses the selected production
LP64 runtime and CUDA 12.9. It does not use relaxed convergence tolerances,
disable spin, change the physical model or temperature, or route CUDA work to
CPU.

## Status and interpretation

This draft documents branch-sensitivity evidence; it does not fix the original
public mismatch. A robust policy shared by both backends for SCC convergence
and stability still needs design and independent validation. Rounding values
or selecting a seed solely to make this structure pass would not establish
that policy or resolve issue #509.
