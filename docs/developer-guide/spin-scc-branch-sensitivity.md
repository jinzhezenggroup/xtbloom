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

### Actual Two-Origin Checkpoint Control

The opt-in `--spin-two-origin-replay [new-output-directory]` mode captures
independently evolved CPU and CUDA pre-states after exactly 19 transitions,
then replays transition 20 from each origin through both backends. Transition
20 is a hindsight-selected diagnostic coordinate from the earlier threshold
crossing, not a searched-for passing point or a prospective holdout.

The typed snapshots retain separate SCC q/d/Q and mixer current/previous
inputs, previous residuals, full df/u/omega histories, residual diagnostics,
counters, statuses, and terminal flags. They also retain wavefunction,
Hamiltonian, raw/publication, and energy outputs. Every mapped field is
restored and checked byte-for-byte before replay; descriptors and allocation
ownership never move between backends. CPU-specific unpublished scratch is
restored from the compatible host checkpoint, while CUDA's arena is restored
on its original device allocations.

CPU `mixer.converged` contains terminal SCC convergence, whereas CUDA
`mixer.residual_converged` is residual-only. The bridge derives the CPU
residual-only diagnostic from the real RMS/maximum thresholds, without
substituting a terminal flag. CUDA's separate SCC RMS has no separate CPU
storage slot: a test-owned proxy preserves it during import, and fresh CPU
post-state comparisons derive that diagnostic from actual CPU mixer RMS.
CPU private next-round q/d/Q maps to the corresponding slices of
`mixer.current_inputs`, not its published wavefunction. The mapper requires
the single-system field-major layout, including both spin channels within
each field. Published multipoles must equal SCC inputs byte-for-byte at the
active pre-transition coordinate, while using distinct storage. Terminal
publications can be raw and are not interchangeable with private next inputs.
Eigenvector coefficients are restored exactly, but post-state
comparison uses the existing sign/degenerate-subspace-aware comparator.

The output directory is created once and must not already exist. Binary
little-endian snapshots and their stdout dtype/count ledger include the real
input geometry and all six pre/post snapshots. The two origins must differ
in at least one transition-relevant input/history field. Both origins are
attempted even when a numerical comparison fails; an invalid restoration or
launch stops rather than executing a misleading control. The mode reports
full post-state/history errors and literal pass/fail at the existing field
tolerances. A no-driver invocation returns 77, and CTest registers
`xtbloom.cuda.scc_spin_two_origin_replay` only with a production LP64 provider.

Validate the complete saved stdout and snapshot directory without running a
model:

```bash
python3 tools/oracle/check_spin_two_origin_replay.py captured-replay.stdout saved-snapshots
```

The directory must match the location recorded by the native invocation.
The checker requires all 294 input/state files and independently recomputes
each typed field comparison at the frozen limits. It treats the native gauge
record as a comparator assertion, not an independently evaluated eigenspace.
Exit 0 means complete diagnostic agreement, exit 1 retains a complete
numerical failure, and exit 2 rejects malformed or incomplete evidence.

Compilation, snapshots, or replay agreement alone do not resolve the original
unchanged-setting public E/F/q mismatch. These scalar-Mulliken, SCC-only
controls do not independently validate a public stationary solution, establish
a basin boundary, or authorize a new solution-selection policy.

### Public FRESH and Strict-WARM Controls

Separate public-C-ABI observations on clean source
`471a7d50e86a8e56d213e3f599e6f5cf39a1486f` reproduce the original endpoint
gap without using the scalar-Mulliken fixture. Both observations use the same
shared library bytes (SHA256
`a0ef437885f2ece3ce10c9e0cb5bb1b82bf57893a970ae1323dfc9767440aef7`),
supplied geometry, explicit two channels, and original numerical settings.
Three independent contexts per coordinate each execute
`FRESH -> strict WARM -> strict WARM` at unchanged geometry and policy.
There is no WARM-to-FRESH retry or selection of a passing seed.

The first observation, Slurm Job 7282, retains **exit 1** for the probe,
outer `srun`, and SSH. Its 36 attempts contain 27 finite/converged CUDA
endpoints and nine CPU rejections. The staged MKL provider shim lacked its
own `libxtbloom_mkl_pthread_tss_bridge.so` dependency, as recorded by `ldd`.
Three CPU FRESH calls return backend-unavailable before SCC; six strict WARM
calls return invalid-argument because no converged predecessor exists.
All CPU outputs remain unchanged. This is a deployment failure, not CPU
nonconvergence or a measured CPU endpoint.

A separately preregistered CPU-only environmental complement, Job 7283,
adds the exact already-built bridge without changing the library, provider
shim, inputs, or numerical policy. Its nine CPU attempts are finite and
converged; probe, outer `srun`, and SSH exit 0. The 27 CUDA observations are
not rerun. Job 7282 remains failed: the two observations must not be reported
as one repaired 36/36 passing invocation.

| Public coordinate | FRESH energy (Eh) | FRESH iterations | Successive WARM iterations |
| --- | ---: | ---: | --- |
| CPU, host descriptors, Job 7283 | -88.55751920411979 | 365 | 9, 6 |
| CUDA, host descriptors, Job 7282 | -88.55856605408331 | 316 | 4, 4 |
| CUDA, device descriptors, Job 7282 | -88.55856605408331 | 316 | 4, 4 |
| CUDA, mixed descriptors, Job 7282 | -88.55856605408331 | 316 | 4, 4 |

All three CPU FRESH outputs are byte-identical. CUDA FRESH energies and
charges are identical across contexts/memory modes; the maximum force spread
is `4.163336342344337e-17 Eh/bohr`. Pair each CPU FRESH context with the
same-index CUDA context in each of the three memory modes. Across these nine
explicitly cross-job comparisons, the maximum energy/force/charge differences are
`1.0468499635294393e-3 Eh`, `8.560817198010169e-4 Eh/bohr`, and
`7.994193958632767e-3 e`. Thus the original public gap is reproduced, not fixed.

| Successive WARM output changes | Energy (Eh) | Maximum force (Eh/bohr) | Maximum atomic charge (e) |
| --- | ---: | ---: | ---: |
| CPU, six comparisons | 1.2079226507921703e-12 | 3.026340120361069e-8 | 7.597233708800388e-8 |
| CUDA, eighteen comparisons | 6.536993168992922e-13 | 2.4009378028679723e-8 | 6.071044778011014e-8 |

These are observed output changes, not reconstructed SCC residuals or new
acceptance tolerances. In particular, the maximum charge changes exceed
`1e-8`; a native convergence flag is not an endpoint charge-parity pass.
CPU WARM restores a wavefunction checkpoint and reinitializes its mixer;
same-epoch CUDA WARM retains mixer history while resetting driver accounting.
These public persistence controls therefore are not identical-internal-state
paired transition tests.

Both jobs run on node1 with the `main` partition, the `gpu:5090:1` resource
request, one CPU/BLAS worker, and explicit ten/five-minute limits. Recorded
`scontrol` metadata confirms the node, partition, limits, and `TresPerNode`
request; it does not provide an independently accounted physical GPU UUID.
Assigned device visibility is preserved. Offline typed audits validate all
36 and nine saved records,
respectively, including policy, memory-space tags, statuses, and output bytes.
The first artifact receipt has SHA256
`69350d7c5f8bbed538e3a23c938cb7d3c5411c00f979cdff2b67a237c05b30db`;
the CPU complement receipt has SHA256
`ad058cecc4a26fe550d4a43e35aac91f90734e67d827439ed1682af4236ed30d`.
Exact scripts, preregistrations, raw arrays, source/runtime/payload receipts,
failure records, and independent audits are linked from issue #509.

Finite output capture and these short restart sequences do not establish
fixed-point stationarity, physical/linear-response stability, independently
correct endpoints, or the basin root cause. The public ABI does not expose
terminal raw-versus-mixed multipoles or SCC residuals. The root correction,
unchanged-setting public parity, and remaining applicable full CUDA validation
remain incomplete.

This draft documents branch-sensitivity evidence; it does not fix the original
public mismatch. A robust policy shared by both backends for SCC convergence
and stability still needs design and independent validation. Rounding values
or selecting a seed solely to make this structure pass would not establish
that policy or resolve issue #509.

## CPU public terminal-buffer diagnostic

When native tests and a compatible CPU LP64 provider are enabled,
`xtbloom_cpu_scc_snapshot_test` builds a separate diagnostic executable from
the real synchronous C API, CPU runtime, and production model/ISA objects.
Its internal context resolver uses handles created by that same executable;
it never interprets a handle belonging to the production shared library.
`XTBLOOM_PUBLIC_SCC_SNAPSHOT_TESTING` is confined to this executable. The
released ABI and default library contain neither its resolver nor its capture
receipt or copying path.

The default invocation tests restricted/unrestricted ragged state, missing
receipts, failure invalidation, empty calls, changed geometry/topology, and
value ownership. It also compares FRESH/WARM/WARM public outputs in captured
and uncaptured contexts. The supported diagnostic protocol is serial direct
`xtbloom_compute` on a CPU GFN2 context, immediately followed by capture.
All energy/force/atomic-charge properties must be requested and every peer
must have converged.
Enqueue, fixed plans, external callbacks, and concurrent context operations
are outside this seam's contract. A call-level rejection or a different model
invalidates the direct-call receipt. A data-level failed peer prevents a
complete converged snapshot; neither call success nor an older checkpoint
may be substituted for that missing terminal state.

For an external molecular input, the optional invocation is:

```console
build/issue-509-cpu-terminal/xtbloom_cpu_scc_snapshot_test --capture input.txt > terminal.json
```

The first input line is `atom_count molecular_charge unpaired_electrons
spin_channels`; each following line is `atomic_number x_bohr y_bohr z_bohr`.
Numbers must preserve the original binary64 values, atom order, charge, and
explicit spin choice. The producer runs FRESH/WARM/WARM with one CPU worker,
300 K, maximum 500 iterations, energy tolerance `1e-10` Ha, charge tolerance
`1e-8`, and modified Broyden history 8/damping 0.4/default determinism. WARM
does not fall back to FRESH. JSON is emitted only after all three calls and
captures succeed, using round-trip binary64 precision. Capture is outside
SCC; this is not a timing producer. Retain failed command exits and stderr
separately rather than presenting absent JSON as a passing run.

The value-owned arrays preserve the actual wavefunction layout: orbital
coefficients, densities, and energy-weighted densities are alpha/beta
spin-major; occupations always have two rows, including restricted cases.
Shell/atom charges and atomic dipoles/quadrupoles instead use the distinct
charge/magnetization convention. Mixer vectors are the literal existing
`qsh`, dipole, quadrupole concatenation. `mixer_current_inputs`,
`mixer_previous_inputs`, and `mixer_previous_residuals` retain their storage
names; do not assume they are a freshly recomputed fixed-point map at the
published density. The SCC free energy excludes the geometry-only terms
present in the full public energy. No retained solver scratch is certified
or serialized as a terminal Fock operator.
The mixer RMS/maximum summaries describe its packed vector, not the maximum
public atomic-charge error or drift.

Before using an endpoint for scientific reasoning, separately qualify the
diagnostic executable against the actual shared-library public E/F/q,
statuses, and iteration counts at identical source, provider, selected ISA,
descriptors, and options. Archive both binary identities and input hashes.
These are distinct producers, even when qualified outputs agree. A different
provider or ISA is not a replay of the earlier pinned MKL observations.
This CPU-only observability step does not establish the CUDA endpoint,
fixed-point stationarity, minimum/stability, independent oracle correctness,
or the root correction required by issue #509.
