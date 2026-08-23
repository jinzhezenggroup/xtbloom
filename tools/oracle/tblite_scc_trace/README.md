# Pinned tblite SCC observer seam

This directory carries the minimal oracle-only patch used to observe tblite's
GFN2 SCC loop. It targets exactly
`e9abc395b122018ed688aecb1c3a65cecaf97beb`; it is not a patch for arbitrary
tblite releases and it is not part of xTBloom's public C ABI.

The patch is licensed `LGPL-3.0-or-later`, matching the files it modifies in
tblite. Verbatim GPLv3 and LGPLv3 texts from the pinned checkout are bundled as
`LICENSES/GPL-3.0.txt` and `LICENSES/LGPL-3.0.txt`. The validation script,
metadata, probe, and tests are original xTBloom tooling; their repository-wide
license remains tracked by #19 rather than being selected implicitly here.

## Immutable inputs

- Upstream: <https://github.com/tblite/tblite.git>
- Revision: `e9abc395b122018ed688aecb1c3a65cecaf97beb`
- Patch: `tblite-e9abc395-scc-observer.patch`
- Patch SHA-256:
  `887294ffb24d24fc0fb27d6f05697e7ef498bc40f0268f34c7056ebe8dfb7928`

`metadata.json` is the machine-readable source of truth. The validator rejects
a patch whose digest or touched-file list differs from that metadata.

## Callback contract

The patch adds a concrete `scf_observer` base type with three overridable no-op
methods. Omitting the optional observer takes the original calculation path.
Passing the base type performs no allocation and changes no numerical state.

### `before_solve(iteration, wfn, pot)`

This callback runs after `add_pot_to_h1` and `set_mixer`, immediately before
`next_density`:

- `iteration` is the one-based outer SCC attempt;
- `wfn%coeff[nao,nao,nspin]` is the assembled effective Hamiltonian;
- `wfn%qsh[nsh,nspin]`, `qat[nat,nspin]`, `dpat[3,nat,nspin]`, and
  `qpat[6,nat,nspin]` are the mixed SCC input;
- `pot%vat`, `vsh`, `vao`, `vdp`, and `vqp` are the assembled potential arrays;
- `set_mixer` has saved the same mixed q/d/Q state used to build the matrix.

`next_density` overwrites `wfn%coeff` with orbital coefficients. A recorder
must therefore deep-copy the effective Hamiltonian inside this callback.

### `after_iteration(iteration, wfn, eelec, elast, pnorm, flags...)`

This callback is emitted only when `next_scf` completed without an error. It
runs after tblite has calculated energy convergence, mixer RMS, and all three
convergence flags, but before per-cycle output:

- `wfn%coeff` contains orbital coefficients, not a Hamiltonian;
- `wfn%emo[nao,nspin]`, `focc[nao,max(2,nspin)]`, and
  `density[nao,nao,nspin]` are the eigensolver result;
- qsh/qat/dpat/qpat are the raw Mulliken output for this iteration;
- `eelec[nat]` is the per-atom electronic SCC energy in Hartree;
- `elast` is `sum(eelec)` before the attempt, so the electronic energy delta is
  `sum(eelec)-elast`;
- `pnorm` is tblite's unweighted RMS of the flattened mixer residual.

For GFN2, the residual is reconstructed as raw minus mixed in exact Fortran
column-major order: `qsh(:,:)`, then `dpat(:,:,:)`, then `qpat(:,:,:)`. `qat`
is derived from qsh and is not part of the mixer vector.

### `finished(iterations, status)`

This callback reports one terminal state after SCC mixer construction:

- `scf_observer_status_converged = 1`;
- `scf_observer_status_max_iterations = 2`;
- `scf_observer_status_failed = 3`.

It is also called when `new_mixer` fails, with `iterations=0`. A later mixer
failure can occur before an iteration counter is incremented; an eigensolver
failure can occur after `before_solve` but before `after_iteration`. Keeping
failure payload out of `after_iteration` prevents stale eigenvectors, density,
or energy from being mistaken for a completed iteration. Input validation that
returns before SCC mixer construction is outside the observer lifecycle.

All callback wavefunction and potential arguments have `intent(in)` and are
borrowed only for the callback duration. An implementation must use allocatable
assignment or another deep copy, never retain pointers into tblite state.
Observer instances are calculation-local; a ragged batch must use one observer
per system rather than module-global state.

Restricted calculations use the captured Hamiltonian directly. In tblite's
unrestricted solver the assembled matrix is multiplied by two before
diagonalization, so version-2 unrestricted traces distinguish assembled and
solver Hamiltonians. The source comment claiming five quadrupole components is
stale; `qpat` is allocated with six components in packed order
`xx,xy,yy,xz,yz,zz`.

The unrestricted observer bundle is separate from the v1 patch:

- patch: `tblite-e9abc395-scc-observer-v2.patch`;
- patch SHA-256:
  `e8291e9ff6a5624a48c760bb246c8a12f41c710c02f71c81da9edd2ea9561c21`;
- metadata: `metadata-v2.json`, which also pins the v2 recorder and main-source
  hashes.

The v2 recorder installs tblite's production `spin_polarization` container in
addition to creating a two-spin wavefunction. This distinction is essential:
`nspin=2` selects alpha/beta orbital storage, while the container supplies the
magnetization potential and `1/2 m^T W m` energy used by unrestricted GFN2.

## Safe application and validation

The validator never patches `--source-root`. It clones the pinned commit into a
temporary directory, applies the patch to the clone's index, checks the staged
diff and hook ordering, verifies reverse application, and confirms that the
source checkout's HEAD and porcelain status did not change:

```bash
python3 tools/oracle/tblite_scc_trace/validate_observer_patch.py \
  --source-root /path/to/tblite
```

To retain a patched detached checkout, provide a path which does not exist and
is outside the source tblite checkout:

```bash
python3 tools/oracle/tblite_scc_trace/validate_observer_patch.py \
  --source-root /path/to/tblite \
  --output-dir build/oracle/tblite-observed
```

The output clone intentionally has the exact patch staged and no commit.

## Numerical probe

`observer_probe.f90` is built from a disposable outer Meson project linked to
the patched tblite subproject. It uses a hand-written H3+ input and does not
depend on tblite's mstore test suite. The probe checks:

- absent observer versus the concrete no-op observer, bit for bit;
- recording observer versus the unobserved final energy and wavefunction;
- paired, monotonic pre/post callbacks over multiple iterations;
- `H*C = S*C*epsilon` for the first captured pre-solve Hamiltonian;
- reconstruction of tblite's first-iteration residual RMS from mixed/raw q/d/Q;
- exactly one converged terminal callback;
- `max_iter=1` produces a complete iteration followed by max-iterations status;
- invalid mixer selector `scf=0` produces no iteration callbacks and exactly one
  failed terminal callback at iteration zero.

On the xTBloom development host, the full command is:

```bash
export FC=/group/software/deepmd-kit-3.1.1/bin/x86_64-conda-linux-gnu-gfortran
export CC=/group/software/deepmd-kit-3.1.1/bin/x86_64-conda-linux-gnu-gcc
export PATH=/home/jzzeng/miniconda3/pkgs/ninja-1.13.2-h171cf75_0/bin:$PATH
export PYTHONPATH=/home/jzzeng/miniconda3/pkgs/meson-1.11.2-pyhcf101f3_0/site-packages
export LIBRARY_PATH=/group/software/deepmd-kit-3.1.1/lib
export LD_LIBRARY_PATH=/group/software/deepmd-kit-3.1.1/lib

/home/jzzeng/miniconda3/bin/python \
  tools/oracle/tblite_scc_trace/validate_observer_patch.py \
  --source-root /home/jzzeng/codes/tblite \
  --probe \
  --lapack mkl-rt
```

The standalone probe still follows tblite's wrap files.  Corpus generation,
described below, instead binds every fallback to a reviewed local commit and
runs Meson with downloads disabled.

## Canonical SCC trace format (`xtbloom-scc-trace-v1`)

`xtbloom_scc_trace.py` implements the versioned interchange format that the
observer recordings will be serialized to and that the CPU/CUDA conformance
tests consume (issue #47). It is pure standard library; it runs before either
Fortran reference is built.

- `xtbloom-scc-trace-v1.schema.json` — the Draft 7 machine-readable JSON
  Schema.
- `xtbloom_scc_trace.py` — canonical writer + structural validation:
  - `validate(trace)` rejects unsupported format versions, malformed
    dimensions, non-finite floats, unpinned provenance, inconsistent q/d/Q
    residuals, and terminal/iteration mismatches with actionable messages;
  - `dumps(trace)` emits canonical JSON (sorted keys, at least 17 significant
    decimal digits per float, trailing newline) so that reading and re-writing
    a trace is byte-identical.

Version 1 is deliberately restricted-only: `input.spin_channels` is one.
`input.unpaired_electrons` is a nonnegative integer and may be nonzero for
tblite's restricted shared-orbital open-shell calculations. Matrices use
logical `[spin=1][row][column]` order. The exception is occupations: tblite
allocates `focc[nao,max(2,nspin)]`, so restricted traces retain both alpha and
beta channels as `[2][nao]`. Issue #51 adds the separate v2 contract below for
unrestricted charge/magnetization arrays and distinct assembled/solver
Hamiltonians while keeping restricted v1 documents valid.

## Unrestricted SCC trace format (`xtbloom-scc-trace-v2`)

Version 2 is a separate schema and corpus contract. It keeps the v1 scalar and
terminal meanings, but represents population and orbital channels explicitly:

- q/d/Q arrays are `[charge, magnetization]`, matching tblite's SCC mixer;
- eigenvalues, occupations, density, and Hamiltonians are `[alpha, beta]`;
- `assembled_hamiltonian` is the matrix after the charge/magnetization
  potentials are assembled;
- `solver_hamiltonian` is the exact alpha/beta matrix passed to the
  eigensolver and must equal `2 * assembled_hamiltonian` elementwise for this
  pinned tblite unrestricted implementation.

Residual flattening is field-major with charge/magnetization channels inside
each field: `qsh(charge, magnetization)`, then `d(charge, magnetization)`, then
`Q(charge, magnetization)`. Its dimension is
`spin_channels * (n_shells + 9*n_atoms)`. The production replay harness uses
the same mixed charge/magnetization state while its eigensolver and density
outputs remain alpha/beta resolved.

### Shapes, ordering, and units

All arrays use the logical order below rather than the recorder's Fortran
memory layout. Values use atomic units unless stated otherwise.

| Field | Logical shape | Meaning and unit |
| --- | --- | --- |
| `input.atomic_numbers` | `[n_atoms]` | Nuclear atomic number, dimensionless |
| `input.positions` | `[n_atoms * 3]` | Atom-major Cartesian `x,y,z`, bohr |
| `input.molecular_charge` | scalar | Charge in units of positive elementary charge |
| `input.temperature` | scalar | Target electronic temperature, kelvin |
| `input.point_charges.positions` | `[n_pc * 3]` | Point-major Cartesian `x,y,z`, bohr |
| `input.point_charges.charges` | `[n_pc]` | External charge, positive elementary charge |
| `input.point_charges.hardnesses` | `[n_pc]` | Positive point-site `gamma_p`, Hartree |
| `statics.overlap` | `[1][nao][nao]` | AO overlap `S`, dimensionless |
| `statics.core_hamiltonian` | `[1][nao][nao]` | Bare AO Hamiltonian `H0`, Hartree |
| `hamiltonian` | `[1][nao][nao]` | Effective Hamiltonian assembled from mixed q/d/Q, Hartree |
| `eigenvalues` | `[1][nao]` | Restricted orbital eigenvalues, Hartree |
| `occupations` | `[2][nao]` | Alpha then beta orbital occupations, electrons |
| `density` | `[1][nao][nao]` | AO density electron-occupation weights, dimensionless |
| `mixed_qsh`, `raw_qsh` | `[1][n_shells]` | Shell Mulliken charges, elementary charge |
| `mixed_qat`, `raw_qat` | `[1][n_atoms]` | Atom charges derived from qsh, elementary charge |
| `mixed_dipoles`, `raw_dipoles` | `[1][n_atoms][3]` | Cartesian `x,y,z`, elementary-charge bohr |
| `mixed_quadrupoles`, `raw_quadrupoles` | `[1][n_atoms][6]` | `xx,xy,yy,xz,yz,zz`, elementary-charge bohr squared |
| `energy` | scalar | `sum(eelec)` after a completed attempt, Hartree |
| `energy_delta` | scalar | `sum(eelec) - elast`, Hartree |

The mixer residual is exactly `raw - mixed`, flattened as the restricted
Fortran arrays `qsh(:)`, then `dpat(:,:)`, then `qpat(:,:)`. In logical JSON
terms this is every shell charge, then atom-major dipole components, then
atom-major quadrupole components. `qat` is derived from qsh and is not part of
the mixer vector.

The residual deliberately combines elementary charge, elementary-charge bohr,
and elementary-charge bohr-squared components. Its `residual_rms` is tblite's
unweighted numerical value

```text
sqrt(sum(residual[i]**2 / len(residual) for i in residual))
```

and therefore has no single physical unit. Runtime validation reconstructs the
vector from mixed/raw q/d/Q and permits only arithmetic roundoff in the RMS,
not a scientific comparison tolerance.

### Completed iterations, convergence, and failures

Each `iterations` entry pairs one `before_solve` payload with the matching
successful `after_iteration` payload. `convergence.energy`, `.population`, and
`.temperature` are the three flags passed by tblite; `.overall` is their
logical conjunction. Per-iteration status is intentionally absent because
`after_iteration` does not provide one. Only `finished` supplies terminal
status: `1` converged, `2` maximum iterations, or `3` failed.

`iterations` may be empty. A mixer-construction failure is represented by
`terminal = {status: 3, converged: false, iterations: 0}`. A later
`mixer%next` failure occurs before the next counter increment and before
`before_solve`, so it has no extra attempt payload and the terminal count
equals the number of completed entries.

If `before_solve` runs but the eigensolver fails before `after_iteration`, the
optional root `failed_attempt` stores only its index, Hamiltonian, and mixed
q/d/Q. It must not contain stale eigenvalues, occupations, density, raw q/d/Q,
residual, energy, or convergence. In this case `terminal.iterations` is the
number of completed entries plus one; otherwise it equals the completed count.
`terminal.converged` is true exactly for status 1, which requires a final
completed entry with `convergence.overall=true`.

`tests/oracle/test_trace_writer.py` keeps the writer and schema synchronized
with two-atom complete, multi-iteration, point-charge, and failure fixtures,
independently of tblite. When the test environment provides `jsonschema`, the
canonical writer outputs are also checked with a Draft 7 validator.

## SCC trace comparator foundation

`xtbloom_scc_compare.py` validates and compares complete traces or one
standalone iteration snapshot.  It is intentionally a read-only comparison
tool: it never generates or updates a golden file.  The replay harness
(`xtbloom_scc_trace_replay`) injects a golden mixed state, executes exactly one
production CPU driver iteration, and emits the snapshot; the wrapper compares
it with `compare_iteration` using the `cpu_replay_v1` profile. The replay plan
caps the driver at that logical iteration after seeding its counter, so a
nonconverged one-step replay reaches a real maximum-iteration terminal instead
of acquiring terminal metadata only during serialization. The wrapper checks
that lifecycle before extracting the standalone iteration. The
independent mixer harness (`xtbloom_scc_trace_mixer`) additionally replays the
pinned golden residual sequence through the production Broyden mixer alone so
a self-consistent flatten-order defect cannot hide behind a matching physical
trajectory.

The built-in tolerance policies have stable versioned identifiers:

- `cpu_closed_loop_v1` compares complete CPU trajectories from the same
  initial guess with default `(atol=1e-8, rtol=1e-9)`, residual overrides of
  `(1e-7, 1e-7)`, and an energy override of `(1e-8, 1e-8)`;
- `cpu_replay_v1` compares one CPU iteration executed separately from its
  injected golden mixed state (same magnitudes as the closed-loop profile);
- `cuda_replay_v1` compares one independently executed iteration with default
  `(atol=1e-9, rtol=1e-10)` and residual overrides of `(1e-8, 1e-8)`.

These version-1 values are anchored by the pinned corpus and manifest.  Every
numeric field uses
`abs(actual - expected) <= atol + rtol * max(abs(actual), abs(expected))`.
Dimensions, provenance pins, layouts, iteration indices/convergence flags, and
terminal status/count metadata are exact.  The documentary
`provenance.oracle_command` string is intentionally ignored: it legitimately
differs between the generator that produced the golden and an independent CPU
capture of the same science.

Compare two complete traces:

```bash
python tools/oracle/tblite_scc_trace/xtbloom_scc_compare.py trace \
  actual.json golden.json \
  --profile cpu_closed_loop_v1 \
  --golden-sha256 <canonical-golden-sha256>
```

Compare an externally produced iteration snapshot to logical iteration 3 of a
complete golden:

```bash
python tools/oracle/tblite_scc_trace/xtbloom_scc_compare.py iteration \
  actual-iteration.json golden.json \
  --iteration 3 \
  --profile cuda_replay_v1
```

The exit-code contract is `0` for a match, `1` for a scientific mismatch, and
`2` for CLI, JSON, schema, profile, canonicalization, or hash errors. Golden
files are opened only for reading. Golden generation and manifest/hash updates
belong to the separate issue #48 workflow.

## Pinned restricted corpus generation (issues #45/#46/#48)

`scc_trace_recorder.f90` + `scc_trace_main.f90` build the oracle recorder
executable that drives the patched pinned tblite GFN2 single point through the
observer seam and streams every `xtbloom-scc-trace-v1` field in a fixed raw
layout.  `generate_scc_corpus.py` is the reproducible corpus pipeline:

1. validates the immutable observer-patch bundle and the local tblite source
   checkout (clean tree, pinned revision reachable);
2. verifies that the local dependency object stores contain every reviewed
   full 40-hex commit, then creates detached clones at exactly those commits;
3. builds the recorder oracle in a disposable outer Meson project with
   `--wrap-mode=nodownload`, forces every dependency to the local fallback,
   and checks Meson's recursive subproject introspection (the build includes
   the oracle-only shell-monopole PCEM container for QM/MM-like cases);
4. runs the five restricted corpus cases and canonicalizes each raw stream
   with `xtbloom_scc_trace.py`, and
5. writes canonical JSON goldens and specs plus a `manifest.json` with their
   SHA-256 digests, every dependency revision, both Fortran source digests,
   the observer patch digest, deterministic environment, exact compiler
   executable and Meson identities, resolved BLAS/LAPACK provider and library
   digests, and the redacted command line.

Regenerating with the same toolchain produces byte-identical goldens (the
manifest records the generation timestamp separately).  Verify committed
goldens without rebuilding:

```bash
python tools/oracle/tblite_scc_trace/generate_scc_corpus.py \
  --source-root /path/to/tblite \
  --corpus-dir data/conformance/scc-traces --check
```

The offline check validates the complete manifest shape and fixed format and
revision, repository-owned patch and source bytes, reviewed dependency pins,
toolchain records, canonical generator-owned specs, canonical trace bytes and
hashes, and agreement between trace and manifest provenance.  It intentionally
does not require a tblite checkout or access the network.

The corpus currently contains `h3_plus`, `ketene`, `nenacl`, and the two
point-charge water cases `water_one_pc_gamma999` and
`water_dimer_6pc_hardness`.  Point-charge cases additionally record the
per-shell `point_charge_shell_potential` (V^PC) and `point_charge_energy`
(q_s V^PC) in every completed iteration (issue #46); plain cases omit them.

## Pinned unrestricted corpus generation (issue #51)

`generate_unrestricted_scc_corpus.py` owns the v2 OH-radical spec, recorder,
schema, and `manifest-v2.json`. It reuses the same pinned tblite revision,
reviewed dependency commits, offline Meson build, deterministic environment,
compiler, and LP64 BLAS/LAPACK provenance as the restricted generator without
rewriting `manifest.json` or any v1 golden.

Generate the unrestricted corpus with the pinned local toolchain:

```bash
export FC=/group/software/deepmd-kit-3.1.1/bin/x86_64-conda-linux-gnu-gfortran
export CC=/group/software/deepmd-kit-3.1.1/bin/x86_64-conda-linux-gnu-gcc
export PATH=/home/jzzeng/miniconda3/pkgs/ninja-1.13.2-h171cf75_0/bin:$PATH
export PYTHONPATH=/home/jzzeng/miniconda3/pkgs/meson-1.11.2-pyhcf101f3_0/site-packages
export LIBRARY_PATH=/group/software/deepmd-kit-3.1.1/lib
export LD_LIBRARY_PATH=/group/software/deepmd-kit-3.1.1/lib

/home/jzzeng/miniconda3/bin/python \
  tools/oracle/tblite_scc_trace/generate_unrestricted_scc_corpus.py \
  --source-root /home/jzzeng/codes/tblite \
  --corpus-dir data/conformance/scc-traces \
  --meson-command '/home/jzzeng/miniconda3/bin/python -m mesonbuild.mesonmain' \
  --lapack mkl-rt --wrap-mode nodownload
```

The offline integrity gate is:

```bash
python3 tools/oracle/tblite_scc_trace/generate_unrestricted_scc_corpus.py \
  --source-root /nonexistent \
  --corpus-dir data/conformance/scc-traces --check
```

The committed `oh_radical` trace converges in 10 iterations. Its first raw
magnetization is independently checked against the pinned oxygen spin constants
so a recorder that creates two orbital channels but omits spin polarization is
rejected. The existing restricted goldens and manifest retain hard-coded
SHA-256 identity checks.

`xtbloom_scc_cpu_trace.py` drives the production CPU GFN2 SCC driver through the
versioned v1/v2 corpora and compares captured trace documents against the
goldens with the matching profile. It is the executable evidence harness for
issues #49/#50/#51 with four modes:

- `--capture`: one case at a time, comparing each complete closed-loop
  trajectory against the pinned golden with `cpu_closed_loop_v1` or
  `cpu_closed_loop_v2`. The capture
  driver includes the self-consistent D4 two-body atom potential (which is
  nonzero even at the zero-charge first mixed state) and starts every system
  from tblite's zero-charge perturbative q/d/Q seed;
- `--batch-capture`: several cases in one ragged driver batch.  Every healthy
  lane must equal its pinned sequential trajectory, systems which converge
  early stop mutating while slower peers continue, and a controlled
  per-system failure lane (NaN in one lane's H0) must neither corrupt nor
  suppress the successful members;
- `--replay`: every golden iteration is replayed from its injected mixed
  q/d/Q state and compared with the matching `cpu_replay_v1` or
  `cpu_replay_v2` single-iteration profile.
  A divergence is therefore assigned to the exact iteration where it first
  appears instead of inheriting Broyden drift; an injected perturbation in a
  later iteration is reported only at that iteration. If the eigensolver fails
  after Hamiltonian assembly, the emitted trace retains only the failed
  attempt's Hamiltonian and mixed q/d/Q for exact failure localization;
- `--mixer`: `xtbloom_scc_trace_mixer` replays the PINNED golden residual
  sequence (raw minus mixed per iteration, in the canonical flatten order)
  through xTBloom's production Broyden mixer alone — no driver, no eigensolver.
  Every state transition must reproduce the golden next mixed state, which
  isolates the mixer's flatten order, damping, history, and transitions from
  the physical trajectory.

The native gates behind these modes are registered as CTest tests
(`xtbloom.gfn2.scc_trace_cpu_closed_loop`, `xtbloom.gfn2.scc_trace_ragged_batch`,
`xtbloom.gfn2.scc_trace_cpu_replay`, `xtbloom.gfn2.scc_trace_mixer_replay`, and
`xtbloom.gfn2.scc_trace_ragged_batch_native`) and are passing acceptance gates
for restricted and unrestricted sequential, replay, mixer-history, and mixed
restricted/unrestricted ragged execution. The mixed test includes a poisoned
unrestricted peer and proves failure isolation plus early-lane freezing across
different wavefunction layouts.
