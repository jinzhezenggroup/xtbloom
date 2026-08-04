# Pinned tblite SCC observer seam

This directory carries the minimal oracle-only patch used to observe tblite's
GFN2 SCC loop. It targets exactly
`e9abc395b122018ed688aecb1c3a65cecaf97beb`; it is not a patch for arbitrary
tblite releases and it is not part of gpuxtb's public C ABI.

The patch is licensed `LGPL-3.0-or-later`, matching the files it modifies in
tblite. Verbatim GPLv3 and LGPLv3 texts from the pinned checkout are bundled as
`LICENSES/GPL-3.0.txt` and `LICENSES/LGPL-3.0.txt`. The validation script,
metadata, probe, and tests are original gpuxtb tooling; their repository-wide
license remains tracked by #19 rather than being selected implicitly here.

## Immutable inputs

- Upstream: <https://github.com/tblite/tblite.git>
- Revision: `e9abc395b122018ed688aecb1c3a65cecaf97beb`
- Patch: `tblite-e9abc395-scc-observer.patch`
- Patch SHA-256:
  `d6a51afc4b3c56d6589a2b5b115ea8b4891600c1161c525939ca3cc16e2b4954`

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
diagonalization, so future unrestricted traces must distinguish assembled and
solver Hamiltonians. The source comment claiming five quadrupole components is
stale; `qpat` is allocated with six components in packed order
`xx,xy,yy,xz,yz,zz`.

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

On the gpuxtb development host, the full command is:

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

Meson fallback dependencies may require network access on the first run. Issue
#45 will turn this validation build into the fully pinned corpus-generation
toolchain; #44 only freezes and validates the observer seam itself.

## Canonical SCC trace format (`gpuxtb-scc-trace-v1`)

`gpuxtb_scc_trace.py` implements the versioned interchange format that the
observer recordings will be serialized to and that the CPU/CUDA conformance
tests consume (issue #47). It is pure standard library; it runs before either
Fortran reference is built.

- `gpuxtb-scc-trace-v1.schema.json` — the machine-readable JSON Schema.
- `gpuxtb_scc_trace.py` — canonical writer + structural validation:
  - `validate(trace)` rejects unsupported format versions, malformed
    dimensions, non-finite floats, unpinned provenance, and terminal/iteration
    mismatches with actionable messages;
  - `dumps(trace)` emits canonical JSON (sorted keys, at least 17 significant
    decimal digits per float, trailing newline) so that reading and re-writing
    a trace is byte-identical.

Matrices are stored in logical `[spin][row][column]` order, multipoles in
`[spin][atom][component]`, and quadrupoles in the `xx, xy, yy, xz, yz, zz`
packing. The mixer residual is shell charges followed by atomic dipoles and
atomic quadrupoles in the exact tblite flattening order.

`tests/oracle/test_trace_writer.py` keeps the writer and schema synchronized
with synthetic fixtures, independently of tblite.
