# GFN2-xTB conformance tools

The corpus is deliberately independent of the xTBloom implementation. Four
closed-shell gas-phase cases use a pinned live tblite calculation as their
primary oracle; the open-shell OH case and three QM/MM cases use pinned xTB
6.7.1. Both command lines explicitly set `--acc 0.0001`: the looser CLI
defaults can leave SCC charge and force residuals larger than xTBloom's primary
`5e-7` acceptance threshold. Coordinates, energies, gradients, and forces use
atomic units; forces are stored explicitly as `-gradient`.

Verify that the committed files still match the manifest:

```sh
python3 tools/conformance/xtbloom_conformance.py check
```

The snapshot importer remains available to audit the historical validation
inputs at the pinned tblite source revision. Snapshot outputs are not the
primary goldens because their original convergence setting was not recorded:

```sh
python3 tools/conformance/xtbloom_conformance.py import-tblite-snapshot \
  --source-root /path/to/tblite
```

Regenerate the four tblite-primary gas cases with a built executable from the
pinned revision. The generator verifies tblite 0.7.0, records the resolved
`libtblite` hash, forces a deterministic single-threaded environment, and
stores the reviewed accuracy in provenance. Output goes to a separate
directory so it cannot silently replace reviewed goldens:

```sh
python3 tools/conformance/xtbloom_conformance.py generate \
  --executable /path/to/tblite --output-dir build/conformance/tblite
python3 tools/conformance/xtbloom_conformance.py compare \
  --actual-dir build/conformance/tblite
```

The xTB 6.7.1 adapter is the primary oracle for OH and QM/MM and remains an
independent live cross-engine oracle for gas-phase diagnostics. It combines the
high-precision energy and Cartesian gradient from xtb's `gradient` artifact
with partial charges, atomic dipoles, and atomic quadrupoles from
`xtbout.json`. Gas-phase inputs are copied to an isolated `coord` file, while
QM/MM JSON is deterministically materialized as described below. The runner
forces OpenMP, OpenBLAS, and xtb itself to one thread, so temporary paths and
thread scheduling do not enter the normalized scientific-output checksum. The
runner also rejects executables whose reported version/revision does not match
the oracle pinned in the manifest. Its command and provenance use the same
explicit `--acc 0.0001` contract as tblite:

```sh
python3 tools/conformance/xtbloom_conformance.py generate-xtb \
  --executable /path/to/xtb --output-dir build/conformance/xtb
python3 tools/conformance/xtbloom_conformance.py compare \
  --actual-dir build/conformance/xtb
```

The corpus also contains versioned QM/MM inputs ending in `.qmmm.json`.  Each
document records QM atomic numbers, symbols, and positions plus every external
point charge's position, charge, and GFN2 hardness (`gamma`) in atomic units.
Symbols must be the exact canonical symbols for the corresponding atomic
numbers. `gamma_mode=explicit` stores standalone gamma values and forbids a
source element, while `gamma_mode=element_hardness` requires source atomic
numbers and verifies every gamma against the pinned GFN2 hardness table.

The live xTB runner validates that document before materializing isolated
`coord`, `pcharge`, and `xcontrol` files with schema
`xtbloom-xtb-pcem-cli-v1`. The exact command names those derived files rather
than pretending that xTB consumes the JSON directly. The machine-readable
input is copied verbatim into its golden; the manifest protects its file hash,
and provenance records the JSON hash plus all three materialized-file hashes.

For the pinned CLI environment, the runner clears all inherited `XTB*`
variables and explicitly sets `XTBPATH` to the directory containing the hashed
GFN2 parameter file. Provenance also records the actually resolved `libxtb`
and parameter-file hashes. Other environment variables remain inherited; that
boundary is stated explicitly in each golden rather than implying a fully
hermetic operating-system environment.

The initial QM/MM set contains a minimal water plus one point charge and the
official xTB water-tetramer PCEM regression represented as 6 QM atoms plus 6
point charges.  The latter is stored with both element-derived H/O hardnesses
and the `gamma=999` point-charge limit.  Regenerate only these cases with:

```bash
python3 tools/conformance/xtbloom_conformance.py generate-xtb \
  --executable /path/to/xtb --output-dir build/conformance/xtb-qmmm \
  --case water_one_pc_gamma999 \
  --case water_dimer_6pc_hardness \
  --case water_dimer_6pc_gamma999
```

When a generated result and committed golden explicitly identify different
reference engines, `compare` uses the manifest's separate cross-engine force
tolerance. This preserves the live tblite/xTB diagnostic without weakening the
tighter primary gate applied to xTBloom and each case's designated oracle.
Explicit `--case` selections may use a non-primary live engine for this check;
for example, an xTB cross-check of the tblite-primary ketene case is:

```sh
python3 tools/conformance/xtbloom_conformance.py generate-xtb \
  --executable /path/to/xtb --output-dir build/conformance/xtb-cross \
  --case ketene
python3 tools/conformance/xtbloom_conformance.py compare \
  --actual-dir build/conformance/xtb-cross --case ketene
```

`oh_radical` is the initial standard GFN2-xTB open-shell gate (`--uhf 1`). Its
committed xTB command does not enable the optional spin-polarization container,
so the manifest explicitly records `spin_channels: 1`: alpha and beta
occupations share one orbital set. The unpaired-electron count and number of
orbital spin channels are independent inputs. The golden requires the
atom-resolved SCC state in addition to energy and forces. Atomic quadrupoles use
xTB's `xx, xy, yy, xz, yz, zz` ordering.
The property names encode their atomic units: charge in elementary charge,
dipole in elementary-charge bohr, and quadrupole in elementary-charge bohr².
For QM/MM cases, `forces_hartree_per_bohr` is the QM force array and
`point_charge_forces_hartree_per_bohr` is the PC force array.  Both are stored
as the exact negative of their corresponding xTB gradient artifacts.  The
packaged xTB 6.7.1 CLI prints `pcgrad` less precisely than the QM `gradient`
artifact, so the corpus validates total QM+PC force conservation at a tolerance
consistent with that text precision.

`compare` also accepts raw tblite JSON (`energy` plus `gradient`) and flat
xTBloom-style JSON. The latter must contain every property required by the
selected case's xTBloom oracle-property set; ordinary cases therefore require
`energy_hartree` plus `forces_hartree_per_bohr`, while a documented
diagnostic-only force may be omitted. SCC-gated cases additionally require
`partial_charges_e`, `atomic_dipoles_e_bohr`, and
`atomic_quadrupoles_e_bohr2`. Each case must be named `<case-id>.json`. This
makes the same comparison entry usable by the reference generators and future
C API integration tests.

## Public C API conformance

Build a shared library, then submit property-compatible ragged batches through
the exported C ABI. Cases request energy, analytic forces, and any golden-backed
charge outputs. The CPU runner submits the manifest's explicit channel count
through the ABI-v2 `spin_channels` suffix:

```bash
env LD_LIBRARY_PATH=/path/to/mkl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  python3 tools/conformance/xtbloom_public_api.py \
    --library build/libxtbloom.so --backend cpu --memory-mode host \
    --actual-dir build/conformance/xtbloom-public

srun --gres=gpu:1 env \
  LD_LIBRARY_PATH=/path/to/mkl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  python3 tools/conformance/xtbloom_public_api.py \
    --library build/libxtbloom.so --backend cuda --memory-mode device \
    --actual-dir build/conformance/xtbloom-public

srun --gres=gpu:1 env \
  LD_LIBRARY_PATH=/path/to/mkl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  python3 tools/conformance/xtbloom_public_api.py \
    --library build/libxtbloom.so --backend cuda --memory-mode mixed \
    --actual-dir build/conformance/xtbloom-public
```

Actual JSON is written before comparison. The primary manifest tolerances are
used unchanged. Energy and QM forces are gated when named by each case's
xTBloom oracle-property set; QM/MM goldens also gate atomic charges and
point-charge forces. `oh_radical` uses the standard
shared-orbital (`spin_channels=1`) xTB semantics and gates energy, force, and
atom-resolved charges on both backends. Spin-polarized (`spin_channels=2`)
inference and analytic forces are exercised on CPU and CUDA separately until an
independently generated spin-polarized golden is committed. Molecular dipoles
are published for requested CPU calculations but are not yet part of the
golden comparison; atomic dipoles and quadrupoles likewise remain diagnostic
oracle state rather than public conformance outputs.

Case-level `xtbloom_backends` metadata keeps interactions on only the released
public backends. The `water_efield` pilot is CPU-only until #237 P3 implements
CUDA interaction execution, so CUDA host/device/mixed batches continue to run
the eight previously supported cases instead of failing the whole ragged call
with `NOT_IMPLEMENTED`. Its pinned tblite 0.7.0 energy remains an independent
oracle. The tblite analytic field gradient uses `+E` per atom instead of the
energy derivative `+q_i E`; that force array remains in the canonical golden
as diagnostic provenance but is excluded from xTBloom oracle comparison.
`xtbloom_invariants.py` central differences of the reported public energy are
the mandatory force evidence for this case.

`--memory-mode device` places every nonempty input and output descriptor in
CUDA memory. `mixed` leaves topology offsets, atomic numbers, energies,
charges, SCC iterations, and per-system status on the host; numerical geometry
and point-charge inputs plus QM/point-charge forces and `scc_converged` use
CUDA pointers. The runner dynamically loads libcudart, performs explicit
host/device copies, frees every allocation on success or failure, and restores
the entry CUDA device. CPU inference accepts only `--memory-mode host`.

## Numerical tolerances and CPU/CUDA agreement

The manifest records absolute (`atol`, `rtol = 0`) tolerances for total energy,
analytic forces, the SCC state (atomic charges, dipoles, quadrupoles), and
external point-charge forces, each with a property-specific `justification`
naming the observed xTBloom-to-oracle margin on the committed corpus. The
thresholds are absolute rather than relative because the corpus intentionally
mixes charged anions, tiny near-zero systems, and large energies where a
relative scale would grant unphysical slack. Behavior gates:

- Every property named by a case's xTBloom oracle-property set must match the
  same pinned live oracle to its own threshold; cases without that metadata
  retain the complete default property set. The cross-engine
  `cross_engine_tolerances` block is used only when both compared documents
  explicitly identify distinct independent reference engines; xTBloom results
  always use the primary tolerances.
- For cases released on both backends, CPU and CUDA must both satisfy the
  primary energy, forces, and charges thresholds (5e-7 each in atomic units),
  so a CPU/CUDA pair on identical inputs can deviate by at most twice that
  value (1e-6) by the triangle inequality. The manifest records this as
  `cpu_cuda_agreement`. CPU-only cases acquire the same parity gate when their
  CUDA execution path is released.
- Within one backend, execution is deterministic for identical descriptors and
  launch configuration: fresh-SCC results are bit-identical across repeated
  calls (the batch-versus-sequential gates below fail at 1e-12), which makes
  debugging reproducible. CUDA results are deterministic for a fixed launch
  geometry and Graph instantiation; compare debug runs backend-by-backend
  rather than mixing CPU/CUDA artifacts.

## Invariance, conservation, and batch-consistency gates

`xtbloom_invariants.py` runs the committed corpus through the same public C ABI
path as the golden runner and checks the exact symmetries an isolated finite
GFN2-xTB system must satisfy. These gates complement, and never replace, golden
comparison: no optimized implementation is accepted solely on agreement with
itself. Each backend is only ever compared with itself at transformed
geometries, so the gate tolerances measure one backend's numerical
reproducibility (measured CPU margins are recorded in the tool header) rather
than cross-engine physics differences. A genuine symmetry break produces errors
orders of magnitude above the gates (for example a translation break shifts
every force component by its full value).

```bash
python3 tools/conformance/xtbloom_invariants.py \
  --library build/libxtbloom.so --backend cpu --memory-mode host

srun --gres=gpu:1 python3 tools/conformance/xtbloom_invariants.py \
  --library build/libxtbloom.so --backend cuda --memory-mode device
```

The gates cover:

- **Batch versus sequential**: one heterogeneous ragged batch of every selected
  case must reproduce each case's sequential single-system solve; identical
  systems duplicated in one homogeneous ragged batch must reproduce the
  sequential solve.
- **Translation invariance**: energy, atomic charges, and analytic forces are
  invariant when the whole system (QM atoms and external point charges
  together) is displaced; tested for two deterministic translations.
- **Rotation covariance**: energy and atomic charges are invariant under a
  proper rotation, while QM and point-charge forces rotate with the structure;
  uniform electric fields rotate with the structure as Cartesian vectors;
  tested with a 37-degree axis rotation and an integer-exact 90-degree rotation
  about z.
- **Force conservation**: the net force on an isolated system vanishes
  componentwise (QM plus point-charge forces for QM/MM cases).
- **Charge conservation**: the summed atomic charges reproduce the declared
  molecular charge.
- **Central finite differences**: every selected case on each released
  backend, QM atom axis, and external point-charge axis is displaced by
  ``+-1e-3`` bohr in isolation, and the numeric force
  ``-(E(+)-E(-)) / (2e-3)`` must match the analytic force published for the
  undisplaced geometry (limit ``1e-5`` Ha/bohr for QM forces, ``1e-7``
  Ha/bohr for point-charge forces). This directly checks the force definition,
  including point-charge force signs, throughout the backend's supported
  corpus.

The invariance gates reuse the golden runner's strict single-shot options
(fresh SCC, charge tolerance 1e-10, energy tolerance 1e-12) so the two paths can
never diverge in convergence policy. Each gate emits one deterministic
`PASS`/`FAIL` line with the measured maximum error and the limit; any failure
makes the tool exit nonzero.

## Difficult-SCC fixture: tmacl ion pair

`data/conformance/inputs/tmacl.xyz` is a reproduced difficult-SCC input for
the separated `Me4N+ / Cl-` ion pair (18 atoms), copied verbatim from
grimme-lab/xtb issue #678. It is intentionally **not** a manifest golden: at
300 K the default Johnson modified-Broyden policy (history 8, damping 0.4)
does not converge under xTBloom or upstream xTB within 250 iterations, so there
is no oracle golden to compare against. The file is instead a machine-readable
fixture whose provenance, baseline status matrix, per-iteration scalar
diagnostics, bounded/coarse path matrix, and mixer-policy sweep are pinned by
`data/conformance/evidence/tmacl-temperature-continuation/manifest.json`.
The native CTest gate `xtbloom.gfn2.scc_temperature_continuation`
(`tests/scc_temperature_continuation_test.cpp`) drives the internal CPU GFN2
SCC driver, checks the exact baseline counts and executes the complete mixer
policy grid, requiring reviewed 300 K policies plus terminal q/d/Q and density
agreement for every successful 300 K cell and path. Its explicit
`--write-evidence` mode is the only supported text-evidence generator;
`tools/conformance/tmacl_evidence.py` verifies fixture, generator, and output
hashes without rewriting them.

This is internal CPU numerical evidence, not an independent oracle or a public
continuation contract. Analytic forces, per-iteration complete q/d/Q,
ordinary-corpus regressions, and the CPU/CUDA/public-ABI implementation matrix
remain open in #217. The current data does not by itself justify choosing
temperature continuation over a deterministic mixer-policy change.
