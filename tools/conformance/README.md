# GFN2-xTB conformance tools

The corpus is deliberately independent of the gpuxtb implementation. Four
closed-shell gas-phase cases use a pinned live tblite calculation as their
primary oracle; the open-shell OH case and three QM/MM cases use pinned xTB
6.7.1. Both command lines explicitly set `--acc 0.0001`: the looser CLI
defaults can leave SCC charge and force residuals larger than gpuxtb's primary
`5e-7` acceptance threshold. Coordinates, energies, gradients, and forces use
atomic units; forces are stored explicitly as `-gradient`.

Verify that the committed files still match the manifest:

```sh
python3 tools/conformance/gpuxtb_conformance.py check
```

The snapshot importer remains available to audit the historical validation
inputs at the pinned tblite source revision. Snapshot outputs are not the
primary goldens because their original convergence setting was not recorded:

```sh
python3 tools/conformance/gpuxtb_conformance.py import-tblite-snapshot \
  --source-root /path/to/tblite
```

Regenerate the four tblite-primary gas cases with a built executable from the
pinned revision. The generator verifies tblite 0.7.0, records the resolved
`libtblite` hash, forces a deterministic single-threaded environment, and
stores the reviewed accuracy in provenance. Output goes to a separate
directory so it cannot silently replace reviewed goldens:

```sh
python3 tools/conformance/gpuxtb_conformance.py generate \
  --executable /path/to/tblite --output-dir build/conformance/tblite
python3 tools/conformance/gpuxtb_conformance.py compare \
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
python3 tools/conformance/gpuxtb_conformance.py generate-xtb \
  --executable /path/to/xtb --output-dir build/conformance/xtb
python3 tools/conformance/gpuxtb_conformance.py compare \
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
`gpuxtb-xtb-pcem-cli-v1`. The exact command names those derived files rather
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
python3 tools/conformance/gpuxtb_conformance.py generate-xtb \
  --executable /path/to/xtb --output-dir build/conformance/xtb-qmmm \
  --case water_one_pc_gamma999 \
  --case water_dimer_6pc_hardness \
  --case water_dimer_6pc_gamma999
```

When a generated result and committed golden explicitly identify different
reference engines, `compare` uses the manifest's separate cross-engine force
tolerance. This preserves the live tblite/xTB diagnostic without weakening the
tighter primary gate applied to gpuxtb and each case's designated oracle.
Explicit `--case` selections may use a non-primary live engine for this check;
for example, an xTB cross-check of the tblite-primary ketene case is:

```sh
python3 tools/conformance/gpuxtb_conformance.py generate-xtb \
  --executable /path/to/xtb --output-dir build/conformance/xtb-cross \
  --case ketene
python3 tools/conformance/gpuxtb_conformance.py compare \
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

`compare` also accepts raw tblite JSON (`energy` plus `gradient`) and
gpuxtb-style JSON (`energy_hartree` plus `forces_hartree_per_bohr`). SCC-gated
cases additionally require `partial_charges_e`, `atomic_dipoles_e_bohr`, and
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
  python3 tools/conformance/gpuxtb_public_api.py \
    --library build/libgpuxtb.so --backend cpu --memory-mode host \
    --actual-dir build/conformance/gpuxtb-public

srun --gres=gpu:1 env \
  LD_LIBRARY_PATH=/path/to/mkl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  python3 tools/conformance/gpuxtb_public_api.py \
    --library build/libgpuxtb.so --backend cuda --memory-mode device \
    --actual-dir build/conformance/gpuxtb-public

srun --gres=gpu:1 env \
  LD_LIBRARY_PATH=/path/to/mkl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  python3 tools/conformance/gpuxtb_public_api.py \
    --library build/libgpuxtb.so --backend cuda --memory-mode mixed \
    --actual-dir build/conformance/gpuxtb-public
```

Actual JSON is written before comparison. The primary manifest tolerances are
used unchanged. Energy and QM forces are gated for every case; QM/MM goldens
also gate atomic charges and point-charge forces. `oh_radical` uses the standard
shared-orbital (`spin_channels=1`) xTB semantics and gates energy, force, and
atom-resolved charges on both backends. Spin-polarized (`spin_channels=2`)
inference and analytic forces are exercised on CPU and CUDA separately until an
independently generated spin-polarized golden is committed. Atomic dipoles and
quadrupoles are not compared because the current C result ABI has no output
buffers for them.

`--memory-mode device` places every nonempty input and output descriptor in
CUDA memory. `mixed` leaves topology offsets, atomic numbers, energies,
charges, SCC iterations, and per-system status on the host; numerical geometry
and point-charge inputs plus QM/point-charge forces and `scc_converged` use
CUDA pointers. The runner dynamically loads libcudart, performs explicit
host/device copies, frees every allocation on success or failure, and restores
the entry CUDA device. CPU inference accepts only `--memory-mode host`.
