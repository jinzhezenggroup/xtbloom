# GFN2-xTB conformance tools

The corpus is deliberately independent of the gpuxtb implementation. Its
committed seed values come from tblite's validation suite at the exact revision
recorded in `data/conformance/manifest.json`. Coordinates, energies, gradients,
and forces use atomic units; forces are stored explicitly as `-gradient`.

Verify that the committed files still match the manifest:

```sh
python3 tools/conformance/gpuxtb_conformance.py check
```

Recreate the corpus from the pinned tblite source checkout, then update the
input and golden SHA-256 values in the manifest as an intentional review step:

```sh
python3 tools/conformance/gpuxtb_conformance.py import-tblite-snapshot \
  --source-root /path/to/tblite
```

Generate independent values with a built tblite executable. Output goes to a
separate directory so it cannot silently replace reviewed goldens:

```sh
python3 tools/conformance/gpuxtb_conformance.py generate \
  --executable /path/to/tblite --output-dir build/conformance/tblite
python3 tools/conformance/gpuxtb_conformance.py compare \
  --actual-dir build/conformance/tblite
```

The xtb 6.7.1 adapter is the supported live second oracle. It combines the
high-precision energy and Cartesian gradient from xtb's `gradient` artifact
with partial charges, atomic dipoles, and atomic quadrupoles from
`xtbout.json`. Gas-phase inputs are copied to an isolated `coord` file, while
QM/MM JSON is deterministically materialized as described below. The runner
forces OpenMP, OpenBLAS, and xtb itself to one thread, so temporary paths and
thread scheduling do not enter the normalized scientific-output checksum. The
runner also rejects executables whose reported version/revision does not match
the oracle pinned in the manifest:

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
tolerance. This records the observed xtb/tblite analytic-gradient spread
without weakening the tighter gpuxtb acceptance gate, which applies to normal
implementation outputs that do not claim to be a reference engine.

`oh_radical` is the initial unrestricted open-shell gate (`--uhf 1`). Its
committed xtb golden requires the atom-resolved SCC state in addition to energy
and forces. Atomic quadrupoles use xtb's `xx, xy, yy, xz, yz, zz` ordering.
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

Build a shared library, then submit all supported restricted cases as one
host-descriptor ragged batch through the exported C ABI:

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
used unchanged. Energy and QM forces are gated for every supported case;
QM/MM goldens also gate atomic charges and point-charge forces. `oh_radical`
is explicitly skipped because its nonzero unpaired-electron count requires
unrestricted SCC, which the current public GFN2 path does not support. Atomic
dipoles and quadrupoles are not compared because C API version 1 has no result
buffers for them.

`--memory-mode device` places every nonempty input and output descriptor in
CUDA memory. `mixed` leaves topology offsets, atomic numbers, energies,
charges, SCC iterations, and per-system status on the host; numerical geometry
and point-charge inputs plus QM/point-charge forces and `scc_converged` use
CUDA pointers. The runner dynamically loads libcudart, performs explicit
host/device copies, frees every allocation on success or failure, and restores
the entry CUDA device. CPU inference accepts only `--memory-mode host`.
