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
`xtbout.json`. The runner copies each input to an isolated `coord` file and
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

`compare` also accepts raw tblite JSON (`energy` plus `gradient`) and
gpuxtb-style JSON (`energy_hartree` plus `forces_hartree_per_bohr`). SCC-gated
cases additionally require `partial_charges_e`, `atomic_dipoles_e_bohr`, and
`atomic_quadrupoles_e_bohr2`. Each case must be named `<case-id>.json`. This
makes the same comparison entry usable by the reference generators and future
C API integration tests.
