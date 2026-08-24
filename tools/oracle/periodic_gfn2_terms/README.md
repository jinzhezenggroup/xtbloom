# Periodic GFN2 term oracle

This directory contains a standalone Fortran probe for the exact tblite
revision `133f91efb94b47f05848e1f86832f40a1accc385`. It extracts numerical
fixtures before SCC composition for the six term families required by issue
#472:

- GFN2 double-exponential and D4 coordination numbers;
- screened nuclear repulsion;
- fixed-charge D4 pair and non-self-consistent ATM contributions;
- overlap, dipole, quadrupole, and zeroth-order Hamiltonian matrices;
- shell-resolved periodic charge Ewald electrostatics; and
- periodic q/d/Q multipole electrostatics.

The probe is independent of xTBloom and links directly to the pinned tblite
Fortran modules. Its line protocol records logical array shapes and preserves
Fortran column-major storage exactly. The normalized JSON corpus documents AO
and shell mappings, quadrupole packing, atomic units, source revisions, build
inputs, compiler/module/runtime hashes, and raw-output hashes. The probe is
repository-authored GPL-3.0-or-later code; tblite remains
LGPL-3.0-or-later. The manifest references the shared tblite source-build
attestation and records only non-system shared libraries from the probe's live
loader closure. The local probe executable is deliberately not hash-pinned:
the reviewed gfortran driver embeds local source and environment paths in its
ELF strings and RPATH. Instead, the path-independent manifest pins the probe
source and build script, compiler and module bytes, shared tblite build
attestation, and resolved non-system runtime closure.

Build and regenerate into the committed corpus only after reviewing the exact
oracle installation:

```bash
bash tools/oracle/periodic_gfn2_terms/build_probe.sh
python3 tools/oracle/periodic_gfn2_terms/periodic_gfn2_terms.py generate \
  --probe build/periodic-gfn2-terms/probe \
  --output-dir data/conformance/periodic/terms
```

The default validation is offline and does not load tblite or xTBloom:

```bash
python3 tools/oracle/periodic_gfn2_terms/periodic_gfn2_terms.py check
python3 -m unittest discover -s tests/oracle -p test_periodic_gfn2_terms.py -v
```

For an installed pinned oracle, regenerate in a temporary directory and
byte-compare all raw outputs, normalized fixtures, finite differences, and
attestation data:

```bash
python3 tools/oracle/periodic_gfn2_terms/periodic_gfn2_terms.py compare \
  --probe build/periodic-gfn2-terms/probe
```

Primary charge fixtures are neutral because tblite's charged periodic Ewald
implementation omits xTBloom's required uniform-background constant. The
multipole fixture retains dipole-only, charge-plus-quadrupole, and full q/d/Q
states; tblite's reviewed direct multipole cutoff is the pinned 100-bohr
implementation value.
