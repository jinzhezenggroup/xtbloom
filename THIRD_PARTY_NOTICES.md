# Third-party notices and provenance

gpuxtb is distributed under `GPL-3.0-or-later`; see `LICENSE`. The
following upstream material is retained under its own compatible terms.
The manifests named below pin the source revisions and content digests used
to produce the redistributed data.

## tblite

Repository: <https://github.com/tblite/tblite>

License: `LGPL-3.0-or-later` (`LICENSES/LGPL-3.0-or-later.txt`; the
incorporated GPLv3 terms are in the project `LICENSE`).

Redistributed or derived material:

- `data/parameters/gfn2.toml`, `gfn2.json`, and `gfn2.hpp` are deterministic
  representations of the GFN2 parameter export from revision
  `fa8a4416e8fe093d0075bc10ac875494c2a449a9`. Exact source paths and hashes
  are recorded in `data/parameters/manifest.json`.
- The Stewart STO-nG tables in `src/model/gfn2/basis.cpp` come from
  `src/tblite/basis/slater.f90` at that revision.
- The element spin constants in `src/model/gfn2/spin.cpp` come from
  `src/tblite/data/spin.f90` at that revision. Their digest is recorded in
  `data/parameters/spin_manifest.json`.
- The SCC observer development tool in `tools/oracle/tblite_scc_trace/`
  carries its own pinned patch metadata and bundled GPL/LGPL texts.

## dftd4 and mctc-lib

dftd4 repository: <https://github.com/dftd4/dftd4>

dftd4 license: `LGPL-3.0-or-later`.

mctc-lib repository: <https://github.com/grimme-lab/mctc-lib>

mctc-lib license: `Apache-2.0` (`LICENSES/Apache-2.0.txt`).

The packed GFN2-D4 reference data in `data/parameters/d4.hpp` is derived
from dftd4 revision `6e1f59c3f39d919a2dbef0601d2576727c8b30e8`.
The D4 electronegativity-weighted coordination data and implementation
conventions also use mctc-lib revision
`e9de066d89f250d1cfb6de3a33f0c27c0e2f855d`. The ordinary GFN2
coordination implementation retains mctc-lib's covalent-radii data and
double-exponential convention, while the H0 implementation retains its
Mantina atomic-radii table. Exact dftd4 source blobs are recorded in
`data/parameters/d4_manifest.json`; exact mctc-lib source paths, blobs, and
hashes are recorded in `data/parameters/mctc_manifest.json`. The original
focused notice and upstream license copies remain in
`data/parameters/d4.NOTICE` and `data/parameters/licenses/`.

## xTB numerical oracle

Repository: <https://github.com/grimme-lab/xtb>

License at the pinned revision: `LGPL-3.0-or-later`.

The conformance corpus uses xTB 6.7.1 revision
`edcfbbe39d411edc225e27315fbda3a204ddb023` as an independent executable
oracle. The repository redistributes normalized numerical outputs and small
test geometries, not xTB source code or binaries. Runtime hashes, command
contracts, and the origin of the QM/MM fixtures are recorded in
`data/conformance/manifest.json`.

## LAMMPS documentation reference

`docs/qmmm.md` cites the LAMMPS QMMM-XTB adapter at revision
`9ab8ca565e0f71d967587e0bca2015f7d689f19f` to document the external
`b + A q` interface convention. No LAMMPS source code, binary, or numerical
table is redistributed by gpuxtb.

## Distribution policy

Generated artifacts remain accompanied by their upstream SPDX identifier,
pinned provenance manifest, and applicable license text. Source archives,
Python wheels, and native CMake installs must retain this file, the project
license, the compatible third-party license texts, and the parameter
provenance manifests.
