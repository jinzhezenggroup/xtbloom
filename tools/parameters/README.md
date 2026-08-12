# GFN-xTB parameter generation

GFN1 and GFN2 use separate canonical exports, schemas, generated headers, and
manifests. They share provenance rules but not an assumption that the same
parameter families or algorithms are present.

## GFN2-xTB

`generate_gfn2.py` treats tblite's structured parameter export as the source
of truth. It records the tblite Git revision, a digest of the relevant source
blobs, the exporter version and binary digest, the upstream LGPL license, and
the digest of every generated artifact.

To refresh the export from a known tblite checkout and executable:

```sh
LC_ALL=C python3 tools/parameters/generate_gfn2.py \
  --refresh \
  --tblite /path/to/tblite \
  --tblite-source /path/to/tblite/source \
  --tblite-revision COMMIT_OR_TAG
```

Use the exact commit or release tag from which the exporter was built. The
generator hashes committed blobs at that revision, so unrelated local edits do
not change the provenance record.

The command writes these deterministic files under `data/parameters/`:

- `gfn2.toml`: the machine-readable tblite export;
- `gfn2.json`: normalized, ordered data used for validation and inspection;
- `gfn2.hpp`: compact trivially-copyable C++ host tables suitable for one-time
  upload to CUDA or a future ROCm backend;
- `manifest.json`: source, licensing, schema, generator, and output checksums.

After editing the generator, regenerate derived files without invoking tblite:

```sh
python3 tools/parameters/generate_gfn2.py
```

CI can detect stale or manually edited files without installing tblite:

```sh
python3 tools/parameters/generate_gfn2.py --check
python3 -m unittest discover -s tests/parameters -p 'test_*.py'
```

The validator intentionally rejects unknown fields. When tblite extends its
parameter schema, update the normalized schema and runtime tables explicitly
instead of silently losing the new values.

## GFN2 D4 reference data

`generate_d4.py` extracts the GFN2 charge-model reference systems from a
pinned dftd4 Git object database. It evaluates the reference polarizabilities
and Casimir--Polder quadrature once, then writes a packed 262-reference C6
matrix for direct C++/CUDA use. No Fortran or dftd4 library is needed at
runtime.

```sh
python3 tools/parameters/generate_d4.py \
  --source-git-dir /path/to/dftd4.git \
  --revision 6e1f59c3f39d919a2dbef0601d2576727c8b30e8 \
  --output-dir data/parameters
```

The generated `d4_manifest.json` records the commit, tree, every parsed Git
blob and SHA-256 digest. `d4.NOTICE` and `data/parameters/licenses/` carry the
dftd4 LGPL-3.0-or-later and mctc-lib Apache-2.0 attribution used by the D4 data
and electronegativity-weighted coordination-number implementation.

## GFN1-xTB

`generate_gfn1.py` treats tblite 0.7.0's structured GFN1 export as the
scientific source of truth and writes a separate `gfn1.toml`, normalized
`gfn1.json`, generated `gfn1.hpp`, and `gfn1_manifest.json`. The generator
validates GFN1-specific selectors such as exponential coordination numbers,
harmonic hardness averaging, atomwise third order, repeated-angular-momentum
shell masks, D3 damping, and the classical halogen parameters. The dxtb GFN1
TOML is recorded only as a non-authoritative semantic cross-check.

The GFN1 D3(BJ) implementation also requires reference coordination numbers,
C6 values, r4/r2 data, and pair vdW radii that are not part of the tblite
method export. `generate_gfn1_d3.py` derives those tables from simple-dftd3
v1.4.0 and mctc-lib v0.5.1 into `gfn1_d3.json`, `gfn1_d3.hpp`, and
`gfn1_d3_manifest.json`. Ordinary checks are fully offline:

```sh
python3 tools/parameters/generate_gfn1.py --check
python3 tools/parameters/generate_gfn1_d3.py --check
```

Refreshing the D3 tables is an explicit provenance operation against reviewed
Git objects. Generate into a separate directory first and review the manifest
and byte diff before replacing canonical files:

```sh
python3 tools/parameters/generate_gfn1_d3.py --refresh \
  --simple-dftd3-source /path/to/simple-dftd3 \
  --mctc-source /path/to/mctc-lib \
  --output-dir build/gfn1-d3-refresh
```

The reviewed pins are simple-dftd3 v1.4.0 revision
`6f0b06fbfa8653a23ca55c453772ce3af4420706` and mctc-lib v0.5.1 revision
`aa89d4bf5c0076fbf169b59eeb9e30185db0e5a5`. Generated files are retained
under their upstream licenses and must not be edited by hand.
