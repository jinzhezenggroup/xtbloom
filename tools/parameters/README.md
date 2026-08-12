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
