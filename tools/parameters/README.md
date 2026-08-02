# GFN2-xTB parameter generation

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
