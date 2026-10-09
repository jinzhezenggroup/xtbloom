# OMol25 diagnostic input conversion

`omol25_inputs.py` validates caller-pinned CSV, ragged NPZ and source-manifest
bytes and creates inputs for the existing finite-list public benchmark runner.
The original CSV row order, IDs, charge, multiplicity and unpaired electrons
are retained. Exact AO counts are checked against generated GFN2 shell
degeneracies, not estimated from atom count. Bohr coordinates use binary64
round-trip serialization. Source and converted-byte digests are recorded.

The output root must be new and outside every Git worktree. Do not distribute
source or converted OMol25 inputs without establishing their legal rights.
The recovered 2026-10-08 cohort does not establish redistribution permission,
a GFN1 cohort, or an independent scientific oracle. The converter never runs
a model or synthesizes energy/force/charge references.

The existing assembler requires a `golden` path. Diagnostic cases therefore
point to an explicitly empty `reference-unavailable.json`, carry
`oracle_role=diagnostic-no-independent-reference`, and declare qualification
false. Successful finite output is not independent E/F/q qualification.
Finite-list reports list missing reference properties and unqualified IDs
across all measured sweeps, set `independent_reference_qualified=false` and
`claim_eligible=false` when references are incomplete, and never grant a
performance claim merely because numerical comparisons pass.

Spin channel count is a required caller choice, not inferred from multiplicity.
The NPZ records multiplicity and unpaired electrons but no channel policy.
Choosing two channels for a new diagnostic experiment does not recover the
historical setting or resolve issue #509. Source molecular/group identifiers
are retained only when present; no near-duplicate grouping identity is invented.

For the recovered canonical cohort, from the repository root:

```bash
python3 benchmarks/omol25_inputs.py \
  --csv /path/to/canonical-selected_samples.csv \
  --npz /path/to/canonical-performance.npz \
  --source-manifest /path/to/source_manifest.json \
  --output-root /absolute/new/external/omol25-diagnostic \
  --csv-sha256 8ba04b03b13afe83f3186a94fed75b1b1d8f5a06b711d5f9380b7cf1e97c0401 \
  --npz-sha256 bc8f7ddc71c83fe14134e8cbabeecb39986a9458c63026d04ea0759eb7547a8c \
  --source-manifest-sha256 8d5016f16a15a267c91a3c041082bad3758037e34bbb04c28d5caecf3c561631 \
  --spin-channels 2
```

The output includes `manifest.json`, `provenance.json`, the empty reference,
per-ID coordinates, and ordered `id-lists/{all,small,medium,large,first256,last256}.txt`.
Small/medium/large mean AO1–38/39–69/70–180, not solver bucket boundaries.
Use the output manifest and one ID list with `benchmarks/run.py`:

```bash
python3 benchmarks/run.py --library /path/to/libxtbloom.so \
  --manifest /absolute/new/external/omol25-diagnostic/manifest.json \
  --case-ids-file /absolute/new/external/omol25-diagnostic/id-lists/last256.txt \
  --engines xtbloom --backends cpu --cuda-memory-modes host \
  --ao-grouping exact-ao --batch-sizes 64 --properties force \
  --warmups 5 --repetitions 30 --require-available \
  --output-json /absolute/external/results.json \
  --output-csv /absolute/external/results.csv
```

This is diagnostic only until independent references and the other scientific
and performance gates pass. For CUDA, run the whole command through finite-time
`srun`, preserve scheduler visibility, and identify the clean selected library.
Use original and exact-AO with the same inputs/library/options; preserve failed
systems and layout-dependent branches rather than excluding them. A single
runner invocation does not implement an interleaved paired comparison.

Hardware-free validation:

```bash
python3 -m unittest -v benchmarks.test_omol25_inputs benchmarks.test_ao_grouping
```

The locked project Python/nox suite always runs converter tests. Bare native
CMake requires only a Python interpreter, so the separate converter CTest is
registered only when that interpreter can import NumPy; otherwise configure
reports it as NOT REGISTERED, not passed. This does not exempt the project
Python gate or change any existing native/scientific acceptance requirement.
