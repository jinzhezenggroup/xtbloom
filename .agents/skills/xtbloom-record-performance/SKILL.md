---
name: xtbloom-record-performance
description: Measure and archive reproducible, correctness-qualified xTBloom latency, throughput, scaling, memory, CUDA Graph, and profiler evidence. Use for performance changes, benchmark harness edits, cross-library comparisons, speed claims, Nsight work, release thresholds, or files under `benchmarks/` and `benchmarks/evidence/`.
---

# Record xTBloom Performance

Make performance evidence reproducible and scientifically admissible. Read `AGENTS.md`, `benchmarks/README.md`, the active issue, and the selected harness before running measurements.

## Define the Claim Before Timing

Write the exact claim and its acceptance threshold. Specify:

- latency, throughput, memory, allocation, transfer, Graph, or kernel metric;
- public API or internal kernel scope;
- hardware, backend, descriptor memory mode, start policy, property set, and workload;
- baseline engine/version and whether it is serial, equal-core, persistent, or includes restart state;
- batch sizes, molecule-size/topology classes, QM/MM coverage, warmups, and sample count.

Do not infer a release-wide claim from one homogeneous molecule, one batch size, one timing, or an unequal resource comparison.

## Qualify Correctness First

1. Build the exact committed revision being measured.
2. Require a clean repository and clean selected-library source. Use dirty-evidence options only for diagnosis; label and reject the result for final evidence.
3. Run the relevant scientific and public validation before timing.
4. Use harness correctness gates such as `--fail-on-correctness`.
5. Preserve failed or unavailable coordinates instead of silently dropping them.
6. Never widen a conformance tolerance to make a timing row eligible.

When numerical output changed, require the independent scientific-evidence workflow before timing. For CUDA claims, require the complete real-GPU runtime and sanitizer matrix as applicable.

## Control the Measurement

Record at minimum:

- repository revision and dirty bit;
- executable/library absolute path and SHA-256;
- compiler, flags, CMake cache identity, CUDA driver/toolkit, GPU and CPU model;
- process affinity, CPU worker count, BLAS provider and thread environment;
- workload inputs, random seed, descriptors, SCC options, FRESH/WARM identity, and requested outputs;
- setup, cold call, warmup policy, synchronization boundary, repetitions, and every raw sample;
- correctness errors, convergence state, SCC iterations, and unavailable reasons.

For CUDA, synchronize at the documented measurement boundary and keep correctness downloads outside the timed interval only when the public protocol says so. For cross-library work, preserve each engine's unavoidable public-API work rather than subtracting it after measurement.

## Use the Repository Harnesses

Use `benchmarks/run.py` for the public cross-library matrix and `benchmarks/natoms_scaling.py` for its documented scaling/start-policy protocol. Do not reconstruct their schemas with an ad hoc script.

Run harness self-tests after code changes:

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple \
  uv run --isolated --locked --only-group nox nox -s python
```

Report a missing optional engine, runtime, GPU coordinate, or workload as `unavailable`; do not reduce the requested matrix without saying so.

## Archive Final Evidence

Create an issue-scoped bundle following the existing layout:

```text
benchmarks/evidence/issue-<N>/<date>-<machine>/
  README.md
  SHA256SUMS
  *.json
  *.csv
  derived-profiler-reports
```

Treat JSON with raw samples and identities as authoritative; keep CSV as a compact view. Git retains only compact evidence: no tracked file under `benchmarks/evidence/` may exceed 1 MiB, and the complete tracked directory may not exceed 16 MiB. Never add a benchmark path to the large-file hook exclusion. If authoritative raw JSON exceeds either budget, upload the exact bytes to durable external storage and commit an `EXTERNAL_ARTIFACTS.tsv` entry containing its immutable URL, exact byte count, SHA-256, producing revision, and retrieval command. If durable storage is unavailable, mark that evidence `UNVERIFIED`; do not shrink the requested matrix or convert the missing artifact to a pass. Do not edit harness artifacts after generation. In `README.md`, record exact argv, environment boundary, build/source revisions, limitations, correctness qualification, and which requested coordinates were unavailable. Hash every retained repository artifact and verify `SHA256SUMS` after the bundle is final; independently verify every external artifact against its recorded SHA-256.

If an old runner or external source produced the data, retain or hash-pin that exact source and command. Never claim the current runner can reproduce a historical workload unless it actually can.

Raw `.nsys-rep`/`.ncu-rep`/`.qdstrm`/`.sqlite`/`.sqlite.dbb`/`.csv.db`/`.prof`
captures can embed the target process environment and credentials. They are
prohibited from the repository: `.gitignore` ignores them and the
`forbid-raw-profiler-captures` pre-commit hook rejects them at commit time
(including `git add -f`). Only sanitized derived CSV/text/JSON summaries from
`nsys stats` or `ncu --csv` console output, with the profiler version and
extraction command recorded, may be archived under `benchmarks/evidence/`.
Do not use `ncu --export` for this purpose: it writes a native `.ncu-rep`.

## State the Narrow Conclusion

Report distributions and raw-sample counts, not one favorable value. Name hardware, workload, batch size, threads, baseline semantics, and correctness gate in every speed claim. Separate development measurements from clean-HEAD release evidence and list remaining matrix gaps in the active issue.
