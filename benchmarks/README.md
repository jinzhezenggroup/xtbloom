# Benchmark harnesses

This directory contains maintainer-facing performance tools. The public
headline results live in the [performance summary](../docs/user-guide/performance.md);
the scripts and method pages here define how evidence is produced and audited.

## Choose the protocol

| Question | Runner | Method | Hardware-free test |
| --- | --- | --- | --- |
| End-to-end gas/QM-MM matrix across xTBloom, xTB, tblite, and dxtb | `run.py` | [Public matrix](matrix.md) | `benchmarks.test_run`, `benchmarks.test_dxtb_adapter` |
| Cross-engine molecule-size scaling and the public README figure | `natoms_cross_engine.py` | [Cross-engine scaling](cross-engine.md) | `benchmarks.test_natoms_cross_engine` |
| CPU FRESH/WARM scaling against explicit references | `natoms_scaling.py` | [FRESH/WARM scaling](fresh-warm.md) | `benchmarks.test_natoms_scaling` |
| Cost of xTBloom-owned CUDA DLPack result arenas | `dlpack_result_memory.py` | [DLPack result memory](dlpack-result-memory.md) | `benchmarks.test_dlpack_result_memory` |
| Dense 62-atom complete-Hessian batch throughput | `hessian.py` | Script module documentation and issue evidence README | `benchmarks.test_hessian` |

These protocols answer different questions. In particular, the public
cross-engine figure and the FRESH/WARM study use different SCC settings,
correctness gates, start policies, workloads, and sample counts. Never combine
their numbers or thresholds.

The public cross-engine selection is maintained in
`natoms_cross_engine_publication.json`. Each engine/backend points to its own
clean evidence series, so a CUDA-only optimization refreshes only xTBloom CUDA
without relabelling unchanged CPU or third-party timings. The generated
`natoms_cross_engine_latest.csv` is the reviewable current data table and
includes per-row revisions, artifact hashes, evidence paths, protocol identity,
and the recorded RTX 5090 identity for CUDA rows.

## Continuous regression signal

`codspeed_inference.py` is a deliberately small `pytest-codspeed` suite for
pull-request regression detection. It measures representative public Python/C
ABI CPU paths (GFN1, GFN2 FRESH/WARM, and a mixed-size ragged batch) with one
xTBloom CPU worker. The primary cases use the deterministic 32-atom alkane from
the scaling protocol, rather than letting a water-only workload reduce the
signal to fixed API and tiny-matrix overhead. The dedicated
`.github/workflows/codspeed.yml` job additionally sets single-threaded BLAS,
forces xTBloom's portable baseline CPU ISA, and pins the reviewed
dynamic-architecture OpenBLAS provider to its `Nehalem` kernel before running
CodSpeed `simulation` mode. The workflow verifies the selected OpenBLAS core at
runtime; `XTBLOOM_CPU_ISA` alone does not control BLAS dispatch.

CodSpeed results are **regression signals, not publication-grade hardware
timings**. They do not replace any protocol above, and they must not be quoted
as absolute latency or throughput evidence. CUDA, cross-engine comparisons,
large-system scaling, Hessian throughput, and hardware-specific ISA claims stay
on their existing audit-ready protocols.

The workflow creates the project environment from `uv.lock`, then installs the
CI-only CodSpeed toolchain from `codspeed-requirements.txt` with hashes required
for every artifact. `codspeed-requirements.in` is the human-maintained input;
regenerate the lock with the pinned workflow version of uv and the command in
its generated header. Both files stay outside project metadata and the PyPI
sdist. The plugin, Action, runner, and modified Valgrind executable do not enter
xTBloom runtime metadata, native installs, sdists, or wheels; distribution
archives retain only the applicable legal notice. Exact provenance and hashes
are recorded in `THIRD_PARTY_NOTICES.md`.

CodSpeed is an informational regression signal. PR results use CodSpeed's
default comparison against the latest successful `main` baseline, but a
reported performance delta does not automatically block a merge. Investigate a
material signal with the applicable reproducible benchmark protocol before
making a performance claim. The first successful `main` run after this workflow
lands establishes the initial baseline, so the introducing PR has no prior
baseline. A reviewed runner, Action, compiler, Python, OpenBLAS, or lock update
starts a new baseline; do not compare values across those environment changes.

To run the same benchmark module in an environment that already has the plugin
installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_CORETYPE=Nehalem OPENBLAS_NUM_THREADS=1 \
  XTBLOOM_CPU_ISA=baseline \
  pytest benchmarks/codspeed_inference.py --codspeed
```

## Evidence requirements

A publishable result must retain:

- the exact clean source revision and selected-library hash;
- compiler, build configuration, runtime providers, hardware, affinity, and
  thread environment;
- workload identity, requested outputs, memory mode, start policy, warmups,
  repetitions, synchronization boundary, sample count, and distribution
  summary;
- convergence and correctness results for every available coordinate;
- explicit `unavailable` or failed rows rather than a silently reduced
  matrix; and
- a README with exact commands, limitations, and SHA-256 coverage.

Git retains compact evidence only. A tracked file under `benchmarks/evidence/`
may not exceed 1 MiB, and the complete tracked directory may not exceed 16 MiB.
When a reproducible raw harness artifact exceeds either budget, omit it rather
than uploading it by default; retain the generated compact result, exact
command, clean source and binary identities, inputs, correctness qualification,
and limitations needed to reproduce the claim. External archival is optional
only when the exact raw bytes are themselves necessary evidence. Final compact
bundles belong under `benchmarks/evidence/issue-<N>/<date>-<machine>/` and must
not be edited after generation.

## Self-tests

Run the relevant test while iterating and the complete hardware-independent
set before changing benchmark documentation or publication logic:

```bash
python3 -m unittest -v benchmarks.test_run
python3 -m unittest -v benchmarks.test_dxtb_adapter
python3 -m unittest -v benchmarks.test_natoms_cross_engine
python3 -m unittest -v benchmarks.test_natoms_scaling
python3 -m unittest -v benchmarks.test_dlpack_result_memory
python3 -m unittest -v benchmarks.test_hessian
python3 -m unittest -v benchmarks.test_evidence_size
```

The plotting test is opt-in because Matplotlib is a publication-only
dependency. Set `XTBLOOM_RUN_PLOT_TEST=1` in an environment that already
provides Matplotlib, or render with the pinned inline-metadata command described
in [the cross-engine method](cross-engine.md).

## Profiler evidence

Raw profiler captures can embed credentials and process environment. Files such
as `*.nsys-rep`, `*.ncu-rep`, `*.qdstrm`, `*.sqlite`, and `*.prof` are
prohibited. Archive only reviewed derived CSV, JSON, or text summaries with the
profiler version and extraction command. See
[profiler evidence policy](profiler-evidence.md).
