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

These protocols answer different questions. In particular, the public
cross-engine figure and the FRESH/WARM study use different SCC settings,
correctness gates, start policies, workloads, and sample counts. Never combine
their numbers or thresholds.

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
