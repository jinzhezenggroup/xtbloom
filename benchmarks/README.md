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
| Input-only AO/risk preregistration and opt-in finite-list runner, not a measured policy | `convergence_grouping.py`, `run.py --ao-grouping ao-risk` | [Convergence grouping prototype](convergence-grouping.md) | `benchmarks.test_convergence_grouping`, `benchmarks.test_convergence_runner` |
| Pinned OMol25 inputs for diagnostic finite-list runs | `omol25_inputs.py` | [Input conversion](omol25-inputs.md) | `benchmarks.test_omol25_inputs` |

These protocols answer different questions. In particular, the public
cross-engine figure and the FRESH/WARM study use different SCC settings,
correctness gates, start policies, workloads, and sample counts. Never combine
their numbers or thresholds.

## Finite-list exact-AO grouping

`run.py` also has an explicit finite-list path for deterministic exact-AO
grouping. The default matrix and the finite-list `original` strategy retain
input order. `exact-ao` runs one grouped layout; `paired` is the opt-in
original-versus-exact-AO comparison. Paired mode is limited to xTBloom CPU
host or CUDA host descriptors with strict FRESH SCC starts. It creates both
layout owner sets once, keeps them alive together, and never reconstructs
contexts per sample. CUDA execution must use the local GPU scheduler where
required.
Paired mode is AO-only: convergence inputs and calibration/holdout partition
flags are rejected before workload or native-library setup. The separately
pinned input-only convergence prototype remains a single-layout experiment.
Every case ID must occur once in the selected manifest; duplicate IDs are
rejected before planning.

The AO key is computed from the QM atoms in each input and the generated
`data/parameters/gfn2.json`: each configured shell contributes `2*l+1`
spatial orbitals using its exact angular momentum. External point charges do
not contribute AOs. The plan sorts by ascending AO count, breaking ties by
canonical input index, then divides each exact-AO group into chunks no larger
than the requested cap. A rare-AO group remains a smaller tail batch. The plan
hash covers strategy, input IDs/order, AO counts, GFN2 parameter hash, cap, and
the resulting batch/index mapping. Spin metadata remains attached to each
whole system; it is not a grouping key, and no SCC result or convergence
outcome is read while planning.

Each property and cap reports planning and one-time owner setup separately,
then per-round synchronous compute, result download, per-system publication,
canonical scatter, end-to-end time, and host/device memory. Paired mode runs
one cold pair, the requested warmup pairs, then every measured pair. Order
starts original/exact-AO and alternates AB/BA continuously across all phases.
Both layout owner sets remain resident for every round, and memory snapshots
state that shared residency. End-to-end excludes planning, one-time setup,
correctness comparisons, and final owner cleanup; one-shot totals and
plan/setup amortization are reported separately with their denominators.
Every requested round retains full per-original-ID E/F/q, SCC iterations,
convergence, status, correctness, and raw NaN slices. Failed, unavailable, and
not-run coordinates remain explicit. Pairwise layout comparison uses the
manifest's existing absolute tolerances, records discrete branch differences,
and never counts as independent scientific qualification. Missing independent
references therefore keep `claim_eligible` false even when layouts agree.
Qualification also requires every cold, warmup, and measured phase to pass;
later successful calls do not erase an earlier failure. Memory observations
are process-lifetime host high-water marks and device-global samples, not
per-layout or per-owner device peaks. Initialized compute options are captured
once per owner outside sweep timing and must match across both layouts.
`numerical_comparison_eligible` is only the numerical/protocol gate;
`claim_eligible` and performance-eligibility flags remain false because the
selected library's exact clean producer/source binding is `UNVERIFIED`.
Source and library hashes alone do not establish that association, even for a
clean runner. This mode therefore reports diagnostic evidence, not an adopted
performance claim.

The optional [controlled build receipt](build-receipts.md) recorder and
read-only checker bind retained source/artifact bytes without promoting this
gate. Paired mode can archive externally pinned preflight/postflight checks;
actual mapped-image/dependency/scientific/runtime/performance admission remains
separate, and every performance flag stays false.

The existing `original` and `exact-ao` single-layout modes remain available
for diagnostic runs. They do not provide interleaved paired evidence.

The JSON reporter emits strict JSON. NaN and infinities use tagged objects,
for example `{"__xtbloom_nonfinite_float__":"NaN"}`, which
`run.restore_json_safe_value` can decode back to IEEE non-finite values.
The top-level provenance object contains SHA-256 digests and sizes for the
manifest and selected input files, without embedding input contents. These
streaming archival hashes are computed before benchmark cells and excluded
from planning and sweep timings. Set `--require-available` on qualification
runs so any unavailable requested row exits nonzero; unavailable rows remain
in the report either way.

Example GPU run with a finite case-ID file (one ID per line; blank lines and
`#` comments are ignored):

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc 'python3 benchmarks/run.py \
  --library /absolute/path/to/libxtbloom.so \
  --manifest /absolute/path/to/manifest.json \
  --case-ids-file /absolute/path/to/case-ids.txt \
  --engines xtbloom --backends cuda --cuda-memory-modes host \
  --ao-grouping paired --batch-sizes 64,256 \
  --properties energy,force --warmups 2 --repetitions 5 \
  --output-json build/benchmarks/ao-exact.json \
  --output-csv build/benchmarks/ao-exact.csv \
  --fail-on-correctness --require-available'
```

This command produces one interleaved comparison row for each property and
cap. JSON keeps full per-round outputs and pairing order; CSV carries the round
schedule, timing pairs, plan hashes, and qualification flags. Original input
manifest identity and independent scientific references remain separate
acceptance gates: paired equality alone cannot satisfy either gate.

For a local CPU smoke check, run both strategies against the same built
library and finite case list. Ensure the configured LP64 linear-algebra
runtime is discoverable by the loader (for example, add its directory to
`LD_LIBRARY_PATH` when needed). This small conformance selection validates the
execution path only; it does not replace the issue's original performance
workload or qualify a speed claim.

```bash
library="$PWD/build/cpu-public/libxtbloom.so"
for strategy in original exact-ao; do
  python3 benchmarks/run.py \
    --library "$library" \
    --case-ids h3_plus,ketene,nenacl,sif5_minus \
    --engines xtbloom --backends cpu \
    --ao-grouping "$strategy" --batch-sizes 2 \
    --properties energy,force --warmups 0 --repetitions 1 \
    --output-json "build/benchmarks/ao-cpu-$strategy.json" \
    --output-csv "build/benchmarks/ao-cpu-$strategy.csv" \
    --fail-on-correctness --require-available
done
```

This CPU run checks the same planner, strict-FRESH adapter, result slices, and
conformance comparisons. It does not replace CUDA correctness or performance
evidence.

The public cross-engine selection is maintained in
`natoms_cross_engine_publication.json`. Each engine/backend points to its own
clean evidence series, so a CUDA-only optimization refreshes only xTBloom CUDA
without relabelling unchanged CPU or third-party timings. The generated
`natoms_cross_engine_latest.csv` is the reviewable current data table and
includes per-row revisions, artifact hashes, evidence paths, protocol identity,
and the recorded RTX 5090 identity for CUDA rows. Manifest declarations are
checked against each evidence bundle's SHA-covered `publication-metadata.json`
rather than trusted as free-form labels.

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
python3 -m unittest -v benchmarks.test_convergence_grouping benchmarks.test_convergence_runner
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
