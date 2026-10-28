# Historical CUDA capture

`historical_alkane_capture.py` uses the original-slot selector and existing
`run.XTBloomAdapter`, not the named-workload CLI or substitute healthy controls.
Prepare and hash-check the exact saved input corpus as described in
[historical input assembly](historical-alkane-inputs.md) before collection.
Empty reference placeholders remain correctness/performance-ineligible.

## Protocol identity

The historical `bridge-raw.jsonl` rows called "strict" recorded maximum 500
iterations, charge tolerance `1e-6`, energy tolerance `1e-8`, 300 K, explicit
FRESH tag 1, and energy/force flags 3. That label does not establish the tighter
conformance helper's `1e-10`/`1e-12` policy. The capture retains those observed
values; it does not retune a failed coordinate. Current independently fixed
GFN2, modified Broyden/history 8/damping 0.4 and default determinism are recorded;
unrecorded historical mixer/ISA details are not retroactively attested.

The matrix distinguishes two requested-property sets:

- EF requests energy and forces, matching the recorded historical request.
- EFq additionally requests atomic charges. This is a separate publication
  coordinate, not a replay of the old EF-only timing or an independent oracle.

Each original 256-slot 32/62 workload and property set has four independent,
persistent owners: main baseline, candidate diagnostics OFF, candidate ON
without a sink, and candidate ON with a sink. The latter two share one pinned
ON binary but not one context. The sixteen owners each receive one cold call,
five excluded warmups and thirty cyclic measured rounds: 576 requested calls
and 147,456 requested peer observations. Every call is FRESH. A warmup is not
the WARM restart policy, and FRESH does not clear topology caches.

## Producer and output requirements

Freeze the source, manifest and source-input hashes, three library/cache/source
identities, compiled diagnostic options, helper version, fixed options and
execution matrix before running. Verify clean source and complete pre/post
binary/cache identities. Do not silently label an older build as current HEAD
or the current baseline as the historical frozen binary.

Run actual model collection only through finite Slurm on an allowed node and
preserve assigned `CUDA_VISIBLE_DEVICES`. The wrapper must independently verify
CUDA ordinal 0 against the assigned physical GPU and record effective CPU
affinity, not infer it from requested binding. Use one CPU/BLAS worker. The
capture never remaps device visibility or enables CPU fallback.

The input JSON provenance is bound by the converter's mandatory source hash and
the capture's exact manifest hash. The separate bridge-raw hash identifies the
historical request protocol, not the input JSON. With an independently verified
library-pin file and scheduled wrapper, the collector command is:

```console
python3 -m benchmarks.historical_alkane_capture \
  --manifest /absolute/capture-inputs/manifest.json \
  --manifest-sha256 <verified-manifest-sha256> \
  --historical-bridge-raw-sha256 <verified-historical-bridge-raw-sha256> \
  --library-pins /absolute/library-pins.json \
  --output-dir /absolute/new-output-directory
```

Add `--describe` for hardware-free manifest validation and protocol description;
it never loads a native library and is not a native runtime pass.

Use a new, exclusively created output directory. Records retain phase, order,
logical IDs, literal call status/error, flags, per-peer convergence/status/
iterations, complete requested floating slices, and synchronized public-invoke
latency. Unrequested charges are marked unrequested rather than filled with
fake zeros. Standard-JSON nonfinite tags preserve failure observations. A
call-level failure poisons the owner; remaining requested calls stay NOT_RUN,
not silently retried. Data-level peer failures retain complete NaN slices and
successful peers. All created owners must be closed even after another fails.

Each ON-sink owner has a separate input-private trace. Correlate every successful
publication with exactly one newly appended, validated diagnostic object; do
not infer logical identity from AO or a positional index alone. ON-without-sink
and OFF have no caller trace. ON sink timings include ledger readback and file
append; ON without a sink distinguishes recording cost. Output validation and
archival happen outside the measured invoke boundary.

## Interpretation boundary

The collector is always diagnostic-only. It does not call the ordinary
correctness gate with empty placeholders, claim independent physics, select
an adopted threshold, or publish primary performance/speedup results.
Compare every matched slot's statuses, convergence, iterations and requested
outputs under a separately frozen numerical gate before interpreting timing
distributions. Keep failed/missing calls, unknown stage timings and tool limits.
No one-shot favorable sample or subset establishes no steady-state overhead.

For this protocol, freeze the diagnostic comparison gate before the first
native call: match workload, property set, phase, phase-local round and original
slot ID across all four owners. Require exact call status, result flags,
per-system status, convergence and iteration counts. Successful requested
binary64 energy/force/charge values must be bitwise identical, including signed
zero. Corresponding failed-peer slices must contain complete quiet NaNs; their
NaN payloads need not match. Missing or call-failed observations do not pass and
must not be reduced to a successful-peer intersection. Compare EF versus EFq
common energy/force outputs and discrete metadata separately to detect a
publication-dependent change. Report differences rather than adopting a new
tolerance after observing them. This is an instrumentation/publication parity
gate, not an independent scientific oracle or a performance adoption gate.

Original2048 geometry/four-topology rotation, broader failure/WARM/memory modes,
full applicable runtime/sanitizer/profiler evidence and scientific qualification
remain separate issue requirements. This protocol does not approve or develop
the owner-gated issue #515 replacement experiment.

The hardware-free self-tests use fake adapters rather than native inference:

```console
python3 -m unittest -v benchmarks.test_historical_alkane_capture
```
