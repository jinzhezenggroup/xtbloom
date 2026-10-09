# Input-only AO and convergence-risk grouping prototype

This page freezes the issue #514 prototype contract before any holdout result is
read. The prototype is a preregistration artifact, not a measured policy or a
production recommendation. It changes no runner defaults and has no optional
machine-learning dependency.

## Input contract and leakage boundary

The JSON input has exactly these scheduling fields:

```json
{
  "schema_version": 1,
  "systems": [
    {
      "case_id": "conformer-001",
      "molecule_group_id": "molecule-001",
      "scheduling_metadata": {
        "atomic_numbers": [6, 1, 1, 1, 1],
        "exact_ao_count": 8,
        "molecular_charge": 0,
        "spin_channels": 1,
        "unpaired_electrons": 0
      }
    }
  ]
}
```

Every system must provide `molecule_group_id`; every system in one molecule or
near-duplicate family uses the same value. A missing identity is an error. The
split is assigned to unique groups before any outcome is read, so conformers
cannot straddle calibration and holdout. The manifest requires exact AO count,
atomic numbers, molecular charge, and at least one explicit spin field
(`spin_channels` or `unpaired_electrons`). The AO count is supplied by the
input-side basis accounting; this prototype does not infer it from atom count.

Scheduling metadata rejects unknown fields and any outcome-shaped field,
including energy, force, status, convergence, iteration, success/failure,
result, target, or label fields. Case IDs are lookup keys and may be used as a
stable planner tie-break; molecule-group IDs are used only for group splitting
and lookup. Neither is encoded as a model feature. The fixed risk score uses
four binary rules: nonzero molecular charge, explicitly nonzero unpaired
electrons, explicitly multiple spin channels, and any atomic number at least
18. Scores 0, 1, 2, and 3–4 map to `low`, `guarded`, `elevated`, and `high`.
Exact AO count and risk band are the candidate planner bucket fields. Every
input ID is annotated; no high-risk or difficult system is filtered.

The policy is static and has no fitting step. If later work fits a policy, it
must use a separate `purpose: calibration-only` document containing only the
case/group IDs and scalar `scc_iterations`, `scc_converged`, and `status`
outcomes. `validate_calibration_input` checks each case/group pair against the
frozen calibration partition before accepting those outcomes; it rejects
holdout group IDs and unknown fields, so nested arbitrary labels cannot hide
holdout identities. The scheduling parser never accepts calibration outcomes.

## Frozen artifacts

Create the plan before collecting any measurements:

```bash
python3 -m benchmarks.convergence_grouping freeze-plan \
  --manifest /path/to/input-only-manifest.json \
  --output /path/to/issue-514-freeze-plan.json
```

The command refuses to overwrite an existing plan. It writes the complete
policy and split plus SHA-256 identities for the policy, split, normalized
input manifest, and whole freeze plan. The split uses the fixed seed
`xtbloom-issue-514-fixed-group-split-v1-2026-10-09`, ranks group IDs by
SHA-256, and assigns `ceil(group_count / 5)` groups to holdout while retaining
at least one calibration group. Input row order does not affect these hashes.
The artifact is marked `preregistered_only` and contains no fitted model,
training result, or holdout measurement.

## Planner integration boundary

`annotate_scheduling_inputs(manifest)` returns a `case_id`-keyed mapping of
exact AO count, risk score/band, explanation reasons, and an AO/risk bucket
key. It deliberately does not return a sorted ID list, batches, or scatter
map. The #513 finite-list planner remains the only component that forms the
complete permutation, applies the batch cap/tail behavior, and restores every
result by its canonical mapping. The comparison remains three distinct
strategies: `original`, #513 `exact-ao`, and experimental `ao-risk`; `original`
stays the default.

The finite-list runner consumes those annotations only with explicit
`--ao-grouping ao-risk`. Batches use ascending exact AO, then the fixed
`high`, `elevated`, `guarded`, `low` band order, then canonical input index.
Every AO/band bucket retains its own cap-bounded tail. Its schema-2 plan hash
covers the annotations and frozen policy identities; existing strategies keep
their schema-1 plan encoding and unchanged behavior without convergence flags.

All three strategies can bind the same prior experiment checkpoint:

```bash
python3 -m benchmarks.run --library /path/to/libxtbloom.so \
  --manifest /path/to/workload-manifest.json --engines xtbloom --backends cuda \
  --cuda-memory-modes host --case-ids-file /path/to/view-holdout-ids.txt \
  --ao-grouping ao-risk --batch-sizes 64 256 --properties force \
  --convergence-manifest /path/to/input-only-manifest.json \
  --convergence-plan /path/to/issue-514-freeze-plan.json \
  --convergence-freeze-sha256 <prior-logical-freeze-sha256> \
  --convergence-workload-sha256 <prior-workload-manifest-file-sha256> \
  --convergence-cohort-sha256 <prior-complete-ordered-cohort-logical-sha256> \
  --convergence-partition holdout --warmups 5 --repetitions 30
```

The command is one strategy, not the balanced paired evaluation orchestrator;
run it through the scheduler and preserve all three strategy orders/blocks
required below. The logical freeze identity is not the freeze file's byte hash.
All expected identities must come from the prior checkpoint, not potentially
rewritten files. The runner reconstructs the entire static policy/split/protocol,
checks every frozen system against actual input AO, charge and explicit spin
metadata, and verifies all frozen input hashes against the separately pinned
workload manifest. Geometry bytes remain provenance, not scheduling features.
Missing or mismatched identities fail before inference, including for frozen
peers outside the selected view.

The workload manifest is parsed from the same captured bytes checked against
its pin. Each frozen input is captured and hashed once, then held in a private
read-only snapshot for metadata parsing and all native batch assembly. Source
geometry changes after capture cannot replace the verified bytes in a sweep.
Snapshot creation and its memory/file footprint belong to planning/binding;
the snapshot is closed after the finite-list run. This protects against ordinary
concurrent source updates, not a malicious process with access to the same
user's private temporary files.

`--convergence-partition` validates rather than filters: supply the exact
view/partition intersection ID file. Its complete ordered ID list must match
the externally pinned cohort digest, so a legitimate roster subset cannot
silently replace a predeclared view. Obtain its logical digest before measurement
with `ordered_case_ids_sha256` (canonical compact JSON list, SHA-256); this is
not the ID text file's byte hash. No system is silently dropped or added.
Original/exact-AO baselines validate the same binding but do not evaluate risk
annotations. Freeze reading, complete-roster hashing and verification, input
inspection, annotations and planner work all count in `planning_ms`. Each row
also reports `planning_inclusive_end_to_end_ms`, which adds the complete
one-time planning cost to each strict-FRESH sweep without assumed amortization;
the existing sweep-only distribution and explicit reusable-plan metric remain
separate. JSON and CSV retain the policy/split/freeze/input identities and risk
bands. This integration supplies no scientific qualification or adoption.
Legacy non-frozen CSV columns and sweep timing fields remain unchanged; the
additional binding/planning-inclusive fields require the opt-in freeze flags.
Every standalone frozen run explicitly has `claim_eligible=false`, even if its
separate `independent_reference_qualified` science check passes. A single
strategy's output is not the complete balanced paired holdout decision.

## Prospective holdout matrix

The eligible comparison is a finite input set, strict FRESH, CUDA public
execution with host descriptors, identical binary, model, SCC tolerances,
requested outputs, thread/device settings, and input IDs for all three
strategies. Use batch caps 64 and 256 for each predeclared coordinate:

- all four frozen OMol25 views (2048 systems total), plus its frozen
  `first256` and `last256` views;
- homogeneous 32-atom and 62-atom controls.

Use five untimed warmups per strategy/coordinate, then 30 paired measurement
blocks. Interleave the six possible orders of `original`, `exact-ao`, and
`ao-risk` five times. Preserve every requested input ID, including difficult,
nonconverged, failed, or unavailable systems. Record expected and attempted
denominators, success sets, per-ID energy/forces/charges, SCC iterations,
status, convergence, failure reason, and complete NaN output slices. Compare
per-ID science against the pinned independent oracle at its existing
tolerances; do not broaden tolerances or remove a failed coordinate.

Report median, p95, maximum, and paired 95% bootstrap intervals for feature
annotation/planning, public preparation, compute, publication/scatter, and
complete end-to-end time. Include per-batch distributions and the final tail
batch, plan and peak process memory, and all planning/prediction costs inside
the end-to-end total. The paired resampling unit is the measurement block; do
not pool unlike cohorts or treat systems within one batch as independent
timing samples. Compute speedup per block as `exact-ao` end-to-end time divided
by `ao-risk` end-to-end time; use a paired percentile bootstrap with 10,000
replicates and fixed seed `xtbloom-issue-514-bootstrap-v1`.

## Decision rule and gates

An `ao-risk` candidate can justify only an opt-in follow-up for named eligible
holdout cohorts when all expected IDs remain present, every scientific gate
passes, its median end-to-end time improves by at least 5% over `exact-ao`, the
paired 95% bootstrap interval for end-to-end speedup excludes 1.0, and no other
eligible cohort regresses by more than 5% in median end-to-end time. Report
compute and tail distributions even when the end-to-end decision is negative.
Any dropped/duplicated ID, scientific or status regression, hidden failed
system, or preparation cost that erases the gain in a completed eligible
evaluation is a no-go. Missing required oracle/GPU/input or incomplete holdout
evidence remains `UNVERIFIED`/`BLOCKED`, not an empirical no-go, and does not
permit issue closure. A no-go or neutral result leaves `original` as default
and does not claim that production has been optimized. A positive decision
still requires a separate production integration issue and the independent
#518 qualification; this prototype alone cannot satisfy either gate.

This prototype contains no qualified holdout timings or original-input/oracle
evidence. No performance result or acceptance decision is claimed here.
