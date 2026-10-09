# CUDA SCC diagnostics

The CUDA SCC diagnostic JSONL format records one instrumented public call per
line. Its v1 call schema is
`xtbloom.cuda.scc_diagnostics.v1`; the machine-readable single-call JSON Schema
is `benchmarks/cuda-scc-diagnostics.schema.json`. The standard-library parser
validates each JSONL line, including cross-field rules not expressible in the
schema, and emits an activity report:

    python3 benchmarks/cuda_diagnostics.py path/to/cuda-diagnostics.jsonl

Use `-` or omit the path to read JSONL from standard input. The command exits
with status 2 and a line-numbered error for malformed input. Its compact JSON
report preserves exact-AO bucket capacities, per-iteration system/channel
activity, submitted-slot slack, and terminal counts. It does not derive a
performance claim. The report schema is
`xtbloom.cuda.scc_diagnostics.report.v1`; its timing policy marks stage timing
unknown and instrumented data ineligible for primary performance tables.

## Build and capture

Build the CUDA library with SCC diagnostics enabled:

    cmake -S . -B build/cuda-diagnostics -G Ninja \
      -DXTBLOOM_ENABLE_CUDA=ON \
      -DXTBLOOM_CUDA_SCC_DIAGNOSTICS=ON \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build build/cuda-diagnostics --parallel

The compile option enables device-ledger recording. Set
`XTBLOOM_CUDA_SCC_DIAGNOSTICS_FILE` to an absolute, writable JSONL path to
enable post-call download and append one record after each successful
synchronous public publication. The parent directory must already exist. The
compile option is default-OFF; when OFF, diagnostic kernels are not registered
in the default library, although the standalone diagnostic kernel test remains
available. When ON without the environment variable, the library records its
device ledger but does not download or append it. A failed public call does not
append a record.

External caller-owned stream capture is an intentional exception. A generic
`Gfn2SccLoopCudaGraphOwner::launch` invoked inside external stream capture omits
owner-ledger kernels so the captured DAG does not retain ledger storage past an
owner reset. A later `write_diagnostics_json` request for that capture returns
`NotSupported` and emits no partial trace. This does not disable diagnostics
for the runtime's private retained-Prepared explicit bounded-capture path or
native periodic path; those remain instrumented.

Run real-GPU capture through Slurm with a finite time limit; replace the sample
public caller with the command that invokes the instrumented library:

    srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
      --time=00:10:00 bash -lc \
      'XTBLOOM_CUDA_SCC_DIAGNOSTICS_FILE=/absolute/output/cuda-diagnostics.jsonl python3 public_caller.py'

Validate and summarize the resulting file:

    python3 benchmarks/cuda_diagnostics.py /absolute/output/cuda-diagnostics.jsonl

The native dump serializes diagnostic contexts within one process. Its optional
`terminal_systems` records carry only positional system indices, not input or
molecule identity. Use a separate, input-private capture file for each process,
and keep a separate runner record of call order and input identity for
correlation. Repeated successful instrumented calls append repeated JSONL
objects in order. Some public requests, including empty calls, bypass SCC and
produce no object; failed public calls also produce no object. Do not
synthesize missing rows or infer molecule identity from AO size, bucket index,
or system index.

## V1 call object

Each line contains the required core fields below and may contain the declared
optional metadata fields. Other fields are rejected.

- `schema`: `xtbloom.cuda.scc_diagnostics.v1`.
- `instrumented`: boolean `true`.
- `execution_mode`: actual mode captured from the device header:
  `device_dispatch_chain`, `device_tail_graph`, or `bounded_fallback`. It is not
  inferred from the prepared graph family or `graph_preference`; for example,
  a periodic public request can execute as `bounded_fallback` even when its
  owner has a graph prepared.
- `fallback_reason`: the actual graph fallback integer code. It is separate
  from static chain selection constraints. The parser preserves unknown codes;
  code `15` is `RuntimeBoundedOverride` when runtime overrides a graph-prepared
  plan to `bounded_fallback` (including auto graph preference). Interpretation
  belongs to the producer's enum.
- `batch_size`: nonnegative system count.
- `maximum_iterations`: positive configured limit.
- `buckets`: exact-AO bucket metadata, with zero-based contiguous
  `bucket_index`, positive `ao`, positive `system_capacity`, and
  `channel_capacity` from one to two channels per system.
- `iterations`: zero-based contiguous numerical-body intervals, each with
  `iteration`, `start_ns`, `end_ns`, integer `plan_failure_record`, and one
  bucket activity record for every metadata bucket.
- `terminal_buckets` (optional): final grouped outcome counts by bucket, with
  `bucket_index`, `converged_systems`, `failed_systems`, `exhausted_systems`,
  and `unfinished_systems`. Native v1 obtains this with the per-system records
  in one grouped post-call readback of canonical SCC state.
- `terminal_systems` (optional): final per-system records with `system_index`,
  `iterations`, integer `status`, and integer `converged` (`0` or `1`). The
  list covers system indices `0..batch_size-1` in order.

Optional metadata is preserved in the summary and never used to infer runtime
selection benefit:

- `plan_token` and `wavefunction_layout_fingerprint`: nonnegative integers
  identifying the native plan and layout fingerprint.
- `start_policy`: integer 1 for FRESH, 2 for WARM, or 0 for unknown white-box
  calls.
- `graph_preference`: integer 0 for auto, 1 for tail, or 2 for chain. This is
  the requested preference; `execution_mode` remains the actual mode.
- `chain_measured_ao_bound`: nonnegative integer. The native v1 producer
  reports 40.
- `diagnostic_device_bytes`: nonnegative integer byte count reported for the
  diagnostic device ledger.
- `chain_selection_constraints`: array of strings describing static
  non-selection guards. These are distinct from the actual `fallback_reason`.

An iteration bucket record contains `bucket_index`, `active_systems`,
`active_channels`, nullable integer `submitted_solver_slots` and
`submitted_backtransform_slots`, plus cumulative `converged_systems`,
`failed_systems`, and `exhausted_systems` counts. Null means the submission
count is unavailable; it never means zero. A nonzero `plan_failure_record`
requires both submitted-slot counts to be null for every bucket because the
interrupted plan may leave stale chain telemetry.

For example, a chain can have fewer submitted solver slots than active
channels after Hamiltonian failures:

    {
      "schema": "xtbloom.cuda.scc_diagnostics.v1",
      "instrumented": true,
      "execution_mode": "device_dispatch_chain",
      "fallback_reason": 0,
      "batch_size": 2,
      "maximum_iterations": 4,
      "diagnostic_device_bytes": 4096,
      "buckets": [
        {"bucket_index": 0, "ao": 2, "system_capacity": 2, "channel_capacity": 3}
      ],
      "iterations": [
        {
          "iteration": 0,
          "start_ns": 100,
          "end_ns": 120,
          "plan_failure_record": 0,
          "buckets": [
            {
              "bucket_index": 0,
              "active_systems": 2,
              "active_channels": 3,
              "submitted_solver_slots": 2,
              "submitted_backtransform_slots": 3,
              "converged_systems": 1,
              "failed_systems": 0,
              "exhausted_systems": 0
            }
          ]
        }
      ]
    }

## Validation rules

The parser rejects booleans where integers are required, extra or missing
fields, duplicate JSON object keys, non-standard JSON constants, duplicate or
gapped bucket and iteration indices, inconsistent bucket capacities, and
invalid timestamp intervals. Bucket system capacities sum to `batch_size`.
Each bucket's channel capacity is between one and two times its system
capacity, reflecting restricted and unrestricted systems.
For legacy restricted wavefunction layouts with a null `spin_channels` view,
the producer reports one channel per system. The parser validates the emitted
counts and does not reconstruct spin mapping from IDs.

Active systems and channels are bounded by their bucket capacities and
non-increasing across numerical bodies. Each active system contributes one or
two active channels, but channel counts are validated independently: when
systems leave a bucket, their spin mix determines how many channels disappear.
The parser does not estimate channels from system counts.

Terminal counts are cumulative snapshots after each iteration and represent
disjoint converged, failed, and exhausted systems. The remaining active count
is implicit as `system_capacity - converged_systems - failed_systems -
exhausted_systems`; it must agree with the next iteration's `active_systems`.
The optional final `terminal_buckets` snapshot adds `unfinished_systems`; its
four disjoint class counts must sum exactly to each bucket's system capacity.
When `terminal_systems` is also present, its status/convergence classes must
match the aggregate bucket counts: `converged=1` is converged, status `7`
(`SCC_NOT_CONVERGED`) is exhausted, other nonzero statuses are failed, and
status `0` with `converged=0` is unfinished. A convergence flag of `1` requires
status `0`. System rows must cover the batch exactly once, and their iteration
counts cannot exceed `maximum_iterations`. Either final snapshot array may be
absent for compatibility; if both are absent, the legacy record remains valid.
The report uses present bucket counts for each bucket's `terminal` summary and
call-level `terminal_totals`, or derives totals from `terminal_systems` when
that is the only snapshot. This also works when `iterations` is empty. It never
invents a numerical-body row from terminal data.
An empty batch has no buckets or numerical bodies. Some empty public calls
bypass SCC and produce no diagnostic object; if an empty trace is emitted, it
has no numerical-body rows. Graph tail/chain execution can also skip the
numerical body after a terminal root check and emit an empty `iterations`
array. Do not synthesize a zero-duration numerical-body row.

In `device_dispatch_chain` mode, each available submitted-slot count must be
no greater than `active_channels`; Hamiltonian failure can reduce the exact
submitted count, so equality is not required. A plan failure makes both counts
null to avoid stale chain telemetry. Null means unavailable and is never
converted to zero. The graph preference and static selection constraints do
not establish that chain execution was selected or beneficial. In
`device_tail_graph` and `bounded_fallback` modes, each available count equals
the full `channel_capacity`, including when a bucket has zero active systems.
Positive-batch bounded fallback enqueues all `maximum_iterations` provider
bodies at full capacity, even after activity reaches zero; retain those rows
in the report and denominator. Tail/chain graph root exits may instead skip
the body and emit no iteration rows. Systems and spin channels are separate
units.

`start_ns` and `end_ns` use CUDA `%globaltimer` and bound the whole
numerical-body envelope. The begin timestamp is taken after the begin-side
diagnostic tally; the end timestamp is taken before the finish-side terminal
tally, excluding both observer tallies from the interval. The interval still
includes device-Graph scheduling and instrumentation launch perturbation. It
is not an additive sum of child-stage profiler timings. V1 does not provide SCC
stage timings; their status is `unknown`, not inferred from the full-body
interval.

Native CUDA kernel tests exercise ledger counts, mixed one/two-channel mapping,
terminal and overflow handling, and replay behavior. The Python parser suite
checks that emitted records preserve those count and capacity contracts,
including all-initial-failure calls whose graph exits before a numerical body.
GPU discovery alone is not execution of these tests. Qualification requires
their recorded scheduler commands and results, together with the original
input-dataset provenance. The issue acceptance ledger records that evidence;
this contract/parser work does not establish a runtime or performance pass.

## Timing use

Collecting diagnostics perturbs execution timing. The report declares
`instrumentation_perturbs_timing: true` and
`eligible_for_primary_performance_tables: false`. Use these records to
understand execution mode, exact-AO buckets, activity, submissions, and
terminal outcomes, not to claim runtime benefit. Measure capture perturbation
separately and use a separate uninstrumented run for primary timing tables. In
the report,
`submitted_solver_slack` and `submitted_backtransform_slack` are
`active_channels - submitted_slots`: positive values are active channels not
submitted, zero is an exact match, and negative values show fixed-capacity
submission beyond current activity. A null submission has null slack.
