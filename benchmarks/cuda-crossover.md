# Restricted SCC component crossover

`cuda_crossover.py` is a default-disabled, component-only harness for the
restricted CUDA SCC dispatch-chain versus monolithic-tail comparison. It
produces diagnostic crossover measurements; no component result decides a
production AO guard or demonstrates a benefit at real public endpoints.

## Isolated public-path graph override

For prospective public endpoint/E2E controls, an independent, default-OFF
`-DXTBLOOM_CUDA_SCC_BENCHMARK_OVERRIDES=ON` build accepts
`XTBLOOM_CUDA_SCC_BENCHMARK_MODE=auto|tail|chain`. Unlike #512's optional
`XTBLOOM_CUDA_SCC_DIAGNOSTICS` device ledger, this option does not instrument
each SCC iteration. Both forced modes use the existing SCC Graph families in
the same native binary and still enter through `xtbloom_compute`. They do not
change equations, provider precision, initial guesses, mixing, tolerances,
spin policy, or the production 40-AO/singleton selector.

The CUDA execution-cache owner captures a value-only preference once at its
creation. Later environment changes do not alter that owner's topology, Graph,
or strict-WARM identity; create a separate owner/context before comparing
another mode. No replay reads the environment or makes a host-side per-iteration
dispatch choice. Default-OFF builds ignore this variable completely. An unset
variable or `auto` in an opt-in build preserves the ordinary selector.

Explicit `tail`/`chain` is restricted to molecular GFN2 with one shared spin
channel, without point charges, periodic charge-response/native-lattice
operators, or an external energy callback. Malformed modes and out-of-scope
requests fail before candidate or caller-output commit. A requested family
that cannot be built returns `NOT_IMPLEMENTED` with its fallback reason rather
than silently measuring the bounded or other Graph family.
The native-periodic strain and host-callback CPU compatibility bridges check
the same frozen owner selection before execution; they cannot bypass these
restrictions to produce a falsely labeled forced-mode result.

Only synchronous `xtbloom_compute`/`xtbloom_plan_compute` may use a forced
family. Public asynchronous enqueue uses a different conditional request Graph
that intentionally captures bounded SCC; explicit forced modes reject that
route with `NOT_SUPPORTED`, before request/caller-output publication, rather
than mislabeling it as chain or tail. `auto` and default-OFF builds retain the
ordinary asynchronous API. The internal asynchronous stage launcher is not the
public enqueue API and still launches its prepared SCC family.

This override is a research prototype, not an accepted dispatch contract or
permission to remove the 40-AO guard. Forced-mode outputs still require
independent oracle, state/status/failure, memory-mode, cache/WARM, and applicable
Graph/sanitizer validation. Keep public setup/preparation/publication costs in
E2E; static artificial masks remain component probes, not declining real SCC
workload throughput. Existing component records and reference failures are not
superseded by enabling the option.

## Component matrix

The fixed matrix contains exact basis AO counts `40, 41, 62, 122, 180`, batch
sizes `1, 8, 32, 64, 128, 256`, and requested active fractions `1, 0.5, 0.25`.
Each coordinate runs from one compiled native executable. AO40, AO41, and AO180
use deterministic synthetic carbon-cluster fixtures; AO41 is the restricted
C10H+ cation with 40 GFN2 valence electrons (60 total electrons). These
synthetic geometries are component probes only. AO62
uses the all-trans C10H22 geometry from the established alkane builder, and
AO122 uses the existing C20H42 SCC fixture. The native harness checks every
system's actual basis orbital count and restricted even-electron state before
it starts timing.

Every call replays the fixture's FRESH initialization, then sets a deterministic
terminal mask and runs one bounded SCC iteration. Chain and tail execute as an
alternating paired order from the same initial state. The selected mask size
rounds the requested fraction to the nearest whole system; both the requested
and realized fractions and the full mask are recorded. A batch of one at a
requested 0.25 fraction therefore records zero active systems.
Masks are static for each one-body probe: this does not measure declining
activity across a converging SCC trajectory. Matching unconverged states are
valid replay results; convergence is not required for this component parity
check.

The measured sample record keeps per-system iterations, status codes, and
convergence flags for both modes, plus paired CUDA event times and full
chain/tail state-parity results. Parity covers energies, eigenvalues,
occupations, density matrices, shell and atomic charges, dipoles, quadrupoles,
iterations, statuses, and convergence. It is a scheduler replay check, not an
independent numerical oracle or public API qualification. The fixture does not
qualify forces, molecular stability, production startup policy, or a real
endpoint. The parser cross-checks paired iteration/status/converged vectors;
`state_parity=true` cannot override contradictory ledgers. Coordinate summaries
report setup time, launch counts, graph executable counts (chain family plus
root; tail root plus body), fixture arena bytes, and the separately
owned FRESH device checkpoint exactly once. Graph subtotals are labeled known
control/table bytes. The retained-device subtotal explicitly excludes opaque
CUDA graph/executable storage and provider allocations; it is not total device
memory use. Matrix metadata records NVML GPU/driver inventory, CMake CUDA
compiler/build identity, and source/binary hashes. Native execution selects CUDA
device ordinal zero, but its association with that inventory is `UNVERIFIED`:
NVML inventory need not follow `CUDA_VISIBLE_DEVICES` remapping. Inventory alone
is not hardware qualification. A failed Git query records the dirty state as
JSON `null` (unknown), never as clean.

The full 90-coordinate matrix requires an explicit `--run`. Every coordinate
has a finite timeout, warmup count, sample count, and conservative host-memory
preflight; coordinates over the configured budget remain in the output as
`unavailable`. The JSONL output preserves failed and unavailable coordinates
alongside every complete sample ledger. Launch failures, including a
non-executable binary, remain explicit failures without stopping the matrix.
Malformed/truncated later JSONL preserves complete prefix records, the raw
stdout, and a parse diagnostic rather than discarding earlier samples. Existing
native CTest names and default invocations remain unchanged; the integrity
module is included in the benchmark CTest and nox suites. No GPU runtime is
started by `--list-grid` or the harness tests.

List the default matrix without launching the native executable:

```bash
python3 benchmarks/cuda_crossover.py --list-grid
```

Run one bounded development slice after building the CUDA production-test
target on a scheduled NVIDIA GPU:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc 'python3 benchmarks/cuda_crossover.py --run \
  --binary build/cuda/xtbloom_cuda_scc_iteration_production_test \
  --cmake-cache build/cuda/CMakeCache.txt \
  --output /tmp/issue-515-crossover-smoke.jsonl \
  --ao-counts 122 --batches 8 --active-fractions 0.5 \
  --warmups 1 --samples 3 --timeout 900'
```

The default publication protocol uses five warmup pairs and 30 measured pairs.
A full matrix is still component evidence; the >40 AO restricted crossover
claim remains open until its required independent real public endpoints and
validation gates pass. Component times do not infer a benefit/no-go decision,
production AO threshold acceptance, force/oracle qualification, or public
endpoint acceptance.

The hardware-independent integrity tests are:

```bash
python3 -m unittest -v benchmarks.test_cuda_crossover
```

## Matched sanitizer controls

The native executable also accepts `--crossover-sanitizer-control AO B DEN MODE`
for exactly `41 1 1`, `122 128 1`, and `180 256 4`, with `MODE` equal to `direct`
or `host-graph`. These use the same fixture, seed515, FRESH checkpoint, static
terminal mask and one-body maximum as the component matrix. The host-Graph
control captures the existing restricted direct launcher, replays it from the
same checkpoint, and checks the complete state and per-peer ledgers against
direct execution. It does not use a device-launched tail. Both modes report a
machine-readable diagnostic record; no GPU or unsupported provider capture is
explicitly unavailable, not a pass.

Six `xtbloom.cuda.scc_iteration_crossover_*` CTests register the controls. Run
them and Compute Sanitizer on the assigned GPU through a finite scheduled job,
preserving the scheduler's device visibility. These controls help distinguish
ordinary execution/capture from device-tail instrumentation reports. They do
not automatically waive an error, inherit an earlier tool disposition, or
qualify forces, an independent oracle, a public endpoint or E2E performance.

## Compact offline table

Summarize the original matrix JSONL without retaining oversized raw samples in
Git. The analyzer reuses the native record validator, requires the complete
requested grid, retains failed/unavailable coordinates, cross-checks duplicate
native records and wrapper outcomes, and excludes warmups from every statistic:

```bash
python3 -m benchmarks.cuda_crossover_summary \
  --input /absolute/external/component-matrix.jsonl \
  --json-output /absolute/external/component-summary.json \
  --csv-output /absolute/external/component-summary.csv \
  --table-csv-output /absolute/external/component-table.csv
python3 -m unittest -v benchmarks.test_cuda_crossover_summary
```

The summary records producing source/binary/cache identities, raw byte count
and SHA-256, analyzer source/dirty identity, static fixture/mask identity,
per-coordinate outcome/ledger counts, setup and known memory/executable
subtotals. It does not infer a materialized geometry hash or native GPU/NVML
association that the producer did not record. Pair ratios are descriptive
`chain_ms/tail_ms` quantiles, not confidence intervals or an adoption decision.
The producing revision remains unchanged when a later clean analyzer processes
an older raw artifact. `claim_eligible` remains false even with complete replay
parity; the real public endpoint, oracle, E2E, declining-activity and sanitizer
gates still own qualification.

The optional table projection retains every requested coordinate and outcome,
with six-significant-digit display floats; exact values, failure reasons,
limitations and producing/analyzer/raw identities remain in the full external
JSON/CSV. The JSON pins the projection's byte count and SHA-256. Projection is
rejected before writing any output if it exceeds 20,000 bytes; it never drops
coordinates to fit. This local table bound does not replace the repository's
1MiB-per-file and 16MiB-total evidence gates. Native integer fields reject
boolean/integral-float aliases, and every paired iteration ledger must respect
the declared one-body bound before any timing pair is usable. Malformed failed
rows retain their diagnostics but contribute no usable statistics.
Schema-complete failures can retain descriptive event distributions with their
failure status; they never become eligible performance results. Unexpected CUDA
initialization errors fail a control instead of being hidden as a no-device skip.
