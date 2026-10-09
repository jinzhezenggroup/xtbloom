# Restricted SCC component crossover

`cuda_crossover.py` is a default-disabled, component-only harness for the
restricted CUDA SCC dispatch-chain versus monolithic-tail comparison. It
produces diagnostic crossover measurements; no component result decides a
production AO guard or demonstrates a benefit at real public endpoints.

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
