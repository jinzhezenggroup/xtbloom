# Issue #256 Stage A batch-one CPU evidence

This directory archives the correctness-qualified public-C-ABI measurements
for PR #259. Stage A uses otherwise-idle context workers for the Mulliken
population and Hamiltonian assembly phases when the CPU batch contains one
system. The final implementation was measured from clean source commit
`133da5837941d6762ec8f3c59ed025af218df25f`; the clean `main` comparator was
measured from `fa0513302a9b4e95c6dca6c57cd8e064b422d6a4`.

## Environment

- Host: `node3`, AMD EPYC 7K62 48-Core Processor, Linux
  6.8.0-110-generic x86_64, glibc 2.35; process affinity CPUs 0-15.
- Build: CMake 4.2.1, Ninja, Release, shared library, CUDA disabled, GCC
  11.4.0 (`/usr/bin/x86_64-linux-gnu-g++-11`).
- Final xTBloom library SHA-256:
  `475dc2bac91bd6811bb21634474e0bd2bc35943e2279161eb6ab2a149eaa3f87`.
- Main comparator library SHA-256:
  `b1519386c96cec23f2b43ef6f1bf00afb42c5b76b0bd5e77eb639ca82b1839d5`.
- MKL LP64 runtime SHA-256:
  `221e89c09644d546cdc6505fc1fdecdf6490a4c57f7da6ca3b48a1c96c4860bd`.
- Threads: 16 xTBloom context workers. The benchmark process had no
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, dynamic
  threading, MKL interface, or MKL threading-layer environment variable set.

Each complete JSON artifact embeds the compiler, CMake cache, dependency,
source revision, library hash, hardware, affinity, argv, and environment
identity used for its run. Issue #348 removed the over-1-MiB
`xtbloom-warm.json` file from the current tree while retaining its compact CSV
view. Its exact historical path, byte count, SHA-256, and retrieval revision
are recorded in `benchmarks/evidence/legacy-large-artifacts.tsv`; the other
under-limit JSON artifacts remain here.

## Protocol

All rows request GFN2-xTB energy and analytic force for a batch containing one
neutral closed-shell alkane at the conformance SCC policy: 500 maximum
iterations, charge tolerance `1e-10`, energy tolerance `1e-12`, and the public
300 K electronic-temperature default. Each cell has 5 untimed warmups and 10
recorded samples. WARM performs one untimed FRESH seed before its warmups.
Setup, result inspection, and serialization are outside the timed public call.

Every retained row is `available`, has eligible clean-source provenance, and
passes correctness. The within-engine gates are `1e-8` Hartree for energy and
the manifest primary `5e-7` Hartree/bohr for force. WARM additionally validates
every raw sample against `xtbloom-fresh.json` under the live cross-engine gates.
The complete JSON retains every timing, iteration count, convergence state,
energy, and force vector; CSV is the compact view. The legacy artifact table
pins the removed WARM JSON bytes.

## Results

Median public-call latency in milliseconds, batch one with 16 context workers:

| atoms | main FRESH | Stage A FRESH | speedup | FRESH iterations | Stage A WARM | WARM iterations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 62 | 77.08 | 69.22 | 1.11x | 18 | 21.29 | 2 |
| 122 | 340.89 | 290.48 | 1.17x | 19 | 70.35 | 2 |
| 242 | 1680.21 | 1279.39 | 1.31x | 17 | 281.98 | 2 |
| 362 | 4687.53 | 3692.19 | 1.27x | 18 | 691.26 | 2 |

The 242-atom FRESH speedup is therefore 1.31x at unchanged 17-iteration
convergence. The original issue target of approximately 261 ms compared a
17-iteration xTBloom FRESH solve with a two-iteration persistent tblite solve;
it is not a like-for-like FRESH target. The remaining single-thread eigensolve
work is tracked by linked sub-issue #258.

## Phase Breakdown

Linux `perf` was unavailable on this host: `perf stat -e task-clock true`
reported restricted observability with `/proc/sys/kernel/perf_event_paranoid`
set to `4`. A temporary environment-gated `std::chrono::steady_clock` trace was
therefore applied to a detached checkout of the final source commit. The exact
uncommitted instrumentation is retained as `phase-trace.patch`; its production
source baseline is `133da5837941d6762ec8f3c59ed025af218df25f` and the traced
library SHA-256 is
`e3059a3de70a9c7f774eb403a59972274d67168dabb2a015711c142024792383`.
The trace was never added to the PR source.

`phase-iteration-samples.csv` losslessly retains 170 phase samples from ten
242-atom FRESH calls, 17 SCC iterations per call. Medians across those samples:

| SCC phase | median (ms) | mean (ms) | min (ms) | p95 (ms) | max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| prepare potentials and Hamiltonian | 4.868 | 4.885 | 4.614 | 5.127 | 5.301 |
| eigensolve and density construction | 56.919 | 56.971 | 56.667 | 57.249 | 58.106 |
| Mulliken population | 2.308 | 2.292 | 1.969 | 2.489 | 2.611 |
| energy assembly and mixer | 2.093 | 2.071 | 1.847 | 2.230 | 2.346 |

The median summed traced SCC time is 1125.95 ms per 17-iteration call. The
instrumented public-call median was 1283.33 ms; the remaining time includes
work outside these four SCC spans, notably pre-SCC setup and analytic forces.
Because the selected source was intentionally dirty with the retained trace,
this supplemental run is not claimed as eligible end-to-end evidence. The
complete clean, uninstrumented JSON artifacts are authoritative for latency
and correctness; this includes the historically retained WARM artifact.

## Commands

The exact benchmark argv is embedded in each complete historical JSON
artifact. The essential final-head commands are also retained below:

```bash
python3 benchmarks/natoms_scaling.py \
  --engine xtbloom --library "$PWD/build/batch1-cpu/libxtbloom.so.0.1.0" \
  --backend cpu --cpu-threads 16 --property force \
  --natoms 62,122,242,362 --batch-sizes 1 \
  --warmups 5 --repetitions 10 --start-mode fresh \
  --energy-atol 1e-8 --force-atol 5e-7 \
  --output-json build/benchmarks/issue-256-stage-a-final/xtbloom-fresh.json \
  --output-csv build/benchmarks/issue-256-stage-a-final/xtbloom-fresh.csv

python3 benchmarks/natoms_scaling.py \
  --engine xtbloom --library "$PWD/build/batch1-cpu/libxtbloom.so.0.1.0" \
  --backend cpu --cpu-threads 16 --property force \
  --natoms 62,122,242,362 --batch-sizes 1 \
  --warmups 5 --repetitions 10 --start-mode warm \
  --energy-atol 1e-8 --force-atol 5e-7 \
  --energy-reference-json \
    build/benchmarks/issue-256-stage-a-final/xtbloom-fresh.json \
  --cross-engine-energy-atol 5e-7 --cross-engine-force-atol 5e-6 \
  --output-json build/benchmarks/issue-256-stage-a-final/xtbloom-warm.json \
  --output-csv build/benchmarks/issue-256-stage-a-final/xtbloom-warm.csv
```

The main comparator used the same FRESH argv with the detached clean-main
runner and library. The phase trace used:

```bash
XTBLOOM_PHASE_TRACE=1 python3 benchmarks/natoms_scaling.py \
  --engine xtbloom --library build/phase-trace/libxtbloom.so.0.1.0 \
  --backend cpu --cpu-threads 16 --property force \
  --natoms 242 --batch-sizes 1 --warmups 0 --repetitions 10 \
  --start-mode fresh --energy-atol 1e-8 --force-atol 5e-7 \
  --allow-dirty-evidence \
  --output-json /tmp/issue-256-phase-trace.json \
  --output-csv /tmp/issue-256-phase-trace.csv
```

The full final-head `build/batch1-cpu` CTest suite passed 43/43 tests, and the
repository benchmark self-tests passed 43/43 tests before this bundle was
assembled. See issue #256 for the complete validation ledger.
