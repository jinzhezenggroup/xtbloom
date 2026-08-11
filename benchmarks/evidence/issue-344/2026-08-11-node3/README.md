# Issue #344 evidence: SCC warm start for browser geometry optimization

Host: `node3`, AMD EPYC 7K62 48-Core Processor, locked to CPU 0 via
`taskset -c 0`, one-threaded CPU backend to mirror the single-threaded wasm
path. The selected `libxtbloom.so` has SHA-256
`8b430df1a56076bf2439dca8ba7609468c92de13e9628d87a05f98262f5d8829` and uses
the host-isolated MKL 2026.0.0 LP64 shim provider. The `natoms` JSON records
its original clean source revision `e1532f87c3448f44c65be10176a83df3f6906a6d`;
the reviewed end-to-end rerun rebuilt byte-identical library bytes from clean
commit `cd50f22e5d83d5da26b923b1101cf335f7269f25`. The numbers below are native
single-threaded CPU samples; the WebAssembly build is ~10-20x slower per
operation, but the SCC-iteration reduction and relative speedup carry over
because the web path runs the same CPU equations through the same public C ABI.

Warm-start policy measured: the browser optimizer's first evaluation starts
SCC fresh; every successive evaluation with the same topology/charge/spin/
options starts strict `SCC_START_WARM` from the previous fully converged
electronic state, with a transparent FRESH fallback when the native gate
rejects WARM (changed identity or no converged predecessor).

## Mechanism evidence (documented harness)

`benchmarks/natoms_scaling.py` FRESH vs WARM per-call latency and SCC
iteration counts on deterministic alkanes (identical geometry per cell,
single thread, 10 warmups, 30 repetitions; WARM derives its checkpoint from
one untimed FRESH seed). `natoms-warm.json` is validated against
`natoms-fresh.json` with the conformance cross-engine energy (`5e-7 Eh`) and
force (`5e-6 Eh/bohr`) gates and is marked `eligible`.

Commands:

```bash
taskset -c 0 python3 benchmarks/natoms_scaling.py \
  --engine xtbloom \
  --library /home/jzzeng/codes/xtbloom2-worktrees/issue-344/build/issue344-cpu/libxtbloom.so \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 --start-mode fresh \
  --output-json build/benchmarks/natoms-issue344-fresh.json \
  --output-csv build/benchmarks/natoms-issue344-fresh.csv

taskset -c 0 python3 benchmarks/natoms_scaling.py \
  --engine xtbloom \
  --library /home/jzzeng/codes/xtbloom2-worktrees/issue-344/build/issue344-cpu/libxtbloom.so \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 --start-mode warm \
  --energy-reference-json build/benchmarks/natoms-issue344-fresh.json \
  --cross-engine-energy-atol 5e-7 --cross-engine-force-atol 5e-6 \
  --output-json build/benchmarks/natoms-issue344-warm.json \
  --output-csv build/benchmarks/natoms-issue344-warm.csv
```

Median per-call SCC iterations (identical geometry): FRESH 17-18, WARM 2.
Median latency (ms):

| molecules | natoms | FRESH | WARM | speedup |
| --- | --- | --- | --- | --- |
| C10H22 | 32 | 19.096 | 6.272 | 3.04x |
| C20H42 | 62 | 74.180 | 21.059 | 3.52x |
| C32H66 | 98 | 193.150 | 48.497 | 3.98x |
| C40H82 | 122 | 314.561 | 73.866 | 4.26x |

Raw samples, energy, force vectors, convergence flags, per-solve SCC counts,
and eligibility are retained in `natoms-{fresh,warm}.json`.

## End-to-end browser optimizer before/after (host-compiled adapter)

The real L-BFGS adapter (`web/xtbloom_web.c`) was compiled against the same
`libxtbloom.so` twice: the "before" variant uses `web/xtbloom_web.c` from
`cbdf755f27ab02b548783bce3573ecb4385ed167` (every SCC solve FRESH), while the
"after" variant uses reviewed commit
`cd50f22e5d83d5da26b923b1101cf335f7269f25` (FRESH then WARM). Both process
identical XYZ (angstrom) inputs, options (neutral singlet, electronic
temperature default, etol `1e-8`, qtol `1e-5`, scc max 250, max move 0.4), 10
or 5 L-BFGS steps with gradient tolerance `1e-12`, 10 warmups and 30 timed
repetitions, `taskset -c 0`. The corrected driver rejects the complete sample
set immediately if any timed repetition returns `ok:0`. Driver source and
molecule inputs are in `driver/`.

Build and water commands (set `AFTER_SOURCE` and `BEFORE_SOURCE` to clean
detached worktrees at the recorded revisions; the other molecules change only
XYZ and use 10 requested optimization steps):

```bash
AFTER_SOURCE=/path/to/clean/xtbloom-after
BEFORE_SOURCE=/path/to/clean/xtbloom-before
cd "$AFTER_SOURCE"
cmake -S . -B build/issue344-review-cpu -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/.cache/rattler/cache/bld/pkgs/mkl-2026.0.0-h0e700b2_915/lib/libmkl_rt.so
cmake --build build/issue344-review-cpu \
  --target xtbloom xtbloom_mkl_lp64_shim --parallel
gcc -O2 -std=gnu11 -I. -Iinclude \
  -Ibuild/issue344-review-cpu/generated/include \
  benchmarks/evidence/issue-344/2026-08-11-node3/driver/optbench.c \
  -o build/issue344-review-cpu/optbench_after \
  -Lbuild/issue344-review-cpu -lxtbloom -lm
gcc -O2 -std=gnu11 \
  -I"$BEFORE_SOURCE" \
  -I. -Iinclude -Ibuild/issue344-review-cpu/generated/include \
  benchmarks/evidence/issue-344/2026-08-11-node3/driver/optbench.c \
  -o build/issue344-review-cpu/optbench_before \
  -Lbuild/issue344-review-cpu -lxtbloom -lm
LD_LIBRARY_PATH=build/issue344-review-cpu taskset -c 0 \
  build/issue344-review-cpu/optbench_after \
  benchmarks/evidence/issue-344/2026-08-11-node3/driver/water.xyz \
  0 0 250 5 1e-12 10 30
LD_LIBRARY_PATH=build/issue344-review-cpu taskset -c 0 \
  build/issue344-review-cpu/optbench_before \
  benchmarks/evidence/issue-344/2026-08-11-node3/driver/water.xyz \
  0 0 250 5 1e-12 10 30
```

Median end-to-end optimization wall time (single-threaded native CPU):

| molecule | natoms | before (ms) | after (ms) | speedup |
| --- | --- | --- | --- | --- |
| water (5 steps) | 3 | 1.834 | 1.399 | 1.31x |
| ethanol | 9 | 15.562 | 13.595 | 1.14x |
| C10H22 | 32 | 135.954 | 108.567 | 1.25x |
| C20H42 | 62 | 448.348 | 346.155 | 1.30x |

"After" runs report `scc_iterations_total` (all solves, including the fresh
first step and any line-search trials): water 41, ethanol 93, C10H22 64,
C20H42 57 over 7-12 SCC solves, with exactly 1 `scc_fresh_solves` and
`scc_warm_fallbacks = 0` in every run. Both variants publish final energies
that agree to `~1e-7..1e-6 Eh` (SCC convergence tolerance), i.e. the warm
trajectory is numerically consistent with the fresh path. Final
energy/force/charge consistency across a 8-frame geometry sequence is also
checked per frame by the deterministic
`xtbloom.cpu.public_inference` `test_geometry_sequence_warm_policy_matches_fresh`
unit test (warm iterations `<=` fresh per frame, energies within
`energy_tolerance`, charges `1e-5`, forces `1e-3` Eh/bohr).

Note: the "before" water run with 10 requested steps hits the demo's
`err_linesearch` boundary on this host while the "after" run completes;
water is therefore reported at 5 steps where both variants are valid.

## Limitations

- Native single-threaded CPU, not wasm. End-to-end wasm timing requires the
  Emscripten build (`.github/workflows/pages.yml`, wasm32/wasm64 parity job);
  the wasm smoke test asserts the adapter's warm-start behavior
  (`scc_warm_solves >= 1`, `scc_fresh_solves == 1`, `scc_warm_fallbacks == 0`)
  and standalone single-point independence in that CI.
- The `natoms` harness measures identical repeated geometry (the strongest
  warm case). The optimizer's successive steps change geometry, so its warm
  benefit is smaller but still ~13-24% end-to-end on these molecules.
- `driver/optbench.c` is an ad hoc reproduction of the real adapter flow for
  the missing wasm coordinate; the authoritative browser tests are
  `xtbloom.web.adapter` (host C mock) and `web/tests/wasm_smoke.mjs` (CI).
