# Issue #315 WebAssembly linear-algebra evidence

This bundle records a correctness-qualified before/after comparison for the
browser-only wasm32 LP64 LAPACKE/CBLAS provider. It supports the narrow claim
that the Eigen 5.0.1 provider solves the supplied neutral-singlet C60 case;
it is not a native CPU/CUDA benchmark or a release-wide speed claim.

## Revisions and artifacts

- Jacobi baseline: `fc7af11845481f94455c3fd8843ef110e2dd39c0`.
- Eigen implementation: `19bfece28ab5560bc3195723e1c66f198c6b6aab`.
- Emscripten: 6.0.6, revision
  `ce75e06884093bcefb86a6b8fd56a5d62a4cc245`.
- CMake: 4.2.1; build type `Release`; wasm32 flags `-m32 -fPIC`.
- Runtime: Node 24.19.0 on Linux x86_64, AMD EPYC 7K62 48-Core Processor.
- Each measurement process was pinned to CPU 0 with `taskset -c 0`.

The baseline wasm/data/side-module hashes were reproduced from a clean detached
worktree at `fc7af1`; they matched the already measured baseline artifacts
byte-for-byte. The Eigen measurement began from a clean worktree at
`19bfece`. The JSON files contain absolute local artifact paths, exact hashes,
sizes, timestamps, every raw sample, and the result used for correctness.

## Protocol

Both engines were initialized once per process. Each workload used the
synchronous public Web adapter entry point `xtbloom_web_compute`, requested
energy, atomic charges, and analytic forces, and used charge 0, no unpaired
electrons, the 300 K default electronic temperature, energy tolerance `1e-8
Eh`, charge tolerance `1e-5 e`, and at most 250 SCC iterations. One warmup was
followed by five retained samples.

The workloads were:

- water, 3 atoms;
- the issue #315 C60 geometry, 60 atoms and 240 orbitals.

The Eigen C60 gate required SCC success in 13 iterations, energy within
`1e-6 Eh` of the native public-C-ABI checkpoint, total charge below `2e-8 e`,
and the selected charge/force checkpoints within `2e-9 e` and `2e-8
Eh/bohr`. Observed maximum checkpoint errors were `1.88e-11 e` and
`4.48e-10 Eh/bohr`. The Jacobi baseline was required to retain its known
`err_compute: eigensolver failed` result; its timing is a failed-call latency,
not a valid completed C60 calculation.

## Results

| Provider/workload | Five-sample median | Range | Scientific result |
| --- | ---: | ---: | --- |
| Jacobi, water | 3.097 ms | 1.044-7.126 ms | success, `-5.06262145 Eh`, SCC 9 |
| Eigen 5.0.1, water | 4.715 ms | 1.194-5.073 ms | same energy/status/iterations |
| Jacobi, C60 | 5351.220 ms | 5348.803-5360.630 ms | eigensolver failure, no result |
| Eigen 5.0.1, C60 | 1044.450 ms | 1037.186-1064.910 ms | success, `-128.249019 Eh`, SCC 13 |

The tiny water timings are visibly noisy and bimodal at this sample count, so
no small-molecule latency conclusion is drawn. For C60, the meaningful outcome
is successful computation in about 1.04 s median where the old provider failed
after about 5.35 s; these are not presented as a 5x speedup because the baseline
did not produce a scientifically valid result.

Artifact payload changed as follows:

| Artifact | Jacobi | Eigen 5.0.1 | Change |
| --- | ---: | ---: | ---: |
| `xtbloom_web.wasm` | 1,138,049 B | 1,138,058 B | +9 B |
| side module / `xtbloom_web.data` | 11,483 B | 136,277 B | +124,794 B |
| complete staged site file payload | 2,073,156 B | 2,428,898 B | +355,742 B (+17.16%) |

## Commands

The clean baseline was rebuilt with the pinned SDK using:

```console
git worktree add --detach /tmp/xtbloom-315-baseline-clean \
  fc7af11845481f94455c3fd8843ef110e2dd39c0
emcmake cmake -S /tmp/xtbloom-315-baseline-clean \
  -B /tmp/xtbloom-315-baseline-build -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF -DXTBLOOM_BUILD_TESTS=OFF \
  -DXTBLOOM_BUILD_WEB_DEMO=ON -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-m32 -fPIC" \
  -DCMAKE_CXX_FLAGS="-m32 -fPIC"
cmake --build /tmp/xtbloom-315-baseline-build --parallel 2
sha256sum /tmp/xtbloom-315-baseline-build/web/site/xtbloom_web.wasm \
  /tmp/xtbloom-315-baseline-build/web/site/xtbloom_web.data \
  /tmp/xtbloom-315-baseline-build/web/libscipy_openblas.so
```

Raw timing commands were:

```console
taskset -c 0 node measure_web.mjs jacobi-baseline \
  fc7af11845481f94455c3fd8843ef110e2dd39c0 \
  build/wasm32-web-baseline/web/site false /tmp/issue315-jacobi.json
taskset -c 0 node measure_web.mjs eigen-5.0.1 \
  19bfece28ab5560bc3195723e1c66f198c6b6aab \
  build/wasm32-web-eigen/web/site true /tmp/issue315-eigen.json
```

`measure_web.mjs` is the exact measurement source used on this machine. Its
absolute imports identify the measured checkout; the raw JSON is authoritative.

## Limitations

- Node's WebAssembly runtime was measured, not a graphical browser event loop,
  network download, cache fill, Worker startup, or rendering.
- One CPU and one wasm32 configuration were measured with five retained samples.
- The baseline C60 row is a deterministic failure and cannot support a speedup
  ratio.
- The staged-site total is the sum of file payload bytes and excludes directory
  metadata, HTTP compression, and transport headers.
