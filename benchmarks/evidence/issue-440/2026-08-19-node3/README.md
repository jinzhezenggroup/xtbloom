# Issue #440 WebAssembly SIMD128 and O3 evidence

This bundle supports the narrow claim that the browser-only, single-threaded
wasm32 public Web adapter reduces correctness-qualified GFN2-xTB C60 compute
latency under Node/V8 when the previous scalar build is replaced by the final
SIMD128 build. It is not a native CPU/CUDA or browser-wide claim.

## Revisions and environment

- Previous scalar baseline: clean detached revision
  `3c21f50195389b093941eb5ed6f1143b8802f96e`.
- Final scalar and SIMD variants: clean revision
  `3b9fac09ecb0b2da1596022ddda786fa9acfb885`.
- Emscripten: 6.0.6, revision
  `ce75e06884093bcefb86a6b8fd56a5d62a4cc245`.
- Compiler bundle SHA-256:
  `6cb7cf45ad85b0b9b466a44cc4bb65ef380e47f040ce73e6f956bde782787f46`.
- CMake 4.2.1; Ninja 1.13.0; `Release`; wasm32 `-m32 -fPIC`.
- Eigen 5.0.1 official archive SHA-256:
  `e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec`.
- Node 24.19.0, V8 13.6.233.17-node.51, Linux x86_64
  6.8.0-110-generic, AMD EPYC 7K62 48-Core Processor.
- Each measurement process was pinned to CPU 0 with `taskset -c 0`.
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` were 1.
  The ending load average was 1.51, 1.45, 1.32.

`build-metadata.txt` records CMake-cache, JavaScript glue, static-library, and
Wasm artifact identities. The JSON reports retain every raw timing sample,
artifact size/hash, runtime identity, source revision, and scientific result.

## Variants and protocol

The variants were built and measured separately:

- `baseline`: previous scalar Web build, with `-O2` on the custom provider,
  adapter, and final link stages (the core Release objects already used `-O3`);
- `scalar`: final `-O3` build with `XTBLOOM_WEB_ENABLE_SIMD=OFF`;
- `simd`: final `-O3` build with `XTBLOOM_WEB_ENABLE_SIMD=ON`.

The synchronous public `xtbloom_web_compute` adapter requested a batch of one
GFN2-xTB system with energy, charges, and analytic forces, charge 0, no
unpaired electrons, energy tolerance `1e-8 Eh`, charge tolerance `1e-5 e`, at
most 250 SCC iterations, and a strict FRESH SCC start. The timed boundary
includes the Emscripten `ccall` and JSON parse. Each workload had one untimed
correctness/warmup call. Two processes retained 10 samples each per variant,
in order `baseline-a`, `scalar-a`, `simd-a`, `simd-b`, `scalar-b`,
`baseline-b`, for 20 samples per result row.

Both water and C60 converged in every warmup and retained call. All variants
reported water energy `-5.06262145 Eh` in 9 SCC iterations and C60 energy
`-128.249019 Eh` in 13 SCC iterations. The harness also required successful
SCC convergence and complete charge/force arrays; C60 additionally had to
match the independent energy and iteration checkpoint. Before timing, all
three sites passed `wasm_smoke.mjs`; both final sites passed the standalone
Eigen LAPACKE/CBLAS test.

## Results

| Variant | Water median (range), ms | C60 median (range), ms | C60 vs baseline |
| --- | ---: | ---: | ---: |
| Previous scalar (O2 Web stages) | 4.393 (0.995-14.344) | 1040.820 (1036.830-1059.835) | 1.000x |
| O3 scalar | 2.877 (0.995-16.097) | 1027.548 (1020.341-1044.837) | 1.013x |
| O3 SIMD128 | 3.938 (0.976-20.992) | 874.412 (869.024-893.495) | 1.190x |

The final SIMD build reduced C60 median latency by 15.99% relative to the
previous scalar build. Within the final revision, SIMD reduced the O3 scalar
median by 14.90% (1.175x); the final scalar median was 1.28% lower than the
previous-build median. The C60 SIMD range is disjoint from both scalar ranges.
Water is visibly bimodal and its ranges overlap, so no small-molecule speed
claim is made.

Complete staged engine payload (`xtbloom_web.js`, main Wasm, and side Wasm):

| Variant | Bytes | Change from baseline |
| --- | ---: | ---: |
| Previous scalar (O2 Web stages) | 1,860,352 | - |
| O3 scalar | 1,883,205 | +22,853 (+1.23%) |
| O3 SIMD128 | 1,899,135 | +38,783 (+2.09%) |

## Commands

The baseline used the following configure shape; the final builds used the
same command with the final source directory and explicitly selected
`XTBLOOM_WEB_ENABLE_SIMD=OFF` or `ON`:

```console
emcmake cmake -S /tmp/xtbloom-441-baseline \
  -B /tmp/xtbloom-441-build/baseline -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF -DXTBLOOM_BUILD_TESTS=OFF \
  -DXTBLOOM_BUILD_WEB_DEMO=ON -DBUILD_SHARED_LIBS=OFF \
  -DXTBLOOM_WEB_EIGEN_ARCHIVE=/tmp/eigen-5.0.1.tar.gz \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-m32 -fPIC" -DCMAKE_CXX_FLAGS="-m32 -fPIC"
cmake --build /tmp/xtbloom-441-build/baseline --parallel 4
cmake --build /tmp/xtbloom-441-build/baseline \
  --target xtbloom_web_linalg_test --parallel 4
node web/tests/wasm_smoke.mjs \
  /tmp/xtbloom-441-build/baseline/web/site
```

One retained process per variant was run as follows, with `a`/`b` substituted
for the run label and the matching source revision recorded in the last field:

```console
taskset -c 0 /tmp/emsdk-6.0.6/node/24.19.0_64bit/bin/node \
  web/tests/wasm_benchmark.mjs \
  /tmp/xtbloom-441-build/simd/web/site simd-a 10 \
  3b9fac09ecb0b2da1596022ddda786fa9acfb885 > simd-a.json
```

## Limitations

- Node/V8 was measured, not Chromium/WebKit Worker scheduling, network
  download, cache fill, startup, rendering, or user-interface latency.
- One x86_64 machine and one wasm32 configuration were measured.
- The baseline and final revisions differ by the PR plus intervening main
  commits (CodSpeed CI and native MKL isolation); the same-revision
  scalar/SIMD pair isolates SIMD at the final source.
- Water is below a stable timing granularity for this protocol; its raw data is
  retained only to satisfy the small-molecule coverage criterion.
