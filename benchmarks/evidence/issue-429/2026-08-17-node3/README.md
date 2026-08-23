# Issue #429 CPU ISA dispatch evidence

This bundle qualifies the retained baseline plus AVX2/FMA Mulliken dispatch on
node3 at committed revision
`dd8193d9b181882dbf41ecddf919724df3ddba0e`. The important comparison uses the
same `libxtbloom.so` and changes only the context-time
`XTBLOOM_CPU_ISA=baseline|avx2` selection. Positive speedup values mean AVX2 is
faster. Every retained JSON row reported `availability=available`,
`correctness.status=pass`, and eligible clean-source evidence.
Subsequent native-library source formatting and a test-only non-AVX2 host guard
rebuilt to the identical candidate-library SHA-256 recorded below.

## Result

For one pinned CPU worker, AVX2/FMA reduced public force-call latency by
9.7-11.6% for FRESH SCC and 3.7-5.0% for WARM SCC over the 32-122 atom alkane
coordinates. Three alternating WARM process rounds were used for the primary
serial result. `XTBLOOM_CPU_ISA=auto` selected the same AVX2 path on this host:
its FRESH medians were 16.904, 62.129, 156.840, and 253.716 ms, and its WARM
medians were 5.899, 19.368, 43.541, and 65.556 ms.

The throughput controls remained small compared with the serial gain. The
32-system FRESH batch improved 3.3% with 16 workers and 0.6% with automatic
48-worker selection. Its three-round WARM median improved 0.5% with automatic
workers and regressed 0.6% with 16 workers. The latter is retained honestly as
a sub-1% control regression, not described as a speedup. The 122-atom
single-system control improved 0.5-1.1% with 16 workers and 0.6-2.5% with
automatic workers. SCC iteration counts were unchanged in every coordinate.

The compact headline values are in `RESULTS.csv`. For rows with multiple
process rounds, that file reports the median of the per-process medians; every
process contained 30 measured calls. Single-process FRESH controls used 20
measured calls, while the primary serial FRESH sweep used 30.
`DISTRIBUTIONS.csv` retains every process/coordinate separately with its sample
count, requested and resolved ISA, min, inclusive Q1, median, inclusive Q3,
max, mean, p95, SCC-iteration median, and correctness status. New runner
artifacts record the resolved ISA through public context creation. These raw
artifacts predate that field, so the summarizer fails closed except for the
exact candidate-library SHA above, whose forced AVX2 context creation proves
that `auto` resolved to AVX2/FMA on node3. The CSV is generated only after
verifying all 38 omitted raw JSON files against `RAW_SHA256SUMS`:

```bash
python3 summarize_distributions.py /path/to/raw-json-directory \
  > DISTRIBUTIONS.csv
```

## Attribution to the retained leaf kernels

The same-binary forced-mode comparison isolates the selected immutable
Mulliken callback table; the generic library, eigensolver provider, input,
thread policy, and all other code bytes are identical. Object-level inspection
showed:

- `features.cpp.o` and `mulliken_kernels_baseline.cpp.o` were compiled with
  `-march=x86-64 -mno-avx -mno-avx2 -mno-fma -fno-lto` and contained no
  VEX/AVX instructions;
- `mulliken_kernels_avx2.cpp.o` alone used
  `-march=x86-64 -mavx2 -mfma -fno-lto` and contained scalar
  `vfmadd*sd`/`vfnmadd*sd` instructions;
- the final library contained distinct hidden baseline and AVX2 population
  and Hamiltonian leaf symbols without adding public exports.

The useful specialization is therefore hardware scalar FMA in the selected
Mulliken contraction and Hamiltonian leaves, not broad whole-binary YMM
vectorization. A diagnostic static `-pg` build of the public CPU inference test
also sampled the baseline Hamiltonian and population callbacks directly and
recorded the corresponding AVX2 leaf call counts. `GPROF_SUMMARY.txt` retains
the exact commands, binary/build/provider hashes, selected flat-profile rows,
and the 0.01-second sampling limitation. This gprof observation is only leaf
attribution, not the retained latency claim. Linux hardware PMU profiling
remained unavailable because `perf_event_paranoid=4`.

## Build and machine identity

- Host: `node3`, Linux 6.8.0-110-generic, glibc 2.35.
- CPU: AMD EPYC 7K62 48-Core Processor, 48 logical CPUs.
- Compiler: GCC 11.4.0; CMake 4.2.1; Ninja Release (`-O3 -DNDEBUG`).
- Candidate library SHA-256:
  `b15e90b7a3d88cc14dcbd0e0baea59d0a82970386dd27e239829c0e80ec8ab73`.
- `compile_commands.json` SHA-256:
  `d7bdee2ac9cf2d6285ce66187d31f204571c1d99004ead89458a3cba44552406`.
- `CMakeCache.txt` SHA-256:
  `e6a25740df858b75e739f7bcabe61eae96aebf5e327a98e40e61d51bc1c73caf`.
- LP64 SciPy OpenBLAS provider SHA-256:
  `b2dfe24b9aa11cf1d1cec8edbca9423b50cfd186b486d59dd4efe45826261a98`.
- `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`; no MKL variables were
  set. Serial runs used `taskset -c 0`, 16-worker runs used `0-15`, and
  automatic-worker runs used `0-47`.

The build command was:

```bash
cmake -S . -B build/issue429-candidate -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=OFF \
  -DXTBLOOM_CPU_LINALG_LIBRARY="$XTBLOOM_LINALG_PROVIDER" \
  -DXTBLOOM_ENABLE_AVX2_DISPATCH=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/issue429-candidate --parallel
```

## Timing protocol and reproduction

Every sample timed one synchronous public-C-ABI `xtbloom_compute` force call
with a persistent context, descriptor, options image, and caller-owned output
buffers. FRESH used the strict conformance SCC thresholds. WARM performed one
untimed FRESH seed, ten WARM warmups, then measured only strict WARM calls; the
matching FRESH artifact supplied the independent energy/force reference.

The primary serial command pattern was:

```bash
taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  XTBLOOM_CPU_ISA=<baseline|avx2|auto> \
  python3 benchmarks/natoms_scaling.py \
  --library build/issue429-candidate/libxtbloom.so \
  --output-json build/issue429-candidate-evidence/<name>.json \
  --output-csv build/issue429-candidate-evidence/<name>.csv \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 --start-mode <fresh|warm> \
  [--energy-reference-json build/issue429-candidate-evidence/<matching-fresh>.json]
```

The controls used the same command with `taskset -c 0-15 --cpu-threads 16` or
`taskset -c 0-47 --cpu-threads 0`, and selected `--natoms 122 --batch-sizes 1`
or `--natoms 32 --batch-sizes 32`. Their FRESH runs used five warmups and 20
samples. The three-round coordinates were run in alternating baseline/AVX2
process order.

The 38 raw JSON documents total 75,964,700 bytes and contain every timing,
energy, force, SCC iteration, command, affinity, environment, compiler,
provider, source, and binary identity. They exceed the repository evidence
budget and are intentionally omitted. `RAW_SHA256SUMS` records their exact
digests and filenames; all compact values required for the claims are retained
in this bundle.

## Correctness qualification

The exact implementation passed the complete 89-test CPU CTest set twice,
once with `XTBLOOM_CPU_ISA=baseline` and once with
`XTBLOOM_CPU_ISA=avx2`. Both runs included public GFN2 conformance and
invariants, restricted/unrestricted SCC traces, energies, forces, charges,
failure behavior, ABI symbols, licensing, and the independent oracle tests.
No tolerance was changed. A separately configured
`-DXTBLOOM_ENABLE_AVX2_DISPATCH=OFF` build linked and passed the focused public
inference, runtime, dispatch, and Mulliken tests.

The object inspection used:

```bash
objdump -d -M intel <generic-or-baseline-object> | \
  rg '^\s*[0-9a-f]+:.*\bv[a-z0-9]+'
objdump -d -M intel <avx2-object> | rg 'vfmadd|vfnmadd|ymm'
readelf -Ws build/issue429-candidate/libxtbloom.so | \
  rg 'mulliken_(population|hamiltonian)_chunk_(baseline|avx2)'
```

The first command produced no generic/baseline matches; the AVX2 and hidden
symbol commands produced the expected results.
