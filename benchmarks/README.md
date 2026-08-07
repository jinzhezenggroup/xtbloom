# Cross-library benchmark harness

`run.py` measures end-to-end inference through gpuxtb's public C API. It keeps
the context, ragged batch descriptors, and result buffers alive between calls,
separates setup, first-call (cold), and steady-state timings, and explicitly
synchronizes CUDA after every measured call. Device result downloads used for
correctness checking happen after timing.
For QM/MM force rows the measured public call requests both QM atomic forces
and external-point-charge forces; both are compared with committed goldens.

Run the complete requested gpuxtb matrix on one allocated GPU:

```bash
srun --gres=gpu:1 env \
  LD_LIBRARY_PATH=/tmp/gpuxtb-reference-env.E0KcEA/lib:/group/software/cuda-12.9.1/lib64:/group/software/deepmd-kit-3.1.1/lib \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
  python3 benchmarks/run.py \
    --library build-cuda-shared/libgpuxtb.so.0.1.0 \
    --tblite-library /path/to/validated/libtblite.so \
    --xtb-library /tmp/gpuxtb-reference-env.E0KcEA/lib/libxtb.so.6.7.1 \
    --xtb-executable /tmp/gpuxtb-reference-env.E0KcEA/bin/xtb \
    --engines gpuxtb,tblite,xtb,dxtb \
    --batch-sizes 1,8,32,128 \
    --properties energy,force \
    --workloads gas,qmmm
```

For a quick CUDA smoke run:

```bash
srun --gres=gpu:1 env \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/deepmd-kit-3.1.1/lib \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
  python3 benchmarks/run.py \
    --library build-cuda-shared/libgpuxtb.so.0.1.0 \
    --engines gpuxtb --backends cuda --cuda-memory-modes host,device \
    --workloads gas --properties energy,force --batch-sizes 1,8 \
    --warmups 1 --repetitions 2
```

The default artifacts are `build/benchmarks/matrix.json` and
`build/benchmarks/matrix.csv`. JSON retains raw samples, environment variables,
hardware/runtime information, exact git revisions, library hashes, correctness
errors, SCC iterations, and memory snapshots. CSV contains the main comparison
columns and embeds raw timing and memory evidence as JSON fields.

The xTB baseline dynamically loads the public C API from `--xtb-library`.
Every logical batch system owns a persistent environment, molecule, calculator,
and result object; measured batch execution is an in-process serial loop over
`xtb_updateMolecule`, `xtb_singlepoint`, and the requested getters. Process
startup and object construction are excluded from steady state and reported as
setup. xTB 6.7.1 has no energy-only C API flag, so an `energy` row still pays
for its native full single-point calculation; this is recorded in each row and
must be considered when comparing it with gpuxtb energy-only inference.

QM/MM xTB rows use `xtb_setExternalCharges` with the corpus' source atomic
numbers, which exactly represents the selected element-hardness case. The
adapter rebinds the same persistent external-charge arrays inside each measured
call because libxtb 6.7.1 otherwise accumulates PC gradients when the object is
reused. This required C API work is included in latency.

The tblite baseline likewise uses its public C API through
`--tblite-library`, with one persistent context, structure, calculator, and
result per logical batch system. Its measured serial loop includes geometry
update, single-point execution, and requested getters, but excludes process
startup and object construction. The tblite 0.6 public C API does not expose
discrete external point charges, so requested QM/MM rows are retained as
explicit `unavailable` coordinates rather than silently omitted.

The dxtb baseline uses one persistent in-process PyTorch `Calculator` and
input tensors from `--dxtb-source`. CPU and CUDA rows are selected with
`--dxtb-backends`; CUDA timing ends with an explicit PyTorch synchronize and
host result download remains outside the measured interval. Every measured
call includes `Calculator.reset()` so dxtb's tensor-identity result cache
cannot turn repeated geometry into a cache-hit benchmark. The active Python
environment must provide dxtb's normal dependencies (including
`tad-libcint` for GFN2); otherwise requested rows remain explicitly
`unavailable`. Discrete QM/MM point-charge mapping is not implemented yet and
is likewise reported as unavailable rather than silently omitted.

Use `--fail-on-correctness` when the command should return status 2 if an
available row exceeds the committed primary conformance tolerance. Artifacts
are written before that status is returned.

Run the hardware-independent harness self-check with:

```bash
python3 -m unittest -v benchmarks.test_run

# Requires PyTorch; uses a fake differentiable dxtb runtime, not dxtb itself.
python3 -m unittest -v benchmarks.test_dxtb_adapter
```

## CPU FRESH/WARM natoms evidence

`natoms_scaling.py` is a public-C-ABI latency sweep for strict gpuxtb SCC start
policy evidence, a persistent public-C-API natoms baseline for tblite, and
explicit xTB diagnostics. A latency row is eligible for a performance claim
only when both its within-engine correctness and explicit FRESH comparison
pass. Each cell retains one context, ragged descriptor, options image, and
caller-owned result buffers. A `warm` run first performs one
untimed `FRESH` call on the identical topology/options identity, then uses
`WARM` for every warmup and measured call. JSON rows retain every latency,
energy, complete requested force vector, SCC iteration count, convergence flag,
and per-system status. FRESH/WARM force vectors are compared numerically; a
finite-only force result cannot pass correctness.
JSON is the authoritative audit artifact. CSV is a compact row summary for
ordinary tools: it omits repeated run identity, seed/cold observable payloads,
measured raw samples, raw latency vectors, and energy/force reference vectors.
Use JSON to reproduce correctness calculations or inspect individual samples.
The default FRESH/WARM force-drift gate is the manifest's primary 5e-7
Hartree/bohr tolerance; its source and hash are stored in the protocol.
Correctness flags may select an equal or tighter gate, but the CLI rejects any
attempt to exceed the committed primary or live cross-engine tolerances. A
supplied FRESH artifact is also rejected if its recorded protocol used wider
energy or force gates.

gpuxtb benchmark rows pin the same SCC convergence settings as the conformance
oracle: 500 iterations, `1e-10` charge tolerance, and `1e-12` energy tolerance,
while retaining the public 300 K electronic-temperature default. A FRESH
artifact is accepted only when those options are present and every measured
raw sample is intact. The loader checks the repetition count, consecutive
sample indices, complete finite energy/force vectors, and recomputes the
recorded reference and maximum-drift summaries from those samples. Dependent
WARM and cross-engine comparisons likewise inspect every measured raw sample
and report the maximum delta across the entire run, not just the seed or cold
snapshot.

The deterministic all-trans-like alkane builder uses tetrahedral terminal and
interior C-H directions. Its hardware-free test checks the full pair-distance
matrix for 32, 62, 98, and 122 atoms and rejects nonbonded separations below
1.75 angstrom.

Output filenames are mandatory and existing files are rejected by default.
Run FRESH and WARM as separate processes and artifacts; the harness never scans
an output directory or merges files from an earlier run. JSON/CSV pairs use an
exclusive pair reservation, unique same-directory staging files, and rollback
on publication failure. A stale partial pair or concurrent/stale reservation is
rejected. The following commands pin one logical CPU and the one-thread BLAS
contract. Replace the library and CPU number with the paths allocated on the
evidence machine.

Final evidence must run from a clean committed checkout. By default the harness
rejects a dirty benchmark repository or dirty selected-library source.
`--allow-dirty-evidence` exists only for development: it labels the run
`development_only_dirty`, writes the diagnostic artifact, and exits nonzero.
Never use it in the commands or evidence bundle below.

Configure and build the exact final branch with an explicit LP64 provider:

```bash
cmake -S . -B build/pr169-cpu-public -G Ninja \
  -DGPUXTB_ENABLE_CUDA=OFF \
  -DGPUXTB_MKL_RT_LIBRARY=/absolute/path/to/libmkl_rt.so \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/pr169-cpu-public --parallel
```

```bash
env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
taskset -c 0 \
python3 benchmarks/natoms_scaling.py \
  --engine gpuxtb \
  --library "$PWD/build/pr169-cpu-public/libgpuxtb.so.0.1.0" \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 --start-mode fresh \
  --output-json build/benchmarks/pr169-fresh.json \
  --output-csv build/benchmarks/pr169-fresh.csv

env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
taskset -c 0 \
python3 benchmarks/natoms_scaling.py \
  --engine gpuxtb \
  --library "$PWD/build/pr169-cpu-public/libgpuxtb.so.0.1.0" \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 --start-mode warm \
  --energy-reference-json build/benchmarks/pr169-fresh.json \
  --cross-engine-energy-atol 5e-7 --cross-engine-force-atol 5e-6 \
  --output-json build/benchmarks/pr169-warm.json \
  --output-csv build/benchmarks/pr169-warm.csv
```

Every gpuxtb WARM, tblite, or xTB invocation requires an explicitly named
gpuxtb FRESH reference; omitting it is an error rather than an unqualified
successful timing. For gpuxtb WARM, the validated FRESH artifact must also use
the same `libgpuxtb` SHA-256 and the same clean source revision. Options and
complete workload identity must match as before.

Run tblite in its own clean process. The first invocation is retained as
`cold_sample`; warmups and raw persistent steady-state samples follow. The
explicit FRESH JSON supplies the energy and force reference. It is read once;
the parsed bytes' SHA-256, path, strict schema/FRESH/row/option/workload checks,
and unique row keys are retained in memory and embedded in every dependent
artifact. tblite's unrelated atomic-charge getter is disabled in this timing
protocol, so its force run times energy plus gradient publication, matching the
requested gpuxtb observable set.

The default live cross-engine gates are read from
`data/conformance/manifest.json`: 5e-7 Hartree for energy and 5e-6
Hartree/bohr for forces. Their manifest path, field, and SHA-256 are stored in
each comparison. The commands spell them out for reviewability. A long-chain
case outside either gate is a failed evidence run; do not widen these values to
produce a performance table.

```bash
env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
taskset -c 0 \
python3 benchmarks/natoms_scaling.py \
  --engine tblite --library /absolute/path/to/libtblite.so \
  --backend cpu --property force \
  --natoms 32,62,98,122 --batch-sizes 1 \
  --warmups 10 --repetitions 30 \
  --energy-reference-json build/benchmarks/pr169-fresh.json \
  --cross-engine-energy-atol 5e-7 --cross-engine-force-atol 5e-6 \
  --output-json build/benchmarks/pr169-tblite.json \
  --output-csv build/benchmarks/pr169-tblite.csv
```

### Known xTB force diagnostic (expected correctness failure)

xTB 6.7.1 force publication is intentionally excluded from the successful
#168 evidence bundle. On the same 32-atom C10H22 input, its energy delta is
1.79205e-8 Hartree (pass), but its maximum force delta is
7.913916573755739e-4 Hartree/bohr (fail versus 5e-6). The result is unchanged
for xTB accuracy settings from 1e-4 through 1e-8, while a repeated invocation
on the persistent state changes forces by only 3.1286e-11 Hartree/bohr. This
rules out the requested SCC accuracy and persistent restart as explanations;
the analytic gradient also has a `1.668e-3` torque residual on the exactly
planar symmetric geometry. `xtb_setAccuracy` clamps values below `1e-4`, so a
nominally smaller value does not tighten this result. A diagnostic 3D jitter
removes that analytic-force pathology, but xTB energy/model drift then grows to
approximately 1.09e-5, 2.60e-5, and 3.61e-5 Hartree for 62, 98, and 122 atoms,
respectively. The benchmark therefore keeps the deterministic physical
geometry and committed gates; the implementation/parameter difference is
follow-up work in #13.

The command below reproduces only the known negative diagnostic. It exits 2,
writes its failure artifact to a temporary directory for inspection, and must
not be included in the #168 latency or speedup table. A correctness-qualified
xTB energy-only matrix also belongs to #13.

```bash
diagnostic_dir="$(mktemp -d)"

env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
taskset -c 0 \
python3 benchmarks/natoms_scaling.py \
  --engine xtb --library /absolute/path/to/libxtb.so.6.7.1 \
  --backend cpu --property force \
  --natoms 32 --batch-sizes 1 \
  --warmups 0 --repetitions 1 \
  --energy-reference-json build/benchmarks/pr169-fresh.json \
  --cross-engine-energy-atol 5e-7 --cross-engine-force-atol 5e-6 \
  --output-json "$diagnostic_dir/xtb-force-failure.json" \
  --output-csv "$diagnostic_dir/xtb-force-failure.csv"
```

The mandatory explicit reference on the gpuxtb WARM command retains direct
FRESH-versus-WARM energy and force deltas plus exact gpuxtb compute-option,
binary, and source-revision identity in that artifact.

Archive the three successful JSON/CSV pairs (gpuxtb FRESH force, gpuxtb WARM
force, and tblite persistent force) without editing them. The JSON top level
and each row record the canonical argv, repository revision and dirty bit,
absolute library path and SHA-256, adjacent CMake cache/compiler evidence when
present, Python/platform/hostname/CPU information, process affinity, thread
environment, protocol counts, and raw samples. CMake builds additionally record
the source Git state, build type, generator, compiler path/version/hash, Release
and linker flags, CUDA selection, and the LP64 LAPACKE+CBLAS provider path/hash.
Meson builds record hashed introspection files, source Git state, project and
subproject versions, build type/options, and dependency provider link
paths/hashes. Meson's configure-time compiler exelist and version are preserved;
an executable receives a content hash only when Meson recorded an absolute path
that remains available. Bare entries are explicitly unresolved rather than
being rebound through the benchmark process's later `PATH`. A reviewable
evidence bundle should also record hashes of all six artifacts:

```bash
sha256sum \
  build/benchmarks/pr169-fresh.json build/benchmarks/pr169-fresh.csv \
  build/benchmarks/pr169-warm.json build/benchmarks/pr169-warm.csv \
  build/benchmarks/pr169-tblite.json build/benchmarks/pr169-tblite.csv
```

For a forensic reproduction of the original five-sample development table,
use `--warmups 3 --repetitions 5`; use the larger counts above for final
performance evidence. Run the self-check without a native library using:

```bash
python3 -m unittest -v benchmarks.test_natoms_scaling
```

## gpuxtb-owned DLPack device results: allocation-cost protocol

`dlpack_result_memory.py` is the narrow issue #214 allocation-cost protocol: it
compares `result_memory="cuda"` (one packed gpuxtb-owned device arena per call,
returned as DLPackResultBuffer producers) with the caller-owned `out=` steady
state on a real NVIDIA GPU through the public `gpuxtb.ArrayBatch` Python API.
The timed interval is a synchronous `perf_counter_ns` window around each public
`compute()` with `torch.cuda.synchronize()` on both sides; the arena mode
closes every producer (and the result) inside the interval so the native
`cudaFree` is measured, while Python garbage collection is kept outside the
window so interpreter GC overhead cannot masquerade as gpuxtb cost. Correctness
is gated per mode against the host CPU `compute_arrays` reference, and the JSON
records raw per-sample latencies, full environment/library identity, and the
timing boundary. The reproducer and NSight-derived memcpy/sync/api evidence are
archived under `benchmarks/evidence/issue-214/`.

```bash
python benchmarks/dlpack_result_memory.py --warmup 30 --repetitions 300 \
  --output benchmarks/evidence/issue-214/<date>-<machine>/dlpack-result-memory.json
```

Hardware-free protocol tests (no CUDA device or provider import needed):

```bash
python3 -m unittest -v benchmarks.test_dlpack_result_memory
```
