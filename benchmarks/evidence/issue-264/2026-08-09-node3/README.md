# Issue #264 large-singleton CUDA eigensolver evidence

This bundle qualifies the issue #264 implementation at clean source commit
`ed0125de9d31d7b6a1f0f231cc17443b603e1579`. The implementation replaces the
CUDA 12.9 large-singleton vector-mode `cusolverDnXsyevBatched` path with a
Graph-capturable device-only sequence for 513 through 1024 orbitals:

1. `cusolverDnDsytrd` symmetric-to-tridiagonal reduction;
2. device Gershgorin bounds and parallel Sturm bisection;
3. shifted inverse iteration with clustered-vector reorthogonalization;
4. `cusolverDnDormtr` backtransform; and
5. the existing generalized backtransform and transactional publication.

The setup-owned provider workspace is reused sequentially for restricted and
spin-expanded singleton solves. No public ABI or result semantics changed.

## Environment and protocol

- Host: `node3`, Ubuntu 22.04, Linux 6.8.0-110-generic.
- CPU: AMD EPYC 7K62, 48 physical/logical cores.
- GPU: NVIDIA GeForce RTX 5090, 32 GiB, `sm_120`.
- Driver/toolkit: driver 580.95.05; CUDA 12.9.86.
- Build: Release, shared, CUDA enabled, GCC 11.4.0, CMake 4.2.1.
- CPU provider: SciPy OpenBLAS32 LP64 shared library recorded in
  `build-metadata.txt`.
- Selected library SHA-256:
  `497ddc7269835bda86ebe997101810b504d442a0bb2721a6e313536777f0d9aa`.

The public-C-ABI timing sweep uses one persistent context, descriptor, options
image, and caller-owned result set per cell. Every row requests analytic
forces for one neutral alkane, uses FRESH SCC, performs one warmup and three
measured calls, and keeps result inspection outside the timed interval. CPU
and CUDA use the same CUDA-enabled library. The committed JSON files retain
all raw timings, energies, forces, SCC states, command arguments, build/cache
identity, affinity, and clean-source checks.

CPU is the same-build performance trend; it is not passed to the CUDA runner
as a reference because the committed `natoms_scaling.py` intentionally keys
reference artifacts by backend. Each timing row independently passes the
harness's finite/converged/repeatability checks. Focused provider tests and
the full public CPU/CUDA conformance matrix are the numerical acceptance gate.
`cpu-cuda-parity-summary.csv` is an additional mechanical comparison of the
complete retained vectors, not a claim that the benchmark harness performed a
cross-backend reference check.

## Results

| Atoms | Orbitals | CPU median (ms) | CUDA median (ms) | CUDA/CPU | CUDA/previous size |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 242 | 482 | 1114.367 | 2153.047 | 1.932 | — |
| 257 | 512 | 1826.171 | 2466.442 | 1.351 | 1.146 |
| 272 | 542 | 1493.312 | 3182.512 | 2.131 | 1.290 |
| 287 | 572 | 1682.525 | 3449.623 | 2.050 | 1.084 |
| 302 | 602 | 1920.931 | 4144.763 | 2.158 | 1.202 |
| 362 | 722 | 3641.140 | 6745.268 | 1.853 | 1.627 |

The historical 257-to-272 step was about 6.6x. On this clean head the same
boundary is 1.290x, so the discrete provider cliff is removed. At 362 atoms,
CUDA is 1.853x the same-build CPU median, satisfying the issue's approximate
2x target.

All twelve CPU/CUDA timing rows are available and report correctness `pass`.
The direct retained-vector diagnostic finds maximum CPU/CUDA differences of
`6.14e-12` Hartree in energy and `1.24e-10` Hartree/bohr in force across the
six coordinates. The independent public conformance and invariant suites also
passed for CPU and CUDA host/device/mixed memory modes.

## Dispatch policy

`dispatch-policy.jsonl` records B=1/8/32/128 decisions at 8, 12, 16, 17, 512,
513, 542, 722, 1024, and 1025 orbitals. Production `kAuto` selects the new
provider only for B=1 and 513-1024 orbitals. Small Jacobi and multi-system
batched divide-and-conquer policy is unchanged; 1025 returns to the explicit
existing provider policy.

## Steady-state profiler evidence

`profile-272.json` is one warmup plus three measured 272-atom public CUDA
calls. Nsight Systems captured setup, four invocations, result publication,
and teardown. `derived-profiler-reports/steady-state-contract-summary.csv`
defines the steady-state window from the first `cudaGraphLaunch` through the
last event before the first teardown free. In that window:

- four public invocations issue exactly four host `cudaGraphLaunch` calls;
- `cudaMalloc`, `cudaMallocAsync`, `cudaMallocHost`, and all free variants are
  zero;
- `cudaDeviceSynchronize`, `cudaStreamQuery`, and `cudaEventQuery` are zero;
- event/stream synchronization and transfers occur at fixed public-call and
  caller-result boundaries, not in proportion to the 17 SCC iterations.

The full derived CUDA API trace, API summary, kernel summary, and GPU memory
time/size summaries are retained. Raw `.nsys-rep`, `.sqlite`, and `.qdstrm`
captures are intentionally excluded because native profiler files can embed
the target environment.

## Compute Sanitizer

Compute Sanitizer 2025.2.1.0 used `--error-exitcode=99`.

| Target | memcheck | racecheck | initcheck | synccheck |
| --- | --- | --- | --- | --- |
| Forced 64-orbital direct provider control | 0 errors | 0 hazards | 0 errors | 0 errors |
| Automatic 542-orbital production Graph smoke | 0 errors | 0 hazards | 0 errors | negative diagnostic |

The direct control exercises the new reduction, device tridiagonal solve,
backtransform, info handling, and eigenpair publication without the existing
Graph-child synchronization instrumentation issue. The production smoke runs
one physical restricted SCC body with the automatic 542-orbital provider.

`production-synccheck-negative.log` is retained rather than hidden. It exits
99 with 256 diagnostics in the pre-existing
`reduce_spin_atomic_charges_kernel` barrier under the full SCC Graph. The
reported kernel is in `src/backends/cuda/gfn2_scc_potential.cu`, which is
byte-identical to `origin/main` for this branch. This bundle does not claim
that full-SCC synccheck passed; the repository-wide CUDA sanitizer release
gate remains tracked by issue #129.

## Validation associated with this head

- Post-review focused CUDA CTest: 4/4 passed.
- Full real-GPU CUDA CTest: 113/113 passed, including public CPU/CUDA
  conformance, CUDA host/device/mixed descriptors, invariants, Graphs, loader,
  and ABI checks.
- Shared CPU Release CTest: 41/41 passed.
- Installed CUDA consumer: `smoke`, `cpu`, and real-GPU `cuda` modes passed;
  install-prefix licensing passed.
- Benchmark unit suites: 36/36 and 7/7 passed.
- Parameter, conformance, oracle, licensing, lockfile, pre-commit, and
  `git diff --check` gates passed.
- Independent CUDA diff review found no source-level correctness blocker. Its
  sanitizer naming/coverage finding was fixed before commit and the affected
  tests were rerun.

## Reproduction

Configure and build:

```bash
cmake -S . -B build/issue264-cuda-dev -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DXTBLOOM_CPU_LINALG_LIBRARY=/home/jzzeng/miniconda3/lib/python3.13/site-packages/scipy_openblas32/lib/libscipy_openblas.so \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/issue264-cuda-dev --parallel
```

Run the CPU sweep; change only `--backend cpu` to `--backend cuda` and the
output names for the CUDA sweep:

```bash
srun --partition=main --nodelist=node3 --gres=gpu:5090:1 \
  --ntasks=1 --cpus-per-task=48 --wait=60 \
  env PYTHONPATH="$PWD/python" \
  LD_LIBRARY_PATH="$PWD/build/issue264-cuda-dev:/home/jzzeng/miniconda3/lib/python3.13/site-packages/scipy_openblas32/lib:/group/software/cuda-12.9.1/targets/x86_64-linux/lib" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
  python3 benchmarks/natoms_scaling.py --engine xtbloom \
  --library "$PWD/build/issue264-cuda-dev/libxtbloom.so.0.1.0" \
  --backend cpu --cpu-threads 16 --property force \
  --natoms 242,257,272,287,302,362 --batch-sizes 1 \
  --warmups 1 --repetitions 3 --start-mode fresh \
  --energy-atol 5e-7 --force-atol 5e-7 \
  --output-json cpu-fresh.json --output-csv cpu-fresh.csv
```

Dispatch report:

```bash
srun --partition=main --nodelist=node3 --gres=gpu:5090:1 --ntasks=1 \
  env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib \
  build/issue264-cuda-dev/xtbloom_cuda_eigensolver_test --dispatch-policy
```

Sanitizer controls use the same `srun`/`LD_LIBRARY_PATH` boundary:

```bash
compute-sanitizer --tool <memcheck|racecheck|initcheck|synccheck> \
  --error-exitcode=99 \
  build/issue264-cuda-dev/xtbloom_cuda_eigensolver_test \
  --tridiagonal-provider-sanitizer

compute-sanitizer --tool <memcheck|racecheck|initcheck|synccheck> \
  --error-exitcode=99 \
  build/issue264-cuda-dev/xtbloom_cuda_scc_iteration_production_test \
  --large-singleton-sanitizer
```

The Nsight target is the CUDA sweep above restricted to `--natoms 272`, with:

```bash
/group/software/cuda-12.9.1/bin/nsys profile \
  --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  --cuda-graph-trace=node --cuda-memory-usage=true \
  --output=nsys-public-272 <benchmark-command>

/group/software/cuda-12.9.1/bin/nsys stats \
  --report cuda_api_trace,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv --force-export=true --force-overwrite=true \
  --output nsys-public-272-derived nsys-public-272.nsys-rep
```

`SHA256SUMS` pins every retained artifact after final formatting.
