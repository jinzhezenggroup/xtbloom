# Issue #350 CUDA overlap and spin-solve staging evidence

This bundle qualifies the retained implementation at clean candidate commit
`7442018c0f063970f29e689d481769827110851a` against clean current-main
baseline `10ffdcd37d148a81cf983236f31d5a61e7606156` on one RTX 5090.

The implementation parallelizes and fuses overlap validation/staging, fuses
the production spin-Hamiltonian validation/staging pass, and lets healthy TRSM
slots borrow the immutable cached overlap Cholesky factor. Invalid or stale
fixed-capacity peers still receive complete identity A/B scratch, and the
nonfinite-Hamiltonian, nonsymmetric-Hamiltonian, then invalid-factor error
precedence is unchanged. The public ABI and GFN2 equations are unchanged.

## Retention decision

The original issue required at least 5% public end-to-end FRESH improvement.
That criterion is **FAIL**. The owner explicitly accepted integration despite
the missed threshold because the reproducible approximately 1% improvement is
still preferable to dropping the change. Nothing in this bundle represents
the original 5% gate as passing.

The final clean-HEAD public measurements used host descriptors, analytic
forces, FRESH SCC, one persistent context per cell, one warmup, 30 measured
calls, one CPU thread, default strict SCC controls, and `5e-7` energy/force
correctness tolerances. Every sample converged in 17 SCC iterations.

| Coordinate | Baseline median (ms) | Candidate median (ms) | Improvement |
| --- | ---: | ---: | ---: |
| C90H182, 272 atoms, B=1 | 3181.270385 | 3135.595538 | 45.674847 ms / 1.435742% |
| C10H22, 32 atoms, B=1 | 80.346583 | 79.571565 | 0.775018 ms / 0.964594% |
| C10H22, 32 atoms, B=32 | 94.469325 | 93.698174 | 0.771151 ms / 0.816297% |

All controls improve, so none approaches the allowed 2% regression limit.
Across the paired raw samples, maximum energy delta is zero and maximum force
delta is `1.39e-16` Hartree/bohr on the primary, `9.02e-17` at 32/B=1, and
`1.25e-16` at 32/B=32.

The raw harness JSON/CSV files embed the local worktree's absolute path, which
contains a retired project name forbidden by repository policy; the B=32 JSON
files also exceed the 1 MiB per-file cap. They are omitted without reducing
the measured matrix. `performance-samples-compact.json` is a mechanical
projection retaining all 30 latency samples, energies, SCC states, iteration
counts, correctness summaries, source hashes, and cross-revision deltas. Each
complete force vector is represented by SHA-256 over its flattened
little-endian IEEE binary64 bytes. `build-metadata.txt` and the compact file
pin the omitted raw artifact hashes and sizes.

## Exact-final-HEAD profile

Nsight Systems 2025.1.3 profiled one warmup plus five measured public calls.
The kernel summaries include seven setup/FRESH instances on each side:

| Kernel metric | Baseline | Candidate |
| --- | ---: | ---: |
| `prepare_overlap_bucket_kernel` median | 42.426559 ms | 0.854909 ms |
| total over seven instances | 297.634852 ms | 6.139722 ms |
| share of reported GPU kernel time | 29.2% | 0.8% |

The kernel median improves 49.627x. The profiled public median improves from
3181.769609 ms to 3132.783643 ms (1.539582%); the unprofiled 30-sample rows
above are the retention result. Only reviewed derived CSV reports are tracked.
`profile-samples-compact.json` retains the profiled public timing and numerical
qualification with the same binary64 force-digest convention.
Raw `.nsys-rep` and SQLite files were omitted and moved to trash after their
hashes, byte counts, and derived reports were recorded.

The complete-profile CUDA API summaries include setup, publication, and
teardown, so their allocation/synchronization totals are not presented as a
steady-state-only count. Allocation-free and Graph/cache invariants are gated
by the production tests and full CUDA CTest instead.

## Runtime safety and validation

Compute Sanitizer 2025.2.1 used `--error-exitcode=99` on the final test binary.
The changed production unrestricted path reports zero memcheck errors, zero
racecheck hazards/errors/warnings, zero initcheck errors, and zero synccheck
errors. The direct 64-system heterogeneous spin-solve control also reports
zero synccheck errors.

The full eigensolver synccheck run exits 99 with 768 reports, all printed
signatures inside NVIDIA cuSOLVER `lansy_M_stage1+0x19a0` during Graph replay.
The literal exit/count/signature is retained in `sanitizer-summary.txt`; the
112120-byte raw text log is omitted and hash-pinned in `build-metadata.txt`
because its tool-emitted whitespace is not a canonical repository artifact.
The row falls under the owner-approved CUDA 12.9.1 / Compute Sanitizer 2025.2.1
provider and Graph disposition in issue #140. It is not claimed as a clean
synccheck pass.

Associated final-head validation:

- explicit-CUDA real-GPU CTest: 109/109 passed;
- CPU core Release CTest: 40/40 passed;
- shared LP64 CPU public-inference CTest: 53/53 passed;
- public CUDA conformance and invariants: host, device, and mixed modes passed;
- benchmark unit suites: 97 passed, one optional plotting test skipped;
- non-editable-wheel Python suite: 309 passed, 43 CUDA/JAX availability skips;
- focused implementation tests and an independent source review passed.

## Reproduction

Configure each source worktree with its own build directory:

```bash
cmake -S . -B build/issue350-final-cuda -G Ninja \
  -DXTBLOOM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/issue350-final-cuda --parallel
```

The exact primary command was run once in each clean baseline/candidate
worktree, substituting only the source-specific library and output paths. The
revisions, library hashes, CMake cache hashes, and raw artifact hashes are
pinned in the compact evidence and `build-metadata.txt`:

```bash
srun --partition=main --nodelist=node3 --nodes=1 --ntasks=1 \
  --cpus-per-task=48 --gres=gpu:5090:1 --wait=60 \
  --export=ALL,LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,OMP_DYNAMIC=FALSE,MKL_DYNAMIC=FALSE,MKL_INTERFACE_LAYER=LP64,MKL_THREADING_LAYER=SEQUENTIAL \
  python3 benchmarks/natoms_scaling.py --engine xtbloom \
  --library "$PWD/build/issue350-final-cuda/libxtbloom.so" \
  --backend cuda --cpu-threads 1 --device-id 0 --property force \
  --natoms 272 --batch-sizes 1 --warmups 1 --repetitions 30 \
  --start-mode fresh --energy-atol 5e-7 --force-atol 5e-7 \
  --output-json primary.json --output-csv primary.csv
```

The controls change only `--natoms 32 --batch-sizes 1,32` and the output
names. The profile changes repetitions to five and wraps the same command:

```bash
/group/software/cuda-12.9.1/bin/nsys profile \
  --force-overwrite=false --trace=cuda,nvtx,osrt --sample=none \
  --cuda-graph-trace=node --cuda-memory-usage=true \
  --output=profile <benchmark-command>

/group/software/cuda-12.9.1/bin/nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv --force-export=true --force-overwrite=true \
  --output profile-derived profile.nsys-rep
```

Sanitizer commands used the same scheduler/runtime boundary:

```bash
/group/software/cuda-12.9.1/bin/compute-sanitizer \
  --tool <memcheck|racecheck|initcheck|synccheck> --error-exitcode=99 \
  build/issue350-final-cuda/xtbloom_cuda_scc_iteration_production_test \
  --unrestricted-parity

/group/software/cuda-12.9.1/bin/compute-sanitizer \
  --tool synccheck --error-exitcode=99 \
  build/issue350-final-cuda/xtbloom_cuda_eigensolver_test

/group/software/cuda-12.9.1/bin/compute-sanitizer \
  --tool synccheck --error-exitcode=99 \
  build/issue350-final-cuda/xtbloom_cuda_eigensolver_test \
  --direct-solve 64 1 20
```

`SHA256SUMS` covers every retained artifact after final formatting.
