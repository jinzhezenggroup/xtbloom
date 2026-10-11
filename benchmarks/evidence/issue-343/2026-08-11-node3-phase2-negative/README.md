# Issue #343 Phase 2: guarded eigensolver recycling is not profitable

## Decision

The guarded same-geometry restricted SCC prototype is numerically safe, but it
does not meet issue #343's performance objective. It is retained on branch
`perf/343-guarded-recycled-eigensolver` for inspection and must not be merged
into production.

- Measured prototype revision: `e855f56c67d7c7a0a9928eb2a324265f3c807167`
- Core prototype commit: `1928f0ed3b8df1f802fcb68fd933cf91125114f1`
- Baseline `main`: `d6d2ae8b62c7431338ef610f934e868dc4423d1c`
- No public ABI or conformance golden was changed.
- No CPU or CUDA speedup is claimed.

The prototype forms the complete projected Hamiltonian `B = C^T H C`, solves
the omitted complement, absorbs up to eight low complement Ritz vectors into
the occupied-plus-buffer space, solves that augmented block, and accepts only
after occupation-tail, frontier-gap, orthogonality, backward-residual, and
residual/gap gates. Any miss performs the existing dense solve in the same SCC
iteration. Accepted iterations are followed by dense correction and dense
confirmation; unrestricted systems and systems below 96 orbitals remain dense.

## Environment and identities

CPU measurements used one affinity-pinned core (`taskset -c 0`) of an AMD EPYC
7K62, GCC 11.4, Release `-O3 -DNDEBUG`, `cpu_threads=1`, and
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`.

| Artifact | SHA-256 |
| --- | --- |
| Prototype `libxtbloom.so.0.1.1` | `689f3c710340a972a540d24f00b6a66e4f1f8d85f710ec0bc943a0388c021a21` |
| Main `libxtbloom.so.0.1.1` | `ddd9de4cc241d3dcb0222ca672750b38a109653daa0a41cfe0c3227a20c140b9` |
| Prototype CPU `CMakeCache.txt` | `f02d408a36946ae5cd3d7db8226e24fdceae53763395db2fe38274b6e4e0ed3f` |
| Main CPU `CMakeCache.txt` | `fa72de3150374c73748506e077ace4f84d1f30419331a721c0b3e5c9efc66d3b` |
| SciPy LP64 OpenBLAS provider | `b2dfe24b9aa11cf1d1cec8edbca9423b50cfd186b486d59dd4efe45826261a98` |
| Pinned tblite 0.7.0 library | `1c2fb4308b398851580af11ccad5eec22314ff45a34a553380277e776a44c3b5` |

The tblite build is the already reviewed 0.7.0 artifact from issue #168. The
JSON files record the full argv, library/build/source identities, workload
hashes, environment, raw samples, observables, and correctness decisions.

GPU measurements used an NVIDIA GeForce RTX 5090 (`sm_120`), driver
580.95.05, CUDA compiler 12.9.86, and the CUDA 12.9.1 runtime cohort from
`/group/software/cuda-12.9.1/targets/x86_64-linux/lib`.

| Artifact | SHA-256 |
| --- | --- |
| `xtbloom_cuda_eigensolver_test` | `7e5d262323d6a06cf839fd3c30edd34cce52b0758e48916c6970439c2a60c284` |
| CUDA `CMakeCache.txt` | `daf47a73e019c1f241d586ba0fc01e6f20336e759d8354918bc28ac8d1dadecc` |

## Public CPU result

The public C ABI FRESH/force protocol used C20H42 (`natoms=62`, 122 orbitals),
batch size one, the default 300 K electronic temperature, 500 maximum SCC
iterations, charge tolerance `1e-10`, energy tolerance `1e-12`, ten warmups,
and 30 measured calls. Every row passed repeatability and convergence checks.

| Build | Median | Min | p95 | SCC iterations |
| --- | ---: | ---: | ---: | ---: |
| Prototype `e855f56` | 78.332982 ms | 78.146482 ms | 78.561673 ms | 19 |
| Main `d6d2ae8` | 69.3676705 ms | 69.287949 ms | 69.480396 ms | 18 |

The prototype is **12.9243% slower** and adds one SCC iteration. The measured
distributions do not overlap at their reported min/p95 bounds.

Correctness remains strong:

- prototype versus main maximum energy difference: `0.0 Eh`;
- prototype versus main maximum force difference: `1.1641158e-11 Eh/bohr`;
- main versus pinned tblite maximum energy difference: `3.6948222e-13 Eh`;
- main versus pinned tblite maximum force difference: `1.2916151e-11 Eh/bohr`;
- tblite comparison status: pass against `5e-7 Eh` energy and
  `5e-6 Eh/bohr` force gates.

The focused strict internal C20H42+D4 control also converged in 34 iterations
for both adaptive and forced-dense schedules. The adaptive run used 29 full
solves and five accepted recycled solves. The two recycle fallbacks are
included in the 29 full solves because a rejected recycle attempt immediately
uses the dense solver in the same SCC iteration, preserving the 34-iteration
total. Final adaptive-versus-dense differences were at most `3.58e-10` in
charge-like state, `1.09e-10` in energy-weighted density, and `5.68e-14 Eh` in
SCC free energy. Reduced full-solve count therefore did not imply reduced
latency.

## GPU provider lower bound

The opt-in CUDA dispatch benchmark measures the production automatic provider
with CUDA-event timing. It used 50 warmups and 100 samples per dimension.

| Dimension | Provider | Mean | Median | Min | Max |
| ---: | --- | ---: | ---: | ---: | ---: |
| 122 | `xsyev_batched` | 8228.081627 us | 8227.583885 us | 8225.312233 us | 8230.367661 us |
| 95 | `xsyev_batched` | 5642.079935 us | 5641.823769 us | 5637.631893 us | 5652.383804 us |
| 35 | `xsyev_batched` | 1256.907849 us | 1256.960034 us | 1255.807996 us | 1258.272052 us |

The `n=95` and `n=35` p50 medians sum to `6898.783803 us`, or **83.8494%** of
the full `n=122` p50 median. Only 16.1506% remains before accounting for the
prototype's three full-size GEMMs, residual analysis, state handling, and any
failed attempt followed by a full solve. The observed two fallback attempts
already make that cost model negative.

An `n=722` random-fixture automatic-provider diagnostic did not finish within
a five-minute scheduler allocation, so it is explicitly unavailable and is
not used for a performance claim. No production CUDA recycle path was built:
the measured lower bound rejected that implementation before CUDA Graph,
workspace, cache, WARM, and publication state were modified.

## Validation

Measured prototype revision `e855f56`:

```text
ctest --test-dir build/issue343-phase2-cpu --output-on-failure
54/54 passed

ctest --test-dir build/issue343-phase2-cpu \
  -R '^xtbloom\.gfn2\.(eigensolver|scc_driver|scc_recycle)$' \
  --output-on-failure
3/3 passed

srun ... env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib \
  ./build/issue343-cuda/xtbloom_cuda_eigensolver_test
passed on RTX 5090

UV_DEFAULT_INDEX=https://pypi.org/simple \
  uv tool run --from prek==0.3.1 prek run \
  --show-diff-on-failure --color=always --all-files
passed

UV_DEFAULT_INDEX=https://pypi.org/simple uv lock --check
passed

git diff --check
passed
```

The full CTest run initially exposed that historical SCC trace replay did not
reconstruct the newly added internal eigensolver provenance. Commit `e855f56`
fixed the harness by representing every replayed prefix as committed dense
iterations; the focused replay tests and subsequent 54-test run passed.

## Reproduction

The final runs used detached clean checkouts under `/tmp/xtbloom-*`, with the
complete SciPy OpenBLAS runtime closure copied to
`/tmp/xtbloom-provider-clean`. Prototype public CPU run:

```bash
env LD_LIBRARY_PATH=/tmp/xtbloom-provider-clean \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0 /home/jzzeng/miniconda3/bin/python3.13 \
  /tmp/xtbloom-issue343-phase2/benchmarks/natoms_scaling.py \
  --engine xtbloom \
  --library /tmp/xtbloom-issue343-phase2/build/phase2-cpu/libxtbloom.so \
  --backend cpu --cpu-threads 1 --property force \
  --natoms 62 --batch-sizes 1 --warmups 10 --repetitions 30 \
  --start-mode fresh \
  --output-json /tmp/xtbloom-issue343-results/prototype-cpu.json \
  --output-csv /tmp/xtbloom-issue343-results/prototype-cpu.csv
```

Main uses the identical command from `/tmp/xtbloom-issue343-main`, with its
runner and `--library /tmp/xtbloom-issue343-main/build/main-cpu/libxtbloom.so`.
The tblite row uses that clean main runner and the same workload and sample
counts with:

```text
--engine tblite
--library /tmp/tblite-pr169-6f7f1e2/libtblite.so.0.7.0
--energy-reference-json /tmp/xtbloom-issue343-results/main-cpu.json
--cross-engine-energy-atol 5e-7
--cross-engine-force-atol 5e-6
```

GPU provider timings:

```bash
for issue343_dimension in 122 95 35; do
  srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --gres=gpu:5090:1 --time=00:10:00 \
    env LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib \
    ./build/issue343-cuda/xtbloom_cuda_eigensolver_test \
    --dispatch-benchmark "$issue343_dimension" 1 50 100
done
```

## Files

- `prototype-cpu.json` / `.csv`: clean final prototype public result.
- `main-cpu.json` / `.csv`: clean main public baseline.
- `tblite-cpu.json` / `.csv`: pinned independent reference comparison.
- `gpu-provider.jsonl`: raw CUDA-event samples for all three dimensions.
- `SHA256SUMS`: hashes for every retained file except the manifest itself.
