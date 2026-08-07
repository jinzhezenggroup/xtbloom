# Issue #214 gpuxtb-owned DLPack device-result allocation evidence

This directory archives issue #214's two remaining acceptance rows:
real CuPy/JAX/PyTorch device-provider import evidence and a durable
allocation/free performance comparison of `result_memory="cuda"` against the
caller-owned `out=` steady-state path. The producer implementation itself was
squash-merged in PR #223 (`44058eb`).

The scientific claim is deliberately narrow and archived with raw samples so it
can be re-derived, not restated from memory:

- On the recorded RTX 5090 / CUDA 12.9 stack, `result_memory="cuda"`
  (gpuxtb-owned device arena, no `out=`) passes the predefined mean-overhead
  gate of no more than 5% above the caller-owned `out=` path. Across 300
  counterbalanced pairs after 30 warmups per mode, arena mean latency is
  `7.521 ms` (`7.495..7.770` min..max) versus `7.820 ms`
  (`7.789..8.224`) for `out=`. The paired arena-minus-out mean is
  `-0.2993 +/- 0.0045 ms` (95% confidence half-width), or `-3.83%`.
- Each arena call adds one packed `cudaMalloc` and releases it with one
  `cudaFree`; the caller-owned mode allocates its reusable output once before
  the calls. Direct trace inspection identifies the 13 regular arena
  allocation/release pairs from 3 warmups + 10 measured calls. Process-wide
  totals also include provider setup and teardown: the profiles record arena
  `cudaMalloc=45`/`cudaFree=51` versus `out=`
  `cudaMalloc=33`/`cudaFree=38`, a total difference of +12 allocations and +13
  frees that is deliberately not substituted for the per-call trace count.
- Neither path performs a device-to-host transfer of result data. The D2H
  memcpy profile is byte-identical between the two modes (49 copies / 1069
  bytes total) and corresponds to the internal numerical-host completion
  report that the synchronous public compute contract already requires;
  result publication is device-to-device in both modes (16 copies /
  8,901,200 bytes).
- Neither path adds an extra device-wide synchronization. `cudaDeviceSynchronize`
  (43), `cudaEventSynchronize` (43), and `cudaStreamSynchronize` (16) call
  counts are identical in the two captures; the per-call completion is the
  existing public completion event.
- Real-provider imports: `cupy.from_dlpack`, `torch.from_dlpack`, and
  `jax.dlpack.from_dlpack` each imported the same arena slice and observed the
  identical device pointer (no host round trip); values match the host CPU
  result to float64 round-off. This is enforced by the committed test
  `python/tests/test_dlpack_producer_cuda.py` (see Evaluation below).

## Environment and build identity

- Host: `node3`, Ubuntu 22.04.5 LTS, Linux 6.8.0-110-generic, x86_64.
- CPU: AMD EPYC 7K62 48-core; `nproc=48`.
- GPU: NVIDIA GeForce RTX 5090 (GB202), `sm_120`, 32 GiB
  (`33668988928` bytes reported by the CUDA runtime).
- Driver: 580.95.05 (reported `cudaDriverGetVersion` = 13000, i.e. CUDA 13.0).
- CUDA runtime: 12.9 (build `cuda_12.9.r12.9/compiler.36037853_0`; reported
  `cudaRuntimeGetVersion` = 12090). CuPy 14.1.1 (cuda13x) and torch 2.13.0
  (+cu130) imported the gpuxtb-produced device bytes; gpuxtb itself is built
  against CUDA 12.9.
- Python: 3.13.9. numpy 2.5.1, pytest 9.1.1, CuPy 14.1.1, JAX 0.11.0 (with
  `jax[cuda12]` compute 12.9 libs), PyTorch 2.13.0 (+cu130).
- Source revision: `8ddd28766db581eafeb6088026e99e45415f29c1` (PR #226 evidence
  branch). The measured native library is the explicit shared CMake build at
  `build/pr226-cuda/libgpuxtb.so.0.1.0`; its SHA-256
  (`53ad262937f1612fceee97f9bc88e0abac2126523e54f226092de5fed12b32ce`)
  is embedded in `dlpack-result-memory.json` together with the exact adjacent
  CMake-cache hash and entries, clean matching source Git identity, configured
  C++/CUDA compiler paths and binary hashes, queried compiler versions, release
  flags, architecture, generator, and CMake version. The committed harness
  rejects missing metadata, dirty or different source trees, and revision
  mismatches before it publishes final evidence.
- Nsight Systems: 2025.1.3.140-251335620677v0.
- The `nsys` captures use the committed runner's `--profile-mode`, driving 3
  warmup + 10 timed calls per mode with explicit producer close per call and
  no Python `gc.collect()` in the timed interval.

## Files

- `dlpack-result-memory.json` — authoritative evidence: 300 raw per-sample
  latencies per mode, summary statistics, per-mode correctness records
  (energy/force/charge parity against the host CPU result), workload and
  timing-boundary descriptions, and full environment/library identity.
- `dlpack-result-memory.csv` — compact raw-sample view of the same
  measurements (`mode, sample, latency_ms`).
- `derived-profiler-reports/` — Nsight System-derived summaries:
  `{arena,out}-kern_sum.csv` (kernel instances/time), `{arena,out}-mem_sum.csv`
  (memcpy/memset time), `{arena,out}-mem_size_sum.csv` (memcpy/memset sizes),
  and `{arena,out}-api_trace.csv` (full CUDA API event trace, raw, including
  every `cudaMalloc`/`cudaFree`/synchronize call).

The `.nsys-rep` captures are not committed; the raw CUDA API event traces in
`*-api_trace.csv` are sufficient to re-derive the allocation and
synchronization counts, per the repository performance-evidence policy.

## Workload and timing boundary

- Molecule: water (O 8, H 1, H 1), 1 system, 3 atoms, charge 0, restricted.
- Inputs: host NumPy flat ragged-batch descriptors on the CUDA backend,
  legacy default stream (`stream=1`).
- Requested properties: energy, forces, atomic charges, SCC iterations,
  SCC convergence, per-system status (all six requested outputs).
- Timing boundary: `perf_counter_ns` around each public
  `ArrayBatch.compute()` call with `torch.cuda.synchronize()` before start and
  after stop. 30 warmup calls per mode, then 300 measured samples. The arena
  mode deterministically closes every returned `DLPackResultBuffer` producer
  (and the result) inside the timed interval so the native `cudaFree` is
  included; Python garbage collection is never inside the timed interval.
- Correctness: before and after timing, each mode is compared against an
  explicit host CPU `compute_arrays(..., backend="cpu")` reference. All four
  gates record finite output, status 0, SCC convergence in 9 iterations, and
  maximum energy/force/charge errors of `3.56e-15`, `4.58e-16`, and
  `1.89e-15`, respectively, within committed tolerances. This is a parity
  gate, not a claim of low error on arbitrary geometries.

## Command lines

```bash
# Allocation/latency benchmark:
srun --gres=gpu:1 --ntasks=1 env PYTHONPATH="$PWD/python" \
  GPUXTB_LIBRARY="$PWD/build/pr226-cuda/libgpuxtb.so.0.1.0" \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64 \
  /tmp/venv-providers/bin/python benchmarks/dlpack_result_memory.py \
  --library build/pr226-cuda/libgpuxtb.so.0.1.0 \
  --warmup 30 --repetitions 300 \
  --output /tmp/pr226-final-8ddd287.YOaNlB/dlpack-result-memory.json

# Profiler capture, executed once with <mode>=arena and once with <mode>=out:
srun --gres=gpu:1 --ntasks=1 env PYTHONPATH="$PWD/python" \
  GPUXTB_LIBRARY="$PWD/build/pr226-cuda/libgpuxtb.so.0.1.0" \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64 \
  /group/software/cuda-12.9.1/bin/nsys profile \
  -o /tmp/pr226-final-8ddd287.YOaNlB/<mode> \
  --force-overwrite=true --cuda-memory-usage=true --trace=cuda,nvtx,osrt \
  /tmp/venv-providers/bin/python benchmarks/dlpack_result_memory.py \
  --library build/pr226-cuda/libgpuxtb.so.0.1.0 \
  --warmup 3 --repetitions 10 --profile-mode <arena|out>

# Derived reports (as committed):
/group/software/cuda-12.9.1/bin/nsys stats \
  --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_trace \
  --format csv --output /tmp/pr226-final-8ddd287.YOaNlB/<mode>-report \
  --force-export=true --force-overwrite=true \
  /tmp/pr226-final-8ddd287.YOaNlB/<mode>.nsys-rep
```

The provider matrix was run against the exact shared CUDA library on a real
GPU under `srun --gres=gpu:1`:

```bash
pytest python/tests/test_dlpack_producer_cuda.py          # 11/11 passed (incl. CuPy/JAX/pointer/stream tests)
pytest python/tests/test_dlpack_producer.py              # 18/18 passed (incl. torch+JAX CPU host producer)
pytest python/tests/test_array_batch_cuda.py              # 14/14 passed (CuPy/JAX/torch device arrays)
pytest python/tests                                      # 211 passed, 0 skipped (ase/dpdata installed)
```

## Evaluation

- `PASS` — allocation/free cost vs `out=`: 300-pair mean 7.521 ms (arena)
  versus 7.820 ms (`out=`), passing the explicit maximum 5% mean-overhead gate
  at -3.83%; one arena alloc/free per call with no added device-wide sync and
  no result D2H (identical synchronization and D2H profiles).
- `PASS` — real CuPy/JAX/torch device-provider import evidence: committed
  CUDA tests import the same arena through `cupy.from_dlpack`,
  `torch.from_dlpack`, and `jax.dlpack.from_dlpack`, assert the identical
  device pointer across providers (zero-copy, no host round trip), and check
  value parity with the host CPU result.
- `PASS` — CPU-provider evidence: torch and JAX additionally import the host
  arena producer zero-copy (pointer and value parity); unsupported
  combinations are reported honestly (CuPy has no CPU-tensor import).
- The benchmark harness unit test `benchmarks/test_dlpack_result_memory.py`
  (11 tests, hardware-free) covers the packed workload, statistics and gate,
  CPU reference status, clean runner and selected-library source/build
  identity, refuse-overwrite guards, required-GPU precondition, and LF-only
  JSON/CSV publication.

## Limitations

- Single molecule (3 atoms), single batch size 1, single GPU: the allocation
  is ~130 bytes of device memory, so the arena cost documents the API overhead
  of the packed-arena path rather than a memory-bandwidth claim. The relative
  allocation/free cost will scale differently with larger outputs and batches.
- Both modes verified on host-numpy inputs; device-resident inputs and larger
  ragged batches were not timed here.
- Profile-mode wall timings include Nsight instrumentation overhead and are
  retained only for allocation, transfer, kernel, and synchronization
  evidence; the 300-pair artifact is authoritative for latency.
