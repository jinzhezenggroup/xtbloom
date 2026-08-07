# Issue #214 gpuxtb-owned DLPack device-result allocation evidence

This directory archives issue #214's two remaining acceptance rows:
real CuPy/JAX/PyTorch device-provider import evidence and a durable
allocation/free performance comparison of `result_memory="cuda"` against the
caller-owned `out=` steady-state path. The producer implementation itself was
squash-merged in PR #223 (`44058eb`).

The scientific claim is deliberately narrow and archived with raw samples so it
can be re-derived, not restated from memory:

- On the recorded RTX 5090 / CUDA 12.9 stack, `result_memory="cuda"`
  (gpuxtb-owned device arena, no `out=`) has the same per-call wall latency as
  the caller-owned `out=` path at 300 warm samples: `7.507 ms` mean
  (`7.493..7.536` min..max) for arena vs `7.830 ms` mean for `out=`. The arena
  path is effectively at parity and not measurably slower in this workload.
- The arena path adds exactly one packed `cudaMalloc` per call and one
  `cudaFree` per call (arena `cudaMalloc=45`/`cudaFree=50` vs `out=`
  `cudaMalloc=32`/`cudaFree=37` over a 10-call timed window plus warmup); the
  difference `+13`/`+13` equals the 13 arena-allocating calls (3 warmup + 10
  timed).
- Neither path performs a device-to-host transfer of result data. The D2H
  memcpy profile is byte-identical between the two modes (49 copies / 1069
  bytes total) and corresponds to the internal numerical-host completion
  report that the synchronous public compute contract already requires;
  result publication is device-to-device in both modes (16 copies / 8.9 MiB).
- Neither path adds an extra device-wide synchronization. `cudaDeviceSynchronize`
  (34) and `cudaEventSynchronize` (43) call counts are identical in the two
  captures; the per-call completion is the existing public completion event.
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
- Source revision: `44058ebbe4161780aea2df1db7ea7c3938684f68` (PR #223 squash
  merge). The measured native library is the CUDA wheel built from that
  revision; its SHA-256 is embedded in `dlpack-result-memory.json`
  (`library_sha256`).
- Nsight Systems: 2025.1.3.140-251335620677v0.
- The `nsys` captures were made with a separate target driving 3 warmup + 10
  timed calls per mode and exporting the same arena release semantics as the
  benchmark (explicit producer close per call, no Python `gc.collect()` in the
  timed interval).

## Files

- `dlpack-result-memory.json` — authoritative evidence: 300 raw per-sample
  latencies per mode, summary statistics, per-mode correctness records
  (energy/force/charge parity against the host CPU result), workload and
  timing-boundary descriptions, and full environment/library identity.
- `dlpack-result-memory.csv` — compact raw-sample view of the same
  measurements (`mode, sample, latency_ms`).
- `derived-profiler-reports/` — Nsight System-derived summaries:
  `{arena,out}-kern_sum.csv` (kernel occupancy/time), `{arena,out}-mem_sum.csv`
  (memcpy/memset totals), and `{arena,out}-api_trace.csv` (full CUDA API event
  trace, raw, including every `cudaMalloc`/`cudaFree`/synchronize call).

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
- Correctness: the first measured call of each mode is compared against a
  host CPU `compute_arrays` reference; recorded max energy/force/charge
  absolute errors are 0.0 (bit-identical CPU/CUDA on this workload). This is
  a parity gate, not a claim of low error on arbitrary geometries.

## Command lines

```bash
# Allocation/latency benchmark (this bundle):
python benchmarks/dlpack_result_memory.py --warmup 30 --repetitions 300 \
  --output benchmarks/evidence/issue-214/2026-08-07-rtx5090/dlpack-result-memory.json

# Profiler captures:
nsys profile -o <mode> --force-overwrite true --cuda-memory-usage=true \
  --trace=cuda,nvtx,osrt python /tmp/nsys_target.py <arena|out> 10

# Derived reports (as committed):
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true <mode>.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum --format csv --force-export=true <mode>.nsys-rep
nsys stats --report cuda_api_trace --format csv --force-export=true <mode>.nsys-rep
```

The provider matrix was run against the freshly built CUDA wheel on a real
GPU under `srun --gres=gpu:1`:

```bash
pytest python/tests/test_dlpack_producer_cuda.py          # 10/10 passed (incl. new CuPy/JAX/pointer tests)
pytest python/tests/test_dlpack_producer.py              # 17/17 passed (incl. torch+JAX CPU host producer)
pytest python/tests/test_array_batch_cuda.py              # 14/14 passed (CuPy/JAX/torch device arrays)
pytest python/tests                                      # 214 passed, 0 skipped (ase/dpdata installed)
```

## Evaluation

- `PASS` — allocation/free cost vs `out=`: 300-sample mean 7.507 ms (arena)
  vs 7.830 ms (`out=`); parity plus one arena alloc/free per call with no
  added device-wide sync, no result D2H (identical sync and D2H profiles).
- `PASS` — real CuPy/JAX/torch device-provider import evidence: committed
  CUDA tests import the same arena through `cupy.from_dlpack`,
  `torch.from_dlpack`, and `jax.dlpack.from_dlpack`, assert the identical
  device pointer across providers (zero-copy, no host round trip), and check
  value parity with the host CPU result.
- `PASS` — CPU-provider evidence: torch and JAX additionally import the host
  arena producer zero-copy (pointer and value parity); unsupported
  combinations are reported honestly (CuPy has no CPU-tensor import).
- The benchmark harness unit test `benchmarks/test_dlpack_result_memory.py`
  (5 tests, hardware-free) covers the packed workload, summary statistics,
  refuse-overwrite guard, required-GPU precondition, and CSV schema.

## Limitations

- Single molecule (3 atoms), single batch size 1, single GPU: the allocation
  is ~130 bytes of device memory, so the arena cost documents the API overhead
  of the packed-arena path rather than a memory-bandwidth claim. The relative
  allocation/free cost will scale differently with larger outputs and batches.
- Both modes verified on host-numpy inputs; device-resident inputs and larger
  ragged batches were not timed here.
- The 58 ms first-sample outlier observed in one exploratory 40-sample run is
  attributed to a cold first-touch allocation and was not reproducible with
  30 warmup calls (none of the 300 committed samples exceeds 7.568 ms).
