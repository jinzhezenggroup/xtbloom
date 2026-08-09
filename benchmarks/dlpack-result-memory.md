# DLPack result-memory allocation protocol

`dlpack_result_memory.py` compares two public CUDA output policies through
`xtbloom.ArrayBatch`:

- `result_memory="cuda"` allocates one xTBloom-owned packed device arena per
  call and exposes its slices as DLPack producers;
- `out=` reuses caller-owned output arrays in steady state.

## Timing and correctness

The timed interval is a synchronous `perf_counter_ns` window around each
public `compute()`, with `torch.cuda.synchronize()` before and after. Arena
and `out=` calls are counterbalanced in AB/BA pairs. Every arena producer is
closed inside the timed interval so its native `cudaFree` is measured.

Correctness is checked before and after timing against an explicit CPU
`compute_arrays` reference, including finite energy, force, and charge parity,
per-system status, and SCC convergence.

The issue #214 acceptance gate requires the arena mean latency to be no more
than 5% above `out=`. That narrow gate documents one API allocation path; it is
not a general CUDA throughput claim.

## Example

```bash
PYTHONPATH="$PWD/python" \
XTBLOOM_LIBRARY=/absolute/path/to/libxtbloom.so \
python3 benchmarks/dlpack_result_memory.py \
  --library /absolute/path/to/libxtbloom.so \
  --warmup 30 \
  --repetitions 300 \
  --output build/benchmarks/dlpack-result-memory.json
```

Real final evidence requires a CUDA build and NVIDIA GPU. JSON records raw
paired samples, confidence bounds, correctness results, environment identity,
and the timing boundary.

The archived
[issue #214 evidence](evidence/issue-214/2026-08-07-rtx5090/README.md)
includes real CuPy, JAX, and PyTorch device-provider imports plus sanitized
Nsight-derived allocation, transfer, and synchronization summaries.

## Validation

```bash
python3 -m unittest -v benchmarks.test_dlpack_result_memory
```
