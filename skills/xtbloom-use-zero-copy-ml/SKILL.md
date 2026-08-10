---
name: xtbloom-use-zero-copy-ml
description: Connect xTBloom to eager NumPy, CuPy, JAX, or PyTorch arrays through ArrayBatch, DLPack, caller-owned outputs, xTBloom-owned CUDA result memory, or the positions-gradient PyTorch operation. Use when an AI coding agent must preserve exact dtype, compact layout, device, CUDA stream, ownership, and lifetime semantics or choose between out=, result_memory="cuda", and xtbloom_torch.
---

# Use xTBloom Zero-Copy ML Interfaces

Choose the simplest public interface that satisfies the data-placement and differentiation requirements. Zero-copy is a strict aliasing contract, not a promise to coerce arbitrary arrays without allocation.

## Preserve the Selected ML Environment

Do not install NumPy, CuPy, JAX, and PyTorch together. Use the framework the
user selected and preserve an existing project or active environment whenever
it encodes a framework, accelerator, or package-index choice.

For a standalone NumPy program, use PEP 723 metadata and run it with `uv run
--script workflow.py` without requiring prior installation:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["xtbloom>=0.1.1"]
# ///
```

For a new standalone CuPy, JAX, or PyTorch program, add only that chosen
framework and the required xTBloom extras to the inline metadata. Do not add
inline metadata to a script that belongs to an existing ML project: PEP 723
would create an isolated environment and either hide the project's framework or
resolve another copy of it. Instead, add xTBloom as a temporary dependency over
the selected environment:

```bash
uv run --no-sync --with 'xtbloom>=0.1.1' python workflow.py
```

Add `--active` when the selected framework lives in an already prepared active
virtual environment rather than the current uv project. `--no-sync` keeps uv
from changing that base environment; the `--with` package is temporary.

## Load the Relevant References

- Read [array-contract.md](references/array-contract.md) before implementing or reviewing any `ArrayBatch`, `compute_arrays`, DLPack, `out=`, or device-result path.
- Read [recipes.md](references/recipes.md) before writing framework-specific code or a PyTorch autograd call.

## Choose the Entry Point

1. Use the ordinary high-level xTBloom calculators when host NumPy inputs and outputs are acceptable. Do not add DLPack complexity without a placement or allocation requirement.
2. Use `ArrayBatch` for a reusable packed ragged batch whose inputs already implement `__dlpack__` and `__dlpack_device__`.
3. Use `compute_arrays` for the same packed contract in a one-shot call.
4. Use `out=` when the caller owns correctly shaped writable output buffers and wants deterministic reuse, especially in a steady-state loop.
5. Use `result_memory="cuda"` when the caller wants xTBloom to allocate device results and export them as DLPack producers without first managing output buffers.
6. Use `xtbloom_torch` only when PyTorch autograd needs the first derivative of energy with respect to positions.

Do not use `ArrayBatch` as an autograd primitive. Its arrays are borrowed eager buffers; `xtbloom_torch` is the only differentiable Python entry point.

## Inspect Before Binding

For every input and caller-owned output, verify:

- exact dtype and logical shape from the tables in `array-contract.md`;
- compact C-contiguous layout when `copy=False`;
- eager, concrete storage rather than a lazy/tracer value;
- CPU, CUDA-host, or CUDA-device placement only;
- the same CUDA device as the resolved xTBloom context for every device array;
- writability for every `out=` buffer;
- a lifetime that extends through the compute call and any later DLPack import.

Never rely on implicit dtype conversion. `copy=True` may ask the producer to pack layout, but it does not change scalar type. `out=` always binds the supplied storage directly and never follows the input copy policy.

## Configure Devices and Streams

1. Require `backend="cuda"` when CUDA device arrays must remain on device. `backend="auto"` is appropriate only when CPU fallback is acceptable and all supplied arrays are compatible with that outcome.
2. Set `device_id` when the application has a specific CUDA device. Cross-device copies are not performed.
3. For `ArrayBatch`, pass a raw `CUstream` handle through `stream=` when integration with a non-default CUDA stream is required. `None` selects the legacy default stream contract.
4. For `xtbloom_torch`, do not pass a raw stream. Execute inside the desired `torch.cuda.stream(...)` context; the operation follows `torch.cuda.current_stream()`.
5. Keep mixed host/device descriptors intentional. Host diagnostics with CUDA numerical outputs are valid and often useful.

Do not add device-wide synchronization merely to make ownership easier. Respect the producer/consumer stream contract and the synchronous `ArrayBatch.compute()` boundary. PyTorch CUDA results remain ordered on the current PyTorch stream.

## Choose an Output Policy

Use `out=` when:

- the loop already owns reusable NumPy, CuPy, or PyTorch buffers;
- output addresses must remain stable;
- avoiding xTBloom device allocation per call matters;
- selected diagnostics should remain on the host while numerical outputs stay on CUDA.

Use `result_memory="cuda"` when:

- the resolved backend is CUDA;
- the caller prefers xTBloom-owned packed device storage;
- the outputs will immediately be imported with `torch.from_dlpack`, `cupy.from_dlpack`, or `jax.dlpack.from_dlpack`;
- a per-call result-arena allocation is acceptable.

A supplied `out=` entry always takes precedence over `result_memory`. This permits mixed caller-owned and xTBloom-owned outputs. JAX arrays are immutable and must never be supplied through `out=`; import an xTBloom-owned DLPack result or create a new JAX array instead.

Keep `per_system_status` and `scc_converged` on host NumPy storage when code needs `ArrayBatchResult.failed_indices`. Device-resident diagnostics cannot be inspected by that helper without an explicit framework-side operation or transfer.

## Preserve DLPack Lifetimes

1. Keep input producers alive until the synchronous `ArrayBatch.compute()` call returns.
2. Keep caller-owned `out=` arrays alive for as long as their results are used.
3. Treat each exported DLPack capsule as single-use, while recognizing that an xTBloom result producer can create a fresh capsule for each export.
4. Close a `DLPackResultBuffer` only after no further exports are needed. Already imported framework tensors retain their own native arena reference.
5. Close the batch/context and result producers deterministically in long-running services; do not depend solely on garbage collection.

## Apply the PyTorch Gradient Boundary

`xtbloom_torch` returns `(energies, forces)` and supports exactly the analytic positions gradient `dE/dR = -F`:

- only `positions` may set `requires_grad=True`;
- the other packed inputs are nondifferentiable metadata or physical parameters;
- a loss depending on `energies` may call `backward()` once to obtain the positions gradient;
- a gradient flowing through the `forces` output is rejected because `dF/dR` is unavailable;
- higher-order differentiation, Hessians, and `create_graph=True` are rejected;
- wrapping the operation in `torch.compile` creates an eager graph break and does not compile the xTBloom call.

Never replace an unsupported derivative with zeros or detach a requested force gradient silently.

## Validate the Integration

Use a small ragged batch on the actual requested backend and check:

- all input and output dtypes and shapes before the call;
- the resolved backend and CUDA device;
- expected pointer reuse for caller-owned `out=` buffers;
- framework imports of every xTBloom-owned result needed after producer cleanup;
- per-system status and convergence rather than only finite aggregate output;
- `positions.grad == -forces` for an energy sum in the PyTorch path;
- explicit rejection of JAX `out=`, non-position autograd, force gradients, and higher-order gradients where those boundaries are relevant.

Report any copy introduced by `copy=True` or by `xtbloom_torch` packing a non-contiguous tensor. Do not call such a path end-to-end zero-copy.
