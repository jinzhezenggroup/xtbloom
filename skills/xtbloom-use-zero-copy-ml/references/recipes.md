# Zero-Copy Array and PyTorch Recipes

## Reusable NumPy Batch

```python
import numpy as np
from xtbloom import ArrayBatch

atom_offsets = np.array([0, 2, 5], dtype=np.int64)
atomic_numbers = np.array([1, 1, 8, 1, 1], dtype=np.int32)
positions = np.ascontiguousarray(positions_bohr, dtype=np.float64)
molecular_charges = np.array([0.0, 0.0], dtype=np.float64)
unpaired_electrons = np.array([0, 0], dtype=np.int32)

with ArrayBatch(
    atom_offsets=atom_offsets,
    atomic_numbers=atomic_numbers,
    positions=positions,
    molecular_charges=molecular_charges,
    unpaired_electrons=unpaired_electrons,
    backend="cpu",
    copy=False,
) as batch:
    result = batch.compute()
    failed = result.failed_indices
```

The explicit `ascontiguousarray` may copy while preparing the input. It makes the later `ArrayBatch` call strict and predictable; do not call the entire pipeline zero-copy unless the original array was already suitable.

## Preallocated CUDA Outputs with Host Diagnostics

```python
import cupy as cp
import numpy as np
from xtbloom import ArrayBatch

# Use an explicit non-default stream because ArrayBatch requires a positive
# native CUstream handle; CuPy's legacy default stream has pointer value zero.
stream = cp.cuda.Stream(non_blocking=True)
nsystems = int(molecular_charges.shape[0])
natoms = int(atomic_numbers.shape[0])

out = {
    "energies": cp.empty((nsystems,), dtype=cp.float64),
    "forces": cp.empty((natoms, 3), dtype=cp.float64),
    "charges": cp.empty((natoms,), dtype=cp.float64),
    "scc_iterations": np.empty((nsystems,), dtype=np.int32),
    "scc_converged": np.empty((nsystems,), dtype=np.uint8),
    "per_system_status": np.empty((nsystems,), dtype=np.int32),
}

with ArrayBatch(
    atom_offsets=atom_offsets,
    atomic_numbers=atomic_numbers,
    positions=positions,
    molecular_charges=molecular_charges,
    unpaired_electrons=unpaired_electrons,
    backend="cuda",
    device_id=int(positions.device.id),
    stream=int(stream.ptr),
) as batch:
    result = batch.compute(out=out)
    assert result.energies is out["energies"]
    failed = result.failed_indices  # Diagnostics were kept as host NumPy arrays.
```

All CUDA arrays must be on the selected device. Host metadata and diagnostics may be mixed with CUDA numerical arrays.

## xTBloom-Owned CUDA Results

```python
import numpy as np
import torch
from xtbloom import ArrayBatch

nsystems = int(molecular_charges.shape[0])
with ArrayBatch(
    atom_offsets=atom_offsets,
    atomic_numbers=atomic_numbers,
    positions=positions,
    molecular_charges=molecular_charges,
    unpaired_electrons=unpaired_electrons,
    backend="cuda",
) as batch:
    host_diagnostics = {
        "scc_iterations": np.empty((nsystems,), dtype=np.int32),
        "scc_converged": np.empty((nsystems,), dtype=np.uint8),
        "per_system_status": np.empty((nsystems,), dtype=np.int32),
    }
    result = batch.compute(
        compute_charges=False,
        out=host_diagnostics,
        result_memory="cuda",
    )
    energy_buffer = result.energies
    force_buffer = result.forces
    try:
        energies = torch.from_dlpack(energy_buffer)
        forces = torch.from_dlpack(force_buffer)
    finally:
        energy_buffer.close()
        force_buffer.close()
        result.close()

# The imported tensors retain native arena references after result cleanup.
loss_input = energies
```

Before `energy_buffer.close()`, an additional import may call `torch.from_dlpack(energy_buffer)` again; xTBloom creates a fresh single-use capsule for every export.

## JAX Consumption Without In-Place Output

```python
import jax.dlpack
import numpy as np

nsystems = int(molecular_charges.shape[0])
host_diagnostics = {
    "scc_iterations": np.empty((nsystems,), dtype=np.int32),
    "scc_converged": np.empty((nsystems,), dtype=np.uint8),
    "per_system_status": np.empty((nsystems,), dtype=np.int32),
}
result = batch.compute(
    compute_charges=False,
    out=host_diagnostics,
    result_memory="cuda",
)
energy_buffer = result.energies
force_buffer = result.forces
try:
    energies_jax = jax.dlpack.from_dlpack(energy_buffer)
    forces_jax = jax.dlpack.from_dlpack(force_buffer)
finally:
    energy_buffer.close()
    force_buffer.close()
    result.close()
```

Do not pass a JAX array in `out=`. JAX arrays are immutable; consuming a DLPack producer creates the appropriate JAX-owned view instead.

## PyTorch Positions Gradient

```python
import torch
from xtbloom import xtbloom_torch

positions = (
    positions.detach().to(dtype=torch.float64).contiguous().requires_grad_(True)
)
atomic_numbers = atomic_numbers.to(dtype=torch.int32).contiguous()
atom_offsets = atom_offsets.to(dtype=torch.int64).contiguous()
molecular_charges = molecular_charges.to(dtype=torch.float64).contiguous()
unpaired_electrons = unpaired_electrons.to(dtype=torch.int32).contiguous()

stream = torch.cuda.Stream(device=positions.device)
with torch.cuda.stream(stream):
    energies, forces = xtbloom_torch(
        positions,
        atomic_numbers,
        atom_offsets,
        molecular_charges,
        unpaired_electrons,
        backend="cuda",
    )
    energies.sum().backward()
    expected_gradient = -forces.detach()

stream.synchronize()
torch.testing.assert_close(positions.grad, expected_gradient)
```

The final synchronization is for host-side verification in this example, not a requirement to add device-wide synchronization to a production pipeline.

Unsupported examples include:

```python
# Not supported: gradient through forces requires dF/dR.
forces.square().sum().backward()

# Not supported: higher-order derivatives are unavailable.
torch.autograd.grad(energies.sum(), positions, create_graph=True)
```

`torch.compile` may wrap surrounding work, but the xTBloom call executes eagerly across a graph break.
