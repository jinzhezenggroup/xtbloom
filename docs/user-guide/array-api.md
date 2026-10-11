# Model-aware Array API and DLPack inference

The public top-level `xtbloom.ArrayBatch` and `xtbloom.compute_arrays()` packed
interfaces accept the same GFN1/GFN2 method spellings as the high-level
calculators while preserving the existing zero-copy DLPack descriptor contract.
GFN2-xTB remains the default for backward compatibility.

```python
import numpy as np
from xtbloom import ArrayBatch

with ArrayBatch(
    atom_offsets=np.array([0, 2], dtype=np.int64),
    atomic_numbers=np.array([1, 1], dtype=np.int32),
    positions=np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=np.float64),
    molecular_charges=np.array([0.0], dtype=np.float64),
    unpaired_electrons=np.array([0], dtype=np.int32),
    method="GFN1-xTB",
    backend="cpu",
) as batch:
    result = batch.compute()
```

`method` accepts `"GFN1-xTB"`/`"GFN1"` and
`"GFN2-xTB"`/`"GFN2"`. The selected stable public C-ABI model tag is written
into the same compute-options structure used by the native implementation; no
GFN2 substitution occurs for a GFN1 request. CPU and CUDA follow their normal
backend capability and error contracts.

All existing packed-array semantics are unchanged: exact dtype/shape checks,
`copy=`, caller-owned `out=` buffers, host or xTBloom-owned CUDA result memory,
DLPack stream negotiation, host/device/mixed descriptors, point charges,
caller-supplied charge response, and peer-local numerical failures.

`xtbloom_torch` is a separately implemented compiled autograd operator, but it
accepts the same GFN1/GFN2 `method` spellings and keeps GFN2-xTB as the default.
Its derivative contract is intentionally narrower than `ArrayBatch`: only the
positions gradient `dE/dR = -F` is supported, while force gradients and
higher-order differentiation remain unsupported.
