# Direct Python geometry optimization

The Python package exposes a small L-BFGS geometry optimizer on top of the same
analytic forces used by single-point inference. It reuses xTBloom's existing
Array-API L-BFGS stepping primitives; it does not add a second native optimizer
or change the C ABI.

For one existing calculator:

```python
import numpy as np
from xtbloom import Calculator, optimize

numbers = np.array([1, 1])
positions = np.array([[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]])

with Calculator(
    "GFN2-xTB",
    numbers,
    positions,
    backend="cpu",
    warm_start=True,
) as calc:
    result = optimize(calc, fmax=5e-4, max_steps=200)

print(result.all_converged)
print(result.positions[0])
print(result.energies[0])
```

`fmax` is the maximum per-atom force norm in Hartree/bohr. The returned
`OptimizationResult` contains one accepted final coordinate array, energy,
force array, convergence flag, and accepted-step count per system. `evaluations`
counts complete energy/force evaluations. The input calculator or structures are
left at the last energy-accepted geometry, never at an unevaluated or rejected
line-search trial.

For a ragged molecular batch, `optimize_batch()` creates one reusable
`BatchCalculator` and keeps every system in one stable ragged topology while
systems converge independently:

```python
from xtbloom import Structure, optimize_batch

systems = [
    Structure([1, 1], np.array([[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]])),
    Structure([8, 1, 1], water_positions),
]
result = optimize_batch(
    systems,
    method="GFN2-xTB",
    backend="cuda",
    warm_start=True,
)
```

A trial is accepted only when its energy does not increase beyond a small
numerical tolerance. Rejected trials are retried from the last accepted
geometry with a shorter step; positive-curvature accepted steps enter the
limited-memory history. Reaching `max_steps` returns the last accepted states
with `converged=False` for unfinished systems. SCC/eigensolver failures and a
line search that stalls at the minimum step raise instead of publishing a bad
geometry.

This is a Python molecular minimizer. Native C-ABI optimization, constraints,
periodic optimization, transition-state search, and cell optimization remain
outside this API.
