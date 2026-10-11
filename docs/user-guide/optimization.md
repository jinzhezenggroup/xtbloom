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
force array, convergence flag, failure flag and message, and accepted-step
count per system. `evaluations` counts complete energy/force evaluations. The
input calculator or structures are left at the last energy-accepted geometry,
never at an unevaluated or rejected line-search trial. That restoration also
occurs before an evaluator or line-search exception escapes.

For a ragged molecular batch, `optimize_batch()` creates one reusable
`BatchCalculator` and keeps every system in one stable ragged topology while
systems converge independently. Every entry must be a distinct mutable
`Structure`; reusing one object in multiple batch slots is rejected before
native resources are acquired because each slot has an independent
accepted-state ledger:

```python
import numpy as np
from xtbloom import Structure, optimize_batch

water_positions = np.array(
    [[0.0, 0.0, -0.45], [0.0, 1.40, 0.55], [0.0, -1.40, 0.55]]
)

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

print(result.failed_indices)
result.raise_for_status()
```

A trial is accepted only when its energy does not increase beyond a small
numerical tolerance. Rejected trials are retried from the last accepted
geometry with a shorter step; positive-curvature accepted steps enter the
limited-memory history. Reaching `max_steps` returns the last accepted states
with `converged=False` for unfinished systems; reaching the step limit is not a
numerical failure.

`optimize()` is strict because its one calculator has no successful peer to
preserve: an SCC/eigensolver failure, non-finite result, or line search that
stalls at the minimum step raises after restoring the last accepted geometry.
`optimize_batch()` keeps those failures local to their systems. Its `failed`
array, `failed_indices`, and input-ordered `failure_messages` identify stopped
systems while successful peers continue. A failed system remains at its last
accepted geometry, or at its original geometry with NaN energy and forces if
no valid baseline was established. Call `result.raise_for_status()` when the
caller wants one combined exception after inspecting or retaining successful
peer results. Invalid evaluator data or a call-level native exception still
aborts the whole optimization and restores every accepted geometry.

This is a higher-level Python molecular minimizer. Native C-ABI optimization,
constraints, periodic optimization, transition-state search, and cell
optimization remain outside this API.
