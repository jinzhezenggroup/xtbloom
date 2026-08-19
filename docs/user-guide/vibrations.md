# Vibrational analysis

xTBloom can turn its numerical Cartesian Hessian into mass-weighted normal modes
without adding a native response implementation. The Python helper keeps the
finite-difference Hessian and the vibrational post-processing as separate
steps, so the raw Hessian remains available for numerical diagnostics.

```python
import numpy as np
from xtbloom import Calculator, vibrations

numbers = np.array([8, 1, 1])
positions = np.array(
    [
        [0.0000000000, 0.0000000000, -0.7357858611],
        [1.4418315287, 0.0000000000, 0.3678929305],
        [-1.4418315287, 0.0000000000, 0.3678929305],
    ]
)
# Atomic masses are explicit inputs in unified atomic mass units (u).
masses = np.array([15.999, 1.008, 1.008])

with Calculator("GFN2-xTB", numbers, positions, backend="cpu") as calc:
    vib = vibrations(calc, masses)

print(vib.frequencies_cm1)
print(vib.modes.shape)
```

`frequencies_cm1` uses signed wavenumbers in cm^-1: a negative value denotes an
imaginary mode. `modes` contains unit-norm Cartesian displacement vectors, and
`mass_weighted_modes` contains the corresponding orthonormal mass-weighted
eigenvectors. The eigenvalues of the projected mass-weighted Hessian are also
available as `eigenvalues`.

By default, `analyze_vibrations()` projects the numerically detected rigid-body
subspace before diagonalization. The rank is three for a single atom, five for
a linear molecule, and six for an ordinary non-linear molecule; the detected
value is returned as `rigid_rank`. Pass `project_rigid=False` when the complete
mass-weighted Cartesian eigenproblem is useful for diagnostics.

The helper accepts atomic masses explicitly rather than choosing isotopes or
maintaining a second periodic-table mass database. The Hessian is expected in
Hartree/bohr^2, positions in bohr, and masses in unified atomic mass units.
`analyze_vibrations()` symmetrizes the Hessian by default because
`Calculator.hessian()` is a finite difference of analytic forces and keeps its
raw antisymmetric residual as a convergence diagnostic.

This API performs normal-mode analysis only. It does not provide thermochemical
partition functions, standard-state corrections, rotational symmetry numbers,
or an analytic/native C-ABI Hessian.
