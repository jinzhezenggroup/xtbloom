# Molecular dynamics with ASE

xTBloom provides molecular energies and analytic forces through its ASE
calculator, so standard ASE molecular-dynamics integrators can drive molecular
trajectories without a separate xTBloom MD engine. This is an adapter workflow:
ASE owns time integration, thermostats, trajectory writing, and MD units, while
xTBloom evaluates the GFN1/GFN2-xTB potential at each geometry.

Install the ASE extra:

```console
pip install "xtbloom[ase]"
```

A minimal velocity-Verlet trajectory is:

```python
import numpy as np
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
from xtbloom.ase import XTBloom

atoms = Atoms(
    "OH2",
    positions=[
        [0.0000, 0.0000, 0.0000],
        [0.7586, 0.0000, 0.5043],
        [-0.7586, 0.0000, 0.5043],
    ],
)
atoms.set_velocities(np.zeros((3, 3)))
atoms.calc = XTBloom(method="GFN2-xTB", backend="cpu", warm_start=True)

dynamics = VelocityVerlet(atoms, timestep=0.5 * units.fs)
dynamics.run(100)
atoms.calc.close()
```

Other ASE integrators and thermostats can use the same calculator contract. The
ASE interface uses eV and Angstrom conventions at its boundary and xTBloom
performs the conversion to its native atomic units internally. Reusing one
`XTBloom` calculator also allows compatible electronic warm starts across
successive geometries, which is useful for MD-like trajectories.

This does **not** add a native xTBloom MD API or integrator. xTBloom does not own
an MD timestep, thermostat, barostat, constraints engine, or trajectory format.
Periodic ASE systems are still rejected because native periodic GFN1/GFN2-xTB
execution is not implemented. For periodic MD, use a potential/backend that
supports the required periodic physics rather than disabling that validation.
