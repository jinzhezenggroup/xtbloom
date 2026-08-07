# QM/MM usage

gpuxtb supports two external electrostatic inputs that can be used together:

1. explicit screened point charges, and
2. a caller-supplied atom-level periodic response operator $b + Aq$.

Both enter the SCC problem. They are not post-processing corrections.

## Explicit point charges

For each point charge provide a position, charge value, and positive screening
parameter `gamma`. gpuxtb computes its interaction with the GFN2 shell
monopoles, includes that potential in every SCC iteration, and returns analytic
forces on both the QM atoms and point charges when requested.

The caller remains responsible for:

- point-charge/point-charge energy and forces;
- topology, exclusions, and classical force-field terms; and
- redistribution of virtual-site forces.

Coincident QM and point-charge positions are valid because the screened
interaction remains finite; their pair force is zero at exact coincidence.

## Periodic response

The optional fields define

```math
\begin{aligned}
\phi &= b + Aq, \\
E &= q^{\mathsf T}b + \frac{1}{2}q^{\mathsf T}Aq.
\end{aligned}
```

where $q$ is the vector of gpuxtb atomic charges. $b$ has one value per atom,
and $A$ is a symmetric per-system matrix. A calling electrostatics program can
use these fields to couple gpuxtb to a periodic environment without requiring
gpuxtb itself to own lattice sums.

gpuxtb holds $b$ and $A$ fixed while differentiating. It therefore excludes
$db/dR$ and $dA/dR$ from returned forces and sets
`GPUXTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES`. The caller must
add those operator derivatives to obtain the force for a coordinate-dependent
periodic model.

## What this does not mean

The response interface is not a lattice/PBC descriptor. gpuxtb does not build
Ewald sums, periodic neighbor lists, or periodic GFN2 multipole interactions.
The external program owns those operations and supplies the resulting `b` and
$A$ fields.

See [theory and equations](../theory/qmmm.md) for the screened interaction,
energy, gradients, pinned xTB reference, and validation case.
