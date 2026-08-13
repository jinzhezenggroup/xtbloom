# External point charges and periodic QM/MM coupling

For input/output responsibilities and Python/native usage, start with the
[QM/MM user guide](../user-guide/qmmm.md). This page records the equations,
reference sources, and pinned numerical evidence.

xTBloom follows the model-specific external-charge paths implemented by xTB
6.7.1. Both are softened Coulomb potentials on shell monopoles, but their
hardness combination differs. Neither adds a direct point-charge field or field
gradient to GFN2 atomic dipole or quadrupole potentials.

For GFN1 shell harmonic hardness $g_s$ and point-site hardness $g_p$, define
$x_{sp}=2/(1/g_s+1/g_p)$ and
$K_{sp}=(\lVert\mathbf R_A-\mathbf R_p\rVert^2+x_{sp}^{-2})^{-1/2}$.
The published GFN1 implementation and oracle fixtures use this equation.

For GFN2, the shell-scaled hardness convention is:

For shell $s$ on atom $A$ and external point charge $p$, define

```math
\begin{aligned}
\gamma_s &= \gamma_A^{\mathrm{element}} h_s, \\
a_{sp} &= \frac{2}{\gamma_s + \gamma_p}, \\
K_{sp} &= \left(\lVert \mathbf R_A - \mathbf R_p \rVert^2
            + a_{sp}^2\right)^{-1/2}, \\
V_s^{\mathrm{PC}} &= \sum_p Q_p K_{sp}.
\end{aligned}
```

$\gamma_A^{\mathrm{element}}$ is the GFN2 element hardness and $h_s$ is the shell
Hubbard scaling factor. $\gamma_p$ is an explicit positive input in the
low-level C API. The potential $V_s^{\mathrm{PC}}$ is geometry dependent but
SCC-iteration independent, so backends compute one value per shell before SCC
and add it to the shell-charge potential on every iteration. The direct
external contributions to atomic dipole and quadrupole potentials are zero.
Those moments still change indirectly through the converged density.

The converged explicit embedding energy is

```math
E_{\mathrm{PC}} = \sum_s q_s V_s^{\mathrm{PC}}.
```

where $q_s$ is the net shell charge and $K_{sp}$ is the appropriate GFN1 or
GFN2 kernel above. xTBloom does not compute point-charge to point-charge
interactions. For $\mathbf d_{Ap} = \mathbf R_A - \mathbf R_p$, define
$D_{sp}=\lVert\mathbf d_{Ap}\rVert^2+x_{sp}^{-2}$ for GFN1 and
$D_{sp}=\lVert\mathbf d_{Ap}\rVert^2+a_{sp}^2$ for GFN2. The coordinate
gradients are

```math
\begin{aligned}
\frac{\partial E_{\mathrm{PC}}}{\partial \mathbf R_A}
  &= -\sum_{s\in A}\sum_p q_s Q_p\,\mathbf d_{Ap}D_{sp}^{-3/2}, \\
\frac{\partial E_{\mathrm{PC}}}{\partial \mathbf R_p}
  &= +\sum_A\sum_{s\in A} q_s Q_p\,\mathbf d_{Ap}D_{sp}^{-3/2}.
\end{aligned}
```

The public API returns forces, so kernels return the negatives of these
gradients. Finite positive hardness keeps the energy finite when a QM atom and
point charge coincide; that geometry is valid and has zero pair gradient.

## Periodic caller-supplied response

The optional atom-level arrays $b$ and $A$ represent the periodic potential

```math
\begin{aligned}
\boldsymbol\phi_{\mathrm{periodic}} &= b + Aq, \\
E_{\mathrm{periodic}} &= q^{\mathsf T}b
  + \frac{1}{2}q^{\mathsf T}Aq.
\end{aligned}
```

where $A$ must be symmetric. The atom potential is broadcast to all shells on
that atom. Consequently, the complete external shell shift is

```math
\Delta V_s = V_s^{\mathrm{PC}} + b_{A(s)} + (Aq)_{A(s)}.
```

The caller constructs $b$ and $A$, handles periodic and classical MM-MM
electrostatics, and adds coordinate derivatives of those arrays. When either
array participates in a calculation, xTBloom sets
`XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES` because its forces
include only the explicit coordinate dependence known to the library.

## Pinned references and initial golden

- xTB interaction and gradients: [`embedding.f90`](https://github.com/grimme-lab/xtb/blob/b31754bf3c7cccf8c242c469b03ae675e04bd608/src/embedding.f90), with SCC insertion in [`scf_module.F90`](https://github.com/grimme-lab/xtb/blob/b31754bf3c7cccf8c242c469b03ae675e04bd608/src/scf_module.F90) and [`scc_core.f90`](https://github.com/grimme-lab/xtb/blob/b31754bf3c7cccf8c242c469b03ae675e04bd608/src/scc_core.f90).
- LAMMPS atom-level $b + Aq$ adapter: [`qmmm_xtb_adapter.f90`](https://github.com/lammps/lammps/blob/9ab8ca565e0f71d967587e0bca2015f7d689f19f/src/QMMM-XTB/qmmm_xtb_adapter.f90).

The executable numerical oracle is xTB 6.7.1 revision
`edcfbbe39d411edc225e27315fbda3a204ddb023`. For a neutral water molecule with
QM coordinates in bohr

```text
O   0.00000000   0.00000000   0.00000000
H   1.43233673   0.00000000   1.10715266
H  -1.43233673   0.00000000   1.10715266
```

and one point charge $Q=+0.5$, $\gamma=0.405771$ at $(4,0,0)$ bohr, the reference
result is

```text
energy = -5.0730682804123326 Eh
QM gradients (Eh/bohr):
  O  -0.0112621578336112   0  -0.0033988314289940
  H   0.0013479723470460   0  -0.0012180034891371
  H   0.0074467401247049   0   0.0012859430397343
point-charge gradient:
      0.0024674453618603   0   0.0033308918783968
```

The summed QM and point-charge gradients are below `4e-18 Eh/bohr`. Central
differences with a `1e-4 bohr` step agree within `3.4e-8 Eh/bohr` for QM
coordinates and `2.3e-8 Eh/bohr` for the point-charge coordinates.
