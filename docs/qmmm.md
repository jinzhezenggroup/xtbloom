# External point charges and periodic QM/MM coupling

gpuxtb follows the external-charge model implemented by xTB 6.7.1. The
reference interaction is a softened Coulomb potential acting directly on GFN2
shell monopoles. It does not add a point-charge electric field or field
gradient directly to the GFN2 atomic dipole or quadrupole potentials.

For shell `s` on atom `A` and external point charge `p`, define

```text
gamma_s = element_hardness[A] * shell_hubbard_scale[s]
a_sp    = 2 / (gamma_s + gamma_p)
K_sp    = 1 / sqrt(|R_A - R_p|^2 + a_sp^2)
V_s^PC  = sum_p Q_p K_sp
```

`gamma_p` is an explicit positive input in the low-level C API. The potential
`V_s^PC` is geometry dependent but SCC-iteration independent, so backends should
compute one value per shell before SCC and add it to the shell-charge potential
on every iteration. The direct external contributions to atomic dipole and
quadrupole potentials are zero. Those moments still change indirectly through
the converged density.

The converged explicit embedding energy is

```text
E_PC = sum_s q_s V_s^PC
```

where `q_s` is the net shell charge. gpuxtb does not compute point-charge to
point-charge interactions. For `d_Ap = R_A - R_p` and
`D_sp = |d_Ap|^2 + a_sp^2`, the coordinate gradients are

```text
dE_PC/dR_A = -sum_(s on A,p) q_s Q_p d_Ap D_sp^(-3/2)
dE_PC/dR_p = +sum_(A,s on A) q_s Q_p d_Ap D_sp^(-3/2)
```

The public API returns forces, so kernels return the negatives of these
gradients. Finite positive hardness keeps the energy finite when a QM atom and
point charge coincide; that geometry is valid and has zero pair gradient.

## Periodic caller-supplied response

The optional atom-level arrays `b` and `A` represent the periodic potential

```text
phi_periodic = b + A q
E_periodic   = q^T b + 0.5 q^T A q
```

where `A` must be symmetric. The atom potential is broadcast to all shells on
that atom. Consequently, the complete external shell shift is

```text
delta V_s = V_s^PC + b_A(s) + (A q)_A(s)
```

The caller constructs `b` and `A`, handles periodic and classical MM-MM
electrostatics, and adds coordinate derivatives of those arrays. When either
array participates in a calculation, gpuxtb sets
`GPUXTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES` because its forces
include only the explicit coordinate dependence known to the library.

## Pinned references and initial golden

- xTB interaction and gradients: [`embedding.f90`](https://github.com/grimme-lab/xtb/blob/b31754bf3c7cccf8c242c469b03ae675e04bd608/src/embedding.f90), with SCC insertion in [`scf_module.F90`](https://github.com/grimme-lab/xtb/blob/b31754bf3c7cccf8c242c469b03ae675e04bd608/src/scf_module.F90) and [`scc_core.f90`](https://github.com/grimme-lab/xtb/blob/b31754bf3c7cccf8c242c469b03ae675e04bd608/src/scc_core.f90).
- LAMMPS atom-level `b + A q` adapter: [`qmmm_xtb_adapter.f90`](https://github.com/lammps/lammps/blob/9ab8ca565e0f71d967587e0bca2015f7d689f19f/src/QMMM-XTB/qmmm_xtb_adapter.f90).

The executable numerical oracle is xTB 6.7.1 revision
`edcfbbe39d411edc225e27315fbda3a204ddb023`. For a neutral water molecule with
QM coordinates in bohr

```text
O   0.00000000   0.00000000   0.00000000
H   1.43233673   0.00000000   1.10715266
H  -1.43233673   0.00000000   1.10715266
```

and one point charge `Q=+0.5`, `gamma=0.405771` at `(4,0,0)` bohr, the reference
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
