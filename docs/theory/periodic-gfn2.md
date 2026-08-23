# Native three-dimensional periodic GFN2-xTB

This page defines the public and numerical contract for native Gamma-point
GFN2-xTB under three-dimensional periodic boundary conditions (PBC). Native
PBC is a property of the quantum system: it supplies a direct lattice,
periodic images, Ewald electrostatics, periodic one-electron matrices, and a
cell derivative. It is not the caller-owned atom-level operator `b + A q`
described in [the QM/MM theory page](qmmm.md).

Only full `XYZ` periodicity is released. A partial-axis mask is invalid even
though the fixed-width ABI tag reserves its individual bits.

## Cell, coordinates, and reciprocal lattice

For one system the public `cell_matrices` slice is a row-major matrix

```math
H = \begin{bmatrix}\mathbf a^T\\\mathbf b^T\\\mathbf c^T\end{bmatrix},
```

in bohr. Fractional row vector `u` maps to the Cartesian row vector
`r = u H`. The cell must be finite, nonsingular, and right-handed:
`det(H) > 0`. The reciprocal row vectors are

```math
B = 2\pi H^{-T},
```

so `a_i dot b_j = 2 pi delta_ij`. Cartesian positions may lie outside the
central cell. They are wrapped through fractional coordinates with the
half-open convention `u - floor(u)`, producing `[0, 1)` in every released
periodic direction.

Integer translations are ordered lexicographically by `(n_a, n_b, n_c)`.
For the Wigner--Seitz minimum-image topology, the zero translation is ordered
first when it is eligible. The retained images are the closest image and every
competitor whose Cartesian distance differs from the minimum by strictly less
than `0.01` bohr. Every retained competitor has the static weight `1/n`.
This xTB-compatible canonical topology is distinct from the smooth
squared-distance weighting used by the pinned tblite implementation. Full
model oracle fixtures therefore stay away from that boundary; dedicated
topology tests cover exact 12/6/4 degeneracies and hostile skew cells.

## Periodic model terms

The Gamma-point basis contains only the orbitals in the central cell. Matrix
elements between two central-cell orbitals are sums over the eligible images
of the ket atom. The complete released energy includes periodic images in:

- exponential coordination numbers and their Cartesian/cell derivatives;
- screened nuclear repulsion;
- D4 coordination, two-body dispersion, and ATM three-body dispersion;
- overlap, dipole, and traceless-quadrupole integrals;
- the coordination-dependent zeroth-order Hamiltonian;
- shell-resolved isotropic second- and third-order SCC electrostatics; and
- complete atomic charge--dipole, dipole--dipole, and charge--quadrupole
  anisotropic electrostatics, including damping and self terms.

The physical sharp cutoffs are 25 bohr for GFN2 coordination and repulsion,
30 bohr for D4 coordination, 50 bohr for D4 two-body dispersion, and 25 bohr
for D4 ATM. Gaussian primitive pairs use the existing dimensionless
product-exponent threshold of 25. Image enumeration is complete for the
corresponding cutoff; a sparse-list builder cutoff is not a physical change.

## Ewald convention

Let `V = det(H)`, `R = n H`, `G = m B`, and `r_ij = r_i - r_j`. The long-range
monopole kernel is the conducting-boundary three-dimensional Ewald sum

```math
J_{ij} =
  \sum_R' \frac{\operatorname{erfc}(\alpha |r_{ij}+R|)}{|r_{ij}+R|}
  + \frac{4\pi}{V}\sum_{G\ne0}
    \frac{e^{-G^2/(4\alpha^2)}}{G^2}\cos(G\cdot r_{ij})
  - \delta_{ij}\frac{2\alpha}{\sqrt\pi}
  - \frac{\pi}{\alpha^2 V}.
```

The prime omits the singular `i = j, R = 0` term. The final constant is the
explicit uniform neutralizing background. For total cell charge `Q` it adds

```math
E_background = -pi Q^2 / (2 alpha^2 V),
phi_background = -pi Q / (alpha^2 V).
```

Consequently a charged calculation is finite and has an explicit convention;
it is never silently interpreted as a bare divergent lattice sum. At fixed
Ewald splitting parameter its direct strain contribution is isotropic,

```math
dE_background / d epsilon_ab =
  +pi Q^2 delta_ab / (2 alpha^2 V).
```

The short-range Klopman--Ohno correction is Wigner--Seitz averaged and added
to `J`. Complete GFN2 atomic multipoles use the corresponding first and second
derivatives of the Ewald Green function, the GFN2 damping functions, the
dipole self term, and the traceless charge--quadrupole self convention. The
reciprocal zero mode is omitted, corresponding to conducting boundary
conditions.

The splitting parameter is a deterministic internal numerical choice, not a
public physical input. Monopole real/reciprocal cutoffs are selected against
binary64 epsilon. The multipole reciprocal cutoff uses `100*sqrt(epsilon)`;
the reviewed compatibility real-space bound is 100 bohr. Every cutoff includes
all boundary vectors. Analytic derivatives treat `alpha` as a fixed Ewald
partition parameter; cancellation between real, reciprocal, self, and
background terms makes the converged result independent of it. Convergence
tests reconstruct the components at more than one `alpha`.

## Forces and cell derivative

The public Cartesian force remains

```math
F_A = -partial F / partial R_A,
```

where `F` is the variational electronic Helmholtz free energy at finite
electronic temperature. The native-PBC result also exposes one row-major
3-by-3 strain derivative per system,

```math
Xi_ab = partial F / partial epsilon_ab.
```

For a finite-difference displacement, Cartesian column vectors and the direct
lattice columns are transformed affinely as

```math
r' = (I + epsilon) r,
H' = H (I + epsilon)^T.
```

Equivalently, row-vector coordinates obey `r' = r (I + epsilon)^T`.
`Xi` has units of Hartree and has the same sign as the energy derivative. It
is often called a virial, but xTBloom does not negate it or divide it by cell
volume. A Cauchy stress, if desired, is a caller-side convention derived from
`Xi/V`. All nine ordered components are published; six symmetric affine modes
(`xx`, `yy`, `zz`, `xy`, `xz`, `yz`) are the minimum finite-difference gate,
and the antisymmetric part must be consistent with rotational invariance.

## State, updates, and unsupported combinations

Restricted and unrestricted GFN2 are both supported. Molecular charge,
unpaired electrons, spin-channel count, electronic temperature, atomic
numbers, atom partition, periodicity, and cell identity participate in plan
and strict-WARM compatibility. A geometry refresh may change positions and
the cell while retaining topology only when every cached image/list remains
valid. A topology or cell-identity change starts a new eligible epoch; CUDA
`WARM` never falls back to `FRESH`.

Native PBC combined with any of the following is rejected during complete
request validation, before caller output is touched:

- continuum solvation;
- explicit point charges;
- caller-supplied `atomic_potential_shifts` or `charge_response_matrix`; or
- an interaction whose periodic energy and Cartesian/cell derivatives have
  not been independently proven.

The implementation does not add classical lattice energy, nuclear background
terms beyond the stated GFN2/Ewald model, or non-Gamma k-point sampling.

## Independent evidence

Neutral full-model energy, Cartesian gradient, and strain-derivative goldens
are generated by tblite 0.7.0 at revision
`133f91efb94b47f05848e1f86832f40a1accc385`. The fixtures are xTBloom-authored
orthogonal, skew, one-atom, and unrestricted cells. The periodic oracle tool
pins the executable, selected shared library, source modules, command, input,
and normalized output hashes. It never uses xTBloom output to create a golden.

Pinned tblite is not the charged-cell oracle: its reviewed monopole matrix
omits the uniform-background constant above. Charged background energy,
potential, and strain are therefore checked by an independent analytic
reconstruction. xTB is used only for focused topology and short-range
diagnostics because its periodic GFN2 multipole and public virial paths are
not a complete full-model reference.
