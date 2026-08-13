# GFN1-xTB implementation contract

GFN1-xTB is a distinct tight-binding model, not a second parameter table for
the GFN2 equations. xTBloom reserves its public model tag and enables each
backend independently only after that backend's complete implementation and
independent evidence are available. This page records the contract those
implementations must satisfy.

## Pinned references

The initial model audit uses three clean owner-requested local checkouts for
readable implementation inspection:

- tblite `133f91efb94b47f05848e1f86832f40a1accc385` as a readable calculator and
  exporter inspection revision;
- xTB `b31754bf3c7cccf8c242c469b03ae675e04bd608` as a readable production and
  analytic-gradient inspection revision; and
- dxtb `b529b5ddb75c0554274955082a189f9f88437cb2` as an independently structured
  implementation cross-check.

Canonical redistributed GFN1 parameter material is pinned to tblite 0.7.0
commit `fa8a4416e8fe093d0075bc10ac875494c2a449a9`. It is an ancestor of the local
tblite checkout. The later checkout was inspected for implementation context,
but its intervening changes include substantive runtime-equation updates as
well as formatting and workflow maintenance. Those later runtime changes are
non-authoritative and are not used to generate the canonical parameter bytes.
Primary closed-shell goldens use the separately pinned live tblite revision
`e9abc395b122018ed688aecb1c3a65cecaf97beb` with explicit `--method gfn1
--acc 0.0001 --grad --json`. xTB 6.7.1 revision
`edcfbbe39d411edc225e27315fbda3a204ddb023` supplies unrestricted,
point-charge, and halogen-specific reference cases. The source, inspection,
and live-oracle roles are intentionally distinct.

The non-TOML atomic and coordination inputs are pinned separately to mctc-lib
v0.5.2 commit `e9de066d89f250d1cfb6de3a33f0c27c0e2f855d`. The generated GFN1 JSON/header
retain the 86-element Pauling electronegativities, Mantina atomic radii, and
4/3-scaled Pyykko--Atsumi covalent radii. Lengths are converted using the
pinned CODATA 2018 expression at mctc working precision; its reviewed IEEE
binary64 value is `1.8897261246204404` bohr/Angstrom. Because those header
bytes derive from both tblite and mctc-lib, they carry
`LGPL-3.0-or-later AND Apache-2.0`.

Redistributed parameter bytes are generated from the reviewed tblite source
and covered by its LGPL-3.0-or-later grant. xTB and dxtb are oracle and review
inputs unless a later provenance manifest explicitly identifies redistributed
material from them.

## Basis, coordination, and zeroth-order Hamiltonian

The canonical tables cover elements H through Rn (`Z = 1..86`), 237 shells,
and 869 non-default symmetric element-pair scales. Unsupported atomic numbers
are invalid requests; there is no fallback element or GFN2 parameter lookup.

For each atom, the first shell encountered at each angular momentum is the
valence shell. A later shell with the same angular momentum is non-valence,
has zero reference occupation, and is orthogonalized against that first
matching shell—not against the immediately preceding shell and not through a
sequential Gram--Schmidt chain. The resulting first-shell valence mask is used
by both reference occupations and H0 scaling.

GFN1 uses the exponential coordination-number model selected by the canonical
export, rather than GFN2's double-exponential convention. For covalent radii
`r_cov,A` and `r_cov,B` and distance `r`, one pair contributes

```math
f_{AB}(r) = \frac{1}{1 + \exp\{-16[(r_{\mathrm{cov},A}+r_{\mathrm{cov},B})/r-1]\}}.
```

The default real-space cutoff is 25 bohr, the maximum-CN cutoff is disabled,
the directed factor and base electronegativity factor are both one, and pairs
are traversed over the lower triangle (including diagonal lattice images).
Pairs are skipped only for `r^2 > 25^2` or `r^2 < 1e-12`; equality at either
boundary remains included. Its analytic distance derivative is the negative
upstream logistic derivative and must feed every CN-dependent term, including
H0 and D3.

GFN1 H0 uses the pinned Pauling electronegativities for the squared element
difference below. The unscaled Mantina radius of each element is used by the
shell-polynomial distance factor and, after the separate canonical halogen
radius scale, by the classical halogen correction.

For shells `i` and `j`, the H0 off-diagonal scale has four branches:

- valence/valence uses the angular-momentum shell scale, the symmetric element
  pair scale (default or override), and the electronegativity factor
  `1 + enscale * (EN_A - EN_B)^2`;
- valence/non-valence uses half the sum of the valence shell's diagonal scale
  and `kpol`;
- non-valence/valence uses the corresponding scale of the other shell; and
- non-valence/non-valence uses `kpol` directly.

Pair overrides and the electronegativity factor apply only to the
valence/valence branch. Shell levels, shell polynomials, and shell CN shifts
come from the exact GFN1 shell table; they must not be interpreted through a
GFN2 shell layout.

## Electrostatics, spin, and classical terms

The remaining GFN1 scientific composition is:

- isotropic ES2 uses shell hardnesses and the harmonic pair average
  `g_ij = 2 / (1/g_i + 1/g_j)` with the canonical GFN1 exponent;
- ES3 is atom-resolved: for atomic charge `q_A` and Hubbard derivative
  `Gamma_A`, `E_ES3,A = Gamma_A q_A^3 / 3` and
  `V_ES3,A = Gamma_A q_A^2`; it is not GFN2's shell-resolved ES3;
- unrestricted spin retains the element and angular-momentum coupling table.
  Repeated shells participate by their angular momentum, so both hydrogen s
  shells receive the s--s coupling; restricted systems have zero spin energy;
- effective nuclear repulsion uses the GFN1 `zeff`, `arep`, `kexp`, and light
  element exponent policy from the canonical table;
- charge-independent D3(BJ) dispersion with the reviewed GFN1 parameters;
- the GFN1 halogen-bond correction; and
- no GFN2 anisotropic AES2/multipole SCC term and no self-consistent D4 term.

D3 uses the pinned simple-dftd3 reference coordination numbers, Gaussian
reference weights, packed reference C6 coefficients, r4/r2 values, and pair
van-der-Waals radii. It includes the two-body C6 and C8 energy, the direct
pair-coordinate derivative, and the `dE/dCN` chain through exponential CN.
The canonical damping constants are `s6`, `s8`, `a1`, and `a2`; `s9 = 0`, so
GFN1 has no Axilrod--Teller--Muto term.

The halogen correction treats Cl, Br, I, and At as donors and N, O, P, and S
as acceptors. The closest non-coincident atom to each donor defines the donor
axis. Candidate donor--acceptor pairs use a 20 bohr cutoff, scaled atomic
radii, the canonical Lennard-Jones-like radial factor, and the sixth-power
angular damping. Its analytic derivative acts on donor, acceptor, and axis
neighbor and must conserve the isolated-system net force.

Shared numerical utilities such as generalized eigensolution, occupations,
density construction, mixing, and failure publication may be reused only when
their equations and state layouts are genuinely model-independent.

The scalar Mulliken charge channel is the active GFN1 SCC variable. GFN2
atomic dipole/quadrupole SCC fields, AES2, and D4 charge state must not be
allocated, mixed, or silently included. Public energy components and forces
must identify GFN1 D3 and halogen contributions separately from GFN2 D4/AES2
semantics.

## Energy, SCC, and forces

At finite electronic temperature, the reported variational energy remains the
electronic Helmholtz free energy

```math
F = E_{\mathrm{GFN1}} - (k_{\mathrm B}T)S_{\mathrm{electronic}}.
```

Forces are the negative coordinate derivative of that converged free energy.
Term-level derivatives, total analytic forces, invariance, and conservation
must be checked independently of xTBloom before the public tag is enabled.
Restricted and unrestricted spin behavior must likewise be compared with the
pinned reference engines rather than assumed to match GFN2.

## Embedding is model-specific until proven

xTB contains distinct GFN1 point-charge potential and gradient paths. The
existing GFN2 screened point-charge implementation therefore cannot be reused
by naming alone. Explicit point charges and caller-supplied periodic `b + A*q`
response must either have GFN1-specific oracle and derivative evidence or be
rejected transactionally for GFN1 requests.

For a GFN1 shell hardness `g_i` and point-site hardness `g_p`, xTB's softened
interaction uses `x = 2 / (1/g_i + 1/g_p)` and
`J = (r^2 + x^-2)^(-1/2)`. This differs from the released GFN2 convention and
requires a separate implementation and finite-difference gate.

## Publication boundary

The stable C ABI already reserves `XTBLOOM_MODEL_GFN1_XTB`. While foundation
work is incomplete, it remains a known but unsupported model: validated GFN1
requests return `XTBLOOM_STATUS_NOT_SUPPORTED` before execution and leave all
caller outputs unchanged. CPU support may be advertised only after complete
energy, requested properties, analytic forces, ragged failure isolation, and
installed public-API evidence pass. CUDA support additionally requires real
GPU host/device/mixed parity and the repository sanitizer matrix.
