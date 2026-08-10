# QM/MM input contract

## Two SCC-level coupling mechanisms

xTBloom supports two external electrostatic mechanisms that may be used
together:

1. explicit screened point charges; and
2. an atom-level caller-supplied operator `b + A q`.

Both enter every SCC iteration and contribute to the reported variational
energy. They are not post-SCC corrections.

## Explicit point charges

For each point charge provide:

- position `R_p` in bohr;
- charge value `Q_p` in elementary-charge units; and
- finite positive screening parameter `gamma_p` in the public atomic-unit
  convention.

For a ragged batch of `B` systems and `P` total points:

- `total_point_charges = P`;
- `point_charge_offsets` contains `B + 1` `int64_t` entries, begins at zero,
  is monotonically nondecreasing, and ends at `P`;
- `point_charge_positions` contains `3P` doubles;
- `point_charge_values` and `point_charge_gammas` each contain `P` doubles.

Supply the complete group when `P` is nonzero. To request returned forces on
the point sites, set `XTBLOOM_COMPUTE_POINT_CHARGE_FORCES` and bind a
`point_charge_forces` output of `3P` doubles. Request atomic forces separately
with `XTBLOOM_COMPUTE_FORCES`.

xTBloom follows a softened Coulomb interaction with GFN2 shell monopoles. For
shell `s` on atom `A` and point `p`:

```text
gamma_s = element_hardness_A * shell_Hubbard_scale_s
a_sp    = 2 / (gamma_s + gamma_p)
K_sp    = (|R_A - R_p|^2 + a_sp^2)^(-1/2)
V_s^PC  = sum_p Q_p K_sp
E_PC    = sum_s q_s V_s^PC
```

The potential is added to the shell-charge SCC channel. No direct point-charge
field or field-gradient term is added to GFN2 atomic dipole or quadrupole
potentials; those moments may still change through the converged density.
Finite screening makes coincident QM/point positions valid, finite, and zero-
force for that exact pair.

`gamma_p` is a model input, not a point position or an automatically
differentiated degree of freedom. If an application makes gamma coordinate-
dependent, that derivative belongs to the application.

## Caller-supplied charge response

Let `q` be the converged vector of xTBloom atomic charges. The external
operator and its variational energy are

```text
phi = b + A q
E_response = q^T b + 0.5 q^T A q
```

The per-atom potential is broadcast to every shell on that atom. With explicit
point charges, the complete external shell shift is

```text
Delta V_s = V_s^PC + b_A(s) + (A q)_A(s).
```

For `N` total QM atoms:

- `atomic_potential_shifts` contains `N` doubles for `b` when supplied;
- each system `i` with `n_i` atoms has one row-major `n_i x n_i` matrix `A_i`;
- every `A_i` must be exactly symmetric;
- `charge_response_offsets` has `B + 1` `int64_t` entries and packs exactly
  `n_i^2` elements for system `i`;
- `total_charge_response_elements = sum_i n_i^2`;
- `charge_response_matrix` contains that many doubles.

At the low-level C ABI, `b` may be present without `A`. A nonzero response
matrix requires the declared count, offsets, and matrix together. Higher-level
bindings may intentionally require `b`, offsets, and `A` as one group; inspect
the interface being integrated.

All operator values must be finite. Use potential/operator units consistent
with atomic charges and the Hartree energy expression above.

## Result flag and differentiation convention

When `b` or `A` participates, xTBloom holds those caller-owned arrays fixed
while differentiating and sets
`XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES`. Treat the flag
as a force-accounting requirement, not an informational detail.

## Backend, lifetime, and warm-state rules

The ordinary C ABI buffer ownership rules apply. CPU uses host views. CUDA may
use host, device, or mixed views on the context's resolved device. Keep views
alive through synchronous compute or through asynchronous completion as
required by the API.

Point-charge and response structure participate in strict warm-state
compatibility. Use `FRESH` when coupling topology or policy changes unless the
installed header explicitly defines the change as compatible. `WARM` never
silently falls back to a new electronic state.

## Non-periodic GFN2 boundary

The `b + A q` interface lets an external program represent a periodic
electrostatic environment, but it does not make the xTBloom GFN2 calculation
periodic. xTBloom does not own:

- a lattice/cell descriptor;
- Ewald or particle-mesh electrostatics;
- periodic neighbor lists or image construction;
- periodic GFN2 multipole interactions; or
- classical MM electrostatics.

The calling program builds the operator, manages periodic images and
classical terms, and supplies all missing derivatives.
