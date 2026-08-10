# QM/MM force and energy accounting

## Included by xTBloom

For a converged system, xTBloom's reported energy includes:

- the supported GFN2-xTB variational energy;
- the screened QM-to-explicit-point-charge interaction `E_PC`; and
- `q^T b + 0.5 q^T A q` when caller operators are supplied.

Requested atomic forces include the negative derivative of the reported
variational energy for coordinate dependence known to xTBloom. Requested point-
charge forces include the direct screened QM/point interaction derivative.
The stationary response of the converged electronic variables is already
handled by the SCC variational formulation.

## Excluded and owned by the caller

The caller must supply or add:

- point-charge/point-charge energy and forces;
- classical bonded and nonbonded MM terms;
- topology, exclusions, switching, and periodic image conventions;
- lattice/Ewald/mesh electrostatic work used to construct `b` and `A`;
- virtual-site force redistribution; and
- explicit coordinate derivatives of `b` and `A`.

For any coordinate `R`, the missing operator-force contribution evaluated at
the converged atomic charges is

```text
F_R^operator = - q^T (db/dR) - 0.5 q^T (dA/dR) q.
```

Apply the expression to every QM, MM, cell, or collective coordinate on which
the external operator depends. Do not add a separate `dq/dR` term: the charge
response is variational because `b + A q` was included in SCC. If a model makes
the point-charge screening gamma coordinate-dependent, its explicit derivative
is also caller-owned.

Whenever the result contains
`XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES`, do not label the
raw xTBloom force as the complete force of a coordinate-dependent embedding.

## Force ledger

Maintain an explicit ledger for each coordinate family:

| Coordinate family | xTBloom contribution | Caller contribution |
| --- | --- | --- |
| QM atoms | GFN2 force; screened QM/point force; fixed-operator variational force | `db/dR`, `dA/dR`, classical boundary terms |
| Explicit MM point sites | screened QM/point force when requested | point-point/MM force field, operator derivatives |
| Virtual sites | force on the explicit site only | redistribute to parent atoms |
| Cell/lattice variables | none as native periodic GFN2 | all lattice, Ewald, and operator derivatives |

An isolated QM plus explicit-point-charge system without `b`, `A`, or other
external fields should conserve total direct force within numerical tolerance.
A subsystem coupled to a fixed external operator need not do so until the
caller's environment and derivative terms are included.

## Validation strategy

Use two distinct finite-difference tests.

### Library contribution at fixed operator

1. Hold `b` and `A` byte-for-byte fixed while displacing a QM or point-charge
   coordinate.
2. Recompute xTBloom energy.
3. Compare the central difference to the returned xTBloom force.

This validates the documented fixed-operator derivative. For explicit point
charges, also check equal-and-opposite summed QM/point force in an otherwise
isolated case and include exact coincidence as a finite zero-pair-force edge
case.

### Complete coupled model

1. Displace the chosen QM, MM, or cell coordinate.
2. Rebuild periodic electrostatics, `b`, and `A` at both displaced states.
3. Recompute xTBloom and all classical/external energy terms.
4. Compare the central difference of the assembled total energy to the sum of
   xTBloom forces and every caller-owned force-ledger term.

Do not compare the second energy difference to raw xTBloom forces alone; the
omitted operator derivative is intentional.

## Failure handling

An API-level success only guarantees that per-system diagnostics were
published. Inspect each `per_system_status`. A failed system's requested
floating-point slices, including QM and point-charge forces, are NaNs; do not
mix them into the MM force field. Preserve successful peer systems in a ragged
batch and decide explicitly whether the simulation retries, rejects, or
isolates the failed item.
