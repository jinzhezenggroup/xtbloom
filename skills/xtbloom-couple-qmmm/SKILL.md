---
name: xtbloom-couple-qmmm
description: Couple xTBloom GFN2-xTB calculations to QM/MM electrostatics using explicit screened point charges and caller-supplied atom-level charge-response operators b + A q. Use when implementing or reviewing point-charge inputs, periodic embedding adapters, QM and MM force accounting, finite-difference validation, or diagnosing missing external-operator derivative terms.
---

# Couple xTBloom to QM/MM

Treat xTBloom as the variational GFN2 subsystem and make the calling
electrostatics or simulation program own every external term that xTBloom does
not calculate.

## Choose the Host Environment

For an agent-generated standalone Python coupling example, add PEP 723 metadata
with `dependencies = ["xtbloom>=0.1.1"]`, then run it with `uv run --script
qmmm.py`. Add only the host package the user actually selected. Preserve an
existing simulation environment when integrating into one; do not impose uv on
a C/C++ or native simulation workflow.

## Workflow

1. Define the coupling boundary before writing descriptors. Decide whether the
   calculation uses explicit point charges, a caller-built `b + A q` operator,
   or both. Read [references/qmmm-input-contract.md](references/qmmm-input-contract.md).
2. Convert positions to bohr, charges to elementary-charge units, and all
   energies, potentials, and forces to consistent atomic units. Reject
   non-finite values; require each point-charge screening `gamma` to be finite
   and positive.
3. Pack one point-charge segment and one response-matrix block per batch item.
   Preserve exact row-major symmetry of every `A` block and bind all buffers as
   caller-owned views with correct byte extents.
4. Request atomic forces, point-charge forces, and atomic charges according to
   the application ledger. Point charges and `b + A q` participate in every SCC
   iteration; do not add them afterward as a one-shot energy correction.
5. After a successful call, inspect every per-system status and the result
   flags. If `XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES` is
   set, add the caller-owned `db/dR` and `dA/dR` force terms before reporting a
   total force.
6. Complete the force ledger in
   [references/force-accounting.md](references/force-accounting.md). Include
   MM-MM electrostatics, force-field terms, exclusions, and virtual-site force
   redistribution outside xTBloom.
7. Validate the library contribution with fixed external operators, then
   validate the assembled coupled model with operators recomputed at displaced
   coordinates. Do not compare a total-energy finite difference to the
   intentionally incomplete fixed-operator force.

## Non-negotiable boundaries

- Explicit point charges and `b + A q` are SCC inputs, not post-processing.
- xTBloom differentiates while holding caller-supplied `b` and `A` fixed. The
  caller owns their coordinate derivatives; no `dq/dR` term should be added
  separately to the variational force.
- xTBloom does not calculate point-charge/point-charge energy or forces,
  classical MM terms, topology/exclusions, or virtual-site redistribution.
- The response interface is not native periodic GFN2. xTBloom has no lattice
  descriptor, Ewald solver, periodic neighbor list, or periodic GFN2 multipole
  implementation. The caller constructs the periodic operator and its
  derivatives.
- A point-charge `gamma` is a screening parameter, not an optimizable spatial
  degree of freedom. Coincident QM and point-charge positions remain finite and
  have zero direct pair force at exact coincidence.
- GFN1-xTB QM/MM is supported on CPU only; GFN2-xTB supports CPU and CUDA.
  ROCm remains an unsupported reserved value.

## Resources

- `references/qmmm-input-contract.md`: equations, array packing, units, and
  non-periodic scope.
- `references/force-accounting.md`: included and caller-owned energy/force
  terms plus validation strategy.
