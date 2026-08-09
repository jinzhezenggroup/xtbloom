# GFN2-xTB model and SCC

xTBloom implements the published GFN2-xTB model rather than fitting a new
variant. The CPU implementation is the readable equation reference inside the
repository, while CUDA implements the same parameters, state, convergence, and
publication semantics.

## Energy model

At a high level, one GFN2-xTB calculation combines:

- the geometry-dependent overlap and zeroth-order Hamiltonian;
- self-consistent isotropic and anisotropic electrostatics using shell charges
  and atomic charge, dipole, and quadrupole moments;
- short-range repulsion;
- self-consistent D4 two-body dispersion;
- the charge-independent D4 Axilrod-Teller-Muto three-body contribution; and
- optional external point-charge and atom-level response operators.

At finite electronic temperature, fractionally occupied orbitals add the
electronic entropy contribution. The public energy is therefore

```math
F = E_{\mathrm{internal}} - (k_{\mathrm B}T)S_{\mathrm{electronic}}.
```

Here $k_{\mathrm B}T$ is the electronic-temperature energy scale used by the
SCC occupations, and $S_{\mathrm{electronic}}$ is the corresponding
dimensionless occupation entropy. At zero temperature, $F$ reduces to the
internal energy.

xTBloom does not add solvation, a lattice Hamiltonian, classical MM-MM energy,
or point-charge/point-charge energy.

## Self-consistent charge cycle

For each system, the implementation conceptually performs these stages:

1. Build geometry, coordination, basis, overlap, core Hamiltonian, repulsion,
   and dispersion inputs.
2. Start from the immutable fresh electronic state or a strictly compatible
   converged checkpoint requested through the low-level ABI.
3. Assemble the SCC potentials from shell monopoles, atomic multipoles,
   the charge-dependent D4 two-body term, and optional external operators.
4. Solve the generalized eigenproblem in the non-orthogonal basis.
5. Determine restricted or unrestricted finite-temperature occupations and
   construct density matrices.
6. Reduce the density into shell charges and atomic charge, dipole, and
   quadrupole moments.
7. Compose the complete free energy and charge residual in a fixed semantic
   order, test both convergence criteria, and otherwise update the state with
   modified-Broyden mixing.

Charge and energy convergence are both required. Reaching the iteration limit
is a per-system numerical result rather than a call-level failure, so other
systems in a ragged batch can still converge and publish results.

## Restricted and unrestricted spin

The molecular charge and unpaired-electron count determine the electron
population. `spin_channels=1` uses shared restricted orbitals;
`spin_channels=2` uses separate unrestricted channels. The Python API defaults
an open-shell system to unrestricted and submits that choice explicitly.

Electron-count and spin-parity inconsistencies are validated for the complete
request before execution. They are input errors, not SCC failures.

## Finite-temperature occupations

Occupations use the public electronic temperature and conserve the requested
electron count within the binary64 publication contract. Exactly degenerate
eigenspaces receive symmetric occupations so a basis rotation or permutation
inside the degenerate subspace does not change published populations.

Some targets cannot be represented by one identical binary64 value across a
multi-orbital degenerate block. CPU and CUDA share a bounded candidate search
that selects the closest representable symmetric state under a documented
quantization bound. Nondegenerate spectra keep the strict count tolerance, and
the zero-temperature Aufbau path is unaffected. The complete rare-path ordering
and acceptance rules are recorded in the
[architecture guide](../developer-guide/architecture.md#compute-semantics).

## Analytic forces

The public force is

```math
\mathbf F_A = -\frac{\partial F}{\partial \mathbf R_A}.
```

for the converged variational free energy. Analytic derivatives include the
geometry dependence of the implemented GFN2 terms and explicit screened
point-charge interaction. They are checked against central finite differences
and independent conformance cases.

Caller-supplied periodic fields $b$ and $A$ are held fixed. Their coordinate
derivatives are excluded because xTBloom does not know how the external
electrostatics program constructed them. See the
[QM/MM theory page](qmmm.md).

## Numerical evidence

Correctness is established through more than total-energy agreement:

- term-level unit tests and force finite differences;
- charge, spin, and SCC edge cases;
- translational invariance and force conservation where applicable;
- public-ABI conformance against hash-pinned xTB/tblite artifacts;
- restricted/unrestricted and CPU/CUDA parity; and
- trace tooling for SCC state and convergence behavior.

Canonical goldens are generated only through the pinned workflows in
[`tools/conformance/README.md`](../../tools/conformance/README.md). They are
independent evidence and are never rewritten to match current xTBloom output.
