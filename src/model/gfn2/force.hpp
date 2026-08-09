#ifndef XTBLOOM_MODEL_GFN2_FORCE_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_FORCE_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/external_point_charges.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/repulsion.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

/*
 * Converged stationary data used by the CPU force composer.
 *
 * density and energy_weighted_density contain the spin-summed row-major AO
 * matrices, one per system in IntegralPlan packing. For unrestricted systems,
 * spin_density contains P_alpha-P_beta and spin_scalar_shell_potentials contains
 * the magnetization-channel derivative dE/dm. They must either both be present
 * or both be null; restricted systems use canonical nulls (or zero slices in a
 * mixed batch). scalar_shell_potentials is the complete charge-channel
 * scalar AO potential by shell: shell electrostatics plus every atom-wise
 * AES2, D4, explicit-charge, and host-supplied periodic contribution mapped
 * onto that shell. Dipole and quadrupole potentials are complete atom-major
 * derivatives of the converged interaction energy.
 *
 * scc_energies already contain the converged SCC free energy,
 * including H0, ES2, ES3, AES2, optional D4 two-body, optional embedding, and
 * finite-temperature entropy. The composer adds geometry-only repulsion and
 * optional D4 ATM. This split matches SccDriverState::free_energies.
 */
struct RestrictedGfn2StationaryInput {
  const double* positions = nullptr;
  const double* coordination_numbers = nullptr;
  std::uint64_t geometry_generation = 0u;

  const double* overlap = nullptr;
  const double* density = nullptr;
  const double* energy_weighted_density = nullptr;

  const double* shell_charges = nullptr;
  const double* atomic_charges = nullptr;
  const double* atomic_dipoles = nullptr;
  const double* atomic_quadrupoles = nullptr;

  const double* scalar_shell_potentials = nullptr;
  const double* atomic_dipole_potentials = nullptr;
  const double* atomic_quadrupole_potentials = nullptr;
  const double* scc_energies = nullptr;

  /* Canonical nulls disable explicit point charges. */
  const double* point_positions = nullptr;
  const double* point_charges = nullptr;
  const double* point_hardnesses = nullptr;

  /* Optional unrestricted stationary response, one matrix/shell per system. */
  const double* spin_density = nullptr;
  const double* spin_scalar_shell_potentials = nullptr;
};

/*
 * Optional diagnostic outputs. Every enabled pointer receives dE/dR, never a
 * force. Each array has 3*total_atoms doubles and is published only after the
 * corresponding stage succeeds. This deliberate progressive publication
 * identifies the last completed physical component if a later stage fails.
 * The D4 fields remain unused when D4 is disabled; external_point_charge
 * remains unused when no point plan is given.
 */
struct RestrictedGfn2ComponentGradients {
  double* electronic = nullptr;
  double* repulsion = nullptr;
  double* es2 = nullptr;
  double* aes2 = nullptr;
  double* d4_two_body = nullptr;
  double* d4_atm = nullptr;
  double* external_point_charge = nullptr;
};

/*
 * Caller-owned unpublished storage. Element counts use doubles. Energy-only
 * execution needs only energy_scratch and component workspaces required by
 * enabled energy terms; all force-only pointers/counts may be null/zero.
 */
struct RestrictedGfn2ForceWorkspace {
  double* energy_scratch = nullptr;
  double* component_energy_scratch = nullptr;
  std::int64_t energy_elements = 0;

  double* total_gradient = nullptr;
  double* component_gradient = nullptr;
  double* force_scratch = nullptr;
  std::int64_t coordinate_elements = 0;

  double* overlap_adjoint = nullptr;
  std::int64_t overlap_adjoint_elements = 0;
  double* dipole_adjoint = nullptr;
  std::int64_t dipole_adjoint_elements = 0;
  double* quadrupole_adjoint = nullptr;
  std::int64_t quadrupole_adjoint_elements = 0;
  double* coordination_adjoint = nullptr;
  std::int64_t coordination_adjoint_elements = 0;

  double* point_force_scratch = nullptr;
  std::int64_t point_force_elements = 0;

  void* integral_workspace = nullptr;
  std::size_t integral_workspace_size = 0u;

  ES2Workspace es2_workspace;
  AES2Workspace aes2_workspace;
  D4Workspace d4_workspace;
};

/*
 * Compose complete restricted or unrestricted GFN2 energies and analytic forces:
 *
 *   E = E_SCC + E_rep + E_D4^ATM,
 *
 *   dE/dR = [P_total:dH0/dR - W_total:dS/dR]
 *          + population-response S/D/Q integral contractions
 *          + unrestricted spin-population overlap response
 *          + ES2 + AES2 + D4(two-body+ATM) + repulsion
 *          + optional explicit point-charge derivatives.
 *
 * H0 and AES2 CN adjoints are contracted through the common GFN2
 * double-exponential coordination model. D4 gradients already include their
 * separate electronegativity-weighted CN chain. ES3 has no explicit
 * coordinate derivative. Host-supplied periodic b/A derivatives remain a
 * caller boundary, but their converged potentials must be included in
 * scalar_shell_potentials so charge-population response is retained.
 *
 * energies is mandatory and overwritten. qm_forces and point_forces use the
 * public F=-dE/dR convention and may be null for energy-only execution. A
 * point-force output requires a non-null external point-charge plan. Public
 * total energies and forces are transactional on failure; optional component
 * diagnostics follow the progressive diagnostic contract described above.
 */
xtbloom_status_t evaluate_restricted_gfn2_energy_forces_cpu(
    const BasisPlan& basis, const IntegralPlan& integrals, const CoordinationPlan& coordination,
    const RepulsionPlan& repulsion, const H0Plan& h0, const MullikenPlan& mulliken,
    const ES2Plan& es2, const ES2GeometryCache& es2_cache, const AES2Plan& aes2,
    const AES2GeometryCache& aes2_cache, const D4Plan* d4, const D4GeometryCache* d4_cache,
    const ExternalPointChargePlan* external_point_charges,
    const RestrictedGfn2StationaryInput& input, double* energies, double* qm_forces,
    double* point_forces, const RestrictedGfn2ComponentGradients& components,
    const RestrictedGfn2ForceWorkspace& workspace, std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_FORCE_HPP
