#ifndef XTBLOOM_MODEL_GFN1_FORCE_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_FORCE_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#include "model/gfn1/basis.hpp"
#include "model/gfn1/coordination.hpp"
#include "model/gfn1/d3.hpp"
#include "model/gfn1/es2.hpp"
#include "model/gfn1/external_point_charges.hpp"
#include "model/gfn1/h0.hpp"
#include "model/gfn1/halogen.hpp"
#include "model/gfn1/integrals.hpp"
#include "model/gfn1/mulliken.hpp"
#include "model/gfn1/repulsion.hpp"

namespace xtbloom::detail::gfn1 {

/* Converged stationary GFN1 data. Density and W are spin-summed matrices. */
struct StationaryInput {
  /* Exact model identity used to seal all independently constructed plans. */
  const std::int32_t* atomic_numbers = nullptr;
  const double* positions = nullptr;
  const double* coordination_numbers = nullptr;
  std::uint64_t geometry_generation = 0u;
  const double* overlap = nullptr;
  const double* density = nullptr;
  const double* energy_weighted_density = nullptr;
  const double* shell_charges = nullptr;
  const double* scalar_shell_potentials = nullptr;
  const double* scc_free_energies = nullptr;

  /* Optional unrestricted stationary response. Both pointers are present or absent. */
  const double* spin_density = nullptr;
  const double* spin_shell_potentials = nullptr;

  /* Canonical nulls disable explicit point charges. */
  const double* point_positions = nullptr;
  const double* point_charges = nullptr;
  const double* point_hardnesses = nullptr;
};

struct ComponentGradients {
  double* electronic = nullptr;
  double* es2 = nullptr;
  double* d3 = nullptr;
  double* repulsion = nullptr;
  double* halogen = nullptr;
  double* external_point_charge = nullptr;
};

struct ForceWorkspace {
  double* energy_scratch = nullptr;
  double* component_energy_scratch = nullptr;
  std::int64_t energy_elements = 0;
  double* total_gradient = nullptr;
  double* component_gradient = nullptr;
  /*
   * Six coordinate-sized unpublished slices in ComponentGradients order.
   * The composer writes requested diagnostics here and publishes every caller
   * output only after the complete energy/force request succeeds.
   */
  double* component_gradient_staging = nullptr;
  std::int64_t component_gradient_staging_elements = 0;
  double* force_scratch = nullptr;
  std::int64_t coordinate_elements = 0;
  double* overlap_adjoint = nullptr;
  std::int64_t overlap_elements = 0;
  double* coordination_adjoint = nullptr;
  std::int64_t atom_elements = 0;
  double* point_force_scratch = nullptr;
  std::int64_t point_force_elements = 0;
  void* integral_workspace = nullptr;
  std::size_t integral_workspace_size = 0u;
  ES2Workspace es2_workspace;
  D3Workspace d3_workspace;
  HalogenWorkspace halogen_workspace;
};

/*
 * Compose GFN1 energy and analytic force at the converged stationary state:
 *
 *   E = F_SCC + E_rep + E_D3 + E_halogen.
 *
 * Periodic b/A operators have no explicit coordinate derivative because the
 * public contract holds them fixed; their converged scalar potentials must
 * nevertheless be included in scalar_shell_potentials so overlap response is
 * retained. Outputs are transactional. Component diagnostics are gradients.
 */
xtbloom_status_t evaluate_gfn1_energy_forces_cpu(
    const BasisPlan& basis, const IntegralPlan& integrals, const CoordinationPlan& coordination,
    const RepulsionPlan& repulsion, const H0Plan& h0, const MullikenPlan& mulliken,
    const ES2Plan& es2, const ES2GeometryCache& es2_cache, const D3Plan& d3,
    const HalogenPlan& halogen, const ExternalPointChargePlan* external,
    const StationaryInput& input, double* energies, double* qm_forces, double* point_forces,
    const ComponentGradients& components, const ForceWorkspace& workspace, std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_FORCE_HPP
