#include <array>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc_iteration_reports.cuh"

namespace gpuxtb::detail::cuda {
namespace {

using BindingDiagnostic = Gfn2SccIterationBindingDiagnostic;
using BindingError = Gfn2SccIterationBindingError;
using BindingField = Gfn2SccIterationBindingField;

constexpr std::uint32_t component_bit(Gfn2SccPotentialComponent component) noexcept {
  return static_cast<std::uint32_t>(component);
}

constexpr std::uint32_t kMandatoryComponents = component_bit(Gfn2SccPotentialComponent::kES2) |
                                               component_bit(Gfn2SccPotentialComponent::kES3) |
                                               component_bit(Gfn2SccPotentialComponent::kAES2);

BindingDiagnostic fail(BindingError error, BindingField field, std::int64_t index = -1) noexcept {
  return {error, field, index};
}

bool enabled(std::uint32_t components, Gfn2SccPotentialComponent component) noexcept {
  return (components & component_bit(component)) != 0u;
}

struct CanonicalStages {
  std::array<Gfn2SccStageId, kGfn2SccIterationMaximumStageReportCount> values{};
  std::int64_t count = 0;

  void append(Gfn2SccStageId stage) noexcept { values[static_cast<std::size_t>(count++)] = stage; }
};

CanonicalStages canonical_stages(std::uint32_t components) noexcept {
  CanonicalStages stages{};
  stages.append(Gfn2SccStageId::kMixedGather);
  stages.append(Gfn2SccStageId::kSpinPotential);
  stages.append(Gfn2SccStageId::kES2Potential);
  stages.append(Gfn2SccStageId::kES3Potential);
  stages.append(Gfn2SccStageId::kAES2Potential);
  if (enabled(components, Gfn2SccPotentialComponent::kD4TwoBody)) {
    stages.append(Gfn2SccStageId::kD4Potential);
  }
  if (enabled(components, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    stages.append(Gfn2SccStageId::kPeriodicPotential);
  }
  stages.append(Gfn2SccStageId::kPotentialCompose);
  stages.append(Gfn2SccStageId::kScalarBridge);
  stages.append(Gfn2SccStageId::kHamiltonian);
  stages.append(Gfn2SccStageId::kEigensolver);
  stages.append(Gfn2SccStageId::kOccupations);
  stages.append(Gfn2SccStageId::kDensity);
  stages.append(Gfn2SccStageId::kMulliken);
  stages.append(Gfn2SccStageId::kSpinRawEnergy);
  stages.append(Gfn2SccStageId::kES2RawEnergy);
  stages.append(Gfn2SccStageId::kES3RawEnergy);
  stages.append(Gfn2SccStageId::kAES2RawEnergy);
  if (enabled(components, Gfn2SccPotentialComponent::kD4TwoBody)) {
    stages.append(Gfn2SccStageId::kD4RawEnergy);
  }
  if (enabled(components, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    stages.append(Gfn2SccStageId::kExplicitPointChargeRawEnergy);
  }
  if (enabled(components, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    stages.append(Gfn2SccStageId::kPeriodicRawEnergy);
  }
  stages.append(Gfn2SccStageId::kClassicalEnergy);
  stages.append(Gfn2SccStageId::kElectronicEnergy);
  stages.append(Gfn2SccStageId::kFreeEnergy);
  stages.append(Gfn2SccStageId::kMixer);
  stages.append(Gfn2SccStageId::kStatePublication);
  return stages;
}

struct ReportSpec {
  Gfn2SccStageCodeFormat format = Gfn2SccStageCodeFormat::kUint32Error;
  Gfn2SccStageDeviceCodeRole role = Gfn2SccStageDeviceCodeRole::kMixedFirstError;
  std::uint64_t peer_mask = 0u;
  gpuxtb_status_t peer_status = GPUXTB_STATUS_INTERNAL_ERROR;
};

bool report_spec(Gfn2SccStageId stage, ReportSpec& spec) noexcept {
  switch (stage) {
    case Gfn2SccStageId::kMixedGather:
      spec.peer_mask = 0xfcu;
      return true;
    case Gfn2SccStageId::kSpinPotential:
    case Gfn2SccStageId::kSpinRawEnergy:
      spec.peer_mask = kGfn2SpinDevicePeerErrorMask;
      return true;
    case Gfn2SccStageId::kES2Potential:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0x380u;
      return true;
    case Gfn2SccStageId::kES3Potential:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0x1cu;
      return true;
    case Gfn2SccStageId::kAES2Potential:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0x706u;
      return true;
    case Gfn2SccStageId::kD4Potential:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0xf0u;
      return true;
    case Gfn2SccStageId::kPeriodicPotential:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0xecu;
      return true;
    case Gfn2SccStageId::kPotentialCompose:
      spec.peer_mask = 0xff0cu;
      return true;
    case Gfn2SccStageId::kScalarBridge:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0xf0u;
      return true;
    case Gfn2SccStageId::kHamiltonian:
      spec.peer_mask = 0x1feu;
      return true;
    case Gfn2SccStageId::kEigensolver:
      spec.peer_mask = 0x3e06u;
      spec.peer_status = GPUXTB_STATUS_EIGENSOLVER_FAILED;
      return true;
    case Gfn2SccStageId::kOccupations:
      spec.peer_mask = 0x3feu;
      spec.peer_status = GPUXTB_STATUS_EIGENSOLVER_FAILED;
      return true;
    case Gfn2SccStageId::kDensity:
      spec.peer_mask = 0x7feu;
      spec.peer_status = GPUXTB_STATUS_EIGENSOLVER_FAILED;
      return true;
    case Gfn2SccStageId::kMulliken:
      spec.peer_mask = 0x1feu;
      return true;
    case Gfn2SccStageId::kES2RawEnergy:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0x980u;
      return true;
    case Gfn2SccStageId::kES3RawEnergy:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0x4cu;
      return true;
    case Gfn2SccStageId::kAES2RawEnergy:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0x1b06u;
      return true;
    case Gfn2SccStageId::kD4RawEnergy:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0xf0u;
      return true;
    case Gfn2SccStageId::kExplicitPointChargeRawEnergy:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0x60u;
      return true;
    case Gfn2SccStageId::kPeriodicRawEnergy:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0x174u;
      return true;
    case Gfn2SccStageId::kClassicalEnergy:
      spec.peer_mask = 0x1feu;
      return true;
    case Gfn2SccStageId::kElectronicEnergy:
      spec.peer_mask = 0xfcu;
      return true;
    case Gfn2SccStageId::kFreeEnergy:
      spec.peer_mask = 0x1ffeu;
      return true;
    case Gfn2SccStageId::kMixer:
      spec.format = Gfn2SccStageCodeFormat::kGpuxtbStatus;
      spec.peer_mask = 0x40u;
      return true;
    case Gfn2SccStageId::kStatePublication:
      spec.role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
      spec.peer_mask = 0x1f8u;
      return true;
    default:
      return false;
  }
}

template <typename T>
T* storage_slot(T* base, std::int64_t slot, std::int64_t stride) noexcept {
  const auto address = reinterpret_cast<std::uintptr_t>(base);
  const auto offset = static_cast<std::uint64_t>(slot) * static_cast<std::uint64_t>(stride) *
                      static_cast<std::uint64_t>(sizeof(T));
  if (offset > std::numeric_limits<std::uintptr_t>::max() - address) return nullptr;
  return reinterpret_cast<T*>(address + static_cast<std::uintptr_t>(offset));
}

BindingDiagnostic validate_storage_range(const void* pointer, std::int64_t available,
                                         std::int64_t required, std::size_t element_size,
                                         std::size_t alignment, std::int64_t index) noexcept {
  if (required <= 0 || available < required) {
    return fail(BindingError::kInsufficientCapacity, BindingField::kStageReports, index);
  }
  if (pointer == nullptr) {
    return fail(BindingError::kNullPointer, BindingField::kStageReports, index);
  }
  if (reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0u) {
    return fail(BindingError::kMisalignedPointer, BindingField::kStageReports, index);
  }
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  const auto elements = static_cast<std::uint64_t>(required);
  if (elements > std::numeric_limits<std::uintptr_t>::max() / element_size) {
    return fail(BindingError::kAddressOverflow, BindingField::kStageReports, index);
  }
  const auto bytes = static_cast<std::uintptr_t>(elements * element_size);
  if (bytes != 0u && bytes - 1u > std::numeric_limits<std::uintptr_t>::max() - address) {
    return fail(BindingError::kAddressOverflow, BindingField::kStageReports, index);
  }
  return {};
}

/*
 * Rebuild every descriptor edge whose identity is fixed by the SCC dataflow.
 * The arena binder owns storage, while this pass owns views: callers only seed
 * immutable inputs (H0/integrals), persistent leaves, and unpublished leaves.
 * Reconstructing the aliases here prevents stale hand-built descriptors from
 * silently crossing a plan boundary and keeps the launch path pointer-free.
 */
void project_zero_copy_views(Gfn2SccIterationProjectedDescriptors& candidate) noexcept {
  auto& plan = candidate.plan;
  auto& input = candidate.input;
  auto& state = candidate.state;
  auto& workspace = candidate.workspace;
  const std::uint64_t token = plan.plan_token;
  const std::uint32_t components = plan.enabled_components;
  const bool mixed_spin = plan.wavefunction_layout.total_spin_channels != plan.topology.batch_size;

  /* One canonical activity ledger fans out to every active-aware primitive. */
  const auto* active = workspace.ledger.active_mask;
  auto* sequence = workspace.ledger.sequence_active;
  plan.eigensolver_batch.active = active;
  plan.eigensolver_batch.active_elements = workspace.ledger.batch_elements;
  plan.occupations_batch.active = active;
  plan.occupations_batch.active_elements = workspace.ledger.batch_elements;
  workspace.activity = {active, sequence, workspace.ledger.batch_elements,
                        workspace.ledger.scalar_elements, token};
  workspace.potential_activity = {active, workspace.ledger.batch_elements, token};
  workspace.hamiltonian_activity = {active, workspace.ledger.batch_elements, token};
  workspace.mulliken_activity = {active, workspace.ledger.batch_elements, token};
  workspace.classical_energy_activity = {active, workspace.ledger.batch_elements, token};
  workspace.free_energy_activity = {active, workspace.ledger.batch_elements, token};

  input.activity_state = {state.scc.iterations, state.scc.system_statuses, state.scc.converged,
                          state.scc.batch_elements, token};
  input.mixed_fields = {state.scc.current_inputs.shell_charges,
                        state.scc.current_inputs.shell_elements,
                        state.scc.current_inputs.atomic_dipoles,
                        state.scc.current_inputs.dipole_elements,
                        state.scc.current_inputs.atomic_quadrupoles,
                        state.scc.current_inputs.quadrupole_elements,
                        token};
  input.mixed_spin = {state.scc.current_inputs.shell_charges,
                      state.scc.current_inputs.shell_elements, token};
  workspace.mixed_topology.shell_charges = state.scc.current_inputs.shell_charges;
  workspace.mixed_topology.shell_elements = state.scc.current_inputs.shell_elements;
  workspace.mixed_topology.atomic_dipoles = state.scc.current_inputs.atomic_dipoles;
  workspace.mixed_topology.dipole_elements = state.scc.current_inputs.dipole_elements;
  workspace.mixed_topology.atomic_quadrupoles = state.scc.current_inputs.atomic_quadrupoles;
  workspace.mixed_topology.quadrupole_elements = state.scc.current_inputs.quadrupole_elements;
  workspace.mixed_topology.plan_token = token;
  workspace.physical_topology.plan_token = token;

  /* Component producer storage is projected once into composer inputs. */
  const auto& storage = workspace.components;
  auto& potential = workspace.potential_components;
  potential = {};
  potential.enabled_components = components;
  potential.es2_shell = storage.es2_shell_potential;
  potential.es2_shell_elements = storage.es2_shell_elements;
  potential.es3_shell = storage.es3_shell_potential;
  potential.es3_shell_elements = storage.es3_shell_elements;
  potential.aes2_atomic = storage.aes2_atomic_potential;
  potential.aes2_atomic_elements = storage.aes2_atomic_elements;
  potential.aes2_dipole = storage.aes2_dipole_potential;
  potential.aes2_dipole_elements = storage.aes2_dipole_elements;
  potential.aes2_quadrupole = storage.aes2_quadrupole_potential;
  potential.aes2_quadrupole_elements = storage.aes2_quadrupole_elements;
  if (enabled(components, Gfn2SccPotentialComponent::kD4TwoBody)) {
    potential.d4_atomic = storage.d4_atomic_potential;
    potential.d4_atomic_elements = storage.d4_atomic_elements;
  }
  if (enabled(components, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    potential.explicit_point_charge_shell = plan.explicit_point_charge_cache.shell_potentials;
    potential.explicit_point_charge_shell_elements =
        plan.explicit_point_charge_cache.shell_elements;
  }
  if (enabled(components, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    potential.periodic_atomic = storage.periodic_atomic_potential;
    potential.periodic_atomic_elements = storage.periodic_atomic_elements;
  }
  potential.plan_token = token;

  workspace.scalar_bridge.fields = {
      workspace.complete_potentials.shell, workspace.complete_potentials.shell_elements,
      workspace.complete_potentials.atomic, workspace.complete_potentials.atom_elements, token};
  workspace.scalar_bridge.plan_token = token;
  /* Mixed-spin composition already folds the physical atomic scalar into the
   * charge-channel shell result. Restricted plans retain the legacy bridge
   * descriptor so their arithmetic and compatibility entry remain exact. */
  input.hamiltonian.shell_scalar_potentials =
      mixed_spin ? workspace.complete_potentials.shell : workspace.scalar_bridge.shell_scalar;
  input.hamiltonian.shell_scalar_elements = mixed_spin
                                                ? workspace.complete_potentials.shell_elements
                                                : workspace.scalar_bridge.shell_elements;
  input.hamiltonian.atomic_dipole_potentials = workspace.complete_potentials.dipole;
  input.hamiltonian.atomic_dipole_elements = workspace.complete_potentials.dipole_elements;
  input.hamiltonian.atomic_quadrupole_potentials = workspace.complete_potentials.quadrupole;
  input.hamiltonian.atomic_quadrupole_elements = workspace.complete_potentials.quadrupole_elements;
  input.hamiltonian.plan_token = token;
  input.eigensolver_hamiltonians = workspace.hamiltonian.matrix;
  input.eigensolver_hamiltonian_elements = workspace.hamiltonian.elements;

  input.occupation_eigenvalues = workspace.staged_eigenpairs.eigenvalues;
  input.occupation_eigenvalue_elements = workspace.staged_eigenpairs.eigenvalue_elements;
  input.density = {workspace.staged_eigenpairs.coefficients,
                   workspace.staged_eigenpairs.coefficient_elements,
                   workspace.staged_eigenpairs.eigenvalues,
                   workspace.staged_eigenpairs.eigenvalue_elements,
                   workspace.staged_occupations.occupations,
                   workspace.staged_occupations.occupation_elements,
                   active,
                   workspace.ledger.batch_elements,
                   token};
  input.mulliken = {workspace.staged_density.density,
                    workspace.staged_density.density_elements,
                    input.hamiltonian.overlap,
                    input.hamiltonian.overlap_elements,
                    input.hamiltonian.dipole_integrals,
                    input.hamiltonian.dipole_integral_elements,
                    input.hamiltonian.quadrupole_integrals,
                    input.hamiltonian.quadrupole_integral_elements,
                    token};
  input.electronic_energy = {workspace.staged_density.density,
                             workspace.staged_density.density_elements,
                             input.hamiltonian.h0,
                             input.hamiltonian.h0_elements,
                             workspace.staged_occupations.entropies,
                             workspace.staged_occupations.entropy_elements,
                             token};

  input.classical_energy = {storage.es2_energy,
                            storage.es2_energy_elements,
                            storage.es3_energy,
                            storage.es3_energy_elements,
                            storage.aes2_energy,
                            storage.aes2_energy_elements,
                            nullptr,
                            0,
                            nullptr,
                            0,
                            nullptr,
                            0,
                            token};
  input.free_energy = {storage.core_energy,
                       storage.core_energy_elements,
                       workspace.staged_occupations.entropies,
                       workspace.staged_occupations.entropy_elements,
                       storage.es2_energy,
                       storage.es2_energy_elements,
                       storage.es3_energy,
                       storage.es3_energy_elements,
                       storage.aes2_energy,
                       storage.aes2_energy_elements,
                       workspace.staged_spin_energies,
                       workspace.staged_spin_energy_elements,
                       nullptr,
                       0,
                       nullptr,
                       0,
                       nullptr,
                       0,
                       token};
  if (enabled(components, Gfn2SccPotentialComponent::kD4TwoBody)) {
    input.classical_energy.d4_two_body = storage.d4_two_body_energy;
    input.classical_energy.d4_two_body_elements = storage.d4_two_body_energy_elements;
    input.free_energy.d4_two_body = storage.d4_two_body_energy;
    input.free_energy.d4_two_body_elements = storage.d4_two_body_energy_elements;
  }
  if (enabled(components, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    input.classical_energy.explicit_point_charge = storage.explicit_point_charge_energy;
    input.classical_energy.explicit_point_charge_elements =
        storage.explicit_point_charge_energy_elements;
    input.free_energy.explicit_point_charge = storage.explicit_point_charge_energy;
    input.free_energy.explicit_point_charge_elements =
        storage.explicit_point_charge_energy_elements;
  }
  if (enabled(components, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    input.classical_energy.periodic_embedding = storage.periodic_embedding_energy;
    input.classical_energy.periodic_embedding_elements = storage.periodic_embedding_energy_elements;
    input.free_energy.periodic_embedding = storage.periodic_embedding_energy;
    input.free_energy.periodic_embedding_elements = storage.periodic_embedding_energy_elements;
  }
  input.raw_multipoles = {workspace.staged_raw_population.qsh,
                          workspace.staged_raw_population.qsh_elements,
                          workspace.staged_raw_population.dipole,
                          workspace.staged_raw_population.dipole_elements,
                          workspace.staged_raw_population.quadrupole,
                          workspace.staged_raw_population.quadrupole_elements,
                          token};
  input.raw_spin = {workspace.staged_raw_population.qsh,
                    workspace.staged_raw_population.qsh_elements, token};
  input.complete_free_energies = workspace.staged_free_energy.free_energy;
  input.complete_free_energy_elements = workspace.staged_free_energy.free_energy_elements;
  input.plan_token = token;

  /* Persistent q/d/Q publication and energy components have exact aliases. */
  state.published = {state.raw_population.qsh,
                     state.raw_population.qsh_elements,
                     state.raw_population.dipole,
                     state.raw_population.dipole_elements,
                     state.raw_population.quadrupole,
                     state.raw_population.quadrupole_elements,
                     token};
  state.free_energy.es2 = state.classical_energy.es2;
  state.free_energy.es2_elements = state.classical_energy.es2_elements;
  state.free_energy.es3 = state.classical_energy.es3;
  state.free_energy.es3_elements = state.classical_energy.es3_elements;
  state.free_energy.aes2 = state.classical_energy.aes2;
  state.free_energy.aes2_elements = state.classical_energy.aes2_elements;
  state.free_energy.spin = state.spin_energies;
  state.free_energy.spin_elements = state.spin_energy_elements;
  state.free_energy.d4_two_body = state.classical_energy.d4_two_body;
  state.free_energy.d4_two_body_elements = state.classical_energy.d4_two_body_elements;
  state.free_energy.explicit_point_charge = state.classical_energy.explicit_point_charge;
  state.free_energy.explicit_point_charge_elements =
      state.classical_energy.explicit_point_charge_elements;
  state.free_energy.periodic_embedding = state.classical_energy.periodic_embedding;
  state.free_energy.periodic_embedding_elements =
      state.classical_energy.periodic_embedding_elements;
  state.publication.wavefunction = {state.eigenpairs, state.occupations, state.density,
                                    state.raw_population, token};
  state.publication.energy = {state.classical_energy, state.free_energy, state.spin_energies,
                              state.spin_energy_elements, token};
  state.publication.mixer = state.mixer;
  state.publication.published = state.published;
  state.publication.scc = state.scc;
  state.publication.plan_token = token;
  state.plan_token = token;

  workspace.staged_free_energy.es2 = workspace.staged_classical_energy.es2;
  workspace.staged_free_energy.es2_elements = workspace.staged_classical_energy.es2_elements;
  workspace.staged_free_energy.es3 = workspace.staged_classical_energy.es3;
  workspace.staged_free_energy.es3_elements = workspace.staged_classical_energy.es3_elements;
  workspace.staged_free_energy.aes2 = workspace.staged_classical_energy.aes2;
  workspace.staged_free_energy.aes2_elements = workspace.staged_classical_energy.aes2_elements;
  workspace.staged_free_energy.spin = workspace.staged_spin_energies;
  workspace.staged_free_energy.spin_elements = workspace.staged_spin_energy_elements;
  workspace.staged_free_energy.d4_two_body = workspace.staged_classical_energy.d4_two_body;
  workspace.staged_free_energy.d4_two_body_elements =
      workspace.staged_classical_energy.d4_two_body_elements;
  workspace.staged_free_energy.explicit_point_charge =
      workspace.staged_classical_energy.explicit_point_charge;
  workspace.staged_free_energy.explicit_point_charge_elements =
      workspace.staged_classical_energy.explicit_point_charge_elements;
  workspace.staged_free_energy.periodic_embedding =
      workspace.staged_classical_energy.periodic_embedding;
  workspace.staged_free_energy.periodic_embedding_elements =
      workspace.staged_classical_energy.periodic_embedding_elements;
  workspace.staged_free_energy.entropy = workspace.staged_occupations.entropies;
  workspace.staged_free_energy.entropy_elements = workspace.staged_occupations.entropy_elements;

  workspace.staged_publication.wavefunction = {
      workspace.staged_eigenpairs, workspace.staged_occupations, workspace.staged_density,
      workspace.staged_raw_population, token};
  workspace.staged_publication.energy = {
      workspace.staged_classical_energy, workspace.staged_free_energy,
      workspace.staged_spin_energies, workspace.staged_spin_energy_elements, token};
  workspace.staged_publication.mixer = workspace.staged_mixer;
  workspace.staged_publication.next_mixed = {workspace.next_mixed.shell_charges,
                                             workspace.next_mixed.shell_elements,
                                             workspace.next_mixed.atomic_dipoles,
                                             workspace.next_mixed.dipole_elements,
                                             workspace.next_mixed.atomic_quadrupoles,
                                             workspace.next_mixed.quadrupole_elements,
                                             token};
  workspace.staged_publication.plan_token = token;
  workspace.publication_workspace.mixed_atomic_charges = workspace.mixed_topology.atomic_charges;
  workspace.publication_workspace.mixed_atomic_charge_elements =
      workspace.mixed_topology.atom_elements;
  workspace.eigensolver_workspace.solver_device_workspace =
      plan.eigensolver_provider.device_workspace;
  workspace.eigensolver_workspace.solver_device_workspace_bytes =
      plan.eigensolver_provider.device_workspace_bytes;
  workspace.eigensolver_workspace.solver_host_workspace = plan.eigensolver_provider.host_workspace;
  workspace.eigensolver_workspace.solver_host_workspace_bytes =
      plan.eigensolver_provider.host_workspace_bytes;
  workspace.spin_output.spin_energies = workspace.staged_spin_energies;
  workspace.spin_output.spin_energy_elements = workspace.staged_spin_energy_elements;
  workspace.spin_output.plan_token = token;
  workspace.plan_token = token;
}

void assign_report_owners(Gfn2SccStageDeviceReport& report, std::int64_t index, std::int64_t batch,
                          const Gfn2SccIterationReportStorage& storage,
                          const Gfn2SccIterationDeviceWorkspace& workspace) noexcept {
  report.system_codes = storage_slot(storage.system_errors, index, batch);
  report.system_code_elements = batch;
  report.device_error = storage_slot(storage.device_errors, index, 1);
  report.device_error_elements = 1;
  report.stage_sequence_active = storage_slot(storage.sequence_latches, index, 1);
  report.stage_sequence_elements = 1;

  switch (report.stage) {
    case Gfn2SccStageId::kD4Potential:
    case Gfn2SccStageId::kD4RawEnergy:
      report.system_codes = workspace.d4_workspace.system_errors;
      break;
    case Gfn2SccStageId::kMixer:
      report.system_codes = workspace.staged_mixer.system_statuses;
      report.device_error = nullptr;
      report.device_error_elements = 0;
      break;
    case Gfn2SccStageId::kStatePublication:
      report.system_codes = workspace.publication_workspace.system_errors;
      report.device_error = workspace.publication_workspace.device_error;
      report.stage_sequence_active = workspace.publication_workspace.sequence_active;
      break;
    default:
      break;
  }

  switch (report.stage) {
    case Gfn2SccStageId::kMixedGather:
    case Gfn2SccStageId::kPotentialCompose:
      report.stage_sequence_active = workspace.potential_workspace.sequence_active;
      break;
    case Gfn2SccStageId::kSpinPotential:
    case Gfn2SccStageId::kSpinRawEnergy:
      report.stage_sequence_active = workspace.spin_workspace.sequence_active;
      break;
    case Gfn2SccStageId::kScalarBridge:
      report.stage_sequence_active = workspace.scalar_bridge.workspace.sequence_active;
      break;
    case Gfn2SccStageId::kPeriodicPotential:
    case Gfn2SccStageId::kPeriodicRawEnergy:
      report.stage_sequence_active = workspace.periodic_workspace.sequence_active;
      break;
    case Gfn2SccStageId::kHamiltonian:
      report.stage_sequence_active = workspace.hamiltonian_workspace.sequence_active;
      break;
    case Gfn2SccStageId::kEigensolver:
      report.stage_sequence_active = workspace.eigensolver_workspace.sequence_active;
      break;
    case Gfn2SccStageId::kOccupations:
      report.stage_sequence_active = workspace.occupations_workspace.sequence_active;
      break;
    case Gfn2SccStageId::kDensity:
      report.stage_sequence_active = workspace.density_workspace.sequence_active;
      break;
    case Gfn2SccStageId::kMulliken:
      report.stage_sequence_active = workspace.mulliken_workspace.sequence_active;
      break;
    case Gfn2SccStageId::kClassicalEnergy:
      report.stage_sequence_active = workspace.classical_energy_workspace.sequence_active;
      break;
    case Gfn2SccStageId::kElectronicEnergy:
      report.stage_sequence_active = workspace.electronic_energy_workspace.sequence_active;
      break;
    case Gfn2SccStageId::kFreeEnergy:
      report.stage_sequence_active = workspace.free_energy_workspace.sequence_active;
      break;
    case Gfn2SccStageId::kMixer:
      report.stage_sequence_active = workspace.mixer_workspace.sequence_active;
      break;
    default:
      break;
  }
}

}  // namespace

Gfn2SccIterationBindingDiagnostic query_gfn2_scc_iteration_report_storage_cuda(
    std::uint32_t enabled_components, std::int64_t batch_size,
    Gfn2SccIterationReportStorageRequirements& requirements) noexcept {
  requirements = {};
  if ((enabled_components & ~kGfn2SccPotentialAllComponents) != 0u ||
      (enabled_components & kMandatoryComponents) != kMandatoryComponents || batch_size <= 0) {
    return fail(BindingError::kInvalidCount, BindingField::kStageReports);
  }

  const CanonicalStages stages = canonical_stages(enabled_components);
  if (stages.count < kGfn2SccIterationBaseStageReportCount ||
      stages.count > kGfn2SccIterationMaximumStageReportCount ||
      batch_size > std::numeric_limits<std::int64_t>::max() / stages.count) {
    return fail(BindingError::kAddressOverflow, BindingField::kStageReports);
  }
  requirements.report_count = stages.count;
  requirements.system_error_elements = stages.count * batch_size;
  requirements.device_error_elements = stages.count;
  requirements.sequence_latch_elements = stages.count;
  return {};
}

Gfn2SccIterationBindingDiagnostic project_gfn2_scc_iteration_reports_cuda(
    const Gfn2SccIterationReportStorage& report_storage,
    const Gfn2SccIterationDevicePlan& plan_seed, const Gfn2SccIterationDeviceInput& input_seed,
    const Gfn2SccIterationDeviceState& state_seed,
    const Gfn2SccIterationDeviceWorkspace& workspace_seed,
    Gfn2SccIterationProjectedDescriptors& projected) noexcept {
  projected = {};
  Gfn2SccIterationReportStorageRequirements requirements{};
  BindingDiagnostic diagnostic = query_gfn2_scc_iteration_report_storage_cuda(
      plan_seed.enabled_components, plan_seed.topology.batch_size, requirements);
  if (diagnostic.error != BindingError::kSuccess) return diagnostic;
  if (report_storage.plan_token != plan_seed.plan_token) {
    return fail(BindingError::kCrossPlan, BindingField::kStageReports);
  }
  diagnostic = validate_storage_range(
      report_storage.system_errors, report_storage.system_error_elements,
      requirements.system_error_elements, sizeof(std::uint32_t), alignof(std::uint32_t), 0);
  if (diagnostic.error != BindingError::kSuccess) return diagnostic;
  diagnostic = validate_storage_range(
      report_storage.device_errors, report_storage.device_error_elements,
      requirements.device_error_elements, sizeof(std::uint32_t), alignof(std::uint32_t), 1);
  if (diagnostic.error != BindingError::kSuccess) return diagnostic;
  diagnostic = validate_storage_range(
      report_storage.sequence_latches, report_storage.sequence_latch_elements,
      requirements.sequence_latch_elements, sizeof(std::uint32_t), alignof(std::uint32_t), 2);
  if (diagnostic.error != BindingError::kSuccess) return diagnostic;

  Gfn2SccIterationProjectedDescriptors candidate{plan_seed, input_seed, state_seed, workspace_seed};
  project_zero_copy_views(candidate);
  for (auto& report : candidate.plan.reports) report = {};
  const CanonicalStages stages = canonical_stages(candidate.plan.enabled_components);
  candidate.plan.report_count = stages.count;
  for (std::int64_t index = 0; index < stages.count; ++index) {
    ReportSpec spec{};
    if (!report_spec(stages.values[static_cast<std::size_t>(index)], spec)) {
      return fail(BindingError::kInvalidStageReport, BindingField::kStageReports, index);
    }
    auto& report = candidate.plan.reports[index];
    report.stage = stages.values[static_cast<std::size_t>(index)];
    report.system_code_format = spec.format;
    report.device_code_role = spec.role;
    report.peer_error_mask = spec.peer_mask;
    report.peer_failure_status = spec.peer_status;
    report.plan_token = candidate.plan.plan_token;
    assign_report_owners(report, index, candidate.plan.topology.batch_size, report_storage,
                         candidate.workspace);

    const bool mixer = report.stage == Gfn2SccStageId::kMixer;
    const std::size_t system_alignment = mixer ? alignof(gpuxtb_status_t) : alignof(std::uint32_t);
    if (report.system_codes == nullptr || (!mixer && report.device_error == nullptr) ||
        report.stage_sequence_active == nullptr) {
      return fail(BindingError::kNullPointer, BindingField::kStageReports, index);
    }
    if (reinterpret_cast<std::uintptr_t>(report.system_codes) % system_alignment != 0u ||
        (!mixer &&
         reinterpret_cast<std::uintptr_t>(report.device_error) % alignof(std::uint32_t) != 0u) ||
        reinterpret_cast<std::uintptr_t>(report.stage_sequence_active) % alignof(std::uint32_t) !=
            0u) {
      return fail(BindingError::kMisalignedPointer, BindingField::kStageReports, index);
    }
  }
  projected = candidate;
  return {};
}

Gfn2SccIterationBindingDiagnostic build_gfn2_scc_iteration_report_binding_cuda(
    const Gfn2SccIterationReportStorage& report_storage,
    const Gfn2SccIterationDevicePlan& plan_seed, const Gfn2SccIterationDeviceInput& input_seed,
    const Gfn2SccIterationDeviceState& state_seed,
    const Gfn2SccIterationDeviceWorkspace& workspace_seed,
    Gfn2SccIterationBinding& binding) noexcept {
  binding = {};
  Gfn2SccIterationProjectedDescriptors projected{};
  const BindingDiagnostic diagnostic = project_gfn2_scc_iteration_reports_cuda(
      report_storage, plan_seed, input_seed, state_seed, workspace_seed, projected);
  if (diagnostic.error != BindingError::kSuccess) return diagnostic;
  return bind_gfn2_scc_iteration_cuda(projected.plan, projected.input, projected.state,
                                      projected.workspace, binding);
}

}  // namespace gpuxtb::detail::cuda
