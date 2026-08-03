#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "backends/cuda/gfn2_scc_iteration_reports.cuh"

namespace {

using namespace gpuxtb::detail;
using namespace gpuxtb::detail::cuda;

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

constexpr std::uint32_t bit(Gfn2SccPotentialComponent component) noexcept {
  return static_cast<std::uint32_t>(component);
}

constexpr std::uint32_t kMandatory = bit(Gfn2SccPotentialComponent::kES2) |
                                     bit(Gfn2SccPotentialComponent::kES3) |
                                     bit(Gfn2SccPotentialComponent::kAES2);

/*
 * Projection is a synchronous descriptor operation, so focused tests use
 * aligned non-dereferenceable UVA-like addresses rather than allocating a GPU.
 * The production validator likewise promises not to inspect device contents.
 */
struct FakeDeviceArena {
  std::uintptr_t cursor = 0x400000000ULL;

  template <typename T>
  T* allocate(std::int64_t elements) noexcept {
    if (elements <= 0) return nullptr;
    constexpr std::uintptr_t alignment = alignof(T) > 64u ? alignof(T) : 64u;
    cursor = (cursor + alignment - 1u) & ~(alignment - 1u);
    T* result = reinterpret_cast<T*>(cursor);
    cursor += static_cast<std::uintptr_t>(elements) * sizeof(T) + 64u;
    return result;
  }

  void* allocate_bytes(std::size_t bytes) noexcept {
    if (bytes == 0u) return nullptr;
    cursor = (cursor + 255u) & ~std::uintptr_t{255u};
    void* result = reinterpret_cast<void*>(cursor);
    cursor += bytes + 256u;
    return result;
  }
};

struct Fixture {
  static constexpr std::uint64_t kToken = 0x10300c0deULL;
  static constexpr std::int64_t kBatch = 2;
  static constexpr std::int64_t kAtoms = 3;
  static constexpr std::int64_t kShells = 4;
  static constexpr std::int64_t kOrbitals = 5;
  static constexpr std::int64_t kMatrices = 13;
  static constexpr std::int64_t kDipoles = 3 * kAtoms;
  static constexpr std::int64_t kQuadrupoles = 6 * kAtoms;

  FakeDeviceArena arena;
  Gfn2SccIterationDevicePlan plan{};
  Gfn2SccIterationDeviceInput input{};
  Gfn2SccIterationDeviceState state{};
  Gfn2SccIterationDeviceWorkspace workspace{};
  Gfn2SccIterationReportStorage storage{};

  template <typename T>
  T* ptr(std::int64_t elements) noexcept {
    return arena.allocate<T>(elements);
  }

  Gfn2EigensolverDeviceResults eigenpairs() noexcept {
    return {ptr<double>(kOrbitals), kOrbitals, ptr<double>(kMatrices), kMatrices, kToken};
  }

  Gfn2OccupationsDeviceResults occupations() noexcept {
    return {ptr<double>(2 * kOrbitals),
            2 * kOrbitals,
            ptr<double>(2 * kBatch),
            2 * kBatch,
            ptr<double>(2 * kBatch),
            2 * kBatch,
            ptr<double>(kBatch),
            kBatch,
            kToken};
  }

  Gfn2DensityDeviceResults density() noexcept {
    return {ptr<double>(kMatrices),
            kMatrices,
            ptr<double>(kMatrices),
            kMatrices,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            kToken};
  }

  Gfn2MullikenDevicePopulation population() noexcept {
    return {ptr<double>(kShells),
            kShells,
            ptr<double>(kAtoms),
            kAtoms,
            ptr<double>(kDipoles),
            kDipoles,
            ptr<double>(kQuadrupoles),
            kQuadrupoles,
            kToken};
  }

  Gfn2SccDeviceMultipoles multipoles() noexcept {
    return {
        ptr<double>(kShells), kShells, ptr<double>(kDipoles), kDipoles, ptr<double>(kQuadrupoles),
        kQuadrupoles,         kToken};
  }

  Gfn2SccClassicalEnergyDeviceDiagnostics classical() noexcept {
    return {ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            kToken};
  }

  Gfn2SccFreeEnergyDeviceDiagnostics free_energy() noexcept {
    return {ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            kToken};
  }

  void make_report_owners() noexcept {
    workspace.d4_workspace.system_errors = ptr<std::uint32_t>(kBatch);
    workspace.d4_workspace.system_error_elements = kBatch;
    workspace.staged_mixer.system_statuses = ptr<gpuxtb_status_t>(kBatch);
    workspace.staged_mixer.batch_elements = kBatch;

    workspace.publication_workspace.system_errors = ptr<std::uint32_t>(kBatch);
    workspace.publication_workspace.system_error_elements = kBatch;
    workspace.publication_workspace.device_error = ptr<std::uint32_t>(1);
    workspace.publication_workspace.device_error_elements = 1;
    workspace.publication_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.publication_workspace.sequence_elements = 1;

    workspace.potential_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.scalar_bridge.workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.periodic_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.hamiltonian_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.eigensolver_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.occupations_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.density_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.mulliken_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.classical_energy_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.electronic_energy_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.free_energy_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.mixer_workspace.sequence_active = ptr<std::uint32_t>(1);

    storage.system_errors = ptr<std::uint32_t>(kGfn2SccIterationMaximumStageReportCount * kBatch);
    storage.system_error_elements = kGfn2SccIterationMaximumStageReportCount * kBatch;
    storage.device_errors = ptr<std::uint32_t>(kGfn2SccIterationMaximumStageReportCount);
    storage.device_error_elements = kGfn2SccIterationMaximumStageReportCount;
    storage.sequence_latches = ptr<std::uint32_t>(kGfn2SccIterationMaximumStageReportCount);
    storage.sequence_latch_elements = kGfn2SccIterationMaximumStageReportCount;
    storage.plan_token = kToken;
  }

  void make_projection_leaves() noexcept {
    workspace.ledger.active_mask = ptr<std::uint8_t>(kBatch);
    workspace.ledger.sequence_active = ptr<std::uint32_t>(1);
    workspace.ledger.batch_elements = kBatch;
    workspace.ledger.scalar_elements = 1;

    state.eigenpairs = eigenpairs();
    state.occupations = occupations();
    state.density = density();
    state.raw_population = population();
    state.classical_energy = classical();
    state.free_energy = free_energy();
    state.scc.current_inputs = multipoles();
    state.scc.iterations = ptr<std::uint64_t>(kBatch);
    state.scc.system_statuses = ptr<gpuxtb_status_t>(kBatch);
    state.scc.converged = ptr<std::uint8_t>(kBatch);
    state.scc.batch_elements = kBatch;

    workspace.staged_eigenpairs = eigenpairs();
    workspace.staged_occupations = occupations();
    workspace.staged_density = density();
    workspace.staged_raw_population = population();
    workspace.staged_classical_energy = classical();
    workspace.staged_free_energy = free_energy();
    workspace.next_mixed = multipoles();
    workspace.mixed_topology.atomic_charges = ptr<double>(kAtoms);
    workspace.mixed_topology.atom_elements = kAtoms;

    auto& produced = workspace.components;
    produced.es2_shell_potential = ptr<double>(kShells);
    produced.es2_shell_elements = kShells;
    produced.es3_shell_potential = ptr<double>(kShells);
    produced.es3_shell_elements = kShells;
    produced.aes2_atomic_potential = ptr<double>(kAtoms);
    produced.aes2_atomic_elements = kAtoms;
    produced.aes2_dipole_potential = ptr<double>(kDipoles);
    produced.aes2_dipole_elements = kDipoles;
    produced.aes2_quadrupole_potential = ptr<double>(kQuadrupoles);
    produced.aes2_quadrupole_elements = kQuadrupoles;
    produced.d4_atomic_potential = ptr<double>(kAtoms);
    produced.d4_atomic_elements = kAtoms;
    produced.periodic_atomic_potential = ptr<double>(kAtoms);
    produced.periodic_atomic_elements = kAtoms;
    produced.es2_energy = ptr<double>(kBatch);
    produced.es2_energy_elements = kBatch;
    produced.es3_energy = ptr<double>(kBatch);
    produced.es3_energy_elements = kBatch;
    produced.aes2_energy = ptr<double>(kBatch);
    produced.aes2_energy_elements = kBatch;
    produced.d4_two_body_energy = ptr<double>(kBatch);
    produced.d4_two_body_energy_elements = kBatch;
    produced.explicit_point_charge_energy = ptr<double>(kBatch);
    produced.explicit_point_charge_energy_elements = kBatch;
    produced.periodic_embedding_energy = ptr<double>(kBatch);
    produced.periodic_embedding_energy_elements = kBatch;
    produced.core_energy = ptr<double>(kBatch);
    produced.core_energy_elements = kBatch;

    workspace.complete_potentials = {ptr<double>(kShells),
                                     kShells,
                                     ptr<double>(kAtoms),
                                     kAtoms,
                                     ptr<double>(kDipoles),
                                     kDipoles,
                                     ptr<double>(kQuadrupoles),
                                     kQuadrupoles,
                                     kToken};
    workspace.scalar_bridge.shell_scalar = ptr<double>(kShells);
    workspace.scalar_bridge.shell_elements = kShells;
    workspace.hamiltonian = {ptr<double>(kMatrices), kMatrices, kToken};

    input.hamiltonian.h0 = ptr<double>(kMatrices);
    input.hamiltonian.h0_elements = kMatrices;
    input.hamiltonian.overlap = ptr<double>(kMatrices);
    input.hamiltonian.overlap_elements = kMatrices;
    input.hamiltonian.dipole_integrals = ptr<double>(3 * kMatrices);
    input.hamiltonian.dipole_integral_elements = 3 * kMatrices;
    input.hamiltonian.quadrupole_integrals = ptr<double>(6 * kMatrices);
    input.hamiltonian.quadrupole_integral_elements = 6 * kMatrices;

    plan.explicit_point_charge_cache.shell_potentials = ptr<double>(kShells);
    plan.explicit_point_charge_cache.shell_elements = kShells;
    plan.eigensolver_provider.device_workspace = arena.allocate_bytes(512u);
    plan.eigensolver_provider.device_workspace_bytes = 512u;
    plan.eigensolver_provider.host_workspace = arena.allocate_bytes(256u);
    plan.eigensolver_provider.host_workspace_bytes = 256u;
  }

  Fixture() noexcept {
    plan.plan_token = kToken;
    plan.topology.batch_size = kBatch;
    plan.enabled_components = kGfn2SccPotentialAllComponents;
    make_report_owners();
    make_projection_leaves();
  }
};

const Gfn2SccStageDeviceReport* find_report(const Gfn2SccIterationDevicePlan& plan,
                                            Gfn2SccStageId stage) noexcept {
  for (std::int64_t index = 0; index < plan.report_count; ++index) {
    if (plan.reports[index].stage == stage) return &plan.reports[index];
  }
  return nullptr;
}

int test_storage_requirements_are_exact_and_fail_closed() {
  Gfn2SccIterationReportStorageRequirements requirements{};
  auto diagnostic = query_gfn2_scc_iteration_report_storage_cuda(kMandatory, 7, requirements);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kSuccess);
  CHECK(requirements.report_count == 19);
  CHECK(requirements.system_error_elements == 19 * 7);
  CHECK(requirements.device_error_elements == 19);
  CHECK(requirements.sequence_latch_elements == 19);

  diagnostic = query_gfn2_scc_iteration_report_storage_cuda(
      kMandatory | bit(Gfn2SccPotentialComponent::kD4TwoBody), 7, requirements);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kSuccess);
  CHECK(requirements.report_count == 21);
  diagnostic = query_gfn2_scc_iteration_report_storage_cuda(
      kMandatory | bit(Gfn2SccPotentialComponent::kExplicitPointCharge), 7, requirements);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kSuccess);
  CHECK(requirements.report_count == 20);
  diagnostic = query_gfn2_scc_iteration_report_storage_cuda(
      kMandatory | bit(Gfn2SccPotentialComponent::kPeriodicEmbedding), 7, requirements);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kSuccess);
  CHECK(requirements.report_count == 21);
  diagnostic =
      query_gfn2_scc_iteration_report_storage_cuda(kGfn2SccPotentialAllComponents, 7, requirements);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kSuccess);
  CHECK(requirements.report_count == 24);

  diagnostic = query_gfn2_scc_iteration_report_storage_cuda(0u, 7, requirements);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidCount);
  CHECK(requirements.report_count == 0);
  diagnostic = query_gfn2_scc_iteration_report_storage_cuda(
      kMandatory, std::numeric_limits<std::int64_t>::max() / 19 + 1, requirements);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kAddressOverflow);
  CHECK(requirements.system_error_elements == 0);
  return 0;
}

int test_canonical_stage_factory_and_diagnostic_owners() {
  Fixture fixture;
  Gfn2SccIterationProjectedDescriptors projected{};
  const auto diagnostic = project_gfn2_scc_iteration_reports_cuda(
      fixture.storage, fixture.plan, fixture.input, fixture.state, fixture.workspace, projected);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kSuccess);

  constexpr std::array<Gfn2SccStageId, 24> expected{{
      Gfn2SccStageId::kMixedGather,
      Gfn2SccStageId::kES2Potential,
      Gfn2SccStageId::kES3Potential,
      Gfn2SccStageId::kAES2Potential,
      Gfn2SccStageId::kD4Potential,
      Gfn2SccStageId::kPeriodicPotential,
      Gfn2SccStageId::kPotentialCompose,
      Gfn2SccStageId::kScalarBridge,
      Gfn2SccStageId::kHamiltonian,
      Gfn2SccStageId::kEigensolver,
      Gfn2SccStageId::kOccupations,
      Gfn2SccStageId::kDensity,
      Gfn2SccStageId::kMulliken,
      Gfn2SccStageId::kES2RawEnergy,
      Gfn2SccStageId::kES3RawEnergy,
      Gfn2SccStageId::kAES2RawEnergy,
      Gfn2SccStageId::kD4RawEnergy,
      Gfn2SccStageId::kExplicitPointChargeRawEnergy,
      Gfn2SccStageId::kPeriodicRawEnergy,
      Gfn2SccStageId::kClassicalEnergy,
      Gfn2SccStageId::kElectronicEnergy,
      Gfn2SccStageId::kFreeEnergy,
      Gfn2SccStageId::kMixer,
      Gfn2SccStageId::kStatePublication,
  }};
  CHECK(projected.plan.report_count == static_cast<std::int64_t>(expected.size()));
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto& report = projected.plan.reports[index];
    CHECK(report.stage == expected[index]);
    CHECK(report.plan_token == Fixture::kToken);
    CHECK(report.system_code_elements == Fixture::kBatch);
    CHECK(report.stage_sequence_elements == 1);
  }

  const auto* d4_potential = find_report(projected.plan, Gfn2SccStageId::kD4Potential);
  const auto* d4_energy = find_report(projected.plan, Gfn2SccStageId::kD4RawEnergy);
  const auto* periodic = find_report(projected.plan, Gfn2SccStageId::kPeriodicPotential);
  const auto* mixer = find_report(projected.plan, Gfn2SccStageId::kMixer);
  const auto* publication = find_report(projected.plan, Gfn2SccStageId::kStatePublication);
  const auto* eigensolver = find_report(projected.plan, Gfn2SccStageId::kEigensolver);
  CHECK(d4_potential != nullptr && d4_energy != nullptr && periodic != nullptr &&
        mixer != nullptr && publication != nullptr && eigensolver != nullptr);
  CHECK(d4_potential->system_codes == fixture.workspace.d4_workspace.system_errors);
  CHECK(d4_energy->system_codes == fixture.workspace.d4_workspace.system_errors);
  CHECK(periodic->stage_sequence_active == fixture.workspace.periodic_workspace.sequence_active);
  CHECK(mixer->system_code_format == Gfn2SccStageCodeFormat::kGpuxtbStatus);
  CHECK(mixer->system_codes == fixture.workspace.staged_mixer.system_statuses);
  CHECK(mixer->device_error == nullptr && mixer->device_error_elements == 0);
  CHECK(publication->system_codes == fixture.workspace.publication_workspace.system_errors);
  CHECK(publication->device_error == fixture.workspace.publication_workspace.device_error);
  CHECK(publication->stage_sequence_active ==
        fixture.workspace.publication_workspace.sequence_active);
  CHECK(eigensolver->peer_failure_status == GPUXTB_STATUS_EIGENSOLVER_FAILED);
  CHECK(eigensolver->peer_error_mask == 0x3e06u);
  return 0;
}

int test_zero_copy_projection_rebuilds_all_dataflow_edges() {
  Fixture fixture;
  Gfn2SccIterationProjectedDescriptors projected{};
  const auto diagnostic = project_gfn2_scc_iteration_reports_cuda(
      fixture.storage, fixture.plan, fixture.input, fixture.state, fixture.workspace, projected);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kSuccess);

  const auto& plan = projected.plan;
  const auto& input = projected.input;
  const auto& state = projected.state;
  const auto& workspace = projected.workspace;
  CHECK(plan.eigensolver_batch.active == workspace.ledger.active_mask);
  CHECK(plan.occupations_batch.active == workspace.ledger.active_mask);
  CHECK(workspace.activity.sequence_active == workspace.ledger.sequence_active);
  CHECK(input.activity_state.iterations == state.scc.iterations);
  CHECK(input.mixed_fields.qsh == state.scc.current_inputs.shell_charges);
  CHECK(workspace.mixed_topology.atomic_dipoles == state.scc.current_inputs.atomic_dipoles);

  CHECK(workspace.potential_components.es2_shell == workspace.components.es2_shell_potential);
  CHECK(workspace.potential_components.explicit_point_charge_shell ==
        plan.explicit_point_charge_cache.shell_potentials);
  CHECK(workspace.potential_components.d4_atomic == workspace.components.d4_atomic_potential);
  CHECK(workspace.scalar_bridge.fields.shell == workspace.complete_potentials.shell);
  CHECK(input.hamiltonian.shell_scalar_potentials == workspace.scalar_bridge.shell_scalar);
  CHECK(input.eigensolver_hamiltonians == workspace.hamiltonian.matrix);
  CHECK(input.density.coefficients == workspace.staged_eigenpairs.coefficients);
  CHECK(input.density.occupations == workspace.staged_occupations.occupations);
  CHECK(input.mulliken.density == workspace.staged_density.density);
  CHECK(input.electronic_energy.entropies == workspace.staged_occupations.entropies);
  CHECK(input.classical_energy.d4_two_body == workspace.components.d4_two_body_energy);
  CHECK(input.free_energy.periodic_embedding == workspace.components.periodic_embedding_energy);
  CHECK(input.raw_multipoles.shell_charges == workspace.staged_raw_population.qsh);
  CHECK(input.complete_free_energies == workspace.staged_free_energy.free_energy);

  CHECK(state.published.shell_charges == state.raw_population.qsh);
  CHECK(state.free_energy.es2 == state.classical_energy.es2);
  CHECK(state.publication.wavefunction.population.qat == state.raw_population.qat);
  CHECK(state.publication.scc.iterations == state.scc.iterations);
  CHECK(workspace.staged_free_energy.entropy == workspace.staged_occupations.entropies);
  CHECK(workspace.staged_free_energy.es3 == workspace.staged_classical_energy.es3);
  CHECK(workspace.staged_publication.wavefunction.density.density ==
        workspace.staged_density.density);
  CHECK(workspace.staged_publication.next_mixed.atomic_quadrupoles ==
        workspace.next_mixed.atomic_quadrupoles);
  CHECK(workspace.publication_workspace.mixed_atomic_charges ==
        workspace.mixed_topology.atomic_charges);
  CHECK(workspace.eigensolver_workspace.solver_device_workspace ==
        plan.eigensolver_provider.device_workspace);
  CHECK(workspace.eigensolver_workspace.solver_host_workspace ==
        plan.eigensolver_provider.host_workspace);
  return 0;
}

int test_disabled_components_are_canonical_null_and_failures_clear_outputs() {
  Fixture fixture;
  fixture.plan.enabled_components = kMandatory;
  Gfn2SccIterationProjectedDescriptors projected{};
  auto diagnostic = project_gfn2_scc_iteration_reports_cuda(
      fixture.storage, fixture.plan, fixture.input, fixture.state, fixture.workspace, projected);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kSuccess);
  CHECK(projected.plan.report_count == kGfn2SccIterationBaseStageReportCount);
  CHECK(projected.workspace.potential_components.d4_atomic == nullptr);
  CHECK(projected.workspace.potential_components.explicit_point_charge_shell == nullptr);
  CHECK(projected.workspace.potential_components.periodic_atomic == nullptr);
  CHECK(projected.input.classical_energy.d4_two_body == nullptr);
  CHECK(projected.input.free_energy.explicit_point_charge == nullptr);
  CHECK(find_report(projected.plan, Gfn2SccStageId::kD4Potential) == nullptr);

  fixture.storage.plan_token ^= 1u;
  projected.plan.plan_token = 0xdeadbeefu;
  diagnostic = project_gfn2_scc_iteration_reports_cuda(fixture.storage, fixture.plan, fixture.input,
                                                       fixture.state, fixture.workspace, projected);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kCrossPlan);
  CHECK(projected.plan.plan_token == 0u);

  fixture.storage.plan_token = Fixture::kToken;
  fixture.storage.system_error_elements = 1;
  diagnostic = project_gfn2_scc_iteration_reports_cuda(fixture.storage, fixture.plan, fixture.input,
                                                       fixture.state, fixture.workspace, projected);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInsufficientCapacity);
  CHECK(projected.plan.plan_token == 0u);

  fixture.storage.system_error_elements =
      kGfn2SccIterationMaximumStageReportCount * Fixture::kBatch;
  fixture.plan.abi_version = 0u;
  Gfn2SccIterationBinding binding{};
  binding.plan.plan_token = 0xdeadbeefu;
  diagnostic = build_gfn2_scc_iteration_report_binding_cuda(
      fixture.storage, fixture.plan, fixture.input, fixture.state, fixture.workspace, binding);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidAbiVersion);
  CHECK(binding.plan.plan_token == 0u);
  CHECK(binding.input.hamiltonian.h0 == nullptr);
  return 0;
}

}  // namespace

int main() {
  const std::array<int (*)(), 4> tests{{
      test_storage_requirements_are_exact_and_fail_closed,
      test_canonical_stage_factory_and_diagnostic_owners,
      test_zero_copy_projection_rebuilds_all_dataflow_edges,
      test_disabled_components_are_canonical_null_and_failures_clear_outputs,
  }};
  for (const auto test : tests) {
    if (const int line = test(); line != 0) {
      std::fprintf(stderr, "CUDA SCC iteration report test failed at line %d\n", line);
      return 1;
    }
  }
  return 0;
}
