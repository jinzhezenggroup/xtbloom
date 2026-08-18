// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_es3.cuh"
#include "backends/cuda/gfn2_scc_iteration_initialize.cuh"
#include "backends/cuda/gfn2_scc_setup_inputs.cuh"
#include "backends/cuda/gfn2_scc_setup_topology.hpp"
#include "model/gfn1/coordination.hpp"

namespace {

using namespace xtbloom::detail;
using namespace xtbloom::detail::cuda;

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

constexpr std::uint64_t kPlanToken = 0x38643901ULL;
constexpr std::int64_t kBatch = 1;
constexpr std::int64_t kAtoms = 2;
constexpr std::int64_t kShells = 4;
constexpr std::int64_t kOrbitals = 4;
constexpr std::int64_t kMatrices = 16;
constexpr std::int64_t kHistory = 2;
constexpr std::uint32_t kGfn1Components =
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3);

template <typename T>
Gfn2SccIterationHostArrayView<T> host_view(const std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), static_cast<std::int64_t>(values.size())};
}

template <typename T>
Gfn2SccSetupHostArray<T> setup_view(const std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), static_cast<std::int64_t>(values.size())};
}

template <typename T>
const T* fake_device_pointer(std::uintptr_t address) noexcept {
  return reinterpret_cast<const T*>(address);
}

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  explicit DeviceAllocation(std::size_t bytes) { (void)allocate(bytes); }
  ~DeviceAllocation() {
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  bool allocate(std::size_t bytes) {
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
    pointer_ = nullptr;
    bytes_ = bytes;
    return cudaMalloc(&pointer_, std::max<std::size_t>(bytes, 1u)) == cudaSuccess;
  }

  [[nodiscard]] void* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

template <typename T>
class DeviceVector {
 public:
  DeviceVector() = default;
  explicit DeviceVector(const std::vector<T>& values) { (void)assign(values); }

  bool assign(const std::vector<T>& values) {
    if (!storage_.allocate(values.size() * sizeof(T))) return false;
    elements_ = values.size();
    return values.empty() || cudaMemcpy(storage_.get(), values.data(), values.size() * sizeof(T),
                                        cudaMemcpyHostToDevice) == cudaSuccess;
  }

  bool copy_to(std::vector<T>& values) const {
    values.resize(elements_);
    return values.empty() || cudaMemcpy(values.data(), storage_.get(), values.size() * sizeof(T),
                                        cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  [[nodiscard]] T* get() const noexcept { return static_cast<T*>(storage_.get()); }
  [[nodiscard]] std::size_t size() const noexcept { return elements_; }

 private:
  DeviceAllocation storage_;
  std::size_t elements_ = 0u;
};

Gfn2SccIterationDevicePlan make_scalar_plan() {
  Gfn2SccIterationDevicePlan plan{};
  plan.abi_version = kGfn2SccIterationAbiVersion;
  plan.enabled_components = kGfn1Components;
  plan.model = XtbModelFlavor::kGfn1;
  plan.plan_token = kPlanToken;
  plan.geometry_generation = 1u;

  plan.topology.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  plan.topology.plan_token = kPlanToken;
  plan.topology.batch_size = kBatch;
  plan.topology.bucket_count = 1;
  plan.topology.total_atoms = kAtoms;
  plan.topology.total_shells = kShells;
  plan.topology.total_orbitals = kOrbitals;
  plan.topology.total_matrix_elements = kMatrices;

  plan.wavefunction_layout.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  plan.wavefunction_layout.plan_token = kPlanToken;
  plan.wavefunction_layout.layout_fingerprint = 0x38643902ULL;
  plan.wavefunction_layout.batch_size = kBatch;
  plan.wavefunction_layout.total_spin_channels = kBatch;
  plan.wavefunction_layout.total_spin_orbitals = kOrbitals;
  plan.wavefunction_layout.total_spin_matrix_elements = kMatrices;
  plan.wavefunction_layout.total_spin_shells = kShells;
  plan.wavefunction_layout.total_spin_atoms = kAtoms;
  plan.wavefunction_layout.spin_channel_count = kBatch;
  plan.wavefunction_layout.spin_channel_offset_count = kBatch + 1;
  plan.wavefunction_layout.spin_orbital_offset_count = kBatch + 1;
  plan.wavefunction_layout.spin_matrix_offset_count = kBatch + 1;
  plan.wavefunction_layout.spin_shell_offset_count = kBatch + 1;
  plan.wavefunction_layout.spin_atom_offset_count = kBatch + 1;
  plan.wavefunction_layout.spin_channels = fake_device_pointer<std::int32_t>(0x10000000u);
  plan.wavefunction_layout.spin_channel_offsets = fake_device_pointer<std::int64_t>(0x10000100u);
  plan.wavefunction_layout.spin_orbital_offsets = fake_device_pointer<std::int64_t>(0x10000200u);
  plan.wavefunction_layout.spin_matrix_offsets = fake_device_pointer<std::int64_t>(0x10000300u);
  plan.wavefunction_layout.spin_shell_offsets = fake_device_pointer<std::int64_t>(0x10000400u);
  plan.wavefunction_layout.spin_atom_offsets = fake_device_pointer<std::int64_t>(0x10000500u);

  plan.spin_batch.batch_size = kBatch;
  plan.spin_batch.total_atoms = kAtoms;
  plan.spin_batch.total_shells = kShells;
  plan.spin_batch.shell_population_elements = kShells;
  plan.spin_batch.plan_token = kPlanToken;
  plan.mixer_policy.history_size = kHistory;
  plan.mixer_policy.atomic_multipole_components = 0;
  plan.mixer_policy.plan_token = kPlanToken;
  plan.state_policy.maximum_iterations = 8u;
  plan.geometry_batch.total_pairs = 1;
  plan.geometry_batch.model = XtbModelFlavor::kGfn1;
  plan.es2_batch.total_matrix_elements = 10;
  plan.es3_batch.model = XtbModelFlavor::kGfn1;

  /* Deliberately nonzero: a disabled GFN1 AES2 leaf must not affect the arena. */
  plan.aes2_batch.total_pairs = 17;
  return plan;
}

struct BoundScalarArena {
  Gfn2SccIterationDevicePlan plan = make_scalar_plan();
  Gfn2SccIterationArenaRequirements requirements{};
  DeviceAllocation arena;
  Gfn2SccIterationDeviceState state{};
  Gfn2SccIterationDeviceWorkspace workspace{};
  Gfn2SccIterationReportStorage reports{};
  bool valid = false;

  BoundScalarArena() {
    const auto query = query_gfn2_scc_iteration_arena_requirements_cuda(
        plan, plan.eigensolver_provider.requirements, requirements);
    if (!query.success() || !arena.allocate(requirements.total_bytes)) return;
    valid = bind_gfn2_scc_iteration_arena_cuda(plan, plan.eigensolver_provider.requirements,
                                               requirements, arena.get(), arena.bytes(), nullptr,
                                               0u, state, workspace, reports)
                .success();
  }
};

struct FreshScalarState {
  std::vector<std::int64_t> atom_offsets{0, kAtoms};
  std::vector<std::int64_t> shell_offsets{0, kShells};
  std::vector<double> qsh{0.10, -0.20, 0.30, -0.40};
  std::vector<double> qat{0.20, -0.20};

  [[nodiscard]] Gfn2SccIterationHostInitialization view(std::uint64_t generation = 3u) const {
    Gfn2SccIterationHostInitialization host{};
    host.mode = Gfn2SccIterationInitializationMode::kFresh;
    host.plan_token = kPlanToken;
    host.initialization_generation = generation;
    host.topology = {host_view(atom_offsets), host_view(shell_offsets), kPlanToken};
    host.wavefunction.plan_token = kPlanToken;
    host.wavefunction.population.shell_charges = host_view(qsh);
    host.wavefunction.population.atomic_charges = host_view(qat);
    host.wavefunction.population.plan_token = kPlanToken;
    return host;
  }
};

struct WarmScalarState {
  std::vector<std::int64_t> atom_offsets{0, kAtoms};
  std::vector<std::int64_t> shell_offsets{0, kShells};
  std::vector<double> eigenvalues{-0.8, -0.3, 0.2, 0.7};
  std::vector<double> coefficients;
  std::vector<double> occupations{1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
  std::vector<double> chemical_potentials{-0.1, -0.1};
  std::vector<double> electron_sums{2.0, 2.0};
  std::vector<double> entropy{0.02};
  std::vector<double> density;
  std::vector<double> weighted_density;
  std::vector<double> band_energy{-1.1};
  std::vector<double> occupation_sum{4.0};
  std::vector<double> density_trace{4.0};
  std::vector<double> weighted_density_trace{-0.7};
  std::vector<double> qsh{0.15, -0.05, 0.20, -0.30};
  std::vector<double> qat{0.10, -0.10};

  std::vector<double> core{-1.2};
  std::vector<double> es2{0.20};
  std::vector<double> es3{0.03};
  std::vector<double> internal{-0.97};
  std::vector<double> free{-0.99};
  std::vector<double> classical_total{0.23};

  std::vector<double> previous_inputs;
  std::vector<double> previous_residuals;
  std::vector<double> df_history;
  std::vector<double> u_history;
  std::vector<double> omega{1.0, 0.5};
  std::vector<double> residual_rms{1.0e-9};
  std::vector<double> residual_maximum{2.0e-9};
  std::vector<std::uint64_t> iterations{3u};
  std::vector<std::uint64_t> restart_counts{1u};
  std::vector<xtbloom_status_t> statuses{XTBLOOM_STATUS_SUCCESS};
  std::vector<std::uint8_t> initialized{1u};
  std::vector<std::uint8_t> residual_converged{1u};
  std::vector<double> previous_free{-1.04};
  std::vector<double> free_change{0.05};
  std::vector<std::uint8_t> converged{1u};

  WarmScalarState()
      : coefficients(static_cast<std::size_t>(kMatrices), 0.0),
        density(static_cast<std::size_t>(kMatrices), 0.0),
        weighted_density(static_cast<std::size_t>(kMatrices), 0.0),
        previous_inputs(static_cast<std::size_t>(kShells), 0.0),
        previous_residuals(static_cast<std::size_t>(kShells), 0.0),
        df_history(static_cast<std::size_t>(kShells * kHistory), 0.0),
        u_history(static_cast<std::size_t>(kShells * kHistory), 0.0) {
    /* The initializer seals the signed energy delta by exact binary64
     * equality, so construct it with the same subtraction it validates. */
    free_change[0] = free[0] - previous_free[0];
    for (std::int64_t index = 0; index < kMatrices; ++index) {
      coefficients[static_cast<std::size_t>(index)] = 0.01 * static_cast<double>(index + 1);
      density[static_cast<std::size_t>(index)] = 0.02 * static_cast<double>(index + 1);
      weighted_density[static_cast<std::size_t>(index)] = -0.01 * static_cast<double>(index + 1);
    }
  }

  [[nodiscard]] Gfn2SccIterationHostInitialization view() const {
    Gfn2SccIterationHostInitialization host{};
    host.mode = Gfn2SccIterationInitializationMode::kWarm;
    host.plan_token = kPlanToken;
    host.initialization_generation = 7u;
    host.topology = {host_view(atom_offsets), host_view(shell_offsets), kPlanToken};

    host.wavefunction.eigenvalues = host_view(eigenvalues);
    host.wavefunction.coefficients = host_view(coefficients);
    host.wavefunction.occupations = host_view(occupations);
    host.wavefunction.chemical_potentials = host_view(chemical_potentials);
    host.wavefunction.electron_sums = host_view(electron_sums);
    host.wavefunction.occupation_entropies = host_view(entropy);
    host.wavefunction.density = host_view(density);
    host.wavefunction.energy_weighted_density = host_view(weighted_density);
    host.wavefunction.band_energies = host_view(band_energy);
    host.wavefunction.occupation_sums = host_view(occupation_sum);
    host.wavefunction.density_traces = host_view(density_trace);
    host.wavefunction.weighted_density_traces = host_view(weighted_density_trace);
    host.wavefunction.population.shell_charges = host_view(qsh);
    host.wavefunction.population.atomic_charges = host_view(qat);
    host.wavefunction.population.plan_token = kPlanToken;
    host.wavefunction.plan_token = kPlanToken;

    host.energy.core = host_view(core);
    host.energy.es2 = host_view(es2);
    host.energy.es3 = host_view(es3);
    host.energy.entropy = host_view(entropy);
    host.energy.internal_energy = host_view(internal);
    host.energy.free_energy = host_view(free);
    host.energy.classical_total = host_view(classical_total);
    host.energy.plan_token = kPlanToken;

    host.mixer.current_inputs = host_view(qsh);
    host.mixer.previous_inputs = host_view(previous_inputs);
    host.mixer.previous_residuals = host_view(previous_residuals);
    host.mixer.df_history = host_view(df_history);
    host.mixer.u_history = host_view(u_history);
    host.mixer.omega = host_view(omega);
    host.mixer.residual_rms = host_view(residual_rms);
    host.mixer.residual_maximum = host_view(residual_maximum);
    host.mixer.iterations = host_view(iterations);
    host.mixer.restart_counts = host_view(restart_counts);
    host.mixer.system_statuses = host_view(statuses);
    host.mixer.initialized = host_view(initialized);
    host.mixer.residual_converged = host_view(residual_converged);
    host.mixer.plan_token = kPlanToken;

    host.scc.current_shell_charges = host_view(qsh);
    host.scc.free_energies = host_view(free);
    host.scc.previous_free_energies = host_view(previous_free);
    host.scc.free_energy_changes = host_view(free_change);
    host.scc.residual_rms = host_view(residual_rms);
    host.scc.iterations = host_view(iterations);
    host.scc.system_statuses = host_view(statuses);
    host.scc.converged = host_view(converged);
    host.scc.plan_token = kPlanToken;
    return host;
  }
};

int test_scalar_arena_and_initializer_contract() {
  BoundScalarArena scalar;
  CHECK(scalar.valid);

  /* GFN1 owns qsh only. True multipole and AES2 storage must therefore be
   * canonical empty, while coordinate-gradient scratch remains 3*nat. */
  CHECK(scalar.state.raw_population.dipole == nullptr);
  CHECK(scalar.state.raw_population.dipole_elements == 0);
  CHECK(scalar.state.raw_population.quadrupole == nullptr);
  CHECK(scalar.state.raw_population.quadrupole_elements == 0);
  CHECK(scalar.state.published.atomic_dipoles == nullptr);
  CHECK(scalar.state.published.dipole_elements == 0);
  CHECK(scalar.state.published.atomic_quadrupoles == nullptr);
  CHECK(scalar.state.published.quadrupole_elements == 0);
  CHECK(scalar.workspace.staged_raw_population.dipole == nullptr);
  CHECK(scalar.workspace.staged_raw_population.quadrupole == nullptr);
  CHECK(scalar.workspace.mixed_topology.atomic_dipoles == nullptr);
  CHECK(scalar.workspace.mixed_topology.atomic_quadrupoles == nullptr);
  CHECK(scalar.workspace.physical_topology.atomic_dipoles == nullptr);
  CHECK(scalar.workspace.physical_topology.atomic_quadrupoles == nullptr);
  CHECK(scalar.workspace.geometry_workspace.gradient_scratch != nullptr);
  CHECK(scalar.workspace.geometry_workspace.gradient_elements == 3 * kAtoms);
  CHECK(scalar.workspace.es2_workspace.gradient_scratch != nullptr);
  CHECK(scalar.workspace.es2_workspace.gradient_elements == 3 * kAtoms);

  CHECK(scalar.workspace.components.aes2_atomic_potential == nullptr);
  CHECK(scalar.workspace.components.aes2_atomic_elements == 0);
  CHECK(scalar.workspace.components.aes2_dipole_potential == nullptr);
  CHECK(scalar.workspace.components.aes2_dipole_elements == 0);
  CHECK(scalar.workspace.components.aes2_quadrupole_potential == nullptr);
  CHECK(scalar.workspace.components.aes2_quadrupole_elements == 0);
  CHECK(scalar.state.free_energy.aes2 == nullptr);
  CHECK(scalar.state.free_energy.aes2_elements == 0);
  CHECK(scalar.workspace.aes2_workspace.pair_scratch == nullptr);
  CHECK(scalar.workspace.aes2_workspace.pair_elements == 0);
  CHECK(scalar.workspace.aes2_workspace.potential_scratch == nullptr);
  CHECK(scalar.workspace.aes2_workspace.potential_elements == 0);
  CHECK(scalar.workspace.aes2_workspace.batch_scratch == nullptr);
  CHECK(scalar.workspace.aes2_workspace.batch_elements == 0);
  CHECK(scalar.workspace.aes2_workspace.gradient_scratch == nullptr);
  CHECK(scalar.workspace.aes2_workspace.gradient_elements == 0);
  CHECK(scalar.workspace.aes2_workspace.coordination_scratch == nullptr);
  CHECK(scalar.workspace.aes2_workspace.coordination_elements == 0);
  CHECK(scalar.workspace.aes2_workspace.scc_peer_error_scratch == nullptr);
  CHECK(scalar.workspace.aes2_workspace.scc_peer_error_elements == 0);

  CHECK(scalar.state.mixer.total_vector_elements == kShells);
  CHECK(scalar.state.mixer.history_elements == kShells * kHistory);
  CHECK(scalar.workspace.mixer_workspace.vector_elements == kShells);

  auto irrelevant_aes2_extent = make_scalar_plan();
  irrelevant_aes2_extent.aes2_batch.total_pairs = 1000000;
  Gfn2SccIterationArenaRequirements same_requirements{};
  CHECK(query_gfn2_scc_iteration_arena_requirements_cuda(
            irrelevant_aes2_extent, irrelevant_aes2_extent.eigensolver_provider.requirements,
            same_requirements)
            .success());
  CHECK(same_requirements.total_bytes == scalar.requirements.total_bytes);
  CHECK(same_requirements.layout_fingerprint == scalar.requirements.layout_fingerprint);

  auto invalid_mixer = make_scalar_plan();
  invalid_mixer.mixer_policy.atomic_multipole_components = 9;
  Gfn2SccIterationArenaRequirements rejected{};
  CHECK(query_gfn2_scc_iteration_arena_requirements_cuda(
            invalid_mixer, invalid_mixer.eigensolver_provider.requirements, rejected)
            .error == Gfn2SccIterationArenaError::kInvalidPlan);

  auto invalid_aes2 = make_scalar_plan();
  invalid_aes2.enabled_components |= static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2);
  CHECK(query_gfn2_scc_iteration_arena_requirements_cuda(
            invalid_aes2, invalid_aes2.eigensolver_provider.requirements, rejected)
            .error == Gfn2SccIterationArenaError::kInvalidPlan);
  CHECK(validate_gfn2_scc_iteration_binding_cuda(invalid_aes2, {}, {}, {}).field ==
        Gfn2SccIterationBindingField::kPlan);

  FreshScalarState fresh;
  Gfn2SccIterationInitializer fresh_initializer;
  auto diagnostic = Gfn2SccIterationInitializer::create(
      scalar.plan, scalar.requirements, scalar.arena.get(), scalar.arena.bytes(), scalar.state,
      scalar.workspace, scalar.reports, fresh.view(), fresh_initializer);
  if (!diagnostic.success()) {
    std::fprintf(stderr, "GFN1 fresh initializer failed: error=%u field=%u index=%lld\n",
                 static_cast<unsigned>(diagnostic.error), static_cast<unsigned>(diagnostic.field),
                 static_cast<long long>(diagnostic.index));
  }
  CHECK(diagnostic.success());
  CHECK(fresh_initializer.valid());

  WarmScalarState warm;
  Gfn2SccIterationInitializer warm_initializer;
  diagnostic = Gfn2SccIterationInitializer::create(
      scalar.plan, scalar.requirements, scalar.arena.get(), scalar.arena.bytes(), scalar.state,
      scalar.workspace, scalar.reports, warm.view(), warm_initializer);
  if (!diagnostic.success()) {
    std::fprintf(stderr, "GFN1 warm initializer failed: error=%u field=%u index=%lld\n",
                 static_cast<unsigned>(diagnostic.error), static_cast<unsigned>(diagnostic.field),
                 static_cast<long long>(diagnostic.index));
  }
  CHECK(diagnostic.success());
  CHECK(warm_initializer.valid());

  /* Supplying the absent model state is not a compatibility option: it must
   * fail synchronously rather than silently growing a GFN1 checkpoint. */
  std::vector<double> synthetic_dipoles(static_cast<std::size_t>(3 * kAtoms), 0.0);
  auto synthetic_fresh = fresh.view(4u);
  synthetic_fresh.wavefunction.population.atomic_dipoles = host_view(synthetic_dipoles);
  diagnostic = Gfn2SccIterationInitializer::create(
      scalar.plan, scalar.requirements, scalar.arena.get(), scalar.arena.bytes(), scalar.state,
      scalar.workspace, scalar.reports, synthetic_fresh, fresh_initializer);
  CHECK(diagnostic.error == Gfn2SccIterationInitializationError::kInvalidExtent);
  CHECK(diagnostic.field == Gfn2SccIterationInitializationField::kPopulation);

  std::vector<double> synthetic_aes2{0.0};
  auto synthetic_warm = warm.view();
  synthetic_warm.energy.aes2 = host_view(synthetic_aes2);
  diagnostic = Gfn2SccIterationInitializer::create(
      scalar.plan, scalar.requirements, scalar.arena.get(), scalar.arena.bytes(), scalar.state,
      scalar.workspace, scalar.reports, synthetic_warm, warm_initializer);
  CHECK(diagnostic.error == Gfn2SccIterationInitializationError::kInvalidExtent);
  CHECK(diagnostic.field == Gfn2SccIterationInitializationField::kEnergy);

  auto corrupt_state = scalar.state;
  ++corrupt_state.mixer.total_vector_elements;
  diagnostic = Gfn2SccIterationInitializer::create(
      scalar.plan, scalar.requirements, scalar.arena.get(), scalar.arena.bytes(), corrupt_state,
      scalar.workspace, scalar.reports, fresh.view(5u), fresh_initializer);
  CHECK(diagnostic.error == Gfn2SccIterationInitializationError::kInvalidExtent);
  CHECK(diagnostic.field == Gfn2SccIterationInitializationField::kMixer);
  return 0;
}

struct Gfn1SetupFixture {
  std::vector<std::int64_t> atom_offsets{0, 2};
  std::vector<std::int32_t> atomic_numbers{1, 1};
  std::vector<double> positions{0.0, 0.0, -0.7, 0.0, 0.0, 0.7};
  std::vector<double> molecular_charges{0.0};
  std::vector<std::int32_t> unpaired_electrons{0};
  std::vector<std::int32_t> spin_channels{1};

  gfn1::BasisPlan basis;
  gfn1::IntegralPlan integrals;
  gfn1::H0Plan h0_plan;
  gfn1::WavefunctionLayout wavefunction;
  gfn1::MullikenPlan mulliken;
  gfn1::ES2Plan es2;
  gfn1::ES3Plan es3;
  gfn1::SpinPopulationLayout spin_layout;
  gfn1::SpinPolarizationPlan spin;
  gfn1::EigensolverPlan eigensolver;
  gfn1::SccMixerPlan mixer;
  gfn1::SccDriverPlan driver;
  gfn1::CoordinationPlan coordination_plan;

  std::vector<double> covalent_radii;
  std::vector<double> h0;
  std::vector<double> overlap;
  std::vector<double> geometry_pair_data;
  std::vector<double> coordination;
  std::vector<std::uint64_t> generations{1u};
  std::vector<double> es2_matrix;

  bool initialize(std::string& error) {
    if (gfn1::make_basis_plan(1, 2, atom_offsets.data(), atomic_numbers.data(), basis, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        gfn1::make_integral_plan(basis, integrals, error) != XTBLOOM_STATUS_SUCCESS ||
        gfn1::make_h0_plan(basis, integrals, atomic_numbers.data(), h0_plan, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        gfn1::make_wavefunction_layout(basis, atomic_numbers.data(), molecular_charges.data(),
                                       unpaired_electrons.data(), spin_channels.data(),
                                       wavefunction, error) != XTBLOOM_STATUS_SUCCESS ||
        gfn1::make_mulliken_plan(basis, integrals, wavefunction, mulliken, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        gfn1::make_es2_plan(basis, atomic_numbers.data(), es2, error) != XTBLOOM_STATUS_SUCCESS ||
        gfn1::make_es3_plan(basis, atomic_numbers.data(), es3, error) != XTBLOOM_STATUS_SUCCESS ||
        gfn1::make_spin_population_layout(basis, spin_channels.data(), spin_layout, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        gfn1::make_spin_polarization_plan(basis, atomic_numbers.data(), spin_layout, spin, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        gfn2::make_eigensolver_plan(gfn1::make_eigensolver_wavefunction_layout(wavefunction),
                                    eigensolver, error) != XTBLOOM_STATUS_SUCCESS ||
        gfn1::make_scc_mixer_plan(wavefunction, 2, 0.4, 1.0e-8, 1.0e-8, mixer, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        gfn1::make_scc_driver_plan(wavefunction, mulliken, es2, es3, spin, eigensolver, mixer,
                                   nullptr, 20u, 300.0 * 3.166808578545117e-6, 1.0e-8, driver,
                                   error) != XTBLOOM_STATUS_SUCCESS ||
        gfn1::make_coordination_plan(1, 2, atom_offsets.data(), atomic_numbers.data(),
                                     coordination_plan, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    covalent_radii = coordination_plan.covalent_radius;
    h0.assign(static_cast<std::size_t>(integrals.total_matrix_elements), 0.0);
    overlap.assign(h0.size(), 0.0);
    geometry_pair_data.assign(static_cast<std::size_t>(kGfn2GeometryPairDataElements), 0.0);
    coordination.assign(2u, 0.0);
    if (gfn1::evaluate_coordination_cpu(coordination_plan, positions.data(), coordination.data(),
                                        error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    es2_matrix.assign(static_cast<std::size_t>(es2.total_matrix_elements()), 0.0);
    return true;
  }

  [[nodiscard]] Gfn1SccSetupInputSources sources() const {
    Gfn1SccSetupInputSources result{};
    result.basis = &basis;
    result.integrals = &integrals;
    result.h0_plan = &h0_plan;
    result.wavefunction = &wavefunction;
    result.es2 = &es2;
    result.es3 = &es3;
    result.spin = &spin;
    result.mulliken = &mulliken;
    result.mixer = &mixer;
    result.driver = &driver;
    result.geometry_generation = 1u;
    result.atomic_numbers = setup_view(atomic_numbers);
    result.positions = setup_view(positions);
    result.covalent_radii = setup_view(covalent_radii);
    result.h0 = setup_view(h0);
    result.overlap = setup_view(overlap);
    result.geometry_cache.pair_data = setup_view(geometry_pair_data);
    result.geometry_cache.coordination_numbers = setup_view(coordination);
    result.geometry_cache.system_generations = setup_view(generations);
    result.es2_cache.coulomb_matrix = setup_view(es2_matrix);
    return result;
  }
};

int test_setup_rejects_noncanonical_gfn1_provenance() {
  Gfn1SetupFixture fixture;
  std::string error;
  CHECK(fixture.initialize(error));

  Gfn2SccSetupTopology topology;
  auto topology_diagnostic = Gfn2SccSetupTopology::create(
      fixture.basis, fixture.integrals, fixture.wavefunction, kPlanToken, topology);
  CHECK(topology_diagnostic.success());

  Gfn2SccSetupInputs baseline;
  CHECK(
      Gfn2SccSetupInputs::create(fixture.sources(), topology.host_topology(), kPlanToken, baseline)
          .success());
  CHECK(baseline.valid());

  const auto expect_rejected = [&](const Gfn1SccSetupInputSources& sources) {
    Gfn2SccSetupInputs output;
    const auto diagnostic =
        Gfn2SccSetupInputs::create(sources, topology.host_topology(), kPlanToken, output);
    return !diagnostic.success() && diagnostic.status == XTBLOOM_STATUS_INVALID_ARGUMENT &&
           !output.valid();
  };

  /* Equal extents are insufficient provenance. GFN1 ES3 must share the basis
   * atom partition and its Gamma3 values must be reconstructed from Z. */
  auto hostile_es3 = fixture.es3;
  hostile_es3.atom_offsets[0] = 1;
  auto sources = fixture.sources();
  sources.es3 = &hostile_es3;
  CHECK(expect_rejected(sources));

  hostile_es3 = fixture.es3;
  hostile_es3.atom_gamma3[1] = hostile_es3.atom_gamma3[0] + 0.125;
  sources = fixture.sources();
  sources.es3 = &hostile_es3;
  CHECK(expect_rejected(sources));

  /* The same reconstruction rule applies to the atom-local spin table and
   * the coordination radii uploaded by the shared geometry leaf. */
  auto hostile_spin = fixture.spin;
  hostile_spin.coupling_matrices[0] += 0.125;
  sources = fixture.sources();
  sources.spin = &hostile_spin;
  CHECK(expect_rejected(sources));

  auto hostile_radii = fixture.covalent_radii;
  hostile_radii[0] += 0.125;
  sources = fixture.sources();
  sources.covalent_radii = setup_view(hostile_radii);
  CHECK(expect_rejected(sources));
  return 0;
}

struct Gfn1Es3DeviceFixture {
  std::vector<std::int64_t> batch_shell_offsets{0, 4};
  /* Allocate one spare entry so intentional metadata aliasing remains in-bounds. */
  std::vector<std::int64_t> atom_offsets{0, 2, 4};
  std::vector<std::int64_t> atom_shell_offsets{0, 2, 4};
  std::vector<std::int64_t> shell_to_atom{0, 0, 1, 1};
  std::vector<double> gamma3{0.10, 0.10, 0.20, 0.20};
  std::vector<double> shell_charges{0.10, 0.20, -0.10, 0.30};
  std::vector<double> potential_seed{17.0, 18.0, 19.0, 20.0};
  std::vector<double> energy_seed{3.0};

  DeviceVector<std::int64_t> d_batch_shell_offsets;
  DeviceVector<std::int64_t> d_atom_offsets;
  DeviceVector<std::int64_t> d_atom_shell_offsets;
  DeviceVector<std::int64_t> d_shell_to_atom;
  DeviceVector<double> d_gamma3;
  DeviceVector<double> d_shell_charges;
  DeviceVector<double> d_potentials;
  DeviceVector<double> d_energies;
  DeviceVector<std::uint32_t> d_error;

  bool initialize() {
    return d_batch_shell_offsets.assign(batch_shell_offsets) &&
           d_atom_offsets.assign(atom_offsets) && d_atom_shell_offsets.assign(atom_shell_offsets) &&
           d_shell_to_atom.assign(shell_to_atom) && d_gamma3.assign(gamma3) &&
           d_shell_charges.assign(shell_charges) && d_potentials.assign(potential_seed) &&
           d_energies.assign(energy_seed) && d_error.assign(std::vector<std::uint32_t>{0u});
  }

  [[nodiscard]] Gfn2ES3DeviceBatch batch() const {
    Gfn2ES3DeviceBatch result{};
    result.batch_size = 1;
    result.total_shells = 4;
    result.batch_shell_offset_count = 2;
    result.shell_gamma3_count = 4;
    result.batch_shell_offsets = d_batch_shell_offsets.get();
    result.shell_gamma3 = d_gamma3.get();
    result.plan_token = kPlanToken;
    result.model = XtbModelFlavor::kGfn1;
    result.total_atoms = 2;
    result.atom_offset_count = 2;
    result.atom_shell_offset_count = 3;
    result.shell_to_atom_count = 4;
    result.atom_offsets = d_atom_offsets.get();
    result.atom_shell_offsets = d_atom_shell_offsets.get();
    result.shell_to_atom = d_shell_to_atom.get();
    return result;
  }

  int reset_outputs() {
    CHECK(d_potentials.assign(potential_seed));
    CHECK(d_energies.assign(energy_seed));
    CHECK(d_error.assign(std::vector<std::uint32_t>{0u}));
    return 0;
  }

  int expect_potential_semantic_rejection(const Gfn2ES3DeviceBatch& batch) {
    CHECK(reset_outputs() == 0);
    CUDA_CHECK(reset_gfn2_es3_device_error_cuda(d_error.get()));
    CUDA_CHECK(evaluate_gfn2_es3_potential_cuda(batch, d_shell_charges.get(), d_potentials.get(),
                                                d_error.get()));
    std::vector<std::uint32_t> actual_error;
    std::vector<double> actual_potential;
    CHECK(d_error.copy_to(actual_error));
    CHECK(d_potentials.copy_to(actual_potential));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(actual_error[0] != static_cast<std::uint32_t>(Gfn2ES3DeviceError::kSuccess));
    CHECK(actual_potential == potential_seed);
    return 0;
  }
};

int test_gfn1_es3_hostile_topology_mapping_and_gamma3() {
  Gfn1Es3DeviceFixture fixture;
  CHECK(fixture.initialize());
  const auto baseline = fixture.batch();

  CUDA_CHECK(reset_gfn2_es3_device_error_cuda(fixture.d_error.get()));
  CUDA_CHECK(evaluate_gfn2_es3_potential_cuda(baseline, fixture.d_shell_charges.get(),
                                              fixture.d_potentials.get(), fixture.d_error.get()));
  CUDA_CHECK(add_gfn2_es3_energy_cuda(baseline, fixture.d_shell_charges.get(),
                                      fixture.d_energies.get(), fixture.d_error.get()));
  std::vector<std::uint32_t> semantic_error;
  std::vector<double> potentials;
  std::vector<double> energies;
  CHECK(fixture.d_error.copy_to(semantic_error));
  CHECK(fixture.d_potentials.copy_to(potentials));
  CHECK(fixture.d_energies.copy_to(energies));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error[0] == static_cast<std::uint32_t>(Gfn2ES3DeviceError::kSuccess));
  CHECK(std::abs(potentials[0] - 0.009) < 1.0e-15);
  CHECK(std::abs(potentials[1] - 0.009) < 1.0e-15);
  CHECK(std::abs(potentials[2] - 0.008) < 1.0e-15);
  CHECK(std::abs(potentials[3] - 0.008) < 1.0e-15);
  const double expected_energy = 3.0 + 0.10 * 0.3 * 0.3 * 0.3 / 3.0 + 0.20 * 0.2 * 0.2 * 0.2 / 3.0;
  CHECK(std::abs(energies[0] - expected_energy) < 1.0e-15);

  /* Host-side alias checks must cover the GFN1-only metadata, not only the
   * inherited shell offsets and Gamma3 arrays. */
  auto aliased = baseline;
  aliased.atom_shell_offsets = aliased.atom_offsets;
  CHECK(evaluate_gfn2_es3_potential_cuda(aliased, fixture.d_shell_charges.get(),
                                         fixture.d_potentials.get(),
                                         fixture.d_error.get()) == cudaErrorInvalidValue);
  aliased = baseline;
  aliased.shell_to_atom = reinterpret_cast<const std::int64_t*>(fixture.d_potentials.get());
  CHECK(evaluate_gfn2_es3_potential_cuda(aliased, fixture.d_shell_charges.get(),
                                         fixture.d_potentials.get(),
                                         fixture.d_error.get()) == cudaErrorInvalidValue);

  /* Every shell must be owned by exactly the atom whose half-open range
   * contains it. Individually in-range arrays can still disagree. */
  const std::vector<std::int64_t> inconsistent_atom_shell_offsets{0, 1, 4};
  CHECK(fixture.d_atom_shell_offsets.assign(inconsistent_atom_shell_offsets));
  CHECK(fixture.expect_potential_semantic_rejection(fixture.batch()) == 0);
  CHECK(fixture.d_atom_shell_offsets.assign(fixture.atom_shell_offsets));

  const std::vector<std::int64_t> invalid_shell_to_atom{0, 0, 1, 2};
  CHECK(fixture.d_shell_to_atom.assign(invalid_shell_to_atom));
  CHECK(fixture.expect_potential_semantic_rejection(fixture.batch()) == 0);
  CHECK(fixture.d_shell_to_atom.assign(fixture.shell_to_atom));

  /* GFN1 Gamma3 is atom-local. All shells owned by one atom must carry the
   * identical setup-broadcast value or potential and energy would disagree. */
  const std::vector<double> inconsistent_gamma3{0.10, 0.11, 0.20, 0.20};
  CHECK(fixture.d_gamma3.assign(inconsistent_gamma3));
  CHECK(fixture.expect_potential_semantic_rejection(fixture.batch()) == 0);
  CHECK(fixture.d_gamma3.assign(fixture.gamma3));

  const std::vector<std::int64_t> invalid_atom_offsets{1, 2, 4};
  CHECK(fixture.d_atom_offsets.assign(invalid_atom_offsets));
  CHECK(fixture.reset_outputs() == 0);
  CUDA_CHECK(reset_gfn2_es3_device_error_cuda(fixture.d_error.get()));
  CUDA_CHECK(add_gfn2_es3_energy_cuda(fixture.batch(), fixture.d_shell_charges.get(),
                                      fixture.d_energies.get(), fixture.d_error.get()));
  CHECK(fixture.d_error.copy_to(semantic_error));
  CHECK(fixture.d_energies.copy_to(energies));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error[0] != static_cast<std::uint32_t>(Gfn2ES3DeviceError::kSuccess));
  CHECK(energies == fixture.energy_seed);
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    if (std::getenv("XTBLOOM_TEST_REQUIRE_DEVICE") != nullptr) {
      std::cerr << "CUDA GFN1 scalar contract test requires a visible device: "
                << (count_status == cudaSuccess ? "none found" : cudaGetErrorString(count_status))
                << '\n';
      return 1;
    }
    std::cout << "CUDA GFN1 scalar contract test skipped: "
              << (count_status == cudaSuccess ? "no visible device"
                                              : cudaGetErrorString(count_status))
              << '\n';
    return 0;
  }

  if (const int status = test_scalar_arena_and_initializer_contract(); status != 0) return status;
  if (const int status = test_setup_rejects_noncanonical_gfn1_provenance(); status != 0) {
    return status;
  }
  return test_gfn1_es3_hostile_topology_mapping_and_gamma3();
}
