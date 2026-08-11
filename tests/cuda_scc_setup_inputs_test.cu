#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "backends/cuda/gfn2_scc_setup_inputs.cuh"
#include "backends/cuda/gfn2_scc_setup_topology.hpp"
#include "data/parameters/d4.hpp"
#include "model/gfn2/coordination.hpp"
#include "tests/support/gfn2_scc_test_case.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::Gfn2RaggedTopologyView;
using xtbloom::detail::Gfn2WavefunctionLayoutView;
using xtbloom::detail::cuda::Gfn2D4DeviceElementData;
using xtbloom::detail::cuda::Gfn2D4DeviceReferenceData;
using xtbloom::detail::cuda::Gfn2SccCacheProvenanceBinding;
using xtbloom::detail::cuda::Gfn2SccIterationDeviceInput;
using xtbloom::detail::cuda::Gfn2SccIterationDevicePlan;
using xtbloom::detail::cuda::Gfn2SccPotentialComponent;
using xtbloom::detail::cuda::Gfn2SccSetupHostArray;
using xtbloom::detail::cuda::Gfn2SccSetupInputs;
using xtbloom::detail::cuda::Gfn2SccSetupInputsError;
using xtbloom::detail::cuda::Gfn2SccSetupInputSources;
using xtbloom::detail::cuda::Gfn2SccSetupTopology;
using xtbloom::test::gfn2::HostSccCase;
using xtbloom::test::gfn2::HostSccCaseOptions;
using xtbloom::test::gfn2::SmallSystemKind;

constexpr std::uint64_t kPlanToken = 0x104104104ULL;

template <typename T>
Gfn2SccSetupHostArray<T> view(const std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), static_cast<std::int64_t>(values.size())};
}

template <typename T>
bool all_zero(const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const T zero{};
  return std::memcmp(&value, &zero, sizeof(T)) == 0;
}

class DeviceArena {
 public:
  DeviceArena() = default;
  ~DeviceArena() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;

  bool allocate(std::size_t bytes) {
    bytes_ = bytes;
    return cudaMalloc(&data_, bytes) == cudaSuccess;
  }
  void* data() const { return data_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0u;
};

struct InputBacking {
  xtbloom::detail::gfn2::CoordinationPlan coordination_plan;
  std::vector<double> geometry_pair_data;
  std::vector<std::uint64_t> geometry_generations;
  std::vector<Gfn2D4DeviceElementData> d4_elements;
  std::vector<Gfn2D4DeviceReferenceData> d4_references;

  bool prepare(const HostSccCase& host, std::string& error) {
    if (xtbloom::detail::gfn2::make_coordination_plan(
            host.batch_size(), host.total_atoms(), host.atom_offsets().data(),
            host.atomic_numbers().data(), coordination_plan, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    geometry_pair_data.resize(static_cast<std::size_t>(host.aes2_plan().total_pairs()) *
                              xtbloom::detail::cuda::kGfn2GeometryPairDataElements);
    for (std::size_t index = 0; index < geometry_pair_data.size(); ++index) {
      geometry_pair_data[index] = 0.001 * static_cast<double>(index + 1u);
    }
    geometry_generations.assign(static_cast<std::size_t>(host.batch_size()),
                                host.options().geometry_generation);
    d4_elements.reserve(xtbloom::parameters::d4::kElements.size());
    for (const auto& element : xtbloom::parameters::d4::kElements) {
      d4_elements.push_back({element.reference_offset, element.reference_count,
                             element.covalent_radius, element.electronegativity,
                             element.effective_charge, element.hardness, element.r4r2});
    }
    d4_references.reserve(xtbloom::parameters::d4::kReferences.size());
    for (const auto& reference : xtbloom::parameters::d4::kReferences) {
      d4_references.push_back(
          {reference.coordination_number, reference.charge, reference.gaussian_count});
    }
    return true;
  }

  Gfn2SccSetupInputSources sources(const HostSccCase& host) const {
    Gfn2SccSetupInputSources result{};
    result.basis = &host.basis_plan();
    result.integrals = &host.integral_plan();
    result.h0_plan = &host.h0_plan();
    result.wavefunction = &host.wavefunction_layout();
    result.es2 = &host.es2_plan();
    result.es3 = &host.es3_plan();
    result.aes2 = &host.aes2_plan();
    result.mulliken = &host.mulliken_plan();
    result.mixer = &host.mixer_plan();
    result.driver = &host.driver_plan();
    result.geometry_generation = host.options().geometry_generation;
    result.atomic_numbers = view(host.atomic_numbers());
    result.positions = view(host.positions());
    result.covalent_radii = view(coordination_plan.covalent_radius);
    result.h0 = view(host.h0());
    result.overlap = view(host.overlap());
    result.dipole_integrals = view(host.dipole_integrals());
    result.quadrupole_integrals = view(host.quadrupole_integrals());
    result.geometry_cache.pair_data = view(geometry_pair_data);
    result.geometry_cache.coordination_numbers = view(host.coordination_numbers());
    result.geometry_cache.system_generations = view(geometry_generations);
    result.es2_cache.coulomb_matrix = {host.es2_cache().coulomb_matrix,
                                       host.es2_cache().matrix_elements};
    result.aes2_cache.pair_data = {host.aes2_cache().pair_data,
                                   host.aes2_cache().pair_data_elements};

    if (host.d4_plan() != nullptr) {
      result.d4.plan = host.d4_plan();
      result.d4.elements = view(d4_elements);
      result.d4.references = view(d4_references);
      result.d4.reference_c6 = {
          xtbloom::parameters::d4::kReferenceC6.data(),
          static_cast<std::int64_t>(xtbloom::parameters::d4::kReferenceC6.size())};
      result.d4.coordination_numbers = {host.d4_cache()->coordination_numbers,
                                        host.d4_cache()->coordination_elements};
    }
    if (host.point_charge_plan() != nullptr) {
      result.point_charges.plan = host.point_charge_plan();
      result.point_charges.positions = view(host.point_charge_positions());
      result.point_charges.charges = view(host.point_charge_charges());
      result.point_charges.hardnesses = view(host.point_charge_hardnesses());
      result.point_charges.shell_potential_cache =
          view(host.explicit_point_charge_shell_potential());
    }
    if (host.periodic_plan() != nullptr) {
      result.periodic.plan = host.periodic_plan();
      result.periodic.shifts = view(host.periodic_shifts());
      result.periodic.response_matrices = view(host.periodic_response_matrices());
    }
    return result;
  }
};

struct SetupFixture {
  HostSccCase host;
  InputBacking backing;
  Gfn2SccSetupTopology topology_owner;
  Gfn2SccSetupInputs inputs_owner;
  DeviceArena topology_arena;
  DeviceArena inputs_arena;
  Gfn2RaggedTopologyView device_topology{};
  Gfn2WavefunctionLayoutView device_wavefunction{};
  cudaStream_t stream = nullptr;

  ~SetupFixture() {
    if (stream != nullptr) {
      (void)cudaStreamSynchronize(stream);
      (void)cudaStreamDestroy(stream);
    }
  }

  bool initialize(const HostSccCaseOptions& options) {
    std::string error;
    if (HostSccCase::create(options, host, error) != XTBLOOM_STATUS_SUCCESS ||
        !backing.prepare(host, error) || cudaStreamCreate(&stream) != cudaSuccess) {
      return false;
    }
    if (!Gfn2SccSetupTopology::create(host.basis_plan(), host.integral_plan(),
                                      host.wavefunction_layout(), kPlanToken, topology_owner)
             .success()) {
      return false;
    }
    if (!topology_arena.allocate(topology_owner.requirements().immutable_device_bytes) ||
        !topology_owner
             .bind_device_arena_and_upload_async(topology_arena.data(), topology_arena.bytes(),
                                                 device_topology, device_wavefunction, stream)
             .success()) {
      return false;
    }
    Gfn2SccSetupInputSources sources = backing.sources(host);
    if (!Gfn2SccSetupInputs::create(sources, topology_owner.host_topology(), kPlanToken,
                                    inputs_owner)
             .success() ||
        !inputs_arena.allocate(inputs_owner.requirements().device_bytes)) {
      return false;
    }
    return true;
  }
};

HostSccCaseOptions four_system_options(bool optional) {
  HostSccCaseOptions options{};
  options.systems = {SmallSystemKind::kH2, SmallSystemKind::kHe, SmallSystemKind::kLiH,
                     SmallSystemKind::kCH2};
  options.enable_d4 = optional;
  options.enable_periodic_embedding = optional;
  options.enable_explicit_point_charges = optional;
  options.geometry_generation = 104u;
  options.maximum_iterations = 17u;
  options.mixer_history = 4;
  options.electronic_temperature = 0.001;
  return options;
}

int test_all_optional_four_system_upload() {
  SetupFixture fixture;
  CHECK(fixture.initialize(four_system_options(true)));
  const auto requirements = fixture.inputs_owner.requirements();
  CHECK(requirements.device_bytes > 0u);
  CHECK(requirements.device_alignment == 256u);

  Gfn2SccIterationDevicePlan plan{};
  Gfn2SccIterationDeviceInput input{};
  CHECK(fixture.inputs_owner
            .bind_device_arena_and_upload_async(
                fixture.device_topology, fixture.device_wavefunction, fixture.inputs_arena.data(),
                fixture.inputs_arena.bytes(), plan, input, fixture.stream)
            .success());
  CUDA_CHECK(cudaStreamSynchronize(fixture.stream));

  const std::uint32_t optional_mask =
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kD4TwoBody) |
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kExplicitPointCharge) |
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kPeriodicEmbedding);
  CHECK(plan.plan_token == kPlanToken);
  CHECK((plan.enabled_components & optional_mask) == optional_mask);
  CHECK(plan.geometry_generation == fixture.host.options().geometry_generation);
  CHECK(plan.topology.atom_offsets == fixture.device_topology.atom_offsets);
  CHECK(plan.wavefunction_layout.spin_channels == fixture.device_wavefunction.spin_channels);
  CHECK(plan.wavefunction_layout.spin_shell_offsets ==
        fixture.device_wavefunction.spin_shell_offsets);
  /* ABI v3: the plan publishes the sealed common projections as the sole
   * borrowing authority, with exact pointer identity to the master topology. */
  CHECK(plan.atom_projection.plan_token == kPlanToken);
  CHECK(plan.atom_projection.atom_offsets == plan.topology.atom_offsets);
  CHECK(plan.shell_ownership_projection.batch_shell_offsets == plan.topology.batch_shell_offsets);
  CHECK(plan.shell_ownership_projection.atom_shell_offsets == plan.topology.atom_shell_offsets);
  CHECK(plan.shell_ownership_projection.shell_to_atom == plan.topology.shell_to_atom);
  CHECK(plan.ao_matrix_projection.batch_orbital_offsets == plan.topology.batch_orbital_offsets);
  CHECK(plan.ao_matrix_projection.matrix_offsets == plan.topology.matrix_offsets);
  CHECK(plan.ao_matrix_projection.shell_orbital_offsets == plan.topology.shell_orbital_offsets);
  CHECK(plan.ao_matrix_projection.orbital_to_shell == plan.topology.orbital_to_shell);
  CHECK(plan.ao_matrix_projection.orbital_to_atom == plan.topology.orbital_to_atom);
  CHECK(plan.ao_bucket_projection.plan_token == kPlanToken);
  CHECK(plan.ao_bucket_projection.bucket_offsets == plan.topology.bucket_offsets);
  CHECK(plan.ao_bucket_projection.bucket_systems == plan.topology.bucket_systems);
  /* Production topology uses kNone pair maps (packed pair offsets live in the
   * setup-owned dense geometry/AES2 domain), so the packed projection stays
   * the canonical empty form. */
  CHECK(plan.packed_all_pair_projection.plan_token == 0u);
  CHECK(plan.element_identity_projection.plan_token == kPlanToken);
  CHECK(plan.element_identity_projection.element_fingerprint != 0u);
  CHECK(plan.publication_plan.wavefunction_layout.spin_channels ==
        fixture.device_wavefunction.spin_channels);
  CHECK(plan.publication_plan.wavefunction_layout.spin_shell_offsets ==
        fixture.device_wavefunction.spin_shell_offsets);
  CHECK(plan.spin_batch.spin_channels == fixture.device_wavefunction.spin_channels);
  CHECK(plan.spin_batch.shell_population_offsets == fixture.device_wavefunction.spin_shell_offsets);
  CHECK(plan.spin_batch.shell_population_elements == fixture.device_wavefunction.total_spin_shells);
  CHECK(fixture.device_wavefunction.total_spin_channels == fixture.host.batch_size());
  CHECK(fixture.device_wavefunction.total_spin_orbitals ==
        fixture.host.basis_plan().total_orbitals);
  CHECK(fixture.device_wavefunction.total_spin_matrix_elements ==
        fixture.host.integral_plan().total_matrix_elements);
  CHECK(fixture.device_wavefunction.total_spin_shells == fixture.host.basis_plan().total_shells);
  CHECK(fixture.device_wavefunction.total_spin_atoms == fixture.host.total_atoms());
  CHECK(plan.d4_batch.atomic_number_hash != 0u);
  CHECK(plan.d4_pairlist_cache.positions != nullptr);
  CHECK(plan.d4_pairlist_cache.position_elements == 3 * fixture.host.total_atoms());
  CHECK(plan.d4_pairlist_cache.coordination_numbers != nullptr);
  CHECK(plan.d4_pairlist_cache.coordination_elements == fixture.host.total_atoms());
  CHECK(plan.d4_pairlist_cache.coordination_generations == nullptr);
  CHECK(plan.d4_pairlist_cache.coordination_generation_elements == 0);
  CHECK(plan.d4_pairlist_cache.coordination_eligible_mask == nullptr);
  CHECK(plan.d4_pairlist_cache.coordination_eligible_elements == 0);
  CHECK(all_zero(plan.d4_pairlist_cache.coordination_pairs));
  CHECK(all_zero(plan.d4_pairlist_cache.two_body_pairs));
  CHECK(all_zero(plan.d4_pairlist_cache.atm_pairs));
  CHECK(plan.d4_pairlist_cache.plan_token == kPlanToken);
  std::int64_t minimum_atoms = std::numeric_limits<std::int64_t>::max();
  for (std::size_t system = 0; system < fixture.host.atom_offsets().size() - 1u; ++system) {
    minimum_atoms = std::min(minimum_atoms, fixture.host.atom_offsets()[system + 1u] -
                                                fixture.host.atom_offsets()[system]);
  }
  CHECK(plan.d4_batch.minimum_atoms_per_system == minimum_atoms);
  CHECK(plan.explicit_point_charge_batch.total_point_charges == fixture.host.batch_size());
  CHECK(plan.periodic_batch.total_matrix_elements ==
        fixture.host.periodic_plan()->total_matrix_elements());
  CHECK(all_zero(plan.eigensolver_provider));
  CHECK(all_zero(plan.overlap_cache));
  CHECK(plan.eigensolver_batch.active == nullptr && plan.eigensolver_batch.active_elements == 0);
  CHECK(plan.occupations_batch.active == nullptr && plan.occupations_batch.active_elements == 0);
  CHECK(input.plan_token == kPlanToken);
  CHECK(input.hamiltonian.h0_elements == fixture.host.integral_plan().total_matrix_elements);

  std::vector<double> h0(fixture.host.h0().size());
  std::vector<double> positions(fixture.host.positions().size());
  std::vector<double> electrons(static_cast<std::size_t>(2 * fixture.host.batch_size()));
  std::vector<Gfn2SccCacheProvenanceBinding> provenance(
      static_cast<std::size_t>(plan.provenance.cache_binding_count));
  CUDA_CHECK(cudaMemcpyAsync(h0.data(), input.hamiltonian.h0, h0.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, fixture.stream));
  CUDA_CHECK(cudaMemcpyAsync(positions.data(), plan.explicit_point_charge_batch.qm_positions,
                             positions.size() * sizeof(double), cudaMemcpyDeviceToHost,
                             fixture.stream));
  CUDA_CHECK(cudaMemcpyAsync(electrons.data(), plan.occupations_batch.electron_counts,
                             electrons.size() * sizeof(double), cudaMemcpyDeviceToHost,
                             fixture.stream));
  CUDA_CHECK(cudaMemcpyAsync(provenance.data(), plan.provenance.cache_bindings,
                             provenance.size() * sizeof(Gfn2SccCacheProvenanceBinding),
                             cudaMemcpyDeviceToHost, fixture.stream));
  CUDA_CHECK(cudaStreamSynchronize(fixture.stream));
  CHECK(h0 == fixture.host.h0());
  CHECK(positions == fixture.host.positions());
  CHECK(provenance.size() == 6u);
  CHECK(provenance.front().provenance.system_geometry_generations ==
        plan.geometry_cache.geometry_generations);
  for (std::int64_t system = 0; system < fixture.host.batch_size(); ++system) {
    CHECK(
        electrons[static_cast<std::size_t>(2 * system)] ==
        fixture.host.wavefunction_layout().alpha_electron_counts[static_cast<std::size_t>(system)]);
    CHECK(
        electrons[static_cast<std::size_t>(2 * system + 1)] ==
        fixture.host.wavefunction_layout().beta_electron_counts[static_cast<std::size_t>(system)]);
  }
  return 0;
}

int test_base_canonical_null_and_transactional_failures() {
  SetupFixture fixture;
  CHECK(fixture.initialize(four_system_options(false)));
  Gfn2SccIterationDevicePlan plan{};
  Gfn2SccIterationDeviceInput input{};
  CHECK(fixture.inputs_owner
            .bind_device_arena_and_upload_async(
                fixture.device_topology, fixture.device_wavefunction, fixture.inputs_arena.data(),
                fixture.inputs_arena.bytes(), plan, input, fixture.stream)
            .success());
  CUDA_CHECK(cudaStreamSynchronize(fixture.stream));
  CHECK(all_zero(plan.d4_batch));
  CHECK(all_zero(plan.d4_parameters));
  CHECK(all_zero(plan.d4_pairlist_cache));
  CHECK(all_zero(plan.explicit_point_charge_batch));
  CHECK(all_zero(plan.explicit_point_charge_cache));
  CHECK(all_zero(plan.periodic_batch));
  CHECK(plan.provenance.cache_binding_count == 3);

  Gfn2SccIterationDevicePlan sentinel_plan{};
  sentinel_plan.plan_token = 999u;
  Gfn2SccIterationDeviceInput sentinel_input{};
  sentinel_input.plan_token = 888u;
  const auto too_small = fixture.inputs_owner.bind_device_arena_and_upload_async(
      fixture.device_topology, fixture.device_wavefunction, fixture.inputs_arena.data(),
      fixture.inputs_arena.bytes() - 1u, sentinel_plan, sentinel_input, fixture.stream);
  CHECK(too_small.error == Gfn2SccSetupInputsError::kInsufficientArena);
  CHECK(sentinel_plan.plan_token == 999u && sentinel_input.plan_token == 888u);

  Gfn2RaggedTopologyView cross_plan = fixture.device_topology;
  ++cross_plan.plan_token;
  const auto cross = fixture.inputs_owner.bind_device_arena_and_upload_async(
      cross_plan, fixture.device_wavefunction, fixture.inputs_arena.data(),
      fixture.inputs_arena.bytes(), sentinel_plan, sentinel_input, fixture.stream);
  CHECK(cross.error == Gfn2SccSetupInputsError::kCrossPlan);
  CHECK(sentinel_plan.plan_token == 999u && sentinel_input.plan_token == 888u);

  Gfn2WavefunctionLayoutView cross_wavefunction = fixture.device_wavefunction;
  ++cross_wavefunction.plan_token;
  const auto wavefunction_cross = fixture.inputs_owner.bind_device_arena_and_upload_async(
      fixture.device_topology, cross_wavefunction, fixture.inputs_arena.data(),
      fixture.inputs_arena.bytes(), sentinel_plan, sentinel_input, fixture.stream);
  CHECK(wavefunction_cross.error == Gfn2SccSetupInputsError::kCrossPlan);
  CHECK(sentinel_plan.plan_token == 999u && sentinel_input.plan_token == 888u);
  return 0;
}

int test_same_size_optional_cross_plan_rejected_transactionally() {
  SetupFixture fixture;
  CHECK(fixture.initialize(four_system_options(true)));

  HostSccCaseOptions mismatched_options = four_system_options(true);
  mismatched_options.systems = {SmallSystemKind::kHe, SmallSystemKind::kH2, SmallSystemKind::kLiH,
                                SmallSystemKind::kCH2};
  HostSccCase mismatched;
  std::string error;
  CHECK(HostSccCase::create(mismatched_options, mismatched, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(mismatched.batch_size() == fixture.host.batch_size());
  CHECK(mismatched.total_atoms() == fixture.host.total_atoms());
  CHECK(mismatched.basis_plan().total_shells == fixture.host.basis_plan().total_shells);
  CHECK(mismatched.integral_plan().total_matrix_elements ==
        fixture.host.integral_plan().total_matrix_elements);
  CHECK(mismatched.atom_offsets() != fixture.host.atom_offsets());

  const auto original_requirements = fixture.inputs_owner.requirements();
  Gfn2SccSetupInputSources sources = fixture.backing.sources(fixture.host);
  sources.point_charges.plan = mismatched.point_charge_plan();
  const auto point_diagnostic = Gfn2SccSetupInputs::create(
      sources, fixture.topology_owner.host_topology(), kPlanToken, fixture.inputs_owner);
  CHECK(point_diagnostic.error == Gfn2SccSetupInputsError::kCrossPlan);
  CHECK(fixture.inputs_owner.valid());
  CHECK(fixture.inputs_owner.requirements().device_bytes == original_requirements.device_bytes);

  sources = fixture.backing.sources(fixture.host);
  sources.periodic.plan = mismatched.periodic_plan();
  const auto periodic_diagnostic = Gfn2SccSetupInputs::create(
      sources, fixture.topology_owner.host_topology(), kPlanToken, fixture.inputs_owner);
  CHECK(periodic_diagnostic.error == Gfn2SccSetupInputsError::kCrossPlan);
  CHECK(fixture.inputs_owner.valid());
  CHECK(fixture.inputs_owner.requirements().device_bytes == original_requirements.device_bytes);
  return 0;
}

}  // namespace

int main() {
  for (const int line : {test_all_optional_four_system_upload(),
                         test_base_canonical_null_and_transactional_failures(),
                         test_same_size_optional_cross_plan_rejected_transactionally()}) {
    if (line != 0) {
      std::fprintf(stderr, "CUDA SCC setup-inputs test failed at line %d\n", line);
      return 1;
    }
  }
  return 0;
}
