#include <algorithm>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_scc_setup_inputs.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr std::size_t kDeviceAlignment = 256u;

struct Segment {
  std::size_t offset = 0u;
  std::int64_t elements = 0;
};

Gfn2SccSetupInputsDiagnostic failure(xtbloom_status_t status, Gfn2SccSetupInputsError error,
                                     Gfn2SccSetupInputsField field,
                                     std::int64_t index = -1) noexcept {
  Gfn2SccSetupInputsDiagnostic diagnostic{};
  diagnostic.status = status;
  diagnostic.error = error;
  diagnostic.field = field;
  diagnostic.index = index;
  return diagnostic;
}

Gfn2SccSetupInputsDiagnostic arena_failure(Gfn2SccSetupInputsError error,
                                           std::size_t required_bytes,
                                           cudaError_t cuda_status = cudaSuccess) noexcept {
  Gfn2SccSetupInputsDiagnostic diagnostic =
      failure(error == Gfn2SccSetupInputsError::kCudaError ? XTBLOOM_STATUS_INTERNAL_ERROR
                                                           : XTBLOOM_STATUS_INVALID_ARGUMENT,
              error, Gfn2SccSetupInputsField::kArena);
  diagnostic.required_bytes = required_bytes;
  diagnostic.cuda_status = cuda_status;
  return diagnostic;
}

bool checked_add(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first > std::numeric_limits<std::size_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

bool checked_multiply(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first != 0u && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t& result) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  result = first * second;
  return true;
}

bool align_cursor(std::size_t cursor, std::size_t alignment, std::size_t& result) noexcept {
  const std::size_t remainder = cursor % alignment;
  return remainder == 0u || checked_add(cursor, alignment - remainder, result);
}

template <typename T>
bool append_segment(std::int64_t elements, std::size_t& cursor, Segment& segment) noexcept {
  if (elements < 0 || static_cast<std::uint64_t>(elements) >
                          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  std::size_t aligned = cursor;
  std::size_t bytes = 0u;
  std::size_t end = 0u;
  if (!align_cursor(cursor, alignof(T), aligned) ||
      !checked_multiply(static_cast<std::size_t>(elements), sizeof(T), bytes) ||
      !checked_add(aligned, bytes, end)) {
    return false;
  }
  segment.offset = aligned;
  segment.elements = elements;
  cursor = end;
  return true;
}

template <typename T>
bool exact_array(const Gfn2SccSetupHostArray<T>& values, std::int64_t expected) noexcept {
  return values.elements == expected && (expected == 0 || values.data != nullptr);
}

template <typename T>
bool exact_vector(const std::vector<T>& values, std::int64_t expected) noexcept {
  return expected >= 0 &&
         static_cast<std::uint64_t>(expected) <=
             static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) &&
         values.size() == static_cast<std::size_t>(expected);
}

template <typename T>
bool vector_matches(const std::vector<T>& values, const T* expected,
                    std::int64_t expected_elements) noexcept {
  return exact_vector(values, expected_elements) &&
         (expected_elements == 0 ||
          (expected != nullptr && std::equal(values.begin(), values.end(), expected)));
}

bool valid_offsets(const std::vector<std::int64_t>& offsets, std::int64_t partitions,
                   std::int64_t endpoint) noexcept {
  if (!exact_vector(offsets, partitions + 1) || offsets.front() != 0 ||
      offsets.back() != endpoint) {
    return false;
  }
  for (std::int64_t index = 0; index < partitions; ++index) {
    if (offsets[static_cast<std::size_t>(index)] < 0 ||
        offsets[static_cast<std::size_t>(index)] > offsets[static_cast<std::size_t>(index + 1)]) {
      return false;
    }
  }
  return true;
}

template <typename T>
void pack_array(std::byte* image, const Segment& segment,
                const Gfn2SccSetupHostArray<T>& source) noexcept {
  if (segment.elements != 0) {
    std::memcpy(image + segment.offset, source.data,
                static_cast<std::size_t>(segment.elements) * sizeof(T));
  }
}

template <typename T>
void pack_vector(std::byte* image, const Segment& segment, const std::vector<T>& source) noexcept {
  if (segment.elements != 0) {
    std::memcpy(image + segment.offset, source.data(), source.size() * sizeof(T));
  }
}

template <typename T>
T* device_pointer(std::byte* arena, const Segment& segment) noexcept {
  return segment.elements == 0 ? nullptr : reinterpret_cast<T*>(arena + segment.offset);
}

template <typename T>
const T* const_device_pointer(std::byte* arena, const Segment& segment) noexcept {
  return device_pointer<T>(arena, segment);
}

bool same_shape(const Gfn2RaggedTopologyView& first,
                const Gfn2RaggedTopologyView& second) noexcept {
  return first.plan_token == second.plan_token && first.batch_size == second.batch_size &&
         first.total_atoms == second.total_atoms && first.total_shells == second.total_shells &&
         first.total_orbitals == second.total_orbitals &&
         first.total_matrix_elements == second.total_matrix_elements &&
         first.bucket_count == second.bucket_count &&
         first.atom_offset_count == second.atom_offset_count &&
         first.batch_shell_offset_count == second.batch_shell_offset_count &&
         first.batch_orbital_offset_count == second.batch_orbital_offset_count &&
         first.matrix_offset_count == second.matrix_offset_count &&
         first.atom_shell_offset_count == second.atom_shell_offset_count &&
         first.shell_orbital_offset_count == second.shell_orbital_offset_count &&
         first.shell_to_atom_count == second.shell_to_atom_count &&
         first.orbital_to_shell_count == second.orbital_to_shell_count &&
         first.orbital_to_atom_count == second.orbital_to_atom_count &&
         first.bucket_offset_count == second.bucket_offset_count &&
         first.bucket_system_count == second.bucket_system_count &&
         first.bucket_orbital_count == second.bucket_orbital_count;
}

bool same_wavefunction_shape(const Gfn2WavefunctionLayoutView& expected,
                             const Gfn2WavefunctionLayoutView& candidate) noexcept {
  return expected.plan_token == candidate.plan_token &&
         expected.layout_fingerprint == candidate.layout_fingerprint &&
         expected.batch_size == candidate.batch_size &&
         expected.total_spin_channels == candidate.total_spin_channels &&
         expected.total_spin_orbitals == candidate.total_spin_orbitals &&
         expected.total_spin_matrix_elements == candidate.total_spin_matrix_elements &&
         expected.total_spin_shells == candidate.total_spin_shells &&
         expected.total_spin_atoms == candidate.total_spin_atoms &&
         expected.spin_channel_count == candidate.spin_channel_count &&
         expected.spin_channel_offset_count == candidate.spin_channel_offset_count &&
         expected.spin_orbital_offset_count == candidate.spin_orbital_offset_count &&
         expected.spin_matrix_offset_count == candidate.spin_matrix_offset_count &&
         expected.spin_shell_offset_count == candidate.spin_shell_offset_count &&
         expected.spin_atom_offset_count == candidate.spin_atom_offset_count;
}

std::int64_t maximum_partition(const std::int64_t* offsets, std::int64_t partitions) noexcept {
  std::int64_t maximum = 0;
  for (std::int64_t index = 0; index < partitions; ++index) {
    maximum = std::max(maximum, offsets[index + 1] - offsets[index]);
  }
  return maximum;
}

std::int64_t minimum_partition(const std::int64_t* offsets, std::int64_t partitions) noexcept {
  std::int64_t minimum = std::numeric_limits<std::int64_t>::max();
  for (std::int64_t index = 0; index < partitions; ++index) {
    minimum = std::min(minimum, offsets[index + 1] - offsets[index]);
  }
  return minimum;
}

}  // namespace

struct Gfn2SccSetupInputs::Impl {
  struct Layout {
    Segment pair_offsets;
    Segment covalent_radii;
    Segment positions;
    Segment atomic_numbers;
    Segment h0;
    Segment overlap;
    Segment dipole_integrals;
    Segment quadrupole_integrals;
    Segment spin_coupling_offsets;
    Segment spin_coupling_matrices;
    Segment dipole_offsets;
    Segment quadrupole_offsets;
    Segment es2_matrix_offsets;
    Segment es2_shell_hardness;
    Segment es3_shell_gamma3;
    Segment aes2_dipole_kernel;
    Segment aes2_quadrupole_kernel;
    Segment aes2_multipole_radius;
    Segment aes2_multipole_valence_cn;
    Segment mulliken_reference_occupations;
    Segment electron_counts;
    Segment temperatures;
    Segment geometry_pair_data;
    Segment geometry_coordination;
    Segment geometry_generations;
    Segment es2_coulomb_matrix;
    Segment aes2_pair_data;
    Segment d4_elements;
    Segment d4_references;
    Segment d4_reference_c6;
    Segment d4_coordination;
    Segment point_charge_offsets;
    Segment point_shell_hardness;
    Segment point_positions;
    Segment point_charges;
    Segment point_hardnesses;
    Segment point_shell_cache;
    Segment periodic_matrix_offsets;
    Segment periodic_shifts;
    Segment periodic_response;
    Segment warm_start_generations;
    Segment provenance_bindings;
    std::size_t total_bytes = 0u;
  } layout;

  std::uint64_t plan_token = 0u;
  std::uint64_t geometry_generation = 0u;
  std::uint64_t warm_start_generation = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t es2_matrix_elements = 0;
  std::int64_t total_pairs = 0;
  std::int64_t minimum_atoms = 0;
  std::int64_t maximum_atoms = 0;
  std::int64_t maximum_shells = 0;
  std::int64_t mixer_history = 0;
  std::uint64_t maximum_iterations = 0u;
  double mixer_damping = 0.0;
  double residual_tolerance = 0.0;
  double mixer_maximum_tolerance = 0.0;
  double energy_tolerance = 0.0;
  double electronic_temperature = 0.0;
  std::uint32_t enabled_components = 0u;
  Gfn2EigensolverOptions eigensolver_options{};
  Gfn2RaggedTopologyView host_topology{};
  Gfn2WavefunctionLayoutView host_wavefunction{};
  /* Setup-owned element identity seal over the exact atomic-number ordering;
   * reused as the plan's CUDA element-identity projection in bind(). */
  Gfn2ElementIdentityProjectionView host_element_identity{};
  std::int64_t spin_coupling_matrix_count = 0;
  std::int64_t mixer_vector_elements = 0;
  bool d4_enabled = false;
  bool point_enabled = false;
  bool periodic_enabled = false;
  std::int64_t point_count = 0;
  std::int64_t periodic_matrix_elements = 0;
  void* upload_image = nullptr;

  ~Impl() {
    if (upload_image != nullptr) {
      (void)cudaFreeHost(upload_image);
    }
  }

  bool make_layout() noexcept {
    std::size_t cursor = 0u;
    std::int64_t atom_coordinates = 0;
    std::int64_t dipoles = 0;
    std::int64_t quadrupoles = 0;
    std::int64_t geometry_pair_elements = 0;
    std::int64_t aes2_pair_elements = 0;
    std::int64_t point_coordinates = 0;
    if (!checked_multiply(total_atoms, 3, atom_coordinates) ||
        !checked_multiply(total_atoms, 3, dipoles) ||
        !checked_multiply(total_atoms, 6, quadrupoles) ||
        !checked_multiply(total_pairs, kGfn2GeometryPairDataElements, geometry_pair_elements) ||
        !checked_multiply(total_pairs, kGfn2AES2PairDataElements, aes2_pair_elements) ||
        !checked_multiply(point_count, 3, point_coordinates)) {
      return false;
    }
    const std::int64_t batch_offsets = batch_size + 1;
    const std::int64_t matrix3 = total_matrix_elements * 3;
    const std::int64_t matrix6 = total_matrix_elements * 6;
    const std::int64_t provenance_count =
        3 + (d4_enabled ? 1 : 0) + (point_enabled ? 1 : 0) + (periodic_enabled ? 1 : 0);
    return append_segment<std::int64_t>(batch_offsets, cursor, layout.pair_offsets) &&
           append_segment<double>(total_atoms, cursor, layout.covalent_radii) &&
           append_segment<double>(atom_coordinates, cursor, layout.positions) &&
           append_segment<std::int32_t>(total_atoms, cursor, layout.atomic_numbers) &&
           append_segment<double>(total_matrix_elements, cursor, layout.h0) &&
           append_segment<double>(total_matrix_elements, cursor, layout.overlap) &&
           append_segment<double>(matrix3, cursor, layout.dipole_integrals) &&
           append_segment<double>(matrix6, cursor, layout.quadrupole_integrals) &&
           append_segment<std::int64_t>(total_atoms + 1, cursor, layout.spin_coupling_offsets) &&
           append_segment<double>(spin_coupling_matrix_count, cursor,
                                  layout.spin_coupling_matrices) &&
           append_segment<std::int64_t>(batch_offsets, cursor, layout.dipole_offsets) &&
           append_segment<std::int64_t>(batch_offsets, cursor, layout.quadrupole_offsets) &&
           append_segment<std::int64_t>(batch_offsets, cursor, layout.es2_matrix_offsets) &&
           append_segment<double>(total_shells, cursor, layout.es2_shell_hardness) &&
           append_segment<double>(total_shells, cursor, layout.es3_shell_gamma3) &&
           append_segment<double>(total_atoms, cursor, layout.aes2_dipole_kernel) &&
           append_segment<double>(total_atoms, cursor, layout.aes2_quadrupole_kernel) &&
           append_segment<double>(total_atoms, cursor, layout.aes2_multipole_radius) &&
           append_segment<double>(total_atoms, cursor, layout.aes2_multipole_valence_cn) &&
           append_segment<double>(total_shells, cursor, layout.mulliken_reference_occupations) &&
           append_segment<double>(2 * batch_size, cursor, layout.electron_counts) &&
           append_segment<double>(batch_size, cursor, layout.temperatures) &&
           append_segment<double>(geometry_pair_elements, cursor, layout.geometry_pair_data) &&
           append_segment<double>(total_atoms, cursor, layout.geometry_coordination) &&
           append_segment<std::uint64_t>(batch_size, cursor, layout.geometry_generations) &&
           append_segment<double>(es2_matrix_elements, cursor, layout.es2_coulomb_matrix) &&
           append_segment<double>(aes2_pair_elements, cursor, layout.aes2_pair_data) &&
           append_segment<Gfn2D4DeviceElementData>(d4_enabled ? layout.d4_elements.elements : 0,
                                                   cursor, layout.d4_elements) &&
           append_segment<Gfn2D4DeviceReferenceData>(d4_enabled ? layout.d4_references.elements : 0,
                                                     cursor, layout.d4_references) &&
           append_segment<double>(d4_enabled ? layout.d4_reference_c6.elements : 0, cursor,
                                  layout.d4_reference_c6) &&
           append_segment<double>(d4_enabled ? total_atoms : 0, cursor,
                                  layout.d4_coordination) &&
           append_segment<std::int64_t>(point_enabled ? batch_offsets : 0, cursor,
                                        layout.point_charge_offsets) &&
           append_segment<double>(point_enabled ? total_shells : 0, cursor,
                                  layout.point_shell_hardness) &&
           append_segment<double>(point_enabled ? point_coordinates : 0, cursor,
                                  layout.point_positions) &&
           append_segment<double>(point_enabled ? point_count : 0, cursor, layout.point_charges) &&
           append_segment<double>(point_enabled ? point_count : 0, cursor,
                                  layout.point_hardnesses) &&
           append_segment<double>(point_enabled ? total_shells : 0, cursor,
                                  layout.point_shell_cache) &&
           append_segment<std::int64_t>(periodic_enabled ? batch_offsets : 0, cursor,
                                        layout.periodic_matrix_offsets) &&
           append_segment<double>(periodic_enabled ? total_atoms : 0, cursor,
                                  layout.periodic_shifts) &&
           append_segment<double>(periodic_enabled ? periodic_matrix_elements : 0, cursor,
                                  layout.periodic_response) &&
           append_segment<std::uint64_t>(warm_start_generation == 0u ? 0 : batch_size, cursor,
                                         layout.warm_start_generations) &&
           append_segment<Gfn2SccCacheProvenanceBinding>(provenance_count, cursor,
                                                         layout.provenance_bindings) &&
           (layout.total_bytes = cursor, true);
  }
};

Gfn2SccSetupInputs::Gfn2SccSetupInputs() noexcept = default;
Gfn2SccSetupInputs::~Gfn2SccSetupInputs() = default;
Gfn2SccSetupInputs::Gfn2SccSetupInputs(Gfn2SccSetupInputs&&) noexcept = default;
Gfn2SccSetupInputs& Gfn2SccSetupInputs::operator=(Gfn2SccSetupInputs&&) noexcept = default;

Gfn2SccSetupInputsDiagnostic Gfn2SccSetupInputs::create(const Gfn2SccSetupInputSources& sources,
                                                        const Gfn2RaggedTopologyView& host_topology,
                                                        std::uint64_t plan_token,
                                                        Gfn2SccSetupInputs& output) noexcept {
  using Error = Gfn2SccSetupInputsError;
  using Field = Gfn2SccSetupInputsField;
  if (plan_token == 0u) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kPlanToken);
  }
  if (host_topology.memory_space != Gfn2PlanMemorySpace::kHost ||
      host_topology.plan_token != plan_token ||
      validate_gfn2_topology_host(host_topology).error != Gfn2PlanSchemaError::kSuccess) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan, Field::kTopology);
  }
  if (sources.basis == nullptr || sources.integrals == nullptr || sources.h0_plan == nullptr ||
      sources.wavefunction == nullptr || sources.es2 == nullptr || sources.es3 == nullptr ||
      sources.aes2 == nullptr || sources.mulliken == nullptr || sources.mixer == nullptr ||
      sources.driver == nullptr) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kRequiredPlans);
  }

  const auto& basis = *sources.basis;
  const auto& integrals = *sources.integrals;
  const auto& h0_plan = *sources.h0_plan;
  const auto& wavefunction = *sources.wavefunction;
  const auto& es2 = *sources.es2;
  const auto& es3 = *sources.es3;
  const auto& aes2 = *sources.aes2;
  const auto& mulliken = *sources.mulliken;
  const auto& mixer = *sources.mixer;
  const auto& driver = *sources.driver;
  const std::int64_t batch = host_topology.batch_size;
  const std::int64_t atoms = host_topology.total_atoms;
  const std::int64_t shells = host_topology.total_shells;
  const std::int64_t orbitals = host_topology.total_orbitals;
  const std::int64_t matrices = host_topology.total_matrix_elements;
  std::int64_t dipole_integral_elements = 0;
  std::int64_t quadrupole_integral_elements = 0;
  if (!checked_multiply(matrices, 3, dipole_integral_elements) ||
      !checked_multiply(matrices, 6, quadrupole_integral_elements)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCountOverflow, Field::kHamiltonian);
  }

  if (basis.batch_size != batch || basis.total_atoms != atoms || basis.total_shells != shells ||
      basis.total_orbitals != orbitals || integrals.batch_size != batch ||
      integrals.total_matrix_elements != matrices || h0_plan.batch_size != batch ||
      h0_plan.total_atoms != atoms || h0_plan.total_shells != shells ||
      h0_plan.total_orbitals != orbitals || h0_plan.total_matrix_elements != matrices ||
      wavefunction.batch_size != batch || wavefunction.total_atoms != atoms ||
      wavefunction.total_shells != shells || wavefunction.total_orbitals != orbitals ||
      !es2.sealed() || es2.batch_size() != batch || es2.total_atoms() != atoms ||
      es2.total_shells() != shells || es3.batch_size != batch || es3.total_shells != shells ||
      !aes2.sealed() || aes2.batch_size() != batch || aes2.total_atoms() != atoms ||
      !mulliken.sealed() || mulliken.batch_size() != batch || mulliken.total_atoms() != atoms ||
      mulliken.total_shells() != shells || mulliken.total_orbitals() != orbitals ||
      mulliken.matrix_elements() != matrices || !mixer.sealed() ||
      !mixer.matches_wavefunction_layout(wavefunction) || !driver.sealed() ||
      driver.batch_size() != batch || driver.maximum_iterations() == 0u) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan, Field::kRequiredPlans);
  }
  if (!vector_matches(basis.atom_offsets, host_topology.atom_offsets, batch + 1) ||
      !vector_matches(basis.batch_shell_offsets, host_topology.batch_shell_offsets, batch + 1) ||
      !vector_matches(basis.batch_orbital_offsets, host_topology.batch_orbital_offsets,
                      batch + 1) ||
      !vector_matches(integrals.matrix_offsets, host_topology.matrix_offsets, batch + 1) ||
      !vector_matches(basis.atom_shell_offsets, host_topology.atom_shell_offsets, atoms + 1) ||
      !vector_matches(basis.shell_orbital_offsets, host_topology.shell_orbital_offsets,
                      shells + 1) ||
      !vector_matches(basis.shell_to_atom, host_topology.shell_to_atom, shells) ||
      wavefunction.atom_offsets != basis.atom_offsets ||
      wavefunction.batch_shell_offsets != basis.batch_shell_offsets ||
      wavefunction.batch_orbital_offsets != basis.batch_orbital_offsets ||
      es2.atom_offsets() != basis.atom_offsets ||
      es2.batch_shell_offsets() != basis.batch_shell_offsets ||
      es2.atom_shell_offsets() != basis.atom_shell_offsets ||
      es2.shell_to_atom() != basis.shell_to_atom || aes2.atom_offsets() != basis.atom_offsets ||
      mulliken.atom_offsets() != basis.atom_offsets ||
      mulliken.batch_shell_offsets() != basis.batch_shell_offsets ||
      mulliken.batch_orbital_offsets() != basis.batch_orbital_offsets ||
      mulliken.matrix_offsets() != integrals.matrix_offsets ||
      mulliken.shell_orbital_offsets() != basis.shell_orbital_offsets ||
      mulliken.shell_to_atom() != basis.shell_to_atom) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan, Field::kRequiredPlans);
  }
  if (!exact_vector(aes2.pair_offsets(), batch + 1) || aes2.pair_offsets().front() != 0 ||
      aes2.total_pairs() < 0 || aes2.pair_offsets().back() != aes2.total_pairs()) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kGeometry);
  }
  if (!exact_vector(es2.matrix_offsets(), batch + 1) || es2.matrix_offsets().front() != 0 ||
      es2.total_matrix_elements() <= 0 ||
      es2.matrix_offsets().back() != es2.total_matrix_elements()) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kES2);
  }

  std::int64_t atom_coordinates = 0;
  std::int64_t geometry_pair_elements = 0;
  std::int64_t aes2_pair_elements = 0;
  if (!checked_multiply(atoms, 3, atom_coordinates) ||
      !checked_multiply(aes2.total_pairs(), kGfn2GeometryPairDataElements,
                        geometry_pair_elements) ||
      !checked_multiply(aes2.total_pairs(), kGfn2AES2PairDataElements, aes2_pair_elements)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCountOverflow, Field::kGeometry);
  }
  if (sources.geometry_generation == 0u || !exact_array(sources.atomic_numbers, atoms) ||
      !exact_array(sources.positions, atom_coordinates) ||
      !exact_array(sources.covalent_radii, atoms) ||
      !exact_array(sources.geometry_cache.pair_data, geometry_pair_elements) ||
      !exact_array(sources.geometry_cache.coordination_numbers, atoms) ||
      !exact_array(sources.geometry_cache.system_generations, batch) ||
      !std::all_of(
          sources.geometry_cache.system_generations.data,
          sources.geometry_cache.system_generations.data + batch,
          [&](std::uint64_t generation) { return generation == sources.geometry_generation; })) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kGeometry);
  }
  if (!exact_array(sources.h0, matrices) || !exact_array(sources.overlap, matrices) ||
      !exact_array(sources.dipole_integrals, dipole_integral_elements) ||
      !exact_array(sources.quadrupole_integrals, quadrupole_integral_elements)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kHamiltonian);
  }
  if (!exact_array(sources.es2_cache.coulomb_matrix, es2.total_matrix_elements()) ||
      !exact_array(sources.aes2_cache.pair_data, aes2_pair_elements)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kES2);
  }
  const bool warm = sources.warm_start_generation != 0u;
  if ((warm && !exact_array(sources.warm_start_generations, batch)) ||
      (!warm && (sources.warm_start_generations.data != nullptr ||
                 sources.warm_start_generations.elements != 0))) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kGeometry);
  }
  const bool valid_eigensolver_strategy =
      sources.eigensolver_options.strategy == Gfn2EigensolverStrategy::kAuto ||
      sources.eigensolver_options.strategy == Gfn2EigensolverStrategy::kBatchedDivideAndConquer ||
      sources.eigensolver_options.strategy == Gfn2EigensolverStrategy::kBatchedJacobi ||
      sources.eigensolver_options.strategy == Gfn2EigensolverStrategy::kTridiagonalBisection;
  if (!std::isfinite(sources.eigensolver_options.minimum_overlap_rcond) ||
      !(sources.eigensolver_options.minimum_overlap_rcond > 0.0) ||
      !std::isfinite(sources.eigensolver_options.symmetry_tolerance) ||
      sources.eigensolver_options.symmetry_tolerance < 0.0 || !valid_eigensolver_strategy ||
      sources.eigensolver_options.jacobi != nullptr) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kEigensolver);
  }

  const bool d4_enabled = sources.d4.plan != nullptr;
  const bool point_enabled = sources.point_charges.plan != nullptr;
  const bool periodic_enabled = sources.periodic.plan != nullptr;
  if (driver.d4_enabled() != d4_enabled ||
      driver.periodic_embedding_enabled() != periodic_enabled) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan,
                   d4_enabled ? Field::kD4 : Field::kPeriodic);
  }
  if (d4_enabled) {
    const auto& d4 = *sources.d4.plan;
    if (!d4.sealed() || d4.batch_size() != batch || d4.total_atoms() != atoms ||
        d4.total_pairs() != aes2.total_pairs() || d4.pair_offsets() != aes2.pair_offsets() ||
        !d4.matches_atomic_numbers(sources.atomic_numbers.data) ||
        !exact_array(sources.d4.coordination_numbers, atoms) || sources.d4.elements.elements <= 0 ||
        sources.d4.references.elements <= 0 ||
        !exact_array(sources.d4.elements, sources.d4.elements.elements) ||
        !exact_array(sources.d4.references, sources.d4.references.elements)) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kD4);
    }
    std::int64_t reference_square = 0;
    if (!checked_multiply(sources.d4.references.elements, sources.d4.references.elements,
                          reference_square) ||
        sources.d4.reference_c6.elements < reference_square ||
        sources.d4.reference_c6.data == nullptr) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kD4);
    }
  } else if (sources.d4.elements.data != nullptr || sources.d4.elements.elements != 0 ||
             sources.d4.references.data != nullptr || sources.d4.references.elements != 0 ||
             sources.d4.reference_c6.data != nullptr || sources.d4.reference_c6.elements != 0 ||
             sources.d4.coordination_numbers.data != nullptr ||
             sources.d4.coordination_numbers.elements != 0) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kD4);
  }

  std::int64_t point_count = 0;
  if (point_enabled) {
    const auto& point = *sources.point_charges.plan;
    point_count = point.total_point_charges;
    std::int64_t point_coordinates = 0;
    if (point.batch_size != batch || point.total_atoms != atoms || point.total_shells != shells) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan, Field::kPointCharges);
    }
    if (point.atom_offsets != basis.atom_offsets ||
        point.batch_shell_offsets != basis.batch_shell_offsets ||
        point.shell_to_atom != basis.shell_to_atom) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan, Field::kPointCharges);
    }
    if (point_count < 0 || !checked_multiply(point_count, 3, point_coordinates) ||
        !valid_offsets(point.point_charge_offsets, batch, point_count) ||
        !exact_vector(point.shell_hardness, shells) ||
        !exact_array(sources.point_charges.positions, point_coordinates) ||
        !exact_array(sources.point_charges.charges, point_count) ||
        !exact_array(sources.point_charges.hardnesses, point_count) ||
        !exact_array(sources.point_charges.shell_potential_cache, shells)) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kPointCharges);
    }
  } else if (sources.point_charges.positions.data != nullptr ||
             sources.point_charges.positions.elements != 0 ||
             sources.point_charges.charges.data != nullptr ||
             sources.point_charges.charges.elements != 0 ||
             sources.point_charges.hardnesses.data != nullptr ||
             sources.point_charges.hardnesses.elements != 0 ||
             sources.point_charges.shell_potential_cache.data != nullptr ||
             sources.point_charges.shell_potential_cache.elements != 0) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kPointCharges);
  }

  std::int64_t periodic_matrices = 0;
  if (periodic_enabled) {
    const auto& periodic = *sources.periodic.plan;
    periodic_matrices = periodic.total_matrix_elements();
    if (!periodic.sealed() || periodic.batch_size() != batch || periodic.total_atoms() != atoms) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan, Field::kPeriodic);
    }
    if (periodic.atom_offsets() != basis.atom_offsets) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan, Field::kPeriodic);
    }
    if (!exact_vector(periodic.atom_offsets(), batch + 1) ||
        !exact_vector(periodic.matrix_offsets(), batch + 1) ||
        !exact_array(sources.periodic.shifts, atoms) ||
        !exact_array(sources.periodic.response_matrices, periodic_matrices)) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kPeriodic);
    }
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int64_t system_atoms = basis.atom_offsets[static_cast<std::size_t>(system + 1)] -
                                        basis.atom_offsets[static_cast<std::size_t>(system)];
      std::int64_t expected_matrix_elements = 0;
      if (!checked_multiply(system_atoms, system_atoms, expected_matrix_elements) ||
          periodic.matrix_offsets()[static_cast<std::size_t>(system + 1)] -
                  periodic.matrix_offsets()[static_cast<std::size_t>(system)] !=
              expected_matrix_elements) {
        return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan, Field::kPeriodic,
                       system);
      }
    }
  } else if (sources.periodic.shifts.data != nullptr || sources.periodic.shifts.elements != 0 ||
             sources.periodic.response_matrices.data != nullptr ||
             sources.periodic.response_matrices.elements != 0) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kPeriodic);
  }

  try {
    /* Spin constants are chemistry- and shell-order-dependent. Build them
     * from the same sealed basis/wavefunction pair instead of accepting a
     * second caller-provided topology authority. */
    gfn2::SpinPolarizationPlan spin;
    std::string spin_error;
    const xtbloom_status_t spin_status =
        gfn2::make_spin_polarization_plan(basis, wavefunction, spin, spin_error);
    if (spin_status != XTBLOOM_STATUS_SUCCESS) {
      return failure(spin_status, Error::kInvalidSource, Field::kRequiredPlans);
    }

    std::unique_ptr<Impl> candidate(new (std::nothrow) Impl());
    if (candidate == nullptr) {
      return failure(XTBLOOM_STATUS_ALLOCATION_FAILED, Error::kAllocationFailed, Field::kArena);
    }
    candidate->plan_token = plan_token;
    candidate->geometry_generation = sources.geometry_generation;
    candidate->warm_start_generation = sources.warm_start_generation;
    candidate->batch_size = batch;
    candidate->total_atoms = atoms;
    candidate->total_shells = shells;
    candidate->total_orbitals = orbitals;
    candidate->total_matrix_elements = matrices;
    candidate->es2_matrix_elements = es2.total_matrix_elements();
    candidate->total_pairs = aes2.total_pairs();
    candidate->minimum_atoms = minimum_partition(host_topology.atom_offsets, batch);
    candidate->maximum_atoms = maximum_partition(host_topology.atom_offsets, batch);
    candidate->maximum_shells = maximum_partition(host_topology.batch_shell_offsets, batch);
    candidate->mixer_history = mixer.history_size();
    candidate->maximum_iterations = driver.maximum_iterations();
    candidate->mixer_damping = mixer.damping();
    candidate->residual_tolerance = mixer.rms_tolerance();
    candidate->mixer_maximum_tolerance = mixer.maximum_tolerance();
    candidate->energy_tolerance = driver.energy_tolerance();
    candidate->electronic_temperature = driver.electronic_temperature();
    candidate->eigensolver_options = sources.eigensolver_options;
    candidate->host_topology = host_topology;
    candidate->host_wavefunction.memory_space = Gfn2PlanMemorySpace::kHost;
    candidate->host_wavefunction.plan_token = plan_token;
    candidate->host_wavefunction.batch_size = batch;
    candidate->host_wavefunction.total_spin_channels = std::accumulate(
        wavefunction.spin_channels.begin(), wavefunction.spin_channels.end(), std::int64_t{0});
    candidate->host_wavefunction.total_spin_orbitals = wavefunction.eigenvalues.element_count;
    candidate->host_wavefunction.total_spin_matrix_elements = wavefunction.density.element_count;
    candidate->host_wavefunction.total_spin_shells = wavefunction.qsh.element_count;
    candidate->host_wavefunction.total_spin_atoms = wavefunction.qat.element_count;
    candidate->host_wavefunction.spin_channel_count = batch;
    candidate->host_wavefunction.spin_channel_offset_count = batch + 1;
    candidate->host_wavefunction.spin_orbital_offset_count = batch + 1;
    candidate->host_wavefunction.spin_matrix_offset_count = batch + 1;
    candidate->host_wavefunction.spin_shell_offset_count = batch + 1;
    candidate->host_wavefunction.spin_atom_offset_count = batch + 1;
    std::vector<std::int64_t> spin_channel_offsets(static_cast<std::size_t>(batch) + 1u, 0);
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int32_t channels = wavefunction.spin_channels[static_cast<std::size_t>(system)];
      if ((channels != 1 && channels != 2) ||
          spin_channel_offsets[static_cast<std::size_t>(system)] >
              std::numeric_limits<std::int64_t>::max() - channels) {
        return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource,
                       Field::kRequiredPlans, system);
      }
      spin_channel_offsets[static_cast<std::size_t>(system + 1)] =
          spin_channel_offsets[static_cast<std::size_t>(system)] + channels;
    }
    Gfn2WavefunctionLayoutView fingerprint_view = candidate->host_wavefunction;
    fingerprint_view.spin_channels = wavefunction.spin_channels.data();
    fingerprint_view.spin_channel_offsets = spin_channel_offsets.data();
    fingerprint_view.spin_orbital_offsets = wavefunction.eigenvalues.system_offsets.data();
    fingerprint_view.spin_matrix_offsets = wavefunction.density.system_offsets.data();
    fingerprint_view.spin_shell_offsets = wavefunction.qsh.system_offsets.data();
    fingerprint_view.spin_atom_offsets = wavefunction.qat.system_offsets.data();
    candidate->host_wavefunction.layout_fingerprint =
        gfn2_wavefunction_layout_fingerprint_host(fingerprint_view);
    if (candidate->host_wavefunction.layout_fingerprint == 0u) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kRequiredPlans);
    }
    candidate->spin_coupling_matrix_count = spin.coupling_offsets.back();
    std::int64_t spin_atom_multipoles = 0;
    if (!checked_multiply(candidate->host_wavefunction.total_spin_atoms, 9, spin_atom_multipoles) ||
        candidate->host_wavefunction.total_spin_shells >
            std::numeric_limits<std::int64_t>::max() - spin_atom_multipoles) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCountOverflow, Field::kRequiredPlans);
    }
    candidate->mixer_vector_elements =
        candidate->host_wavefunction.total_spin_shells + spin_atom_multipoles;
    candidate->d4_enabled = d4_enabled;
    candidate->point_enabled = point_enabled;
    candidate->periodic_enabled = periodic_enabled;
    candidate->point_count = point_count;
    candidate->periodic_matrix_elements = periodic_matrices;
    candidate->enabled_components =
        static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
        static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
        static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2) |
        (d4_enabled ? static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kD4TwoBody) : 0u) |
        (point_enabled ? static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kExplicitPointCharge)
                       : 0u) |
        (periodic_enabled
             ? static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kPeriodicEmbedding)
             : 0u);

    if (d4_enabled) {
      candidate->layout.d4_elements.elements = sources.d4.elements.elements;
      candidate->layout.d4_references.elements = sources.d4.references.elements;
      candidate->layout.d4_reference_c6.elements = sources.d4.reference_c6.elements;
    }
    if (!candidate->make_layout()) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCountOverflow, Field::kArena);
    }

    cudaError_t cuda_status =
        cudaMallocHost(&candidate->upload_image, candidate->layout.total_bytes);
    if (cuda_status != cudaSuccess) {
      Gfn2SccSetupInputsDiagnostic diagnostic = failure(
          cuda_status == cudaErrorMemoryAllocation ? XTBLOOM_STATUS_ALLOCATION_FAILED
                                                   : XTBLOOM_STATUS_INTERNAL_ERROR,
          cuda_status == cudaErrorMemoryAllocation ? Error::kAllocationFailed : Error::kCudaError,
          Field::kArena);
      diagnostic.required_bytes = candidate->layout.total_bytes;
      diagnostic.cuda_status = cuda_status;
      return diagnostic;
    }
    auto* const image = static_cast<std::byte*>(candidate->upload_image);
    std::memset(image, 0, candidate->layout.total_bytes);

    Gfn2SccSetupHostArray<std::int64_t> pair_offsets{aes2.pair_offsets().data(), batch + 1};
    std::vector<std::int64_t> physical_dipole_offsets(static_cast<std::size_t>(batch) + 1u);
    std::vector<std::int64_t> physical_quadrupole_offsets(static_cast<std::size_t>(batch) + 1u);
    for (std::int64_t system = 0; system <= batch; ++system) {
      if (!checked_multiply(basis.atom_offsets[static_cast<std::size_t>(system)], 3,
                            physical_dipole_offsets[static_cast<std::size_t>(system)]) ||
          !checked_multiply(basis.atom_offsets[static_cast<std::size_t>(system)], 6,
                            physical_quadrupole_offsets[static_cast<std::size_t>(system)])) {
        return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCountOverflow, Field::kHamiltonian);
      }
    }
    Gfn2SccSetupHostArray<std::int64_t> dipole_offsets{physical_dipole_offsets.data(), batch + 1};
    Gfn2SccSetupHostArray<std::int64_t> quadrupole_offsets{physical_quadrupole_offsets.data(),
                                                           batch + 1};
    Gfn2SccSetupHostArray<double> es2_hardness{es2.shell_hardness().data(), shells};
    Gfn2SccSetupHostArray<std::int64_t> es2_matrix_offsets{es2.matrix_offsets().data(), batch + 1};
    Gfn2SccSetupHostArray<double> es3_gamma{es3.shell_gamma3.data(), shells};
    Gfn2SccSetupHostArray<double> aes2_dipole{aes2.dipole_kernel().data(), atoms};
    Gfn2SccSetupHostArray<double> aes2_quadrupole{aes2.quadrupole_kernel().data(), atoms};
    Gfn2SccSetupHostArray<double> aes2_radius{aes2.multipole_radius().data(), atoms};
    Gfn2SccSetupHostArray<double> aes2_valence{aes2.multipole_valence_cn().data(), atoms};
    Gfn2SccSetupHostArray<double> reference_occupations{
        mulliken.reference_shell_occupations().data(), shells};
    std::vector<double> electron_counts(static_cast<std::size_t>(2 * batch));
    std::vector<double> temperatures(static_cast<std::size_t>(batch),
                                     driver.electronic_temperature());
    for (std::int64_t system = 0; system < batch; ++system) {
      electron_counts[static_cast<std::size_t>(2 * system)] =
          wavefunction.alpha_electron_counts[static_cast<std::size_t>(system)];
      electron_counts[static_cast<std::size_t>(2 * system + 1)] =
          wavefunction.beta_electron_counts[static_cast<std::size_t>(system)];
    }

    pack_array(image, candidate->layout.pair_offsets, pair_offsets);
    pack_array(image, candidate->layout.covalent_radii, sources.covalent_radii);
    pack_array(image, candidate->layout.positions, sources.positions);
    pack_array(image, candidate->layout.atomic_numbers, sources.atomic_numbers);
    /* Element identity is setup-owned term-specific input: seal the exact
     * atomic-number ordering and plan token now so the device plan can borrow
     * a CUDA element-identity projection without re-deriving atomic numbers. */
    if (project_gfn2_element_identity_projection_host(sources.atomic_numbers.data,
                                                      sources.atomic_numbers.elements, plan_token,
                                                      candidate->host_element_identity)
            .error != Gfn2PlanSchemaError::kSuccess) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kD4);
    }
    pack_array(image, candidate->layout.h0, sources.h0);
    pack_array(image, candidate->layout.overlap, sources.overlap);
    pack_array(image, candidate->layout.dipole_integrals, sources.dipole_integrals);
    pack_array(image, candidate->layout.quadrupole_integrals, sources.quadrupole_integrals);
    pack_vector(image, candidate->layout.spin_coupling_offsets, spin.coupling_offsets);
    pack_vector(image, candidate->layout.spin_coupling_matrices, spin.coupling_matrices);
    pack_array(image, candidate->layout.dipole_offsets, dipole_offsets);
    pack_array(image, candidate->layout.quadrupole_offsets, quadrupole_offsets);
    pack_array(image, candidate->layout.es2_matrix_offsets, es2_matrix_offsets);
    pack_array(image, candidate->layout.es2_shell_hardness, es2_hardness);
    pack_array(image, candidate->layout.es3_shell_gamma3, es3_gamma);
    pack_array(image, candidate->layout.aes2_dipole_kernel, aes2_dipole);
    pack_array(image, candidate->layout.aes2_quadrupole_kernel, aes2_quadrupole);
    pack_array(image, candidate->layout.aes2_multipole_radius, aes2_radius);
    pack_array(image, candidate->layout.aes2_multipole_valence_cn, aes2_valence);
    pack_array(image, candidate->layout.mulliken_reference_occupations, reference_occupations);
    pack_vector(image, candidate->layout.electron_counts, electron_counts);
    pack_vector(image, candidate->layout.temperatures, temperatures);
    pack_array(image, candidate->layout.geometry_pair_data, sources.geometry_cache.pair_data);
    pack_array(image, candidate->layout.geometry_coordination,
               sources.geometry_cache.coordination_numbers);
    pack_array(image, candidate->layout.geometry_generations,
               sources.geometry_cache.system_generations);
    pack_array(image, candidate->layout.es2_coulomb_matrix, sources.es2_cache.coulomb_matrix);
    pack_array(image, candidate->layout.aes2_pair_data, sources.aes2_cache.pair_data);
    if (d4_enabled) {
      pack_array(image, candidate->layout.d4_elements, sources.d4.elements);
      pack_array(image, candidate->layout.d4_references, sources.d4.references);
      pack_array(image, candidate->layout.d4_reference_c6, sources.d4.reference_c6);
      pack_array(image, candidate->layout.d4_coordination, sources.d4.coordination_numbers);
    }
    if (point_enabled) {
      Gfn2SccSetupHostArray<std::int64_t> point_offsets{
          sources.point_charges.plan->point_charge_offsets.data(), batch + 1};
      Gfn2SccSetupHostArray<double> point_shell_hardness{
          sources.point_charges.plan->shell_hardness.data(), shells};
      pack_array(image, candidate->layout.point_charge_offsets, point_offsets);
      pack_array(image, candidate->layout.point_shell_hardness, point_shell_hardness);
      pack_array(image, candidate->layout.point_positions, sources.point_charges.positions);
      pack_array(image, candidate->layout.point_charges, sources.point_charges.charges);
      pack_array(image, candidate->layout.point_hardnesses, sources.point_charges.hardnesses);
      pack_array(image, candidate->layout.point_shell_cache,
                 sources.point_charges.shell_potential_cache);
    }
    if (periodic_enabled) {
      Gfn2SccSetupHostArray<std::int64_t> periodic_offsets{
          sources.periodic.plan->matrix_offsets().data(), batch + 1};
      pack_array(image, candidate->layout.periodic_matrix_offsets, periodic_offsets);
      pack_array(image, candidate->layout.periodic_shifts, sources.periodic.shifts);
      pack_array(image, candidate->layout.periodic_response, sources.periodic.response_matrices);
    }
    if (warm) {
      pack_array(image, candidate->layout.warm_start_generations, sources.warm_start_generations);
    }

    Gfn2SccSetupInputs replacement;
    replacement.impl_ = std::move(candidate);
    output = std::move(replacement);
    return {};
  } catch (const std::bad_alloc&) {
    return failure(XTBLOOM_STATUS_ALLOCATION_FAILED, Error::kAllocationFailed, Field::kArena);
  } catch (...) {
    return failure(XTBLOOM_STATUS_INTERNAL_ERROR, Error::kInvalidSource, Field::kRequiredPlans);
  }
}

bool Gfn2SccSetupInputs::valid() const noexcept { return impl_ != nullptr; }

std::size_t Gfn2SccSetupInputs::retained_host_bytes() const noexcept {
  if (impl_ == nullptr) return 0u;
  return sizeof(*impl_) + (impl_->upload_image == nullptr ? 0u : impl_->layout.total_bytes);
}

Gfn2SccSetupInputsRequirements Gfn2SccSetupInputs::requirements() const noexcept {
  Gfn2SccSetupInputsRequirements result{};
  if (impl_ != nullptr) {
    result.device_bytes = impl_->layout.total_bytes;
  }
  return result;
}

Gfn2SccSetupInputsDiagnostic Gfn2SccSetupInputs::bind_device_arena_and_upload_async(
    const Gfn2RaggedTopologyView& device_topology,
    const Gfn2WavefunctionLayoutView& device_wavefunction, void* device_arena,
    std::size_t device_arena_bytes, Gfn2SccIterationDevicePlan& plan_seed,
    Gfn2SccIterationDeviceInput& input_seed, cudaStream_t stream) const noexcept {
  using Error = Gfn2SccSetupInputsError;
  using Field = Gfn2SccSetupInputsField;
  if (impl_ == nullptr) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kRequiredPlans);
  }
  if (device_topology.memory_space != Gfn2PlanMemorySpace::kCudaDevice ||
      validate_gfn2_topology_binding(device_topology, Gfn2PlanMemorySpace::kCudaDevice).error !=
          Gfn2PlanSchemaError::kSuccess ||
      !same_shape(impl_->host_topology, device_topology)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan, Field::kTopology);
  }
  if (validate_gfn2_wavefunction_layout_binding(device_topology, device_wavefunction,
                                                Gfn2PlanMemorySpace::kCudaDevice)
              .error != Gfn2PlanSchemaError::kSuccess ||
      !same_wavefunction_shape(impl_->host_wavefunction, device_wavefunction)) {
    /* Publication, mixer, and all spin-aware leaves must share the descriptor
     * transaction published by the topology owner. */
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan, Field::kRequiredPlans);
  }
  const std::size_t required = impl_->layout.total_bytes;
  if (device_arena == nullptr) {
    return arena_failure(Error::kNullArena, required);
  }
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(device_arena);
  if (address % kDeviceAlignment != 0u) {
    return arena_failure(Error::kMisalignedArena, required);
  }
  if (device_arena_bytes < required) {
    return arena_failure(Error::kInsufficientArena, required);
  }
  if (required != 0u && required - 1u > std::numeric_limits<std::uintptr_t>::max() - address) {
    return arena_failure(Error::kCountOverflow, required);
  }
  cudaPointerAttributes attributes{};
  const cudaError_t attribute_status = cudaPointerGetAttributes(&attributes, device_arena);
  if (attribute_status != cudaSuccess) {
    (void)cudaGetLastError();
    return arena_failure(Error::kInvalidArenaMemory, required, attribute_status);
  }
  if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) {
    return arena_failure(Error::kInvalidArenaMemory, required);
  }

  auto* const arena = static_cast<std::byte*>(device_arena);
  const auto cptr = [arena](const Segment& segment, auto* type) {
    using T = std::remove_pointer_t<decltype(type)>;
    return const_device_pointer<T>(arena, segment);
  };
  const auto mptr = [arena](const Segment& segment, auto* type) {
    using T = std::remove_pointer_t<decltype(type)>;
    return device_pointer<T>(arena, segment);
  };
  const std::int64_t batch = impl_->batch_size;
  const std::int64_t atoms = impl_->total_atoms;
  const std::int64_t shells = impl_->total_shells;
  const std::int64_t orbitals = impl_->total_orbitals;
  const std::int64_t matrices = impl_->total_matrix_elements;
  const std::uint64_t token = impl_->plan_token;

  Gfn2SccIterationDevicePlan candidate{};
  candidate.abi_version = kGfn2SccIterationAbiVersion;
  candidate.enabled_components = impl_->enabled_components;
  candidate.plan_token = token;
  candidate.geometry_generation = impl_->geometry_generation;
  candidate.topology = device_topology;
  candidate.wavefunction_layout = device_wavefunction;
  /* ABI v3: publish the sealed common projections as the plan's sole borrowing
   * authority.  Every leaf below already subscribes these exact arrays; the
   * iteration binding validator proves each leaf's pointer/count identity
   * against these projections once instead of re-deriving the master.  The
   * binders return success and clear the projection on rejection, so the
   * published-token check below is the fail-closed gate. */
  static_cast<void>(bind_gfn2_atom_projection_cuda(device_topology, candidate.atom_projection));
  static_cast<void>(bind_gfn2_shell_ownership_projection_cuda(
      device_topology, candidate.shell_ownership_projection));
  static_cast<void>(
      bind_gfn2_ao_matrix_projection_cuda(device_topology, candidate.ao_matrix_projection));
  if (device_topology.pair_map_kind == Gfn2PairMapKind::kPackedLowerTriangle) {
    static_cast<void>(bind_gfn2_packed_all_pair_projection_cuda(
        device_topology, candidate.packed_all_pair_projection));
  }
  if (device_topology.bucket_count != 0) {
    static_cast<void>(
        bind_gfn2_ao_bucket_projection_cuda(device_topology, candidate.ao_bucket_projection));
  }
  /* The topology binders clear the projection on rejection and return success,
   * so setup must fail closed when any sealed projection is not published. */
  if (candidate.atom_projection.plan_token != token ||
      candidate.shell_ownership_projection.plan_token != token ||
      candidate.ao_matrix_projection.plan_token != token ||
      (device_topology.pair_map_kind == Gfn2PairMapKind::kPackedLowerTriangle &&
       candidate.packed_all_pair_projection.plan_token != token) ||
      (device_topology.bucket_count != 0 && candidate.ao_bucket_projection.plan_token != token)) {
    return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kCrossPlan, Field::kTopology);
  }
  /* Element identity is setup-owned: reuse the host seal but name the uploaded
   * device atomic-number array for the CUDA descriptor. */
  {
    const std::int32_t* device_atomic_numbers =
        impl_->layout.atomic_numbers.elements == 0
            ? nullptr
            : cptr(impl_->layout.atomic_numbers, static_cast<std::int32_t*>(nullptr));
    static_cast<void>(bind_gfn2_element_identity_projection_cuda(
        impl_->host_element_identity, device_atomic_numbers,
        candidate.element_identity_projection));
    if (candidate.element_identity_projection.plan_token != token ||
        impl_->host_element_identity.element_fingerprint == 0u) {
      return failure(XTBLOOM_STATUS_INVALID_ARGUMENT, Error::kInvalidSource, Field::kD4);
    }
  }
  candidate.activity_policy = {batch, impl_->maximum_iterations, token};
  candidate.state_policy = {impl_->maximum_iterations, impl_->residual_tolerance,
                            impl_->energy_tolerance, token};
  candidate.mixer_policy = {impl_->mixer_history, impl_->mixer_damping, impl_->residual_tolerance,
                            impl_->mixer_maximum_tolerance, token};

  candidate.geometry_batch = {batch,
                              atoms,
                              impl_->total_pairs,
                              batch + 1,
                              batch + 1,
                              atoms,
                              3 * atoms,
                              token,
                              device_topology.atom_offsets,
                              cptr(impl_->layout.pair_offsets, static_cast<std::int64_t*>(nullptr)),
                              cptr(impl_->layout.covalent_radii, static_cast<double*>(nullptr))};
  candidate.geometry_cache = {
      mptr(impl_->layout.geometry_pair_data, static_cast<double*>(nullptr)),
      impl_->layout.geometry_pair_data.elements,
      mptr(impl_->layout.geometry_coordination, static_cast<double*>(nullptr)),
      atoms,
      mptr(impl_->layout.geometry_generations, static_cast<std::uint64_t*>(nullptr)),
      batch,
      token};
  candidate.scc_batch = {batch,
                         shells,
                         atoms,
                         batch + 1,
                         batch + 1,
                         token,
                         device_topology.batch_shell_offsets,
                         device_topology.atom_offsets};
  candidate.spin_batch = {
      batch,
      atoms,
      shells,
      device_wavefunction.total_spin_shells,
      batch + 1,
      batch + 1,
      atoms + 1,
      batch + 1,
      batch,
      atoms + 1,
      impl_->spin_coupling_matrix_count,
      token,
      device_topology.atom_offsets,
      device_topology.batch_shell_offsets,
      device_topology.atom_shell_offsets,
      device_wavefunction.spin_shell_offsets,
      device_wavefunction.spin_channels,
      cptr(impl_->layout.spin_coupling_offsets, static_cast<std::int64_t*>(nullptr)),
      cptr(impl_->layout.spin_coupling_matrices, static_cast<double*>(nullptr))};
  candidate.potential_batch = {
      batch,
      atoms,
      shells,
      token,
      batch + 1,
      batch + 1,
      batch + 1,
      batch + 1,
      batch + 1,
      batch + 1,
      shells,
      device_topology.atom_offsets,
      device_topology.batch_shell_offsets,
      device_topology.batch_shell_offsets,
      device_topology.atom_offsets,
      cptr(impl_->layout.dipole_offsets, static_cast<std::int64_t*>(nullptr)),
      cptr(impl_->layout.quadrupole_offsets, static_cast<std::int64_t*>(nullptr)),
      device_topology.shell_to_atom};
  candidate.es2_batch = {
      batch,
      atoms,
      shells,
      impl_->es2_matrix_elements,
      token,
      batch + 1,
      batch + 1,
      atoms + 1,
      batch + 1,
      shells,
      shells,
      device_topology.atom_offsets,
      device_topology.batch_shell_offsets,
      device_topology.atom_shell_offsets,
      cptr(impl_->layout.es2_matrix_offsets, static_cast<std::int64_t*>(nullptr)),
      device_topology.shell_to_atom,
      cptr(impl_->layout.es2_shell_hardness, static_cast<double*>(nullptr))};
  candidate.es2_cache = {mptr(impl_->layout.es2_coulomb_matrix, static_cast<double*>(nullptr)),
                         impl_->es2_matrix_elements, impl_->geometry_generation, token};
  candidate.es3_batch = {batch,
                         shells,
                         batch + 1,
                         shells,
                         device_topology.batch_shell_offsets,
                         cptr(impl_->layout.es3_shell_gamma3, static_cast<double*>(nullptr)),
                         token};
  candidate.aes2_batch = {
      batch,
      atoms,
      impl_->total_pairs,
      token,
      batch + 1,
      batch + 1,
      atoms,
      atoms,
      atoms,
      atoms,
      device_topology.atom_offsets,
      cptr(impl_->layout.pair_offsets, static_cast<std::int64_t*>(nullptr)),
      cptr(impl_->layout.aes2_dipole_kernel, static_cast<double*>(nullptr)),
      cptr(impl_->layout.aes2_quadrupole_kernel, static_cast<double*>(nullptr)),
      cptr(impl_->layout.aes2_multipole_radius, static_cast<double*>(nullptr)),
      cptr(impl_->layout.aes2_multipole_valence_cn, static_cast<double*>(nullptr))};
  candidate.aes2_cache = {mptr(impl_->layout.aes2_pair_data, static_cast<double*>(nullptr)),
                          impl_->layout.aes2_pair_data.elements, impl_->geometry_generation, token};

  if (impl_->d4_enabled) {
    candidate.d4_batch = {
        batch,
        atoms,
        impl_->total_pairs,
        token,
        gfn2_d4_atomic_number_hash(static_cast<const std::int32_t*>(static_cast<const void*>(
                                       static_cast<std::byte*>(impl_->upload_image) +
                                       impl_->layout.atomic_numbers.offset)),
                                   atoms),
        device_topology.atom_offsets,
        cptr(impl_->layout.pair_offsets, static_cast<std::int64_t*>(nullptr)),
        cptr(impl_->layout.atomic_numbers, static_cast<std::int32_t*>(nullptr)),
        impl_->minimum_atoms};
    candidate.d4_parameters = {
        cptr(impl_->layout.d4_elements, static_cast<Gfn2D4DeviceElementData*>(nullptr)),
        impl_->layout.d4_elements.elements,
        cptr(impl_->layout.d4_references, static_cast<Gfn2D4DeviceReferenceData*>(nullptr)),
        impl_->layout.d4_references.elements,
        cptr(impl_->layout.d4_reference_c6, static_cast<double*>(nullptr)),
        impl_->layout.d4_reference_c6.elements};
    /* Setup owns the stable CN outlet address, while runtime supplies the
     * final provenance and committed role projections after preprocessing has
     * been bound. The setup positions support fixed-geometry standalone SCC
     * composition tests; production runtime replaces them with the exact
     * committed-position outlet before validation. */
    candidate.d4_pairlist_cache = {
        cptr(impl_->layout.positions, static_cast<double*>(nullptr)),
        3 * atoms,
        mptr(impl_->layout.d4_coordination, static_cast<double*>(nullptr)),
        atoms,
        nullptr,
        0,
        nullptr,
        0,
        {},
        {},
        {},
        token};
  }
  if (impl_->point_enabled) {
    candidate.explicit_point_charge_batch = {
        batch,
        atoms,
        shells,
        impl_->point_count,
        device_topology.atom_offsets,
        device_topology.batch_shell_offsets,
        cptr(impl_->layout.point_charge_offsets, static_cast<std::int64_t*>(nullptr)),
        device_topology.shell_to_atom,
        cptr(impl_->layout.point_shell_hardness, static_cast<double*>(nullptr)),
        cptr(impl_->layout.positions, static_cast<double*>(nullptr)),
        cptr(impl_->layout.point_positions, static_cast<double*>(nullptr)),
        cptr(impl_->layout.point_charges, static_cast<double*>(nullptr)),
        cptr(impl_->layout.point_hardnesses, static_cast<double*>(nullptr)),
        token};
    candidate.explicit_point_charge_cache = {
        mptr(impl_->layout.point_shell_cache, static_cast<double*>(nullptr)), shells,
        impl_->geometry_generation, token};
  }
  if (impl_->periodic_enabled) {
    candidate.periodic_batch = {
        batch,
        atoms,
        impl_->periodic_matrix_elements,
        batch + 1,
        batch + 1,
        atoms,
        impl_->periodic_matrix_elements,
        token,
        device_topology.atom_offsets,
        cptr(impl_->layout.periodic_matrix_offsets, static_cast<std::int64_t*>(nullptr)),
        cptr(impl_->layout.periodic_shifts, static_cast<double*>(nullptr)),
        cptr(impl_->layout.periodic_response, static_cast<double*>(nullptr)),
        impl_->geometry_generation};
  }

  candidate.scalar_bridge_batch = {device_topology, batch + 1, batch + 1,
                                   device_topology.batch_shell_offsets,
                                   device_topology.atom_offsets};
  candidate.hamiltonian_batch = {batch,
                                 atoms,
                                 shells,
                                 orbitals,
                                 matrices,
                                 token,
                                 batch + 1,
                                 batch + 1,
                                 batch + 1,
                                 batch + 1,
                                 atoms + 1,
                                 shells + 1,
                                 shells,
                                 orbitals,
                                 orbitals,
                                 device_topology.atom_offsets,
                                 device_topology.batch_shell_offsets,
                                 device_topology.batch_orbital_offsets,
                                 device_topology.matrix_offsets,
                                 device_topology.atom_shell_offsets,
                                 device_topology.shell_orbital_offsets,
                                 device_topology.shell_to_atom,
                                 device_topology.orbital_to_shell,
                                 device_topology.orbital_to_atom};
  candidate.eigensolver_batch = {batch,
                                 orbitals,
                                 matrices,
                                 batch + 1,
                                 batch + 1,
                                 batch,
                                 0,
                                 token,
                                 device_topology.batch_orbital_offsets,
                                 device_topology.matrix_offsets,
                                 device_topology.bucket_systems,
                                 nullptr};
  candidate.eigensolver_options = impl_->eigensolver_options;
  candidate.occupations_batch = {batch,
                                 orbitals,
                                 batch + 1,
                                 2 * batch,
                                 batch,
                                 0,
                                 token,
                                 device_topology.batch_orbital_offsets,
                                 cptr(impl_->layout.electron_counts, static_cast<double*>(nullptr)),
                                 cptr(impl_->layout.temperatures, static_cast<double*>(nullptr)),
                                 nullptr};
  candidate.density_batch = {batch,
                             orbitals,
                             matrices,
                             batch + 1,
                             batch + 1,
                             token,
                             device_topology.batch_orbital_offsets,
                             device_topology.matrix_offsets};
  candidate.mulliken_batch = {
      batch,
      atoms,
      shells,
      orbitals,
      matrices,
      impl_->maximum_atoms,
      impl_->maximum_shells,
      token,
      batch + 1,
      batch + 1,
      batch + 1,
      batch + 1,
      atoms + 1,
      shells + 1,
      shells,
      shells,
      device_topology.atom_offsets,
      device_topology.batch_shell_offsets,
      device_topology.batch_orbital_offsets,
      device_topology.matrix_offsets,
      device_topology.atom_shell_offsets,
      device_topology.shell_orbital_offsets,
      device_topology.shell_to_atom,
      cptr(impl_->layout.mulliken_reference_occupations, static_cast<double*>(nullptr))};
  candidate.electronic_energy_batch = {batch, matrices, batch + 1, token,
                                       device_topology.matrix_offsets};
  candidate.classical_energy_batch = {batch, impl_->enabled_components, token};
  candidate.free_energy_batch = {batch, impl_->enabled_components, impl_->electronic_temperature,
                                 token};
  candidate.publication_plan = {batch,
                                atoms,
                                shells,
                                orbitals,
                                matrices,
                                impl_->mixer_vector_elements,
                                impl_->mixer_history,
                                batch + 1,
                                batch + 1,
                                batch + 1,
                                batch + 1,
                                shells,
                                device_topology.atom_offsets,
                                device_topology.batch_shell_offsets,
                                device_topology.batch_orbital_offsets,
                                device_topology.matrix_offsets,
                                device_topology.shell_to_atom,
                                device_wavefunction,
                                impl_->maximum_iterations,
                                impl_->residual_tolerance,
                                impl_->energy_tolerance,
                                token};

  auto* const provenance =
      mptr(impl_->layout.provenance_bindings, static_cast<Gfn2SccCacheProvenanceBinding*>(nullptr));
  candidate.provenance = {
      provenance,
      impl_->layout.provenance_bindings.elements,
      impl_->geometry_generation,
      cptr(impl_->layout.warm_start_generations, static_cast<std::uint64_t*>(nullptr)),
      impl_->layout.warm_start_generations.elements,
      impl_->warm_start_generation,
      token};

  /* Provenance records contain one device address, so create() cannot seal
   * their final bytes. Patch the pinned image immediately before its one H2D
   * submission. Bind is single-owner setup and remains allocation-free; the
   * lifetime contract already forbids rebinding while a prior upload is live. */
  std::array<Gfn2SccCacheProvenanceBinding, 6> records{};
  std::size_t record_count = 0u;
  const auto batch_provenance = [&](Gfn2SccStageId stage) {
    Gfn2SccCacheProvenanceBinding binding{};
    binding.provenance = {Gfn2PlanMemorySpace::kCudaDevice,
                          Gfn2GenerationScope::kBatch,
                          token,
                          impl_->geometry_generation,
                          batch,
                          0,
                          nullptr};
    binding.owner_stage = stage;
    records[record_count++] = binding;
  };
  Gfn2SccCacheProvenanceBinding geometry{};
  geometry.provenance = {Gfn2PlanMemorySpace::kCudaDevice,
                         Gfn2GenerationScope::kPerSystem,
                         token,
                         0u,
                         batch,
                         batch,
                         candidate.geometry_cache.geometry_generations};
  geometry.owner_stage = Gfn2SccStageId::kGeometry;
  records[record_count++] = geometry;
  batch_provenance(Gfn2SccStageId::kES2Potential);
  batch_provenance(Gfn2SccStageId::kAES2Potential);
  if (impl_->d4_enabled) {
    batch_provenance(Gfn2SccStageId::kD4Potential);
  }
  if (impl_->point_enabled) {
    batch_provenance(Gfn2SccStageId::kExplicitPointChargePotential);
  }
  if (impl_->periodic_enabled) {
    batch_provenance(Gfn2SccStageId::kPeriodicPotential);
  }
  if (record_count != static_cast<std::size_t>(impl_->layout.provenance_bindings.elements)) {
    return failure(XTBLOOM_STATUS_INTERNAL_ERROR, Error::kInvalidSource, Field::kGeometry);
  }
  std::memcpy(
      static_cast<std::byte*>(impl_->upload_image) + impl_->layout.provenance_bindings.offset,
      records.data(), record_count * sizeof(Gfn2SccCacheProvenanceBinding));

  Gfn2SccIterationDeviceInput input{};
  input.hamiltonian = {cptr(impl_->layout.h0, static_cast<double*>(nullptr)),
                       matrices,
                       cptr(impl_->layout.overlap, static_cast<double*>(nullptr)),
                       matrices,
                       cptr(impl_->layout.dipole_integrals, static_cast<double*>(nullptr)),
                       impl_->layout.dipole_integrals.elements,
                       cptr(impl_->layout.quadrupole_integrals, static_cast<double*>(nullptr)),
                       impl_->layout.quadrupole_integrals.elements,
                       nullptr,
                       0,
                       nullptr,
                       0,
                       nullptr,
                       0,
                       token};
  input.plan_token = token;

  const cudaError_t status =
      cudaMemcpyAsync(device_arena, impl_->upload_image, required, cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) {
    return arena_failure(Error::kCudaError, required, status);
  }
  plan_seed = candidate;
  input_seed = input;
  return {};
}

}  // namespace xtbloom::detail::cuda
