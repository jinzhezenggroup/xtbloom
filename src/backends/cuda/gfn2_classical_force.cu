#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/gfn2_classical_force.cuh"
#include "backends/cuda/gfn2_parameters.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr double kRepulsionCutoffSquared = 25.0 * 25.0;
constexpr double kMinimumDistanceSquared = 1.0e-24;
constexpr std::int64_t kInt64Maximum = 9223372036854775807LL;
constexpr std::uint32_t kPrimitiveInactiveMarker = 0xffffffffu;

[[nodiscard]] constexpr bool component_enabled(std::uint32_t mask,
                                               Gfn2ClassicalForceComponent component) noexcept {
  return (mask & static_cast<std::uint32_t>(component)) != 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
  bool empty = true;
};

template <typename T>
bool make_range(const T* pointer, std::int64_t elements, AddressRange* range) noexcept {
  if (range == nullptr || elements < 0) {
    return false;
  }
  if (elements == 0) {
    *range = {};
    return true;
  }
  if (pointer == nullptr || reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) != 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(T);
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  *range = {begin, begin + bytes, false};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return !first.empty && !second.empty && first.begin < second.end && second.begin < first.end;
}

template <std::size_t N>
bool ranges_are_disjoint(const std::array<AddressRange, N>& ranges, std::size_t count) noexcept {
  if (count > N) {
    return false;
  }
  for (std::size_t first = 0; first < count; ++first) {
    for (std::size_t second = first + 1; second < count; ++second) {
      if (ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

template <std::size_t N, typename T>
bool append_range(std::array<AddressRange, N>& ranges, std::size_t* count, const T* pointer,
                  std::int64_t elements) noexcept {
  if (count == nullptr || *count >= N || !make_range(pointer, elements, &ranges[*count])) {
    return false;
  }
  ++*count;
  return true;
}

bool checked_product(std::int64_t first, std::int64_t second, std::int64_t* result) noexcept {
  if (result == nullptr || first < 0 || second < 0 ||
      (first != 0 && second > kInt64Maximum / first)) {
    return false;
  }
  *result = first * second;
  return true;
}

template <typename T>
bool aligned_pointer(const T* pointer) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) == 0u;
}

bool validate_common_descriptors(const Gfn2ClassicalForceDevicePlan& plan,
                                 const Gfn2ForceDeviceActivity& activity,
                                 const Gfn2ClassicalForceDeviceInput& input,
                                 const Gfn2ClassicalForceDeviceOutput& output,
                                 const Gfn2ClassicalForceDeviceWorkspace& workspace,
                                 std::uint32_t* system_errors,
                                 std::uint32_t* device_error) noexcept {
  const bool extent_representable =
      plan.batch_size > 0 && plan.total_atoms > 0 && plan.total_shells > 0 &&
      plan.batch_size <= static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max()) &&
      plan.total_atoms <= kInt64Maximum / 6 && plan.total_atoms <= kInt64Maximum / 3;
  if (!extent_representable || plan.plan_token == 0u || plan.geometry_generation == 0u ||
      (plan.enabled_components & ~kGfn2ClassicalForceAllComponents) != 0u ||
      !aligned_pointer(plan.atom_offsets) || !aligned_pointer(plan.atomic_numbers) ||
      activity.requested_mask == nullptr || activity.system_statuses == nullptr ||
      activity.batch_elements != plan.batch_size || activity.plan_token != plan.plan_token ||
      input.plan_token != plan.plan_token || output.plan_token != plan.plan_token ||
      workspace.plan_token != plan.plan_token || input.position_elements != plan.total_atoms * 3 ||
      output.force_elements != plan.total_atoms * 3 ||
      workspace.gradient_elements < plan.total_atoms * 3 ||
      workspace.force_elements < plan.total_atoms * 3 ||
      workspace.coordination_elements < plan.total_atoms ||
      workspace.selected_elements < plan.batch_size ||
      workspace.primitive_system_error_elements < plan.batch_size ||
      workspace.primitive_device_error_elements != 1 || workspace.sequence_elements != 1 ||
      !aligned_pointer(input.positions) || !aligned_pointer(output.forces) ||
      !aligned_pointer(workspace.gradient_scratch) || !aligned_pointer(workspace.force_scratch) ||
      !aligned_pointer(workspace.coordination_adjoints) ||
      !aligned_pointer(workspace.selected_mask) ||
      !aligned_pointer(workspace.primitive_system_errors) ||
      !aligned_pointer(workspace.primitive_device_error) ||
      !aligned_pointer(workspace.sequence_active) || !aligned_pointer(system_errors) ||
      !aligned_pointer(device_error)) {
    return false;
  }

  const bool needs_es2 =
      component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kES2);
  const bool needs_aes2 =
      component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kAES2);
  const bool needs_d4 =
      component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4TwoBody) ||
      component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4ATM);

  if (needs_es2 &&
      (input.shell_elements != plan.total_shells || !aligned_pointer(input.shell_charges) ||
       plan.es2_batch.batch_size != plan.batch_size ||
       plan.es2_batch.total_atoms != plan.total_atoms ||
       plan.es2_batch.total_shells != plan.total_shells ||
       plan.es2_batch.plan_token != plan.plan_token ||
       plan.es2_batch.atom_offsets != plan.atom_offsets ||
       plan.es2_cache.plan_token != plan.plan_token ||
       plan.es2_cache.geometry_generation != plan.geometry_generation)) {
    return false;
  }
  if (needs_aes2 &&
      (input.coordination_elements != plan.total_atoms || input.atom_elements != plan.total_atoms ||
       input.dipole_elements != plan.total_atoms * 3 ||
       input.quadrupole_elements != plan.total_atoms * 6 ||
       !aligned_pointer(input.coordination_numbers) || !aligned_pointer(input.atomic_charges) ||
       !aligned_pointer(input.atomic_dipoles) || !aligned_pointer(input.atomic_quadrupoles) ||
       plan.aes2_batch.batch_size != plan.batch_size ||
       plan.aes2_batch.total_atoms != plan.total_atoms ||
       plan.aes2_batch.plan_token != plan.plan_token ||
       plan.aes2_batch.atom_offsets != plan.atom_offsets ||
       plan.aes2_cache.plan_token != plan.plan_token ||
       plan.aes2_cache.geometry_generation != plan.geometry_generation ||
       plan.geometry_batch.batch_size != plan.batch_size ||
       plan.geometry_batch.total_atoms != plan.total_atoms ||
       plan.geometry_batch.plan_token != plan.plan_token ||
       plan.geometry_batch.atom_offsets != plan.atom_offsets ||
       plan.geometry_cache.plan_token != plan.plan_token ||
       workspace.aes2_workspace.gradient_elements < plan.total_atoms * 3 ||
       workspace.aes2_workspace.coordination_elements < plan.total_atoms ||
       !aligned_pointer(workspace.aes2_workspace.gradient_scratch) ||
       !aligned_pointer(workspace.aes2_workspace.coordination_scratch) ||
       workspace.geometry_workspace.gradient_elements < plan.total_atoms * 3 ||
       workspace.geometry_workspace.sequence_elements != 1 ||
       !aligned_pointer(workspace.geometry_workspace.gradient_scratch) ||
       !aligned_pointer(workspace.geometry_workspace.sequence_active))) {
    return false;
  }
  if (needs_d4 && (plan.d4_batch.batch_size != plan.batch_size ||
                   plan.d4_batch.total_atoms != plan.total_atoms ||
                   plan.d4_batch.plan_token != plan.plan_token ||
                   plan.d4_batch.atom_offsets != plan.atom_offsets ||
                   plan.d4_batch.atomic_numbers != plan.atomic_numbers ||
                   plan.d4_cache.plan_token != plan.plan_token ||
                   plan.d4_cache.geometry_generation != plan.geometry_generation ||
                   workspace.d4_workspace.system_errors != workspace.primitive_system_errors ||
                   workspace.d4_workspace.system_error_elements < plan.batch_size)) {
    return false;
  }
  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4TwoBody) &&
      (input.atom_elements != plan.total_atoms || !aligned_pointer(input.atomic_charges))) {
    return false;
  }

  std::int64_t geometry_pair_elements = 0;
  std::int64_t aes2_pair_elements = 0;
  std::int64_t d4_pair_elements = 0;
  std::int64_t d4_weight_elements = 0;
  if ((needs_aes2 && (!checked_product(plan.geometry_batch.total_pairs,
                                       kGfn2GeometryPairDataElements, &geometry_pair_elements) ||
                      !checked_product(plan.aes2_batch.total_pairs, kGfn2AES2PairDataElements,
                                       &aes2_pair_elements))) ||
      (needs_d4 &&
       (!checked_product(plan.d4_batch.total_pairs, kGfn2D4PairDataElements, &d4_pair_elements) ||
        !checked_product(plan.total_atoms, kGfn2D4MaximumReferences, &d4_weight_elements)))) {
    return false;
  }

  /*
   * Keep this fixed-capacity inventory synchronized with the nested primitive
   * calls below. Validation is a hot-path host operation and must not allocate.
   * d4_workspace.system_errors is the documented exact projection of
   * primitive_system_errors, so that range is represented only once.
   */
  std::array<AddressRange, 32> writes{};
  std::size_t write_count = 0u;
  if (!append_range(writes, &write_count, output.forces, plan.total_atoms * 3) ||
      !append_range(writes, &write_count, workspace.gradient_scratch, plan.total_atoms * 3) ||
      !append_range(writes, &write_count, workspace.force_scratch, plan.total_atoms * 3) ||
      !append_range(writes, &write_count, workspace.coordination_adjoints, plan.total_atoms) ||
      !append_range(writes, &write_count, workspace.selected_mask, plan.batch_size) ||
      !append_range(writes, &write_count, workspace.primitive_system_errors, plan.batch_size) ||
      !append_range(writes, &write_count, workspace.primitive_device_error, 1) ||
      !append_range(writes, &write_count, workspace.sequence_active, 1) ||
      !append_range(writes, &write_count, system_errors, plan.batch_size) ||
      !append_range(writes, &write_count, device_error, 1)) {
    return false;
  }
  if (needs_aes2 &&
      (!append_range(writes, &write_count, workspace.aes2_workspace.pair_scratch,
                     workspace.aes2_workspace.pair_elements) ||
       !append_range(writes, &write_count, workspace.aes2_workspace.potential_scratch,
                     workspace.aes2_workspace.potential_elements) ||
       !append_range(writes, &write_count, workspace.aes2_workspace.batch_scratch,
                     workspace.aes2_workspace.batch_elements) ||
       !append_range(writes, &write_count, workspace.aes2_workspace.gradient_scratch,
                     plan.total_atoms * 3) ||
       !append_range(writes, &write_count, workspace.aes2_workspace.coordination_scratch,
                     plan.total_atoms) ||
       !append_range(writes, &write_count, workspace.aes2_workspace.scc_peer_error_scratch,
                     workspace.aes2_workspace.scc_peer_error_elements) ||
       !append_range(writes, &write_count, workspace.geometry_workspace.pair_scratch,
                     workspace.geometry_workspace.pair_elements) ||
       !append_range(writes, &write_count, workspace.geometry_workspace.coordination_scratch,
                     workspace.geometry_workspace.coordination_elements) ||
       !append_range(writes, &write_count, workspace.geometry_workspace.gradient_scratch,
                     plan.total_atoms * 3) ||
       !append_range(writes, &write_count, workspace.geometry_workspace.sequence_active, 1))) {
    return false;
  }
  if (needs_d4 &&
      (!append_range(writes, &write_count, workspace.d4_workspace.weights, d4_weight_elements) ||
       !append_range(writes, &write_count, workspace.d4_workspace.weight_cn_derivatives,
                     d4_weight_elements) ||
       !append_range(writes, &write_count, workspace.d4_workspace.weight_charge_derivatives,
                     d4_weight_elements) ||
       !append_range(writes, &write_count, workspace.d4_workspace.atom_scratch, plan.total_atoms) ||
       !append_range(writes, &write_count, workspace.d4_workspace.coordination_adjoints,
                     plan.total_atoms) ||
       !append_range(writes, &write_count, workspace.d4_workspace.batch_scratch, plan.batch_size) ||
       !append_range(writes, &write_count, workspace.d4_workspace.gradient_scratch,
                     plan.total_atoms * 3))) {
    return false;
  }
  if (!ranges_are_disjoint(writes, write_count)) {
    return false;
  }

  std::array<AddressRange, 40> reads{};
  std::size_t read_count = 0u;
  if (!append_range(reads, &read_count, plan.atom_offsets, plan.batch_size + 1) ||
      !append_range(reads, &read_count, plan.atomic_numbers, plan.total_atoms) ||
      !append_range(reads, &read_count, activity.requested_mask, plan.batch_size) ||
      !append_range(reads, &read_count, activity.system_statuses, plan.batch_size) ||
      !append_range(reads, &read_count, input.positions, plan.total_atoms * 3)) {
    return false;
  }
  if (needs_es2 &&
      (!append_range(reads, &read_count, plan.es2_batch.batch_shell_offsets, plan.batch_size + 1) ||
       !append_range(reads, &read_count, plan.es2_batch.atom_shell_offsets, plan.total_atoms + 1) ||
       !append_range(reads, &read_count, plan.es2_batch.matrix_offsets, plan.batch_size + 1) ||
       !append_range(reads, &read_count, plan.es2_batch.shell_to_atom, plan.total_shells) ||
       !append_range(reads, &read_count, plan.es2_batch.shell_hardness, plan.total_shells) ||
       !append_range(reads, &read_count, plan.es2_cache.coulomb_matrix,
                     plan.es2_batch.total_matrix_elements) ||
       !append_range(reads, &read_count, input.shell_charges, plan.total_shells))) {
    return false;
  }
  if (needs_aes2 &&
      (!append_range(reads, &read_count, plan.aes2_batch.pair_offsets, plan.batch_size + 1) ||
       !append_range(reads, &read_count, plan.aes2_batch.dipole_kernel, plan.total_atoms) ||
       !append_range(reads, &read_count, plan.aes2_batch.quadrupole_kernel, plan.total_atoms) ||
       !append_range(reads, &read_count, plan.aes2_batch.multipole_radius, plan.total_atoms) ||
       !append_range(reads, &read_count, plan.aes2_batch.multipole_valence_cn, plan.total_atoms) ||
       !append_range(reads, &read_count, plan.aes2_cache.pair_data, aes2_pair_elements) ||
       !append_range(reads, &read_count, input.coordination_numbers, plan.total_atoms) ||
       !append_range(reads, &read_count, input.atomic_charges, plan.total_atoms) ||
       !append_range(reads, &read_count, input.atomic_dipoles, plan.total_atoms * 3) ||
       !append_range(reads, &read_count, input.atomic_quadrupoles, plan.total_atoms * 6) ||
       !append_range(reads, &read_count, plan.geometry_batch.pair_offsets, plan.batch_size + 1) ||
       !append_range(reads, &read_count, plan.geometry_batch.covalent_radii, plan.total_atoms) ||
       !append_range(reads, &read_count, plan.geometry_cache.pair_data, geometry_pair_elements) ||
       !append_range(reads, &read_count, plan.geometry_cache.coordination_numbers,
                     plan.total_atoms) ||
       !append_range(reads, &read_count, plan.geometry_cache.geometry_generations,
                     plan.batch_size))) {
    return false;
  }
  if (needs_d4 &&
      (!append_range(reads, &read_count, plan.d4_batch.pair_offsets, plan.batch_size + 1) ||
       !append_range(reads, &read_count, plan.d4_parameters.elements,
                     plan.d4_parameters.element_count) ||
       !append_range(reads, &read_count, plan.d4_parameters.references,
                     plan.d4_parameters.reference_count) ||
       !append_range(reads, &read_count, plan.d4_parameters.reference_c6,
                     plan.d4_parameters.reference_c6_elements) ||
       !append_range(reads, &read_count, plan.d4_cache.pair_data, d4_pair_elements) ||
       !append_range(reads, &read_count, plan.d4_cache.coordination_numbers, plan.total_atoms))) {
    return false;
  }
  /* q is already represented by AES2 when both charge-dependent components run. */
  if (!needs_aes2 &&
      component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4TwoBody) &&
      !append_range(reads, &read_count, input.atomic_charges, plan.total_atoms)) {
    return false;
  }

  for (std::size_t read_index = 0; read_index < read_count; ++read_index) {
    for (std::size_t write_index = 0; write_index < write_count; ++write_index) {
      const AddressRange& read = reads[read_index];
      const AddressRange& write = writes[write_index];
      if (ranges_overlap(read, write)) {
        return false;
      }
    }
  }
  return true;
}

__device__ bool sequence_is_open(const std::uint32_t* sequence_active) {
  return atomicAdd(const_cast<std::uint32_t*>(sequence_active), 0u) == 1u;
}

__device__ bool system_is_selected(const Gfn2ClassicalForceDeviceWorkspace& workspace,
                                   const std::uint32_t* system_errors, std::int64_t system) {
  return sequence_is_open(workspace.sequence_active) && workspace.selected_mask[system] == 1u &&
         atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) == 0u;
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error,
                                    Gfn2ClassicalForceDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system, 0u, code) == 0u) {
    atomicCAS(device_error, 0u, code);
  }
}

__device__ void record_plan_error(std::uint32_t* device_error, std::uint32_t* sequence_active,
                                  Gfn2ClassicalForceDeviceError error) {
  atomicCAS(device_error, 0u, static_cast<std::uint32_t>(error));
  atomicExch(sequence_active, 0u);
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) == 0u ? 1u : 0u;
  }
}

__global__ void common_topology_preflight_kernel(Gfn2ClassicalForceDevicePlan plan,
                                                 std::uint32_t* device_error,
                                                 std::uint32_t* sequence_active) {
  if (blockIdx.x != 0 || !sequence_is_open(sequence_active)) {
    return;
  }
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = plan.atom_offsets[0] == 0 && plan.atom_offsets[plan.batch_size] == plan.total_atoms;
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < plan.batch_size; system += blockDim.x) {
    const std::int64_t begin = plan.atom_offsets[system];
    const std::int64_t end = plan.atom_offsets[system + 1];
    if (begin < 0 || begin > end || end > plan.total_atoms) {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && valid == 0) {
    record_plan_error(device_error, sequence_active,
                      Gfn2ClassicalForceDeviceError::kInvalidTopology);
  }
}

__global__ void gate_and_seed_kernel(Gfn2ClassicalForceDevicePlan plan,
                                     Gfn2ForceDeviceActivity activity,
                                     Gfn2ClassicalForceDeviceOutput output,
                                     Gfn2ClassicalForceDeviceWorkspace workspace,
                                     std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int selected;
  if (threadIdx.x == 0) {
    selected = 0;
    workspace.selected_mask[system] = 0u;
    if (sequence_is_open(workspace.sequence_active) &&
        atomicAdd(system_errors + system, 0u) == 0u) {
      const std::uint8_t requested = activity.requested_mask[system];
      if (requested > 1u) {
        record_system_error(system_errors, system, device_error,
                            Gfn2ClassicalForceDeviceError::kInvalidActivity);
      } else if (requested == 1u && activity.system_statuses[system] == GPUXTB_STATUS_SUCCESS) {
        selected = 1;
        workspace.selected_mask[system] = 1u;
      }
    }
  }
  __syncthreads();
  if (selected == 0) {
    return;
  }

  const std::int64_t atom_begin = plan.atom_offsets[system];
  const std::int64_t atom_end = plan.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    bool finite_seed = true;
    for (int axis = 0; axis < 3; ++axis) {
      const double seed = output.forces[coordinate + axis];
      if (!isfinite(seed)) {
        finite_seed = false;
      } else {
        workspace.force_scratch[coordinate + axis] = seed;
        workspace.gradient_scratch[coordinate + axis] = 0.0;
      }
    }
    workspace.coordination_adjoints[atom] = 0.0;
    if (!finite_seed) {
      record_system_error(system_errors, system, device_error,
                          Gfn2ClassicalForceDeviceError::kNonfiniteForceSeed);
    }
  }
}

__device__ bool add_finite_atomic(double* target, double contribution) {
  if (!isfinite(contribution)) {
    return false;
  }
  const double previous = atomic_add_fp64(target, contribution);
  return isfinite(previous) && isfinite(previous + contribution);
}

__global__ void repulsion_gradient_kernel(Gfn2ClassicalForceDevicePlan plan,
                                          const double* positions,
                                          Gfn2ClassicalForceDeviceWorkspace workspace,
                                          std::uint32_t* system_errors,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = system_is_selected(workspace, system_errors, system) ? 1 : 0;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  const std::int64_t begin = plan.atom_offsets[system];
  const std::int64_t end = plan.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    const std::int32_t atomic_number = plan.atomic_numbers[atom];
    const std::int64_t coordinate = atom * 3;
    if (atomic_number < 1 ||
        atomic_number > static_cast<std::int32_t>(parameters::gfn2::kElementCount) ||
        !isfinite(positions[coordinate]) || !isfinite(positions[coordinate + 1]) ||
        !isfinite(positions[coordinate + 2])) {
      atomicExch(&valid, 0);
    } else {
      const parameters::gfn2::ElementParameters element = g_gfn2_elements[atomic_number - 1];
      if (element.atomic_number != atomic_number || !(element.arep > 0.0) ||
          !(element.zeff > 0.0) || !isfinite(element.arep) || !isfinite(element.zeff)) {
        atomicExch(&valid, 0);
      }
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2ClassicalForceDeviceError::kRepulsionFailure);
    }
    return;
  }

  for (std::int64_t upper = begin + 1 + threadIdx.x; upper < end; upper += blockDim.x) {
    const std::int32_t upper_number = plan.atomic_numbers[upper];
    const parameters::gfn2::ElementParameters upper_element = g_gfn2_elements[upper_number - 1];
    const double upper_sqrt_alpha = sqrt(upper_element.arep);
    const std::int64_t upper_coordinate = upper * 3;
    for (std::int64_t lower = begin; lower < upper; ++lower) {
      const std::int64_t lower_coordinate = lower * 3;
      const double dx = positions[upper_coordinate] - positions[lower_coordinate];
      const double dy = positions[upper_coordinate + 1] - positions[lower_coordinate + 1];
      const double dz = positions[upper_coordinate + 2] - positions[lower_coordinate + 2];
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      if (!(distance_squared > kMinimumDistanceSquared) || !isfinite(distance_squared)) {
        atomicExch(&valid, 0);
        continue;
      }
      if (distance_squared > kRepulsionCutoffSquared) {
        continue;
      }
      const std::int32_t lower_number = plan.atomic_numbers[lower];
      const parameters::gfn2::ElementParameters lower_element = g_gfn2_elements[lower_number - 1];
      const double distance = sqrt(distance_squared);
      const bool light_pair = upper_number <= 2 && lower_number <= 2;
      const double exponent =
          light_pair ? g_gfn2_global.repulsion_klight : g_gfn2_global.repulsion_kexp;
      const double distance_power = light_pair ? distance : distance * sqrt(distance);
      const double pair_alpha = upper_sqrt_alpha * sqrt(lower_element.arep);
      const double pair_energy =
          upper_element.zeff * lower_element.zeff * exp(-pair_alpha * distance_power) / distance;
      const double force_scale =
          (pair_alpha * exponent * distance_power + 1.0) * pair_energy / distance_squared;
      const double force[3] = {force_scale * dx, force_scale * dy, force_scale * dz};
      bool finite_pair = isfinite(pair_energy) && isfinite(force_scale);
      for (int axis = 0; axis < 3; ++axis) {
        /* Existing repulsion primitive publishes force; this composer stores dE/dR. */
        finite_pair =
            add_finite_atomic(workspace.gradient_scratch + upper_coordinate + axis, -force[axis]) &&
            add_finite_atomic(workspace.gradient_scratch + lower_coordinate + axis, force[axis]) &&
            finite_pair;
      }
      if (!finite_pair) {
        atomicExch(&valid, 0);
      }
    }
  }
  __syncthreads();
  if (valid == 0 && threadIdx.x == 0) {
    record_system_error(system_errors, system, device_error,
                        Gfn2ClassicalForceDeviceError::kRepulsionFailure);
  }
}

__global__ void es2_topology_preflight_kernel(Gfn2ClassicalForceDevicePlan plan,
                                              std::uint32_t* device_error,
                                              std::uint32_t* sequence_active) {
  if (blockIdx.x != 0 || !sequence_is_open(sequence_active)) {
    return;
  }
  const Gfn2ES2DeviceBatch batch = plan.es2_batch;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = batch.atom_offsets[0] == 0 &&
            batch.atom_offsets[batch.batch_size] == batch.total_atoms &&
            batch.batch_shell_offsets[0] == 0 &&
            batch.batch_shell_offsets[batch.batch_size] == batch.total_shells &&
            batch.matrix_offsets[0] == 0 &&
            batch.matrix_offsets[batch.batch_size] == batch.total_matrix_elements;
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t shell_begin = batch.batch_shell_offsets[system];
    const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
    const std::int64_t matrix_begin = batch.matrix_offsets[system];
    const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
    const std::int64_t shells = shell_end - shell_begin;
    const bool square_ok = shells >= 0 && (shells == 0 || shells <= kInt64Maximum / shells);
    bool system_valid = atom_begin >= 0 && atom_begin <= atom_end &&
                        atom_end <= batch.total_atoms && shell_begin >= 0 &&
                        shell_begin <= shell_end && shell_end <= batch.total_shells &&
                        matrix_begin >= 0 && matrix_begin <= matrix_end &&
                        matrix_end <= batch.total_matrix_elements && square_ok &&
                        matrix_end - matrix_begin == shells * shells;
    if (system_valid) {
      system_valid = batch.atom_shell_offsets[atom_begin] == shell_begin &&
                     batch.atom_shell_offsets[atom_end] == shell_end;
    }
    if (!system_valid) {
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t atom = threadIdx.x; atom < batch.total_atoms; atom += blockDim.x) {
    const std::int64_t shell_begin = batch.atom_shell_offsets[atom];
    const std::int64_t shell_end = batch.atom_shell_offsets[atom + 1];
    if (shell_begin < 0 || shell_begin >= shell_end || shell_end > batch.total_shells) {
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t shell = threadIdx.x; shell < batch.total_shells; shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    if (atom < 0 || atom >= batch.total_atoms || shell < batch.atom_shell_offsets[atom] ||
        shell >= batch.atom_shell_offsets[atom + 1] || !(batch.shell_hardness[shell] > 0.0) ||
        !isfinite(batch.shell_hardness[shell])) {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && valid == 0) {
    record_plan_error(device_error, sequence_active,
                      Gfn2ClassicalForceDeviceError::kInvalidTopology);
  }
}

__global__ void es2_gradient_kernel(Gfn2ClassicalForceDevicePlan plan,
                                    Gfn2ClassicalForceDeviceInput input,
                                    Gfn2ClassicalForceDeviceWorkspace workspace,
                                    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = system_is_selected(workspace, system_errors, system) ? 1 : 0;
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  const Gfn2ES2DeviceBatch batch = plan.es2_batch;
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t shells = shell_end - shell_begin;
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!isfinite(input.positions[coordinate]) || !isfinite(input.positions[coordinate + 1]) ||
        !isfinite(input.positions[coordinate + 2])) {
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    if (!isfinite(input.shell_charges[shell])) {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2ClassicalForceDeviceError::kES2Failure);
    }
    return;
  }

  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    double contribution[3] = {0.0, 0.0, 0.0};
    bool finite_result = true;
    for (std::int64_t peer = atom_begin; finite_result && peer < atom_end; ++peer) {
      if (peer == atom) {
        continue;
      }
      const std::int64_t upper = atom > peer ? atom : peer;
      const std::int64_t lower = atom > peer ? peer : atom;
      const std::int64_t upper_shell_begin = batch.atom_shell_offsets[upper];
      const std::int64_t upper_shell_end = batch.atom_shell_offsets[upper + 1];
      const std::int64_t lower_shell_begin = batch.atom_shell_offsets[lower];
      const std::int64_t lower_shell_end = batch.atom_shell_offsets[lower + 1];
      double weighted = 0.0;
      for (std::int64_t upper_shell = upper_shell_begin; upper_shell < upper_shell_end;
           ++upper_shell) {
        for (std::int64_t lower_shell = lower_shell_begin; lower_shell < lower_shell_end;
             ++lower_shell) {
          const std::int64_t matrix =
              matrix_begin + (upper_shell - shell_begin) * shells + lower_shell - shell_begin;
          const double kernel = plan.es2_cache.coulomb_matrix[matrix];
          double term = input.shell_charges[upper_shell] * kernel;
          term *= input.shell_charges[lower_shell];
          term *= kernel;
          term *= kernel;
          const double updated = weighted + term;
          if (!(kernel > 0.0) || !isfinite(kernel) || !isfinite(term) || !isfinite(updated)) {
            finite_result = false;
            break;
          }
          weighted = updated;
        }
      }
      const double sign = atom == upper ? -1.0 : 1.0;
      for (int axis = 0; finite_result && axis < 3; ++axis) {
        const double displacement =
            input.positions[upper * 3 + axis] - input.positions[lower * 3 + axis];
        const double term = sign * weighted * displacement;
        const double updated = contribution[axis] + term;
        if (!isfinite(displacement) || !isfinite(term) || !isfinite(updated)) {
          finite_result = false;
        } else {
          contribution[axis] = updated;
        }
      }
    }
    const std::int64_t coordinate = atom * 3;
    for (int axis = 0; finite_result && axis < 3; ++axis) {
      const double updated = workspace.gradient_scratch[coordinate + axis] + contribution[axis];
      if (!isfinite(updated)) {
        finite_result = false;
      } else {
        workspace.gradient_scratch[coordinate + axis] = updated;
      }
    }
    if (!finite_result) {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0 && threadIdx.x == 0) {
    record_system_error(system_errors, system, device_error,
                        Gfn2ClassicalForceDeviceError::kES2Failure);
  }
}

__global__ void prepare_primitive_stage_kernel(std::int64_t batch_size,
                                               Gfn2ClassicalForceDeviceWorkspace workspace,
                                               const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size) {
    const bool active = sequence_is_open(workspace.sequence_active) &&
                        workspace.selected_mask[system] == 1u &&
                        atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) == 0u;
    workspace.primitive_system_errors[system] = active ? 0u : kPrimitiveInactiveMarker;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0 && !sequence_is_open(workspace.sequence_active)) {
    *workspace.primitive_device_error = kPrimitiveInactiveMarker;
  }
}

__global__ void merge_primitive_stage_kernel(std::int64_t batch_size,
                                             Gfn2ClassicalForceDeviceWorkspace workspace,
                                             Gfn2ClassicalForceDeviceError mapped_error,
                                             bool device_code_is_plan_only,
                                             std::uint32_t* system_errors,
                                             std::uint32_t* device_error) {
  __shared__ int device_code_localized;
  __shared__ std::uint32_t primitive_device_code;
  if (threadIdx.x == 0) {
    device_code_localized = 0;
    primitive_device_code = atomicAdd(workspace.primitive_device_error, 0u);
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < batch_size; system += blockDim.x) {
    const std::uint32_t raw = workspace.primitive_system_errors[system];
    if (raw != 0u && raw != kPrimitiveInactiveMarker && workspace.selected_mask[system] == 1u) {
      record_system_error(system_errors, system, device_error, mapped_error);
      if (raw == primitive_device_code) {
        atomicExch(&device_code_localized, 1);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && primitive_device_code != 0u &&
      primitive_device_code != kPrimitiveInactiveMarker &&
      (device_code_is_plan_only || device_code_localized == 0)) {
    record_plan_error(device_error, workspace.sequence_active,
                      Gfn2ClassicalForceDeviceError::kInvalidTopology);
  }
}

__global__ void finalize_force_kernel(Gfn2ClassicalForceDevicePlan plan,
                                      Gfn2ClassicalForceDeviceWorkspace workspace,
                                      std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!system_is_selected(workspace, system_errors, system)) {
    return;
  }
  const std::int64_t begin = plan.atom_offsets[system];
  const std::int64_t end = plan.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    bool finite_result = true;
    for (int axis = 0; axis < 3; ++axis) {
      const double updated = workspace.force_scratch[coordinate + axis] -
                             workspace.gradient_scratch[coordinate + axis];
      if (!isfinite(updated)) {
        finite_result = false;
      } else {
        workspace.force_scratch[coordinate + axis] = updated;
      }
    }
    if (!finite_result) {
      record_system_error(system_errors, system, device_error,
                          Gfn2ClassicalForceDeviceError::kNonfiniteForceArithmetic);
    }
  }
}

__global__ void publish_force_kernel(Gfn2ClassicalForceDevicePlan plan,
                                     Gfn2ClassicalForceDeviceOutput output,
                                     Gfn2ClassicalForceDeviceWorkspace workspace,
                                     const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!system_is_selected(workspace, system_errors, system)) {
    return;
  }
  const std::int64_t begin = plan.atom_offsets[system];
  const std::int64_t end = plan.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    output.forces[coordinate] = workspace.force_scratch[coordinate];
    output.forces[coordinate + 1] = workspace.force_scratch[coordinate + 1];
    output.forces[coordinate + 2] = workspace.force_scratch[coordinate + 2];
  }
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

cudaError_t prepare_primitive_stage(const Gfn2ClassicalForceDevicePlan& plan,
                                    const Gfn2ClassicalForceDeviceWorkspace& workspace,
                                    const std::uint32_t* system_errors,
                                    cudaStream_t stream) noexcept {
  cudaError_t status = cudaMemsetAsync(workspace.primitive_device_error, 0,
                                       sizeof(*workspace.primitive_device_error), stream);
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks =
      static_cast<unsigned int>((plan.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  prepare_primitive_stage_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(plan.batch_size,
                                                                          workspace, system_errors);
  return check_launch();
}

cudaError_t merge_primitive_stage(const Gfn2ClassicalForceDevicePlan& plan,
                                  const Gfn2ClassicalForceDeviceWorkspace& workspace,
                                  Gfn2ClassicalForceDeviceError mapped_error,
                                  bool device_code_is_plan_only, std::uint32_t* system_errors,
                                  std::uint32_t* device_error, cudaStream_t stream) noexcept {
  merge_primitive_stage_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      plan.batch_size, workspace, mapped_error, device_code_is_plan_only, system_errors,
      device_error);
  return check_launch();
}

}  // namespace

cudaError_t reset_gfn2_classical_force_device_errors_cuda(std::int64_t batch_size,
                                                          std::uint32_t* system_errors,
                                                          std::uint32_t* device_error,
                                                          cudaStream_t stream) noexcept {
  if (batch_size <= 0 || !aligned_pointer(system_errors) || !aligned_pointer(device_error) ||
      static_cast<std::uint64_t>(batch_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                     sizeof(*system_errors))) {
    return cudaErrorInvalidValue;
  }
  AddressRange systems;
  AddressRange device;
  if (!make_range(system_errors, batch_size, &systems) || !make_range(device_error, 1, &device) ||
      ranges_overlap(systems, device)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  return status == cudaSuccess ? cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream)
                               : status;
}

cudaError_t add_gfn2_classical_forces_cuda(
    const Gfn2ClassicalForceDevicePlan& plan, const Gfn2ForceDeviceActivity& activity,
    const Gfn2ClassicalForceDeviceInput& input, const Gfn2ClassicalForceDeviceOutput& output,
    const Gfn2ClassicalForceDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!validate_common_descriptors(plan, activity, input, output, workspace, system_errors,
                                   device_error)) {
    return cudaErrorInvalidValue;
  }

  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  common_topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(plan, device_error,
                                                                       workspace.sequence_active);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  gate_and_seed_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock, 0, stream>>>(
      plan, activity, output, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }

  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kRepulsion)) {
    repulsion_gradient_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock, 0,
                                stream>>>(plan, input.positions, workspace, system_errors,
                                          device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }

  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kES2)) {
    es2_topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(plan, device_error,
                                                                      workspace.sequence_active);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
    es2_gradient_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock, 0,
                          stream>>>(plan, input, workspace, system_errors, device_error);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }

  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kAES2)) {
    status = prepare_primitive_stage(plan, workspace, system_errors, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = add_gfn2_aes2_vjp_cuda(
        plan.aes2_batch, plan.aes2_cache, input.positions, input.coordination_numbers,
        plan.geometry_generation, input.atomic_charges, input.atomic_dipoles,
        input.atomic_quadrupoles, workspace.gradient_scratch, workspace.coordination_adjoints,
        workspace.aes2_workspace, workspace.primitive_system_errors,
        workspace.primitive_device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = merge_primitive_stage(plan, workspace, Gfn2ClassicalForceDeviceError::kAES2Failure,
                                   false, system_errors, device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }

    status = prepare_primitive_stage(plan, workspace, system_errors, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = add_gfn2_coordination_vjp_cuda(
        plan.geometry_batch, plan.geometry_cache, plan.geometry_generation,
        workspace.coordination_adjoints, workspace.gradient_scratch, workspace.geometry_workspace,
        workspace.primitive_system_errors, workspace.primitive_device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = merge_primitive_stage(plan, workspace,
                                   Gfn2ClassicalForceDeviceError::kAES2CoordinationFailure, false,
                                   system_errors, device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
  }

  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4TwoBody)) {
    status = prepare_primitive_stage(plan, workspace, system_errors, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = add_gfn2_d4_two_body_gradient_cuda(plan.d4_batch, plan.d4_parameters, plan.d4_cache,
                                                input.atomic_charges, workspace.gradient_scratch,
                                                workspace.d4_workspace,
                                                workspace.primitive_device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status =
        merge_primitive_stage(plan, workspace, Gfn2ClassicalForceDeviceError::kD4TwoBodyFailure,
                              true, system_errors, device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
  }

  if (component_enabled(plan.enabled_components, Gfn2ClassicalForceComponent::kD4ATM)) {
    status = prepare_primitive_stage(plan, workspace, system_errors, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = add_gfn2_d4_atm_gradient_cuda(plan.d4_batch, plan.d4_parameters, plan.d4_cache,
                                           workspace.gradient_scratch, workspace.d4_workspace,
                                           workspace.primitive_device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
    status = merge_primitive_stage(plan, workspace, Gfn2ClassicalForceDeviceError::kD4ATMFailure,
                                   true, system_errors, device_error, stream);
    if (status != cudaSuccess) {
      return status;
    }
  }

  finalize_force_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock, 0,
                          stream>>>(plan, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_force_kernel<<<static_cast<unsigned int>(plan.batch_size), kThreadsPerBlock, 0, stream>>>(
      plan, output, workspace, system_errors);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
