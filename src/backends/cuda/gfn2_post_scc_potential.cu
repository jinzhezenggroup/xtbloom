#include <cuda_runtime.h>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_post_scc_potential.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 128;
constexpr std::uint32_t kMandatoryComponents =
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
    static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2);
constexpr std::uint32_t kInvalidRequestedMask = 1u;
constexpr std::uint32_t kIneligibleGeometry = 2u;
constexpr std::uint32_t kStaleGeometry = 3u;
constexpr std::uint32_t kInvalidGeometryEpoch = 4u;
constexpr std::uint32_t kInvalidEligibilityMask = 5u;
constexpr std::uint32_t kClosedStageWithoutCode = 0x00fffffeu;

template <typename T>
bool aligned_pointer(const T* pointer) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) == 0u;
}

bool component_enabled(std::uint32_t mask, Gfn2SccPotentialComponent component) noexcept {
  return (mask & static_cast<std::uint32_t>(component)) != 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                AddressRange* result) noexcept {
  if (result == nullptr || elements < 0 || (pointer == nullptr) != (elements == 0) ||
      static_cast<std::uint64_t>(elements) >
          std::numeric_limits<std::size_t>::max() / element_size) {
    return false;
  }
  if (elements == 0) {
    *result = {};
    return true;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  *result = {begin, begin + bytes};
  return true;
}

bool overlaps(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin != first.end && second.begin != second.end && first.begin < second.end &&
         second.begin < first.end;
}

/* Public outputs are one terminal transaction and may not double as scratch. */
bool public_results_are_disjoint(const Gfn2PostSccPotentialDevicePlan& plan,
                                 const Gfn2PostSccPotentialDeviceInput& input,
                                 const Gfn2PostSccPotentialDeviceResults& results,
                                 const Gfn2PostSccPotentialDeviceIntermediates& intermediates,
                                 const Gfn2PostSccPotentialDeviceWorkspace& workspace,
                                 const Gfn2PostSccPotentialDeviceDiagnostics& diagnostics,
                                 const Gfn2GeometryEpochConsumerDevice* geometry,
                                 std::int64_t batch_size) noexcept {
  std::array<AddressRange, 5> public_writes{};
  if (!make_range(results.complete.shell, results.complete.shell_elements, sizeof(double),
                  &public_writes[0]) ||
      !make_range(results.complete.atomic, results.complete.atom_elements, sizeof(double),
                  &public_writes[1]) ||
      !make_range(results.complete.dipole, results.complete.dipole_elements, sizeof(double),
                  &public_writes[2]) ||
      !make_range(results.complete.quadrupole, results.complete.quadrupole_elements, sizeof(double),
                  &public_writes[3]) ||
      !make_range(results.shell_scalar, results.shell_scalar_elements, sizeof(double),
                  &public_writes[4])) {
    return false;
  }
  for (std::size_t lhs = 0; lhs < public_writes.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < public_writes.size(); ++rhs) {
      if (overlaps(public_writes[lhs], public_writes[rhs])) {
        return false;
      }
    }
  }

  std::array<AddressRange, 44> protected_ranges{};
  std::size_t count = 0u;
  auto append = [&](const void* pointer, std::int64_t elements, std::size_t element_size) {
    return count < protected_ranges.size() &&
           make_range(pointer, elements, element_size, &protected_ranges[count++]);
  };
  if (!append(input.activity.requested_mask, batch_size, sizeof(std::uint8_t)) ||
      !append(input.activity.system_statuses, batch_size, sizeof(xtbloom_status_t)) ||
      !append(input.raw_shell_charges, input.shell_elements, sizeof(double)) ||
      !append(input.raw_atomic_charges, input.atom_elements, sizeof(double)) ||
      !append(input.raw_atomic_dipoles, input.dipole_elements, sizeof(double)) ||
      !append(input.raw_atomic_quadrupoles, input.quadrupole_elements, sizeof(double)) ||
      !append(intermediates.es2_shell, intermediates.es2_shell_elements, sizeof(double)) ||
      !append(intermediates.es3_shell, intermediates.es3_shell_elements, sizeof(double)) ||
      !append(intermediates.aes2_atomic, intermediates.aes2_atomic_elements, sizeof(double)) ||
      !append(intermediates.aes2_dipole, intermediates.aes2_dipole_elements, sizeof(double)) ||
      !append(intermediates.aes2_quadrupole, intermediates.aes2_quadrupole_elements,
              sizeof(double)) ||
      !append(intermediates.complete.shell, intermediates.complete.shell_elements,
              sizeof(double)) ||
      !append(intermediates.complete.atomic, intermediates.complete.atom_elements,
              sizeof(double)) ||
      !append(intermediates.complete.dipole, intermediates.complete.dipole_elements,
              sizeof(double)) ||
      !append(intermediates.complete.quadrupole, intermediates.complete.quadrupole_elements,
              sizeof(double)) ||
      !append(intermediates.shell_scalar, intermediates.shell_scalar_elements, sizeof(double)) ||
      !append(workspace.active_mask, workspace.active_elements, sizeof(std::uint8_t)) ||
      !append(workspace.sequence_active, 1, sizeof(std::uint32_t)) ||
      !append(workspace.stage_system_errors, batch_size, sizeof(std::uint32_t)) ||
      !append(workspace.stage_device_error, 1, sizeof(std::uint32_t)) ||
      !append(workspace.es2.matrix_scratch, workspace.es2.matrix_elements, sizeof(double)) ||
      !append(workspace.es2.shell_scratch, workspace.es2.shell_elements, sizeof(double)) ||
      !append(workspace.aes2.potential_scratch, workspace.aes2.potential_elements,
              sizeof(double)) ||
      !append(workspace.aes2.scc_peer_error_scratch, workspace.aes2.scc_peer_error_elements,
              sizeof(std::uint32_t)) ||
      !append(workspace.composition.shell_scratch, workspace.composition.shell_elements,
              sizeof(double)) ||
      !append(workspace.composition.atom_scratch, workspace.composition.atom_elements,
              sizeof(double)) ||
      !append(workspace.composition.dipole_scratch, workspace.composition.dipole_elements,
              sizeof(double)) ||
      !append(workspace.composition.quadrupole_scratch, workspace.composition.quadrupole_elements,
              sizeof(double)) ||
      !append(workspace.composition.sequence_active, 1, sizeof(std::uint32_t)) ||
      !append(workspace.scalar_bridge.shell_scratch, workspace.scalar_bridge.shell_elements,
              sizeof(double)) ||
      !append(workspace.scalar_bridge.sequence_active, 1, sizeof(std::uint32_t)) ||
      !append(diagnostics.system_errors, batch_size, sizeof(std::uint32_t)) ||
      !append(diagnostics.device_error, 1, sizeof(std::uint32_t))) {
    return false;
  }
  if (geometry != nullptr &&
      (!append(geometry->epoch.value, 1, sizeof(std::uint64_t)) ||
       !append(geometry->committed_generations, batch_size, sizeof(std::uint64_t)) ||
       !append(geometry->eligible_mask, batch_size, sizeof(std::uint8_t)))) {
    return false;
  }
  /* Optional component descriptors are completely opaque when their SCC bit
   * is disabled. This is required so an energy/force owner can leave unused
   * backend-specific views unbound without the refresh path inspecting them. */
  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kD4TwoBody) &&
      (!append(intermediates.d4_atomic, intermediates.d4_atomic_elements, sizeof(double)) ||
       !append(workspace.d4.weights, workspace.d4.weight_elements, sizeof(double)) ||
       !append(workspace.d4.weight_charge_derivatives, workspace.d4.weight_elements,
               sizeof(double)) ||
       !append(workspace.d4.atom_scratch, workspace.d4.atom_elements, sizeof(double)))) {
    return false;
  }
  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kPeriodicEmbedding) &&
      (!append(intermediates.periodic_atomic, intermediates.periodic_atomic_elements,
               sizeof(double)) ||
       !append(workspace.periodic.potential_scratch, workspace.periodic.atom_elements,
               sizeof(double)) ||
       !append(workspace.periodic.sequence_active, workspace.periodic.sequence_elements,
               sizeof(std::uint32_t)))) {
    return false;
  }
  for (const AddressRange& public_write : public_writes) {
    for (std::size_t index = 0; index < count; ++index) {
      if (overlaps(public_write, protected_ranges[index])) {
        return false;
      }
    }
  }
  const auto no_public_overlap = [&](const void* pointer, std::int64_t elements,
                                     std::size_t element_size) {
    AddressRange read{};
    if (!make_range(pointer, elements, element_size, &read)) {
      return false;
    }
    for (const AddressRange& public_write : public_writes) {
      if (overlaps(public_write, read)) {
        return false;
      }
    }
    return true;
  };
  const auto& batch = plan.potential_batch;
  if (!no_public_overlap(batch.atom_offsets, batch.atom_offset_count, sizeof(std::int64_t)) ||
      !no_public_overlap(batch.batch_shell_offsets, batch.batch_shell_offset_count,
                         sizeof(std::int64_t)) ||
      !no_public_overlap(batch.qsh_offsets, batch.qsh_offset_count, sizeof(std::int64_t)) ||
      !no_public_overlap(batch.qat_offsets, batch.qat_offset_count, sizeof(std::int64_t)) ||
      !no_public_overlap(batch.dipole_offsets, batch.dipole_offset_count, sizeof(std::int64_t)) ||
      !no_public_overlap(batch.quadrupole_offsets, batch.quadrupole_offset_count,
                         sizeof(std::int64_t)) ||
      !no_public_overlap(batch.shell_to_atom, batch.shell_to_atom_count, sizeof(std::int64_t)) ||
      !no_public_overlap(plan.es2_batch.atom_offsets, plan.es2_batch.atom_offset_count,
                         sizeof(std::int64_t)) ||
      !no_public_overlap(plan.es2_batch.batch_shell_offsets,
                         plan.es2_batch.batch_shell_offset_count, sizeof(std::int64_t)) ||
      !no_public_overlap(plan.es2_batch.atom_shell_offsets, plan.es2_batch.atom_shell_offset_count,
                         sizeof(std::int64_t)) ||
      !no_public_overlap(plan.es2_batch.matrix_offsets, plan.es2_batch.matrix_offset_count,
                         sizeof(std::int64_t)) ||
      !no_public_overlap(plan.es2_batch.shell_to_atom, plan.es2_batch.shell_to_atom_count,
                         sizeof(std::int64_t)) ||
      !no_public_overlap(plan.es2_batch.shell_hardness, plan.es2_batch.shell_hardness_count,
                         sizeof(double)) ||
      !no_public_overlap(plan.es2_cache.coulomb_matrix, plan.es2_cache.matrix_elements,
                         sizeof(double)) ||
      !no_public_overlap(plan.es3_batch.batch_shell_offsets,
                         plan.es3_batch.batch_shell_offset_count, sizeof(std::int64_t)) ||
      !no_public_overlap(plan.es3_batch.shell_gamma3, plan.es3_batch.shell_gamma3_count,
                         sizeof(double)) ||
      !no_public_overlap(plan.aes2_batch.atom_offsets, plan.aes2_batch.atom_offset_count,
                         sizeof(std::int64_t)) ||
      !no_public_overlap(plan.aes2_batch.pair_offsets, plan.aes2_batch.pair_offset_count,
                         sizeof(std::int64_t)) ||
      !no_public_overlap(plan.aes2_batch.dipole_kernel, plan.aes2_batch.dipole_kernel_count,
                         sizeof(double)) ||
      !no_public_overlap(plan.aes2_batch.quadrupole_kernel, plan.aes2_batch.quadrupole_kernel_count,
                         sizeof(double)) ||
      !no_public_overlap(plan.aes2_batch.multipole_radius, plan.aes2_batch.multipole_radius_count,
                         sizeof(double)) ||
      !no_public_overlap(plan.aes2_batch.multipole_valence_cn,
                         plan.aes2_batch.multipole_valence_cn_count, sizeof(double)) ||
      !no_public_overlap(plan.aes2_cache.pair_data, plan.aes2_cache.pair_data_elements,
                         sizeof(double))) {
    return false;
  }
  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kD4TwoBody) &&
      (!no_public_overlap(plan.d4_batch.atom_offsets, plan.d4_batch.batch_size + 1,
                          sizeof(std::int64_t)) ||
       !no_public_overlap(plan.d4_batch.pair_offsets, plan.d4_batch.batch_size + 1,
                          sizeof(std::int64_t)) ||
       !no_public_overlap(plan.d4_batch.atomic_numbers, plan.d4_batch.total_atoms,
                          sizeof(std::int32_t)) ||
       !no_public_overlap(plan.d4_parameters.elements, plan.d4_parameters.element_count,
                          sizeof(Gfn2D4DeviceElementData)) ||
       !no_public_overlap(plan.d4_parameters.references, plan.d4_parameters.reference_count,
                          sizeof(Gfn2D4DeviceReferenceData)) ||
       !no_public_overlap(plan.d4_parameters.reference_c6, plan.d4_parameters.reference_c6_elements,
                          sizeof(double)) ||
       !no_public_overlap(plan.d4_cache.pair_data, plan.d4_cache.pair_data_elements,
                          sizeof(double)) ||
       !no_public_overlap(plan.d4_cache.coordination_numbers, plan.d4_cache.coordination_elements,
                          sizeof(double)))) {
    return false;
  }
  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kExplicitPointCharge) &&
      !no_public_overlap(plan.external_point_charge_cache.shell_potentials,
                         plan.external_point_charge_cache.shell_elements, sizeof(double))) {
    return false;
  }
  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kPeriodicEmbedding) &&
      (!no_public_overlap(plan.periodic_batch.atom_offsets, plan.periodic_batch.atom_offset_count,
                          sizeof(std::int64_t)) ||
       !no_public_overlap(plan.periodic_batch.matrix_offsets,
                          plan.periodic_batch.matrix_offset_count, sizeof(std::int64_t)) ||
       !no_public_overlap(plan.periodic_batch.shifts, plan.periodic_batch.shift_elements,
                          sizeof(double)) ||
       !no_public_overlap(plan.periodic_batch.response_matrices,
                          plan.periodic_batch.response_elements, sizeof(double)))) {
    return false;
  }
  return true;
}

bool validate_common(const Gfn2PostSccPotentialDevicePlan& plan,
                     const Gfn2PostSccPotentialDeviceInput& input,
                     const Gfn2PostSccPotentialDeviceResults& results,
                     const Gfn2PostSccPotentialDeviceIntermediates& intermediates,
                     const Gfn2PostSccPotentialDeviceWorkspace& workspace,
                     const Gfn2PostSccPotentialDeviceDiagnostics& diagnostics,
                     const Gfn2GeometryEpochConsumerDevice* geometry) noexcept {
  const auto& batch = plan.potential_batch;
  if (plan.plan_token == 0u || plan.geometry_generation == 0u ||
      batch.plan_token != plan.plan_token || batch.batch_size <= 0 || batch.total_atoms <= 0 ||
      batch.total_shells <= 0 ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max()) ||
      (plan.enabled_components & ~kGfn2SccPotentialAllComponents) != 0u ||
      (plan.enabled_components & kMandatoryComponents) != kMandatoryComponents ||
      input.plan_token != plan.plan_token || input.activity.plan_token != plan.plan_token ||
      results.plan_token != plan.plan_token || results.complete.plan_token != plan.plan_token ||
      intermediates.plan_token != plan.plan_token ||
      intermediates.complete.plan_token != plan.plan_token ||
      workspace.plan_token != plan.plan_token ||
      workspace.composition.plan_token != plan.plan_token ||
      workspace.scalar_bridge.plan_token != plan.plan_token ||
      diagnostics.plan_token != plan.plan_token ||
      input.activity.batch_elements != batch.batch_size ||
      diagnostics.batch_elements != batch.batch_size ||
      workspace.active_elements != batch.batch_size || workspace.sequence_elements != 1 ||
      workspace.stage_system_error_elements != batch.batch_size ||
      workspace.stage_device_error_elements != 1 || input.shell_elements != batch.total_shells ||
      input.atom_elements != batch.total_atoms ||
      input.dipole_elements != batch.total_atoms * kGfn2SccPotentialDipoleComponents ||
      input.quadrupole_elements != batch.total_atoms * kGfn2SccPotentialQuadrupoleComponents ||
      results.complete.shell_elements != batch.total_shells ||
      results.complete.atom_elements != batch.total_atoms ||
      results.complete.dipole_elements != input.dipole_elements ||
      results.complete.quadrupole_elements != input.quadrupole_elements ||
      results.shell_scalar_elements != batch.total_shells ||
      intermediates.complete.shell_elements != batch.total_shells ||
      intermediates.complete.atom_elements != batch.total_atoms ||
      intermediates.complete.dipole_elements != input.dipole_elements ||
      intermediates.complete.quadrupole_elements != input.quadrupole_elements ||
      intermediates.shell_scalar_elements != batch.total_shells ||
      intermediates.es2_shell_elements != batch.total_shells ||
      intermediates.es3_shell_elements != batch.total_shells ||
      intermediates.aes2_atomic_elements != batch.total_atoms ||
      intermediates.aes2_dipole_elements != input.dipole_elements ||
      intermediates.aes2_quadrupole_elements != input.quadrupole_elements ||
      !aligned_pointer(input.activity.requested_mask) ||
      !aligned_pointer(input.activity.system_statuses) ||
      !aligned_pointer(input.raw_shell_charges) || !aligned_pointer(input.raw_atomic_charges) ||
      !aligned_pointer(input.raw_atomic_dipoles) ||
      !aligned_pointer(input.raw_atomic_quadrupoles) || !aligned_pointer(results.complete.shell) ||
      !aligned_pointer(results.complete.atomic) || !aligned_pointer(results.complete.dipole) ||
      !aligned_pointer(results.complete.quadrupole) || !aligned_pointer(results.shell_scalar) ||
      !aligned_pointer(intermediates.es2_shell) || !aligned_pointer(intermediates.es3_shell) ||
      !aligned_pointer(intermediates.aes2_atomic) || !aligned_pointer(intermediates.aes2_dipole) ||
      !aligned_pointer(intermediates.aes2_quadrupole) ||
      !aligned_pointer(intermediates.complete.shell) ||
      !aligned_pointer(intermediates.complete.atomic) ||
      !aligned_pointer(intermediates.complete.dipole) ||
      !aligned_pointer(intermediates.complete.quadrupole) ||
      !aligned_pointer(intermediates.shell_scalar) || !aligned_pointer(workspace.active_mask) ||
      !aligned_pointer(workspace.sequence_active) ||
      !aligned_pointer(workspace.stage_system_errors) ||
      !aligned_pointer(workspace.stage_device_error) ||
      !aligned_pointer(diagnostics.system_errors) || !aligned_pointer(diagnostics.device_error)) {
    return false;
  }
  if (geometry != nullptr &&
      (geometry->plan_token != plan.plan_token || geometry->epoch.plan_token != plan.plan_token ||
       geometry->epoch.value_elements != 1 || geometry->batch_elements != batch.batch_size ||
       !aligned_pointer(geometry->epoch.value) ||
       !aligned_pointer(geometry->committed_generations) ||
       !aligned_pointer(geometry->eligible_mask))) {
    return false;
  }

  if (geometry != nullptr) {
    std::array<AddressRange, 3> reads{};
    std::array<AddressRange, 4> gate_writes{};
    if (!make_range(geometry->epoch.value, 1, sizeof(std::uint64_t), &reads[0]) ||
        !make_range(geometry->committed_generations, batch.batch_size, sizeof(std::uint64_t),
                    &reads[1]) ||
        !make_range(geometry->eligible_mask, batch.batch_size, sizeof(std::uint8_t), &reads[2]) ||
        !make_range(workspace.active_mask, batch.batch_size, sizeof(std::uint8_t),
                    &gate_writes[0]) ||
        !make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &gate_writes[1]) ||
        !make_range(diagnostics.system_errors, batch.batch_size, sizeof(std::uint32_t),
                    &gate_writes[2]) ||
        !make_range(diagnostics.device_error, 1, sizeof(std::uint32_t), &gate_writes[3])) {
      return false;
    }
    for (const AddressRange& read : reads) {
      for (const AddressRange& write : gate_writes) {
        if (overlaps(read, write)) {
          return false;
        }
      }
    }
  }

  const auto& bridge = plan.scalar_bridge_batch;
  const auto& topology = bridge.topology;
  if (topology.plan_token != plan.plan_token || topology.batch_size != batch.batch_size ||
      topology.total_atoms != batch.total_atoms || topology.total_shells != batch.total_shells ||
      topology.atom_offsets != batch.atom_offsets ||
      topology.batch_shell_offsets != batch.batch_shell_offsets ||
      topology.shell_to_atom != batch.shell_to_atom || bridge.qsh_offsets != batch.qsh_offsets ||
      bridge.qat_offsets != batch.qat_offsets ||
      bridge.qsh_offset_count != batch.qsh_offset_count ||
      bridge.qat_offset_count != batch.qat_offset_count) {
    return false;
  }

  if (plan.es2_batch.plan_token != plan.plan_token ||
      plan.es2_batch.batch_size != batch.batch_size ||
      plan.es2_batch.total_atoms != batch.total_atoms ||
      plan.es2_batch.total_shells != batch.total_shells ||
      plan.es2_cache.plan_token != plan.plan_token ||
      plan.es2_cache.geometry_generation != plan.geometry_generation ||
      plan.es3_batch.plan_token != plan.plan_token ||
      plan.es3_batch.batch_size != batch.batch_size ||
      plan.es3_batch.total_shells != batch.total_shells ||
      plan.aes2_batch.plan_token != plan.plan_token ||
      plan.aes2_batch.batch_size != batch.batch_size ||
      plan.aes2_batch.total_atoms != batch.total_atoms ||
      plan.aes2_cache.plan_token != plan.plan_token ||
      plan.aes2_cache.geometry_generation != plan.geometry_generation) {
    return false;
  }

  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kD4TwoBody)) {
    if (plan.d4_batch.plan_token != plan.plan_token ||
        plan.d4_batch.batch_size != batch.batch_size ||
        plan.d4_batch.total_atoms != batch.total_atoms ||
        plan.d4_cache.plan_token != plan.plan_token ||
        plan.d4_cache.geometry_generation != plan.geometry_generation ||
        intermediates.d4_atomic_elements != batch.total_atoms ||
        !aligned_pointer(intermediates.d4_atomic) ||
        workspace.d4.system_errors != workspace.stage_system_errors ||
        workspace.d4.system_error_elements != batch.batch_size) {
      return false;
    }
  }

  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    if (plan.external_point_charge_batch.plan_token != plan.plan_token ||
        plan.external_point_charge_batch.batch_size != batch.batch_size ||
        plan.external_point_charge_batch.total_atoms != batch.total_atoms ||
        plan.external_point_charge_batch.total_shells != batch.total_shells ||
        plan.external_point_charge_cache.plan_token != plan.plan_token ||
        plan.external_point_charge_cache.geometry_generation != plan.geometry_generation ||
        plan.external_point_charge_cache.shell_elements != batch.total_shells ||
        !aligned_pointer(plan.external_point_charge_cache.shell_potentials)) {
      return false;
    }
  }

  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    if (plan.periodic_batch.plan_token != plan.plan_token ||
        plan.periodic_batch.geometry_generation != plan.geometry_generation ||
        plan.periodic_batch.batch_size != batch.batch_size ||
        plan.periodic_batch.total_atoms != batch.total_atoms ||
        intermediates.periodic_atomic_elements != batch.total_atoms ||
        !aligned_pointer(intermediates.periodic_atomic)) {
      return false;
    }
  }

  return public_results_are_disjoint(plan, input, results, intermediates, workspace, diagnostics,
                                     geometry, batch.batch_size);
}

__global__ void initialize_activity_kernel(Gfn2ForceDeviceActivity source,
                                           Gfn2GeometryEpochConsumerDevice geometry,
                                           int dynamic_epoch, std::uint8_t* active_mask,
                                           std::uint32_t* sequence_active,
                                           std::uint32_t* system_errors,
                                           std::uint32_t* device_error) {
  __shared__ std::uint64_t epoch;
  __shared__ int invalid_geometry;
  if (threadIdx.x == 0) {
    *sequence_active = 1u;
    epoch = dynamic_epoch != 0 ? *geometry.epoch.value : 0u;
    invalid_geometry = dynamic_epoch != 0 && epoch == 0u ? 1 : 0;
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < source.batch_elements; system += blockDim.x) {
    active_mask[system] = 0u;
    const std::uint8_t requested = source.requested_mask[system];
    if (requested > 1u) {
      system_errors[system] = gfn2_post_scc_potential_error(Gfn2PostSccPotentialStage::kActivity,
                                                            kInvalidRequestedMask);
    } else if (requested == 1u && source.system_statuses[system] == XTBLOOM_STATUS_SUCCESS) {
      if (dynamic_epoch == 0) {
        active_mask[system] = 1u;
        continue;
      }
      const std::uint8_t eligible = geometry.eligible_mask[system];
      if (eligible > 1u) {
        atomicExch(&invalid_geometry, 1);
      } else if (eligible == 0u) {
        system_errors[system] = gfn2_post_scc_potential_error(Gfn2PostSccPotentialStage::kActivity,
                                                              kIneligibleGeometry);
      } else if (geometry.committed_generations[system] != epoch) {
        system_errors[system] =
            gfn2_post_scc_potential_error(Gfn2PostSccPotentialStage::kActivity, kStaleGeometry);
      } else {
        active_mask[system] = 1u;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && invalid_geometry != 0) {
    const std::uint32_t raw = epoch == 0u ? kInvalidGeometryEpoch : kInvalidEligibilityMask;
    atomicCAS(device_error, 0u,
              gfn2_post_scc_potential_error(Gfn2PostSccPotentialStage::kActivity, raw));
    atomicExch(sequence_active, 0u);
  }
}

/*
 * Fold one primitive's diagnostics into the post-SCC ledger. plan_only_code
 * is true for split SCC APIs whose device scalar is explicitly plan-only.
 * Mixed primitives classify a device code as peer-local when the same code is
 * present in at least one stage_system_errors entry.
 */
__global__ void merge_stage_kernel(std::int64_t batch_size, Gfn2PostSccPotentialStage stage,
                                   const std::uint32_t* stage_system_errors,
                                   const std::uint32_t* stage_device_error,
                                   const std::uint32_t* stage_sequence_active, int plan_only_code,
                                   std::uint8_t* active_mask, std::uint32_t* sequence_active,
                                   std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size) {
    const std::uint32_t raw = stage_system_errors[system];
    if (raw != 0u && active_mask[system] == 1u) {
      atomicCAS(system_errors + system, 0u, gfn2_post_scc_potential_error(stage, raw));
      active_mask[system] = 0u;
    }
  }
  if (system != 0) {
    return;
  }

  const std::uint32_t raw_device = atomicAdd(const_cast<std::uint32_t*>(stage_device_error), 0u);
  bool peer_summary = false;
  if (raw_device != 0u && plan_only_code == 0) {
    for (std::int64_t peer = 0; peer < batch_size; ++peer) {
      if (stage_system_errors[peer] == raw_device) {
        peer_summary = true;
        break;
      }
    }
  }
  const bool stage_closed = stage_sequence_active != nullptr &&
                            atomicAdd(const_cast<std::uint32_t*>(stage_sequence_active), 0u) != 1u;
  if ((raw_device != 0u && !peer_summary) || stage_closed) {
    const std::uint32_t raw = raw_device == 0u ? kClosedStageWithoutCode : raw_device;
    atomicCAS(device_error, 0u, gfn2_post_scc_potential_error(stage, raw));
    atomicExch(sequence_active, 0u);
  }
}

__global__ void publish_results_kernel(Gfn2SccPotentialDeviceBatch batch,
                                       const std::uint8_t* active_mask,
                                       const std::uint32_t* sequence_active,
                                       Gfn2SccPotentialDeviceResults staged,
                                       const double* staged_shell_scalar,
                                       Gfn2SccPotentialDeviceResults results,
                                       double* shell_scalar) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (atomicAdd(const_cast<std::uint32_t*>(sequence_active), 0u) != 1u ||
      active_mask[system] != 1u) {
    return;
  }
  const std::int64_t shell_begin = batch.qsh_offsets[system];
  const std::int64_t shell_end = batch.qsh_offsets[system + 1];
  const std::int64_t atom_begin = batch.qat_offsets[system];
  const std::int64_t atom_end = batch.qat_offsets[system + 1];
  const std::int64_t dipole_begin = batch.dipole_offsets[system];
  const std::int64_t dipole_end = batch.dipole_offsets[system + 1];
  const std::int64_t quadrupole_begin = batch.quadrupole_offsets[system];
  const std::int64_t quadrupole_end = batch.quadrupole_offsets[system + 1];
  const std::int64_t topology_shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t topology_shell_end = batch.batch_shell_offsets[system + 1];
  for (std::int64_t index = shell_begin + threadIdx.x; index < shell_end; index += blockDim.x) {
    results.shell[index] = staged.shell[index];
  }
  for (std::int64_t index = atom_begin + threadIdx.x; index < atom_end; index += blockDim.x) {
    results.atomic[index] = staged.atomic[index];
  }
  for (std::int64_t index = dipole_begin + threadIdx.x; index < dipole_end; index += blockDim.x) {
    results.dipole[index] = staged.dipole[index];
  }
  for (std::int64_t index = quadrupole_begin + threadIdx.x; index < quadrupole_end;
       index += blockDim.x) {
    results.quadrupole[index] = staged.quadrupole[index];
  }
  for (std::int64_t index = topology_shell_begin + threadIdx.x; index < topology_shell_end;
       index += blockDim.x) {
    shell_scalar[index] = staged_shell_scalar[index];
  }
}

cudaError_t check_launch() noexcept { return cudaGetLastError(); }

}  // namespace

static cudaError_t refresh_post_scc_potentials_impl(
    const Gfn2PostSccPotentialDevicePlan& plan, const Gfn2PostSccPotentialDeviceInput& input,
    const Gfn2PostSccPotentialDeviceResults& results,
    const Gfn2PostSccPotentialDeviceIntermediates& intermediates,
    const Gfn2PostSccPotentialDeviceWorkspace& workspace,
    const Gfn2PostSccPotentialDeviceDiagnostics& diagnostics,
    const Gfn2GeometryEpochConsumerDevice* geometry, cudaStream_t stream) noexcept {
  if (!validate_common(plan, input, results, intermediates, workspace, diagnostics, geometry)) {
    return cudaErrorInvalidValue;
  }

  const std::int64_t batch_size = plan.potential_batch.batch_size;
  const std::size_t system_error_bytes =
      static_cast<std::size_t>(batch_size) * sizeof(std::uint32_t);
  cudaError_t status = cudaMemsetAsync(diagnostics.system_errors, 0, system_error_bytes, stream);
  if (status == cudaSuccess) {
    status =
        cudaMemsetAsync(diagnostics.device_error, 0, sizeof(*diagnostics.device_error), stream);
  }
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int gate_blocks =
      static_cast<unsigned int>((batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const Gfn2GeometryEpochConsumerDevice consumer =
      geometry == nullptr ? Gfn2GeometryEpochConsumerDevice{} : *geometry;
  initialize_activity_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      input.activity, consumer, geometry == nullptr ? 0 : 1, workspace.active_mask,
      workspace.sequence_active, diagnostics.system_errors, diagnostics.device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }

  const Gfn2SccIterationDeviceActivity activity{workspace.active_mask, workspace.sequence_active,
                                                batch_size, 1, plan.plan_token};
  const auto merge_stage = [&](Gfn2PostSccPotentialStage stage, bool plan_only,
                               const std::uint32_t* stage_sequence) {
    merge_stage_kernel<<<gate_blocks, kThreadsPerBlock, 0, stream>>>(
        batch_size, stage, workspace.stage_system_errors, workspace.stage_device_error,
        stage_sequence, plan_only ? 1 : 0, workspace.active_mask, workspace.sequence_active,
        diagnostics.system_errors, diagnostics.device_error);
    return check_launch();
  };

  status = reset_gfn2_es2_scc_errors_cuda(batch_size, workspace.stage_system_errors,
                                          workspace.stage_device_error, stream);
  if (status == cudaSuccess) {
    status = evaluate_gfn2_es2_scc_potential_cuda(
        plan.es2_batch, plan.es2_cache, plan.geometry_generation, activity, input.raw_shell_charges,
        intermediates.es2_shell, workspace.es2, workspace.stage_system_errors,
        workspace.stage_device_error, stream);
  }
  if (status == cudaSuccess) {
    status = merge_stage(Gfn2PostSccPotentialStage::kES2, true, nullptr);
  }
  if (status != cudaSuccess) {
    return status;
  }

  status = reset_gfn2_es3_scc_errors_cuda(batch_size, workspace.stage_system_errors,
                                          workspace.stage_device_error, stream);
  if (status == cudaSuccess) {
    status = evaluate_gfn2_es3_scc_potential_cuda(
        plan.es3_batch, activity, input.raw_shell_charges, intermediates.es3_shell,
        workspace.stage_system_errors, workspace.stage_device_error, stream);
  }
  if (status == cudaSuccess) {
    status = merge_stage(Gfn2PostSccPotentialStage::kES3, true, nullptr);
  }
  if (status != cudaSuccess) {
    return status;
  }

  status = reset_gfn2_aes2_device_errors_cuda(batch_size, workspace.stage_system_errors,
                                              workspace.stage_device_error, stream);
  if (status == cudaSuccess) {
    status = evaluate_gfn2_aes2_scc_potential_cuda(
        plan.aes2_batch, plan.aes2_cache, plan.geometry_generation, activity,
        input.raw_atomic_charges, input.raw_atomic_dipoles, input.raw_atomic_quadrupoles,
        intermediates.aes2_atomic, intermediates.aes2_dipole, intermediates.aes2_quadrupole,
        workspace.aes2, workspace.stage_system_errors, workspace.stage_device_error, stream);
  }
  if (status == cudaSuccess) {
    status = merge_stage(Gfn2PostSccPotentialStage::kAES2, true, nullptr);
  }
  if (status != cudaSuccess) {
    return status;
  }

  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kD4TwoBody)) {
    status = reset_gfn2_d4_device_errors_cuda(batch_size, workspace.stage_system_errors,
                                              workspace.stage_device_error, stream);
    if (status == cudaSuccess) {
      status = evaluate_gfn2_d4_scc_potential_cuda(
          plan.d4_batch, plan.d4_parameters, plan.d4_cache, plan.geometry_generation,
          input.raw_atomic_charges, activity, intermediates.d4_atomic, workspace.d4,
          workspace.stage_device_error, stream);
    }
    if (status == cudaSuccess) {
      status = merge_stage(Gfn2PostSccPotentialStage::kD4, false, nullptr);
    }
    if (status != cudaSuccess) {
      return status;
    }
  }

  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    status = reset_gfn2_periodic_embedding_scc_device_errors_cuda(
        batch_size, workspace.stage_system_errors, workspace.stage_device_error, stream);
    if (status == cudaSuccess) {
      status = evaluate_gfn2_periodic_embedding_scc_potential_cuda(
          plan.periodic_batch, plan.geometry_generation, input.raw_atomic_charges, activity,
          intermediates.periodic_atomic, workspace.periodic, workspace.stage_system_errors,
          workspace.stage_device_error, stream);
    }
    if (status == cudaSuccess) {
      status = merge_stage(Gfn2PostSccPotentialStage::kPeriodicEmbedding, true, nullptr);
    }
    if (status != cudaSuccess) {
      return status;
    }
  }

  Gfn2SccPotentialDeviceComponents components{};
  components.enabled_components = plan.enabled_components;
  components.es2_shell = intermediates.es2_shell;
  components.es2_shell_elements = intermediates.es2_shell_elements;
  components.es3_shell = intermediates.es3_shell;
  components.es3_shell_elements = intermediates.es3_shell_elements;
  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kExplicitPointCharge)) {
    components.explicit_point_charge_shell = plan.external_point_charge_cache.shell_potentials;
    components.explicit_point_charge_shell_elements =
        plan.external_point_charge_cache.shell_elements;
  }
  components.aes2_atomic = intermediates.aes2_atomic;
  components.aes2_atomic_elements = intermediates.aes2_atomic_elements;
  components.aes2_dipole = intermediates.aes2_dipole;
  components.aes2_dipole_elements = intermediates.aes2_dipole_elements;
  components.aes2_quadrupole = intermediates.aes2_quadrupole;
  components.aes2_quadrupole_elements = intermediates.aes2_quadrupole_elements;
  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kD4TwoBody)) {
    components.d4_atomic = intermediates.d4_atomic;
    components.d4_atomic_elements = intermediates.d4_atomic_elements;
  }
  if (component_enabled(plan.enabled_components, Gfn2SccPotentialComponent::kPeriodicEmbedding)) {
    components.periodic_atomic = intermediates.periodic_atomic;
    components.periodic_atomic_elements = intermediates.periodic_atomic_elements;
  }
  components.plan_token = plan.plan_token;

  status = reset_gfn2_scc_potential_device_errors_cuda(batch_size, workspace.stage_system_errors,
                                                       workspace.stage_device_error, stream);
  if (status == cudaSuccess) {
    status = compose_gfn2_scc_potentials_cuda(
        plan.potential_batch, components, activity, intermediates.complete, workspace.composition,
        workspace.stage_system_errors, workspace.stage_device_error, stream);
  }
  if (status == cudaSuccess) {
    status = merge_stage(Gfn2PostSccPotentialStage::kComposition, false,
                         workspace.composition.sequence_active);
  }
  if (status != cudaSuccess) {
    return status;
  }

  status = reset_gfn2_scc_bridge_device_errors_cuda(
      batch_size, workspace.stage_system_errors, workspace.stage_device_error,
      workspace.scalar_bridge.sequence_active, stream);
  if (status == cudaSuccess) {
    const Gfn2SccBridgeDevicePotentialFields fields{
        intermediates.complete.shell, intermediates.complete.shell_elements,
        intermediates.complete.atomic, intermediates.complete.atom_elements, plan.plan_token};
    status = collect_gfn2_scc_shell_scalar_potential_cuda(
        plan.scalar_bridge_batch, fields, activity, intermediates.shell_scalar,
        intermediates.shell_scalar_elements, workspace.scalar_bridge, workspace.stage_system_errors,
        workspace.stage_device_error, stream);
  }
  if (status == cudaSuccess) {
    status = merge_stage(Gfn2PostSccPotentialStage::kScalarBridge, false,
                         workspace.scalar_bridge.sequence_active);
  }
  if (status != cudaSuccess) {
    return status;
  }

  publish_results_kernel<<<static_cast<unsigned int>(batch_size), kThreadsPerBlock, 0, stream>>>(
      plan.potential_batch, workspace.active_mask, workspace.sequence_active,
      intermediates.complete, intermediates.shell_scalar, results.complete, results.shell_scalar);
  return check_launch();
}

cudaError_t refresh_gfn2_post_scc_potentials_cuda(
    const Gfn2PostSccPotentialDevicePlan& plan, const Gfn2PostSccPotentialDeviceInput& input,
    const Gfn2PostSccPotentialDeviceResults& results,
    const Gfn2PostSccPotentialDeviceIntermediates& intermediates,
    const Gfn2PostSccPotentialDeviceWorkspace& workspace,
    const Gfn2PostSccPotentialDeviceDiagnostics& diagnostics, cudaStream_t stream) noexcept {
  return refresh_post_scc_potentials_impl(plan, input, results, intermediates, workspace,
                                          diagnostics, nullptr, stream);
}

cudaError_t refresh_gfn2_post_scc_potentials_cuda(
    const Gfn2PostSccPotentialDevicePlan& plan, const Gfn2PostSccPotentialDeviceInput& input,
    const Gfn2PostSccPotentialDeviceResults& results,
    const Gfn2PostSccPotentialDeviceIntermediates& intermediates,
    const Gfn2PostSccPotentialDeviceWorkspace& workspace,
    const Gfn2PostSccPotentialDeviceDiagnostics& diagnostics,
    const Gfn2GeometryEpochConsumerDevice& geometry, cudaStream_t stream) noexcept {
  return refresh_post_scc_potentials_impl(plan, input, results, intermediates, workspace,
                                          diagnostics, &geometry, stream);
}

}  // namespace xtbloom::detail::cuda
