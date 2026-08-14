#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_force_composition.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

constexpr __host__ __device__ bool component_enabled(
    std::uint32_t mask, Gfn2ForceCompositionComponent component) noexcept {
  return (mask & static_cast<std::uint32_t>(component)) != 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

bool make_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                AddressRange* range) noexcept {
  if (range == nullptr || elements < 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes == 0u) {
    if (pointer != nullptr) {
      return false;
    }
    *range = {};
    return true;
  }
  if (pointer == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  *range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& lhs, const AddressRange& rhs) noexcept {
  return lhs.begin != lhs.end && rhs.begin != rhs.end && lhs.begin < rhs.end && rhs.begin < lhs.end;
}

bool valid_optional(const void* pointer, std::int64_t elements, std::int64_t required_elements,
                    std::size_t alignment, bool enabled) noexcept {
  if (!enabled || required_elements == 0) {
    return pointer == nullptr && elements == 0;
  }
  return elements == required_elements && is_aligned(pointer, alignment);
}

bool validate_launch(const Gfn2ForceCompositionDeviceBatch& batch,
                     const Gfn2ForceDeviceActivity& activity,
                     const Gfn2ForceCompositionDeviceInput& input,
                     const Gfn2ForceCompositionDeviceOutput& output,
                     const Gfn2ForceCompositionDeviceWorkspace& workspace,
                     std::uint32_t* system_errors, std::uint32_t* plan_error) noexcept {
  if (batch.batch_size <= 0 ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.total_atoms <= 0 || batch.total_atoms > std::numeric_limits<std::int64_t>::max() / 3 ||
      batch.total_point_charges < 0 ||
      batch.total_point_charges > std::numeric_limits<std::int64_t>::max() / 3 ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.point_charge_offset_count != batch.batch_size + 1 || batch.plan_token == 0u ||
      (batch.enabled_components & ~kGfn2ForceCompositionAllComponents) != 0u ||
      activity.plan_token != batch.plan_token || input.plan_token != batch.plan_token ||
      output.plan_token != batch.plan_token || workspace.plan_token != batch.plan_token ||
      activity.batch_elements != batch.batch_size ||
      !is_aligned(activity.requested_mask, alignof(std::uint8_t)) ||
      !is_aligned(activity.system_statuses, alignof(xtbloom_status_t)) ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.point_charge_offsets, alignof(std::int64_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(plan_error, alignof(std::uint32_t)) || workspace.sequence_elements != 1 ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t))) {
    return false;
  }

  const std::int64_t qm_elements = batch.total_atoms * 3;
  const std::int64_t point_elements = batch.total_point_charges * 3;
  if (qm_elements / 3 != batch.total_atoms ||
      (batch.total_point_charges != 0 && point_elements / 3 != batch.total_point_charges)) {
    return false;
  }
  const bool qm_output = output.qm_forces != nullptr || output.qm_force_elements != 0;
  const bool point_output = output.point_forces != nullptr || output.point_force_elements != 0;
  const bool electronic = component_enabled(batch.enabled_components,
                                            Gfn2ForceCompositionComponent::kElectronicGradient);
  const bool classical =
      component_enabled(batch.enabled_components, Gfn2ForceCompositionComponent::kClassicalForce);
  const bool explicit_pc = component_enabled(
      batch.enabled_components, Gfn2ForceCompositionComponent::kExplicitPointChargeForce);
  const bool electric_field = input.electric_field_vectors != nullptr ||
                              input.electric_field_vector_elements != 0 ||
                              input.atomic_charges != nullptr || input.atomic_charge_elements != 0;
  if ((!qm_output && !point_output) || (!qm_output && (electronic || classical)) ||
      (point_output && !explicit_pc) || (qm_output && !(electronic || classical || explicit_pc)) ||
      !valid_optional(output.qm_forces, output.qm_force_elements, qm_elements, alignof(double),
                      qm_output) ||
      !valid_optional(output.point_forces, output.point_force_elements, point_elements,
                      alignof(double), point_output) ||
      !valid_optional(workspace.qm_force_scratch, workspace.qm_force_elements, qm_elements,
                      alignof(double), qm_output) ||
      !valid_optional(workspace.point_force_scratch, workspace.point_force_elements, point_elements,
                      alignof(double), point_output) ||
      !valid_optional(input.electronic_gradients, input.electronic_gradient_elements, qm_elements,
                      alignof(double), qm_output && electronic) ||
      !valid_optional(input.classical_forces, input.classical_force_elements, qm_elements,
                      alignof(double), qm_output && classical) ||
      !valid_optional(input.explicit_qm_forces, input.explicit_qm_force_elements, qm_elements,
                      alignof(double),
                      qm_output && explicit_pc && batch.total_point_charges != 0) ||
      !valid_optional(input.explicit_point_forces, input.explicit_point_force_elements,
                      point_elements, alignof(double), point_output && explicit_pc) ||
      !valid_optional(input.electric_field_vectors, input.electric_field_vector_elements,
                      batch.batch_size * 3, alignof(double), electric_field) ||
      !valid_optional(input.atomic_charges, input.atomic_charge_elements, batch.total_atoms,
                      alignof(double), electric_field)) {
    return false;
  }

  std::array<AddressRange, 10> reads{};
  std::array<AddressRange, 5> writes{};
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(std::int64_t), &reads[0]) ||
      !make_range(batch.point_charge_offsets, batch.point_charge_offset_count, sizeof(std::int64_t),
                  &reads[1]) ||
      !make_range(activity.requested_mask, activity.batch_elements, sizeof(std::uint8_t),
                  &reads[2]) ||
      !make_range(activity.system_statuses, activity.batch_elements, sizeof(xtbloom_status_t),
                  &reads[3]) ||
      !make_range(input.electronic_gradients, input.electronic_gradient_elements, sizeof(double),
                  &reads[4]) ||
      !make_range(input.classical_forces, input.classical_force_elements, sizeof(double),
                  &reads[5]) ||
      !make_range(input.explicit_qm_forces, input.explicit_qm_force_elements, sizeof(double),
                  &reads[6]) ||
      !make_range(input.explicit_point_forces, input.explicit_point_force_elements, sizeof(double),
                  &reads[7]) ||
      !make_range(input.electric_field_vectors, input.electric_field_vector_elements,
                  sizeof(double), &reads[8]) ||
      !make_range(input.atomic_charges, input.atomic_charge_elements, sizeof(double), &reads[9]) ||
      !make_range(output.qm_forces, output.qm_force_elements, sizeof(double), &writes[0]) ||
      !make_range(output.point_forces, output.point_force_elements, sizeof(double), &writes[1]) ||
      !make_range(workspace.qm_force_scratch, workspace.qm_force_elements, sizeof(double),
                  &writes[2]) ||
      !make_range(workspace.point_force_scratch, workspace.point_force_elements, sizeof(double),
                  &writes[3]) ||
      !make_range(workspace.sequence_active, workspace.sequence_elements, sizeof(std::uint32_t),
                  &writes[4])) {
    return false;
  }

  AddressRange system_error_range;
  AddressRange plan_error_range;
  if (!make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &system_error_range) ||
      !make_range(plan_error, 1, sizeof(std::uint32_t), &plan_error_range)) {
    return false;
  }
  for (std::size_t lhs = 0; lhs < writes.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < writes.size(); ++rhs) {
      if (ranges_overlap(writes[lhs], writes[rhs])) {
        return false;
      }
    }
    if (ranges_overlap(writes[lhs], system_error_range) ||
        ranges_overlap(writes[lhs], plan_error_range)) {
      return false;
    }
    for (const AddressRange& read : reads) {
      if (ranges_overlap(writes[lhs], read)) {
        return false;
      }
    }
  }
  if (ranges_overlap(system_error_range, plan_error_range)) {
    return false;
  }
  for (const AddressRange& read : reads) {
    if (ranges_overlap(system_error_range, read) || ranges_overlap(plan_error_range, read)) {
      return false;
    }
  }
  return true;
}

__device__ void record_plan_error(std::uint32_t* plan_error,
                                  Gfn2ForceCompositionDeviceError error) {
  atomicCAS(plan_error, static_cast<std::uint32_t>(Gfn2ForceCompositionDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    Gfn2ForceCompositionDeviceError error) {
  atomicCAS(system_errors + system,
            static_cast<std::uint32_t>(Gfn2ForceCompositionDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__global__ void preflight_force_topology_kernel(Gfn2ForceCompositionDeviceBatch batch,
                                                std::uint32_t* plan_error) {
  if (atomicAdd(plan_error, 0u) != 0u) {
    return;
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t point_begin = batch.point_charge_offsets[system];
    const std::int64_t point_end = batch.point_charge_offsets[system + 1];
    if (atom_begin < 0 || atom_begin >= atom_end || atom_end > batch.total_atoms ||
        point_begin < 0 || point_begin > point_end || point_end > batch.total_point_charges ||
        (system == 0 && (atom_begin != 0 || point_begin != 0)) ||
        (system + 1 == batch.batch_size &&
         (atom_end != batch.total_atoms || point_end != batch.total_point_charges))) {
      record_plan_error(plan_error, Gfn2ForceCompositionDeviceError::kInvalidOffsets);
    }
  }
}

__global__ void capture_force_sequence_kernel(const std::uint32_t* plan_error,
                                              Gfn2ForceCompositionDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) == 0u ? 1u : 0u;
  }
}

__global__ void compose_force_kernel(Gfn2ForceCompositionDeviceBatch batch,
                                     Gfn2ForceDeviceActivity activity,
                                     Gfn2ForceCompositionDeviceInput input,
                                     Gfn2ForceCompositionDeviceOutput output,
                                     Gfn2ForceCompositionDeviceWorkspace workspace,
                                     std::uint32_t* system_errors,
                                     const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ std::int64_t atom_begin;
  __shared__ std::int64_t atom_end;
  __shared__ std::int64_t point_begin;
  __shared__ std::int64_t point_end;
  if (threadIdx.x == 0) {
    active = 0;
    if (atomicAdd(workspace.sequence_active, 0u) == 1u &&
        atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) == 0u &&
        atomicAdd(system_errors + system, 0u) == 0u) {
      const std::uint8_t requested = activity.requested_mask[system];
      if (requested > 1u) {
        record_system_error(system_errors, system,
                            Gfn2ForceCompositionDeviceError::kInvalidRequestedMask);
      } else if (requested == 1u && activity.system_statuses[system] == XTBLOOM_STATUS_SUCCESS) {
        atom_begin = batch.atom_offsets[system];
        atom_end = batch.atom_offsets[system + 1];
        point_begin = batch.point_charge_offsets[system];
        point_end = batch.point_charge_offsets[system + 1];
        active = 1;
      }
    }
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const bool electronic = component_enabled(batch.enabled_components,
                                            Gfn2ForceCompositionComponent::kElectronicGradient);
  const bool classical =
      component_enabled(batch.enabled_components, Gfn2ForceCompositionComponent::kClassicalForce);
  const bool explicit_pc = component_enabled(
      batch.enabled_components, Gfn2ForceCompositionComponent::kExplicitPointChargeForce);
  const bool electric_field = input.electric_field_vectors != nullptr;

  if (output.qm_forces != nullptr) {
    for (std::int64_t coordinate = atom_begin * 3 + threadIdx.x; coordinate < atom_end * 3;
         coordinate += blockDim.x) {
      double value = 0.0;
      if (electronic) {
        const double gradient = input.electronic_gradients[coordinate];
        if (!isfinite(gradient)) {
          record_system_error(system_errors, system,
                              Gfn2ForceCompositionDeviceError::kNonfiniteElectronicGradient);
          atomicExch(&active, 0);
          continue;
        }
        value = -gradient;
      }
      if (classical) {
        const double force = input.classical_forces[coordinate];
        if (!isfinite(force)) {
          record_system_error(system_errors, system,
                              Gfn2ForceCompositionDeviceError::kNonfiniteClassicalForce);
          atomicExch(&active, 0);
          continue;
        }
        value += force;
        if (!isfinite(value)) {
          record_system_error(system_errors, system,
                              Gfn2ForceCompositionDeviceError::kNonfiniteForceArithmetic);
          atomicExch(&active, 0);
          continue;
        }
      }
      if (explicit_pc && batch.total_point_charges != 0) {
        const double force = input.explicit_qm_forces[coordinate];
        if (!isfinite(force)) {
          record_system_error(system_errors, system,
                              Gfn2ForceCompositionDeviceError::kNonfiniteExplicitQmForce);
          atomicExch(&active, 0);
          continue;
        }
        value += force;
        if (!isfinite(value)) {
          record_system_error(system_errors, system,
                              Gfn2ForceCompositionDeviceError::kNonfiniteForceArithmetic);
          atomicExch(&active, 0);
          continue;
        }
      }
      if (electric_field) {
        const std::int64_t atom = coordinate / 3;
        const std::int64_t axis = coordinate % 3;
        const double charge = input.atomic_charges[atom];
        const double field = input.electric_field_vectors[system * 3 + axis];
        const double force = charge * field;
        if (!isfinite(charge) || !isfinite(field) || !isfinite(force)) {
          record_system_error(system_errors, system,
                              Gfn2ForceCompositionDeviceError::kNonfiniteElectricFieldForce);
          atomicExch(&active, 0);
          continue;
        }
        value += force;
        if (!isfinite(value)) {
          record_system_error(system_errors, system,
                              Gfn2ForceCompositionDeviceError::kNonfiniteForceArithmetic);
          atomicExch(&active, 0);
          continue;
        }
      }
      workspace.qm_force_scratch[coordinate] = value;
    }
  }
  if (output.point_forces != nullptr) {
    for (std::int64_t coordinate = point_begin * 3 + threadIdx.x; coordinate < point_end * 3;
         coordinate += blockDim.x) {
      const double force = input.explicit_point_forces[coordinate];
      if (!isfinite(force)) {
        record_system_error(system_errors, system,
                            Gfn2ForceCompositionDeviceError::kNonfiniteExplicitPointForce);
        atomicExch(&active, 0);
        continue;
      }
      workspace.point_force_scratch[coordinate] = force;
    }
  }
  __syncthreads();
  if (active == 0 || atomicAdd(system_errors + system, 0u) != 0u) {
    return;
  }

  if (output.qm_forces != nullptr) {
    for (std::int64_t coordinate = atom_begin * 3 + threadIdx.x; coordinate < atom_end * 3;
         coordinate += blockDim.x) {
      output.qm_forces[coordinate] = workspace.qm_force_scratch[coordinate];
    }
  }
  if (output.point_forces != nullptr) {
    for (std::int64_t coordinate = point_begin * 3 + threadIdx.x; coordinate < point_end * 3;
         coordinate += blockDim.x) {
      output.point_forces[coordinate] = workspace.point_force_scratch[coordinate];
    }
  }
}

}  // namespace

cudaError_t reset_gfn2_force_composition_device_errors_cuda(std::int64_t batch_size,
                                                            std::uint32_t* system_errors,
                                                            std::uint32_t* plan_error,
                                                            cudaStream_t stream) noexcept {
  if (batch_size <= 0 ||
      static_cast<std::uint64_t>(batch_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                     sizeof(*system_errors)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(plan_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  AddressRange systems;
  AddressRange plan;
  if (!make_range(system_errors, batch_size, sizeof(*system_errors), &systems) ||
      !make_range(plan_error, 1, sizeof(*plan_error), &plan) || ranges_overlap(systems, plan)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaMemsetAsync(plan_error, 0, sizeof(*plan_error), stream);
}

cudaError_t compose_gfn2_forces_cuda(const Gfn2ForceCompositionDeviceBatch& batch,
                                     const Gfn2ForceDeviceActivity& activity,
                                     const Gfn2ForceCompositionDeviceInput& input,
                                     const Gfn2ForceCompositionDeviceOutput& output,
                                     const Gfn2ForceCompositionDeviceWorkspace& workspace,
                                     std::uint32_t* system_errors, std::uint32_t* plan_error,
                                     cudaStream_t stream) noexcept {
  if (!validate_launch(batch, activity, input, output, workspace, system_errors, plan_error)) {
    return cudaErrorInvalidValue;
  }
  preflight_force_topology_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, plan_error);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  capture_force_sequence_kernel<<<1, 1, 0, stream>>>(plan_error, workspace);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  compose_force_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                         stream>>>(batch, activity, input, output, workspace, system_errors,
                                   plan_error);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
