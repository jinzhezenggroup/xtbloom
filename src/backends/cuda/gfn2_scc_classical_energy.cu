#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc_classical_energy.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

constexpr __host__ __device__ std::uint32_t component_bit(
    Gfn2SccClassicalEnergyComponent component) noexcept {
  return static_cast<std::uint32_t>(component);
}

constexpr __host__ __device__ bool component_enabled(
    std::uint32_t mask, Gfn2SccClassicalEnergyComponent component) noexcept {
  return (mask & component_bit(component)) != 0u;
}

__device__ bool sequence_is_active(const Gfn2SccClassicalEnergyDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2SccClassicalEnergyDeviceError::kSuccess);
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error,
                                    Gfn2SccClassicalEnergyDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system,
                static_cast<std::uint32_t>(Gfn2SccClassicalEnergyDeviceError::kSuccess),
                code) == static_cast<std::uint32_t>(Gfn2SccClassicalEnergyDeviceError::kSuccess)) {
    atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2SccClassicalEnergyDeviceError::kSuccess),
              code);
  }
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        Gfn2SccClassicalEnergyDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                static_cast<std::uint32_t>(Gfn2SccClassicalEnergyDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

__device__ void load_component(double* values, int component, bool enabled, const double* source,
                               std::int64_t system, Gfn2SccClassicalEnergyDeviceError error,
                               bool* finite, std::uint32_t* system_errors,
                               std::uint32_t* device_error) {
  if (!enabled) {
    values[component] = 0.0;
    return;
  }
  const double value = source[system];
  values[component] = value;
  if (!isfinite(value)) {
    record_system_error(system_errors, system, device_error, error);
    *finite = false;
  }
}

__device__ double evaluate_field_energy(const Gfn2SccClassicalEnergyDeviceBatch& batch,
                                        const Gfn2SccClassicalEnergyDeviceInput& input,
                                        std::int64_t system, bool* finite,
                                        std::uint32_t* system_errors, std::uint32_t* device_error) {
  if (batch.electric_field.plan_token == 0u) return 0.0;
  const std::int64_t begin = batch.electric_field.atom_offsets[system];
  const std::int64_t end = batch.electric_field.atom_offsets[system + 1];
  double energy = 0.0;
  for (std::int64_t atom = begin; atom < end; ++atom) {
    double contribution = input.electric_field_potentials.atomic[atom] *
                          input.electric_field_multipoles.atomic_charges[atom];
    if (batch.model == XtbModelFlavor::kGfn2) {
#pragma unroll
      for (int component = 0; component < 3; ++component) {
        const std::int64_t index = atom * 3 + component;
        contribution = fma(input.electric_field_potentials.dipole[index],
                           input.electric_field_multipoles.atomic_dipoles[index], contribution);
      }
    }
    energy += contribution;
    if (!isfinite(contribution) || !isfinite(energy)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccClassicalEnergyDeviceError::kNonfiniteElectricField);
      *finite = false;
      return 0.0;
    }
  }
  return energy;
}

/*
 * One lane owns one system. Seven values are cheap enough to preflight and sum
 * serially, preserving the CPU component order without a lossy parallel tree.
 */
__global__ void aggregate_classical_energy_kernel(Gfn2SccClassicalEnergyDeviceBatch batch,
                                                  Gfn2SccClassicalEnergyDeviceInput input,
                                                  Gfn2SccClassicalEnergyDeviceActivity activity,
                                                  Gfn2SccClassicalEnergyDeviceWorkspace workspace,
                                                  std::uint32_t* system_errors,
                                                  std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= batch.batch_size || !sequence_is_active(workspace) ||
      !system_is_valid(system_errors, system)) {
    return;
  }

  const std::uint8_t active = activity.active_mask == nullptr ? 1u : activity.active_mask[system];
  if (active == 0u) {
    return;
  }
  if (active != 1u) {
    record_system_error(system_errors, system, device_error,
                        Gfn2SccClassicalEnergyDeviceError::kInvalidActiveMask);
    return;
  }

  double values[kGfn2SccClassicalStorageComponents - 1];
  bool finite = true;
  load_component(values, 0,
                 component_enabled(batch.enabled_components, Gfn2SccClassicalEnergyComponent::kES2),
                 input.es2, system, Gfn2SccClassicalEnergyDeviceError::kNonfiniteES2, &finite,
                 system_errors, device_error);
  load_component(values, 1,
                 component_enabled(batch.enabled_components, Gfn2SccClassicalEnergyComponent::kES3),
                 input.es3, system, Gfn2SccClassicalEnergyDeviceError::kNonfiniteES3, &finite,
                 system_errors, device_error);
  load_component(
      values, 2,
      component_enabled(batch.enabled_components, Gfn2SccClassicalEnergyComponent::kAES2),
      input.aes2, system, Gfn2SccClassicalEnergyDeviceError::kNonfiniteAES2, &finite, system_errors,
      device_error);
  load_component(
      values, 3,
      component_enabled(batch.enabled_components, Gfn2SccClassicalEnergyComponent::kD4TwoBody),
      input.d4_two_body, system, Gfn2SccClassicalEnergyDeviceError::kNonfiniteD4TwoBody, &finite,
      system_errors, device_error);
  load_component(values, 4,
                 component_enabled(batch.enabled_components,
                                   Gfn2SccClassicalEnergyComponent::kExplicitPointCharge),
                 input.explicit_point_charge, system,
                 Gfn2SccClassicalEnergyDeviceError::kNonfiniteExplicitPointCharge, &finite,
                 system_errors, device_error);
  values[5] = evaluate_field_energy(batch, input, system, &finite, system_errors, device_error);
  load_component(values, 6,
                 component_enabled(batch.enabled_components,
                                   Gfn2SccClassicalEnergyComponent::kPeriodicEmbedding),
                 input.periodic_embedding, system,
                 Gfn2SccClassicalEnergyDeviceError::kNonfinitePeriodicEmbedding, &finite,
                 system_errors, device_error);
  if (!finite) {
    return;
  }

  double total = 0.0;
#pragma unroll
  for (int component = 0; component < kGfn2SccClassicalStorageComponents - 1; ++component) {
    const double updated = total + values[component];
    if (!isfinite(updated)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SccClassicalEnergyDeviceError::kNonfiniteTotalArithmetic);
      return;
    }
    total = updated;
  }

#pragma unroll
  for (int component = 0; component < 5; ++component) {
    workspace.component_scratch[static_cast<std::int64_t>(component) * batch.batch_size + system] =
        values[component];
  }
  /* Preserve the legacy [periodic,total] slots and append the new field
   * diagnostic so field-free arena/report consumers retain their offsets. */
  workspace.component_scratch[5 * batch.batch_size + system] = values[6];
  workspace.component_scratch[6 * batch.batch_size + system] = total;
  if (workspace.component_elements >= batch.batch_size * kGfn2SccClassicalStorageComponents) {
    workspace.component_scratch[7 * batch.batch_size + system] = values[5];
  }
}

__global__ void publish_classical_energy_kernel(Gfn2SccClassicalEnergyDeviceBatch batch,
                                                Gfn2SccClassicalEnergyDeviceActivity activity,
                                                Gfn2SccClassicalEnergyDeviceDiagnostics diagnostics,
                                                Gfn2SccClassicalEnergyDeviceWorkspace workspace,
                                                const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= batch.batch_size || !sequence_is_active(workspace) ||
      !system_is_valid(system_errors, system) ||
      (activity.active_mask != nullptr && activity.active_mask[system] != 1u)) {
    return;
  }

  diagnostics.es2[system] = workspace.component_scratch[system];
  diagnostics.es3[system] = workspace.component_scratch[batch.batch_size + system];
  if (diagnostics.aes2 != nullptr) {
    diagnostics.aes2[system] = workspace.component_scratch[2 * batch.batch_size + system];
  }
  diagnostics.d4_two_body[system] = workspace.component_scratch[3 * batch.batch_size + system];
  diagnostics.explicit_point_charge[system] =
      workspace.component_scratch[4 * batch.batch_size + system];
  if (diagnostics.electric_field != nullptr) {
    diagnostics.electric_field[system] =
        batch.electric_field.plan_token != 0u
            ? workspace.component_scratch[7 * batch.batch_size + system]
            : 0.0;
  }
  diagnostics.periodic_embedding[system] =
      workspace.component_scratch[5 * batch.batch_size + system];
  diagnostics.classical_total[system] = workspace.component_scratch[6 * batch.batch_size + system];
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                AddressRange* range) noexcept {
  if (elements < 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
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

bool valid_input_component(const double* pointer, std::int64_t elements, std::int64_t batch_size,
                           bool enabled) noexcept {
  if (!enabled) {
    return pointer == nullptr && elements == 0;
  }
  return elements == batch_size && is_aligned(pointer, alignof(double));
}

bool valid_output(double* pointer, std::int64_t elements, std::int64_t batch_size) noexcept {
  return elements == batch_size && is_aligned(pointer, alignof(double));
}

bool valid_optional_output(double* pointer, std::int64_t elements,
                           std::int64_t batch_size) noexcept {
  return (pointer == nullptr && elements == 0) || valid_output(pointer, elements, batch_size);
}

bool valid_field_binding(const Gfn2SccClassicalEnergyDeviceBatch& batch,
                         const Gfn2SccClassicalEnergyDeviceInput& input) noexcept {
  const auto& field = batch.electric_field;
  const auto& multipoles = input.electric_field_multipoles;
  const auto& potentials = input.electric_field_potentials;
  if (field.plan_token == 0u) {
    return field.batch_size == 0 && field.total_atoms == 0 && field.atom_offset_count == 0 &&
           field.atom_offsets == nullptr && multipoles.plan_token == 0u &&
           multipoles.atomic_charges == nullptr && multipoles.atom_elements == 0 &&
           multipoles.atomic_dipoles == nullptr && multipoles.dipole_elements == 0 &&
           potentials.plan_token == 0u && potentials.atomic == nullptr &&
           potentials.atom_elements == 0 && potentials.dipole == nullptr &&
           potentials.dipole_elements == 0;
  }
  const bool multipoles_enabled = batch.model == XtbModelFlavor::kGfn2;
  return field.plan_token == batch.plan_token && field.batch_size == batch.batch_size &&
         field.total_atoms > 0 &&
         field.total_atoms <= std::numeric_limits<std::int64_t>::max() / 3 &&
         field.atom_offset_count == batch.batch_size + 1 &&
         is_aligned(field.atom_offsets, alignof(std::int64_t)) &&
         multipoles.plan_token == batch.plan_token &&
         multipoles.atom_elements == field.total_atoms &&
         multipoles.dipole_elements == (multipoles_enabled ? field.total_atoms * 3 : 0) &&
         is_aligned(multipoles.atomic_charges, alignof(double)) &&
         (!multipoles_enabled || is_aligned(multipoles.atomic_dipoles, alignof(double))) &&
         (multipoles_enabled || multipoles.atomic_dipoles == nullptr) &&
         potentials.plan_token == batch.plan_token &&
         potentials.atom_elements == field.total_atoms &&
         potentials.dipole_elements == (multipoles_enabled ? field.total_atoms * 3 : 0) &&
         is_aligned(potentials.atomic, alignof(double)) &&
         (!multipoles_enabled || is_aligned(potentials.dipole, alignof(double))) &&
         (multipoles_enabled || potentials.dipole == nullptr);
}

bool validate_launch(const Gfn2SccClassicalEnergyDeviceBatch& batch,
                     const Gfn2SccClassicalEnergyDeviceInput& input,
                     const Gfn2SccClassicalEnergyDeviceActivity& activity,
                     const Gfn2SccClassicalEnergyDeviceDiagnostics& diagnostics,
                     const Gfn2SccClassicalEnergyDeviceWorkspace& workspace,
                     std::uint32_t* system_errors, std::uint32_t* device_error) noexcept {
  const bool aes2_enabled =
      component_enabled(batch.enabled_components, Gfn2SccClassicalEnergyComponent::kAES2);
  const bool d4_enabled =
      component_enabled(batch.enabled_components, Gfn2SccClassicalEnergyComponent::kD4TwoBody);
  if (batch.batch_size <= 0 || !valid_xtb_model_flavor(batch.model) ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      (batch.enabled_components & ~kGfn2SccClassicalAllComponents) != 0u ||
      (batch.model == XtbModelFlavor::kGfn1 && (aes2_enabled || d4_enabled)) ||
      batch.plan_token == 0u || input.plan_token != batch.plan_token ||
      activity.plan_token != batch.plan_token || diagnostics.plan_token != batch.plan_token ||
      workspace.plan_token != batch.plan_token ||
      !valid_input_component(
          input.es2, input.es2_elements, batch.batch_size,
          component_enabled(batch.enabled_components, Gfn2SccClassicalEnergyComponent::kES2)) ||
      !valid_input_component(
          input.es3, input.es3_elements, batch.batch_size,
          component_enabled(batch.enabled_components, Gfn2SccClassicalEnergyComponent::kES3)) ||
      !valid_input_component(
          input.aes2, input.aes2_elements, batch.batch_size,
          component_enabled(batch.enabled_components, Gfn2SccClassicalEnergyComponent::kAES2)) ||
      !valid_input_component(input.d4_two_body, input.d4_two_body_elements, batch.batch_size,
                             component_enabled(batch.enabled_components,
                                               Gfn2SccClassicalEnergyComponent::kD4TwoBody)) ||
      !valid_input_component(
          input.explicit_point_charge, input.explicit_point_charge_elements, batch.batch_size,
          component_enabled(batch.enabled_components,
                            Gfn2SccClassicalEnergyComponent::kExplicitPointCharge)) ||
      !valid_input_component(
          input.periodic_embedding, input.periodic_embedding_elements, batch.batch_size,
          component_enabled(batch.enabled_components,
                            Gfn2SccClassicalEnergyComponent::kPeriodicEmbedding)) ||
      !valid_field_binding(batch, input) ||
      !((activity.active_mask == nullptr && activity.elements == 0) ||
        (activity.elements == batch.batch_size &&
         is_aligned(activity.active_mask, alignof(std::uint8_t)))) ||
      !valid_output(diagnostics.es2, diagnostics.es2_elements, batch.batch_size) ||
      !valid_output(diagnostics.es3, diagnostics.es3_elements, batch.batch_size) ||
      !(batch.model == XtbModelFlavor::kGfn1
            ? diagnostics.aes2 == nullptr && diagnostics.aes2_elements == 0
            : valid_output(diagnostics.aes2, diagnostics.aes2_elements, batch.batch_size)) ||
      !valid_output(diagnostics.d4_two_body, diagnostics.d4_two_body_elements, batch.batch_size) ||
      !valid_output(diagnostics.explicit_point_charge, diagnostics.explicit_point_charge_elements,
                    batch.batch_size) ||
      !valid_optional_output(diagnostics.electric_field, diagnostics.electric_field_elements,
                             batch.batch_size) ||
      !valid_output(diagnostics.periodic_embedding, diagnostics.periodic_embedding_elements,
                    batch.batch_size) ||
      !valid_output(diagnostics.classical_total, diagnostics.classical_total_elements,
                    batch.batch_size) ||
      batch.batch_size >
          std::numeric_limits<std::int64_t>::max() / kGfn2SccClassicalStorageComponents ||
      workspace.component_elements <
          batch.batch_size * (batch.electric_field.plan_token != 0u
                                  ? kGfn2SccClassicalStorageComponents
                                  : kGfn2SccClassicalDiagnosticComponents) ||
      workspace.sequence_elements < 1 ||
      !is_aligned(workspace.component_scratch, alignof(double)) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return false;
  }

  const std::array<const double*, kGfn2SccClassicalInputComponents> input_pointers{
      input.es2,
      input.es3,
      input.aes2,
      input.d4_two_body,
      input.explicit_point_charge,
      input.periodic_embedding};
  const std::array<std::int64_t, kGfn2SccClassicalInputComponents> input_elements{
      input.es2_elements,
      input.es3_elements,
      input.aes2_elements,
      input.d4_two_body_elements,
      input.explicit_point_charge_elements,
      input.periodic_embedding_elements};
  std::array<AddressRange, kGfn2SccClassicalInputComponents + 6> reads{};
  for (std::size_t component = 0; component < input_pointers.size(); ++component) {
    if (!make_range(input_pointers[component], input_elements[component], sizeof(double),
                    &reads[component])) {
      return false;
    }
  }
  if (!make_range(activity.active_mask, activity.elements, sizeof(*activity.active_mask),
                  &reads[kGfn2SccClassicalInputComponents]) ||
      !make_range(batch.electric_field.atom_offsets, batch.electric_field.atom_offset_count,
                  sizeof(std::int64_t), &reads[kGfn2SccClassicalInputComponents + 1]) ||
      !make_range(input.electric_field_multipoles.atomic_charges,
                  input.electric_field_multipoles.atom_elements, sizeof(double),
                  &reads[kGfn2SccClassicalInputComponents + 2]) ||
      !make_range(input.electric_field_multipoles.atomic_dipoles,
                  input.electric_field_multipoles.dipole_elements, sizeof(double),
                  &reads[kGfn2SccClassicalInputComponents + 3]) ||
      !make_range(input.electric_field_potentials.atomic,
                  input.electric_field_potentials.atom_elements, sizeof(double),
                  &reads[kGfn2SccClassicalInputComponents + 4]) ||
      !make_range(input.electric_field_potentials.dipole,
                  input.electric_field_potentials.dipole_elements, sizeof(double),
                  &reads[kGfn2SccClassicalInputComponents + 5])) {
    return false;
  }

  const std::array<double*, kGfn2SccClassicalStorageComponents> output_pointers{
      diagnostics.es2,
      diagnostics.es3,
      diagnostics.aes2,
      diagnostics.d4_two_body,
      diagnostics.explicit_point_charge,
      diagnostics.periodic_embedding,
      diagnostics.classical_total,
      diagnostics.electric_field};
  const std::array<std::int64_t, kGfn2SccClassicalStorageComponents> output_elements{
      diagnostics.es2_elements,
      diagnostics.es3_elements,
      diagnostics.aes2_elements,
      diagnostics.d4_two_body_elements,
      diagnostics.explicit_point_charge_elements,
      diagnostics.periodic_embedding_elements,
      diagnostics.classical_total_elements,
      diagnostics.electric_field_elements};
  std::array<AddressRange, kGfn2SccClassicalStorageComponents + 4> writes{};
  for (std::size_t component = 0; component < output_pointers.size(); ++component) {
    if (!make_range(output_pointers[component], output_elements[component], sizeof(double),
                    &writes[component])) {
      return false;
    }
  }
  if (!make_range(workspace.component_scratch, workspace.component_elements,
                  sizeof(*workspace.component_scratch),
                  &writes[kGfn2SccClassicalStorageComponents]) ||
      !make_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active),
                  &writes[kGfn2SccClassicalStorageComponents + 1]) ||
      !make_range(system_errors, batch.batch_size, sizeof(*system_errors),
                  &writes[kGfn2SccClassicalStorageComponents + 2]) ||
      !make_range(device_error, 1, sizeof(*device_error),
                  &writes[kGfn2SccClassicalStorageComponents + 3])) {
    return false;
  }
  for (std::size_t lhs = 0; lhs < writes.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < writes.size(); ++rhs) {
      if (ranges_overlap(writes[lhs], writes[rhs])) {
        return false;
      }
    }
    for (const AddressRange& read : reads) {
      if (ranges_overlap(writes[lhs], read)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

cudaError_t reset_gfn2_scc_classical_energy_device_errors_cuda(std::int64_t batch_size,
                                                               std::uint32_t* system_errors,
                                                               std::uint32_t* device_error,
                                                               cudaStream_t stream) noexcept {
  if (batch_size <= 0 ||
      static_cast<std::uint64_t>(batch_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                     sizeof(*system_errors)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  AddressRange system_range;
  AddressRange device_range;
  if (!make_range(system_errors, batch_size, sizeof(*system_errors), &system_range) ||
      !make_range(device_error, 1, sizeof(*device_error), &device_range) ||
      ranges_overlap(system_range, device_range)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t evaluate_gfn2_scc_classical_energy_cuda(
    const Gfn2SccClassicalEnergyDeviceBatch& batch, const Gfn2SccClassicalEnergyDeviceInput& input,
    const Gfn2SccClassicalEnergyDeviceActivity& activity,
    const Gfn2SccClassicalEnergyDeviceDiagnostics& diagnostics,
    const Gfn2SccClassicalEnergyDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!validate_launch(batch, input, activity, diagnostics, workspace, system_errors,
                       device_error)) {
    return cudaErrorInvalidValue;
  }

  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks =
      static_cast<unsigned int>((batch.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  aggregate_classical_energy_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, input, activity, workspace, system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_classical_energy_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch, activity, diagnostics, workspace, system_errors);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
