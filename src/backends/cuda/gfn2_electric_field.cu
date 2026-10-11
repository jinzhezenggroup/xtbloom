#include <cuda_runtime.h>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_electric_field.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

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
    if (pointer != nullptr) return false;
    *range = {};
    return true;
  }
  if (pointer == nullptr) return false;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
  *range = {begin, begin + bytes};
  return true;
}

bool overlaps(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin != first.end && second.begin != second.end && first.begin < second.end &&
         second.begin < first.end;
}

bool validate_launch(const Gfn2ElectricFieldDeviceBatch& batch,
                     const Gfn2ElectricFieldDeviceInput& input,
                     const Gfn2ElectricFieldDevicePotentials& potentials,
                     std::uint32_t* system_errors, std::uint32_t* plan_error) noexcept {
  if (batch.batch_size <= 0 ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.total_atoms <= 0 || batch.total_atoms > std::numeric_limits<std::int64_t>::max() / 3 ||
      batch.atom_offset_count != batch.batch_size + 1 || batch.plan_token == 0u ||
      input.plan_token != batch.plan_token || potentials.plan_token != batch.plan_token ||
      input.vector_elements != batch.batch_size * 3 ||
      input.position_elements != batch.total_atoms * 3 ||
      potentials.atom_elements != batch.total_atoms ||
      potentials.dipole_elements != batch.total_atoms * 3 ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(input.vectors, alignof(double)) ||
      !is_aligned(input.positions, alignof(double)) ||
      !is_aligned(potentials.atomic, alignof(double)) ||
      !is_aligned(potentials.dipole, alignof(double)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(plan_error, alignof(std::uint32_t))) {
    return false;
  }

  std::array<AddressRange, 3> reads{};
  std::array<AddressRange, 4> writes{};
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(std::int64_t), &reads[0]) ||
      !make_range(input.vectors, input.vector_elements, sizeof(double), &reads[1]) ||
      !make_range(input.positions, input.position_elements, sizeof(double), &reads[2]) ||
      !make_range(potentials.atomic, potentials.atom_elements, sizeof(double), &writes[0]) ||
      !make_range(potentials.dipole, potentials.dipole_elements, sizeof(double), &writes[1]) ||
      !make_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writes[2]) ||
      !make_range(plan_error, 1, sizeof(std::uint32_t), &writes[3])) {
    return false;
  }
  for (std::size_t lhs = 0; lhs < writes.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < writes.size(); ++rhs) {
      if (overlaps(writes[lhs], writes[rhs])) return false;
    }
    for (const AddressRange& read : reads) {
      if (overlaps(writes[lhs], read)) return false;
    }
  }
  return true;
}

__device__ void record_system_error(std::uint32_t* errors, std::int64_t system,
                                    Gfn2ElectricFieldDeviceError error) {
  atomicCAS(errors + system, 0u, static_cast<std::uint32_t>(error));
}

__global__ void preflight_offsets_kernel(Gfn2ElectricFieldDeviceBatch batch,
                                         std::uint32_t* plan_error) {
  if (threadIdx.x == 0 &&
      (batch.atom_offsets[0] != 0 || batch.atom_offsets[batch.batch_size] != batch.total_atoms)) {
    atomicCAS(plan_error, 0u,
              static_cast<std::uint32_t>(Gfn2ElectricFieldDeviceError::kInvalidOffsets));
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t begin = batch.atom_offsets[system];
    const std::int64_t end = batch.atom_offsets[system + 1];
    if (begin < 0 || begin >= end || end > batch.total_atoms) {
      atomicCAS(plan_error, 0u,
                static_cast<std::uint32_t>(Gfn2ElectricFieldDeviceError::kInvalidOffsets));
    }
  }
}

__global__ void preflight_values_kernel(Gfn2ElectricFieldDeviceBatch batch,
                                        Gfn2ElectricFieldDeviceInput input,
                                        std::uint32_t* system_errors,
                                        const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) != 0u) return;
  __shared__ double field[3];
  __shared__ int finite_field;
  if (threadIdx.x == 0) finite_field = 1;
  __syncthreads();
  if (threadIdx.x < 3) {
    field[threadIdx.x] = input.vectors[system * 3 + threadIdx.x];
    if (!isfinite(field[threadIdx.x])) {
      atomicExch(&finite_field, 0);
      record_system_error(system_errors, system, Gfn2ElectricFieldDeviceError::kNonfiniteVector);
    }
  }
  __syncthreads();
  /* Preserve one deterministic classification for malformed field vectors.
   * Arithmetic derived from an already-invalid vector must not race the
   * primary kNonfiniteVector diagnostic. */
  if (finite_field == 0) return;
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    const double x = input.positions[atom * 3];
    const double y = input.positions[atom * 3 + 1];
    const double z = input.positions[atom * 3 + 2];
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
      record_system_error(system_errors, system, Gfn2ElectricFieldDeviceError::kNonfinitePosition);
      continue;
    }
    const double potential = -fma(field[2], z, fma(field[1], y, field[0] * x));
    if (!isfinite(potential)) {
      record_system_error(system_errors, system,
                          Gfn2ElectricFieldDeviceError::kNonfinitePotentialArithmetic);
    }
  }
}

__global__ void publish_potentials_kernel(Gfn2ElectricFieldDeviceBatch batch,
                                          Gfn2ElectricFieldDeviceInput input,
                                          Gfn2ElectricFieldDevicePotentials potentials,
                                          const std::uint32_t* system_errors,
                                          const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) != 0u ||
      atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) != 0u) {
    return;
  }
  __shared__ double field[3];
  if (threadIdx.x < 3) field[threadIdx.x] = input.vectors[system * 3 + threadIdx.x];
  __syncthreads();
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    potentials.atomic[atom] =
        -fma(field[2], input.positions[atom * 3 + 2],
             fma(field[1], input.positions[atom * 3 + 1], field[0] * input.positions[atom * 3]));
    potentials.dipole[atom * 3] = -field[0];
    potentials.dipole[atom * 3 + 1] = -field[1];
    potentials.dipole[atom * 3 + 2] = -field[2];
  }
}

}  // namespace

cudaError_t reset_gfn2_electric_field_device_errors_cuda(std::int64_t batch_size,
                                                         std::uint32_t* system_errors,
                                                         std::uint32_t* plan_error,
                                                         cudaStream_t stream) noexcept {
  if (batch_size <= 0 || !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(plan_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  AddressRange systems{};
  AddressRange plan{};
  if (!make_range(system_errors, batch_size, sizeof(std::uint32_t), &systems) ||
      !make_range(plan_error, 1, sizeof(std::uint32_t), &plan) || overlaps(systems, plan)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(std::uint32_t), stream);
  return status == cudaSuccess ? cudaMemsetAsync(plan_error, 0, sizeof(std::uint32_t), stream)
                               : status;
}

cudaError_t refresh_gfn2_electric_field_potentials_cuda(
    const Gfn2ElectricFieldDeviceBatch& batch, const Gfn2ElectricFieldDeviceInput& input,
    const Gfn2ElectricFieldDevicePotentials& potentials, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream) noexcept {
  if (!validate_launch(batch, input, potentials, system_errors, plan_error)) {
    return cudaErrorInvalidValue;
  }
  preflight_offsets_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, plan_error);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  preflight_values_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                            stream>>>(batch, input, system_errors, plan_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  publish_potentials_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, input, potentials, system_errors, plan_error);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
