#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_es3.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

__device__ void record_error(std::uint32_t* device_error, Gfn2ES3DeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2ES3DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

/*
 * CUDA 12.9 resolves the C++ isnormal overload as a host-only constexpr
 * function in device code. Keep this tiny device-native predicate local so
 * the fallback decision is identical without relying on relaxed constexpr.
 * Callers handle zero before reaching this helper.
 */
__device__ bool is_normal_double(double value) {
  constexpr double kMinNormalDouble = 2.2250738585072014e-308;
  return isfinite(value) && fabs(value) >= kMinNormalDouble;
}

/*
 * Keep the ordinary operation order identical to the CPU path. Retry through
 * a mantissa/exponent decomposition when an intermediate overflowed,
 * underflowed, or lost range even though the final binary64 result may still
 * be representable. This is the device equivalent of the CPU path's
 * wider-intermediate fallback and also handles finite caller-supplied Gamma3
 * values outside the generated GFN2 range.
 */
__device__ bool shell_potential(double gamma3, double charge, double* result) {
  if (charge == 0.0 || gamma3 == 0.0) {
    *result = 0.0;
    return true;
  }
  const double square = charge * charge;
  double value = square * gamma3;
  if (is_normal_double(square) && is_normal_double(gamma3) && is_normal_double(value)) {
    *result = value;
    return true;
  }
  int charge_exponent = 0;
  int gamma_exponent = 0;
  const double charge_mantissa = frexp(charge, &charge_exponent);
  const double gamma_mantissa = frexp(gamma3, &gamma_exponent);
  const double mantissa = charge_mantissa * charge_mantissa * gamma_mantissa;
  value = scalbn(mantissa, 2 * charge_exponent + gamma_exponent);
  if (!isfinite(value)) {
    return false;
  }
  *result = value;
  return true;
}

/* Match the CPU expression first, then recover representable q^3 cases. */
__device__ bool shell_energy(double gamma3, double charge, double* result) {
  if (charge == 0.0 || gamma3 == 0.0) {
    *result = 0.0;
    return true;
  }
  const double square = charge * charge;
  const double cube = square * charge;
  const double scaled = cube * gamma3;
  double value = scaled / 3.0;
  if (is_normal_double(square) && is_normal_double(cube) && is_normal_double(gamma3) &&
      is_normal_double(scaled) && is_normal_double(value)) {
    *result = value;
    return true;
  }
  int charge_exponent = 0;
  int gamma_exponent = 0;
  const double charge_mantissa = frexp(charge, &charge_exponent);
  const double gamma_mantissa = frexp(gamma3, &gamma_exponent);
  const double mantissa =
      charge_mantissa * charge_mantissa * charge_mantissa * gamma_mantissa / 3.0;
  value = scalbn(mantissa, 3 * charge_exponent + gamma_exponent);
  if (!isfinite(value)) {
    return false;
  }
  *result = value;
  return true;
}

struct ShellRange {
  std::int64_t begin;
  std::int64_t end;
};

/* Validate the ragged partition before any pointer indexed by its values. */
__device__ void load_and_validate_range(const Gfn2ES3DeviceBatch& batch, std::int64_t system,
                                        ShellRange* range, int* valid,
                                        std::uint32_t* device_error) {
  if (threadIdx.x == 0) {
    /* Preserve an upstream failure and avoid dereferencing dependent inputs. */
    if (atomicAdd(device_error, 0u) != static_cast<std::uint32_t>(Gfn2ES3DeviceError::kSuccess)) {
      *valid = 0;
    } else {
      range->begin = batch.batch_shell_offsets[system];
      range->end = batch.batch_shell_offsets[system + 1];
      *valid = range->begin >= 0 && range->begin <= range->end &&
               range->end <= batch.total_shells && (system != 0 || range->begin == 0) &&
               (system + 1 != batch.batch_size || range->end == batch.total_shells);
      if (*valid == 0) {
        record_error(device_error, Gfn2ES3DeviceError::kInvalidOffsets);
      }
    }
  }
  __syncthreads();
}

/* Validate immutable parameters and SCC charges as one system-local phase. */
__device__ void validate_shell_inputs(const Gfn2ES3DeviceBatch& batch, const ShellRange& range,
                                      const double* shell_charges, int* valid,
                                      std::uint32_t* device_error) {
  for (std::int64_t shell = range.begin + threadIdx.x; shell < range.end; shell += blockDim.x) {
    if (!isfinite(batch.shell_gamma3[shell])) {
      record_error(device_error, Gfn2ES3DeviceError::kNonfiniteGamma3);
      atomicExch(valid, 0);
    } else if (!isfinite(shell_charges[shell])) {
      record_error(device_error, Gfn2ES3DeviceError::kNonfiniteShellCharge);
      atomicExch(valid, 0);
    }
  }
  __syncthreads();
}

__global__ void es3_potential_kernel(Gfn2ES3DeviceBatch batch, const double* shell_charges,
                                     double* shell_potentials, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ ShellRange range;
  __shared__ int valid;
  load_and_validate_range(batch, system, &range, &valid, device_error);
  if (valid == 0) {
    return;
  }
  validate_shell_inputs(batch, range, shell_charges, &valid, device_error);
  if (valid == 0) {
    return;
  }

  /* Preflight all outputs before publishing any result for this system. */
  for (std::int64_t shell = range.begin + threadIdx.x; shell < range.end; shell += blockDim.x) {
    double potential = 0.0;
    if (!shell_potential(batch.shell_gamma3[shell], shell_charges[shell], &potential)) {
      record_error(device_error, Gfn2ES3DeviceError::kNonfinitePotentialArithmetic);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t shell = range.begin + threadIdx.x; shell < range.end; shell += blockDim.x) {
    double potential = 0.0;
    (void)shell_potential(batch.shell_gamma3[shell], shell_charges[shell], &potential);
    shell_potentials[shell] = potential;
  }
}

__global__ void es3_energy_kernel(Gfn2ES3DeviceBatch batch, const double* shell_charges,
                                  double* energies, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ ShellRange range;
  __shared__ int valid;
  load_and_validate_range(batch, system, &range, &valid, device_error);
  if (valid == 0) {
    return;
  }
  validate_shell_inputs(batch, range, shell_charges, &valid, device_error);
  if (valid == 0) {
    return;
  }

  /*
   * One thread follows CPU shell order exactly. Typical xTB systems have only
   * a few shells per atom, while batch members still execute concurrently.
   * This avoids reduction-order drift in SCC energies and makes the checked
   * accumulation transactional for each system.
   */
  if (threadIdx.x == 0) {
    double energy = energies[system];
    if (!isfinite(energy)) {
      record_error(device_error, Gfn2ES3DeviceError::kNonfiniteEnergySeed);
      return;
    }
    for (std::int64_t shell = range.begin; shell < range.end; ++shell) {
      double contribution = 0.0;
      if (!shell_energy(batch.shell_gamma3[shell], shell_charges[shell], &contribution)) {
        record_error(device_error, Gfn2ES3DeviceError::kNonfiniteEnergyArithmetic);
        return;
      }
      const double updated = energy + contribution;
      if (!isfinite(updated)) {
        record_error(device_error, Gfn2ES3DeviceError::kNonfiniteEnergyArithmetic);
        return;
      }
      energy = updated;
    }
    energies[system] = energy;
  }
}

__device__ void record_es3_scc_plan_error(std::uint32_t* plan_error, Gfn2ES3DeviceError error) {
  atomicCAS(plan_error, static_cast<std::uint32_t>(Gfn2ES3DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_es3_scc_system_error(std::uint32_t* system_errors, std::int64_t system,
                                            Gfn2ES3DeviceError error) {
  atomicCAS(system_errors + system, static_cast<std::uint32_t>(Gfn2ES3DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool es3_scc_member_is_active(const Gfn2SccIterationDeviceActivity& activity,
                                         const std::uint32_t* system_errors,
                                         const std::uint32_t* plan_error, std::int64_t system) {
  if (atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) != 1u ||
      activity.active_mask[system] != 1u) {
    return false;
  }
  return atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) == 0u &&
         atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) == 0u;
}

__global__ void es3_scc_plan_preflight_kernel(Gfn2ES3DeviceBatch batch,
                                              Gfn2SccIterationDeviceActivity activity,
                                              std::uint32_t* plan_error) {
  __shared__ int sequence_active;
  __shared__ int any_active;
  if (threadIdx.x == 0) {
    sequence_active = atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) == 1u;
    any_active = 0;
  }
  __syncthreads();
  if (sequence_active == 0) {
    return;
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] == 1u) {
      atomicExch(&any_active, 1);
    }
  }
  __syncthreads();
  if (any_active == 0 || atomicAdd(plan_error, 0u) != 0u) {
    return;
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] != 1u) {
      continue;
    }
    const std::int64_t begin = batch.batch_shell_offsets[system];
    const std::int64_t end = batch.batch_shell_offsets[system + 1];
    const bool endpoints_valid = begin >= 0 && begin <= end && end <= batch.total_shells;
    const bool boundary_valid = (system != 0 || begin == 0) &&
                                (system + 1 != batch.batch_size || end == batch.total_shells);
    if (!endpoints_valid || !boundary_valid) {
      record_es3_scc_plan_error(plan_error, Gfn2ES3DeviceError::kInvalidOffsets);
    }
  }
}

__global__ void es3_scc_potential_kernel(Gfn2ES3DeviceBatch batch,
                                         Gfn2SccIterationDeviceActivity activity,
                                         const double* shell_charges, double* shell_potentials,
                                         std::uint32_t* system_errors,
                                         const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    active = es3_scc_member_is_active(activity, system_errors, plan_error, system) ? 1 : 0;
    valid = 1;
  }
  __syncthreads();
  if (active == 0) {
    return;
  }
  const std::int64_t begin = batch.batch_shell_offsets[system];
  const std::int64_t end = batch.batch_shell_offsets[system + 1];
  for (std::int64_t shell = begin + threadIdx.x; shell < end; shell += blockDim.x) {
    const double gamma3 = batch.shell_gamma3[shell];
    const double charge = shell_charges[shell];
    double value = 0.0;
    if (!isfinite(gamma3)) {
      record_es3_scc_system_error(system_errors, system, Gfn2ES3DeviceError::kNonfiniteGamma3);
      atomicExch(&valid, 0);
    } else if (!isfinite(charge)) {
      record_es3_scc_system_error(system_errors, system, Gfn2ES3DeviceError::kNonfiniteShellCharge);
      atomicExch(&valid, 0);
    } else if (!shell_potential(gamma3, charge, &value)) {
      record_es3_scc_system_error(system_errors, system,
                                  Gfn2ES3DeviceError::kNonfinitePotentialArithmetic);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }
  for (std::int64_t shell = begin + threadIdx.x; shell < end; shell += blockDim.x) {
    double value = 0.0;
    (void)shell_potential(batch.shell_gamma3[shell], shell_charges[shell], &value);
    shell_potentials[shell] = value;
  }
}

__global__ void es3_scc_energy_kernel(Gfn2ES3DeviceBatch batch,
                                      Gfn2SccIterationDeviceActivity activity,
                                      const double* shell_charges, double* component_energies,
                                      std::uint32_t* system_errors,
                                      const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || !es3_scc_member_is_active(activity, system_errors, plan_error, system)) {
    return;
  }
  const std::int64_t begin = batch.batch_shell_offsets[system];
  const std::int64_t end = batch.batch_shell_offsets[system + 1];
  double energy = 0.0;
  for (std::int64_t shell = begin; shell < end; ++shell) {
    const double gamma3 = batch.shell_gamma3[shell];
    const double charge = shell_charges[shell];
    double contribution = 0.0;
    if (!isfinite(gamma3)) {
      record_es3_scc_system_error(system_errors, system, Gfn2ES3DeviceError::kNonfiniteGamma3);
      return;
    }
    if (!isfinite(charge)) {
      record_es3_scc_system_error(system_errors, system, Gfn2ES3DeviceError::kNonfiniteShellCharge);
      return;
    }
    if (!shell_energy(gamma3, charge, &contribution)) {
      record_es3_scc_system_error(system_errors, system,
                                  Gfn2ES3DeviceError::kNonfiniteEnergyArithmetic);
      return;
    }
    const double updated = energy + contribution;
    if (!isfinite(updated)) {
      record_es3_scc_system_error(system_errors, system,
                                  Gfn2ES3DeviceError::kNonfiniteEnergyArithmetic);
      return;
    }
    energy = updated;
  }
  component_energies[system] = energy;
}

bool count_bytes(std::int64_t count, std::size_t element_size, std::size_t* bytes) noexcept {
  if (count < 0 ||
      static_cast<std::uint64_t>(count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      static_cast<std::uint64_t>(count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  *bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                    std::size_t second_bytes) noexcept {
  if (first_bytes == 0u || second_bytes == 0u) {
    return false;
  }
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  const auto first_end = first_begin + first_bytes;
  const auto second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

cudaError_t validate_common_launcher_arguments(const Gfn2ES3DeviceBatch& batch,
                                               std::uint32_t* device_error) noexcept {
  if (batch.batch_size <= 0 || batch.total_shells <= 0 ||
      batch.batch_size == std::numeric_limits<std::int64_t>::max() ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.shell_gamma3_count != batch.total_shells || batch.batch_shell_offsets == nullptr ||
      batch.shell_gamma3 == nullptr || device_error == nullptr ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_gamma3, alignof(double)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  if (static_cast<std::uint64_t>(batch.batch_size) >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return cudaErrorInvalidConfiguration;
  }

  std::size_t offset_bytes = 0;
  std::size_t shell_bytes = 0;
  if (!count_bytes(batch.batch_shell_offset_count, sizeof(std::int64_t), &offset_bytes) ||
      !count_bytes(batch.total_shells, sizeof(double), &shell_bytes) ||
      ranges_overlap(device_error, sizeof(*device_error), batch.batch_shell_offsets,
                     offset_bytes) ||
      ranges_overlap(device_error, sizeof(*device_error), batch.shell_gamma3, shell_bytes)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t validate_scc_launcher_arguments(const Gfn2ES3DeviceBatch& batch,
                                            const Gfn2SccIterationDeviceActivity& activity,
                                            std::uint32_t* system_errors,
                                            std::uint32_t* plan_error) noexcept {
  cudaError_t status = validate_common_launcher_arguments(batch, plan_error);
  if (status != cudaSuccess || batch.plan_token == 0u || activity.active_mask == nullptr ||
      activity.sequence_active == nullptr || activity.batch_elements != batch.batch_size ||
      activity.sequence_elements != 1 || activity.plan_token != batch.plan_token ||
      system_errors == nullptr || !is_aligned(activity.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  std::size_t offset_bytes = 0;
  std::size_t shell_bytes = 0;
  std::size_t system_error_bytes = 0;
  if (!count_bytes(batch.batch_shell_offset_count, sizeof(std::int64_t), &offset_bytes) ||
      !count_bytes(batch.total_shells, sizeof(double), &shell_bytes) ||
      !count_bytes(batch.batch_size, sizeof(std::uint32_t), &system_error_bytes) ||
      ranges_overlap(system_errors, system_error_bytes, plan_error, sizeof(*plan_error)) ||
      ranges_overlap(activity.active_mask, static_cast<std::size_t>(batch.batch_size),
                     system_errors, system_error_bytes) ||
      ranges_overlap(activity.active_mask, static_cast<std::size_t>(batch.batch_size), plan_error,
                     sizeof(*plan_error)) ||
      ranges_overlap(activity.sequence_active, sizeof(*activity.sequence_active), system_errors,
                     system_error_bytes) ||
      ranges_overlap(activity.sequence_active, sizeof(*activity.sequence_active), plan_error,
                     sizeof(*plan_error)) ||
      ranges_overlap(activity.active_mask, static_cast<std::size_t>(batch.batch_size),
                     activity.sequence_active, sizeof(*activity.sequence_active)) ||
      ranges_overlap(activity.active_mask, static_cast<std::size_t>(batch.batch_size),
                     batch.batch_shell_offsets, offset_bytes) ||
      ranges_overlap(activity.active_mask, static_cast<std::size_t>(batch.batch_size),
                     batch.shell_gamma3, shell_bytes) ||
      ranges_overlap(activity.sequence_active, sizeof(*activity.sequence_active),
                     batch.batch_shell_offsets, offset_bytes) ||
      ranges_overlap(activity.sequence_active, sizeof(*activity.sequence_active),
                     batch.shell_gamma3, shell_bytes) ||
      ranges_overlap(system_errors, system_error_bytes, batch.batch_shell_offsets, offset_bytes) ||
      ranges_overlap(system_errors, system_error_bytes, batch.shell_gamma3, shell_bytes)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t reset_gfn2_es3_device_error_cuda(std::uint32_t* device_error,
                                             cudaStream_t stream) noexcept {
  if (!is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t reset_gfn2_es3_scc_errors_cuda(std::int64_t batch_size, std::uint32_t* system_errors,
                                           std::uint32_t* plan_error,
                                           cudaStream_t stream) noexcept {
  std::size_t system_error_bytes = 0;
  if (batch_size <= 0 || system_errors == nullptr || plan_error == nullptr ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(plan_error, alignof(std::uint32_t)) ||
      !count_bytes(batch_size, sizeof(std::uint32_t), &system_error_bytes) ||
      ranges_overlap(system_errors, system_error_bytes, plan_error, sizeof(*plan_error))) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(system_errors, 0, system_error_bytes, stream);
  return status == cudaSuccess ? cudaMemsetAsync(plan_error, 0, sizeof(*plan_error), stream)
                               : status;
}

cudaError_t evaluate_gfn2_es3_potential_cuda(const Gfn2ES3DeviceBatch& batch,
                                             const double* shell_charges, double* shell_potentials,
                                             std::uint32_t* device_error,
                                             cudaStream_t stream) noexcept {
  cudaError_t status = validate_common_launcher_arguments(batch, device_error);
  std::size_t offset_bytes = 0;
  std::size_t shell_bytes = 0;
  if (status != cudaSuccess || shell_charges == nullptr || shell_potentials == nullptr ||
      !count_bytes(batch.batch_shell_offset_count, sizeof(std::int64_t), &offset_bytes) ||
      !count_bytes(batch.total_shells, sizeof(double), &shell_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, shell_charges, shell_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, batch.shell_gamma3, shell_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, batch.batch_shell_offsets, offset_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, batch.batch_shell_offsets, offset_bytes) ||
      ranges_overlap(device_error, sizeof(*device_error), shell_charges, shell_bytes) ||
      ranges_overlap(device_error, sizeof(*device_error), shell_potentials, shell_bytes)) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }

  es3_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                         stream>>>(batch, shell_charges, shell_potentials, device_error);
  return cudaGetLastError();
}

cudaError_t add_gfn2_es3_energy_cuda(const Gfn2ES3DeviceBatch& batch, const double* shell_charges,
                                     double* energies, std::uint32_t* device_error,
                                     cudaStream_t stream) noexcept {
  cudaError_t status = validate_common_launcher_arguments(batch, device_error);
  std::size_t offset_bytes = 0;
  std::size_t shell_bytes = 0;
  std::size_t energy_bytes = 0;
  if (status != cudaSuccess || shell_charges == nullptr || energies == nullptr ||
      !count_bytes(batch.batch_shell_offset_count, sizeof(std::int64_t), &offset_bytes) ||
      !count_bytes(batch.total_shells, sizeof(double), &shell_bytes) ||
      !count_bytes(batch.batch_size, sizeof(double), &energy_bytes) ||
      ranges_overlap(energies, energy_bytes, shell_charges, shell_bytes) ||
      ranges_overlap(energies, energy_bytes, batch.shell_gamma3, shell_bytes) ||
      ranges_overlap(energies, energy_bytes, batch.batch_shell_offsets, offset_bytes) ||
      ranges_overlap(device_error, sizeof(*device_error), shell_charges, shell_bytes) ||
      ranges_overlap(device_error, sizeof(*device_error), energies, energy_bytes)) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }

  es3_energy_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, shell_charges, energies, device_error);
  return cudaGetLastError();
}

cudaError_t evaluate_gfn2_es3_scc_potential_cuda(
    const Gfn2ES3DeviceBatch& batch, const Gfn2SccIterationDeviceActivity& activity,
    const double* mixed_shell_charges, double* shell_potentials, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream) noexcept {
  cudaError_t status = validate_scc_launcher_arguments(batch, activity, system_errors, plan_error);
  std::size_t shell_bytes = 0;
  std::size_t system_error_bytes = 0;
  std::size_t offset_bytes = 0;
  if (status != cudaSuccess || mixed_shell_charges == nullptr || shell_potentials == nullptr ||
      !is_aligned(mixed_shell_charges, alignof(double)) ||
      !is_aligned(shell_potentials, alignof(double)) ||
      !count_bytes(batch.total_shells, sizeof(double), &shell_bytes) ||
      !count_bytes(batch.batch_size, sizeof(std::uint32_t), &system_error_bytes) ||
      !count_bytes(batch.batch_shell_offset_count, sizeof(std::int64_t), &offset_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, mixed_shell_charges, shell_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, batch.shell_gamma3, shell_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, batch.batch_shell_offsets, offset_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, activity.active_mask,
                     static_cast<std::size_t>(batch.batch_size)) ||
      ranges_overlap(shell_potentials, shell_bytes, activity.sequence_active,
                     sizeof(*activity.sequence_active)) ||
      ranges_overlap(shell_potentials, shell_bytes, system_errors, system_error_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, plan_error, sizeof(*plan_error)) ||
      ranges_overlap(mixed_shell_charges, shell_bytes, system_errors, system_error_bytes) ||
      ranges_overlap(mixed_shell_charges, shell_bytes, plan_error, sizeof(*plan_error)) ||
      ranges_overlap(mixed_shell_charges, shell_bytes, activity.active_mask,
                     static_cast<std::size_t>(batch.batch_size)) ||
      ranges_overlap(mixed_shell_charges, shell_bytes, activity.sequence_active,
                     sizeof(*activity.sequence_active)) ||
      ranges_overlap(mixed_shell_charges, shell_bytes, batch.batch_shell_offsets, offset_bytes)) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  es3_scc_plan_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, activity, plan_error);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  es3_scc_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                             stream>>>(batch, activity, mixed_shell_charges, shell_potentials,
                                       system_errors, plan_error);
  return cudaPeekAtLastError();
}

cudaError_t evaluate_gfn2_es3_scc_energy_cuda(
    const Gfn2ES3DeviceBatch& batch, const Gfn2SccIterationDeviceActivity& activity,
    const double* raw_shell_charges, double* component_energies, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream) noexcept {
  cudaError_t status = validate_scc_launcher_arguments(batch, activity, system_errors, plan_error);
  std::size_t shell_bytes = 0;
  std::size_t energy_bytes = 0;
  std::size_t system_error_bytes = 0;
  std::size_t offset_bytes = 0;
  if (status != cudaSuccess || raw_shell_charges == nullptr || component_energies == nullptr ||
      !is_aligned(raw_shell_charges, alignof(double)) ||
      !is_aligned(component_energies, alignof(double)) ||
      !count_bytes(batch.total_shells, sizeof(double), &shell_bytes) ||
      !count_bytes(batch.batch_size, sizeof(double), &energy_bytes) ||
      !count_bytes(batch.batch_size, sizeof(std::uint32_t), &system_error_bytes) ||
      !count_bytes(batch.batch_shell_offset_count, sizeof(std::int64_t), &offset_bytes) ||
      ranges_overlap(component_energies, energy_bytes, raw_shell_charges, shell_bytes) ||
      ranges_overlap(component_energies, energy_bytes, batch.shell_gamma3, shell_bytes) ||
      ranges_overlap(component_energies, energy_bytes, batch.batch_shell_offsets, offset_bytes) ||
      ranges_overlap(component_energies, energy_bytes, activity.active_mask,
                     static_cast<std::size_t>(batch.batch_size)) ||
      ranges_overlap(component_energies, energy_bytes, activity.sequence_active,
                     sizeof(*activity.sequence_active)) ||
      ranges_overlap(component_energies, energy_bytes, system_errors, system_error_bytes) ||
      ranges_overlap(component_energies, energy_bytes, plan_error, sizeof(*plan_error)) ||
      ranges_overlap(raw_shell_charges, shell_bytes, system_errors, system_error_bytes) ||
      ranges_overlap(raw_shell_charges, shell_bytes, plan_error, sizeof(*plan_error)) ||
      ranges_overlap(raw_shell_charges, shell_bytes, activity.active_mask,
                     static_cast<std::size_t>(batch.batch_size)) ||
      ranges_overlap(raw_shell_charges, shell_bytes, activity.sequence_active,
                     sizeof(*activity.sequence_active)) ||
      ranges_overlap(raw_shell_charges, shell_bytes, batch.batch_shell_offsets, offset_bytes)) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  es3_scc_plan_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, activity, plan_error);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  es3_scc_energy_kernel<<<static_cast<unsigned int>(batch.batch_size), 1, 0, stream>>>(
      batch, activity, raw_shell_charges, component_energies, system_errors, plan_error);
  return cudaPeekAtLastError();
}

}  // namespace gpuxtb::detail::cuda
