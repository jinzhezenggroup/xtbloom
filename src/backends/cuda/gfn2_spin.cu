#include <array>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_spin.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 64;
constexpr std::int64_t kMaximumAtomShells = 3;
constexpr std::int64_t kInt64Maximum = 9223372036854775807LL;

struct SystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t shell_begin;
  std::int64_t shell_end;
  std::int64_t population_begin;
  std::int64_t population_end;
  std::int32_t spin_channels;
};

__device__ std::uint32_t load_u32(const std::uint32_t* value) {
  return atomicAdd(const_cast<std::uint32_t*>(value), 0u);
}

__device__ bool stage_is_open(const Gfn2SpinDeviceWorkspace& workspace) {
  return load_u32(workspace.sequence_active) == 1u;
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return load_u32(system_errors + system) ==
         static_cast<std::uint32_t>(Gfn2SpinDeviceError::kSuccess);
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2SpinDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system, static_cast<std::uint32_t>(Gfn2SpinDeviceError::kSuccess),
                code) == static_cast<std::uint32_t>(Gfn2SpinDeviceError::kSuccess)) {
    atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2SpinDeviceError::kSuccess), code);
  }
}

__device__ bool valid_nonempty_range(std::int64_t begin, std::int64_t end, std::int64_t total) {
  return begin >= 0 && begin < end && end <= total;
}

/*
 * Snapshot both upstream gates before a peer-local error makes device_error
 * sticky.  Peer failures recorded later must not suppress healthy publication
 * in the same stage, while a pre-existing plan failure closes the whole call.
 */
__global__ void capture_stage_kernel(Gfn2SccIterationDeviceActivity activity,
                                     const std::uint32_t* device_error,
                                     Gfn2SpinDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        load_u32(activity.sequence_active) == 1u &&
                load_u32(device_error) == static_cast<std::uint32_t>(Gfn2SpinDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

/*
 * Validate only canonical active slices.  This ordering deliberately permits
 * inactive peers to retain poisoned offsets and matrices without observation.
 */
__global__ void topology_preflight_kernel(Gfn2SpinDeviceBatch batch,
                                          Gfn2SccIterationDeviceActivity activity,
                                          Gfn2SpinDeviceWorkspace workspace,
                                          std::uint32_t* system_errors,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!stage_is_open(workspace) || !system_is_valid(system_errors, system)) {
    return;
  }

  const std::uint8_t active = activity.active_mask[system];
  if (active == 0u) {
    return;
  }
  if (active != 1u) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SpinDeviceError::kInvalidActiveMask);
    }
    return;
  }

  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    ranges.population_begin = batch.shell_population_offsets[system];
    ranges.population_end = batch.shell_population_offsets[system + 1];
    ranges.spin_channels = batch.spin_channels[system];

    valid = valid_nonempty_range(ranges.population_begin, ranges.population_end,
                                 batch.shell_population_elements)
                ? 1
                : 0;
    if (ranges.spin_channels != 1 && ranges.spin_channels != 2) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SpinDeviceError::kInvalidSpinChannels);
      valid = 0;
    }
    /*
     * A restricted member has no spin term.  Its exact-zero result depends
     * only on the output slice, so atom/shell/coupling topology remains
     * completely unread.  Setup owns validation of that immutable metadata;
     * this device stage preflights it only for unrestricted consumers.
     */
    if (valid != 0 && ranges.spin_channels == 2) {
      ranges.atom_begin = batch.atom_offsets[system];
      ranges.atom_end = batch.atom_offsets[system + 1];
      ranges.shell_begin = batch.batch_shell_offsets[system];
      ranges.shell_end = batch.batch_shell_offsets[system + 1];
      valid =
          valid_nonempty_range(ranges.atom_begin, ranges.atom_end, batch.total_atoms) &&
                  valid_nonempty_range(ranges.shell_begin, ranges.shell_end, batch.total_shells) &&
                  batch.atom_shell_offsets[ranges.atom_begin] == ranges.shell_begin &&
                  batch.atom_shell_offsets[ranges.atom_end] == ranges.shell_end
              ? 1
              : 0;
      if (valid != 0) {
        const std::int64_t system_shells = ranges.shell_end - ranges.shell_begin;
        if (system_shells > kInt64Maximum / 2 ||
            ranges.population_end - ranges.population_begin != 2 * system_shells) {
          valid = 0;
        }
      }
    }
    if (valid == 0 && system_is_valid(system_errors, system)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SpinDeviceError::kInvalidOffsets);
    }
  }
  __syncthreads();
  if (valid == 0 || ranges.spin_channels == 1) {
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t shell_begin = batch.atom_shell_offsets[atom];
    const std::int64_t shell_end = batch.atom_shell_offsets[atom + 1];
    const std::int64_t coupling_begin = batch.coupling_offsets[atom];
    const std::int64_t coupling_end = batch.coupling_offsets[atom + 1];
    const bool shell_range_valid =
        valid_nonempty_range(shell_begin, shell_end, batch.total_shells) &&
        shell_begin >= ranges.shell_begin && shell_end <= ranges.shell_end;
    /* Subtract only after range validation so poisoned device offsets cannot
     * trigger signed overflow before being converted into a peer-local error. */
    const std::int64_t shells = shell_range_valid ? shell_end - shell_begin : 0;
    const bool atom_valid = shell_range_valid && shells <= kMaximumAtomShells &&
                            coupling_begin >= 0 && coupling_begin <= coupling_end &&
                            coupling_end <= batch.coupling_matrix_count &&
                            coupling_end - coupling_begin == shells * shells;
    if (!atom_valid) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SpinDeviceError::kInvalidCoupling);
      atomicExch(&valid, 0);
    }
  }
}

/*
 * A block owns one ragged system.  Restricted peers take the zero-only branch
 * before numerical input or W is read.  Unrestricted peers first preflight
 * their complete population and coupling slices, then write unpublished
 * potentials in the exact CPU atom/row/column order.  Parallel zeroing and
 * finite preflight are safe, but the numerical FMA sequence is intentionally
 * serialized to preserve reference rounding.
 */
__global__ void evaluate_spin_kernel(Gfn2SpinDeviceBatch batch, Gfn2SpinDeviceInput input,
                                     Gfn2SccIterationDeviceActivity activity,
                                     Gfn2SpinDeviceWorkspace workspace,
                                     std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;

  if (threadIdx.x == 0) {
    valid = stage_is_open(workspace) && activity.active_mask[system] == 1u &&
                    system_is_valid(system_errors, system)
                ? 1
                : 0;
    if (valid != 0) {
      ranges.population_begin = batch.shell_population_offsets[system];
      ranges.population_end = batch.shell_population_offsets[system + 1];
      ranges.spin_channels = batch.spin_channels[system];
      if (ranges.spin_channels == 2) {
        ranges.atom_begin = batch.atom_offsets[system];
        ranges.atom_end = batch.atom_offsets[system + 1];
        ranges.shell_begin = batch.batch_shell_offsets[system];
        ranges.shell_end = batch.batch_shell_offsets[system + 1];
      }
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t element = ranges.population_begin + threadIdx.x;
       element < ranges.population_end; element += blockDim.x) {
    workspace.potential_scratch[element] = 0.0;
  }
  if (threadIdx.x == 0) {
    workspace.energy_scratch[system] = 0.0;
  }
  __syncthreads();

  /* Restricted output is exact zero and intentionally tolerates poison input. */
  if (ranges.spin_channels == 1) {
    return;
  }

  for (std::int64_t element = ranges.population_begin + threadIdx.x;
       element < ranges.population_end; element += blockDim.x) {
    if (!isfinite(input.shell_populations[element])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2SpinDeviceError::kNonfinitePopulation);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t coupling_begin = batch.coupling_offsets[atom];
    const std::int64_t coupling_end = batch.coupling_offsets[atom + 1];
    for (std::int64_t coupling = coupling_begin; coupling < coupling_end; ++coupling) {
      if (!isfinite(batch.coupling_matrices[coupling])) {
        record_system_error(system_errors, system, device_error,
                            Gfn2SpinDeviceError::kInvalidCoupling);
        atomicExch(&valid, 0);
      }
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  const std::int64_t system_shells = ranges.shell_end - ranges.shell_begin;
  const std::int64_t magnetization_base = ranges.population_begin + system_shells;
  if (threadIdx.x == 0) {
    bool arithmetic_valid = true;
    double energy = 0.0;
    for (std::int64_t atom = ranges.atom_begin; atom < ranges.atom_end; ++atom) {
      const std::int64_t shell_begin = batch.atom_shell_offsets[atom];
      const std::int64_t shell_end = batch.atom_shell_offsets[atom + 1];
      const std::int64_t shells = shell_end - shell_begin;
      const std::int64_t coupling_begin = batch.coupling_offsets[atom];
      for (std::int64_t row = 0; row < shells; ++row) {
        double potential = 0.0;
        for (std::int64_t column = 0; column < shells; ++column) {
          const std::int64_t population =
              magnetization_base + shell_begin - ranges.shell_begin + column;
          potential = fma(batch.coupling_matrices[coupling_begin + row * shells + column],
                          input.shell_populations[population], potential);
        }
        if (!isfinite(potential)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2SpinDeviceError::kNonfinitePotentialArithmetic);
          arithmetic_valid = false;
          break;
        }
        const std::int64_t population = magnetization_base + shell_begin - ranges.shell_begin + row;
        workspace.potential_scratch[population] = potential;
        energy = fma(0.5 * input.shell_populations[population], potential, energy);
        if (!isfinite(energy)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2SpinDeviceError::kNonfiniteEnergyArithmetic);
          arithmetic_valid = false;
          break;
        }
      }
      if (!arithmetic_valid) {
        break;
      }
    }
    if (arithmetic_valid) {
      workspace.energy_scratch[system] = energy;
    }
  }
}

__global__ void publish_spin_kernel(Gfn2SpinDeviceBatch batch,
                                    Gfn2SccIterationDeviceActivity activity,
                                    Gfn2SpinDeviceOutput output, Gfn2SpinDeviceWorkspace workspace,
                                    const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!stage_is_open(workspace) || activity.active_mask[system] != 1u ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t population_begin = batch.shell_population_offsets[system];
  const std::int64_t population_end = batch.shell_population_offsets[system + 1];
  for (std::int64_t element = population_begin + threadIdx.x; element < population_end;
       element += blockDim.x) {
    output.shell_potentials[element] = workspace.potential_scratch[element];
  }
  if (threadIdx.x == 0) {
    output.spin_energies[system] = workspace.energy_scratch[system];
  }
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

struct MemoryRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                MemoryRange* range) noexcept {
  if (elements < 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes == 0u) {
    *range = {};
    return pointer == nullptr;
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

bool overlaps(const MemoryRange& first, const MemoryRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t Count>
bool pairwise_disjoint(const std::array<MemoryRange, Count>& ranges) noexcept {
  for (std::size_t first = 0; first < Count; ++first) {
    for (std::size_t second = first + 1u; second < Count; ++second) {
      if (overlaps(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t reset_gfn2_spin_device_errors_cuda(std::int64_t batch_size,
                                               std::uint32_t* system_errors,
                                               std::uint32_t* device_error,
                                               cudaStream_t stream) noexcept {
  if (batch_size <= 0 || !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  MemoryRange systems;
  MemoryRange diagnostic;
  if (!make_range(system_errors, batch_size, sizeof(*system_errors), &systems) ||
      !make_range(device_error, 1, sizeof(*device_error), &diagnostic) ||
      overlaps(systems, diagnostic)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  return status == cudaSuccess ? cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream)
                               : status;
}

cudaError_t evaluate_gfn2_spin_polarization_cuda(
    const Gfn2SpinDeviceBatch& batch, const Gfn2SpinDeviceInput& input,
    const Gfn2SccIterationDeviceActivity& activity, const Gfn2SpinDeviceOutput& output,
    const Gfn2SpinDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_shells <= 0 ||
      batch.shell_population_elements <= 0 || batch.total_atoms == kInt64Maximum ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.plan_token == 0u || input.plan_token != batch.plan_token ||
      activity.plan_token != batch.plan_token || output.plan_token != batch.plan_token ||
      workspace.plan_token != batch.plan_token || batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.atom_shell_offset_count != batch.total_atoms + 1 ||
      batch.shell_population_offset_count != batch.batch_size + 1 ||
      batch.spin_channel_count != batch.batch_size ||
      batch.coupling_offset_count != batch.total_atoms + 1 || batch.coupling_matrix_count <= 0 ||
      input.shell_population_elements != batch.shell_population_elements ||
      activity.batch_elements != batch.batch_size || activity.sequence_elements != 1 ||
      output.spin_energy_elements != batch.batch_size ||
      output.shell_potential_elements != batch.shell_population_elements ||
      workspace.energy_elements < batch.batch_size ||
      workspace.potential_elements < batch.shell_population_elements ||
      workspace.sequence_elements < 1 || !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atom_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_population_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.spin_channels, alignof(std::int32_t)) ||
      !is_aligned(batch.coupling_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.coupling_matrices, alignof(double)) ||
      !is_aligned(input.shell_populations, alignof(double)) ||
      !is_aligned(activity.active_mask, alignof(std::uint8_t)) ||
      !is_aligned(activity.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(output.spin_energies, alignof(double)) ||
      !is_aligned(output.shell_potentials, alignof(double)) ||
      !is_aligned(workspace.energy_scratch, alignof(double)) ||
      !is_aligned(workspace.potential_scratch, alignof(double)) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max())
               ? cudaErrorInvalidConfiguration
               : cudaErrorInvalidValue;
  }

  std::array<MemoryRange, 10> reads;
  std::array<MemoryRange, 7> writes;
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                  &reads[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count,
                  sizeof(*batch.batch_shell_offsets), &reads[1]) ||
      !make_range(batch.atom_shell_offsets, batch.atom_shell_offset_count,
                  sizeof(*batch.atom_shell_offsets), &reads[2]) ||
      !make_range(batch.shell_population_offsets, batch.shell_population_offset_count,
                  sizeof(*batch.shell_population_offsets), &reads[3]) ||
      !make_range(batch.spin_channels, batch.spin_channel_count, sizeof(*batch.spin_channels),
                  &reads[4]) ||
      !make_range(batch.coupling_offsets, batch.coupling_offset_count,
                  sizeof(*batch.coupling_offsets), &reads[5]) ||
      !make_range(batch.coupling_matrices, batch.coupling_matrix_count,
                  sizeof(*batch.coupling_matrices), &reads[6]) ||
      !make_range(input.shell_populations, input.shell_population_elements,
                  sizeof(*input.shell_populations), &reads[7]) ||
      !make_range(activity.active_mask, activity.batch_elements, sizeof(*activity.active_mask),
                  &reads[8]) ||
      !make_range(activity.sequence_active, 1, sizeof(*activity.sequence_active), &reads[9]) ||
      !make_range(output.spin_energies, output.spin_energy_elements, sizeof(*output.spin_energies),
                  &writes[0]) ||
      !make_range(output.shell_potentials, output.shell_potential_elements,
                  sizeof(*output.shell_potentials), &writes[1]) ||
      !make_range(workspace.energy_scratch, batch.batch_size, sizeof(*workspace.energy_scratch),
                  &writes[2]) ||
      !make_range(workspace.potential_scratch, batch.shell_population_elements,
                  sizeof(*workspace.potential_scratch), &writes[3]) ||
      !make_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active), &writes[4]) ||
      !make_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[5]) ||
      !make_range(device_error, 1, sizeof(*device_error), &writes[6]) ||
      !pairwise_disjoint(writes)) {
    return cudaErrorInvalidValue;
  }
  for (const MemoryRange& read : reads) {
    for (const MemoryRange& write : writes) {
      if (overlaps(read, write)) {
        return cudaErrorInvalidValue;
      }
    }
  }

  capture_stage_kernel<<<1, 1, 0, stream>>>(activity, device_error, workspace);
  cudaError_t status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, activity, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  evaluate_spin_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                         stream>>>(batch, input, activity, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_spin_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, activity, output, workspace, system_errors);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
