#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_mulliken.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 64;
constexpr int kMultipoleComponents = 9;
constexpr std::int64_t kMaximumInt64 = 9223372036854775807LL;

struct SystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t shell_begin;
  std::int64_t shell_end;
  std::int64_t orbital_begin;
  std::int64_t orbital_end;
  std::int64_t matrix_begin;
  std::int64_t matrix_end;
};

__device__ bool sequence_is_active(const Gfn2MullikenDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kSuccess);
}

__device__ bool system_is_active(const Gfn2MullikenDeviceActivity& activity, std::int64_t system) {
  return activity.active_mask[system] == 1u;
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2MullikenDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system,
                static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kSuccess),
                code) == static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kSuccess)) {
    atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kSuccess), code);
  }
}

__device__ bool valid_closed_range(std::int64_t begin, std::int64_t end, std::int64_t total) {
  return begin >= 0 && end >= 0 && begin <= end && end <= total;
}

__device__ bool checked_square(std::int64_t value, std::int64_t* square) {
  if (value < 0 || (value != 0 && value > kMaximumInt64 / value)) {
    return false;
  }
  *square = value * value;
  return true;
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        Gfn2MullikenDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                static_cast<std::uint32_t>(Gfn2MullikenDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

/*
 * Validate one active member without trusting any offset arithmetic. Inactive
 * terminal members return after reading only their activity byte, so stale or
 * poisoned numerical/topology storage belonging solely to them is irrelevant.
 */
__global__ void topology_preflight_kernel(Gfn2MullikenDeviceBatch batch,
                                          Gfn2MullikenDeviceActivity activity,
                                          Gfn2MullikenDeviceWorkspace workspace,
                                          std::uint32_t* system_errors,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system)) {
    return;
  }
  const std::uint8_t active = activity.active_mask[system];
  if (active == 0u) {
    return;
  }
  if (active != 1u) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kInvalidActiveMask);
    }
    return;
  }

  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = 1;
    ranges.atom_begin = batch.atom_offsets[system];
    ranges.atom_end = batch.atom_offsets[system + 1];
    ranges.shell_begin = batch.batch_shell_offsets[system];
    ranges.shell_end = batch.batch_shell_offsets[system + 1];
    ranges.orbital_begin = batch.batch_orbital_offsets[system];
    ranges.orbital_end = batch.batch_orbital_offsets[system + 1];
    ranges.matrix_begin = batch.matrix_offsets[system];
    ranges.matrix_end = batch.matrix_offsets[system + 1];
    if (!valid_closed_range(ranges.atom_begin, ranges.atom_end, batch.total_atoms) ||
        !valid_closed_range(ranges.shell_begin, ranges.shell_end, batch.total_shells) ||
        !valid_closed_range(ranges.orbital_begin, ranges.orbital_end, batch.total_orbitals) ||
        !valid_closed_range(ranges.matrix_begin, ranges.matrix_end, batch.total_matrix_elements)) {
      valid = 0;
    }
    if (valid != 0) {
      const std::int64_t atoms = ranges.atom_end - ranges.atom_begin;
      const std::int64_t shells = ranges.shell_end - ranges.shell_begin;
      const std::int64_t orbitals = ranges.orbital_end - ranges.orbital_begin;
      std::int64_t expected_matrix = 0;
      valid = atoms > 0 && shells > 0 && orbitals > 0 && atoms <= batch.maximum_system_atoms &&
              shells <= batch.maximum_system_shells && checked_square(orbitals, &expected_matrix) &&
              ranges.matrix_end - ranges.matrix_begin == expected_matrix;
    }
    if (valid != 0) {
      valid = batch.atom_shell_offsets[ranges.atom_begin] == ranges.shell_begin &&
              batch.atom_shell_offsets[ranges.atom_end] == ranges.shell_end &&
              batch.shell_orbital_offsets[ranges.shell_begin] == ranges.orbital_begin &&
              batch.shell_orbital_offsets[ranges.shell_end] == ranges.orbital_end;
    }
    if (valid != 0 && system == 0) {
      valid = ranges.atom_begin == 0 && ranges.shell_begin == 0 && ranges.orbital_begin == 0 &&
              ranges.matrix_begin == 0;
    }
    if (valid != 0 && system + 1 == batch.batch_size) {
      valid = ranges.atom_end == batch.total_atoms && ranges.shell_end == batch.total_shells &&
              ranges.orbital_end == batch.total_orbitals &&
              ranges.matrix_end == batch.total_matrix_elements;
    }
    if (valid == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kInvalidOffsets);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t shell_begin = batch.atom_shell_offsets[atom];
    const std::int64_t shell_end = batch.atom_shell_offsets[atom + 1];
    if (!valid_closed_range(shell_begin, shell_end, batch.total_shells) ||
        shell_begin < ranges.shell_begin || shell_end > ranges.shell_end ||
        shell_begin == shell_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kInvalidOffsets);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t shell = ranges.shell_begin + threadIdx.x; shell < ranges.shell_end;
       shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const std::int64_t orbital_begin = batch.shell_orbital_offsets[shell];
    const std::int64_t orbital_end = batch.shell_orbital_offsets[shell + 1];
    bool shell_valid = atom >= ranges.atom_begin && atom < ranges.atom_end;
    if (shell_valid) {
      shell_valid =
          shell >= batch.atom_shell_offsets[atom] && shell < batch.atom_shell_offsets[atom + 1];
    }
    shell_valid = shell_valid &&
                  valid_closed_range(orbital_begin, orbital_end, batch.total_orbitals) &&
                  orbital_begin >= ranges.orbital_begin && orbital_end <= ranges.orbital_end &&
                  orbital_begin < orbital_end;
    if (!shell_valid) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kInvalidShellMetadata);
      atomicExch(&valid, 0);
    }
    if (!isfinite(batch.reference_shell_occupations[shell])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kNonfiniteReferenceOccupation);
      atomicExch(&valid, 0);
    }
  }
}

__global__ void shell_population_kernel(Gfn2MullikenDeviceBatch batch,
                                        Gfn2MullikenDeviceInput input,
                                        Gfn2MullikenDeviceActivity activity,
                                        Gfn2MullikenDeviceWorkspace workspace,
                                        std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t global = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t system = global / batch.maximum_system_shells;
  const std::int64_t local_shell = global - system * batch.maximum_system_shells;
  if (!sequence_is_active(workspace) || !system_is_active(activity, system) ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shells = batch.batch_shell_offsets[system + 1] - shell_begin;
  if (local_shell >= shells) {
    return;
  }
  const std::int64_t shell = shell_begin + local_shell;
  const std::int64_t ket_begin = batch.shell_orbital_offsets[shell];
  const std::int64_t ket_end = batch.shell_orbital_offsets[shell + 1];
  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbitals = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t contractions = (ket_end - ket_begin) * orbitals;

  double sum = 0.0;
  bool finite = true;
  for (std::int64_t local = threadIdx.x; local < contractions; local += blockDim.x) {
    const std::int64_t ket = ket_begin + local / orbitals;
    const std::int64_t bra = orbital_begin + local % orbitals;
    const std::int64_t matrix =
        matrix_begin + (bra - orbital_begin) * orbitals + ket - orbital_begin;
    const double density = input.density[matrix];
    const double overlap = input.overlap[matrix];
    if (!isfinite(density)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kNonfiniteDensity);
      finite = false;
    }
    if (!isfinite(overlap)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kNonfiniteIntegral);
      finite = false;
    }
    if (finite) {
      sum = fma(-density, overlap, sum);
      if (!isfinite(sum)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2MullikenDeviceError::kNonfiniteContraction);
        finite = false;
      }
    }
  }

  __shared__ double reduction[kThreadsPerBlock];
  reduction[threadIdx.x] = finite ? sum : 0.0;
  __syncthreads();
  for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const double updated = reduction[threadIdx.x] + reduction[threadIdx.x + stride];
      if (!isfinite(updated)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2MullikenDeviceError::kNonfiniteContraction);
      } else {
        reduction[threadIdx.x] = updated;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && system_is_valid(system_errors, system)) {
    const double charge = reduction[0] + batch.reference_shell_occupations[shell];
    if (!isfinite(charge)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kNonfiniteContraction);
    } else {
      workspace.qsh_scratch[shell] = charge;
    }
  }
}

__global__ void multipole_population_kernel(Gfn2MullikenDeviceBatch batch,
                                            Gfn2MullikenDeviceInput input,
                                            Gfn2MullikenDeviceActivity activity,
                                            Gfn2MullikenDeviceWorkspace workspace,
                                            std::uint32_t* system_errors,
                                            std::uint32_t* device_error) {
  const std::int64_t global = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t system = global / batch.maximum_system_atoms;
  const std::int64_t local_atom = global - system * batch.maximum_system_atoms;
  if (!sequence_is_active(workspace) || !system_is_active(activity, system) ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atoms = batch.atom_offsets[system + 1] - atom_begin;
  if (local_atom >= atoms) {
    return;
  }
  const std::int64_t atom = atom_begin + local_atom;
  const std::int64_t first_shell = batch.atom_shell_offsets[atom];
  const std::int64_t last_shell = batch.atom_shell_offsets[atom + 1];
  const std::int64_t ket_begin = batch.shell_orbital_offsets[first_shell];
  const std::int64_t ket_end = batch.shell_orbital_offsets[last_shell];
  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbitals = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t contractions = (ket_end - ket_begin) * orbitals;

  double sum[kMultipoleComponents] = {};
  bool finite = true;
  for (std::int64_t local = threadIdx.x; local < contractions; local += blockDim.x) {
    const std::int64_t ket = ket_begin + local / orbitals;
    const std::int64_t bra = orbital_begin + local % orbitals;
    const std::int64_t matrix =
        matrix_begin + (bra - orbital_begin) * orbitals + ket - orbital_begin;
    const double density = input.density[matrix];
    if (!isfinite(density)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kNonfiniteDensity);
      finite = false;
    }
    for (int component = 0; component < kMultipoleComponents; ++component) {
      const double integral =
          component < kGfn2MullikenDipoleComponents
              ? input.dipole_integrals[component * batch.total_matrix_elements + matrix]
              : input.quadrupole_integrals[(component - kGfn2MullikenDipoleComponents) *
                                               batch.total_matrix_elements +
                                           matrix];
      if (!isfinite(integral)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2MullikenDeviceError::kNonfiniteIntegral);
        finite = false;
      }
      if (finite) {
        sum[component] = fma(-density, integral, sum[component]);
        if (!isfinite(sum[component])) {
          record_system_error(system_errors, system, device_error,
                              Gfn2MullikenDeviceError::kNonfiniteContraction);
          finite = false;
        }
      }
    }
  }

  __shared__ double reduction[kMultipoleComponents][kThreadsPerBlock];
  for (int component = 0; component < kMultipoleComponents; ++component) {
    reduction[component][threadIdx.x] = finite ? sum[component] : 0.0;
  }
  __syncthreads();
  for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      for (int component = 0; component < kMultipoleComponents; ++component) {
        const double updated =
            reduction[component][threadIdx.x] + reduction[component][threadIdx.x + stride];
        if (!isfinite(updated)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2MullikenDeviceError::kNonfiniteContraction);
        } else {
          reduction[component][threadIdx.x] = updated;
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && system_is_valid(system_errors, system)) {
    for (int component = 0; component < kGfn2MullikenDipoleComponents; ++component) {
      workspace.dipole_scratch[atom * kGfn2MullikenDipoleComponents + component] =
          reduction[component][0];
    }
    for (int component = 0; component < kGfn2MullikenQuadrupoleComponents; ++component) {
      workspace.quadrupole_scratch[atom * kGfn2MullikenQuadrupoleComponents + component] =
          reduction[component + kGfn2MullikenDipoleComponents][0];
    }
  }
}

__global__ void atom_population_kernel(Gfn2MullikenDeviceBatch batch,
                                       Gfn2MullikenDeviceActivity activity,
                                       Gfn2MullikenDeviceWorkspace workspace,
                                       std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t global = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t system = global / batch.maximum_system_atoms;
  const std::int64_t local_atom = global - system * batch.maximum_system_atoms;
  if (!sequence_is_active(workspace) || !system_is_active(activity, system) ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atoms = batch.atom_offsets[system + 1] - atom_begin;
  if (local_atom >= atoms || threadIdx.x != 0) {
    return;
  }
  const std::int64_t atom = atom_begin + local_atom;
  const std::int64_t shell_begin = batch.atom_shell_offsets[atom];
  const std::int64_t shell_end = batch.atom_shell_offsets[atom + 1];
  double charge = 0.0;
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
    charge += workspace.qsh_scratch[shell];
    if (!isfinite(charge)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kNonfinitePopulationReduction);
      return;
    }
  }
  workspace.qat_scratch[atom] = charge;
}

__global__ void publish_population_kernel(Gfn2MullikenDeviceBatch batch,
                                          Gfn2MullikenDeviceActivity activity,
                                          Gfn2MullikenDevicePopulation population,
                                          Gfn2MullikenDeviceWorkspace workspace,
                                          const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || !system_is_active(activity, system) ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    population.qsh[shell] = workspace.qsh_scratch[shell];
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    population.qat[atom] = workspace.qat_scratch[atom];
    for (int component = 0; component < kGfn2MullikenDipoleComponents; ++component) {
      const std::int64_t index = atom * kGfn2MullikenDipoleComponents + component;
      population.dipole[index] = workspace.dipole_scratch[index];
    }
    for (int component = 0; component < kGfn2MullikenQuadrupoleComponents; ++component) {
      const std::int64_t index = atom * kGfn2MullikenQuadrupoleComponents + component;
      population.quadrupole[index] = workspace.quadrupole_scratch[index];
    }
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
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes == 0u) {
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

cudaError_t reset_gfn2_mulliken_device_errors_cuda(std::int64_t batch_size,
                                                   std::uint32_t* system_errors,
                                                   std::uint32_t* device_error,
                                                   cudaStream_t stream) noexcept {
  if (batch_size <= 0 || !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      static_cast<std::uint64_t>(batch_size) >
          std::numeric_limits<std::size_t>::max() / sizeof(*system_errors)) {
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

cudaError_t evaluate_gfn2_mulliken_population_cuda(
    const Gfn2MullikenDeviceBatch& batch, const Gfn2MullikenDeviceInput& input,
    const Gfn2MullikenDeviceActivity& activity, const Gfn2MullikenDevicePopulation& population,
    const Gfn2MullikenDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_shells <= 0 ||
      batch.total_orbitals <= 0 || batch.total_matrix_elements <= 0 ||
      batch.maximum_system_atoms <= 0 || batch.maximum_system_shells <= 0 ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.total_atoms > kMaximumInt64 / kGfn2MullikenQuadrupoleComponents ||
      batch.total_matrix_elements > kMaximumInt64 / kGfn2MullikenQuadrupoleComponents ||
      batch.plan_token == 0u || input.plan_token != batch.plan_token ||
      activity.plan_token != batch.plan_token || population.plan_token != batch.plan_token ||
      workspace.plan_token != batch.plan_token || batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.batch_orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.atom_shell_offset_count != batch.total_atoms + 1 ||
      batch.shell_orbital_offset_count != batch.total_shells + 1 ||
      batch.shell_to_atom_count != batch.total_shells ||
      batch.reference_occupation_count != batch.total_shells ||
      input.density_elements != batch.total_matrix_elements ||
      input.overlap_elements != batch.total_matrix_elements ||
      input.dipole_integral_elements !=
          batch.total_matrix_elements * kGfn2MullikenDipoleComponents ||
      input.quadrupole_integral_elements !=
          batch.total_matrix_elements * kGfn2MullikenQuadrupoleComponents ||
      activity.elements != batch.batch_size || population.qsh_elements != batch.total_shells ||
      population.qat_elements != batch.total_atoms ||
      population.dipole_elements != batch.total_atoms * kGfn2MullikenDipoleComponents ||
      population.quadrupole_elements != batch.total_atoms * kGfn2MullikenQuadrupoleComponents ||
      workspace.qsh_elements < batch.total_shells || workspace.qat_elements < batch.total_atoms ||
      workspace.dipole_elements < batch.total_atoms * kGfn2MullikenDipoleComponents ||
      workspace.quadrupole_elements < batch.total_atoms * kGfn2MullikenQuadrupoleComponents ||
      workspace.sequence_elements < 1 || !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atom_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_to_atom, alignof(std::int64_t)) ||
      !is_aligned(batch.reference_shell_occupations, alignof(double)) ||
      !is_aligned(input.density, alignof(double)) || !is_aligned(input.overlap, alignof(double)) ||
      !is_aligned(input.dipole_integrals, alignof(double)) ||
      !is_aligned(input.quadrupole_integrals, alignof(double)) ||
      !is_aligned(activity.active_mask, alignof(std::uint8_t)) ||
      !is_aligned(population.qsh, alignof(double)) ||
      !is_aligned(population.qat, alignof(double)) ||
      !is_aligned(population.dipole, alignof(double)) ||
      !is_aligned(population.quadrupole, alignof(double)) ||
      !is_aligned(workspace.qsh_scratch, alignof(double)) ||
      !is_aligned(workspace.qat_scratch, alignof(double)) ||
      !is_aligned(workspace.dipole_scratch, alignof(double)) ||
      !is_aligned(workspace.quadrupole_scratch, alignof(double)) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max())
               ? cudaErrorInvalidConfiguration
               : cudaErrorInvalidValue;
  }
  if (batch.maximum_system_atoms > kMaximumInt64 / batch.batch_size ||
      batch.maximum_system_shells > kMaximumInt64 / batch.batch_size) {
    return cudaErrorInvalidConfiguration;
  }
  const std::int64_t atom_blocks = batch.maximum_system_atoms * batch.batch_size;
  const std::int64_t shell_blocks = batch.maximum_system_shells * batch.batch_size;
  if (atom_blocks > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      shell_blocks > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    return cudaErrorInvalidConfiguration;
  }

  std::array<MemoryRange, 23> ranges;
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                  &ranges[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count,
                  sizeof(*batch.batch_shell_offsets), &ranges[1]) ||
      !make_range(batch.batch_orbital_offsets, batch.batch_orbital_offset_count,
                  sizeof(*batch.batch_orbital_offsets), &ranges[2]) ||
      !make_range(batch.matrix_offsets, batch.matrix_offset_count, sizeof(*batch.matrix_offsets),
                  &ranges[3]) ||
      !make_range(batch.atom_shell_offsets, batch.atom_shell_offset_count,
                  sizeof(*batch.atom_shell_offsets), &ranges[4]) ||
      !make_range(batch.shell_orbital_offsets, batch.shell_orbital_offset_count,
                  sizeof(*batch.shell_orbital_offsets), &ranges[5]) ||
      !make_range(batch.shell_to_atom, batch.shell_to_atom_count, sizeof(*batch.shell_to_atom),
                  &ranges[6]) ||
      !make_range(batch.reference_shell_occupations, batch.reference_occupation_count,
                  sizeof(*batch.reference_shell_occupations), &ranges[7]) ||
      !make_range(input.density, input.density_elements, sizeof(*input.density), &ranges[8]) ||
      !make_range(input.overlap, input.overlap_elements, sizeof(*input.overlap), &ranges[9]) ||
      !make_range(input.dipole_integrals, input.dipole_integral_elements,
                  sizeof(*input.dipole_integrals), &ranges[10]) ||
      !make_range(input.quadrupole_integrals, input.quadrupole_integral_elements,
                  sizeof(*input.quadrupole_integrals), &ranges[11]) ||
      !make_range(activity.active_mask, activity.elements, sizeof(*activity.active_mask),
                  &ranges[12]) ||
      !make_range(population.qsh, population.qsh_elements, sizeof(*population.qsh), &ranges[13]) ||
      !make_range(population.qat, population.qat_elements, sizeof(*population.qat), &ranges[14]) ||
      !make_range(population.dipole, population.dipole_elements, sizeof(*population.dipole),
                  &ranges[15]) ||
      !make_range(population.quadrupole, population.quadrupole_elements,
                  sizeof(*population.quadrupole), &ranges[16]) ||
      !make_range(workspace.qsh_scratch, workspace.qsh_elements, sizeof(*workspace.qsh_scratch),
                  &ranges[17]) ||
      !make_range(workspace.qat_scratch, workspace.qat_elements, sizeof(*workspace.qat_scratch),
                  &ranges[18]) ||
      !make_range(workspace.dipole_scratch, workspace.dipole_elements,
                  sizeof(*workspace.dipole_scratch), &ranges[19]) ||
      !make_range(workspace.quadrupole_scratch, workspace.quadrupole_elements,
                  sizeof(*workspace.quadrupole_scratch), &ranges[20]) ||
      !make_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active), &ranges[21]) ||
      !make_range(system_errors, batch.batch_size, sizeof(*system_errors), &ranges[22]) ||
      !pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  MemoryRange diagnostic;
  if (!make_range(device_error, 1, sizeof(*device_error), &diagnostic)) {
    return cudaErrorInvalidValue;
  }
  for (const MemoryRange& range : ranges) {
    if (overlaps(range, diagnostic)) {
      return cudaErrorInvalidValue;
    }
  }

  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace);
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
  shell_population_kernel<<<static_cast<unsigned int>(shell_blocks), kThreadsPerBlock, 0, stream>>>(
      batch, input, activity, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  multipole_population_kernel<<<static_cast<unsigned int>(atom_blocks), kThreadsPerBlock, 0,
                                stream>>>(batch, input, activity, workspace, system_errors,
                                          device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  atom_population_kernel<<<static_cast<unsigned int>(atom_blocks), kThreadsPerBlock, 0, stream>>>(
      batch, activity, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_population_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, activity, population, workspace, system_errors);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
