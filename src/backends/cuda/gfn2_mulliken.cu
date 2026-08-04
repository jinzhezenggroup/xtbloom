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

struct SpinSystemRanges {
  SystemRanges topology;
  std::int64_t spin_begin;
  std::int64_t spin_end;
  std::int64_t density_begin;
  std::int64_t density_end;
  std::int64_t shell_population_begin;
  std::int64_t shell_population_end;
  std::int64_t atom_population_begin;
  std::int64_t atom_population_end;
  std::int32_t spin_channels;
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

__device__ bool checked_product(std::int64_t first, std::int64_t second, std::int64_t* product) {
  if (first < 0 || second < 0 || (first != 0 && second > kMaximumInt64 / first)) {
    return false;
  }
  *product = first * second;
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

/*
 * Validate the WavefunctionLayout spin projection without weakening the
 * inactive-member contract. Only an active member may expose its spin count or
 * spin-dependent offsets to this kernel.
 */
__global__ void spin_topology_preflight_kernel(Gfn2MullikenDeviceBatch batch,
                                               Gfn2WavefunctionLayoutView layout,
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

  __shared__ SpinSystemRanges ranges;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = 1;
    SystemRanges& topology = ranges.topology;
    topology.atom_begin = batch.atom_offsets[system];
    topology.atom_end = batch.atom_offsets[system + 1];
    topology.shell_begin = batch.batch_shell_offsets[system];
    topology.shell_end = batch.batch_shell_offsets[system + 1];
    topology.orbital_begin = batch.batch_orbital_offsets[system];
    topology.orbital_end = batch.batch_orbital_offsets[system + 1];
    topology.matrix_begin = batch.matrix_offsets[system];
    topology.matrix_end = batch.matrix_offsets[system + 1];
    ranges.spin_begin = layout.spin_channel_offsets[system];
    ranges.spin_end = layout.spin_channel_offsets[system + 1];
    ranges.density_begin = layout.spin_matrix_offsets[system];
    ranges.density_end = layout.spin_matrix_offsets[system + 1];
    ranges.shell_population_begin = layout.spin_shell_offsets[system];
    ranges.shell_population_end = layout.spin_shell_offsets[system + 1];
    ranges.atom_population_begin = layout.spin_atom_offsets[system];
    ranges.atom_population_end = layout.spin_atom_offsets[system + 1];
    ranges.spin_channels = layout.spin_channels[system];

    if (!valid_closed_range(topology.atom_begin, topology.atom_end, batch.total_atoms) ||
        !valid_closed_range(topology.shell_begin, topology.shell_end, batch.total_shells) ||
        !valid_closed_range(topology.orbital_begin, topology.orbital_end, batch.total_orbitals) ||
        !valid_closed_range(topology.matrix_begin, topology.matrix_end,
                            batch.total_matrix_elements) ||
        !valid_closed_range(ranges.spin_begin, ranges.spin_end, layout.total_spin_channels) ||
        !valid_closed_range(ranges.density_begin, ranges.density_end,
                            layout.total_spin_matrix_elements) ||
        !valid_closed_range(ranges.shell_population_begin, ranges.shell_population_end,
                            layout.total_spin_shells) ||
        !valid_closed_range(ranges.atom_population_begin, ranges.atom_population_end,
                            layout.total_spin_atoms)) {
      valid = 0;
    }
    if (valid != 0 && ranges.spin_channels != 1 && ranges.spin_channels != 2) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kInvalidSpinChannels);
      valid = 0;
    }
    if (valid != 0) {
      const std::int64_t atoms = topology.atom_end - topology.atom_begin;
      const std::int64_t shells = topology.shell_end - topology.shell_begin;
      const std::int64_t orbitals = topology.orbital_end - topology.orbital_begin;
      std::int64_t expected_matrix = 0;
      std::int64_t expected_density = 0;
      std::int64_t expected_shell_population = 0;
      std::int64_t expected_atom_population = 0;
      valid = atoms > 0 && shells > 0 && orbitals > 0 && atoms <= batch.maximum_system_atoms &&
              shells <= batch.maximum_system_shells && checked_square(orbitals, &expected_matrix) &&
              checked_product(expected_matrix, ranges.spin_channels, &expected_density) &&
              checked_product(shells, ranges.spin_channels, &expected_shell_population) &&
              checked_product(atoms, ranges.spin_channels, &expected_atom_population) &&
              topology.matrix_end - topology.matrix_begin == expected_matrix &&
              ranges.spin_end - ranges.spin_begin == ranges.spin_channels &&
              ranges.density_end - ranges.density_begin == expected_density &&
              ranges.shell_population_end - ranges.shell_population_begin ==
                  expected_shell_population &&
              ranges.atom_population_end - ranges.atom_population_begin == expected_atom_population;
    }
    if (valid != 0) {
      valid = batch.atom_shell_offsets[topology.atom_begin] == topology.shell_begin &&
              batch.atom_shell_offsets[topology.atom_end] == topology.shell_end &&
              batch.shell_orbital_offsets[topology.shell_begin] == topology.orbital_begin &&
              batch.shell_orbital_offsets[topology.shell_end] == topology.orbital_end;
    }
    if (valid != 0 && system == 0) {
      valid = topology.atom_begin == 0 && topology.shell_begin == 0 &&
              topology.orbital_begin == 0 && topology.matrix_begin == 0 && ranges.spin_begin == 0 &&
              ranges.density_begin == 0 && ranges.shell_population_begin == 0 &&
              ranges.atom_population_begin == 0;
    }
    if (valid != 0 && system + 1 == batch.batch_size) {
      valid = topology.atom_end == batch.total_atoms && topology.shell_end == batch.total_shells &&
              topology.orbital_end == batch.total_orbitals &&
              topology.matrix_end == batch.total_matrix_elements &&
              ranges.spin_end == layout.total_spin_channels &&
              ranges.density_end == layout.total_spin_matrix_elements &&
              ranges.shell_population_end == layout.total_spin_shells &&
              ranges.atom_population_end == layout.total_spin_atoms;
    }
    if (valid == 0 && system_is_valid(system_errors, system)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kInvalidOffsets);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  const SystemRanges& topology = ranges.topology;
  for (std::int64_t atom = topology.atom_begin + threadIdx.x; atom < topology.atom_end;
       atom += blockDim.x) {
    const std::int64_t shell_begin = batch.atom_shell_offsets[atom];
    const std::int64_t shell_end = batch.atom_shell_offsets[atom + 1];
    if (!valid_closed_range(shell_begin, shell_end, batch.total_shells) ||
        shell_begin < topology.shell_begin || shell_end > topology.shell_end ||
        shell_begin == shell_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kInvalidOffsets);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t shell = topology.shell_begin + threadIdx.x; shell < topology.shell_end;
       shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const std::int64_t orbital_begin = batch.shell_orbital_offsets[shell];
    const std::int64_t orbital_end = batch.shell_orbital_offsets[shell + 1];
    bool shell_valid = atom >= topology.atom_begin && atom < topology.atom_end;
    if (shell_valid) {
      shell_valid =
          shell >= batch.atom_shell_offsets[atom] && shell < batch.atom_shell_offsets[atom + 1];
    }
    shell_valid = shell_valid &&
                  valid_closed_range(orbital_begin, orbital_end, batch.total_orbitals) &&
                  orbital_begin >= topology.orbital_begin && orbital_end <= topology.orbital_end &&
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

__global__ void spin_shell_population_kernel(
    Gfn2MullikenDeviceBatch batch, Gfn2WavefunctionLayoutView layout, Gfn2MullikenDeviceInput input,
    Gfn2MullikenDeviceActivity activity, Gfn2MullikenDeviceWorkspace workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t global = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t shell_span = 2 * batch.maximum_system_shells;
  const std::int64_t system = global / shell_span;
  const std::int64_t system_local = global - system * shell_span;
  const std::int64_t spin = system_local / batch.maximum_system_shells;
  const std::int64_t local_shell = system_local - spin * batch.maximum_system_shells;
  if (!sequence_is_active(workspace) || !system_is_active(activity, system) ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int32_t nspin = layout.spin_channels[system];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shells = batch.batch_shell_offsets[system + 1] - shell_begin;
  if (spin >= nspin || local_shell >= shells) {
    return;
  }
  const std::int64_t shell = shell_begin + local_shell;
  const std::int64_t ket_begin = batch.shell_orbital_offsets[shell];
  const std::int64_t ket_end = batch.shell_orbital_offsets[shell + 1];
  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbitals = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t density_begin =
      layout.spin_matrix_offsets[system] + spin * orbitals * orbitals;
  const std::int64_t contractions = (ket_end - ket_begin) * orbitals;

  double sum = 0.0;
  bool finite = true;
  for (std::int64_t local = threadIdx.x; local < contractions; local += blockDim.x) {
    const std::int64_t ket = ket_begin + local / orbitals;
    const std::int64_t bra = orbital_begin + local % orbitals;
    const std::int64_t local_matrix = (bra - orbital_begin) * orbitals + ket - orbital_begin;
    const double density = input.density[density_begin + local_matrix];
    const double overlap = input.overlap[matrix_begin + local_matrix];
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
    const std::int64_t output = layout.spin_shell_offsets[system] + spin * shells + local_shell;
    workspace.qsh_scratch[output] = reduction[0];
  }
}

__global__ void spin_multipole_population_kernel(
    Gfn2MullikenDeviceBatch batch, Gfn2WavefunctionLayoutView layout, Gfn2MullikenDeviceInput input,
    Gfn2MullikenDeviceActivity activity, Gfn2MullikenDeviceWorkspace workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t global = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t atom_span = 2 * batch.maximum_system_atoms;
  const std::int64_t system = global / atom_span;
  const std::int64_t system_local = global - system * atom_span;
  const std::int64_t spin = system_local / batch.maximum_system_atoms;
  const std::int64_t local_atom = system_local - spin * batch.maximum_system_atoms;
  if (!sequence_is_active(workspace) || !system_is_active(activity, system) ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int32_t nspin = layout.spin_channels[system];
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atoms = batch.atom_offsets[system + 1] - atom_begin;
  if (spin >= nspin || local_atom >= atoms) {
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
  const std::int64_t density_begin =
      layout.spin_matrix_offsets[system] + spin * orbitals * orbitals;
  const std::int64_t contractions = (ket_end - ket_begin) * orbitals;

  double sum[kMultipoleComponents] = {};
  bool finite = true;
  for (std::int64_t local = threadIdx.x; local < contractions; local += blockDim.x) {
    const std::int64_t ket = ket_begin + local / orbitals;
    const std::int64_t bra = orbital_begin + local % orbitals;
    const std::int64_t local_matrix = (bra - orbital_begin) * orbitals + ket - orbital_begin;
    const std::int64_t matrix = matrix_begin + local_matrix;
    const double density = input.density[density_begin + local_matrix];
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
    const std::int64_t atom_population = layout.spin_atom_offsets[system];
    const std::int64_t channel_atom = spin * atoms + local_atom;
    for (int component = 0; component < kGfn2MullikenDipoleComponents; ++component) {
      workspace.dipole_scratch[(atom_population + channel_atom) * kGfn2MullikenDipoleComponents +
                               component] = reduction[component][0];
    }
    for (int component = 0; component < kGfn2MullikenQuadrupoleComponents; ++component) {
      workspace
          .quadrupole_scratch[(atom_population + channel_atom) * kGfn2MullikenQuadrupoleComponents +
                              component] = reduction[component + kGfn2MullikenDipoleComponents][0];
    }
  }
}

__global__ void spin_conversion_kernel(Gfn2MullikenDeviceBatch batch,
                                       Gfn2WavefunctionLayoutView layout,
                                       Gfn2MullikenDeviceActivity activity,
                                       Gfn2MullikenDeviceWorkspace workspace,
                                       std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || !system_is_active(activity, system) ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int32_t nspin = layout.spin_channels[system];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shells = batch.batch_shell_offsets[system + 1] - shell_begin;
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atoms = batch.atom_offsets[system + 1] - atom_begin;
  const std::int64_t qsh_begin = layout.spin_shell_offsets[system];
  const std::int64_t qat_begin = layout.spin_atom_offsets[system];

  for (std::int64_t local_shell = threadIdx.x; local_shell < shells; local_shell += blockDim.x) {
    const double alpha = workspace.qsh_scratch[qsh_begin + local_shell];
    const double reference = batch.reference_shell_occupations[shell_begin + local_shell];
    if (nspin == 1) {
      const double charge = alpha + reference;
      if (!isfinite(charge)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2MullikenDeviceError::kNonfiniteSpinConversion);
      } else {
        workspace.qsh_scratch[qsh_begin + local_shell] = charge;
      }
    } else {
      const double beta = workspace.qsh_scratch[qsh_begin + shells + local_shell];
      const double charge = alpha + beta + reference;
      /*
       * alpha/beta are negative electronic contractions (-N_alpha/-N_beta),
       * so alpha - beta is the public m = N_beta - N_alpha convention.
       */
      const double magnetization = alpha - beta;
      if (!isfinite(charge) || !isfinite(magnetization)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2MullikenDeviceError::kNonfiniteSpinConversion);
      } else {
        workspace.qsh_scratch[qsh_begin + local_shell] = charge;
        workspace.qsh_scratch[qsh_begin + shells + local_shell] = magnetization;
      }
    }
  }

  if (nspin == 2) {
    for (std::int64_t local_atom = threadIdx.x; local_atom < atoms; local_atom += blockDim.x) {
      for (int component = 0; component < kGfn2MullikenDipoleComponents; ++component) {
        const std::int64_t alpha_index =
            (qat_begin + local_atom) * kGfn2MullikenDipoleComponents + component;
        const std::int64_t beta_index =
            (qat_begin + atoms + local_atom) * kGfn2MullikenDipoleComponents + component;
        const double alpha = workspace.dipole_scratch[alpha_index];
        const double beta = workspace.dipole_scratch[beta_index];
        const double charge = alpha + beta;
        /* The contraction scratch carries the electronic minus sign. */
        const double magnetization = alpha - beta;
        if (!isfinite(charge) || !isfinite(magnetization)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2MullikenDeviceError::kNonfiniteSpinConversion);
        } else {
          workspace.dipole_scratch[alpha_index] = charge;
          workspace.dipole_scratch[beta_index] = magnetization;
        }
      }
      for (int component = 0; component < kGfn2MullikenQuadrupoleComponents; ++component) {
        const std::int64_t alpha_index =
            (qat_begin + local_atom) * kGfn2MullikenQuadrupoleComponents + component;
        const std::int64_t beta_index =
            (qat_begin + atoms + local_atom) * kGfn2MullikenQuadrupoleComponents + component;
        const double alpha = workspace.quadrupole_scratch[alpha_index];
        const double beta = workspace.quadrupole_scratch[beta_index];
        const double charge = alpha + beta;
        /* The contraction scratch carries the electronic minus sign. */
        const double magnetization = alpha - beta;
        if (!isfinite(charge) || !isfinite(magnetization)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2MullikenDeviceError::kNonfiniteSpinConversion);
        } else {
          workspace.quadrupole_scratch[alpha_index] = charge;
          workspace.quadrupole_scratch[beta_index] = magnetization;
        }
      }
    }
  }
}

__global__ void spin_atom_population_kernel(Gfn2MullikenDeviceBatch batch,
                                            Gfn2WavefunctionLayoutView layout,
                                            Gfn2MullikenDeviceActivity activity,
                                            Gfn2MullikenDeviceWorkspace workspace,
                                            std::uint32_t* system_errors,
                                            std::uint32_t* device_error) {
  const std::int64_t global = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t atom_span = 2 * batch.maximum_system_atoms;
  const std::int64_t system = global / atom_span;
  const std::int64_t system_local = global - system * atom_span;
  const std::int64_t channel = system_local / batch.maximum_system_atoms;
  const std::int64_t local_atom = system_local - channel * batch.maximum_system_atoms;
  if (!sequence_is_active(workspace) || !system_is_active(activity, system) ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int32_t nspin = layout.spin_channels[system];
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atoms = batch.atom_offsets[system + 1] - atom_begin;
  if (channel >= nspin || local_atom >= atoms || threadIdx.x != 0) {
    return;
  }
  const std::int64_t atom = atom_begin + local_atom;
  const std::int64_t shell_begin = batch.atom_shell_offsets[atom];
  const std::int64_t shell_end = batch.atom_shell_offsets[atom + 1];
  const std::int64_t system_shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shells = batch.batch_shell_offsets[system + 1] - system_shell_begin;
  const std::int64_t qsh_begin = layout.spin_shell_offsets[system];
  double population = 0.0;
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
    population += workspace.qsh_scratch[qsh_begin + channel * shells + shell - system_shell_begin];
    if (!isfinite(population)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2MullikenDeviceError::kNonfinitePopulationReduction);
      return;
    }
  }
  workspace.qat_scratch[layout.spin_atom_offsets[system] + channel * atoms + local_atom] =
      population;
}

__global__ void publish_spin_population_kernel(Gfn2MullikenDeviceBatch batch,
                                               Gfn2WavefunctionLayoutView layout,
                                               Gfn2MullikenDeviceActivity activity,
                                               Gfn2MullikenDevicePopulation population,
                                               Gfn2MullikenDeviceWorkspace workspace,
                                               const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || !system_is_active(activity, system) ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t qsh_begin = layout.spin_shell_offsets[system];
  const std::int64_t qsh_end = layout.spin_shell_offsets[system + 1];
  const std::int64_t qat_begin = layout.spin_atom_offsets[system];
  const std::int64_t qat_end = layout.spin_atom_offsets[system + 1];
  for (std::int64_t index = qsh_begin + threadIdx.x; index < qsh_end; index += blockDim.x) {
    population.qsh[index] = workspace.qsh_scratch[index];
  }
  for (std::int64_t index = qat_begin + threadIdx.x; index < qat_end; index += blockDim.x) {
    population.qat[index] = workspace.qat_scratch[index];
    for (int component = 0; component < kGfn2MullikenDipoleComponents; ++component) {
      const std::int64_t component_index = index * kGfn2MullikenDipoleComponents + component;
      population.dipole[component_index] = workspace.dipole_scratch[component_index];
    }
    for (int component = 0; component < kGfn2MullikenQuadrupoleComponents; ++component) {
      const std::int64_t component_index = index * kGfn2MullikenQuadrupoleComponents + component;
      population.quadrupole[component_index] = workspace.quadrupole_scratch[component_index];
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

cudaError_t evaluate_gfn2_mulliken_population_spin_cuda(
    const Gfn2MullikenDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const Gfn2MullikenDeviceInput& input, const Gfn2MullikenDeviceActivity& activity,
    const Gfn2MullikenDevicePopulation& population, const Gfn2MullikenDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_shells <= 0 ||
      batch.total_orbitals <= 0 || batch.total_matrix_elements <= 0 ||
      layout.batch_size != batch.batch_size || layout.total_spin_channels <= 0 ||
      layout.total_spin_matrix_elements <= 0 || layout.total_spin_shells <= 0 ||
      layout.total_spin_atoms <= 0 || batch.maximum_system_atoms <= 0 ||
      batch.maximum_system_shells <= 0 ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.total_atoms > kMaximumInt64 / kGfn2MullikenQuadrupoleComponents ||
      batch.total_matrix_elements > kMaximumInt64 / kGfn2MullikenQuadrupoleComponents ||
      layout.total_spin_atoms > kMaximumInt64 / kGfn2MullikenQuadrupoleComponents ||
      batch.plan_token == 0u || input.plan_token != batch.plan_token ||
      activity.plan_token != batch.plan_token || population.plan_token != batch.plan_token ||
      workspace.plan_token != batch.plan_token || layout.plan_token != batch.plan_token ||
      layout.memory_space != Gfn2PlanMemorySpace::kCudaDevice ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.batch_orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.atom_shell_offset_count != batch.total_atoms + 1 ||
      batch.shell_orbital_offset_count != batch.total_shells + 1 ||
      batch.shell_to_atom_count != batch.total_shells ||
      batch.reference_occupation_count != batch.total_shells ||
      layout.spin_channel_offset_count != batch.batch_size + 1 ||
      layout.spin_matrix_offset_count != batch.batch_size + 1 ||
      layout.spin_shell_offset_count != batch.batch_size + 1 ||
      layout.spin_atom_offset_count != batch.batch_size + 1 ||
      layout.spin_channel_count != batch.batch_size ||
      layout.total_spin_channels < batch.batch_size ||
      layout.total_spin_channels > 2 * batch.batch_size ||
      input.density_elements != layout.total_spin_matrix_elements ||
      input.overlap_elements != batch.total_matrix_elements ||
      input.dipole_integral_elements !=
          batch.total_matrix_elements * kGfn2MullikenDipoleComponents ||
      input.quadrupole_integral_elements !=
          batch.total_matrix_elements * kGfn2MullikenQuadrupoleComponents ||
      activity.elements != batch.batch_size ||
      population.qsh_elements != layout.total_spin_shells ||
      population.qat_elements != layout.total_spin_atoms ||
      population.dipole_elements != layout.total_spin_atoms * kGfn2MullikenDipoleComponents ||
      population.quadrupole_elements !=
          layout.total_spin_atoms * kGfn2MullikenQuadrupoleComponents ||
      workspace.qsh_elements < layout.total_spin_shells ||
      workspace.qat_elements < layout.total_spin_atoms ||
      workspace.dipole_elements < layout.total_spin_atoms * kGfn2MullikenDipoleComponents ||
      workspace.quadrupole_elements < layout.total_spin_atoms * kGfn2MullikenQuadrupoleComponents ||
      workspace.sequence_elements < 1 || !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atom_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_to_atom, alignof(std::int64_t)) ||
      !is_aligned(batch.reference_shell_occupations, alignof(double)) ||
      !is_aligned(layout.spin_channel_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(layout.spin_channels, alignof(std::int32_t)) ||
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
  if (batch.maximum_system_atoms > kMaximumInt64 / 2 ||
      batch.maximum_system_shells > kMaximumInt64 / 2) {
    return cudaErrorInvalidConfiguration;
  }
  const std::int64_t atom_span = 2 * batch.maximum_system_atoms;
  const std::int64_t shell_span = 2 * batch.maximum_system_shells;
  if (atom_span > kMaximumInt64 / batch.batch_size ||
      shell_span > kMaximumInt64 / batch.batch_size) {
    return cudaErrorInvalidConfiguration;
  }
  const std::int64_t atom_blocks = atom_span * batch.batch_size;
  const std::int64_t shell_blocks = shell_span * batch.batch_size;
  if (atom_blocks > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      shell_blocks > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    return cudaErrorInvalidConfiguration;
  }

  std::array<MemoryRange, 28> ranges;
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
      !make_range(layout.spin_channel_offsets, layout.spin_channel_offset_count,
                  sizeof(*layout.spin_channel_offsets), &ranges[8]) ||
      !make_range(layout.spin_matrix_offsets, layout.spin_matrix_offset_count,
                  sizeof(*layout.spin_matrix_offsets), &ranges[9]) ||
      !make_range(layout.spin_shell_offsets, layout.spin_shell_offset_count,
                  sizeof(*layout.spin_shell_offsets), &ranges[10]) ||
      !make_range(layout.spin_atom_offsets, layout.spin_atom_offset_count,
                  sizeof(*layout.spin_atom_offsets), &ranges[11]) ||
      !make_range(layout.spin_channels, layout.spin_channel_count, sizeof(*layout.spin_channels),
                  &ranges[12]) ||
      !make_range(input.density, input.density_elements, sizeof(*input.density), &ranges[13]) ||
      !make_range(input.overlap, input.overlap_elements, sizeof(*input.overlap), &ranges[14]) ||
      !make_range(input.dipole_integrals, input.dipole_integral_elements,
                  sizeof(*input.dipole_integrals), &ranges[15]) ||
      !make_range(input.quadrupole_integrals, input.quadrupole_integral_elements,
                  sizeof(*input.quadrupole_integrals), &ranges[16]) ||
      !make_range(activity.active_mask, activity.elements, sizeof(*activity.active_mask),
                  &ranges[17]) ||
      !make_range(population.qsh, population.qsh_elements, sizeof(*population.qsh), &ranges[18]) ||
      !make_range(population.qat, population.qat_elements, sizeof(*population.qat), &ranges[19]) ||
      !make_range(population.dipole, population.dipole_elements, sizeof(*population.dipole),
                  &ranges[20]) ||
      !make_range(population.quadrupole, population.quadrupole_elements,
                  sizeof(*population.quadrupole), &ranges[21]) ||
      !make_range(workspace.qsh_scratch, workspace.qsh_elements, sizeof(*workspace.qsh_scratch),
                  &ranges[22]) ||
      !make_range(workspace.qat_scratch, workspace.qat_elements, sizeof(*workspace.qat_scratch),
                  &ranges[23]) ||
      !make_range(workspace.dipole_scratch, workspace.dipole_elements,
                  sizeof(*workspace.dipole_scratch), &ranges[24]) ||
      !make_range(workspace.quadrupole_scratch, workspace.quadrupole_elements,
                  sizeof(*workspace.quadrupole_scratch), &ranges[25]) ||
      !make_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active), &ranges[26]) ||
      !make_range(system_errors, batch.batch_size, sizeof(*system_errors), &ranges[27]) ||
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
  spin_topology_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                                   stream>>>(batch, layout, activity, workspace, system_errors,
                                             device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  spin_shell_population_kernel<<<static_cast<unsigned int>(shell_blocks), kThreadsPerBlock, 0,
                                 stream>>>(batch, layout, input, activity, workspace, system_errors,
                                           device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  spin_multipole_population_kernel<<<static_cast<unsigned int>(atom_blocks), kThreadsPerBlock, 0,
                                     stream>>>(batch, layout, input, activity, workspace,
                                               system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  spin_conversion_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                           stream>>>(batch, layout, activity, workspace, system_errors,
                                     device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  spin_atom_population_kernel<<<static_cast<unsigned int>(atom_blocks), kThreadsPerBlock, 0,
                                stream>>>(batch, layout, activity, workspace, system_errors,
                                          device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_spin_population_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                                   stream>>>(batch, layout, activity, population, workspace,
                                             system_errors);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
