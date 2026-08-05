#include <array>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kDipoleComponents = 3;
constexpr std::int64_t kQuadrupoleComponents = 6;
constexpr std::int64_t kMaximumInt64 = 9223372036854775807LL;

static_assert((kThreadsPerBlock & (kThreadsPerBlock - 1)) == 0,
              "SCC residual reduction requires a power-of-two block size");

__device__ void record_error(std::uint32_t* device_error, Gfn2SccDeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2SccDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

/* Device offsets are validated before any system kernel indexes a field. */
__global__ void topology_preflight_kernel(Gfn2SccDeviceBatch batch, std::uint32_t* device_error) {
  if (atomicAdd(device_error, 0u) != static_cast<std::uint32_t>(Gfn2SccDeviceError::kSuccess)) {
    return;
  }
  if (threadIdx.x == 0 && (batch.shell_offsets[0] != 0 || batch.atom_offsets[0] != 0 ||
                           batch.shell_offsets[batch.batch_size] != batch.total_shells ||
                           batch.atom_offsets[batch.batch_size] != batch.total_atoms)) {
    record_error(device_error, Gfn2SccDeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t shell_begin = batch.shell_offsets[system];
    const std::int64_t shell_end = batch.shell_offsets[system + 1];
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    if (shell_begin < 0 || shell_begin >= shell_end || shell_end > batch.total_shells ||
        atom_begin < 0 || atom_begin >= atom_end || atom_end > batch.total_atoms) {
      record_error(device_error, Gfn2SccDeviceError::kInvalidOffsets);
      continue;
    }
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t atoms = atom_end - atom_begin;
    if (atoms > (kMaximumInt64 - shells) / (kDipoleComponents + kQuadrupoleComponents)) {
      record_error(device_error, Gfn2SccDeviceError::kInvalidOffsets);
    }
  }
}

/*
 * Snapshot upstream/topology validity before any per-system numerical failure
 * can set the same sticky scalar. This is what preserves peer isolation when
 * blocks are scheduled at different times.
 */
__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                               static_cast<std::uint32_t>(Gfn2SccDeviceError::kSuccess)
                           ? 1u
                           : 0u;
  }
}

__device__ void accumulate_field_residual(const double* current, const double* next_mixed,
                                          const double* raw, std::int64_t begin, std::int64_t end,
                                          double* local_square, int* valid,
                                          std::uint32_t* device_error) {
  for (std::int64_t element = begin + threadIdx.x; element < end; element += blockDim.x) {
    const double current_value = current[element];
    const double mixed_value = next_mixed[element];
    const double raw_value = raw[element];
    if (!isfinite(current_value)) {
      record_error(device_error, Gfn2SccDeviceError::kNonfiniteCurrentMultipole);
      atomicExch(valid, 0);
      continue;
    }
    if (!isfinite(mixed_value)) {
      record_error(device_error, Gfn2SccDeviceError::kNonfiniteMixedMultipole);
      atomicExch(valid, 0);
      continue;
    }
    if (!isfinite(raw_value)) {
      record_error(device_error, Gfn2SccDeviceError::kNonfiniteRawMultipole);
      atomicExch(valid, 0);
      continue;
    }
    const double residual = raw_value - current_value;
    const double square = residual * residual;
    const double updated = *local_square + square;
    if (!isfinite(residual) || !isfinite(square) || !isfinite(updated)) {
      record_error(device_error, Gfn2SccDeviceError::kNonfiniteResidual);
      atomicExch(valid, 0);
      continue;
    }
    *local_square = updated;
  }
}

__device__ void publish_field(const double* next_mixed, const double* raw, double* current,
                              double* published, std::int64_t begin, std::int64_t end,
                              bool converged) {
  for (std::int64_t element = begin + threadIdx.x; element < end; element += blockDim.x) {
    const double mixed_value = next_mixed[element];
    current[element] = mixed_value;
    published[element] = converged ? raw[element] : mixed_value;
  }
}

__device__ bool known_system_status(gpuxtb_status_t status) {
  return status >= GPUXTB_STATUS_SUCCESS && status <= GPUXTB_STATUS_EIGENSOLVER_FAILED;
}

__device__ double quiet_nan() {
  return __longlong_as_double(static_cast<long long>(0x7ff8000000000000ULL));
}

/* Match the CPU driver's accounting for a failed active SCC attempt. */
__device__ void commit_numeric_failure(Gfn2SccDeviceState state, std::int64_t system) {
  const double nan = quiet_nan();
  state.free_energies[system] = nan;
  state.previous_free_energies[system] = nan;
  state.free_energy_changes[system] = nan;
  state.residual_rms[system] = nan;
  ++state.iterations[system];
  state.system_statuses[system] = GPUXTB_STATUS_INTERNAL_ERROR;
}

__global__ void update_state_kernel(Gfn2SccDeviceBatch batch, Gfn2SccDevicePolicy policy,
                                    Gfn2SccDeviceConstMultipoles next_mixed,
                                    Gfn2SccDeviceConstMultipoles raw,
                                    const double* complete_free_energies,
                                    Gfn2SccDeviceMultipoles published, Gfn2SccDeviceState state,
                                    Gfn2SccDeviceWorkspace workspace, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  __shared__ int valid;
  __shared__ double partial_square[kThreadsPerBlock];
  __shared__ double old_energy;
  __shared__ double new_energy;
  __shared__ double energy_delta;
  __shared__ double system_residual_rms;
  __shared__ std::uint64_t new_iteration;
  __shared__ int system_converged;
  __shared__ gpuxtb_status_t new_status;

  if (threadIdx.x == 0) {
    active = 0;
    valid = 1;
    if (atomicAdd(workspace.sequence_active, 0u) == 1u) {
      const gpuxtb_status_t status = state.system_statuses[system];
      const std::uint8_t converged = state.converged[system];
      const std::uint64_t iteration = state.iterations[system];
      if (!known_system_status(status)) {
        record_error(device_error, Gfn2SccDeviceError::kInvalidState);
        state.system_statuses[system] = GPUXTB_STATUS_INTERNAL_ERROR;
        valid = 0;
      } else if (status == GPUXTB_STATUS_SUCCESS) {
        if (converged > 1u) {
          record_error(device_error, Gfn2SccDeviceError::kInvalidState);
          state.system_statuses[system] = GPUXTB_STATUS_INTERNAL_ERROR;
          valid = 0;
        } else if (converged == 0u && iteration < policy.maximum_iterations) {
          active = 1;
        }
      }
    }
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const std::int64_t shell_begin = batch.shell_offsets[system];
  const std::int64_t shell_end = batch.shell_offsets[system + 1];
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t dipole_begin = atom_begin * kDipoleComponents;
  const std::int64_t dipole_end = atom_end * kDipoleComponents;
  const std::int64_t quadrupole_begin = atom_begin * kQuadrupoleComponents;
  const std::int64_t quadrupole_end = atom_end * kQuadrupoleComponents;

  if (threadIdx.x == 0) {
    const std::uint64_t iteration = state.iterations[system];
    old_energy = iteration == 0u ? 0.0 : state.free_energies[system];
    new_energy = complete_free_energies[system];
    energy_delta = new_energy - old_energy;
    if (!isfinite(old_energy) || !isfinite(new_energy)) {
      record_error(device_error, Gfn2SccDeviceError::kNonfiniteFreeEnergy);
      valid = 0;
    } else if (!isfinite(energy_delta)) {
      record_error(device_error, Gfn2SccDeviceError::kNonfiniteEnergyDelta);
      valid = 0;
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      commit_numeric_failure(state, system);
    }
    return;
  }

  double local_square = 0.0;
  accumulate_field_residual(state.current_inputs.shell_charges, next_mixed.shell_charges,
                            raw.shell_charges, shell_begin, shell_end, &local_square, &valid,
                            device_error);
  accumulate_field_residual(state.current_inputs.atomic_dipoles, next_mixed.atomic_dipoles,
                            raw.atomic_dipoles, dipole_begin, dipole_end, &local_square, &valid,
                            device_error);
  accumulate_field_residual(state.current_inputs.atomic_quadrupoles, next_mixed.atomic_quadrupoles,
                            raw.atomic_quadrupoles, quadrupole_begin, quadrupole_end, &local_square,
                            &valid, device_error);
  partial_square[threadIdx.x] = local_square;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      const double updated = partial_square[threadIdx.x] + partial_square[threadIdx.x + offset];
      if (!isfinite(updated)) {
        record_error(device_error, Gfn2SccDeviceError::kNonfiniteResidual);
        atomicExch(&valid, 0);
      } else {
        partial_square[threadIdx.x] = updated;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && valid != 0) {
    const std::int64_t dimension =
        (shell_end - shell_begin) +
        (atom_end - atom_begin) * (kDipoleComponents + kQuadrupoleComponents);
    system_residual_rms = sqrt(partial_square[0]) / sqrt(static_cast<double>(dimension));
    if (!isfinite(system_residual_rms)) {
      record_error(device_error, Gfn2SccDeviceError::kNonfiniteResidual);
      valid = 0;
    } else {
      new_iteration = state.iterations[system] + 1u;
      system_converged = system_residual_rms < policy.residual_rms_tolerance &&
                                 fabs(energy_delta) < policy.energy_tolerance
                             ? 1
                             : 0;
      new_status = system_converged == 0 && new_iteration >= policy.maximum_iterations
                       ? GPUXTB_STATUS_SCC_NOT_CONVERGED
                       : GPUXTB_STATUS_SUCCESS;
    }
  }
  __syncthreads();
  if (valid == 0) {
    if (threadIdx.x == 0) {
      commit_numeric_failure(state, system);
    }
    return;
  }

  const bool converged = system_converged != 0;
  publish_field(next_mixed.shell_charges, raw.shell_charges, state.current_inputs.shell_charges,
                published.shell_charges, shell_begin, shell_end, converged);
  publish_field(next_mixed.atomic_dipoles, raw.atomic_dipoles, state.current_inputs.atomic_dipoles,
                published.atomic_dipoles, dipole_begin, dipole_end, converged);
  publish_field(next_mixed.atomic_quadrupoles, raw.atomic_quadrupoles,
                state.current_inputs.atomic_quadrupoles, published.atomic_quadrupoles,
                quadrupole_begin, quadrupole_end, converged);
  __syncthreads();

  if (threadIdx.x == 0) {
    state.previous_free_energies[system] = old_energy;
    state.free_energies[system] = new_energy;
    state.free_energy_changes[system] = energy_delta;
    state.residual_rms[system] = system_residual_rms;
    state.iterations[system] = new_iteration;
    state.converged[system] = converged ? 1u : 0u;
    state.system_statuses[system] = new_status;
  }
}

bool checked_multiply(std::int64_t value, std::int64_t factor, std::int64_t* result) noexcept {
  if (value < 0 || factor < 0 ||
      (value != 0 && factor > std::numeric_limits<std::int64_t>::max() / value)) {
    return false;
  }
  *result = value * factor;
  return true;
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_address_range(const void* pointer, std::int64_t elements, std::size_t element_size,
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

bool ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

bool same_range(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin == second.begin && first.end == second.end;
}

template <std::size_t Count>
bool pairwise_disjoint(const std::array<AddressRange, Count>& ranges) noexcept {
  for (std::size_t first = 0u; first < Count; ++first) {
    for (std::size_t second = first + 1u; second < Count; ++second) {
      if (ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

bool valid_const_multipoles(const Gfn2SccDeviceConstMultipoles& view, std::int64_t shell_elements,
                            std::int64_t dipole_elements, std::int64_t quadrupole_elements,
                            std::uint64_t token) noexcept {
  return view.plan_token == token && view.shell_elements == shell_elements &&
         view.dipole_elements == dipole_elements &&
         view.quadrupole_elements == quadrupole_elements &&
         is_aligned(view.shell_charges, alignof(double)) &&
         is_aligned(view.atomic_dipoles, alignof(double)) &&
         is_aligned(view.atomic_quadrupoles, alignof(double));
}

bool valid_multipoles(const Gfn2SccDeviceMultipoles& view, std::int64_t shell_elements,
                      std::int64_t dipole_elements, std::int64_t quadrupole_elements,
                      std::uint64_t token) noexcept {
  return view.plan_token == token && view.shell_elements == shell_elements &&
         view.dipole_elements == dipole_elements &&
         view.quadrupole_elements == quadrupole_elements &&
         is_aligned(view.shell_charges, alignof(double)) &&
         is_aligned(view.atomic_dipoles, alignof(double)) &&
         is_aligned(view.atomic_quadrupoles, alignof(double));
}

}  // namespace

cudaError_t reset_gfn2_scc_device_error_cuda(std::uint32_t* device_error,
                                             cudaStream_t stream) noexcept {
  AddressRange range;
  if (!is_aligned(device_error, alignof(std::uint32_t)) ||
      !make_address_range(device_error, 1, sizeof(*device_error), &range)) {
    return cudaErrorInvalidValue;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t update_gfn2_scc_state_cuda(
    const Gfn2SccDeviceBatch& batch, const Gfn2SccDevicePolicy& policy,
    const Gfn2SccDeviceConstMultipoles& next_mixed, const Gfn2SccDeviceConstMultipoles& raw,
    const double* complete_free_energies, const Gfn2SccDeviceMultipoles& published,
    const Gfn2SccDeviceState& state, const Gfn2SccDeviceWorkspace& workspace,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  std::int64_t dipole_elements = 0;
  std::int64_t quadrupole_elements = 0;
  const bool dimensions_valid =
      checked_multiply(batch.total_atoms, kDipoleComponents, &dipole_elements) &&
      checked_multiply(batch.total_atoms, kQuadrupoleComponents, &quadrupole_elements);
  if (batch.batch_size <= 0 || batch.total_shells <= 0 || batch.total_atoms <= 0 ||
      batch.batch_size == std::numeric_limits<std::int64_t>::max() ||
      batch.batch_size > std::numeric_limits<int>::max() ||
      batch.shell_offset_count != batch.batch_size + 1 ||
      batch.atom_offset_count != batch.batch_size + 1 || batch.plan_token == 0u ||
      !dimensions_valid || !is_aligned(batch.shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      policy.plan_token != batch.plan_token || policy.maximum_iterations == 0u ||
      !std::isfinite(policy.residual_rms_tolerance) || !(policy.residual_rms_tolerance > 0.0) ||
      !std::isfinite(policy.energy_tolerance) || !(policy.energy_tolerance > 0.0) ||
      !valid_const_multipoles(next_mixed, batch.total_shells, dipole_elements, quadrupole_elements,
                              batch.plan_token) ||
      !valid_const_multipoles(raw, batch.total_shells, dipole_elements, quadrupole_elements,
                              batch.plan_token) ||
      !valid_multipoles(published, batch.total_shells, dipole_elements, quadrupole_elements,
                        batch.plan_token) ||
      state.plan_token != batch.plan_token || state.batch_elements != batch.batch_size ||
      !valid_multipoles(state.current_inputs, batch.total_shells, dipole_elements,
                        quadrupole_elements, batch.plan_token) ||
      !is_aligned(complete_free_energies, alignof(double)) ||
      !is_aligned(state.free_energies, alignof(double)) ||
      !is_aligned(state.previous_free_energies, alignof(double)) ||
      !is_aligned(state.free_energy_changes, alignof(double)) ||
      !is_aligned(state.residual_rms, alignof(double)) ||
      !is_aligned(state.iterations, alignof(std::uint64_t)) ||
      !is_aligned(state.system_statuses, alignof(gpuxtb_status_t)) ||
      !is_aligned(state.converged, alignof(std::uint8_t)) ||
      workspace.plan_token != batch.plan_token || workspace.elements != 1 ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }

  std::array<AddressRange, 9> reads;
  std::array<AddressRange, 15> writes;
  if (!make_address_range(batch.shell_offsets, batch.shell_offset_count,
                          sizeof(*batch.shell_offsets), &reads[0]) ||
      !make_address_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                          &reads[1]) ||
      !make_address_range(next_mixed.shell_charges, batch.total_shells, sizeof(double),
                          &reads[2]) ||
      !make_address_range(next_mixed.atomic_dipoles, dipole_elements, sizeof(double), &reads[3]) ||
      !make_address_range(next_mixed.atomic_quadrupoles, quadrupole_elements, sizeof(double),
                          &reads[4]) ||
      !make_address_range(raw.shell_charges, batch.total_shells, sizeof(double), &reads[5]) ||
      !make_address_range(raw.atomic_dipoles, dipole_elements, sizeof(double), &reads[6]) ||
      !make_address_range(raw.atomic_quadrupoles, quadrupole_elements, sizeof(double), &reads[7]) ||
      !make_address_range(complete_free_energies, batch.batch_size, sizeof(double), &reads[8]) ||
      !make_address_range(published.shell_charges, batch.total_shells, sizeof(double),
                          &writes[0]) ||
      !make_address_range(published.atomic_dipoles, dipole_elements, sizeof(double), &writes[1]) ||
      !make_address_range(published.atomic_quadrupoles, quadrupole_elements, sizeof(double),
                          &writes[2]) ||
      !make_address_range(state.current_inputs.shell_charges, batch.total_shells, sizeof(double),
                          &writes[3]) ||
      !make_address_range(state.current_inputs.atomic_dipoles, dipole_elements, sizeof(double),
                          &writes[4]) ||
      !make_address_range(state.current_inputs.atomic_quadrupoles, quadrupole_elements,
                          sizeof(double), &writes[5]) ||
      !make_address_range(state.free_energies, batch.batch_size, sizeof(double), &writes[6]) ||
      !make_address_range(state.previous_free_energies, batch.batch_size, sizeof(double),
                          &writes[7]) ||
      !make_address_range(state.free_energy_changes, batch.batch_size, sizeof(double),
                          &writes[8]) ||
      !make_address_range(state.residual_rms, batch.batch_size, sizeof(double), &writes[9]) ||
      !make_address_range(state.iterations, batch.batch_size, sizeof(std::uint64_t), &writes[10]) ||
      !make_address_range(state.system_statuses, batch.batch_size, sizeof(gpuxtb_status_t),
                          &writes[11]) ||
      !make_address_range(state.converged, batch.batch_size, sizeof(std::uint8_t), &writes[12]) ||
      !make_address_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[13]) ||
      !make_address_range(device_error, 1, sizeof(std::uint32_t), &writes[14]) ||
      !pairwise_disjoint(writes)) {
    return cudaErrorInvalidValue;
  }

  for (std::size_t write = 0u; write < writes.size(); ++write) {
    for (std::size_t read = 0u; read < reads.size(); ++read) {
      const bool allowed_in_place =
          write < 3u && read == write + 2u && same_range(writes[write], reads[read]);
      if (ranges_overlap(writes[write], reads[read]) && !allowed_in_place) {
        return cudaErrorInvalidValue;
      }
    }
  }

  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, device_error);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  update_state_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, policy, next_mixed, raw, complete_free_energies, published, state, workspace,
      device_error);
  return cudaPeekAtLastError();
}

}  // namespace gpuxtb::detail::cuda
