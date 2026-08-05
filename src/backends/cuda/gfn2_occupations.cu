#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_occupations.cuh"
#include "model/gfn2/occupation_binary64_policy.hpp"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kPublishThreads = 128;
namespace occupation_policy = gpuxtb::detail::gfn2::binary64_policy;

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kSuccess);
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2OccupationsDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system,
                static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kSuccess),
                code) == static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kSuccess)) {
    atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kSuccess), code);
  }
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                               static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kSuccess)
                           ? 1u
                           : 0u;
  }
}

__device__ double actual_electron_sum(const double* occupations, std::int64_t count) {
  double sum = 0.0;
  double compensation = 0.0;
  for (std::int64_t orbital = 0; orbital < count; ++orbital) {
    const double corrected = occupations[orbital] - compensation;
    const double updated = sum + corrected;
    compensation = (updated - sum) - corrected;
    sum = updated;
  }
  return sum;
}

struct SpinResult {
  double chemical_potential;
  double electron_sum;
  double entropy;
};

__device__ bool fill_one_spin(const double* eigenvalues, std::int64_t count, double electron_count,
                              double temperature, double* occupations, SpinResult* result,
                              Gfn2OccupationsDeviceError* error) {
  result->chemical_potential = 0.0;
  result->electron_sum = 0.0;
  result->entropy = 0.0;
  if (temperature == 0.0) {
    const std::int64_t full = min(static_cast<std::int64_t>(floor(electron_count)), count);
    for (std::int64_t orbital = 0; orbital < count; ++orbital) {
      occupations[orbital] = orbital < full ? 1.0 : 0.0;
    }
    if (full < count) {
      occupations[full] = electron_count - static_cast<double>(full);
    }
    result->electron_sum = actual_electron_sum(occupations, count);
    return true;
  }
  if (electron_count == 0.0) {
    for (std::int64_t orbital = 0; orbital < count; ++orbital) {
      occupations[orbital] = 0.0;
    }
    return true;
  }

  const double capacity = static_cast<double>(count);
  if (electron_count == capacity) {
    for (std::int64_t orbital = 0; orbital < count; ++orbital) {
      occupations[orbital] = 1.0;
    }
    result->chemical_potential =
        occupation_policy::saturated_affine(eigenvalues[count - 1], 50.0, temperature);
    result->electron_sum = capacity;
    return isfinite(result->chemical_potential);
  }

  const bool solve_holes = electron_count > 0.5 * capacity;
  const double quantity_target = solve_holes ? capacity - electron_count : electron_count;
  if (!(quantity_target > 0.0) || !isfinite(quantity_target)) {
    *error = Gfn2OccupationsDeviceError::kElectronConservationFailure;
    return false;
  }
  occupation_policy::Root root{};
  if (!occupation_policy::solve_root(eigenvalues, count, quantity_target, temperature, solve_holes,
                                     root)) {
    *error = Gfn2OccupationsDeviceError::kChemicalPotentialBracketFailure;
    return false;
  }
  /*
   * The shared binary64 solver retries only after adjacent root spacing is
   * exhausted, the ordinary root tolerance is still unmet, the final bracket
   * straddles the target, and exactly one multi-member degenerate block changes
   * occupation across it. The retry merely changes the translated reference;
   * this ordinary acceptance gate still decides whether the root is usable.
   */
  occupation_policy::Publication publication{};
  if (!occupation_policy::select_publication(eigenvalues, count, quantity_target, temperature,
                                             solve_holes, root, publication)) {
    *error = Gfn2OccupationsDeviceError::kElectronConservationFailure;
    return false;
  }
  const double root_tolerance = occupation_policy::root_acceptance_tolerance(quantity_target, root);
  if (!isfinite(root.quantity) ||
      occupation_policy::absolute(root.quantity - quantity_target) > root_tolerance) {
    *error = Gfn2OccupationsDeviceError::kElectronConservationFailure;
    return false;
  }
  result->chemical_potential =
      occupation_policy::saturated_affine(root.energy_reference, root.scaled_mu, temperature);
  const std::int64_t largest_degenerate_block =
      occupation_policy::largest_degenerate_block(eigenvalues, count);
  if (largest_degenerate_block == count && quantity_target / capacity == 0.0) {
    /* Match the CPU's canonical analytic root when a valid subnormal total
     * population cannot be divided into a nonzero binary64 per-member value.
     * This avoids choosing an arbitrary side of the device Fermi jump. */
    const double log_fraction =
        occupation_policy::logarithm(quantity_target) - occupation_policy::logarithm(capacity);
    const double log_complement =
        occupation_policy::logarithm_one_plus(-occupation_policy::exponential(log_fraction));
    const double canonical_scaled_mu =
        solve_holes ? log_complement - log_fraction : log_fraction - log_complement;
    result->chemical_potential = occupation_policy::saturated_affine(
        root.energy_reference, canonical_scaled_mu, temperature);
  }
  for (std::int64_t orbital = 0; orbital < count; ++orbital) {
    occupations[orbital] = occupation_policy::published_occupation(eigenvalues, orbital,
                                                                   temperature, root, publication);
  }
  const double entropy =
      occupation_policy::publication_entropy(eigenvalues, count, temperature, root, publication);
  if (!isfinite(result->chemical_potential) || !isfinite(entropy)) {
    *error = Gfn2OccupationsDeviceError::kNonfiniteEntropy;
    return false;
  }
  result->electron_sum = actual_electron_sum(occupations, count);
  result->entropy = entropy;
  return isfinite(result->electron_sum);
}

__global__ void evaluate_kernel(Gfn2OccupationsDeviceBatch batch, Gfn2WavefunctionLayoutView layout,
                                const double* eigenvalues, Gfn2OccupationsDeviceWorkspace workspace,
                                std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || atomicAdd(workspace.sequence_active, 0u) == 0u ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::uint8_t active = batch.active[system];
  if (active == 0u) {
    return;
  }
  if (active != 1u) {
    record_system_error(system_errors, system, device_error,
                        Gfn2OccupationsDeviceError::kInvalidActiveMask);
    return;
  }

  const std::int64_t begin = batch.orbital_offsets[system];
  const std::int64_t end = batch.orbital_offsets[system + 1];
  if (begin < 0 || begin >= end || end > batch.total_orbitals || (system == 0 && begin != 0) ||
      (system + 1 == batch.batch_size && end != batch.total_orbitals)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2OccupationsDeviceError::kInvalidOffsets);
    return;
  }
  const std::int64_t count = end - begin;
  const bool spin_layout = layout.spin_channels != nullptr;
  std::int64_t spin_orbital_begin = begin;
  std::uint8_t spin_channels = 1u;
  if (spin_layout) {
    const std::int32_t configured_channels = layout.spin_channels[system];
    if (configured_channels != 1 && configured_channels != 2) {
      record_system_error(system_errors, system, device_error,
                          Gfn2OccupationsDeviceError::kInvalidSpinLayout);
      return;
    }
    spin_channels = static_cast<std::uint8_t>(configured_channels);
    const std::int64_t channel_begin = layout.spin_channel_offsets[system];
    const std::int64_t channel_end = layout.spin_channel_offsets[system + 1];
    spin_orbital_begin = layout.spin_orbital_offsets[system];
    const std::int64_t spin_orbital_end = layout.spin_orbital_offsets[system + 1];
    if (channel_begin < 0 || channel_end - channel_begin != spin_channels ||
        channel_end > layout.total_spin_channels || spin_orbital_begin < 0 ||
        spin_orbital_end - spin_orbital_begin != static_cast<std::int64_t>(spin_channels) * count ||
        spin_orbital_end > layout.total_spin_orbitals ||
        (system == 0 && (channel_begin != 0 || spin_orbital_begin != 0)) ||
        (system + 1 == batch.batch_size && (channel_end != layout.total_spin_channels ||
                                            spin_orbital_end != layout.total_spin_orbitals))) {
      record_system_error(system_errors, system, device_error,
                          Gfn2OccupationsDeviceError::kInvalidSpinLayout);
      return;
    }
  }
  const double temperature = batch.temperatures[system];
  if (!(temperature >= 0.0) || !isfinite(temperature)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2OccupationsDeviceError::kInvalidTemperature);
    return;
  }
  for (int spin = 0; spin < 2; ++spin) {
    const double electron_count = batch.electron_counts[system * 2 + spin];
    if (!(electron_count >= 0.0) || electron_count > static_cast<double>(count) ||
        !isfinite(electron_count)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2OccupationsDeviceError::kInvalidElectronCount);
      return;
    }
  }
  for (std::uint8_t spin = 0u; spin < spin_channels; ++spin) {
    const std::int64_t spectrum_begin =
        spin_orbital_begin + static_cast<std::int64_t>(spin) * count;
    for (std::int64_t orbital = 0; orbital < count; ++orbital) {
      const double eigenvalue = eigenvalues[spectrum_begin + orbital];
      if (!isfinite(eigenvalue)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2OccupationsDeviceError::kNonfiniteEigenvalue);
        return;
      }
      if (orbital != 0 && eigenvalue < eigenvalues[spectrum_begin + orbital - 1]) {
        record_system_error(system_errors, system, device_error,
                            Gfn2OccupationsDeviceError::kUnsortedEigenvalues);
        return;
      }
    }
  }

  const std::int64_t occupation_base = begin * 2;
  SpinResult spin_results[2];
  for (int spin = 0; spin < 2; ++spin) {
    const std::int64_t spectrum_begin =
        spin_orbital_begin + (spin_channels == 2u ? static_cast<std::int64_t>(spin) * count : 0);
    Gfn2OccupationsDeviceError error = Gfn2OccupationsDeviceError::kSuccess;
    if (!fill_one_spin(eigenvalues + spectrum_begin, count,
                       batch.electron_counts[system * 2 + spin], temperature,
                       workspace.occupation_scratch + occupation_base + spin * count,
                       spin_results + spin, &error)) {
      record_system_error(system_errors, system, device_error, error);
      return;
    }
  }
  const double total_entropy = spin_results[0].entropy + spin_results[1].entropy;
  if (!isfinite(total_entropy)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2OccupationsDeviceError::kNonfiniteEntropy);
    return;
  }
  for (int spin = 0; spin < 2; ++spin) {
    workspace.chemical_potential_scratch[system * 2 + spin] = spin_results[spin].chemical_potential;
    workspace.electron_sum_scratch[system * 2 + spin] = spin_results[spin].electron_sum;
  }
  workspace.entropy_scratch[system] = total_entropy;
}

/* Lowest active system index wins, making diagnostics reproducible across block schedules. */
__global__ void canonicalize_device_error_kernel(std::int64_t batch_size,
                                                 const std::uint32_t* sequence_active,
                                                 const std::uint32_t* system_errors,
                                                 std::uint32_t* device_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      atomicAdd(const_cast<std::uint32_t*>(sequence_active), 0u) == 0u) {
    return;
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::uint32_t error = atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u);
    if (error != static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kSuccess)) {
      atomicExch(device_error, error);
      return;
    }
  }
  atomicExch(device_error, static_cast<std::uint32_t>(Gfn2OccupationsDeviceError::kSuccess));
}

__global__ void publish_kernel(Gfn2OccupationsDeviceBatch batch,
                               Gfn2OccupationsDeviceResults results,
                               Gfn2OccupationsDeviceWorkspace workspace,
                               const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (atomicAdd(workspace.sequence_active, 0u) == 0u || batch.active[system] != 1u ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t begin = batch.orbital_offsets[system];
  const std::int64_t end = batch.orbital_offsets[system + 1];
  const std::int64_t occupation_begin = begin * 2;
  const std::int64_t occupation_end = end * 2;
  for (std::int64_t element = occupation_begin + threadIdx.x; element < occupation_end;
       element += blockDim.x) {
    results.occupations[element] = workspace.occupation_scratch[element];
  }
  if (threadIdx.x == 0) {
    results.chemical_potentials[system * 2] = workspace.chemical_potential_scratch[system * 2];
    results.chemical_potentials[system * 2 + 1] =
        workspace.chemical_potential_scratch[system * 2 + 1];
    results.electron_sums[system * 2] = workspace.electron_sum_scratch[system * 2];
    results.electron_sums[system * 2 + 1] = workspace.electron_sum_scratch[system * 2 + 1];
    results.entropies[system] = workspace.entropy_scratch[system];
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

template <std::size_t FirstCount, std::size_t SecondCount>
bool disjoint_sets(const std::array<AddressRange, FirstCount>& first,
                   const std::array<AddressRange, SecondCount>& second) noexcept {
  for (const AddressRange& first_range : first) {
    for (const AddressRange& second_range : second) {
      if (ranges_overlap(first_range, second_range)) {
        return false;
      }
    }
  }
  return true;
}

bool valid_bindings(const Gfn2OccupationsDeviceBatch& batch,
                    const Gfn2WavefunctionLayoutView* layout, const double* eigenvalues,
                    std::int64_t eigenvalue_elements, const Gfn2OccupationsDeviceResults& results,
                    const Gfn2OccupationsDeviceWorkspace& workspace, std::uint32_t* system_errors,
                    std::uint32_t* device_error) noexcept {
  std::int64_t two_batch = 0;
  std::int64_t two_orbitals = 0;
  const bool spin_layout = layout != nullptr;
  const std::int64_t spectrum_elements =
      spin_layout ? layout->total_spin_orbitals : batch.total_orbitals;
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<int>::max() ||
      batch.total_orbitals <= 0 || batch.plan_token == 0u ||
      !checked_multiply(batch.batch_size, 2, &two_batch) ||
      !checked_multiply(batch.total_orbitals, 2, &two_orbitals) ||
      batch.orbital_offset_count != batch.batch_size + 1 ||
      batch.electron_count_elements != two_batch ||
      batch.temperature_elements != batch.batch_size || batch.active_elements != batch.batch_size ||
      eigenvalue_elements != spectrum_elements || results.plan_token != batch.plan_token ||
      results.occupation_elements != two_orbitals ||
      results.chemical_potential_elements != two_batch ||
      results.electron_sum_elements != two_batch || results.entropy_elements != batch.batch_size ||
      workspace.plan_token != batch.plan_token || workspace.occupation_elements != two_orbitals ||
      workspace.chemical_potential_elements != two_batch ||
      workspace.electron_sum_elements != two_batch ||
      workspace.entropy_elements != batch.batch_size || workspace.sequence_active_elements != 1 ||
      !is_aligned(batch.orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.electron_counts, alignof(double)) ||
      !is_aligned(batch.temperatures, alignof(double)) || batch.active == nullptr ||
      !is_aligned(eigenvalues, alignof(double)) ||
      !is_aligned(results.occupations, alignof(double)) ||
      !is_aligned(results.chemical_potentials, alignof(double)) ||
      !is_aligned(results.electron_sums, alignof(double)) ||
      !is_aligned(results.entropies, alignof(double)) ||
      !is_aligned(workspace.occupation_scratch, alignof(double)) ||
      !is_aligned(workspace.chemical_potential_scratch, alignof(double)) ||
      !is_aligned(workspace.electron_sum_scratch, alignof(double)) ||
      !is_aligned(workspace.entropy_scratch, alignof(double)) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      (spin_layout &&
       (layout->memory_space != Gfn2PlanMemorySpace::kCudaDevice ||
        layout->plan_token != batch.plan_token || layout->batch_size != batch.batch_size ||
        layout->total_spin_channels < batch.batch_size || layout->total_spin_channels > two_batch ||
        layout->total_spin_orbitals < batch.total_orbitals ||
        layout->total_spin_orbitals > two_orbitals ||
        layout->spin_channel_count != batch.batch_size ||
        layout->spin_channel_offset_count != batch.batch_size + 1 ||
        layout->spin_orbital_offset_count != batch.batch_size + 1 ||
        !is_aligned(layout->spin_channels, alignof(std::int32_t)) ||
        !is_aligned(layout->spin_channel_offsets, alignof(std::int64_t)) ||
        !is_aligned(layout->spin_orbital_offsets, alignof(std::int64_t))))) {
    return false;
  }

  std::array<AddressRange, 8> inputs{};
  std::array<AddressRange, 11> writable{};
  if (!make_address_range(batch.orbital_offsets, batch.orbital_offset_count, sizeof(std::int64_t),
                          &inputs[0]) ||
      !make_address_range(batch.electron_counts, two_batch, sizeof(double), &inputs[1]) ||
      !make_address_range(batch.temperatures, batch.batch_size, sizeof(double), &inputs[2]) ||
      !make_address_range(batch.active, batch.batch_size, sizeof(std::uint8_t), &inputs[3]) ||
      !make_address_range(eigenvalues, spectrum_elements, sizeof(double), &inputs[4]) ||
      !make_address_range(spin_layout ? layout->spin_channels : nullptr,
                          spin_layout ? batch.batch_size : 0, sizeof(std::int32_t), &inputs[5]) ||
      !make_address_range(spin_layout ? layout->spin_channel_offsets : nullptr,
                          spin_layout ? batch.batch_size + 1 : 0, sizeof(std::int64_t),
                          &inputs[6]) ||
      !make_address_range(spin_layout ? layout->spin_orbital_offsets : nullptr,
                          spin_layout ? batch.batch_size + 1 : 0, sizeof(std::int64_t),
                          &inputs[7]) ||
      !make_address_range(results.occupations, two_orbitals, sizeof(double), &writable[0]) ||
      !make_address_range(results.chemical_potentials, two_batch, sizeof(double), &writable[1]) ||
      !make_address_range(results.electron_sums, two_batch, sizeof(double), &writable[2]) ||
      !make_address_range(results.entropies, batch.batch_size, sizeof(double), &writable[3]) ||
      !make_address_range(workspace.occupation_scratch, two_orbitals, sizeof(double),
                          &writable[4]) ||
      !make_address_range(workspace.chemical_potential_scratch, two_batch, sizeof(double),
                          &writable[5]) ||
      !make_address_range(workspace.electron_sum_scratch, two_batch, sizeof(double),
                          &writable[6]) ||
      !make_address_range(workspace.entropy_scratch, batch.batch_size, sizeof(double),
                          &writable[7]) ||
      !make_address_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writable[8]) ||
      !make_address_range(system_errors, batch.batch_size, sizeof(std::uint32_t), &writable[9]) ||
      !make_address_range(device_error, 1, sizeof(std::uint32_t), &writable[10]) ||
      !pairwise_disjoint(inputs) || !pairwise_disjoint(writable) ||
      !disjoint_sets(inputs, writable)) {
    return false;
  }
  return true;
}

}  // namespace

cudaError_t reset_gfn2_occupations_device_errors_cuda(std::int64_t batch_size,
                                                      std::uint32_t* system_errors,
                                                      std::uint32_t* device_error,
                                                      cudaStream_t stream) noexcept {
  if (batch_size <= 0 || !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  AddressRange system_range;
  AddressRange device_range;
  if (!make_address_range(system_errors, batch_size, sizeof(std::uint32_t), &system_range) ||
      !make_address_range(device_error, 1, sizeof(std::uint32_t), &device_range) ||
      ranges_overlap(system_range, device_range)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(std::uint32_t), stream);
  return status == cudaSuccess ? cudaMemsetAsync(device_error, 0, sizeof(std::uint32_t), stream)
                               : status;
}

cudaError_t evaluate_gfn2_restricted_occupations_cuda(
    const Gfn2OccupationsDeviceBatch& batch, const double* eigenvalues,
    std::int64_t eigenvalue_elements, const Gfn2OccupationsDeviceResults& results,
    const Gfn2OccupationsDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!valid_bindings(batch, nullptr, eigenvalues, eigenvalue_elements, results, workspace,
                      system_errors, device_error)) {
    return cudaErrorInvalidValue;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  evaluate_kernel<<<static_cast<unsigned int>(batch.batch_size), 1, 0, stream>>>(
      batch, {}, eigenvalues, workspace, system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  canonicalize_device_error_kernel<<<1, 1, 0, stream>>>(batch.batch_size, workspace.sequence_active,
                                                        system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_kernel<<<static_cast<unsigned int>(batch.batch_size), kPublishThreads, 0, stream>>>(
      batch, results, workspace, system_errors);
  return cudaGetLastError();
}

cudaError_t evaluate_gfn2_occupations_cuda(
    const Gfn2OccupationsDeviceBatch& batch, const Gfn2WavefunctionLayoutView& layout,
    const double* eigenvalues, std::int64_t eigenvalue_elements,
    const Gfn2OccupationsDeviceResults& results, const Gfn2OccupationsDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!valid_bindings(batch, &layout, eigenvalues, eigenvalue_elements, results, workspace,
                      system_errors, device_error)) {
    return cudaErrorInvalidValue;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace.sequence_active);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  evaluate_kernel<<<static_cast<unsigned int>(batch.batch_size), 1, 0, stream>>>(
      batch, layout, eigenvalues, workspace, system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  canonicalize_device_error_kernel<<<1, 1, 0, stream>>>(batch.batch_size, workspace.sequence_active,
                                                        system_errors, device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_kernel<<<static_cast<unsigned int>(batch.batch_size), kPublishThreads, 0, stream>>>(
      batch, results, workspace, system_errors);
  return cudaGetLastError();
}

}  // namespace gpuxtb::detail::cuda
