#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_d4.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr double kChargeScalingHeight = 3.0;
constexpr double kChargeScalingSteepness = 2.0;
constexpr double kReferenceWeightFactor = 6.0;
constexpr double kMinimumWeightNorm = 1.4916681462400413e-154;
constexpr std::int64_t kMaximumInt64 = 9223372036854775807LL;

static_assert(kThreadsPerBlock > 0);
static_assert((kThreadsPerBlock & (kThreadsPerBlock - 1)) == 0,
              "the energy reduction requires a power-of-two block size");

__device__ void record_error(std::uint32_t* device_error, Gfn2D4DeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool sequence_is_valid(const std::uint32_t* device_error) {
  return atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
         static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess);
}

/*
 * Read the sticky error once per block. A per-thread atomic read creates severe
 * contention on a single address for large batches, while a non-atomic load
 * may race with record_error from a block that started earlier.
 */
__device__ bool block_sequence_is_valid(const std::uint32_t* device_error,
                                        std::uint32_t& shared_error) {
  if (threadIdx.x == 0) {
    shared_error = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u);
  }
  __syncthreads();
  return shared_error == static_cast<std::uint32_t>(Gfn2D4DeviceError::kSuccess);
}

/* Match the CPU plan's checked n*(n-1)/2 construction without signed overflow. */
__device__ bool packed_pair_count(std::int64_t atoms, std::int64_t& pairs) {
  if (atoms < 0 || (atoms > 0 && atoms - 1 > kMaximumInt64 / atoms)) {
    return false;
  }
  pairs = atoms * (atoms - 1) / 2;
  return true;
}

__device__ double charge_scale(double a, double c, double qref, double qmod) {
  if (qmod < 0.0) {
    return exp(a);
  }
  return exp(a * (1.0 - exp(c * (1.0 - qref / qmod))));
}

__device__ double charge_scale_derivative(double a, double c, double qref, double qmod) {
  if (qmod < 0.0) {
    return 0.0;
  }
  const double inner = exp(c * (1.0 - qref / qmod));
  return -a * c * inner * charge_scale(a, c, qref, qmod) * qref / (qmod * qmod);
}

__global__ void topology_preflight_kernel(Gfn2D4DeviceBatch batch,
                                          Gfn2D4DeviceParameters parameters,
                                          std::uint32_t* device_error) {
  if (blockIdx.x != 0) {
    return;
  }
  __shared__ std::uint32_t shared_error;
  __shared__ std::uint64_t partial_hash[kThreadsPerBlock];
  if (!block_sequence_is_valid(device_error, shared_error)) {
    return;
  }
  if (threadIdx.x == 0 && (batch.atom_offsets[0] != 0 || batch.pair_offsets[0] != 0 ||
                           batch.atom_offsets[batch.batch_size] != batch.total_atoms ||
                           batch.pair_offsets[batch.batch_size] != batch.total_pairs)) {
    record_error(device_error, Gfn2D4DeviceError::kInvalidOffsets);
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t pair_begin = batch.pair_offsets[system];
    const std::int64_t pair_end = batch.pair_offsets[system + 1];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > batch.total_atoms || pair_begin < 0 ||
        pair_begin > pair_end || pair_end > batch.total_pairs) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidOffsets);
      continue;
    }
    const std::int64_t atoms = atom_end - atom_begin;
    std::int64_t expected_pairs = 0;
    if (!packed_pair_count(atoms, expected_pairs) || pair_end - pair_begin != expected_pairs) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidOffsets);
    }
  }
  std::uint64_t local_hash = 0u;
  for (std::int64_t atom = threadIdx.x; atom < batch.total_atoms; atom += blockDim.x) {
    const std::int32_t atomic_number = batch.atomic_numbers[atom];
    local_hash ^= gfn2_d4_atomic_number_hash_contribution(atomic_number, atom);
    if (atomic_number <= 0 || atomic_number > parameters.element_count) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidAtomicNumber);
      continue;
    }
    const Gfn2D4DeviceElementData element = parameters.elements[atomic_number - 1];
    if (element.reference_count == 0 || element.reference_count > kGfn2D4MaximumReferences ||
        static_cast<std::int64_t>(element.reference_offset) + element.reference_count >
            parameters.reference_count ||
        !(element.effective_charge > 0.0) || !isfinite(element.effective_charge) ||
        !(element.hardness > 0.0) || !isfinite(element.hardness)) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidParameterData);
    }
  }
  partial_hash[threadIdx.x] = local_hash;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial_hash[threadIdx.x] ^= partial_hash[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const std::uint64_t atomic_number_hash =
        partial_hash[0] ^ 0x243f6a8885a308d3ULL ^
        gfn2_d4_hash_mix(static_cast<std::uint64_t>(batch.total_atoms));
    if (atomic_number_hash != batch.atomic_number_hash) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidAtomicNumber);
    }
  }
}

__global__ void prepare_weights_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                       Gfn2D4DeviceCache cache, const double* atomic_charges,
                                       Gfn2D4DeviceWorkspace workspace,
                                       std::uint32_t* device_error) {
  __shared__ std::uint32_t shared_error;
  if (!block_sequence_is_valid(device_error, shared_error)) {
    return;
  }
  const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (atom >= batch.total_atoms) {
    return;
  }
  const double coordination = cache.coordination_numbers[atom];
  const double charge = atomic_charges[atom];
  if (!isfinite(coordination)) {
    record_error(device_error, Gfn2D4DeviceError::kInvalidCoordination);
    return;
  }
  if (!isfinite(charge)) {
    record_error(device_error, Gfn2D4DeviceError::kNonfiniteCharge);
    return;
  }

  const std::int32_t atomic_number = batch.atomic_numbers[atom];
  if (atomic_number <= 0 || atomic_number > parameters.element_count) {
    record_error(device_error, Gfn2D4DeviceError::kInvalidAtomicNumber);
    return;
  }
  const Gfn2D4DeviceElementData element = parameters.elements[atomic_number - 1];
  if (element.reference_count == 0 || element.reference_count > kGfn2D4MaximumReferences ||
      static_cast<std::int64_t>(element.reference_offset) + element.reference_count >
          parameters.reference_count) {
    record_error(device_error, Gfn2D4DeviceError::kInvalidParameterData);
    return;
  }

  const std::int64_t output_offset = atom * kGfn2D4MaximumReferences;
  for (std::int64_t local = 0; local < kGfn2D4MaximumReferences; ++local) {
    workspace.weights[output_offset + local] = 0.0;
    workspace.weight_charge_derivatives[output_offset + local] = 0.0;
  }

  double normalization = 0.0;
  double maximum_reference_cn = -1.7976931348623157e308;
  for (std::int64_t local = 0; local < element.reference_count; ++local) {
    const Gfn2D4DeviceReferenceData reference =
        parameters.references[static_cast<std::int64_t>(element.reference_offset) + local];
    if (reference.gaussian_count == 0 || !isfinite(reference.coordination_number) ||
        !isfinite(reference.charge)) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidParameterData);
      return;
    }
    maximum_reference_cn = fmax(maximum_reference_cn, reference.coordination_number);
    for (std::int64_t gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
      const double factor = static_cast<double>(gaussian) * kReferenceWeightFactor;
      const double delta = coordination - reference.coordination_number;
      normalization += exp(-factor * delta * delta);
    }
  }
  const double inverse_normalization =
      fabs(normalization) > kMinimumWeightNorm ? 1.0 / normalization : 0.0;
  const double qmod = charge + element.effective_charge;
  const double charge_steepness = element.hardness * kChargeScalingSteepness;

  for (std::int64_t local = 0; local < element.reference_count; ++local) {
    const Gfn2D4DeviceReferenceData reference =
        parameters.references[static_cast<std::int64_t>(element.reference_offset) + local];
    double numerator = 0.0;
    for (std::int64_t gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
      const double factor = static_cast<double>(gaussian) * kReferenceWeightFactor;
      const double delta = coordination - reference.coordination_number;
      numerator += exp(-factor * delta * delta);
    }
    double cn_weight = numerator * inverse_normalization;
    if (!isfinite(cn_weight) || inverse_normalization == 0.0) {
      cn_weight = fabs(maximum_reference_cn - reference.coordination_number) < 1.0e-12 ? 1.0 : 0.0;
    }
    const double qref = reference.charge + element.effective_charge;
    const double scaling = charge_scale(kChargeScalingHeight, charge_steepness, qref, qmod);
    const double derivative =
        cn_weight * charge_scale_derivative(kChargeScalingHeight, charge_steepness, qref, qmod);
    const double weight = cn_weight * scaling;
    if (!isfinite(weight) || !isfinite(derivative)) {
      record_error(device_error, Gfn2D4DeviceError::kNonfiniteArithmetic);
      return;
    }
    workspace.weights[output_offset + local] = weight;
    workspace.weight_charge_derivatives[output_offset + local] = derivative;
  }
}

struct PairCoefficient {
  double c6;
  double first_charge;
  double second_charge;
};

__device__ PairCoefficient pair_coefficient(Gfn2D4DeviceBatch batch,
                                            Gfn2D4DeviceParameters parameters,
                                            Gfn2D4DeviceWorkspace workspace, std::int64_t first,
                                            std::int64_t second) {
  const Gfn2D4DeviceElementData first_element =
      parameters.elements[batch.atomic_numbers[first] - 1];
  const Gfn2D4DeviceElementData second_element =
      parameters.elements[batch.atomic_numbers[second] - 1];
  const std::int64_t first_weight = first * kGfn2D4MaximumReferences;
  const std::int64_t second_weight = second * kGfn2D4MaximumReferences;
  PairCoefficient result{0.0, 0.0, 0.0};
  for (std::int64_t first_ref = 0; first_ref < first_element.reference_count; ++first_ref) {
    const std::int64_t global_first = first_element.reference_offset + first_ref;
    const double first_value = workspace.weights[first_weight + first_ref];
    const double first_derivative = workspace.weight_charge_derivatives[first_weight + first_ref];
    for (std::int64_t second_ref = 0; second_ref < second_element.reference_count; ++second_ref) {
      const std::int64_t global_second = second_element.reference_offset + second_ref;
      const double reference_c6 =
          parameters.reference_c6[global_first * parameters.reference_count + global_second];
      const double second_value = workspace.weights[second_weight + second_ref];
      const double second_derivative =
          workspace.weight_charge_derivatives[second_weight + second_ref];
      result.c6 += first_value * second_value * reference_c6;
      result.first_charge += first_derivative * second_value * reference_c6;
      result.second_charge += first_value * second_derivative * reference_c6;
    }
  }
  return result;
}

__global__ void potential_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                                 Gfn2D4DeviceCache cache, Gfn2D4DeviceWorkspace workspace,
                                 std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ std::uint32_t shared_error;
  if (!block_sequence_is_valid(device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    double potential = 0.0;
    for (std::int64_t other = begin; other < end; ++other) {
      if (other == atom) {
        continue;
      }
      const std::int64_t first = atom < other ? atom : other;
      const std::int64_t second = atom < other ? other : atom;
      const std::int64_t local_first = first - begin;
      const std::int64_t local_second = second - begin;
      const std::int64_t packed_pair =
          batch.pair_offsets[system] + local_second * (local_second - 1) / 2 + local_first;
      const double damping = cache.pair_data[packed_pair * kGfn2D4PairDataElements + 3];
      if (!isfinite(damping) || damping < 0.0) {
        record_error(device_error, Gfn2D4DeviceError::kInvalidDamping);
        return;
      }
      if (damping == 0.0) {
        continue;
      }
      const PairCoefficient coefficient =
          pair_coefficient(batch, parameters, workspace, first, second);
      const double derivative =
          atom == first ? coefficient.first_charge : coefficient.second_charge;
      potential -= derivative * damping;
      if (!isfinite(potential)) {
        record_error(device_error, Gfn2D4DeviceError::kNonfiniteArithmetic);
        return;
      }
    }
    workspace.atom_scratch[atom] = potential;
  }
}

__global__ void energy_kernel(Gfn2D4DeviceBatch batch, Gfn2D4DeviceParameters parameters,
                              Gfn2D4DeviceCache cache, Gfn2D4DeviceWorkspace workspace,
                              std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ double partial[kThreadsPerBlock];
  __shared__ std::uint32_t shared_error;
  if (!block_sequence_is_valid(device_error, shared_error)) {
    return;
  }
  const std::int64_t begin = batch.atom_offsets[system];
  const std::int64_t end = batch.atom_offsets[system + 1];
  const std::int64_t atom_count = end - begin;
  const std::int64_t pair_begin = batch.pair_offsets[system];
  const std::int64_t pair_end = batch.pair_offsets[system + 1];
  double local_energy = 0.0;
  for (std::int64_t local_pair = threadIdx.x; local_pair < pair_end - pair_begin;
       local_pair += blockDim.x) {
    std::int64_t local_second =
        static_cast<std::int64_t>((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(local_pair))) * 0.5);
    while (local_second * (local_second - 1) / 2 > local_pair) {
      --local_second;
    }
    while (local_second + 1 < atom_count && (local_second + 1) * local_second / 2 <= local_pair) {
      ++local_second;
    }
    const std::int64_t local_first = local_pair - local_second * (local_second - 1) / 2;
    const std::int64_t first = begin + local_first;
    const std::int64_t second = begin + local_second;
    const double damping = cache.pair_data[(pair_begin + local_pair) * kGfn2D4PairDataElements + 3];
    if (!isfinite(damping) || damping < 0.0) {
      record_error(device_error, Gfn2D4DeviceError::kInvalidDamping);
      local_energy = 0.0;
      break;
    }
    if (damping != 0.0) {
      const PairCoefficient coefficient =
          pair_coefficient(batch, parameters, workspace, first, second);
      local_energy -= coefficient.c6 * damping;
      if (!isfinite(local_energy)) {
        record_error(device_error, Gfn2D4DeviceError::kNonfiniteArithmetic);
        local_energy = 0.0;
        break;
      }
    }
  }
  partial[threadIdx.x] = local_energy;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && sequence_is_valid(device_error) && isfinite(partial[0])) {
    workspace.batch_scratch[system] = partial[0];
  }
}

__global__ void publish_kernel(std::int64_t count, const double* scratch, double* output,
                               const std::uint32_t* device_error) {
  __shared__ std::uint32_t shared_error;
  if (!block_sequence_is_valid(device_error, shared_error)) {
    return;
  }
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = scratch[index];
  }
}

bool is_aligned(const void* pointer, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

/*
 * Form a half-open byte range without dereferencing the CUDA address. Empty
 * ranges are normalized so a NULL zero-pair cache does not alias anything.
 */
bool make_address_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                        AddressRange& range) {
  if (elements < 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes == 0u) {
    range = {};
    return true;
  }
  if (pointer == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t ReadCount, std::size_t WriteCount>
bool writable_ranges_are_disjoint(const std::array<AddressRange, ReadCount>& read_ranges,
                                  const std::array<AddressRange, WriteCount>& write_ranges) {
  for (std::size_t write = 0; write < WriteCount; ++write) {
    for (const AddressRange& read_range : read_ranges) {
      if (ranges_overlap(write_ranges[write], read_range)) {
        return false;
      }
    }
    for (std::size_t other = write + 1u; other < WriteCount; ++other) {
      if (ranges_overlap(write_ranges[write], write_ranges[other])) {
        return false;
      }
    }
  }
  return true;
}

cudaError_t launch_grid(std::int64_t count, unsigned int* blocks) {
  if (count <= 0 || blocks == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t needed =
      (static_cast<std::uint64_t>(count) + kThreadsPerBlock - 1u) / kThreadsPerBlock;
  if (needed > std::numeric_limits<unsigned int>::max()) {
    return cudaErrorInvalidConfiguration;
  }
  *blocks = static_cast<unsigned int>(needed);
  return cudaSuccess;
}

cudaError_t check_launch() { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t reset_gfn2_d4_device_error_cuda(std::uint32_t* device_error,
                                            cudaStream_t stream) noexcept {
  AddressRange error_range;
  if (device_error == nullptr || !is_aligned(device_error, alignof(std::uint32_t)) ||
      !make_address_range(device_error, 1, sizeof(*device_error), error_range)) {
    return cudaErrorInvalidValue;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t evaluate_gfn2_d4_two_body_cuda(
    const Gfn2D4DeviceBatch& batch, const Gfn2D4DeviceParameters& parameters,
    const Gfn2D4DeviceCache& cache, const double* atomic_charges, double* energies,
    double* atomic_potentials, const Gfn2D4DeviceWorkspace& workspace, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  const bool reference_square_representable =
      parameters.reference_count > 0 &&
      parameters.reference_count <=
          std::numeric_limits<std::int64_t>::max() / parameters.reference_count;
  const bool workspace_extents_representable =
      batch.total_atoms > 0 &&
      batch.total_atoms <= std::numeric_limits<std::int64_t>::max() / kGfn2D4MaximumReferences &&
      batch.total_pairs >= 0 &&
      batch.total_pairs <= std::numeric_limits<std::int64_t>::max() / kGfn2D4PairDataElements;
  const std::int64_t expected_pair_elements =
      workspace_extents_representable ? batch.total_pairs * kGfn2D4PairDataElements : 0;
  const std::int64_t expected_weight_elements =
      workspace_extents_representable ? batch.total_atoms * kGfn2D4MaximumReferences : 0;
  const std::int64_t required_reference_c6_elements =
      reference_square_representable ? parameters.reference_count * parameters.reference_count : 0;
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_pairs < 0 ||
      batch.batch_size > std::numeric_limits<unsigned int>::max() || batch.plan_token == 0u ||
      !workspace_extents_representable || batch.atom_offsets == nullptr ||
      batch.pair_offsets == nullptr || batch.atomic_numbers == nullptr ||
      parameters.elements == nullptr || parameters.element_count <= 0 ||
      parameters.references == nullptr || parameters.reference_c6 == nullptr ||
      !reference_square_representable ||
      parameters.reference_c6_elements < required_reference_c6_elements ||
      cache.plan_token != batch.plan_token || cache.geometry_generation == 0u ||
      (batch.total_pairs != 0 && cache.pair_data == nullptr) ||
      cache.pair_data_elements != expected_pair_elements || cache.coordination_numbers == nullptr ||
      cache.coordination_elements != batch.total_atoms || atomic_charges == nullptr ||
      energies == nullptr || atomic_potentials == nullptr || workspace.weights == nullptr ||
      workspace.weight_charge_derivatives == nullptr ||
      workspace.weight_elements < expected_weight_elements || workspace.atom_scratch == nullptr ||
      workspace.atom_elements < batch.total_atoms || workspace.batch_scratch == nullptr ||
      workspace.batch_elements < batch.batch_size || device_error == nullptr ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.pair_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atomic_numbers, alignof(std::int32_t)) ||
      !is_aligned(parameters.elements, alignof(Gfn2D4DeviceElementData)) ||
      !is_aligned(parameters.references, alignof(Gfn2D4DeviceReferenceData)) ||
      !is_aligned(parameters.reference_c6, alignof(double)) ||
      (cache.pair_data != nullptr && !is_aligned(cache.pair_data, alignof(double))) ||
      !is_aligned(cache.coordination_numbers, alignof(double)) ||
      !is_aligned(atomic_charges, alignof(double)) || !is_aligned(energies, alignof(double)) ||
      !is_aligned(atomic_potentials, alignof(double)) ||
      !is_aligned(workspace.weights, alignof(double)) ||
      !is_aligned(workspace.weight_charge_derivatives, alignof(double)) ||
      !is_aligned(workspace.atom_scratch, alignof(double)) ||
      !is_aligned(workspace.batch_scratch, alignof(double)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }

  std::array<AddressRange, 9> read_ranges;
  std::array<AddressRange, 7> write_ranges;
  if (!make_address_range(batch.atom_offsets, batch.batch_size + 1, sizeof(*batch.atom_offsets),
                          read_ranges[0]) ||
      !make_address_range(batch.pair_offsets, batch.batch_size + 1, sizeof(*batch.pair_offsets),
                          read_ranges[1]) ||
      !make_address_range(batch.atomic_numbers, batch.total_atoms, sizeof(*batch.atomic_numbers),
                          read_ranges[2]) ||
      !make_address_range(parameters.elements, parameters.element_count,
                          sizeof(*parameters.elements), read_ranges[3]) ||
      !make_address_range(parameters.references, parameters.reference_count,
                          sizeof(*parameters.references), read_ranges[4]) ||
      !make_address_range(parameters.reference_c6, parameters.reference_c6_elements,
                          sizeof(*parameters.reference_c6), read_ranges[5]) ||
      !make_address_range(cache.pair_data, expected_pair_elements, sizeof(*cache.pair_data),
                          read_ranges[6]) ||
      !make_address_range(cache.coordination_numbers, batch.total_atoms,
                          sizeof(*cache.coordination_numbers), read_ranges[7]) ||
      !make_address_range(atomic_charges, batch.total_atoms, sizeof(*atomic_charges),
                          read_ranges[8]) ||
      !make_address_range(workspace.weights, workspace.weight_elements, sizeof(*workspace.weights),
                          write_ranges[0]) ||
      !make_address_range(workspace.weight_charge_derivatives, workspace.weight_elements,
                          sizeof(*workspace.weight_charge_derivatives), write_ranges[1]) ||
      !make_address_range(workspace.atom_scratch, workspace.atom_elements,
                          sizeof(*workspace.atom_scratch), write_ranges[2]) ||
      !make_address_range(workspace.batch_scratch, workspace.batch_elements,
                          sizeof(*workspace.batch_scratch), write_ranges[3]) ||
      !make_address_range(energies, batch.batch_size, sizeof(*energies), write_ranges[4]) ||
      !make_address_range(atomic_potentials, batch.total_atoms, sizeof(*atomic_potentials),
                          write_ranges[5]) ||
      !make_address_range(device_error, 1, sizeof(*device_error), write_ranges[6]) ||
      !writable_ranges_are_disjoint(read_ranges, write_ranges)) {
    return cudaErrorInvalidValue;
  }

  unsigned int atom_blocks = 0;
  unsigned int batch_blocks = 0;
  cudaError_t status = launch_grid(batch.total_atoms, &atom_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  status = launch_grid(batch.batch_size, &batch_blocks);
  if (status != cudaSuccess) {
    return status;
  }

  topology_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, parameters, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  prepare_weights_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, atomic_charges, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  energy_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, parameters, cache, workspace, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_kernel<<<atom_blocks, kThreadsPerBlock, 0, stream>>>(
      batch.total_atoms, workspace.atom_scratch, atomic_potentials, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_kernel<<<batch_blocks, kThreadsPerBlock, 0, stream>>>(
      batch.batch_size, workspace.batch_scratch, energies, device_error);
  return check_launch();
}

}  // namespace gpuxtb::detail::cuda
