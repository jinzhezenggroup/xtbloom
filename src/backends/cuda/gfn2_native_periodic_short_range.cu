// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime_api.h>
#include <stdio.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/gfn2_native_periodic_short_range.cuh"
#include "backends/cuda/gfn2_parameters.cuh"
#include "backends/cuda/periodic_wrap.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr double kCoordinationCutoffBohr = 25.0;
constexpr double kCoordinationCutoffSquared = kCoordinationCutoffBohr * kCoordinationCutoffBohr;
constexpr double kMinimumDistanceSquared = 1.0e-12;
/* The released CPU repulsion path permits a smaller positive distance than
 * coordination because its screened exponential remains finite there. */
constexpr double kRepulsionMinimumDistanceSquared = 1.0e-24;
constexpr double kFirstSteepness = 10.0;
constexpr double kSecondSteepness = 20.0;
constexpr double kSecondRadiusShiftBohr = 2.0;

using DeviceError = Gfn2NativePeriodicShortRangeDeviceError;

bool checked_multiply(std::int64_t value, std::int64_t factor, std::int64_t* result) noexcept {
  if (result == nullptr || value < 0 || factor < 0 ||
      (factor != 0 && value > std::numeric_limits<std::int64_t>::max() / factor)) {
    return false;
  }
  *result = value * factor;
  return true;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
  bool empty = true;
};

template <typename T>
bool make_range(const T* pointer, std::int64_t elements, AddressRange* range) noexcept {
  if (range == nullptr || elements < 0) return false;
  if (elements == 0) {
    *range = {};
    return pointer == nullptr;
  }
  if (pointer == nullptr || reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) != 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(T);
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
  *range = {begin, begin + bytes, false};
  return true;
}

bool overlaps(const AddressRange& first, const AddressRange& second) noexcept {
  return !first.empty && !second.empty && first.begin < second.end && second.begin < first.end;
}

template <std::size_t N>
bool all_disjoint(const std::array<AddressRange, N>& ranges) noexcept {
  for (std::size_t first = 0u; first < ranges.size(); ++first) {
    for (std::size_t second = first + 1u; second < ranges.size(); ++second) {
      if (overlaps(ranges[first], ranges[second])) return false;
    }
  }
  return true;
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    DeviceError error) {
  atomicCAS(system_errors + system, static_cast<std::uint32_t>(DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_device_error(std::uint32_t* device_error, DeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors + system), 0u) ==
         static_cast<std::uint32_t>(DeviceError::kSuccess);
}

__device__ bool is_origin(const Gfn2CudaPeriodicTranslation& translation) {
  return translation.index[0] == 0 && translation.index[1] == 0 && translation.index[2] == 0 &&
         translation.cartesian[0] == 0.0 && translation.cartesian[1] == 0.0 &&
         translation.cartesian[2] == 0.0;
}

__device__ bool wrap_position(const Gfn2CudaPeriodicTopologyView& topology, std::int64_t system,
                              const double* position, double* wrapped) {
  return wrap_periodic_position(topology, system, position, wrapped);
}

__device__ bool image_displacement(const double* center, const double* image,
                                   const Gfn2CudaPeriodicTranslation& translation,
                                   double displacement[3], double& distance_squared) {
  for (int component = 0; component < 3; ++component) {
    const double translated = image[component] + translation.cartesian[component];
    displacement[component] = translated - center[component];
    if (!isfinite(displacement[component]) ||
        fabs(displacement[component]) > kCoordinationCutoffBohr) {
      return false;
    }
  }
  distance_squared = fma(displacement[0], displacement[0],
                         fma(displacement[1], displacement[1], displacement[2] * displacement[2]));
  return isfinite(distance_squared) && distance_squared <= kCoordinationCutoffSquared;
}

__device__ double logistic(double argument) {
  if (argument >= 0.0) {
    const double exponential = exp(-argument);
    return 1.0 / (1.0 + exponential);
  }
  const double exponential = exp(argument);
  return exponential / (1.0 + exponential);
}

__device__ double coordination_pair(double distance, double radius) {
  const double inverse_distance = 1.0 / distance;
  const double first = logistic(kFirstSteepness * (radius * inverse_distance - 1.0));
  const double shifted_radius = radius + kSecondRadiusShiftBohr;
  const double second = logistic(kSecondSteepness * (shifted_radius * inverse_distance - 1.0));
  return first * second;
}

/* Return d f / d |r| divided by |r| for the two-logistic GFN2 CN
 * contribution.  Keeping this expression beside coordination_pair() is
 * important: the native periodic force path must differentiate exactly the
 * image-aware function used to build the committed CN values, rather than
 * silently falling back to the nonperiodic pair cache. */
__device__ bool coordination_pair_derivative_over_distance(double distance, double radius,
                                                           double& derivative_over_distance) {
  if (!(distance > 0.0) || !isfinite(distance) || !isfinite(radius)) return false;
  const double inverse_distance = 1.0 / distance;
  const double shifted_radius = radius + kSecondRadiusShiftBohr;
  const double first_argument = kFirstSteepness * (radius * inverse_distance - 1.0);
  const double second_argument = kSecondSteepness * (shifted_radius * inverse_distance - 1.0);
  const double first = logistic(first_argument);
  const double second = logistic(second_argument);
  const double radial_numerator =
      kFirstSteepness * radius * first * (1.0 - first) * second +
      kSecondSteepness * shifted_radius * second * (1.0 - second) * first;
  derivative_over_distance =
      -radial_numerator * inverse_distance * inverse_distance * inverse_distance;
  return isfinite(derivative_over_distance);
}

__device__ bool repulsion_pair(const Gfn2NativePeriodicShortRangeDeviceBatch& batch,
                               std::int64_t first, std::int64_t second, double distance_squared,
                               double& energy, double& gradient_scale) {
  const std::int32_t first_number = batch.atomic_numbers[first];
  const std::int32_t second_number = batch.atomic_numbers[second];
  if (first_number < 1 ||
      first_number > static_cast<std::int32_t>(parameters::gfn2::kElementCount) ||
      second_number < 1 ||
      second_number > static_cast<std::int32_t>(parameters::gfn2::kElementCount)) {
    return false;
  }
  const auto first_element = g_gfn2_elements[first_number - 1];
  const auto second_element = g_gfn2_elements[second_number - 1];
  const double distance = sqrt(distance_squared);
  const bool light_pair = first_number <= 2 && second_number <= 2;
  const double exponent =
      light_pair ? g_gfn2_global.repulsion_klight : g_gfn2_global.repulsion_kexp;
  /* GFN2's repulsion convention uses d for H/H and d*sqrt(d) otherwise;
   * `exponent` is the fitted derivative multiplier, not the exponent of the
   * distance power in the energy. */
  const double distance_power = light_pair ? distance : distance * sqrt(distance);
  const double pair_alpha = sqrt(first_element.arep) * sqrt(second_element.arep);
  const double pair_charge = first_element.zeff * second_element.zeff;
  energy = pair_charge * exp(-pair_alpha * distance_power) / distance;
  gradient_scale = -(pair_alpha * distance_power * exponent + 1.0) * energy / distance_squared;
  return isfinite(energy) && isfinite(gradient_scale);
}

__global__ void native_periodic_short_range_kernel(
    Gfn2NativePeriodicShortRangeDeviceBatch batch,
    Gfn2NativePeriodicShortRangeDeviceWorkspace workspace, double* coordination_numbers,
    double* repulsion_energies, double* repulsion_gradients, double* repulsion_strain,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= batch.topology.batch_size) return;

  __shared__ std::int64_t atom_begin;
  __shared__ std::int64_t atom_end;
  __shared__ std::int64_t translation_begin;
  __shared__ std::int64_t translation_end;
  __shared__ int valid;

  if (threadIdx.x == 0) {
    valid = 1;
    atom_begin = batch.topology.atom_offsets[system];
    atom_end = batch.topology.atom_offsets[system + 1];
    translation_begin = batch.topology.translation_offsets[system];
    translation_end = batch.topology.translation_offsets[system + 1];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > batch.topology.total_atoms ||
        translation_begin < 0 || translation_begin > translation_end ||
        translation_end > batch.topology.total_translations ||
        translation_begin == translation_end ||
        batch.topology.periodic_axes[system] != XTBLOOM_PERIODIC_AXES_NONE &&
            batch.topology.periodic_axes[system] != XTBLOOM_PERIODIC_AXES_XYZ) {
      valid = 0;
      record_system_error(system_errors, system, DeviceError::kInvalidOffsets);
      record_device_error(device_error, DeviceError::kInvalidTopology);
    }
  }
  __syncthreads();
  if (valid == 0) return;

  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int32_t atomic_number = batch.atomic_numbers[atom];
    if (atomic_number < 1 ||
        atomic_number > static_cast<std::int32_t>(parameters::gfn2::kElementCount)) {
      record_system_error(system_errors, system, DeviceError::kInvalidAtomicNumber);
      atomicExch(&valid, 0);
      continue;
    }
    if (!(batch.covalent_radii[atom] > 0.0) || !isfinite(batch.covalent_radii[atom])) {
      record_system_error(system_errors, system, DeviceError::kInvalidCovalentRadius);
      atomicExch(&valid, 0);
    }
    const double* const position = batch.positions + atom * 3;
    double* const wrapped = workspace.wrapped_positions + atom * 3;
    if (!isfinite(position[0]) || !isfinite(position[1]) || !isfinite(position[2]) ||
        !wrap_position(batch.topology, system, position, wrapped)) {
      record_system_error(system_errors, system, DeviceError::kNonfinitePosition);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) return;
  /* One thread owns one atom's CN sum.  This avoids atomics and therefore
   * keeps repeated launches deterministic while matching the ordered CPU
   * accumulation to ordinary binary64 roundoff. */
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const double* const center = workspace.wrapped_positions + atom * 3;
    const double radius = batch.covalent_radii[atom];
    double sum = 0.0;
    for (std::int64_t image_atom = atom_begin; image_atom < atom_end; ++image_atom) {
      const double* const image = workspace.wrapped_positions + image_atom * 3;
      const double pair_radius = radius + batch.covalent_radii[image_atom];
      for (std::int64_t translation_index = translation_begin; translation_index < translation_end;
           ++translation_index) {
        const auto& translation = batch.topology.translations[translation_index];
        if (atom == image_atom && is_origin(translation)) continue;
        double displacement[3]{};
        double distance_squared = 0.0;
        if (!image_displacement(center, image, translation, displacement, distance_squared))
          continue;
        if (distance_squared < kMinimumDistanceSquared) {
          record_system_error(system_errors, system, DeviceError::kCoincidentImage);
          atomicExch(&valid, 0);
          continue;
        }
        sum += coordination_pair(sqrt(distance_squared), pair_radius);
      }
    }
    if (!isfinite(sum)) {
      record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
      atomicExch(&valid, 0);
    }
    workspace.coordination[atom] = sum;
  }
  __syncthreads();
  if (valid == 0) return;

  /* The force/strain accumulation is written in the same unique-pair order
   * as evaluate_periodic_repulsion_cpu.  Thread zero handles energy and
   * strain; each atom thread independently evaluates its Cartesian gradient,
   * which avoids cross-thread atomics in this correctness-first primitive. */
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const double* const center = workspace.wrapped_positions + atom * 3;
    double gradient[3] = {0.0, 0.0, 0.0};
    for (std::int64_t image_atom = atom_begin; image_atom < atom_end; ++image_atom) {
      if (image_atom == atom) continue;
      const double* const image = workspace.wrapped_positions + image_atom * 3;
      for (std::int64_t translation_index = translation_begin; translation_index < translation_end;
           ++translation_index) {
        const auto& translation = batch.topology.translations[translation_index];
        double displacement[3]{};
        double distance_squared = 0.0;
        if (!image_displacement(center, image, translation, displacement, distance_squared))
          continue;
        if (distance_squared <= kRepulsionMinimumDistanceSquared) {
          record_system_error(system_errors, system, DeviceError::kCoincidentImage);
          atomicExch(&valid, 0);
          continue;
        }
        double energy = 0.0;
        double scale = 0.0;
        if (!repulsion_pair(batch, atom, image_atom, distance_squared, energy, scale)) {
          record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
          atomicExch(&valid, 0);
          continue;
        }
        /* rij is center - image, exactly the sign used by the CPU periodic
         * repulsion derivative. */
        for (int component = 0; component < 3; ++component) {
          gradient[component] += scale * (-displacement[component]);
        }
      }
    }
    if (!isfinite(gradient[0]) || !isfinite(gradient[1]) || !isfinite(gradient[2])) {
      record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
      atomicExch(&valid, 0);
    }
    workspace.repulsion_gradients[atom * 3] = gradient[0];
    workspace.repulsion_gradients[atom * 3 + 1] = gradient[1];
    workspace.repulsion_gradients[atom * 3 + 2] = gradient[2];
  }

  if (threadIdx.x == 0) {
    double energy_sum = 0.0;
    double strain[9] = {};
    for (std::int64_t first = atom_begin; first < atom_end; ++first) {
      const double* const center = workspace.wrapped_positions + first * 3;
      for (std::int64_t second = atom_begin; second <= first; ++second) {
        const double* const image = workspace.wrapped_positions + second * 3;
        for (std::int64_t translation_index = translation_begin;
             translation_index < translation_end; ++translation_index) {
          const auto& translation = batch.topology.translations[translation_index];
          if (first == second && is_origin(translation)) continue;
          double displacement[3]{};
          double distance_squared = 0.0;
          if (!image_displacement(center, image, translation, displacement, distance_squared))
            continue;
          if (distance_squared <= kRepulsionMinimumDistanceSquared) {
            record_system_error(system_errors, system, DeviceError::kCoincidentImage);
            valid = 0;
            continue;
          }
          double pair_energy = 0.0;
          double gradient_scale = 0.0;
          if (!repulsion_pair(batch, first, second, distance_squared, pair_energy,
                              gradient_scale)) {
            record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
            valid = 0;
            continue;
          }
          energy_sum += pair_energy;
          const double first_gradient[3] = {gradient_scale * (-displacement[0]),
                                            gradient_scale * (-displacement[1]),
                                            gradient_scale * (-displacement[2])};
          const double strain_scale = first == second ? 0.5 : 1.0;
          for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
              strain[row * 3 + column] +=
                  strain_scale * first_gradient[row] * (-displacement[column]);
            }
          }
        }
      }
    }
    if (!isfinite(energy_sum)) {
      record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
      valid = 0;
    }
    workspace.repulsion_energies[system] = energy_sum;
    for (int component = 0; component < 9; ++component) {
      workspace.repulsion_strain[system * 9 + component] = strain[component];
    }
  }
  __syncthreads();
  if (valid == 0 || !system_is_valid(system_errors, system)) return;

  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    coordination_numbers[atom] = workspace.coordination[atom];
    repulsion_gradients[atom * 3] = workspace.repulsion_gradients[atom * 3];
    repulsion_gradients[atom * 3 + 1] = workspace.repulsion_gradients[atom * 3 + 1];
    repulsion_gradients[atom * 3 + 2] = workspace.repulsion_gradients[atom * 3 + 2];
  }
  if (threadIdx.x == 0) {
    repulsion_energies[system] = workspace.repulsion_energies[system];
    for (int component = 0; component < 9; ++component) {
      repulsion_strain[system * 9 + component] = workspace.repulsion_strain[system * 9 + component];
    }
  }
}

/* Capture the sequence-wide topology status before the peer-local VJP starts.
 * A later invalid peer belongs in system_errors; only a pre-existing device
 * contract failure should suppress the complete composed force transaction. */
__global__ void capture_native_coordination_sequence_kernel(const std::uint32_t* device_error,
                                                            std::uint32_t* sequence_active) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *sequence_active = atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                               static_cast<std::uint32_t>(DeviceError::kSuccess)
                           ? 1u
                           : 0u;
  }
}

/* The native forward CN uses wrapped central-cell coordinates and traverses
 * every image translation.  This reverse leaf mirrors that exact traversal;
 * wrapping is a piecewise translation, so its Cartesian Jacobian is the
 * identity away from the already-rejected cell-boundary discontinuity. */
__global__ void native_periodic_coordination_vjp_kernel(
    Gfn2NativePeriodicShortRangeDeviceBatch batch,
    Gfn2NativePeriodicShortRangeDeviceWorkspace workspace, const double* dE_dcn, double* gradients,
    double* gradient_scratch, const std::uint32_t* sequence_active, std::uint32_t* system_errors,
    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= batch.topology.batch_size ||
      atomicAdd(const_cast<std::uint32_t*>(sequence_active), 0u) != 1u) {
    return;
  }

  if (batch.active_mask != nullptr) {
    const std::uint8_t selected = batch.active_mask[system];
    if (selected > 1u) {
      record_system_error(system_errors, system, DeviceError::kInvalidOffsets);
      record_device_error(device_error, DeviceError::kInvalidTopology);
      return;
    }
    if (selected == 0u) return;
  }

  __shared__ std::int64_t atom_begin;
  __shared__ std::int64_t atom_end;
  __shared__ std::int64_t translation_begin;
  __shared__ std::int64_t translation_end;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    atom_begin = batch.topology.atom_offsets[system];
    atom_end = batch.topology.atom_offsets[system + 1];
    translation_begin = batch.topology.translation_offsets[system];
    translation_end = batch.topology.translation_offsets[system + 1];
    valid = atom_begin >= 0 && atom_begin <= atom_end && atom_end <= batch.topology.total_atoms &&
            translation_begin >= 0 && translation_begin < translation_end &&
            translation_end <= batch.topology.total_translations;
    if (!valid) {
      record_system_error(system_errors, system, DeviceError::kInvalidOffsets);
      record_device_error(device_error, DeviceError::kInvalidTopology);
    }
  }
  __syncthreads();
  if (valid == 0) return;

  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const double* const position = workspace.wrapped_positions + atom * 3;
    if (!isfinite(position[0]) || !isfinite(position[1]) || !isfinite(position[2]) ||
        !isfinite(batch.covalent_radii[atom]) || !(batch.covalent_radii[atom] > 0.0) ||
        !isfinite(dE_dcn[atom]) || !isfinite(gradients[atom * 3]) ||
        !isfinite(gradients[atom * 3 + 1]) || !isfinite(gradients[atom * 3 + 2])) {
      record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0 || !system_is_valid(system_errors, system)) return;

  for (std::int64_t target = atom_begin + threadIdx.x; target < atom_end; target += blockDim.x) {
    double contribution[3] = {gradients[target * 3], gradients[target * 3 + 1],
                              gradients[target * 3 + 2]};
    for (std::int64_t center_atom = atom_begin; center_atom < atom_end; ++center_atom) {
      const double center_radius = batch.covalent_radii[center_atom];
      const double center_weight = dE_dcn[center_atom];
      const double* const center = workspace.wrapped_positions + center_atom * 3;
      for (std::int64_t image_atom = atom_begin; image_atom < atom_end; ++image_atom) {
        /* A self-image CN term depends on the cell but not on the Cartesian
         * position of that atom: center and image move together, so the two
         * Cartesian derivatives cancel exactly. */
        if (center_atom == image_atom) continue;
        const double* const image = workspace.wrapped_positions + image_atom * 3;
        const double pair_radius = center_radius + batch.covalent_radii[image_atom];
        for (std::int64_t translation_index = translation_begin;
             translation_index < translation_end; ++translation_index) {
          const auto& translation = batch.topology.translations[translation_index];
          double displacement[3]{};
          double distance_squared = 0.0;
          if (!image_displacement(center, image, translation, displacement, distance_squared)) {
            continue;
          }
          if (distance_squared < kMinimumDistanceSquared) {
            record_system_error(system_errors, system, DeviceError::kCoincidentImage);
            atomicExch(&valid, 0);
            continue;
          }
          double derivative_over_distance = 0.0;
          if (!coordination_pair_derivative_over_distance(sqrt(distance_squared), pair_radius,
                                                          derivative_over_distance)) {
            record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
            atomicExch(&valid, 0);
            continue;
          }
          const double weighted_scale = center_weight * derivative_over_distance;
          const double sign = target == center_atom ? -1.0 : (target == image_atom ? 1.0 : 0.0);
          if (sign == 0.0) continue;
          for (int axis = 0; axis < 3; ++axis) {
            contribution[axis] += sign * weighted_scale * displacement[axis];
            if (!isfinite(contribution[axis])) atomicExch(&valid, 0);
          }
        }
      }
    }
    if (valid != 0 && isfinite(contribution[0]) && isfinite(contribution[1]) &&
        isfinite(contribution[2])) {
      gradient_scratch[target * 3] = contribution[0];
      gradient_scratch[target * 3 + 1] = contribution[1];
      gradient_scratch[target * 3 + 2] = contribution[2];
    } else {
      record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0 || !system_is_valid(system_errors, system)) return;
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    gradients[atom * 3] = gradient_scratch[atom * 3];
    gradients[atom * 3 + 1] = gradient_scratch[atom * 3 + 1];
    gradients[atom * 3 + 2] = gradient_scratch[atom * 3 + 2];
  }
}

}  // namespace

cudaError_t reset_gfn2_native_periodic_short_range_errors_cuda(std::int64_t batch_size,
                                                               std::uint32_t* system_errors,
                                                               std::uint32_t* device_error,
                                                               cudaStream_t stream) noexcept {
  if (batch_size <= 0 || system_errors == nullptr || device_error == nullptr ||
      static_cast<std::uint64_t>(batch_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                     sizeof(*system_errors))) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  if (status != cudaSuccess) return status;
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t evaluate_gfn2_native_periodic_short_range_cuda(
    const Gfn2NativePeriodicShortRangeDeviceBatch& batch,
    const Gfn2NativePeriodicShortRangeDeviceWorkspace& workspace, double* coordination_numbers,
    double* repulsion_energies, double* repulsion_gradients, double* repulsion_strain,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const auto& topology = batch.topology;
  if (topology.batch_size <= 0 || topology.total_atoms < 0 || topology.total_translations <= 0 ||
      topology.batch_size > std::numeric_limits<unsigned int>::max() ||
      topology.atom_offset_count != topology.batch_size + 1 ||
      topology.translation_offset_count != topology.batch_size + 1 ||
      topology.cell_elements != topology.batch_size * 9 ||
      topology.periodic_axes_elements != topology.batch_size || topology.plan_token == 0u ||
      topology.atom_offsets == nullptr || topology.cell_matrices == nullptr ||
      topology.periodic_axes == nullptr || topology.translation_offsets == nullptr ||
      topology.translations == nullptr || batch.atomic_number_elements != topology.total_atoms ||
      batch.position_elements < 0 || batch.covalent_radius_elements != topology.total_atoms ||
      workspace.plan_token != topology.plan_token || workspace.wrapped_position_elements < 0 ||
      workspace.coordination_elements != topology.total_atoms ||
      workspace.repulsion_energy_elements != topology.batch_size ||
      workspace.repulsion_gradient_elements < 0 || workspace.repulsion_strain_elements < 0 ||
      coordination_numbers == nullptr || repulsion_energies == nullptr ||
      repulsion_gradients == nullptr || repulsion_strain == nullptr || system_errors == nullptr ||
      device_error == nullptr) {
    return cudaErrorInvalidValue;
  }

  std::int64_t atom_coordinate_elements = 0;
  std::int64_t strain_elements = 0;
  if (!checked_multiply(topology.total_atoms, 3, &atom_coordinate_elements) ||
      !checked_multiply(topology.batch_size, 9, &strain_elements) ||
      batch.position_elements != atom_coordinate_elements ||
      workspace.wrapped_position_elements != atom_coordinate_elements ||
      workspace.repulsion_gradient_elements != atom_coordinate_elements ||
      workspace.repulsion_strain_elements != strain_elements) {
    return cudaErrorInvalidValue;
  }

  std::array<AddressRange, 16> ranges{};
  ranges[0] = {};
  const bool ranges_ok =
      make_range(topology.atom_offsets, topology.atom_offset_count, &ranges[0]) &&
      make_range(topology.cell_matrices, topology.cell_elements, &ranges[1]) &&
      make_range(topology.periodic_axes, topology.periodic_axes_elements, &ranges[2]) &&
      make_range(topology.translation_offsets, topology.translation_offset_count, &ranges[3]) &&
      make_range(topology.translations, topology.total_translations, &ranges[4]) &&
      make_range(batch.atomic_numbers, batch.atomic_number_elements, &ranges[5]) &&
      make_range(batch.positions, batch.position_elements, &ranges[6]) &&
      make_range(batch.covalent_radii, batch.covalent_radius_elements, &ranges[7]) &&
      make_range(workspace.wrapped_positions, workspace.wrapped_position_elements, &ranges[8]) &&
      make_range(workspace.coordination, workspace.coordination_elements, &ranges[9]) &&
      make_range(workspace.repulsion_energies, workspace.repulsion_energy_elements, &ranges[10]) &&
      make_range(workspace.repulsion_gradients, workspace.repulsion_gradient_elements,
                 &ranges[11]) &&
      make_range(workspace.repulsion_strain, workspace.repulsion_strain_elements, &ranges[12]) &&
      make_range(coordination_numbers, topology.total_atoms, &ranges[13]) &&
      make_range(repulsion_energies, topology.batch_size, &ranges[14]) &&
      make_range(repulsion_gradients, atom_coordinate_elements, &ranges[15]);
  if (!ranges_ok) {
    return cudaErrorInvalidValue;
  }
  AddressRange strain_range{};
  AddressRange errors_range{};
  AddressRange device_error_range{};
  if (!make_range(repulsion_strain, strain_elements, &strain_range) ||
      !make_range(system_errors, topology.batch_size, &errors_range) ||
      !make_range(device_error, 1, &device_error_range) ||
      overlaps(errors_range, device_error_range) || overlaps(strain_range, errors_range) ||
      overlaps(strain_range, device_error_range)) {
    return cudaErrorInvalidValue;
  }
  /* The evaluator normally writes distinct output arrays, but the production
   * preprocessing transaction intentionally reuses the native workspace as
   * the repulsion outlet.  Permit only those exact workspace/output aliases;
   * all other read/write and output/output overlaps remain rejected. */
  const std::array<AddressRange, 5> scratch_writes{ranges[8], ranges[9], ranges[10], ranges[11],
                                                   ranges[12]};
  for (const auto& write : scratch_writes) {
    const bool exact_strain_alias =
        write.begin == strain_range.begin && write.end == strain_range.end;
    if ((!exact_strain_alias && overlaps(write, strain_range)) || overlaps(write, errors_range) ||
        overlaps(write, device_error_range)) {
      return cudaErrorInvalidValue;
    }
  }
  for (std::size_t read = 0u; read < 8u; ++read) {
    for (const auto& write : scratch_writes) {
      if (overlaps(ranges[read], write)) {
        return cudaErrorInvalidValue;
      }
    }
    if (overlaps(ranges[read], strain_range) || overlaps(ranges[read], errors_range) ||
        overlaps(ranges[read], device_error_range)) {
      return cudaErrorInvalidValue;
    }
  }
  const std::array<AddressRange, 3> outputs{ranges[13], ranges[14], ranges[15]};
  const std::array<AddressRange, 3> aliases{ranges[9], ranges[10], ranges[11]};
  for (std::size_t output = 0u; output < outputs.size(); ++output) {
    if (overlaps(outputs[output], strain_range) || overlaps(outputs[output], errors_range) ||
        overlaps(outputs[output], device_error_range)) {
      return cudaErrorInvalidValue;
    }
    for (std::size_t read = 0u; read < 8u; ++read) {
      if (overlaps(outputs[output], ranges[read])) {
        return cudaErrorInvalidValue;
      }
    }
    for (std::size_t other = output + 1u; other < outputs.size(); ++other) {
      if (overlaps(outputs[output], outputs[other])) {
        return cudaErrorInvalidValue;
      }
    }
    for (std::size_t scratch = 0u; scratch < aliases.size(); ++scratch) {
      if (overlaps(outputs[output], aliases[scratch]) && output != scratch) {
        return cudaErrorInvalidValue;
      }
    }
  }

  native_periodic_short_range_kernel<<<static_cast<unsigned int>(topology.batch_size),
                                       kThreadsPerBlock, 0, stream>>>(
      batch, workspace, coordination_numbers, repulsion_energies, repulsion_gradients,
      repulsion_strain, system_errors, device_error);
  return cudaGetLastError();
}

cudaError_t add_gfn2_native_periodic_coordination_vjp_cuda(
    const Gfn2NativePeriodicShortRangeDeviceBatch& batch,
    const Gfn2NativePeriodicShortRangeDeviceWorkspace& workspace, const double* dE_dcn,
    double* gradients, double* gradient_scratch, std::int64_t gradient_elements,
    std::uint32_t* sequence_active, std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  const auto& topology = batch.topology;
  std::int64_t coordinate_elements = 0;
  if (topology.batch_size <= 0 || topology.total_atoms < 0 || topology.total_translations <= 0 ||
      topology.batch_size > std::numeric_limits<unsigned int>::max() ||
      topology.atom_offset_count != topology.batch_size + 1 ||
      topology.translation_offset_count != topology.batch_size + 1 ||
      topology.cell_elements != topology.batch_size * 9 ||
      topology.periodic_axes_elements != topology.batch_size || topology.plan_token == 0u ||
      topology.atom_offsets == nullptr || topology.cell_matrices == nullptr ||
      topology.periodic_axes == nullptr || topology.translation_offsets == nullptr ||
      topology.translations == nullptr || batch.position_elements < 0 ||
      batch.covalent_radius_elements != topology.total_atoms ||
      workspace.plan_token != topology.plan_token ||
      workspace.wrapped_position_elements != topology.total_atoms * 3 ||
      gradient_elements != topology.total_atoms * 3 || dE_dcn == nullptr || gradients == nullptr ||
      gradient_scratch == nullptr || sequence_active == nullptr || system_errors == nullptr ||
      device_error == nullptr || !checked_multiply(topology.total_atoms, 3, &coordinate_elements) ||
      batch.position_elements != coordinate_elements ||
      (batch.active_mask == nullptr ? batch.active_mask_elements != 0
                                    : batch.active_mask_elements != topology.batch_size)) {
    return cudaErrorInvalidValue;
  }

  std::array<AddressRange, 11> ranges{};
  if (!make_range(batch.topology.atom_offsets, batch.topology.atom_offset_count, &ranges[0]) ||
      !make_range(batch.topology.cell_matrices, batch.topology.cell_elements, &ranges[1]) ||
      !make_range(batch.topology.periodic_axes, batch.topology.periodic_axes_elements,
                  &ranges[2]) ||
      !make_range(batch.topology.translation_offsets, batch.topology.translation_offset_count,
                  &ranges[3]) ||
      !make_range(batch.topology.translations, batch.topology.total_translations, &ranges[4]) ||
      !make_range(batch.positions, batch.position_elements, &ranges[5]) ||
      !make_range(batch.covalent_radii, batch.covalent_radius_elements, &ranges[6]) ||
      !make_range(batch.active_mask, batch.active_mask_elements, &ranges[7]) ||
      !make_range(workspace.wrapped_positions, workspace.wrapped_position_elements, &ranges[8]) ||
      !make_range(dE_dcn, topology.total_atoms, &ranges[9]) ||
      !make_range(gradients, gradient_elements, &ranges[10])) {
    return cudaErrorInvalidValue;
  }
  AddressRange scratch_range{}, sequence_range{}, errors_range{}, device_error_range{};
  if (!make_range(gradient_scratch, gradient_elements, &scratch_range) ||
      !make_range(sequence_active, 1, &sequence_range) ||
      !make_range(system_errors, topology.batch_size, &errors_range) ||
      !make_range(device_error, 1, &device_error_range)) {
    return cudaErrorInvalidValue;
  }
  for (std::size_t read = 0u; read < ranges.size(); ++read) {
    if (overlaps(ranges[read], scratch_range) || overlaps(ranges[read], sequence_range) ||
        overlaps(ranges[read], errors_range) || overlaps(ranges[read], device_error_range)) {
      return cudaErrorInvalidValue;
    }
  }
  if (overlaps(gradients == nullptr ? AddressRange{} : ranges[10], scratch_range) ||
      overlaps(errors_range, device_error_range) || overlaps(sequence_range, errors_range) ||
      overlaps(sequence_range, device_error_range) || overlaps(scratch_range, errors_range) ||
      overlaps(scratch_range, device_error_range)) {
    return cudaErrorInvalidValue;
  }

  capture_native_coordination_sequence_kernel<<<1, 1, 0, stream>>>(device_error, sequence_active);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  native_periodic_coordination_vjp_kernel<<<static_cast<unsigned int>(topology.batch_size),
                                            kThreadsPerBlock, 0, stream>>>(
      batch, workspace, dE_dcn, gradients, gradient_scratch, sequence_active, system_errors,
      device_error);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
