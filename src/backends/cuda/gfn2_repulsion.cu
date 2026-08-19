#include <cmath>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/gfn2_parameters.cuh"
#include "backends/cuda/gfn2_repulsion.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr double kCutoffSquaredBohr = 25.0 * 25.0;
constexpr double kGfn2MinimumDistanceSquared = 1.0e-24;
constexpr double kGfn1MinimumDistanceSquared = 1.0e-12;

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

__device__ void record_error(std::uint32_t* device_error, Gfn2RepulsionDeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2RepulsionDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__global__ void gfn2_repulsion_kernel(Gfn2RepulsionDeviceBatch batch, double* energies,
                                      double* forces, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ std::int64_t begin;
  __shared__ std::int64_t end;
  __shared__ int valid;
  __shared__ double energy_sums[kThreadsPerBlock];

  if (threadIdx.x == 0) {
    begin = batch.atom_offsets[system];
    end = batch.atom_offsets[system + 1];
    valid = begin >= 0 && begin <= end && end <= batch.total_atoms && (system != 0 || begin == 0) &&
            (system + 1 != batch.batch_size || end == batch.total_atoms);
    if (valid == 0) {
      record_error(device_error, Gfn2RepulsionDeviceError::kInvalidOffsets);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  /*
   * Validation is a separate phase so every atom is checked, including atoms
   * in empty/singleton systems and atoms whose pairs lie outside the cutoff.
   */
  for (std::int64_t atom = begin + threadIdx.x; atom < end; atom += blockDim.x) {
    const std::int32_t atomic_number = batch.atomic_numbers[atom];
    if (batch.model == XtbModelFlavor::kGfn1) {
      if (atomic_number < 1 || batch.sqrt_alpha == nullptr || batch.effective_charge == nullptr ||
          !(batch.sqrt_alpha[atom] > 0.0) || !(batch.effective_charge[atom] > 0.0) ||
          !isfinite(batch.sqrt_alpha[atom]) || !isfinite(batch.effective_charge[atom])) {
        record_error(device_error, Gfn2RepulsionDeviceError::kInvalidAtomicNumberOrParameter);
        atomicExch(&valid, 0);
      }
    } else {
      if (atomic_number < 1 ||
          atomic_number > static_cast<std::int32_t>(parameters::gfn2::kElementCount)) {
        record_error(device_error, Gfn2RepulsionDeviceError::kInvalidAtomicNumberOrParameter);
        atomicExch(&valid, 0);
        continue;
      }
      const parameters::gfn2::ElementParameters element = g_gfn2_elements[atomic_number - 1];
      if (element.atomic_number != atomic_number || !(element.arep > 0.0) ||
          !(element.zeff > 0.0) || !isfinite(element.arep) || !isfinite(element.zeff)) {
        record_error(device_error, Gfn2RepulsionDeviceError::kInvalidAtomicNumberOrParameter);
        atomicExch(&valid, 0);
      }
    }

    const std::int64_t coordinate = atom * 3;
    if (!isfinite(batch.positions[coordinate]) || !isfinite(batch.positions[coordinate + 1]) ||
        !isfinite(batch.positions[coordinate + 2])) {
      record_error(device_error, Gfn2RepulsionDeviceError::kNonfinitePosition);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  double local_energy = 0.0;
  for (std::int64_t first = begin + 1 + threadIdx.x; first < end; first += blockDim.x) {
    const std::int32_t first_atomic_number = batch.atomic_numbers[first];
    const parameters::gfn2::ElementParameters first_element =
        batch.model == XtbModelFlavor::kGfn2 ? g_gfn2_elements[first_atomic_number - 1]
                                             : parameters::gfn2::ElementParameters{};
    const double first_sqrt_alpha =
        batch.model == XtbModelFlavor::kGfn1 ? batch.sqrt_alpha[first] : sqrt(first_element.arep);
    const std::int64_t first_coordinate = first * 3;
    const double first_x = batch.positions[first_coordinate];
    const double first_y = batch.positions[first_coordinate + 1];
    const double first_z = batch.positions[first_coordinate + 2];
    double first_fx = 0.0;
    double first_fy = 0.0;
    double first_fz = 0.0;

    for (std::int64_t second = begin; second < first; ++second) {
      const std::int64_t second_coordinate = second * 3;
      const double dx = first_x - batch.positions[second_coordinate];
      const double dy = first_y - batch.positions[second_coordinate + 1];
      const double dz = first_z - batch.positions[second_coordinate + 2];
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      const double minimum_distance_squared = batch.model == XtbModelFlavor::kGfn1
                                                  ? kGfn1MinimumDistanceSquared
                                                  : kGfn2MinimumDistanceSquared;
      if ((batch.model == XtbModelFlavor::kGfn1 && distance_squared < minimum_distance_squared) ||
          (batch.model == XtbModelFlavor::kGfn2 && distance_squared <= minimum_distance_squared)) {
        /* GFN1 follows tblite and silently excludes diagonal/self-image style
         * zero-distance pairs. GFN2 retains its released coincident-atom error. */
        if (batch.model == XtbModelFlavor::kGfn2) {
          record_error(device_error, Gfn2RepulsionDeviceError::kCoincidentAtoms);
        }
        continue;
      }
      if (distance_squared > kCutoffSquaredBohr) {
        continue;
      }

      const std::int32_t second_atomic_number = batch.atomic_numbers[second];
      const parameters::gfn2::ElementParameters second_element =
          batch.model == XtbModelFlavor::kGfn2 ? g_gfn2_elements[second_atomic_number - 1]
                                               : parameters::gfn2::ElementParameters{};
      const double distance = sqrt(distance_squared);
      const bool light_pair = batch.model == XtbModelFlavor::kGfn2 && first_atomic_number <= 2 &&
                              second_atomic_number <= 2;
      const double exponent =
          batch.model == XtbModelFlavor::kGfn1
              ? 1.5
              : (light_pair ? g_gfn2_global.repulsion_klight : g_gfn2_global.repulsion_kexp);
      /* GFN1 always uses r^1.5. GFN2 preserves its H/He r branch. */
      const double distance_power = batch.model == XtbModelFlavor::kGfn1 || !light_pair
                                        ? distance * sqrt(distance)
                                        : distance;
      const double second_sqrt_alpha = batch.model == XtbModelFlavor::kGfn1
                                           ? batch.sqrt_alpha[second]
                                           : sqrt(second_element.arep);
      const double pair_alpha = first_sqrt_alpha * second_sqrt_alpha;
      const double pair_charge =
          batch.model == XtbModelFlavor::kGfn1
              ? batch.effective_charge[first] * batch.effective_charge[second]
              : first_element.zeff * second_element.zeff;
      const double pair_energy = pair_charge * exp(-pair_alpha * distance_power) / distance;
      local_energy += pair_energy;

      if (forces != nullptr) {
        const double force_scale =
            (pair_alpha * exponent * distance_power + 1.0) * pair_energy / distance_squared;
        const double fx = force_scale * dx;
        const double fy = force_scale * dy;
        const double fz = force_scale * dz;
        first_fx += fx;
        first_fy += fy;
        first_fz += fz;
        atomic_add_fp64(&forces[second_coordinate], -fx);
        atomic_add_fp64(&forces[second_coordinate + 1], -fy);
        atomic_add_fp64(&forces[second_coordinate + 2], -fz);
      }
    }

    if (forces != nullptr) {
      atomic_add_fp64(&forces[first_coordinate], first_fx);
      atomic_add_fp64(&forces[first_coordinate + 1], first_fy);
      atomic_add_fp64(&forces[first_coordinate + 2], first_fz);
    }
  }

  energy_sums[threadIdx.x] = local_energy;
  __syncthreads();
  for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      energy_sums[threadIdx.x] += energy_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomic_add_fp64(&energies[system], energy_sums[0]);
  }
}

}  // namespace

cudaError_t add_gfn2_repulsion_cuda(const Gfn2RepulsionDeviceBatch& batch, double* energies,
                                    double* forces, std::uint32_t* device_error,
                                    cudaStream_t stream) noexcept {
  if (!valid_xtb_model_flavor(batch.model) || batch.batch_size <= 0 || batch.total_atoms <= 0 ||
      batch.atom_offsets == nullptr || batch.atomic_numbers == nullptr ||
      batch.positions == nullptr || energies == nullptr || device_error == nullptr ||
      (batch.model == XtbModelFlavor::kGfn1 &&
       (batch.sqrt_alpha_elements != batch.total_atoms ||
        batch.effective_charge_elements != batch.total_atoms)) ||
      (batch.model == XtbModelFlavor::kGfn2 &&
       (batch.sqrt_alpha != nullptr || batch.effective_charge != nullptr ||
        batch.sqrt_alpha_elements != 0 || batch.effective_charge_elements != 0))) {
    return cudaErrorInvalidValue;
  }
  /* CUDA devices expose at most INT_MAX blocks in grid.x. */
  if (static_cast<std::uint64_t>(batch.batch_size) >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return cudaErrorInvalidConfiguration;
  }
  if (batch.total_atoms > std::numeric_limits<std::int64_t>::max() / 3) {
    return cudaErrorInvalidValue;
  }

  std::array<AddressRange, 5> reads{};
  AddressRange energy_range{};
  AddressRange force_range{};
  AddressRange error_range{};
  const std::int64_t coordinates = batch.total_atoms * 3;
  if (!make_range(batch.atom_offsets, batch.batch_size + 1, &reads[0]) ||
      !make_range(batch.atomic_numbers, batch.total_atoms, &reads[1]) ||
      !make_range(batch.positions, coordinates, &reads[2]) ||
      !make_range(batch.sqrt_alpha, batch.sqrt_alpha_elements, &reads[3]) ||
      !make_range(batch.effective_charge, batch.effective_charge_elements, &reads[4]) ||
      !make_range(energies, batch.batch_size, &energy_range) ||
      !make_range(forces, forces == nullptr ? 0 : coordinates, &force_range) ||
      !make_range(device_error, 1, &error_range) || overlaps(energy_range, force_range) ||
      overlaps(energy_range, error_range) || overlaps(force_range, error_range)) {
    return cudaErrorInvalidValue;
  }
  for (const AddressRange& read : reads) {
    if (overlaps(read, energy_range) || overlaps(read, force_range) ||
        overlaps(read, error_range)) {
      return cudaErrorInvalidValue;
    }
  }

  const cudaError_t reset_status = cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
  if (reset_status != cudaSuccess) {
    return reset_status;
  }

  gfn2_repulsion_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                          stream>>>(batch, energies, forces, device_error);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
