#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/gfn2_integral_force.cuh"
#include "backends/cuda/gfn2_integrals.cuh"
#include "backends/cuda/periodic_wrap.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 64;
constexpr int kCompactH0ThreadsPerBlock = 16;
constexpr std::int64_t kMaximumInt64 = 9223372036854775807LL;
constexpr double kSqrtThree = 1.732050807568877293527446341505872367;
constexpr double kSqrtPiCubed = 5.5683279968317061;
constexpr double kMaximumCoordinate = 3.3519519824856493e153;
constexpr int kMaximumCartesianBlock = 36;
constexpr int kMultipoleComponents = 9;

struct SystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t shell_begin;
  std::int64_t shell_end;
  std::int64_t orbital_begin;
  std::int64_t orbital_end;
  std::int64_t matrix_begin;
  std::int64_t matrix_end;
  std::int64_t shell_pair_begin;
  std::int64_t shell_pair_end;
};

__device__ bool sequence_is_active(const Gfn2IntegralDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess);
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2IntegralDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system,
                static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess),
                code) == static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess)) {
    atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess), code);
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
                                        Gfn2IntegralDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

/*
 * One block validates one member. Every outer offset is range-checked before
 * subtraction, multiplication, or use as an index, including adversarial
 * INT64_MIN/INT64_MAX device values.
 */
__global__ void topology_preflight_kernel(Gfn2IntegralDeviceBatch batch, const double* positions,
                                          Gfn2IntegralDeviceWorkspace workspace,
                                          std::uint32_t* system_errors,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system)) {
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
    ranges.shell_pair_begin = batch.shell_pair_offsets[system];
    ranges.shell_pair_end = batch.shell_pair_offsets[system + 1];

    if (!valid_closed_range(ranges.atom_begin, ranges.atom_end, batch.total_atoms) ||
        !valid_closed_range(ranges.shell_begin, ranges.shell_end, batch.total_shells) ||
        !valid_closed_range(ranges.orbital_begin, ranges.orbital_end, batch.total_orbitals) ||
        !valid_closed_range(ranges.matrix_begin, ranges.matrix_end, batch.total_matrix_elements) ||
        !valid_closed_range(ranges.shell_pair_begin, ranges.shell_pair_end,
                            batch.total_shell_pair_elements)) {
      valid = 0;
    }
    if (valid != 0) {
      const std::int64_t shells = ranges.shell_end - ranges.shell_begin;
      const std::int64_t orbitals = ranges.orbital_end - ranges.orbital_begin;
      std::int64_t expected_pairs = 0;
      std::int64_t expected_matrix = 0;
      valid = shells <= batch.maximum_system_shells && checked_square(shells, &expected_pairs) &&
              checked_square(orbitals, &expected_matrix) &&
              ranges.shell_pair_end - ranges.shell_pair_begin == expected_pairs &&
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
              ranges.matrix_begin == 0 && ranges.shell_pair_begin == 0;
    }
    if (valid != 0 && system + 1 == batch.batch_size) {
      valid = ranges.atom_end == batch.total_atoms && ranges.shell_end == batch.total_shells &&
              ranges.orbital_end == batch.total_orbitals &&
              ranges.matrix_end == batch.total_matrix_elements &&
              ranges.shell_pair_end == batch.total_shell_pair_elements;
    }
    if (valid == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidOffsets);
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
                          Gfn2IntegralDeviceError::kInvalidOffsets);
      atomicExch(&valid, 0);
    }
    const std::int64_t coordinate = atom * 3;
    for (int axis = 0; axis < 3; ++axis) {
      const double value = positions[coordinate + axis];
      if (!isfinite(value)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2IntegralDeviceError::kNonfinitePosition);
        atomicExch(&valid, 0);
      } else if (fabs(value) > kMaximumCoordinate) {
        record_system_error(system_errors, system, device_error,
                            Gfn2IntegralDeviceError::kCoordinateDifferenceOverflow);
        atomicExch(&valid, 0);
      }
    }
  }

  for (std::int64_t shell = ranges.shell_begin + threadIdx.x; shell < ranges.shell_end;
       shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const std::uint8_t angular_momentum = batch.angular_momenta[shell];
    const std::int64_t orbital_begin = batch.shell_orbital_offsets[shell];
    const std::int64_t orbital_end = batch.shell_orbital_offsets[shell + 1];
    const std::int64_t primitive_begin = batch.shell_primitive_offsets[shell];
    const std::int64_t primitive_end = batch.shell_primitive_offsets[shell + 1];
    bool shell_valid = atom >= ranges.atom_begin && atom < ranges.atom_end;
    if (shell_valid) {
      shell_valid =
          shell >= batch.atom_shell_offsets[atom] && shell < batch.atom_shell_offsets[atom + 1];
    }
    shell_valid =
        shell_valid && angular_momentum <= 2u &&
        valid_closed_range(orbital_begin, orbital_end, batch.total_orbitals) &&
        orbital_begin >= ranges.orbital_begin && orbital_end <= ranges.orbital_end &&
        orbital_end - orbital_begin == 2 * static_cast<std::int64_t>(angular_momentum) + 1 &&
        valid_closed_range(primitive_begin, primitive_end, batch.total_primitives) &&
        primitive_begin < primitive_end;
    if (!shell_valid) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidShellMetadata);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  const std::int64_t primitive_begin = batch.shell_primitive_offsets[ranges.shell_begin];
  const std::int64_t primitive_end = batch.shell_primitive_offsets[ranges.shell_end];
  for (std::int64_t primitive = primitive_begin + threadIdx.x; primitive < primitive_end;
       primitive += blockDim.x) {
    const double exponent = batch.primitive_exponents[primitive];
    const double coefficient = batch.primitive_coefficients[primitive];
    if (!(exponent > 0.0) || !isfinite(exponent) || !isfinite(coefficient)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidPrimitiveData);
      atomicExch(&valid, 0);
    }
  }
}

__device__ int cartesian_count(std::uint8_t angular_momentum) {
  const int l = static_cast<int>(angular_momentum);
  return (l + 1) * (l + 2) / 2;
}

__device__ int spherical_count(std::uint8_t angular_momentum) {
  return 2 * static_cast<int>(angular_momentum) + 1;
}

__device__ void cartesian_exponent(std::uint8_t angular_momentum, int function, int* x, int* y,
                                   int* z) {
  if (angular_momentum == 0u) {
    *x = 0;
    *y = 0;
    *z = 0;
  } else if (angular_momentum == 1u) {
    *x = function == 0 ? 1 : 0;
    *y = function == 1 ? 1 : 0;
    *z = function == 2 ? 1 : 0;
  } else {
    constexpr int exponents[6][3] = {{2, 0, 0}, {1, 1, 0}, {1, 0, 1},
                                     {0, 2, 0}, {0, 1, 1}, {0, 0, 2}};
    *x = exponents[function][0];
    *y = exponents[function][1];
    *z = exponents[function][2];
  }
}

__device__ void multipole_power(int component, int* x, int* y, int* z) {
  constexpr int powers[kMultipoleComponents][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                                                   {2, 0, 0}, {1, 1, 0}, {0, 2, 0},
                                                   {1, 0, 1}, {0, 1, 1}, {0, 0, 2}};
  *x = powers[component][0];
  *y = powers[component][1];
  *z = powers[component][2];
}

/* Compact task contents are plan-owned, but guard their indices so a damaged
 * immutable arena fails closed instead of becoming an unchecked device read. */
__device__ bool resolve_shell_pair_task(const Gfn2IntegralDeviceBatch& batch,
                                        const Gfn2IntegralShellPairTask* tasks,
                                        std::int64_t global_pair, std::int64_t maximum_pair_blocks,
                                        std::uint32_t* system_errors, std::uint32_t* device_error,
                                        std::int64_t& system, std::int64_t& local_pair,
                                        std::int64_t& task_bra_shell,
                                        std::int64_t& task_ket_shell) {
  system = tasks == nullptr ? global_pair / maximum_pair_blocks
                            : static_cast<std::int64_t>(tasks[global_pair].system);
  local_pair = tasks == nullptr ? global_pair - system * maximum_pair_blocks
                                : static_cast<std::int64_t>(tasks[global_pair].local_pair);
  task_bra_shell = tasks == nullptr ? -1 : static_cast<std::int64_t>(tasks[global_pair].bra_shell);
  task_ket_shell = tasks == nullptr ? -1 : static_cast<std::int64_t>(tasks[global_pair].ket_shell);
  if (system >= 0 && system < batch.batch_size) {
    return true;
  }
  if (threadIdx.x == 0) {
    for (std::int64_t member = 0; member < batch.batch_size; ++member) {
      record_system_error(system_errors, member, device_error,
                          Gfn2IntegralDeviceError::kInvalidShellMetadata);
    }
  }
  return false;
}

/* tblite real-spherical rows in [-l,...,+l] and CCA Cartesian columns. */
__device__ double spherical_coefficient(std::uint8_t angular_momentum, int spherical,
                                        int cartesian) {
  if (angular_momentum == 0u) {
    return spherical == 0 && cartesian == 0 ? 1.0 : 0.0;
  }
  if (angular_momentum == 1u) {
    const int selected = spherical == 0 ? 1 : (spherical == 1 ? 2 : 0);
    return cartesian == selected ? 1.0 : 0.0;
  }
  if (spherical == 0) {
    return cartesian == 1 ? kSqrtThree : 0.0;
  }
  if (spherical == 1) {
    return cartesian == 4 ? kSqrtThree : 0.0;
  }
  if (spherical == 2) {
    return cartesian == 0 || cartesian == 3 ? -0.5 : (cartesian == 5 ? 1.0 : 0.0);
  }
  if (spherical == 3) {
    return cartesian == 2 ? kSqrtThree : 0.0;
  }
  return cartesian == 0 ? 0.5 * kSqrtThree : (cartesian == 3 ? -0.5 * kSqrtThree : 0.0);
}

__device__ void make_axis_overlap(double product_minus_i, double product_minus_j,
                                  double inverse_twice_sum, int maximum_a, int maximum_b,
                                  double overlap[6][3]) {
#pragma unroll
  for (int a = 0; a < 6; ++a) {
#pragma unroll
    for (int b = 0; b < 3; ++b) {
      overlap[a][b] = 0.0;
    }
  }
  overlap[0][0] = 1.0;
  for (int a = 1; a <= maximum_a; ++a) {
    overlap[a][0] = product_minus_i * overlap[a - 1][0];
    if (a > 1) {
      overlap[a][0] += static_cast<double>(a - 1) * inverse_twice_sum * overlap[a - 2][0];
    }
  }
  for (int b = 1; b <= maximum_b; ++b) {
    overlap[0][b] = product_minus_j * overlap[0][b - 1];
    if (b > 1) {
      overlap[0][b] += static_cast<double>(b - 1) * inverse_twice_sum * overlap[0][b - 2];
    }
    for (int a = 1; a <= maximum_a; ++a) {
      overlap[a][b] = product_minus_i * overlap[a - 1][b] +
                      static_cast<double>(b) * inverse_twice_sum * overlap[a - 1][b - 1];
      if (a > 1) {
        overlap[a][b] += static_cast<double>(a - 1) * inverse_twice_sum * overlap[a - 2][b];
      }
    }
  }
}

/*
 * Evaluate one unique shell pair. Cartesian primitive contractions are staged
 * in shared memory before the spherical transform, preserving the CPU
 * contraction order while letting up to 36 lanes own independent functions.
 */
__global__ void integral_shell_pair_kernel(Gfn2IntegralDeviceBatch batch, const double* positions,
                                           Gfn2IntegralDeviceWorkspace workspace,
                                           std::uint32_t* system_errors,
                                           std::uint32_t* device_error,
                                           const Gfn2IntegralShellPairTask* tasks,
                                           std::int64_t maximum_pair_blocks) {
  const std::int64_t global_pair = static_cast<std::int64_t>(blockIdx.x);
  std::int64_t system = 0;
  std::int64_t local_pair = 0;
  std::int64_t task_bra_shell = -1;
  std::int64_t task_ket_shell = -1;
  if (!resolve_shell_pair_task(batch, tasks, global_pair, maximum_pair_blocks, system_errors,
                               device_error, system, local_pair, task_bra_shell, task_ket_shell)) {
    return;
  }
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system)) {
    return;
  }

  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t shells = shell_end - shell_begin;
  const bool compact_task_valid =
      tasks == nullptr ||
      (task_bra_shell >= shell_begin && task_bra_shell < shell_end &&
       task_ket_shell >= shell_begin && task_ket_shell < shell_end &&
       local_pair == (task_bra_shell - shell_begin) * shells + task_ket_shell - shell_begin);
  if (local_pair >= shells * shells || !compact_task_valid) {
    if (tasks != nullptr && threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidShellMetadata);
    }
    return;
  }
  const std::int64_t bra_shell =
      tasks == nullptr ? shell_begin + local_pair / shells : task_bra_shell;
  const std::int64_t ket_shell =
      tasks == nullptr ? shell_begin + local_pair % shells : task_ket_shell;
  const std::int64_t bra_atom = batch.shell_to_atom[bra_shell];
  const std::int64_t ket_atom = batch.shell_to_atom[ket_shell];
  if (bra_atom > ket_atom || (bra_atom == ket_atom && bra_shell > ket_shell)) {
    return;
  }

  const std::uint8_t bra_l = batch.angular_momenta[bra_shell];
  const std::uint8_t ket_l = batch.angular_momenta[ket_shell];
  const int bra_cartesian_count = cartesian_count(bra_l);
  const int ket_cartesian_count = cartesian_count(ket_l);
  const int cartesian_block_size = bra_cartesian_count * ket_cartesian_count;
  const int bra_spherical_count = spherical_count(bra_l);
  const int ket_spherical_count = spherical_count(ket_l);
  const int spherical_block_size = bra_spherical_count * ket_spherical_count;
  const double vector[3] = {positions[ket_atom * 3] - positions[bra_atom * 3],
                            positions[ket_atom * 3 + 1] - positions[bra_atom * 3 + 1],
                            positions[ket_atom * 3 + 2] - positions[bra_atom * 3 + 2]};
  if (!isfinite(vector[0]) || !isfinite(vector[1]) || !isfinite(vector[2])) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kCoordinateDifferenceOverflow);
    }
    return;
  }
  const double distance_squared =
      vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
  if (!isfinite(distance_squared)) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kCoordinateDifferenceOverflow);
    }
    return;
  }

  const bool multipoles_enabled = batch.model == XtbModelFlavor::kGfn2;
  __shared__ double cartesian_overlap[kMaximumCartesianBlock];
  __shared__ double cartesian_multipole[kMultipoleComponents * kMaximumCartesianBlock];
  const int cartesian_index = static_cast<int>(threadIdx.x);
  if (cartesian_index < cartesian_block_size) {
    const int bra_cartesian = cartesian_index / ket_cartesian_count;
    const int ket_cartesian = cartesian_index % ket_cartesian_count;
    int bra_power[3];
    int ket_power[3];
    cartesian_exponent(bra_l, bra_cartesian, &bra_power[0], &bra_power[1], &bra_power[2]);
    cartesian_exponent(ket_l, ket_cartesian, &ket_power[0], &ket_power[1], &ket_power[2]);
    double overlap_value = 0.0;
    double multipoles[kMultipoleComponents] = {};
    const std::int64_t bra_primitive_begin = batch.shell_primitive_offsets[bra_shell];
    const std::int64_t bra_primitive_end = batch.shell_primitive_offsets[bra_shell + 1];
    const std::int64_t ket_primitive_begin = batch.shell_primitive_offsets[ket_shell];
    const std::int64_t ket_primitive_end = batch.shell_primitive_offsets[ket_shell + 1];

    for (std::int64_t ket_primitive = ket_primitive_begin; ket_primitive < ket_primitive_end;
         ++ket_primitive) {
      const double ket_alpha = batch.primitive_exponents[ket_primitive];
      for (std::int64_t bra_primitive = bra_primitive_begin; bra_primitive < bra_primitive_end;
           ++bra_primitive) {
        const double bra_alpha = batch.primitive_exponents[bra_primitive];
        const double alpha_sum = ket_alpha + bra_alpha;
        const double inverse_sum = 1.0 / alpha_sum;
        const double product_exponent = ket_alpha * bra_alpha * distance_squared * inverse_sum;
        if (product_exponent > batch.integral_cutoff) {
          continue;
        }
        const double sqrt_inverse_sum = sqrt(inverse_sum);
        const double primitive_prefactor = exp(-product_exponent) * kSqrtPiCubed *
                                           sqrt_inverse_sum * sqrt_inverse_sum * sqrt_inverse_sum *
                                           batch.primitive_coefficients[ket_primitive] *
                                           batch.primitive_coefficients[bra_primitive];
        const double inverse_twice_sum = 0.5 * inverse_sum;
        double axis[3][6][3];
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          const double product_minus_i = -vector[coordinate] * bra_alpha * inverse_sum;
          const double product_minus_j = +vector[coordinate] * ket_alpha * inverse_sum;
          make_axis_overlap(product_minus_i, product_minus_j, inverse_twice_sum,
                            static_cast<int>(ket_l) + 2, static_cast<int>(bra_l), axis[coordinate]);
        }
        const double x = axis[0][ket_power[0]][bra_power[0]];
        const double y = axis[1][ket_power[1]][bra_power[1]];
        const double z = axis[2][ket_power[2]][bra_power[2]];
        overlap_value += primitive_prefactor * x * y * z;
        if (multipoles_enabled) {
          for (int component = 0; component < kMultipoleComponents; ++component) {
            int moment_power[3];
            multipole_power(component, &moment_power[0], &moment_power[1], &moment_power[2]);
            multipoles[component] += primitive_prefactor *
                                     axis[0][ket_power[0] + moment_power[0]][bra_power[0]] *
                                     axis[1][ket_power[1] + moment_power[1]][bra_power[1]] *
                                     axis[2][ket_power[2] + moment_power[2]][bra_power[2]];
          }
        }
      }
    }
    bool finite = isfinite(overlap_value);
    cartesian_overlap[cartesian_index] = overlap_value;
    if (multipoles_enabled) {
      for (int component = 0; component < kMultipoleComponents; ++component) {
        cartesian_multipole[component * kMaximumCartesianBlock + cartesian_index] =
            multipoles[component];
        finite = finite && isfinite(multipoles[component]);
      }
    }
    if (!finite) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kNonfiniteIntegralArithmetic);
    }
  }
  __syncthreads();
  if (!system_is_valid(system_errors, system)) {
    return;
  }

  const int spherical_index = static_cast<int>(threadIdx.x);
  if (spherical_index >= spherical_block_size) {
    return;
  }
  const int bra_ao = spherical_index / ket_spherical_count;
  const int ket_ao = spherical_index % ket_spherical_count;
  if (bra_shell == ket_shell && bra_ao > ket_ao) {
    return;
  }
  double overlap_value = 0.0;
  double raw_multipoles[kMultipoleComponents] = {};
  for (int bra_cartesian = 0; bra_cartesian < bra_cartesian_count; ++bra_cartesian) {
    const double bra_coefficient = spherical_coefficient(bra_l, bra_ao, bra_cartesian);
    if (bra_coefficient == 0.0) {
      continue;
    }
    for (int ket_cartesian = 0; ket_cartesian < ket_cartesian_count; ++ket_cartesian) {
      const double ket_coefficient = spherical_coefficient(ket_l, ket_ao, ket_cartesian);
      if (ket_coefficient == 0.0) {
        continue;
      }
      const int index = bra_cartesian * ket_cartesian_count + ket_cartesian;
      overlap_value += bra_coefficient * cartesian_overlap[index] * ket_coefficient;
      if (multipoles_enabled) {
        for (int component = 0; component < kMultipoleComponents; ++component) {
          raw_multipoles[component] +=
              bra_coefficient * cartesian_multipole[component * kMaximumCartesianBlock + index] *
              ket_coefficient;
        }
      }
    }
  }

  double dipole[3] = {raw_multipoles[0], raw_multipoles[1], raw_multipoles[2]};
  const double trace = 0.5 * (raw_multipoles[3] + raw_multipoles[5] + raw_multipoles[8]);
  double quadrupole[6] = {1.5 * raw_multipoles[3] - trace, 1.5 * raw_multipoles[4],
                          1.5 * raw_multipoles[5] - trace, 1.5 * raw_multipoles[6],
                          1.5 * raw_multipoles[7],         1.5 * raw_multipoles[8] - trace};
  bool finite = isfinite(overlap_value);
  if (multipoles_enabled) {
    for (double value : dipole) {
      finite = finite && isfinite(value);
    }
    for (double value : quadrupole) {
      finite = finite && isfinite(value);
    }
  }
  if (!finite) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kNonfiniteIntegralArithmetic);
    return;
  }

  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbital_count = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t bra_orbital = batch.shell_orbital_offsets[bra_shell] - orbital_begin + bra_ao;
  const std::int64_t ket_orbital = batch.shell_orbital_offsets[ket_shell] - orbital_begin + ket_ao;
  const std::int64_t forward = matrix_begin + bra_orbital * orbital_count + ket_orbital;
  const std::int64_t reverse = matrix_begin + ket_orbital * orbital_count + bra_orbital;
  workspace.overlap_scratch[forward] = overlap_value;
  workspace.overlap_scratch[reverse] = overlap_value;
  if (!multipoles_enabled) return;
  for (int component = 0; component < 3; ++component) {
    workspace.dipole_scratch[component * batch.total_matrix_elements + forward] = dipole[component];
  }
  for (int component = 0; component < 6; ++component) {
    workspace.quadrupole_scratch[component * batch.total_matrix_elements + forward] =
        quadrupole[component];
  }
  if (bra_shell == ket_shell) {
    for (int component = 0; component < 3; ++component) {
      workspace.dipole_scratch[component * batch.total_matrix_elements + reverse] =
          dipole[component];
    }
    for (int component = 0; component < 6; ++component) {
      workspace.quadrupole_scratch[component * batch.total_matrix_elements + reverse] =
          quadrupole[component];
    }
    return;
  }

  double shifted_dipole[3];
  for (int component = 0; component < 3; ++component) {
    shifted_dipole[component] = dipole[component] + vector[component] * overlap_value;
    workspace.dipole_scratch[component * batch.total_matrix_elements + reverse] =
        shifted_dipole[component];
  }
  const double shift[6] = {
      2.0 * vector[0] * dipole[0] + vector[0] * vector[0] * overlap_value,
      vector[0] * dipole[1] + vector[1] * dipole[0] + vector[0] * vector[1] * overlap_value,
      2.0 * vector[1] * dipole[1] + vector[1] * vector[1] * overlap_value,
      vector[0] * dipole[2] + vector[2] * dipole[0] + vector[0] * vector[2] * overlap_value,
      vector[1] * dipole[2] + vector[2] * dipole[1] + vector[1] * vector[2] * overlap_value,
      2.0 * vector[2] * dipole[2] + vector[2] * vector[2] * overlap_value};
  const double shift_trace = 0.5 * (shift[0] + shift[2] + shift[5]);
  for (int component = 0; component < 6; ++component) {
    const bool diagonal = component == 0 || component == 2 || component == 5;
    const double shifted =
        quadrupole[component] + 1.5 * shift[component] - (diagonal ? shift_trace : 0.0);
    if (!isfinite(shifted)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kNonfiniteIntegralArithmetic);
      return;
    }
    workspace.quadrupole_scratch[component * batch.total_matrix_elements + reverse] = shifted;
  }
}

__device__ void make_ss_moments(const double vector[3], double bra_alpha, double inverse_sum,
                                double moments[3][4]) {
  const double inverse_twice_sum = 0.5 * inverse_sum;
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    const double product_minus_i = -vector[coordinate] * bra_alpha * inverse_sum;
    moments[coordinate][0] = 1.0;
    for (int power = 1; power <= 3; ++power) {
      moments[coordinate][power] = product_minus_i * moments[coordinate][power - 1];
      if (power > 1) {
        moments[coordinate][power] +=
            static_cast<double>(power - 1) * inverse_twice_sum * moments[coordinate][power - 2];
      }
    }
  }
}

/* The generic shell-pair kernel assigns a complete 64-thread CTA to an ss
 * pair even though only one Cartesian/spherical lane carries work. Packing
 * independent immutable ss tasks removes that structural underfill without
 * changing the primitive contraction order within any task. */
__global__ void integral_ss_task_kernel(Gfn2IntegralDeviceBatch batch, const double* positions,
                                        Gfn2IntegralDeviceWorkspace workspace,
                                        std::uint32_t* system_errors, std::uint32_t* device_error,
                                        const Gfn2IntegralShellPairTask* tasks,
                                        std::int64_t task_count) {
  const std::int64_t task_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (task_index >= task_count || !sequence_is_active(workspace)) return;
  const Gfn2IntegralShellPairTask task = tasks[task_index];
  const std::int64_t system = static_cast<std::int64_t>(task.system);
  if (system < 0 || system >= batch.batch_size) {
    for (std::int64_t member = 0; member < batch.batch_size; ++member) {
      record_system_error(system_errors, member, device_error,
                          Gfn2IntegralDeviceError::kInvalidShellMetadata);
    }
    return;
  }
  if (!system_is_valid(system_errors, system)) return;
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t local_pair = static_cast<std::int64_t>(task.local_pair);
  const std::int64_t bra_shell = static_cast<std::int64_t>(task.bra_shell);
  const std::int64_t ket_shell = static_cast<std::int64_t>(task.ket_shell);
  if (local_pair >= shells * shells || bra_shell < shell_begin || bra_shell >= shell_end ||
      ket_shell < shell_begin || ket_shell >= shell_end ||
      local_pair != (bra_shell - shell_begin) * shells + ket_shell - shell_begin) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kInvalidShellMetadata);
    return;
  }
  const std::int64_t bra_atom = batch.shell_to_atom[bra_shell];
  const std::int64_t ket_atom = batch.shell_to_atom[ket_shell];
  if (batch.angular_momenta[bra_shell] != 0u || batch.angular_momenta[ket_shell] != 0u ||
      bra_atom > ket_atom || (bra_atom == ket_atom && bra_shell > ket_shell)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kInvalidShellMetadata);
    return;
  }
  const double vector[3] = {positions[ket_atom * 3] - positions[bra_atom * 3],
                            positions[ket_atom * 3 + 1] - positions[bra_atom * 3 + 1],
                            positions[ket_atom * 3 + 2] - positions[bra_atom * 3 + 2]};
  const double distance_squared =
      vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
  if (!isfinite(vector[0]) || !isfinite(vector[1]) || !isfinite(vector[2]) ||
      !isfinite(distance_squared)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kCoordinateDifferenceOverflow);
    return;
  }

  const bool multipoles_enabled = batch.model == XtbModelFlavor::kGfn2;
  double overlap_value = 0.0;
  double raw_multipoles[kMultipoleComponents] = {};
  const std::int64_t bra_primitive_begin = batch.shell_primitive_offsets[bra_shell];
  const std::int64_t bra_primitive_end = batch.shell_primitive_offsets[bra_shell + 1];
  const std::int64_t ket_primitive_begin = batch.shell_primitive_offsets[ket_shell];
  const std::int64_t ket_primitive_end = batch.shell_primitive_offsets[ket_shell + 1];
  for (std::int64_t ket_primitive = ket_primitive_begin; ket_primitive < ket_primitive_end;
       ++ket_primitive) {
    const double ket_alpha = batch.primitive_exponents[ket_primitive];
    for (std::int64_t bra_primitive = bra_primitive_begin; bra_primitive < bra_primitive_end;
         ++bra_primitive) {
      const double bra_alpha = batch.primitive_exponents[bra_primitive];
      const double alpha_sum = ket_alpha + bra_alpha;
      const double inverse_sum = 1.0 / alpha_sum;
      const double product_exponent = ket_alpha * bra_alpha * distance_squared * inverse_sum;
      if (product_exponent > batch.integral_cutoff) continue;
      const double sqrt_inverse_sum = sqrt(inverse_sum);
      const double primitive_prefactor = exp(-product_exponent) * kSqrtPiCubed * sqrt_inverse_sum *
                                         sqrt_inverse_sum * sqrt_inverse_sum *
                                         batch.primitive_coefficients[ket_primitive] *
                                         batch.primitive_coefficients[bra_primitive];
      overlap_value += primitive_prefactor;
      if (!multipoles_enabled) continue;
      double moments[3][4];
      make_ss_moments(vector, bra_alpha, inverse_sum, moments);
      for (int component = 0; component < kMultipoleComponents; ++component) {
        int power[3];
        multipole_power(component, &power[0], &power[1], &power[2]);
        raw_multipoles[component] += primitive_prefactor * moments[0][power[0]] *
                                     moments[1][power[1]] * moments[2][power[2]];
      }
    }
  }
  bool finite = isfinite(overlap_value);
  for (int component = 0; multipoles_enabled && component < kMultipoleComponents; ++component) {
    finite = finite && isfinite(raw_multipoles[component]);
  }
  if (!finite) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kNonfiniteIntegralArithmetic);
    return;
  }

  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbital_count = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t bra_orbital = batch.shell_orbital_offsets[bra_shell] - orbital_begin;
  const std::int64_t ket_orbital = batch.shell_orbital_offsets[ket_shell] - orbital_begin;
  const std::int64_t forward = matrix_begin + bra_orbital * orbital_count + ket_orbital;
  const std::int64_t reverse = matrix_begin + ket_orbital * orbital_count + bra_orbital;
  workspace.overlap_scratch[forward] = overlap_value;
  workspace.overlap_scratch[reverse] = overlap_value;
  if (!multipoles_enabled) return;
  const double dipole[3] = {raw_multipoles[0], raw_multipoles[1], raw_multipoles[2]};
  const double trace = 0.5 * (raw_multipoles[3] + raw_multipoles[5] + raw_multipoles[8]);
  const double quadrupole[6] = {1.5 * raw_multipoles[3] - trace, 1.5 * raw_multipoles[4],
                                1.5 * raw_multipoles[5] - trace, 1.5 * raw_multipoles[6],
                                1.5 * raw_multipoles[7],         1.5 * raw_multipoles[8] - trace};
  for (int component = 0; component < 3; ++component) {
    workspace.dipole_scratch[component * batch.total_matrix_elements + forward] = dipole[component];
  }
  for (int component = 0; component < 6; ++component) {
    workspace.quadrupole_scratch[component * batch.total_matrix_elements + forward] =
        quadrupole[component];
  }
  if (bra_shell == ket_shell) {
    for (int component = 0; component < 3; ++component) {
      workspace.dipole_scratch[component * batch.total_matrix_elements + reverse] =
          dipole[component];
    }
    for (int component = 0; component < 6; ++component) {
      workspace.quadrupole_scratch[component * batch.total_matrix_elements + reverse] =
          quadrupole[component];
    }
    return;
  }
  double shifted_dipole[3];
  for (int component = 0; component < 3; ++component) {
    shifted_dipole[component] = dipole[component] + vector[component] * overlap_value;
    workspace.dipole_scratch[component * batch.total_matrix_elements + reverse] =
        shifted_dipole[component];
  }
  const double shift[6] = {
      2.0 * vector[0] * dipole[0] + vector[0] * vector[0] * overlap_value,
      vector[0] * dipole[1] + vector[1] * dipole[0] + vector[0] * vector[1] * overlap_value,
      2.0 * vector[1] * dipole[1] + vector[1] * vector[1] * overlap_value,
      vector[0] * dipole[2] + vector[2] * dipole[0] + vector[0] * vector[2] * overlap_value,
      vector[1] * dipole[2] + vector[2] * dipole[1] + vector[1] * vector[2] * overlap_value,
      2.0 * vector[2] * dipole[2] + vector[2] * vector[2] * overlap_value};
  const double shift_trace = 0.5 * (shift[0] + shift[2] + shift[5]);
  for (int component = 0; component < 6; ++component) {
    const bool diagonal = component == 0 || component == 2 || component == 5;
    const double shifted =
        quadrupole[component] + 1.5 * shift[component] - (diagonal ? shift_trace : 0.0);
    if (!isfinite(shifted)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kNonfiniteIntegralArithmetic);
      return;
    }
    workspace.quadrupole_scratch[component * batch.total_matrix_elements + reverse] = shifted;
  }
}

__global__ void publish_integrals_kernel(Gfn2IntegralDeviceBatch batch, const double* overlap_in,
                                         const double* dipole_in, const double* quadrupole_in,
                                         double* overlap, double* dipole, double* quadrupole,
                                         Gfn2IntegralDeviceWorkspace workspace,
                                         const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t tile = static_cast<std::int64_t>(blockIdx.y);
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t begin = batch.matrix_offsets[system];
  const std::int64_t end = batch.matrix_offsets[system + 1];
  const std::int64_t stride = batch.linear_tiles_per_system * blockDim.x;
  for (std::int64_t element = begin + tile * blockDim.x + threadIdx.x; element < end;
       element += stride) {
    overlap[element] = overlap_in[element];
    if (batch.model != XtbModelFlavor::kGfn2) continue;
    for (int component = 0; component < 3; ++component) {
      dipole[component * batch.total_matrix_elements + element] =
          dipole_in[component * batch.total_matrix_elements + element];
    }
    for (int component = 0; component < 6; ++component) {
      quadrupole[component * batch.total_matrix_elements + element] =
          quadrupole_in[component * batch.total_matrix_elements + element];
    }
  }
}

__global__ void h0_preflight_kernel(Gfn2IntegralDeviceBatch batch, Gfn2H0DevicePlan plan,
                                    const double* coordination_numbers, const double* overlap,
                                    Gfn2IntegralDeviceWorkspace workspace,
                                    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t tile = static_cast<std::int64_t>(blockIdx.y);
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system)) {
    return;
  }
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = 1;
  }
  __syncthreads();
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t pair_begin = batch.shell_pair_offsets[system];
  const std::int64_t pair_end = batch.shell_pair_offsets[system + 1];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
  const std::int64_t stride = batch.linear_tiles_per_system * blockDim.x;
  for (std::int64_t atom = atom_begin + tile * blockDim.x + threadIdx.x; atom < atom_end;
       atom += stride) {
    if (!(plan.atomic_radii[atom] > 0.0) || !isfinite(plan.atomic_radii[atom]) ||
        !isfinite(coordination_numbers[atom])) {
      record_system_error(system_errors, system, device_error,
                          isfinite(coordination_numbers[atom])
                              ? Gfn2IntegralDeviceError::kInvalidH0Parameter
                              : Gfn2IntegralDeviceError::kInvalidCoordination);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t shell = shell_begin + tile * blockDim.x + threadIdx.x; shell < shell_end;
       shell += stride) {
    if (!isfinite(plan.shell_levels[shell]) || !isfinite(plan.shell_coordination_scale[shell]) ||
        !isfinite(plan.shell_polynomial[shell])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidH0Parameter);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t pair = pair_begin + tile * blockDim.x + threadIdx.x; pair < pair_end;
       pair += stride) {
    if (!(plan.shell_pair_scale[pair] > 0.0) || !isfinite(plan.shell_pair_scale[pair])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidH0Parameter);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t element = matrix_begin + tile * blockDim.x + threadIdx.x; element < matrix_end;
       element += stride) {
    if (!isfinite(overlap[element])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kNonfiniteOverlap);
      atomicExch(&valid, 0);
    }
  }
}

__global__ void h0_shell_pair_kernel(Gfn2IntegralDeviceBatch batch, Gfn2H0DevicePlan plan,
                                     const double* positions, const double* coordination_numbers,
                                     const double* overlap, Gfn2IntegralDeviceWorkspace workspace,
                                     std::uint32_t* system_errors, std::uint32_t* device_error,
                                     const Gfn2IntegralShellPairTask* tasks,
                                     std::int64_t maximum_pair_blocks) {
  const std::int64_t global_pair = static_cast<std::int64_t>(blockIdx.x);
  std::int64_t system = 0;
  std::int64_t local_pair = 0;
  std::int64_t task_first_shell = -1;
  std::int64_t task_second_shell = -1;
  if (!resolve_shell_pair_task(batch, tasks, global_pair, maximum_pair_blocks, system_errors,
                               device_error, system, local_pair, task_first_shell,
                               task_second_shell)) {
    return;
  }
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t shells = shell_end - shell_begin;
  const bool compact_task_valid =
      tasks == nullptr ||
      (task_first_shell >= shell_begin && task_first_shell < shell_end &&
       task_second_shell >= shell_begin && task_second_shell < shell_end &&
       local_pair == (task_first_shell - shell_begin) * shells + task_second_shell - shell_begin);
  if (local_pair >= shells * shells || !compact_task_valid) {
    if (tasks != nullptr && threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidShellMetadata);
    }
    return;
  }
  const std::int64_t first_shell =
      tasks == nullptr ? shell_begin + local_pair / shells : task_first_shell;
  const std::int64_t second_shell =
      tasks == nullptr ? shell_begin + local_pair % shells : task_second_shell;
  const std::int64_t first_atom = batch.shell_to_atom[first_shell];
  const std::int64_t second_atom = batch.shell_to_atom[second_shell];
  const double first_level =
      plan.shell_levels[first_shell] -
      plan.shell_coordination_scale[first_shell] * coordination_numbers[first_atom];
  const double second_level =
      plan.shell_levels[second_shell] -
      plan.shell_coordination_scale[second_shell] * coordination_numbers[second_atom];
  double spatial_scale = 1.0;
  if (first_atom != second_atom) {
    const double dx = positions[first_atom * 3] - positions[second_atom * 3];
    const double dy = positions[first_atom * 3 + 1] - positions[second_atom * 3 + 1];
    const double dz = positions[first_atom * 3 + 2] - positions[second_atom * 3 + 2];
    const double distance = sqrt(dx * dx + dy * dy + dz * dz);
    const double reduced_distance =
        sqrt(distance / (plan.atomic_radii[first_atom] + plan.atomic_radii[second_atom]));
    const double polynomial = (1.0 + plan.shell_polynomial[first_shell] * reduced_distance) *
                              (1.0 + plan.shell_polynomial[second_shell] * reduced_distance);
    const std::int64_t pair = batch.shell_pair_offsets[system] + local_pair;
    spatial_scale = plan.shell_pair_scale[pair] * polynomial;
  }
  const double factor = 0.5 * (first_level + second_level) * spatial_scale;
  if (!isfinite(factor)) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kNonfiniteH0Arithmetic);
    }
    return;
  }
  const std::int64_t first_begin = batch.shell_orbital_offsets[first_shell];
  const std::int64_t first_end = batch.shell_orbital_offsets[first_shell + 1];
  const std::int64_t second_begin = batch.shell_orbital_offsets[second_shell];
  const std::int64_t second_end = batch.shell_orbital_offsets[second_shell + 1];
  const std::int64_t first_count = first_end - first_begin;
  const std::int64_t second_count = second_end - second_begin;
  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbitals = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  for (std::int64_t local = threadIdx.x; local < first_count * second_count; local += blockDim.x) {
    const std::int64_t first_ao = first_begin + local / second_count;
    const std::int64_t second_ao = second_begin + local % second_count;
    const std::int64_t matrix =
        matrix_begin + (first_ao - orbital_begin) * orbitals + second_ao - orbital_begin;
    const double value = overlap[matrix] * factor;
    if (!isfinite(value)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kNonfiniteH0Arithmetic);
    } else {
      workspace.h0_scratch[matrix] = value;
    }
  }
}

__global__ void h0_ss_task_kernel(Gfn2IntegralDeviceBatch batch, Gfn2H0DevicePlan plan,
                                  const double* positions, const double* coordination_numbers,
                                  const double* overlap, Gfn2IntegralDeviceWorkspace workspace,
                                  std::uint32_t* system_errors, std::uint32_t* device_error,
                                  const Gfn2IntegralShellPairTask* tasks, std::int64_t task_count) {
  const std::int64_t task_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (task_index >= task_count || !sequence_is_active(workspace)) return;
  const Gfn2IntegralShellPairTask task = tasks[task_index];
  const std::int64_t system = static_cast<std::int64_t>(task.system);
  if (system < 0 || system >= batch.batch_size) {
    for (std::int64_t member = 0; member < batch.batch_size; ++member) {
      record_system_error(system_errors, member, device_error,
                          Gfn2IntegralDeviceError::kInvalidShellMetadata);
    }
    return;
  }
  if (!system_is_valid(system_errors, system)) return;
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t local_pair = static_cast<std::int64_t>(task.local_pair);
  const std::int64_t first_shell = static_cast<std::int64_t>(task.bra_shell);
  const std::int64_t second_shell = static_cast<std::int64_t>(task.ket_shell);
  if (local_pair >= shells * shells || first_shell < shell_begin || first_shell >= shell_end ||
      second_shell < shell_begin || second_shell >= shell_end ||
      local_pair != (first_shell - shell_begin) * shells + second_shell - shell_begin) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kInvalidShellMetadata);
    return;
  }
  if (batch.angular_momenta[first_shell] != 0u || batch.angular_momenta[second_shell] != 0u) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kInvalidShellMetadata);
    return;
  }
  const std::int64_t first_atom = batch.shell_to_atom[first_shell];
  const std::int64_t second_atom = batch.shell_to_atom[second_shell];
  const double first_level =
      plan.shell_levels[first_shell] -
      plan.shell_coordination_scale[first_shell] * coordination_numbers[first_atom];
  const double second_level =
      plan.shell_levels[second_shell] -
      plan.shell_coordination_scale[second_shell] * coordination_numbers[second_atom];
  double spatial_scale = 1.0;
  if (first_atom != second_atom) {
    const double dx = positions[first_atom * 3] - positions[second_atom * 3];
    const double dy = positions[first_atom * 3 + 1] - positions[second_atom * 3 + 1];
    const double dz = positions[first_atom * 3 + 2] - positions[second_atom * 3 + 2];
    const double distance = sqrt(dx * dx + dy * dy + dz * dz);
    const double reduced_distance =
        sqrt(distance / (plan.atomic_radii[first_atom] + plan.atomic_radii[second_atom]));
    const double polynomial = (1.0 + plan.shell_polynomial[first_shell] * reduced_distance) *
                              (1.0 + plan.shell_polynomial[second_shell] * reduced_distance);
    const std::int64_t pair = batch.shell_pair_offsets[system] + local_pair;
    spatial_scale = plan.shell_pair_scale[pair] * polynomial;
  }
  const double factor = 0.5 * (first_level + second_level) * spatial_scale;
  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbitals = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t first_ao = batch.shell_orbital_offsets[first_shell];
  const std::int64_t second_ao = batch.shell_orbital_offsets[second_shell];
  const std::int64_t matrix =
      matrix_begin + (first_ao - orbital_begin) * orbitals + second_ao - orbital_begin;
  const double value = overlap[matrix] * factor;
  if (!isfinite(factor) || !isfinite(value)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kNonfiniteH0Arithmetic);
    return;
  }
  workspace.h0_scratch[matrix] = value;
}

__global__ void publish_h0_kernel(Gfn2IntegralDeviceBatch batch,
                                  Gfn2IntegralDeviceWorkspace workspace, double* hamiltonian,
                                  const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t tile = static_cast<std::int64_t>(blockIdx.y);
  if (!sequence_is_active(workspace) || !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t begin = batch.matrix_offsets[system];
  const std::int64_t end = batch.matrix_offsets[system + 1];
  const std::int64_t stride = batch.linear_tiles_per_system * blockDim.x;
  for (std::int64_t element = begin + tile * blockDim.x + threadIdx.x; element < end;
       element += stride) {
    hamiltonian[element] = workspace.h0_scratch[element];
  }
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

template <typename T>
bool required_pointer(const T* pointer, std::int64_t elements) noexcept {
  return elements == 0 || is_aligned(pointer, alignof(T));
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

template <std::size_t ReadCount, std::size_t WriteCount>
bool writes_are_disjoint(const std::array<MemoryRange, ReadCount>& reads,
                         const std::array<MemoryRange, WriteCount>& writes) noexcept {
  for (std::size_t write = 0; write < WriteCount; ++write) {
    for (const MemoryRange& read : reads) {
      if (overlaps(writes[write], read)) {
        return false;
      }
    }
    for (std::size_t peer = write + 1; peer < WriteCount; ++peer) {
      if (overlaps(writes[write], writes[peer])) {
        return false;
      }
    }
  }
  return true;
}

cudaError_t validate_common(const Gfn2IntegralDeviceBatch& batch,
                            const Gfn2IntegralDeviceWorkspace& workspace,
                            std::uint32_t* system_errors, std::uint32_t* device_error,
                            std::int64_t* maximum_pair_blocks, unsigned int* grid_blocks) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_shells <= 0 ||
      batch.total_orbitals <= 0 || batch.total_primitives <= 0 ||
      batch.total_matrix_elements <= 0 || batch.total_shell_pair_elements <= 0 ||
      batch.maximum_system_shells <= 0 || batch.linear_tiles_per_system <= 0 ||
      batch.linear_tiles_per_system > kGfn2IntegralLinearBlockBudget ||
      !(batch.integral_cutoff > 0.0) || !std::isfinite(batch.integral_cutoff) ||
      batch.plan_token == 0u || workspace.plan_token != batch.plan_token ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.total_atoms == kMaximumInt64 || batch.total_shells == kMaximumInt64 ||
      batch.total_atoms > kMaximumInt64 / 3 || !valid_xtb_model_flavor(batch.model) ||
      batch.use_compact_tasks > 1u || batch.reserved != 0u ||
      batch.forward_generic_task_count < 0 || batch.forward_ss_task_count < 0 ||
      batch.h0_generic_task_count < 0 || batch.h0_ss_task_count < 0 ||
      batch.force_generic_task_count < 0 || batch.force_ss_task_count < 0 ||
      batch.forward_generic_task_count >
          static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.forward_ss_task_count > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.h0_generic_task_count > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.h0_ss_task_count > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.force_generic_task_count > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      batch.force_ss_task_count > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      (batch.model == XtbModelFlavor::kGfn2 &&
       batch.total_matrix_elements > kMaximumInt64 / kGfn2IntegralQuadrupoleComponents) ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.batch_orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.shell_pair_offset_count != batch.batch_size + 1 ||
      batch.atom_shell_offset_count != batch.total_atoms + 1 ||
      batch.shell_orbital_offset_count != batch.total_shells + 1 ||
      batch.shell_primitive_offset_count != batch.total_shells + 1 ||
      batch.shell_to_atom_count != batch.total_shells ||
      batch.angular_momentum_count != batch.total_shells ||
      batch.primitive_exponent_count != batch.total_primitives ||
      batch.primitive_coefficient_count != batch.total_primitives ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_pair_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atom_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_primitive_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_to_atom, alignof(std::int64_t)) || batch.angular_momenta == nullptr ||
      !is_aligned(batch.primitive_exponents, alignof(double)) ||
      !is_aligned(batch.primitive_coefficients, alignof(double)) ||
      !required_pointer(batch.forward_generic_tasks, batch.forward_generic_task_count) ||
      !required_pointer(batch.forward_ss_tasks, batch.forward_ss_task_count) ||
      !required_pointer(batch.h0_generic_tasks, batch.h0_generic_task_count) ||
      !required_pointer(batch.h0_ss_tasks, batch.h0_ss_task_count) ||
      !required_pointer(batch.force_generic_tasks, batch.force_generic_task_count) ||
      !required_pointer(batch.force_ss_tasks, batch.force_ss_task_count) ||
      workspace.sequence_elements < 1 ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max())
               ? cudaErrorInvalidConfiguration
               : cudaErrorInvalidValue;
  }
  const bool tasks_absent =
      batch.forward_generic_task_count == 0 && batch.forward_ss_task_count == 0 &&
      batch.h0_generic_task_count == 0 && batch.h0_ss_task_count == 0 &&
      batch.force_generic_task_count == 0 && batch.force_ss_task_count == 0 &&
      batch.forward_generic_tasks == nullptr && batch.forward_ss_tasks == nullptr &&
      batch.h0_generic_tasks == nullptr && batch.h0_ss_tasks == nullptr &&
      batch.force_generic_tasks == nullptr && batch.force_ss_tasks == nullptr;
  const bool tasks_present =
      batch.forward_generic_task_count + batch.forward_ss_task_count > 0 &&
      batch.h0_generic_task_count + batch.h0_ss_task_count > 0 &&
      required_pointer(batch.forward_generic_tasks, batch.forward_generic_task_count) &&
      required_pointer(batch.forward_ss_tasks, batch.forward_ss_task_count) &&
      required_pointer(batch.h0_generic_tasks, batch.h0_generic_task_count) &&
      required_pointer(batch.h0_ss_tasks, batch.h0_ss_task_count) &&
      required_pointer(batch.force_generic_tasks, batch.force_generic_task_count) &&
      required_pointer(batch.force_ss_tasks, batch.force_ss_task_count);
  if ((!tasks_absent && !tasks_present) || (batch.use_compact_tasks != 0u && !tasks_present)) {
    return cudaErrorInvalidValue;
  }
  if (batch.use_compact_tasks != 0u) {
    *maximum_pair_blocks = 0;
    *grid_blocks = 0u;
    return cudaSuccess;
  }
  if (batch.maximum_system_shells > kMaximumInt64 / batch.maximum_system_shells) {
    return cudaErrorInvalidConfiguration;
  }
  *maximum_pair_blocks = batch.maximum_system_shells * batch.maximum_system_shells;
  if (*maximum_pair_blocks > kMaximumInt64 / batch.batch_size) {
    return cudaErrorInvalidConfiguration;
  }
  const std::int64_t blocks = *maximum_pair_blocks * batch.batch_size;
  if (blocks <= 0 || blocks > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    return cudaErrorInvalidConfiguration;
  }
  *grid_blocks = static_cast<unsigned int>(blocks);
  return cudaSuccess;
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

bool make_gfn2_integral_linear_launch_shape(std::int64_t batch_size, std::int64_t tiles_per_system,
                                            Gfn2IntegralLinearLaunchShape& shape) noexcept {
  if (batch_size <= 0 || batch_size > std::numeric_limits<int>::max() || tiles_per_system <= 0 ||
      tiles_per_system > kGfn2IntegralLinearBlockBudget) {
    return false;
  }
  shape = {static_cast<std::uint32_t>(batch_size), static_cast<std::uint32_t>(tiles_per_system)};
  return true;
}

cudaError_t reset_gfn2_integral_device_errors_cuda(std::int64_t batch_size,
                                                   std::uint32_t* system_errors,
                                                   std::uint32_t* device_error,
                                                   cudaStream_t stream) noexcept {
  if (batch_size <= 0 || system_errors == nullptr || device_error == nullptr ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
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

cudaError_t evaluate_gfn2_integrals_cuda(const Gfn2IntegralDeviceBatch& batch,
                                         const double* positions, double* overlap, double* dipole,
                                         double* quadrupole,
                                         const Gfn2IntegralDeviceWorkspace& workspace,
                                         std::uint32_t* system_errors, std::uint32_t* device_error,
                                         cudaStream_t stream) noexcept {
  std::int64_t maximum_pair_blocks = 0;
  unsigned int grid_blocks = 0u;
  cudaError_t status = validate_common(batch, workspace, system_errors, device_error,
                                       &maximum_pair_blocks, &grid_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  const bool multipoles_enabled = batch.model == XtbModelFlavor::kGfn2;
  const std::int64_t dipole_elements =
      multipoles_enabled ? batch.total_matrix_elements * kGfn2IntegralDipoleComponents : 0;
  const std::int64_t quadrupole_elements =
      multipoles_enabled ? batch.total_matrix_elements * kGfn2IntegralQuadrupoleComponents : 0;
  if (positions == nullptr || overlap == nullptr ||
      workspace.overlap_elements < batch.total_matrix_elements ||
      workspace.dipole_elements != dipole_elements ||
      workspace.quadrupole_elements != quadrupole_elements ||
      !required_pointer(positions, batch.total_atoms * 3) ||
      !required_pointer(overlap, batch.total_matrix_elements) ||
      !required_pointer(dipole, dipole_elements) ||
      !required_pointer(quadrupole, quadrupole_elements) ||
      !required_pointer(workspace.overlap_scratch, batch.total_matrix_elements) ||
      !required_pointer(workspace.dipole_scratch, dipole_elements) ||
      !required_pointer(workspace.quadrupole_scratch, quadrupole_elements)) {
    return cudaErrorInvalidValue;
  }

  const bool compact = batch.use_compact_tasks != 0u;
  if (compact) grid_blocks = static_cast<unsigned int>(batch.forward_generic_task_count);
  std::array<MemoryRange, 15> reads;
  std::array<MemoryRange, 9> writes;
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                  &reads[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count,
                  sizeof(*batch.batch_shell_offsets), &reads[1]) ||
      !make_range(batch.batch_orbital_offsets, batch.batch_orbital_offset_count,
                  sizeof(*batch.batch_orbital_offsets), &reads[2]) ||
      !make_range(batch.matrix_offsets, batch.matrix_offset_count, sizeof(*batch.matrix_offsets),
                  &reads[3]) ||
      !make_range(batch.shell_pair_offsets, batch.shell_pair_offset_count,
                  sizeof(*batch.shell_pair_offsets), &reads[4]) ||
      !make_range(batch.atom_shell_offsets, batch.atom_shell_offset_count,
                  sizeof(*batch.atom_shell_offsets), &reads[5]) ||
      !make_range(batch.shell_orbital_offsets, batch.shell_orbital_offset_count,
                  sizeof(*batch.shell_orbital_offsets), &reads[6]) ||
      !make_range(batch.shell_primitive_offsets, batch.shell_primitive_offset_count,
                  sizeof(*batch.shell_primitive_offsets), &reads[7]) ||
      !make_range(batch.shell_to_atom, batch.shell_to_atom_count, sizeof(*batch.shell_to_atom),
                  &reads[8]) ||
      !make_range(batch.angular_momenta, batch.angular_momentum_count,
                  sizeof(*batch.angular_momenta), &reads[9]) ||
      !make_range(batch.primitive_exponents, batch.primitive_exponent_count,
                  sizeof(*batch.primitive_exponents), &reads[10]) ||
      !make_range(batch.primitive_coefficients, batch.primitive_coefficient_count,
                  sizeof(*batch.primitive_coefficients), &reads[11]) ||
      !make_range(positions, batch.total_atoms * 3, sizeof(*positions), &reads[12]) ||
      !make_range(batch.forward_generic_tasks, compact ? batch.forward_generic_task_count : 0,
                  sizeof(*batch.forward_generic_tasks), &reads[13]) ||
      !make_range(batch.forward_ss_tasks, compact ? batch.forward_ss_task_count : 0,
                  sizeof(*batch.forward_ss_tasks), &reads[14]) ||
      !make_range(overlap, batch.total_matrix_elements, sizeof(*overlap), &writes[0]) ||
      !make_range(dipole, dipole_elements, sizeof(*dipole), &writes[1]) ||
      !make_range(quadrupole, quadrupole_elements, sizeof(*quadrupole), &writes[2]) ||
      !make_range(workspace.overlap_scratch, batch.total_matrix_elements,
                  sizeof(*workspace.overlap_scratch), &writes[3]) ||
      !make_range(workspace.dipole_scratch, dipole_elements, sizeof(*workspace.dipole_scratch),
                  &writes[4]) ||
      !make_range(workspace.quadrupole_scratch, quadrupole_elements,
                  sizeof(*workspace.quadrupole_scratch), &writes[5]) ||
      !make_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active), &writes[6]) ||
      !make_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[7]) ||
      !make_range(device_error, 1, sizeof(*device_error), &writes[8]) ||
      !writes_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }

  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, positions, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  if (compact && batch.forward_ss_task_count > 0) {
    const auto ss_grid = static_cast<unsigned int>(
        (batch.forward_ss_task_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    integral_ss_task_kernel<<<ss_grid, kThreadsPerBlock, 0, stream>>>(
        batch, positions, workspace, system_errors, device_error, batch.forward_ss_tasks,
        batch.forward_ss_task_count);
    status = check_launch();
    if (status != cudaSuccess) return status;
  }
  if (!compact || grid_blocks != 0u) {
    integral_shell_pair_kernel<<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
        batch, positions, workspace, system_errors, device_error,
        compact ? batch.forward_generic_tasks : nullptr, maximum_pair_blocks);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }
  Gfn2IntegralLinearLaunchShape linear_shape{};
  if (!make_gfn2_integral_linear_launch_shape(batch.batch_size, batch.linear_tiles_per_system,
                                              linear_shape)) {
    return cudaErrorInvalidValue;
  }
  const dim3 linear_grid(linear_shape.systems, linear_shape.tiles, 1u);
  publish_integrals_kernel<<<linear_grid, kThreadsPerBlock, 0, stream>>>(
      batch, workspace.overlap_scratch, workspace.dipole_scratch, workspace.quadrupole_scratch,
      overlap, dipole, quadrupole, workspace, system_errors);
  return check_launch();
}

cudaError_t evaluate_gfn2_h0_cuda(const Gfn2IntegralDeviceBatch& batch,
                                  const Gfn2H0DevicePlan& plan, const double* positions,
                                  const double* coordination_numbers, const double* overlap,
                                  double* hamiltonian, const Gfn2IntegralDeviceWorkspace& workspace,
                                  std::uint32_t* system_errors, std::uint32_t* device_error,
                                  cudaStream_t stream) noexcept {
  std::int64_t maximum_pair_blocks = 0;
  unsigned int grid_blocks = 0u;
  cudaError_t status = validate_common(batch, workspace, system_errors, device_error,
                                       &maximum_pair_blocks, &grid_blocks);
  if (status != cudaSuccess || plan.plan_token != batch.plan_token ||
      plan.atomic_radius_count != batch.total_atoms ||
      plan.shell_level_count != batch.total_shells ||
      plan.shell_coordination_scale_count != batch.total_shells ||
      plan.shell_polynomial_count != batch.total_shells ||
      plan.shell_pair_scale_count != batch.total_shell_pair_elements || positions == nullptr ||
      coordination_numbers == nullptr || overlap == nullptr || hamiltonian == nullptr ||
      workspace.h0_elements < batch.total_matrix_elements ||
      !required_pointer(plan.atomic_radii, batch.total_atoms) ||
      !required_pointer(plan.shell_levels, batch.total_shells) ||
      !required_pointer(plan.shell_coordination_scale, batch.total_shells) ||
      !required_pointer(plan.shell_polynomial, batch.total_shells) ||
      !required_pointer(plan.shell_pair_scale, batch.total_shell_pair_elements) ||
      !required_pointer(positions, batch.total_atoms * 3) ||
      !required_pointer(coordination_numbers, batch.total_atoms) ||
      !required_pointer(overlap, batch.total_matrix_elements) ||
      !required_pointer(hamiltonian, batch.total_matrix_elements) ||
      !required_pointer(workspace.h0_scratch, batch.total_matrix_elements)) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }

  const bool compact = batch.use_compact_tasks != 0u;
  if (compact) grid_blocks = static_cast<unsigned int>(batch.h0_generic_task_count);
  std::array<MemoryRange, 22> reads;
  std::array<MemoryRange, 5> writes;
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                  &reads[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count,
                  sizeof(*batch.batch_shell_offsets), &reads[1]) ||
      !make_range(batch.batch_orbital_offsets, batch.batch_orbital_offset_count,
                  sizeof(*batch.batch_orbital_offsets), &reads[2]) ||
      !make_range(batch.matrix_offsets, batch.matrix_offset_count, sizeof(*batch.matrix_offsets),
                  &reads[3]) ||
      !make_range(batch.shell_pair_offsets, batch.shell_pair_offset_count,
                  sizeof(*batch.shell_pair_offsets), &reads[4]) ||
      !make_range(batch.atom_shell_offsets, batch.atom_shell_offset_count,
                  sizeof(*batch.atom_shell_offsets), &reads[5]) ||
      !make_range(batch.shell_orbital_offsets, batch.shell_orbital_offset_count,
                  sizeof(*batch.shell_orbital_offsets), &reads[6]) ||
      !make_range(batch.shell_primitive_offsets, batch.shell_primitive_offset_count,
                  sizeof(*batch.shell_primitive_offsets), &reads[7]) ||
      !make_range(batch.shell_to_atom, batch.shell_to_atom_count, sizeof(*batch.shell_to_atom),
                  &reads[8]) ||
      !make_range(batch.angular_momenta, batch.angular_momentum_count,
                  sizeof(*batch.angular_momenta), &reads[9]) ||
      !make_range(batch.primitive_exponents, batch.primitive_exponent_count,
                  sizeof(*batch.primitive_exponents), &reads[10]) ||
      !make_range(batch.primitive_coefficients, batch.primitive_coefficient_count,
                  sizeof(*batch.primitive_coefficients), &reads[11]) ||
      !make_range(plan.atomic_radii, batch.total_atoms, sizeof(*plan.atomic_radii), &reads[12]) ||
      !make_range(plan.shell_levels, batch.total_shells, sizeof(*plan.shell_levels), &reads[13]) ||
      !make_range(plan.shell_coordination_scale, batch.total_shells,
                  sizeof(*plan.shell_coordination_scale), &reads[14]) ||
      !make_range(plan.shell_polynomial, batch.total_shells, sizeof(*plan.shell_polynomial),
                  &reads[15]) ||
      !make_range(plan.shell_pair_scale, batch.total_shell_pair_elements,
                  sizeof(*plan.shell_pair_scale), &reads[16]) ||
      !make_range(positions, batch.total_atoms * 3, sizeof(*positions), &reads[17]) ||
      !make_range(coordination_numbers, batch.total_atoms, sizeof(*coordination_numbers),
                  &reads[18]) ||
      !make_range(overlap, batch.total_matrix_elements, sizeof(*overlap), &reads[19]) ||
      !make_range(batch.h0_generic_tasks, compact ? batch.h0_generic_task_count : 0,
                  sizeof(*batch.h0_generic_tasks), &reads[20]) ||
      !make_range(batch.h0_ss_tasks, compact ? batch.h0_ss_task_count : 0,
                  sizeof(*batch.h0_ss_tasks), &reads[21]) ||
      !make_range(hamiltonian, batch.total_matrix_elements, sizeof(*hamiltonian), &writes[0]) ||
      !make_range(workspace.h0_scratch, batch.total_matrix_elements, sizeof(*workspace.h0_scratch),
                  &writes[1]) ||
      !make_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active), &writes[2]) ||
      !make_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[3]) ||
      !make_range(device_error, 1, sizeof(*device_error), &writes[4]) ||
      !writes_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }

  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, positions, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  Gfn2IntegralLinearLaunchShape linear_shape{};
  if (!make_gfn2_integral_linear_launch_shape(batch.batch_size, batch.linear_tiles_per_system,
                                              linear_shape)) {
    return cudaErrorInvalidValue;
  }
  const dim3 linear_grid(linear_shape.systems, linear_shape.tiles, 1u);
  h0_preflight_kernel<<<linear_grid, kThreadsPerBlock, 0, stream>>>(
      batch, plan, coordination_numbers, overlap, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  if (compact && batch.h0_ss_task_count > 0) {
    const auto ss_grid = static_cast<unsigned int>((batch.h0_ss_task_count + kThreadsPerBlock - 1) /
                                                   kThreadsPerBlock);
    h0_ss_task_kernel<<<ss_grid, kThreadsPerBlock, 0, stream>>>(
        batch, plan, positions, coordination_numbers, overlap, workspace, system_errors,
        device_error, batch.h0_ss_tasks, batch.h0_ss_task_count);
    status = check_launch();
    if (status != cudaSuccess) return status;
  }
  if (!compact || grid_blocks != 0u) {
    const int h0_threads = compact ? kCompactH0ThreadsPerBlock : kThreadsPerBlock;
    h0_shell_pair_kernel<<<grid_blocks, h0_threads, 0, stream>>>(
        batch, plan, positions, coordination_numbers, overlap, workspace, system_errors,
        device_error, compact ? batch.h0_generic_tasks : nullptr, maximum_pair_blocks);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }
  publish_h0_kernel<<<linear_grid, kThreadsPerBlock, 0, stream>>>(batch, workspace, hamiltonian,
                                                                  system_errors);
  return check_launch();
}

namespace {

__device__ bool force_sequence_is_active(const Gfn2IntegralForceDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
}

__device__ bool force_member_is_active(const Gfn2ForceDeviceActivity& activity, std::int64_t system,
                                       std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::uint8_t requested = activity.requested_mask[system];
  if (requested > 1u) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidActiveMask);
    }
    return false;
  }
  return requested == 1u && activity.system_statuses[system] == XTBLOOM_STATUS_SUCCESS &&
         system_is_valid(system_errors, system);
}

__device__ bool force_add_atomic(double* target, double contribution) {
  if (!isfinite(contribution)) {
    return false;
  }
  const double previous = atomic_add_fp64(target, contribution);
  return isfinite(previous) && isfinite(previous + contribution);
}

__global__ void capture_force_sequence_kernel(const std::uint32_t* device_error,
                                              Gfn2IntegralForceDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                static_cast<std::uint32_t>(Gfn2IntegralDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

/*
 * Validate every indexing relationship with one CTA per system before any
 * tiled force preflight reads system-wide primitive bounds. A kernel boundary
 * is the inter-CTA synchronization point: keeping this phase single-CTA means
 * malformed shell metadata cannot race a peer tile into an out-of-range read.
 */
__global__ void integral_force_topology_preflight_kernel(Gfn2IntegralDeviceBatch batch,
                                                         Gfn2ForceDeviceActivity activity,
                                                         Gfn2IntegralForceDeviceWorkspace workspace,
                                                         std::uint32_t* system_errors,
                                                         std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!force_sequence_is_active(workspace) ||
      !force_member_is_active(activity, system, system_errors, device_error)) {
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
    ranges.shell_pair_begin = batch.shell_pair_offsets[system];
    ranges.shell_pair_end = batch.shell_pair_offsets[system + 1];
    if (!valid_closed_range(ranges.atom_begin, ranges.atom_end, batch.total_atoms) ||
        !valid_closed_range(ranges.shell_begin, ranges.shell_end, batch.total_shells) ||
        !valid_closed_range(ranges.orbital_begin, ranges.orbital_end, batch.total_orbitals) ||
        !valid_closed_range(ranges.matrix_begin, ranges.matrix_end, batch.total_matrix_elements) ||
        !valid_closed_range(ranges.shell_pair_begin, ranges.shell_pair_end,
                            batch.total_shell_pair_elements)) {
      valid = 0;
    }
    if (valid != 0) {
      const std::int64_t shells = ranges.shell_end - ranges.shell_begin;
      const std::int64_t orbitals = ranges.orbital_end - ranges.orbital_begin;
      std::int64_t expected_pairs = 0;
      std::int64_t expected_matrix = 0;
      valid = shells <= batch.maximum_system_shells && checked_square(shells, &expected_pairs) &&
              checked_square(orbitals, &expected_matrix) &&
              ranges.shell_pair_end - ranges.shell_pair_begin == expected_pairs &&
              ranges.matrix_end - ranges.matrix_begin == expected_matrix &&
              batch.atom_shell_offsets[ranges.atom_begin] == ranges.shell_begin &&
              batch.atom_shell_offsets[ranges.atom_end] == ranges.shell_end &&
              batch.shell_orbital_offsets[ranges.shell_begin] == ranges.orbital_begin &&
              batch.shell_orbital_offsets[ranges.shell_end] == ranges.orbital_end;
    }
    if (valid != 0 && system == 0) {
      valid = ranges.atom_begin == 0 && ranges.shell_begin == 0 && ranges.orbital_begin == 0 &&
              ranges.matrix_begin == 0 && ranges.shell_pair_begin == 0;
    }
    if (valid != 0 && system + 1 == batch.batch_size) {
      valid = ranges.atom_end == batch.total_atoms && ranges.shell_end == batch.total_shells &&
              ranges.orbital_end == batch.total_orbitals &&
              ranges.matrix_end == batch.total_matrix_elements &&
              ranges.shell_pair_end == batch.total_shell_pair_elements;
    }
    if (valid == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidOffsets);
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
                          Gfn2IntegralDeviceError::kInvalidOffsets);
      atomicExch(&valid, 0);
    }
  }
  for (std::int64_t shell = ranges.shell_begin + threadIdx.x; shell < ranges.shell_end;
       shell += blockDim.x) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const std::uint8_t angular_momentum = batch.angular_momenta[shell];
    const std::int64_t orbital_begin = batch.shell_orbital_offsets[shell];
    const std::int64_t orbital_end = batch.shell_orbital_offsets[shell + 1];
    const std::int64_t primitive_begin = batch.shell_primitive_offsets[shell];
    const std::int64_t primitive_end = batch.shell_primitive_offsets[shell + 1];
    bool shell_valid = atom >= ranges.atom_begin && atom < ranges.atom_end;
    if (shell_valid) {
      shell_valid =
          shell >= batch.atom_shell_offsets[atom] && shell < batch.atom_shell_offsets[atom + 1];
    }
    shell_valid =
        shell_valid && angular_momentum <= 2u &&
        valid_closed_range(orbital_begin, orbital_end, batch.total_orbitals) &&
        orbital_begin >= ranges.orbital_begin && orbital_end <= ranges.orbital_end &&
        orbital_end - orbital_begin == 2 * static_cast<std::int64_t>(angular_momentum) + 1 &&
        valid_closed_range(primitive_begin, primitive_end, batch.total_primitives) &&
        primitive_begin < primitive_end;
    if (!shell_valid) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidShellMetadata);
      atomicExch(&valid, 0);
    }
  }
}

/*
 * After topology validation has completed for the whole grid, numerical
 * inputs and disjoint output seeds can be scanned by bounded topology-fixed
 * tiles. Every write has unique element ownership; a peer tile's error only
 * suppresses the later publication kernel.
 */
__global__ void integral_force_numerical_preflight_kernel(
    Gfn2IntegralDeviceBatch batch, Gfn2ForceDeviceActivity activity,
    Gfn2IntegralForceDeviceInput input, Gfn2IntegralForceDeviceOutput output,
    Gfn2IntegralForceDeviceWorkspace workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t tile = static_cast<std::int64_t>(blockIdx.y);
  if (!force_sequence_is_active(workspace) ||
      !force_member_is_active(activity, system, system_errors, device_error)) {
    return;
  }

  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
  const std::int64_t primitive_begin = batch.shell_primitive_offsets[shell_begin];
  const std::int64_t primitive_end = batch.shell_primitive_offsets[shell_end];
  const std::int64_t stride = batch.linear_tiles_per_system * blockDim.x;

  for (std::int64_t atom = atom_begin + tile * blockDim.x + threadIdx.x; atom < atom_end;
       atom += stride) {
    const std::int64_t coordinate = atom * 3;
    for (int axis = 0; axis < 3; ++axis) {
      if (!isfinite(input.positions[coordinate + axis])) {
        record_system_error(system_errors, system, device_error,
                            Gfn2IntegralDeviceError::kNonfinitePosition);
      }
      const double seed = output.gradients[coordinate + axis];
      if (!isfinite(seed)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2IntegralDeviceError::kNonfiniteGradientSeed);
      } else {
        workspace.gradient_scratch[coordinate + axis] = seed;
      }
    }
  }
  for (std::int64_t primitive = primitive_begin + tile * blockDim.x + threadIdx.x;
       primitive < primitive_end; primitive += stride) {
    if (!(batch.primitive_exponents[primitive] > 0.0) ||
        !isfinite(batch.primitive_exponents[primitive]) ||
        !isfinite(batch.primitive_coefficients[primitive])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidPrimitiveData);
    }
  }
  const std::int64_t matrix_elements = batch.total_matrix_elements;
  for (std::int64_t matrix = matrix_begin + tile * blockDim.x + threadIdx.x; matrix < matrix_end;
       matrix += stride) {
    if (!isfinite(input.overlap_adjoint[matrix])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kNonfiniteAdjoint);
    }
    for (int component = 0;
         batch.model == XtbModelFlavor::kGfn2 && component < kGfn2IntegralDipoleComponents;
         ++component) {
      if (!isfinite(input.dipole_adjoint[component * matrix_elements + matrix])) {
        record_system_error(system_errors, system, device_error,
                            Gfn2IntegralDeviceError::kNonfiniteAdjoint);
      }
    }
    for (int component = 0;
         batch.model == XtbModelFlavor::kGfn2 && component < kGfn2IntegralQuadrupoleComponents;
         ++component) {
      if (!isfinite(input.quadrupole_adjoint[component * matrix_elements + matrix])) {
        record_system_error(system_errors, system, device_error,
                            Gfn2IntegralDeviceError::kNonfiniteAdjoint);
      }
    }
  }
}

__device__ void multipole_shift_pullback_device(const double vector[3], double overlap,
                                                const double dipole[3],
                                                const double reverse_dipole_adjoint[3],
                                                const double reverse_quadrupole_adjoint[6],
                                                double* overlap_adjoint, double dipole_adjoint[3],
                                                double vector_adjoint[3]) {
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    *overlap_adjoint += reverse_dipole_adjoint[coordinate] * vector[coordinate];
    dipole_adjoint[coordinate] += reverse_dipole_adjoint[coordinate];
    vector_adjoint[coordinate] += reverse_dipole_adjoint[coordinate] * overlap;
  }
  const double diagonal_sum =
      reverse_quadrupole_adjoint[0] + reverse_quadrupole_adjoint[2] + reverse_quadrupole_adjoint[5];
  const double raw[6] = {1.5 * reverse_quadrupole_adjoint[0] - 0.5 * diagonal_sum,
                         1.5 * reverse_quadrupole_adjoint[1],
                         1.5 * reverse_quadrupole_adjoint[2] - 0.5 * diagonal_sum,
                         1.5 * reverse_quadrupole_adjoint[3],
                         1.5 * reverse_quadrupole_adjoint[4],
                         1.5 * reverse_quadrupole_adjoint[5] - 0.5 * diagonal_sum};
  const double x = vector[0];
  const double y = vector[1];
  const double z = vector[2];
  const double dx = dipole[0];
  const double dy = dipole[1];
  const double dz = dipole[2];
  const double axx = raw[0];
  const double axy = raw[1];
  const double ayy = raw[2];
  const double axz = raw[3];
  const double ayz = raw[4];
  const double azz = raw[5];
  *overlap_adjoint +=
      axx * x * x + axy * x * y + ayy * y * y + axz * x * z + ayz * y * z + azz * z * z;
  dipole_adjoint[0] += 2.0 * axx * x + axy * y + axz * z;
  dipole_adjoint[1] += axy * x + 2.0 * ayy * y + ayz * z;
  dipole_adjoint[2] += axz * x + ayz * y + 2.0 * azz * z;
  vector_adjoint[0] +=
      axx * (2.0 * dx + 2.0 * x * overlap) + axy * (dy + y * overlap) + axz * (dz + z * overlap);
  vector_adjoint[1] +=
      axy * (dx + x * overlap) + ayy * (2.0 * dy + 2.0 * y * overlap) + ayz * (dz + z * overlap);
  vector_adjoint[2] +=
      axz * (dx + x * overlap) + ayz * (dy + y * overlap) + azz * (2.0 * dz + 2.0 * z * overlap);
}

__device__ void add_multipole_shift_coefficient_pullback(const double vector[3],
                                                         const double reverse_dipole_adjoint[3],
                                                         const double reverse_quadrupole_adjoint[6],
                                                         double* overlap_adjoint,
                                                         double dipole_adjoint[3]) {
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    *overlap_adjoint += reverse_dipole_adjoint[coordinate] * vector[coordinate];
    dipole_adjoint[coordinate] += reverse_dipole_adjoint[coordinate];
  }
  const double diagonal_sum =
      reverse_quadrupole_adjoint[0] + reverse_quadrupole_adjoint[2] + reverse_quadrupole_adjoint[5];
  const double raw[6] = {1.5 * reverse_quadrupole_adjoint[0] - 0.5 * diagonal_sum,
                         1.5 * reverse_quadrupole_adjoint[1],
                         1.5 * reverse_quadrupole_adjoint[2] - 0.5 * diagonal_sum,
                         1.5 * reverse_quadrupole_adjoint[3],
                         1.5 * reverse_quadrupole_adjoint[4],
                         1.5 * reverse_quadrupole_adjoint[5] - 0.5 * diagonal_sum};
  const double x = vector[0];
  const double y = vector[1];
  const double z = vector[2];
  *overlap_adjoint += raw[0] * x * x + raw[1] * x * y + raw[2] * y * y + raw[3] * x * z +
                      raw[4] * y * z + raw[5] * z * z;
  dipole_adjoint[0] += 2.0 * raw[0] * x + raw[1] * y + raw[3] * z;
  dipole_adjoint[1] += raw[1] * x + 2.0 * raw[2] * y + raw[4] * z;
  dipole_adjoint[2] += raw[3] * x + raw[4] * y + 2.0 * raw[5] * z;
}

__device__ double ss_moment_derivative(const double moments[4], int exponent, double ket_alpha) {
  double derivative = 2.0 * ket_alpha * moments[exponent + 1];
  if (exponent > 0) derivative -= static_cast<double>(exponent) * moments[exponent - 1];
  return derivative;
}

__global__ void integral_force_ss_task_kernel(
    Gfn2IntegralDeviceBatch batch, Gfn2ForceDeviceActivity activity,
    Gfn2IntegralForceDeviceInput input, Gfn2IntegralForceDeviceWorkspace workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    const Gfn2IntegralShellPairTask* tasks, std::int64_t task_count) {
  const std::int64_t task_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (task_index >= task_count || !force_sequence_is_active(workspace)) return;
  const Gfn2IntegralShellPairTask task = tasks[task_index];
  const std::int64_t system = static_cast<std::int64_t>(task.system);
  if (system < 0 || system >= batch.batch_size) {
    for (std::int64_t member = 0; member < batch.batch_size; ++member) {
      record_system_error(system_errors, member, device_error,
                          Gfn2IntegralDeviceError::kInvalidShellMetadata);
    }
    return;
  }
  if (!force_member_is_active(activity, system, system_errors, device_error)) return;
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t local_pair = static_cast<std::int64_t>(task.local_pair);
  const std::int64_t bra_shell = static_cast<std::int64_t>(task.bra_shell);
  const std::int64_t ket_shell = static_cast<std::int64_t>(task.ket_shell);
  if (local_pair >= shells * shells || bra_shell < shell_begin || bra_shell >= shell_end ||
      ket_shell < shell_begin || ket_shell >= shell_end ||
      local_pair != (bra_shell - shell_begin) * shells + ket_shell - shell_begin) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kInvalidShellMetadata);
    return;
  }
  const std::int64_t bra_atom = batch.shell_to_atom[bra_shell];
  const std::int64_t ket_atom = batch.shell_to_atom[ket_shell];
  if (batch.angular_momenta[bra_shell] != 0u || batch.angular_momenta[ket_shell] != 0u ||
      bra_atom >= ket_atom) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kInvalidShellMetadata);
    return;
  }
  const double vector[3] = {input.positions[ket_atom * 3] - input.positions[bra_atom * 3],
                            input.positions[ket_atom * 3 + 1] - input.positions[bra_atom * 3 + 1],
                            input.positions[ket_atom * 3 + 2] - input.positions[bra_atom * 3 + 2]};
  const double distance_squared =
      vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
  if (!isfinite(vector[0]) || !isfinite(vector[1]) || !isfinite(vector[2]) ||
      !isfinite(distance_squared)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kCoordinateDifferenceOverflow);
    return;
  }

  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbital_count = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t bra_orbital = batch.shell_orbital_offsets[bra_shell] - orbital_begin;
  const std::int64_t ket_orbital = batch.shell_orbital_offsets[ket_shell] - orbital_begin;
  const std::int64_t forward = matrix_begin + bra_orbital * orbital_count + ket_orbital;
  const std::int64_t reverse = matrix_begin + ket_orbital * orbital_count + bra_orbital;
  const std::int64_t matrix_elements = batch.total_matrix_elements;
  double overlap_adjoint = input.overlap_adjoint[forward] + input.overlap_adjoint[reverse];
  double dipole_adjoint[3] = {};
  double reverse_dipole_adjoint[3] = {};
  double quadrupole_adjoint[6] = {};
  double reverse_quadrupole_adjoint[6] = {};
  const bool multipoles_enabled = batch.model == XtbModelFlavor::kGfn2;
  if (multipoles_enabled) {
    for (int component = 0; component < 3; ++component) {
      dipole_adjoint[component] = input.dipole_adjoint[component * matrix_elements + forward];
      reverse_dipole_adjoint[component] =
          input.dipole_adjoint[component * matrix_elements + reverse];
    }
    for (int component = 0; component < 6; ++component) {
      const double forward_adjoint =
          input.quadrupole_adjoint[component * matrix_elements + forward];
      const double reverse_adjoint =
          input.quadrupole_adjoint[component * matrix_elements + reverse];
      quadrupole_adjoint[component] = forward_adjoint + reverse_adjoint;
      reverse_quadrupole_adjoint[component] = reverse_adjoint;
    }
    add_multipole_shift_coefficient_pullback(vector, reverse_dipole_adjoint,
                                             reverse_quadrupole_adjoint, &overlap_adjoint,
                                             dipole_adjoint);
  }
  double raw_multipole_adjoint[kMultipoleComponents] = {dipole_adjoint[0], dipole_adjoint[1],
                                                        dipole_adjoint[2]};
  if (multipoles_enabled) {
    const double diagonal_sum =
        quadrupole_adjoint[0] + quadrupole_adjoint[2] + quadrupole_adjoint[5];
    raw_multipole_adjoint[3] = 1.5 * quadrupole_adjoint[0] - 0.5 * diagonal_sum;
    raw_multipole_adjoint[4] = 1.5 * quadrupole_adjoint[1];
    raw_multipole_adjoint[5] = 1.5 * quadrupole_adjoint[2] - 0.5 * diagonal_sum;
    raw_multipole_adjoint[6] = 1.5 * quadrupole_adjoint[3];
    raw_multipole_adjoint[7] = 1.5 * quadrupole_adjoint[4];
    raw_multipole_adjoint[8] = 1.5 * quadrupole_adjoint[5] - 0.5 * diagonal_sum;
  }

  double overlap = 0.0;
  double dipole[3] = {};
  double derivative[3] = {};
  bool finite = true;
  const std::int64_t bra_primitive_begin = batch.shell_primitive_offsets[bra_shell];
  const std::int64_t bra_primitive_end = batch.shell_primitive_offsets[bra_shell + 1];
  const std::int64_t ket_primitive_begin = batch.shell_primitive_offsets[ket_shell];
  const std::int64_t ket_primitive_end = batch.shell_primitive_offsets[ket_shell + 1];
  for (std::int64_t ket_primitive = ket_primitive_begin; ket_primitive < ket_primitive_end;
       ++ket_primitive) {
    const double ket_alpha = batch.primitive_exponents[ket_primitive];
    for (std::int64_t bra_primitive = bra_primitive_begin; bra_primitive < bra_primitive_end;
         ++bra_primitive) {
      const double bra_alpha = batch.primitive_exponents[bra_primitive];
      const double alpha_sum = ket_alpha + bra_alpha;
      const double inverse_sum = 1.0 / alpha_sum;
      const double product_exponent = ket_alpha * bra_alpha * distance_squared * inverse_sum;
      if (product_exponent > batch.integral_cutoff) continue;
      const double sqrt_inverse_sum = sqrt(inverse_sum);
      const double primitive_prefactor = exp(-product_exponent) * kSqrtPiCubed * sqrt_inverse_sum *
                                         sqrt_inverse_sum * sqrt_inverse_sum *
                                         batch.primitive_coefficients[ket_primitive] *
                                         batch.primitive_coefficients[bra_primitive];
      double moments[3][4];
      make_ss_moments(vector, bra_alpha, inverse_sum, moments);
      overlap += primitive_prefactor;
      for (int coordinate = 0; coordinate < 3; ++coordinate) {
        const double gradient =
            primitive_prefactor * ss_moment_derivative(moments[coordinate], 0, ket_alpha);
        derivative[coordinate] += overlap_adjoint * gradient;
        finite = finite && isfinite(gradient);
      }
      for (int component = 0; multipoles_enabled && component < kMultipoleComponents; ++component) {
        int power[3];
        multipole_power(component, &power[0], &power[1], &power[2]);
        const double value = primitive_prefactor * moments[0][power[0]] * moments[1][power[1]] *
                             moments[2][power[2]];
        if (component < 3) dipole[component] += value;
        finite = finite && isfinite(value);
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          const double gradient =
              primitive_prefactor *
              ss_moment_derivative(moments[coordinate], power[coordinate], ket_alpha) *
              moments[(coordinate + 1) % 3][power[(coordinate + 1) % 3]] *
              moments[(coordinate + 2) % 3][power[(coordinate + 2) % 3]];
          derivative[coordinate] += raw_multipole_adjoint[component] * gradient;
          finite = finite && isfinite(gradient);
        }
      }
      finite = finite && isfinite(primitive_prefactor);
    }
  }
  finite = finite && isfinite(overlap);
  double vector_adjoint[3] = {};
  if (multipoles_enabled) {
    double ignored_overlap_adjoint = 0.0;
    double ignored_dipole_adjoint[3] = {};
    multipole_shift_pullback_device(vector, overlap, dipole, reverse_dipole_adjoint,
                                    reverse_quadrupole_adjoint, &ignored_overlap_adjoint,
                                    ignored_dipole_adjoint, vector_adjoint);
  }
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    derivative[coordinate] += vector_adjoint[coordinate];
    finite = finite && isfinite(dipole[coordinate]) && isfinite(derivative[coordinate]);
  }
  if (!finite) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kNonfiniteGradientArithmetic);
    return;
  }
  bool atomic_finite = true;
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    atomic_finite = force_add_atomic(workspace.gradient_scratch + ket_atom * 3 + coordinate,
                                     derivative[coordinate]) &&
                    force_add_atomic(workspace.gradient_scratch + bra_atom * 3 + coordinate,
                                     -derivative[coordinate]) &&
                    atomic_finite;
  }
  if (!atomic_finite) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kNonfiniteGradientArithmetic);
  }
}

__global__ void integral_force_shell_pair_kernel(
    Gfn2IntegralDeviceBatch batch, Gfn2ForceDeviceActivity activity,
    Gfn2IntegralForceDeviceInput input, Gfn2IntegralForceDeviceWorkspace workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    const Gfn2IntegralShellPairTask* tasks, std::int64_t maximum_pair_blocks) {
  const std::int64_t global_pair = static_cast<std::int64_t>(blockIdx.x);
  std::int64_t system = 0;
  std::int64_t local_pair = 0;
  std::int64_t task_bra_shell = -1;
  std::int64_t task_ket_shell = -1;
  if (!resolve_shell_pair_task(batch, tasks, global_pair, maximum_pair_blocks, system_errors,
                               device_error, system, local_pair, task_bra_shell, task_ket_shell)) {
    return;
  }
  if (!force_sequence_is_active(workspace) ||
      !force_member_is_active(activity, system, system_errors, device_error)) {
    return;
  }
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t shells = shell_end - shell_begin;
  const bool compact_task_valid =
      tasks == nullptr ||
      (task_bra_shell >= shell_begin && task_bra_shell < shell_end &&
       task_ket_shell >= shell_begin && task_ket_shell < shell_end &&
       local_pair == (task_bra_shell - shell_begin) * shells + task_ket_shell - shell_begin);
  if (local_pair >= shells * shells || !compact_task_valid) {
    if (tasks != nullptr && threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidShellMetadata);
    }
    return;
  }
  const std::int64_t bra_shell =
      tasks == nullptr ? shell_begin + local_pair / shells : task_bra_shell;
  const std::int64_t ket_shell =
      tasks == nullptr ? shell_begin + local_pair % shells : task_ket_shell;
  const std::int64_t bra_atom = batch.shell_to_atom[bra_shell];
  const std::int64_t ket_atom = batch.shell_to_atom[ket_shell];
  if (bra_atom >= ket_atom) {
    return;
  }

  const std::uint8_t bra_l = batch.angular_momenta[bra_shell];
  const std::uint8_t ket_l = batch.angular_momenta[ket_shell];
  const int bra_cartesian_count = cartesian_count(bra_l);
  const int ket_cartesian_count = cartesian_count(ket_l);
  const int cartesian_block_size = bra_cartesian_count * ket_cartesian_count;
  const int bra_spherical_count = spherical_count(bra_l);
  const int ket_spherical_count = spherical_count(ket_l);
  const int spherical_block_size = bra_spherical_count * ket_spherical_count;
  const double vector[3] = {input.positions[ket_atom * 3] - input.positions[bra_atom * 3],
                            input.positions[ket_atom * 3 + 1] - input.positions[bra_atom * 3 + 1],
                            input.positions[ket_atom * 3 + 2] - input.positions[bra_atom * 3 + 2]};
  if (!isfinite(vector[0]) || !isfinite(vector[1]) || !isfinite(vector[2])) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kCoordinateDifferenceOverflow);
    }
    return;
  }
  const double distance_squared =
      vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
  if (!isfinite(distance_squared)) {
    if (threadIdx.x == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kCoordinateDifferenceOverflow);
    }
    return;
  }

  __shared__ double cartesian_overlap[kMaximumCartesianBlock];
  __shared__ double cartesian_overlap_gradient[3 * kMaximumCartesianBlock];
  __shared__ double cartesian_multipole[kMultipoleComponents * kMaximumCartesianBlock];
  __shared__ double cartesian_multipole_gradient[3 * kMultipoleComponents * kMaximumCartesianBlock];
  __shared__ double partial_gradient[3][kThreadsPerBlock];
  const bool multipoles_enabled = batch.model == XtbModelFlavor::kGfn2;
  const int cartesian_index = static_cast<int>(threadIdx.x);
  if (cartesian_index < cartesian_block_size) {
    const int bra_cartesian = cartesian_index / ket_cartesian_count;
    const int ket_cartesian = cartesian_index % ket_cartesian_count;
    int bra_power[3];
    int ket_power[3];
    cartesian_exponent(bra_l, bra_cartesian, &bra_power[0], &bra_power[1], &bra_power[2]);
    cartesian_exponent(ket_l, ket_cartesian, &ket_power[0], &ket_power[1], &ket_power[2]);
    double overlap_value = 0.0;
    double overlap_gradient[3] = {};
    double multipoles[kMultipoleComponents] = {};
    double multipole_gradient[3][kMultipoleComponents] = {};
    const std::int64_t bra_primitive_begin = batch.shell_primitive_offsets[bra_shell];
    const std::int64_t bra_primitive_end = batch.shell_primitive_offsets[bra_shell + 1];
    const std::int64_t ket_primitive_begin = batch.shell_primitive_offsets[ket_shell];
    const std::int64_t ket_primitive_end = batch.shell_primitive_offsets[ket_shell + 1];
    for (std::int64_t ket_primitive = ket_primitive_begin; ket_primitive < ket_primitive_end;
         ++ket_primitive) {
      const double ket_alpha = batch.primitive_exponents[ket_primitive];
      for (std::int64_t bra_primitive = bra_primitive_begin; bra_primitive < bra_primitive_end;
           ++bra_primitive) {
        const double bra_alpha = batch.primitive_exponents[bra_primitive];
        const double alpha_sum = ket_alpha + bra_alpha;
        const double inverse_sum = 1.0 / alpha_sum;
        const double product_exponent = ket_alpha * bra_alpha * distance_squared * inverse_sum;
        if (product_exponent > batch.integral_cutoff) {
          continue;
        }
        const double sqrt_inverse_sum = sqrt(inverse_sum);
        const double primitive_prefactor = exp(-product_exponent) * kSqrtPiCubed *
                                           sqrt_inverse_sum * sqrt_inverse_sum * sqrt_inverse_sum *
                                           batch.primitive_coefficients[ket_primitive] *
                                           batch.primitive_coefficients[bra_primitive];
        double axis[3][6][3];
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          const int maximum_ket_power = static_cast<int>(ket_l) + (multipoles_enabled ? 3 : 1);
          make_axis_overlap(-vector[coordinate] * bra_alpha * inverse_sum,
                            vector[coordinate] * ket_alpha * inverse_sum, 0.5 * inverse_sum,
                            maximum_ket_power, static_cast<int>(bra_l), axis[coordinate]);
        }
        const double one_dimensional[3] = {axis[0][ket_power[0]][bra_power[0]],
                                           axis[1][ket_power[1]][bra_power[1]],
                                           axis[2][ket_power[2]][bra_power[2]]};
        overlap_value +=
            primitive_prefactor * one_dimensional[0] * one_dimensional[1] * one_dimensional[2];
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          double derivative_1d =
              2.0 * ket_alpha * axis[coordinate][ket_power[coordinate] + 1][bra_power[coordinate]];
          if (ket_power[coordinate] > 0) {
            derivative_1d -= static_cast<double>(ket_power[coordinate]) *
                             axis[coordinate][ket_power[coordinate] - 1][bra_power[coordinate]];
          }
          overlap_gradient[coordinate] += primitive_prefactor * derivative_1d *
                                          one_dimensional[(coordinate + 1) % 3] *
                                          one_dimensional[(coordinate + 2) % 3];
        }
        for (int component = 0; multipoles_enabled && component < kMultipoleComponents;
             ++component) {
          int moment_power[3];
          multipole_power(component, &moment_power[0], &moment_power[1], &moment_power[2]);
          const double moment_axis[3] = {axis[0][ket_power[0] + moment_power[0]][bra_power[0]],
                                         axis[1][ket_power[1] + moment_power[1]][bra_power[1]],
                                         axis[2][ket_power[2] + moment_power[2]][bra_power[2]]};
          multipoles[component] +=
              primitive_prefactor * moment_axis[0] * moment_axis[1] * moment_axis[2];
          for (int coordinate = 0; coordinate < 3; ++coordinate) {
            const int exponent = ket_power[coordinate] + moment_power[coordinate];
            double derivative_1d =
                2.0 * ket_alpha * axis[coordinate][exponent + 1][bra_power[coordinate]];
            if (exponent > 0) {
              derivative_1d -= static_cast<double>(exponent) *
                               axis[coordinate][exponent - 1][bra_power[coordinate]];
            }
            multipole_gradient[coordinate][component] += primitive_prefactor * derivative_1d *
                                                         moment_axis[(coordinate + 1) % 3] *
                                                         moment_axis[(coordinate + 2) % 3];
          }
        }
      }
    }
    bool finite = isfinite(overlap_value);
    cartesian_overlap[cartesian_index] = overlap_value;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      cartesian_overlap_gradient[coordinate * kMaximumCartesianBlock + cartesian_index] =
          overlap_gradient[coordinate];
      finite = finite && isfinite(overlap_gradient[coordinate]);
      for (int component = 0; multipoles_enabled && component < kMultipoleComponents; ++component) {
        cartesian_multipole_gradient[(coordinate * kMultipoleComponents + component) *
                                         kMaximumCartesianBlock +
                                     cartesian_index] = multipole_gradient[coordinate][component];
        finite = finite && isfinite(multipole_gradient[coordinate][component]);
      }
    }
    for (int component = 0; multipoles_enabled && component < kMultipoleComponents; ++component) {
      cartesian_multipole[component * kMaximumCartesianBlock + cartesian_index] =
          multipoles[component];
      finite = finite && isfinite(multipoles[component]);
    }
    if (!finite) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kNonfiniteGradientArithmetic);
    }
  }
  __syncthreads();

  double derivative[3] = {};
  bool finite_derivative = system_is_valid(system_errors, system);
  const int spherical_index = static_cast<int>(threadIdx.x);
  if (finite_derivative && spherical_index < spherical_block_size) {
    const int bra_ao = spherical_index / ket_spherical_count;
    const int ket_ao = spherical_index % ket_spherical_count;
    double overlap = 0.0;
    double overlap_gradient[3] = {};
    double raw_multipoles[kMultipoleComponents] = {};
    double raw_multipole_gradient[3][kMultipoleComponents] = {};
    for (int bra_cartesian = 0; bra_cartesian < bra_cartesian_count; ++bra_cartesian) {
      const double bra_coefficient = spherical_coefficient(bra_l, bra_ao, bra_cartesian);
      if (bra_coefficient == 0.0) {
        continue;
      }
      for (int ket_cartesian = 0; ket_cartesian < ket_cartesian_count; ++ket_cartesian) {
        const double ket_coefficient = spherical_coefficient(ket_l, ket_ao, ket_cartesian);
        if (ket_coefficient == 0.0) {
          continue;
        }
        const int index = bra_cartesian * ket_cartesian_count + ket_cartesian;
        const double coefficient = bra_coefficient * ket_coefficient;
        overlap += coefficient * cartesian_overlap[index];
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          overlap_gradient[coordinate] +=
              coefficient * cartesian_overlap_gradient[coordinate * kMaximumCartesianBlock + index];
        }
        for (int component = 0; multipoles_enabled && component < kMultipoleComponents;
             ++component) {
          raw_multipoles[component] +=
              coefficient * cartesian_multipole[component * kMaximumCartesianBlock + index];
          for (int coordinate = 0; coordinate < 3; ++coordinate) {
            raw_multipole_gradient[coordinate][component] +=
                coefficient *
                cartesian_multipole_gradient[(coordinate * kMultipoleComponents + component) *
                                                 kMaximumCartesianBlock +
                                             index];
          }
        }
      }
    }
    const double dipole[3] = {raw_multipoles[0], raw_multipoles[1], raw_multipoles[2]};
    double quadrupole_gradient[3][6] = {};
    for (int coordinate = 0; multipoles_enabled && coordinate < 3; ++coordinate) {
      const double trace =
          0.5 * (raw_multipole_gradient[coordinate][3] + raw_multipole_gradient[coordinate][5] +
                 raw_multipole_gradient[coordinate][8]);
      quadrupole_gradient[coordinate][0] = 1.5 * raw_multipole_gradient[coordinate][3] - trace;
      quadrupole_gradient[coordinate][1] = 1.5 * raw_multipole_gradient[coordinate][4];
      quadrupole_gradient[coordinate][2] = 1.5 * raw_multipole_gradient[coordinate][5] - trace;
      quadrupole_gradient[coordinate][3] = 1.5 * raw_multipole_gradient[coordinate][6];
      quadrupole_gradient[coordinate][4] = 1.5 * raw_multipole_gradient[coordinate][7];
      quadrupole_gradient[coordinate][5] = 1.5 * raw_multipole_gradient[coordinate][8] - trace;
    }
    const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
    const std::int64_t orbital_count = batch.batch_orbital_offsets[system + 1] - orbital_begin;
    const std::int64_t matrix_begin = batch.matrix_offsets[system];
    const std::int64_t bra_orbital =
        batch.shell_orbital_offsets[bra_shell] - orbital_begin + bra_ao;
    const std::int64_t ket_orbital =
        batch.shell_orbital_offsets[ket_shell] - orbital_begin + ket_ao;
    const std::int64_t forward = matrix_begin + bra_orbital * orbital_count + ket_orbital;
    const std::int64_t reverse = matrix_begin + ket_orbital * orbital_count + bra_orbital;
    const std::int64_t matrix_elements = batch.total_matrix_elements;
    double overlap_adjoint = input.overlap_adjoint[forward] + input.overlap_adjoint[reverse];
    double dipole_adjoint[3] = {};
    double reverse_dipole_adjoint[3] = {};
    double quadrupole_adjoint[6] = {};
    double reverse_quadrupole_adjoint[6] = {};
    if (batch.model == XtbModelFlavor::kGfn2) {
      for (int component = 0; component < 3; ++component) {
        dipole_adjoint[component] = input.dipole_adjoint[component * matrix_elements + forward];
        reverse_dipole_adjoint[component] =
            input.dipole_adjoint[component * matrix_elements + reverse];
      }
      for (int component = 0; component < 6; ++component) {
        const double forward_adjoint =
            input.quadrupole_adjoint[component * matrix_elements + forward];
        const double reverse_adjoint =
            input.quadrupole_adjoint[component * matrix_elements + reverse];
        quadrupole_adjoint[component] = forward_adjoint + reverse_adjoint;
        reverse_quadrupole_adjoint[component] = reverse_adjoint;
      }
    }
    double vector_adjoint[3] = {};
    if (batch.model == XtbModelFlavor::kGfn2) {
      multipole_shift_pullback_device(vector, overlap, dipole, reverse_dipole_adjoint,
                                      reverse_quadrupole_adjoint, &overlap_adjoint, dipole_adjoint,
                                      vector_adjoint);
    }
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      derivative[coordinate] =
          vector_adjoint[coordinate] + overlap_adjoint * overlap_gradient[coordinate];
      for (int component = 0; batch.model == XtbModelFlavor::kGfn2 && component < 3; ++component) {
        derivative[coordinate] +=
            dipole_adjoint[component] * raw_multipole_gradient[coordinate][component];
      }
      for (int component = 0; batch.model == XtbModelFlavor::kGfn2 && component < 6; ++component) {
        derivative[coordinate] +=
            quadrupole_adjoint[component] * quadrupole_gradient[coordinate][component];
      }
      finite_derivative = finite_derivative && isfinite(derivative[coordinate]);
    }
    if (!finite_derivative) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kNonfiniteGradientArithmetic);
    }
  }
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    partial_gradient[coordinate][threadIdx.x] =
        finite_derivative && spherical_index < spherical_block_size ? derivative[coordinate] : 0.0;
  }
  __syncthreads();
  for (int offset = kThreadsPerBlock / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      for (int coordinate = 0; coordinate < 3; ++coordinate) {
        partial_gradient[coordinate][threadIdx.x] +=
            partial_gradient[coordinate][threadIdx.x + offset];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && system_is_valid(system_errors, system)) {
    bool finite = true;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      finite = force_add_atomic(workspace.gradient_scratch + ket_atom * 3 + coordinate,
                                partial_gradient[coordinate][0]) &&
               force_add_atomic(workspace.gradient_scratch + bra_atom * 3 + coordinate,
                                -partial_gradient[coordinate][0]) &&
               finite;
    }
    if (!finite) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kNonfiniteGradientArithmetic);
    }
  }
}

/* Reconstruct the image-specific H0 multiplier used by the periodic forward
 * evaluator.  The public overlap/H0 matrices are image sums, so this factor
 * must stay local to each image during the reverse overlap contraction. */
__device__ bool native_h0_image_factor(const Gfn2IntegralDeviceBatch& batch,
                                       const Gfn2H0DevicePlan& h0_plan,
                                       const double* coordination_numbers, std::int64_t system,
                                       std::int64_t bra_atom, std::int64_t ket_atom,
                                       std::int64_t bra_shell, std::int64_t ket_shell,
                                       double distance_squared, double& factor) {
  const double bra_level =
      h0_plan.shell_levels[bra_shell] -
      h0_plan.shell_coordination_scale[bra_shell] * coordination_numbers[bra_atom];
  const double ket_level =
      h0_plan.shell_levels[ket_shell] -
      h0_plan.shell_coordination_scale[ket_shell] * coordination_numbers[ket_atom];
  const double distance = sqrt(distance_squared);
  const double radius_sum = h0_plan.atomic_radii[bra_atom] + h0_plan.atomic_radii[ket_atom];
  const double reduced = sqrt(distance / radius_sum);
  const double bra_polynomial = 1.0 + h0_plan.shell_polynomial[bra_shell] * reduced;
  const double ket_polynomial = 1.0 + h0_plan.shell_polynomial[ket_shell] * reduced;
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_count = batch.batch_shell_offsets[system + 1] - shell_begin;
  const std::int64_t local_bra = bra_shell - shell_begin;
  const std::int64_t local_ket = ket_shell - shell_begin;
  const double pair_scale =
      h0_plan
          .shell_pair_scale[batch.shell_pair_offsets[system] + local_bra * shell_count + local_ket];
  factor = 0.5 * (bra_level + ket_level) * pair_scale * bra_polynomial * ket_polynomial;
  return isfinite(factor) && isfinite(distance) && isfinite(radius_sum) && radius_sum > 0.0 &&
         isfinite(reduced) && isfinite(bra_polynomial) && isfinite(ket_polynomial) &&
         isfinite(pair_scale);
}

/* The central onsite block is not represented by an image translation in the
 * native preprocessing queue. Its H0 multiplier is the level average only;
 * keeping this helper separate avoids treating the zero vector as a valid
 * non-onsite image and makes the per-image adjoint split explicit. */
__device__ bool native_h0_onsite_factor(const Gfn2H0DevicePlan& h0_plan,
                                        const double* coordination_numbers, std::int64_t bra_atom,
                                        std::int64_t ket_atom, std::int64_t bra_shell,
                                        std::int64_t ket_shell, double& factor) {
  const double bra_level =
      h0_plan.shell_levels[bra_shell] -
      h0_plan.shell_coordination_scale[bra_shell] * coordination_numbers[bra_atom];
  const double ket_level =
      h0_plan.shell_levels[ket_shell] -
      h0_plan.shell_coordination_scale[ket_shell] * coordination_numbers[ket_atom];
  factor = 0.5 * (bra_level + ket_level);
  return isfinite(bra_level) && isfinite(ket_level) && isfinite(factor);
}

/*
 * Image-aware reverse of the native periodic S/D/Q evaluator.  The forward
 * path publishes one canonical atom pair for every translation and mirrors a
 * distinct-atom block into the reverse matrix (with the ket-origin multipole
 * shift).  This kernel reconstructs each image contribution from the same
 * primitive recurrence, applies the reverse shift pullback, and accumulates
 * dE/dR directly into the transactional gradient scratch.  One lane owns a
 * task deliberately: periodic image counts are topology-bounded and this
 * scalar implementation keeps the reverse ordering auditable while parity
 * evidence is collected.
 */
__global__ void native_periodic_integral_force_kernel(
    Gfn2IntegralDeviceBatch batch, Gfn2ForceDeviceActivity activity,
    Gfn2IntegralForceDeviceInput input, Gfn2IntegralForceDeviceWorkspace workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, std::int64_t image_task_count,
    std::int64_t image_system_stride, std::int64_t maximum_pair_blocks) {
  if (threadIdx.x != 0 || !force_sequence_is_active(workspace)) return;
  const std::int64_t global_task = static_cast<std::int64_t>(blockIdx.x);
  const bool image_task = global_task < image_task_count;
  const std::int64_t task = image_task ? global_task : global_task - image_task_count;
  if (image_system_stride <= 0 || maximum_pair_blocks <= 0) return;
  const std::int64_t system =
      image_task ? global_task / image_system_stride : task / maximum_pair_blocks;
  if (system < 0 || system >= batch.batch_size ||
      !force_member_is_active(activity, system, system_errors, device_error)) {
    return;
  }

  const std::int64_t remainder =
      image_task ? global_task - system * image_system_stride : task - system * maximum_pair_blocks;
  const std::int64_t image_slot = image_task ? remainder / maximum_pair_blocks : -1;
  const std::int64_t local_pair =
      image_task ? remainder - image_slot * maximum_pair_blocks : remainder;
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t shell_count = shell_end - shell_begin;
  std::int64_t expected_pairs = 0;
  if (shell_count <= 0 || !checked_square(shell_count, &expected_pairs) || local_pair < 0 ||
      local_pair >= expected_pairs) {
    return;  // rectangular queue padding is an ordinary no-op
  }
  const std::int64_t bra_shell = shell_begin + local_pair / shell_count;
  const std::int64_t ket_shell = shell_begin + local_pair % shell_count;
  const std::int64_t bra_atom = batch.shell_to_atom[bra_shell];
  const std::int64_t ket_atom = batch.shell_to_atom[ket_shell];
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  if (bra_atom < atom_begin || bra_atom >= atom_end || ket_atom < atom_begin ||
      ket_atom >= atom_end || (image_task ? bra_atom > ket_atom : bra_atom != ket_atom)) {
    return;
  }
  /* The forward evaluator publishes a transposed (reverse) matrix block only
   * for distinct atoms in an image pair.  Onsite blocks and same-atom image
   * blocks are intentionally forward-only; including their reverse adjoints
   * here would differentiate a matrix element that was never written. */
  const bool publish_reverse = image_task && bra_atom != ket_atom;

  double vector[3] = {0.0, 0.0, 0.0};
  double distance_squared = 0.0;
  if (image_task) {
    const auto periodic = input.native_periodic;
    const std::int64_t translation_begin = periodic.translation_offsets[system];
    const std::int64_t translation_end = periodic.translation_offsets[system + 1];
    if (translation_begin < 0 || translation_end <= translation_begin ||
        translation_end > periodic.translation_elements || image_slot < 0 ||
        image_slot >= translation_end - translation_begin) {
      record_system_error(system_errors, system, device_error,
                          Gfn2IntegralDeviceError::kInvalidOffsets);
      return;
    }
    const auto translation = periodic.translations[translation_begin + image_slot];
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      const double difference = periodic_wrap_detail::rounded_subtract(
          input.positions[ket_atom * 3 + coordinate], input.positions[bra_atom * 3 + coordinate]);
      vector[coordinate] =
          periodic_wrap_detail::rounded_subtract(difference, translation.cartesian[coordinate]);
      if (!isfinite(vector[coordinate])) {
        record_system_error(system_errors, system, device_error,
                            Gfn2IntegralDeviceError::kCoordinateDifferenceOverflow);
        return;
      }
    }
    const double x_squared = periodic_wrap_detail::rounded_multiply(vector[0], vector[0]);
    const double y_squared = periodic_wrap_detail::rounded_multiply(vector[1], vector[1]);
    const double z_squared = periodic_wrap_detail::rounded_multiply(vector[2], vector[2]);
    distance_squared = periodic_wrap_detail::rounded_add(
        periodic_wrap_detail::rounded_add(x_squared, y_squared), z_squared);
    const double cutoff_squared = periodic.realspace_cutoff * periodic.realspace_cutoff;
    if (!isfinite(distance_squared) ||
        distance_squared < 2.220446049250313080847263336181640625e-16 ||
        distance_squared > cutoff_squared || fabs(vector[0]) > periodic.realspace_cutoff ||
        fabs(vector[1]) > periodic.realspace_cutoff ||
        fabs(vector[2]) > periodic.realspace_cutoff) {
      return;
    }
  }

  const std::uint8_t bra_l = batch.angular_momenta[bra_shell];
  const std::uint8_t ket_l = batch.angular_momenta[ket_shell];
  const int bra_cartesian_count = cartesian_count(bra_l);
  const int ket_cartesian_count = cartesian_count(ket_l);
  const int bra_spherical_count = spherical_count(bra_l);
  const int ket_spherical_count = spherical_count(ket_l);
  if (bra_l > 2u || ket_l > 2u ||
      bra_cartesian_count * ket_cartesian_count > kMaximumCartesianBlock) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kInvalidShellMetadata);
    return;
  }

  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbital_count = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t bra_shell_orbital = batch.shell_orbital_offsets[bra_shell] - orbital_begin;
  const std::int64_t ket_shell_orbital = batch.shell_orbital_offsets[ket_shell] - orbital_begin;
  const std::int64_t matrix_elements = batch.total_matrix_elements;
  const bool multipoles_enabled = batch.model == XtbModelFlavor::kGfn2;
  double image_h0_factor = 0.0;
  if (image_task &&
      !native_h0_image_factor(batch, input.h0_plan, input.coordination_numbers, system, bra_atom,
                              ket_atom, bra_shell, ket_shell, distance_squared, image_h0_factor)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kNonfiniteH0Arithmetic);
    return;
  }
  double local_h0_factor = image_h0_factor;
  if (!image_task && !native_h0_onsite_factor(input.h0_plan, input.coordination_numbers, bra_atom,
                                              ket_atom, bra_shell, ket_shell, local_h0_factor)) {
    record_system_error(system_errors, system, device_error,
                        Gfn2IntegralDeviceError::kNonfiniteH0Arithmetic);
    return;
  }
  for (int bra_ao = 0; bra_ao < bra_spherical_count; ++bra_ao) {
    for (int ket_ao = 0; ket_ao < ket_spherical_count; ++ket_ao) {
      const std::int64_t bra_orbital = bra_shell_orbital + bra_ao;
      const std::int64_t ket_orbital = ket_shell_orbital + ket_ao;
      const std::int64_t forward = matrix_begin + bra_orbital * orbital_count + ket_orbital;
      const std::int64_t reverse = matrix_begin + ket_orbital * orbital_count + bra_orbital;
      /* The periodic H0 reverse deliberately leaves dH0/dS out of the
       * global overlap adjoint. Add only this task's local multiplier beside
       * the matching image derivative; otherwise an image-summed adjoint
       * would be applied to every image and cross-multiply all translations. */
      double overlap_adjoint =
          input.overlap_adjoint[forward] + input.density[forward] * local_h0_factor;
      if (publish_reverse) {
        overlap_adjoint +=
            input.overlap_adjoint[reverse] + input.density[reverse] * local_h0_factor;
      }
      double dipole_adjoint[3] = {};
      double reverse_dipole_adjoint[3] = {};
      double quadrupole_adjoint[6] = {};
      double reverse_quadrupole_adjoint[6] = {};
      if (multipoles_enabled) {
        for (int component = 0; component < 3; ++component) {
          dipole_adjoint[component] = input.dipole_adjoint[component * matrix_elements + forward];
          if (publish_reverse) {
            reverse_dipole_adjoint[component] =
                input.dipole_adjoint[component * matrix_elements + reverse];
          }
        }
        for (int component = 0; component < 6; ++component) {
          quadrupole_adjoint[component] =
              input.quadrupole_adjoint[component * matrix_elements + forward];
          if (publish_reverse) {
            reverse_quadrupole_adjoint[component] =
                input.quadrupole_adjoint[component * matrix_elements + reverse];
            quadrupole_adjoint[component] += reverse_quadrupole_adjoint[component];
          }
        }
        if (publish_reverse) {
          add_multipole_shift_coefficient_pullback(vector, reverse_dipole_adjoint,
                                                   reverse_quadrupole_adjoint, &overlap_adjoint,
                                                   dipole_adjoint);
        }
      }
      int bra_power[3];
      int ket_power[3];
      /* Spherical rows are sparse Cartesian combinations.  Accumulating the
       * transformed primitive contractions here avoids materializing image
       * tensors while retaining the exact forward recurrence. */
      double derivative[3] = {};
      bool finite = true;
      for (int bra_cartesian = 0; bra_cartesian < bra_cartesian_count; ++bra_cartesian) {
        const double bra_coefficient = spherical_coefficient(bra_l, bra_ao, bra_cartesian);
        if (bra_coefficient == 0.0) continue;
        cartesian_exponent(bra_l, bra_cartesian, &bra_power[0], &bra_power[1], &bra_power[2]);
        for (int ket_cartesian = 0; ket_cartesian < ket_cartesian_count; ++ket_cartesian) {
          const double ket_coefficient = spherical_coefficient(ket_l, ket_ao, ket_cartesian);
          if (ket_coefficient == 0.0) continue;
          cartesian_exponent(ket_l, ket_cartesian, &ket_power[0], &ket_power[1], &ket_power[2]);
          double overlap = 0.0;
          double overlap_gradient[3] = {};
          double multipoles[kMultipoleComponents] = {};
          double multipole_gradient[3][kMultipoleComponents] = {};
          const std::int64_t bra_primitive_begin = batch.shell_primitive_offsets[bra_shell];
          const std::int64_t bra_primitive_end = batch.shell_primitive_offsets[bra_shell + 1];
          const std::int64_t ket_primitive_begin = batch.shell_primitive_offsets[ket_shell];
          const std::int64_t ket_primitive_end = batch.shell_primitive_offsets[ket_shell + 1];
          for (std::int64_t ket_primitive = ket_primitive_begin; ket_primitive < ket_primitive_end;
               ++ket_primitive) {
            const double ket_alpha = batch.primitive_exponents[ket_primitive];
            for (std::int64_t bra_primitive = bra_primitive_begin;
                 bra_primitive < bra_primitive_end; ++bra_primitive) {
              const double bra_alpha = batch.primitive_exponents[bra_primitive];
              const double alpha_sum = ket_alpha + bra_alpha;
              const double inverse_sum = 1.0 / alpha_sum;
              const double product_exponent =
                  ket_alpha * bra_alpha * distance_squared * inverse_sum;
              if (!(alpha_sum > 0.0) || !isfinite(alpha_sum) || !isfinite(inverse_sum) ||
                  !isfinite(product_exponent)) {
                finite = false;
                continue;
              }
              if (product_exponent > batch.integral_cutoff) continue;
              const double sqrt_inverse_sum = sqrt(inverse_sum);
              const double primitive_prefactor =
                  exp(-product_exponent) * kSqrtPiCubed * sqrt_inverse_sum * sqrt_inverse_sum *
                  sqrt_inverse_sum * batch.primitive_coefficients[ket_primitive] *
                  batch.primitive_coefficients[bra_primitive];
              if (!isfinite(primitive_prefactor)) {
                finite = false;
                continue;
              }
              double axis[3][6][3];
              for (int coordinate = 0; coordinate < 3; ++coordinate) {
                const int maximum_ket_power =
                    static_cast<int>(ket_l) + (multipoles_enabled ? 3 : 1);
                make_axis_overlap(-vector[coordinate] * bra_alpha * inverse_sum,
                                  vector[coordinate] * ket_alpha * inverse_sum, 0.5 * inverse_sum,
                                  maximum_ket_power, static_cast<int>(bra_l), axis[coordinate]);
              }
              const double one_dimensional[3] = {axis[0][ket_power[0]][bra_power[0]],
                                                 axis[1][ket_power[1]][bra_power[1]],
                                                 axis[2][ket_power[2]][bra_power[2]]};
              overlap += primitive_prefactor * one_dimensional[0] * one_dimensional[1] *
                         one_dimensional[2];
              for (int coordinate = 0; coordinate < 3; ++coordinate) {
                double derivative_1d =
                    2.0 * ket_alpha *
                    axis[coordinate][ket_power[coordinate] + 1][bra_power[coordinate]];
                if (ket_power[coordinate] > 0) {
                  derivative_1d -=
                      static_cast<double>(ket_power[coordinate]) *
                      axis[coordinate][ket_power[coordinate] - 1][bra_power[coordinate]];
                }
                overlap_gradient[coordinate] += primitive_prefactor * derivative_1d *
                                                one_dimensional[(coordinate + 1) % 3] *
                                                one_dimensional[(coordinate + 2) % 3];
              }
              for (int component = 0; multipoles_enabled && component < kMultipoleComponents;
                   ++component) {
                int moment_power[3];
                multipole_power(component, &moment_power[0], &moment_power[1], &moment_power[2]);
                const double moment_axis[3] = {
                    axis[0][ket_power[0] + moment_power[0]][bra_power[0]],
                    axis[1][ket_power[1] + moment_power[1]][bra_power[1]],
                    axis[2][ket_power[2] + moment_power[2]][bra_power[2]]};
                multipoles[component] +=
                    primitive_prefactor * moment_axis[0] * moment_axis[1] * moment_axis[2];
                for (int coordinate = 0; coordinate < 3; ++coordinate) {
                  const int exponent = ket_power[coordinate] + moment_power[coordinate];
                  double derivative_1d =
                      2.0 * ket_alpha * axis[coordinate][exponent + 1][bra_power[coordinate]];
                  if (exponent > 0) {
                    derivative_1d -= static_cast<double>(exponent) *
                                     axis[coordinate][exponent - 1][bra_power[coordinate]];
                  }
                  multipole_gradient[coordinate][component] += primitive_prefactor * derivative_1d *
                                                               moment_axis[(coordinate + 1) % 3] *
                                                               moment_axis[(coordinate + 2) % 3];
                }
              }
            }
          }
          finite = finite && isfinite(overlap);
          for (int coordinate = 0; coordinate < 3; ++coordinate) {
            finite = finite && isfinite(overlap_gradient[coordinate]);
          }
          for (int component = 0; multipoles_enabled && component < kMultipoleComponents;
               ++component) {
            finite = finite && isfinite(multipoles[component]);
            for (int coordinate = 0; coordinate < 3; ++coordinate) {
              finite = finite && isfinite(multipole_gradient[coordinate][component]);
            }
          }
          double vector_adjoint[3] = {};
          if (multipoles_enabled && publish_reverse) {
            double ignored_overlap_adjoint = 0.0;
            double ignored_dipole_adjoint[3] = {};
            multipole_shift_pullback_device(vector, overlap, multipoles, reverse_dipole_adjoint,
                                            reverse_quadrupole_adjoint, &ignored_overlap_adjoint,
                                            ignored_dipole_adjoint, vector_adjoint);
          }
          const double coefficient = bra_coefficient * ket_coefficient;
          for (int coordinate = 0; coordinate < 3; ++coordinate) {
            double value =
                vector_adjoint[coordinate] + overlap_adjoint * overlap_gradient[coordinate];
            if (multipoles_enabled) {
              for (int component = 0; component < 3; ++component) {
                value += dipole_adjoint[component] * multipole_gradient[coordinate][component];
              }
              const double trace =
                  0.5 * (multipole_gradient[coordinate][3] + multipole_gradient[coordinate][5] +
                         multipole_gradient[coordinate][8]);
              const double quadrupole_gradient[6] = {
                  1.5 * multipole_gradient[coordinate][3] - trace,
                  1.5 * multipole_gradient[coordinate][4],
                  1.5 * multipole_gradient[coordinate][5] - trace,
                  1.5 * multipole_gradient[coordinate][6],
                  1.5 * multipole_gradient[coordinate][7],
                  1.5 * multipole_gradient[coordinate][8] - trace};
              for (int component = 0; component < 6; ++component) {
                value += quadrupole_adjoint[component] * quadrupole_gradient[component];
              }
            }
            derivative[coordinate] += coefficient * value;
            finite = finite && isfinite(value) && isfinite(derivative[coordinate]);
          }
        }
      }
      if (!finite) {
        record_system_error(system_errors, system, device_error,
                            Gfn2IntegralDeviceError::kNonfiniteGradientArithmetic);
        return;
      }
      if (bra_atom != ket_atom) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          if (!force_add_atomic(workspace.gradient_scratch + ket_atom * 3 + coordinate,
                                derivative[coordinate]) ||
              !force_add_atomic(workspace.gradient_scratch + bra_atom * 3 + coordinate,
                                -derivative[coordinate])) {
            record_system_error(system_errors, system, device_error,
                                Gfn2IntegralDeviceError::kNonfiniteGradientArithmetic);
            return;
          }
        }
      }
    }
  }
}

__global__ void publish_integral_force_kernel(Gfn2IntegralDeviceBatch batch,
                                              Gfn2ForceDeviceActivity activity,
                                              Gfn2IntegralForceDeviceOutput output,
                                              Gfn2IntegralForceDeviceWorkspace workspace,
                                              const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t tile = static_cast<std::int64_t>(blockIdx.y);
  if (!force_sequence_is_active(workspace) || activity.requested_mask[system] != 1u ||
      activity.system_statuses[system] != XTBLOOM_STATUS_SUCCESS ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t stride = batch.linear_tiles_per_system * blockDim.x;
  for (std::int64_t atom = atom_begin + tile * blockDim.x + threadIdx.x; atom < atom_end;
       atom += stride) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      output.gradients[atom * 3 + coordinate] = workspace.gradient_scratch[atom * 3 + coordinate];
    }
  }
}

cudaError_t validate_integral_force_descriptors(
    const Gfn2IntegralDeviceBatch& batch, const Gfn2ForceDeviceActivity& activity,
    const Gfn2IntegralForceDeviceInput& input, const Gfn2IntegralForceDeviceOutput& output,
    const Gfn2IntegralForceDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, std::int64_t* maximum_pair_blocks,
    unsigned int* grid_blocks) noexcept {
  Gfn2IntegralDeviceWorkspace common_workspace{};
  common_workspace.sequence_active = workspace.sequence_active;
  common_workspace.sequence_elements = workspace.sequence_elements;
  common_workspace.plan_token = workspace.plan_token;
  cudaError_t status = validate_common(batch, common_workspace, system_errors, device_error,
                                       maximum_pair_blocks, grid_blocks);
  const std::int64_t coordinates = batch.total_atoms * 3;
  const bool multipoles_enabled = batch.model == XtbModelFlavor::kGfn2;
  const auto& periodic = input.native_periodic;
  const bool native_enabled = periodic.plan_token != 0u;
  const bool native_fields_empty =
      periodic.translation_offset_elements == 0 && periodic.translation_elements == 0 &&
      periodic.max_translations_per_system == 0 && periodic.realspace_cutoff == 0.0 &&
      periodic.translation_offsets == nullptr && periodic.translations == nullptr;
  const std::int64_t dipoles =
      multipoles_enabled ? batch.total_matrix_elements * kGfn2IntegralDipoleComponents : 0;
  const std::int64_t quadrupoles =
      multipoles_enabled ? batch.total_matrix_elements * kGfn2IntegralQuadrupoleComponents : 0;
  if (status != cudaSuccess || activity.plan_token != batch.plan_token ||
      input.plan_token != batch.plan_token || output.plan_token != batch.plan_token ||
      workspace.plan_token != batch.plan_token || activity.batch_elements != batch.batch_size ||
      input.position_elements < coordinates ||
      input.overlap_adjoint_elements < batch.total_matrix_elements ||
      input.dipole_adjoint_elements < dipoles || input.quadrupole_adjoint_elements < quadrupoles ||
      (!multipoles_enabled &&
       (input.dipole_adjoint != nullptr || input.dipole_adjoint_elements != 0 ||
        input.quadrupole_adjoint != nullptr || input.quadrupole_adjoint_elements != 0)) ||
      (native_enabled &&
       (periodic.plan_token != batch.plan_token ||
        periodic.translation_offset_elements != batch.batch_size + 1 ||
        periodic.translation_elements <= 0 || periodic.max_translations_per_system <= 0 ||
        periodic.max_translations_per_system > periodic.translation_elements ||
        !(periodic.realspace_cutoff > 0.0) || !std::isfinite(periodic.realspace_cutoff) ||
        !std::isfinite(periodic.realspace_cutoff * periodic.realspace_cutoff) ||
        !is_aligned(periodic.translation_offsets, alignof(std::int64_t)) ||
        !is_aligned(periodic.translations, alignof(Gfn2CudaPeriodicTranslation)))) ||
      (!native_enabled && !native_fields_empty) ||
      (native_enabled &&
       (input.density_elements < batch.total_matrix_elements ||
        input.coordination_elements < batch.total_atoms ||
        input.h0_plan.plan_token != batch.plan_token ||
        input.h0_plan.atomic_radius_count < batch.total_atoms ||
        input.h0_plan.shell_level_count < batch.total_shells ||
        input.h0_plan.shell_coordination_scale_count < batch.total_shells ||
        input.h0_plan.shell_polynomial_count < batch.total_shells ||
        input.h0_plan.shell_pair_scale_count < batch.total_shell_pair_elements ||
        !required_pointer(input.density, batch.total_matrix_elements) ||
        !required_pointer(input.coordination_numbers, batch.total_atoms) ||
        !required_pointer(input.h0_plan.atomic_radii, batch.total_atoms) ||
        !required_pointer(input.h0_plan.shell_levels, batch.total_shells) ||
        !required_pointer(input.h0_plan.shell_coordination_scale, batch.total_shells) ||
        !required_pointer(input.h0_plan.shell_polynomial, batch.total_shells) ||
        !required_pointer(input.h0_plan.shell_pair_scale, batch.total_shell_pair_elements))) ||
      output.gradient_elements < coordinates || workspace.gradient_elements < coordinates ||
      !is_aligned(activity.requested_mask, alignof(std::uint8_t)) ||
      !is_aligned(activity.system_statuses, alignof(xtbloom_status_t)) ||
      !required_pointer(input.positions, coordinates) ||
      !required_pointer(input.overlap_adjoint, batch.total_matrix_elements) ||
      !required_pointer(input.dipole_adjoint, dipoles) ||
      !required_pointer(input.quadrupole_adjoint, quadrupoles) ||
      !required_pointer(output.gradients, coordinates) ||
      !required_pointer(workspace.gradient_scratch, coordinates)) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const bool compact = batch.use_compact_tasks != 0u;
  if (compact) *grid_blocks = static_cast<unsigned int>(batch.force_generic_task_count);
  std::array<MemoryRange, 29> reads;
  std::array<MemoryRange, 5> writes;
  if (!make_range(batch.atom_offsets, batch.atom_offset_count, sizeof(*batch.atom_offsets),
                  &reads[0]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_count,
                  sizeof(*batch.batch_shell_offsets), &reads[1]) ||
      !make_range(batch.batch_orbital_offsets, batch.batch_orbital_offset_count,
                  sizeof(*batch.batch_orbital_offsets), &reads[2]) ||
      !make_range(batch.matrix_offsets, batch.matrix_offset_count, sizeof(*batch.matrix_offsets),
                  &reads[3]) ||
      !make_range(batch.shell_pair_offsets, batch.shell_pair_offset_count,
                  sizeof(*batch.shell_pair_offsets), &reads[4]) ||
      !make_range(batch.atom_shell_offsets, batch.atom_shell_offset_count,
                  sizeof(*batch.atom_shell_offsets), &reads[5]) ||
      !make_range(batch.shell_orbital_offsets, batch.shell_orbital_offset_count,
                  sizeof(*batch.shell_orbital_offsets), &reads[6]) ||
      !make_range(batch.shell_primitive_offsets, batch.shell_primitive_offset_count,
                  sizeof(*batch.shell_primitive_offsets), &reads[7]) ||
      !make_range(batch.shell_to_atom, batch.shell_to_atom_count, sizeof(*batch.shell_to_atom),
                  &reads[8]) ||
      !make_range(batch.angular_momenta, batch.angular_momentum_count,
                  sizeof(*batch.angular_momenta), &reads[9]) ||
      !make_range(batch.primitive_exponents, batch.primitive_exponent_count,
                  sizeof(*batch.primitive_exponents), &reads[10]) ||
      !make_range(batch.primitive_coefficients, batch.primitive_coefficient_count,
                  sizeof(*batch.primitive_coefficients), &reads[11]) ||
      !make_range(activity.requested_mask, batch.batch_size, sizeof(*activity.requested_mask),
                  &reads[12]) ||
      !make_range(activity.system_statuses, batch.batch_size, sizeof(*activity.system_statuses),
                  &reads[13]) ||
      !make_range(input.positions, coordinates, sizeof(*input.positions), &reads[14]) ||
      !make_range(input.overlap_adjoint, batch.total_matrix_elements,
                  sizeof(*input.overlap_adjoint), &reads[15]) ||
      !make_range(input.dipole_adjoint, dipoles, sizeof(*input.dipole_adjoint), &reads[16]) ||
      !make_range(input.quadrupole_adjoint, quadrupoles, sizeof(*input.quadrupole_adjoint),
                  &reads[17]) ||
      !make_range(input.density, native_enabled ? batch.total_matrix_elements : 0,
                  sizeof(*input.density), &reads[22]) ||
      !make_range(input.coordination_numbers, native_enabled ? batch.total_atoms : 0,
                  sizeof(*input.coordination_numbers), &reads[23]) ||
      !make_range(input.h0_plan.atomic_radii, native_enabled ? batch.total_atoms : 0,
                  sizeof(*input.h0_plan.atomic_radii), &reads[24]) ||
      !make_range(input.h0_plan.shell_levels, native_enabled ? batch.total_shells : 0,
                  sizeof(*input.h0_plan.shell_levels), &reads[25]) ||
      !make_range(input.h0_plan.shell_coordination_scale, native_enabled ? batch.total_shells : 0,
                  sizeof(*input.h0_plan.shell_coordination_scale), &reads[26]) ||
      !make_range(input.h0_plan.shell_polynomial, native_enabled ? batch.total_shells : 0,
                  sizeof(*input.h0_plan.shell_polynomial), &reads[27]) ||
      !make_range(input.h0_plan.shell_pair_scale,
                  native_enabled ? batch.total_shell_pair_elements : 0,
                  sizeof(*input.h0_plan.shell_pair_scale), &reads[28]) ||
      !make_range(batch.force_generic_tasks, compact ? batch.force_generic_task_count : 0,
                  sizeof(*batch.force_generic_tasks), &reads[18]) ||
      !make_range(batch.force_ss_tasks, compact ? batch.force_ss_task_count : 0,
                  sizeof(*batch.force_ss_tasks), &reads[19]) ||
      !make_range(periodic.translation_offsets,
                  native_enabled ? periodic.translation_offset_elements : 0,
                  sizeof(*periodic.translation_offsets), &reads[20]) ||
      !make_range(periodic.translations, native_enabled ? periodic.translation_elements : 0,
                  sizeof(*periodic.translations), &reads[21]) ||
      !make_range(output.gradients, coordinates, sizeof(*output.gradients), &writes[0]) ||
      !make_range(workspace.gradient_scratch, coordinates, sizeof(*workspace.gradient_scratch),
                  &writes[1]) ||
      !make_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active), &writes[2]) ||
      !make_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[3]) ||
      !make_range(device_error, 1, sizeof(*device_error), &writes[4]) ||
      !writes_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t reset_gfn2_integral_force_device_errors_cuda(std::int64_t batch_size,
                                                         std::uint32_t* system_errors,
                                                         std::uint32_t* device_error,
                                                         cudaStream_t stream) noexcept {
  return reset_gfn2_integral_device_errors_cuda(batch_size, system_errors, device_error, stream);
}

cudaError_t add_gfn2_integral_gradient_cuda(
    const Gfn2IntegralDeviceBatch& batch, const Gfn2ForceDeviceActivity& activity,
    const Gfn2IntegralForceDeviceInput& input, const Gfn2IntegralForceDeviceOutput& output,
    const Gfn2IntegralForceDeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  std::int64_t maximum_pair_blocks = 0;
  unsigned int grid_blocks = 0u;
  cudaError_t status =
      validate_integral_force_descriptors(batch, activity, input, output, workspace, system_errors,
                                          device_error, &maximum_pair_blocks, &grid_blocks);
  if (status != cudaSuccess) {
    return status;
  }
  capture_force_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  Gfn2IntegralLinearLaunchShape linear_shape{};
  if (!make_gfn2_integral_linear_launch_shape(batch.batch_size, batch.linear_tiles_per_system,
                                              linear_shape)) {
    return cudaErrorInvalidValue;
  }
  const dim3 linear_grid(linear_shape.systems, linear_shape.tiles, 1u);
  integral_force_topology_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size),
                                             kThreadsPerBlock, 0, stream>>>(
      batch, activity, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  integral_force_numerical_preflight_kernel<<<linear_grid, kThreadsPerBlock, 0, stream>>>(
      batch, activity, input, output, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  if (input.native_periodic.plan_token != 0u) {
    const std::int64_t shell_limit = batch.maximum_system_shells;
    if (shell_limit <= 0 || shell_limit > kMaximumInt64 / shell_limit) {
      return cudaErrorInvalidValue;
    }
    const std::int64_t maximum_pair_blocks = shell_limit * shell_limit;
    const std::int64_t max_translations = input.native_periodic.max_translations_per_system;
    if (max_translations <= 0 || max_translations > kMaximumInt64 / maximum_pair_blocks) {
      return cudaErrorInvalidValue;
    }
    const std::int64_t image_system_stride = max_translations * maximum_pair_blocks;
    if (batch.batch_size > kMaximumInt64 / image_system_stride) {
      return cudaErrorInvalidValue;
    }
    const std::int64_t image_tasks = batch.batch_size * image_system_stride;
    if (batch.batch_size > kMaximumInt64 / maximum_pair_blocks) {
      return cudaErrorInvalidValue;
    }
    const std::int64_t onsite_tasks = batch.batch_size * maximum_pair_blocks;
    if (image_tasks > kMaximumInt64 - onsite_tasks ||
        image_tasks + onsite_tasks >
            static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max())) {
      return cudaErrorInvalidConfiguration;
    }
    native_periodic_integral_force_kernel<<<static_cast<unsigned int>(image_tasks + onsite_tasks),
                                            kThreadsPerBlock, 0, stream>>>(
        batch, activity, input, workspace, system_errors, device_error, image_tasks,
        image_system_stride, maximum_pair_blocks);
    status = check_launch();
    if (status != cudaSuccess) return status;
    publish_integral_force_kernel<<<linear_grid, kThreadsPerBlock, 0, stream>>>(
        batch, activity, output, workspace, system_errors);
    return check_launch();
  }
  if (batch.use_compact_tasks != 0u && batch.force_ss_task_count > 0) {
    const auto ss_grid = static_cast<unsigned int>(
        (batch.force_ss_task_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    integral_force_ss_task_kernel<<<ss_grid, kThreadsPerBlock, 0, stream>>>(
        batch, activity, input, workspace, system_errors, device_error, batch.force_ss_tasks,
        batch.force_ss_task_count);
    status = check_launch();
    if (status != cudaSuccess) return status;
  }
  if (grid_blocks != 0u) {
    integral_force_shell_pair_kernel<<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
        batch, activity, input, workspace, system_errors, device_error,
        batch.use_compact_tasks != 0u ? batch.force_generic_tasks : nullptr, maximum_pair_blocks);
    status = check_launch();
    if (status != cudaSuccess) {
      return status;
    }
  }
  publish_integral_force_kernel<<<linear_grid, kThreadsPerBlock, 0, stream>>>(
      batch, activity, output, workspace, system_errors);
  return check_launch();
}

}  // namespace xtbloom::detail::cuda
