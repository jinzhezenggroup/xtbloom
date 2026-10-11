#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/gfn2_h0_force.cuh"
#include "backends/cuda/periodic_wrap.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 128;
constexpr std::int64_t kMaximumInt64 = 9223372036854775807LL;
constexpr double kMinimumDistanceSquared = 1.0e-24;
constexpr double kSqrtPiCubed = 5.5683279968317061;
constexpr int kNativeMaximumCartesianBlock = 36;
constexpr double kNativeMinimumImageDistanceSquared = 2.220446049250313080847263336181640625e-16;

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

__device__ bool sequence_is_active(const Gfn2H0ForceDeviceWorkspace& workspace) {
  return atomicAdd(workspace.sequence_active, 0u) == 1u;
}

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2H0ForceDeviceError::kSuccess);
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2H0ForceDeviceError error) {
  const std::uint32_t success = static_cast<std::uint32_t>(Gfn2H0ForceDeviceError::kSuccess);
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system, success, code) == success) {
    atomicCAS(device_error, success, code);
  }
}

__device__ bool valid_range(std::int64_t begin, std::int64_t end, std::int64_t total) {
  return begin >= 0 && begin <= end && end <= total;
}

__device__ bool checked_square(std::int64_t value, std::int64_t* square) {
  if (value < 0 || (value != 0 && value > kMaximumInt64 / value)) {
    return false;
  }
  *square = value * value;
  return true;
}

__device__ int native_cartesian_count(std::uint8_t angular_momentum) {
  const int l = static_cast<int>(angular_momentum);
  return (l + 1) * (l + 2) / 2;
}

__device__ int native_spherical_count(std::uint8_t angular_momentum) {
  return 2 * static_cast<int>(angular_momentum) + 1;
}

__device__ void native_cartesian_exponent(std::uint8_t angular_momentum, int function, int* x,
                                          int* y, int* z) {
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

__device__ void native_make_axis_overlap(double product_minus_i, double product_minus_j,
                                         double inverse_twice_sum, int maximum_a, int maximum_b,
                                         double overlap[6][3]) {
#pragma unroll
  for (int a = 0; a < 6; ++a) {
#pragma unroll
    for (int b = 0; b < 3; ++b) overlap[a][b] = 0.0;
  }
  overlap[0][0] = 1.0;
  for (int a = 1; a <= maximum_a; ++a) {
    overlap[a][0] = product_minus_i * overlap[a - 1][0];
    if (a > 1) overlap[a][0] += static_cast<double>(a - 1) * inverse_twice_sum * overlap[a - 2][0];
  }
  for (int b = 1; b <= maximum_b; ++b) {
    overlap[0][b] = product_minus_j * overlap[0][b - 1];
    if (b > 1) overlap[0][b] += static_cast<double>(b - 1) * inverse_twice_sum * overlap[0][b - 2];
    for (int a = 1; a <= maximum_a; ++a) {
      overlap[a][b] = product_minus_i * overlap[a - 1][b] +
                      static_cast<double>(b) * inverse_twice_sum * overlap[a - 1][b - 1];
      if (a > 1)
        overlap[a][b] += static_cast<double>(a - 1) * inverse_twice_sum * overlap[a - 2][b];
    }
  }
}

__device__ double native_spherical_coefficient(std::uint8_t angular_momentum, int spherical,
                                               int cartesian) {
  constexpr double sqrt_three = 1.732050807568877293527446341505872367;
  if (angular_momentum == 0u) return spherical == 0 && cartesian == 0 ? 1.0 : 0.0;
  if (angular_momentum == 1u) {
    const int selected = spherical == 0 ? 1 : (spherical == 1 ? 2 : 0);
    return cartesian == selected ? 1.0 : 0.0;
  }
  if (spherical == 0) return cartesian == 1 ? sqrt_three : 0.0;
  if (spherical == 1) return cartesian == 4 ? sqrt_three : 0.0;
  if (spherical == 2) {
    return cartesian == 0 || cartesian == 3 ? -0.5 : (cartesian == 5 ? 1.0 : 0.0);
  }
  if (spherical == 3) return cartesian == 2 ? sqrt_three : 0.0;
  return cartesian == 0 ? 0.5 * sqrt_three : (cartesian == 3 ? -0.5 * sqrt_three : 0.0);
}

/* Recompute one periodic overlap block in the same primitive order as the
 * forward native evaluator.  H0's reverse needs the per-image overlap rather
 * than only the already-summed periodic matrix. */
__device__ bool native_overlap_block(const Gfn2IntegralDeviceBatch& batch, std::int64_t bra_shell,
                                     std::int64_t ket_shell, const double vector[3],
                                     double distance_squared, int bra_ao, int ket_ao,
                                     double& value) {
  const std::uint8_t bra_l = batch.angular_momenta[bra_shell];
  const std::uint8_t ket_l = batch.angular_momenta[ket_shell];
  const int bra_cartesian_count = native_cartesian_count(bra_l);
  const int ket_cartesian_count = native_cartesian_count(ket_l);
  const int spherical_bra_count = native_spherical_count(bra_l);
  const int spherical_ket_count = native_spherical_count(ket_l);
  if (bra_cartesian_count * ket_cartesian_count > kNativeMaximumCartesianBlock || bra_ao < 0 ||
      bra_ao >= spherical_bra_count || ket_ao < 0 || ket_ao >= spherical_ket_count) {
    return false;
  }
  /* Keep the reduction order identical to integral_shell_pair_kernel: one
   * Cartesian contraction owns the complete ket-primitive/bra-primitive
   * sequence, and only then is the sparse spherical transform applied.  A
   * primitive-major spherical accumulation is mathematically equivalent but
   * changes cancellation in diffuse image blocks enough to perturb the
   * periodic H0 force at the requested binary64 precision. */
  double cartesian_overlap[kNativeMaximumCartesianBlock] = {};
  for (int cartesian_index = 0; cartesian_index < bra_cartesian_count * ket_cartesian_count;
       ++cartesian_index) {
    const int bra_cartesian = cartesian_index / ket_cartesian_count;
    const int ket_cartesian = cartesian_index % ket_cartesian_count;
    int bra_power[3];
    int ket_power[3];
    native_cartesian_exponent(bra_l, bra_cartesian, &bra_power[0], &bra_power[1], &bra_power[2]);
    native_cartesian_exponent(ket_l, ket_cartesian, &ket_power[0], &ket_power[1], &ket_power[2]);
    double overlap_value = 0.0;
    for (std::int64_t ket_primitive = batch.shell_primitive_offsets[ket_shell];
         ket_primitive < batch.shell_primitive_offsets[ket_shell + 1]; ++ket_primitive) {
      const double ket_alpha = batch.primitive_exponents[ket_primitive];
      for (std::int64_t bra_primitive = batch.shell_primitive_offsets[bra_shell];
           bra_primitive < batch.shell_primitive_offsets[bra_shell + 1]; ++bra_primitive) {
        const double bra_alpha = batch.primitive_exponents[bra_primitive];
        const double alpha_sum = ket_alpha + bra_alpha;
        const double inverse_sum = 1.0 / alpha_sum;
        const double product_exponent = ket_alpha * bra_alpha * distance_squared * inverse_sum;
        if (!(alpha_sum > 0.0) || !isfinite(alpha_sum) || !isfinite(inverse_sum) ||
            !isfinite(product_exponent)) {
          return false;
        }
        if (product_exponent > batch.integral_cutoff) continue;
        const double sqrt_inverse_sum = sqrt(inverse_sum);
        const double primitive_prefactor = exp(-product_exponent) * kSqrtPiCubed *
                                           sqrt_inverse_sum * sqrt_inverse_sum * sqrt_inverse_sum *
                                           batch.primitive_coefficients[ket_primitive] *
                                           batch.primitive_coefficients[bra_primitive];
        if (!isfinite(primitive_prefactor)) return false;
        double axis[3][6][3];
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          native_make_axis_overlap(-vector[coordinate] * bra_alpha * inverse_sum,
                                   vector[coordinate] * ket_alpha * inverse_sum, 0.5 * inverse_sum,
                                   static_cast<int>(ket_l) + 2, static_cast<int>(bra_l),
                                   axis[coordinate]);
        }
        const double local = primitive_prefactor * axis[0][ket_power[0]][bra_power[0]] *
                             axis[1][ket_power[1]][bra_power[1]] *
                             axis[2][ket_power[2]][bra_power[2]];
        if (!isfinite(local)) return false;
        overlap_value += local;
      }
    }
    if (!isfinite(overlap_value)) return false;
    cartesian_overlap[cartesian_index] = overlap_value;
  }

  value = 0.0;
  for (int bra_cartesian = 0; bra_cartesian < bra_cartesian_count; ++bra_cartesian) {
    const double bra_coefficient = native_spherical_coefficient(bra_l, bra_ao, bra_cartesian);
    if (bra_coefficient == 0.0) continue;
    for (int ket_cartesian = 0; ket_cartesian < ket_cartesian_count; ++ket_cartesian) {
      const double ket_coefficient = native_spherical_coefficient(ket_l, ket_ao, ket_cartesian);
      if (ket_coefficient == 0.0) continue;
      const int index = bra_cartesian * ket_cartesian_count + ket_cartesian;
      value += bra_coefficient * cartesian_overlap[index] * ket_coefficient;
    }
  }
  return isfinite(value);
}

__device__ bool native_h0_factor(const Gfn2IntegralDeviceBatch& batch, const Gfn2H0DevicePlan& h0,
                                 const double* coordination, std::int64_t system,
                                 std::int64_t bra_atom, std::int64_t ket_atom,
                                 std::int64_t bra_shell, std::int64_t ket_shell, bool onsite,
                                 double distance_squared, double& factor, double& spatial_scale,
                                 double& spatial_scale_derivative, double& distance) {
  const double bra_level =
      h0.shell_levels[bra_shell] - h0.shell_coordination_scale[bra_shell] * coordination[bra_atom];
  const double ket_level =
      h0.shell_levels[ket_shell] - h0.shell_coordination_scale[ket_shell] * coordination[ket_atom];
  if (!isfinite(bra_level) || !isfinite(ket_level)) return false;
  const double average = 0.5 * (bra_level + ket_level);
  if (!isfinite(average)) return false;
  spatial_scale = 1.0;
  spatial_scale_derivative = 0.0;
  distance = 0.0;
  if (!onsite) {
    distance = sqrt(distance_squared);
    const double radius_sum = h0.atomic_radii[bra_atom] + h0.atomic_radii[ket_atom];
    if (!(distance > 0.0) || !isfinite(distance) || !(radius_sum > 0.0) || !isfinite(radius_sum))
      return false;
    const double reduced = sqrt(distance / radius_sum);
    const double bra_polynomial = 1.0 + h0.shell_polynomial[bra_shell] * reduced;
    const double ket_polynomial = 1.0 + h0.shell_polynomial[ket_shell] * reduced;
    const std::int64_t shell_begin = batch.batch_shell_offsets[system];
    const std::int64_t shell_count = batch.batch_shell_offsets[system + 1] - shell_begin;
    const std::int64_t local_bra = bra_shell - shell_begin;
    const std::int64_t local_ket = ket_shell - shell_begin;
    const double pair_scale =
        h0.shell_pair_scale[batch.shell_pair_offsets[system] + local_bra * shell_count + local_ket];
    const double polynomial_derivative = (h0.shell_polynomial[bra_shell] * ket_polynomial +
                                          h0.shell_polynomial[ket_shell] * bra_polynomial) *
                                         reduced / (2.0 * distance);
    spatial_scale = pair_scale * bra_polynomial * ket_polynomial;
    spatial_scale_derivative = pair_scale * polynomial_derivative;
    if (!(pair_scale > 0.0) || !isfinite(pair_scale) || !isfinite(spatial_scale) ||
        !isfinite(spatial_scale_derivative))
      return false;
  }
  factor = average * spatial_scale;
  return isfinite(factor);
}

__device__ bool add_finite_atomic(double* target, double contribution) {
  if (!isfinite(contribution)) {
    return false;
  }
  const double previous = atomic_add_fp64(target, contribution);
  return isfinite(previous) && isfinite(previous + contribution);
}

__global__ void capture_sequence_kernel(const std::uint32_t* device_error,
                                        Gfn2H0ForceDeviceWorkspace workspace) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspace.sequence_active =
        atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
                static_cast<std::uint32_t>(Gfn2H0ForceDeviceError::kSuccess)
            ? 1u
            : 0u;
  }
}

/*
 * Return true only for requested members whose terminal SCC status permits a
 * stationary force. The status is intentionally not read for mask==0.
 */
__device__ bool load_force_gate(const Gfn2ForceDeviceActivity& activity, std::int64_t system,
                                int* selected, std::uint32_t* system_errors,
                                std::uint32_t* device_error) {
  if (threadIdx.x == 0) {
    *selected = 0;
    const std::uint8_t requested = activity.requested_mask[system];
    if (requested > 1u) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidActiveMask);
    } else if (requested == 1u && activity.system_statuses[system] == XTBLOOM_STATUS_SUCCESS) {
      *selected = 1;
    }
  }
  __syncthreads();
  return *selected != 0 && system_is_valid(system_errors, system);
}

__device__ bool load_ranges(const Gfn2IntegralDeviceBatch& batch, std::int64_t system,
                            SystemRanges* ranges) {
  ranges->atom_begin = batch.atom_offsets[system];
  ranges->atom_end = batch.atom_offsets[system + 1];
  ranges->shell_begin = batch.batch_shell_offsets[system];
  ranges->shell_end = batch.batch_shell_offsets[system + 1];
  ranges->orbital_begin = batch.batch_orbital_offsets[system];
  ranges->orbital_end = batch.batch_orbital_offsets[system + 1];
  ranges->matrix_begin = batch.matrix_offsets[system];
  ranges->matrix_end = batch.matrix_offsets[system + 1];
  ranges->shell_pair_begin = batch.shell_pair_offsets[system];
  ranges->shell_pair_end = batch.shell_pair_offsets[system + 1];
  if (!valid_range(ranges->atom_begin, ranges->atom_end, batch.total_atoms) ||
      !valid_range(ranges->shell_begin, ranges->shell_end, batch.total_shells) ||
      !valid_range(ranges->orbital_begin, ranges->orbital_end, batch.total_orbitals) ||
      !valid_range(ranges->matrix_begin, ranges->matrix_end, batch.total_matrix_elements) ||
      !valid_range(ranges->shell_pair_begin, ranges->shell_pair_end,
                   batch.total_shell_pair_elements)) {
    return false;
  }
  std::int64_t expected_matrix = 0;
  std::int64_t expected_pairs = 0;
  const std::int64_t orbitals = ranges->orbital_end - ranges->orbital_begin;
  const std::int64_t shells = ranges->shell_end - ranges->shell_begin;
  return checked_square(orbitals, &expected_matrix) && checked_square(shells, &expected_pairs) &&
         ranges->matrix_end - ranges->matrix_begin == expected_matrix &&
         ranges->shell_pair_end - ranges->shell_pair_begin == expected_pairs &&
         batch.atom_shell_offsets[ranges->atom_begin] == ranges->shell_begin &&
         batch.atom_shell_offsets[ranges->atom_end] == ranges->shell_end &&
         batch.shell_orbital_offsets[ranges->shell_begin] == ranges->orbital_begin &&
         batch.shell_orbital_offsets[ranges->shell_end] == ranges->orbital_end &&
         (system != 0 ||
          (ranges->atom_begin == 0 && ranges->shell_begin == 0 && ranges->orbital_begin == 0 &&
           ranges->matrix_begin == 0 && ranges->shell_pair_begin == 0)) &&
         (system + 1 != batch.batch_size ||
          (ranges->atom_end == batch.total_atoms && ranges->shell_end == batch.total_shells &&
           ranges->orbital_end == batch.total_orbitals &&
           ranges->matrix_end == batch.total_matrix_elements &&
           ranges->shell_pair_end == batch.total_shell_pair_elements));
}

__global__ void preflight_and_seed_kernel(Gfn2IntegralDeviceBatch batch, Gfn2H0DevicePlan h0_plan,
                                          Gfn2ForceDeviceActivity activity,
                                          Gfn2H0ForceDeviceInput input,
                                          Gfn2H0ForceDeviceOutput output,
                                          Gfn2H0ForceDeviceWorkspace workspace,
                                          std::uint32_t* system_errors,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t tile = static_cast<std::int64_t>(blockIdx.y);
  __shared__ SystemRanges ranges;
  __shared__ int selected;
  __shared__ int valid;
  if (!sequence_is_active(workspace) ||
      !load_force_gate(activity, system, &selected, system_errors, device_error)) {
    return;
  }
  if (threadIdx.x == 0) {
    valid = load_ranges(batch, system, &ranges) ? 1 : 0;
    if (valid == 0) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidOffsets);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  const std::int64_t stride = batch.linear_tiles_per_system * blockDim.x;
  for (std::int64_t atom = ranges.atom_begin + tile * blockDim.x + threadIdx.x;
       atom < ranges.atom_end; atom += stride) {
    const std::int64_t shell_begin = batch.atom_shell_offsets[atom];
    const std::int64_t shell_end = batch.atom_shell_offsets[atom + 1];
    if (!valid_range(shell_begin, shell_end, batch.total_shells) ||
        shell_begin < ranges.shell_begin || shell_end > ranges.shell_end ||
        shell_begin == shell_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidOffsets);
      atomicExch(&valid, 0);
    }
    const double radius = h0_plan.atomic_radii[atom];
    const double coordination = input.coordination_numbers[atom];
    if (!(radius > 0.0) || !isfinite(radius)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidH0Parameter);
      atomicExch(&valid, 0);
    }
    if (!isfinite(coordination)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteInput);
      atomicExch(&valid, 0);
    }
    const std::int64_t coordinate = atom * 3;
    for (int axis = 0; axis < 3; ++axis) {
      if (!isfinite(input.positions[coordinate + axis])) {
        record_system_error(system_errors, system, device_error,
                            Gfn2H0ForceDeviceError::kNonfinitePosition);
        atomicExch(&valid, 0);
      }
      const double seed = output.gradients[coordinate + axis];
      if (!isfinite(seed)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2H0ForceDeviceError::kNonfiniteOutputSeed);
        atomicExch(&valid, 0);
      } else {
        workspace.gradient_scratch[coordinate + axis] = seed;
      }
    }
    const double cn_seed = output.coordination_adjoint[atom];
    if (!isfinite(cn_seed)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteOutputSeed);
      atomicExch(&valid, 0);
    } else {
      workspace.coordination_adjoint_scratch[atom] = cn_seed;
    }
  }

  for (std::int64_t shell = ranges.shell_begin + tile * blockDim.x + threadIdx.x;
       shell < ranges.shell_end; shell += stride) {
    const std::int64_t atom = batch.shell_to_atom[shell];
    const std::int64_t orbital_begin = batch.shell_orbital_offsets[shell];
    const std::int64_t orbital_end = batch.shell_orbital_offsets[shell + 1];
    if (atom < ranges.atom_begin || atom >= ranges.atom_end ||
        shell < batch.atom_shell_offsets[atom] || shell >= batch.atom_shell_offsets[atom + 1] ||
        orbital_begin < ranges.orbital_begin || orbital_begin >= orbital_end ||
        orbital_end > ranges.orbital_end) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidShellMetadata);
      atomicExch(&valid, 0);
    }
    if (!isfinite(h0_plan.shell_levels[shell]) ||
        !isfinite(h0_plan.shell_coordination_scale[shell]) ||
        !isfinite(h0_plan.shell_polynomial[shell])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidH0Parameter);
      atomicExch(&valid, 0);
    }
  }

  for (std::int64_t pair = ranges.shell_pair_begin + tile * blockDim.x + threadIdx.x;
       pair < ranges.shell_pair_end; pair += stride) {
    const double scale = h0_plan.shell_pair_scale[pair];
    if (!(scale > 0.0) || !isfinite(scale)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kInvalidH0Parameter);
      atomicExch(&valid, 0);
    }
  }

  for (std::int64_t matrix = ranges.matrix_begin + tile * blockDim.x + threadIdx.x;
       matrix < ranges.matrix_end; matrix += stride) {
    const double overlap = input.overlap[matrix];
    const double density = input.density[matrix];
    const double weighted = input.energy_weighted_density[matrix];
    const double seed = output.overlap_adjoint[matrix];
    if (!isfinite(overlap) || !isfinite(density) || !isfinite(weighted)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteInput);
      atomicExch(&valid, 0);
    }
    const double pulay_seed = seed - weighted;
    if (!isfinite(seed)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteOutputSeed);
      atomicExch(&valid, 0);
    } else if (!isfinite(pulay_seed)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
      atomicExch(&valid, 0);
    } else {
      workspace.overlap_adjoint_scratch[matrix] = pulay_seed;
    }
  }
}

__global__ void contract_h0_pulay_kernel(Gfn2IntegralDeviceBatch batch, Gfn2H0DevicePlan h0_plan,
                                         Gfn2ForceDeviceActivity activity,
                                         Gfn2H0ForceDeviceInput input,
                                         Gfn2H0ForceDeviceWorkspace workspace,
                                         std::uint32_t* system_errors,
                                         std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int selected;
  if (!sequence_is_active(workspace) ||
      !load_force_gate(activity, system, &selected, system_errors, device_error)) {
    return;
  }

  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t shell_count = shell_end - shell_begin;
  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbital_count = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t pair_begin = batch.shell_pair_offsets[system];
  const std::int64_t pair_count = shell_count * shell_count;

  for (std::int64_t local_pair = threadIdx.x; local_pair < pair_count; local_pair += blockDim.x) {
    const std::int64_t first_shell = shell_begin + local_pair / shell_count;
    const std::int64_t second_shell = shell_begin + local_pair % shell_count;
    const std::int64_t first_atom = batch.shell_to_atom[first_shell];
    const std::int64_t second_atom = batch.shell_to_atom[second_shell];
    const double first_level =
        h0_plan.shell_levels[first_shell] -
        h0_plan.shell_coordination_scale[first_shell] * input.coordination_numbers[first_atom];
    const double second_level =
        h0_plan.shell_levels[second_shell] -
        h0_plan.shell_coordination_scale[second_shell] * input.coordination_numbers[second_atom];
    const double average_level = 0.5 * (first_level + second_level);
    double spatial_scale = 1.0;
    double spatial_scale_derivative = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double distance = 0.0;
    bool finite = isfinite(first_level) && isfinite(second_level) && isfinite(average_level);

    if (finite && first_atom != second_atom) {
      dx = input.positions[first_atom * 3] - input.positions[second_atom * 3];
      dy = input.positions[first_atom * 3 + 1] - input.positions[second_atom * 3 + 1];
      dz = input.positions[first_atom * 3 + 2] - input.positions[second_atom * 3 + 2];
      if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2H0ForceDeviceError::kCoordinateDifferenceOverflow);
        finite = false;
      }
      const double distance_squared = dx * dx + dy * dy + dz * dz;
      if (finite && !isfinite(distance_squared)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2H0ForceDeviceError::kCoordinateDifferenceOverflow);
        finite = false;
      } else if (finite && distance_squared <= kMinimumDistanceSquared) {
        record_system_error(system_errors, system, device_error,
                            Gfn2H0ForceDeviceError::kCoincidentAtoms);
        finite = false;
      }
      if (finite) {
        distance = sqrt(distance_squared);
        const double radius_sum =
            h0_plan.atomic_radii[first_atom] + h0_plan.atomic_radii[second_atom];
        const double reduced_distance = sqrt(distance / radius_sum);
        const double first_polynomial =
            1.0 + h0_plan.shell_polynomial[first_shell] * reduced_distance;
        const double second_polynomial =
            1.0 + h0_plan.shell_polynomial[second_shell] * reduced_distance;
        const double pair_scale = h0_plan.shell_pair_scale[pair_begin + local_pair];
        spatial_scale = pair_scale * first_polynomial * second_polynomial;
        const double polynomial_derivative =
            (h0_plan.shell_polynomial[first_shell] * second_polynomial +
             h0_plan.shell_polynomial[second_shell] * first_polynomial) *
            reduced_distance / (2.0 * distance);
        spatial_scale_derivative = pair_scale * polynomial_derivative;
        finite = isfinite(radius_sum) && radius_sum > 0.0 && isfinite(distance) &&
                 isfinite(reduced_distance) && isfinite(first_polynomial) &&
                 isfinite(second_polynomial) && isfinite(spatial_scale) &&
                 isfinite(spatial_scale_derivative);
      }
    }

    const double factor = average_level * spatial_scale;
    if (!finite || !isfinite(factor)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
      continue;
    }

    double block_weight = 0.0;
    const std::int64_t first_orbital_begin = batch.shell_orbital_offsets[first_shell];
    const std::int64_t first_orbital_end = batch.shell_orbital_offsets[first_shell + 1];
    const std::int64_t second_orbital_begin = batch.shell_orbital_offsets[second_shell];
    const std::int64_t second_orbital_end = batch.shell_orbital_offsets[second_shell + 1];
    for (std::int64_t first_orbital = first_orbital_begin;
         finite && first_orbital < first_orbital_end; ++first_orbital) {
      const std::int64_t row = first_orbital - orbital_begin;
      for (std::int64_t second_orbital = second_orbital_begin; second_orbital < second_orbital_end;
           ++second_orbital) {
        const std::int64_t column = second_orbital - orbital_begin;
        const std::int64_t matrix = matrix_begin + row * orbital_count + column;
        const double density = input.density[matrix];
        const double overlap = input.overlap[matrix];
        const double overlap_contribution = density * factor;
        const double overlap_updated =
            workspace.overlap_adjoint_scratch[matrix] + overlap_contribution;
        const double weight_contribution = density * overlap;
        const double weight_updated = block_weight + weight_contribution;
        if (!isfinite(overlap_contribution) || !isfinite(overlap_updated) ||
            !isfinite(weight_contribution) || !isfinite(weight_updated)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
          finite = false;
          break;
        }
        workspace.overlap_adjoint_scratch[matrix] = overlap_updated;
        block_weight = weight_updated;
      }
    }
    if (!finite) {
      continue;
    }

    const double level_weight = 0.5 * block_weight * spatial_scale;
    const double first_cn = -h0_plan.shell_coordination_scale[first_shell] * level_weight;
    const double second_cn = -h0_plan.shell_coordination_scale[second_shell] * level_weight;
    if (!isfinite(level_weight) ||
        !add_finite_atomic(workspace.coordination_adjoint_scratch + first_atom, first_cn) ||
        !add_finite_atomic(workspace.coordination_adjoint_scratch + second_atom, second_cn)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
      continue;
    }

    if (first_atom != second_atom) {
      const double radial_derivative = block_weight * average_level * spatial_scale_derivative;
      const double coordinate_scale = radial_derivative / distance;
      const double contribution[3] = {coordinate_scale * dx, coordinate_scale * dy,
                                      coordinate_scale * dz};
      for (int axis = 0; axis < 3; ++axis) {
        if (!add_finite_atomic(workspace.gradient_scratch + first_atom * 3 + axis,
                               contribution[axis]) ||
            !add_finite_atomic(workspace.gradient_scratch + second_atom * 3 + axis,
                               -contribution[axis])) {
          record_system_error(system_errors, system, device_error,
                              Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
        }
      }
    }
  }
}

/* Image-aware counterpart of contract_h0_pulay_kernel.  The periodic forward
 * evaluator sums each image into the public matrix, so the reverse pass must
 * reconstruct the same local overlap contribution before applying the H0
 * distance/CN chain.  One CTA owns one ragged system; the scalar inner loops
 * intentionally trade throughput for a deterministic, allocation-free bridge
 * while the periodic force kernels are being brought to parity. */
__global__ void contract_native_periodic_h0_pulay_kernel(
    Gfn2IntegralDeviceBatch batch, Gfn2H0DevicePlan h0_plan, Gfn2ForceDeviceActivity activity,
    Gfn2H0ForceDeviceInput input, Gfn2H0ForceDeviceWorkspace workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int selected;
  if (!sequence_is_active(workspace) ||
      !load_force_gate(activity, system, &selected, system_errors, device_error)) {
    return;
  }
  if (threadIdx.x != 0) return;

  const auto periodic = input.native_periodic;
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t orbital_begin = batch.batch_orbital_offsets[system];
  const std::int64_t orbital_count = batch.batch_orbital_offsets[system + 1] - orbital_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t shell_count = shell_end - shell_begin;
  if (shell_count <= 0 || orbital_count <= 0 || periodic.translation_offsets == nullptr ||
      periodic.translations == nullptr || periodic.translation_offset_elements <= system + 1) {
    record_system_error(system_errors, system, device_error,
                        Gfn2H0ForceDeviceError::kInvalidOffsets);
    return;
  }
  const std::int64_t translation_begin = periodic.translation_offsets[system];
  const std::int64_t translation_end = periodic.translation_offsets[system + 1];
  if (translation_begin < 0 || translation_end <= translation_begin ||
      translation_end > periodic.translation_elements) {
    record_system_error(system_errors, system, device_error,
                        Gfn2H0ForceDeviceError::kInvalidOffsets);
    return;
  }

  auto process_pair = [&](std::int64_t bra_atom, std::int64_t ket_atom, std::int64_t bra_shell,
                          std::int64_t ket_shell, const double vector[3], double distance_squared,
                          bool onsite) {
    double factor = 0.0;
    double spatial_scale = 1.0;
    double spatial_scale_derivative = 0.0;
    double distance = 0.0;
    if (!native_h0_factor(batch, h0_plan, input.coordination_numbers, system, bra_atom, ket_atom,
                          bra_shell, ket_shell, onsite, distance_squared, factor, spatial_scale,
                          spatial_scale_derivative, distance)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
      return;
    }
    const std::int64_t bra_orbital_begin = batch.shell_orbital_offsets[bra_shell];
    const std::int64_t bra_orbital_end = batch.shell_orbital_offsets[bra_shell + 1];
    const std::int64_t ket_orbital_begin = batch.shell_orbital_offsets[ket_shell];
    const std::int64_t ket_orbital_end = batch.shell_orbital_offsets[ket_shell + 1];
    double block_weight = 0.0;
    bool finite = true;
    for (std::int64_t bra_orbital = bra_orbital_begin; bra_orbital < bra_orbital_end && finite;
         ++bra_orbital) {
      const std::int64_t row = bra_orbital - orbital_begin;
      const int bra_ao = static_cast<int>(bra_orbital - bra_orbital_begin);
      for (std::int64_t ket_orbital = ket_orbital_begin; ket_orbital < ket_orbital_end;
           ++ket_orbital) {
        const std::int64_t column = ket_orbital - orbital_begin;
        const int ket_ao = static_cast<int>(ket_orbital - ket_orbital_begin);
        const std::int64_t forward = matrix_begin + row * orbital_count + column;
        double local_overlap = 0.0;
        if (!native_overlap_block(batch, bra_shell, ket_shell, vector, distance_squared, bra_ao,
                                  ket_ao, local_overlap)) {
          finite = false;
          break;
        }
        const double density = input.density[forward];
        const double weight_updated = block_weight + density * local_overlap;
        const double h0_adjoint_contribution = density * factor;
        const double overlap_adjoint_updated =
            workspace.overlap_adjoint_scratch[forward] + h0_adjoint_contribution;
        if (!isfinite(h0_adjoint_contribution) || !isfinite(weight_updated) ||
            !isfinite(overlap_adjoint_updated)) {
          finite = false;
          break;
        }
        /* Unlike the molecular matrix, a periodic H0 entry is an image sum.
         * Keep P*dH0/dS local to the image task: the downstream integral VJP
         * adds this contribution beside that image's dS.  Publishing the sum
         * here would multiply every image's dS by every other image's H0
         * factor.  The scalar is still checked above because it participates
         * in the same finite-input contract as the molecular leaf. */
        workspace.overlap_adjoint_scratch[forward] = overlap_adjoint_updated;
        block_weight = weight_updated;
        if (!onsite && bra_atom != ket_atom) {
          const std::int64_t reverse = matrix_begin + column * orbital_count + row;
          const double reverse_density = input.density[reverse];
          const double reverse_h0_adjoint_contribution = reverse_density * factor;
          const double reverse_weight = block_weight + reverse_density * local_overlap;
          const double reverse_overlap_adjoint_updated =
              workspace.overlap_adjoint_scratch[reverse] + reverse_h0_adjoint_contribution;
          if (!isfinite(reverse_h0_adjoint_contribution) || !isfinite(reverse_weight) ||
              !isfinite(reverse_overlap_adjoint_updated)) {
            finite = false;
            break;
          }
          workspace.overlap_adjoint_scratch[reverse] = reverse_overlap_adjoint_updated;
          block_weight = reverse_weight;
        }
      }
    }
    if (!finite) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
      return;
    }
    const double level_weight = 0.5 * block_weight * spatial_scale;
    if (!isfinite(level_weight) ||
        !add_finite_atomic(workspace.coordination_adjoint_scratch + bra_atom,
                           -h0_plan.shell_coordination_scale[bra_shell] * level_weight) ||
        !add_finite_atomic(workspace.coordination_adjoint_scratch + ket_atom,
                           -h0_plan.shell_coordination_scale[ket_shell] * level_weight)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
      return;
    }
    if (!onsite && bra_atom != ket_atom) {
      const double radial =
          block_weight * 0.5 *
          (h0_plan.shell_levels[bra_shell] -
           h0_plan.shell_coordination_scale[bra_shell] * input.coordination_numbers[bra_atom] +
           h0_plan.shell_levels[ket_shell] -
           h0_plan.shell_coordination_scale[ket_shell] * input.coordination_numbers[ket_atom]) *
          spatial_scale_derivative;
      const double scale = radial / distance;
      for (int axis = 0; axis < 3; ++axis) {
        const double contribution = scale * vector[axis];
        if (!add_finite_atomic(workspace.gradient_scratch + ket_atom * 3 + axis, contribution) ||
            !add_finite_atomic(workspace.gradient_scratch + bra_atom * 3 + axis, -contribution)) {
          record_system_error(system_errors, system, device_error,
                              Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
        }
      }
    }
  };

  /* Image traversal mirrors native_periodic_integral_kernel: atom pairs are
   * canonicalized with bra <= ket, while distinct-atom reverse matrix blocks
   * are handled inside process_pair. */
  for (std::int64_t image = translation_begin; image < translation_end; ++image) {
    const auto translation = periodic.translations[image];
    if (!isfinite(translation.cartesian[0]) || !isfinite(translation.cartesian[1]) ||
        !isfinite(translation.cartesian[2])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2H0ForceDeviceError::kNonfiniteArithmetic);
      return;
    }
    for (std::int64_t ket_atom = atom_begin; ket_atom < atom_end; ++ket_atom) {
      for (std::int64_t bra_atom = atom_begin; bra_atom <= ket_atom; ++bra_atom) {
        double vector[3];
        for (int axis = 0; axis < 3; ++axis) {
          const double difference = periodic_wrap_detail::rounded_subtract(
              input.positions[ket_atom * 3 + axis], input.positions[bra_atom * 3 + axis]);
          vector[axis] =
              periodic_wrap_detail::rounded_subtract(difference, translation.cartesian[axis]);
        }
        const double x_squared = periodic_wrap_detail::rounded_multiply(vector[0], vector[0]);
        const double y_squared = periodic_wrap_detail::rounded_multiply(vector[1], vector[1]);
        const double z_squared = periodic_wrap_detail::rounded_multiply(vector[2], vector[2]);
        const double distance_squared = periodic_wrap_detail::rounded_add(
            periodic_wrap_detail::rounded_add(x_squared, y_squared), z_squared);
        if (!isfinite(distance_squared) || distance_squared < kNativeMinimumImageDistanceSquared ||
            distance_squared > periodic.realspace_cutoff * periodic.realspace_cutoff ||
            fabs(vector[0]) > periodic.realspace_cutoff ||
            fabs(vector[1]) > periodic.realspace_cutoff ||
            fabs(vector[2]) > periodic.realspace_cutoff) {
          continue;
        }
        for (std::int64_t ket_shell = batch.atom_shell_offsets[ket_atom];
             ket_shell < batch.atom_shell_offsets[ket_atom + 1]; ++ket_shell) {
          for (std::int64_t bra_shell = batch.atom_shell_offsets[bra_atom];
               bra_shell < batch.atom_shell_offsets[bra_atom + 1]; ++bra_shell) {
            process_pair(bra_atom, ket_atom, bra_shell, ket_shell, vector, distance_squared, false);
          }
        }
      }
    }
  }
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    for (std::int64_t ket_shell = batch.atom_shell_offsets[atom];
         ket_shell < batch.atom_shell_offsets[atom + 1]; ++ket_shell) {
      for (std::int64_t bra_shell = batch.atom_shell_offsets[atom];
           bra_shell < batch.atom_shell_offsets[atom + 1]; ++bra_shell) {
        const double zero[3] = {0.0, 0.0, 0.0};
        process_pair(atom, atom, bra_shell, ket_shell, zero, 0.0, true);
      }
    }
  }
}

__global__ void publish_kernel(Gfn2IntegralDeviceBatch batch, Gfn2ForceDeviceActivity activity,
                               Gfn2H0ForceDeviceOutput output, Gfn2H0ForceDeviceWorkspace workspace,
                               const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const std::int64_t tile = static_cast<std::int64_t>(blockIdx.y);
  if (!sequence_is_active(workspace) || activity.requested_mask[system] != 1u ||
      activity.system_statuses[system] != XTBLOOM_STATUS_SUCCESS ||
      !system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
  const std::int64_t stride = batch.linear_tiles_per_system * blockDim.x;
  for (std::int64_t matrix = matrix_begin + tile * blockDim.x + threadIdx.x; matrix < matrix_end;
       matrix += stride) {
    output.overlap_adjoint[matrix] = workspace.overlap_adjoint_scratch[matrix];
  }
  for (std::int64_t atom = atom_begin + tile * blockDim.x + threadIdx.x; atom < atom_end;
       atom += stride) {
    output.coordination_adjoint[atom] = workspace.coordination_adjoint_scratch[atom];
    const std::int64_t coordinate = atom * 3;
    output.gradients[coordinate] = workspace.gradient_scratch[coordinate];
    output.gradients[coordinate + 1] = workspace.gradient_scratch[coordinate + 1];
    output.gradients[coordinate + 2] = workspace.gradient_scratch[coordinate + 2];
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
  for (std::size_t write = 0u; write < WriteCount; ++write) {
    for (const MemoryRange& read : reads) {
      if (overlaps(writes[write], read)) {
        return false;
      }
    }
    for (std::size_t other = write + 1u; other < WriteCount; ++other) {
      if (overlaps(writes[write], writes[other])) {
        return false;
      }
    }
  }
  return true;
}

cudaError_t validate_descriptors(
    const Gfn2IntegralDeviceBatch& batch, const Gfn2H0DevicePlan& h0_plan,
    const Gfn2ForceDeviceActivity& activity, const Gfn2H0ForceDeviceInput& input,
    const Gfn2H0ForceDeviceOutput& output, const Gfn2H0ForceDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error) noexcept {
  const auto& periodic = input.native_periodic;
  const bool native_enabled = periodic.plan_token != 0u;
  const bool native_fields_empty =
      periodic.translation_offset_elements == 0 && periodic.translation_elements == 0 &&
      periodic.max_translations_per_system == 0 && periodic.realspace_cutoff == 0.0 &&
      periodic.translation_offsets == nullptr && periodic.translations == nullptr;
  if (batch.batch_size <= 0 || batch.batch_size > std::numeric_limits<int>::max() ||
      batch.total_atoms < 0 || batch.total_shells < 0 || batch.total_orbitals < 0 ||
      batch.total_matrix_elements < 0 || batch.total_shell_pair_elements < 0 ||
      batch.total_atoms > kMaximumInt64 / 3 || batch.linear_tiles_per_system <= 0 ||
      batch.linear_tiles_per_system > kGfn2IntegralLinearBlockBudget || batch.plan_token == 0u ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.batch_shell_offset_count != batch.batch_size + 1 ||
      batch.batch_orbital_offset_count != batch.batch_size + 1 ||
      batch.matrix_offset_count != batch.batch_size + 1 ||
      batch.shell_pair_offset_count != batch.batch_size + 1 ||
      batch.atom_shell_offset_count < batch.total_atoms + 1 ||
      batch.shell_orbital_offset_count < batch.total_shells + 1 ||
      batch.shell_to_atom_count < batch.total_shells ||
      h0_plan.atomic_radius_count < batch.total_atoms ||
      h0_plan.shell_level_count < batch.total_shells ||
      h0_plan.shell_coordination_scale_count < batch.total_shells ||
      h0_plan.shell_polynomial_count < batch.total_shells ||
      h0_plan.shell_pair_scale_count < batch.total_shell_pair_elements ||
      h0_plan.plan_token != batch.plan_token || activity.plan_token != batch.plan_token ||
      input.plan_token != batch.plan_token || output.plan_token != batch.plan_token ||
      workspace.plan_token != batch.plan_token || activity.batch_elements != batch.batch_size ||
      input.position_elements < batch.total_atoms * 3 ||
      input.coordination_elements < batch.total_atoms ||
      input.overlap_elements < batch.total_matrix_elements ||
      input.density_elements < batch.total_matrix_elements ||
      input.energy_weighted_density_elements < batch.total_matrix_elements ||
      output.overlap_adjoint_elements < batch.total_matrix_elements ||
      output.coordination_adjoint_elements < batch.total_atoms ||
      output.gradient_elements < batch.total_atoms * 3 ||
      workspace.overlap_adjoint_elements < batch.total_matrix_elements ||
      workspace.coordination_adjoint_elements < batch.total_atoms ||
      workspace.gradient_elements < batch.total_atoms * 3 || workspace.sequence_elements < 1 ||
      !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.batch_orbital_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.matrix_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_pair_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.atom_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.shell_orbital_offsets, alignof(std::int64_t)) ||
      !required_pointer(batch.shell_to_atom, batch.total_shells) ||
      !required_pointer(h0_plan.atomic_radii, batch.total_atoms) ||
      !required_pointer(h0_plan.shell_levels, batch.total_shells) ||
      !required_pointer(h0_plan.shell_coordination_scale, batch.total_shells) ||
      !required_pointer(h0_plan.shell_polynomial, batch.total_shells) ||
      !required_pointer(h0_plan.shell_pair_scale, batch.total_shell_pair_elements) ||
      !is_aligned(activity.requested_mask, alignof(std::uint8_t)) ||
      !is_aligned(activity.system_statuses, alignof(xtbloom_status_t)) ||
      !required_pointer(input.positions, batch.total_atoms * 3) ||
      !required_pointer(input.coordination_numbers, batch.total_atoms) ||
      !required_pointer(input.overlap, batch.total_matrix_elements) ||
      !required_pointer(input.density, batch.total_matrix_elements) ||
      !required_pointer(input.energy_weighted_density, batch.total_matrix_elements) ||
      !required_pointer(output.overlap_adjoint, batch.total_matrix_elements) ||
      !required_pointer(output.coordination_adjoint, batch.total_atoms) ||
      !required_pointer(output.gradients, batch.total_atoms * 3) ||
      !required_pointer(workspace.overlap_adjoint_scratch, batch.total_matrix_elements) ||
      !required_pointer(workspace.coordination_adjoint_scratch, batch.total_atoms) ||
      !required_pointer(workspace.gradient_scratch, batch.total_atoms * 3) ||
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
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return batch.batch_size > std::numeric_limits<int>::max() ? cudaErrorInvalidConfiguration
                                                              : cudaErrorInvalidValue;
  }

  std::array<MemoryRange, 22> reads;
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
      !make_range(batch.shell_to_atom, batch.total_shells, sizeof(*batch.shell_to_atom),
                  &reads[7]) ||
      !make_range(h0_plan.atomic_radii, batch.total_atoms, sizeof(*h0_plan.atomic_radii),
                  &reads[8]) ||
      !make_range(h0_plan.shell_levels, batch.total_shells, sizeof(*h0_plan.shell_levels),
                  &reads[9]) ||
      !make_range(h0_plan.shell_coordination_scale, batch.total_shells,
                  sizeof(*h0_plan.shell_coordination_scale), &reads[10]) ||
      !make_range(h0_plan.shell_polynomial, batch.total_shells, sizeof(*h0_plan.shell_polynomial),
                  &reads[11]) ||
      !make_range(h0_plan.shell_pair_scale, batch.total_shell_pair_elements,
                  sizeof(*h0_plan.shell_pair_scale), &reads[12]) ||
      !make_range(activity.requested_mask, batch.batch_size, sizeof(*activity.requested_mask),
                  &reads[13]) ||
      !make_range(activity.system_statuses, batch.batch_size, sizeof(*activity.system_statuses),
                  &reads[14]) ||
      !make_range(input.positions, batch.total_atoms * 3, sizeof(*input.positions), &reads[15]) ||
      !make_range(input.coordination_numbers, batch.total_atoms,
                  sizeof(*input.coordination_numbers), &reads[16]) ||
      !make_range(input.overlap, batch.total_matrix_elements, sizeof(*input.overlap), &reads[17]) ||
      !make_range(input.density, batch.total_matrix_elements, sizeof(*input.density), &reads[18]) ||
      !make_range(input.energy_weighted_density, batch.total_matrix_elements,
                  sizeof(*input.energy_weighted_density), &reads[19]) ||
      !make_range(periodic.translation_offsets,
                  native_enabled ? periodic.translation_offset_elements : 0,
                  sizeof(*periodic.translation_offsets), &reads[20]) ||
      !make_range(periodic.translations, native_enabled ? periodic.translation_elements : 0,
                  sizeof(*periodic.translations), &reads[21]) ||
      !make_range(output.overlap_adjoint, batch.total_matrix_elements,
                  sizeof(*output.overlap_adjoint), &writes[0]) ||
      !make_range(output.coordination_adjoint, batch.total_atoms,
                  sizeof(*output.coordination_adjoint), &writes[1]) ||
      !make_range(output.gradients, batch.total_atoms * 3, sizeof(*output.gradients), &writes[2]) ||
      !make_range(workspace.overlap_adjoint_scratch, batch.total_matrix_elements,
                  sizeof(*workspace.overlap_adjoint_scratch), &writes[3]) ||
      !make_range(workspace.coordination_adjoint_scratch, batch.total_atoms,
                  sizeof(*workspace.coordination_adjoint_scratch), &writes[4]) ||
      !make_range(workspace.gradient_scratch, batch.total_atoms * 3,
                  sizeof(*workspace.gradient_scratch), &writes[5]) ||
      !make_range(workspace.sequence_active, 1, sizeof(*workspace.sequence_active), &writes[6]) ||
      !make_range(system_errors, batch.batch_size, sizeof(*system_errors), &writes[7]) ||
      !make_range(device_error, 1, sizeof(*device_error), &writes[8]) ||
      !writes_are_disjoint(reads, writes)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t check_launch() noexcept { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t reset_gfn2_h0_force_device_errors_cuda(std::int64_t batch_size,
                                                   std::uint32_t* system_errors,
                                                   std::uint32_t* device_error,
                                                   cudaStream_t stream) noexcept {
  if (batch_size <= 0 ||
      static_cast<std::uint64_t>(batch_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                     sizeof(*system_errors)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  MemoryRange systems;
  MemoryRange device;
  if (!make_range(system_errors, batch_size, sizeof(*system_errors), &systems) ||
      !make_range(device_error, 1, sizeof(*device_error), &device) || overlaps(systems, device)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  return status == cudaSuccess ? cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream)
                               : status;
}

cudaError_t add_gfn2_h0_pulay_gradient_cuda(
    const Gfn2IntegralDeviceBatch& batch, const Gfn2H0DevicePlan& h0_plan,
    const Gfn2ForceDeviceActivity& activity, const Gfn2H0ForceDeviceInput& input,
    const Gfn2H0ForceDeviceOutput& output, const Gfn2H0ForceDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  cudaError_t status = validate_descriptors(batch, h0_plan, activity, input, output, workspace,
                                            system_errors, device_error);
  if (status != cudaSuccess) {
    return status;
  }
  capture_sequence_kernel<<<1, 1, 0, stream>>>(device_error, workspace);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks = static_cast<unsigned int>(batch.batch_size);
  Gfn2IntegralLinearLaunchShape linear_shape{};
  if (!make_gfn2_integral_linear_launch_shape(batch.batch_size, batch.linear_tiles_per_system,
                                              linear_shape)) {
    return cudaErrorInvalidValue;
  }
  const dim3 linear_grid(linear_shape.systems, linear_shape.tiles, 1u);
  preflight_and_seed_kernel<<<linear_grid, kThreadsPerBlock, 0, stream>>>(
      batch, h0_plan, activity, input, output, workspace, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  if (input.native_periodic.plan_token != 0u) {
    contract_native_periodic_h0_pulay_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        batch, h0_plan, activity, input, workspace, system_errors, device_error);
  } else {
    contract_h0_pulay_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        batch, h0_plan, activity, input, workspace, system_errors, device_error);
  }
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_kernel<<<linear_grid, kThreadsPerBlock, 0, stream>>>(batch, activity, output, workspace,
                                                               system_errors);
  return check_launch();
}

}  // namespace xtbloom::detail::cuda
