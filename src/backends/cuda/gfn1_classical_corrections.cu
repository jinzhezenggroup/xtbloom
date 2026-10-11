#include "backends/cuda/gfn1_classical_corrections.cuh"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace xtbloom::detail::cuda {
namespace {

constexpr double kD3WeightFactor = 4.0;
constexpr double kD3Cutoff = 50.0;
constexpr double kD3SwitchWidth = 0.05;
constexpr double kD3S6 = 1.0;
constexpr double kD3S8 = 2.4;
constexpr double kCoordinationSteepness = 16.0;
constexpr double kCoordinationCutoff = 25.0;
constexpr double kCoordinationMinimumSquared = 1.0e-12;
constexpr double kD3MinimumSquared = 2.2204460492503131e-16;
constexpr double kHalogenCutoff = 20.0;
constexpr double kHalogenDamping = 0.44;
constexpr double kMaximumFinite = 0x1.fffffffffffffp+1023;
constexpr std::int64_t kInt64Maximum = 9223372036854775807LL;

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

template <std::size_t N, typename T>
bool append_range(std::array<AddressRange, N>& ranges, std::size_t* count, const T* pointer,
                  std::int64_t elements) noexcept {
  if (count == nullptr || *count >= N || !make_range(pointer, elements, &ranges[*count])) {
    return false;
  }
  ++*count;
  return true;
}

template <std::size_t N>
bool ranges_are_disjoint(const std::array<AddressRange, N>& ranges, std::size_t count) noexcept {
  if (count > N) return false;
  for (std::size_t first = 0; first < count; ++first) {
    for (std::size_t second = first + 1; second < count; ++second) {
      if (overlaps(ranges[first], ranges[second])) return false;
    }
  }
  return true;
}

__device__ void record_system_error(std::uint32_t* errors, std::int64_t system,
                                    Gfn1ClassicalCorrectionDeviceError error) {
  atomicCAS(errors + system,
            static_cast<std::uint32_t>(Gfn1ClassicalCorrectionDeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ double smooth_cutoff(double distance, double* derivative) {
  const double inner = kD3Cutoff - kD3SwitchWidth;
  if (distance <= inner) {
    *derivative = 0.0;
    return 1.0;
  }
  if (distance >= kD3Cutoff) {
    *derivative = 0.0;
    return 0.0;
  }
  const double x = (kD3Cutoff - distance) / kD3SwitchWidth;
  *derivative = -30.0 * x * x * (1.0 - x) * (1.0 - x) / kD3SwitchWidth;
  return x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
}

__device__ bool finite3(const double* values) {
  return isfinite(values[0]) && isfinite(values[1]) && isfinite(values[2]);
}

__device__ double distance_between(const double* positions, std::int64_t first, std::int64_t second,
                                   double delta[3]) {
  double r2 = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    delta[axis] = positions[3 * second + axis] - positions[3 * first + axis];
    r2 += delta[axis] * delta[axis];
  }
  return sqrt(r2);
}

__device__ bool prepare_d3_weights(const Gfn1ClassicalCorrectionDevicePlan& plan,
                                   std::int64_t begin, std::int64_t end, const double* coordination,
                                   const Gfn1ClassicalCorrectionDeviceWorkspace& workspace) {
  for (std::int64_t atom = begin; atom < end; ++atom) {
    const std::uint8_t count = plan.reference_counts[atom];
    if (count == 0u || count > kGfn1D3MaximumReferences || !isfinite(coordination[atom])) {
      return false;
    }
    const std::int64_t offset = atom * kGfn1D3MaximumReferences;
    double norm = 0.0;
    double derivative_norm = 0.0;
    double maximum_cn = -kMaximumFinite;
    for (std::int64_t ref = 0; ref < kGfn1D3MaximumReferences; ++ref) {
      workspace.weights[offset + ref] = 0.0;
      workspace.weight_cn_derivatives[offset + ref] = 0.0;
      if (ref >= count) continue;
      const double ref_cn = plan.reference_cn[offset + ref];
      if (!isfinite(ref_cn)) return false;
      maximum_cn = fmax(maximum_cn, ref_cn);
      const double delta = ref_cn - coordination[atom];
      const double u = exp(-kD3WeightFactor * delta * delta);
      workspace.weights[offset + ref] = u;
      norm += u;
      derivative_norm += 2.0 * kD3WeightFactor * delta * u;
    }
    const double inverse_norm = 1.0 / norm;
    for (std::int64_t ref = 0; ref < count; ++ref) {
      const double ref_cn = plan.reference_cn[offset + ref];
      const double delta = ref_cn - coordination[atom];
      const double u = workspace.weights[offset + ref];
      double weight = u * inverse_norm;
      if (!isfinite(weight)) weight = ref_cn == maximum_cn ? 1.0 : 0.0;
      double derivative = 2.0 * kD3WeightFactor * delta * u * inverse_norm -
                          u * derivative_norm * inverse_norm * inverse_norm;
      if (!isfinite(derivative)) derivative = 0.0;
      workspace.weights[offset + ref] = weight;
      workspace.weight_cn_derivatives[offset + ref] = derivative;
    }
  }
  return true;
}

__device__ bool pair_coefficient(const Gfn1ClassicalCorrectionDevicePlan& plan,
                                 std::int64_t packed_pair, std::int64_t first, std::int64_t second,
                                 const Gfn1ClassicalCorrectionDeviceWorkspace& workspace,
                                 double* c6, double* first_cn, double* second_cn) {
  *c6 = 0.0;
  *first_cn = 0.0;
  *second_cn = 0.0;
  const std::int64_t first_offset = first * kGfn1D3MaximumReferences;
  const std::int64_t second_offset = second * kGfn1D3MaximumReferences;
  const std::int64_t c6_offset = packed_pair * kGfn1D3ReferencePairStride;
  const std::int64_t first_count = plan.reference_counts[first];
  const std::int64_t second_count = plan.reference_counts[second];
  for (std::int64_t i = 0; i < first_count; ++i) {
    for (std::int64_t j = 0; j < second_count; ++j) {
      const double ref = plan.reference_c6[c6_offset + i * kGfn1D3MaximumReferences + j];
      const double wi = workspace.weights[first_offset + i];
      const double wj = workspace.weights[second_offset + j];
      const double di = workspace.weight_cn_derivatives[first_offset + i];
      const double dj = workspace.weight_cn_derivatives[second_offset + j];
      if (!isfinite(ref) || !isfinite(wi) || !isfinite(wj) || !isfinite(di) || !isfinite(dj)) {
        return false;
      }
      *c6 += wi * wj * ref;
      *first_cn += di * wj * ref;
      *second_cn += wi * dj * ref;
      if (!isfinite(*c6) || !isfinite(*first_cn) || !isfinite(*second_cn)) return false;
    }
  }
  return true;
}

__device__ bool add_d3(const Gfn1ClassicalCorrectionDevicePlan& plan, std::int64_t system,
                       const double* positions, const double* coordination, double* energy,
                       double* gradient, const Gfn1ClassicalCorrectionDeviceWorkspace& workspace) {
  const std::int64_t begin = plan.atom_offsets[system];
  const std::int64_t end = plan.atom_offsets[system + 1];
  if (!prepare_d3_weights(plan, begin, end, coordination, workspace)) return false;
  for (std::int64_t atom = begin; atom < end; ++atom) workspace.coordination_adjoints[atom] = 0.0;

  for (std::int64_t second = begin + 1; second < end; ++second) {
    const std::int64_t local_second = second - begin;
    for (std::int64_t first = begin; first < second; ++first) {
      const std::int64_t local_first = first - begin;
      const std::int64_t packed =
          plan.pair_offsets[system] + local_second * (local_second - 1) / 2 + local_first;
      double dx = positions[3 * first] - positions[3 * second];
      double dy = positions[3 * first + 1] - positions[3 * second + 1];
      double dz = positions[3 * first + 2] - positions[3 * second + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!isfinite(r2)) return false;
      if (r2 < kD3MinimumSquared || r2 > kD3Cutoff * kD3Cutoff) continue;
      double c6 = 0.0, dfirst = 0.0, dsecond = 0.0;
      if (!pair_coefficient(plan, packed, first, second, workspace, &c6, &dfirst, &dsecond)) {
        return false;
      }
      const double r = sqrt(r2);
      const double rd = plan.pair_damping_radii[packed];
      const double rr = plan.pair_rrij[packed];
      if (!(rd > 0.0) || !(rr > 0.0) || !isfinite(rd) || !isfinite(rr)) return false;
      const double rd2 = rd * rd;
      const double rd4 = rd2 * rd2;
      const double rd6 = rd4 * rd2;
      const double rd8 = rd4 * rd4;
      const double r4 = r2 * r2;
      const double t6 = 1.0 / (r4 * r2 + rd6);
      const double t8 = 1.0 / (r4 * r4 + rd8);
      const double phi = kD3S6 * t6 + kD3S8 * rr * t8;
      const double derivative_over_distance =
          kD3S6 * (-6.0 * r4 * t6 * t6) + kD3S8 * rr * (-8.0 * r4 * r2 * t8 * t8);
      double cutoff_derivative = 0.0;
      const double cutoff = smooth_cutoff(r, &cutoff_derivative);
      const double damping = cutoff * phi;
      const double e = -c6 * damping;
      if (!isfinite(e)) return false;
      if (energy != nullptr) *energy += e;
      workspace.coordination_adjoints[first] += -dfirst * damping;
      workspace.coordination_adjoints[second] += -dsecond * damping;
      if (gradient != nullptr) {
        const double scale =
            -c6 * (cutoff * derivative_over_distance + cutoff_derivative * phi / r);
        gradient[3 * first] += scale * dx;
        gradient[3 * first + 1] += scale * dy;
        gradient[3 * first + 2] += scale * dz;
        gradient[3 * second] -= scale * dx;
        gradient[3 * second + 1] -= scale * dy;
        gradient[3 * second + 2] -= scale * dz;
      }
    }
  }

  if (gradient != nullptr) {
    for (std::int64_t second = begin + 1; second < end; ++second) {
      for (std::int64_t first = begin; first < second; ++first) {
        const double dx = positions[3 * first] - positions[3 * second];
        const double dy = positions[3 * first + 1] - positions[3 * second + 1];
        const double dz = positions[3 * first + 2] - positions[3 * second + 2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < kCoordinationMinimumSquared || r2 > kCoordinationCutoff * kCoordinationCutoff)
          continue;
        const double r = sqrt(r2);
        const double radius = plan.covalent_radii[first] + plan.covalent_radii[second];
        if (!(radius > 0.0) || !isfinite(radius)) return false;
        const double argument = kCoordinationSteepness * (radius / r - 1.0);
        double logistic_derivative = 0.0;
        if (argument >= 0.0) {
          const double ex = exp(-argument);
          const double den = 1.0 + ex;
          logistic_derivative = ex / (den * den);
        } else {
          const double ex = exp(argument);
          const double den = 1.0 + ex;
          logistic_derivative = ex / (den * den);
        }
        const double derivative = -kCoordinationSteepness * radius / (r * r) * logistic_derivative;
        const double adj =
            workspace.coordination_adjoints[first] + workspace.coordination_adjoints[second];
        const double scale = adj * derivative / r;
        gradient[3 * first] += scale * dx;
        gradient[3 * first + 1] += scale * dy;
        gradient[3 * first + 2] += scale * dz;
        gradient[3 * second] -= scale * dx;
        gradient[3 * second + 1] -= scale * dy;
        gradient[3 * second + 2] -= scale * dz;
      }
    }
  }
  return energy == nullptr || isfinite(*energy);
}

__device__ bool add_halogen(const Gfn1ClassicalCorrectionDevicePlan& plan, std::int64_t system,
                            const double* positions, double* energy, double* gradient,
                            const Gfn1ClassicalCorrectionDeviceWorkspace& workspace) {
  const std::int64_t begin = plan.atom_offsets[system];
  const std::int64_t end = plan.atom_offsets[system + 1];
  for (std::int64_t donor = begin; donor < end; ++donor) {
    workspace.axis_neighbors[donor] = -1;
    if (plan.halogen_donor[donor] == 0u) continue;
    bool active = false;
    for (std::int64_t acceptor = begin; acceptor < end; ++acceptor) {
      if (plan.halogen_acceptor[acceptor] == 0u) continue;
      double delta[3];
      const double distance = distance_between(positions, donor, acceptor, delta);
      if (!isfinite(distance)) return false;
      if (distance <= kHalogenCutoff) {
        if (!(distance > 0.0)) return false;
        active = true;
      }
    }
    /* CPU GFN1 requires an axis only when an acceptor is actually active.
     * Isolated donors and donor-only systems contribute exact zero. */
    if (!active) continue;
    double nearest = kMaximumFinite;
    for (std::int64_t other = begin; other < end; ++other) {
      if (other == donor) continue;
      double delta[3];
      const double r = distance_between(positions, donor, other, delta);
      if (!isfinite(r)) return false;
      if (r > 0.0 && r < nearest) {
        nearest = r;
        workspace.axis_neighbors[donor] = other;
      }
    }
    if (workspace.axis_neighbors[donor] < 0) return false;
  }

  for (std::int64_t donor = begin; donor < end; ++donor) {
    if (plan.halogen_donor[donor] == 0u) continue;
    const std::int64_t neighbor = workspace.axis_neighbors[donor];
    if (neighbor < 0) continue;
    for (std::int64_t acceptor = begin; acceptor < end; ++acceptor) {
      if (plan.halogen_acceptor[acceptor] == 0u || acceptor == neighbor) continue;
      double da[3], dn[3];
      const double ra = distance_between(positions, donor, acceptor, da);
      if (!isfinite(ra) || ra > kHalogenCutoff) continue;
      const double rn = distance_between(positions, donor, neighbor, dn);
      if (!(ra > 0.0) || !(rn > 0.0) || !isfinite(rn)) return false;
      double ua[3], un[3];
      double cosine = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        ua[axis] = da[axis] / ra;
        un[axis] = dn[axis] / rn;
        cosine += ua[axis] * un[axis];
      }
      const double base = 0.5 * (1.0 - cosine);
      const double b2 = base * base;
      const double angular = b2 * b2 * b2;
      const double r0 = plan.halogen_scaled_radii[donor] + plan.halogen_scaled_radii[acceptor];
      if (!(r0 > 0.0) || !isfinite(r0)) return false;
      const double ratio = ra / r0;
      const double ratio2 = ratio * ratio;
      const double ratio6 = ratio2 * ratio2 * ratio2;
      const double ratio12 = ratio6 * ratio6;
      const double den = 1.0 + ratio12;
      const double radial = (1.0 - kHalogenDamping * ratio6) / den;
      const double strength = plan.halogen_bond_strength[donor];
      if (!isfinite(strength)) return false;
      const double e = strength * angular * radial;
      if (!isfinite(e)) return false;
      if (energy != nullptr) *energy += e;
      if (gradient != nullptr) {
        const double radial_dr = 6.0 * ratio6 / ra *
                                 (kHalogenDamping * ratio12 - 2.0 * ratio6 - kHalogenDamping) /
                                 (den * den);
        const double angular_dc = -3.0 * b2 * b2 * base;
        for (int axis = 0; axis < 3; ++axis) {
          const double dc_da = (un[axis] - cosine * ua[axis]) / ra;
          const double dc_dn = (ua[axis] - cosine * un[axis]) / rn;
          const double acceptor_gradient =
              strength * (radial * angular_dc * dc_da + angular * radial_dr * ua[axis]);
          const double neighbor_gradient = strength * radial * angular_dc * dc_dn;
          gradient[3 * acceptor + axis] += acceptor_gradient;
          gradient[3 * neighbor + axis] += neighbor_gradient;
          gradient[3 * donor + axis] -= acceptor_gradient + neighbor_gradient;
        }
      }
    }
  }
  return energy == nullptr || isfinite(*energy);
}

__global__ void corrections_kernel(Gfn1ClassicalCorrectionDevicePlan plan, const double* positions,
                                   const double* coordination, const std::uint8_t* active_mask,
                                   double* energies, double* gradients,
                                   Gfn1ClassicalCorrectionDeviceWorkspace workspace,
                                   std::uint32_t* system_errors, const std::uint32_t* plan_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || system >= plan.batch_size || system_errors[system] != 0u ||
      atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) != 0u)
    return;
  const std::uint8_t active = active_mask == nullptr ? 1u : active_mask[system];
  if (active == 0u) return;
  if (active != 1u) {
    record_system_error(system_errors, system,
                        Gfn1ClassicalCorrectionDeviceError::kInvalidActiveMask);
    return;
  }
  const std::int64_t begin = plan.atom_offsets[system];
  const std::int64_t end = plan.atom_offsets[system + 1];
  if (begin < 0 || begin > end || end > plan.total_atoms) {
    record_system_error(system_errors, system,
                        Gfn1ClassicalCorrectionDeviceError::kInvalidTopology);
    return;
  }
  for (std::int64_t atom = begin; atom < end; ++atom) {
    const std::uint8_t references = plan.reference_counts[atom];
    if (!finite3(positions + 3 * atom) || !isfinite(coordination[atom]) ||
        !(plan.covalent_radii[atom] > 0.0) || !isfinite(plan.covalent_radii[atom]) ||
        !(plan.halogen_scaled_radii[atom] > 0.0) || !isfinite(plan.halogen_scaled_radii[atom]) ||
        !isfinite(plan.halogen_bond_strength[atom]) || references == 0u ||
        references > kGfn1D3MaximumReferences || plan.halogen_donor[atom] > 1u ||
        plan.halogen_acceptor[atom] > 1u) {
      record_system_error(system_errors, system,
                          Gfn1ClassicalCorrectionDeviceError::kNonfiniteInput);
      return;
    }
    const std::int64_t reference_offset = atom * kGfn1D3MaximumReferences;
    for (std::int64_t reference = 0; reference < references; ++reference) {
      if (!isfinite(plan.reference_cn[reference_offset + reference])) {
        record_system_error(system_errors, system,
                            Gfn1ClassicalCorrectionDeviceError::kNonfiniteInput);
        return;
      }
    }
  }
  const std::int64_t pair_begin = plan.pair_offsets[system];
  const std::int64_t pair_end = plan.pair_offsets[system + 1];
  for (std::int64_t pair = pair_begin; pair < pair_end; ++pair) {
    if (!(plan.pair_rrij[pair] > 0.0) || !isfinite(plan.pair_rrij[pair]) ||
        !(plan.pair_damping_radii[pair] > 0.0) || !isfinite(plan.pair_damping_radii[pair])) {
      record_system_error(system_errors, system,
                          Gfn1ClassicalCorrectionDeviceError::kNonfiniteInput);
      return;
    }
    const std::int64_t reference_offset = pair * kGfn1D3ReferencePairStride;
    for (std::int64_t reference = 0; reference < kGfn1D3ReferencePairStride; ++reference) {
      if (!isfinite(plan.reference_c6[reference_offset + reference])) {
        record_system_error(system_errors, system,
                            Gfn1ClassicalCorrectionDeviceError::kNonfiniteInput);
        return;
      }
    }
  }

  double* energy_candidate = energies == nullptr ? nullptr : workspace.batch_scratch + system;
  if (energy_candidate != nullptr) {
    *energy_candidate = energies[system];
    if (!isfinite(*energy_candidate)) {
      record_system_error(system_errors, system,
                          Gfn1ClassicalCorrectionDeviceError::kNonfiniteInput);
      return;
    }
  }
  double* gradient_candidate = gradients == nullptr ? nullptr : workspace.gradient_scratch;
  if (gradient_candidate != nullptr) {
    for (std::int64_t coordinate = 3 * begin; coordinate < 3 * end; ++coordinate) {
      gradient_candidate[coordinate] = gradients[coordinate];
      if (!isfinite(gradient_candidate[coordinate])) {
        record_system_error(system_errors, system,
                            Gfn1ClassicalCorrectionDeviceError::kNonfiniteInput);
        return;
      }
    }
  }
  if (!add_d3(plan, system, positions, coordination, energy_candidate, gradient_candidate,
              workspace) ||
      !add_halogen(plan, system, positions, energy_candidate, gradient_candidate, workspace)) {
    record_system_error(system_errors, system,
                        Gfn1ClassicalCorrectionDeviceError::kNonfiniteArithmetic);
    return;
  }
  if (energy_candidate != nullptr && !isfinite(*energy_candidate)) {
    record_system_error(system_errors, system,
                        Gfn1ClassicalCorrectionDeviceError::kNonfiniteArithmetic);
    return;
  }
  if (gradient_candidate != nullptr) {
    for (std::int64_t coordinate = 3 * begin; coordinate < 3 * end; ++coordinate) {
      if (!isfinite(gradient_candidate[coordinate])) {
        record_system_error(system_errors, system,
                            Gfn1ClassicalCorrectionDeviceError::kNonfiniteArithmetic);
        return;
      }
    }
  }
  if (energy_candidate != nullptr) energies[system] = *energy_candidate;
  if (gradient_candidate != nullptr) {
    for (std::int64_t coordinate = 3 * begin; coordinate < 3 * end; ++coordinate) {
      gradients[coordinate] = gradient_candidate[coordinate];
    }
  }
}

__global__ void topology_preflight_kernel(Gfn1ClassicalCorrectionDevicePlan plan,
                                          std::uint32_t* plan_error) {
  if (blockIdx.x != 0) return;
  __shared__ int valid;
  if (threadIdx.x == 0) {
    valid = atomicAdd(plan_error, 0u) == 0u && plan.atom_offsets[0] == 0 &&
            plan.atom_offsets[plan.batch_size] == plan.total_atoms && plan.pair_offsets[0] == 0 &&
            plan.pair_offsets[plan.batch_size] == plan.total_pairs;
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < plan.batch_size; system += blockDim.x) {
    const std::int64_t atom_begin = plan.atom_offsets[system];
    const std::int64_t atom_end = plan.atom_offsets[system + 1];
    const std::int64_t pair_begin = plan.pair_offsets[system];
    const std::int64_t pair_end = plan.pair_offsets[system + 1];
    const std::int64_t atoms = atom_end - atom_begin;
    const bool product_safe = atoms >= 0 && atoms <= 1 + kInt64Maximum / 2 &&
                              (atoms < 2 || atoms <= kInt64Maximum / (atoms - 1));
    const std::int64_t expected_pairs = product_safe ? atoms * (atoms - 1) / 2 : -1;
    if (atom_begin < 0 || atom_begin >= atom_end || atom_end > plan.total_atoms || pair_begin < 0 ||
        pair_begin > pair_end || pair_end > plan.total_pairs || !product_safe ||
        pair_end - pair_begin != expected_pairs) {
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && valid == 0) {
    atomicCAS(plan_error, 0u,
              static_cast<std::uint32_t>(Gfn1ClassicalCorrectionDeviceError::kInvalidTopology));
  }
}

}  // namespace

bool validate_gfn1_classical_correction_binding(
    const Gfn1ClassicalCorrectionDevicePlan& plan, const double* positions,
    const double* coordination_numbers, const std::uint8_t* active_mask, double* energies,
    double* gradients, const Gfn1ClassicalCorrectionDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* plan_error) noexcept {
  std::int64_t weight_elements = 0;
  std::int64_t reference_c6_elements = 0;
  std::int64_t coordinate_elements = 0;
  if (plan.batch_size <= 0 || plan.total_atoms <= 0 || plan.total_pairs < 0 ||
      plan.batch_size > static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max()) ||
      plan.total_atoms > kInt64Maximum / 3 ||
      plan.total_atoms > kInt64Maximum / kGfn1D3MaximumReferences ||
      plan.total_pairs > kInt64Maximum / kGfn1D3ReferencePairStride) {
    return false;
  }
  weight_elements = plan.total_atoms * kGfn1D3MaximumReferences;
  reference_c6_elements = plan.total_pairs * kGfn1D3ReferencePairStride;
  coordinate_elements = plan.total_atoms * 3;
  if (plan.model != XtbModelFlavor::kGfn1 || plan.plan_token == 0u ||
      plan.atom_offset_elements != plan.batch_size + 1 ||
      plan.pair_offset_elements != plan.batch_size + 1 ||
      plan.covalent_radius_elements != plan.total_atoms ||
      plan.reference_count_elements != plan.total_atoms ||
      plan.reference_cn_elements != weight_elements ||
      plan.reference_c6_elements != reference_c6_elements ||
      plan.pair_rrij_elements != plan.total_pairs ||
      plan.pair_damping_radius_elements != plan.total_pairs ||
      plan.halogen_scaled_radius_elements != plan.total_atoms ||
      plan.halogen_bond_strength_elements != plan.total_atoms ||
      plan.halogen_donor_elements != plan.total_atoms ||
      plan.halogen_acceptor_elements != plan.total_atoms ||
      workspace.plan_token != plan.plan_token || workspace.weight_elements != weight_elements ||
      workspace.weight_cn_derivative_elements != weight_elements ||
      workspace.coordination_adjoint_elements != plan.total_atoms ||
      workspace.axis_neighbor_elements != plan.total_atoms ||
      workspace.batch_scratch_elements != plan.batch_size ||
      workspace.gradient_scratch_elements != coordinate_elements ||
      (energies == nullptr && gradients == nullptr)) {
    return false;
  }

  std::array<AddressRange, 10> writes{};
  std::size_t write_count = 0u;
  if (!append_range(writes, &write_count, energies, energies == nullptr ? 0 : plan.batch_size) ||
      !append_range(writes, &write_count, gradients,
                    gradients == nullptr ? 0 : coordinate_elements) ||
      !append_range(writes, &write_count, workspace.weights, weight_elements) ||
      !append_range(writes, &write_count, workspace.weight_cn_derivatives, weight_elements) ||
      !append_range(writes, &write_count, workspace.coordination_adjoints, plan.total_atoms) ||
      !append_range(writes, &write_count, workspace.axis_neighbors, plan.total_atoms) ||
      !append_range(writes, &write_count, workspace.batch_scratch, plan.batch_size) ||
      !append_range(writes, &write_count, workspace.gradient_scratch, coordinate_elements) ||
      !append_range(writes, &write_count, system_errors, plan.batch_size) ||
      !append_range(writes, &write_count, plan_error, 1) ||
      !ranges_are_disjoint(writes, write_count)) {
    return false;
  }

  std::array<AddressRange, 15> reads{};
  std::size_t read_count = 0u;
  if (!append_range(reads, &read_count, plan.atom_offsets, plan.atom_offset_elements) ||
      !append_range(reads, &read_count, plan.pair_offsets, plan.pair_offset_elements) ||
      !append_range(reads, &read_count, plan.covalent_radii, plan.covalent_radius_elements) ||
      !append_range(reads, &read_count, plan.reference_counts, plan.reference_count_elements) ||
      !append_range(reads, &read_count, plan.reference_cn, plan.reference_cn_elements) ||
      !append_range(reads, &read_count, plan.reference_c6, plan.reference_c6_elements) ||
      !append_range(reads, &read_count, plan.pair_rrij, plan.pair_rrij_elements) ||
      !append_range(reads, &read_count, plan.pair_damping_radii,
                    plan.pair_damping_radius_elements) ||
      !append_range(reads, &read_count, plan.halogen_scaled_radii,
                    plan.halogen_scaled_radius_elements) ||
      !append_range(reads, &read_count, plan.halogen_bond_strength,
                    plan.halogen_bond_strength_elements) ||
      !append_range(reads, &read_count, plan.halogen_donor, plan.halogen_donor_elements) ||
      !append_range(reads, &read_count, plan.halogen_acceptor, plan.halogen_acceptor_elements) ||
      !append_range(reads, &read_count, positions, coordinate_elements) ||
      !append_range(reads, &read_count, coordination_numbers, plan.total_atoms) ||
      !append_range(reads, &read_count, active_mask,
                    active_mask == nullptr ? 0 : plan.batch_size)) {
    return false;
  }
  for (std::size_t read = 0u; read < read_count; ++read) {
    for (std::size_t write = 0u; write < write_count; ++write) {
      if (overlaps(reads[read], writes[write])) return false;
    }
  }
  return true;
}

cudaError_t add_gfn1_classical_corrections_cuda(
    const Gfn1ClassicalCorrectionDevicePlan& plan, const double* positions,
    const double* coordination_numbers, const std::uint8_t* active_mask, double* energies,
    double* gradients, const Gfn1ClassicalCorrectionDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* plan_error, cudaStream_t stream) noexcept {
  if (!validate_gfn1_classical_correction_binding(plan, positions, coordination_numbers,
                                                  active_mask, energies, gradients, workspace,
                                                  system_errors, plan_error)) {
    return cudaErrorInvalidValue;
  }
  topology_preflight_kernel<<<1, 256, 0, stream>>>(plan, plan_error);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  corrections_kernel<<<static_cast<unsigned int>(plan.batch_size), 1, 0, stream>>>(
      plan, positions, coordination_numbers, active_mask, energies, gradients, workspace,
      system_errors, plan_error);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
