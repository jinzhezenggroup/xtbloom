#include "model/gfn2/aes2.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "data/parameters/gfn2.hpp"

namespace gpuxtb::detail::gfn2 {

struct AES2PlanData final {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_pairs = 0;
  std::int64_t pair_data_elements = 0;
  std::int64_t potential_scratch_elements = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> pair_offsets;
  std::vector<double> dipole_kernel;
  std::vector<double> quadrupole_kernel;
  std::vector<double> multipole_radius;
  std::vector<double> multipole_valence_cn;
};

namespace {

constexpr std::int64_t kPairStride = 5;
constexpr std::int64_t kPotentialStride = 10;
constexpr double kMinimumDistanceSquared = 1.0e-12;

static_assert(parameters::gfn2::kGlobal.multipole_dmp3 == 3.0,
              "GFN2 AES2 compact kernel specializes dmp3=3");
static_assert(parameters::gfn2::kGlobal.multipole_dmp5 == 4.0,
              "GFN2 AES2 compact kernel specializes dmp5=4");
static_assert(parameters::gfn2::kGlobal.multipole_kexp == 4.0,
              "GFN2 AES2 multipole-radius exponent changed");
static_assert(parameters::gfn2::kGlobal.multipole_shift == 1.2,
              "GFN2 AES2 multipole-radius shift changed");
static_assert(parameters::gfn2::kGlobal.multipole_rmax == 5.0,
              "GFN2 AES2 maximum multipole radius changed");
static_assert(std::is_trivially_copyable_v<AES2GeometryCache>);
static_assert(std::is_standard_layout_v<AES2GeometryCache>);
static_assert(std::is_trivially_copyable_v<AES2Workspace>);
static_assert(std::is_standard_layout_v<AES2Workspace>);

struct MemoryRange {
  const void* data = nullptr;
  std::size_t size_bytes = 0;
};

bool representable_as_size(std::int64_t value) {
  return value >= 0 &&
         static_cast<std::uint64_t>(value) <=
             static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) &&
         static_cast<std::uint64_t>(value) <=
             static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

bool checked_multiply(std::int64_t value, std::int64_t factor, std::int64_t& result) {
  if (value < 0 || factor < 0 ||
      (value != 0 && factor > std::numeric_limits<std::int64_t>::max() / value)) {
    return false;
  }
  result = value * factor;
  return true;
}

bool checked_pair_count(std::int64_t atom_count, std::int64_t& result) {
  if (atom_count < 0) {
    return false;
  }
  if (atom_count <= 1) {
    result = 0;
    return true;
  }
  if ((atom_count & 1) == 0) {
    return checked_multiply(atom_count / 2, atom_count - 1, result);
  }
  return checked_multiply(atom_count, (atom_count - 1) / 2, result);
}

bool checked_add(std::int64_t increment, std::int64_t& total) {
  if (increment < 0 || total > std::numeric_limits<std::int64_t>::max() - increment) {
    return false;
  }
  total += increment;
  return true;
}

bool count_bytes(std::int64_t count, std::size_t element_size, std::size_t& bytes) {
  if (!representable_as_size(count) ||
      static_cast<std::uint64_t>(count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                    std::size_t second_bytes) {
  if (first_bytes == 0u || second_bytes == 0u) {
    return false;
  }
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  const auto first_end = first_begin + first_bytes;
  const auto second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

bool ranges_are_disjoint(const MemoryRange* ranges, std::size_t count) {
  for (std::size_t first = 0; first < count; ++first) {
    for (std::size_t second = first + 1u; second < count; ++second) {
      if (ranges_overlap(ranges[first].data, ranges[first].size_bytes, ranges[second].data,
                         ranges[second].size_bytes)) {
        return false;
      }
    }
  }
  return true;
}

bool is_aligned(const void* pointer, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

std::array<MemoryRange, 8> plan_storage_ranges(const AES2Plan& plan) {
  return {{{&plan, sizeof(plan)},
           {plan.identity(), sizeof(AES2PlanData)},
           {plan.atom_offsets().data(), plan.atom_offsets().capacity() * sizeof(std::int64_t)},
           {plan.pair_offsets().data(), plan.pair_offsets().capacity() * sizeof(std::int64_t)},
           {plan.dipole_kernel().data(), plan.dipole_kernel().capacity() * sizeof(double)},
           {plan.quadrupole_kernel().data(), plan.quadrupole_kernel().capacity() * sizeof(double)},
           {plan.multipole_radius().data(), plan.multipole_radius().capacity() * sizeof(double)},
           {plan.multipole_valence_cn().data(),
            plan.multipole_valence_cn().capacity() * sizeof(double)}}};
}

bool overlaps_plan_storage(const AES2Plan& plan, const void* data, std::size_t bytes) {
  for (const MemoryRange& range : plan_storage_ranges(plan)) {
    if (ranges_overlap(data, bytes, range.data, range.size_bytes)) {
      return true;
    }
  }
  return false;
}

bool overlaps_control_storage(const AES2Plan& plan, const AES2GeometryCache& cache,
                              const AES2Workspace& workspace, const void* data, std::size_t bytes) {
  return overlaps_plan_storage(plan, data, bytes) ||
         ranges_overlap(data, bytes, &cache, sizeof(cache)) ||
         ranges_overlap(data, bytes, &workspace, sizeof(workspace));
}

gpuxtb_status_t validate_basis(const BasisPlan& basis, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      !representable_as_size(basis.batch_size) || !representable_as_size(basis.total_atoms) ||
      !representable_as_size(basis.total_shells) ||
      basis.batch_size == std::numeric_limits<std::int64_t>::max()) {
    error = "AES2 requires a positive, representable basis plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch_count = static_cast<std::size_t>(basis.batch_size);
  const std::size_t atom_count = static_cast<std::size_t>(basis.total_atoms);
  const std::size_t shell_count = static_cast<std::size_t>(basis.total_shells);
  if (basis.atom_offsets.size() != batch_count + 1u ||
      basis.batch_shell_offsets.size() != batch_count + 1u ||
      basis.atom_shell_offsets.size() != atom_count + 1u ||
      basis.shell_to_atom.size() != shell_count ||
      basis.principal_quantum_numbers.size() != shell_count ||
      basis.angular_momenta.size() != shell_count || basis.slater_exponents.size() != shell_count ||
      basis.atom_offsets.front() != 0 || basis.atom_offsets.back() != basis.total_atoms ||
      basis.batch_shell_offsets.front() != 0 ||
      basis.batch_shell_offsets.back() != basis.total_shells ||
      basis.atom_shell_offsets.front() != 0 ||
      basis.atom_shell_offsets.back() != basis.total_shells) {
    error = "AES2 basis plan is incomplete or internally inconsistent";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const std::int64_t atom_begin = basis.atom_offsets[batch];
    const std::int64_t atom_end = basis.atom_offsets[batch + 1u];
    const std::int64_t shell_begin = basis.batch_shell_offsets[batch];
    const std::int64_t shell_end = basis.batch_shell_offsets[batch + 1u];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > basis.total_atoms ||
        shell_begin < 0 || shell_begin > shell_end || shell_end > basis.total_shells ||
        shell_begin != basis.atom_shell_offsets[static_cast<std::size_t>(atom_begin)] ||
        shell_end != basis.atom_shell_offsets[static_cast<std::size_t>(atom_end)]) {
      error = "AES2 basis offsets are not valid ragged partitions";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    const std::int64_t shell_begin = basis.atom_shell_offsets[atom];
    const std::int64_t shell_end = basis.atom_shell_offsets[atom + 1u];
    if (shell_begin < 0 || shell_begin >= shell_end || shell_end > basis.total_shells) {
      error = "AES2 atom-to-shell offsets are invalid";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::size_t shell_index = static_cast<std::size_t>(shell);
      if (basis.shell_to_atom[shell_index] != static_cast<std::int64_t>(atom) ||
          basis.angular_momenta[shell_index] > 2u || !(basis.slater_exponents[shell_index] > 0.0) ||
          !std::isfinite(basis.slater_exponents[shell_index])) {
        error = "AES2 basis shell metadata is invalid";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_plan(const AES2Plan& plan, std::string& error) {
  if (!plan.sealed() || plan.batch_size() <= 0 || plan.total_atoms() <= 0 ||
      plan.total_pairs() < 0 || plan.pair_data_elements() < 0 ||
      plan.potential_scratch_elements() <= 0 || !representable_as_size(plan.batch_size()) ||
      !representable_as_size(plan.total_atoms()) || !representable_as_size(plan.total_pairs()) ||
      !representable_as_size(plan.pair_data_elements()) ||
      !representable_as_size(plan.potential_scratch_elements()) ||
      plan.batch_size() == std::numeric_limits<std::int64_t>::max()) {
    error = "AES2 plan is default-constructed, moved-from, or has invalid dimensions";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch_count = static_cast<std::size_t>(plan.batch_size());
  const std::size_t atom_count = static_cast<std::size_t>(plan.total_atoms());
  std::int64_t expected_pair_elements = 0;
  std::int64_t expected_potential_elements = 0;
  if (!checked_multiply(plan.total_pairs(), kPairStride, expected_pair_elements) ||
      !checked_multiply(plan.total_atoms(), kPotentialStride, expected_potential_elements) ||
      plan.pair_data_elements() != expected_pair_elements ||
      plan.potential_scratch_elements() != expected_potential_elements ||
      plan.atom_offsets().size() != batch_count + 1u ||
      plan.pair_offsets().size() != batch_count + 1u || plan.dipole_kernel().size() != atom_count ||
      plan.quadrupole_kernel().size() != atom_count ||
      plan.multipole_radius().size() != atom_count ||
      plan.multipole_valence_cn().size() != atom_count || plan.atom_offsets().front() != 0 ||
      plan.atom_offsets().back() != plan.total_atoms() || plan.pair_offsets().front() != 0 ||
      plan.pair_offsets().back() != plan.total_pairs()) {
    error = "AES2 plan storage is incomplete or internally inconsistent";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const std::int64_t atom_begin = plan.atom_offsets()[batch];
    const std::int64_t atom_end = plan.atom_offsets()[batch + 1u];
    const std::int64_t pair_begin = plan.pair_offsets()[batch];
    const std::int64_t pair_end = plan.pair_offsets()[batch + 1u];
    std::int64_t expected_pairs = 0;
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > plan.total_atoms() ||
        pair_begin < 0 || pair_begin > pair_end || pair_end > plan.total_pairs() ||
        !checked_pair_count(atom_end - atom_begin, expected_pairs) ||
        pair_end - pair_begin != expected_pairs) {
      error = "AES2 plan offsets are not valid ragged atom/pair partitions";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    if (!std::isfinite(plan.dipole_kernel()[atom]) ||
        !std::isfinite(plan.quadrupole_kernel()[atom]) || !(plan.multipole_radius()[atom] > 0.0) ||
        plan.multipole_radius()[atom] > parameters::gfn2::kGlobal.multipole_rmax ||
        !std::isfinite(plan.multipole_radius()[atom]) ||
        !(plan.multipole_valence_cn()[atom] > 0.0) ||
        !std::isfinite(plan.multipole_valence_cn()[atom])) {
      error = "AES2 plan contains an invalid element parameter";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_cache(const AES2Plan& plan, const AES2GeometryCache& cache,
                               std::string& error) {
  if (cache.pair_data_elements != plan.pair_data_elements() ||
      cache.plan_identity != plan.identity() ||
      (cache.pair_data_elements != 0 &&
       (cache.pair_data == nullptr || !is_aligned(cache.pair_data, alignof(double))))) {
    error = "AES2 geometry cache is missing or belongs to a different plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t pair = 0; pair < plan.total_pairs(); ++pair) {
    const std::size_t base = static_cast<std::size_t>(pair * kPairStride);
    if (!std::isfinite(cache.pair_data[base]) || !std::isfinite(cache.pair_data[base + 1u]) ||
        !std::isfinite(cache.pair_data[base + 2u]) || !(cache.pair_data[base + 3u] >= 0.0) ||
        !std::isfinite(cache.pair_data[base + 3u]) || !(cache.pair_data[base + 4u] >= 0.0) ||
        !std::isfinite(cache.pair_data[base + 4u])) {
      error = "AES2 geometry cache contains an invalid pair kernel";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_workspace_pointer(double* pointer, std::int64_t available,
                                           std::int64_t required, const char* message,
                                           std::string& error) {
  if (available < required ||
      (required != 0 && (pointer == nullptr || !is_aligned(pointer, alignof(double))))) {
    error = message;
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_finite_array(const double* values, std::int64_t count,
                                      const char* null_message, const char* finite_message,
                                      std::string& error, bool require_nonnegative = false) {
  if (values == nullptr || !is_aligned(values, alignof(double))) {
    error = null_message;
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index]) || (require_nonnegative && values[index] < 0.0)) {
      error = finite_message;
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

/* Stable logistic form of 1/(1+exp(-argument)). */
double logistic(double argument) {
  if (argument >= 0.0) {
    const double exponential = std::exp(-argument);
    return 1.0 / (1.0 + exponential);
  }
  const double exponential = std::exp(argument);
  return exponential / (1.0 + exponential);
}

double multipole_radius(const AES2Plan& plan, std::size_t atom, double coordination_number) {
  const double argument = parameters::gfn2::kGlobal.multipole_kexp *
                          (coordination_number - plan.multipole_valence_cn()[atom] -
                           parameters::gfn2::kGlobal.multipole_shift);
  return plan.multipole_radius()[atom] +
         (parameters::gfn2::kGlobal.multipole_rmax - plan.multipole_radius()[atom]) *
             logistic(argument);
}

double multipole_radius_cn_derivative(const AES2Plan& plan, std::size_t atom,
                                      double coordination_number) {
  const double argument = parameters::gfn2::kGlobal.multipole_kexp *
                          (coordination_number - plan.multipole_valence_cn()[atom] -
                           parameters::gfn2::kGlobal.multipole_shift);
  const double fraction = logistic(argument);
  return (parameters::gfn2::kGlobal.multipole_rmax - plan.multipole_radius()[atom]) *
         parameters::gfn2::kGlobal.multipole_kexp * fraction * (1.0 - fraction);
}

bool pair_kernels(double distance, double radius, double& kernel3, double& kernel5) {
  const double inverse = 1.0 / distance;
  const double inverse2 = inverse * inverse;
  const double inverse3 = inverse2 * inverse;
  const double inverse5 = inverse3 * inverse2;
  const double scaled = radius * inverse;
  const double scaled2 = scaled * scaled;
  const double scaled3 = scaled2 * scaled;
  const double scaled4 = scaled2 * scaled2;
  kernel3 = inverse3 / (1.0 + 6.0 * scaled3);
  kernel5 = inverse5 / (1.0 + 6.0 * scaled4);
  if (kernel3 >= 0.0 && std::isfinite(kernel3) && kernel5 >= 0.0 && std::isfinite(kernel5)) {
    return true;
  }

  /*
   * Algebraically equivalent denominators avoid inf/inf for extreme but
   * representable distances. Long double is only a range-recovery path; the
   * ordinary binary64 operation order remains the fast/reference path.
   */
  const long double wide_distance = static_cast<long double>(distance);
  const long double wide_radius = static_cast<long double>(radius);
  const long double distance2 = wide_distance * wide_distance;
  const long double distance3 = distance2 * wide_distance;
  const long double radius2 = wide_radius * wide_radius;
  const long double radius3 = radius2 * wide_radius;
  const long double radius4 = radius2 * radius2;
  const long double denominator3 = distance3 + 6.0L * radius3;
  const long double denominator5 = distance3 * distance2 + 6.0L * radius4 * wide_distance;
  const long double wide3 = 1.0L / denominator3;
  const long double wide5 = 1.0L / denominator5;
  if (!(wide3 >= 0.0L) || !std::isfinite(wide3) || !(wide5 >= 0.0L) || !std::isfinite(wide5) ||
      wide3 > static_cast<long double>(std::numeric_limits<double>::max()) ||
      wide5 > static_cast<long double>(std::numeric_limits<double>::max())) {
    return false;
  }
  kernel3 = static_cast<double>(wide3);
  kernel5 = static_cast<double>(wide5);
  return std::isfinite(kernel3) && std::isfinite(kernel5);
}

bool add_value(double contribution, double& target) {
  if (!std::isfinite(contribution)) {
    return false;
  }
  const double updated = target + contribution;
  if (!std::isfinite(updated)) {
    return false;
  }
  target = updated;
  return true;
}

std::array<double, 6> packed_pair_tensor(double dx, double dy, double dz, double kernel5) {
  return {{dx * dx * kernel5, 2.0 * dx * dy * kernel5, dy * dy * kernel5, 2.0 * dx * dz * kernel5,
           2.0 * dy * dz * kernel5, dz * dz * kernel5}};
}

double packed_dot(const std::array<double, 6>& tensor, const double* quadrupole) {
  double result = 0.0;
  for (std::size_t component = 0; component < tensor.size(); ++component) {
    result += tensor[component] * quadrupole[component];
  }
  return result;
}

bool initialize_onsite_potential(const AES2Plan& plan, const double* dipoles,
                                 const double* quadrupoles, double* scratch) {
  const std::size_t atom_count = static_cast<std::size_t>(plan.total_atoms());
  double* const charge_potential = scratch;
  double* const dipole_potential = scratch + atom_count;
  double* const quadrupole_potential = scratch + atom_count * 4u;
  constexpr std::array<double, 6> scale{{1.0, 2.0, 1.0, 2.0, 2.0, 1.0}};
  std::fill_n(charge_potential, atom_count, 0.0);
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    for (std::size_t component = 0; component < 3u; ++component) {
      const std::size_t index = atom * 3u + component;
      dipole_potential[index] = 2.0 * plan.dipole_kernel()[atom] * dipoles[index];
      if (!std::isfinite(dipole_potential[index])) {
        return false;
      }
    }
    for (std::size_t component = 0; component < 6u; ++component) {
      const std::size_t index = atom * 6u + component;
      quadrupole_potential[index] =
          2.0 * plan.quadrupole_kernel()[atom] * scale[component] * quadrupoles[index];
      if (!std::isfinite(quadrupole_potential[index])) {
        return false;
      }
    }
  }
  return true;
}

bool add_pair_potential(const double* pair_data, std::size_t first, std::size_t second,
                        const double* charges, const double* dipoles, const double* quadrupoles,
                        double* charge_potential, double* dipole_potential,
                        double* quadrupole_potential) {
  const double dx = pair_data[0];
  const double dy = pair_data[1];
  const double dz = pair_data[2];
  const double kernel3 = pair_data[3];
  const double kernel5 = pair_data[4];
  const std::array<double, 3> displacement{{dx, dy, dz}};
  const std::array<double, 3> sd{{dx * kernel3, dy * kernel3, dz * kernel3}};
  const std::array<double, 6> sq = packed_pair_tensor(dx, dy, dz, kernel5);
  const double distance2 = dx * dx + dy * dy + dz * dz;
  const double isotropic_dd = distance2 * kernel5;

  const double* const first_dipole = dipoles + first * 3u;
  const double* const second_dipole = dipoles + second * 3u;
  const double* const first_quadrupole = quadrupoles + first * 6u;
  const double* const second_quadrupole = quadrupoles + second * 6u;
  double first_projection = 0.0;
  double second_projection = 0.0;
  double first_sd_dot = 0.0;
  double second_sd_dot = 0.0;
  for (std::size_t component = 0; component < 3u; ++component) {
    first_projection += displacement[component] * first_dipole[component];
    second_projection += displacement[component] * second_dipole[component];
    first_sd_dot += sd[component] * first_dipole[component];
    second_sd_dot += sd[component] * second_dipole[component];
  }

  if (!std::isfinite(distance2) || !std::isfinite(isotropic_dd) ||
      !std::isfinite(first_projection) || !std::isfinite(second_projection) ||
      !std::isfinite(first_sd_dot) || !std::isfinite(second_sd_dot) ||
      !add_value(second_sd_dot + packed_dot(sq, second_quadrupole), charge_potential[first]) ||
      !add_value(-first_sd_dot + packed_dot(sq, first_quadrupole), charge_potential[second])) {
    return false;
  }

  for (std::size_t component = 0; component < 3u; ++component) {
    const double first_dd = isotropic_dd * second_dipole[component] -
                            3.0 * kernel5 * displacement[component] * second_projection;
    const double second_dd = isotropic_dd * first_dipole[component] -
                             3.0 * kernel5 * displacement[component] * first_projection;
    if (!add_value(-charges[second] * sd[component] + first_dd,
                   dipole_potential[first * 3u + component]) ||
        !add_value(charges[first] * sd[component] + second_dd,
                   dipole_potential[second * 3u + component])) {
      return false;
    }
  }

  for (std::size_t component = 0; component < 6u; ++component) {
    if (!std::isfinite(sq[component]) ||
        !add_value(charges[second] * sq[component], quadrupole_potential[first * 6u + component]) ||
        !add_value(charges[first] * sq[component], quadrupole_potential[second * 6u + component])) {
      return false;
    }
  }
  return true;
}

bool onsite_energy(const AES2Plan& plan, std::size_t atom, const double* dipole,
                   const double* quadrupole, double& energy) {
  constexpr std::array<double, 6> scale{{1.0, 2.0, 1.0, 2.0, 2.0, 1.0}};
  double dipole_norm2 = 0.0;
  double quadrupole_norm2 = 0.0;
  for (std::size_t component = 0; component < 3u; ++component) {
    dipole_norm2 += dipole[component] * dipole[component];
  }
  for (std::size_t component = 0; component < 6u; ++component) {
    quadrupole_norm2 += scale[component] * quadrupole[component] * quadrupole[component];
  }
  energy =
      plan.dipole_kernel()[atom] * dipole_norm2 + plan.quadrupole_kernel()[atom] * quadrupole_norm2;
  return std::isfinite(dipole_norm2) && std::isfinite(quadrupole_norm2) && std::isfinite(energy);
}

bool pair_energy(const double* pair_data, std::size_t first, std::size_t second,
                 const double* charges, const double* dipoles, const double* quadrupoles,
                 double& energy) {
  const double dx = pair_data[0];
  const double dy = pair_data[1];
  const double dz = pair_data[2];
  const double kernel3 = pair_data[3];
  const double kernel5 = pair_data[4];
  const std::array<double, 3> displacement{{dx, dy, dz}};
  const std::array<double, 6> sq = packed_pair_tensor(dx, dy, dz, kernel5);
  const double* const first_dipole = dipoles + first * 3u;
  const double* const second_dipole = dipoles + second * 3u;
  const double* const first_quadrupole = quadrupoles + first * 6u;
  const double* const second_quadrupole = quadrupoles + second * 6u;

  double first_projection = 0.0;
  double second_projection = 0.0;
  double dipole_dot = 0.0;
  double charge_dipole_numerator = 0.0;
  for (std::size_t component = 0; component < 3u; ++component) {
    first_projection += displacement[component] * first_dipole[component];
    second_projection += displacement[component] * second_dipole[component];
    dipole_dot += first_dipole[component] * second_dipole[component];
    charge_dipole_numerator +=
        displacement[component] *
        (charges[first] * second_dipole[component] - charges[second] * first_dipole[component]);
  }
  const double distance2 = dx * dx + dy * dy + dz * dz;
  const double charge_dipole = kernel3 * charge_dipole_numerator;
  const double dipole_dipole =
      kernel5 * (distance2 * dipole_dot - 3.0 * first_projection * second_projection);
  const double charge_quadrupole = charges[first] * packed_dot(sq, second_quadrupole) +
                                   charges[second] * packed_dot(sq, first_quadrupole);
  energy = charge_dipole + dipole_dipole + charge_quadrupole;
  return std::isfinite(first_projection) && std::isfinite(second_projection) &&
         std::isfinite(dipole_dot) && std::isfinite(charge_dipole_numerator) &&
         std::isfinite(distance2) && std::isfinite(charge_dipole) && std::isfinite(dipole_dipole) &&
         std::isfinite(charge_quadrupole) && std::isfinite(energy);
}

bool pair_vjp(const double* pair_data, double average_radius, double first_radius_cn_derivative,
              double second_radius_cn_derivative, std::size_t first, std::size_t second,
              const double* charges, const double* dipoles, const double* quadrupoles,
              double* gradient_scratch, double* coordination_scratch) {
  const std::array<double, 3> displacement{{pair_data[0], pair_data[1], pair_data[2]}};
  const double kernel3 = pair_data[3];
  const double kernel5 = pair_data[4];
  const double distance = std::hypot(std::hypot(displacement[0], displacement[1]), displacement[2]);
  if (!(distance > 0.0) || !std::isfinite(distance) || !(average_radius > 0.0) ||
      !std::isfinite(average_radius) || !std::isfinite(first_radius_cn_derivative) ||
      !std::isfinite(second_radius_cn_derivative)) {
    return false;
  }

  /* Reconstruct f3/f5 without storing two more pair scalars. */
  const double scaled = average_radius / distance;
  const double scaled2 = scaled * scaled;
  const double scaled3 = scaled2 * scaled;
  const double scaled4 = scaled2 * scaled2;
  const double damping3 = 1.0 / (1.0 + 6.0 * scaled3);
  const double damping5 = 1.0 / (1.0 + 6.0 * scaled4);
  if (!(damping3 >= 0.0) || !std::isfinite(damping3) || !(damping5 >= 0.0) ||
      !std::isfinite(damping5)) {
    return false;
  }

  const double* const first_dipole = dipoles + first * 3u;
  const double* const second_dipole = dipoles + second * 3u;
  const double* const first_quadrupole = quadrupoles + first * 6u;
  const double* const second_quadrupole = quadrupoles + second * 6u;
  std::array<double, 3> charge_dipole_vector{};
  std::array<double, 3> tensor_vector{};
  double first_projection = 0.0;
  double second_projection = 0.0;
  double dipole_dot = 0.0;
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    charge_dipole_vector[axis] =
        charges[first] * second_dipole[axis] - charges[second] * first_dipole[axis];
    first_projection += displacement[axis] * first_dipole[axis];
    second_projection += displacement[axis] * second_dipole[axis];
    dipole_dot += first_dipole[axis] * second_dipole[axis];
  }
  const double charge_dipole_numerator = displacement[0] * charge_dipole_vector[0] +
                                         displacement[1] * charge_dipole_vector[1] +
                                         displacement[2] * charge_dipole_vector[2];
  const double distance2 = displacement[0] * displacement[0] + displacement[1] * displacement[1] +
                           displacement[2] * displacement[2];
  const double dipole_dipole_numerator =
      distance2 * dipole_dot - 3.0 * first_projection * second_projection;

  /* T = q_first Q_second + q_second Q_first in symmetric packed form. */
  const std::array<double, 6> tensor{{
      charges[first] * second_quadrupole[0] + charges[second] * first_quadrupole[0],
      charges[first] * second_quadrupole[1] + charges[second] * first_quadrupole[1],
      charges[first] * second_quadrupole[2] + charges[second] * first_quadrupole[2],
      charges[first] * second_quadrupole[3] + charges[second] * first_quadrupole[3],
      charges[first] * second_quadrupole[4] + charges[second] * first_quadrupole[4],
      charges[first] * second_quadrupole[5] + charges[second] * first_quadrupole[5],
  }};
  tensor_vector[0] =
      tensor[0] * displacement[0] + tensor[1] * displacement[1] + tensor[3] * displacement[2];
  tensor_vector[1] =
      tensor[1] * displacement[0] + tensor[2] * displacement[1] + tensor[4] * displacement[2];
  tensor_vector[2] =
      tensor[3] * displacement[0] + tensor[4] * displacement[1] + tensor[5] * displacement[2];
  const double charge_quadrupole_numerator = displacement[0] * tensor_vector[0] +
                                             displacement[1] * tensor_vector[1] +
                                             displacement[2] * tensor_vector[2];
  const double inverse_distance = 1.0 / distance;
  const double kernel3_distance_derivative = -3.0 * damping3 * kernel3 * inverse_distance;
  const double kernel5_distance_derivative = -(1.0 + 4.0 * damping5) * kernel5 * inverse_distance;
  const double inverse_radius = 1.0 / average_radius;
  const double kernel3_radius_derivative = -3.0 * (1.0 - damping3) * kernel3 * inverse_radius;
  const double kernel5_radius_derivative = -4.0 * (1.0 - damping5) * kernel5 * inverse_radius;
  const double kernel5_numerator = dipole_dipole_numerator + charge_quadrupole_numerator;
  const double energy_radius_derivative = kernel3_radius_derivative * charge_dipole_numerator +
                                          kernel5_radius_derivative * kernel5_numerator;
  if (!std::isfinite(charge_dipole_numerator) || !std::isfinite(distance2) ||
      !std::isfinite(dipole_dipole_numerator) || !std::isfinite(charge_quadrupole_numerator) ||
      !std::isfinite(kernel3_distance_derivative) || !std::isfinite(kernel5_distance_derivative) ||
      !std::isfinite(kernel3_radius_derivative) || !std::isfinite(kernel5_radius_derivative) ||
      !std::isfinite(kernel5_numerator) || !std::isfinite(energy_radius_derivative)) {
    return false;
  }

  for (std::size_t axis = 0; axis < 3u; ++axis) {
    const double dipole_dipole_derivative =
        2.0 * dipole_dot * displacement[axis] -
        3.0 * (first_projection * second_dipole[axis] + second_projection * first_dipole[axis]);
    const double pair_gradient =
        kernel3 * charge_dipole_vector[axis] +
        kernel3_distance_derivative * charge_dipole_numerator * displacement[axis] *
            inverse_distance +
        kernel5 * (dipole_dipole_derivative + 2.0 * tensor_vector[axis]) +
        kernel5_distance_derivative * kernel5_numerator * displacement[axis] * inverse_distance;
    const std::size_t first_coordinate = first * 3u + axis;
    const std::size_t second_coordinate = second * 3u + axis;
    if (!add_value(pair_gradient, gradient_scratch[first_coordinate]) ||
        !add_value(-pair_gradient, gradient_scratch[second_coordinate])) {
      return false;
    }
  }

  /* average_radius = (mrad_first + mrad_second)/2. */
  return add_value(0.5 * energy_radius_derivative * first_radius_cn_derivative,
                   coordination_scratch[first]) &&
         add_value(0.5 * energy_radius_derivative * second_radius_cn_derivative,
                   coordination_scratch[second]);
}

}  // namespace

namespace {

const std::vector<std::int64_t> kEmptyInt64Vector;
const std::vector<double> kEmptyDoubleVector;

}  // namespace

AES2Plan::AES2Plan(std::shared_ptr<const AES2PlanData> data) noexcept : data_(std::move(data)) {}

bool AES2Plan::sealed() const noexcept { return data_ != nullptr; }

std::int64_t AES2Plan::batch_size() const noexcept {
  return data_ == nullptr ? 0 : data_->batch_size;
}

std::int64_t AES2Plan::total_atoms() const noexcept {
  return data_ == nullptr ? 0 : data_->total_atoms;
}

std::int64_t AES2Plan::total_pairs() const noexcept {
  return data_ == nullptr ? 0 : data_->total_pairs;
}

std::int64_t AES2Plan::pair_data_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->pair_data_elements;
}

std::int64_t AES2Plan::potential_scratch_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->potential_scratch_elements;
}

std::int64_t AES2Plan::gradient_scratch_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->total_atoms * 3;
}

std::int64_t AES2Plan::coordination_scratch_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->total_atoms;
}

const std::vector<std::int64_t>& AES2Plan::atom_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->atom_offsets;
}

const std::vector<std::int64_t>& AES2Plan::pair_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->pair_offsets;
}

const std::vector<double>& AES2Plan::dipole_kernel() const noexcept {
  return data_ == nullptr ? kEmptyDoubleVector : data_->dipole_kernel;
}

const std::vector<double>& AES2Plan::quadrupole_kernel() const noexcept {
  return data_ == nullptr ? kEmptyDoubleVector : data_->quadrupole_kernel;
}

const std::vector<double>& AES2Plan::multipole_radius() const noexcept {
  return data_ == nullptr ? kEmptyDoubleVector : data_->multipole_radius;
}

const std::vector<double>& AES2Plan::multipole_valence_cn() const noexcept {
  return data_ == nullptr ? kEmptyDoubleVector : data_->multipole_valence_cn;
}

bool AES2Plan::overlaps_storage(const void* data, std::size_t size_bytes) const noexcept {
  return size_bytes != 0u && (data_ == nullptr || overlaps_plan_storage(*this, data, size_bytes));
}

const AES2PlanData* AES2Plan::identity() const noexcept { return data_.get(); }

gpuxtb_status_t make_aes2_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                               AES2Plan& plan, std::string& error) {
  gpuxtb_status_t status = validate_basis(basis, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (atomic_numbers == nullptr || !is_aligned(atomic_numbers, alignof(std::int32_t))) {
    error = "AES2 atomic numbers must not be NULL or misaligned";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  try {
    auto created = std::make_shared<AES2PlanData>();
    created->batch_size = basis.batch_size;
    created->total_atoms = basis.total_atoms;
    created->atom_offsets = basis.atom_offsets;
    created->pair_offsets.resize(static_cast<std::size_t>(basis.batch_size) + 1u);
    created->pair_offsets[0] = 0;
    for (std::int64_t batch = 0; batch < basis.batch_size; ++batch) {
      const std::int64_t atom_count = basis.atom_offsets[static_cast<std::size_t>(batch + 1)] -
                                      basis.atom_offsets[static_cast<std::size_t>(batch)];
      std::int64_t pair_count = 0;
      if (!checked_pair_count(atom_count, pair_count) ||
          !checked_add(pair_count, created->total_pairs)) {
        error = "AES2 ragged pair count exceeds supported dimensions";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      created->pair_offsets[static_cast<std::size_t>(batch + 1)] = created->total_pairs;
    }
    if (!checked_multiply(created->total_pairs, kPairStride, created->pair_data_elements) ||
        !checked_multiply(created->total_atoms, kPotentialStride,
                          created->potential_scratch_elements) ||
        !representable_as_size(created->pair_data_elements) ||
        !representable_as_size(created->potential_scratch_elements)) {
      error = "AES2 plan dimensions exceed host storage limits";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    const std::size_t atom_count = static_cast<std::size_t>(basis.total_atoms);
    created->dipole_kernel.resize(atom_count);
    created->quadrupole_kernel.resize(atom_count);
    created->multipole_radius.resize(atom_count);
    created->multipole_valence_cn.resize(atom_count);
    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::size_t atom_index = static_cast<std::size_t>(atom);
      const std::int32_t atomic_number = atomic_numbers[atom_index];
      const auto* element =
          parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number ||
          !std::isfinite(element->dipole_kernel) || !std::isfinite(element->quadrupole_kernel) ||
          !(element->multipole_radius > 0.0) ||
          element->multipole_radius > parameters::gfn2::kGlobal.multipole_rmax ||
          !std::isfinite(element->multipole_radius) || !(element->multipole_valence_cn > 0.0) ||
          !std::isfinite(element->multipole_valence_cn)) {
        error = "AES2 plan contains an unsupported element or invalid parameter";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }

      const std::int64_t shell_begin = basis.atom_shell_offsets[atom_index];
      const std::int64_t shell_end = basis.atom_shell_offsets[atom_index + 1u];
      if (shell_end - shell_begin != element->shell_count ||
          element->shell_offset > parameters::gfn2::kShells.size() ||
          element->shell_count > parameters::gfn2::kShells.size() - element->shell_offset) {
        error = "AES2 atomic numbers do not match the supplied basis shell layout";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        const std::size_t shell_index = static_cast<std::size_t>(shell);
        const std::size_t local_shell = static_cast<std::size_t>(shell - shell_begin);
        const auto& parameter = parameters::gfn2::kShells[element->shell_offset + local_shell];
        if (basis.shell_to_atom[shell_index] != atom ||
            basis.principal_quantum_numbers[shell_index] != parameter.principal_quantum_number ||
            basis.angular_momenta[shell_index] != parameter.angular_momentum ||
            basis.slater_exponents[shell_index] != parameter.slater) {
          error = "AES2 atomic numbers do not match the supplied basis shell metadata";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
      }

      created->dipole_kernel[atom_index] = element->dipole_kernel;
      created->quadrupole_kernel[atom_index] = element->quadrupole_kernel;
      created->multipole_radius[atom_index] = element->multipole_radius;
      created->multipole_valence_cn[atom_index] = element->multipole_valence_cn;
    }

    AES2Plan completed(std::move(created));
    status = validate_plan(completed, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
    plan = std::move(completed);
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN2 AES2 plan";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN2 AES2 plan dimensions exceed host container limits";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

gpuxtb_status_t update_aes2_geometry_cache_cpu(
    const AES2Plan& plan, const double* positions, const double* coordination_numbers,
    std::uint64_t geometry_generation, double* pair_storage, std::size_t pair_storage_elements,
    const AES2Workspace& workspace, AES2GeometryCache& cache, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(positions, plan.total_atoms() * 3, "AES2 positions are invalid",
                                 "AES2 positions contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(
      coordination_numbers, plan.total_atoms(), "AES2 coordination numbers are invalid",
      "AES2 coordination numbers must be finite and nonnegative", error, true);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const std::size_t required_elements = static_cast<std::size_t>(plan.pair_data_elements());
  if (pair_storage_elements < required_elements ||
      (required_elements != 0u &&
       (pair_storage == nullptr || !is_aligned(pair_storage, alignof(double))))) {
    error = "AES2 pair cache storage is NULL, misaligned, or too small";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_workspace_pointer(workspace.pair_scratch, workspace.pair_elements,
                                      plan.pair_data_elements(),
                                      "AES2 pair scratch is NULL, misaligned, or too small", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::size_t position_bytes = 0;
  std::size_t atom_bytes = 0;
  std::size_t pair_bytes = 0;
  if (!count_bytes(plan.total_atoms() * 3, sizeof(double), position_bytes) ||
      !count_bytes(plan.total_atoms(), sizeof(double), atom_bytes) ||
      !count_bytes(plan.pair_data_elements(), sizeof(double), pair_bytes)) {
    error = "AES2 cache dimensions exceed addressable host storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::array<MemoryRange, 4> active{{{positions, position_bytes},
                                           {coordination_numbers, atom_bytes},
                                           {pair_storage, pair_bytes},
                                           {workspace.pair_scratch, pair_bytes}}};
  if (!ranges_are_disjoint(active.data(), active.size())) {
    error = "AES2 cache inputs, output, and scratch must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (const MemoryRange& range : active) {
    if (overlaps_control_storage(plan, cache, workspace, range.data, range.size_bytes)) {
      error = "AES2 cache buffers must not overlap plan or descriptor storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  std::int64_t pair = 0;
  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    const std::int64_t atom_begin = plan.atom_offsets()[static_cast<std::size_t>(batch)];
    const std::int64_t atom_end = plan.atom_offsets()[static_cast<std::size_t>(batch + 1)];
    for (std::int64_t second = atom_begin; second < atom_end; ++second) {
      const std::size_t second_index = static_cast<std::size_t>(second);
      for (std::int64_t first = atom_begin; first < second; ++first) {
        const std::size_t first_index = static_cast<std::size_t>(first);
        const double dx = positions[first_index * 3u] - positions[second_index * 3u];
        const double dy = positions[first_index * 3u + 1u] - positions[second_index * 3u + 1u];
        const double dz = positions[first_index * 3u + 2u] - positions[second_index * 3u + 2u];
        const double distance = std::hypot(std::hypot(dx, dy), dz);
        const double distance_squared = distance * distance;
        if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz) ||
            !std::isfinite(distance) || distance_squared < kMinimumDistanceSquared) {
          error = "AES2 is undefined for coincident, near-coincident, or unrepresentable atoms";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        const double first_radius =
            multipole_radius(plan, first_index, coordination_numbers[first_index]);
        const double second_radius =
            multipole_radius(plan, second_index, coordination_numbers[second_index]);
        const double average_radius = 0.5 * (first_radius + second_radius);
        double kernel3 = 0.0;
        double kernel5 = 0.0;
        if (!(first_radius > 0.0) || !std::isfinite(first_radius) || !(second_radius > 0.0) ||
            !std::isfinite(second_radius) || !(average_radius > 0.0) ||
            !std::isfinite(average_radius) ||
            !pair_kernels(distance, average_radius, kernel3, kernel5)) {
          error = "AES2 damping-radius or pair-kernel arithmetic failed";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        const std::size_t base = static_cast<std::size_t>(pair * kPairStride);
        workspace.pair_scratch[base] = dx;
        workspace.pair_scratch[base + 1u] = dy;
        workspace.pair_scratch[base + 2u] = dz;
        workspace.pair_scratch[base + 3u] = kernel3;
        workspace.pair_scratch[base + 4u] = kernel5;
        ++pair;
      }
    }
  }
  if (pair != plan.total_pairs()) {
    error = "AES2 internal pair enumeration disagrees with the sealed plan";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }

  if (required_elements != 0u) {
    std::memcpy(pair_storage, workspace.pair_scratch, pair_bytes);
  }
  cache = AES2GeometryCache{pair_storage, plan.pair_data_elements(), geometry_generation,
                            plan.identity()};
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t evaluate_aes2_potential_cpu(const AES2Plan& plan, const AES2GeometryCache& cache,
                                            const double* atomic_charges,
                                            const double* atomic_dipoles,
                                            const double* atomic_quadrupoles,
                                            double* charge_potentials, double* dipole_potentials,
                                            double* quadrupole_potentials,
                                            const AES2Workspace& workspace, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_cache(plan, cache, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status =
      validate_finite_array(atomic_charges, plan.total_atoms(), "AES2 atomic charges are invalid",
                            "AES2 atomic charges contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(atomic_dipoles, plan.total_atoms() * 3,
                                 "AES2 atomic dipoles are invalid",
                                 "AES2 atomic dipoles contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(atomic_quadrupoles, plan.total_atoms() * 6,
                                 "AES2 atomic quadrupoles are invalid",
                                 "AES2 atomic quadrupoles contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (charge_potentials == nullptr || dipole_potentials == nullptr ||
      quadrupole_potentials == nullptr || !is_aligned(charge_potentials, alignof(double)) ||
      !is_aligned(dipole_potentials, alignof(double)) ||
      !is_aligned(quadrupole_potentials, alignof(double))) {
    error = "AES2 potential outputs must not be NULL or misaligned";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_workspace_pointer(
      workspace.potential_scratch, workspace.potential_elements, plan.potential_scratch_elements(),
      "AES2 potential scratch is NULL, misaligned, or too small", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::size_t atom_bytes = 0;
  std::size_t dipole_bytes = 0;
  std::size_t quadrupole_bytes = 0;
  std::size_t pair_bytes = 0;
  std::size_t scratch_bytes = 0;
  if (!count_bytes(plan.total_atoms(), sizeof(double), atom_bytes) ||
      !count_bytes(plan.total_atoms() * 3, sizeof(double), dipole_bytes) ||
      !count_bytes(plan.total_atoms() * 6, sizeof(double), quadrupole_bytes) ||
      !count_bytes(plan.pair_data_elements(), sizeof(double), pair_bytes) ||
      !count_bytes(plan.potential_scratch_elements(), sizeof(double), scratch_bytes)) {
    error = "AES2 potential dimensions exceed addressable host storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::array<MemoryRange, 8> active{{{atomic_charges, atom_bytes},
                                           {atomic_dipoles, dipole_bytes},
                                           {atomic_quadrupoles, quadrupole_bytes},
                                           {cache.pair_data, pair_bytes},
                                           {charge_potentials, atom_bytes},
                                           {dipole_potentials, dipole_bytes},
                                           {quadrupole_potentials, quadrupole_bytes},
                                           {workspace.potential_scratch, scratch_bytes}}};
  if (!ranges_are_disjoint(active.data(), active.size())) {
    error = "AES2 potential inputs, outputs, cache, and scratch must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (const MemoryRange& range : active) {
    if (overlaps_control_storage(plan, cache, workspace, range.data, range.size_bytes)) {
      error = "AES2 potential buffers must not overlap plan or descriptor storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  double* const scratch_charge = workspace.potential_scratch;
  double* const scratch_dipole =
      workspace.potential_scratch + static_cast<std::size_t>(plan.total_atoms());
  double* const scratch_quadrupole =
      workspace.potential_scratch + static_cast<std::size_t>(plan.total_atoms()) * 4u;
  if (!initialize_onsite_potential(plan, atomic_dipoles, atomic_quadrupoles,
                                   workspace.potential_scratch)) {
    error = "AES2 onsite potential arithmetic exceeded floating-point range";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  std::int64_t pair = 0;
  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    const std::int64_t atom_begin = plan.atom_offsets()[static_cast<std::size_t>(batch)];
    const std::int64_t atom_end = plan.atom_offsets()[static_cast<std::size_t>(batch + 1)];
    for (std::int64_t second = atom_begin; second < atom_end; ++second) {
      for (std::int64_t first = atom_begin; first < second; ++first) {
        const std::size_t pair_base = static_cast<std::size_t>(pair * kPairStride);
        if (!add_pair_potential(cache.pair_data + pair_base, static_cast<std::size_t>(first),
                                static_cast<std::size_t>(second), atomic_charges, atomic_dipoles,
                                atomic_quadrupoles, scratch_charge, scratch_dipole,
                                scratch_quadrupole)) {
          error = "AES2 pair potential arithmetic exceeded floating-point range";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        ++pair;
      }
    }
  }
  if (pair != plan.total_pairs()) {
    error = "AES2 internal pair enumeration disagrees with the geometry cache";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }

  std::memcpy(charge_potentials, scratch_charge, atom_bytes);
  std::memcpy(dipole_potentials, scratch_dipole, dipole_bytes);
  std::memcpy(quadrupole_potentials, scratch_quadrupole, quadrupole_bytes);
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_aes2_energy_cpu(const AES2Plan& plan, const AES2GeometryCache& cache,
                                    const double* atomic_charges, const double* atomic_dipoles,
                                    const double* atomic_quadrupoles, double* energies,
                                    const AES2Workspace& workspace, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_cache(plan, cache, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status =
      validate_finite_array(atomic_charges, plan.total_atoms(), "AES2 atomic charges are invalid",
                            "AES2 atomic charges contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(atomic_dipoles, plan.total_atoms() * 3,
                                 "AES2 atomic dipoles are invalid",
                                 "AES2 atomic dipoles contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(atomic_quadrupoles, plan.total_atoms() * 6,
                                 "AES2 atomic quadrupoles are invalid",
                                 "AES2 atomic quadrupoles contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(energies, plan.batch_size(), "AES2 energies are invalid",
                                 "AES2 input energies contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_workspace_pointer(
      workspace.batch_scratch, workspace.batch_elements, plan.batch_size(),
      "AES2 batch scratch is NULL, misaligned, or too small", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::size_t atom_bytes = 0;
  std::size_t dipole_bytes = 0;
  std::size_t quadrupole_bytes = 0;
  std::size_t pair_bytes = 0;
  std::size_t batch_bytes = 0;
  if (!count_bytes(plan.total_atoms(), sizeof(double), atom_bytes) ||
      !count_bytes(plan.total_atoms() * 3, sizeof(double), dipole_bytes) ||
      !count_bytes(plan.total_atoms() * 6, sizeof(double), quadrupole_bytes) ||
      !count_bytes(plan.pair_data_elements(), sizeof(double), pair_bytes) ||
      !count_bytes(plan.batch_size(), sizeof(double), batch_bytes)) {
    error = "AES2 energy dimensions exceed addressable host storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::array<MemoryRange, 6> active{{{atomic_charges, atom_bytes},
                                           {atomic_dipoles, dipole_bytes},
                                           {atomic_quadrupoles, quadrupole_bytes},
                                           {cache.pair_data, pair_bytes},
                                           {energies, batch_bytes},
                                           {workspace.batch_scratch, batch_bytes}}};
  if (!ranges_are_disjoint(active.data(), active.size())) {
    error = "AES2 energy inputs, output, cache, and scratch must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (const MemoryRange& range : active) {
    if (overlaps_control_storage(plan, cache, workspace, range.data, range.size_bytes)) {
      error = "AES2 energy buffers must not overlap plan or descriptor storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  std::fill_n(workspace.batch_scratch, static_cast<std::size_t>(plan.batch_size()), 0.0);
  std::int64_t pair = 0;
  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    const std::int64_t atom_begin = plan.atom_offsets()[static_cast<std::size_t>(batch)];
    const std::int64_t atom_end = plan.atom_offsets()[static_cast<std::size_t>(batch + 1)];
    double contribution = 0.0;
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::size_t atom_index = static_cast<std::size_t>(atom);
      double onsite = 0.0;
      if (!onsite_energy(plan, atom_index, atomic_dipoles + atom_index * 3u,
                         atomic_quadrupoles + atom_index * 6u, onsite) ||
          !add_value(onsite, contribution)) {
        error = "AES2 onsite energy arithmetic exceeded floating-point range";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
    for (std::int64_t second = atom_begin; second < atom_end; ++second) {
      for (std::int64_t first = atom_begin; first < second; ++first) {
        const std::size_t pair_base = static_cast<std::size_t>(pair * kPairStride);
        double pair_contribution = 0.0;
        if (!pair_energy(cache.pair_data + pair_base, static_cast<std::size_t>(first),
                         static_cast<std::size_t>(second), atomic_charges, atomic_dipoles,
                         atomic_quadrupoles, pair_contribution) ||
            !add_value(pair_contribution, contribution)) {
          error = "AES2 pair energy arithmetic exceeded floating-point range";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        ++pair;
      }
    }
    const double updated = energies[batch] + contribution;
    if (!std::isfinite(updated)) {
      error = "AES2 accumulated energy exceeded floating-point range";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    workspace.batch_scratch[batch] = contribution;
  }
  if (pair != plan.total_pairs()) {
    error = "AES2 internal pair enumeration disagrees with the geometry cache";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }

  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    energies[batch] += workspace.batch_scratch[batch];
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_aes2_vjp_cpu(const AES2Plan& plan, const AES2GeometryCache& cache,
                                 const double* positions, const double* coordination_numbers,
                                 std::uint64_t geometry_generation, const double* atomic_charges,
                                 const double* atomic_dipoles, const double* atomic_quadrupoles,
                                 double* gradients, double* coordination_adjoints,
                                 const AES2Workspace& workspace, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_cache(plan, cache, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (cache.geometry_generation != geometry_generation) {
    error = "AES2 VJP inputs do not match the cached geometry generation";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status =
      validate_finite_array(positions, plan.total_atoms() * 3, "AES2 VJP positions are invalid",
                            "AES2 VJP positions contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(
      coordination_numbers, plan.total_atoms(), "AES2 VJP coordination numbers are invalid",
      "AES2 VJP coordination numbers must be finite and nonnegative", error, true);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(atomic_charges, plan.total_atoms(), "AES2 VJP charges are invalid",
                                 "AES2 VJP charges contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status =
      validate_finite_array(atomic_dipoles, plan.total_atoms() * 3, "AES2 VJP dipoles are invalid",
                            "AES2 VJP dipoles contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(atomic_quadrupoles, plan.total_atoms() * 6,
                                 "AES2 VJP quadrupoles are invalid",
                                 "AES2 VJP quadrupoles contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(gradients, plan.gradient_scratch_elements(),
                                 "AES2 gradient output is invalid",
                                 "AES2 input gradients contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_finite_array(coordination_adjoints, plan.coordination_scratch_elements(),
                                 "AES2 coordination-adjoint output is invalid",
                                 "AES2 input coordination adjoints contain NaN or infinity", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_workspace_pointer(
      workspace.gradient_scratch, workspace.gradient_elements, plan.gradient_scratch_elements(),
      "AES2 gradient scratch is NULL, misaligned, or too small", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_workspace_pointer(
      workspace.coordination_scratch, workspace.coordination_elements,
      plan.coordination_scratch_elements(),
      "AES2 coordination scratch is NULL, misaligned, or too small", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::size_t atom_bytes = 0;
  std::size_t position_bytes = 0;
  std::size_t dipole_bytes = 0;
  std::size_t quadrupole_bytes = 0;
  std::size_t pair_bytes = 0;
  if (!count_bytes(plan.total_atoms(), sizeof(double), atom_bytes) ||
      !count_bytes(plan.gradient_scratch_elements(), sizeof(double), position_bytes) ||
      !count_bytes(plan.total_atoms() * 3, sizeof(double), dipole_bytes) ||
      !count_bytes(plan.total_atoms() * 6, sizeof(double), quadrupole_bytes) ||
      !count_bytes(plan.pair_data_elements(), sizeof(double), pair_bytes)) {
    error = "AES2 VJP dimensions exceed addressable host storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::array<MemoryRange, 10> active{{
      {positions, position_bytes},
      {coordination_numbers, atom_bytes},
      {atomic_charges, atom_bytes},
      {atomic_dipoles, dipole_bytes},
      {atomic_quadrupoles, quadrupole_bytes},
      {cache.pair_data, pair_bytes},
      {gradients, position_bytes},
      {coordination_adjoints, atom_bytes},
      {workspace.gradient_scratch, position_bytes},
      {workspace.coordination_scratch, atom_bytes},
  }};
  if (!ranges_are_disjoint(active.data(), active.size())) {
    error = "AES2 VJP inputs, outputs, cache, and scratch must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (const MemoryRange& range : active) {
    if (overlaps_control_storage(plan, cache, workspace, range.data, range.size_bytes)) {
      error = "AES2 VJP buffers must not overlap plan or descriptor storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  std::fill_n(workspace.gradient_scratch,
              static_cast<std::size_t>(plan.gradient_scratch_elements()), 0.0);
  std::fill_n(workspace.coordination_scratch,
              static_cast<std::size_t>(plan.coordination_scratch_elements()), 0.0);
  std::int64_t pair = 0;
  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    const std::int64_t atom_begin = plan.atom_offsets()[static_cast<std::size_t>(batch)];
    const std::int64_t atom_end = plan.atom_offsets()[static_cast<std::size_t>(batch + 1)];
    for (std::int64_t second = atom_begin; second < atom_end; ++second) {
      const std::size_t second_index = static_cast<std::size_t>(second);
      for (std::int64_t first = atom_begin; first < second; ++first) {
        const std::size_t first_index = static_cast<std::size_t>(first);
        const std::size_t pair_base = static_cast<std::size_t>(pair * kPairStride);
        const double dx = positions[first_index * 3u] - positions[second_index * 3u];
        const double dy = positions[first_index * 3u + 1u] - positions[second_index * 3u + 1u];
        const double dz = positions[first_index * 3u + 2u] - positions[second_index * 3u + 2u];
        const double distance = std::hypot(std::hypot(dx, dy), dz);
        const double distance_squared = distance * distance;
        const double first_radius =
            multipole_radius(plan, first_index, coordination_numbers[first_index]);
        const double second_radius =
            multipole_radius(plan, second_index, coordination_numbers[second_index]);
        const double average_radius = 0.5 * (first_radius + second_radius);
        double expected_kernel3 = 0.0;
        double expected_kernel5 = 0.0;
        if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz) ||
            !std::isfinite(distance) || distance_squared < kMinimumDistanceSquared ||
            !(first_radius > 0.0) || !std::isfinite(first_radius) || !(second_radius > 0.0) ||
            !std::isfinite(second_radius) || !(average_radius > 0.0) ||
            !std::isfinite(average_radius) ||
            !pair_kernels(distance, average_radius, expected_kernel3, expected_kernel5)) {
          error = "AES2 VJP geometry or damping-radius arithmetic failed";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        if (cache.pair_data[pair_base] != dx || cache.pair_data[pair_base + 1u] != dy ||
            cache.pair_data[pair_base + 2u] != dz ||
            cache.pair_data[pair_base + 3u] != expected_kernel3 ||
            cache.pair_data[pair_base + 4u] != expected_kernel5) {
          error = "AES2 VJP positions or coordination numbers disagree with the geometry cache";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        const double first_cn_derivative =
            multipole_radius_cn_derivative(plan, first_index, coordination_numbers[first_index]);
        const double second_cn_derivative =
            multipole_radius_cn_derivative(plan, second_index, coordination_numbers[second_index]);
        if (!pair_vjp(cache.pair_data + pair_base, average_radius, first_cn_derivative,
                      second_cn_derivative, first_index, second_index, atomic_charges,
                      atomic_dipoles, atomic_quadrupoles, workspace.gradient_scratch,
                      workspace.coordination_scratch)) {
          error = "AES2 coordinate/CN VJP arithmetic exceeded floating-point range";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        ++pair;
      }
    }
  }
  if (pair != plan.total_pairs()) {
    error = "AES2 internal pair enumeration disagrees with the geometry cache";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }

  for (std::size_t coordinate = 0;
       coordinate < static_cast<std::size_t>(plan.gradient_scratch_elements()); ++coordinate) {
    if (!std::isfinite(gradients[coordinate] + workspace.gradient_scratch[coordinate])) {
      error = "AES2 accumulated gradient exceeded floating-point range";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t atom = 0; atom < static_cast<std::size_t>(plan.coordination_scratch_elements());
       ++atom) {
    if (!std::isfinite(coordination_adjoints[atom] + workspace.coordination_scratch[atom])) {
      error = "AES2 accumulated coordination adjoint exceeded floating-point range";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t coordinate = 0;
       coordinate < static_cast<std::size_t>(plan.gradient_scratch_elements()); ++coordinate) {
    gradients[coordinate] += workspace.gradient_scratch[coordinate];
  }
  for (std::size_t atom = 0; atom < static_cast<std::size_t>(plan.coordination_scratch_elements());
       ++atom) {
    coordination_adjoints[atom] += workspace.coordination_scratch[atom];
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail::gfn2
