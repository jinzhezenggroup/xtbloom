// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/periodic_multipole.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "data/parameters/gfn2.hpp"

namespace xtbloom::detail::gfn2 {

struct PeriodicMultipolePlanData final {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t matrix_elements = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<Lattice3D> lattices;
  std::vector<double> alphas;
  std::vector<std::int64_t> direct_translation_offsets;
  std::vector<std::int64_t> reciprocal_translation_offsets;
  std::vector<LatticeTranslation> direct_translations;
  std::vector<LatticeTranslation> reciprocal_translations;
  std::vector<double> dipole_kernel;
  std::vector<double> quadrupole_kernel;
  std::vector<double> multipole_radius;
  std::vector<double> multipole_valence_cn;
  const PeriodicShortRangePlanData* topology_identity = nullptr;
};

namespace {

using Vec3 = std::array<double, 3>;
using Vec6 = std::array<double, 6>;
using Mat3 = std::array<double, 9>;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrtPi = 1.7724538509055160272981674833411451828;
constexpr double kWignerToleranceSquared = 0.3;
constexpr double kWignerThreshold = 1.4901161193847656e-8;
constexpr double kAlphaTolerance = std::sqrt(std::numeric_limits<double>::epsilon());
constexpr double kMultipoleCutoff = 100.0;
constexpr double kMultipoleReciprocalConvergence = 100.0 * kAlphaTolerance;
constexpr double kDampingExponent3 = 3.0;
constexpr double kDampingExponent5 = 4.0;

static_assert(parameters::gfn2::kGlobal.multipole_dmp3 == kDampingExponent3,
              "periodic multipole evaluator specializes dmp3=3");
static_assert(parameters::gfn2::kGlobal.multipole_dmp5 == kDampingExponent5,
              "periodic multipole evaluator specializes dmp5=4");
static_assert(parameters::gfn2::kGlobal.multipole_kexp == 4.0,
              "periodic multipole evaluator specializes kexp=4");
static_assert(parameters::gfn2::kGlobal.multipole_shift == 1.2,
              "periodic multipole evaluator specializes shift=1.2");
static_assert(parameters::gfn2::kGlobal.multipole_rmax == 5.0,
              "periodic multipole evaluator specializes rmax=5");

struct WscImage {
  Vec3 vector{};  // wrapped center - wrapped image - lattice translation
  double weight = 0.0;
  Vec3 weight_gradient{};  // derivative with respect to wrapped center - image
  Mat3 weight_strain{};    // affine cell/coordinate derivative
};

struct MultipoleTerms {
  Vec3 sd{};
  Mat3 dd{};
  Vec6 sq{};
};

struct MultipoleDerivatives {
  double radius = 0.0;
  Vec3 gradient{};
  Mat3 strain{};
};

struct MemoryRange {
  const void* data = nullptr;
  std::size_t bytes = 0u;
};

bool ranges_overlap(const MemoryRange& first, const MemoryRange& second) noexcept {
  if (first.bytes == 0u || second.bytes == 0u) return false;
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first.data);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second.data);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first.bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second.bytes) {
    return true;
  }
  return first_begin < second_begin + second.bytes && second_begin < first_begin + first.bytes;
}

bool all_disjoint(const std::vector<MemoryRange>& ranges) noexcept {
  for (std::size_t first = 0; first < ranges.size(); ++first) {
    for (std::size_t second = first + 1u; second < ranges.size(); ++second) {
      if (ranges_overlap(ranges[first], ranges[second])) return false;
    }
  }
  return true;
}

bool aligned_double(const void* data) noexcept {
  return data != nullptr && reinterpret_cast<std::uintptr_t>(data) % alignof(double) == 0u;
}

bool finite_array(const double* data, std::size_t count) noexcept {
  if (data == nullptr) return false;
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(data[index])) return false;
  }
  return true;
}

bool finite_matrix(const double* data, std::size_t count) noexcept {
  return finite_array(data, count);
}

double dot(const Vec3& first, const Vec3& second) noexcept {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

double norm(const Vec3& value) noexcept { return std::sqrt(dot(value, value)); }

Vec3 negate(const Vec3& value) noexcept { return {-value[0], -value[1], -value[2]}; }

Vec3 subtract(const Vec3& first, const Vec3& second) noexcept {
  return {first[0] - second[0], first[1] - second[1], first[2] - second[2]};
}

Vec3 scale(const Vec3& value, double factor) noexcept {
  return {factor * value[0], factor * value[1], factor * value[2]};
}

Mat3 outer(const Vec3& first, const Vec3& second) noexcept {
  Mat3 result{};
  for (std::size_t row = 0; row < 3u; ++row) {
    for (std::size_t column = 0; column < 3u; ++column) {
      result[row * 3u + column] = first[row] * second[column];
    }
  }
  return result;
}

void add_scaled(Vec3& target, const Vec3& value, double factor = 1.0) noexcept {
  for (std::size_t axis = 0; axis < 3u; ++axis) target[axis] += factor * value[axis];
}

void add_scaled(Mat3& target, const Mat3& value, double factor = 1.0) noexcept {
  for (std::size_t component = 0; component < 9u; ++component) {
    target[component] += factor * value[component];
  }
}

void add_scaled(Vec6& target, const Vec6& value, double factor = 1.0) noexcept {
  for (std::size_t component = 0; component < 6u; ++component) {
    target[component] += factor * value[component];
  }
}

Vec6 packed_outer(const Vec3& vector, double factor = 1.0) noexcept {
  return {factor * vector[0] * vector[0],       factor * 2.0 * vector[0] * vector[1],
          factor * vector[1] * vector[1],       factor * 2.0 * vector[0] * vector[2],
          factor * 2.0 * vector[1] * vector[2], factor * vector[2] * vector[2]};
}

double packed_dot(const Vec6& tensor, const double* quadrupole) noexcept {
  double result = 0.0;
  for (std::size_t component = 0; component < 6u; ++component) {
    result += tensor[component] * quadrupole[component];
  }
  return result;
}

bool is_origin(const LatticeTranslation& translation) noexcept {
  return translation.index[0] == 0 && translation.index[1] == 0 && translation.index[2] == 0;
}

double smooth_shape(double delta) noexcept {
  const double x = std::min(1.0, std::max(0.0, delta) / kWignerToleranceSquared);
  return std::max(0.0, 1.0 - 10.0 * x * x * x + 15.0 * x * x * x * x - 6.0 * x * x * x * x * x);
}

double smooth_shape_derivative(double delta) noexcept {
  const double x = std::min(1.0, std::max(0.0, delta) / kWignerToleranceSquared);
  return -30.0 * x * x * (1.0 - x) * (1.0 - x) / kWignerToleranceSquared;
}

bool finite_lattice(const Lattice3D& lattice) noexcept {
  if (!(lattice.volume > 0.0) || !std::isfinite(lattice.volume)) return false;
  for (double value : lattice.direct) {
    if (!std::isfinite(value)) return false;
  }
  for (double value : lattice.reciprocal) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

double reciprocal_multipole_decay(double distance, double alpha, double volume) noexcept {
  const double squared = distance * distance;
  return 4.0 * kPi * std::exp(-0.25 * squared / (alpha * alpha)) / volume;
}

double direct_multipole_decay(double distance, double alpha) noexcept {
  const double argument = alpha * distance;
  return (std::erfc(argument) + 2.0 * argument * std::exp(-argument * argument) / kSqrtPi) /
         (distance * distance * distance);
}

double select_alpha(const Lattice3D& lattice) noexcept {
  double direct_min = std::numeric_limits<double>::infinity();
  double reciprocal_min = std::numeric_limits<double>::infinity();
  for (std::size_t row = 0; row < 3u; ++row) {
    Vec3 direct{};
    Vec3 reciprocal{};
    for (std::size_t component = 0; component < 3u; ++component) {
      direct[component] = lattice.direct[row * 3u + component];
      reciprocal[component] = lattice.reciprocal[row * 3u + component];
    }
    direct_min = std::min(direct_min, norm(direct));
    reciprocal_min = std::min(reciprocal_min, norm(reciprocal));
  }
  if (!(direct_min > 0.0) || !(reciprocal_min > 0.0) || !std::isfinite(direct_min) ||
      !std::isfinite(reciprocal_min)) {
    return 0.25;
  }
  const auto difference = [&](double alpha) {
    return (reciprocal_multipole_decay(4.0 * reciprocal_min, alpha, lattice.volume) -
            reciprocal_multipole_decay(5.0 * reciprocal_min, alpha, lattice.volume)) -
           (direct_multipole_decay(2.0 * direct_min, alpha) -
            direct_multipole_decay(3.0 * direct_min, alpha));
  };

  double alpha = kAlphaTolerance;
  double diff = difference(alpha);
  while (diff < -kAlphaTolerance && alpha <= std::numeric_limits<double>::max()) {
    alpha *= 2.0;
    diff = difference(alpha);
  }
  if (!std::isfinite(alpha) || alpha == kAlphaTolerance) return 0.25;

  double lower = 0.5 * alpha;
  while (diff < kAlphaTolerance && alpha <= std::numeric_limits<double>::max()) {
    alpha *= 2.0;
    diff = difference(alpha);
  }
  if (!std::isfinite(alpha)) return 0.25;
  double upper = alpha;
  alpha = 0.5 * (lower + upper);
  diff = difference(alpha);
  int iterations = 0;
  while (std::abs(diff) > kAlphaTolerance && iterations <= 30) {
    if (diff < 0.0) {
      lower = alpha;
    } else {
      upper = alpha;
    }
    alpha = 0.5 * (lower + upper);
    diff = difference(alpha);
    ++iterations;
  }
  return iterations > 30 ? 0.25 : alpha;
}

double search_reciprocal_cutoff(double alpha, double volume) noexcept {
  double cutoff = kAlphaTolerance;
  double value = reciprocal_multipole_decay(cutoff, alpha, volume);
  while (value > kMultipoleReciprocalConvergence && cutoff <= std::numeric_limits<double>::max()) {
    cutoff *= 2.0;
    value = reciprocal_multipole_decay(cutoff, alpha, volume);
  }
  double lower = 0.5 * cutoff;
  double lower_value = reciprocal_multipole_decay(lower, alpha, volume);
  double upper = cutoff;
  double upper_value = value;
  for (int iteration = 0; iteration < 30; ++iteration) {
    if (lower_value - upper_value <= kMultipoleReciprocalConvergence) break;
    cutoff = 0.5 * (lower + upper);
    value = reciprocal_multipole_decay(cutoff, alpha, volume);
    if (value >= kMultipoleReciprocalConvergence) {
      lower = cutoff;
      lower_value = value;
    } else {
      upper = cutoff;
      upper_value = value;
    }
  }
  return cutoff;
}

bool build_wsc_images(const Lattice3D& lattice, const Vec3& rij, bool self,
                      std::vector<WscImage>& images, std::string& error) {
  std::vector<LatticeTranslation> translations;
  xtbloom_status_t status = make_lattice_translations(
      lattice, kWignerThreshold, LatticeOriginPolicy::kInclude, translations, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return false;

  struct Candidate {
    Vec3 vector{};
    double squared = 0.0;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(translations.size());
  double minimum = std::numeric_limits<double>::infinity();
  for (const auto& translation : translations) {
    if (self && is_origin(translation)) continue;
    const Vec3 vector{{rij[0] - translation.cartesian[0], rij[1] - translation.cartesian[1],
                       rij[2] - translation.cartesian[2]}};
    const double squared = dot(vector, vector);
    /* tblite compares the squared distance with sqrt(epsilon), rather than
     * epsilon itself.  Keep the same predicate here so a near-coincident
     * image is excluded from both Wigner--Seitz selection and the kernel
     * evaluation; otherwise the selector could choose an image that the
     * matrix routine subsequently skips. */
    if (!(squared >= kWignerThreshold) || !std::isfinite(squared)) continue;
    candidates.push_back({vector, squared});
    minimum = std::min(minimum, squared);
  }
  if (!std::isfinite(minimum)) {
    error = "periodic multipole Wigner-Seitz image search found no image";
    return false;
  }

  double shape_sum = 0.0;
  for (const auto& candidate : candidates) {
    shape_sum += smooth_shape(std::max(0.0, candidate.squared - minimum));
  }
  if (!(shape_sum > 0.0) || !std::isfinite(shape_sum)) {
    error = "periodic multipole Wigner-Seitz image weights are invalid";
    return false;
  }

  const auto reference =
      std::find_if(candidates.begin(), candidates.end(),
                   [&](const Candidate& candidate) { return candidate.squared == minimum; });
  if (reference == candidates.end()) {
    error = "periodic multipole Wigner-Seitz reference image is missing";
    return false;
  }
  Vec3 sum_gradient{};
  Mat3 sum_strain{};
  std::vector<double> shapes;
  std::vector<Vec3> shape_gradients;
  std::vector<Mat3> shape_strains;
  shapes.reserve(candidates.size());
  shape_gradients.reserve(candidates.size());
  shape_strains.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    const double delta = std::max(0.0, candidate.squared - minimum);
    const double derivative = smooth_shape_derivative(delta);
    const Vec3 difference = subtract(candidate.vector, reference->vector);
    const Vec3 squared_gradient = scale(difference, 2.0);
    Mat3 squared_strain{};
    for (std::size_t row = 0; row < 3u; ++row) {
      for (std::size_t column = 0; column < 3u; ++column) {
        squared_strain[row * 3u + column] =
            2.0 * (candidate.vector[row] * candidate.vector[column] -
                   reference->vector[row] * reference->vector[column]);
      }
    }
    const double shape = smooth_shape(delta);
    const Vec3 shape_gradient = scale(squared_gradient, derivative);
    Mat3 shape_strain{};
    for (std::size_t component = 0; component < 9u; ++component) {
      shape_strain[component] = derivative * squared_strain[component];
    }
    shapes.push_back(shape);
    shape_gradients.push_back(shape_gradient);
    shape_strains.push_back(shape_strain);
    add_scaled(sum_gradient, shape_gradient);
    add_scaled(sum_strain, shape_strain);
  }

  images.clear();
  images.reserve(candidates.size());
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    WscImage image;
    image.vector = candidates[index].vector;
    image.weight = shapes[index] / shape_sum;
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      image.weight_gradient[axis] =
          (shape_gradients[index][axis] - image.weight * sum_gradient[axis]) / shape_sum;
    }
    for (std::size_t component = 0; component < 9u; ++component) {
      image.weight_strain[component] =
          (shape_strains[index][component] - image.weight * sum_strain[component]) / shape_sum;
    }
    if (image.weight > 0.0) images.push_back(image);
  }
  if (images.empty()) {
    error = "periodic multipole Wigner-Seitz image weights are empty";
    return false;
  }
  return true;
}

MultipoleTerms matrix_terms(const Vec3& vector, double radius, double alpha,
                            const std::vector<LatticeTranslation>& direct) {
  MultipoleTerms result{};
  const double alpha_squared = alpha * alpha;
  for (const auto& translation : direct) {
    const Vec3 value{{vector[0] + translation.cartesian[0], vector[1] + translation.cartesian[1],
                      vector[2] + translation.cartesian[2]}};
    const double distance = norm(value);
    if (!(distance >= kWignerThreshold)) continue;
    const double inverse = 1.0 / distance;
    const double inverse_squared = inverse * inverse;
    const double inverse_cubed = inverse_squared * inverse;
    const double inverse_fifth = inverse_cubed * inverse_squared;
    const double scaled = radius * inverse;
    const double scaled_squared = scaled * scaled;
    const double damping3 = 1.0 / (1.0 + 6.0 * scaled_squared * scaled);
    const double damping5 = 1.0 / (1.0 + 6.0 * scaled_squared * scaled_squared);
    const double argument = alpha * distance;
    const double argument_squared = argument * argument;
    const double exponential = std::exp(-argument_squared) / kSqrtPi;
    const double erf_term = -std::erf(argument) * inverse;
    const double e1 = inverse_squared * (erf_term + 2.0 * exponential * alpha);
    const double e2 = inverse_squared * (e1 + 4.0 * exponential * alpha_squared * alpha / 3.0);
    const double sd_kernel = damping3 * inverse_cubed + e1;
    const double dd_diagonal = damping5 * inverse_cubed + e1;
    const double dd_tensor = damping5 * inverse_fifth + e2;
    add_scaled(result.sd, scale(value, sd_kernel));
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      result.dd[axis * 3u + axis] += dd_diagonal;
    }
    const Mat3 vv = outer(value, value);
    add_scaled(result.dd, vv, -3.0 * dd_tensor);
    const Vec6 packed = packed_outer(value, dd_tensor);
    add_scaled(result.sq, packed);
    /* The traceless quadrupole basis subtracts the scalar diagonal kernel;
     * it is not an outer product with the displacement.  This distinction
     * is easy to miss because the off-diagonal packed components do carry a
     * factor of two. */
    const double trace = dd_diagonal / 3.0;
    result.sq[0] -= trace;
    result.sq[2] -= trace;
    result.sq[5] -= trace;
  }

  return result;
}

MultipoleTerms matrix_terms_with_volume(const Vec3& vector, double radius, double alpha,
                                        double volume,
                                        const std::vector<LatticeTranslation>& direct,
                                        const std::vector<LatticeTranslation>& reciprocal) {
  MultipoleTerms result = matrix_terms(vector, radius, alpha, direct);
  const double factor = 4.0 * kPi / volume;
  const double alpha_squared = alpha * alpha;
  for (const auto& translation : reciprocal) {
    const Vec3 g{{translation.cartesian[0], translation.cartesian[1], translation.cartesian[2]}};
    const double g2 = dot(g, g);
    if (!(g2 >= std::numeric_limits<double>::epsilon())) continue;
    const double exponential = factor * std::exp(-0.25 * g2 / alpha_squared) / g2;
    const double phase = dot(vector, g);
    const double sine = std::sin(phase) * exponential;
    const double cosine = std::cos(phase) * exponential;
    add_scaled(result.sd, scale(g, sine));
    add_scaled(result.dd, outer(g, g), cosine);
    add_scaled(result.sq, packed_outer(g, -cosine / 3.0));
  }
  return result;
}

Vec3 quadrupole_vector(const double* quadrupole, const Vec3& vector) noexcept {
  return {quadrupole[0] * vector[0] + quadrupole[1] * vector[1] + quadrupole[3] * vector[2],
          quadrupole[1] * vector[0] + quadrupole[2] * vector[1] + quadrupole[4] * vector[2],
          quadrupole[3] * vector[0] + quadrupole[4] * vector[1] + quadrupole[5] * vector[2]};
}

double quadrupole_scalar(const double* quadrupole, const Vec3& vector) noexcept {
  return dot(vector, quadrupole_vector(quadrupole, vector));
}

double pair_energy(double qi, double qj, const double* mi, const double* mj, const double* ti,
                   const double* tj, const MultipoleTerms& terms) noexcept {
  Vec3 first{{mi[0], mi[1], mi[2]}};
  Vec3 second{{mj[0], mj[1], mj[2]}};
  const Vec3 charge_dipole = subtract(scale(first, qj), scale(second, qi));
  const Vec3 dd_first{
      {terms.dd[0] * second[0] + terms.dd[1] * second[1] + terms.dd[2] * second[2],
       terms.dd[3] * second[0] + terms.dd[4] * second[1] + terms.dd[5] * second[2],
       terms.dd[6] * second[0] + terms.dd[7] * second[1] + terms.dd[8] * second[2]}};
  Vec6 charge_quadrupole{};
  for (std::size_t component = 0; component < 6u; ++component) {
    charge_quadrupole[component] = qi * tj[component] + qj * ti[component];
  }
  return dot(charge_dipole, terms.sd) + dot(first, dd_first) +
         packed_dot(terms.sq, charge_quadrupole.data());
}

MultipoleDerivatives derivative_terms(const Vec3& rij, double qi, double qj, const Vec3& mi,
                                      const Vec3& mj, const double* ti, const double* tj,
                                      double radius, double alpha,
                                      const std::vector<LatticeTranslation>& direct,
                                      const std::vector<LatticeTranslation>& reciprocal,
                                      double volume) {
  MultipoleDerivatives result{};
  const double alpha_squared = alpha * alpha;

  for (const auto& translation : direct) {
    const Vec3 vector{{rij[0] + translation.cartesian[0], rij[1] + translation.cartesian[1],
                       rij[2] + translation.cartesian[2]}};
    const double distance = norm(vector);
    if (!(distance >= kWignerThreshold)) continue;
    const double squared = distance * distance;
    const double inverse = 1.0 / distance;
    const double inverse_squared = inverse * inverse;
    const double g3 = inverse_squared * inverse;
    const double g5 = g3 * inverse_squared;
    const double g7 = g5 * inverse_squared;
    const double argument = alpha * distance;
    const double argument_squared = argument * argument;
    const double exponential = std::exp(-argument_squared) / kSqrtPi;
    const double erf_term = -std::erf(argument) * inverse;
    const double e1 = inverse_squared * (erf_term + exponential * (2.0 * alpha_squared) / alpha);
    const double e2 = inverse_squared * (e1 + exponential * (2.0 * alpha_squared) *
                                                  (2.0 * alpha_squared) / (3.0 * alpha));
    const double e3 =
        inverse_squared * (e2 + exponential * (2.0 * alpha_squared) * (2.0 * alpha_squared) *
                                    (2.0 * alpha_squared) / (15.0 * alpha));
    const double scaled = radius * inverse;
    const double scaled_squared = scaled * scaled;
    const double damping3 = 1.0 / (1.0 + 6.0 * scaled_squared * scaled);
    const double damping5 = 1.0 / (1.0 + 6.0 * scaled_squared * scaled_squared);
    const double ddamping3 = -3.0 * damping3 - 3.0 * damping3 * (damping3 - 1.0);
    const double ddamping5 = -5.0 * damping5 - 4.0 * (damping5 * damping5 - damping5);

    const double dpiqj = dot(vector, mi) * qj;
    const double qidpj = dot(vector, mj) * qi;
    const double charge_dipole_difference = dpiqj - qidpj;
    Vec3 g_sd = scale(vector, -(ddamping3 * g5) * charge_dipole_difference);
    add_scaled(g_sd, subtract(scale(mj, qi), scale(mi, qj)), damping3 * g3);

    Mat3 tab{};
    for (std::size_t a = 0; a < 3u; ++a) {
      for (std::size_t b = 0; b < 3u; ++b) tab[a * 3u + b] = 3.0 * vector[a] * vector[b] * e2;
      tab[a * 3u + a] -= e1;
    }
    for (std::size_t k = 0; k < 3u; ++k) {
      for (std::size_t a = 0; a < 3u; ++a) {
        g_sd[k] += qj * tab[a * 3u + k] * mi[a] - qi * tab[a * 3u + k] * mj[a];
      }
    }
    result.radius += 3.0 * charge_dipole_difference * kDampingExponent3 * damping3 * g3 *
                     (damping3 / radius) * (radius * inverse) * (radius * inverse) *
                     (radius * inverse);
    add_scaled(result.gradient, g_sd);
    add_scaled(result.strain, outer(vector, g_sd), -0.5);
    add_scaled(result.strain, outer(g_sd, vector), -0.5);

    const double dipole_dot = dot(mj, mi);
    const double dipole_i_projection = dot(mi, vector);
    const double dipole_j_projection = dot(mj, vector);
    const double dipole_energy =
        dipole_dot * squared - 3.0 * dipole_j_projection * dipole_i_projection;
    Vec3 g_dd = scale(vector, -2.0 * damping5 * g5 * dipole_dot);
    const Vec3 dipole_linear = {dipole_i_projection * mj[0] + dipole_j_projection * mi[0],
                                dipole_i_projection * mj[1] + dipole_j_projection * mi[1],
                                dipole_i_projection * mj[2] + dipole_j_projection * mi[2]};
    add_scaled(g_dd, dipole_linear, 3.0 * damping5 * g5);
    add_scaled(g_dd, vector, -dipole_energy * ddamping5 * g7);

    std::array<std::array<std::array<double, 3>, 3>, 3> tabc{};
    for (std::size_t c = 0; c < 3u; ++c) {
      for (std::size_t a = 0; a < 3u; ++a) {
        for (std::size_t b = 0; b < 3u; ++b) {
          tabc[a][b][c] = -15.0 * vector[a] * vector[b] * vector[c] * e3;
        }
      }
      for (std::size_t a = 0; a < 3u; ++a) {
        tabc[a][a][c] += 3.0 * e2 * vector[c];
        tabc[c][a][c] += 3.0 * e2 * vector[a];
        tabc[a][c][c] += 3.0 * e2 * vector[a];
      }
    }
    for (std::size_t k = 0; k < 3u; ++k) {
      for (std::size_t a = 0; a < 3u; ++a) {
        for (std::size_t b = 0; b < 3u; ++b) {
          g_dd[k] += mi[a] * tabc[b][a][k] * mj[b];
        }
      }
    }
    result.radius += 3.0 * dipole_energy * kDampingExponent5 * damping5 * g5 * (damping5 / radius) *
                     (radius * inverse) * (radius * inverse) * (radius * inverse) *
                     (radius * inverse);
    add_scaled(result.gradient, g_dd);
    add_scaled(result.strain, outer(vector, g_dd), -0.5);
    add_scaled(result.strain, outer(g_dd, vector), -0.5);

    Vec6 tensor{};
    for (std::size_t component = 0; component < 6u; ++component) {
      tensor[component] = qj * ti[component] + qi * tj[component];
    }
    const double quadrupole_energy =
        tensor[0] * vector[0] * vector[0] + tensor[2] * vector[1] * vector[1] +
        tensor[5] * vector[2] * vector[2] + 2.0 * tensor[1] * vector[0] * vector[1] +
        2.0 * tensor[3] * vector[0] * vector[2] + 2.0 * tensor[4] * vector[1] * vector[2];
    const Vec3 tj_vector = quadrupole_vector(tj, vector);
    const Vec3 ti_vector = quadrupole_vector(ti, vector);
    Vec3 g_sq = scale(vector, -quadrupole_energy * ddamping5 * g7);
    add_scaled(g_sq, tj_vector, -2.0 * damping5 * g5 * qi);
    add_scaled(g_sq, ti_vector, -2.0 * damping5 * g5 * qj);
    for (std::size_t k = 0; k < 3u; ++k) {
      double tj_contract = 0.0;
      double ti_contract = 0.0;
      for (std::size_t a = 0; a < 3u; ++a) {
        for (std::size_t b = 0; b < 3u; ++b) {
          const double tj_weight = (a == b) ? 1.0 : 2.0;
          (void)tj_weight;
        }
      }
      tj_contract = tabc[0][0][k] * tj[0] + 2.0 * tabc[1][0][k] * tj[1] +
                    2.0 * tabc[2][0][k] * tj[3] + tabc[1][1][k] * tj[2] +
                    2.0 * tabc[2][1][k] * tj[4] + tabc[2][2][k] * tj[5];
      ti_contract = tabc[0][0][k] * ti[0] + 2.0 * tabc[1][0][k] * ti[1] +
                    2.0 * tabc[2][0][k] * ti[3] + tabc[1][1][k] * ti[2] +
                    2.0 * tabc[2][1][k] * ti[4] + tabc[2][2][k] * ti[5];
      g_sq[k] += (-qi * tj_contract - qj * ti_contract) / 3.0;
    }
    result.radius += quadrupole_energy * 3.0 * kDampingExponent5 * damping5 * g5 *
                     (damping5 / radius) * (radius * inverse) * (radius * inverse) *
                     (radius * inverse) * (radius * inverse);
    add_scaled(result.gradient, g_sq);
    add_scaled(result.strain, outer(vector, g_sq), -0.5);
    add_scaled(result.strain, outer(g_sq, vector), -0.5);
  }

  const double factor = 4.0 * kPi / volume;
  for (const auto& translation : reciprocal) {
    const Vec3 g{{translation.cartesian[0], translation.cartesian[1], translation.cartesian[2]}};
    const double g2 = dot(g, g);
    if (!(g2 >= std::numeric_limits<double>::epsilon())) continue;
    const double exponential = factor * std::exp(-0.25 * g2 / alpha_squared) / g2;
    const double phase = dot(rij, g);
    const double cosine = std::cos(phase) * exponential;
    const double sine = std::sin(phase) * exponential;
    const double dpiqj = dot(g, mi) * qj;
    const double qidpj = dot(g, mj) * qi;
    const double difference = dpiqj - qidpj;
    add_scaled(result.gradient, g, -cosine * difference);
    const Mat3 kernel = [&]() {
      Mat3 value = outer(g, g);
      const double factor_value = 2.0 / g2 + 0.5 / alpha_squared;
      for (std::size_t component = 0; component < 9u; ++component) value[component] *= factor_value;
      value[0] -= 1.0;
      value[4] -= 1.0;
      value[8] -= 1.0;
      return value;
    }();
    add_scaled(result.strain, kernel, sine * difference);
    const Vec3 charge_dipole = subtract(scale(mi, qj), scale(mj, qi));
    Mat3 charge_dipole_outer = outer(g, charge_dipole);
    add_scaled(charge_dipole_outer, outer(charge_dipole, g));
    add_scaled(result.strain, charge_dipole_outer, -0.5 * sine);

    const double dipole_i_projection = dot(mi, g);
    const double dipole_j_projection = dot(mj, g);
    add_scaled(result.gradient, g, sine * dipole_i_projection * dipole_j_projection);
    const Vec3 dipole_linear = {mi[0] * dipole_j_projection + mj[0] * dipole_i_projection,
                                mi[1] * dipole_j_projection + mj[1] * dipole_i_projection,
                                mi[2] * dipole_j_projection + mj[2] * dipole_i_projection};
    Mat3 dipole_outer = outer(g, dipole_linear);
    add_scaled(dipole_outer, outer(dipole_linear, g));
    add_scaled(result.strain, kernel, cosine * dipole_i_projection * dipole_j_projection);
    add_scaled(result.strain, dipole_outer, -0.5 * cosine);

    const double qiqpj = qi * quadrupole_scalar(tj, g);
    const double qpiqj = qj * quadrupole_scalar(ti, g);
    const Vec3 tj_vector = quadrupole_vector(tj, g);
    const Vec3 ti_vector = quadrupole_vector(ti, g);
    add_scaled(result.gradient, g, -sine * (qiqpj + qpiqj) / 3.0);
    add_scaled(result.strain, kernel, -cosine * (qiqpj + qpiqj) / 3.0);
    const Vec3 quad_linear = {qi * tj_vector[0] + qj * ti_vector[0],
                              qi * tj_vector[1] + qj * ti_vector[1],
                              qi * tj_vector[2] + qj * ti_vector[2]};
    Mat3 quad_outer = outer(g, quad_linear);
    add_scaled(quad_outer, outer(quad_linear, g));
    add_scaled(result.strain, quad_outer, cosine / 3.0);
  }
  return result;
}

double multipole_radius_value(const PeriodicMultipolePlanData& data, std::size_t atom,
                              double coordination) noexcept {
  const double argument =
      parameters::gfn2::kGlobal.multipole_kexp *
      (coordination - data.multipole_valence_cn[atom] - parameters::gfn2::kGlobal.multipole_shift);
  const double fraction = argument >= 0.0 ? 1.0 / (1.0 + std::exp(-argument))
                                          : std::exp(argument) / (1.0 + std::exp(argument));
  return data.multipole_radius[atom] +
         (parameters::gfn2::kGlobal.multipole_rmax - data.multipole_radius[atom]) * fraction;
}

double multipole_radius_cn_derivative(const PeriodicMultipolePlanData& data, std::size_t atom,
                                      double coordination) noexcept {
  const double argument =
      parameters::gfn2::kGlobal.multipole_kexp *
      (coordination - data.multipole_valence_cn[atom] - parameters::gfn2::kGlobal.multipole_shift);
  const double fraction = argument >= 0.0 ? 1.0 / (1.0 + std::exp(-argument))
                                          : std::exp(argument) / (1.0 + std::exp(argument));
  /* tblite's periodic q/d/Q path publishes this signed derivative.  The
   * negative sign is retained for oracle compatibility (the underlying
   * damping-radius routine stores the derivative of its descending logistic
   * representation). */
  return -(parameters::gfn2::kGlobal.multipole_rmax - data.multipole_radius[atom]) *
         parameters::gfn2::kGlobal.multipole_kexp * fraction * (1.0 - fraction);
}

std::size_t pair_offset(const PeriodicMultipolePlanData& data, std::size_t system, std::size_t row,
                        std::size_t column) noexcept {
  const std::size_t atom_begin = static_cast<std::size_t>(data.atom_offsets[system]);
  const std::size_t local_atoms =
      static_cast<std::size_t>(data.atom_offsets[system + 1u]) - atom_begin;
  return static_cast<std::size_t>(data.matrix_offsets[system]) + row * local_atoms + column;
}

void add_matrix(const PeriodicMultipolePlanData& data, std::size_t system, std::size_t row,
                std::size_t column, const MultipoleTerms& terms, double weight, double* sd,
                double* dd, double* sq) noexcept {
  const std::size_t pair = pair_offset(data, system, row, column);
  for (std::size_t component = 0; component < 3u; ++component) {
    sd[3u * pair + component] += weight * terms.sd[component];
  }
  for (std::size_t component = 0; component < 9u; ++component) {
    dd[9u * pair + component] += weight * terms.dd[component];
  }
  for (std::size_t component = 0; component < 6u; ++component) {
    sq[6u * pair + component] += weight * terms.sq[component];
  }
}

void accumulate_potentials_and_energy(const PeriodicMultipolePlanData& data, std::size_t system,
                                      const double* charges, const double* dipoles,
                                      const double* quadrupoles, const double* sd, const double* dd,
                                      const double* sq, double* charge_potentials,
                                      double* dipole_potentials, double* quadrupole_potentials,
                                      double& energy) noexcept {
  const std::size_t atom_begin = static_cast<std::size_t>(data.atom_offsets[system]);
  const std::size_t atom_end = static_cast<std::size_t>(data.atom_offsets[system + 1u]);
  const std::size_t atom_count = atom_end - atom_begin;
  constexpr std::array<double, 6> quadrupole_scale{{1.0, 2.0, 1.0, 2.0, 2.0, 1.0}};
  /* The packed matrix follows tblite's Fortran layout: pair(row, column) is
   * A(:, jat=row, iat=column).  A non-transposed GEMV therefore publishes
   * the potential at row, while a transposed GEMV publishes the charge
   * potential at column. */
  for (std::size_t local = 0; local < atom_count; ++local) {
    const std::size_t atom = atom_begin + local;
    charge_potentials[atom] = 0.0;
    for (std::size_t component = 0; component < 3u; ++component) {
      dipole_potentials[atom * 3u + component] = 0.0;
    }
    for (std::size_t component = 0; component < 6u; ++component) {
      quadrupole_potentials[atom * 6u + component] = 0.0;
    }
  }

  for (std::size_t local_row = 0; local_row < atom_count; ++local_row) {
    const std::size_t row_atom = atom_begin + local_row;
    const Vec3 row_dipole{
        {dipoles[row_atom * 3u], dipoles[row_atom * 3u + 1u], dipoles[row_atom * 3u + 2u]}};
    const double* row_quadrupole = quadrupoles + row_atom * 6u;
    for (std::size_t local_column = 0; local_column < atom_count; ++local_column) {
      const std::size_t column_atom = atom_begin + local_column;
      const std::size_t pair = pair_offset(data, system, local_row, local_column);
      const double* charge_dipole = sd + 3u * pair;
      const double* dipole_matrix = dd + 9u * pair;
      const double* charge_quad = sq + 6u * pair;

      /* A^T*d and A^T*Q contribute to the scalar potential of the
       * source/column atom. */
      charge_potentials[column_atom] +=
          dot(Vec3{{charge_dipole[0], charge_dipole[1], charge_dipole[2]}}, row_dipole);
      charge_potentials[column_atom] +=
          packed_dot(Vec6{{charge_quad[0], charge_quad[1], charge_quad[2], charge_quad[3],
                           charge_quad[4], charge_quad[5]}},
                     row_quadrupole);

      /* A*q and A*d publish the vector potential at the matrix row. */
      for (std::size_t row_component = 0; row_component < 3u; ++row_component) {
        dipole_potentials[row_atom * 3u + row_component] +=
            charge_dipole[row_component] * charges[column_atom];
        for (std::size_t column_component = 0; column_component < 3u; ++column_component) {
          dipole_potentials[row_atom * 3u + row_component] +=
              dipole_matrix[row_component * 3u + column_component] *
              dipoles[column_atom * 3u + column_component];
        }
      }
      for (std::size_t component = 0; component < 6u; ++component) {
        quadrupole_potentials[row_atom * 6u + component] +=
            charge_quad[component] * charges[column_atom];
      }

      /* get_energy in tblite is d^T(A_sd q + 1/2 A_dd d) +
       * Q^T A_sq q.  Keep this contraction independent of the potential
       * assembly so a transpose/layout regression cannot double-count it. */
      energy += dot(row_dipole, Vec3{{charge_dipole[0], charge_dipole[1], charge_dipole[2]}}) *
                charges[column_atom];
      double dd_energy = 0.0;
      for (std::size_t row_component = 0; row_component < 3u; ++row_component) {
        for (std::size_t column_component = 0; column_component < 3u; ++column_component) {
          dd_energy += dipoles[row_atom * 3u + row_component] *
                       dipole_matrix[row_component * 3u + column_component] *
                       dipoles[column_atom * 3u + column_component];
        }
      }
      energy += 0.5 * dd_energy;
      energy += packed_dot(Vec6{{charge_quad[0], charge_quad[1], charge_quad[2], charge_quad[3],
                                 charge_quad[4], charge_quad[5]}},
                           row_quadrupole) *
                charges[column_atom];
    }
  }

  /* The AXC potentials are derivatives of the quadratic onsite energies. */
  for (std::size_t local = 0; local < atom_count; ++local) {
    const std::size_t atom = atom_begin + local;
    const double* dipole = dipoles + atom * 3u;
    const double* quadrupole = quadrupoles + atom * 6u;
    for (std::size_t component = 0; component < 3u; ++component) {
      dipole_potentials[atom * 3u + component] +=
          2.0 * data.dipole_kernel[atom] * dipole[component];
    }
    for (std::size_t component = 0; component < 6u; ++component) {
      quadrupole_potentials[atom * 6u + component] +=
          2.0 * data.quadrupole_kernel[atom] * quadrupole_scale[component] * quadrupole[component];
    }
    energy += data.dipole_kernel[atom] *
              dot(Vec3{{dipole[0], dipole[1], dipole[2]}}, Vec3{{dipole[0], dipole[1], dipole[2]}});
    double quadrupole_norm = 0.0;
    for (std::size_t component = 0; component < 6u; ++component) {
      quadrupole_norm +=
          quadrupole_scale[component] * quadrupole[component] * quadrupole[component];
    }
    energy += data.quadrupole_kernel[atom] * quadrupole_norm;
  }
}

bool validate_plan(const PeriodicMultipolePlan& plan, std::string& error) {
  if (!plan.sealed() || plan.batch_size() <= 0 || plan.total_atoms() <= 0 ||
      plan.atom_offsets().size() != static_cast<std::size_t>(plan.batch_size() + 1) ||
      plan.atom_offsets().front() != 0 || plan.atom_offsets().back() != plan.total_atoms() ||
      plan.matrix_offsets().size() != static_cast<std::size_t>(plan.batch_size() + 1) ||
      plan.lattice(0).volume <= 0.0) {
    error = "periodic multipole plan is incomplete or internally inconsistent";
    return false;
  }
  return true;
}

}  // namespace

std::int64_t PeriodicMultipolePlan::batch_size() const noexcept {
  return data_ == nullptr ? 0 : data_->batch_size;
}

std::int64_t PeriodicMultipolePlan::total_atoms() const noexcept {
  return data_ == nullptr ? 0 : data_->total_atoms;
}

std::int64_t PeriodicMultipolePlan::matrix_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->matrix_elements;
}

const std::vector<std::int64_t>& PeriodicMultipolePlan::atom_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ == nullptr ? empty : data_->atom_offsets;
}

const std::vector<std::int64_t>& PeriodicMultipolePlan::matrix_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ == nullptr ? empty : data_->matrix_offsets;
}

const Lattice3D& PeriodicMultipolePlan::lattice(std::int64_t system) const noexcept {
  static const Lattice3D empty;
  return data_ == nullptr || system < 0 || system >= data_->batch_size
             ? empty
             : data_->lattices[static_cast<std::size_t>(system)];
}

double PeriodicMultipolePlan::alpha(std::int64_t system) const noexcept {
  return data_ == nullptr || system < 0 || system >= data_->batch_size
             ? 0.0
             : data_->alphas[static_cast<std::size_t>(system)];
}

const std::vector<double>& PeriodicMultipolePlan::dipole_kernel() const noexcept {
  static const std::vector<double> empty;
  return data_ == nullptr ? empty : data_->dipole_kernel;
}

const std::vector<double>& PeriodicMultipolePlan::quadrupole_kernel() const noexcept {
  static const std::vector<double> empty;
  return data_ == nullptr ? empty : data_->quadrupole_kernel;
}

const std::vector<double>& PeriodicMultipolePlan::multipole_radius() const noexcept {
  static const std::vector<double> empty;
  return data_ == nullptr ? empty : data_->multipole_radius;
}

const std::vector<double>& PeriodicMultipolePlan::multipole_valence_cn() const noexcept {
  static const std::vector<double> empty;
  return data_ == nullptr ? empty : data_->multipole_valence_cn;
}

bool PeriodicMultipolePlan::overlaps_storage(const void* pointer,
                                             std::size_t bytes) const noexcept {
  if (bytes == 0u) return false;
  if (pointer == nullptr || data_ == nullptr) return true;
  const MemoryRange active{pointer, bytes};
  const MemoryRange object{this, sizeof(*this)};
  const MemoryRange data_range{data_.get(), sizeof(*data_)};
  if (ranges_overlap(active, object) || ranges_overlap(active, data_range)) return true;
  auto overlaps_vector = [&](const auto& values) {
    return ranges_overlap(active, {values.data(), values.capacity() * sizeof(values[0])});
  };
  return overlaps_vector(data_->atom_offsets) || overlaps_vector(data_->matrix_offsets) ||
         overlaps_vector(data_->lattices) || overlaps_vector(data_->alphas) ||
         overlaps_vector(data_->direct_translation_offsets) ||
         overlaps_vector(data_->reciprocal_translation_offsets) ||
         overlaps_vector(data_->direct_translations) ||
         overlaps_vector(data_->reciprocal_translations) || overlaps_vector(data_->dipole_kernel) ||
         overlaps_vector(data_->quadrupole_kernel) || overlaps_vector(data_->multipole_radius) ||
         overlaps_vector(data_->multipole_valence_cn);
}

xtbloom_status_t make_periodic_multipole_plan(const AES2Plan& aes2,
                                              const PeriodicShortRangePlan& topology,
                                              PeriodicMultipolePlan& plan, std::string& error) {
  if (!aes2.sealed() || !topology.sealed() || aes2.batch_size() != topology.batch_size() ||
      aes2.total_atoms() != topology.total_atoms() ||
      aes2.atom_offsets() != topology.atom_offsets()) {
    error = "periodic multipole plan dimensions do not match AES2 and periodic topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto created = std::make_shared<PeriodicMultipolePlanData>();
    created->batch_size = topology.batch_size();
    created->total_atoms = topology.total_atoms();
    created->atom_offsets = topology.atom_offsets();
    created->matrix_offsets.assign(static_cast<std::size_t>(created->batch_size + 1), 0);
    created->lattices.resize(static_cast<std::size_t>(created->batch_size));
    created->alphas.resize(static_cast<std::size_t>(created->batch_size));
    created->direct_translation_offsets.assign(static_cast<std::size_t>(created->batch_size + 1),
                                               0);
    created->reciprocal_translation_offsets.assign(
        static_cast<std::size_t>(created->batch_size + 1), 0);
    created->dipole_kernel = aes2.dipole_kernel();
    created->quadrupole_kernel = aes2.quadrupole_kernel();
    created->multipole_radius = aes2.multipole_radius();
    created->multipole_valence_cn = aes2.multipole_valence_cn();
    created->topology_identity = topology.identity();
    std::int64_t matrix_total = 0;
    for (std::size_t system = 0; system < static_cast<std::size_t>(created->batch_size); ++system) {
      created->lattices[system] = topology.lattice(static_cast<std::int64_t>(system));
      if (!finite_lattice(created->lattices[system])) {
        error = "periodic multipole topology contains an invalid lattice";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const std::int64_t atom_count =
          created->atom_offsets[system + 1u] - created->atom_offsets[system];
      if (atom_count < 0 ||
          (atom_count > 0 && atom_count > std::numeric_limits<std::int64_t>::max() / atom_count)) {
        error = "periodic multipole matrix dimensions overflow";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      matrix_total += atom_count * atom_count;
      created->matrix_offsets[system + 1u] = matrix_total;
      const double alpha = select_alpha(created->lattices[system]);
      created->alphas[system] = alpha;
      std::vector<LatticeTranslation> direct;
      std::vector<LatticeTranslation> reciprocal;
      xtbloom_status_t status =
          make_lattice_translations(created->lattices[system], kMultipoleCutoff,
                                    LatticeOriginPolicy::kInclude, direct, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      Lattice3D reciprocal_lattice;
      status =
          make_lattice_3d(created->lattices[system].reciprocal.data(), reciprocal_lattice, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      const double reciprocal_cutoff =
          search_reciprocal_cutoff(alpha, created->lattices[system].volume);
      status = make_lattice_translations(reciprocal_lattice, reciprocal_cutoff,
                                         LatticeOriginPolicy::kExclude, reciprocal, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      created->direct_translation_offsets[system + 1u] =
          created->direct_translation_offsets[system] + static_cast<std::int64_t>(direct.size());
      created->reciprocal_translation_offsets[system + 1u] =
          created->reciprocal_translation_offsets[system] +
          static_cast<std::int64_t>(reciprocal.size());
      created->direct_translations.insert(created->direct_translations.end(), direct.begin(),
                                          direct.end());
      created->reciprocal_translations.insert(created->reciprocal_translations.end(),
                                              reciprocal.begin(), reciprocal.end());
    }
    created->matrix_elements = matrix_total;
    plan = PeriodicMultipolePlan(std::move(created));
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate periodic multipole plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t evaluate_periodic_multipole_cpu(
    const PeriodicMultipolePlan& plan, const double* positions, const double* coordination_numbers,
    const double* atomic_charges, const double* atomic_dipoles, const double* atomic_quadrupoles,
    double* charge_dipole_matrix, double* dipole_dipole_matrix, double* charge_quadrupole_matrix,
    double* charge_potentials, double* dipole_potentials, double* quadrupole_potentials,
    double* energies, double* gradients, double* strain_derivatives, double* coordination_adjoint,
    std::string& error) {
  if (!validate_plan(plan, error)) return XTBLOOM_STATUS_INVALID_ARGUMENT;
  const std::size_t atom_count = static_cast<std::size_t>(plan.total_atoms());
  const std::size_t batch_count = static_cast<std::size_t>(plan.batch_size());
  const std::size_t matrix_count = static_cast<std::size_t>(plan.matrix_elements());
  const std::size_t coordinate_count = atom_count * 3u;
  const std::size_t quadrupole_count = atom_count * 6u;
  if (!aligned_double(positions) || !aligned_double(coordination_numbers) ||
      !aligned_double(atomic_charges) || !aligned_double(atomic_dipoles) ||
      !aligned_double(atomic_quadrupoles) || !aligned_double(charge_dipole_matrix) ||
      !aligned_double(dipole_dipole_matrix) || !aligned_double(charge_quadrupole_matrix) ||
      !aligned_double(charge_potentials) || !aligned_double(dipole_potentials) ||
      !aligned_double(quadrupole_potentials) || !aligned_double(energies) ||
      !aligned_double(gradients) || !aligned_double(strain_derivatives) ||
      (coordination_adjoint != nullptr && !aligned_double(coordination_adjoint)) ||
      !finite_array(positions, coordinate_count) ||
      !finite_array(coordination_numbers, atom_count) ||
      !finite_array(atomic_charges, atom_count) ||
      !finite_array(atomic_dipoles, coordinate_count) ||
      !finite_array(atomic_quadrupoles, quadrupole_count)) {
    error = "periodic multipole inputs are malformed or nonfinite";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::vector<MemoryRange> ranges{{positions, coordinate_count * sizeof(double)},
                                  {coordination_numbers, atom_count * sizeof(double)},
                                  {atomic_charges, atom_count * sizeof(double)},
                                  {atomic_dipoles, coordinate_count * sizeof(double)},
                                  {atomic_quadrupoles, quadrupole_count * sizeof(double)},
                                  {charge_dipole_matrix, matrix_count * 3u * sizeof(double)},
                                  {dipole_dipole_matrix, matrix_count * 9u * sizeof(double)},
                                  {charge_quadrupole_matrix, matrix_count * 6u * sizeof(double)},
                                  {charge_potentials, atom_count * sizeof(double)},
                                  {dipole_potentials, coordinate_count * sizeof(double)},
                                  {quadrupole_potentials, quadrupole_count * sizeof(double)},
                                  {energies, batch_count * sizeof(double)},
                                  {gradients, coordinate_count * sizeof(double)},
                                  {strain_derivatives, batch_count * 9u * sizeof(double)}};
  if (coordination_adjoint != nullptr) {
    ranges.push_back({coordination_adjoint, atom_count * sizeof(double)});
  }
  if (!all_disjoint(ranges)) {
    error = "periodic multipole inputs and outputs must not overlap";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (const MemoryRange& range : ranges) {
    if (plan.overlaps_storage(range.data, range.bytes)) {
      error = "periodic multipole buffers overlap immutable plan storage";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  try {
    std::vector<double> staged_sd(matrix_count * 3u, 0.0);
    std::vector<double> staged_dd(matrix_count * 9u, 0.0);
    std::vector<double> staged_sq(matrix_count * 6u, 0.0);
    std::vector<double> staged_charge(atom_count, 0.0);
    std::vector<double> staged_dipole(coordinate_count, 0.0);
    std::vector<double> staged_quadrupole(quadrupole_count, 0.0);
    std::vector<double> staged_energy(batch_count, 0.0);
    std::vector<double> staged_gradient(coordinate_count, 0.0);
    std::vector<double> staged_strain(batch_count * 9u, 0.0);
    std::vector<double> staged_adjoint(atom_count, 0.0);

    for (std::size_t system = 0; system < batch_count; ++system) {
      const std::size_t atom_begin = static_cast<std::size_t>(plan.atom_offsets()[system]);
      const std::size_t atom_end = static_cast<std::size_t>(plan.atom_offsets()[system + 1u]);
      const std::size_t local_atoms = atom_end - atom_begin;
      const double alpha = plan.alpha(static_cast<std::int64_t>(system));
      const std::int64_t direct_begin = plan.data_->direct_translation_offsets[system];
      const std::int64_t direct_end = plan.data_->direct_translation_offsets[system + 1u];
      const std::int64_t reciprocal_begin = plan.data_->reciprocal_translation_offsets[system];
      const std::int64_t reciprocal_end = plan.data_->reciprocal_translation_offsets[system + 1u];
      const std::vector<LatticeTranslation> direct(
          plan.data_->direct_translations.begin() + direct_begin,
          plan.data_->direct_translations.begin() + direct_end);
      const std::vector<LatticeTranslation> reciprocal(
          plan.data_->reciprocal_translations.begin() + reciprocal_begin,
          plan.data_->reciprocal_translations.begin() + reciprocal_end);
      std::vector<Vec3> wrapped(local_atoms);
      std::string local_error;
      for (std::size_t local = 0; local < local_atoms; ++local) {
        const xtbloom_status_t status = wrap_cartesian(
            plan.lattice(static_cast<std::int64_t>(system)), positions + (atom_begin + local) * 3u,
            wrapped[local].data(), local_error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          error = local_error;
          return status;
        }
      }
      std::vector<std::vector<WscImage>> pair_images(local_atoms * local_atoms);
      for (std::size_t center = 0; center < local_atoms; ++center) {
        for (std::size_t image = 0; image < center; ++image) {
          if (!build_wsc_images(plan.lattice(static_cast<std::int64_t>(system)),
                                subtract(wrapped[center], wrapped[image]), false,
                                pair_images[center * local_atoms + image], local_error)) {
            error = local_error;
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
        }
        if (!build_wsc_images(plan.lattice(static_cast<std::int64_t>(system)), Vec3{}, true,
                              pair_images[center * local_atoms + center], local_error)) {
          error = local_error;
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }

      for (std::size_t center = 0; center < local_atoms; ++center) {
        for (std::size_t image = 0; image < center; ++image) {
          const std::size_t first = atom_begin + center;
          const std::size_t second = atom_begin + image;
          const double radius =
              0.5 * (multipole_radius_value(*plan.data_, first, coordination_numbers[first]) +
                     multipole_radius_value(*plan.data_, second, coordination_numbers[second]));
          for (const WscImage& wsc : pair_images[center * local_atoms + image]) {
            const Vec3 vector = wsc.vector;
            const MultipoleTerms forward = matrix_terms_with_volume(
                vector, radius, alpha, plan.lattice(static_cast<std::int64_t>(system)).volume,
                direct, reciprocal);
            const MultipoleTerms reverse = matrix_terms_with_volume(
                negate(vector), radius, alpha,
                plan.lattice(static_cast<std::int64_t>(system)).volume, direct, reciprocal);
            add_matrix(*plan.data_, system, image, center, forward, wsc.weight, staged_sd.data(),
                       staged_dd.data(), staged_sq.data());
            add_matrix(*plan.data_, system, center, image, reverse, wsc.weight, staged_sd.data(),
                       staged_dd.data(), staged_sq.data());

            const Vec3 derivative_vector = negate(vector);
            const Vec3 mi{atomic_dipoles[first * 3u], atomic_dipoles[first * 3u + 1u],
                          atomic_dipoles[first * 3u + 2u]};
            const Vec3 mj{atomic_dipoles[second * 3u], atomic_dipoles[second * 3u + 1u],
                          atomic_dipoles[second * 3u + 2u]};
            const MultipoleDerivatives derivative = derivative_terms(
                derivative_vector, atomic_charges[first], atomic_charges[second], mi, mj,
                atomic_quadrupoles + first * 6u, atomic_quadrupoles + second * 6u, radius, alpha,
                direct, reciprocal, plan.lattice(static_cast<std::int64_t>(system)).volume);
            const MultipoleTerms energy_terms = matrix_terms_with_volume(
                derivative_vector, radius, alpha,
                plan.lattice(static_cast<std::int64_t>(system)).volume, direct, reciprocal);
            const double image_energy = pair_energy(
                atomic_charges[first], atomic_charges[second], mi.data(), mj.data(),
                atomic_quadrupoles + first * 6u, atomic_quadrupoles + second * 6u, energy_terms);
            Vec3 weighted_gradient = scale(derivative.gradient, wsc.weight);
            add_scaled(weighted_gradient, wsc.weight_gradient, image_energy);
            for (std::size_t axis = 0; axis < 3u; ++axis) {
              staged_gradient[first * 3u + axis] += weighted_gradient[axis];
              staged_gradient[second * 3u + axis] -= weighted_gradient[axis];
            }
            for (std::size_t component = 0; component < 9u; ++component) {
              staged_strain[system * 9u + component] += wsc.weight * derivative.strain[component] +
                                                        image_energy * wsc.weight_strain[component];
            }
            /* The ordered matrix contains both transpose directions, whereas
             * get_damat_sdq_* contracts the unordered image expression once.
             * The radius VJP therefore receives the two equal ordered
             * contributions before the per-atom average radius chain rule. */
            staged_adjoint[first] += wsc.weight * derivative.radius;
            staged_adjoint[second] += wsc.weight * derivative.radius;
          }
        }

        const std::size_t atom = atom_begin + center;
        const double radius = multipole_radius_value(*plan.data_, atom, coordination_numbers[atom]);
        const Vec3 mi{atomic_dipoles[atom * 3u], atomic_dipoles[atom * 3u + 1u],
                      atomic_dipoles[atom * 3u + 2u]};
        for (const WscImage& wsc : pair_images[center * local_atoms + center]) {
          const MultipoleTerms matrix = matrix_terms_with_volume(
              wsc.vector, radius, alpha, plan.lattice(static_cast<std::int64_t>(system)).volume,
              direct, reciprocal);
          add_matrix(*plan.data_, system, center, center, matrix, wsc.weight, staged_sd.data(),
                     staged_dd.data(), staged_sq.data());
          const Vec3 derivative_vector = negate(wsc.vector);
          const MultipoleDerivatives derivative = derivative_terms(
              derivative_vector, atomic_charges[atom], atomic_charges[atom], mi, mi,
              atomic_quadrupoles + atom * 6u, atomic_quadrupoles + atom * 6u, radius, alpha, direct,
              reciprocal, plan.lattice(static_cast<std::int64_t>(system)).volume);
          const MultipoleTerms energy_terms = matrix_terms_with_volume(
              derivative_vector, radius, alpha,
              plan.lattice(static_cast<std::int64_t>(system)).volume, direct, reciprocal);
          const double image_energy = pair_energy(
              atomic_charges[atom], atomic_charges[atom], mi.data(), mi.data(),
              atomic_quadrupoles + atom * 6u, atomic_quadrupoles + atom * 6u, energy_terms);
          for (std::size_t component = 0; component < 9u; ++component) {
            staged_strain[system * 9u + component] +=
                0.5 * (wsc.weight * derivative.strain[component] +
                       image_energy * wsc.weight_strain[component]);
          }
          staged_adjoint[atom] += wsc.weight * derivative.radius;
        }
      }

      const double self_dd = -4.0 * alpha * alpha * alpha / (3.0 * kSqrtPi);
      const double self_sq = 4.0 * alpha * alpha * alpha / (9.0 * kSqrtPi);
      for (std::size_t local = 0; local < local_atoms; ++local) {
        const std::size_t pair = pair_offset(*plan.data_, system, local, local);
        staged_dd[9u * pair] += self_dd;
        staged_dd[9u * pair + 4u] += self_dd;
        staged_dd[9u * pair + 8u] += self_dd;
        staged_sq[6u * pair] += self_sq;
        staged_sq[6u * pair + 2u] += self_sq;
        staged_sq[6u * pair + 5u] += self_sq;
      }

      accumulate_potentials_and_energy(*plan.data_, system, atomic_charges, atomic_dipoles,
                                       atomic_quadrupoles, staged_sd.data(), staged_dd.data(),
                                       staged_sq.data(), staged_charge.data(), staged_dipole.data(),
                                       staged_quadrupole.data(), staged_energy[system]);
      for (std::size_t local = 0; local < local_atoms; ++local) {
        const std::size_t atom = atom_begin + local;
        staged_adjoint[atom] *=
            multipole_radius_cn_derivative(*plan.data_, atom, coordination_numbers[atom]);
      }
    }

    if (!finite_matrix(staged_sd.data(), staged_sd.size()) ||
        !finite_matrix(staged_dd.data(), staged_dd.size()) ||
        !finite_matrix(staged_sq.data(), staged_sq.size()) ||
        !finite_array(staged_charge.data(), staged_charge.size()) ||
        !finite_array(staged_dipole.data(), staged_dipole.size()) ||
        !finite_array(staged_quadrupole.data(), staged_quadrupole.size()) ||
        !finite_array(staged_energy.data(), staged_energy.size()) ||
        !finite_array(staged_gradient.data(), staged_gradient.size()) ||
        !finite_array(staged_strain.data(), staged_strain.size()) ||
        !finite_array(staged_adjoint.data(), staged_adjoint.size())) {
      error = "periodic multipole arithmetic exceeded floating-point range";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    std::copy(staged_sd.begin(), staged_sd.end(), charge_dipole_matrix);
    std::copy(staged_dd.begin(), staged_dd.end(), dipole_dipole_matrix);
    std::copy(staged_sq.begin(), staged_sq.end(), charge_quadrupole_matrix);
    std::copy(staged_charge.begin(), staged_charge.end(), charge_potentials);
    std::copy(staged_dipole.begin(), staged_dipole.end(), dipole_potentials);
    std::copy(staged_quadrupole.begin(), staged_quadrupole.end(), quadrupole_potentials);
    std::copy(staged_energy.begin(), staged_energy.end(), energies);
    std::copy(staged_gradient.begin(), staged_gradient.end(), gradients);
    std::copy(staged_strain.begin(), staged_strain.end(), strain_derivatives);
    if (coordination_adjoint != nullptr) {
      std::copy(staged_adjoint.begin(), staged_adjoint.end(), coordination_adjoint);
    }
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate periodic multipole evaluation scratch";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

}  // namespace xtbloom::detail::gfn2
