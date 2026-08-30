// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/periodic_ewald.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace xtbloom::detail::gfn2 {

struct PeriodicEwaldPlanData final {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_matrix_elements = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<double> shell_hardness;
  std::vector<double> alphas;
  std::vector<double> direct_cutoffs;
  std::vector<double> reciprocal_cutoffs;
  std::vector<std::int64_t> direct_translation_offsets;
  std::vector<std::int64_t> reciprocal_translation_offsets;
  std::vector<LatticeTranslation> direct_translations;
  std::vector<LatticeTranslation> reciprocal_translations;
  const PeriodicShortRangePlanData* topology_identity = nullptr;
};

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrtPi = 1.7724538509055160272981674833411451828;
constexpr double kWignerToleranceSquared = 0.3;
constexpr double kWignerThreshold = 1.4901161193847656e-8;
constexpr double kBinary64Epsilon = std::numeric_limits<double>::epsilon();
constexpr double kAlphaTolerance = std::sqrt(kBinary64Epsilon);

struct WscImage {
  std::array<double, 3> vector{};
  double weight = 0.0;
  std::array<double, 3> weight_gradient{};
  std::array<double, 9> weight_strain{};
};

std::array<double, 3> direct_derivative(const std::array<double, 3>& vector, double alpha);
std::array<double, 9> outer(const std::array<double, 3>& lhs, const std::array<double, 3>& rhs);
void add_inplace(std::array<double, 3>& target, const std::array<double, 3>& value,
                 double scale = 1.0);
void add_inplace(std::array<double, 9>& target, const std::array<double, 9>& value,
                 double scale = 1.0);

bool finite_array(const double* data, std::size_t count) {
  if (data == nullptr) return false;
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(data[i])) return false;
  }
  return true;
}

double norm(const std::array<double, 3>& value) {
  return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
}

double dot(const std::array<double, 3>& lhs, const std::array<double, 3>& rhs) {
  return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

std::array<double, 3> subtract(const std::array<double, 3>& lhs, const std::array<double, 3>& rhs) {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

std::array<double, 3> lattice_vector(const LatticeTranslation& value) { return value.cartesian; }

bool is_origin(const LatticeTranslation& value) {
  return value.index[0] == 0 && value.index[1] == 0 && value.index[2] == 0;
}

double smooth_shape(double delta) {
  const double x = std::min(1.0, std::max(0.0, delta) / kWignerToleranceSquared);
  return std::max(0.0, 1.0 - 10.0 * x * x * x + 15.0 * x * x * x * x - 6.0 * x * x * x * x * x);
}

double smooth_shape_derivative(double delta) {
  const double x = std::min(1.0, std::max(0.0, delta) / kWignerToleranceSquared);
  return -30.0 * x * x * (1.0 - x) * (1.0 - x) / kWignerToleranceSquared;
}

bool valid_dimensions(const ES2Plan& es2, const PeriodicShortRangePlan& topology,
                      std::string& error) {
  if (!es2.sealed() || !topology.sealed() || es2.batch_size() <= 0 ||
      es2.batch_size() != topology.batch_size() || es2.total_atoms() != topology.total_atoms() ||
      es2.atom_offsets() != topology.atom_offsets()) {
    error = "periodic Ewald plan dimensions do not match ES2 and topology plans";
    return false;
  }
  return true;
}

double ewald_decay_direct(double distance, double alpha) {
  return std::erfc(alpha * distance) / distance;
}

double ewald_decay_reciprocal(double distance, double alpha, double volume) {
  return 4.0 * kPi * std::exp(-0.25 * distance * distance / (alpha * alpha)) /
         (volume * distance * distance);
}

double select_alpha(const Lattice3D& lattice) {
  double direct_min = std::numeric_limits<double>::infinity();
  double reciprocal_min = std::numeric_limits<double>::infinity();
  for (std::size_t row = 0; row < 3u; ++row) {
    std::array<double, 3> direct{};
    std::array<double, 3> reciprocal{};
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      direct[axis] = lattice.direct[row * 3u + axis];
      reciprocal[axis] = lattice.reciprocal[row * 3u + axis];
    }
    direct_min = std::min(direct_min, norm(direct));
    reciprocal_min = std::min(reciprocal_min, norm(reciprocal));
  }
  auto difference = [&](double alpha) {
    return (ewald_decay_reciprocal(4.0 * reciprocal_min, alpha, lattice.volume) -
            ewald_decay_reciprocal(5.0 * reciprocal_min, alpha, lattice.volume)) -
           (ewald_decay_direct(2.0 * direct_min, alpha) -
            ewald_decay_direct(3.0 * direct_min, alpha));
  };
  double alpha = kAlphaTolerance;
  double diff = difference(alpha);
  while (diff < -kAlphaTolerance && std::isfinite(alpha)) {
    alpha *= 2.0;
    diff = difference(alpha);
  }
  if (!std::isfinite(alpha) || alpha == kAlphaTolerance) return 0.25;
  double left = 0.5 * alpha;
  while (diff < kAlphaTolerance && std::isfinite(alpha)) {
    alpha *= 2.0;
    diff = difference(alpha);
  }
  if (!std::isfinite(alpha)) return 0.25;
  double right = alpha;
  alpha = 0.5 * (left + right);
  diff = difference(alpha);
  int iterations = 0;
  while (std::abs(diff) > kAlphaTolerance && iterations <= 30) {
    if (diff < 0.0) {
      left = alpha;
    } else {
      right = alpha;
    }
    alpha = 0.5 * (left + right);
    diff = difference(alpha);
    ++iterations;
  }
  return iterations > 30 ? 0.25 : alpha;
}

double search_cutoff_direct(double alpha) {
  double cutoff = kAlphaTolerance;
  double value = ewald_decay_direct(cutoff, alpha);
  while (value > kBinary64Epsilon && std::isfinite(cutoff)) {
    cutoff *= 2.0;
    value = ewald_decay_direct(cutoff, alpha);
  }
  double left = 0.5 * cutoff;
  double left_value = ewald_decay_direct(left, alpha);
  double right = cutoff;
  double right_value = value;
  for (int iteration = 0; iteration < 30; ++iteration) {
    if (left_value - right_value <= kBinary64Epsilon) break;
    cutoff = 0.5 * (left + right);
    value = ewald_decay_direct(cutoff, alpha);
    if (value >= kBinary64Epsilon) {
      left = cutoff;
      left_value = value;
    } else {
      right = cutoff;
      right_value = value;
    }
  }
  return cutoff;
}

double search_cutoff_reciprocal(double alpha, double volume) {
  double cutoff = kAlphaTolerance;
  double value = ewald_decay_reciprocal(cutoff, alpha, volume);
  while (value > kBinary64Epsilon && std::isfinite(cutoff)) {
    cutoff *= 2.0;
    value = ewald_decay_reciprocal(cutoff, alpha, volume);
  }
  double left = 0.5 * cutoff;
  double left_value = ewald_decay_reciprocal(left, alpha, volume);
  double right = cutoff;
  double right_value = value;
  for (int iteration = 0; iteration < 30; ++iteration) {
    if (left_value - right_value <= kBinary64Epsilon) break;
    cutoff = 0.5 * (left + right);
    value = ewald_decay_reciprocal(cutoff, alpha, volume);
    if (value >= kBinary64Epsilon) {
      left = cutoff;
      left_value = value;
    } else {
      right = cutoff;
      right_value = value;
    }
  }
  return cutoff;
}

double direct_sum(const std::array<double, 3>& rij,
                  const std::vector<LatticeTranslation>& translations, double alpha) {
  double result = 0.0;
  for (const auto& translation : translations) {
    const auto vector = subtract(rij, lattice_vector(translation));
    const double distance = norm(vector);
    if (distance < kWignerThreshold) continue;
    result += ewald_decay_direct(distance, alpha);
  }
  return result;
}

std::array<double, 3> direct_derivative_sum(const std::array<double, 3>& rij,
                                            const std::vector<LatticeTranslation>& translations,
                                            double alpha) {
  std::array<double, 3> result{};
  for (const auto& translation : translations) {
    add_inplace(result, direct_derivative(subtract(rij, lattice_vector(translation)), alpha));
  }
  return result;
}

std::array<double, 9> direct_strain_sum(const std::array<double, 3>& rij,
                                        const std::vector<LatticeTranslation>& translations,
                                        double alpha) {
  std::array<double, 9> result{};
  for (const auto& translation : translations) {
    const auto vector = subtract(rij, lattice_vector(translation));
    add_inplace(result, outer(direct_derivative(vector, alpha), vector));
  }
  return result;
}

double reciprocal_sum(const std::array<double, 3>& rij,
                      const std::vector<LatticeTranslation>& translations, double alpha,
                      double volume) {
  double result = 0.0;
  for (const auto& translation : translations) {
    const auto vector = lattice_vector(translation);
    const double g2 = dot(vector, vector);
    if (g2 < kBinary64Epsilon) continue;
    result += 4.0 * kPi / volume * std::exp(-0.25 * g2 / (alpha * alpha)) *
              std::cos(dot(rij, vector)) / g2;
  }
  return result;
}

bool build_wsc_images(const Lattice3D& lattice, const std::array<double, 3>& rij, bool self,
                      std::vector<WscImage>& images, std::string& error) {
  std::vector<LatticeTranslation> translations;
  const xtbloom_status_t status = make_lattice_translations(
      lattice, kWignerThreshold, LatticeOriginPolicy::kInclude, translations, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return false;

  double minimum = std::numeric_limits<double>::infinity();
  struct Candidate {
    std::array<double, 3> vector{};
    double squared = 0.0;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(translations.size());
  for (const auto& translation : translations) {
    if (self && is_origin(translation)) continue;
    const auto vector = subtract(rij, lattice_vector(translation));
    const double squared = dot(vector, vector);
    if (!(squared >= kBinary64Epsilon) || !std::isfinite(squared)) continue;
    candidates.push_back({vector, squared});
    minimum = std::min(minimum, squared);
  }
  if (!std::isfinite(minimum)) {
    error = "periodic Ewald Wigner-Seitz image search found no image";
    return false;
  }

  double shape_sum = 0.0;
  for (const auto& candidate : candidates) {
    shape_sum += smooth_shape(std::max(0.0, candidate.squared - minimum));
  }
  if (!(shape_sum > 0.0) || !std::isfinite(shape_sum)) {
    error = "periodic Ewald Wigner-Seitz image weights are invalid";
    return false;
  }
  std::array<double, 3> sum_gradient{};
  std::array<double, 9> sum_strain{};
  std::vector<double> shapes;
  std::vector<std::array<double, 3>> shape_gradients;
  std::vector<std::array<double, 9>> shape_strains;
  shapes.reserve(candidates.size());
  shape_gradients.reserve(candidates.size());
  shape_strains.reserve(candidates.size());
  const auto reference = std::find_if(candidates.begin(), candidates.end(),
                                      [&](const Candidate& c) { return c.squared == minimum; });
  for (const auto& candidate : candidates) {
    const double delta = std::max(0.0, candidate.squared - minimum);
    const double derivative = smooth_shape_derivative(delta);
    const auto difference = subtract(candidate.vector, reference->vector);
    std::array<double, 3> gradient{};
    std::array<double, 9> strain{};
    for (std::size_t axis = 0; axis < 3u; ++axis) gradient[axis] = 2.0 * difference[axis];
    for (std::size_t row = 0; row < 3u; ++row) {
      for (std::size_t column = 0; column < 3u; ++column) {
        strain[row * 3u + column] = 2.0 * (candidate.vector[row] * candidate.vector[column] -
                                           reference->vector[row] * reference->vector[column]);
      }
    }
    const double shape = smooth_shape(delta);
    shape_gradients.push_back(
        {derivative * gradient[0], derivative * gradient[1], derivative * gradient[2]});
    shape_strains.push_back({});
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      shape_gradients.back()[axis] = derivative * gradient[axis];
      sum_gradient[axis] += shape_gradients.back()[axis];
    }
    for (std::size_t index = 0; index < 9u; ++index) {
      shape_strains.back()[index] = derivative * strain[index];
      sum_strain[index] += shape_strains.back()[index];
    }
    shapes.push_back(shape);
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
  return !images.empty();
}

std::array<double, 3> direct_derivative(const std::array<double, 3>& vector, double alpha) {
  const double distance = norm(vector);
  if (distance < kWignerThreshold) return {};
  const double squared = distance * distance;
  const double coefficient = -std::erfc(alpha * distance) / (squared * distance) -
                             2.0 * alpha * std::exp(-squared * alpha * alpha) / (kSqrtPi * squared);
  return {coefficient * vector[0], coefficient * vector[1], coefficient * vector[2]};
}

std::array<double, 9> outer(const std::array<double, 3>& lhs, const std::array<double, 3>& rhs) {
  std::array<double, 9> result{};
  for (std::size_t row = 0; row < 3u; ++row) {
    for (std::size_t column = 0; column < 3u; ++column) {
      result[row * 3u + column] = lhs[row] * rhs[column];
    }
  }
  return result;
}

void add_inplace(std::array<double, 3>& target, const std::array<double, 3>& value, double scale) {
  for (std::size_t axis = 0; axis < 3u; ++axis) target[axis] += scale * value[axis];
}

void add_inplace(std::array<double, 9>& target, const std::array<double, 9>& value, double scale) {
  for (std::size_t index = 0; index < 9u; ++index) target[index] += scale * value[index];
}

void add_flat_vector(double* target, std::size_t atom, const std::array<double, 3>& value,
                     double scale = 1.0) {
  for (std::size_t axis = 0; axis < 3u; ++axis) target[atom * 3u + axis] += scale * value[axis];
}

void add_flat_tensor(double* target, std::size_t system, const std::array<double, 9>& value,
                     double scale = 1.0) {
  for (std::size_t index = 0; index < 9u; ++index)
    target[system * 9u + index] += scale * value[index];
}

std::array<double, 3> reciprocal_derivative(const std::array<double, 3>& rij,
                                            const std::vector<LatticeTranslation>& translations,
                                            double alpha, double volume) {
  std::array<double, 3> result{};
  const double alpha_squared = alpha * alpha;
  for (const auto& translation : translations) {
    const auto vector = lattice_vector(translation);
    const double g2 = dot(vector, vector);
    if (g2 < kBinary64Epsilon) continue;
    const double exponential = 4.0 * kPi / volume * std::exp(-0.25 * g2 / alpha_squared) / g2;
    const double sine = std::sin(dot(rij, vector));
    add_inplace(result, vector, -sine * exponential);
  }
  return result;
}

std::array<double, 9> reciprocal_strain(const std::array<double, 3>& rij,
                                        const std::vector<LatticeTranslation>& translations,
                                        double alpha, double volume) {
  std::array<double, 9> result{};
  const double alpha_squared = alpha * alpha;
  for (const auto& translation : translations) {
    const auto vector = lattice_vector(translation);
    const double g2 = dot(vector, vector);
    if (g2 < kBinary64Epsilon) continue;
    const double exponential = 4.0 * kPi / volume * std::exp(-0.25 * g2 / alpha_squared) / g2;
    const double cosine = std::cos(dot(rij, vector)) * exponential;
    auto term = outer(vector, vector);
    for (std::size_t index = 0; index < 9u; ++index) {
      term[index] *= (2.0 / g2 + 0.5 / alpha_squared);
      if (index == 0u || index == 4u || index == 8u) term[index] -= 1.0;
    }
    add_inplace(result, term, cosine);
  }
  return result;
}

std::array<double, 3> correction_derivative(const std::array<double, 3>& vector, double gamma) {
  const double squared = dot(vector, vector);
  const double distance = std::sqrt(squared);
  if (distance < kWignerThreshold) return {};
  const double inverse_gamma_squared = 1.0 / (gamma * gamma);
  const double damped = -1.0 / std::pow(squared + inverse_gamma_squared, 1.5);
  const double bare = 1.0 / (squared * distance);
  return {(damped + bare) * vector[0], (damped + bare) * vector[1], (damped + bare) * vector[2]};
}

double correction_value(const std::array<double, 3>& vector, double gamma) {
  const double distance = norm(vector);
  if (distance < kWignerThreshold) return 0.0;
  return 1.0 / std::sqrt(distance * distance + 1.0 / (gamma * gamma)) - 1.0 / distance;
}

std::size_t matrix_index(const PeriodicEwaldPlanData& plan, std::size_t system, std::size_t row,
                         std::size_t column) {
  const std::size_t shell_count = static_cast<std::size_t>(plan.batch_shell_offsets[system + 1u] -
                                                           plan.batch_shell_offsets[system]);
  return static_cast<std::size_t>(plan.matrix_offsets[system]) + row * shell_count + column;
}

}  // namespace

std::int64_t PeriodicEwaldPlan::batch_size() const noexcept {
  return data_ == nullptr ? 0 : data_->batch_size;
}

std::int64_t PeriodicEwaldPlan::total_atoms() const noexcept {
  return data_ == nullptr ? 0 : data_->total_atoms;
}

std::int64_t PeriodicEwaldPlan::total_shells() const noexcept {
  return data_ == nullptr ? 0 : data_->total_shells;
}

std::int64_t PeriodicEwaldPlan::total_matrix_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->total_matrix_elements;
}

const std::vector<std::int64_t>& PeriodicEwaldPlan::atom_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ == nullptr ? empty : data_->atom_offsets;
}

const std::vector<std::int64_t>& PeriodicEwaldPlan::batch_shell_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ == nullptr ? empty : data_->batch_shell_offsets;
}

const std::vector<std::int64_t>& PeriodicEwaldPlan::matrix_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ == nullptr ? empty : data_->matrix_offsets;
}

double PeriodicEwaldPlan::alpha(std::int64_t system) const noexcept {
  return data_ == nullptr || system < 0 || system >= data_->batch_size
             ? 0.0
             : data_->alphas[static_cast<std::size_t>(system)];
}

double PeriodicEwaldPlan::direct_cutoff(std::int64_t system) const noexcept {
  return data_ == nullptr || system < 0 || system >= data_->batch_size
             ? 0.0
             : data_->direct_cutoffs[static_cast<std::size_t>(system)];
}

double PeriodicEwaldPlan::reciprocal_cutoff(std::int64_t system) const noexcept {
  return data_ == nullptr || system < 0 || system >= data_->batch_size
             ? 0.0
             : data_->reciprocal_cutoffs[static_cast<std::size_t>(system)];
}

bool PeriodicEwaldPlan::overlaps_storage(const void* pointer, std::size_t bytes) const noexcept {
  if (pointer == nullptr && bytes != 0u) return true;
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return true;
  const auto end = begin + bytes;
  auto overlaps = [&](const void* data, std::size_t size) {
    if (data == nullptr || size == 0u) return false;
    const auto candidate = reinterpret_cast<std::uintptr_t>(data);
    if (candidate > std::numeric_limits<std::uintptr_t>::max() - size) return true;
    return begin < candidate + size && candidate < end;
  };
  if (overlaps(this, sizeof(*this)) || overlaps(data_.get(), sizeof(PeriodicEwaldPlanData))) {
    return true;
  }
  if (data_ == nullptr) return false;
  return overlaps(data_->atom_offsets.data(),
                  data_->atom_offsets.capacity() * sizeof(std::int64_t)) ||
         overlaps(data_->batch_shell_offsets.data(),
                  data_->batch_shell_offsets.capacity() * sizeof(std::int64_t)) ||
         overlaps(data_->atom_shell_offsets.data(),
                  data_->atom_shell_offsets.capacity() * sizeof(std::int64_t)) ||
         overlaps(data_->matrix_offsets.data(),
                  data_->matrix_offsets.capacity() * sizeof(std::int64_t)) ||
         overlaps(data_->shell_hardness.data(),
                  data_->shell_hardness.capacity() * sizeof(double)) ||
         overlaps(data_->alphas.data(), data_->alphas.capacity() * sizeof(double)) ||
         overlaps(data_->direct_cutoffs.data(),
                  data_->direct_cutoffs.capacity() * sizeof(double)) ||
         overlaps(data_->reciprocal_cutoffs.data(),
                  data_->reciprocal_cutoffs.capacity() * sizeof(double)) ||
         overlaps(data_->direct_translation_offsets.data(),
                  data_->direct_translation_offsets.capacity() * sizeof(std::int64_t)) ||
         overlaps(data_->reciprocal_translation_offsets.data(),
                  data_->reciprocal_translation_offsets.capacity() * sizeof(std::int64_t)) ||
         overlaps(data_->direct_translations.data(),
                  data_->direct_translations.capacity() * sizeof(LatticeTranslation)) ||
         overlaps(data_->reciprocal_translations.data(),
                  data_->reciprocal_translations.capacity() * sizeof(LatticeTranslation));
}

xtbloom_status_t make_periodic_ewald_plan(const ES2Plan& es2,
                                          const PeriodicShortRangePlan& topology,
                                          PeriodicEwaldPlan& plan, std::string& error) {
  if (!valid_dimensions(es2, topology, error)) return XTBLOOM_STATUS_INVALID_ARGUMENT;
  try {
    PeriodicEwaldPlanData created;
    created.batch_size = es2.batch_size();
    created.total_atoms = es2.total_atoms();
    created.total_shells = es2.total_shells();
    created.total_matrix_elements = es2.total_matrix_elements();
    created.atom_offsets = es2.atom_offsets();
    created.batch_shell_offsets = es2.batch_shell_offsets();
    created.atom_shell_offsets = es2.atom_shell_offsets();
    created.matrix_offsets = es2.matrix_offsets();
    created.shell_hardness = es2.shell_hardness();
    created.topology_identity = topology.identity();
    const std::size_t batch_count = static_cast<std::size_t>(created.batch_size);
    created.alphas.resize(batch_count);
    created.direct_cutoffs.resize(batch_count);
    created.reciprocal_cutoffs.resize(batch_count);
    created.direct_translation_offsets.assign(batch_count + 1u, 0);
    created.reciprocal_translation_offsets.assign(batch_count + 1u, 0);
    for (std::size_t system = 0; system < batch_count; ++system) {
      const Lattice3D& lattice = topology.lattice(static_cast<std::int64_t>(system));
      const double alpha = select_alpha(lattice);
      const double direct_cutoff = search_cutoff_direct(alpha);
      const double reciprocal_cutoff = search_cutoff_reciprocal(alpha, lattice.volume);
      created.alphas[system] = alpha;
      created.direct_cutoffs[system] = direct_cutoff;
      created.reciprocal_cutoffs[system] = reciprocal_cutoff;
      std::vector<LatticeTranslation> direct;
      std::vector<LatticeTranslation> reciprocal;
      xtbloom_status_t status = make_lattice_translations(
          lattice, direct_cutoff, LatticeOriginPolicy::kInclude, direct, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      Lattice3D reciprocal_lattice;
      status = make_lattice_3d(lattice.reciprocal.data(), reciprocal_lattice, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      status = make_lattice_translations(reciprocal_lattice, reciprocal_cutoff,
                                         LatticeOriginPolicy::kExclude, reciprocal, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      created.direct_translation_offsets[system + 1u] =
          created.direct_translation_offsets[system] + static_cast<std::int64_t>(direct.size());
      created.reciprocal_translation_offsets[system + 1u] =
          created.reciprocal_translation_offsets[system] +
          static_cast<std::int64_t>(reciprocal.size());
      created.direct_translations.insert(created.direct_translations.end(), direct.begin(),
                                         direct.end());
      created.reciprocal_translations.insert(created.reciprocal_translations.end(),
                                             reciprocal.begin(), reciprocal.end());
    }
    plan = PeriodicEwaldPlan(std::make_shared<const PeriodicEwaldPlanData>(std::move(created)));
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate periodic Ewald plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t evaluate_periodic_ewald_cpu(const PeriodicEwaldPlan& plan,
                                             const PeriodicShortRangePlan& topology,
                                             const double* positions, const double* shell_charges,
                                             double* coulomb_matrix, double* shell_potentials,
                                             double* energies, double* gradients,
                                             double* strain_derivatives, std::string& error) {
  if (!plan.sealed() || !topology.sealed() ||
      plan.data_->topology_identity != topology.identity() ||
      plan.batch_size() != topology.batch_size() || plan.total_atoms() != topology.total_atoms() ||
      plan.atom_offsets() != topology.atom_offsets() || positions == nullptr ||
      shell_charges == nullptr || coulomb_matrix == nullptr || shell_potentials == nullptr ||
      energies == nullptr || gradients == nullptr || strain_derivatives == nullptr ||
      !finite_array(positions, static_cast<std::size_t>(plan.total_atoms()) * 3u) ||
      !finite_array(shell_charges, static_cast<std::size_t>(plan.total_shells()))) {
    error = "periodic Ewald inputs are malformed or nonfinite";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  try {
    const std::size_t matrix_count = static_cast<std::size_t>(plan.total_matrix_elements());
    const std::size_t shell_count = static_cast<std::size_t>(plan.total_shells());
    const std::size_t atom_count = static_cast<std::size_t>(plan.total_atoms());
    const std::size_t batch_count = static_cast<std::size_t>(plan.batch_size());
    std::vector<double> staged_matrix(matrix_count, 0.0);
    std::vector<double> staged_potential(shell_count, 0.0);
    std::vector<double> staged_energy(batch_count, 0.0);
    std::vector<double> staged_gradient(atom_count * 3u, 0.0);
    std::vector<double> staged_strain(batch_count * 9u, 0.0);

    for (std::size_t system = 0; system < batch_count; ++system) {
      const std::size_t atom_begin = static_cast<std::size_t>(plan.atom_offsets()[system]);
      const std::size_t atom_end = static_cast<std::size_t>(plan.atom_offsets()[system + 1u]);
      const std::size_t shell_begin = static_cast<std::size_t>(plan.batch_shell_offsets()[system]);
      const std::size_t shell_end =
          static_cast<std::size_t>(plan.batch_shell_offsets()[system + 1u]);
      const std::size_t molecule_atoms = atom_end - atom_begin;
      const std::size_t molecule_shells = shell_end - shell_begin;
      const Lattice3D& lattice = topology.lattice(static_cast<std::int64_t>(system));
      const double alpha = plan.alpha(static_cast<std::int64_t>(system));
      const auto direct_begin = plan.data_->direct_translation_offsets[system];
      const auto direct_end = plan.data_->direct_translation_offsets[system + 1u];
      const auto reciprocal_begin = plan.data_->reciprocal_translation_offsets[system];
      const auto reciprocal_end = plan.data_->reciprocal_translation_offsets[system + 1u];
      const std::vector<LatticeTranslation> direct(
          plan.data_->direct_translations.begin() + direct_begin,
          plan.data_->direct_translations.begin() + direct_end);
      const std::vector<LatticeTranslation> reciprocal(
          plan.data_->reciprocal_translations.begin() + reciprocal_begin,
          plan.data_->reciprocal_translations.begin() + reciprocal_end);
      std::vector<std::array<double, 3>> wrapped(molecule_atoms);
      for (std::size_t atom = 0; atom < molecule_atoms; ++atom) {
        const xtbloom_status_t status = wrap_cartesian(
            lattice, positions + (atom_begin + atom) * 3u, wrapped[atom].data(), error);
        if (status != XTBLOOM_STATUS_SUCCESS) return status;
      }
      std::vector<std::vector<WscImage>> pair_images(molecule_atoms * molecule_atoms);
      for (std::size_t center = 0; center < molecule_atoms; ++center) {
        for (std::size_t image = 0; image < center; ++image) {
          if (!build_wsc_images(lattice, subtract(wrapped[center], wrapped[image]), false,
                                pair_images[center * molecule_atoms + image], error)) {
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
        }
        if (!build_wsc_images(lattice, {}, true, pair_images[center * molecule_atoms + center],
                              error)) {
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }

      for (std::size_t center = 0; center < molecule_atoms; ++center) {
        for (std::size_t image = 0; image < center; ++image) {
          const auto rij = subtract(wrapped[center], wrapped[image]);
          const double base = direct_sum(rij, direct, alpha) +
                              reciprocal_sum(rij, reciprocal, alpha, lattice.volume);
          const auto d_direct = direct_derivative_sum(rij, direct, alpha);
          const auto d_reciprocal = reciprocal_derivative(rij, reciprocal, alpha, lattice.volume);
          const auto d_strain_direct = direct_strain_sum(rij, direct, alpha);
          const auto d_strain_reciprocal =
              reciprocal_strain(rij, reciprocal, alpha, lattice.volume);
          const auto& images = pair_images[center * molecule_atoms + image];
          const std::size_t center_shell_begin =
              static_cast<std::size_t>(plan.data_->atom_shell_offsets[atom_begin + center]);
          const std::size_t center_shell_end =
              static_cast<std::size_t>(plan.data_->atom_shell_offsets[atom_begin + center + 1u]);
          const std::size_t image_shell_begin =
              static_cast<std::size_t>(plan.data_->atom_shell_offsets[atom_begin + image]);
          const std::size_t image_shell_end =
              static_cast<std::size_t>(plan.data_->atom_shell_offsets[atom_begin + image + 1u]);
          for (const auto& image_data : images) {
            const double weight = image_data.weight;
            for (std::size_t shell_i = center_shell_begin; shell_i < center_shell_end; ++shell_i) {
              for (std::size_t shell_j = image_shell_begin; shell_j < image_shell_end; ++shell_j) {
                const std::size_t local_i = shell_i - shell_begin;
                const std::size_t local_j = shell_j - shell_begin;
                const double gamma = 0.5 * (plan.data_->shell_hardness[shell_i] +
                                            plan.data_->shell_hardness[shell_j]);
                const double correction = correction_value(image_data.vector, gamma);
                const double value = base + correction;
                const std::size_t ij = matrix_index(*plan.data_, system, local_i, local_j);
                const std::size_t ji = matrix_index(*plan.data_, system, local_j, local_i);
                staged_matrix[ij] += weight * value;
                staged_matrix[ji] += weight * value;
                const auto derivative = [&]() {
                  std::array<double, 3> result{};
                  add_inplace(result, d_direct, weight);
                  add_inplace(result, d_reciprocal, weight);
                  add_inplace(result, correction_derivative(image_data.vector, gamma), weight);
                  add_inplace(result, image_data.weight_gradient, correction);
                  return result;
                }();
                const auto strain = [&]() {
                  std::array<double, 9> result{};
                  add_inplace(result, d_strain_direct, weight);
                  add_inplace(result, d_strain_reciprocal, weight);
                  add_inplace(
                      result,
                      outer(correction_derivative(image_data.vector, gamma), image_data.vector),
                      weight);
                  add_inplace(result, image_data.weight_strain, correction);
                  return result;
                }();
                const double factor = shell_charges[shell_i] * shell_charges[shell_j];
                add_flat_vector(staged_gradient.data(), atom_begin + center, derivative, factor);
                add_flat_vector(staged_gradient.data(), atom_begin + image, derivative, -factor);
                add_flat_tensor(staged_strain.data(), system, strain, factor);
              }
            }
          }
        }

        const double base_self = direct_sum({}, direct, alpha) +
                                 reciprocal_sum({}, reciprocal, alpha, lattice.volume) -
                                 2.0 * alpha / kSqrtPi;
        const auto d_strain_self_direct = direct_strain_sum({}, direct, alpha);
        const auto d_strain_self_reciprocal =
            reciprocal_strain({}, reciprocal, alpha, lattice.volume);
        const auto& self_images = pair_images[center * molecule_atoms + center];
        const std::size_t atom_shell_begin =
            static_cast<std::size_t>(plan.data_->atom_shell_offsets[atom_begin + center]);
        const std::size_t atom_shell_end =
            static_cast<std::size_t>(plan.data_->atom_shell_offsets[atom_begin + center + 1u]);
        for (std::size_t shell_i = atom_shell_begin; shell_i < atom_shell_end; ++shell_i) {
          for (std::size_t shell_j = atom_shell_begin; shell_j <= shell_i; ++shell_j) {
            const double gamma =
                0.5 * (plan.data_->shell_hardness[shell_i] + plan.data_->shell_hardness[shell_j]);
            double value = 0.0;
            std::array<double, 9> derivative_strain{};
            for (const auto& image_data : self_images) {
              const double correction = correction_value(image_data.vector, gamma);
              value += image_data.weight * (base_self + correction + gamma);
              add_inplace(derivative_strain, d_strain_self_direct, image_data.weight);
              add_inplace(derivative_strain, d_strain_self_reciprocal, image_data.weight);
              add_inplace(derivative_strain,
                          outer(correction_derivative(image_data.vector, gamma), image_data.vector),
                          image_data.weight);
              add_inplace(derivative_strain, image_data.weight_strain, correction);
            }
            const std::size_t local_i = shell_i - shell_begin;
            const std::size_t local_j = shell_j - shell_begin;
            const std::size_t ij = matrix_index(*plan.data_, system, local_i, local_j);
            staged_matrix[ij] += value;
            if (shell_i != shell_j) {
              staged_matrix[matrix_index(*plan.data_, system, local_j, local_i)] += value;
            }
            const double factor = shell_i == shell_j
                                      ? 0.5 * shell_charges[shell_i] * shell_charges[shell_i]
                                      : shell_charges[shell_i] * shell_charges[shell_j];
            add_flat_tensor(staged_strain.data(), system, derivative_strain, factor);
          }
        }
      }

      /*
       * The shell matrix intentionally retains tblite's neutral-cell
       * convention: its entries contain the pair, self, and Klopman--Ohno
       * terms, while the uniform background is a batch-level contribution.
       * Keeping the background out of the matrix preserves the frozen
       * neutral matrix oracle and avoids making a zero-total-charge result
       * depend on an otherwise unobservable constant shift.
       */
      const double total_charge = [&]() {
        double result = 0.0;
        for (std::size_t shell = shell_begin; shell < shell_end; ++shell)
          result += shell_charges[shell];
        return result;
      }();
      const double alpha_squared = alpha * alpha;
      const double background_factor = -kPi / (alpha_squared * lattice.volume);
      if (!std::isfinite(total_charge) || !(alpha_squared > 0.0) ||
          !std::isfinite(background_factor)) {
        error = "periodic Ewald background contribution is invalid";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }

      for (std::size_t row = 0; row < molecule_shells; ++row) {
        double potential = 0.0;
        for (std::size_t column = 0; column < molecule_shells; ++column) {
          potential += staged_matrix[matrix_index(*plan.data_, system, row, column)] *
                       shell_charges[shell_begin + column];
        }
        /* phi_background = -pi Q/(alpha^2 V). */
        staged_potential[shell_begin + row] = potential + background_factor * total_charge;
        staged_energy[system] += 0.5 * shell_charges[shell_begin + row] * potential;
      }
      /* E_background = -pi Q^2/(2 alpha^2 V). */
      staged_energy[system] += 0.5 * background_factor * total_charge * total_charge;
      /* dE_background/d epsilon_ab is isotropic at fixed alpha. */
      const double background_strain = -0.5 * background_factor * total_charge * total_charge;
      for (const std::size_t diagonal : {std::size_t{0}, std::size_t{4}, std::size_t{8}})
        staged_strain[system * 9u + diagonal] += background_strain;
    }
    if (!finite_array(staged_matrix.data(), staged_matrix.size()) ||
        !finite_array(staged_potential.data(), staged_potential.size()) ||
        !finite_array(staged_energy.data(), staged_energy.size()) ||
        !finite_array(staged_gradient.data(), staged_gradient.size()) ||
        !finite_array(staged_strain.data(), staged_strain.size())) {
      error = "periodic Ewald evaluation overflowed";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    std::copy(staged_matrix.begin(), staged_matrix.end(), coulomb_matrix);
    std::copy(staged_potential.begin(), staged_potential.end(), shell_potentials);
    std::copy(staged_energy.begin(), staged_energy.end(), energies);
    std::copy(staged_gradient.begin(), staged_gradient.end(), gradients);
    std::copy(staged_strain.begin(), staged_strain.end(), strain_derivatives);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate periodic Ewald evaluation scratch";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

}  // namespace xtbloom::detail::gfn2
