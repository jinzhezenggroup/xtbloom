#include "model/gfn2/alpb.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace xtbloom::detail::gfn2 {
namespace {

constexpr double kP16Zeta = 1.028;
constexpr double kP16ZetaOver16 = kP16Zeta / 16.0;
constexpr double kSphereInertiaFactor = 2.0 / 5.0;
struct ShapeDescriptor {
  double value = 0.0;
  double volume = 0.0;
  double center[3]{};
  double inverse_inertia[9]{};
};

bool representable_atom_count(std::int64_t atom_count) {
  if (atom_count < 0 ||
      static_cast<std::uint64_t>(atom_count) > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  const auto count = static_cast<std::uint64_t>(atom_count);
  if (count == 0u) {
    return true;
  }
  return count <= std::numeric_limits<std::size_t>::max() / 3u &&
         count <= std::numeric_limits<std::size_t>::max() / count / sizeof(double);
}

bool valid_settings(const AlpbPolarSettings& settings) {
  const bool valid_model =
      settings.model == AlpbPolarModel::kAlpb || settings.model == AlpbPolarModel::kGbsa;
  const bool valid_kernel =
      settings.kernel == AlpbBornKernel::kP16 || settings.kernel == AlpbBornKernel::kStill;
  if (!valid_model || !valid_kernel || !std::isfinite(settings.dielectric_constant) ||
      settings.dielectric_constant <= 0.0) {
    return false;
  }
  return std::isfinite(1.0 / settings.dielectric_constant);
}

double determinant_3x3(const double* matrix) {
  return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
         matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
         matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

bool invert_symmetric_3x3(const double* matrix, double determinant, double* inverse) {
  if (!(determinant > 0.0) || !std::isfinite(determinant)) {
    return false;
  }
  const double inverse_determinant = 1.0 / determinant;
  inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inverse_determinant;
  inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inverse_determinant;
  inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inverse_determinant;
  inverse[3] = inverse[1];
  inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inverse_determinant;
  inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inverse_determinant;
  inverse[6] = inverse[2];
  inverse[7] = inverse[5];
  inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inverse_determinant;
  for (std::size_t element = 0; element < 9u; ++element) {
    if (!std::isfinite(inverse[element])) {
      return false;
    }
  }
  return true;
}

bool compute_shape_descriptor(std::size_t atom_count, const double* positions,
                              const double* cavity_radii, bool need_inverse,
                              ShapeDescriptor& shape) {
  shape = ShapeDescriptor{};
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    const double radius = cavity_radii[atom];
    const double weight = radius * radius * radius;
    shape.volume += weight;
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      shape.center[axis] = std::fma(weight, positions[atom * 3u + axis], shape.center[axis]);
    }
  }
  if (!(shape.volume > 0.0) || !std::isfinite(shape.volume)) {
    return false;
  }
  for (double& coordinate : shape.center) {
    coordinate /= shape.volume;
  }

  double inertia[9]{};
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    const double radius = cavity_radii[atom];
    const double radius_squared = radius * radius;
    const double weight = radius_squared * radius;
    const double x = positions[atom * 3u] - shape.center[0];
    const double y = positions[atom * 3u + 1u] - shape.center[1];
    const double z = positions[atom * 3u + 2u] - shape.center[2];
    const double distance_squared = x * x + y * y + z * z;
    const double diagonal = distance_squared + kSphereInertiaFactor * radius_squared;
    inertia[0] += weight * (diagonal - x * x);
    inertia[1] -= weight * x * y;
    inertia[2] -= weight * x * z;
    inertia[4] += weight * (diagonal - y * y);
    inertia[5] -= weight * y * z;
    inertia[8] += weight * (diagonal - z * z);
  }
  inertia[3] = inertia[1];
  inertia[6] = inertia[2];
  inertia[7] = inertia[5];

  const double determinant = determinant_3x3(inertia);
  if (!(determinant > 0.0) || !std::isfinite(determinant)) {
    return false;
  }
  shape.value = std::sqrt(std::cbrt(determinant) / (kSphereInertiaFactor * shape.volume));
  if (!(shape.value > 0.0) || !std::isfinite(shape.value)) {
    return false;
  }
  return !need_inverse || invert_symmetric_3x3(inertia, determinant, shape.inverse_inertia);
}

xtbloom_status_t validate_geometry_inputs(std::int64_t atom_count, const double* positions,
                                          const double* born_radii, const double* cavity_radii,
                                          const AlpbPolarSettings& settings, std::string& error) {
  if (!representable_atom_count(atom_count)) {
    error = "ALPB polar term requires a nonnegative, representable atom count";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!valid_settings(settings)) {
    error = "ALPB polar settings contain an invalid model, kernel, or dielectric constant";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_count == 0) {
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (positions == nullptr || born_radii == nullptr ||
      (settings.model == AlpbPolarModel::kAlpb && cavity_radii == nullptr)) {
    error = "ALPB polar geometry inputs must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const auto count = static_cast<std::size_t>(atom_count);
  for (std::size_t coordinate = 0; coordinate < count * 3u; ++coordinate) {
    if (!std::isfinite(positions[coordinate])) {
      error = "ALPB polar positions contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t atom = 0; atom < count; ++atom) {
    if (!(born_radii[atom] > 0.0) || !std::isfinite(born_radii[atom])) {
      error = "ALPB polar Born radii must be finite and positive";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (settings.model == AlpbPolarModel::kAlpb &&
        (!(cavity_radii[atom] > 0.0) || !std::isfinite(cavity_radii[atom]))) {
      error = "ALPB polar cavity radii must be finite and positive";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (settings.model == AlpbPolarModel::kAlpb) {
      const double squared_radius = cavity_radii[atom] * cavity_radii[atom];
      const double cavity_weight = squared_radius * cavity_radii[atom];
      if (!(cavity_weight > 0.0) || !std::isfinite(cavity_weight)) {
        error = "ALPB polar cavity radii exceed the stable binary64 range";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  for (std::size_t first = 0; first < count; ++first) {
    for (std::size_t second = 0; second < first; ++second) {
      const double dx = positions[first * 3u] - positions[second * 3u];
      const double dy = positions[first * 3u + 1u] - positions[second * 3u + 1u];
      const double dz = positions[first * 3u + 2u] - positions[second * 3u + 2u];
      const double distance_measure = settings.kernel == AlpbBornKernel::kP16
                                          ? std::hypot(dx, dy, dz)
                                          : dx * dx + dy * dy + dz * dz;
      const double radius_product = born_radii[first] * born_radii[second];
      if (!std::isfinite(distance_measure) || !(radius_product > 0.0) ||
          !std::isfinite(radius_product)) {
        error = "ALPB polar pair inputs exceed the stable binary64 range";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

double p16_inverse_distance(double distance, double first_radius, double second_radius) {
  const double geometric_radius = std::sqrt(first_radius * second_radius);
  const double argument = geometric_radius / (geometric_radius + kP16ZetaOver16 * distance);
  const double argument2 = argument * argument;
  const double argument4 = argument2 * argument2;
  const double argument8 = argument4 * argument4;
  const double argument16 = argument8 * argument8;
  return 1.0 / (distance + geometric_radius * argument16);
}

double still_inverse_distance(double distance_squared, double first_radius, double second_radius) {
  const double radius_product = first_radius * second_radius;
  const double screened_squared =
      distance_squared + radius_product * std::exp(-0.25 * distance_squared / radius_product);
  return 1.0 / std::sqrt(screened_squared);
}

}  // namespace

xtbloom_status_t build_alpb_polar_matrix_cpu(std::int64_t atom_count, const double* positions,
                                             const double* born_radii, const double* cavity_radii,
                                             const AlpbPolarSettings& settings, double* matrix,
                                             std::string& error) {
  xtbloom_status_t status =
      validate_geometry_inputs(atom_count, positions, born_radii, cavity_radii, settings, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (atom_count == 0) {
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (matrix == nullptr) {
    error = "ALPB polar matrix output must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  ShapeDescriptor shape;
  if (settings.model == AlpbPolarModel::kAlpb &&
      !compute_shape_descriptor(static_cast<std::size_t>(atom_count), positions, cavity_radii,
                                false, shape)) {
    error = "ALPB polar cavity has an invalid electrostatic-size descriptor";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const double inverse_dielectric = 1.0 / settings.dielectric_constant;
  const double alpha_beta =
      settings.model == AlpbPolarModel::kAlpb ? kAlpbAlpha * inverse_dielectric : 0.0;
  const double dielectric_scale = (inverse_dielectric - 1.0) / (1.0 + alpha_beta);
  const double size_correction =
      settings.model == AlpbPolarModel::kAlpb ? dielectric_scale * alpha_beta / shape.value : 0.0;

  const auto count = static_cast<std::size_t>(atom_count);
  const auto matrix_element = [&](std::size_t first, std::size_t second) {
    if (first == second) {
      return dielectric_scale / born_radii[first] + size_correction;
    }
    const double dx = positions[first * 3u] - positions[second * 3u];
    const double dy = positions[first * 3u + 1u] - positions[second * 3u + 1u];
    const double dz = positions[first * 3u + 2u] - positions[second * 3u + 2u];
    const double inverse_distance =
        settings.kernel == AlpbBornKernel::kP16
            ? p16_inverse_distance(std::hypot(dx, dy, dz), born_radii[first], born_radii[second])
            : still_inverse_distance(dx * dx + dy * dy + dz * dz, born_radii[first],
                                     born_radii[second]);
    return dielectric_scale * inverse_distance + size_correction;
  };

  // Validate a complete candidate matrix before touching caller storage. The
  // second pass deliberately recomputes the symmetric elements, trading extra
  // arithmetic for O(1) transactional workspace on the O(N^2) SCC hot path.
  for (std::size_t first = 0; first < count; ++first) {
    if (!std::isfinite(matrix_element(first, first))) {
      error = "ALPB polar matrix arithmetic exceeded the stable binary64 range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t second = 0; second < first; ++second) {
      if (!std::isfinite(matrix_element(first, second))) {
        error = "ALPB polar matrix arithmetic exceeded the stable binary64 range";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }

  for (std::size_t first = 0; first < count; ++first) {
    matrix[first * count + first] = matrix_element(first, first);
    for (std::size_t second = 0; second < first; ++second) {
      const double element = matrix_element(first, second);
      matrix[first * count + second] = element;
      matrix[second * count + first] = element;
    }
  }

  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t evaluate_alpb_polar_cpu(std::int64_t atom_count, const double* matrix,
                                         const double* atomic_charges, double* atomic_potentials,
                                         double* energy, std::string& error) {
  if (!representable_atom_count(atom_count)) {
    error = "ALPB polar evaluation requires a nonnegative, representable atom count";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (energy == nullptr || (atom_count != 0 && (matrix == nullptr || atomic_charges == nullptr ||
                                                atomic_potentials == nullptr))) {
    error = "ALPB polar evaluation inputs and outputs must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_count == 0) {
    *energy = 0.0;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  const auto count = static_cast<std::size_t>(atom_count);
  for (std::size_t atom = 0; atom < count; ++atom) {
    if (!std::isfinite(atomic_charges[atom])) {
      error = "ALPB polar charges contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t element = 0; element < count * count; ++element) {
    if (!std::isfinite(matrix[element])) {
      error = "ALPB polar matrix contains NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  try {
    std::vector<double> candidate_potentials(count);
    double polar_energy = 0.0;
    for (std::size_t row = 0; row < count; ++row) {
      double potential = 0.0;
      for (std::size_t column = 0; column < count; ++column) {
        potential = std::fma(matrix[row * count + column], atomic_charges[column], potential);
      }
      if (!std::isfinite(potential)) {
        error = "ALPB polar potential arithmetic exceeded the stable binary64 range";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      candidate_potentials[row] = potential;
      polar_energy = std::fma(0.5 * atomic_charges[row], potential, polar_energy);
    }
    if (!std::isfinite(polar_energy)) {
      error = "ALPB polar energy arithmetic exceeded the stable binary64 range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    std::copy(candidate_potentials.begin(), candidate_potentials.end(), atomic_potentials);
    *energy = polar_energy;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate transactional ALPB polar potential storage";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_alpb_polar_gradient_cpu(std::int64_t atom_count, const double* positions,
                                             const double* born_radii, const double* cavity_radii,
                                             const double* atomic_charges,
                                             const AlpbPolarSettings& settings, double* gradients,
                                             std::string& error) {
  xtbloom_status_t status =
      validate_geometry_inputs(atom_count, positions, born_radii, cavity_radii, settings, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (atom_count == 0) {
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (atomic_charges == nullptr || gradients == nullptr) {
    error = "ALPB polar gradient charges and output must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const auto count = static_cast<std::size_t>(atom_count);
  for (std::size_t atom = 0; atom < count; ++atom) {
    if (!std::isfinite(atomic_charges[atom]) || !std::isfinite(gradients[atom * 3u]) ||
        !std::isfinite(gradients[atom * 3u + 1u]) || !std::isfinite(gradients[atom * 3u + 2u])) {
      error = "ALPB polar charges and accumulated gradients must be finite";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  if (settings.kernel == AlpbBornKernel::kP16) {
    for (std::size_t first = 0; first < count; ++first) {
      for (std::size_t second = 0; second < first; ++second) {
        const double dx = positions[first * 3u] - positions[second * 3u];
        const double dy = positions[first * 3u + 1u] - positions[second * 3u + 1u];
        const double dz = positions[first * 3u + 2u] - positions[second * 3u + 2u];
        if (std::hypot(dx, dy, dz) == 0.0) {
          error = "the P16 polar coordinate derivative is undefined for coincident atoms";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }
    }
  }

  ShapeDescriptor shape;
  if (settings.model == AlpbPolarModel::kAlpb &&
      !compute_shape_descriptor(count, positions, cavity_radii, true, shape)) {
    error = "ALPB polar cavity has an invalid electrostatic-size derivative";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const double inverse_dielectric = 1.0 / settings.dielectric_constant;
  const double alpha_beta =
      settings.model == AlpbPolarModel::kAlpb ? kAlpbAlpha * inverse_dielectric : 0.0;
  const double dielectric_scale = (inverse_dielectric - 1.0) / (1.0 + alpha_beta);

  try {
    std::vector<double> candidate(gradients, gradients + count * 3u);
    for (std::size_t first = 0; first < count; ++first) {
      for (std::size_t second = 0; second < first; ++second) {
        const double dx = positions[first * 3u] - positions[second * 3u];
        const double dy = positions[first * 3u + 1u] - positions[second * 3u + 1u];
        const double dz = positions[first * 3u + 2u] - positions[second * 3u + 2u];
        const double charge_product = atomic_charges[first] * atomic_charges[second];
        double radial_scale = 0.0;
        double direction_scale = 1.0;
        if (settings.kernel == AlpbBornKernel::kP16) {
          const double distance = std::hypot(dx, dy, dz);
          const double geometric_radius = std::sqrt(born_radii[first] * born_radii[second]);
          const double argument = geometric_radius / (geometric_radius + kP16ZetaOver16 * distance);
          const double argument2 = argument * argument;
          const double argument4 = argument2 * argument2;
          const double argument8 = argument4 * argument4;
          const double argument16 = argument8 * argument8;
          const double screened_distance = distance + geometric_radius * argument16;
          radial_scale = dielectric_scale * charge_product *
                         (1.0 - kP16Zeta * argument * argument16) /
                         (screened_distance * screened_distance);
          direction_scale = 1.0 / distance;
        } else {
          const double distance_squared = dx * dx + dy * dy + dz * dz;
          const double radius_product = born_radii[first] * born_radii[second];
          const double exponential = std::exp(-0.25 * distance_squared / radius_product);
          const double screened_squared = distance_squared + radius_product * exponential;
          radial_scale = dielectric_scale * charge_product * (1.0 - 0.25 * exponential) /
                         (screened_squared * std::sqrt(screened_squared));
        }

        const double gx = radial_scale * dx * direction_scale;
        const double gy = radial_scale * dy * direction_scale;
        const double gz = radial_scale * dz * direction_scale;
        candidate[first * 3u] -= gx;
        candidate[first * 3u + 1u] -= gy;
        candidate[first * 3u + 2u] -= gz;
        candidate[second * 3u] += gx;
        candidate[second * 3u + 1u] += gy;
        candidate[second * 3u + 2u] += gz;
      }
    }

    if (settings.model == AlpbPolarModel::kAlpb) {
      double total_charge = 0.0;
      for (std::size_t atom = 0; atom < count; ++atom) {
        total_charge += atomic_charges[atom];
      }
      const double inverse_trace =
          shape.inverse_inertia[0] + shape.inverse_inertia[4] + shape.inverse_inertia[8];
      const double energy_scale =
          -0.5 * dielectric_scale * alpha_beta * total_charge * total_charge / shape.value;
      for (std::size_t atom = 0; atom < count; ++atom) {
        const double radius = cavity_radii[atom];
        const double weight = radius * radius * radius;
        const double x = positions[atom * 3u] - shape.center[0];
        const double y = positions[atom * 3u + 1u] - shape.center[1];
        const double z = positions[atom * 3u + 2u] - shape.center[2];
        const double inverse_x = shape.inverse_inertia[0] * x + shape.inverse_inertia[1] * y +
                                 shape.inverse_inertia[2] * z;
        const double inverse_y = shape.inverse_inertia[3] * x + shape.inverse_inertia[4] * y +
                                 shape.inverse_inertia[5] * z;
        const double inverse_z = shape.inverse_inertia[6] * x + shape.inverse_inertia[7] * y +
                                 shape.inverse_inertia[8] * z;
        const double scale = energy_scale * weight / 3.0;
        candidate[atom * 3u] += scale * (inverse_trace * x - inverse_x);
        candidate[atom * 3u + 1u] += scale * (inverse_trace * y - inverse_y);
        candidate[atom * 3u + 2u] += scale * (inverse_trace * z - inverse_z);
      }
    }
    for (const double component : candidate) {
      if (!std::isfinite(component)) {
        error = "ALPB polar gradient arithmetic exceeded the stable binary64 range";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    std::copy(candidate.begin(), candidate.end(), gradients);
  } catch (const std::bad_alloc&) {
    error = "failed to allocate transactional ALPB polar gradient storage";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }

  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn2
