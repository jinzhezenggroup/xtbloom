// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/lattice.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace xtbloom::detail::gfn2 {
namespace {

constexpr long double kTwoPi = 6.2831853071795864769252867665590057683943387987502L;

std::array<long double, 3> cross(const std::array<long double, 3>& first,
                                 const std::array<long double, 3>& second) {
  return {
      first[1] * second[2] - first[2] * second[1],
      first[2] * second[0] - first[0] * second[2],
      first[0] * second[1] - first[1] * second[0],
  };
}

long double dot(const std::array<long double, 3>& first, const std::array<long double, 3>& second) {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

long double norm(const std::array<long double, 3>& value) {
  return std::hypot(value[0], value[1], value[2]);
}

bool finite_vector(const double* value) {
  return value != nullptr && std::isfinite(value[0]) && std::isfinite(value[1]) &&
         std::isfinite(value[2]);
}

bool store_double(long double value, double& output, bool require_positive = false);

bool finite_lattice(const Lattice3D& lattice) {
  if (!(lattice.volume > 0.0) || !std::isfinite(lattice.volume) ||
      !std::all_of(lattice.direct.begin(), lattice.direct.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::all_of(lattice.reciprocal.begin(), lattice.reciprocal.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::all_of(lattice.plane_spacing.begin(), lattice.plane_spacing.end(),
                   [](double value) { return value > 0.0 && std::isfinite(value); })) {
    return false;
  }
  if (!valid_lattice_cell_3d(lattice.direct.data())) {
    return false;
  }

  std::array<double, 9> reciprocal{};
  std::array<double, 3> spacing{};
  double scale = 0.0;
  for (double value : lattice.direct) {
    scale = std::max(scale, std::abs(value));
  }
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    return false;
  }
  std::array<std::array<long double, 3>, 3> normalized{};
  for (std::size_t vector = 0; vector < 3u; ++vector) {
    for (std::size_t component = 0; component < 3u; ++component) {
      normalized[vector][component] =
          static_cast<long double>(lattice.direct[vector * 3u + component]) /
          static_cast<long double>(scale);
    }
  }
  const std::array<std::array<long double, 3>, 3> cofactors{cross(normalized[1], normalized[2]),
                                                            cross(normalized[2], normalized[0]),
                                                            cross(normalized[0], normalized[1])};
  const long double scaled_determinant = dot(normalized[0], cofactors[0]);
  if (!(scaled_determinant > 0.0L) || !std::isfinite(scaled_determinant)) {
    return false;
  }
  const long double scale_long = static_cast<long double>(scale);
  const long double volume = scaled_determinant * scale_long * scale_long * scale_long;
  const long double reciprocal_scale = kTwoPi / (scale_long * scaled_determinant);
  for (std::size_t vector = 0; vector < 3u; ++vector) {
    if (!store_double(scale_long * scaled_determinant / norm(cofactors[vector]), spacing[vector],
                      true)) {
      return false;
    }
    for (std::size_t component = 0; component < 3u; ++component) {
      if (!store_double(reciprocal_scale * cofactors[vector][component],
                        reciprocal[vector * 3u + component])) {
        return false;
      }
    }
  }
  double stored_volume = 0.0;
  if (!store_double(volume, stored_volume, true) || stored_volume != lattice.volume) {
    return false;
  }
  for (std::size_t element = 0; element < reciprocal.size(); ++element) {
    if (reciprocal[element] != lattice.reciprocal[element]) {
      return false;
    }
  }
  for (std::size_t vector = 0; vector < spacing.size(); ++vector) {
    if (spacing[vector] != lattice.plane_spacing[vector]) {
      return false;
    }
  }
  return true;
}

bool store_double(long double value, double& output, bool require_positive) {
  if (!std::isfinite(value) ||
      std::abs(value) > static_cast<long double>(std::numeric_limits<double>::max()) ||
      (require_positive &&
       value < static_cast<long double>(std::numeric_limits<double>::denorm_min()))) {
    return false;
  }
  output = static_cast<double>(value);
  return std::isfinite(output) && (!require_positive || output > 0.0);
}

xtbloom_status_t validate_transform_arguments(const Lattice3D& lattice, const double* input,
                                              double* output, const char* quantity,
                                              std::string& error) {
  if (!finite_lattice(lattice)) {
    error = "lattice geometry is incomplete or contains nonfinite values";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (input == nullptr || output == nullptr) {
    error = std::string(quantity) + " input and output must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!finite_vector(input)) {
    error = std::string(quantity) + " input contains NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

bool checked_product(std::size_t factor, std::size_t& product) {
  if (factor != 0u && product > std::numeric_limits<std::size_t>::max() / factor) {
    return false;
  }
  product *= factor;
  return true;
}

bool repeat_count(double cutoff, double plane_spacing, std::int64_t& repeat) {
  const long double ratio =
      static_cast<long double>(cutoff) / static_cast<long double>(plane_spacing);
  const long double rounded = std::ceil(ratio);
  if (!std::isfinite(rounded) || rounded < 0.0L ||
      rounded > static_cast<long double>((std::numeric_limits<std::int64_t>::max() - 1) / 2)) {
    return false;
  }
  repeat = static_cast<std::int64_t>(rounded);
  return true;
}

bool make_translation(const Lattice3D& lattice, std::int64_t first, std::int64_t second,
                      std::int64_t third, LatticeTranslation& translation) {
  translation.index = {first, second, third};
  for (std::size_t component = 0; component < 3u; ++component) {
    const long double value = static_cast<long double>(first) * lattice.direct[component] +
                              static_cast<long double>(second) * lattice.direct[3u + component] +
                              static_cast<long double>(third) * lattice.direct[6u + component];
    if (!store_double(value, translation.cartesian[component])) {
      return false;
    }
    if (translation.cartesian[component] == 0.0) {
      translation.cartesian[component] = 0.0;
    }
  }
  return true;
}

}  // namespace

bool valid_lattice_cell_3d(const double* direct) noexcept {
  return valid_lattice_cell_3d_binary64(direct);
}

xtbloom_status_t make_lattice_3d(const double* direct, Lattice3D& lattice, std::string& error) {
  if (direct == nullptr) {
    error = "direct lattice must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  double scale = 0.0;
  for (std::size_t element = 0; element < 9u; ++element) {
    if (!std::isfinite(direct[element])) {
      error = "direct lattice contains NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    scale = std::max(scale, std::abs(direct[element]));
  }
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    error = "direct lattice is singular";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!valid_lattice_cell_3d(direct)) {
    std::array<std::array<long double, 3>, 3> vectors{};
    for (std::size_t vector = 0; vector < 3u; ++vector) {
      for (std::size_t component = 0; component < 3u; ++component) {
        vectors[vector][component] = static_cast<long double>(direct[vector * 3u + component]);
      }
    }
    const long double determinant = dot(vectors[0], cross(vectors[1], vectors[2]));
    error = determinant < 0.0L ? "direct lattice must be right-handed"
                               : "direct lattice is singular or numerically ill-conditioned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::array<std::array<long double, 3>, 3> normalized{};
  for (std::size_t vector = 0; vector < 3u; ++vector) {
    for (std::size_t component = 0; component < 3u; ++component) {
      normalized[vector][component] = static_cast<long double>(direct[vector * 3u + component]) /
                                      static_cast<long double>(scale);
    }
  }
  const std::array<long double, 3> cross_bc = cross(normalized[1], normalized[2]);
  const std::array<long double, 3> cross_ca = cross(normalized[2], normalized[0]);
  const std::array<long double, 3> cross_ab = cross(normalized[0], normalized[1]);
  const long double scaled_determinant = dot(normalized[0], cross_bc);
  /* valid_lattice_cell_3d() already established handedness and the shared
   * norm-product conditioning threshold. */

  Lattice3D created;
  std::copy_n(direct, 9u, created.direct.begin());
  const long double scale_long = static_cast<long double>(scale);
  const long double volume = scaled_determinant * scale_long * scale_long * scale_long;
  if (!store_double(volume, created.volume, true)) {
    error = "direct lattice volume is outside the binary64 range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const std::array<std::array<long double, 3>, 3> cofactors{cross_bc, cross_ca, cross_ab};
  const long double reciprocal_scale = kTwoPi / (scale_long * scaled_determinant);
  for (std::size_t vector = 0; vector < 3u; ++vector) {
    const long double cofactor_norm = norm(cofactors[vector]);
    if (!(cofactor_norm > 0.0L) || !std::isfinite(cofactor_norm)) {
      error = "direct lattice is singular";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const long double plane_spacing = scale_long * scaled_determinant / cofactor_norm;
    if (!store_double(plane_spacing, created.plane_spacing[vector], true)) {
      error = "lattice plane spacing is outside the binary64 range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t component = 0; component < 3u; ++component) {
      if (!store_double(reciprocal_scale * cofactors[vector][component],
                        created.reciprocal[vector * 3u + component])) {
        error = "reciprocal lattice is outside the binary64 range";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }

  lattice = created;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t fractional_to_cartesian(const Lattice3D& lattice, const double* fractional,
                                         double* cartesian, std::string& error) {
  const xtbloom_status_t status =
      validate_transform_arguments(lattice, fractional, cartesian, "fractional coordinate", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  std::array<double, 3> result{};
  for (std::size_t component = 0; component < 3u; ++component) {
    const long double value =
        static_cast<long double>(fractional[0]) * lattice.direct[component] +
        static_cast<long double>(fractional[1]) * lattice.direct[3u + component] +
        static_cast<long double>(fractional[2]) * lattice.direct[6u + component];
    if (!store_double(value, result[component])) {
      error = "fractional-to-Cartesian conversion overflowed binary64";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  std::copy(result.begin(), result.end(), cartesian);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t cartesian_to_fractional(const Lattice3D& lattice, const double* cartesian,
                                         double* fractional, std::string& error) {
  const xtbloom_status_t status =
      validate_transform_arguments(lattice, cartesian, fractional, "Cartesian coordinate", error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  std::array<double, 3> result{};
  for (std::size_t vector = 0; vector < 3u; ++vector) {
    long double value = 0.0L;
    for (std::size_t component = 0; component < 3u; ++component) {
      value += static_cast<long double>(lattice.reciprocal[vector * 3u + component]) *
               static_cast<long double>(cartesian[component]);
    }
    if (!store_double(value / kTwoPi, result[vector])) {
      error = "Cartesian-to-fractional conversion overflowed binary64";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  std::copy(result.begin(), result.end(), fractional);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t wrap_fractional(const double* fractional, double* wrapped, std::string& error) {
  if (fractional == nullptr || wrapped == nullptr) {
    error = "fractional wrapping input and output must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!finite_vector(fractional)) {
    error = "fractional wrapping input contains NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::array<double, 3> result{};
  for (std::size_t component = 0; component < 3u; ++component) {
    result[component] = fractional[component] - std::floor(fractional[component]);
    if (result[component] >= 1.0) {
      result[component] = 0.0;
    }
    if (result[component] == 0.0) {
      result[component] = 0.0;
    }
  }
  std::copy(result.begin(), result.end(), wrapped);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t wrap_cartesian(const Lattice3D& lattice, const double* cartesian, double* wrapped,
                                std::string& error) {
  if (wrapped == nullptr) {
    error = "Cartesian wrapping output must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::array<double, 3> fractional{};
  xtbloom_status_t status = cartesian_to_fractional(lattice, cartesian, fractional.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = wrap_fractional(fractional.data(), fractional.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  return fractional_to_cartesian(lattice, fractional.data(), wrapped, error);
}

xtbloom_status_t make_lattice_translations(const Lattice3D& lattice, double cutoff,
                                           LatticeOriginPolicy origin_policy,
                                           std::vector<LatticeTranslation>& translations,
                                           std::string& error) {
  if (!finite_lattice(lattice)) {
    error = "lattice geometry is incomplete or contains nonfinite values";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!(cutoff >= 0.0) || !std::isfinite(cutoff)) {
    error = "lattice-image cutoff must be finite and nonnegative";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (origin_policy != LatticeOriginPolicy::kInclude &&
      origin_policy != LatticeOriginPolicy::kExclude) {
    error = "lattice-image origin policy is invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::array<std::int64_t, 3> repeat{};
  std::size_t count = 1u;
  for (std::size_t vector = 0; vector < 3u; ++vector) {
    if (!repeat_count(cutoff, lattice.plane_spacing[vector], repeat[vector])) {
      error = "lattice-image repeat count is outside the supported integer range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const auto width = static_cast<std::uint64_t>(repeat[vector]) * 2u + 1u;
    if (width > std::numeric_limits<std::size_t>::max() ||
        !checked_product(static_cast<std::size_t>(width), count)) {
      error = "lattice-image count overflows the host address space";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  if (origin_policy == LatticeOriginPolicy::kExclude) {
    --count;
  }

  try {
    std::vector<LatticeTranslation> created;
    if (count > created.max_size()) {
      error = "lattice-image count exceeds the vector implementation limit";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    created.reserve(count);
    if (origin_policy == LatticeOriginPolicy::kInclude) {
      created.push_back({});
    }

    for (std::int64_t first = -repeat[0]; first <= repeat[0]; ++first) {
      for (std::int64_t second = -repeat[1]; second <= repeat[1]; ++second) {
        for (std::int64_t third = -repeat[2]; third <= repeat[2]; ++third) {
          if (first == 0 && second == 0 && third == 0) {
            continue;
          }
          LatticeTranslation translation;
          if (!make_translation(lattice, first, second, third, translation)) {
            error = "lattice translation is outside the binary64 range";
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
          created.push_back(translation);
        }
      }
    }
    if (created.size() != count) {
      error = "lattice-image enumeration produced an inconsistent count";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    translations = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate lattice translations";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

}  // namespace xtbloom::detail::gfn2
