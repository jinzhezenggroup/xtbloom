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
long double wide_absolute(long double value) { return value < 0.0L ? -value : value; }

bool wide_finite(long double value) {
  const long double maximum = std::numeric_limits<long double>::max();
  return value == value && value <= maximum && value >= -maximum;
}

long double wide_scalbn(long double value, int exponent) {
  return static_cast<long double>(std::scalbn(static_cast<double>(value), exponent));
}

long double wide_nextafter(long double value, long double direction) {
  return static_cast<long double>(
      std::nextafter(static_cast<double>(value), static_cast<double>(direction)));
}

long double wide_square_root(long double value) {
  return static_cast<long double>(std::sqrt(static_cast<double>(value)));
}

long double wide_ceil(long double value) {
  return static_cast<long double>(std::ceil(static_cast<double>(value)));
}

/* A compact double-double accumulator supplies the inverse accuracy normally
 * obtained from an extended long double. MSVC intentionally aliases long
 * double to double, so relying on the language type would make Cartesian
 * wrapping and fractional recovery platform-dependent. */
struct DoubleDouble {
  double high = 0.0;
  double low = 0.0;
};

DoubleDouble two_sum(double first, double second) {
  const double high = first + second;
  const double second_virtual = high - first;
  const double low = (first - (high - second_virtual)) + (second - second_virtual);
  return {high, low};
}

DoubleDouble quick_two_sum(double first, double second) {
  const double high = first + second;
  return {high, second - (high - first)};
}

DoubleDouble add(DoubleDouble first, DoubleDouble second) {
  DoubleDouble result = two_sum(first.high, second.high);
  result.low += first.low + second.low;
  return quick_two_sum(result.high, result.low);
}

DoubleDouble subtract(DoubleDouble first, DoubleDouble second) {
  return add(first, {-second.high, -second.low});
}

DoubleDouble multiply(double first, double second) {
  const double high = first * second;
  return {high, std::fma(first, second, -high)};
}

DoubleDouble multiply(DoubleDouble value, double factor) {
  DoubleDouble result = multiply(value.high, factor);
  result.low += value.low * factor;
  return quick_two_sum(result.high, result.low);
}

DoubleDouble scale_double_double(DoubleDouble value, int exponent) {
  return {std::scalbn(value.high, exponent), std::scalbn(value.low, exponent)};
}

DoubleDouble dot_double_double(const std::array<DoubleDouble, 3>& first,
                               const std::array<double, 3>& second) {
  DoubleDouble result = multiply(first[0], second[0]);
  result = add(result, multiply(first[1], second[1]));
  return add(result, multiply(first[2], second[2]));
}

/* Return dot(coefficients, values) as mantissa * 2^exponent. Each product is
 * normalized independently before summation, so a vector spanning the full
 * binary64 exponent range does not discard the only component relevant to a
 * sparse reciprocal row. */
DoubleDouble scaled_dot_double_double(const std::array<DoubleDouble, 3>& coefficients,
                                      const double* values, int& exponent) {
  std::array<DoubleDouble, 3> products{};
  std::array<int, 3> product_exponents{};
  std::array<bool, 3> active{};
  exponent = std::numeric_limits<int>::min();
  for (std::size_t component = 0; component < 3u; ++component) {
    const double coefficient_magnitude =
        std::max(std::abs(coefficients[component].high),
                 std::abs(coefficients[component].low));
    if (!(coefficient_magnitude > 0.0) || values[component] == 0.0) continue;
    int coefficient_exponent = 0;
    int value_exponent = 0;
    (void)std::frexp(coefficient_magnitude, &coefficient_exponent);
    const double value_mantissa = std::frexp(values[component], &value_exponent);
    const DoubleDouble coefficient_mantissa =
        scale_double_double(coefficients[component], -coefficient_exponent);
    products[component] = multiply(coefficient_mantissa, value_mantissa);
    product_exponents[component] = coefficient_exponent + value_exponent;
    active[component] = true;
    exponent = std::max(exponent, product_exponents[component]);
  }
  if (exponent == std::numeric_limits<int>::min()) {
    exponent = 0;
    return {};
  }
  DoubleDouble result{};
  for (std::size_t component = 0; component < 3u; ++component) {
    if (active[component]) {
      result = add(result, scale_double_double(
                               products[component], product_exponents[component] - exponent));
    }
  }
  return result;
}

std::array<DoubleDouble, 3> cross_double_double(const std::array<double, 3>& first,
                                                const std::array<double, 3>& second) {
  return {
      subtract(multiply(first[1], second[2]), multiply(first[2], second[1])),
      subtract(multiply(first[2], second[0]), multiply(first[0], second[2])),
      subtract(multiply(first[0], second[1]), multiply(first[1], second[0])),
  };
}

double divide_double_double(DoubleDouble numerator, DoubleDouble denominator) {
  const double first = numerator.high / denominator.high;
  DoubleDouble remainder = subtract(numerator, multiply(denominator, first));
  const double second = (remainder.high + remainder.low) / denominator.high;
  remainder = subtract(remainder, multiply(denominator, second));
  const double third = (remainder.high + remainder.low) / denominator.high;
  const DoubleDouble leading = two_sum(first, second);
  return leading.high + (leading.low + third);
}

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

bool finite_vector(const double* value) {
  return value != nullptr && std::isfinite(value[0]) && std::isfinite(value[1]) &&
         std::isfinite(value[2]);
}

bool store_double(long double value, double& output, bool require_positive = false);

/* Store value * 2^exponent without ever materializing the power of two. This
 * keeps valid anisotropic cells representable even on platforms where
 * long double has the same exponent range as binary64. */
bool store_scaled_double(long double value, int exponent, double& output,
                         bool require_positive = false) {
  const long double scaled = wide_scalbn(value, exponent);
  return store_double(scaled, output, require_positive);
}

/* Store a positive binary64 lower bound for value * 2^exponent. Repeat counts
 * divide by this value, so nearest rounding is unsafe when it rounds a plane
 * spacing up: ceil(cutoff / spacing) could then omit a reachable outer image.
 * Scaling the rounded result back to the normalized exponent provides the
 * rounding-direction certificate even when long double is binary64. */
bool store_scaled_positive_lower_bound(long double value, int exponent, double& output) {
  if (!(value > 0.0L) || !wide_finite(value)) {
    return false;
  }
  const long double scaled = wide_scalbn(value, exponent);
  if (!(scaled > 0.0L) || !wide_finite(scaled) ||
      scaled > static_cast<long double>(std::numeric_limits<double>::max())) {
    return false;
  }
  double rounded = static_cast<double>(scaled);
  long double recovered = wide_scalbn(static_cast<long double>(rounded), -exponent);
  if (recovered > value) {
    rounded = std::nextafter(rounded, 0.0);
    recovered = wide_scalbn(static_cast<long double>(rounded), -exponent);
  }
  if (!(rounded > 0.0) || !std::isfinite(rounded) || !wide_finite(recovered) ||
      recovered > value) {
    return false;
  }
  output = rounded;
  return true;
}

struct LongDoubleInterval {
  long double lower = 0.0L;
  long double upper = 0.0L;
};

long double round_down(long double value) {
  return wide_nextafter(value, -std::numeric_limits<long double>::infinity());
}

long double round_up(long double value) {
  return wide_nextafter(value, std::numeric_limits<long double>::infinity());
}

LongDoubleInterval product_interval(long double lhs, long double rhs) {
  const long double product = lhs * rhs;
  if (product == 0.0L && (lhs == 0.0L || rhs == 0.0L)) {
    return {0.0L, 0.0L};
  }
  return {round_down(product), round_up(product)};
}

LongDoubleInterval subtract_interval(const LongDoubleInterval& lhs,
                                     const LongDoubleInterval& rhs) {
  return {round_down(lhs.lower - rhs.upper), round_up(lhs.upper - rhs.lower)};
}

LongDoubleInterval add_interval(const LongDoubleInterval& lhs,
                                const LongDoubleInterval& rhs) {
  return {round_down(lhs.lower + rhs.lower), round_up(lhs.upper + rhs.upper)};
}

LongDoubleInterval multiply_exact_interval(long double exact,
                                           const LongDoubleInterval& interval) {
  if (exact == 0.0L) return {0.0L, 0.0L};
  if (exact > 0.0L) {
    return {round_down(exact * interval.lower), round_up(exact * interval.upper)};
  }
  return {round_down(exact * interval.upper), round_up(exact * interval.lower)};
}

std::array<LongDoubleInterval, 3> cross_interval(const std::array<long double, 3>& first,
                                                 const std::array<long double, 3>& second) {
  return {
      subtract_interval(product_interval(first[1], second[2]),
                        product_interval(first[2], second[1])),
      subtract_interval(product_interval(first[2], second[0]),
                        product_interval(first[0], second[2])),
      subtract_interval(product_interval(first[0], second[1]),
                        product_interval(first[1], second[0])),
  };
}

LongDoubleInterval dot_exact_interval(
    const std::array<long double, 3>& exact,
    const std::array<LongDoubleInterval, 3>& interval) {
  LongDoubleInterval result = multiply_exact_interval(exact[0], interval[0]);
  result = add_interval(result, multiply_exact_interval(exact[1], interval[1]));
  return add_interval(result, multiply_exact_interval(exact[2], interval[2]));
}

long double norm_upper_bound(const std::array<LongDoubleInterval, 3>& vector) {
  long double squared_upper = 0.0L;
  for (const LongDoubleInterval& component : vector) {
    const long double magnitude =
        std::max(wide_absolute(component.lower), wide_absolute(component.upper));
    const long double square_upper = round_up(magnitude * magnitude);
    squared_upper = round_up(squared_upper + square_upper);
  }
  return round_up(wide_square_root(squared_upper));
}

struct ScaledLatticeDerivation {
  std::array<std::array<long double, 3>, 3> normalized{};
  std::array<int, 3> row_exponents{};
  std::array<std::array<long double, 3>, 3> cofactors{};
  long double determinant = 0.0L;
  std::array<double, 3> plane_spacing{};
  std::array<double, 9> reciprocal{};
  double volume = 0.0;
};

/* Normalize each row by an exact binary power. Per-row scaling is essential:
 * one global scale can underflow a short vector in a valid anisotropic cell.
 * Keeping the normalized components as binary64 also makes this derivation
 * independent of whether the host offers an extended long-double format. */
bool normalize_lattice_rows(const double* direct,
                            std::array<std::array<long double, 3>, 3>& normalized,
                            std::array<int, 3>& row_exponents) {
  for (std::size_t vector = 0; vector < 3u; ++vector) {
    double maximum = 0.0;
    for (std::size_t component = 0; component < 3u; ++component) {
      const double value = direct[vector * 3u + component];
      if (!std::isfinite(value)) return false;
      maximum = std::max(maximum, std::abs(value));
    }
    if (!(maximum > 0.0)) return false;
    int exponent = 0;
    (void)std::frexp(maximum, &exponent);
    row_exponents[vector] = exponent;
    for (std::size_t component = 0; component < 3u; ++component) {
      const double scaled = std::scalbn(direct[vector * 3u + component], -exponent);
      if (!std::isfinite(scaled)) return false;
      normalized[vector][component] = static_cast<long double>(scaled);
    }
  }
  return true;
}

/* Derive every stored lattice quantity from the same exponent-safe image.
 * Outward intervals certify the plane-spacing lower bounds, while frexp/scalbn
 * composition avoids overflow in a representable volume or reciprocal row. */
bool derive_lattice_geometry(const double* direct, ScaledLatticeDerivation& derived) {
  if (!normalize_lattice_rows(direct, derived.normalized, derived.row_exponents)) return false;
  derived.cofactors = {cross(derived.normalized[1], derived.normalized[2]),
                       cross(derived.normalized[2], derived.normalized[0]),
                       cross(derived.normalized[0], derived.normalized[1])};
  derived.determinant = dot(derived.normalized[0], derived.cofactors[0]);
  if (!(derived.determinant > 0.0L) || !wide_finite(derived.determinant)) return false;

  const std::array<std::array<LongDoubleInterval, 3>, 3> cofactor_intervals{
      cross_interval(derived.normalized[1], derived.normalized[2]),
      cross_interval(derived.normalized[2], derived.normalized[0]),
      cross_interval(derived.normalized[0], derived.normalized[1])};
  const LongDoubleInterval determinant_interval =
      dot_exact_interval(derived.normalized[0], cofactor_intervals[0]);
  if (!(determinant_interval.lower > 0.0L) ||
      !wide_finite(determinant_interval.lower)) {
    return false;
  }

  for (std::size_t vector = 0; vector < 3u; ++vector) {
    const long double cofactor_norm_upper = norm_upper_bound(cofactor_intervals[vector]);
    if (!(cofactor_norm_upper > 0.0L) || !wide_finite(cofactor_norm_upper)) return false;
    const long double spacing_lower =
        round_down(determinant_interval.lower / cofactor_norm_upper);
    if (!store_scaled_positive_lower_bound(spacing_lower, derived.row_exponents[vector],
                                           derived.plane_spacing[vector])) {
      return false;
    }
    for (std::size_t component = 0; component < 3u; ++component) {
      const long double normalized_reciprocal =
          (kTwoPi / derived.determinant) * derived.cofactors[vector][component];
      if (!store_scaled_double(normalized_reciprocal, -derived.row_exponents[vector],
                               derived.reciprocal[vector * 3u + component])) {
        return false;
      }
    }
  }

  const int volume_exponent =
      derived.row_exponents[0] + derived.row_exponents[1] + derived.row_exponents[2];
  return store_scaled_double(derived.determinant, volume_exponent, derived.volume, true);
}

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

  ScaledLatticeDerivation derived;
  if (!derive_lattice_geometry(lattice.direct.data(), derived) ||
      derived.volume != lattice.volume) {
    return false;
  }
  for (std::size_t element = 0; element < derived.reciprocal.size(); ++element) {
    if (derived.reciprocal[element] != lattice.reciprocal[element]) {
      return false;
    }
  }
  for (std::size_t vector = 0; vector < derived.plane_spacing.size(); ++vector) {
    if (derived.plane_spacing[vector] != lattice.plane_spacing[vector]) {
      return false;
    }
  }
  return true;
}

bool store_double(long double value, double& output, bool require_positive) {
  if (!wide_finite(value) ||
      wide_absolute(value) > static_cast<long double>(std::numeric_limits<double>::max()) ||
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
  const long double rounded = wide_ceil(ratio);
  if (!wide_finite(rounded) || rounded < 0.0L ||
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

  for (std::size_t element = 0; element < 9u; ++element) {
    if (!std::isfinite(direct[element])) {
      error = "direct lattice contains NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  if (!valid_lattice_cell_3d(direct)) {
    std::array<std::array<long double, 3>, 3> normalized{};
    std::array<int, 3> row_exponents{};
    const long double determinant =
        normalize_lattice_rows(direct, normalized, row_exponents)
            ? dot(normalized[0], cross(normalized[1], normalized[2]))
            : 0.0L;
    error = determinant < 0.0L ? "direct lattice must be right-handed"
                               : "direct lattice is singular or numerically ill-conditioned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  ScaledLatticeDerivation derived;
  if (!derive_lattice_geometry(direct, derived)) {
    error = "lattice geometry is outside the supported binary64 range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  Lattice3D created;
  std::copy_n(direct, 9u, created.direct.begin());
  created.plane_spacing = derived.plane_spacing;
  created.reciprocal = derived.reciprocal;
  created.volume = derived.volume;

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

  ScaledLatticeDerivation derived;
  if (!derive_lattice_geometry(lattice.direct.data(), derived)) {
    error = "lattice inverse could not be derived in binary64 range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::array<double, 3> result{};
  std::array<std::array<double, 3>, 3> normalized_direct{};
  for (std::size_t vector = 0; vector < 3u; ++vector) {
    for (std::size_t component = 0; component < 3u; ++component) {
      normalized_direct[vector][component] =
          static_cast<double>(derived.normalized[vector][component]);
    }
  }
  const std::array<std::array<DoubleDouble, 3>, 3> cofactor_double_double{
      cross_double_double(normalized_direct[1], normalized_direct[2]),
      cross_double_double(normalized_direct[2], normalized_direct[0]),
      cross_double_double(normalized_direct[0], normalized_direct[1])};
  const DoubleDouble determinant_double_double =
      dot_double_double(cofactor_double_double[0], normalized_direct[0]);
  for (std::size_t vector = 0; vector < 3u; ++vector) {
    int numerator_exponent = 0;
    const DoubleDouble numerator =
        scaled_dot_double_double(cofactor_double_double[vector], cartesian, numerator_exponent);
    const double normalized_fractional =
        divide_double_double(numerator, determinant_double_double);
    /* Reconstruct the inverse directly from the immutable direct cell. Going
     * through the stored 2*pi reciprocal and dividing by 2*pi adds two
     * binary64 rounding steps; for an exact lattice translation those steps
     * can turn integer 1 into nextafter(1, 0), defeating canonical wrapping. */
    if (!store_scaled_double(normalized_fractional,
                             numerator_exponent - derived.row_exponents[vector],
                             result[vector])) {
      error = "Cartesian-to-fractional conversion overflowed binary64";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  /* Certify an exactly representable forward-transform preimage. The inverse
   * of a rounded Cartesian binary64 vector can straddle the nearest fractional
   * value by one ULP even with double-double arithmetic. Search only the three
   * immediate binary64 candidates per coordinate and accept a triplet solely
   * when the public forward transform reconstructs every Cartesian component
   * bit-for-bit; this is not an epsilon snap. */
  std::array<std::array<double, 5>, 3> fractional_candidates{};
  for (std::size_t component = 0; component < 3u; ++component) {
    const double lower =
        std::nextafter(result[component], -std::numeric_limits<double>::infinity());
    const double upper =
        std::nextafter(result[component], std::numeric_limits<double>::infinity());
    fractional_candidates[component] = {
        std::nextafter(lower, -std::numeric_limits<double>::infinity()),
        lower,
        result[component],
        upper,
        std::nextafter(upper, std::numeric_limits<double>::infinity()),
    };
  }
  bool certified = false;
  for (std::size_t first = 0; first < 5u && !certified; ++first) {
    for (std::size_t second = 0; second < 5u && !certified; ++second) {
      for (std::size_t third = 0; third < 5u; ++third) {
        const std::array<double, 3> candidate{fractional_candidates[0][first],
                                               fractional_candidates[1][second],
                                               fractional_candidates[2][third]};
        std::array<double, 3> reconstructed{};
        bool matches = true;
        for (std::size_t component = 0; component < 3u; ++component) {
          const long double value =
              static_cast<long double>(candidate[0]) * lattice.direct[component] +
              static_cast<long double>(candidate[1]) * lattice.direct[3u + component] +
              static_cast<long double>(candidate[2]) * lattice.direct[6u + component];
          if (!store_double(value, reconstructed[component]) ||
              reconstructed[component] != cartesian[component]) {
            matches = false;
            break;
          }
        }
        if (matches) {
          result = candidate;
          certified = true;
          break;
        }
      }
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
  /* A rounded inverse can place an exact integer lattice translation just
   * below or above an integer. Snap only when an integer triplet reconstructed
   * through the same public forward transform reproduces the supplied
   * Cartesian binary64 vector exactly. This certificate fixes integer
   * translations without consuming a genuine nextafter(1, 0) coordinate. */
  std::array<std::array<double, 2>, 3> integer_candidates{};
  std::array<std::size_t, 3> candidate_counts{};
  for (std::size_t component = 0; component < 3u; ++component) {
    const double lower = std::floor(fractional[component]);
    const double upper = std::ceil(fractional[component]);
    integer_candidates[component][0] = lower;
    candidate_counts[component] = 1u;
    if (upper != lower) {
      integer_candidates[component][1] = upper;
      candidate_counts[component] = 2u;
    }
  }
  bool certified_integer = false;
  for (std::size_t first = 0; first < candidate_counts[0] && !certified_integer; ++first) {
    for (std::size_t second = 0; second < candidate_counts[1] && !certified_integer; ++second) {
      for (std::size_t third = 0; third < candidate_counts[2]; ++third) {
        const std::array<double, 3> candidate{integer_candidates[0][first],
                                               integer_candidates[1][second],
                                               integer_candidates[2][third]};
        std::array<double, 3> reconstructed{};
        std::string reconstruction_error;
        if (fractional_to_cartesian(lattice, candidate.data(), reconstructed.data(),
                                    reconstruction_error) == XTBLOOM_STATUS_SUCCESS &&
            reconstructed[0] == cartesian[0] && reconstructed[1] == cartesian[1] &&
            reconstructed[2] == cartesian[2]) {
          fractional = candidate;
          certified_integer = true;
          break;
        }
      }
    }
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
