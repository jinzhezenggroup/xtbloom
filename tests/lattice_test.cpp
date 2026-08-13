// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/lattice.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559005768;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

bool same_translation(const xtbloom::detail::gfn2::LatticeTranslation& first,
                      const xtbloom::detail::gfn2::LatticeTranslation& second) {
  return first.index == second.index && first.cartesian == second.cartesian;
}

double direct_reciprocal_dot(const xtbloom::detail::gfn2::Lattice3D& lattice, std::size_t direct,
                             std::size_t reciprocal) {
  double result = 0.0;
  for (std::size_t component = 0; component < 3u; ++component) {
    result +=
        lattice.direct[direct * 3u + component] * lattice.reciprocal[reciprocal * 3u + component];
  }
  return result;
}

int test_orthogonal_geometry_and_wrap() {
  constexpr std::array<double, 9> direct{
      2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0,
  };
  xtbloom::detail::gfn2::Lattice3D lattice;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(direct.data(), lattice, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());
  CHECK(lattice.direct == direct);
  CHECK(lattice.volume == 24.0);
  const std::array<double, 3> exact_spacing{2.0, 3.0, 4.0};
  for (std::size_t vector = 0; vector < 3u; ++vector) {
    CHECK(lattice.plane_spacing[vector] <= exact_spacing[vector]);
    CHECK(near(lattice.plane_spacing[vector], exact_spacing[vector], 8.0e-15));
  }
  CHECK(near(lattice.reciprocal[0], kTwoPi / 2.0, 5.0e-16));
  CHECK(near(lattice.reciprocal[4], kTwoPi / 3.0, 5.0e-16));
  CHECK(near(lattice.reciprocal[8], kTwoPi / 4.0, 5.0e-16));

  constexpr std::array<double, 3> fractional{0.25, -0.5, 1.25};
  std::array<double, 3> cartesian{};
  CHECK(xtbloom::detail::gfn2::fractional_to_cartesian(lattice, fractional.data(), cartesian.data(),
                                                       error) == XTBLOOM_STATUS_SUCCESS);
  CHECK((cartesian == std::array<double, 3>{0.5, -1.5, 5.0}));
  std::array<double, 3> recovered{};
  CHECK(xtbloom::detail::gfn2::cartesian_to_fractional(lattice, cartesian.data(), recovered.data(),
                                                       error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t component = 0; component < 3u; ++component) {
    CHECK(near(recovered[component], fractional[component], 2.0e-16));
  }

  std::array<double, 3> wrapped_fractional{};
  CHECK(xtbloom::detail::gfn2::wrap_fractional(fractional.data(), wrapped_fractional.data(),
                                               error) == XTBLOOM_STATUS_SUCCESS);
  CHECK((wrapped_fractional == std::array<double, 3>{0.25, 0.5, 0.25}));
  std::array<double, 3> wrapped_cartesian{};
  CHECK(xtbloom::detail::gfn2::wrap_cartesian(lattice, cartesian.data(), wrapped_cartesian.data(),
                                              error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(wrapped_cartesian[0], 0.5, 2.0e-16));
  CHECK(near(wrapped_cartesian[1], 1.5, 2.0e-16));
  CHECK(near(wrapped_cartesian[2], 1.0, 2.0e-16));

  std::array<double, 3> alias{1.0, -0.0, -2.25};
  CHECK(xtbloom::detail::gfn2::wrap_fractional(alias.data(), alias.data(), error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK((alias == std::array<double, 3>{0.0, 0.0, 0.75}));
  CHECK(!std::signbit(alias[0]) && !std::signbit(alias[1]));
  return 0;
}

int test_skew_geometry_and_translation_invariance() {
  constexpr std::array<double, 9> direct{
      2.0, 0.0, 0.0, 0.5, 1.5, 0.0, -0.2, 0.4, 1.2,
  };
  xtbloom::detail::gfn2::Lattice3D lattice;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(direct.data(), lattice, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(near(lattice.volume, 3.6, 8.0e-16));
  for (std::size_t first = 0; first < 3u; ++first) {
    for (std::size_t second = 0; second < 3u; ++second) {
      CHECK(near(direct_reciprocal_dot(lattice, first, second), first == second ? kTwoPi : 0.0,
                 1.0e-15));
    }
  }

  constexpr std::array<double, 3> fractional{0.13, 0.72, 0.41};
  std::array<double, 3> cartesian{};
  CHECK(xtbloom::detail::gfn2::fractional_to_cartesian(lattice, fractional.data(), cartesian.data(),
                                                       error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 3> recovered{};
  CHECK(xtbloom::detail::gfn2::cartesian_to_fractional(lattice, cartesian.data(), recovered.data(),
                                                       error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t component = 0; component < 3u; ++component) {
    CHECK(near(recovered[component], fractional[component], 3.0e-16));
  }

  /* Add 3*a - 2*b + 4*c and require the same canonical representative. */
  std::array<double, 3> translated = cartesian;
  for (std::size_t component = 0; component < 3u; ++component) {
    translated[component] +=
        3.0 * direct[component] - 2.0 * direct[3u + component] + 4.0 * direct[6u + component];
  }
  std::array<double, 3> wrapped_original{};
  std::array<double, 3> wrapped_translated{};
  CHECK(xtbloom::detail::gfn2::wrap_cartesian(lattice, cartesian.data(), wrapped_original.data(),
                                              error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::wrap_cartesian(lattice, translated.data(), wrapped_translated.data(),
                                              error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t component = 0; component < 3u; ++component) {
    CHECK(near(wrapped_translated[component], wrapped_original[component], 1.5e-15));
  }
  return 0;
}

int test_canonical_translation_order_and_origin_policy() {
  constexpr std::array<double, 9> direct{
      2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0,
  };
  xtbloom::detail::gfn2::Lattice3D lattice;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(direct.data(), lattice, error) ==
        XTBLOOM_STATUS_SUCCESS);

  std::vector<xtbloom::detail::gfn2::LatticeTranslation> included;
  CHECK(xtbloom::detail::gfn2::make_lattice_translations(
            lattice, 3.1, xtbloom::detail::gfn2::LatticeOriginPolicy::kInclude, included, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(included.size() == 75u);
  CHECK((included.front().index == std::array<std::int64_t, 3>{0, 0, 0}));
  CHECK((included.front().cartesian == std::array<double, 3>{0.0, 0.0, 0.0}));
  CHECK((included[1].index == std::array<std::int64_t, 3>{-2, -2, -1}));
  CHECK((included[1].cartesian == std::array<double, 3>{-4.0, -6.0, -4.0}));
  CHECK((included.back().index == std::array<std::int64_t, 3>{2, 2, 1}));

  std::vector<xtbloom::detail::gfn2::LatticeTranslation> excluded;
  CHECK(xtbloom::detail::gfn2::make_lattice_translations(
            lattice, 3.1, xtbloom::detail::gfn2::LatticeOriginPolicy::kExclude, excluded, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(excluded.size() == 74u);
  CHECK(excluded.front().index == included[1].index);
  CHECK(excluded.back().index == included.back().index);
  CHECK(std::none_of(excluded.begin(), excluded.end(), [](const auto& translation) {
    return translation.index == std::array<std::int64_t, 3>{0, 0, 0};
  }));

  std::vector<xtbloom::detail::gfn2::LatticeTranslation> repeated;
  CHECK(xtbloom::detail::gfn2::make_lattice_translations(
            lattice, 3.1, xtbloom::detail::gfn2::LatticeOriginPolicy::kInclude, repeated, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(std::equal(included.begin(), included.end(), repeated.begin(), repeated.end(),
                   same_translation));

  CHECK(xtbloom::detail::gfn2::make_lattice_translations(
            lattice, 0.0, xtbloom::detail::gfn2::LatticeOriginPolicy::kInclude, repeated, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK((repeated.size() == 1u && repeated[0].index == std::array<std::int64_t, 3>{0, 0, 0}));
  CHECK(xtbloom::detail::gfn2::make_lattice_translations(
            lattice, 0.0, xtbloom::detail::gfn2::LatticeOriginPolicy::kExclude, repeated, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(repeated.empty());
  return 0;
}

bool contains_index(const std::vector<xtbloom::detail::gfn2::LatticeTranslation>& translations,
                    const std::array<std::int64_t, 3>& index) {
  return std::any_of(translations.begin(), translations.end(),
                     [&](const auto& translation) { return translation.index == index; });
}

int test_small_skew_cell_cutoff_completeness() {
  /* A small, strongly skewed but well-conditioned cell stresses plane bounds. */
  constexpr std::array<double, 9> direct{
      0.20, 0.00, 0.00, 0.17, 0.16, 0.00, 0.13, 0.07, 0.18,
  };
  xtbloom::detail::gfn2::Lattice3D lattice;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(direct.data(), lattice, error) ==
        XTBLOOM_STATUS_SUCCESS);
  constexpr double cutoff = 0.61;
  std::vector<xtbloom::detail::gfn2::LatticeTranslation> translations;
  CHECK(xtbloom::detail::gfn2::make_lattice_translations(
            lattice, cutoff, xtbloom::detail::gfn2::LatticeOriginPolicy::kInclude, translations,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(!translations.empty());

  std::array<std::int64_t, 3> repeat{};
  for (std::size_t vector = 0; vector < 3u; ++vector) {
    repeat[vector] = static_cast<std::int64_t>(std::ceil(cutoff / lattice.plane_spacing[vector]));
  }
  const std::size_t expected = static_cast<std::size_t>(2 * repeat[0] + 1) *
                               static_cast<std::size_t>(2 * repeat[1] + 1) *
                               static_cast<std::size_t>(2 * repeat[2] + 1);
  CHECK(translations.size() == expected);

  /*
   * Brute-force nearby indices and representative wrapped point differences.
   * Every image capable of producing a separation within cutoff must occur in
   * the rectangular image set, including images whose bare norm exceeds it.
   */
  constexpr std::array<double, 5> differences{-0.999, -0.5, 0.0, 0.5, 0.999};
  for (std::int64_t first = -repeat[0] - 2; first <= repeat[0] + 2; ++first) {
    for (std::int64_t second = -repeat[1] - 2; second <= repeat[1] + 2; ++second) {
      for (std::int64_t third = -repeat[2] - 2; third <= repeat[2] + 2; ++third) {
        bool can_reach_cutoff = false;
        for (double dx : differences) {
          for (double dy : differences) {
            for (double dz : differences) {
              const std::array<double, 3> fractional{
                  static_cast<double>(first) + dx,
                  static_cast<double>(second) + dy,
                  static_cast<double>(third) + dz,
              };
              std::array<double, 3> cartesian{};
              CHECK(xtbloom::detail::gfn2::fractional_to_cartesian(lattice, fractional.data(),
                                                                   cartesian.data(), error) ==
                    XTBLOOM_STATUS_SUCCESS);
              const double distance = std::hypot(cartesian[0], cartesian[1], cartesian[2]);
              can_reach_cutoff = can_reach_cutoff || distance <= cutoff;
            }
          }
        }
        if (can_reach_cutoff) {
          CHECK(contains_index(translations, {first, second, third}));
        }
      }
    }
  }
  return 0;
}

int test_rounding_boundary_image_completeness() {
  /* Nearest rounding raises the first plane spacing for this cell. Using that
   * rounded-up value directly makes ceil(cutoff / spacing) equal four even
   * though the fifth image can bring two wrapped points inside the cutoff. */
  constexpr std::array<double, 9> direct{
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.01, 0.0, 1.0,
  };
  constexpr double cutoff = 3.999800014998750263828;
  xtbloom::detail::gfn2::Lattice3D lattice;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(direct.data(), lattice, error) ==
        XTBLOOM_STATUS_SUCCESS);
  const long double exact_spacing =
      1.0L / std::hypot(static_cast<long double>(direct[6]), static_cast<long double>(direct[8]));
  CHECK(static_cast<long double>(lattice.plane_spacing[0]) <= exact_spacing);

  std::vector<xtbloom::detail::gfn2::LatticeTranslation> translations;
  CHECK(xtbloom::detail::gfn2::make_lattice_translations(
            lattice, cutoff, xtbloom::detail::gfn2::LatticeOriginPolicy::kInclude, translations,
            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(contains_index(translations, {5, 0, 0}));

  const long double dx = static_cast<long double>(std::nextafter(-1.0, 0.0));
  constexpr long double dz = -0.03999600039996000638842L;
  const long double x =
      (5.0L + dx) * static_cast<long double>(direct[0]) + dz * static_cast<long double>(direct[6]);
  const long double z = dz * static_cast<long double>(direct[8]);
  CHECK(std::hypot(x, z) <= static_cast<long double>(cutoff));
  return 0;
}

int test_integer_cartesian_translations_wrap_canonically() {
  constexpr std::array<double, 9> direct{
      0.79979393826591227, -0.27902125012705659, -1.9036126812397094,
      -1.9438839310276417, -0.55375873092674577, -1.12666756478776,
      -0.3563602655153677, 1.2829454330487802,   -0.74540186630715821,
  };
  xtbloom::detail::gfn2::Lattice3D lattice;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(direct.data(), lattice, error) ==
        XTBLOOM_STATUS_SUCCESS);

  for (std::size_t vector = 0; vector < 3u; ++vector) {
    std::array<double, 3> wrapped{};
    CHECK(xtbloom::detail::gfn2::wrap_cartesian(lattice, direct.data() + vector * 3u,
                                                wrapped.data(), error) == XTBLOOM_STATUS_SUCCESS);
    CHECK((wrapped == std::array<double, 3>{0.0, 0.0, 0.0}));
  }

  /* Improving the inverse must not introduce an epsilon snap that erases a
   * genuine fractional coordinate immediately below the cell boundary. */
  const std::array<double, 3> fractional{
      0.125,
      std::nextafter(1.0, 0.0),
      0.375,
  };
  std::array<double, 3> cartesian{};
  std::array<double, 3> recovered{};
  CHECK(xtbloom::detail::gfn2::fractional_to_cartesian(lattice, fractional.data(), cartesian.data(),
                                                       error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::cartesian_to_fractional(lattice, cartesian.data(), recovered.data(),
                                                       error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t component = 0; component < 3u; ++component) {
    CHECK(near(recovered[component], fractional[component], 2.0e-16));
  }
  return 0;
}

int test_validation_and_transactional_outputs() {
  constexpr std::array<double, 9> valid{
      2.0, 0.0, 0.0, 0.3, 1.7, 0.0, -0.2, 0.4, 1.3,
  };
  xtbloom::detail::gfn2::Lattice3D lattice;
  lattice.volume = 17.0;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(nullptr, lattice, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(lattice.volume == 17.0);

  std::array<double, 9> invalid = valid;
  invalid[4] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(invalid.data(), lattice, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  invalid = valid;
  invalid[8] = std::numeric_limits<double>::infinity();
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(invalid.data(), lattice, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

  constexpr std::array<double, 9> singular{
      1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0,
  };
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(singular.data(), lattice, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  constexpr std::array<double, 9> ill_conditioned{
      1.0, 0.0, 0.0, 1.0, 1.0e-16, 0.0, 0.0, 0.0, 1.0,
  };
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(ill_conditioned.data(), lattice, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  constexpr std::array<double, 9> left_handed{
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
  };
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(left_handed.data(), lattice, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

  CHECK(xtbloom::detail::gfn2::make_lattice_3d(valid.data(), lattice, error) ==
        XTBLOOM_STATUS_SUCCESS);
  constexpr std::array<double, 3> finite{0.1, 0.2, 0.3};
  std::array<double, 3> output{7.0, 8.0, 9.0};
  CHECK(xtbloom::detail::gfn2::fractional_to_cartesian(lattice, nullptr, output.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK((output == std::array<double, 3>{7.0, 8.0, 9.0}));
  CHECK(xtbloom::detail::gfn2::cartesian_to_fractional(lattice, finite.data(), nullptr, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  std::array<double, 3> nonfinite = finite;
  nonfinite[1] = std::numeric_limits<double>::infinity();
  CHECK(xtbloom::detail::gfn2::wrap_fractional(nonfinite.data(), output.data(), error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK((output == std::array<double, 3>{7.0, 8.0, 9.0}));

  std::vector<xtbloom::detail::gfn2::LatticeTranslation> translations(1u);
  translations[0].index = {7, 8, 9};
  CHECK(xtbloom::detail::gfn2::make_lattice_translations(
            lattice, -1.0, xtbloom::detail::gfn2::LatticeOriginPolicy::kInclude, translations,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(translations.size() == 1u && translations[0].index[0] == 7);
  CHECK(xtbloom::detail::gfn2::make_lattice_translations(
            lattice, std::numeric_limits<double>::infinity(),
            xtbloom::detail::gfn2::LatticeOriginPolicy::kInclude, translations,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_lattice_translations(
            lattice, 1.0, static_cast<xtbloom::detail::gfn2::LatticeOriginPolicy>(42), translations,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(translations.size() == 1u && translations[0].index[0] == 7);

  xtbloom::detail::gfn2::Lattice3D corrupted = lattice;
  corrupted.reciprocal[0] =
      std::nextafter(corrupted.reciprocal[0], std::numeric_limits<double>::infinity());
  CHECK(xtbloom::detail::gfn2::fractional_to_cartesian(corrupted, finite.data(), output.data(),
                                                       error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  corrupted = lattice;
  corrupted.volume = std::nextafter(corrupted.volume, std::numeric_limits<double>::infinity());
  CHECK(xtbloom::detail::gfn2::make_lattice_translations(
            corrupted, 1.0, xtbloom::detail::gfn2::LatticeOriginPolicy::kInclude, translations,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(translations.size() == 1u && translations[0].index[0] == 7);
  return 0;
}

int test_binary64_scale_range() {
  /* Normalization keeps well-shaped cells valid across a broad exponent range. */
  std::string error;
  for (double length : {1.0e-100, 1.0e100}) {
    const std::array<double, 9> direct{
        length, 0.0, 0.0, 0.0, length, 0.0, 0.0, 0.0, length,
    };
    xtbloom::detail::gfn2::Lattice3D lattice;
    CHECK(xtbloom::detail::gfn2::make_lattice_3d(direct.data(), lattice, error) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(lattice.volume > 0.0 && std::isfinite(lattice.volume));
    constexpr std::array<double, 3> fractional{0.125, 0.5, 0.875};
    std::array<double, 3> cartesian{};
    std::array<double, 3> recovered{};
    CHECK(xtbloom::detail::gfn2::fractional_to_cartesian(
              lattice, fractional.data(), cartesian.data(), error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom::detail::gfn2::cartesian_to_fractional(
              lattice, cartesian.data(), recovered.data(), error) == XTBLOOM_STATUS_SUCCESS);
    for (std::size_t component = 0; component < 3u; ++component) {
      CHECK(near(recovered[component], fractional[component], 3.0e-16));
    }
  }

  /* These derived quantities are all representable, despite the direct rows
   * spanning 400 decimal orders. A global normalization loses the short row
   * when long double has only the binary64 exponent range. */
  constexpr std::array<double, 9> anisotropic{
      1.0e200, 0.0, 0.0, 0.0, 1.0e200, 0.0, 0.0, 0.0, 1.0e-200,
  };
  xtbloom::detail::gfn2::Lattice3D anisotropic_lattice;
  CHECK(xtbloom::detail::gfn2::valid_lattice_cell_3d(anisotropic.data()));
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(anisotropic.data(), anisotropic_lattice, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(near(anisotropic_lattice.volume / 1.0e200, 1.0, 8.0e-16));
  CHECK(anisotropic_lattice.plane_spacing[0] <= 1.0e200);
  CHECK(anisotropic_lattice.plane_spacing[1] <= 1.0e200);
  CHECK(anisotropic_lattice.plane_spacing[2] <= 1.0e-200);
  CHECK(std::isfinite(anisotropic_lattice.reciprocal[8]));
  constexpr std::array<double, 3> fractional{0.125, 0.5, 0.875};
  std::array<double, 3> cartesian{};
  std::array<double, 3> recovered{};
  CHECK(xtbloom::detail::gfn2::fractional_to_cartesian(anisotropic_lattice, fractional.data(),
                                                       cartesian.data(),
                                                       error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::cartesian_to_fractional(anisotropic_lattice, cartesian.data(),
                                                       recovered.data(),
                                                       error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t component = 0; component < 3u; ++component) {
    CHECK(near(recovered[component], fractional[component], 4.0e-16));
  }

  /* Large, mildly skew rows exercise cancellation in the normalized
   * determinant without overflowing the unscaled cofactor products. */
  constexpr double scale = 1.0e106;
  constexpr double epsilon = 3.0e-7;
  constexpr std::array<double, 9> large_skew{
      scale,
      scale,
      scale,
      scale,
      scale * (1.0 + epsilon),
      scale,
      scale,
      scale,
      scale * (1.0 + epsilon),
  };
  xtbloom::detail::gfn2::Lattice3D large_skew_lattice;
  CHECK(xtbloom::detail::gfn2::valid_lattice_cell_3d(large_skew.data()));
  CHECK(xtbloom::detail::gfn2::make_lattice_3d(large_skew.data(), large_skew_lattice, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(large_skew_lattice.volume > 0.0 && std::isfinite(large_skew_lattice.volume));
  CHECK(std::all_of(large_skew_lattice.plane_spacing.begin(),
                    large_skew_lattice.plane_spacing.end(),
                    [](double value) { return value > 0.0 && std::isfinite(value); }));
  return 0;
}

int test_binary64_conditioning_boundary() {
  /* This descriptor exposed the former long-double host versus binary64
   * device split. The contract now uses one shared binary64 predicate and
   * accepts it consistently. */
  constexpr std::array<double, 9> former_divergence_probe{
      1.0, 0.0, 0.0, 1.0, 2.0097183471152322e-14, 0.0, 0.0, 0.0, 1.0,
  };
  CHECK(xtbloom::detail::gfn2::valid_lattice_cell_3d(former_divergence_probe.data()));

  std::array<double, 9> rejected{
      1.0, 0.0, 0.0, 1.0, 0x1p-46, 0.0, 0.0, 0.0, 1.0,
  };
  CHECK(!xtbloom::detail::gfn2::valid_lattice_cell_3d(rejected.data()));
  std::array<double, 9> accepted = rejected;
  accepted[4] = std::nextafter(rejected[4], std::numeric_limits<double>::infinity());
  CHECK(xtbloom::detail::gfn2::valid_lattice_cell_3d(accepted.data()));
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_orthogonal_geometry_and_wrap(); line != 0) {
    return line;
  }
  if (const int line = test_skew_geometry_and_translation_invariance(); line != 0) {
    return line;
  }
  if (const int line = test_canonical_translation_order_and_origin_policy(); line != 0) {
    return line;
  }
  if (const int line = test_small_skew_cell_cutoff_completeness(); line != 0) {
    return line;
  }
  if (const int line = test_rounding_boundary_image_completeness(); line != 0) {
    return line;
  }
  if (const int line = test_integer_cartesian_translations_wrap_canonically(); line != 0) {
    return line;
  }
  if (const int line = test_validation_and_transactional_outputs(); line != 0) {
    return line;
  }
  if (const int line = test_binary64_scale_range(); line != 0) {
    return line;
  }
  if (const int line = test_binary64_conditioning_boundary(); line != 0) {
    return line;
  }
  return 0;
}
