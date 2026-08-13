// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_MODEL_GFN2_LATTICE_HPP
#define XTBLOOM_MODEL_GFN2_LATTICE_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

#if defined(__CUDACC__)
#define XTBLOOM_LATTICE_HD __host__ __device__
#else
#define XTBLOOM_LATTICE_HD
#endif

namespace lattice_binary64_detail {

/*
 * Keep the public cell predicate bit-for-bit reproducible in ordinary C++ and
 * CUDA device code.  Volatile temporaries are intentional: they prevent a
 * backend from contracting the determinant into FMAs or retaining wider
 * intermediates at the acceptance boundary.
 */
XTBLOOM_LATTICE_HD inline double rounded_add(double lhs, double rhs) noexcept {
  volatile double result = lhs + rhs;
  return result;
}

XTBLOOM_LATTICE_HD inline double rounded_subtract(double lhs, double rhs) noexcept {
  volatile double result = lhs - rhs;
  return result;
}

XTBLOOM_LATTICE_HD inline double rounded_multiply(double lhs, double rhs) noexcept {
  volatile double result = lhs * rhs;
  return result;
}

XTBLOOM_LATTICE_HD inline double rounded_divide(double lhs, double rhs) noexcept {
  volatile double result = lhs / rhs;
  return result;
}

XTBLOOM_LATTICE_HD inline double rounded_square_root(double value) noexcept {
#if defined(__CUDA_ARCH__)
  volatile double result = sqrt(value);
#else
  volatile double result = std::sqrt(value);
#endif
  return result;
}

XTBLOOM_LATTICE_HD inline double absolute(double value) noexcept {
  return value < 0.0 ? -value : value;
}

XTBLOOM_LATTICE_HD inline bool finite(double value) noexcept {
  constexpr double maximum = 1.79769313486231570814527423731704357e308;
  return value == value && value <= maximum && value >= -maximum;
}

XTBLOOM_LATTICE_HD inline double squared_norm(const double* row) noexcept {
  return rounded_add(
      rounded_add(rounded_multiply(row[0], row[0]), rounded_multiply(row[1], row[1])),
      rounded_multiply(row[2], row[2]));
}

}  // namespace lattice_binary64_detail

/*
 * Binary64-only form of the scale-aware cell predicate shared by host
 * validation and the stream-ordered CUDA request gate.  Each row is first
 * scaled by its largest component and normalized to unit length before
 * evaluating det(a,b,c) > 64 epsilon. The explicit rounded operations above
 * make cells on the threshold receive one stable answer on every released
 * backend.
 */
XTBLOOM_LATTICE_HD inline bool valid_lattice_cell_3d_binary64(const double* direct) noexcept {
  using namespace lattice_binary64_detail;
  if (direct == nullptr) return false;

  double normalized[9]{};
  for (int row_index = 0; row_index < 3; ++row_index) {
    const double* source = direct + row_index * 3;
    double maximum = 0.0;
    for (int component = 0; component < 3; ++component) {
      if (!finite(source[component])) return false;
      const double magnitude = absolute(source[component]);
      if (magnitude > maximum) maximum = magnitude;
    }
    if (!(maximum > 0.0) || !finite(maximum)) return false;
    for (int component = 0; component < 3; ++component) {
      normalized[row_index * 3 + component] = rounded_divide(source[component], maximum);
    }
    const double row_norm = rounded_square_root(squared_norm(normalized + row_index * 3));
    if (!(row_norm > 0.0) || !finite(row_norm)) return false;
    for (int component = 0; component < 3; ++component) {
      normalized[row_index * 3 + component] =
          rounded_divide(normalized[row_index * 3 + component], row_norm);
    }
  }

  const double cross_x = rounded_subtract(rounded_multiply(normalized[4], normalized[8]),
                                          rounded_multiply(normalized[5], normalized[7]));
  const double cross_y = rounded_subtract(rounded_multiply(normalized[5], normalized[6]),
                                          rounded_multiply(normalized[3], normalized[8]));
  const double cross_z = rounded_subtract(rounded_multiply(normalized[3], normalized[7]),
                                          rounded_multiply(normalized[4], normalized[6]));
  const double determinant = rounded_add(rounded_add(rounded_multiply(normalized[0], cross_x),
                                                     rounded_multiply(normalized[1], cross_y)),
                                         rounded_multiply(normalized[2], cross_z));
  if (!(determinant > 0.0) || !finite(determinant)) return false;

  /* 64 * DBL_EPSILON = 2^-46 exactly. */
  return determinant > 0x1p-46;
}

/*
 * Backend-neutral three-dimensional lattice geometry. Direct and reciprocal
 * vectors are stored as three row-major xyz vectors. The reciprocal vectors
 * include 2*pi, so direct[i] dot reciprocal[j] = 2*pi delta(i,j). Volume and
 * all vector components use atomic units.
 */
struct Lattice3D {
  std::array<double, 9> direct{};
  std::array<double, 9> reciprocal{};
  /* Conservative binary64 lower bounds used for complete image enumeration. */
  std::array<double, 3> plane_spacing{};
  double volume = 0.0;
};

/* Integer lattice coordinates and their Cartesian translation in bohr. */
struct LatticeTranslation {
  std::array<std::int64_t, 3> index{};
  std::array<double, 3> cartesian{};
};

/*
 * Shared scale-aware predicate for the public ABI and internal lattice
 * construction. The determinant is evaluated relative to the product of the
 * three direct-vector norms, making the acceptance contract invariant under
 * independent changes of the lattice-vector lengths.
 */
bool valid_lattice_cell_3d(const double* direct) noexcept;

enum class LatticeOriginPolicy : std::int32_t {
  kInclude = 0,
  kExclude = 1,
};

/*
 * Construct a right-handed, nonsingular three-dimensional lattice from nine
 * row-major doubles. The output is unchanged when validation or allocation
 * fails. Left-handed cells are rejected instead of silently changing their
 * orientation because later strain derivatives depend on that convention.
 */
xtbloom_status_t make_lattice_3d(const double* direct, Lattice3D& lattice, std::string& error);

/* Convert one xyz vector between Cartesian bohr and fractional coordinates. */
xtbloom_status_t fractional_to_cartesian(const Lattice3D& lattice, const double* fractional,
                                         double* cartesian, std::string& error);
xtbloom_status_t cartesian_to_fractional(const Lattice3D& lattice, const double* cartesian,
                                         double* fractional, std::string& error);

/*
 * Wrap one vector into the half-open central cell [0, 1)^3. Input and output
 * may alias exactly. Cartesian wrapping converts through fractional space, so
 * rigid integer-lattice translations have one canonical representative.
 */
xtbloom_status_t wrap_fractional(const double* fractional, double* wrapped, std::string& error);
xtbloom_status_t wrap_cartesian(const Lattice3D& lattice, const double* cartesian, double* wrapped,
                                std::string& error);

/*
 * Generate the complete rectangular lattice-image superset required by a
 * real-space cutoff. For each lattice vector, the repeat count is
 * ceil(cutoff / opposing-plane spacing). Images are deliberately not filtered
 * by the norm of the bare translation: a longer translation can still bring
 * two wrapped points within cutoff across a cell boundary.
 *
 * With kInclude the origin is first. All remaining integer triplets use
 * lexicographic (i, j, k) order, which is deterministic across backends.
 */
xtbloom_status_t make_lattice_translations(const Lattice3D& lattice, double cutoff,
                                           LatticeOriginPolicy origin_policy,
                                           std::vector<LatticeTranslation>& translations,
                                           std::string& error);

}  // namespace xtbloom::detail::gfn2

#undef XTBLOOM_LATTICE_HD

#endif  // XTBLOOM_MODEL_GFN2_LATTICE_HPP
