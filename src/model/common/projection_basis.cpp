// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/common/projection_basis.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace xtbloom::detail::common {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoOverPi = 2.0 / kPi;
const std::array<double, 12> kZeta{{std::pow(1.5, 17.0), std::pow(1.5, 13.0), std::pow(1.5, 10.0),
                                    std::pow(1.5, 7.0), std::pow(1.5, 5.0), std::pow(1.5, 3.0),
                                    std::pow(1.5, 2.0), std::pow(1.5, 1.0), 1.0, 1.0 / 1.5,
                                    1.0 / 2.25, 1.0 / 3.375}};
constexpr std::array<double, 5> kDoubleFactorial{{1.0, 1.0, 3.0, 15.0, 105.0}};

bool count_fits(std::int64_t value, std::size_t element_size, bool sentinel = false) {
  if (value < 0) return false;
  const auto count = static_cast<std::uint64_t>(value) + (sentinel ? 1u : 0u);
  return count >= static_cast<std::uint64_t>(value) &&
         count <= std::numeric_limits<std::size_t>::max() / element_size;
}

bool checked_add(std::int64_t increment, std::int64_t& total) {
  if (increment < 0 || total > std::numeric_limits<std::int64_t>::max() - increment) return false;
  total += increment;
  return true;
}

/* Normalization for a Cartesian Gaussian primitive, matching tblite's STO
 * expansion convention and the spherical transforms used by integrals.cpp. */
double primitive_normalization(std::size_t angular_momentum, double exponent) {
  return std::pow(kTwoOverPi * exponent, 0.75) *
         std::pow(std::sqrt(4.0 * exponent), static_cast<double>(angular_momentum)) /
         std::sqrt(kDoubleFactorial[angular_momentum]);
}

/*
 * Normalize one contracted radial function. The embedded auxiliary-basis table
 * gives raw contraction coefficients; PySCF normalizes each contracted spherical
 * shell before constructing int1e_ovlp.  The radial normalization integral is
 * evaluated for the m=0 real spherical component, whose Cartesian polynomial
 * is z^l in the local integral convention.
 */
double contraction_norm(std::size_t angular_momentum, const std::array<double, 12>& exponents,
                        const std::array<double, 12>& coefficients) {
  double integral_factor = 1.0;
  if (angular_momentum == 1u) integral_factor = 0.5;
  if (angular_momentum == 2u) integral_factor = 0.75;
  double norm_squared = 0.0;
  for (std::size_t first = 0; first < exponents.size(); ++first) {
    for (std::size_t second = 0; second < exponents.size(); ++second) {
      const double sum = exponents[first] + exponents[second];
      /* The angular polynomial changes only the power of the Gaussian
       * denominator.  Keep pi^(3/2) separate: pow(pi / sum, 3/2+l) would
       * incorrectly introduce an extra factor of pi^l for p/d shells. */
      const double radial =
          std::pow(kPi, 1.5) * std::pow(sum, -1.5 - static_cast<double>(angular_momentum));
      norm_squared += coefficients[first] * coefficients[second] * radial;
    }
  }
  norm_squared *= integral_factor;
  return norm_squared > 0.0 && std::isfinite(norm_squared) ? 1.0 / std::sqrt(norm_squared) : 0.0;
}

}  // namespace

xtbloom_status_t make_external_projection_basis(std::int64_t atom_count, BasisPlan& basis,
                                                std::string& error) {
  if (atom_count <= 0 || !count_fits(atom_count, sizeof(std::int64_t), true)) {
    error = "external energy projection basis requires a positive representable atom count";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  constexpr std::int64_t kShellsPerAtom = 36;
  constexpr std::int64_t kOrbitalsPerAtom = 108;
  constexpr std::int64_t kCartesianPerAtom = 120;
  constexpr std::int64_t kPrimitivesPerAtom = kShellsPerAtom * 12;
  if (atom_count > std::numeric_limits<std::int64_t>::max() / kShellsPerAtom ||
      atom_count > std::numeric_limits<std::int64_t>::max() / kOrbitalsPerAtom ||
      atom_count > std::numeric_limits<std::int64_t>::max() / kCartesianPerAtom ||
      atom_count > std::numeric_limits<std::int64_t>::max() / kPrimitivesPerAtom) {
    error = "external energy projection basis dimensions overflow the supported index range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    BasisPlan created;
    created.batch_size = 1;
    created.total_atoms = atom_count;
    created.total_shells = atom_count * kShellsPerAtom;
    created.total_orbitals = atom_count * kOrbitalsPerAtom;
    created.total_cartesian_orbitals = atom_count * kCartesianPerAtom;
    created.total_primitives = atom_count * kPrimitivesPerAtom;
    created.maximum_angular_momentum = 2u;
    created.minimum_primitive_exponent = *std::min_element(kZeta.begin(), kZeta.end());

    created.atom_offsets = {0, atom_count};
    created.batch_shell_offsets = {0, created.total_shells};
    created.batch_orbital_offsets = {0, created.total_orbitals};
    created.batch_cartesian_orbital_offsets = {0, created.total_cartesian_orbitals};
    created.batch_primitive_offsets = {0, created.total_primitives};
    created.atom_shell_offsets.resize(static_cast<std::size_t>(atom_count + 1));
    created.atom_orbital_offsets.resize(static_cast<std::size_t>(atom_count + 1));
    created.atom_cartesian_orbital_offsets.resize(static_cast<std::size_t>(atom_count + 1));
    created.atom_primitive_offsets.resize(static_cast<std::size_t>(atom_count + 1));

    const std::size_t shell_count = static_cast<std::size_t>(created.total_shells);
    const std::size_t orbital_count = static_cast<std::size_t>(created.total_orbitals);
    const std::size_t cartesian_count = static_cast<std::size_t>(created.total_cartesian_orbitals);
    const std::size_t primitive_count = static_cast<std::size_t>(created.total_primitives);
    created.shell_orbital_offsets.resize(shell_count + 1);
    created.shell_cartesian_orbital_offsets.resize(shell_count + 1);
    created.shell_primitive_offsets.resize(shell_count + 1);
    created.shell_to_atom.resize(shell_count);
    created.principal_quantum_numbers.resize(shell_count);
    created.angular_momenta.resize(shell_count);
    created.shell_is_valence.assign(shell_count, 1u);
    created.slater_exponents.assign(shell_count, 1.0);
    created.primitive_exponents.resize(primitive_count);
    created.primitive_coefficients.resize(primitive_count);

    std::int64_t shell = 0;
    std::int64_t orbital = 0;
    std::int64_t cartesian = 0;
    std::int64_t primitive = 0;
    for (std::int64_t atom = 0; atom < atom_count; ++atom) {
      const std::size_t atom_index = static_cast<std::size_t>(atom);
      created.atom_shell_offsets[atom_index] = shell;
      created.atom_orbital_offsets[atom_index] = orbital;
      created.atom_cartesian_orbital_offsets[atom_index] = cartesian;
      created.atom_primitive_offsets[atom_index] = primitive;
      for (std::size_t angular_momentum = 0; angular_momentum <= 2u; ++angular_momentum) {
        const std::int64_t shell_orbitals = 2 * static_cast<std::int64_t>(angular_momentum) + 1;
        const std::int64_t shell_cartesian = (static_cast<std::int64_t>(angular_momentum) + 1) *
                                             (static_cast<std::int64_t>(angular_momentum) + 2) / 2;
        for (std::size_t radial = 0; radial < kZeta.size(); ++radial) {
          const std::size_t shell_index = static_cast<std::size_t>(shell);
          const std::size_t primitive_index = static_cast<std::size_t>(primitive);
          created.shell_orbital_offsets[shell_index] = orbital;
          created.shell_cartesian_orbital_offsets[shell_index] = cartesian;
          created.shell_primitive_offsets[shell_index] = primitive;
          created.shell_to_atom[shell_index] = atom;
          created.principal_quantum_numbers[shell_index] =
              static_cast<std::uint8_t>(angular_momentum + 1u);
          created.angular_momenta[shell_index] = static_cast<std::uint8_t>(angular_momentum);
          created.slater_exponents[shell_index] = kZeta[radial];

          std::array<double, 12> normalized_coefficients{};
          for (std::size_t gaussian = 0; gaussian < kZeta.size(); ++gaussian) {
            created.primitive_exponents[primitive_index + gaussian] = kZeta[gaussian];
            const double raw = gaussian == radial ? 1.0 : (radial + 1u == gaussian ? -1.0 : 0.0);
            normalized_coefficients[gaussian] =
                raw * primitive_normalization(angular_momentum, kZeta[gaussian]);
          }
          if (radial + 1u == kZeta.size()) {
            normalized_coefficients.fill(0.0);
            for (std::size_t gaussian = 0; gaussian < kZeta.size(); ++gaussian) {
              normalized_coefficients[gaussian] =
                  (gaussian + 1u == kZeta.size() ? 1.0 : 0.0) *
                  primitive_normalization(angular_momentum, kZeta[gaussian]);
            }
          }
          const double scale = contraction_norm(angular_momentum, kZeta, normalized_coefficients);
          if (!(scale > 0.0) || !std::isfinite(scale)) {
            error = "external energy projection basis contraction normalization failed";
            return XTBLOOM_STATUS_INTERNAL_ERROR;
          }
          for (std::size_t gaussian = 0; gaussian < kZeta.size(); ++gaussian) {
            created.primitive_coefficients[primitive_index + gaussian] =
                normalized_coefficients[gaussian] * scale;
          }

          ++shell;
          orbital += shell_orbitals;
          cartesian += shell_cartesian;
          primitive += static_cast<std::int64_t>(kZeta.size());
          created.shell_primitive_offsets[static_cast<std::size_t>(shell)] = primitive;
        }
      }
      created.atom_shell_offsets[atom_index + 1u] = shell;
      created.atom_orbital_offsets[atom_index + 1u] = orbital;
      created.atom_cartesian_orbital_offsets[atom_index + 1u] = cartesian;
      created.atom_primitive_offsets[atom_index + 1u] = primitive;
    }

    if (shell != created.total_shells || orbital != created.total_orbitals ||
        cartesian != created.total_cartesian_orbitals || primitive != created.total_primitives) {
      error = "external energy projection basis construction produced inconsistent offsets";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    created.shell_orbital_offsets[shell_count] = orbital;
    created.shell_cartesian_orbital_offsets[shell_count] = cartesian;
    created.shell_primitive_offsets[shell_count] = primitive;
    basis = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the external energy projection basis";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "external energy projection basis dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

}  // namespace xtbloom::detail::common
