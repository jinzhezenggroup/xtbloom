#include "model/gfn2/integrals.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace xtbloom::detail::gfn2 {
namespace {

constexpr std::size_t kMaximumCartesianFunctions = 6;
constexpr std::size_t kMaximumSphericalFunctions = 5;
constexpr std::size_t kMaximumCartesianBlock =
    kMaximumCartesianFunctions * kMaximumCartesianFunctions;
constexpr std::size_t kMaximumSphericalBlock =
    kMaximumSphericalFunctions * kMaximumSphericalFunctions;
constexpr std::size_t kDipoleComponents = 3;
constexpr std::size_t kQuadrupoleComponents = 6;
constexpr std::size_t kMultipoleComponents = kDipoleComponents + kQuadrupoleComponents;
constexpr double kSqrtThree = 1.732050807568877293527446341505872367;
constexpr double kSqrtPiCubed = 5.5683279968317061;

struct alignas(double) IntegralWorkspace {
  std::array<double, kMaximumCartesianBlock> cartesian;
  std::array<double, 3 * kMaximumCartesianBlock> cartesian_gradient;
  std::array<double, kMaximumSphericalBlock> spherical;
  std::array<double, 3 * kMaximumSphericalBlock> spherical_gradient;
  /* Components are [x,y,z,xx,xy,yy,xz,yz,zz], each as one shell block. */
  std::array<double, kMultipoleComponents * kMaximumCartesianBlock> cartesian_multipole;
  std::array<double, kMultipoleComponents * kMaximumSphericalBlock> spherical_multipole;
  /* Derivative layout is [coordinate][component][shell-block element]. */
  std::array<double, 3 * kMultipoleComponents * kMaximumCartesianBlock>
      cartesian_multipole_gradient;
  std::array<double, 3 * kMultipoleComponents * kMaximumSphericalBlock>
      spherical_multipole_gradient;
};

struct CartesianExponent {
  std::uint8_t x;
  std::uint8_t y;
  std::uint8_t z;
};

/* CCA Cartesian order used by tblite's native integral implementation. */
constexpr std::array<CartesianExponent, 1> kCartesianS{{{0, 0, 0}}};
constexpr std::array<CartesianExponent, 3> kCartesianP{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
constexpr std::array<CartesianExponent, 6> kCartesianD{
    {{2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2}}};

/* Powers of the ket-centered operator in dipole/quadrupole component order. */
constexpr std::array<CartesianExponent, kMultipoleComponents> kMultipolePowers{{
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
    {2, 0, 0},
    {1, 1, 0},
    {0, 2, 0},
    {1, 0, 1},
    {0, 1, 1},
    {0, 0, 2},
}};

struct SphericalTransform {
  std::size_t spherical_count;
  std::size_t cartesian_count;
  /* Row-major [spherical][cartesian], padded to 5-by-6. */
  std::array<double, kMaximumSphericalFunctions * kMaximumCartesianFunctions> coefficient;
};

constexpr SphericalTransform kTransformS{
    1, 1, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

/* Standard real spherical ordering [-1, 0, +1] = [y, z, x]. */
constexpr SphericalTransform kTransformP{
    3, 3, {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

/*
 * tblite's normalized real d harmonics in [-2, -1, 0, +1, +2] order.
 * Cartesian columns are [xx, xy, xz, yy, yz, zz].
 */
constexpr SphericalTransform kTransformD{5,
                                         6,
                                         {0.0,
                                          kSqrtThree,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          kSqrtThree,
                                          0.0,
                                          -0.5,
                                          0.0,
                                          0.0,
                                          -0.5,
                                          0.0,
                                          1.0,
                                          0.0,
                                          0.0,
                                          kSqrtThree,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.5 * kSqrtThree,
                                          0.0,
                                          0.0,
                                          -0.5 * kSqrtThree,
                                          0.0,
                                          0.0}};

bool representable_as_size(std::int64_t value) {
  return value >= 0 && static_cast<std::uint64_t>(value) <= std::numeric_limits<std::size_t>::max();
}

bool checked_square_add(std::int64_t value, std::int64_t& total) {
  if (value < 0) {
    return false;
  }
  if (value == 0) {
    return true;
  }
  if (value > std::numeric_limits<std::int64_t>::max() / value) {
    return false;
  }
  const std::int64_t square = value * value;
  if (total > std::numeric_limits<std::int64_t>::max() - square) {
    return false;
  }
  total += square;
  return true;
}

std::size_t spherical_count(std::uint8_t angular_momentum) {
  return 2u * static_cast<std::size_t>(angular_momentum) + 1u;
}

std::size_t cartesian_count(std::uint8_t angular_momentum) {
  const std::size_t l = angular_momentum;
  return (l + 1u) * (l + 2u) / 2u;
}

const CartesianExponent* cartesian_exponents(std::uint8_t angular_momentum) {
  switch (angular_momentum) {
    case 0:
      return kCartesianS.data();
    case 1:
      return kCartesianP.data();
    case 2:
      return kCartesianD.data();
    default:
      return nullptr;
  }
}

const SphericalTransform* spherical_transform(std::uint8_t angular_momentum) {
  switch (angular_momentum) {
    case 0:
      return &kTransformS;
    case 1:
      return &kTransformP;
    case 2:
      return &kTransformD;
    default:
      return nullptr;
  }
}

xtbloom_status_t validate_basis(const BasisPlan& basis, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      basis.total_orbitals <= 0 || basis.maximum_angular_momentum > 2u ||
      !representable_as_size(basis.batch_size) || !representable_as_size(basis.total_atoms) ||
      !representable_as_size(basis.total_shells) || !representable_as_size(basis.total_orbitals) ||
      !representable_as_size(basis.total_primitives)) {
    error = "integral basis has unsupported or unrepresentable dimensions";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch_count = static_cast<std::size_t>(basis.batch_size);
  const std::size_t atom_count = static_cast<std::size_t>(basis.total_atoms);
  const std::size_t shell_count = static_cast<std::size_t>(basis.total_shells);
  const std::size_t primitive_count = static_cast<std::size_t>(basis.total_primitives);
  if (basis.atom_offsets.size() != batch_count + 1u ||
      basis.batch_shell_offsets.size() != batch_count + 1u ||
      basis.batch_orbital_offsets.size() != batch_count + 1u ||
      basis.atom_shell_offsets.size() != atom_count + 1u ||
      basis.atom_orbital_offsets.size() != atom_count + 1u ||
      basis.shell_orbital_offsets.size() != shell_count + 1u ||
      basis.shell_primitive_offsets.size() != shell_count + 1u ||
      basis.shell_to_atom.size() != shell_count || basis.angular_momenta.size() != shell_count ||
      basis.primitive_exponents.size() != primitive_count ||
      basis.primitive_coefficients.size() != primitive_count) {
    error = "integral basis is incomplete or internally inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (basis.atom_offsets.front() != 0 || basis.atom_offsets.back() != basis.total_atoms ||
      basis.batch_shell_offsets.front() != 0 ||
      basis.batch_shell_offsets.back() != basis.total_shells ||
      basis.batch_orbital_offsets.front() != 0 ||
      basis.batch_orbital_offsets.back() != basis.total_orbitals ||
      basis.atom_shell_offsets.front() != 0 ||
      basis.atom_shell_offsets.back() != basis.total_shells ||
      basis.atom_orbital_offsets.front() != 0 ||
      basis.atom_orbital_offsets.back() != basis.total_orbitals ||
      basis.shell_orbital_offsets.front() != 0 ||
      basis.shell_orbital_offsets.back() != basis.total_orbitals ||
      basis.shell_primitive_offsets.front() != 0 ||
      basis.shell_primitive_offsets.back() != basis.total_primitives) {
    error = "integral basis offsets do not span their stored dimensions";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    if (basis.atom_offsets[batch] < 0 ||
        basis.atom_offsets[batch] > basis.atom_offsets[batch + 1] ||
        basis.atom_offsets[batch + 1] > basis.total_atoms ||
        basis.batch_shell_offsets[batch] > basis.batch_shell_offsets[batch + 1] ||
        basis.batch_orbital_offsets[batch] > basis.batch_orbital_offsets[batch + 1]) {
      error = "integral basis batch offsets are not a valid ragged partition";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::size_t first_atom = static_cast<std::size_t>(basis.atom_offsets[batch]);
    const std::size_t last_atom = static_cast<std::size_t>(basis.atom_offsets[batch + 1]);
    if (basis.batch_shell_offsets[batch] != basis.atom_shell_offsets[first_atom] ||
        basis.batch_shell_offsets[batch + 1] != basis.atom_shell_offsets[last_atom] ||
        basis.batch_orbital_offsets[batch] != basis.atom_orbital_offsets[first_atom] ||
        basis.batch_orbital_offsets[batch + 1] != basis.atom_orbital_offsets[last_atom]) {
      error = "integral basis batch offsets disagree with atom offsets";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    if (basis.atom_shell_offsets[atom] > basis.atom_shell_offsets[atom + 1] ||
        basis.atom_orbital_offsets[atom] > basis.atom_orbital_offsets[atom + 1]) {
      error = "integral basis atom offsets are not monotone";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t shell = 0; shell < shell_count; ++shell) {
    const std::uint8_t angular_momentum = basis.angular_momenta[shell];
    const std::int64_t orbital_count =
        basis.shell_orbital_offsets[shell + 1] - basis.shell_orbital_offsets[shell];
    const std::int64_t primitive_begin = basis.shell_primitive_offsets[shell];
    const std::int64_t primitive_end = basis.shell_primitive_offsets[shell + 1];
    if (angular_momentum > 2u ||
        orbital_count != static_cast<std::int64_t>(spherical_count(angular_momentum)) ||
        primitive_begin < 0 || primitive_begin >= primitive_end ||
        primitive_end > basis.total_primitives || basis.shell_to_atom[shell] < 0 ||
        basis.shell_to_atom[shell] >= basis.total_atoms) {
      error = "integral basis contains an invalid s, p, or d shell";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t primitive = primitive_begin; primitive < primitive_end; ++primitive) {
      const std::size_t index = static_cast<std::size_t>(primitive);
      if (!(basis.primitive_exponents[index] > 0.0) ||
          !std::isfinite(basis.primitive_exponents[index]) ||
          !std::isfinite(basis.primitive_coefficients[index])) {
        error = "integral basis contains invalid primitive data";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_plan(const BasisPlan& basis, const IntegralPlan& plan,
                               std::string& error) {
  xtbloom_status_t status = validate_basis(basis, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (plan.batch_size != basis.batch_size || plan.total_matrix_elements < 0 ||
      !std::isfinite(plan.integral_cutoff) || !(plan.integral_cutoff > 0.0) ||
      plan.workspace_size_bytes < sizeof(IntegralWorkspace) ||
      plan.matrix_offsets.size() != static_cast<std::size_t>(basis.batch_size + 1) ||
      plan.matrix_offsets.front() != 0 ||
      plan.matrix_offsets.back() != plan.total_matrix_elements ||
      !representable_as_size(plan.total_matrix_elements) ||
      static_cast<std::uint64_t>(plan.total_matrix_elements) >
          std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    error = "integral plan is incomplete or incompatible with the basis";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::int64_t expected = 0;
  for (std::int64_t batch = 0; batch < basis.batch_size; ++batch) {
    if (plan.matrix_offsets[static_cast<std::size_t>(batch)] != expected) {
      error = "integral matrix offsets do not match the basis dimensions";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::int64_t orbitals = basis.batch_orbital_offsets[static_cast<std::size_t>(batch + 1)] -
                                  basis.batch_orbital_offsets[static_cast<std::size_t>(batch)];
    if (!checked_square_add(orbitals, expected)) {
      error = "integral matrix dimensions overflow the supported index range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  if (expected != plan.total_matrix_elements) {
    error = "integral matrix offsets do not span the packed output";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_evaluation(const BasisPlan& basis, const IntegralPlan& plan,
                                     const double* positions, const void* workspace,
                                     std::size_t workspace_size, std::string& error) {
  xtbloom_status_t status = validate_plan(basis, plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (positions == nullptr || workspace == nullptr) {
    error = "integral positions and workspace must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (workspace_size < plan.workspace_size_bytes ||
      reinterpret_cast<std::uintptr_t>(workspace) % alignof(double) != 0u) {
    error = "integral workspace is too small or not aligned for double";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t atom_count = static_cast<std::size_t>(basis.total_atoms);
  if (atom_count > std::numeric_limits<std::size_t>::max() / 3u) {
    error = "integral geometry dimensions exceed host limits";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const double maximum_coordinate = 0.25 * std::sqrt(std::numeric_limits<double>::max());
  for (std::size_t coordinate = 0; coordinate < atom_count * 3u; ++coordinate) {
    if (!std::isfinite(positions[coordinate])) {
      error = "integral positions contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    /*
     * Pair vectors and their squared norms/products are formed directly in the
     * hot loops. This bound leaves enough margin for three squared displacement
     * components even when the two coordinates have opposite signs.
     */
    if (std::abs(positions[coordinate]) > maximum_coordinate) {
      error = "integral positions are too large for finite pair arithmetic";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

/*
 * One-dimensional Hermite overlap recurrence normalized by the s-s integral.
 * a is the ket angular exponent about center i and b is the bra exponent about
 * center j. GFN2 needs a<=2 for overlap, a<=3 for first derivatives, and
 * a<=5 for derivatives of second moments relative to the ket center.
 */
void make_axis_overlap(double product_minus_i, double product_minus_j, double inverse_twice_sum,
                       std::size_t maximum_a, std::size_t maximum_b, double overlap[6][3]) {
  for (std::size_t a = 0; a < 6u; ++a) {
    for (std::size_t b = 0; b < 3u; ++b) {
      overlap[a][b] = 0.0;
    }
  }
  overlap[0][0] = 1.0;
  for (std::size_t a = 1; a <= maximum_a; ++a) {
    overlap[a][0] = product_minus_i * overlap[a - 1][0];
    if (a > 1u) {
      overlap[a][0] += static_cast<double>(a - 1u) * inverse_twice_sum * overlap[a - 2][0];
    }
  }
  for (std::size_t b = 1; b <= maximum_b; ++b) {
    overlap[0][b] = product_minus_j * overlap[0][b - 1];
    if (b > 1u) {
      overlap[0][b] += static_cast<double>(b - 1u) * inverse_twice_sum * overlap[0][b - 2];
    }
    for (std::size_t a = 1; a <= maximum_a; ++a) {
      overlap[a][b] = product_minus_i * overlap[a - 1][b] +
                      static_cast<double>(b) * inverse_twice_sum * overlap[a - 1][b - 1];
      if (a > 1u) {
        overlap[a][b] += static_cast<double>(a - 1u) * inverse_twice_sum * overlap[a - 2][b];
      }
    }
  }
}

void transform_shell_pair(const SphericalTransform& bra, const SphericalTransform& ket,
                          const double* cartesian, double* spherical) {
  for (std::size_t spherical_bra = 0; spherical_bra < bra.spherical_count; ++spherical_bra) {
    for (std::size_t spherical_ket = 0; spherical_ket < ket.spherical_count; ++spherical_ket) {
      double value = 0.0;
      for (std::size_t cartesian_bra = 0; cartesian_bra < bra.cartesian_count; ++cartesian_bra) {
        const double bra_coefficient =
            bra.coefficient[spherical_bra * kMaximumCartesianFunctions + cartesian_bra];
        if (bra_coefficient == 0.0) {
          continue;
        }
        for (std::size_t cartesian_ket = 0; cartesian_ket < ket.cartesian_count; ++cartesian_ket) {
          const double ket_coefficient =
              ket.coefficient[spherical_ket * kMaximumCartesianFunctions + cartesian_ket];
          if (ket_coefficient == 0.0) {
            continue;
          }
          value += bra_coefficient *
                   cartesian[cartesian_bra * ket.cartesian_count + cartesian_ket] * ket_coefficient;
        }
      }
      spherical[spherical_bra * ket.spherical_count + spherical_ket] = value;
    }
  }
}

/* Project raw second moments to tblite's traceless Cartesian quadrupoles. */
void make_quadrupole_traceless(double* multipoles, std::size_t block_size) {
  for (std::size_t element = 0; element < block_size; ++element) {
    const double trace =
        0.5 * (multipoles[3u * block_size + element] + multipoles[5u * block_size + element] +
               multipoles[8u * block_size + element]);
    multipoles[3u * block_size + element] = 1.5 * multipoles[3u * block_size + element] - trace;
    multipoles[4u * block_size + element] *= 1.5;
    multipoles[5u * block_size + element] = 1.5 * multipoles[5u * block_size + element] - trace;
    multipoles[6u * block_size + element] *= 1.5;
    multipoles[7u * block_size + element] *= 1.5;
    multipoles[8u * block_size + element] = 1.5 * multipoles[8u * block_size + element] - trace;
  }
}

/*
 * Pull back the explicit ket-to-bra origin translation used for the reverse
 * matrix element. Local moment derivatives are handled separately by the
 * Gaussian recurrence; this helper returns only adjoints of local S/D and the
 * explicit displacement dependence.
 */
void add_multipole_shift_pullback(
    const double vector[3], double overlap, const std::array<double, kDipoleComponents>& dipole,
    const std::array<double, kDipoleComponents>& reverse_dipole_adjoint,
    const std::array<double, kQuadrupoleComponents>& reverse_quadrupole_adjoint,
    double& overlap_adjoint, std::array<double, kDipoleComponents>& dipole_adjoint,
    std::array<double, 3>& vector_adjoint) {
  for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
    overlap_adjoint += reverse_dipole_adjoint[coordinate] * vector[coordinate];
    dipole_adjoint[coordinate] += reverse_dipole_adjoint[coordinate];
    vector_adjoint[coordinate] += reverse_dipole_adjoint[coordinate] * overlap;
  }

  /* Pull the traceless projection back to the six raw second-moment shifts. */
  const double diagonal_sum =
      reverse_quadrupole_adjoint[0] + reverse_quadrupole_adjoint[2] + reverse_quadrupole_adjoint[5];
  const std::array<double, kQuadrupoleComponents> raw_shift_adjoint{
      1.5 * reverse_quadrupole_adjoint[0] - 0.5 * diagonal_sum,
      1.5 * reverse_quadrupole_adjoint[1],
      1.5 * reverse_quadrupole_adjoint[2] - 0.5 * diagonal_sum,
      1.5 * reverse_quadrupole_adjoint[3],
      1.5 * reverse_quadrupole_adjoint[4],
      1.5 * reverse_quadrupole_adjoint[5] - 0.5 * diagonal_sum,
  };

  const double x = vector[0];
  const double y = vector[1];
  const double z = vector[2];
  const double dx = dipole[0];
  const double dy = dipole[1];
  const double dz = dipole[2];
  const double axx = raw_shift_adjoint[0];
  const double axy = raw_shift_adjoint[1];
  const double ayy = raw_shift_adjoint[2];
  const double axz = raw_shift_adjoint[3];
  const double ayz = raw_shift_adjoint[4];
  const double azz = raw_shift_adjoint[5];

  overlap_adjoint +=
      axx * x * x + axy * x * y + ayy * y * y + axz * x * z + ayz * y * z + azz * z * z;
  dipole_adjoint[0] += 2.0 * axx * x + axy * y + axz * z;
  dipole_adjoint[1] += axy * x + 2.0 * ayy * y + ayz * z;
  dipole_adjoint[2] += axz * x + ayz * y + 2.0 * azz * z;

  vector_adjoint[0] +=
      axx * (2.0 * dx + 2.0 * x * overlap) + axy * (dy + y * overlap) + axz * (dz + z * overlap);
  vector_adjoint[1] +=
      axy * (dx + x * overlap) + ayy * (2.0 * dy + 2.0 * y * overlap) + ayz * (dz + z * overlap);
  vector_adjoint[2] +=
      axz * (dx + x * overlap) + ayz * (dy + y * overlap) + azz * (2.0 * dz + 2.0 * z * overlap);
}

void compute_shell_pair(const BasisPlan& basis, std::size_t bra_shell, std::size_t ket_shell,
                        const double vector[3], double integral_cutoff, bool with_gradient,
                        bool with_multipoles, IntegralWorkspace& workspace) {
  const std::uint8_t bra_l = basis.angular_momenta[bra_shell];
  const std::uint8_t ket_l = basis.angular_momenta[ket_shell];
  const std::size_t bra_cartesian_count = cartesian_count(bra_l);
  const std::size_t ket_cartesian_count = cartesian_count(ket_l);
  const std::size_t cartesian_block_size = bra_cartesian_count * ket_cartesian_count;
  std::fill_n(workspace.cartesian.data(), cartesian_block_size, 0.0);
  if (with_gradient) {
    std::fill_n(workspace.cartesian_gradient.data(), 3u * cartesian_block_size, 0.0);
  }
  if (with_multipoles) {
    std::fill_n(workspace.cartesian_multipole.data(), kMultipoleComponents * cartesian_block_size,
                0.0);
    if (with_gradient) {
      std::fill_n(workspace.cartesian_multipole_gradient.data(),
                  3u * kMultipoleComponents * cartesian_block_size, 0.0);
    }
  }

  const CartesianExponent* bra_exponents = cartesian_exponents(bra_l);
  const CartesianExponent* ket_exponents = cartesian_exponents(ket_l);
  const std::int64_t bra_primitive_begin = basis.shell_primitive_offsets[bra_shell];
  const std::int64_t bra_primitive_end = basis.shell_primitive_offsets[bra_shell + 1u];
  const std::int64_t ket_primitive_begin = basis.shell_primitive_offsets[ket_shell];
  const std::int64_t ket_primitive_end = basis.shell_primitive_offsets[ket_shell + 1u];
  const double distance_squared =
      vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];

  for (std::int64_t ket_primitive = ket_primitive_begin; ket_primitive < ket_primitive_end;
       ++ket_primitive) {
    const std::size_t ket_primitive_index = static_cast<std::size_t>(ket_primitive);
    const double ket_alpha = basis.primitive_exponents[ket_primitive_index];
    for (std::int64_t bra_primitive = bra_primitive_begin; bra_primitive < bra_primitive_end;
         ++bra_primitive) {
      const std::size_t bra_primitive_index = static_cast<std::size_t>(bra_primitive);
      const double bra_alpha = basis.primitive_exponents[bra_primitive_index];
      const double alpha_sum = ket_alpha + bra_alpha;
      const double inverse_sum = 1.0 / alpha_sum;
      const double product_exponent = ket_alpha * bra_alpha * distance_squared * inverse_sum;
      if (product_exponent > integral_cutoff) {
        continue;
      }

      const double sqrt_inverse_sum = std::sqrt(inverse_sum);
      const double primitive_prefactor = std::exp(-product_exponent) * kSqrtPiCubed *
                                         sqrt_inverse_sum * sqrt_inverse_sum * sqrt_inverse_sum *
                                         basis.primitive_coefficients[ket_primitive_index] *
                                         basis.primitive_coefficients[bra_primitive_index];
      const double inverse_twice_sum = 0.5 * inverse_sum;
      double axis[3][6][3];
      for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
        const double product_minus_i = -vector[coordinate] * bra_alpha * inverse_sum;
        const double product_minus_j = +vector[coordinate] * ket_alpha * inverse_sum;
        const std::size_t moment_order =
            with_multipoles ? (with_gradient ? 3u : 2u) : (with_gradient ? 1u : 0u);
        make_axis_overlap(product_minus_i, product_minus_j, inverse_twice_sum,
                          static_cast<std::size_t>(ket_l) + moment_order, bra_l, axis[coordinate]);
      }

      for (std::size_t bra_cartesian = 0; bra_cartesian < bra_cartesian_count; ++bra_cartesian) {
        const CartesianExponent bra = bra_exponents[bra_cartesian];
        for (std::size_t ket_cartesian = 0; ket_cartesian < ket_cartesian_count; ++ket_cartesian) {
          const CartesianExponent ket = ket_exponents[ket_cartesian];
          const std::array<std::size_t, 3> ket_power{ket.x, ket.y, ket.z};
          const std::array<std::size_t, 3> bra_power{bra.x, bra.y, bra.z};
          const double x = axis[0][ket.x][bra.x];
          const double y = axis[1][ket.y][bra.y];
          const double z = axis[2][ket.z][bra.z];
          const std::size_t cartesian_index = bra_cartesian * ket_cartesian_count + ket_cartesian;
          workspace.cartesian[cartesian_index] += primitive_prefactor * x * y * z;

          if (with_multipoles) {
            for (std::size_t component = 0; component < kMultipoleComponents; ++component) {
              const CartesianExponent power = kMultipolePowers[component];
              const std::array<std::size_t, 3> moment_power{power.x, power.y, power.z};
              const std::array<double, 3> one_dimensional{
                  axis[0][ket_power[0] + moment_power[0]][bra_power[0]],
                  axis[1][ket_power[1] + moment_power[1]][bra_power[1]],
                  axis[2][ket_power[2] + moment_power[2]][bra_power[2]],
              };
              workspace.cartesian_multipole[component * cartesian_block_size + cartesian_index] +=
                  primitive_prefactor * one_dimensional[0] * one_dimensional[1] *
                  one_dimensional[2];

              if (with_gradient) {
                for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
                  const std::size_t exponent = ket_power[coordinate] + moment_power[coordinate];
                  double derivative_1d =
                      2.0 * ket_alpha * axis[coordinate][exponent + 1u][bra_power[coordinate]];
                  if (exponent > 0u) {
                    derivative_1d -= static_cast<double>(exponent) *
                                     axis[coordinate][exponent - 1u][bra_power[coordinate]];
                  }
                  const std::size_t first_other = (coordinate + 1u) % 3u;
                  const std::size_t second_other = (coordinate + 2u) % 3u;
                  const std::size_t gradient_index =
                      (coordinate * kMultipoleComponents + component) * cartesian_block_size +
                      cartesian_index;
                  workspace.cartesian_multipole_gradient[gradient_index] +=
                      primitive_prefactor * derivative_1d * one_dimensional[first_other] *
                      one_dimensional[second_other];
                }
              }
            }
          }

          if (with_gradient) {
            const std::array<double, 3> other_axis_product{y * z, x * z, x * y};
            for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
              const std::size_t a = ket_power[coordinate];
              const std::size_t b = bra_power[coordinate];
              double derivative_1d = 2.0 * ket_alpha * axis[coordinate][a + 1u][b];
              if (a > 0u) {
                derivative_1d -= static_cast<double>(a) * axis[coordinate][a - 1u][b];
              }
              workspace.cartesian_gradient[coordinate * cartesian_block_size + cartesian_index] +=
                  primitive_prefactor * derivative_1d * other_axis_product[coordinate];
            }
          }
        }
      }
    }
  }

  const SphericalTransform& bra_transform = *spherical_transform(bra_l);
  const SphericalTransform& ket_transform = *spherical_transform(ket_l);
  transform_shell_pair(bra_transform, ket_transform, workspace.cartesian.data(),
                       workspace.spherical.data());
  if (with_gradient) {
    for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
      transform_shell_pair(bra_transform, ket_transform,
                           workspace.cartesian_gradient.data() + coordinate * cartesian_block_size,
                           workspace.spherical_gradient.data() + coordinate *
                                                                     bra_transform.spherical_count *
                                                                     ket_transform.spherical_count);
    }
  }
  if (with_multipoles) {
    const std::size_t spherical_block_size =
        bra_transform.spherical_count * ket_transform.spherical_count;
    for (std::size_t component = 0; component < kMultipoleComponents; ++component) {
      transform_shell_pair(bra_transform, ket_transform,
                           workspace.cartesian_multipole.data() + component * cartesian_block_size,
                           workspace.spherical_multipole.data() + component * spherical_block_size);
    }
    /* tblite stores Q = 3*rr/2 - r^2*I/2 in [xx,xy,yy,xz,yz,zz] order. */
    make_quadrupole_traceless(workspace.spherical_multipole.data(), spherical_block_size);
    if (with_gradient) {
      for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
        double* spherical_gradient = workspace.spherical_multipole_gradient.data() +
                                     coordinate * kMultipoleComponents * spherical_block_size;
        for (std::size_t component = 0; component < kMultipoleComponents; ++component) {
          transform_shell_pair(
              bra_transform, ket_transform,
              workspace.cartesian_multipole_gradient.data() +
                  (coordinate * kMultipoleComponents + component) * cartesian_block_size,
              spherical_gradient + component * spherical_block_size);
        }
        make_quadrupole_traceless(spherical_gradient, spherical_block_size);
      }
    }
  }
}

}  // namespace

xtbloom_status_t make_integral_plan(const BasisPlan& basis, IntegralPlan& plan, std::string& error,
                                    double integral_cutoff) {
  xtbloom_status_t status = validate_basis(basis, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (!std::isfinite(integral_cutoff) || !(integral_cutoff > 0.0)) {
    error = "integral cutoff must be finite and positive";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    IntegralPlan created;
    created.batch_size = basis.batch_size;
    created.integral_cutoff = integral_cutoff;
    created.workspace_size_bytes = sizeof(IntegralWorkspace);
    created.matrix_offsets.resize(static_cast<std::size_t>(basis.batch_size + 1));
    for (std::int64_t batch = 0; batch < basis.batch_size; ++batch) {
      created.matrix_offsets[static_cast<std::size_t>(batch)] = created.total_matrix_elements;
      const std::int64_t orbitals =
          basis.batch_orbital_offsets[static_cast<std::size_t>(batch + 1)] -
          basis.batch_orbital_offsets[static_cast<std::size_t>(batch)];
      if (!checked_square_add(orbitals, created.total_matrix_elements) ||
          static_cast<std::uint64_t>(created.total_matrix_elements) >
              std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        error = "integral matrix dimensions overflow the supported index range";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    created.matrix_offsets.back() = created.total_matrix_elements;
    plan = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN2 integral plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN2 integral plan dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t evaluate_overlap_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                      const double* positions, double* overlap, void* workspace,
                                      std::size_t workspace_size, std::string& error) {
  xtbloom_status_t status =
      validate_evaluation(basis, plan, positions, workspace, workspace_size, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (overlap == nullptr) {
    error = "overlap output must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::fill_n(overlap, static_cast<std::size_t>(plan.total_matrix_elements), 0.0);
  auto& scratch = *static_cast<IntegralWorkspace*>(workspace);
  for (std::int64_t batch = 0; batch < basis.batch_size; ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    const std::int64_t atom_begin = basis.atom_offsets[batch_index];
    const std::int64_t atom_end = basis.atom_offsets[batch_index + 1u];
    const std::int64_t orbital_begin = basis.batch_orbital_offsets[batch_index];
    const std::size_t orbital_count =
        static_cast<std::size_t>(basis.batch_orbital_offsets[batch_index + 1u] - orbital_begin);
    double* matrix = overlap + static_cast<std::size_t>(plan.matrix_offsets[batch_index]);

    for (std::int64_t ket_atom = atom_begin; ket_atom < atom_end; ++ket_atom) {
      const std::size_t ket_atom_index = static_cast<std::size_t>(ket_atom);
      for (std::int64_t bra_atom = atom_begin; bra_atom <= ket_atom; ++bra_atom) {
        const std::size_t bra_atom_index = static_cast<std::size_t>(bra_atom);
        const double vector[3]{
            positions[ket_atom_index * 3u] - positions[bra_atom_index * 3u],
            positions[ket_atom_index * 3u + 1u] - positions[bra_atom_index * 3u + 1u],
            positions[ket_atom_index * 3u + 2u] - positions[bra_atom_index * 3u + 2u]};
        const std::int64_t ket_shell_begin = basis.atom_shell_offsets[ket_atom_index];
        const std::int64_t ket_shell_end = basis.atom_shell_offsets[ket_atom_index + 1u];
        const std::int64_t bra_shell_begin = basis.atom_shell_offsets[bra_atom_index];
        const std::int64_t bra_shell_end = basis.atom_shell_offsets[bra_atom_index + 1u];
        for (std::int64_t ket_shell = ket_shell_begin; ket_shell < ket_shell_end; ++ket_shell) {
          for (std::int64_t bra_shell = bra_shell_begin; bra_shell < bra_shell_end; ++bra_shell) {
            if (ket_atom == bra_atom && bra_shell > ket_shell) {
              continue;
            }
            const std::size_t ket_shell_index = static_cast<std::size_t>(ket_shell);
            const std::size_t bra_shell_index = static_cast<std::size_t>(bra_shell);
            compute_shell_pair(basis, bra_shell_index, ket_shell_index, vector,
                               plan.integral_cutoff, false, false, scratch);

            const std::size_t bra_spherical_count =
                spherical_count(basis.angular_momenta[bra_shell_index]);
            const std::size_t ket_spherical_count =
                spherical_count(basis.angular_momenta[ket_shell_index]);
            const std::size_t bra_orbital = static_cast<std::size_t>(
                basis.shell_orbital_offsets[bra_shell_index] - orbital_begin);
            const std::size_t ket_orbital = static_cast<std::size_t>(
                basis.shell_orbital_offsets[ket_shell_index] - orbital_begin);
            for (std::size_t bra_ao = 0; bra_ao < bra_spherical_count; ++bra_ao) {
              for (std::size_t ket_ao = 0; ket_ao < ket_spherical_count; ++ket_ao) {
                const double value = scratch.spherical[bra_ao * ket_spherical_count + ket_ao];
                matrix[(bra_orbital + bra_ao) * orbital_count + ket_orbital + ket_ao] = value;
                if (bra_shell != ket_shell) {
                  matrix[(ket_orbital + ket_ao) * orbital_count + bra_orbital + bra_ao] = value;
                }
              }
            }
          }
        }
      }
    }
  }

  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t evaluate_multipole_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                        const double* positions, double* dipole, double* quadrupole,
                                        void* workspace, std::size_t workspace_size,
                                        std::string& error) {
  xtbloom_status_t status =
      validate_evaluation(basis, plan, positions, workspace, workspace_size, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (dipole == nullptr || quadrupole == nullptr) {
    error = "multipole output buffers must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t matrix_elements = static_cast<std::size_t>(plan.total_matrix_elements);
  if (matrix_elements >
      std::numeric_limits<std::size_t>::max() / kQuadrupoleComponents / sizeof(double)) {
    error = "multipole output dimensions exceed host limits";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::fill_n(dipole, kDipoleComponents * matrix_elements, 0.0);
  std::fill_n(quadrupole, kQuadrupoleComponents * matrix_elements, 0.0);
  auto& scratch = *static_cast<IntegralWorkspace*>(workspace);

  for (std::int64_t batch = 0; batch < basis.batch_size; ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    const std::int64_t atom_begin = basis.atom_offsets[batch_index];
    const std::int64_t atom_end = basis.atom_offsets[batch_index + 1u];
    const std::int64_t orbital_begin = basis.batch_orbital_offsets[batch_index];
    const std::size_t orbital_count =
        static_cast<std::size_t>(basis.batch_orbital_offsets[batch_index + 1u] - orbital_begin);
    const std::size_t matrix_offset = static_cast<std::size_t>(plan.matrix_offsets[batch_index]);

    for (std::int64_t ket_atom = atom_begin; ket_atom < atom_end; ++ket_atom) {
      const std::size_t ket_atom_index = static_cast<std::size_t>(ket_atom);
      for (std::int64_t bra_atom = atom_begin; bra_atom <= ket_atom; ++bra_atom) {
        const std::size_t bra_atom_index = static_cast<std::size_t>(bra_atom);
        const double vector[3]{
            positions[ket_atom_index * 3u] - positions[bra_atom_index * 3u],
            positions[ket_atom_index * 3u + 1u] - positions[bra_atom_index * 3u + 1u],
            positions[ket_atom_index * 3u + 2u] - positions[bra_atom_index * 3u + 2u]};
        const std::int64_t ket_shell_begin = basis.atom_shell_offsets[ket_atom_index];
        const std::int64_t ket_shell_end = basis.atom_shell_offsets[ket_atom_index + 1u];
        const std::int64_t bra_shell_begin = basis.atom_shell_offsets[bra_atom_index];
        const std::int64_t bra_shell_end = basis.atom_shell_offsets[bra_atom_index + 1u];

        for (std::int64_t ket_shell = ket_shell_begin; ket_shell < ket_shell_end; ++ket_shell) {
          for (std::int64_t bra_shell = bra_shell_begin; bra_shell < bra_shell_end; ++bra_shell) {
            if (ket_atom == bra_atom && bra_shell > ket_shell) {
              continue;
            }
            const std::size_t ket_shell_index = static_cast<std::size_t>(ket_shell);
            const std::size_t bra_shell_index = static_cast<std::size_t>(bra_shell);
            compute_shell_pair(basis, bra_shell_index, ket_shell_index, vector,
                               plan.integral_cutoff, false, true, scratch);

            const std::size_t bra_count = spherical_count(basis.angular_momenta[bra_shell_index]);
            const std::size_t ket_count = spherical_count(basis.angular_momenta[ket_shell_index]);
            const std::size_t block_size = bra_count * ket_count;
            const std::size_t bra_orbital = static_cast<std::size_t>(
                basis.shell_orbital_offsets[bra_shell_index] - orbital_begin);
            const std::size_t ket_orbital = static_cast<std::size_t>(
                basis.shell_orbital_offsets[ket_shell_index] - orbital_begin);

            for (std::size_t bra_ao = 0; bra_ao < bra_count; ++bra_ao) {
              for (std::size_t ket_ao = 0; ket_ao < ket_count; ++ket_ao) {
                /*
                 * A same-shell, same-center operator block is exactly
                 * symmetric. Use one computed AO triangle for both sides so
                 * different Cartesian-to-spherical accumulation orders do
                 * not leave one-ULP antisymmetric noise in the matrix.
                 */
                if (bra_shell == ket_shell && bra_ao > ket_ao) {
                  continue;
                }
                const std::size_t block_index = bra_ao * ket_count + ket_ao;
                const std::size_t forward =
                    matrix_offset + (bra_orbital + bra_ao) * orbital_count + ket_orbital + ket_ao;
                const std::size_t reverse =
                    matrix_offset + (ket_orbital + ket_ao) * orbital_count + bra_orbital + bra_ao;
                const double overlap = scratch.spherical[block_index];
                std::array<double, kDipoleComponents> local_dipole{};
                for (std::size_t component = 0; component < kDipoleComponents; ++component) {
                  local_dipole[component] =
                      scratch.spherical_multipole[component * block_size + block_index];
                  dipole[component * matrix_elements + forward] = local_dipole[component];
                  if (bra_shell == ket_shell) {
                    dipole[component * matrix_elements + reverse] = local_dipole[component];
                  }
                }
                std::array<double, kQuadrupoleComponents> local_quadrupole{};
                for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
                  local_quadrupole[component] =
                      scratch.spherical_multipole[(component + kDipoleComponents) * block_size +
                                                  block_index];
                  quadrupole[component * matrix_elements + forward] = local_quadrupole[component];
                  if (bra_shell == ket_shell) {
                    quadrupole[component * matrix_elements + reverse] = local_quadrupole[component];
                  }
                }

                if (bra_shell == ket_shell) {
                  continue;
                }
                std::array<double, kDipoleComponents> shifted_dipole{};
                for (std::size_t component = 0; component < kDipoleComponents; ++component) {
                  shifted_dipole[component] = local_dipole[component] + vector[component] * overlap;
                  dipole[component * matrix_elements + reverse] = shifted_dipole[component];
                }

                std::array<double, kQuadrupoleComponents> shift{};
                shift[0] = 2.0 * vector[0] * local_dipole[0] + vector[0] * vector[0] * overlap;
                shift[1] = vector[0] * local_dipole[1] + vector[1] * local_dipole[0] +
                           vector[0] * vector[1] * overlap;
                shift[2] = 2.0 * vector[1] * local_dipole[1] + vector[1] * vector[1] * overlap;
                shift[3] = vector[0] * local_dipole[2] + vector[2] * local_dipole[0] +
                           vector[0] * vector[2] * overlap;
                shift[4] = vector[1] * local_dipole[2] + vector[2] * local_dipole[1] +
                           vector[1] * vector[2] * overlap;
                shift[5] = 2.0 * vector[2] * local_dipole[2] + vector[2] * vector[2] * overlap;
                const double trace = 0.5 * (shift[0] + shift[2] + shift[5]);
                for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
                  const bool diagonal = component == 0u || component == 2u || component == 5u;
                  const double shifted = local_quadrupole[component] + 1.5 * shift[component] -
                                         (diagonal ? trace : 0.0);
                  quadrupole[component * matrix_elements + reverse] = shifted;
                }
              }
            }
          }
        }
      }
    }
  }

  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_multipole_gradient_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                            const double* positions, const double* dE_ddipole,
                                            const double* dE_dquadrupole, double* gradients,
                                            void* workspace, std::size_t workspace_size,
                                            std::string& error) {
  xtbloom_status_t status =
      validate_evaluation(basis, plan, positions, workspace, workspace_size, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (dE_ddipole == nullptr || dE_dquadrupole == nullptr || gradients == nullptr) {
    error = "multipole derivatives and gradients must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t matrix_elements = static_cast<std::size_t>(plan.total_matrix_elements);
  if (matrix_elements >
      std::numeric_limits<std::size_t>::max() / kQuadrupoleComponents / sizeof(double)) {
    error = "multipole derivative dimensions exceed host limits";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t element = 0; element < kDipoleComponents * matrix_elements; ++element) {
    if (!std::isfinite(dE_ddipole[element])) {
      error = "dipole derivatives contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t element = 0; element < kQuadrupoleComponents * matrix_elements; ++element) {
    if (!std::isfinite(dE_dquadrupole[element])) {
      error = "quadrupole derivatives contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  auto& scratch = *static_cast<IntegralWorkspace*>(workspace);
  for (std::int64_t batch = 0; batch < basis.batch_size; ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    const std::int64_t atom_begin = basis.atom_offsets[batch_index];
    const std::int64_t atom_end = basis.atom_offsets[batch_index + 1u];
    const std::int64_t orbital_begin = basis.batch_orbital_offsets[batch_index];
    const std::size_t orbital_count =
        static_cast<std::size_t>(basis.batch_orbital_offsets[batch_index + 1u] - orbital_begin);
    const std::size_t matrix_offset = static_cast<std::size_t>(plan.matrix_offsets[batch_index]);

    /* All onsite moments are invariant when their atom and operator origin co-move. */
    for (std::int64_t ket_atom = atom_begin; ket_atom < atom_end; ++ket_atom) {
      const std::size_t ket_atom_index = static_cast<std::size_t>(ket_atom);
      for (std::int64_t bra_atom = atom_begin; bra_atom < ket_atom; ++bra_atom) {
        const std::size_t bra_atom_index = static_cast<std::size_t>(bra_atom);
        const double vector[3]{
            positions[ket_atom_index * 3u] - positions[bra_atom_index * 3u],
            positions[ket_atom_index * 3u + 1u] - positions[bra_atom_index * 3u + 1u],
            positions[ket_atom_index * 3u + 2u] - positions[bra_atom_index * 3u + 2u]};
        const std::int64_t ket_shell_begin = basis.atom_shell_offsets[ket_atom_index];
        const std::int64_t ket_shell_end = basis.atom_shell_offsets[ket_atom_index + 1u];
        const std::int64_t bra_shell_begin = basis.atom_shell_offsets[bra_atom_index];
        const std::int64_t bra_shell_end = basis.atom_shell_offsets[bra_atom_index + 1u];

        for (std::int64_t ket_shell = ket_shell_begin; ket_shell < ket_shell_end; ++ket_shell) {
          for (std::int64_t bra_shell = bra_shell_begin; bra_shell < bra_shell_end; ++bra_shell) {
            const std::size_t ket_shell_index = static_cast<std::size_t>(ket_shell);
            const std::size_t bra_shell_index = static_cast<std::size_t>(bra_shell);
            compute_shell_pair(basis, bra_shell_index, ket_shell_index, vector,
                               plan.integral_cutoff, true, true, scratch);

            const std::size_t bra_count = spherical_count(basis.angular_momenta[bra_shell_index]);
            const std::size_t ket_count = spherical_count(basis.angular_momenta[ket_shell_index]);
            const std::size_t block_size = bra_count * ket_count;
            const std::size_t bra_orbital = static_cast<std::size_t>(
                basis.shell_orbital_offsets[bra_shell_index] - orbital_begin);
            const std::size_t ket_orbital = static_cast<std::size_t>(
                basis.shell_orbital_offsets[ket_shell_index] - orbital_begin);
            std::array<double, 3> pair_gradient{};

            for (std::size_t bra_ao = 0; bra_ao < bra_count; ++bra_ao) {
              for (std::size_t ket_ao = 0; ket_ao < ket_count; ++ket_ao) {
                const std::size_t block_index = bra_ao * ket_count + ket_ao;
                const std::size_t forward =
                    matrix_offset + (bra_orbital + bra_ao) * orbital_count + ket_orbital + ket_ao;
                const std::size_t reverse =
                    matrix_offset + (ket_orbital + ket_ao) * orbital_count + bra_orbital + bra_ao;
                const double overlap = scratch.spherical[block_index];

                std::array<double, kDipoleComponents> dipole{};
                std::array<double, kDipoleComponents> dipole_adjoint{};
                std::array<double, kDipoleComponents> reverse_dipole_adjoint{};
                for (std::size_t component = 0; component < kDipoleComponents; ++component) {
                  dipole[component] =
                      scratch.spherical_multipole[component * block_size + block_index];
                  const double forward_adjoint = dE_ddipole[component * matrix_elements + forward];
                  const double reverse_adjoint = dE_ddipole[component * matrix_elements + reverse];
                  dipole_adjoint[component] = forward_adjoint;
                  reverse_dipole_adjoint[component] = reverse_adjoint;
                }

                std::array<double, kQuadrupoleComponents> quadrupole_adjoint{};
                std::array<double, kQuadrupoleComponents> reverse_quadrupole_adjoint{};
                for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
                  const double forward_adjoint =
                      dE_dquadrupole[component * matrix_elements + forward];
                  const double reverse_adjoint =
                      dE_dquadrupole[component * matrix_elements + reverse];
                  quadrupole_adjoint[component] = forward_adjoint + reverse_adjoint;
                  reverse_quadrupole_adjoint[component] = reverse_adjoint;
                }

                double overlap_adjoint = 0.0;
                std::array<double, 3> vector_adjoint{};
                add_multipole_shift_pullback(vector, overlap, dipole, reverse_dipole_adjoint,
                                             reverse_quadrupole_adjoint, overlap_adjoint,
                                             dipole_adjoint, vector_adjoint);

                for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
                  double derivative =
                      vector_adjoint[coordinate] +
                      overlap_adjoint *
                          scratch.spherical_gradient[coordinate * block_size + block_index];
                  const double* multipole_gradient = scratch.spherical_multipole_gradient.data() +
                                                     coordinate * kMultipoleComponents * block_size;
                  for (std::size_t component = 0; component < kDipoleComponents; ++component) {
                    derivative += dipole_adjoint[component] *
                                  multipole_gradient[component * block_size + block_index];
                  }
                  for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
                    derivative += quadrupole_adjoint[component] *
                                  multipole_gradient[(component + kDipoleComponents) * block_size +
                                                     block_index];
                  }
                  pair_gradient[coordinate] += derivative;
                }
              }
            }

            for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
              gradients[ket_atom_index * 3u + coordinate] += pair_gradient[coordinate];
              gradients[bra_atom_index * 3u + coordinate] -= pair_gradient[coordinate];
            }
          }
        }
      }
    }
  }

  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_overlap_gradient_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                          const double* positions, const double* dE_doverlap,
                                          double* gradients, void* workspace,
                                          std::size_t workspace_size, std::string& error) {
  xtbloom_status_t status =
      validate_evaluation(basis, plan, positions, workspace, workspace_size, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (dE_doverlap == nullptr || gradients == nullptr) {
    error = "overlap derivatives and gradients must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t element = 0; element < plan.total_matrix_elements; ++element) {
    if (!std::isfinite(dE_doverlap[static_cast<std::size_t>(element)])) {
      error = "overlap derivatives contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  auto& scratch = *static_cast<IntegralWorkspace*>(workspace);
  for (std::int64_t batch = 0; batch < basis.batch_size; ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    const std::int64_t atom_begin = basis.atom_offsets[batch_index];
    const std::int64_t atom_end = basis.atom_offsets[batch_index + 1u];
    const std::int64_t orbital_begin = basis.batch_orbital_offsets[batch_index];
    const std::size_t orbital_count =
        static_cast<std::size_t>(basis.batch_orbital_offsets[batch_index + 1u] - orbital_begin);
    const double* matrix_adjoint =
        dE_doverlap + static_cast<std::size_t>(plan.matrix_offsets[batch_index]);

    /* Intra-atomic overlap is invariant when all shells on the atom co-move. */
    for (std::int64_t ket_atom = atom_begin; ket_atom < atom_end; ++ket_atom) {
      const std::size_t ket_atom_index = static_cast<std::size_t>(ket_atom);
      for (std::int64_t bra_atom = atom_begin; bra_atom < ket_atom; ++bra_atom) {
        const std::size_t bra_atom_index = static_cast<std::size_t>(bra_atom);
        const double vector[3]{
            positions[ket_atom_index * 3u] - positions[bra_atom_index * 3u],
            positions[ket_atom_index * 3u + 1u] - positions[bra_atom_index * 3u + 1u],
            positions[ket_atom_index * 3u + 2u] - positions[bra_atom_index * 3u + 2u]};
        const std::int64_t ket_shell_begin = basis.atom_shell_offsets[ket_atom_index];
        const std::int64_t ket_shell_end = basis.atom_shell_offsets[ket_atom_index + 1u];
        const std::int64_t bra_shell_begin = basis.atom_shell_offsets[bra_atom_index];
        const std::int64_t bra_shell_end = basis.atom_shell_offsets[bra_atom_index + 1u];
        for (std::int64_t ket_shell = ket_shell_begin; ket_shell < ket_shell_end; ++ket_shell) {
          for (std::int64_t bra_shell = bra_shell_begin; bra_shell < bra_shell_end; ++bra_shell) {
            const std::size_t ket_shell_index = static_cast<std::size_t>(ket_shell);
            const std::size_t bra_shell_index = static_cast<std::size_t>(bra_shell);
            compute_shell_pair(basis, bra_shell_index, ket_shell_index, vector,
                               plan.integral_cutoff, true, false, scratch);

            const std::size_t bra_spherical_count =
                spherical_count(basis.angular_momenta[bra_shell_index]);
            const std::size_t ket_spherical_count =
                spherical_count(basis.angular_momenta[ket_shell_index]);
            const std::size_t spherical_block_size = bra_spherical_count * ket_spherical_count;
            const std::size_t bra_orbital = static_cast<std::size_t>(
                basis.shell_orbital_offsets[bra_shell_index] - orbital_begin);
            const std::size_t ket_orbital = static_cast<std::size_t>(
                basis.shell_orbital_offsets[ket_shell_index] - orbital_begin);
            std::array<double, 3> pair_gradient{};
            for (std::size_t bra_ao = 0; bra_ao < bra_spherical_count; ++bra_ao) {
              for (std::size_t ket_ao = 0; ket_ao < ket_spherical_count; ++ket_ao) {
                const double adjoint =
                    matrix_adjoint[(bra_orbital + bra_ao) * orbital_count + ket_orbital + ket_ao] +
                    matrix_adjoint[(ket_orbital + ket_ao) * orbital_count + bra_orbital + bra_ao];
                const std::size_t block_index = bra_ao * ket_spherical_count + ket_ao;
                for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
                  pair_gradient[coordinate] +=
                      adjoint *
                      scratch.spherical_gradient[coordinate * spherical_block_size + block_index];
                }
              }
            }
            for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
              gradients[ket_atom_index * 3u + coordinate] += pair_gradient[coordinate];
              gradients[bra_atom_index * 3u + coordinate] -= pair_gradient[coordinate];
            }
          }
        }
      }
    }
  }

  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn2
