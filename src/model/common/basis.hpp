#ifndef XTBLOOM_MODEL_COMMON_BASIS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_COMMON_BASIS_HPP

#include <cstdint>
#include <vector>

namespace xtbloom::detail::common {

/*
 * Geometry-independent basis metadata shared by model-specific setup paths.
 *
 * Atoms, shells, spherical orbitals, Cartesian orbitals, and primitives are
 * stored batch-major and atom-major. Every offset array is a zero-based
 * half-open partition. Primitive coefficients include the Cartesian Gaussian
 * normalization used by tblite before its spherical transformation.
 *
 * shell_is_valence records model-owned setup semantics rather than inferring
 * them later from shell ordering. GFN1 needs this distinction for hydrogen's
 * repeated 1s/2s shells; GFN2 marks every current shell as valence.
 */
struct BasisPlan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t total_cartesian_orbitals = 0;
  std::int64_t total_primitives = 0;
  std::uint8_t maximum_angular_momentum = 0;
  double minimum_primitive_exponent = 0.0;

  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> batch_orbital_offsets;
  std::vector<std::int64_t> batch_cartesian_orbital_offsets;
  std::vector<std::int64_t> batch_primitive_offsets;

  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> atom_orbital_offsets;
  std::vector<std::int64_t> atom_cartesian_orbital_offsets;
  std::vector<std::int64_t> atom_primitive_offsets;

  std::vector<std::int64_t> shell_orbital_offsets;
  std::vector<std::int64_t> shell_cartesian_orbital_offsets;
  std::vector<std::int64_t> shell_primitive_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::uint8_t> principal_quantum_numbers;
  std::vector<std::uint8_t> angular_momenta;
  std::vector<std::uint8_t> shell_is_valence;
  std::vector<double> slater_exponents;

  std::vector<double> primitive_exponents;
  std::vector<double> primitive_coefficients;
};

}  // namespace xtbloom::detail::common

#endif  // XTBLOOM_MODEL_COMMON_BASIS_HPP
