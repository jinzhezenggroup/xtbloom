#ifndef XTBLOOM_MODEL_GFN2_BASIS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_BASIS_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

/*
 * Geometry-independent GFN2 basis metadata for a complete ragged batch.
 *
 * Atoms, shells, spherical orbitals, Cartesian orbitals, and primitives are
 * all stored batch-major and then atom-major. Every *_offsets array uses
 * zero-based half-open ranges. Primitive coefficients include the Cartesian
 * Gaussian normalization used by tblite before its spherical transformation.
 * Plan construction may allocate; downstream integral evaluation only reads
 * these arrays and can copy them directly to a device backend.
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
  std::vector<double> slater_exponents;

  std::vector<double> primitive_exponents;
  std::vector<double> primitive_coefficients;
};

/*
 * Expand the generated, pinned GFN2 element parameters into a reusable basis
 * plan. The STO-nG expansion and normalization follow tblite's
 * tblite_basis_slater implementation exactly, including its special 6s/6p
 * tables. Repeated angular-momentum shells, if introduced by a future
 * parameter set, receive tblite's first-shell Gram-Schmidt treatment.
 */
xtbloom_status_t make_basis_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                 const std::int64_t* atom_offsets,
                                 const std::int32_t* atomic_numbers, BasisPlan& plan,
                                 std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_BASIS_HPP
