#ifndef XTBLOOM_MODEL_COMMON_BASIS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_COMMON_BASIS_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>
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

/*
 * Heap storage retained by a sealed basis plan. Runtime workspace reporting
 * uses vector capacities because those are the actual owned allocations, not
 * merely the currently populated element counts. Keep this inventory beside
 * BasisPlan so every backend accounts for newly added metadata together.
 */
inline std::size_t basis_plan_resident_bytes(const BasisPlan& plan) noexcept {
  const auto vector_bytes = [](const auto& values) noexcept {
    using Value = typename std::decay_t<decltype(values)>::value_type;
    return values.capacity() * sizeof(Value);
  };
  return vector_bytes(plan.atom_offsets) + vector_bytes(plan.batch_shell_offsets) +
         vector_bytes(plan.batch_orbital_offsets) +
         vector_bytes(plan.batch_cartesian_orbital_offsets) +
         vector_bytes(plan.batch_primitive_offsets) + vector_bytes(plan.atom_shell_offsets) +
         vector_bytes(plan.atom_orbital_offsets) +
         vector_bytes(plan.atom_cartesian_orbital_offsets) +
         vector_bytes(plan.atom_primitive_offsets) + vector_bytes(plan.shell_orbital_offsets) +
         vector_bytes(plan.shell_cartesian_orbital_offsets) +
         vector_bytes(plan.shell_primitive_offsets) + vector_bytes(plan.shell_to_atom) +
         vector_bytes(plan.principal_quantum_numbers) + vector_bytes(plan.angular_momenta) +
         vector_bytes(plan.shell_is_valence) + vector_bytes(plan.slater_exponents) +
         vector_bytes(plan.primitive_exponents) + vector_bytes(plan.primitive_coefficients);
}

}  // namespace xtbloom::detail::common

#endif  // XTBLOOM_MODEL_COMMON_BASIS_HPP
