#ifndef XTBLOOM_MODEL_GFN2_ALPB_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_ALPB_HPP

#include <cstdint>
#include <string>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

/* Born interaction kernels implemented by tblite's ALPB/GBSA polar term. */
enum class AlpbBornKernel : std::int32_t {
  kStill = 1,
  kP16 = 2,
};

/*
 * ALPB adds the electrostatic-size correction; GBSA is its conducting-limit
 * generalized-Born form without that correction.
 */
enum class AlpbPolarModel : std::int32_t {
  kAlpb = 1,
  kGbsa = 2,
};

struct AlpbPolarSettings {
  AlpbPolarModel model = AlpbPolarModel::kAlpb;
  AlpbBornKernel kernel = AlpbBornKernel::kP16;
  double dielectric_constant = 1.0;
};

/*
 * Code-compatible ALPB constant from tblite revision 133f91ef (v0.7.0-15).
 * The tblite prose specification contains a transposed pair of digits
 * (0.571214); the executable implementation uses 0.571412.
 */
inline constexpr double kAlpbAlpha = 0.571412;

/*
 * Build the symmetric row-major polar interaction matrix J in atomic units.
 * Positions, Born radii, and optional ALPB cavity radii are in bohr, so J is
 * in Hartree/electron^2. `cavity_radii` is required for ALPB and ignored for
 * GBSA. A zero-atom input is an empty no-op and permits null array pointers.
 * The caller owns all storage and must keep output disjoint from input.
 */
xtbloom_status_t build_alpb_polar_matrix_cpu(std::int64_t atom_count, const double* positions,
                                             const double* born_radii, const double* cavity_radii,
                                             const AlpbPolarSettings& settings, double* matrix,
                                             std::string& error);

/*
 * Evaluate V = J q and E = 0.5 q^T J q for a previously built row-major J.
 * Atomic charges are in electrons, potentials in Hartree/electron, and the
 * energy is in Hartree. A zero-atom evaluation publishes exactly zero energy
 * and permits null matrix, charge, and potential arrays.
 */
xtbloom_status_t evaluate_alpb_polar_cpu(std::int64_t atom_count, const double* matrix,
                                         const double* atomic_charges, double* atomic_potentials,
                                         double* energy, std::string& error);

/*
 * Accumulate dE/dR in Hartree/bohr for fixed atomic charges, Born radii, and
 * cavity radii. Derivatives of a future geometry-dependent Born-radius model
 * must be composed separately through its radius Jacobian. A zero-atom input
 * is an empty no-op and permits null array pointers. The P16 coordinate
 * derivative is undefined at exact atom coincidence, so this call rejects
 * that geometry even though the corresponding matrix elements remain finite.
 */
xtbloom_status_t add_alpb_polar_gradient_cpu(std::int64_t atom_count, const double* positions,
                                             const double* born_radii, const double* cavity_radii,
                                             const double* atomic_charges,
                                             const AlpbPolarSettings& settings, double* gradients,
                                             std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_ALPB_HPP
