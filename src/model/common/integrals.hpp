#ifndef XTBLOOM_MODEL_COMMON_INTEGRALS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_COMMON_INTEGRALS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cpu_dispatch/features.hpp"
#include "model/common/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::common {

inline constexpr std::size_t kIntegralMaximumCartesianFunctions = 6u;
inline constexpr std::size_t kIntegralMaximumSphericalFunctions = 5u;
inline constexpr std::size_t kIntegralMaximumCartesianBlock =
    kIntegralMaximumCartesianFunctions * kIntegralMaximumCartesianFunctions;
inline constexpr std::size_t kIntegralMaximumSphericalBlock =
    kIntegralMaximumSphericalFunctions * kIntegralMaximumSphericalFunctions;
inline constexpr std::size_t kIntegralDipoleComponents = 3u;
inline constexpr std::size_t kIntegralQuadrupoleComponents = 6u;
inline constexpr std::size_t kIntegralMultipoleComponents =
    kIntegralDipoleComponents + kIntegralQuadrupoleComponents;

/*
 * The fixed-size shell-pair scratch is shared by the molecular evaluator and
 * the periodic image evaluator.  Keeping the layout in the internal header
 * lets the latter reuse exactly the same recurrence and spherical transform
 * without changing the molecular loop or adding per-image allocations.
 */
struct alignas(double) IntegralWorkspace {
  std::array<double, kIntegralMaximumCartesianBlock> cartesian;
  std::array<double, 3u * kIntegralMaximumCartesianBlock> cartesian_gradient;
  std::array<double, kIntegralMaximumSphericalBlock> spherical;
  std::array<double, 3u * kIntegralMaximumSphericalBlock> spherical_gradient;
  /* Components are [x,y,z,xx,xy,yy,xz,yz,zz], one shell block each. */
  std::array<double, kIntegralMultipoleComponents * kIntegralMaximumCartesianBlock>
      cartesian_multipole;
  std::array<double, kIntegralMultipoleComponents * kIntegralMaximumSphericalBlock>
      spherical_multipole;
  /* Derivative layout is [coordinate][component][shell-block element]. */
  std::array<double, 3u * kIntegralMultipoleComponents * kIntegralMaximumCartesianBlock>
      cartesian_multipole_gradient;
  std::array<double, 3u * kIntegralMultipoleComponents * kIntegralMaximumSphericalBlock>
      spherical_multipole_gradient;
};

/*
 * Evaluate one shell pair into IntegralWorkspace.  This is intentionally an
 * internal, allocation-free bridge: periodic callers pass the same vectors
 * and flags as the molecular evaluator, preserving the reviewed shell-pair
 * arithmetic and all spherical/multipole conventions.
 */
void compute_shell_pair_cpu(const BasisPlan& basis, std::size_t bra_shell, std::size_t ket_shell,
                            const double vector[3], double integral_cutoff, bool with_gradient,
                            bool with_multipoles, IntegralWorkspace& workspace);

/* Geometry-independent packed layout for ragged one-electron matrices. */
struct IntegralPlan {
  std::int64_t batch_size = 0;
  std::int64_t total_matrix_elements = 0;
  double integral_cutoff = 0.0;
  std::size_t workspace_size_bytes = 0;
  std::vector<std::int64_t> matrix_offsets;
};

/* Validate the immutable basis/integral relationship without numerical inputs. */
xtbloom_status_t validate_integral_plan(const BasisPlan& basis, const IntegralPlan& plan,
                                        std::string& error);

/* tblite's default dimensionless Gaussian-product cutoff at accuracy 1.0. */
inline constexpr double kDefaultIntegralCutoff = 25.0;

/*
 * Context-selected implementation of the multipole-gradient shell-pair
 * arithmetic. The opaque workspace points to the evaluator's preallocated
 * integral scratch buffer; keeping its private layout out of this header gives
 * each ISA variant a narrow, allocation-free boundary without expanding the
 * model interface.
 */
using MultipoleGradientShellPairKernel = void (*)(const BasisPlan&, std::size_t, std::size_t,
                                                  const double*, double, void*) noexcept;

struct IntegralKernelTable {
  MultipoleGradientShellPairKernel multipole_gradient_shell_pair = nullptr;
  CpuIsa isa = CpuIsa::kBaseline;
};

[[nodiscard]] const IntegralKernelTable& integral_baseline_kernels() noexcept;
[[nodiscard]] const IntegralKernelTable& integral_avx2_fma_kernels() noexcept;
[[nodiscard]] const IntegralKernelTable& integral_kernels_for_cpu_isa(CpuIsa isa) noexcept;

xtbloom_status_t make_integral_plan(const BasisPlan& basis, IntegralPlan& plan, std::string& error,
                                    double integral_cutoff = kDefaultIntegralCutoff);

xtbloom_status_t evaluate_overlap_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                      const double* positions, double* overlap, void* workspace,
                                      std::size_t workspace_size, std::string& error);

/*
 * Evaluate cross overlap between two one-system Gaussian bases.  The matrix
 * is row-major [bra_orbital, ket_orbital] and is intentionally not assumed to
 * be symmetric.  This private research primitive is used for the native GFN2
 * AO basis (bra) and the auxiliary projection basis (ket).
 */
xtbloom_status_t evaluate_cross_overlap_system_cpu(const BasisPlan& bra_basis,
                                                   const BasisPlan& ket_basis,
                                                   const double* bra_positions,
                                                   const double* ket_positions, double* overlap,
                                                   void* workspace, std::size_t workspace_size,
                                                   std::string& error);

/*
 * Evaluate d(cross-overlap)/dR for co-moving atom centers.  The output is
 * [derivative_atom, xyz, bra_orbital, ket_orbital].  For a pair of distinct
 * centers the ket-center derivative is added to the ket atom and its negative
 * to the bra atom; same-center terms cancel under rigid translation.
 */
xtbloom_status_t evaluate_cross_overlap_gradient_system_cpu(
    const BasisPlan& bra_basis, const BasisPlan& ket_basis, const double* bra_positions,
    const double* ket_positions, double* overlap_gradient, void* workspace,
    std::size_t workspace_size, std::string& error);

/* Evaluate the full AO-overlap Cartesian derivative for one cached system.
 * The output layout is [atom, xyz, nao, nao] in row-major AO order. This
 * research-only export reuses the same shell-pair recurrence as the reviewed
 * force path and is consumed by the external energy projection/Pulay callback. */
xtbloom_status_t evaluate_overlap_gradient_system_cpu(
    const BasisPlan& basis, const IntegralPlan& plan, const double* positions,
    double* overlap_gradient, void* workspace, std::size_t workspace_size, std::string& error);

xtbloom_status_t evaluate_multipole_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                        const double* positions, double* dipole, double* quadrupole,
                                        void* workspace, std::size_t workspace_size,
                                        std::string& error);

xtbloom_status_t add_multipole_gradient_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                            const double* positions, const double* dE_ddipole,
                                            const double* dE_dquadrupole, double* gradients,
                                            void* workspace, std::size_t workspace_size,
                                            std::string& error, CpuIsa cpu_isa = CpuIsa::kBaseline);

xtbloom_status_t add_overlap_gradient_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                          const double* positions, const double* dE_doverlap,
                                          double* gradients, void* workspace,
                                          std::size_t workspace_size, std::string& error);

}  // namespace xtbloom::detail::common

#endif  // XTBLOOM_MODEL_COMMON_INTEGRALS_HPP
