#ifndef XTBLOOM_MODEL_COMMON_INTEGRALS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_COMMON_INTEGRALS_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/common/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::common {

/* Geometry-independent packed layout for ragged one-electron matrices. */
struct IntegralPlan {
  std::int64_t batch_size = 0;
  std::int64_t total_matrix_elements = 0;
  double integral_cutoff = 0.0;
  std::size_t workspace_size_bytes = 0;
  std::vector<std::int64_t> matrix_offsets;
};

/* tblite's default dimensionless Gaussian-product cutoff at accuracy 1.0. */
inline constexpr double kDefaultIntegralCutoff = 25.0;

xtbloom_status_t make_integral_plan(const BasisPlan& basis, IntegralPlan& plan, std::string& error,
                                    double integral_cutoff = kDefaultIntegralCutoff);

xtbloom_status_t evaluate_overlap_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                      const double* positions, double* overlap, void* workspace,
                                      std::size_t workspace_size, std::string& error);

xtbloom_status_t evaluate_multipole_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                        const double* positions, double* dipole, double* quadrupole,
                                        void* workspace, std::size_t workspace_size,
                                        std::string& error);

xtbloom_status_t add_multipole_gradient_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                            const double* positions, const double* dE_ddipole,
                                            const double* dE_dquadrupole, double* gradients,
                                            void* workspace, std::size_t workspace_size,
                                            std::string& error);

xtbloom_status_t add_overlap_gradient_cpu(const BasisPlan& basis, const IntegralPlan& plan,
                                          const double* positions, const double* dE_doverlap,
                                          double* gradients, void* workspace,
                                          std::size_t workspace_size, std::string& error);

}  // namespace xtbloom::detail::common

#endif  // XTBLOOM_MODEL_COMMON_INTEGRALS_HPP
