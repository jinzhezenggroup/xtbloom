#ifndef XTBLOOM_MODEL_COMMON_STO_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_COMMON_STO_HPP

#include <cstddef>
#include <cstdint>

namespace xtbloom::detail::common {

inline constexpr std::size_t kMaximumContractedPrimitives = 12;

/* Select one pinned tblite Stewart STO-nG row. */
bool sto_table(std::uint8_t n, std::uint8_t l, std::uint8_t ng, const double*& alpha,
               const double*& coefficients) noexcept;

/* Expand and Cartesian-normalize one STO-nG shell. */
void expand_sto_shell(std::uint8_t n, std::uint8_t l, std::uint8_t ng, double slater,
                      double* alpha, double* coefficients) noexcept;

/* Apply tblite's first-matching-shell orthogonalization and renormalization. */
void orthogonalize_to_first(const double* first_alpha, const double* first_coefficients,
                            std::size_t first_count, double* alpha, double* coefficients,
                            std::size_t base_count) noexcept;

}  // namespace xtbloom::detail::common

#endif  // XTBLOOM_MODEL_COMMON_STO_HPP
