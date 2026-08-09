#ifndef XTBLOOM_MODEL_GFN2_ES3_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_ES3_HPP

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

/*
 * Geometry-independent storage for the shell-resolved GFN2 onsite cubic
 * charge term (ES3). shell_gamma3 follows BasisPlan shell order and stores
 *
 *   Gamma3_s = gam3_Z * {1, 1/2, 1/4}_{l=s,p,d}.
 *
 * Plan construction may allocate. Steady-state potential and energy
 * evaluation only read the plan and allocate no dynamic memory.
 */
struct ES3Plan {
  std::int64_t batch_size = 0;
  std::int64_t total_shells = 0;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<double> shell_gamma3;
};

/*
 * Non-owning, address-space-neutral ES3 data used by elementwise backends.
 * A CUDA or future ROCm backend can construct the same view over device
 * copies without depending on std::vector. Explicit buffer counts let the CPU
 * path reject truncated owning plans before dereferencing them. CPU evaluation
 * requires the pointers to be host-accessible for the full advertised ranges.
 */
struct ES3View {
  std::int64_t batch_size = 0;
  std::int64_t total_shells = 0;
  std::int64_t batch_shell_offset_count = 0;
  std::int64_t shell_gamma3_count = 0;
  const std::int64_t* batch_shell_offsets = nullptr;
  const double* shell_gamma3 = nullptr;
};

static_assert(std::is_trivially_copyable_v<ES3View>);
static_assert(std::is_standard_layout_v<ES3View>);

/*
 * Build ES3 parameters for an existing basis. atomic_numbers must be in the
 * exact batch-major atom order used to construct basis. The complete shell
 * metadata is cross-checked so a same-size but mismatched element list is
 * rejected rather than silently selecting the wrong gam3 values.
 */
xtbloom_status_t make_es3_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                               ES3Plan& plan, std::string& error);

/* Return a lightweight view over a plan; the plan must outlive the view. */
[[nodiscard]] ES3View make_es3_view(const ES3Plan& plan) noexcept;

/*
 * Overwrite one shell potential per charge using
 *
 *   v_s = Gamma3_s q_s^2.
 *
 * The routine validates the complete view, all charges, and all arithmetic
 * before modifying shell_potentials. Thus every reported failure leaves the
 * output unchanged. Input and output arrays must not overlap.
 */
xtbloom_status_t evaluate_es3_potential_cpu(ES3View view, const double* shell_charges,
                                            double* shell_potentials, std::string& error);

/*
 * Accumulate one Hartree energy per ragged batch member using
 *
 *   E3 = sum_s Gamma3_s q_s^3 / 3.
 *
 * Existing energies must be finite. The complete operation is preflighted,
 * including accumulated-output range, so a failure leaves every energy
 * unchanged. Input and output arrays must not overlap.
 */
xtbloom_status_t add_es3_energy_cpu(ES3View view, const double* shell_charges, double* energies,
                                    std::string& error);

/*
 * Overwrite the ES3 shell potential of exactly one ragged batch member.
 * shell_charges and shell_potentials address the complete packed arrays, while
 * numerical validation and arithmetic touch only system's shell slice. This
 * allows an SCC worker to prepare the Hamiltonian for a successful member
 * while peers may fail independently. Structural/binding errors return
 * INVALID_ARGUMENT; invalid target numerical data and range failures return
 * INTERNAL_ERROR and the target slice may be partially modified, so callers
 * must treat the whole target system as failed. The routine allocates no
 * memory and needs no scratch.
 */
xtbloom_status_t evaluate_es3_potential_system_cpu(ES3View view, std::int64_t system,
                                                   const double* shell_charges,
                                                   double* shell_potentials, std::string& error);

/*
 * Accumulate E3 for exactly one ragged batch member. shell_charges addresses
 * the complete packed array, while numerical validation and arithmetic touch
 * only system's shell slice. Structural/binding errors return
 * XTBLOOM_STATUS_INVALID_ARGUMENT; invalid target numerical data and range
 * failures return XTBLOOM_STATUS_INTERNAL_ERROR. accumulated_energy is unchanged
 * on every failure. The routine allocates no memory and needs no scratch.
 */
xtbloom_status_t add_es3_energy_system_cpu(ES3View view, std::int64_t system,
                                           const double* shell_charges, double& accumulated_energy,
                                           std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_ES3_HPP
