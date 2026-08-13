#ifndef XTBLOOM_MODEL_GFN1_ES3_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_ES3_HPP

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "model/gfn1/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn1 {

/*
 * Geometry-independent GFN1 atom-resolved third-order electrostatics. Unlike
 * GFN2, Gamma3 is indexed by atom and consumes atomic rather than shell
 * charges. This distinction is explicit in both owning and non-owning layouts.
 */
struct ES3Plan {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<double> atom_gamma3;
};

struct ES3View {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t atom_offset_count = 0;
  std::int64_t atom_gamma3_count = 0;
  const std::int64_t* atom_offsets = nullptr;
  const double* atom_gamma3 = nullptr;
};

static_assert(std::is_trivially_copyable_v<ES3View>);
static_assert(std::is_standard_layout_v<ES3View>);

xtbloom_status_t make_es3_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                               ES3Plan& plan, std::string& error);

[[nodiscard]] ES3View make_es3_view(const ES3Plan& plan) noexcept;

/* Overwrite v_A = Gamma3_A q_A^2 without allocating or partially publishing. */
xtbloom_status_t evaluate_es3_potential_cpu(ES3View view, const double* atomic_charges,
                                            double* atomic_potentials, std::string& error);

/* Accumulate E3 = sum_A Gamma3_A q_A^3 / 3 into one energy per batch member. */
xtbloom_status_t add_es3_energy_cpu(ES3View view, const double* atomic_charges, double* energies,
                                    std::string& error);

/*
 * One-system variants inspect only the target ragged slice. Numerical failure
 * is therefore peer-local; the energy accumulator is unchanged on failure.
 */
xtbloom_status_t evaluate_es3_potential_system_cpu(ES3View view, std::int64_t system,
                                                   const double* atomic_charges,
                                                   double* atomic_potentials,
                                                   std::string& error);

xtbloom_status_t add_es3_energy_system_cpu(ES3View view, std::int64_t system,
                                           const double* atomic_charges,
                                           double& accumulated_energy, std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_ES3_HPP
