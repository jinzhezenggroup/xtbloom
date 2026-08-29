// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_MODEL_GFN2_PERIODIC_INTEGRALS_HPP
#define XTBLOOM_MODEL_GFN2_PERIODIC_INTEGRALS_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cpu_dispatch/features.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/periodic_topology.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

/* The definition remains private to periodic_topology.cpp. */
struct PeriodicShortRangePlanData;

/* Pinned tblite caps its exponent-derived one-electron image radius here. */
inline constexpr double kPeriodicIntegralMaximumCutoffBohr = 40.0;
inline constexpr std::size_t kPeriodicIntegralWorkspaceAlignment = 64u;

struct PeriodicIntegralPlanData {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_matrix_elements = 0;
  double integral_cutoff = 0.0;
  double minimum_primitive_exponent = 0.0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> translation_offsets;
  std::vector<LatticeTranslation> translations;

  /* Geometry-independent image radius used for each system. */
  std::vector<double> realspace_cutoffs;

  /* Raw workspace layout. Every offset is aligned for the pointed-to type. */
  std::size_t integral_workspace_offset = 0u;
  std::size_t overlap_scratch_offset = 0u;
  std::size_t dipole_scratch_offset = 0u;
  std::size_t quadrupole_scratch_offset = 0u;
  std::size_t h0_scratch_offset = 0u;
  std::size_t coordination_scratch_offset = 0u;
  std::size_t gradient_scratch_offset = 0u;
  std::size_t strain_scratch_offset = 0u;
  std::size_t workspace_size_bytes = 0u;

  const PeriodicShortRangePlanData* topology_identity = nullptr;
};

struct PeriodicIntegralTranslationView {
  const LatticeTranslation* data = nullptr;
  std::int64_t size = 0;
};

/*
 * Immutable image topology and caller-owned workspace schema for periodic
 * one-electron matrices. The image list is a complete rectangular superset
 * for the Gaussian product threshold; unlike the Wigner--Seitz short-range
 * pair topology, one-electron Gamma-point sums apply no 1/n image weights.
 */
class PeriodicIntegralPlan {
 public:
  PeriodicIntegralPlan() noexcept = default;
  PeriodicIntegralPlan(const PeriodicIntegralPlan&) noexcept = default;
  PeriodicIntegralPlan(PeriodicIntegralPlan&&) noexcept = default;
  PeriodicIntegralPlan& operator=(const PeriodicIntegralPlan&) noexcept = default;
  PeriodicIntegralPlan& operator=(PeriodicIntegralPlan&&) noexcept = default;
  ~PeriodicIntegralPlan() = default;

  [[nodiscard]] bool sealed() const noexcept { return data_ != nullptr; }
  [[nodiscard]] std::int64_t batch_size() const noexcept {
    return data_ == nullptr ? 0 : data_->batch_size;
  }
  [[nodiscard]] std::int64_t total_atoms() const noexcept {
    return data_ == nullptr ? 0 : data_->total_atoms;
  }
  [[nodiscard]] std::int64_t total_matrix_elements() const noexcept {
    return data_ == nullptr ? 0 : data_->total_matrix_elements;
  }
  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept {
    static const std::vector<std::int64_t> empty;
    return data_ == nullptr ? empty : data_->atom_offsets;
  }
  [[nodiscard]] const std::vector<std::int64_t>& matrix_offsets() const noexcept {
    static const std::vector<std::int64_t> empty;
    return data_ == nullptr ? empty : data_->matrix_offsets;
  }
  [[nodiscard]] double realspace_cutoff(std::int64_t system) const noexcept {
    if (data_ == nullptr || system < 0 || system >= data_->batch_size) return 0.0;
    const auto index = static_cast<std::size_t>(system);
    return index < data_->realspace_cutoffs.size() ? data_->realspace_cutoffs[index] : 0.0;
  }
  [[nodiscard]] PeriodicIntegralTranslationView translations(std::int64_t system) const noexcept {
    if (data_ == nullptr || system < 0 || system >= data_->batch_size ||
        data_->translation_offsets.size() != static_cast<std::size_t>(data_->batch_size + 1)) {
      return {};
    }
    const auto index = static_cast<std::size_t>(system);
    const auto begin = data_->translation_offsets[index];
    const auto end = data_->translation_offsets[index + 1u];
    if (begin < 0 || end < begin || static_cast<std::uint64_t>(end) > data_->translations.size()) {
      return {};
    }
    return {data_->translations.data() + static_cast<std::size_t>(begin), end - begin};
  }
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept {
    return data_ == nullptr ? 0u : data_->workspace_size_bytes;
  }
  [[nodiscard]] const PeriodicIntegralPlanData* identity() const noexcept { return data_.get(); }
  [[nodiscard]] const PeriodicIntegralPlanData* data() const noexcept { return data_.get(); }

  /* Reject numerical buffers forged inside immutable plan-owned vectors. */
  [[nodiscard]] bool overlaps_storage(const void* pointer, std::size_t bytes) const noexcept;

 private:
  explicit PeriodicIntegralPlan(std::shared_ptr<const PeriodicIntegralPlanData> data) noexcept
      : data_(std::move(data)) {}

  std::shared_ptr<const PeriodicIntegralPlanData> data_;

  friend xtbloom_status_t make_periodic_integral_plan(const BasisPlan&, const IntegralPlan&,
                                                      const PeriodicShortRangePlan&,
                                                      PeriodicIntegralPlan&, std::string&);
};

/* Build the immutable image lists and one raw workspace layout. */
xtbloom_status_t make_periodic_integral_plan(const BasisPlan& basis, const IntegralPlan& integrals,
                                             const PeriodicShortRangePlan& periodic,
                                             PeriodicIntegralPlan& plan, std::string& error);

/*
 * Evaluate the Gamma-point overlap, atomic dipole/quadrupole integrals, and
 * coordination-dependent H0 in one image traversal. Matrix buffers use the
 * packed layouts from IntegralPlan: one dense matrix per batch item, with
 * component-major storage for multipoles. The raw workspace is caller-owned
 * and must be at least PeriodicIntegralPlan::workspace_size_bytes() bytes,
 * aligned to 64 bytes. Outputs are published only after all staged values are
 * finite.
 */
xtbloom_status_t evaluate_periodic_integrals_h0_cpu(
    const BasisPlan& basis, const IntegralPlan& integrals, const H0Plan& h0,
    const PeriodicIntegralPlan& periodic, const PeriodicShortRangePlan& topology,
    const PeriodicShortRangeGeometry& geometry,
    const PeriodicShortRangeWorkspace& geometry_workspace, const double* coordination_numbers,
    double* overlap, double* dipole, double* quadrupole, double* hamiltonian, void* workspace,
    std::size_t workspace_size, std::string& error);

/*
 * Contract the same four matrix outputs in reverse mode. dE_dcn, gradients,
 * and strain_derivatives are accumulators and are changed only after the
 * complete traversal is finite. Strain is row-major dE/d epsilon following
 * r' = r (I + epsilon)^T and H' = H (I + epsilon)^T; self images therefore
 * have no Cartesian coordinate derivative but retain a cell derivative.
 */
xtbloom_status_t add_periodic_integrals_h0_vjp_cpu(
    const BasisPlan& basis, const IntegralPlan& integrals, const H0Plan& h0,
    const PeriodicIntegralPlan& periodic, const PeriodicShortRangePlan& topology,
    const PeriodicShortRangeGeometry& geometry,
    const PeriodicShortRangeWorkspace& geometry_workspace, const double* coordination_numbers,
    const double* dE_doverlap, const double* dE_ddipole, const double* dE_dquadrupole,
    const double* dE_dhamiltonian, double* dE_dcn, double* gradients, double* strain_derivatives,
    void* workspace, std::size_t workspace_size, std::string& error,
    CpuIsa cpu_isa = CpuIsa::kBaseline);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_PERIODIC_INTEGRALS_HPP
