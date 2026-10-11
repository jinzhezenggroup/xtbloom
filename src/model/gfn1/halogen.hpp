#ifndef XTBLOOM_MODEL_GFN1_HALOGEN_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_HALOGEN_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn1 {

inline constexpr std::size_t kHalogenWorkspaceAlignment = 64u;

struct HalogenPlanData;

/*
 * Immutable geometry-independent data for the classical GFN1 halogen-bond
 * correction. Radii are the 1.3-scaled Mantina atomic radii in bohr and bond
 * strengths are in Hartree. Setup may allocate; evaluation does not.
 */
class HalogenPlan {
 public:
  HalogenPlan() noexcept = default;
  HalogenPlan(const HalogenPlan&) noexcept = default;
  HalogenPlan(HalogenPlan&&) noexcept = default;
  HalogenPlan& operator=(const HalogenPlan&) noexcept = default;
  HalogenPlan& operator=(HalogenPlan&&) noexcept = default;
  ~HalogenPlan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_atoms() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] bool matches_atomic_numbers(const std::int32_t* atomic_numbers) const noexcept;
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] const HalogenPlanData* identity() const noexcept;

 private:
  explicit HalogenPlan(std::shared_ptr<const HalogenPlanData> data) noexcept;
  std::shared_ptr<const HalogenPlanData> data_;

  friend xtbloom_status_t make_halogen_plan(std::int64_t, std::int64_t, const std::int64_t*,
                                            const std::int32_t*, HalogenPlan&, std::string&);
};

/*
 * Canonical caller-owned unpublished storage. axis_neighbors records the
 * globally nearest positive-distance atom for each active donor, while the
 * remaining arrays stage the complete outputs until evaluation succeeds.
 */
struct HalogenWorkspace {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  std::int64_t* axis_neighbors = nullptr;
  std::int64_t axis_neighbor_elements = 0;
  double* batch_scratch = nullptr;
  std::int64_t batch_elements = 0;
  double* force_scratch = nullptr;
  std::int64_t force_elements = 0;
  const HalogenPlanData* plan_identity = nullptr;
};

/* Build a reusable GFN1 halogen plan from one fixed ragged topology. */
xtbloom_status_t make_halogen_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                   const std::int64_t* atom_offsets,
                                   const std::int32_t* atomic_numbers, HalogenPlan& plan,
                                   std::string& error);

/* Bind a sufficiently large 64-byte-aligned caller allocation to the plan. */
xtbloom_status_t bind_halogen_workspace(const HalogenPlan& plan, void* workspace,
                                        std::size_t workspace_size, HalogenWorkspace& view,
                                        std::string& error);

/*
 * Accumulate the GFN1 halogen energy and optional analytic forces. Positions
 * use bohr, energies use Hartree, and forces use Hartree/bohr. For each donor,
 * the lowest-index atom at the strictly smallest positive distance defines
 * the donor axis; all donor-acceptor pairs at or below 20 bohr contribute.
 * The complete request is validated and staged before caller outputs change.
 */
xtbloom_status_t add_halogen_cpu(const HalogenPlan& plan, const double* positions, double* energies,
                                 double* forces, const HalogenWorkspace& workspace,
                                 std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_HALOGEN_HPP
