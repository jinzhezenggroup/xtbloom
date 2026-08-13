#ifndef XTBLOOM_MODEL_GFN1_D3_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN1_D3_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn1 {

inline constexpr std::size_t kD3WorkspaceAlignment = 64u;
inline constexpr std::size_t kD3MaximumReferences = 7u;

struct D3PlanData;

/*
 * Immutable ragged topology and GFN1 D3(BJ) element-pair parameters.
 *
 * Setup precomputes charge-independent pair damping data. Evaluation consumes
 * the ordinary GFN1 exponential coordination numbers; it must not substitute
 * GFN2's D4-specific coordination model. GFN1 has s9 == 0, so this plan owns
 * no ATM topology or workspace.
 */
class D3Plan {
 public:
  D3Plan() noexcept = default;
  D3Plan(const D3Plan&) noexcept = default;
  D3Plan(D3Plan&&) noexcept = default;
  D3Plan& operator=(const D3Plan&) noexcept = default;
  D3Plan& operator=(D3Plan&&) noexcept = default;
  ~D3Plan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_atoms() const noexcept;
  [[nodiscard]] std::int64_t total_pairs() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& pair_offsets() const noexcept;
  [[nodiscard]] bool matches_atomic_numbers(const std::int32_t* atomic_numbers) const noexcept;
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;
  [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
  [[nodiscard]] const D3PlanData* identity() const noexcept;

 private:
  explicit D3Plan(std::shared_ptr<const D3PlanData> data) noexcept;
  std::shared_ptr<const D3PlanData> data_;

  friend xtbloom_status_t make_d3_plan(std::int64_t, std::int64_t, const std::int64_t*,
                                       const std::int32_t*, D3Plan&, std::string&);
};

/*
 * Canonical caller-owned scratch. Counts are numbers of doubles.
 * bind_d3_workspace seals the pointer layout to one plan identity; callers
 * may retain the descriptor but must not mutate its fields before evaluation.
 */
struct D3Workspace {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0u;
  double* weights = nullptr;
  double* weight_cn_derivatives = nullptr;
  std::int64_t weight_elements = 0;
  double* coordination_adjoints = nullptr;
  std::int64_t atom_elements = 0;
  double* batch_scratch = nullptr;
  std::int64_t batch_elements = 0;
  double* gradient_scratch = nullptr;
  std::int64_t gradient_elements = 0;
  const D3PlanData* plan_identity = nullptr;
};

xtbloom_status_t make_d3_plan(std::int64_t batch_size, std::int64_t total_atoms,
                              const std::int64_t* atom_offsets, const std::int32_t* atomic_numbers,
                              D3Plan& plan, std::string& error);

xtbloom_status_t bind_d3_workspace(const D3Plan& plan, void* workspace, std::size_t workspace_size,
                                   D3Workspace& view, std::string& error);

/*
 * Accumulate charge-independent GFN1 D3(BJ) energy and optional gradients.
 *
 * Positions are in bohr, energies in Hartree, and gradients are dE/dR in
 * Hartree/bohr (not forces). The gradient includes both the explicit pair
 * derivative and the C6-reference interpolation path through GFN1 CN. Every
 * input and accumulator is validated before caller output is modified. A
 * successful call allocates nothing and preserves s9 == 0 by evaluating no
 * three-body term.
 */
xtbloom_status_t add_d3_cpu(const D3Plan& plan, const double* positions,
                            const double* coordination_numbers, double* energies, double* gradients,
                            const D3Workspace& workspace, std::string& error);

}  // namespace xtbloom::detail::gfn1

#endif  // XTBLOOM_MODEL_GFN1_D3_HPP
