#ifndef GPUXTB_MODEL_GFN2_PERIODIC_EMBEDDING_HPP
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_MODEL_GFN2_PERIODIC_EMBEDDING_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail::gfn2 {

struct PeriodicEmbeddingPlanData;

/*
 * Immutable ragged-atom topology for a host-supplied periodic QM/MM charge
 * response. Each system owns one dense row-major n_atom by n_atom matrix.
 * Construction may allocate metadata; evaluation uses caller-owned buffers
 * only and allocates nothing on its successful hot path.
 */
class PeriodicEmbeddingPlan {
 public:
  PeriodicEmbeddingPlan() noexcept = default;
  PeriodicEmbeddingPlan(const PeriodicEmbeddingPlan&) noexcept = default;
  PeriodicEmbeddingPlan(PeriodicEmbeddingPlan&&) noexcept = default;
  PeriodicEmbeddingPlan& operator=(const PeriodicEmbeddingPlan&) noexcept = default;
  PeriodicEmbeddingPlan& operator=(PeriodicEmbeddingPlan&&) noexcept = default;
  ~PeriodicEmbeddingPlan() = default;

  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_atoms() const noexcept;
  [[nodiscard]] std::int64_t total_matrix_elements() const noexcept;
  [[nodiscard]] std::int64_t maximum_atoms() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& matrix_offsets() const noexcept;
  [[nodiscard]] bool overlaps_storage(const void* data, std::size_t size_bytes) const noexcept;
  [[nodiscard]] const PeriodicEmbeddingPlanData* identity() const noexcept;

 private:
  explicit PeriodicEmbeddingPlan(std::shared_ptr<const PeriodicEmbeddingPlanData> data) noexcept;
  std::shared_ptr<const PeriodicEmbeddingPlanData> data_;

  friend gpuxtb_status_t make_periodic_embedding_plan(std::int64_t batch_size,
                                                      std::int64_t total_atoms,
                                                      const std::int64_t* atom_offsets,
                                                      PeriodicEmbeddingPlan& plan,
                                                      std::string& error);
};

/*
 * Backend-neutral binding for
 *
 *   v = b + A q,  E = q^T b + 1/2 q^T A q.
 *
 * b, q, and v use the plan's packed atom order. A contains one dense
 * row-major symmetric matrix per system in matrix_offsets() order. Its upper
 * and lower entries must compare exactly equal as doubles; +0.0 and -0.0 are
 * therefore accepted as equal. Potentials and energies are overwritten, not
 * accumulated. system_statuses records independent numerical success or
 * failure for every member.
 */
struct PeriodicEmbeddingView {
  const double* shifts = nullptr;
  std::int64_t shift_elements = 0;
  const double* response_matrices = nullptr;
  std::int64_t response_elements = 0;
  const double* atomic_charges = nullptr;
  std::int64_t charge_elements = 0;
  double* atomic_potentials = nullptr;
  std::int64_t potential_elements = 0;
  double* energies = nullptr;
  std::int64_t energy_elements = 0;
  gpuxtb_status_t* system_statuses = nullptr;
  std::int64_t status_elements = 0;
  const PeriodicEmbeddingPlanData* plan_identity = nullptr;
};

/*
 * Compact scratch for one system. maximum_atoms() doubles suffice for both
 * the serial batch wrapper and a one-system worker. Distinct workspaces allow
 * different systems to be evaluated concurrently.
 */
struct PeriodicEmbeddingWorkspace {
  double* potential_scratch = nullptr;
  std::int64_t potential_elements = 0;
  const PeriodicEmbeddingPlanData* plan_identity = nullptr;
};

/* Build a plan from a monotone atom partition. Empty systems are supported. */
gpuxtb_status_t make_periodic_embedding_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                             const std::int64_t* atom_offsets,
                                             PeriodicEmbeddingPlan& plan, std::string& error);

/*
 * Validate and bind caller-owned numerical storage. Supplied element counts
 * may exceed the required extents, but only the required prefixes are active.
 * Across an evaluation, all active inputs, outputs, statuses, scratch, error
 * objects, descriptors, the plan object, and immutable plan backing storage
 * must be mutually disjoint. Exact and partial overlaps are rejected. The
 * output descriptor is unchanged on failure.
 */
gpuxtb_status_t bind_periodic_embedding_view(
    const PeriodicEmbeddingPlan& plan, const double* shifts, std::size_t shift_elements,
    const double* response_matrices, std::size_t response_elements, const double* atomic_charges,
    std::size_t charge_elements, double* atomic_potentials, std::size_t potential_elements,
    double* energies, std::size_t energy_elements, gpuxtb_status_t* system_statuses,
    std::size_t status_elements, PeriodicEmbeddingView& view, std::string& error);

/*
 * Bind caller-owned scratch. Scratch must not overlap the plan, destination
 * descriptor, or error object; evaluation additionally rejects overlap with
 * any active view range. The output descriptor is unchanged on failure.
 */
gpuxtb_status_t bind_periodic_embedding_workspace(const PeriodicEmbeddingPlan& plan,
                                                  double* potential_scratch,
                                                  std::size_t potential_elements,
                                                  PeriodicEmbeddingWorkspace& workspace,
                                                  std::string& error);

/*
 * Evaluate one system. Structural errors publish nothing. Nonfinite values,
 * a nonsymmetric A, or floating-point range failure leave that system's v and
 * E unchanged and publish GPUXTB_STATUS_INTERNAL_ERROR only to its status.
 * Calling different systems concurrently requires distinct workspaces and
 * distinct std::string objects; concurrent calls for the same system are not
 * supported.
 */
gpuxtb_status_t evaluate_periodic_embedding_system_cpu(const PeriodicEmbeddingPlan& plan,
                                                       std::int64_t system,
                                                       const PeriodicEmbeddingView& view,
                                                       const PeriodicEmbeddingWorkspace& workspace,
                                                       std::string& error);

/*
 * Serial ragged-batch wrapper. Structural validation is whole-call atomic.
 * Numerical failures are isolated: all peers are still attempted, and the
 * call returns GPUXTB_STATUS_INTERNAL_ERROR after processing the full batch.
 */
gpuxtb_status_t evaluate_periodic_embedding_batch_cpu(const PeriodicEmbeddingPlan& plan,
                                                      const PeriodicEmbeddingView& view,
                                                      const PeriodicEmbeddingWorkspace& workspace,
                                                      std::string& error);

}  // namespace gpuxtb::detail::gfn2

#endif  // GPUXTB_MODEL_GFN2_PERIODIC_EMBEDDING_HPP
