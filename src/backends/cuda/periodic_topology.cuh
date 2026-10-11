// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_BACKENDS_CUDA_PERIODIC_TOPOLOGY_CUH
#define XTBLOOM_BACKENDS_CUDA_PERIODIC_TOPOLOGY_CUH

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::cuda {

/*
 * Canonical device-side topology errors.  The device validator writes only a
 * small POD record so callers can inspect a rejected topology without
 * downloading any of the potentially large translation arrays.
 */
enum class Gfn2CudaPeriodicTopologyError : std::uint32_t {
  kSuccess = 0u,
  kInvalidDimensions = 1u,
  kInvalidOffsets = 2u,
  kInvalidPeriodicity = 3u,
  kUnsupportedPeriodicity = 4u,
  kInvalidCell = 5u,
  kInvalidTranslations = 6u,
  kNonfiniteTranslation = 7u,
  kInvalidPlanIdentity = 8u,
  kAllocationFailed = 9u,
  kCudaFailure = 10u,
};

enum class Gfn2CudaPeriodicTopologyField : std::uint32_t {
  kNone = 0u,
  kDimensions = 1u,
  kAtomOffsets = 2u,
  kCellMatrices = 3u,
  kPeriodicAxes = 4u,
  kTranslationOffsets = 5u,
  kTranslations = 6u,
  kPlanIdentity = 7u,
};

struct Gfn2CudaPeriodicTopologyDeviceDiagnostic {
  std::uint32_t error = static_cast<std::uint32_t>(Gfn2CudaPeriodicTopologyError::kSuccess);
  std::uint32_t field = static_cast<std::uint32_t>(Gfn2CudaPeriodicTopologyField::kNone);
  std::int64_t index = -1;
};

/* Host-visible result of construction or launch validation. */
struct Gfn2CudaPeriodicTopologyDiagnostic {
  xtbloom_status_t status = XTBLOOM_STATUS_SUCCESS;
  Gfn2CudaPeriodicTopologyError error = Gfn2CudaPeriodicTopologyError::kSuccess;
  Gfn2CudaPeriodicTopologyField field = Gfn2CudaPeriodicTopologyField::kNone;
  std::int64_t index = -1;
  std::int32_t cuda_status = static_cast<std::int32_t>(cudaSuccess);

  [[nodiscard]] bool success() const noexcept { return status == XTBLOOM_STATUS_SUCCESS; }
};

/* One translation in the row-major, atom-independent image list. */
struct Gfn2CudaPeriodicTranslation {
  std::int64_t index[3]{};
  double cartesian[3]{};
};

/*
 * Device view consumed by future periodic CUDA terms.  All arrays are
 * immutable after upload and remain owned by Gfn2CudaPeriodicTopology.
 * `translation_offsets[i] : translation_offsets[i + 1]` is the complete
 * rectangular image list for system i; the origin is always first.
 */
struct Gfn2CudaPeriodicTopologyView {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_translations = 0;
  std::int64_t atom_offset_count = 0;
  std::int64_t cell_elements = 0;
  std::int64_t periodic_axes_elements = 0;
  std::int64_t translation_offset_count = 0;
  std::uint64_t plan_token = 0u;
  std::uint64_t cell_generation = 0u;
  const std::int64_t* atom_offsets = nullptr;
  const double* cell_matrices = nullptr;
  const std::int32_t* periodic_axes = nullptr;
  const std::int64_t* translation_offsets = nullptr;
  const Gfn2CudaPeriodicTranslation* translations = nullptr;
};

static_assert(std::is_trivially_copyable_v<Gfn2CudaPeriodicTranslation>);
static_assert(std::is_standard_layout_v<Gfn2CudaPeriodicTranslation>);
static_assert(std::is_trivially_copyable_v<Gfn2CudaPeriodicTopologyView>);
static_assert(std::is_standard_layout_v<Gfn2CudaPeriodicTopologyView>);
static_assert(std::is_trivially_copyable_v<Gfn2CudaPeriodicTopologyDeviceDiagnostic>);
static_assert(std::is_standard_layout_v<Gfn2CudaPeriodicTopologyDeviceDiagnostic>);

/*
 * The owner accepts complete HOST arrays.  Device/mixed public descriptors are
 * staged by the existing runtime before this setup-time owner is called.  The
 * explicit generations and token prevent an accidentally reused device view
 * from being mistaken for a compatible fixed plan.
 */
struct Gfn2CudaPeriodicTopologyInput {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  const std::int64_t* atom_offsets = nullptr;
  const double* cell_matrices = nullptr;
  const std::int32_t* periodic_axes = nullptr;
  double image_cutoff = 25.0;
  std::uint64_t plan_token = 0u;
  std::uint64_t cell_generation = 0u;
};

/*
 * Fixed-plan Graph/cache identity.  Content fingerprint covers cells, masks,
 * offsets, cutoff, and canonical translations.  Geometry generation remains
 * separate because geometry refreshes may reuse immutable topology storage.
 * requested_result_flags and unsupported_interaction_mask make a graph built
 * for one publication contract ineligible for a different one.
 */
struct Gfn2CudaPeriodicGraphIdentity {
  std::uint64_t plan_token = 0u;
  std::uint64_t topology_fingerprint = 0u;
  std::uint64_t cell_generation = 0u;
  std::uint64_t geometry_generation = 0u;
  std::uint32_t requested_result_flags = 0u;
  std::uint32_t unsupported_interaction_mask = 0u;
  std::uint8_t strain_requested = 0u;
  std::uint8_t reserved[7]{};
};

static_assert(std::is_trivially_copyable_v<Gfn2CudaPeriodicGraphIdentity>);
static_assert(std::is_standard_layout_v<Gfn2CudaPeriodicGraphIdentity>);

[[nodiscard]] bool same_gfn2_cuda_periodic_graph_identity(
    const Gfn2CudaPeriodicGraphIdentity& first,
    const Gfn2CudaPeriodicGraphIdentity& second) noexcept;

/*
 * Setup-time CUDA owner for canonical native-cell translations.  Construction
 * is transactional: `output` is unchanged when host validation, allocation,
 * or any asynchronous upload fails.  Uploads are ordered on `stream`; the
 * returned owner must outlive all work that consumes its device_view().
 */
class Gfn2CudaPeriodicTopology {
 public:
  Gfn2CudaPeriodicTopology() noexcept = default;
  ~Gfn2CudaPeriodicTopology();

  Gfn2CudaPeriodicTopology(const Gfn2CudaPeriodicTopology&) = delete;
  Gfn2CudaPeriodicTopology& operator=(const Gfn2CudaPeriodicTopology&) = delete;
  Gfn2CudaPeriodicTopology(Gfn2CudaPeriodicTopology&& other) noexcept;
  Gfn2CudaPeriodicTopology& operator=(Gfn2CudaPeriodicTopology&& other) noexcept;

  [[nodiscard]] static Gfn2CudaPeriodicTopologyDiagnostic create(
      const Gfn2CudaPeriodicTopologyInput& input, cudaStream_t stream,
      Gfn2CudaPeriodicTopology& output) noexcept;

  /*
   * Device-side semantic validation.  The launcher performs only structural
   * pointer/count checks and enqueues one allocation-free kernel, so a valid
   * call is safe during CUDA Graph capture.
   */
  [[nodiscard]] static Gfn2CudaPeriodicTopologyDiagnostic validate(
      const Gfn2CudaPeriodicTopologyView& view,
      Gfn2CudaPeriodicTopologyDeviceDiagnostic* device_diagnostic,
      cudaStream_t stream = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] Gfn2CudaPeriodicTopologyView device_view() const noexcept;
  [[nodiscard]] Gfn2CudaPeriodicGraphIdentity graph_identity(
      std::uint64_t geometry_generation, std::uint32_t requested_result_flags,
      bool strain_requested, std::uint32_t unsupported_interaction_mask) const noexcept;

  [[nodiscard]] std::int32_t device_id() const noexcept { return device_id_; }
  [[nodiscard]] double image_cutoff() const noexcept { return image_cutoff_; }
  [[nodiscard]] std::uint64_t plan_token() const noexcept { return plan_token_; }
  [[nodiscard]] std::uint64_t cell_generation() const noexcept { return cell_generation_; }
  [[nodiscard]] std::uint64_t topology_fingerprint() const noexcept {
    return topology_fingerprint_;
  }

  /* Host mirrors are used for independent CPU/CUDA topology parity checks. */
  [[nodiscard]] const std::vector<std::int64_t>& host_atom_offsets() const noexcept {
    return atom_offsets_;
  }
  [[nodiscard]] const std::vector<double>& host_cell_matrices() const noexcept {
    return cell_matrices_;
  }
  [[nodiscard]] const std::vector<std::int32_t>& host_periodic_axes() const noexcept {
    return periodic_axes_;
  }
  [[nodiscard]] const std::vector<std::int64_t>& host_translation_offsets() const noexcept {
    return translation_offsets_;
  }
  [[nodiscard]] const std::vector<Gfn2CudaPeriodicTranslation>& host_translations() const noexcept {
    return translations_;
  }

 private:
  void release_device() noexcept;
  void move_from(Gfn2CudaPeriodicTopology&& other) noexcept;

  std::int32_t device_id_ = -1;
  std::int64_t batch_size_ = 0;
  std::int64_t total_atoms_ = 0;
  double image_cutoff_ = 0.0;
  std::uint64_t plan_token_ = 0u;
  std::uint64_t cell_generation_ = 0u;
  std::uint64_t topology_fingerprint_ = 0u;

  std::vector<std::int64_t> atom_offsets_;
  std::vector<double> cell_matrices_;
  std::vector<std::int32_t> periodic_axes_;
  std::vector<std::int64_t> translation_offsets_;
  std::vector<Gfn2CudaPeriodicTranslation> translations_;

  std::int64_t* device_atom_offsets_ = nullptr;
  double* device_cell_matrices_ = nullptr;
  std::int32_t* device_periodic_axes_ = nullptr;
  std::int64_t* device_translation_offsets_ = nullptr;
  Gfn2CudaPeriodicTranslation* device_translations_ = nullptr;
};

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_PERIODIC_TOPOLOGY_CUH
