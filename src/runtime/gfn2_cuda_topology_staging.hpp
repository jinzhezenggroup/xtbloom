#ifndef GPUXTB_RUNTIME_GFN2_CUDA_TOPOLOGY_STAGING_HPP
#define GPUXTB_RUNTIME_GFN2_CUDA_TOPOLOGY_STAGING_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail {

/*
 * Result of one synchronous topology-staging transaction.  A match reuses the
 * committed topology directly.  A candidate must be followed by either
 * commit_candidate() after all topology-derived plans have been constructed,
 * or abort_candidate() when that external construction fails.
 */
enum class Gfn2CudaTopologyStageDisposition : std::uint32_t {
  kNone = 0u,
  kMatchesCommitted = 1u,
  kCandidate = 2u,
};

enum class Gfn2CudaTopologyStagingError : std::uint32_t {
  kSuccess = 0u,
  kNotInitialized = 1u,
  kCandidatePending = 2u,
  kInvalidDimensions = 3u,
  kInvalidDescriptor = 4u,
  kInvalidMetadata = 5u,
  kCountOverflow = 6u,
  kAllocationFailed = 7u,
  kCudaError = 8u,
  kNoCandidate = 9u,
};

enum class Gfn2CudaTopologyStagingField : std::uint32_t {
  kNone = 0u,
  kStream = 1u,
  kDimensions = 2u,
  kAtomOffsets = 3u,
  kAtomicNumbers = 4u,
  kMolecularCharges = 5u,
  kUnpairedElectrons = 6u,
  kPointChargeOffsets = 7u,
  kChargeResponseOffsets = 8u,
  kArena = 9u,
  kEvent = 10u,
  kSpinChannels = 11u,
};

/* CUDA-free diagnostic. cuda_status stores the numeric cudaError_t value. */
struct Gfn2CudaTopologyStagingDiagnostic {
  gpuxtb_status_t status = GPUXTB_STATUS_SUCCESS;
  Gfn2CudaTopologyStagingError error = Gfn2CudaTopologyStagingError::kSuccess;
  Gfn2CudaTopologyStagingField field = Gfn2CudaTopologyStagingField::kNone;
  Gfn2CudaTopologyStageDisposition disposition = Gfn2CudaTopologyStageDisposition::kNone;
  std::int64_t index = -1;
  std::int32_t cuda_status = 0;

  [[nodiscard]] bool success() const noexcept { return status == GPUXTB_STATUS_SUCCESS; }
};

/*
 * Canonical CPU representation produced from the packed device candidate.
 * It is source agnostic: absent point offsets are all zero and absent response
 * offsets are the dense per-molecule square prefixes derived from atom_offsets.
 */
struct Gfn2CudaTopologyHostSnapshot {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_point_charges = 0;
  std::int64_t total_charge_response_elements = 0;
  bool periodic_enabled = false;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> point_charge_offsets;
  std::vector<std::int64_t> charge_response_offsets;
};

/* Opaque device views.  Addresses are diagnostics/bridge inputs, not host pointers. */
struct Gfn2CudaTopologyDeviceKeyIdentity {
  std::uintptr_t packed_base = 0u;
  std::size_t packed_bytes = 0u;
  std::uintptr_t atom_offsets = 0u;
  std::uintptr_t atomic_numbers = 0u;
  std::uintptr_t molecular_charges = 0u;
  std::uintptr_t unpaired_electrons = 0u;
  std::uintptr_t spin_channels = 0u;
  std::uintptr_t point_charge_offsets = 0u;
  std::uintptr_t charge_response_offsets = 0u;
};

/* Fixed-topology observability used to prove stable storage and zero full D2H. */
struct Gfn2CudaTopologyStagingIdentity {
  std::int32_t device_id = -1;
  std::uintptr_t stream = 0u;
  std::uintptr_t completion_event = 0u;
  std::uintptr_t pinned_report = 0u;
  std::uintptr_t committed_owner = 0u;
  std::uintptr_t pending_owner = 0u;
  Gfn2CudaTopologyDeviceKeyIdentity staging{};
  Gfn2CudaTopologyDeviceKeyIdentity committed{};
  Gfn2CudaTopologyDeviceKeyIdentity pending{};
  std::uintptr_t pinned_staging = 0u;
  std::size_t pinned_staging_bytes = 0u;
  std::uint64_t committed_generation = 0u;
  std::uint64_t stage_submissions = 0u;
  std::uint64_t report_downloads = 0u;
  std::uint64_t full_metadata_downloads = 0u;
  std::uint8_t completion_event_disable_timing = 0u;
  std::uint8_t candidate_pending = 0u;
};

/*
 * Context-owned mixed-memory topology staging owner.
 *
 * stage_and_validate() performs pointer/type/device validation before touching
 * caller storage.  It synchronously snapshots every HOST leaf into pinned
 * runtime storage, enqueues all H2D/D2D transfers and one validation/comparison
 * kernel on the owner stream, and waits only on its disable-timing event.
 * Device inputs are never downloaded merely to validate them.
 *
 * The class is intentionally CUDA-free at this boundary so the same runtime
 * transaction can later be mirrored by a ROCm owner.  It is not internally
 * concurrent; the context transaction mutex must serialize calls.
 */
class Gfn2CudaTopologyStaging {
 public:
  Gfn2CudaTopologyStaging(std::int32_t device_id, void* stream) noexcept;
  ~Gfn2CudaTopologyStaging();

  Gfn2CudaTopologyStaging(const Gfn2CudaTopologyStaging&) = delete;
  Gfn2CudaTopologyStaging& operator=(const Gfn2CudaTopologyStaging&) = delete;

  [[nodiscard]] bool valid() const noexcept;

  [[nodiscard]] Gfn2CudaTopologyStagingDiagnostic stage_and_validate(const gpuxtb_batch_t& batch,
                                                                     std::string& error);

  /*
   * Complete the fallible CUDA copy/event phase without changing the
   * committed owner.  The enclosing runtime can then validate its result
   * bridge before reaching the no-fail publication point.
   */
  [[nodiscard]] Gfn2CudaTopologyStagingDiagnostic prepare_candidate_commit(std::string& error);

  /* True only while the no-fail ownership publication invariant is sealed. */
  [[nodiscard]] bool candidate_publishable() const noexcept;

  /* Publish a successfully prepared candidate using ownership moves only.
   * False reports an invariant violation instead of silently doing nothing. */
  [[nodiscard]] bool publish_candidate() noexcept;

  /* Compatibility convenience for users that do not need two-phase commit. */
  [[nodiscard]] Gfn2CudaTopologyStagingDiagnostic commit_candidate(std::string& error);

  /* Discard a staged candidate without modifying the committed canonical key. */
  void abort_candidate() noexcept;

  [[nodiscard]] const Gfn2CudaTopologyHostSnapshot* committed_snapshot() const noexcept;
  [[nodiscard]] const Gfn2CudaTopologyHostSnapshot* candidate_snapshot() const noexcept;
  [[nodiscard]] Gfn2CudaTopologyDeviceKeyIdentity committed_device_key() const noexcept;
  [[nodiscard]] Gfn2CudaTopologyDeviceKeyIdentity candidate_device_key() const noexcept;
  [[nodiscard]] Gfn2CudaTopologyStagingIdentity identity() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpuxtb::detail

#endif  // GPUXTB_RUNTIME_GFN2_CUDA_TOPOLOGY_STAGING_HPP
