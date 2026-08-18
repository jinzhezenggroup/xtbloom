#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_SCC_SETUP_TOPOLOGY_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_SCC_SETUP_TOPOLOGY_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "backends/common/gfn2_plan_schema.hpp"
#include "backends/cuda/gfn2_eigensolver.cuh"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn1/wavefunction.hpp"
#include "model/gfn2/wavefunction.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::cuda {

/* Setup failures are synchronous; device semantic validation remains in the
 * existing common-schema diagnostic returned by bind_gfn2_topology_cuda. */
enum class Gfn2SccSetupTopologyError : std::uint32_t {
  kSuccess = 0u,
  kInvalidPlan = 1u,
  kCountOverflow = 2u,
  kAllocationFailed = 3u,
  kNullArena = 4u,
  kMisalignedArena = 5u,
  kInsufficientArena = 6u,
  kInvalidArenaMemory = 7u,
  kCudaError = 8u,
};

enum class Gfn2SccSetupTopologyField : std::uint32_t {
  kNone = 0u,
  kPlanToken = 1u,
  kBasis = 2u,
  kIntegrals = 3u,
  kWavefunction = 4u,
  kOrbitalMap = 5u,
  kBuckets = 6u,
  kHostTopology = 7u,
  kArena = 8u,
};

/* Rich internal diagnostic suitable for translating into the public C status
 * plus a human-readable last-error string at the eventual API boundary. */
struct Gfn2SccSetupTopologyDiagnostic {
  xtbloom_status_t status = XTBLOOM_STATUS_SUCCESS;
  Gfn2SccSetupTopologyError error = Gfn2SccSetupTopologyError::kSuccess;
  Gfn2SccSetupTopologyField field = Gfn2SccSetupTopologyField::kNone;
  std::int64_t index = -1;
  std::size_t required_bytes = 0u;
  cudaError_t cuda_status = cudaSuccess;
  Gfn2PlanSchemaDiagnostic schema{};

  [[nodiscard]] bool success() const noexcept {
    return status == XTBLOOM_STATUS_SUCCESS && error == Gfn2SccSetupTopologyError::kSuccess;
  }
};

struct Gfn2SccSetupTopologyRequirements {
  std::size_t immutable_device_bytes = 0u;
  std::size_t device_alignment = alignof(std::int64_t);
};

/*
 * Move-only owner of the immutable host topology blueprint used by the CUDA
 * SCC setup path. The implementation owns every array referenced by
 * host_topology(), the host-only eigensolver bucket records, and one pinned
 * packed upload image, so moving the owner does not invalidate descriptor
 * addresses and upload does not require pageable-memory staging.
 *
 * Device storage is deliberately caller-owned in this slice. After a successful
 * upload, both this owner and the device arena must outlive work queued on the
 * upload stream; the returned device descriptor remains valid until the arena
 * is released or rebound. A consumer on another stream must first wait on an
 * event recorded after this upload. Moving, replacing through create(...,
 * output), or destroying an owner with an upload still in flight is forbidden,
 * because those operations may release the immutable pinned transfer image.
 */
class Gfn2SccSetupTopology {
 public:
  Gfn2SccSetupTopology() noexcept;
  ~Gfn2SccSetupTopology();
  Gfn2SccSetupTopology(Gfn2SccSetupTopology&&) noexcept;
  Gfn2SccSetupTopology& operator=(Gfn2SccSetupTopology&&) noexcept;
  Gfn2SccSetupTopology(const Gfn2SccSetupTopology&) = delete;
  Gfn2SccSetupTopology& operator=(const Gfn2SccSetupTopology&) = delete;

  /* Transactionally build and host-validate the common topology, then prepare
   * its pinned immutable upload image. output is unchanged on invalid plans,
   * overflow, CUDA setup failure, or allocation failure. */
  [[nodiscard]] static Gfn2SccSetupTopologyDiagnostic create(
      const gfn2::BasisPlan& basis, const gfn2::IntegralPlan& integrals,
      const gfn2::WavefunctionLayout& wavefunction, std::uint64_t plan_token,
      Gfn2SccSetupTopology& output) noexcept;

  /* GFN1 owns a scalar SCC wavefunction. This overload projects its physical
   * charge/magnetization packing into the common CUDA topology schema while
   * synthesizing only the unused multipole field extents required by that
   * schema; no GFN2 numerical parameters are introduced. */
  [[nodiscard]] static Gfn2SccSetupTopologyDiagnostic create(
      const gfn1::BasisPlan& basis, const gfn1::IntegralPlan& integrals,
      const gfn1::WavefunctionLayout& wavefunction, std::uint64_t plan_token,
      Gfn2SccSetupTopology& output) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const Gfn2RaggedTopologyView& host_topology() const noexcept;
  [[nodiscard]] const Gfn2WavefunctionLayoutView& host_wavefunction_layout() const noexcept;
  [[nodiscard]] const std::vector<Gfn2EigensolverBucket>& eigensolver_buckets() const noexcept;
  [[nodiscard]] Gfn2SccSetupTopologyRequirements requirements() const noexcept;
  /* Host bytes retained by copied topology vectors and the pinned upload image. */
  [[nodiscard]] std::size_t retained_host_bytes() const noexcept;

  /* Preflight the complete caller arena, enqueue every immutable upload on
   * stream, and publish the device descriptor only after all copies were
   * accepted by CUDA. No synchronization or allocation is performed. The
   * descriptor is ready for immediate use only on stream; other streams need
   * explicit event ordering as described by the class lifetime contract. */
  [[nodiscard]] Gfn2SccSetupTopologyDiagnostic bind_device_arena_and_upload_async(
      void* device_arena, std::size_t device_arena_bytes, Gfn2RaggedTopologyView& device_topology,
      cudaStream_t stream = nullptr) const noexcept;

  /*
   * Upload the same immutable arena while also publishing the exact
   * WavefunctionLayout spin projection.  The two output descriptors are one
   * transaction: either both remain unchanged or both name the uploaded plan.
   */
  [[nodiscard]] Gfn2SccSetupTopologyDiagnostic bind_device_arena_and_upload_async(
      void* device_arena, std::size_t device_arena_bytes, Gfn2RaggedTopologyView& device_topology,
      Gfn2WavefunctionLayoutView& device_wavefunction,
      cudaStream_t stream = nullptr) const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_SCC_SETUP_TOPOLOGY_HPP
