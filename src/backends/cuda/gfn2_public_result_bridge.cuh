#ifndef GPUXTB_BACKENDS_CUDA_GFN2_PUBLIC_RESULT_BRIDGE_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_PUBLIC_RESULT_BRIDGE_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail::cuda {

inline constexpr std::uint32_t kGfn2PublicResultBridgeAbiVersion = 1u;

/*
 * Aggregate bridge failures are control-plane failures. A non-success value
 * suppresses every CUDA caller-buffer write and tells the synchronous runtime
 * not to commit any pinned host staging bytes or the pending result flags.
 */
enum class Gfn2PublicResultBridgeError : std::uint32_t {
  kSuccess = 0u,
  kInvalidAbiVersion = 1u,
  kPlanTokenMismatch = 2u,
  kInternalPublicationFailure = 3u,
  kInvalidEpoch = 4u,
  kInvalidFlags = 5u,
  kInvalidExtents = 6u,
  kInvalidDestinations = 7u,
};

/* Host and CUDA outputs are both staged before the caller-visible commit. */
enum class Gfn2PublicResultRoute : std::uint32_t {
  kAbsent = 0u,
  kHost = 1u,
  kCudaDevice = 2u,
};

/* Immutable shape, property request, and public flag image for one call. */
struct Gfn2PublicResultBridgeDevicePlan {
  std::uint32_t abi_version = kGfn2PublicResultBridgeAbiVersion;
  std::uint32_t requested_properties = 0u;
  std::uint32_t result_flags = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t plan_token = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_point_charges = 0;
};

/*
 * Internal publication arena. Floating-point failure slices are already quiet
 * NaNs and the three diagnostic arrays already contain their final public
 * per-peer values; this bridge deliberately performs no semantic translation.
 */
struct Gfn2PublicResultBridgeDeviceInput {
  const double* energies = nullptr;
  std::int64_t energy_elements = 0;
  const double* qm_forces = nullptr;
  std::int64_t qm_force_elements = 0;
  const double* atomic_charges = nullptr;
  std::int64_t atomic_charge_elements = 0;
  const double* point_forces = nullptr;
  std::int64_t point_force_elements = 0;

  const std::int32_t* iterations = nullptr;
  const std::uint8_t* converged = nullptr;
  const gpuxtb_status_t* system_statuses = nullptr;
  std::int64_t batch_elements = 0;

  /* Control values produced by internal inference publication. */
  const std::uint32_t* publication_plan_error = nullptr;
  const std::uint64_t* publication_epoch_snapshot = nullptr;
  const std::uint64_t* current_geometry_epoch = nullptr;
  std::uint64_t plan_token = 0u;
};

/*
 * Runtime-owned device shadow image produced by prepare. Every active public
 * field has a private slice, including fields routed to HOST. The final commit
 * kernel reads only this sealed image, so no recoverable prepare/settlement
 * failure can have touched a caller CUDA destination.
 */
struct Gfn2PublicResultBridgeDeviceStaging {
  double* energies = nullptr;
  std::int64_t energy_elements = 0;
  double* qm_forces = nullptr;
  std::int64_t qm_force_elements = 0;
  double* atomic_charges = nullptr;
  std::int64_t atomic_charge_elements = 0;
  double* point_forces = nullptr;
  std::int64_t point_force_elements = 0;

  std::int32_t* iterations = nullptr;
  std::uint8_t* converged = nullptr;
  gpuxtb_status_t* system_statuses = nullptr;
  std::int64_t batch_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * A host route intentionally has no caller pointer here. Host caller memory is
 * outside the asynchronous CUDA lifetime and is committed later by the owning
 * runtime from Gfn2PublicResultBridgeHostStaging.
 */
struct Gfn2PublicResultBridgeDestination {
  Gfn2PublicResultRoute route = Gfn2PublicResultRoute::kAbsent;
  void* device_data = nullptr;
  std::int64_t elements = 0;
};

struct Gfn2PublicResultBridgeDeviceDestinations {
  Gfn2PublicResultBridgeDestination energies{};
  Gfn2PublicResultBridgeDestination qm_forces{};
  Gfn2PublicResultBridgeDestination atomic_charges{};
  Gfn2PublicResultBridgeDestination point_forces{};
  Gfn2PublicResultBridgeDestination iterations{};
  Gfn2PublicResultBridgeDestination converged{};
  Gfn2PublicResultBridgeDestination system_statuses{};
  std::uint64_t plan_token = 0u;
};

struct Gfn2PublicResultBridgeHostBuffer {
  void* data = nullptr;
  std::int64_t elements = 0;
};

/*
 * Every nonempty host buffer and control must refer to fixed pinned storage
 * whose lifetime extends through stream completion. pending_result_flags is a
 * runtime-owned host value, not the public result.flags field itself.
 */
struct Gfn2PublicResultBridgeHostStaging {
  Gfn2PublicResultBridgeHostBuffer energies{};
  Gfn2PublicResultBridgeHostBuffer qm_forces{};
  Gfn2PublicResultBridgeHostBuffer atomic_charges{};
  Gfn2PublicResultBridgeHostBuffer point_forces{};
  Gfn2PublicResultBridgeHostBuffer iterations{};
  Gfn2PublicResultBridgeHostBuffer converged{};
  Gfn2PublicResultBridgeHostBuffer system_statuses{};
  void* control = nullptr;
  std::int64_t control_elements = 0;
  std::uint32_t* pending_result_flags = nullptr;
  std::uint64_t plan_token = 0u;
};

/* One device-produced record is downloaded after all requested host results. */
struct Gfn2PublicResultBridgeControl {
  std::uint32_t aggregate_error = static_cast<std::uint32_t>(Gfn2PublicResultBridgeError::kSuccess);
  std::uint32_t internal_publication_plan_error = 0u;
  std::uint64_t publication_epoch_snapshot = 0u;
  std::uint64_t current_geometry_epoch = 0u;
  std::uint64_t plan_token = 0u;
};

struct Gfn2PublicResultBridgeDeviceDiagnostics {
  Gfn2PublicResultBridgeControl* control = nullptr;
  std::int64_t control_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2PublicResultBridgeDevicePlan>);
static_assert(std::is_standard_layout_v<Gfn2PublicResultBridgeDevicePlan>);
static_assert(std::is_trivially_copyable_v<Gfn2PublicResultBridgeDeviceInput>);
static_assert(std::is_standard_layout_v<Gfn2PublicResultBridgeDeviceInput>);
static_assert(std::is_trivially_copyable_v<Gfn2PublicResultBridgeDeviceStaging>);
static_assert(std::is_standard_layout_v<Gfn2PublicResultBridgeDeviceStaging>);
static_assert(std::is_trivially_copyable_v<Gfn2PublicResultBridgeDestination>);
static_assert(std::is_standard_layout_v<Gfn2PublicResultBridgeDestination>);
static_assert(std::is_trivially_copyable_v<Gfn2PublicResultBridgeDeviceDestinations>);
static_assert(std::is_standard_layout_v<Gfn2PublicResultBridgeDeviceDestinations>);
static_assert(std::is_trivially_copyable_v<Gfn2PublicResultBridgeHostStaging>);
static_assert(std::is_standard_layout_v<Gfn2PublicResultBridgeHostStaging>);
static_assert(std::is_trivially_copyable_v<Gfn2PublicResultBridgeControl>);
static_assert(std::is_standard_layout_v<Gfn2PublicResultBridgeControl>);
static_assert(std::is_trivially_copyable_v<Gfn2PublicResultBridgeDeviceDiagnostics>);
static_assert(std::is_standard_layout_v<Gfn2PublicResultBridgeDeviceDiagnostics>);

/*
 * Prepare one transactional public-result image on stream.
 *
 * The preflight kernel seals the aggregate gate before copying every active
 * field into runtime-owned device staging. Requested host routes are then
 * downloaded into pinned staging, followed by the control record. This phase
 * never writes a caller CUDA destination. The owner must wait for completion,
 * accept the control record, and settle every other recoverable transaction
 * phase before calling commit_gfn2_public_results_cuda(). No allocation, host
 * polling, callback, or synchronization is performed here.
 */
cudaError_t prepare_gfn2_public_results_cuda(
    const Gfn2PublicResultBridgeDevicePlan& plan, const Gfn2PublicResultBridgeDeviceInput& input,
    const Gfn2PublicResultBridgeDeviceStaging& device_staging,
    const Gfn2PublicResultBridgeDeviceDestinations& destinations,
    const Gfn2PublicResultBridgeHostStaging& staging,
    const Gfn2PublicResultBridgeDeviceDiagnostics& diagnostics,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Final caller-visible CUDA commit. A successful return means the explicit
 * commit kernel was accepted by the stream; a later asynchronous device fault
 * cannot be rolled back and must be reported honestly by the synchronous
 * owner. HOST outputs and result.flags remain an owner-side commit after that
 * stream completion succeeds.
 */
cudaError_t commit_gfn2_public_results_cuda(
    const Gfn2PublicResultBridgeDevicePlan& plan,
    const Gfn2PublicResultBridgeDeviceStaging& device_staging,
    const Gfn2PublicResultBridgeDeviceDestinations& destinations,
    const Gfn2PublicResultBridgeDeviceDiagnostics& diagnostics,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_PUBLIC_RESULT_BRIDGE_CUH
