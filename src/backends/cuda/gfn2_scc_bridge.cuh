#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_BRIDGE_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_BRIDGE_CUH

#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "backends/common/gfn2_plan_schema.hpp"
#include "backends/cuda/gfn2_scc_iteration_control.cuh"

namespace gpuxtb::detail::cuda {

/* Errors produced by the bridge itself. Plan errors are published only through
 * downstream_plan_error; peer-local errors are stored in system_errors. */
enum class Gfn2SccBridgeDeviceError : std::uint32_t {
  kSuccess = 0u,
  kInvalidTopologyOffsets = 1u,
  kInvalidFieldOffsets = 2u,
  kInvalidShellToAtom = 3u,
  kInvalidActiveMask = 4u,
  kNonfiniteShellPotential = 5u,
  kNonfiniteAtomicPotential = 6u,
  kNonfiniteScalarPotentialArithmetic = 7u,
  kUpstreamPlanFailure = 8u,
};

/*
 * Adapter boundary between the common immutable plan schema and SCC field
 * layouts. topology is a complete, setup-time validated CUDA-device view; the
 * hot bridge deliberately rechecks only the partitions that it dereferences.
 * qsh/qat offsets locate charge-channel fields and may have a nonzero common
 * base, so they need not equal the topology-major shell/atom offsets.
 */
struct Gfn2SccBridgeDeviceBatch {
  Gfn2RaggedTopologyView topology{};
  std::int64_t qsh_offset_count = 0;
  std::int64_t qat_offset_count = 0;
  const std::int64_t* qsh_offsets = nullptr;
  const std::int64_t* qat_offsets = nullptr;
};

/* Field-layout scalar components produced by the SCC potential composer. */
struct Gfn2SccBridgeDevicePotentialFields {
  const double* shell = nullptr;
  std::int64_t shell_elements = 0;
  const double* atomic = nullptr;
  std::int64_t atom_elements = 0;
  std::uint64_t plan_token = 0u;
};

/*
 * Status entering an SCC edge. requested_active may be null+zero to request
 * every member. system_errors is the shared sticky per-system array and may
 * already contain failures from an upstream stage.
 *
 * upstream_device_error is the upstream stage's first-error diagnostic.
 * upstream_sequence_active must point to the upstream workspace's device-side
 * plan/sequence latch after that stage completed. A value other than one is an
 * unconditional fail-closed signal and takes precedence over peer_error_mask;
 * this preserves a later plan failure hidden behind an earlier peer diagnostic.
 * peer_error_mask classifies error codes 1..63 that are known to be numerical
 * or otherwise per-system; bit N corresponds to code N. Success and classified
 * peer errors keep the sequence open. Every other nonzero code, including an
 * unknown code >= 64, is conservatively treated as a plan error.
 */
struct Gfn2SccBridgeDeviceStageInput {
  const std::uint8_t* requested_active = nullptr;
  std::int64_t active_elements = 0;
  std::uint32_t* system_errors = nullptr;
  std::int64_t system_error_elements = 0;
  const std::uint32_t* upstream_device_error = nullptr;
  std::int64_t upstream_device_error_elements = 0;
  const std::uint32_t* upstream_sequence_active = nullptr;
  std::int64_t upstream_sequence_elements = 0;
  std::uint64_t peer_error_mask = 0u;
  std::uint64_t plan_token = 0u;
};

/*
 * topology-major shell scalar potential and normalized downstream gate.
 * downstream_plan_error intentionally contains only sequence-wide failures;
 * healthy peers can therefore enter primitives whose legacy device_error
 * snapshot treats every nonzero value as a whole-sequence failure.
 */
struct Gfn2SccBridgeDeviceOutput {
  double* shell_scalar = nullptr;
  std::int64_t shell_elements = 0;
  std::uint8_t* downstream_active = nullptr;
  std::int64_t active_elements = 0;
  std::uint32_t* downstream_plan_error = nullptr;
  std::int64_t plan_error_elements = 0;
  std::uint64_t plan_token = 0u;
};

/* Caller-owned unpublished shell values and one sequence latch. */
struct Gfn2SccBridgeDeviceWorkspace {
  double* shell_scratch = nullptr;
  std::int64_t shell_elements = 0;
  std::uint32_t* sequence_active = nullptr;
  std::int64_t sequence_elements = 0;
  std::uint64_t plan_token = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccBridgeDeviceBatch>);
static_assert(std::is_standard_layout_v<Gfn2SccBridgeDeviceBatch>);
static_assert(std::is_trivially_copyable_v<Gfn2SccBridgeDevicePotentialFields>);
static_assert(std::is_standard_layout_v<Gfn2SccBridgeDevicePotentialFields>);
static_assert(std::is_trivially_copyable_v<Gfn2SccBridgeDeviceStageInput>);
static_assert(std::is_standard_layout_v<Gfn2SccBridgeDeviceStageInput>);
static_assert(std::is_trivially_copyable_v<Gfn2SccBridgeDeviceOutput>);
static_assert(std::is_standard_layout_v<Gfn2SccBridgeDeviceOutput>);
static_assert(std::is_trivially_copyable_v<Gfn2SccBridgeDeviceWorkspace>);
static_assert(std::is_standard_layout_v<Gfn2SccBridgeDeviceWorkspace>);

/* Explicitly reopen a bridge edge. The reset is asynchronous and Graph-safe;
 * shared upstream/per-system diagnostics remain the caller's responsibility. */
cudaError_t reset_gfn2_scc_bridge_stage_cuda(std::int64_t batch_size,
                                             std::uint8_t* downstream_active,
                                             std::uint32_t* downstream_plan_error,
                                             std::uint32_t* sequence_active,
                                             cudaStream_t stream = nullptr) noexcept;

/* Reset diagnostics for the canonical bridge without allocating a redundant
 * downstream activity array.  sequence_active remains a stage-local report
 * latch and is opened by the subsequent canonical collect launch. */
cudaError_t reset_gfn2_scc_bridge_device_errors_cuda(std::int64_t batch_size,
                                                     std::uint32_t* system_errors,
                                                     std::uint32_t* device_error,
                                                     std::uint32_t* sequence_active,
                                                     cudaStream_t stream = nullptr) noexcept;

/*
 * Collect complete scalar shell potentials in the exact CPU expansion order:
 *
 *   complete_vsh = field_vsh + field_vat(shell_to_atom)
 *
 * The launch allocates, transfers, and synchronizes nothing. Publication is
 * transactional per system, inactive/failed members are not inspected, all
 * descriptors are copied into CUDA Graph nodes by value, and only the supplied
 * stream is used. Pointer memory-space provenance is a setup-time binding
 * responsibility; this hot API accepts CUDA-device/UVA addresses only.
 */
cudaError_t collect_gfn2_scc_shell_scalar_potential_cuda(
    const Gfn2SccBridgeDeviceBatch& batch, const Gfn2SccBridgeDevicePotentialFields& potential,
    const Gfn2SccBridgeDeviceStageInput& stage, const Gfn2SccBridgeDeviceOutput& output,
    const Gfn2SccBridgeDeviceWorkspace& workspace, cudaStream_t stream = nullptr) noexcept;

/*
 * Canonical SCC edge: complete_vsh = field_vsh + field_vat(shell_to_atom).
 * The canonical ledger is the only sequence/member authority; no requested or
 * downstream activity copy is produced.  The stage-local sequence latch in
 * workspace is retained solely for normalized diagnostics and Graph replay.
 */
cudaError_t collect_gfn2_scc_shell_scalar_potential_cuda(
    const Gfn2SccBridgeDeviceBatch& batch, const Gfn2SccBridgeDevicePotentialFields& potential,
    const Gfn2SccIterationDeviceActivity& activity, double* shell_scalar,
    std::int64_t shell_elements, const Gfn2SccBridgeDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_BRIDGE_CUH
