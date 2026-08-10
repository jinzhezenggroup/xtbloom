#ifndef XTBLOOM_BACKENDS_CUDA_GFN2_SCC_ITERATION_ARENA_CUH
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_CUDA_GFN2_SCC_ITERATION_ARENA_CUH

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration_reports.cuh"

namespace xtbloom::detail::cuda {

inline constexpr std::uint32_t kGfn2SccIterationArenaAbiVersion = 4u;
inline constexpr std::size_t kGfn2SccIterationArenaAlignment = 256u;

/* Synchronous setup failure; no allocation, transfer, or device work occurs. */
enum class Gfn2SccIterationArenaError : std::uint32_t {
  kSuccess = 0u,
  kInvalidPlan = 1u,
  kInvalidProviderRequirements = 2u,
  kSizeOverflow = 3u,
  kStaleRequirements = 4u,
  kNullArena = 5u,
  kMisalignedArena = 6u,
  kInsufficientArena = 7u,
  kInvalidProviderHostWorkspace = 8u,
};

struct Gfn2SccIterationArenaDiagnostic {
  Gfn2SccIterationArenaError error = Gfn2SccIterationArenaError::kSuccess;
  std::size_t required_bytes = 0u;
  std::size_t provided_bytes = 0u;

  [[nodiscard]] bool success() const noexcept {
    return error == Gfn2SccIterationArenaError::kSuccess;
  }
};

/*
 * Exact deterministic layout sealed from the plan shape and provider query.
 * persistent contains the public wavefunction, energy trace, mixer, and SCC
 * state. workspace contains every unpublished transaction, primitive scratch,
 * ledger, publication diagnostic, and stage-report backing range. The solver
 * device workspace is the final separately aligned slice of the same arena.
 *
 * plan_token rejects cross-plan reuse directly. layout_fingerprint prevents
 * binding requirements computed for another topology, component mask, mixer
 * depth, or provider workspace size. It is a layout identity only and
 * deliberately excludes pointer values and geometry generations.
 */
struct Gfn2SccIterationArenaRequirements {
  std::uint32_t abi_version = kGfn2SccIterationArenaAbiVersion;
  std::uint32_t reserved = 0u;
  std::size_t alignment = kGfn2SccIterationArenaAlignment;
  std::size_t persistent_offset = 0u;
  std::size_t persistent_bytes = 0u;
  std::size_t workspace_offset = 0u;
  std::size_t workspace_bytes = 0u;
  std::size_t provider_device_offset = 0u;
  std::size_t provider_device_bytes = 0u;
  std::size_t total_bytes = 0u;
  std::uint64_t plan_token = 0u;
  std::uint64_t layout_fingerprint = 0u;
};

static_assert(std::is_trivially_copyable_v<Gfn2SccIterationArenaDiagnostic>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationArenaDiagnostic>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationArenaRequirements>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationArenaRequirements>);

/*
 * Compute the exact single-arena byte requirement with checked arithmetic.
 * plan must already contain the immutable topology, enabled-component mask,
 * primitive extents, policies, provider handles/buckets, and plan token. The
 * provider requirement sizes may be zero, but must fit in int64_t because the
 * iteration validator records every active address range with signed counts.
 */
[[nodiscard]] Gfn2SccIterationArenaDiagnostic query_gfn2_scc_iteration_arena_requirements_cuda(
    const Gfn2SccIterationDevicePlan& plan,
    const Gfn2EigensolverWorkspaceRequirements& provider_requirements,
    Gfn2SccIterationArenaRequirements& requirements) noexcept;

/*
 * Project one caller-owned aligned arena into the complete mutable iteration
 * state, workspace, and raw stage-diagnostic capacity. The separate report
 * factory owns canonical report ordering and semantics; immutable H0,
 * integral, topology, cache, and downstream input projections are likewise
 * outside this binder's responsibility.
 *
 * The binder never reads or writes arena contents. Public state must be
 * initialized separately before launch. All writable primitive ranges are
 * disjoint. Exact aliases exist only as descriptor projections: published
 * qsh/d/Q reuse raw population, staged free-energy entropy reuses staged
 * occupation entropy, component views reuse their producer storage, and the
 * publication views reuse the public/staged descriptors. The physical charge
 * channel projection is separate from the spin-ragged SCC multipoles so
 * legacy component kernels never index across a magnetization slice.
 *
 * provider_host_workspace remains caller-owned host storage outside the
 * device arena. arena, that host workspace, plan storage, and provider handles
 * must outlive every queued use of the resulting binding.
 * provider_requirements must be the same sealed result supplied to the query
 * call. On success the binder records that single authority in plan's provider
 * leaf together with the projected device and host workspace addresses.
 */
[[nodiscard]] Gfn2SccIterationArenaDiagnostic bind_gfn2_scc_iteration_arena_cuda(
    Gfn2SccIterationDevicePlan& plan,
    const Gfn2EigensolverWorkspaceRequirements& provider_requirements,
    const Gfn2SccIterationArenaRequirements& requirements, void* arena, std::size_t arena_bytes,
    void* provider_host_workspace, std::size_t provider_host_workspace_bytes,
    Gfn2SccIterationDeviceState& state, Gfn2SccIterationDeviceWorkspace& workspace,
    Gfn2SccIterationReportStorage& report_storage) noexcept;

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_GFN2_SCC_ITERATION_ARENA_CUH
