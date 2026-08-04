#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "backends/cuda/gfn2_scc_iteration_arena.cuh"

namespace {

using namespace gpuxtb::detail;
using namespace gpuxtb::detail::cuda;

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

struct DeviceAllocation {
  void* pointer = nullptr;

  explicit DeviceAllocation(std::size_t bytes) {
    if (cudaMalloc(&pointer, bytes) != cudaSuccess) pointer = nullptr;
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  ~DeviceAllocation() {
    if (pointer != nullptr) (void)cudaFree(pointer);
  }
};

constexpr std::uint64_t kPlanToken = 0xa101c0deULL;

Gfn2SccIterationDevicePlan make_plan() {
  Gfn2SccIterationDevicePlan plan{};
  plan.abi_version = kGfn2SccIterationAbiVersion;
  plan.enabled_components = kGfn2SccPotentialAllComponents;
  plan.plan_token = kPlanToken;
  plan.topology.plan_token = kPlanToken;
  plan.topology.batch_size = 3;
  plan.topology.bucket_count = 2;
  plan.topology.total_atoms = 7;
  plan.topology.total_shells = 11;
  plan.topology.total_orbitals = 15;
  plan.topology.total_matrix_elements = 83;
  plan.mixer_policy.history_size = 6;
  plan.geometry_batch.total_pairs = 8;
  plan.es2_batch.total_matrix_elements = 51;
  plan.aes2_batch.total_pairs = 8;
  plan.d4_batch.total_pairs = 8;
  plan.eigensolver_provider.requirements.solver_device_workspace_bytes = 130u;
  plan.eigensolver_provider.requirements.solver_host_workspace_bytes = 96u;
  return plan;
}

template <typename T>
std::size_t offset_from(const void* base, const T* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) - reinterpret_cast<std::uintptr_t>(base);
}

bool pointer_in_arena(const void* base, std::size_t bytes, const void* pointer) {
  if (pointer == nullptr) return false;
  const auto begin = reinterpret_cast<std::uintptr_t>(base);
  const auto value = reinterpret_cast<std::uintptr_t>(pointer);
  return value >= begin && value < begin + bytes;
}

int test_query_and_complete_projection_without_arena_writes() {
  auto plan = make_plan();
  Gfn2SccIterationArenaRequirements requirements{};
  const auto query = query_gfn2_scc_iteration_arena_requirements_cuda(
      plan, plan.eigensolver_provider.requirements, requirements);
  CHECK(query.success());
  CHECK(requirements.abi_version == kGfn2SccIterationArenaAbiVersion);
  CHECK(requirements.alignment == kGfn2SccIterationArenaAlignment);
  CHECK(requirements.plan_token == kPlanToken);
  CHECK(requirements.layout_fingerprint != 0u);
  CHECK(requirements.persistent_offset == 0u);
  CHECK(requirements.persistent_bytes != 0u);
  CHECK(requirements.workspace_offset % kGfn2SccIterationArenaAlignment == 0u);
  CHECK(requirements.workspace_bytes != 0u);
  CHECK(requirements.provider_device_offset % kGfn2SccIterationArenaAlignment == 0u);
  CHECK(requirements.provider_device_bytes == 130u);
  CHECK(requirements.total_bytes % kGfn2SccIterationArenaAlignment == 0u);
  CHECK(requirements.total_bytes >=
        requirements.provider_device_offset + requirements.provider_device_bytes);

  DeviceAllocation arena(requirements.total_bytes);
  CHECK(arena.pointer != nullptr);
  CHECK(reinterpret_cast<std::uintptr_t>(arena.pointer) % requirements.alignment == 0u);
  CHECK(cudaMemset(arena.pointer, 0xa5, requirements.total_bytes) == cudaSuccess);

  alignas(std::max_align_t) std::array<std::byte, 96> host_workspace{};
  Gfn2SccIterationDeviceState state{};
  Gfn2SccIterationDeviceWorkspace workspace{};
  Gfn2SccIterationReportStorage reports{};
  const auto provider_requirements = plan.eigensolver_provider.requirements;
  plan.eigensolver_provider.requirements = {};
  const auto diagnostic = bind_gfn2_scc_iteration_arena_cuda(
      plan, provider_requirements, requirements, arena.pointer, requirements.total_bytes,
      host_workspace.data(), host_workspace.size(), state, workspace, reports);
  if (!diagnostic.success()) {
    std::fprintf(stderr, "arena bind failed: error=%u required=%zu provided=%zu\n",
                 static_cast<unsigned>(diagnostic.error), diagnostic.required_bytes,
                 diagnostic.provided_bytes);
  }
  CHECK(diagnostic.success());

  CHECK(plan.eigensolver_provider.device_workspace ==
        static_cast<std::byte*>(arena.pointer) + requirements.provider_device_offset);
  CHECK(plan.eigensolver_provider.device_workspace_bytes == requirements.provider_device_bytes);
  CHECK(plan.eigensolver_provider.host_workspace == host_workspace.data());
  CHECK(plan.eigensolver_provider.host_workspace_bytes == host_workspace.size());
  CHECK(state.plan_token == kPlanToken);
  CHECK(workspace.plan_token == kPlanToken);
  CHECK(reports.plan_token == kPlanToken);
  CHECK(reports.system_error_elements == 24 * plan.topology.batch_size);
  CHECK(reports.device_error_elements == 24);
  CHECK(reports.sequence_latch_elements == 24);

  CHECK(pointer_in_arena(arena.pointer, requirements.total_bytes, state.eigenpairs.eigenvalues));
  CHECK(pointer_in_arena(arena.pointer, requirements.total_bytes,
                         workspace.publication_workspace.next_statuses));
  CHECK(pointer_in_arena(arena.pointer, requirements.total_bytes, reports.system_errors));
  CHECK(state.published.shell_charges == state.raw_population.qsh);
  CHECK(state.published.atomic_dipoles == state.raw_population.dipole);
  CHECK(state.published.atomic_quadrupoles == state.raw_population.quadrupole);
  CHECK(workspace.staged_free_energy.entropy == workspace.staged_occupations.entropies);
  CHECK(workspace.publication_workspace.mixed_atomic_charges ==
        workspace.mixed_topology.atomic_charges);
  CHECK(workspace.eigensolver_workspace.solver_device_workspace ==
        plan.eigensolver_provider.device_workspace);
  CHECK(workspace.eigensolver_workspace.solver_host_workspace == host_workspace.data());
  CHECK(workspace.eigensolver_workspace.compact_system_elements == plan.topology.batch_size);
  CHECK(workspace.eigensolver_workspace.compact_source_slot_elements == plan.topology.batch_size);
  CHECK(workspace.eigensolver_workspace.bucket_activity_elements == plan.topology.bucket_count);
  CHECK(pointer_in_arena(arena.pointer, requirements.total_bytes,
                         workspace.eigensolver_workspace.compact_systems));
  CHECK(pointer_in_arena(arena.pointer, requirements.total_bytes,
                         workspace.eigensolver_workspace.compact_source_slots));
  CHECK(pointer_in_arena(arena.pointer, requirements.total_bytes,
                         workspace.eigensolver_workspace.bucket_activity));

  /* The setup pass projects descriptors only; caller initialization owns bytes. */
  std::vector<std::byte> snapshot(requirements.total_bytes);
  CHECK(cudaMemcpy(snapshot.data(), arena.pointer, snapshot.size(), cudaMemcpyDeviceToHost) ==
        cudaSuccess);
  for (const std::byte value : snapshot) CHECK(value == std::byte{0xa5});
  return 0;
}

int test_deterministic_rebind_offsets() {
  auto plan = make_plan();
  Gfn2SccIterationArenaRequirements requirements{};
  CHECK(query_gfn2_scc_iteration_arena_requirements_cuda(
            plan, plan.eigensolver_provider.requirements, requirements)
            .success());
  DeviceAllocation first_arena(requirements.total_bytes);
  DeviceAllocation second_arena(requirements.total_bytes);
  CHECK(first_arena.pointer != nullptr);
  CHECK(second_arena.pointer != nullptr);
  alignas(std::max_align_t) std::array<std::byte, 96> host_workspace{};

  Gfn2SccIterationDeviceState first_state{};
  Gfn2SccIterationDeviceWorkspace first_workspace{};
  Gfn2SccIterationReportStorage first_reports{};
  const auto provider_requirements = plan.eigensolver_provider.requirements;
  CHECK(bind_gfn2_scc_iteration_arena_cuda(plan, provider_requirements, requirements,
                                           first_arena.pointer, requirements.total_bytes,
                                           host_workspace.data(), host_workspace.size(),
                                           first_state, first_workspace, first_reports)
            .success());

  const std::array<std::size_t, 6> first_offsets{{
      offset_from(first_arena.pointer, first_state.eigenpairs.eigenvalues),
      offset_from(first_arena.pointer, first_state.scc.system_statuses),
      offset_from(first_arena.pointer, first_workspace.ledger.active_mask),
      offset_from(first_arena.pointer, first_workspace.mixer_workspace.beta),
      offset_from(first_arena.pointer, first_reports.system_errors),
      offset_from(first_arena.pointer,
                  static_cast<std::byte*>(plan.eigensolver_provider.device_workspace)),
  }};

  Gfn2SccIterationDeviceState second_state{};
  Gfn2SccIterationDeviceWorkspace second_workspace{};
  Gfn2SccIterationReportStorage second_reports{};
  CHECK(bind_gfn2_scc_iteration_arena_cuda(plan, provider_requirements, requirements,
                                           second_arena.pointer, requirements.total_bytes,
                                           host_workspace.data(), host_workspace.size(),
                                           second_state, second_workspace, second_reports)
            .success());
  const std::array<std::size_t, 6> second_offsets{{
      offset_from(second_arena.pointer, second_state.eigenpairs.eigenvalues),
      offset_from(second_arena.pointer, second_state.scc.system_statuses),
      offset_from(second_arena.pointer, second_workspace.ledger.active_mask),
      offset_from(second_arena.pointer, second_workspace.mixer_workspace.beta),
      offset_from(second_arena.pointer, second_reports.system_errors),
      offset_from(second_arena.pointer,
                  static_cast<std::byte*>(plan.eigensolver_provider.device_workspace)),
  }};
  CHECK(first_offsets == second_offsets);
  return 0;
}

int test_fail_closed_arena_and_host_workspace_checks() {
  auto plan = make_plan();
  const auto original_plan = plan;
  Gfn2SccIterationArenaRequirements requirements{};
  CHECK(query_gfn2_scc_iteration_arena_requirements_cuda(
            plan, plan.eigensolver_provider.requirements, requirements)
            .success());
  DeviceAllocation allocation(requirements.total_bytes + requirements.alignment);
  CHECK(allocation.pointer != nullptr);
  alignas(std::max_align_t) std::array<std::byte, 96> host_workspace{};

  const auto expect_failure = [&](Gfn2SccIterationArenaError expected, void* arena,
                                  std::size_t arena_bytes, void* host, std::size_t host_bytes,
                                  const Gfn2SccIterationArenaRequirements& supplied) {
    Gfn2SccIterationDeviceState state{};
    Gfn2SccIterationDeviceWorkspace workspace{};
    Gfn2SccIterationReportStorage reports{};
    state.plan_token = 1u;
    workspace.plan_token = 1u;
    reports.plan_token = 1u;
    const auto diagnostic = bind_gfn2_scc_iteration_arena_cuda(
        plan, plan.eigensolver_provider.requirements, supplied, arena, arena_bytes, host,
        host_bytes, state, workspace, reports);
    return diagnostic.error == expected && state.plan_token == 0u && workspace.plan_token == 0u &&
           reports.plan_token == 0u && std::memcmp(&plan, &original_plan, sizeof(plan)) == 0;
  };

  CHECK(expect_failure(Gfn2SccIterationArenaError::kNullArena, nullptr, requirements.total_bytes,
                       host_workspace.data(), host_workspace.size(), requirements));
  CHECK(expect_failure(Gfn2SccIterationArenaError::kMisalignedArena,
                       static_cast<std::byte*>(allocation.pointer) + 1, requirements.total_bytes,
                       host_workspace.data(), host_workspace.size(), requirements));
  CHECK(expect_failure(Gfn2SccIterationArenaError::kInsufficientArena, allocation.pointer,
                       requirements.total_bytes - 1u, host_workspace.data(), host_workspace.size(),
                       requirements));
  const auto maximum_aligned_address =
      std::numeric_limits<std::uintptr_t>::max() &
      ~(static_cast<std::uintptr_t>(kGfn2SccIterationArenaAlignment) - 1u);
  CHECK(expect_failure(Gfn2SccIterationArenaError::kSizeOverflow,
                       reinterpret_cast<void*>(maximum_aligned_address), requirements.total_bytes,
                       host_workspace.data(), host_workspace.size(), requirements));
  CHECK(expect_failure(Gfn2SccIterationArenaError::kInvalidProviderHostWorkspace,
                       allocation.pointer, requirements.total_bytes, nullptr, host_workspace.size(),
                       requirements));
  CHECK(expect_failure(Gfn2SccIterationArenaError::kInvalidProviderHostWorkspace,
                       allocation.pointer, requirements.total_bytes, host_workspace.data(),
                       host_workspace.size() - 1u, requirements));
  CHECK(expect_failure(Gfn2SccIterationArenaError::kInvalidProviderHostWorkspace,
                       allocation.pointer, requirements.total_bytes, host_workspace.data() + 1,
                       host_workspace.size(), requirements));

  auto stale = requirements;
  stale.layout_fingerprint ^= 1u;
  CHECK(expect_failure(Gfn2SccIterationArenaError::kStaleRequirements, allocation.pointer,
                       requirements.total_bytes, host_workspace.data(), host_workspace.size(),
                       stale));
  stale = requirements;
  stale.total_bytes += requirements.alignment;
  CHECK(expect_failure(Gfn2SccIterationArenaError::kStaleRequirements, allocation.pointer,
                       requirements.total_bytes, host_workspace.data(), host_workspace.size(),
                       stale));
  stale = requirements;
  stale.plan_token ^= 1u;
  CHECK(expect_failure(Gfn2SccIterationArenaError::kStaleRequirements, allocation.pointer,
                       requirements.total_bytes, host_workspace.data(), host_workspace.size(),
                       stale));
  stale = requirements;
  stale.alignment /= 2u;
  CHECK(expect_failure(Gfn2SccIterationArenaError::kStaleRequirements, allocation.pointer,
                       requirements.total_bytes, host_workspace.data(), host_workspace.size(),
                       stale));
  stale = requirements;
  stale.abi_version += 1u;
  CHECK(expect_failure(Gfn2SccIterationArenaError::kStaleRequirements, allocation.pointer,
                       requirements.total_bytes, host_workspace.data(), host_workspace.size(),
                       stale));
  return 0;
}

int test_cross_plan_and_query_overflow_fail_closed() {
  auto first = make_plan();
  Gfn2SccIterationArenaRequirements requirements{};
  CHECK(query_gfn2_scc_iteration_arena_requirements_cuda(
            first, first.eigensolver_provider.requirements, requirements)
            .success());

  auto second = first;
  second.plan_token += 1u;
  second.topology.plan_token = second.plan_token;
  Gfn2SccIterationDeviceState state{};
  Gfn2SccIterationDeviceWorkspace workspace{};
  Gfn2SccIterationReportStorage reports{};
  const auto cross_plan = bind_gfn2_scc_iteration_arena_cuda(
      second, second.eigensolver_provider.requirements, requirements,
      reinterpret_cast<void*>(0x100000000ULL), requirements.total_bytes,
      reinterpret_cast<void*>(0x200000000ULL), 96u, state, workspace, reports);
  CHECK(cross_plan.error == Gfn2SccIterationArenaError::kStaleRequirements);
  CHECK(state.plan_token == 0u);
  CHECK(workspace.plan_token == 0u);
  CHECK(reports.plan_token == 0u);

  auto stale_provider = first.eigensolver_provider.requirements;
  stale_provider.solver_device_workspace_bytes += kGfn2SccIterationArenaAlignment;
  state.plan_token = 1u;
  workspace.plan_token = 1u;
  reports.plan_token = 1u;
  const auto provider_mismatch = bind_gfn2_scc_iteration_arena_cuda(
      first, stale_provider, requirements, reinterpret_cast<void*>(0x100000000ULL),
      requirements.total_bytes, reinterpret_cast<void*>(0x200000000ULL), 96u, state, workspace,
      reports);
  CHECK(provider_mismatch.error == Gfn2SccIterationArenaError::kStaleRequirements);
  CHECK(state.plan_token == 0u);
  CHECK(workspace.plan_token == 0u);
  CHECK(reports.plan_token == 0u);

  auto overflow = make_plan();
  overflow.topology.total_atoms = std::numeric_limits<std::int64_t>::max();
  requirements.layout_fingerprint = 9u;
  auto diagnostic = query_gfn2_scc_iteration_arena_requirements_cuda(
      overflow, overflow.eigensolver_provider.requirements, requirements);
  CHECK(diagnostic.error == Gfn2SccIterationArenaError::kSizeOverflow);
  CHECK(requirements.layout_fingerprint == 0u);

  auto invalid_provider = make_plan();
  invalid_provider.eigensolver_provider.requirements.solver_device_workspace_bytes =
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) + 1u;
  diagnostic = query_gfn2_scc_iteration_arena_requirements_cuda(
      invalid_provider, invalid_provider.eigensolver_provider.requirements, requirements);
  CHECK(diagnostic.error == Gfn2SccIterationArenaError::kInvalidProviderRequirements);
  CHECK(requirements.total_bytes == 0u);
  return 0;
}

int test_zero_sized_provider_workspace_is_canonical_null() {
  auto plan = make_plan();
  plan.eigensolver_provider.requirements = {};
  Gfn2SccIterationArenaRequirements requirements{};
  CHECK(query_gfn2_scc_iteration_arena_requirements_cuda(
            plan, plan.eigensolver_provider.requirements, requirements)
            .success());
  CHECK(requirements.provider_device_bytes == 0u);
  CHECK(requirements.total_bytes % requirements.alignment == 0u);

  DeviceAllocation arena(requirements.total_bytes);
  CHECK(arena.pointer != nullptr);
  alignas(std::max_align_t) std::array<std::byte, 16> unused_host_storage{};
  Gfn2SccIterationDeviceState state{};
  Gfn2SccIterationDeviceWorkspace workspace{};
  Gfn2SccIterationReportStorage reports{};
  CHECK(bind_gfn2_scc_iteration_arena_cuda(plan, plan.eigensolver_provider.requirements,
                                           requirements, arena.pointer, requirements.total_bytes,
                                           unused_host_storage.data(), unused_host_storage.size(),
                                           state, workspace, reports)
            .success());
  CHECK(plan.eigensolver_provider.device_workspace == nullptr);
  CHECK(plan.eigensolver_provider.device_workspace_bytes == 0u);
  CHECK(plan.eigensolver_provider.host_workspace == nullptr);
  CHECK(plan.eigensolver_provider.host_workspace_bytes == 0u);
  CHECK(workspace.eigensolver_workspace.solver_device_workspace == nullptr);
  CHECK(workspace.eigensolver_workspace.solver_host_workspace == nullptr);
  return 0;
}

}  // namespace

int main() {
  const std::array<int (*)(), 5> tests{{
      test_query_and_complete_projection_without_arena_writes,
      test_deterministic_rebind_offsets,
      test_fail_closed_arena_and_host_workspace_checks,
      test_cross_plan_and_query_overflow_fail_closed,
      test_zero_sized_provider_workspace_is_canonical_null,
  }};
  for (const auto test : tests) {
    if (const int line = test(); line != 0) {
      std::fprintf(stderr, "CUDA SCC iteration arena test failed at line %d\n", line);
      return 1;
    }
  }
  return 0;
}
