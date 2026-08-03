#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc_iteration_control.cuh"

namespace gpuxtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr unsigned long long kNoIndexedCode = std::numeric_limits<unsigned long long>::max();

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool checked_bytes(std::int64_t elements, std::size_t element_size, std::size_t* bytes) noexcept {
  if (elements < 0 ||
      (elements != 0 && static_cast<std::uint64_t>(elements) >
                            std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  *bytes = static_cast<std::size_t>(elements) * element_size;
  return true;
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

bool make_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                AddressRange* range) noexcept {
  std::size_t bytes = 0u;
  if (!checked_bytes(elements, element_size, &bytes) || (bytes != 0u && pointer == nullptr)) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return false;
  }
  *range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t N>
bool pairwise_disjoint(const AddressRange (&ranges)[N]) noexcept {
  for (std::size_t first = 0u; first < N; ++first) {
    for (std::size_t second = first + 1u; second < N; ++second) {
      if (ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

bool valid_ledger(const Gfn2SccIterationDeviceLedger& ledger, std::int64_t batch_size,
                  std::uint64_t plan_token, AddressRange (&writes)[5]) noexcept {
  return ledger.batch_elements == batch_size && ledger.scalar_elements == 1 &&
         ledger.plan_token == plan_token && is_aligned(ledger.active_mask, alignof(std::uint8_t)) &&
         is_aligned(ledger.pending_statuses, alignof(gpuxtb_status_t)) &&
         is_aligned(ledger.system_failure_records, alignof(std::uint64_t)) &&
         is_aligned(ledger.plan_failure_record, alignof(std::uint64_t)) &&
         is_aligned(ledger.sequence_active, alignof(std::uint32_t)) &&
         make_range(ledger.active_mask, batch_size, sizeof(std::uint8_t), &writes[0]) &&
         make_range(ledger.pending_statuses, batch_size, sizeof(gpuxtb_status_t), &writes[1]) &&
         make_range(ledger.system_failure_records, batch_size, sizeof(std::uint64_t), &writes[2]) &&
         make_range(ledger.plan_failure_record, 1, sizeof(std::uint64_t), &writes[3]) &&
         make_range(ledger.sequence_active, 1, sizeof(std::uint32_t), &writes[4]) &&
         pairwise_disjoint(writes);
}

bool overlaps_any(const AddressRange& range, const AddressRange* ranges,
                  std::size_t count) noexcept {
  for (std::size_t index = 0u; index < count; ++index) {
    if (ranges_overlap(range, ranges[index])) {
      return true;
    }
  }
  return false;
}

__device__ bool known_status(gpuxtb_status_t status) {
  return status >= GPUXTB_STATUS_SUCCESS && status <= GPUXTB_STATUS_EIGENSOLVER_FAILED;
}

__device__ bool aligned_device_pointer(const void* pointer, std::size_t alignment) {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

__device__ bool device_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                             std::uintptr_t* begin, std::uintptr_t* end) {
  if (elements < 0 || pointer == nullptr) {
    return false;
  }
  const std::uint64_t count = static_cast<std::uint64_t>(elements);
  if (count > static_cast<std::uint64_t>(~std::uintptr_t{0}) / element_size) {
    return false;
  }
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
  const std::uintptr_t bytes = static_cast<std::uintptr_t>(count * element_size);
  if (bytes > ~std::uintptr_t{0} - address) {
    return false;
  }
  *begin = address;
  *end = address + bytes;
  return true;
}

__device__ bool device_ranges_overlap(std::uintptr_t first_begin, std::uintptr_t first_end,
                                      const void* second_pointer, std::int64_t second_elements,
                                      std::size_t second_element_size) {
  std::uintptr_t second_begin = 0u;
  std::uintptr_t second_end = 0u;
  return !device_range(second_pointer, second_elements, second_element_size, &second_begin,
                       &second_end) ||
         (first_begin < second_end && second_begin < first_end);
}

__device__ bool provenance_generation_aliases_ledger(const Gfn2GeometryCacheProvenanceView& view,
                                                     const Gfn2SccIterationDeviceLedger& ledger) {
  if (view.generation_scope != Gfn2GenerationScope::kPerSystem) {
    return false;
  }
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
  if (!device_range(view.system_geometry_generations, view.system_generation_count,
                    sizeof(std::uint64_t), &begin, &end)) {
    return true;
  }
  return device_ranges_overlap(begin, end, ledger.active_mask, ledger.batch_elements,
                               sizeof(std::uint8_t)) ||
         device_ranges_overlap(begin, end, ledger.pending_statuses, ledger.batch_elements,
                               sizeof(gpuxtb_status_t)) ||
         device_ranges_overlap(begin, end, ledger.system_failure_records, ledger.batch_elements,
                               sizeof(std::uint64_t)) ||
         device_ranges_overlap(begin, end, ledger.plan_failure_record, 1, sizeof(std::uint64_t)) ||
         device_ranges_overlap(begin, end, ledger.sequence_active, 1, sizeof(std::uint32_t));
}

__device__ bool classified_peer(std::uint32_t code, std::uint64_t mask) {
  return code != 0u && code < 64u && (mask & (std::uint64_t{1} << code)) != 0u;
}

__device__ std::uint32_t load_system_code(const Gfn2SccStageDeviceReport& report,
                                          std::int64_t system) {
  if (report.system_code_format == Gfn2SccStageCodeFormat::kGpuxtbStatus) {
    return static_cast<std::uint32_t>(
        static_cast<const gpuxtb_status_t*>(report.system_codes)[system]);
  }
  return static_cast<const std::uint32_t*>(report.system_codes)[system];
}

__device__ void close_plan_sequence(const Gfn2SccIterationDeviceLedger& ledger,
                                    std::uint64_t record) {
  if (threadIdx.x == 0) {
    if (*ledger.plan_failure_record == 0u) {
      *ledger.plan_failure_record = record;
    }
    *ledger.sequence_active = 0u;
  }
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < ledger.batch_elements; system += blockDim.x) {
    ledger.active_mask[system] = 0u;
  }
}

__global__ void derive_activity_kernel(Gfn2SccIterationDevicePolicy policy,
                                       Gfn2SccIterationDeviceStateInput state,
                                       Gfn2SccIterationDeviceProvenance provenance,
                                       Gfn2SccIterationDeviceLedger ledger) {
  __shared__ int invalid_state;
  __shared__ int any_active;
  __shared__ unsigned long long plan_record;

  if (threadIdx.x == 0) {
    invalid_state = 0;
    any_active = 0;
    plan_record = 0u;
    *ledger.plan_failure_record = 0u;
    *ledger.sequence_active = 1u;
  }
  // All workers may update the shared flags below.  Publish thread 0's
  // initialization first so a worker cannot have its atomic update erased by
  // a late initialization store.
  __syncthreads();
  for (std::int64_t system = threadIdx.x; system < policy.batch_size; system += blockDim.x) {
    const gpuxtb_status_t status = state.system_statuses[system];
    const std::uint8_t converged = state.converged[system];
    const std::uint64_t iterations = state.iterations[system];
    ledger.system_failure_records[system] = 0u;
    ledger.pending_statuses[system] = status;
    if (!known_status(status) || converged > 1u) {
      ledger.active_mask[system] = 0u;
      atomicExch(&invalid_state, 1);
      continue;
    }
    const bool active = status == GPUXTB_STATUS_SUCCESS && converged == 0u &&
                        iterations < policy.maximum_iterations;
    ledger.active_mask[system] = active ? 1u : 0u;
    if (active) {
      atomicExch(&any_active, 1);
    }
  }
  __syncthreads();

  if (invalid_state != 0) {
    close_plan_sequence(
        ledger, gfn2_scc_stage_failure_record(
                    Gfn2SccStageId::kActivity,
                    static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kInvalidState)));
    return;
  }

  if (threadIdx.x == 0) {
    for (std::int64_t binding_index = 0; binding_index < provenance.cache_binding_count;
         ++binding_index) {
      const Gfn2SccCacheProvenanceBinding& binding = provenance.cache_bindings[binding_index];
      const Gfn2GeometryCacheProvenanceView& view = binding.provenance;
      const bool cross_plan = view.plan_token != policy.plan_token;
      const bool owner_in_domain = gfn2_scc_stage_id_in_domain(binding.owner_stage);
      const bool owner_valid = gfn2_scc_stage_id_is_valid(binding.owner_stage);
      const bool common_valid = owner_valid && binding.reserved == 0u &&
                                view.memory_space == Gfn2PlanMemorySpace::kCudaDevice &&
                                !cross_plan && view.batch_size == policy.batch_size;
      bool scope_valid = false;
      if (view.generation_scope == Gfn2GenerationScope::kBatch) {
        scope_valid =
            view.system_geometry_generations == nullptr && view.system_generation_count == 0;
      } else if (view.generation_scope == Gfn2GenerationScope::kPerSystem) {
        scope_valid =
            view.geometry_generation == 0u && view.system_generation_count == policy.batch_size &&
            aligned_device_pointer(view.system_geometry_generations, alignof(std::uint64_t)) &&
            !provenance_generation_aliases_ledger(view, ledger);
      }
      if (!common_valid || !scope_valid) {
        const Gfn2SccStageId stage = owner_valid ? binding.owner_stage : Gfn2SccStageId::kActivity;
        // An out-of-domain enum is malformed provenance, even when another
        // field also looks cross-plan.  Never encode unchecked enum bits into
        // the canonical stage-qualified failure record.
        const Gfn2SccIterationControlCode code =
            !owner_in_domain ? Gfn2SccIterationControlCode::kInvalidProvenance
                             : (cross_plan ? Gfn2SccIterationControlCode::kCrossPlan
                                           : Gfn2SccIterationControlCode::kInvalidProvenance);
        plan_record = gfn2_scc_stage_failure_record(stage, static_cast<std::uint32_t>(code));
        break;
      }
      if (any_active != 0 && view.generation_scope == Gfn2GenerationScope::kBatch &&
          view.geometry_generation != provenance.expected_geometry_generation) {
        plan_record = gfn2_scc_stage_failure_record(
            binding.owner_stage,
            static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kStaleGeneration));
        break;
      }
    }
  }
  __syncthreads();

  if (plan_record != 0u) {
    close_plan_sequence(ledger, plan_record);
    return;
  }

  for (std::int64_t system = threadIdx.x; system < policy.batch_size; system += blockDim.x) {
    if (ledger.active_mask[system] != 1u) {
      continue;
    }
    for (std::int64_t binding_index = 0; binding_index < provenance.cache_binding_count;
         ++binding_index) {
      const Gfn2SccCacheProvenanceBinding& binding = provenance.cache_bindings[binding_index];
      const Gfn2GeometryCacheProvenanceView& view = binding.provenance;
      if (view.generation_scope == Gfn2GenerationScope::kPerSystem &&
          view.system_geometry_generations[system] != provenance.expected_geometry_generation) {
        ledger.system_failure_records[system] = gfn2_scc_stage_failure_record(
            binding.owner_stage,
            static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kStaleGeneration));
        ledger.pending_statuses[system] = GPUXTB_STATUS_INTERNAL_ERROR;
        ledger.active_mask[system] = 0u;
        break;
      }
    }
    if (ledger.active_mask[system] == 1u && provenance.warm_start_generations != nullptr &&
        provenance.warm_start_generations[system] != provenance.expected_warm_start_generation) {
      ledger.system_failure_records[system] = gfn2_scc_stage_failure_record(
          Gfn2SccStageId::kWarmStartProvenance,
          static_cast<std::uint32_t>(Gfn2SccIterationControlCode::kStaleGeneration));
      ledger.pending_statuses[system] = GPUXTB_STATUS_INTERNAL_ERROR;
      ledger.active_mask[system] = 0u;
    }
  }
}

__global__ void normalize_stage_kernel(Gfn2SccStageDeviceReport report,
                                       Gfn2SccIterationDeviceLedger ledger) {
  __shared__ int run;
  __shared__ int invalid_activity;
  __shared__ int matching_device_peer_count;
  __shared__ unsigned long long indexed_plan;
  __shared__ std::uint32_t device_code;
  __shared__ std::uint32_t stage_sequence;
  __shared__ unsigned long long selected_plan_record;

  if (threadIdx.x == 0) {
    run = *ledger.sequence_active == 1u && *ledger.plan_failure_record == 0u ? 1 : 0;
    invalid_activity = 0;
    matching_device_peer_count = 0;
    indexed_plan = kNoIndexedCode;
    device_code = 0u;
    stage_sequence = 1u;
    selected_plan_record = 0u;
  }
  __syncthreads();
  if (run == 0) {
    return;
  }

  if (threadIdx.x == 0) {
    if (report.device_error != nullptr) {
      device_code = *report.device_error;
    }
    if (report.stage_sequence_active != nullptr) {
      stage_sequence = *report.stage_sequence_active;
    }
  }
  __syncthreads();

  for (std::int64_t system = threadIdx.x; system < ledger.batch_elements; system += blockDim.x) {
    const std::uint8_t active = ledger.active_mask[system];
    if (active > 1u) {
      atomicExch(&invalid_activity, 1);
      continue;
    }
    if (active == 0u || report.system_codes == nullptr) {
      continue;
    }
    const std::uint32_t code = load_system_code(report, system);
    if (code == 0u) {
      continue;
    }
    if (classified_peer(code, report.peer_error_mask)) {
      // The device scalar identifies one specific sticky peer error.  A
      // different classified peer code cannot localize that scalar. Plan-only
      // scalars deliberately never participate in peer localization, even if
      // their integer value collides with this per-system peer domain.
      if (report.device_code_role == Gfn2SccStageDeviceCodeRole::kMixedFirstError &&
          code == device_code) {
        atomicAdd(&matching_device_peer_count, 1);
      }
    } else {
      const unsigned long long indexed = (static_cast<unsigned long long>(system) << 32u) | code;
      atomicMin(&indexed_plan, indexed);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const bool latch_failed = stage_sequence != 1u;
    const bool has_system_plan = indexed_plan != kNoIndexedCode;
    const bool device_is_peer =
        report.device_code_role == Gfn2SccStageDeviceCodeRole::kMixedFirstError &&
        classified_peer(device_code, report.peer_error_mask);
    const bool device_is_plan = device_code != 0u && !device_is_peer;
    const bool peer_is_unlocalized = device_is_peer && matching_device_peer_count == 0;
    if (invalid_activity != 0) {
      selected_plan_record =
          gfn2_scc_stage_failure_record(report.stage, kGfn2SccStageMalformedReportCode);
    } else if (latch_failed || has_system_plan || device_is_plan || peer_is_unlocalized) {
      std::uint32_t raw_code = kGfn2SccStageSequenceClosedCode;
      // device_error is the stage's sticky first failure.  Preserve a plan
      // code from it even if indexed diagnostics or the sequence latch also
      // report a plan failure.
      if (device_is_plan) {
        raw_code = device_code;
      } else if (has_system_plan) {
        raw_code = static_cast<std::uint32_t>(indexed_plan);
      } else if (peer_is_unlocalized && !latch_failed) {
        raw_code = kGfn2SccStageUnlocalizedPeerCode;
      }
      selected_plan_record = gfn2_scc_stage_failure_record(report.stage, raw_code);
    }
  }
  __syncthreads();

  if (selected_plan_record != 0u) {
    close_plan_sequence(ledger, selected_plan_record);
    return;
  }

  if (report.system_codes == nullptr) {
    return;
  }
  for (std::int64_t system = threadIdx.x; system < ledger.batch_elements; system += blockDim.x) {
    if (ledger.active_mask[system] != 1u) {
      continue;
    }
    const std::uint32_t code = load_system_code(report, system);
    if (!classified_peer(code, report.peer_error_mask)) {
      continue;
    }
    if (ledger.system_failure_records[system] == 0u) {
      ledger.system_failure_records[system] = gfn2_scc_stage_failure_record(report.stage, code);
    }
    ledger.pending_statuses[system] = report.peer_failure_status;
    ledger.active_mask[system] = 0u;
  }
}

}  // namespace

cudaError_t derive_gfn2_scc_iteration_activity_cuda(
    const Gfn2SccIterationDevicePolicy& policy, const Gfn2SccIterationDeviceStateInput& state,
    const Gfn2SccIterationDeviceProvenance& provenance, const Gfn2SccIterationDeviceLedger& ledger,
    cudaStream_t stream) noexcept {
  if (policy.batch_size <= 0 || policy.maximum_iterations == 0u || policy.plan_token == 0u ||
      state.batch_elements != policy.batch_size || state.plan_token != policy.plan_token ||
      provenance.plan_token != policy.plan_token || provenance.cache_binding_count < 0 ||
      !is_aligned(state.iterations, alignof(std::uint64_t)) ||
      !is_aligned(state.system_statuses, alignof(gpuxtb_status_t)) ||
      !is_aligned(state.converged, alignof(std::uint8_t)) ||
      (provenance.cache_binding_count == 0
           ? provenance.cache_bindings != nullptr || provenance.expected_geometry_generation != 0u
           : !is_aligned(provenance.cache_bindings, alignof(Gfn2SccCacheProvenanceBinding)) ||
                 provenance.expected_geometry_generation == 0u) ||
      ((provenance.warm_start_generations == nullptr && provenance.warm_start_elements == 0 &&
        provenance.expected_warm_start_generation == 0u)
           ? false
           : (!is_aligned(provenance.warm_start_generations, alignof(std::uint64_t)) ||
              provenance.warm_start_elements != policy.batch_size ||
              provenance.expected_warm_start_generation == 0u))) {
    return cudaErrorInvalidValue;
  }

  AddressRange writes[5];
  if (!valid_ledger(ledger, policy.batch_size, policy.plan_token, writes)) {
    return cudaErrorInvalidValue;
  }
  AddressRange reads[5];
  if (!make_range(state.iterations, policy.batch_size, sizeof(std::uint64_t), &reads[0]) ||
      !make_range(state.system_statuses, policy.batch_size, sizeof(gpuxtb_status_t), &reads[1]) ||
      !make_range(state.converged, policy.batch_size, sizeof(std::uint8_t), &reads[2]) ||
      !make_range(provenance.cache_bindings, provenance.cache_binding_count,
                  sizeof(Gfn2SccCacheProvenanceBinding), &reads[3]) ||
      !make_range(provenance.warm_start_generations, provenance.warm_start_elements,
                  sizeof(std::uint64_t), &reads[4])) {
    return cudaErrorInvalidValue;
  }
  for (const AddressRange& read : reads) {
    if (overlaps_any(read, writes, 5u)) {
      return cudaErrorInvalidValue;
    }
  }

  derive_activity_kernel<<<1, kThreadsPerBlock, 0, stream>>>(policy, state, provenance, ledger);
  return cudaPeekAtLastError();
}

cudaError_t normalize_gfn2_scc_stage_cuda(const Gfn2SccStageDeviceReport& report,
                                          const Gfn2SccIterationDeviceLedger& ledger,
                                          cudaStream_t stream) noexcept {
  if (!gfn2_scc_stage_id_is_valid(report.stage) || report.plan_token == 0u ||
      report.plan_token != ledger.plan_token ||
      (report.system_code_format != Gfn2SccStageCodeFormat::kUint32Error &&
       report.system_code_format != Gfn2SccStageCodeFormat::kGpuxtbStatus) ||
      (report.system_codes == nullptr
           ? report.system_code_elements != 0
           : report.system_code_elements != ledger.batch_elements ||
                 !is_aligned(report.system_codes, alignof(std::uint32_t))) ||
      (report.device_error == nullptr
           ? report.device_error_elements != 0
           : report.device_error_elements != 1 ||
                 !is_aligned(report.device_error, alignof(std::uint32_t))) ||
      (report.stage_sequence_active == nullptr
           ? report.stage_sequence_elements != 0
           : report.stage_sequence_elements != 1 ||
                 !is_aligned(report.stage_sequence_active, alignof(std::uint32_t))) ||
      !gfn2_scc_stage_device_code_role_is_valid(report.device_code_role) ||
      (report.device_code_role == Gfn2SccStageDeviceCodeRole::kPlanOnly &&
       report.device_error == nullptr) ||
      (report.peer_error_mask & 1u) != 0u ||
      (report.peer_failure_status != GPUXTB_STATUS_INTERNAL_ERROR &&
       report.peer_failure_status != GPUXTB_STATUS_EIGENSOLVER_FAILED)) {
    return cudaErrorInvalidValue;
  }

  AddressRange writes[5];
  if (ledger.batch_elements <= 0 ||
      !valid_ledger(ledger, ledger.batch_elements, ledger.plan_token, writes)) {
    return cudaErrorInvalidValue;
  }
  AddressRange reads[3];
  if (!make_range(report.system_codes, report.system_code_elements, sizeof(std::uint32_t),
                  &reads[0]) ||
      !make_range(report.device_error, report.device_error_elements, sizeof(std::uint32_t),
                  &reads[1]) ||
      !make_range(report.stage_sequence_active, report.stage_sequence_elements,
                  sizeof(std::uint32_t), &reads[2])) {
    return cudaErrorInvalidValue;
  }
  for (const AddressRange& read : reads) {
    if (overlaps_any(read, writes, 5u)) {
      return cudaErrorInvalidValue;
    }
  }

  normalize_stage_kernel<<<1, kThreadsPerBlock, 0, stream>>>(report, ledger);
  return cudaPeekAtLastError();
}

}  // namespace gpuxtb::detail::cuda
