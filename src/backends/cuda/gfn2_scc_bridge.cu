#include <array>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_scc_bridge.cuh"

namespace xtbloom::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

__device__ std::uint32_t load_error(const std::uint32_t* error) {
  return atomicAdd(const_cast<std::uint32_t*>(error), 0u);
}

__device__ bool sequence_is_active(const Gfn2SccBridgeDeviceWorkspace& workspace) {
  return load_error(workspace.sequence_active) == 1u;
}

__device__ bool system_is_valid(const Gfn2SccBridgeDeviceStageInput& stage, std::int64_t system) {
  return load_error(stage.system_errors + system) == 0u;
}

__device__ void record_plan_error(const Gfn2SccBridgeDeviceOutput& output,
                                  const Gfn2SccBridgeDeviceWorkspace& workspace,
                                  Gfn2SccBridgeDeviceError error) {
  atomicCAS(output.downstream_plan_error, 0u, static_cast<std::uint32_t>(error));
  atomicExch(workspace.sequence_active, 0u);
}

__device__ void record_system_error(const Gfn2SccBridgeDeviceStageInput& stage, std::int64_t system,
                                    Gfn2SccBridgeDeviceError error) {
  atomicCAS(stage.system_errors + system, 0u, static_cast<std::uint32_t>(error));
}

__device__ bool valid_range(std::int64_t begin, std::int64_t end, std::int64_t extent) {
  return begin >= 0 && end >= begin && end <= extent;
}

__global__ void capture_upstream_stage_kernel(Gfn2SccBridgeDeviceStageInput stage,
                                              Gfn2SccBridgeDeviceOutput output,
                                              Gfn2SccBridgeDeviceWorkspace workspace) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (load_error(output.downstream_plan_error) != 0u) {
    *workspace.sequence_active = 0u;
    return;
  }
  const std::uint32_t error = load_error(stage.upstream_device_error);
  const std::uint32_t upstream_sequence = load_error(stage.upstream_sequence_active);
  const bool classified_peer =
      error < 64u && (stage.peer_error_mask & (std::uint64_t{1} << error)) != 0u;
  if (upstream_sequence == 1u && (error == 0u || classified_peer)) {
    *workspace.sequence_active = 1u;
  } else {
    *output.downstream_plan_error =
        error != 0u && !classified_peer
            ? error
            : static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kUpstreamPlanFailure);
    *workspace.sequence_active = 0u;
  }
}

/* The embedded common topology was fully validated at binding time. This
 * device preflight protects Graph replay against stale/corrupted partitions
 * before any field or shell map is dereferenced. */
__global__ void topology_preflight_kernel(Gfn2SccBridgeDeviceBatch batch,
                                          Gfn2SccBridgeDevicePotentialFields potential,
                                          Gfn2SccBridgeDeviceOutput output,
                                          Gfn2SccBridgeDeviceWorkspace workspace) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  const Gfn2RaggedTopologyView& topology = batch.topology;
  __shared__ int run;
  __shared__ int valid;
  __shared__ std::int64_t atom_begin;
  __shared__ std::int64_t atom_end;
  __shared__ std::int64_t shell_begin;
  __shared__ std::int64_t shell_end;
  if (threadIdx.x == 0) {
    run = sequence_is_active(workspace) ? 1 : 0;
  }
  __syncthreads();
  if (run == 0) {
    return;
  }
  if (threadIdx.x == 0) {
    atom_begin = topology.atom_offsets[system];
    atom_end = topology.atom_offsets[system + 1];
    shell_begin = topology.batch_shell_offsets[system];
    shell_end = topology.batch_shell_offsets[system + 1];
    const std::int64_t qsh_begin = batch.qsh_offsets[system];
    const std::int64_t qsh_end = batch.qsh_offsets[system + 1];
    const std::int64_t qat_begin = batch.qat_offsets[system];
    const std::int64_t qat_end = batch.qat_offsets[system + 1];
    valid = valid_range(atom_begin, atom_end, topology.total_atoms) &&
                    valid_range(shell_begin, shell_end, topology.total_shells)
                ? 1
                : 0;
    if (valid != 0 && system == 0 && (atom_begin != 0 || shell_begin != 0)) {
      valid = 0;
    }
    if (valid != 0 && system + 1 == topology.batch_size &&
        (atom_end != topology.total_atoms || shell_end != topology.total_shells)) {
      valid = 0;
    }
    if (valid == 0) {
      record_plan_error(output, workspace, Gfn2SccBridgeDeviceError::kInvalidTopologyOffsets);
    } else if (!valid_range(qsh_begin, qsh_end, potential.shell_elements) ||
               !valid_range(qat_begin, qat_end, potential.atom_elements) ||
               qsh_end - qsh_begin != shell_end - shell_begin ||
               qat_end - qat_begin != atom_end - atom_begin) {
      valid = 0;
      record_plan_error(output, workspace, Gfn2SccBridgeDeviceError::kInvalidFieldOffsets);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const std::int64_t atom = topology.shell_to_atom[shell];
    if (atom < atom_begin || atom >= atom_end) {
      record_plan_error(output, workspace, Gfn2SccBridgeDeviceError::kInvalidShellToAtom);
    }
  }
}

__global__ void collect_shell_scalar_kernel(Gfn2SccBridgeDeviceBatch batch,
                                            Gfn2SccBridgeDevicePotentialFields potential,
                                            Gfn2SccBridgeDeviceStageInput stage,
                                            Gfn2SccBridgeDeviceWorkspace workspace) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  if (threadIdx.x == 0) {
    active = 0;
    if (sequence_is_active(workspace) && system_is_valid(stage, system)) {
      const std::uint8_t requested =
          stage.requested_active == nullptr ? 1u : stage.requested_active[system];
      if (requested == 1u) {
        active = 1;
      } else if (requested != 0u) {
        record_system_error(stage, system, Gfn2SccBridgeDeviceError::kInvalidActiveMask);
      }
    }
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const Gfn2RaggedTopologyView& topology = batch.topology;
  const std::int64_t atom_begin = topology.atom_offsets[system];
  const std::int64_t shell_begin = topology.batch_shell_offsets[system];
  const std::int64_t shell_end = topology.batch_shell_offsets[system + 1];
  const std::int64_t qsh_begin = batch.qsh_offsets[system];
  const std::int64_t qat_begin = batch.qat_offsets[system];
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const std::int64_t atom = topology.shell_to_atom[shell];
    const double shell_value = potential.shell[qsh_begin + shell - shell_begin];
    const double atomic_value = potential.atomic[qat_begin + atom - atom_begin];
    if (!isfinite(shell_value)) {
      record_system_error(stage, system, Gfn2SccBridgeDeviceError::kNonfiniteShellPotential);
      continue;
    }
    if (!isfinite(atomic_value)) {
      record_system_error(stage, system, Gfn2SccBridgeDeviceError::kNonfiniteAtomicPotential);
      continue;
    }
    /* Do not contract or reassociate: this is tblite add_vat_to_vsh order. */
    const double complete = shell_value + atomic_value;
    if (!isfinite(complete)) {
      record_system_error(stage, system,
                          Gfn2SccBridgeDeviceError::kNonfiniteScalarPotentialArithmetic);
      continue;
    }
    workspace.shell_scratch[shell] = complete;
  }
}

__global__ void publish_shell_scalar_and_gate_kernel(Gfn2SccBridgeDeviceBatch batch,
                                                     Gfn2SccBridgeDeviceStageInput stage,
                                                     Gfn2SccBridgeDeviceOutput output,
                                                     Gfn2SccBridgeDeviceWorkspace workspace) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int publish;
  if (threadIdx.x == 0) {
    const std::uint8_t requested =
        stage.requested_active == nullptr ? 1u : stage.requested_active[system];
    publish =
        sequence_is_active(workspace) && requested == 1u && system_is_valid(stage, system) ? 1 : 0;
    output.downstream_active[system] = publish == 1 ? 1u : 0u;
  }
  __syncthreads();
  if (publish == 0) {
    return;
  }
  const std::int64_t shell_begin = batch.topology.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.topology.batch_shell_offsets[system + 1];
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    output.shell_scalar[shell] = workspace.shell_scratch[shell];
  }
}

__device__ bool canonical_sequence_is_open(const Gfn2SccIterationDeviceActivity& activity,
                                           const Gfn2SccBridgeDeviceWorkspace& workspace) {
  return load_error(activity.sequence_active) == 1u && sequence_is_active(workspace);
}

__device__ bool canonical_system_is_active(const Gfn2SccIterationDeviceActivity& activity,
                                           const Gfn2SccBridgeDeviceWorkspace& workspace,
                                           const std::uint32_t* system_errors,
                                           std::int64_t system) {
  /* Preserve the canonical order: sequence, member, then stage diagnostics.
   * No offset or numerical field is touched until all three gates pass. */
  return canonical_sequence_is_open(activity, workspace) && activity.active_mask[system] == 1u &&
         load_error(system_errors + system) == 0u;
}

__device__ bool canonical_bridge_plan_error(std::uint32_t error) {
  return error == static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kInvalidTopologyOffsets) ||
         error == static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kInvalidFieldOffsets) ||
         error == static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kInvalidShellToAtom) ||
         error == static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kUpstreamPlanFailure);
}

__device__ void record_canonical_plan_error(const Gfn2SccBridgeDeviceWorkspace& workspace,
                                            std::uint32_t* device_error,
                                            Gfn2SccBridgeDeviceError error) {
  atomicCAS(device_error, 0u, static_cast<std::uint32_t>(error));
  atomicExch(workspace.sequence_active, 0u);
}

__device__ void record_canonical_system_error(std::uint32_t* system_errors,
                                              std::uint32_t* device_error, std::int64_t system,
                                              Gfn2SccBridgeDeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  /* The canonical bridge report is PlanOnly. Peer codes live exclusively in
   * the indexed array so normalization can disable one member without
   * promoting that code through the plan-only scalar. */
  atomicCAS(system_errors + system, 0u, code);
  (void)device_error;
}

__global__ void canonical_bridge_preflight_kernel(Gfn2SccBridgeDeviceBatch batch,
                                                  Gfn2SccBridgeDevicePotentialFields potential,
                                                  Gfn2SccIterationDeviceActivity activity,
                                                  Gfn2SccBridgeDeviceWorkspace workspace,
                                                  std::uint32_t* device_error) {
  __shared__ int sequence_open;
  __shared__ int any_active;
  if (threadIdx.x == 0) {
    sequence_open = load_error(activity.sequence_active) == 1u ? 1 : 0;
    any_active = 0;
    *workspace.sequence_active =
        sequence_open != 0 && !canonical_bridge_plan_error(load_error(device_error)) ? 1u : 0u;
  }
  __syncthreads();
  if (sequence_open == 0 || !sequence_is_active(workspace)) {
    return;
  }
  const Gfn2RaggedTopologyView& topology = batch.topology;
  for (std::int64_t system = threadIdx.x; system < topology.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] == 1u) {
      atomicExch(&any_active, 1);
    }
  }
  __syncthreads();
  if (any_active == 0 || !sequence_is_active(workspace)) {
    return;
  }

  for (std::int64_t system = threadIdx.x; system < topology.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] != 1u) {
      continue;
    }
    const std::int64_t atom_begin = topology.atom_offsets[system];
    const std::int64_t atom_end = topology.atom_offsets[system + 1];
    const std::int64_t shell_begin = topology.batch_shell_offsets[system];
    const std::int64_t shell_end = topology.batch_shell_offsets[system + 1];
    const std::int64_t qsh_begin = batch.qsh_offsets[system];
    const std::int64_t qsh_end = batch.qsh_offsets[system + 1];
    const std::int64_t qat_begin = batch.qat_offsets[system];
    const std::int64_t qat_end = batch.qat_offsets[system + 1];
    bool valid = valid_range(atom_begin, atom_end, topology.total_atoms) &&
                 valid_range(shell_begin, shell_end, topology.total_shells);
    if (valid && system == 0) {
      valid = atom_begin == 0 && shell_begin == 0;
    }
    if (valid && system + 1 == topology.batch_size) {
      valid = atom_end == topology.total_atoms && shell_end == topology.total_shells;
    }
    if (!valid) {
      record_canonical_plan_error(workspace, device_error,
                                  Gfn2SccBridgeDeviceError::kInvalidTopologyOffsets);
      continue;
    }
    if (!valid_range(qsh_begin, qsh_end, potential.shell_elements) ||
        !valid_range(qat_begin, qat_end, potential.atom_elements) ||
        qsh_end - qsh_begin != shell_end - shell_begin ||
        qat_end - qat_begin != atom_end - atom_begin) {
      record_canonical_plan_error(workspace, device_error,
                                  Gfn2SccBridgeDeviceError::kInvalidFieldOffsets);
      continue;
    }
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::int64_t atom = topology.shell_to_atom[shell];
      if (atom < atom_begin || atom >= atom_end) {
        record_canonical_plan_error(workspace, device_error,
                                    Gfn2SccBridgeDeviceError::kInvalidShellToAtom);
        break;
      }
    }
  }
}

__global__ void collect_canonical_shell_scalar_kernel(Gfn2SccBridgeDeviceBatch batch,
                                                      Gfn2SccBridgeDevicePotentialFields potential,
                                                      Gfn2SccIterationDeviceActivity activity,
                                                      Gfn2SccBridgeDeviceWorkspace workspace,
                                                      std::uint32_t* system_errors,
                                                      std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ int active;
  if (threadIdx.x == 0) {
    active = canonical_system_is_active(activity, workspace, system_errors, system) ? 1 : 0;
  }
  __syncthreads();
  if (active == 0) {
    return;
  }

  const Gfn2RaggedTopologyView& topology = batch.topology;
  const std::int64_t atom_begin = topology.atom_offsets[system];
  const std::int64_t shell_begin = topology.batch_shell_offsets[system];
  const std::int64_t shell_end = topology.batch_shell_offsets[system + 1];
  const std::int64_t qsh_begin = batch.qsh_offsets[system];
  const std::int64_t qat_begin = batch.qat_offsets[system];
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const std::int64_t atom = topology.shell_to_atom[shell];
    const double shell_value = potential.shell[qsh_begin + shell - shell_begin];
    const double atomic_value = potential.atomic[qat_begin + atom - atom_begin];
    if (!isfinite(shell_value)) {
      record_canonical_system_error(system_errors, device_error, system,
                                    Gfn2SccBridgeDeviceError::kNonfiniteShellPotential);
      continue;
    }
    if (!isfinite(atomic_value)) {
      record_canonical_system_error(system_errors, device_error, system,
                                    Gfn2SccBridgeDeviceError::kNonfiniteAtomicPotential);
      continue;
    }
    const double complete = shell_value + atomic_value;
    if (!isfinite(complete)) {
      record_canonical_system_error(system_errors, device_error, system,
                                    Gfn2SccBridgeDeviceError::kNonfiniteScalarPotentialArithmetic);
      continue;
    }
    workspace.shell_scratch[shell] = complete;
  }
}

__global__ void publish_canonical_shell_scalar_kernel(Gfn2SccBridgeDeviceBatch batch,
                                                      Gfn2SccIterationDeviceActivity activity,
                                                      double* shell_scalar,
                                                      Gfn2SccBridgeDeviceWorkspace workspace,
                                                      const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!canonical_system_is_active(activity, workspace, system_errors, system)) {
    return;
  }
  const std::int64_t shell_begin = batch.topology.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.topology.batch_shell_offsets[system + 1];
  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    shell_scalar[shell] = workspace.shell_scratch[shell];
  }
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::int64_t elements, std::size_t element_size,
                AddressRange* range) noexcept {
  if (elements < 0 || element_size == 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes == 0u) {
    *range = {};
    return pointer == nullptr;
  }
  if (pointer == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  *range = {begin, begin + bytes};
  return true;
}

bool overlaps(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t ReadCount, std::size_t WriteCount>
bool disjoint_bindings(const std::array<AddressRange, ReadCount>& reads,
                       const std::array<AddressRange, WriteCount>& writes) noexcept {
  for (std::size_t first = 0; first < writes.size(); ++first) {
    for (std::size_t second = first + 1u; second < writes.size(); ++second) {
      if (overlaps(writes[first], writes[second])) {
        return false;
      }
    }
    for (const AddressRange& read : reads) {
      if (overlaps(writes[first], read)) {
        return false;
      }
    }
  }
  return true;
}

bool valid_host_binding(const Gfn2SccBridgeDeviceBatch& batch,
                        const Gfn2SccBridgeDevicePotentialFields& potential,
                        const Gfn2SccBridgeDeviceStageInput& stage,
                        const Gfn2SccBridgeDeviceOutput& output,
                        const Gfn2SccBridgeDeviceWorkspace& workspace) noexcept {
  const Gfn2RaggedTopologyView& topology = batch.topology;
  if (topology.memory_space != Gfn2PlanMemorySpace::kCudaDevice || topology.plan_token == 0u ||
      topology.batch_size <= 0 || topology.total_atoms < 0 || topology.total_shells < 0 ||
      topology.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      topology.atom_offset_count != topology.batch_size + 1 ||
      topology.batch_shell_offset_count != topology.batch_size + 1 ||
      topology.shell_to_atom_count != topology.total_shells ||
      batch.qsh_offset_count != topology.batch_size + 1 ||
      batch.qat_offset_count != topology.batch_size + 1 ||
      potential.plan_token != topology.plan_token || stage.plan_token != topology.plan_token ||
      output.plan_token != topology.plan_token || workspace.plan_token != topology.plan_token ||
      potential.shell_elements < topology.total_shells ||
      potential.atom_elements < topology.total_atoms ||
      !((stage.requested_active == nullptr && stage.active_elements == 0) ||
        (stage.active_elements == topology.batch_size &&
         is_aligned(stage.requested_active, alignof(std::uint8_t)))) ||
      stage.system_error_elements != topology.batch_size ||
      stage.upstream_device_error_elements != 1 || stage.upstream_sequence_elements != 1 ||
      (stage.peer_error_mask & 1u) != 0u || output.shell_elements != topology.total_shells ||
      output.active_elements != topology.batch_size || output.plan_error_elements != 1 ||
      workspace.shell_elements < topology.total_shells || workspace.sequence_elements < 1 ||
      !is_aligned(topology.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(topology.batch_shell_offsets, alignof(std::int64_t)) ||
      (topology.total_shells != 0 && !is_aligned(topology.shell_to_atom, alignof(std::int64_t))) ||
      !is_aligned(batch.qsh_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.qat_offsets, alignof(std::int64_t)) ||
      (potential.shell_elements != 0 && !is_aligned(potential.shell, alignof(double))) ||
      (potential.atom_elements != 0 && !is_aligned(potential.atomic, alignof(double))) ||
      !is_aligned(stage.system_errors, alignof(std::uint32_t)) ||
      !is_aligned(stage.upstream_device_error, alignof(std::uint32_t)) ||
      !is_aligned(stage.upstream_sequence_active, alignof(std::uint32_t)) ||
      (topology.total_shells != 0 && !is_aligned(output.shell_scalar, alignof(double))) ||
      !is_aligned(output.downstream_active, alignof(std::uint8_t)) ||
      !is_aligned(output.downstream_plan_error, alignof(std::uint32_t)) ||
      (topology.total_shells != 0 && !is_aligned(workspace.shell_scratch, alignof(double))) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t))) {
    return false;
  }

  std::array<AddressRange, 10> reads{};
  std::array<AddressRange, 6> writes{};
  if (!make_range(topology.atom_offsets, topology.atom_offset_count, sizeof(std::int64_t),
                  &reads[0]) ||
      !make_range(topology.batch_shell_offsets, topology.batch_shell_offset_count,
                  sizeof(std::int64_t), &reads[1]) ||
      !make_range(topology.shell_to_atom, topology.shell_to_atom_count, sizeof(std::int64_t),
                  &reads[2]) ||
      !make_range(batch.qsh_offsets, batch.qsh_offset_count, sizeof(std::int64_t), &reads[3]) ||
      !make_range(batch.qat_offsets, batch.qat_offset_count, sizeof(std::int64_t), &reads[4]) ||
      !make_range(potential.shell, potential.shell_elements, sizeof(double), &reads[5]) ||
      !make_range(potential.atomic, potential.atom_elements, sizeof(double), &reads[6]) ||
      !make_range(stage.requested_active, stage.active_elements, sizeof(std::uint8_t), &reads[7]) ||
      !make_range(stage.upstream_device_error, 1, sizeof(std::uint32_t), &reads[8]) ||
      !make_range(stage.upstream_sequence_active, 1, sizeof(std::uint32_t), &reads[9]) ||
      !make_range(stage.system_errors, stage.system_error_elements, sizeof(std::uint32_t),
                  &writes[0]) ||
      !make_range(output.shell_scalar, output.shell_elements, sizeof(double), &writes[1]) ||
      !make_range(output.downstream_active, output.active_elements, sizeof(std::uint8_t),
                  &writes[2]) ||
      !make_range(output.downstream_plan_error, 1, sizeof(std::uint32_t), &writes[3]) ||
      !make_range(workspace.shell_scratch, workspace.shell_elements, sizeof(double), &writes[4]) ||
      !make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[5])) {
    return false;
  }
  return disjoint_bindings(reads, writes);
}

bool valid_canonical_host_binding(const Gfn2SccBridgeDeviceBatch& batch,
                                  const Gfn2SccBridgeDevicePotentialFields& potential,
                                  const Gfn2SccIterationDeviceActivity& activity,
                                  double* shell_scalar, std::int64_t shell_elements,
                                  const Gfn2SccBridgeDeviceWorkspace& workspace,
                                  std::uint32_t* system_errors,
                                  std::uint32_t* device_error) noexcept {
  const Gfn2RaggedTopologyView& topology = batch.topology;
  if (topology.memory_space != Gfn2PlanMemorySpace::kCudaDevice || topology.plan_token == 0u ||
      topology.batch_size <= 0 || topology.total_atoms < 0 || topology.total_shells < 0 ||
      topology.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
      topology.atom_offset_count != topology.batch_size + 1 ||
      topology.batch_shell_offset_count != topology.batch_size + 1 ||
      topology.shell_to_atom_count != topology.total_shells ||
      batch.qsh_offset_count != topology.batch_size + 1 ||
      batch.qat_offset_count != topology.batch_size + 1 ||
      potential.plan_token != topology.plan_token || activity.plan_token != topology.plan_token ||
      workspace.plan_token != topology.plan_token ||
      potential.shell_elements < topology.total_shells ||
      potential.atom_elements < topology.total_atoms || activity.active_mask == nullptr ||
      activity.sequence_active == nullptr || activity.batch_elements != topology.batch_size ||
      activity.sequence_elements != 1 || shell_elements != topology.total_shells ||
      workspace.shell_elements < topology.total_shells || workspace.sequence_elements < 1 ||
      !is_aligned(topology.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(topology.batch_shell_offsets, alignof(std::int64_t)) ||
      (topology.total_shells != 0 && !is_aligned(topology.shell_to_atom, alignof(std::int64_t))) ||
      !is_aligned(batch.qsh_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.qat_offsets, alignof(std::int64_t)) ||
      (potential.shell_elements != 0 && !is_aligned(potential.shell, alignof(double))) ||
      (potential.atom_elements != 0 && !is_aligned(potential.atomic, alignof(double))) ||
      !is_aligned(activity.active_mask, alignof(std::uint8_t)) ||
      !is_aligned(activity.sequence_active, alignof(std::uint32_t)) ||
      (topology.total_shells != 0 && !is_aligned(shell_scalar, alignof(double))) ||
      (topology.total_shells != 0 && !is_aligned(workspace.shell_scratch, alignof(double))) ||
      !is_aligned(workspace.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return false;
  }

  std::array<AddressRange, 9> reads{};
  std::array<AddressRange, 5> writes{};
  if (!make_range(topology.atom_offsets, topology.atom_offset_count, sizeof(std::int64_t),
                  &reads[0]) ||
      !make_range(topology.batch_shell_offsets, topology.batch_shell_offset_count,
                  sizeof(std::int64_t), &reads[1]) ||
      !make_range(topology.shell_to_atom, topology.shell_to_atom_count, sizeof(std::int64_t),
                  &reads[2]) ||
      !make_range(batch.qsh_offsets, batch.qsh_offset_count, sizeof(std::int64_t), &reads[3]) ||
      !make_range(batch.qat_offsets, batch.qat_offset_count, sizeof(std::int64_t), &reads[4]) ||
      !make_range(potential.shell, potential.shell_elements, sizeof(double), &reads[5]) ||
      !make_range(potential.atomic, potential.atom_elements, sizeof(double), &reads[6]) ||
      !make_range(activity.active_mask, activity.batch_elements, sizeof(std::uint8_t), &reads[7]) ||
      !make_range(activity.sequence_active, 1, sizeof(std::uint32_t), &reads[8]) ||
      !make_range(shell_scalar, shell_elements, sizeof(double), &writes[0]) ||
      !make_range(workspace.shell_scratch, workspace.shell_elements, sizeof(double), &writes[1]) ||
      !make_range(workspace.sequence_active, 1, sizeof(std::uint32_t), &writes[2]) ||
      !make_range(system_errors, topology.batch_size, sizeof(std::uint32_t), &writes[3]) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &writes[4])) {
    return false;
  }
  return disjoint_bindings(reads, writes);
}

}  // namespace

cudaError_t reset_gfn2_scc_bridge_stage_cuda(std::int64_t batch_size,
                                             std::uint8_t* downstream_active,
                                             std::uint32_t* downstream_plan_error,
                                             std::uint32_t* sequence_active,
                                             cudaStream_t stream) noexcept {
  if (batch_size <= 0 ||
      static_cast<std::uint64_t>(batch_size) > std::numeric_limits<std::size_t>::max() ||
      !is_aligned(downstream_active, alignof(std::uint8_t)) ||
      !is_aligned(downstream_plan_error, alignof(std::uint32_t)) ||
      !is_aligned(sequence_active, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  AddressRange active_range;
  AddressRange plan_range;
  AddressRange sequence_range;
  if (!make_range(downstream_active, batch_size, sizeof(std::uint8_t), &active_range) ||
      !make_range(downstream_plan_error, 1, sizeof(std::uint32_t), &plan_range) ||
      !make_range(sequence_active, 1, sizeof(std::uint32_t), &sequence_range) ||
      overlaps(active_range, plan_range) || overlaps(active_range, sequence_range) ||
      overlaps(plan_range, sequence_range)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status =
      cudaMemsetAsync(downstream_active, 0, static_cast<std::size_t>(batch_size), stream);
  if (status == cudaSuccess) {
    status = cudaMemsetAsync(downstream_plan_error, 0, sizeof(std::uint32_t), stream);
  }
  return status == cudaSuccess ? cudaMemsetAsync(sequence_active, 0, sizeof(std::uint32_t), stream)
                               : status;
}

cudaError_t reset_gfn2_scc_bridge_device_errors_cuda(std::int64_t batch_size,
                                                     std::uint32_t* system_errors,
                                                     std::uint32_t* device_error,
                                                     std::uint32_t* sequence_active,
                                                     cudaStream_t stream) noexcept {
  if (batch_size <= 0 ||
      static_cast<std::uint64_t>(batch_size) > std::numeric_limits<std::size_t>::max() ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !is_aligned(sequence_active, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  AddressRange systems;
  AddressRange device;
  AddressRange sequence;
  if (!make_range(system_errors, batch_size, sizeof(std::uint32_t), &systems) ||
      !make_range(device_error, 1, sizeof(std::uint32_t), &device) ||
      !make_range(sequence_active, 1, sizeof(std::uint32_t), &sequence) ||
      overlaps(systems, device) || overlaps(systems, sequence) || overlaps(device, sequence)) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(std::uint32_t), stream);
  if (status == cudaSuccess) {
    status = cudaMemsetAsync(device_error, 0, sizeof(std::uint32_t), stream);
  }
  return status == cudaSuccess ? cudaMemsetAsync(sequence_active, 0, sizeof(std::uint32_t), stream)
                               : status;
}

cudaError_t collect_gfn2_scc_shell_scalar_potential_cuda(
    const Gfn2SccBridgeDeviceBatch& batch, const Gfn2SccBridgeDevicePotentialFields& potential,
    const Gfn2SccBridgeDeviceStageInput& stage, const Gfn2SccBridgeDeviceOutput& output,
    const Gfn2SccBridgeDeviceWorkspace& workspace, cudaStream_t stream) noexcept {
  if (!valid_host_binding(batch, potential, stage, output, workspace)) {
    return batch.topology.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max())
               ? cudaErrorInvalidConfiguration
               : cudaErrorInvalidValue;
  }

  capture_upstream_stage_kernel<<<1, 1, 0, stream>>>(stage, output, workspace);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  topology_preflight_kernel<<<static_cast<unsigned int>(batch.topology.batch_size),
                              kThreadsPerBlock, 0, stream>>>(batch, potential, output, workspace);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  collect_shell_scalar_kernel<<<static_cast<unsigned int>(batch.topology.batch_size),
                                kThreadsPerBlock, 0, stream>>>(batch, potential, stage, workspace);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_shell_scalar_and_gate_kernel<<<static_cast<unsigned int>(batch.topology.batch_size),
                                         kThreadsPerBlock, 0, stream>>>(batch, stage, output,
                                                                        workspace);
  return cudaPeekAtLastError();
}

cudaError_t collect_gfn2_scc_shell_scalar_potential_cuda(
    const Gfn2SccBridgeDeviceBatch& batch, const Gfn2SccBridgeDevicePotentialFields& potential,
    const Gfn2SccIterationDeviceActivity& activity, double* shell_scalar,
    std::int64_t shell_elements, const Gfn2SccBridgeDeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!valid_canonical_host_binding(batch, potential, activity, shell_scalar, shell_elements,
                                    workspace, system_errors, device_error)) {
    return batch.topology.batch_size > static_cast<std::int64_t>(std::numeric_limits<int>::max())
               ? cudaErrorInvalidConfiguration
               : cudaErrorInvalidValue;
  }
  canonical_bridge_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, potential, activity,
                                                                        workspace, device_error);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  collect_canonical_shell_scalar_kernel<<<static_cast<unsigned int>(batch.topology.batch_size),
                                          kThreadsPerBlock, 0, stream>>>(
      batch, potential, activity, workspace, system_errors, device_error);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return status;
  }
  publish_canonical_shell_scalar_kernel<<<static_cast<unsigned int>(batch.topology.batch_size),
                                          kThreadsPerBlock, 0, stream>>>(
      batch, activity, shell_scalar, workspace, system_errors);
  return cudaPeekAtLastError();
}

}  // namespace xtbloom::detail::cuda
