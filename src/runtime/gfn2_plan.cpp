#include "runtime/gfn2_plan.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime/backend.hpp"
#include "runtime/gfn2_cpu_execution.hpp"
#include "runtime/request.hpp"
#include "runtime/validation.hpp"
#if defined(XTBLOOM_HAS_CUDA)
#include "runtime/gfn2_cuda_execution.hpp"
#endif

namespace xtbloom::detail {
namespace {

constexpr std::size_t kHostWorkspaceAlignment = 64u;
#if defined(XTBLOOM_HAS_CUDA)
constexpr std::size_t kDeviceWorkspaceAlignment = 256u;
#endif

template <typename T>
void copy_host_elements(const void* source, std::size_t count, std::vector<T>& destination) {
  destination.resize(count);
  if (count != 0u) {
    std::memcpy(destination.data(), source, count * sizeof(T));
  }
}

struct FixedTopology {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_point_charges = 0;
  std::int64_t total_charge_response_elements = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> point_charge_offsets;
  std::vector<std::int64_t> charge_response_offsets;
  bool point_offsets_present = false;
  bool response_offsets_present = false;
  bool potential_shifts_present = false;
  bool response_matrix_present = false;

  [[nodiscard]] std::size_t retained_host_bytes() const noexcept {
    /* Plan compute reads these canonical snapshots on every topology match.
     * Count capacities because their complete heap reservations remain owned
     * by the plan even when a vector's logical size is smaller. */
    const auto vector_bytes = [](const auto& values) noexcept {
      return values.capacity() * sizeof(values[0]);
    };
    return vector_bytes(atom_offsets) + vector_bytes(atomic_numbers) +
           vector_bytes(molecular_charges) + vector_bytes(unpaired_electrons) +
           vector_bytes(spin_channels) + vector_bytes(point_charge_offsets) +
           vector_bytes(charge_response_offsets);
  }

  void capture(const xtbloom_batch_t& batch) {
    batch_size = batch.batch_size;
    total_atoms = batch.total_atoms;
    total_point_charges = batch.total_point_charges;
    total_charge_response_elements = batch.total_charge_response_elements;
    copy_host_elements(batch.atom_offsets.data, static_cast<std::size_t>(batch_size + 1),
                       atom_offsets);
    copy_host_elements(batch.atomic_numbers.data, static_cast<std::size_t>(total_atoms),
                       atomic_numbers);
    copy_host_elements(batch.molecular_charges.data, static_cast<std::size_t>(batch_size),
                       molecular_charges);
    copy_host_elements(batch.unpaired_electrons.data, static_cast<std::size_t>(batch_size),
                       unpaired_electrons);
    const bool spin_present = batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
                              batch.spin_channels.data != nullptr &&
                              batch.spin_channels.size_bytes != 0u;
    if (spin_present) {
      copy_host_elements(batch.spin_channels.data, static_cast<std::size_t>(batch_size),
                         spin_channels);
    } else {
      spin_channels.assign(static_cast<std::size_t>(batch_size), 1);
    }
    point_offsets_present = batch.point_charge_offsets.data != nullptr;
    if (point_offsets_present) {
      copy_host_elements(batch.point_charge_offsets.data, static_cast<std::size_t>(batch_size + 1),
                         point_charge_offsets);
    } else {
      point_charge_offsets.clear();
    }
    response_offsets_present = batch.charge_response_offsets.data != nullptr;
    if (response_offsets_present) {
      copy_host_elements(batch.charge_response_offsets.data,
                         static_cast<std::size_t>(batch_size + 1), charge_response_offsets);
    } else {
      charge_response_offsets.clear();
    }
    potential_shifts_present = batch.atomic_potential_shifts.data != nullptr;
    response_matrix_present = batch.charge_response_matrix.data != nullptr;
  }

  [[nodiscard]] bool matches(const xtbloom_batch_t& batch) const noexcept {
    if (batch.batch_size != batch_size || batch.total_atoms != total_atoms ||
        batch.total_point_charges != total_point_charges ||
        batch.total_charge_response_elements != total_charge_response_elements ||
        (batch.point_charge_offsets.data != nullptr) != point_offsets_present ||
        (batch.charge_response_offsets.data != nullptr) != response_offsets_present ||
        (batch.atomic_potential_shifts.data != nullptr) != potential_shifts_present ||
        (batch.charge_response_matrix.data != nullptr) != response_matrix_present) {
      return false;
    }
    const bool spin_present = batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
                              batch.spin_channels.data != nullptr &&
                              batch.spin_channels.size_bytes != 0u;
    const auto bytes_equal = [](const void* data, const auto& expected) {
      return expected.empty() ||
             std::memcmp(data, expected.data(), expected.size() * sizeof(expected[0])) == 0;
    };
    if (!bytes_equal(batch.atom_offsets.data, atom_offsets) ||
        !bytes_equal(batch.atomic_numbers.data, atomic_numbers) ||
        !bytes_equal(batch.molecular_charges.data, molecular_charges) ||
        !bytes_equal(batch.unpaired_electrons.data, unpaired_electrons) ||
        (point_offsets_present &&
         !bytes_equal(batch.point_charge_offsets.data, point_charge_offsets)) ||
        (response_offsets_present &&
         !bytes_equal(batch.charge_response_offsets.data, charge_response_offsets))) {
      return false;
    }
    if (spin_present) {
      return bytes_equal(batch.spin_channels.data, spin_channels);
    }
    return std::all_of(spin_channels.begin(), spin_channels.end(),
                       [](std::int32_t channels) { return channels == 1; });
  }
};

xtbloom_compute_options_t normalize_plan_policy(const xtbloom_compute_options_t& options) noexcept {
  /* ABI-v1 callers own only the first 48 bytes. Copy the validated prefix
   * field-by-field so plan creation never reads the optional v2 suffix. */
  xtbloom_compute_options_t policy{};
  policy.struct_size = XTBLOOM_COMPUTE_OPTIONS_V2_SIZE;
  policy.api_version = XTBLOOM_API_VERSION;
  policy.model = options.model;
  policy.flags = options.flags;
  policy.max_scc_iterations = options.max_scc_iterations;
  policy.reserved = 0u;
  policy.charge_tolerance = options.charge_tolerance;
  policy.energy_tolerance = options.energy_tolerance;
  policy.electronic_temperature = options.electronic_temperature;
  policy.scc_start_mode = XTBLOOM_SCC_START_FRESH;
  policy.reserved_v2 = 0u;
  return policy;
}

bool plan_policy_matches(const xtbloom_compute_options_t& policy,
                         const xtbloom_compute_options_t& options) noexcept {
  return options.model == policy.model && options.flags == policy.flags &&
         options.max_scc_iterations == policy.max_scc_iterations &&
         options.charge_tolerance == policy.charge_tolerance &&
         options.energy_tolerance == policy.energy_tolerance &&
         options.electronic_temperature == policy.electronic_temperature;
}

/* CPU plan identity compares host-readable topology bytes on every compute.
 * CUDA plans use the backend's canonical mixed-memory topology staging, which
 * validates pointer ownership before it snapshots or compares caller bytes. */
bool host_resident(xtbloom_memory_space_t memory_space) noexcept {
  return memory_space == XTBLOOM_MEMORY_HOST;
}

bool topology_host_resident(const xtbloom_batch_t& batch) noexcept {
  const bool spin_present = batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
                            batch.spin_channels.data != nullptr &&
                            batch.spin_channels.size_bytes != 0u;
  const bool point_present =
      batch.point_charge_offsets.data != nullptr && batch.point_charge_offsets.size_bytes != 0u;
  const bool response_present = batch.charge_response_offsets.data != nullptr &&
                                batch.charge_response_offsets.size_bytes != 0u;
  return host_resident(static_cast<xtbloom_memory_space_t>(batch.atom_offsets.memory_space)) &&
         host_resident(static_cast<xtbloom_memory_space_t>(batch.atomic_numbers.memory_space)) &&
         host_resident(static_cast<xtbloom_memory_space_t>(batch.molecular_charges.memory_space)) &&
         host_resident(
             static_cast<xtbloom_memory_space_t>(batch.unpaired_electrons.memory_space)) &&
         (!spin_present ||
          host_resident(static_cast<xtbloom_memory_space_t>(batch.spin_channels.memory_space))) &&
         (!point_present || host_resident(static_cast<xtbloom_memory_space_t>(
                                batch.point_charge_offsets.memory_space))) &&
         (!response_present || host_resident(static_cast<xtbloom_memory_space_t>(
                                   batch.charge_response_offsets.memory_space)));
}

}  // namespace

struct Gfn2Plan::Impl {
  xtbloom_backend_t backend = XTBLOOM_BACKEND_CPU;
  Context* context = nullptr;
  xtbloom_compute_options_t policy{};
  FixedTopology topology;
  std::shared_ptr<Gfn2CpuExecutionCache> cpu_cache;
#if defined(XTBLOOM_HAS_CUDA)
  std::shared_ptr<Gfn2CudaExecutionCache> cuda_cache;
#endif
  /* Retained host and device storage is captured from the plan-owned prepared
   * cache at creation, so workspace queries remain stable across computes and
   * across independent plans on the same context. */
  std::size_t cpu_persistent_bytes = 0u;
  std::uint64_t cuda_host_workspace_bytes = 0u;
  std::uint64_t cuda_device_workspace_bytes = 0u;
};

Gfn2Plan::Gfn2Plan() : impl_(std::make_unique<Impl>()) {}
Gfn2Plan::~Gfn2Plan() = default;

bool Gfn2Plan::valid() const noexcept { return impl_ != nullptr && impl_->context != nullptr; }

Context* Gfn2Plan::context() const noexcept { return impl_ == nullptr ? nullptr : impl_->context; }

void Gfn2Plan::destroy() noexcept {
  /* Prepared caches belong to the plan, while the context pointer remains a
   * borrowed lifetime binding. Callers must still destroy the plan first. */
  impl_->context = nullptr;
  impl_->cpu_cache.reset();
#if defined(XTBLOOM_HAS_CUDA)
  impl_->cuda_cache.reset();
#endif
  impl_->policy = {};
  impl_->topology = {};
  impl_->cpu_persistent_bytes = 0u;
  impl_->cuda_host_workspace_bytes = 0u;
  impl_->cuda_device_workspace_bytes = 0u;
}

xtbloom_status_t Gfn2Plan::create(Context& context, const xtbloom_batch_t& batch,
                                  const xtbloom_compute_options_t& options, std::string& error) {
  if (impl_->context != nullptr) {
    error = "plan has already been created";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (context.backend == XTBLOOM_BACKEND_AUTO || context.backend > XTBLOOM_BACKEND_CUDA) {
    error = "plan creation requires a resolved CPU or CUDA context backend";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (context.backend == XTBLOOM_BACKEND_ROCM) {
    error = "the ROCm backend is reserved but not implemented";
    return XTBLOOM_STATUS_NOT_SUPPORTED;
  }

  DescriptorValidationResult validation =
      validate_plan_descriptor_structure(context.backend, &batch, &options);
  if (!validation.ok()) {
    error = std::move(validation.error);
    return validation.status;
  }
  if (options.model == XTBLOOM_MODEL_GFN1_XTB) {
    error = "GFN1-xTB is reserved by the ABI but is not implemented yet";
    return XTBLOOM_STATUS_NOT_SUPPORTED;
  }

  impl_->backend = context.backend;
  impl_->context = &context;
  impl_->policy = normalize_plan_policy(options);
  impl_->topology.batch_size = batch.batch_size;
  impl_->topology.total_atoms = batch.total_atoms;
  impl_->topology.total_point_charges = batch.total_point_charges;
  impl_->topology.total_charge_response_elements = batch.total_charge_response_elements;

  if (context.backend == XTBLOOM_BACKEND_CPU) {
    if (!topology_host_resident(batch)) {
      error =
          "CPU plan identity requires host-resident topology buffers (offsets, element numbers, "
          "charges, spin channels)";
      impl_->context = nullptr;
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    impl_->topology.capture(batch);
    impl_->cpu_cache = std::make_shared<Gfn2CpuExecutionCache>(context.cpu_threads);
    bool reused = false;
    xtbloom_status_t status =
        prepare_restricted_gfn2_cpu(*impl_->cpu_cache, batch, impl_->policy, reused, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      impl_->context = nullptr;
      return status;
    }
    impl_->cpu_persistent_bytes =
        persistent_workspace_bytes_restricted_gfn2_cpu(*impl_->cpu_cache) +
        impl_->topology.retained_host_bytes();
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

#if defined(XTBLOOM_HAS_CUDA)
  impl_->cuda_cache = std::make_shared<Gfn2CudaExecutionCache>(context.device_id, context.stream);
  {
    xtbloom_status_t status = impl_->cuda_cache->prepare_topology_only(batch, impl_->policy, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      impl_->context = nullptr;
      return status;
    }
    const Gfn2CudaExecutionIdentity identity = impl_->cuda_cache->identity();
    impl_->cuda_host_workspace_bytes = identity.retained_host_workspace_bytes;
    impl_->cuda_device_workspace_bytes = identity.retained_device_workspace_bytes;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
#else
  error = "the xtbloom library was built without CUDA support";
  impl_->context = nullptr;
  return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
#endif
}

xtbloom_status_t Gfn2Plan::query_workspace(std::uint32_t compute_flags,
                                           xtbloom_workspace_query_t& query, std::string& error) {
  if (impl_->context == nullptr || impl_->backend == XTBLOOM_BACKEND_AUTO) {
    error = "plan is not created or has been destroyed";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  constexpr std::uint32_t kKnownComputeFlags =
      XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES | XTBLOOM_COMPUTE_ATOMIC_CHARGES |
      XTBLOOM_COMPUTE_POINT_CHARGE_FORCES | XTBLOOM_COMPUTE_DIPOLE_MOMENTS;
  if (query.reserved != 0u || query.reserved_v2 != 0u || compute_flags == 0u ||
      (compute_flags & ~kKnownComputeFlags) != 0u) {
    error = "workspace query contains unknown flags or nonzero reserved fields";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (compute_flags != impl_->policy.flags) {
    error = "workspace query flags do not match the plan compute policy";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (impl_->backend == XTBLOOM_BACKEND_CPU) {
    /* The captured value includes per-system state, copied inputs, policy
     * keys, worker/publication metadata, and all flag-dependent output staging. */
    query.host_required_bytes = static_cast<std::uint64_t>(impl_->cpu_persistent_bytes);
    query.host_required_alignment = static_cast<std::uint32_t>(kHostWorkspaceAlignment);
    query.device_required_bytes = 0u;
    query.device_required_alignment = 1u;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
#if defined(XTBLOOM_HAS_CUDA)
  if (impl_->cuda_cache == nullptr) {
    error = "plan does not own a CUDA GFN2 execution cache";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  query.host_required_bytes = impl_->cuda_host_workspace_bytes;
  query.host_required_alignment = static_cast<std::uint32_t>(kHostWorkspaceAlignment);
  query.device_required_bytes = impl_->cuda_device_workspace_bytes;
  query.device_required_alignment = static_cast<std::uint32_t>(kDeviceWorkspaceAlignment);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
#else
  error = "the xtbloom library was built without CUDA support";
  return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
#endif
}

xtbloom_status_t Gfn2Plan::compute(const xtbloom_batch_t& batch,
                                   const xtbloom_compute_options_t& options,
                                   xtbloom_batch_result_t& result, std::string& error) {
  if (impl_->context == nullptr || impl_->backend == XTBLOOM_BACKEND_AUTO) {
    error = "plan is not created or has been destroyed";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  /* A corrupted plan (destroyed, mismatched topology, or created for a
   * different request) must fail before any caller output is modified. The
   * descriptor structure is validated first, then the immutable topology is
   * compared against the plan identity. */
  try {
    DescriptorValidationResult validation =
        impl_->backend == XTBLOOM_BACKEND_CUDA
            ? validate_compute_descriptor_structure(impl_->backend, &batch, &options, &result)
            : validate_compute_descriptors(impl_->backend, &batch, &options, &result);
    if (!validation.ok()) {
      error = std::move(validation.error);
      return validation.status;
    }
  } catch (const std::bad_alloc&) {
    error = "failed to allocate temporary storage while validating a plan compute request";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::exception& exception) {
    error = exception.what();
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  } catch (...) {
    error = "unknown exception while validating a plan compute request";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  if (!plan_policy_matches(impl_->policy, options)) {
    error = "the compute options do not match the fixed plan policy";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  if (impl_->backend == XTBLOOM_BACKEND_CPU) {
    if (!topology_host_resident(batch)) {
      error =
          "CPU plan compute requires host-resident topology buffers (offsets, element numbers, "
          "charges, spin channels)";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (!impl_->topology.matches(batch)) {
      error = "the batch topology does not match the fixed plan topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (impl_->cpu_cache == nullptr) {
      error = "plan does not own a CPU GFN2 execution cache";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    return execute_restricted_gfn2_cpu(*impl_->cpu_cache, batch, options, result, error);
  }
#if defined(XTBLOOM_HAS_CUDA)
  if (impl_->cuda_cache == nullptr) {
    error = "plan does not own a CUDA GFN2 execution cache";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  return execute_restricted_gfn2_cuda_plan(*impl_->cuda_cache, batch, options, result, error);
#else
  error = "the xtbloom library was built without CUDA support";
  return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
#endif
}

xtbloom_status_t Gfn2Plan::enqueue(const xtbloom_batch_t& batch,
                                   const xtbloom_compute_options_t& options,
                                   const xtbloom_batch_result_t& result,
                                   RequestSubmission& submission, std::string& error) {
  if (impl_->context == nullptr || impl_->backend == XTBLOOM_BACKEND_AUTO) {
    error = "plan is not created or has been destroyed";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (impl_->backend == XTBLOOM_BACKEND_CPU) {
    error = "asynchronous plan enqueue is not supported by the CPU backend";
    return XTBLOOM_STATUS_NOT_SUPPORTED;
  }

  try {
    const DescriptorValidationResult validation =
        validate_compute_descriptor_structure(impl_->backend, &batch, &options, &result);
    if (!validation.ok()) {
      error = validation.error;
      return validation.status;
    }
  } catch (const std::bad_alloc&) {
    error = "failed to allocate temporary storage while validating a plan enqueue request";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::exception& exception) {
    error = exception.what();
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  } catch (...) {
    error = "unknown exception while validating a plan enqueue request";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  if (!plan_policy_matches(impl_->policy, options)) {
    error = "the compute options do not match the fixed plan policy";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
#if defined(XTBLOOM_HAS_CUDA)
  if (impl_->cuda_cache == nullptr) {
    error = "plan does not own a CUDA GFN2 execution cache";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  return enqueue_restricted_gfn2_cuda_plan(impl_->cuda_cache, batch, options, result, submission,
                                           error);
#else
  error = "the xtbloom library was built without CUDA support";
  return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
#endif
}

}  // namespace xtbloom::detail
