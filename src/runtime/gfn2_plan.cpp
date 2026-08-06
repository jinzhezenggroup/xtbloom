#include "runtime/gfn2_plan.hpp"
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

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
#include "runtime/validation.hpp"
#if defined(GPUXTB_HAS_CUDA)
#include "runtime/gfn2_cuda_execution.hpp"
#endif

namespace gpuxtb::detail {
namespace {

constexpr std::size_t kHostWorkspaceAlignment = 64u;
#if defined(GPUXTB_HAS_CUDA)
constexpr std::size_t kDeviceWorkspaceAlignment = 256u;
#endif

gpuxtb_compute_options_t default_plan_options() noexcept {
  gpuxtb_compute_options_t options{};
  options.struct_size = GPUXTB_COMPUTE_OPTIONS_V2_SIZE;
  options.api_version = GPUXTB_API_VERSION;
  options.model = GPUXTB_MODEL_GFN2_XTB;
  options.flags = GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES;
  options.max_scc_iterations = 250;
  options.charge_tolerance = 1.0e-6;
  options.energy_tolerance = 1.0e-8;
  options.electronic_temperature = GPUXTB_DEFAULT_ELECTRONIC_TEMPERATURE;
  options.scc_start_mode = GPUXTB_SCC_START_FRESH;
  return options;
}

void hash_append(std::uint64_t& hash, const void* bytes, std::size_t count) noexcept {
  const auto* cursor = static_cast<const std::uint8_t*>(bytes);
  for (std::size_t index = 0u; index < count; ++index) {
    hash ^= cursor[index];
    hash *= 0x100000001b3ULL;
  }
}

void hash_append_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
  hash_append(hash, &value, sizeof(value));
}

void hash_append_i64(std::uint64_t& hash, std::int64_t value) noexcept {
  const auto raw = static_cast<std::uint64_t>(value);
  hash_append(hash, &raw, sizeof(raw));
}

void hash_append_i32(std::uint64_t& hash, std::int32_t value) noexcept {
  const auto raw = static_cast<std::uint32_t>(value);
  hash_append(hash, &raw, sizeof(raw));
}

template <typename T>
void hash_elements(std::uint64_t& hash, const void* data, std::int64_t count) noexcept {
  hash_append_i64(hash, count);
  if (count > 0) {
    hash_append(hash, data, static_cast<std::size_t>(count) * sizeof(T));
  }
}

/* FNV-1a fingerprint of the immutable topology (atom offsets, element numbers,
 * charges, unpaired electrons, spin channels, point and response structure).
 * Geometry and periodic numerical values are excluded. Uses only scalar reads
 * so repeated plan_compute calls perform zero steady-state allocations. */
std::uint64_t topology_fingerprint(const gpuxtb_batch_t& batch) noexcept {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  hash_append_u64(hash, 1u); /* fingerprint schema version */
  hash_append_i64(hash, batch.batch_size);
  hash_append_i64(hash, batch.total_atoms);
  hash_append_i64(hash, batch.total_point_charges);
  hash_append_i64(hash, batch.total_charge_response_elements);
  hash_elements<std::int64_t>(hash, batch.atom_offsets.data, batch.batch_size + 1);
  hash_elements<std::int32_t>(hash, batch.atomic_numbers.data, batch.total_atoms);
  hash_elements<double>(hash, batch.molecular_charges.data, batch.batch_size);
  hash_elements<std::int32_t>(hash, batch.unpaired_electrons.data, batch.batch_size);
  const bool spin_present = batch.struct_size >= GPUXTB_BATCH_V2_SIZE &&
                            batch.spin_channels.data != nullptr &&
                            batch.spin_channels.size_bytes != 0u;
  if (spin_present) {
    hash_elements<std::int32_t>(hash, batch.spin_channels.data, batch.batch_size);
  } else {
    hash_append_i64(hash, batch.batch_size);
    for (std::int64_t system = 0; system < batch.batch_size; ++system) {
      hash_append_i32(hash, 1);
    }
  }
  /* A missing point-charge or response structure is distinct from any real
   * structure; hash a presence marker and the offsets only when supplied. */
  hash_append_u64(hash, batch.point_charge_offsets.data != nullptr ? 1u : 0u);
  if (batch.point_charge_offsets.data != nullptr) {
    hash_elements<std::int64_t>(hash, batch.point_charge_offsets.data, batch.batch_size + 1);
  }
  hash_append_u64(hash, batch.charge_response_offsets.data != nullptr ? 1u : 0u);
  if (batch.charge_response_offsets.data != nullptr) {
    hash_elements<std::int64_t>(hash, batch.charge_response_offsets.data, batch.batch_size + 1);
  }
  return hash;
}

/* Plan identity compares host-readable topology bytes on every plan compute.
 * CUDA callers may keep the immutable topology host-resident while numerical
 * geometry stays device-resident, which matches the public mixed-mode
 * convention; a device-resident topology is rejected for a plan. */
bool host_resident(gpuxtb_memory_space_t memory_space) noexcept {
  return memory_space == GPUXTB_MEMORY_HOST;
}

bool topology_host_resident(const gpuxtb_batch_t& batch) noexcept {
  const bool spin_present = batch.struct_size >= GPUXTB_BATCH_V2_SIZE &&
                            batch.spin_channels.data != nullptr &&
                            batch.spin_channels.size_bytes != 0u;
  const bool point_present =
      batch.point_charge_offsets.data != nullptr && batch.point_charge_offsets.size_bytes != 0u;
  const bool response_present = batch.charge_response_offsets.data != nullptr &&
                                batch.charge_response_offsets.size_bytes != 0u;
  return host_resident(static_cast<gpuxtb_memory_space_t>(batch.atom_offsets.memory_space)) &&
         host_resident(static_cast<gpuxtb_memory_space_t>(batch.atomic_numbers.memory_space)) &&
         host_resident(static_cast<gpuxtb_memory_space_t>(batch.molecular_charges.memory_space)) &&
         host_resident(static_cast<gpuxtb_memory_space_t>(batch.unpaired_electrons.memory_space)) &&
         (!spin_present ||
          host_resident(static_cast<gpuxtb_memory_space_t>(batch.spin_channels.memory_space))) &&
         (!point_present || host_resident(static_cast<gpuxtb_memory_space_t>(
                                batch.point_charge_offsets.memory_space))) &&
         (!response_present || host_resident(static_cast<gpuxtb_memory_space_t>(
                                   batch.charge_response_offsets.memory_space)));
}

#if defined(GPUXTB_HAS_CUDA)
std::size_t host_output_staging_bytes(std::int64_t batch_size, std::int64_t total_atoms,
                                      std::int64_t total_points, std::uint32_t flags) noexcept {
  const std::size_t batch = static_cast<std::size_t>(batch_size);
  const std::size_t atoms = static_cast<std::size_t>(total_atoms);
  const std::size_t points = static_cast<std::size_t>(total_points);
  std::size_t bytes = 0u;
  if ((flags & GPUXTB_COMPUTE_ENERGY) != 0u) {
    bytes += batch * sizeof(double);
  }
  if ((flags & GPUXTB_COMPUTE_FORCES) != 0u) {
    bytes += 3u * atoms * sizeof(double);
  }
  if ((flags & GPUXTB_COMPUTE_ATOMIC_CHARGES) != 0u) {
    bytes += atoms * sizeof(double);
  }
  if ((flags & GPUXTB_COMPUTE_POINT_CHARGE_FORCES) != 0u) {
    bytes += 3u * points * sizeof(double);
  }
  bytes += batch * (sizeof(std::int32_t) + sizeof(std::uint8_t) + sizeof(std::int32_t));
  return bytes;
}
#endif  // GPUXTB_HAS_CUDA

}  // namespace

struct Gfn2Plan::Impl {
  gpuxtb_backend_t backend = GPUXTB_BACKEND_CPU;
  Context* context = nullptr;
  std::uint64_t topology_fingerprint = 0u;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_point_charges = 0;
  /* Host and device staging reserved for the pre-warmed identity, used by the
   * workspace query until a CUDA runtime is measured after compute. CPU values
   * are captured at plan creation (topology + spin only) so they stay correct
   * even after another topology replaced the shared cache's prepared systems. */
  std::size_t cpu_persistent_bytes = 0u;
  std::uint64_t cuda_host_workspace_bytes = 0u;
  std::uint64_t cuda_base_device_bytes = 0u;
  std::uint64_t cuda_force_device_bytes = 0u;
  bool cuda_prepared = false;
};

Gfn2Plan::Gfn2Plan() : impl_(std::make_unique<Impl>()) {}
Gfn2Plan::~Gfn2Plan() = default;

bool Gfn2Plan::valid() const noexcept { return impl_ != nullptr && impl_->context != nullptr; }

void Gfn2Plan::destroy() noexcept {
  /* The plan borrows the context's execution caches; destruction only clears
   * our reference. The caller is required to destroy the plan before the
   * context. */
  impl_->context = nullptr;
  impl_->topology_fingerprint = 0u;
  impl_->batch_size = 0;
  impl_->total_atoms = 0;
  impl_->total_point_charges = 0;
  impl_->cpu_persistent_bytes = 0u;
  impl_->cuda_host_workspace_bytes = 0u;
  impl_->cuda_base_device_bytes = 0u;
  impl_->cuda_force_device_bytes = 0u;
  impl_->cuda_prepared = false;
}

gpuxtb_status_t Gfn2Plan::create(Context& context, const gpuxtb_batch_t& batch,
                                 std::string& error) {
  if (impl_->context != nullptr) {
    error = "plan has already been created";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (context.backend == GPUXTB_BACKEND_AUTO || context.backend > GPUXTB_BACKEND_CUDA) {
    error = "plan creation requires a resolved CPU or CUDA context backend";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (context.backend == GPUXTB_BACKEND_ROCM) {
    error = "the ROCm backend is reserved but not implemented";
    return GPUXTB_STATUS_NOT_SUPPORTED;
  }

  /* Validate the complete descriptor set exactly like gpuxtb_compute would,
   * so a plan is never created from a corrupted or inconsistent request. A
   * plan has no result descriptor at creation, so the plan-specific structure
   * validation skips output-buffer checks. */
  const gpuxtb_compute_options_t setup_options = default_plan_options();
  DescriptorValidationResult validation =
      validate_plan_descriptor_structure(context.backend, &batch, &setup_options);
  if (!validation.ok()) {
    error = std::move(validation.error);
    return validation.status;
  }
  if (!topology_host_resident(batch)) {
    error =
        "plan identity requires host-resident topology buffers (offsets, element numbers, "
        "charges, spin channels)";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  impl_->backend = context.backend;
  impl_->context = &context;
  impl_->topology_fingerprint = topology_fingerprint(batch);
  impl_->batch_size = batch.batch_size;
  impl_->total_atoms = batch.total_atoms;
  impl_->total_point_charges = batch.total_point_charges;
  impl_->cuda_prepared = false;

  if (context.backend == GPUXTB_BACKEND_CPU) {
    if (context.gfn2_cpu_execution_cache == nullptr) {
      error = "CPU context does not own a GFN2 execution cache";
      impl_->context = nullptr;
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    bool reused = false;
    gpuxtb_status_t status = prepare_restricted_gfn2_cpu(*context.gfn2_cpu_execution_cache, batch,
                                                         setup_options, reused, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      impl_->context = nullptr;
      return status;
    }
    impl_->cpu_persistent_bytes =
        persistent_workspace_bytes_restricted_gfn2_cpu(*context.gfn2_cpu_execution_cache);
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  }

#if defined(GPUXTB_HAS_CUDA)
  if (context.gfn2_cuda_execution_cache == nullptr) {
    error = "CUDA context does not own a GFN2 execution cache";
    impl_->context = nullptr;
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  {
    bool reused = false;
    gpuxtb_status_t status =
        context.gfn2_cuda_execution_cache->prepare_host(batch, setup_options, reused, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      impl_->context = nullptr;
      return status;
    }
    const Gfn2CudaExecutionIdentity identity = context.gfn2_cuda_execution_cache->identity();
    impl_->cuda_host_workspace_bytes = identity.provider_host_workspace_bytes;
    impl_->cuda_base_device_bytes =
        identity.topology_arena_bytes + identity.input_arena_bytes +
        identity.iteration_arena_bytes + identity.eigensolver_setup_arena_bytes +
        identity.numerical_refresh_arena_bytes + identity.inference_arena_bytes;
    impl_->cuda_force_device_bytes =
        identity.force_immutable_arena_bytes + identity.force_execution_arena_bytes;
    impl_->cuda_prepared = true;
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  }
#else
  error = "the gpuxtb library was built without CUDA support";
  impl_->context = nullptr;
  return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
#endif
}

gpuxtb_status_t Gfn2Plan::query_workspace(std::uint32_t compute_flags,
                                          gpuxtb_workspace_query_t& query, std::string& error) {
  if (impl_->context == nullptr || impl_->backend == GPUXTB_BACKEND_AUTO) {
    error = "plan is not created or has been destroyed";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (impl_->backend == GPUXTB_BACKEND_CPU) {
    /* Persistent bytes are captured at plan creation (topology + spin only),
     * so the answer stays valid even after another topology replaced the
     * shared cache's prepared systems. Add only the flag-dependent output
     * staging on top. */
    std::size_t host_bytes = impl_->cpu_persistent_bytes;
    const std::size_t batch = static_cast<std::size_t>(impl_->batch_size);
    const std::size_t atoms = static_cast<std::size_t>(impl_->total_atoms);
    const std::size_t points = static_cast<std::size_t>(impl_->total_point_charges);
    if ((compute_flags & GPUXTB_COMPUTE_ENERGY) != 0u) {
      host_bytes += batch * sizeof(double);
    }
    if ((compute_flags & GPUXTB_COMPUTE_FORCES) != 0u) {
      host_bytes += 3u * atoms * sizeof(double);
    }
    if ((compute_flags & GPUXTB_COMPUTE_ATOMIC_CHARGES) != 0u) {
      host_bytes += atoms * sizeof(double);
    }
    if ((compute_flags & GPUXTB_COMPUTE_POINT_CHARGE_FORCES) != 0u) {
      host_bytes += 3u * points * sizeof(double);
    }
    host_bytes += batch * (sizeof(std::int32_t) + sizeof(std::uint8_t) + sizeof(std::int32_t));
    host_bytes += batch * (3u * (atoms == 0u ? 1u : atoms) * sizeof(double));
    query.host_required_bytes = static_cast<std::uint64_t>(host_bytes);
    query.host_required_alignment = static_cast<std::uint32_t>(kHostWorkspaceAlignment);
    query.device_required_bytes = 0u;
    query.device_required_alignment = 1u;
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  }
#if defined(GPUXTB_HAS_CUDA)
  if (impl_->context->gfn2_cuda_execution_cache == nullptr) {
    error = "CUDA context does not own a GFN2 execution cache";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  const Gfn2CudaExecutionIdentity identity = impl_->context->gfn2_cuda_execution_cache->identity();
  const std::uint64_t base_device =
      impl_->cuda_prepared
          ? impl_->cuda_base_device_bytes
          : identity.topology_arena_bytes + identity.input_arena_bytes +
                identity.iteration_arena_bytes + identity.eigensolver_setup_arena_bytes +
                identity.numerical_refresh_arena_bytes + identity.inference_arena_bytes;
  const std::uint64_t force_device =
      impl_->cuda_prepared
          ? impl_->cuda_force_device_bytes
          : identity.force_immutable_arena_bytes + identity.force_execution_arena_bytes;
  const std::uint64_t device_bytes =
      base_device + ((compute_flags & GPUXTB_COMPUTE_FORCES) != 0u ? force_device : 0u);
  const std::uint64_t host_bytes =
      (impl_->cuda_prepared ? impl_->cuda_host_workspace_bytes
                            : static_cast<std::uint64_t>(identity.provider_host_workspace_bytes)) +
      static_cast<std::uint64_t>(host_output_staging_bytes(
          impl_->batch_size, impl_->total_atoms, impl_->total_point_charges, compute_flags));
  query.host_required_bytes = host_bytes;
  query.host_required_alignment = static_cast<std::uint32_t>(kHostWorkspaceAlignment);
  query.device_required_bytes = device_bytes;
  query.device_required_alignment = static_cast<std::uint32_t>(kDeviceWorkspaceAlignment);
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
#else
  error = "the gpuxtb library was built without CUDA support";
  return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
#endif
}

gpuxtb_status_t Gfn2Plan::compute(const gpuxtb_batch_t& batch,
                                  const gpuxtb_compute_options_t& options,
                                  gpuxtb_batch_result_t& result, std::string& error) {
  if (impl_->context == nullptr || impl_->backend == GPUXTB_BACKEND_AUTO) {
    error = "plan is not created or has been destroyed";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  /* A corrupted plan (destroyed, mismatched topology, or created for a
   * different request) must fail before any caller output is modified. The
   * descriptor structure is validated first, then the immutable topology is
   * compared against the plan identity. */
  try {
    DescriptorValidationResult validation =
        impl_->backend == GPUXTB_BACKEND_CUDA
            ? validate_compute_descriptor_structure(impl_->backend, &batch, &options, &result)
            : validate_compute_descriptors(impl_->backend, &batch, &options, &result);
    if (!validation.ok()) {
      error = std::move(validation.error);
      return validation.status;
    }
  } catch (const std::bad_alloc&) {
    error = "failed to allocate temporary storage while validating a plan compute request";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  } catch (const std::exception& exception) {
    error = exception.what();
    return GPUXTB_STATUS_INTERNAL_ERROR;
  } catch (...) {
    error = "unknown exception while validating a plan compute request";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }

  if (!topology_host_resident(batch)) {
    error =
        "plan compute requires host-resident topology buffers (offsets, element numbers, "
        "charges, spin channels)";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (topology_fingerprint(batch) != impl_->topology_fingerprint) {
    error = "the batch topology does not match the fixed plan topology";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  if (options.model == GPUXTB_MODEL_GFN1_XTB) {
    return GPUXTB_STATUS_NOT_SUPPORTED;
  }

  if (impl_->backend == GPUXTB_BACKEND_CPU) {
    if (impl_->context->gfn2_cpu_execution_cache == nullptr) {
      error = "CPU context does not own a GFN2 execution cache";
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    return execute_restricted_gfn2_cpu(*impl_->context->gfn2_cpu_execution_cache, batch, options,
                                       result, error);
  }
#if defined(GPUXTB_HAS_CUDA)
  if (impl_->context->gfn2_cuda_execution_cache == nullptr) {
    error = "CUDA context does not own a GFN2 execution cache";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  return execute_restricted_gfn2_cuda(*impl_->context->gfn2_cuda_execution_cache, batch, options,
                                      result, error);
#else
  error = "the gpuxtb library was built without CUDA support";
  return GPUXTB_STATUS_BACKEND_UNAVAILABLE;
#endif
}

}  // namespace gpuxtb::detail
