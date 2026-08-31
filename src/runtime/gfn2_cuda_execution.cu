// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn1_classical_corrections.cuh"
#include "backends/cuda/gfn2_d4.cuh"
#include "backends/cuda/gfn2_electric_field.cuh"
#include "backends/cuda/gfn2_energy_force_execution.cuh"
#include "backends/cuda/gfn2_external_point_charges.cuh"
#include "backends/cuda/gfn2_geometry.cuh"
#include "backends/cuda/gfn2_inference_publication.cuh"
#include "backends/cuda/gfn2_integral_tasks.hpp"
#include "backends/cuda/gfn2_preprocessing.cuh"
#include "backends/cuda/gfn2_public_result_bridge.cuh"
#include "backends/cuda/gfn2_scc_iteration_arena.cuh"
#include "backends/cuda/gfn2_scc_iteration_initialize.cuh"
#include "backends/cuda/gfn2_scc_iteration_reports.cuh"
#include "backends/cuda/gfn2_scc_loop.cuh"
#include "backends/cuda/gfn2_scc_setup_eigensolver.cuh"
#include "backends/cuda/gfn2_scc_setup_inputs.cuh"
#include "backends/cuda/gfn2_scc_setup_topology.hpp"
#include "backends/cuda/gfn2_terminal_classical_energy.cuh"
#include "data/parameters/d4.hpp"
#include "data/parameters/gfn1.hpp"
#include "data/parameters/gfn1_d3.hpp"
#include "model/gfn1/basis.hpp"
#include "model/gfn1/coordination.hpp"
#include "model/gfn1/d3.hpp"
#include "model/gfn1/es2.hpp"
#include "model/gfn1/es3.hpp"
#include "model/gfn1/external_point_charges.hpp"
#include "model/gfn1/h0.hpp"
#include "model/gfn1/halogen.hpp"
#include "model/gfn1/integrals.hpp"
#include "model/gfn1/mulliken.hpp"
#include "model/gfn1/repulsion.hpp"
#include "model/gfn1/scc_driver.hpp"
#include "model/gfn1/scc_mixer.hpp"
#include "model/gfn1/spin.hpp"
#include "model/gfn1/wavefunction.hpp"
#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/eigensolver.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/es3.hpp"
#include "model/gfn2/external_point_charges.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/lattice.hpp"
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/periodic_embedding.hpp"
#include "model/gfn2/repulsion.hpp"
#include "model/gfn2/scc_driver.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"
#include "runtime/backend.hpp"
#include "runtime/cuda_descriptor_validation.hpp"
#include "runtime/gfn2_cpu_execution.hpp"
#include "runtime/gfn2_cuda_execution.hpp"
#include "runtime/gfn2_cuda_topology_staging.hpp"
#include "runtime/nvidia_host_api.h"
#include "runtime/request.hpp"

namespace xtbloom::detail {
namespace {

using namespace xtbloom::detail::cuda;
using namespace xtbloom::detail::gfn2;

constexpr std::int32_t kDefaultMixerHistory = 8;
constexpr double kDefaultMixerDamping = 0.4;
constexpr std::uint64_t kInitialGeometryGeneration = 1u;
constexpr std::uint64_t kInitialStateGeneration = 1u;
constexpr std::size_t kArenaAlignment = 256u;
#ifdef XTBLOOM_CUDA_TEST_HOOKS
std::atomic<std::uint32_t> g_admission_alias_test_hook{
    static_cast<std::uint32_t>(Gfn2CudaAdmissionAliasTestHook::kNone)};
#endif
/* One committed physical superset serves every pair consumer.  D4 two-body
 * reaches 50 bohr; narrower roles re-evaluate positions with their own
 * inclusive 25/30-bohr predicates and never infer physical membership solely
 * from list presence. */
constexpr double kD4PairlistBuilderCutoffBohr = 50.0;

xtbloom_status_t configured_integral_task_schedule(const Gfn2IntegralTaskAccounting& accounting,
                                                   bool& compact, std::string& error) {
  const char* configured = std::getenv("XTBLOOM_CUDA_SHELL_PAIR_SCHEDULE");
  if (configured == nullptr) {
    compact = prefer_gfn2_compact_integral_tasks(accounting);
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (std::strcmp(configured, "compact") == 0) {
    compact = true;
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (std::strcmp(configured, "legacy") == 0) {
    compact = false;
    return XTBLOOM_STATUS_SUCCESS;
  }
  error = "XTBLOOM_CUDA_SHELL_PAIR_SCHEDULE must be exactly compact or legacy";
  return XTBLOOM_STATUS_INVALID_ARGUMENT;
}

#if defined(XTBLOOM_CUDA_TEST_HOOKS)
std::atomic<std::uint32_t> g_execution_test_fault{
    static_cast<std::uint32_t>(Gfn2CudaExecutionTestFault::kNone)};
std::atomic<std::uint64_t> g_native_lattice_allocation_faults{0u};
std::atomic<std::uint64_t> g_native_lattice_completion_faults{0u};
std::atomic<std::uint64_t> g_native_lattice_teardown_faults{0u};
std::atomic<std::uint64_t> g_quarantined_native_lattice_arenas{0u};
std::atomic<std::uint64_t> g_quarantined_native_lattice_bytes{0u};
std::atomic<std::uint64_t> g_request_graph_build_attempts{0u};
std::atomic<std::uint64_t> g_request_graph_build_successes{0u};

bool consume_execution_test_fault(Gfn2CudaExecutionTestFault fault) noexcept {
  std::uint32_t expected = static_cast<std::uint32_t>(fault);
  return g_execution_test_fault.compare_exchange_strong(
      expected, static_cast<std::uint32_t>(Gfn2CudaExecutionTestFault::kNone),
      std::memory_order_acq_rel, std::memory_order_relaxed);
}
#endif

Gfn2CudaSccStartMode public_scc_start_mode(const xtbloom_compute_options_t& options) noexcept {
  return options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE &&
                 options.scc_start_mode == XTBLOOM_SCC_START_WARM
             ? Gfn2CudaSccStartMode::kWarm
             : Gfn2CudaSccStartMode::kFresh;
}

/* ABI-v3 mixer and reproducibility controls form one complete suffix. A
 * caller that supplies only a prefix of the suffix receives the established
 * production defaults; no individual field becomes visible early. */
bool has_compute_options_v3(const xtbloom_compute_options_t& options) noexcept {
  return options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V3_SIZE;
}

xtbloom_scc_mixer_t public_scc_mixer(const xtbloom_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.scc_mixer : XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
}

std::int32_t public_scc_mixer_history(const xtbloom_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.scc_mixer_history : kDefaultMixerHistory;
}

double public_scc_mixer_damping(const xtbloom_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.scc_mixer_damping : kDefaultMixerDamping;
}

xtbloom_determinism_t public_determinism(const xtbloom_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.determinism : XTBLOOM_DETERMINISM_DEFAULT;
}

xtbloom_status_t validate_public_execution_policy(const xtbloom_compute_options_t& options,
                                                  std::string& error) {
  if (!has_compute_options_v3(options)) return XTBLOOM_STATUS_SUCCESS;
  if (options.scc_mixer != XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN) {
    error = "CUDA GFN2 setup supports only modified-Broyden SCC mixing";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (options.scc_mixer_history < 1 || options.scc_mixer_history > 64) {
    error = "CUDA GFN2 SCC mixer history must be between 1 and 64";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!std::isfinite(options.scc_mixer_damping) || options.scc_mixer_damping <= 0.0 ||
      options.scc_mixer_damping > 1.0) {
    error = "CUDA GFN2 SCC mixer damping must be finite and in (0, 1]";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (options.determinism != XTBLOOM_DETERMINISM_DEFAULT &&
      options.determinism != XTBLOOM_DETERMINISM_REPRODUCIBLE) {
    error = "CUDA GFN2 determinism policy is unknown";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (options.reserved_v3 != 0u) {
    error = "CUDA GFN2 compute-options reserved_v3 field must be zero";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

std::uintptr_t opaque_address(const void* pointer) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer);
}

template <typename T>
Gfn2SccSetupHostArray<T> setup_array(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), static_cast<std::int64_t>(values.size())};
}

template <typename T>
std::size_t vector_bytes(const std::vector<T>& values) noexcept {
  return values.capacity() * sizeof(T);
}

template <typename T>
Gfn2SccIterationHostArrayView<T> initialization_array(const T* values,
                                                      std::int64_t elements) noexcept {
  return {elements == 0 ? nullptr : values, elements};
}

bool checked_bytes(std::int64_t elements, std::size_t element_size, std::size_t& bytes) noexcept {
  if (elements < 0) {
    return false;
  }
  const auto count = static_cast<std::uint64_t>(elements);
  if (count > std::numeric_limits<std::size_t>::max() / element_size) {
    return false;
  }
  bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

/* Asynchronous calls cannot inspect device-resident interaction offsets on
 * the host without waiting behind earlier owner-stream work. A HOST payload
 * paired with device descriptors is therefore accepted only when its complete
 * declared view fits the released one-block-per-system staging arena. Keep
 * this metadata-only admission check before any fixed-topology comparison is
 * submitted so an unsupported asynchronous layout remains transactional and
 * nonblocking. */
xtbloom_status_t validate_nonblocking_interaction_staging(const xtbloom_batch_t& batch,
                                                          std::string& error) {
  if (batch.struct_size < XTBLOOM_BATCH_V3_SIZE || batch.total_interactions == 0 ||
      batch.interaction_descriptors.memory_space != XTBLOOM_MEMORY_CUDA_DEVICE ||
      batch.interaction_payload.memory_space != XTBLOOM_MEMORY_HOST) {
    return XTBLOOM_STATUS_SUCCESS;
  }

  std::size_t released_payload_capacity = 0u;
  if (!checked_bytes(batch.batch_size, 32u, released_payload_capacity)) {
    error = "CUDA released interaction payload capacity overflows size_t";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  constexpr std::size_t kPayloadAlignmentSlack = alignof(double) - 1u;
  if (released_payload_capacity >
          std::numeric_limits<std::size_t>::max() - kPayloadAlignmentSlack ||
      batch.interaction_payload.size_bytes > released_payload_capacity + kPayloadAlignmentSlack) {
    error =
        "asynchronous CUDA interaction staging requires a device descriptor/host payload view "
        "to fit the fixed released payload capacity";
    return XTBLOOM_STATUS_NOT_SUPPORTED;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_address_range(const void* pointer, std::size_t bytes, AddressRange& range) noexcept {
  if (pointer == nullptr || bytes == 0u) return false;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > UINTPTR_MAX - begin) return false;
  range = {begin, begin + bytes};
  return true;
}

bool address_ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

bool address_range_contains(const AddressRange& owner, const AddressRange& child) noexcept {
  return owner.begin <= child.begin && child.end <= owner.end;
}

bool checked_elements(std::int64_t elements, std::int64_t factor, std::int64_t& product) noexcept {
  if (elements < 0 || factor < 0 ||
      (factor != 0 && elements > std::numeric_limits<std::int64_t>::max() / factor)) {
    return false;
  }
  product = elements * factor;
  return true;
}

bool checked_triangle(std::int64_t atoms, std::int64_t& pairs) noexcept {
  if (atoms < 0) {
    return false;
  }
  if (atoms <= 1) {
    pairs = 0;
    return true;
  }
  const std::int64_t lower = atoms - 1;
  if ((atoms & 1) == 0) {
    return checked_elements(atoms / 2, lower, pairs);
  }
  return checked_elements(atoms, lower / 2, pairs);
}

bool native_lattice_active(const xtbloom_batch_t& batch) noexcept {
  return batch.struct_size >= XTBLOOM_BATCH_V4_SIZE &&
         (batch.cell_matrices.data != nullptr || batch.cell_matrices.size_bytes != 0u ||
          batch.periodic_axes.data != nullptr || batch.periodic_axes.size_bytes != 0u);
}

/*
 * CUDA backend objects are linked directly into white-box test executables,
 * while validation.cpp remains hidden inside libxtbloom. Keep this staged HOST
 * content check local instead of creating a link-time dependency on internal
 * shared-library symbols. The predicate itself is shared with the CPU ABI and
 * lattice builder through model/gfn2/lattice.cpp.
 */
xtbloom_status_t validate_host_native_lattice_request(const xtbloom_batch_t& batch,
                                                      std::string& error,
                                                      bool allow_native_periodic = false) {
  if (!native_lattice_active(batch)) return XTBLOOM_STATUS_SUCCESS;

  std::int64_t cell_elements = 0;
  std::size_t cell_bytes = 0u;
  std::size_t axes_bytes = 0u;
  if (batch.batch_size <= 0 || !checked_elements(batch.batch_size, 9, cell_elements) ||
      !checked_bytes(cell_elements, sizeof(double), cell_bytes) ||
      !checked_bytes(batch.batch_size, sizeof(std::int32_t), axes_bytes)) {
    error = "CUDA native-cell extent overflows the host address space";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (batch.cell_matrices.memory_space != XTBLOOM_MEMORY_HOST ||
      batch.periodic_axes.memory_space != XTBLOOM_MEMORY_HOST ||
      batch.cell_matrices.data == nullptr || batch.periodic_axes.data == nullptr ||
      batch.cell_matrices.size_bytes < cell_bytes || batch.periodic_axes.size_bytes < axes_bytes) {
    error = "CUDA native-cell semantic validation requires complete HOST staging buffers";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  bool periodic = false;
  for (std::int64_t system = 0; system < batch.batch_size; ++system) {
    std::int32_t mask = 0;
    std::array<double, 9> cell{};
    std::memcpy(&mask,
                static_cast<const std::byte*>(batch.periodic_axes.data) +
                    static_cast<std::size_t>(system) * sizeof(mask),
                sizeof(mask));
    std::memcpy(cell.data(),
                static_cast<const std::byte*>(batch.cell_matrices.data) +
                    static_cast<std::size_t>(system) * sizeof(cell),
                sizeof(cell));

    if (mask == XTBLOOM_PERIODIC_AXES_NONE) {
      if (!std::all_of(cell.begin(), cell.end(), [](double value) { return value == 0.0; })) {
        error = "a nonperiodic batch item must use an all-zero cell matrix";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      continue;
    }
    if ((mask & ~XTBLOOM_PERIODIC_AXES_XYZ) != 0) {
      error = "periodic_axes contains unknown mask bits";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (mask != XTBLOOM_PERIODIC_AXES_XYZ) {
      error = "one- and two-dimensional periodic axes are reserved but not supported";
      return XTBLOOM_STATUS_NOT_SUPPORTED;
    }
    if (!gfn2::valid_lattice_cell_3d(cell.data())) {
      error = "a periodic cell must be finite, right-handed, and nonsingular";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    periodic = true;
  }
  if (periodic) {
    /* Native PBC and the caller-owned embedding/interaction inputs are not
     * yet compositionally proven.  Keep this check after cell staging and
     * semantic validation so mixed host/device descriptors are handled with
     * the same precise refusal as an all-host request. */
    if (batch.total_point_charges != 0) {
      error = "native periodic PBC cannot be combined with explicit point charges";
      return XTBLOOM_STATUS_NOT_SUPPORTED;
    }
    if (batch.atomic_potential_shifts.data != nullptr ||
        batch.atomic_potential_shifts.size_bytes != 0u) {
      error = "native periodic PBC cannot be combined with atomic_potential_shifts";
      return XTBLOOM_STATUS_NOT_SUPPORTED;
    }
    if (batch.total_charge_response_elements != 0 ||
        batch.charge_response_offsets.data != nullptr ||
        batch.charge_response_offsets.size_bytes != 0u ||
        batch.charge_response_matrix.data != nullptr ||
        batch.charge_response_matrix.size_bytes != 0u) {
      error = "native periodic PBC cannot be combined with charge_response_matrix or b+Aq";
      return XTBLOOM_STATUS_NOT_SUPPORTED;
    }
    if (batch.struct_size >= XTBLOOM_BATCH_V3_SIZE && batch.total_interactions != 0) {
      error = "native periodic PBC cannot be combined with interaction attachments";
      return XTBLOOM_STATUS_NOT_SUPPORTED;
    }
    if (!allow_native_periodic) {
      error =
          "native lattice/PBC descriptors are valid but native periodic execution is not "
          "implemented yet";
      return XTBLOOM_STATUS_NOT_IMPLEMENTED;
    }
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

template <typename T>
xtbloom_status_t copy_host_buffer(const char* name, const xtbloom_const_buffer_t& buffer,
                                  std::int64_t elements, std::vector<T>& output, std::string& error,
                                  bool allow_absent = false) {
  std::size_t required = 0u;
  if (!checked_bytes(elements, sizeof(T), required)) {
    error = std::string(name) + " extent overflows size_t";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (elements == 0 && buffer.data == nullptr && allow_absent) {
    output.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (buffer.memory_space != XTBLOOM_MEMORY_HOST || buffer.reserved != 0u ||
      (required != 0u && buffer.data == nullptr) || buffer.size_bytes < required) {
    error = std::string(name) + " is not a sufficiently large host buffer";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  output.resize(static_cast<std::size_t>(elements));
  if (required != 0u) {
    std::memcpy(output.data(), buffer.data, required);
  }
  return XTBLOOM_STATUS_SUCCESS;
}

std::string cuda_error_message(const char* operation, cudaError_t status);
inline std::string cuda_error_message(const std::string& operation, cudaError_t status) {
  return cuda_error_message(operation.c_str(), status);
}

/*
 * Native periodic CUDA execution currently uses the CPU periodic evaluator as
 * its scientific reference bridge.  This helper owns the only host ingress
 * used by that bridge: device buffers are copied synchronously after the
 * owner stream has been fenced, while host buffers are copied with memcpy so
 * under-aligned ABI byte views remain valid.  The caller has already run the
 * complete public descriptor validator, but retaining the memory-space and
 * extent checks here keeps the internal bridge fail-closed when it is called
 * from a white-box runtime test.
 */
template <typename T>
xtbloom_status_t copy_cuda_or_host_buffer(const char* name, const xtbloom_const_buffer_t& buffer,
                                          std::int64_t elements, cudaStream_t stream,
                                          std::vector<T>& output, std::string& error,
                                          bool allow_absent = false) {
  std::size_t required = 0u;
  if (!checked_bytes(elements, sizeof(T), required)) {
    error = std::string(name) + " extent overflows size_t";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (elements == 0 && buffer.data == nullptr && allow_absent) {
    output.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (buffer.reserved != 0u || buffer.size_bytes < required ||
      (required != 0u && buffer.data == nullptr) ||
      (buffer.memory_space != XTBLOOM_MEMORY_HOST &&
       buffer.memory_space != XTBLOOM_MEMORY_CUDA_DEVICE)) {
    error = std::string(name) + " is not a sufficiently sized host or CUDA buffer";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  output.resize(static_cast<std::size_t>(elements));
  if (required == 0u) return XTBLOOM_STATUS_SUCCESS;
  if (buffer.memory_space == XTBLOOM_MEMORY_HOST) {
    std::memcpy(output.data(), buffer.data, required);
    return XTBLOOM_STATUS_SUCCESS;
  }
  const cudaError_t status =
      cudaMemcpy(output.data(), buffer.data, required, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    error = cuda_error_message(std::string("CUDA ") + name + " download", status);
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  (void)stream;
  return XTBLOOM_STATUS_SUCCESS;
}

template <typename T>
xtbloom_status_t copy_host_result_buffer(const char* name, const std::vector<T>& source,
                                         xtbloom_buffer_t& destination, cudaStream_t stream,
                                         std::string& error) {
  const std::size_t bytes = source.size() * sizeof(T);
  if (bytes == 0u) return XTBLOOM_STATUS_SUCCESS;
  if (destination.data == nullptr || destination.size_bytes < bytes || destination.reserved != 0u) {
    error = std::string(name) + " output is not a sufficiently sized buffer";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (destination.memory_space == XTBLOOM_MEMORY_HOST) {
    std::memcpy(destination.data, source.data(), bytes);
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (destination.memory_space != XTBLOOM_MEMORY_CUDA_DEVICE) {
    error = std::string(name) + " output has an unknown memory space";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const cudaError_t status =
      cudaMemcpyAsync(destination.data, source.data(), bytes, cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) {
    error = cuda_error_message(std::string("CUDA ") + name + " upload", status);
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

template <typename T>
xtbloom_const_buffer_t host_const_buffer(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

template <typename T>
xtbloom_buffer_t host_result_buffer(std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0u};
}

struct NativePeriodicHostBridgeStorage {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> point_offsets;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> potential_shifts;
  std::vector<std::int64_t> response_offsets;
  std::vector<double> response_matrix;
  std::vector<std::uint8_t> interaction_descriptors;
  std::vector<std::uint8_t> interaction_payload;
  std::vector<double> cell_matrices;
  std::vector<std::int32_t> periodic_axes;

  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<xtbloom_status_t> system_statuses;
  std::vector<double> dipole_moments;
  std::vector<double> strain_derivatives;

  xtbloom_batch_t batch{};
  xtbloom_batch_result_t result{};

  xtbloom_status_t stage_batch(const xtbloom_batch_t& source, cudaStream_t stream,
                               std::string& error) {
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t descriptor_bytes = 0;
    if (!checked_elements(source.total_atoms, 3, coordinates) ||
        !checked_elements(source.total_point_charges, 3, point_coordinates) ||
        (source.struct_size >= XTBLOOM_BATCH_V3_SIZE && source.total_interactions < 0) ||
        (source.struct_size >= XTBLOOM_BATCH_V3_SIZE &&
         !checked_elements(source.total_interactions,
                           static_cast<std::int64_t>(sizeof(xtbloom_interaction_t)),
                           descriptor_bytes))) {
      error = "native periodic CUDA bridge input extent overflows int64_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    xtbloom_status_t status = copy_cuda_or_host_buffer(
        "atom_offsets", source.atom_offsets, source.batch_size + 1, stream, atom_offsets, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_cuda_or_host_buffer("atomic_numbers", source.atomic_numbers, source.total_atoms,
                                      stream, atomic_numbers, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_cuda_or_host_buffer("positions", source.positions, coordinates, stream, positions,
                                      error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_cuda_or_host_buffer("molecular_charges", source.molecular_charges,
                                      source.batch_size, stream, molecular_charges, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_cuda_or_host_buffer("unpaired_electrons", source.unpaired_electrons,
                                      source.batch_size, stream, unpaired_electrons, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const bool spin_present =
        source.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
        (source.spin_channels.data != nullptr || source.spin_channels.size_bytes != 0u);
    if (spin_present) {
      status = copy_cuda_or_host_buffer("spin_channels", source.spin_channels, source.batch_size,
                                        stream, spin_channels, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    } else {
      spin_channels.assign(static_cast<std::size_t>(source.batch_size), 1);
    }

    if (source.total_point_charges != 0 || source.point_charge_offsets.data != nullptr ||
        source.point_charge_offsets.size_bytes != 0u) {
      status = copy_cuda_or_host_buffer("point_charge_offsets", source.point_charge_offsets,
                                        source.batch_size + 1, stream, point_offsets, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    } else {
      point_offsets.assign(static_cast<std::size_t>(source.batch_size + 1), 0);
    }
    status = copy_cuda_or_host_buffer("point_charge_positions", source.point_charge_positions,
                                      point_coordinates, stream, point_positions, error, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        copy_cuda_or_host_buffer("point_charge_values", source.point_charge_values,
                                 source.total_point_charges, stream, point_values, error, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        copy_cuda_or_host_buffer("point_charge_gammas", source.point_charge_gammas,
                                 source.total_point_charges, stream, point_gammas, error, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    const bool shifts_present = source.atomic_potential_shifts.data != nullptr ||
                                source.atomic_potential_shifts.size_bytes != 0u;
    if (shifts_present) {
      status = copy_cuda_or_host_buffer("atomic_potential_shifts", source.atomic_potential_shifts,
                                        source.total_atoms, stream, potential_shifts, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }
    const bool response_present = source.total_charge_response_elements != 0 ||
                                  source.charge_response_offsets.data != nullptr ||
                                  source.charge_response_offsets.size_bytes != 0u ||
                                  source.charge_response_matrix.data != nullptr ||
                                  source.charge_response_matrix.size_bytes != 0u;
    if (response_present) {
      status = copy_cuda_or_host_buffer("charge_response_offsets", source.charge_response_offsets,
                                        source.batch_size + 1, stream, response_offsets, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      status = copy_cuda_or_host_buffer("charge_response_matrix", source.charge_response_matrix,
                                        source.total_charge_response_elements, stream,
                                        response_matrix, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }

    if (source.struct_size >= XTBLOOM_BATCH_V3_SIZE && source.total_interactions != 0) {
      status = copy_cuda_or_host_buffer("interaction_descriptors", source.interaction_descriptors,
                                        descriptor_bytes, stream, interaction_descriptors, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      status =
          copy_cuda_or_host_buffer("interaction_payload", source.interaction_payload,
                                   static_cast<std::int64_t>(source.interaction_payload.size_bytes),
                                   stream, interaction_payload, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }

    const bool lattice_present =
        source.struct_size >= XTBLOOM_BATCH_V4_SIZE &&
        (source.cell_matrices.data != nullptr || source.cell_matrices.size_bytes != 0u ||
         source.periodic_axes.data != nullptr || source.periodic_axes.size_bytes != 0u);
    if (lattice_present) {
      status = copy_cuda_or_host_buffer("cell_matrices", source.cell_matrices,
                                        source.batch_size * 9, stream, cell_matrices, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      status = copy_cuda_or_host_buffer("periodic_axes", source.periodic_axes, source.batch_size,
                                        stream, periodic_axes, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }

    std::memcpy(&batch, &source, std::min<std::size_t>(source.struct_size, sizeof(batch)));
    batch.atom_offsets = host_const_buffer(atom_offsets);
    batch.atomic_numbers = host_const_buffer(atomic_numbers);
    batch.positions = host_const_buffer(positions);
    batch.molecular_charges = host_const_buffer(molecular_charges);
    batch.unpaired_electrons = host_const_buffer(unpaired_electrons);
    if (source.struct_size >= XTBLOOM_BATCH_V2_SIZE) {
      batch.spin_channels = host_const_buffer(spin_channels);
    }
    batch.point_charge_offsets = host_const_buffer(point_offsets);
    batch.point_charge_positions = host_const_buffer(point_positions);
    batch.point_charge_values = host_const_buffer(point_values);
    batch.point_charge_gammas = host_const_buffer(point_gammas);
    batch.atomic_potential_shifts = host_const_buffer(potential_shifts);
    batch.charge_response_offsets = host_const_buffer(response_offsets);
    batch.charge_response_matrix = host_const_buffer(response_matrix);
    if (source.struct_size >= XTBLOOM_BATCH_V3_SIZE) {
      batch.interaction_descriptors = host_const_buffer(interaction_descriptors);
      batch.interaction_payload = host_const_buffer(interaction_payload);
    }
    if (source.struct_size >= XTBLOOM_BATCH_V4_SIZE) {
      batch.cell_matrices = host_const_buffer(cell_matrices);
      batch.periodic_axes = host_const_buffer(periodic_axes);
    }
    return XTBLOOM_STATUS_SUCCESS;
  }

  void bind_result(const xtbloom_batch_result_t& source, const xtbloom_compute_options_t& options) {
    result = {};
    result.struct_size = source.struct_size;
    result.api_version = source.api_version;
    result.reserved = source.reserved;
    const std::size_t batch_size = static_cast<std::size_t>(batch.batch_size);
    const std::size_t atoms = static_cast<std::size_t>(batch.total_atoms);
    const std::size_t points = static_cast<std::size_t>(batch.total_point_charges);
    const auto has = [&](std::uint32_t flag) { return (options.flags & flag) != 0u; };
    if (has(XTBLOOM_COMPUTE_ENERGY)) energies.resize(batch_size);
    if (has(XTBLOOM_COMPUTE_FORCES)) forces.resize(atoms * 3u);
    if (has(XTBLOOM_COMPUTE_ATOMIC_CHARGES)) atomic_charges.resize(atoms);
    if (has(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) point_forces.resize(points * 3u);
    iterations.resize(batch_size);
    converged.resize(batch_size);
    system_statuses.resize(batch_size);
    if (has(XTBLOOM_COMPUTE_DIPOLE_MOMENTS) && source.struct_size >= XTBLOOM_BATCH_RESULT_V2_SIZE) {
      dipole_moments.resize(batch_size * 3u);
    }
    if (has(XTBLOOM_COMPUTE_STRAIN_DERIVATIVES) &&
        source.struct_size >= XTBLOOM_BATCH_RESULT_V3_SIZE) {
      strain_derivatives.resize(batch_size * 9u);
    }
    result.energies = host_result_buffer(energies);
    result.forces = host_result_buffer(forces);
    result.atomic_charges = host_result_buffer(atomic_charges);
    result.point_charge_forces = host_result_buffer(point_forces);
    result.scc_iterations = host_result_buffer(iterations);
    result.scc_converged = host_result_buffer(converged);
    result.per_system_status = host_result_buffer(system_statuses);
    if (source.struct_size >= XTBLOOM_BATCH_RESULT_V2_SIZE) {
      result.dipole_moments = host_result_buffer(dipole_moments);
    }
    if (source.struct_size >= XTBLOOM_BATCH_RESULT_V3_SIZE) {
      result.strain_derivatives = host_result_buffer(strain_derivatives);
    }
  }

  xtbloom_status_t publish_result(xtbloom_batch_result_t& destination, cudaStream_t stream,
                                  std::string& error) {
    xtbloom_status_t status =
        copy_host_result_buffer("energies", energies, destination.energies, stream, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_host_result_buffer("forces", forces, destination.forces, stream, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_host_result_buffer("atomic_charges", atomic_charges, destination.atomic_charges,
                                     stream, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_host_result_buffer("point_charge_forces", point_forces,
                                     destination.point_charge_forces, stream, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_host_result_buffer("scc_iterations", iterations, destination.scc_iterations,
                                     stream, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_host_result_buffer("scc_converged", converged, destination.scc_converged, stream,
                                     error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_host_result_buffer("per_system_status", system_statuses,
                                     destination.per_system_status, stream, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    if (result.struct_size >= XTBLOOM_BATCH_RESULT_V2_SIZE) {
      status = copy_host_result_buffer("dipole_moments", dipole_moments, destination.dipole_moments,
                                       stream, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }
    if (result.struct_size >= XTBLOOM_BATCH_RESULT_V3_SIZE) {
      status = copy_host_result_buffer("strain_derivatives", strain_derivatives,
                                       destination.strain_derivatives, stream, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }
    return cudaStreamSynchronize(stream) == cudaSuccess ? XTBLOOM_STATUS_SUCCESS
                                                        : XTBLOOM_STATUS_INTERNAL_ERROR;
  }
};

std::uint64_t hash_mix(std::uint64_t value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31u);
}

void hash_append(std::uint64_t& hash, std::uint64_t value) noexcept {
  hash ^= hash_mix(value + 0x9e3779b97f4a7c15ULL + (hash << 6u) + (hash >> 2u));
}

std::uint64_t double_bits(double value) noexcept {
  std::uint64_t bits = 0u;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <typename T>
void hash_append_vector(std::uint64_t& hash, const std::vector<T>& values) noexcept {
  hash_append(hash, values.size());
  for (const T value : values) {
    if constexpr (std::is_same_v<T, double>) {
      hash_append(hash, double_bits(value));
    } else {
      hash_append(hash, static_cast<std::uint64_t>(value));
    }
  }
}

class DeviceArena {
 public:
  DeviceArena() = default;
  ~DeviceArena() { reset(); }
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;
  DeviceArena(DeviceArena&&) = delete;
  DeviceArena& operator=(DeviceArena&&) = delete;

  cudaError_t allocate(std::size_t bytes) noexcept {
    reset();
    bytes_ = bytes;
    return bytes == 0u ? cudaSuccess : cudaMalloc(&pointer_, bytes);
  }

  void reset() noexcept {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
    pointer_ = nullptr;
    bytes_ = 0u;
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

class PinnedArena {
 public:
  PinnedArena() = default;
  ~PinnedArena() { reset(); }
  PinnedArena(const PinnedArena&) = delete;
  PinnedArena& operator=(const PinnedArena&) = delete;
  PinnedArena(PinnedArena&&) = delete;
  PinnedArena& operator=(PinnedArena&&) = delete;

  cudaError_t allocate(std::size_t bytes, bool inject_allocation_failure = false) noexcept {
    if (bytes == 0u) {
      reset();
      return cudaSuccess;
    }

    /* Preserve the old image until a complete replacement exists. In
     * particular, a failed growth must not publish a nonzero capacity beside
     * a null pointer, because callers use capacity to decide whether the
     * retained image is reusable. */
    void* replacement = nullptr;
    const cudaError_t status =
        inject_allocation_failure ? cudaErrorMemoryAllocation : cudaMallocHost(&replacement, bytes);
    if (status != cudaSuccess) return status;

    void* const previous = pointer_;
    pointer_ = replacement;
    bytes_ = bytes;
    if (previous != nullptr) (void)cudaFreeHost(previous);
    return cudaSuccess;
  }

  void reset() noexcept {
    if (pointer_ != nullptr) {
      (void)cudaFreeHost(pointer_);
    }
    pointer_ = nullptr;
    bytes_ = 0u;
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

  /* Relinquish an image that cannot be proved idle. CUDA may still reference
   * it, so the caller deliberately leaves it allocated instead of risking a
   * use-after-free in a noexcept teardown path. */
  void* release_without_free() noexcept {
    void* const released = pointer_;
    pointer_ = nullptr;
    bytes_ = 0u;
    return released;
  }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

class CudaEvent {
 public:
  CudaEvent() = default;
  ~CudaEvent() { reset(); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&&) = delete;
  CudaEvent& operator=(CudaEvent&&) = delete;

  cudaError_t create(unsigned int flags) noexcept {
    reset();
    return cudaEventCreateWithFlags(&event_, flags);
  }

  void reset() noexcept {
    if (event_ != nullptr) {
      (void)cudaEventDestroy(event_);
    }
    event_ = nullptr;
  }

  cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

class CudaStream {
 public:
  CudaStream() = default;
  ~CudaStream() { reset(); }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&&) = delete;
  CudaStream& operator=(CudaStream&&) = delete;

  cudaError_t create(unsigned int flags) noexcept {
    reset();
    return cudaStreamCreateWithFlags(&stream_, flags);
  }

  void reset() noexcept {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
    stream_ = nullptr;
  }

  cudaStream_t get() const noexcept { return stream_; }
  bool valid() const noexcept { return stream_ != nullptr; }

 private:
  cudaStream_t stream_ = nullptr;
};

bool align_up(std::size_t value, std::size_t alignment, std::size_t& aligned) noexcept {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return false;
  }
  const std::size_t mask = alignment - 1u;
  if (value > std::numeric_limits<std::size_t>::max() - mask) {
    return false;
  }
  aligned = (value + mask) & ~mask;
  return true;
}

class ArenaLayout {
 public:
  template <typename T, typename Count>
  std::size_t append(Count elements) {
    static_assert(std::is_integral_v<Count>, "arena element counts must be integral");
    if constexpr (std::is_signed_v<Count>) {
      if (elements < 0) {
        valid_ = false;
        return 0u;
      }
    }
    using UnsignedCount = std::make_unsigned_t<Count>;
    const auto count = static_cast<UnsignedCount>(elements);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      valid_ = false;
      return 0u;
    }
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(T);
    const std::size_t alignment = std::max<std::size_t>(alignof(T), 8u);
    std::size_t offset = 0u;
    if (!align_up(bytes_, alignment, offset)) {
      valid_ = false;
      return 0u;
    }
    if (offset > std::numeric_limits<std::size_t>::max() - bytes) {
      valid_ = false;
      return 0u;
    }
    bytes_ = offset + bytes;
    return offset;
  }

  bool valid() const noexcept {
    std::size_t ignored = 0u;
    return valid_ && align_up(bytes_, kArenaAlignment, ignored);
  }
  std::size_t bytes() const noexcept {
    std::size_t aligned = 0u;
    return valid_ && align_up(bytes_, kArenaAlignment, aligned) ? aligned : 0u;
  }

 private:
  std::size_t bytes_ = 0u;
  bool valid_ = true;
};

struct InteractionStagingLayout {
  std::size_t descriptor_offset = 0u;
  std::size_t payload_offset = 0u;
  std::size_t descriptor_snapshot_offset = 0u;
  std::size_t descriptor_capacity_bytes = 0u;
  std::size_t payload_capacity_bytes = 0u;
  std::size_t arena_bytes = 0u;
  bool valid = false;
};

InteractionStagingLayout make_interaction_staging_layout(
    std::size_t descriptor_capacity_bytes, std::size_t payload_capacity_bytes) noexcept {
  ArenaLayout layout;
  InteractionStagingLayout result{};
  constexpr std::size_t kPayloadAlignmentSlack = alignof(double) - 1u;
  if (payload_capacity_bytes >
      std::numeric_limits<std::size_t>::max() - 2u * kPayloadAlignmentSlack) {
    return result;
  }
  result.descriptor_offset = layout.append<std::byte>(descriptor_capacity_bytes);
  /* Reverse-mixed staging preserves the caller payload base modulo double
   * alignment. Reserve the maximum displacement without increasing the
   * released one-block-per-system capacity reported to callers. */
  result.payload_offset =
      layout.append<std::byte>(payload_capacity_bytes + 2u * kPayloadAlignmentSlack);
  /* Device-descriptor/host-payload submissions need a bounded D2H snapshot
   * before the caller's host payload may be released.  Keep that readback
   * image separate from the compact descriptor image built for the H2D. */
  result.descriptor_snapshot_offset = layout.append<std::byte>(descriptor_capacity_bytes);
  result.descriptor_capacity_bytes = descriptor_capacity_bytes;
  result.payload_capacity_bytes = payload_capacity_bytes;
  result.arena_bytes = layout.bytes();
  result.valid = layout.valid();
  return result;
}

template <typename T>
T* arena_pointer(void* arena, std::size_t offset) noexcept {
  return reinterpret_cast<T*>(static_cast<std::byte*>(arena) + offset);
}

template <typename T>
T* arena_pointer_if(void* arena, std::size_t offset, std::int64_t elements) noexcept {
  return elements == 0 ? nullptr : arena_pointer<T>(arena, offset);
}

/* Stable numerical leaves and transaction metadata passed by value to the
 * runtime-owned refresh kernels. CUDA types stay confined to this translation
 * unit; the public internal boundary above remains usable by a future HIP
 * implementation. */
struct NumericalRefreshDeviceSources {
  const double* positions = nullptr;
  const double* point_positions = nullptr;
  const double* point_values = nullptr;
  const double* point_gammas = nullptr;
  const double* periodic_shifts = nullptr;
  const double* periodic_response = nullptr;
  const xtbloom_interaction_t* interaction_descriptors = nullptr;
  const std::byte* interaction_payload = nullptr;
  /* Range/alignment validation always refers to the caller's declared
   * payload view.  When a HOST payload is compacted into the fixed staging
   * arena, field bytes are instead addressed by system_index below. */
  std::uintptr_t interaction_payload_address = 0u;
  std::uint64_t interaction_payload_bytes = 0u;
  std::int64_t total_interactions = 0;
  std::uint32_t prevalidated_interaction_error = kGfn2RequestErrorNone;
  std::uint8_t interaction_payload_compacted_by_system = 0u;
  /* A HOST payload paired with device descriptors is snapshotted at identical
   * offsets when it fits the released fixed-capacity arena.  Device semantic
   * validation still uses the caller view's address for alignment checks, but
   * reads bytes from this plan-owned device image after enqueue returns. */
  std::uint8_t interaction_payload_staged_at_offsets = 0u;
  std::uint8_t prevalidated_reserved_interaction = 0u;
};

/* The start-mode scalar is written by a kernel argument so steady-state
 * admission neither transfers host bytes nor retains a caller-owned pointer. */
__global__ void stage_request_start_mode_kernel(std::uint32_t start_mode,
                                                std::uint32_t* staged_start_mode) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *staged_start_mode = start_mode;
  }
}

/*
 * A request admitted past all known semantic gates consumes the preceding
 * public checkpoint. WARM moves the token into request scratch so the later
 * reset can validate it after numerical refresh. FRESH clears the old token
 * but publishes no scratch checkpoint. Classified semantic rejections leave
 * the checkpoint intact; unknown/internal request codes consume it safely.
 */
struct RequestAdmissionDeviceBinding {
  std::int64_t batch_size = 0;
  const std::uint32_t* request_error = nullptr;
  const std::uint32_t* start_mode = nullptr;
  std::uint64_t* warm_checkpoint_generations = nullptr;
  std::uint32_t* warm_checkpoint_batch_ready = nullptr;
  std::uint64_t* admitted_warm_checkpoint_generations = nullptr;
  cudaGraphConditionalHandle execution_condition = 0u;
  cudaGraphConditionalHandle start_mode_condition = 0u;
};

static_assert(std::is_trivially_copyable_v<RequestAdmissionDeviceBinding>);

__global__ void consume_request_checkpoint_kernel(RequestAdmissionDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= binding.batch_size) return;
  const std::uint32_t request_error = *binding.request_error;
  if (request_error != kGfn2RequestErrorNone &&
      request_error <= kGfn2RequestErrorTopologyMismatch) {
    /* A fully classified semantic rejection never starts numerical execution,
     * so the preceding public checkpoint remains reusable. Unknown codes are
     * treated as internal failures below and consume the token defensively. */
    binding.admitted_warm_checkpoint_generations[system] = 0u;
    return;
  }
  const std::uint32_t start_mode = *binding.start_mode;
  const bool warm = start_mode == 1u;
  const std::uint64_t checkpoint = atomicExch(
      reinterpret_cast<unsigned long long*>(binding.warm_checkpoint_generations + system), 0ULL);
  binding.admitted_warm_checkpoint_generations[system] = warm ? checkpoint : 0u;
}

__global__ void set_request_execution_conditions_kernel(RequestAdmissionDeviceBinding binding) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const bool warm = *binding.start_mode == 1u;
  *binding.warm_checkpoint_batch_ready = 0u;
  cudaGraphSetConditional(binding.start_mode_condition, warm ? 1u : 0u);
  cudaGraphSetConditional(binding.execution_condition, *binding.request_error == 0u ? 1u : 0u);
}

struct NumericalRefreshDeviceBinding {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_matrices = 0;
  std::int64_t total_point_charges = 0;
  std::int64_t total_response_elements = 0;
  std::int64_t geometry_pair_elements = 0;
  std::int64_t es2_elements = 0;
  std::int64_t aes2_elements = 0;
  std::uint8_t multipoles_enabled = 1u;
  std::uint8_t aes2_enabled = 1u;
  std::uint8_t d4_enabled = 0u;
  std::uint8_t point_enabled = 0u;
  std::uint8_t periodic_enabled = 0u;
  std::uint64_t plan_token = 0u;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* shell_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* geometry_pair_offsets = nullptr;
  const std::int64_t* es2_offsets = nullptr;
  const std::int64_t* point_offsets = nullptr;
  const std::int64_t* response_offsets = nullptr;
  const std::uint8_t* requested = nullptr;
  const std::uint8_t* preprocessing_published = nullptr;
  std::uint8_t* eligible = nullptr;
  /* Canonical eigensolver/SCC activity leaf snapshotted by overlap refactor. */
  std::uint8_t* factor_active = nullptr;
  std::uint64_t* committed_generations = nullptr;
  /* The immediately preceding successfully committed numerical epoch. Warm
   * reset uses this edge instead of rewriting checkpoint generations, so two
   * refreshes without an intervening inference cannot chain a stale state. */
  std::uint64_t* refresh_predecessor_generations = nullptr;
  /* Bound after the inference arena is built. A failed refresh must revoke
   * both the predecessor edge and its consumable warm checkpoint. */
  std::uint64_t* warm_checkpoint_generations = nullptr;

  double* candidate_positions = nullptr;
  double* candidate_point_positions = nullptr;
  double* candidate_point_values = nullptr;
  double* candidate_point_gammas = nullptr;
  double* candidate_periodic_shifts = nullptr;
  double* candidate_periodic_response = nullptr;

  double* committed_positions = nullptr;
  double* committed_point_positions = nullptr;
  double* committed_point_values = nullptr;
  double* committed_point_gammas = nullptr;
  double* committed_periodic_shifts = nullptr;
  double* committed_periodic_response = nullptr;

  /* Electric-field attachments are dense normalized numerical state. Exact
   * presence is retained separately so absent and attached-zero remain
   * distinct under strict WARM. */
  std::uint8_t* candidate_field_attached = nullptr;
  double* candidate_field_vectors = nullptr;
  double* candidate_field_atomic_potentials = nullptr;
  double* candidate_field_dipole_potentials = nullptr;
  std::uint8_t* committed_field_attached = nullptr;
  double* committed_field_vectors = nullptr;
  double* committed_field_atomic_potentials = nullptr;
  double* committed_field_dipole_potentials = nullptr;
  std::uint8_t* warm_checkpoint_field_attached = nullptr;
  double* warm_checkpoint_field_vectors = nullptr;
  std::int64_t total_interactions = 0;
  const xtbloom_interaction_t* interaction_descriptors = nullptr;
  const std::byte* interaction_payload = nullptr;
  std::uint64_t interaction_payload_bytes = 0u;
  std::uint32_t* interaction_request_error = nullptr;
  std::uint32_t* request_error = nullptr;
  std::uint32_t* field_system_errors = nullptr;
  std::uint32_t* field_plan_error = nullptr;

  const double* candidate_geometry_pairs = nullptr;
  const double* candidate_coordination = nullptr;
  const double* candidate_overlap = nullptr;
  const double* candidate_dipole = nullptr;
  const double* candidate_quadrupole = nullptr;
  const double* candidate_h0 = nullptr;
  const double* candidate_es2 = nullptr;
  const double* candidate_aes2 = nullptr;

  double* public_geometry_pairs = nullptr;
  double* public_coordination = nullptr;
  double* public_overlap = nullptr;
  double* public_dipole = nullptr;
  double* public_quadrupole = nullptr;
  double* public_h0 = nullptr;
  double* public_es2 = nullptr;
  double* public_aes2 = nullptr;

  const double* candidate_d4_coordination = nullptr;
  double* public_d4_coordination = nullptr;
  const double* candidate_point_shell = nullptr;
  double* public_point_shell = nullptr;

  const std::uint32_t* preprocessing_plan_error = nullptr;
  const std::uint32_t* d4_system_errors = nullptr;
  const std::uint32_t* d4_device_error = nullptr;
  const std::uint32_t* point_system_errors = nullptr;
  const std::uint32_t* point_plan_error = nullptr;
  std::uint32_t* periodic_system_errors = nullptr;
  std::uint32_t* periodic_plan_error = nullptr;
  const std::uint64_t* factor_generations = nullptr;
  const std::uint32_t* factor_statuses = nullptr;
  const std::uint64_t* geometry_epoch = nullptr;
};

static_assert(std::is_trivially_copyable_v<NumericalRefreshDeviceSources>);
static_assert(std::is_trivially_copyable_v<NumericalRefreshDeviceBinding>);

__global__ void stage_gfn2_numerical_inputs_kernel(NumericalRefreshDeviceBinding binding,
                                                   NumericalRefreshDeviceSources sources) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  const bool use_source = binding.requested[system] == 1u;

  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  for (std::int64_t index = atom_begin * 3 + threadIdx.x; index < atom_end * 3;
       index += blockDim.x) {
    binding.candidate_positions[index] =
        use_source ? sources.positions[index] : binding.committed_positions[index];
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    if (binding.periodic_enabled != 0u) {
      binding.candidate_periodic_shifts[atom] =
          use_source ? sources.periodic_shifts[atom] : binding.committed_periodic_shifts[atom];
    }
  }

  if (binding.point_enabled != 0u) {
    const std::int64_t point_begin = binding.point_offsets[system];
    const std::int64_t point_end = binding.point_offsets[system + 1];
    for (std::int64_t index = point_begin * 3 + threadIdx.x; index < point_end * 3;
         index += blockDim.x) {
      binding.candidate_point_positions[index] =
          use_source ? sources.point_positions[index] : binding.committed_point_positions[index];
    }
    for (std::int64_t point = point_begin + threadIdx.x; point < point_end; point += blockDim.x) {
      binding.candidate_point_values[point] =
          use_source ? sources.point_values[point] : binding.committed_point_values[point];
      binding.candidate_point_gammas[point] =
          use_source ? sources.point_gammas[point] : binding.committed_point_gammas[point];
    }
  }

  if (binding.periodic_enabled != 0u) {
    const std::int64_t response_begin = binding.response_offsets[system];
    const std::int64_t response_end = binding.response_offsets[system + 1];
    for (std::int64_t index = response_begin + threadIdx.x; index < response_end;
         index += blockDim.x) {
      binding.candidate_periodic_response[index] = use_source
                                                       ? sources.periodic_response[index]
                                                       : binding.committed_periodic_response[index];
    }
  }
}

std::uint32_t interaction_type_mask_host(std::int32_t type) noexcept {
  switch (type) {
    case XTBLOOM_INTERACTION_ELECTRIC_FIELD:
      return UINT32_C(1) << 0;
    case XTBLOOM_INTERACTION_ELECTRIC_FIELD_GRADIENT:
      return UINT32_C(1) << 1;
    case XTBLOOM_INTERACTION_POINT_CHARGES_MULTIPOLE:
      return UINT32_C(1) << 2;
    case XTBLOOM_INTERACTION_ATOMIC_POTENTIAL_GRID:
      return UINT32_C(1) << 3;
    case XTBLOOM_INTERACTION_ALPB_SOLVATION:
      return UINT32_C(1) << 4;
    case XTBLOOM_INTERACTION_GBSA_SOLVATION:
      return UINT32_C(1) << 5;
    case XTBLOOM_INTERACTION_GB_SOLVATION:
      return UINT32_C(1) << 6;
    case XTBLOOM_INTERACTION_GBE_SOLVATION:
      return UINT32_C(1) << 7;
    case XTBLOOM_INTERACTION_DDX_SOLVATION:
      return UINT32_C(1) << 8;
    case XTBLOOM_INTERACTION_D3_DISPERSION:
      return UINT32_C(1) << 9;
    case XTBLOOM_INTERACTION_D4_VARIANT_DISPERSION:
      return UINT32_C(1) << 10;
    case XTBLOOM_INTERACTION_HALOGEN_BOND:
      return UINT32_C(1) << 11;
    default:
      return 0u;
  }
}

__device__ bool known_interaction_type_device(std::int32_t type) noexcept {
  switch (type) {
    case XTBLOOM_INTERACTION_ELECTRIC_FIELD:
    case XTBLOOM_INTERACTION_ELECTRIC_FIELD_GRADIENT:
    case XTBLOOM_INTERACTION_POINT_CHARGES_MULTIPOLE:
    case XTBLOOM_INTERACTION_ATOMIC_POTENTIAL_GRID:
    case XTBLOOM_INTERACTION_ALPB_SOLVATION:
    case XTBLOOM_INTERACTION_GBSA_SOLVATION:
    case XTBLOOM_INTERACTION_GB_SOLVATION:
    case XTBLOOM_INTERACTION_GBE_SOLVATION:
    case XTBLOOM_INTERACTION_DDX_SOLVATION:
    case XTBLOOM_INTERACTION_D3_DISPERSION:
    case XTBLOOM_INTERACTION_D4_VARIANT_DISPERSION:
    case XTBLOOM_INTERACTION_HALOGEN_BOND:
      return true;
    default:
      return false;
  }
}

/* Normalize arbitrary descriptor/payload placement into one dense field per
 * system. The single-block scan is deliberately bounded by the public
 * descriptor count and performs no compact-payload assumption. The shared
 * request error codes are consumed by both the persistent-state admission gate
 * and the public result bridge. */
__global__ void normalize_gfn2_interactions_kernel(NumericalRefreshDeviceBinding binding,
                                                   NumericalRefreshDeviceSources sources) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (sources.prevalidated_interaction_error != kGfn2RequestErrorNone) {
    *binding.interaction_request_error = sources.prevalidated_interaction_error;
    return;
  }
  for (std::int64_t system = 0; system < binding.batch_size; ++system) {
    const bool retain = binding.requested[system] == 0u;
    binding.candidate_field_attached[system] =
        retain ? binding.committed_field_attached[system] : 0u;
    binding.candidate_field_vectors[3 * system] =
        retain ? binding.committed_field_vectors[3 * system] : 0.0;
    binding.candidate_field_vectors[3 * system + 1] =
        retain ? binding.committed_field_vectors[3 * system + 1] : 0.0;
    binding.candidate_field_vectors[3 * system + 2] =
        retain ? binding.committed_field_vectors[3 * system + 2] : 0.0;
  }
  bool reserved_tag = sources.prevalidated_reserved_interaction != 0u;
  for (std::int64_t index = 0; index < sources.total_interactions; ++index) {
    const xtbloom_interaction_t item = sources.interaction_descriptors[index];
    if (item.flags != 0u || item.type == XTBLOOM_INTERACTION_NONE ||
        !known_interaction_type_device(item.type) || item.system_index < 0 ||
        item.system_index >= binding.batch_size ||
        item.payload_offset > sources.interaction_payload_bytes ||
        item.payload_size > sources.interaction_payload_bytes - item.payload_offset) {
      *binding.interaction_request_error = kGfn2RequestErrorInvalid;
      return;
    }
    const std::uintptr_t payload_base = sources.interaction_payload_address;
    if (item.payload_offset > static_cast<std::uint64_t>(UINTPTR_MAX - payload_base)) {
      *binding.interaction_request_error = kGfn2RequestErrorInvalid;
      return;
    }
    const std::uintptr_t block_address =
        payload_base + static_cast<std::uintptr_t>(item.payload_offset);
    if (item.type != XTBLOOM_INTERACTION_ELECTRIC_FIELD) {
      if (item.payload_size < sizeof(std::int32_t) || block_address % alignof(std::int32_t) != 0u) {
        *binding.interaction_request_error = kGfn2RequestErrorInvalid;
        return;
      } else {
        reserved_tag = true;
      }
      continue;
    }
    if (item.payload_size != 32u || block_address % alignof(double) != 0u) {
      *binding.interaction_request_error = kGfn2RequestErrorInvalid;
      return;
    }
    const auto* block = sources.interaction_payload_compacted_by_system != 0u
                            ? sources.interaction_payload + 32u * item.system_index
                            : (sources.interaction_payload_staged_at_offsets != 0u
                                   ? sources.interaction_payload + item.payload_offset
                                   : reinterpret_cast<const std::byte*>(block_address));
    const auto version = *reinterpret_cast<const std::int32_t*>(block);
    const auto reserved = *reinterpret_cast<const std::int32_t*>(block + sizeof(std::int32_t));
    const auto* field = reinterpret_cast<const double*>(block + 2 * sizeof(std::int32_t));
    if (version != 1 || reserved != 0 || !isfinite(field[0]) || !isfinite(field[1]) ||
        !isfinite(field[2])) {
      *binding.interaction_request_error = kGfn2RequestErrorInvalid;
      return;
    }
  }
  for (std::int64_t lhs = 0; lhs < sources.total_interactions; ++lhs) {
    const xtbloom_interaction_t left = sources.interaction_descriptors[lhs];
    for (std::int64_t rhs = lhs + 1; rhs < sources.total_interactions; ++rhs) {
      const xtbloom_interaction_t right = sources.interaction_descriptors[rhs];
      if (left.system_index == right.system_index && left.type == right.type) {
        *binding.interaction_request_error = kGfn2RequestErrorInvalid;
        return;
      }
    }
  }
  if (reserved_tag) {
    *binding.interaction_request_error = kGfn2RequestErrorNotImplemented;
    return;
  }
  for (std::int64_t index = 0; index < sources.total_interactions; ++index) {
    const xtbloom_interaction_t item = sources.interaction_descriptors[index];
    if (binding.requested[item.system_index] == 0u) continue;
    const auto* block = sources.interaction_payload_compacted_by_system != 0u
                            ? sources.interaction_payload + 32u * item.system_index
                            : (sources.interaction_payload_staged_at_offsets != 0u
                                   ? sources.interaction_payload + item.payload_offset
                                   : reinterpret_cast<const std::byte*>(
                                         sources.interaction_payload_address +
                                         static_cast<std::uintptr_t>(item.payload_offset)));
    const auto* field = reinterpret_cast<const double*>(block + 2 * sizeof(std::int32_t));
    binding.candidate_field_attached[item.system_index] = 1u;
    binding.candidate_field_vectors[3 * item.system_index] = field[0];
    binding.candidate_field_vectors[3 * item.system_index + 1] = field[1];
    binding.candidate_field_vectors[3 * item.system_index + 2] = field[2];
  }
}

__global__ void merge_gfn2_request_error_kernel(const std::uint32_t* interaction_error,
                                                std::uint32_t* request_error) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const std::uint32_t error = *interaction_error;
    if (error != 0u) atomicMax(request_error, error);
  }
}

/* Strict WARM compatibility is a call-wide admission decision. It is checked
 * after interaction normalization has produced the candidate field image but
 * before preprocessing advances the epoch or any persistent cache publishes.
 * A separate validation pass avoids the peer race that would occur if one
 * block started consuming its checkpoint while another discovered a mismatch. */
__global__ void validate_gfn2_warm_field_identity_kernel(
    NumericalRefreshDeviceBinding binding, const std::uint8_t* checkpoint_field_attached,
    const double* checkpoint_field_vectors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= binding.batch_size || *binding.request_error != kGfn2RequestErrorNone) return;
  const bool same_field =
      binding.candidate_field_attached[system] == checkpoint_field_attached[system] &&
      binding.candidate_field_vectors[3 * system] == checkpoint_field_vectors[3 * system] &&
      binding.candidate_field_vectors[3 * system + 1] == checkpoint_field_vectors[3 * system + 1] &&
      binding.candidate_field_vectors[3 * system + 2] == checkpoint_field_vectors[3 * system + 2];
  if (!same_field) {
    atomicCAS(binding.request_error, kGfn2RequestErrorNone, kGfn2RequestErrorWarmIncompatible);
  }
}

__global__ void validate_gfn2_periodic_refresh_kernel(NumericalRefreshDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size || binding.periodic_enabled == 0u ||
      binding.preprocessing_published[system] == 0u ||
      atomicAdd(binding.periodic_plan_error, 0u) != 0u) {
    return;
  }
  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    if (!isfinite(binding.candidate_periodic_shifts[atom])) {
      atomicCAS(binding.periodic_system_errors + system, 0u, 1u);
    }
  }
  __syncthreads();
  if (atomicAdd(binding.periodic_system_errors + system, 0u) != 0u) return;

  const std::int64_t count = atom_end - atom_begin;
  const std::int64_t response_begin = binding.response_offsets[system];
  const std::int64_t response_end = binding.response_offsets[system + 1];
  if (count < 0 || count > 3037000499LL || response_begin < 0 || response_end < response_begin ||
      response_end - response_begin != count * count) {
    if (threadIdx.x == 0) atomicCAS(binding.periodic_plan_error, 0u, 1u);
    return;
  }
  for (std::int64_t local = threadIdx.x; local < count * count; local += blockDim.x) {
    const double value = binding.candidate_periodic_response[response_begin + local];
    if (!isfinite(value)) {
      atomicCAS(binding.periodic_system_errors + system, 0u, 2u);
      continue;
    }
    const std::int64_t row = local / count;
    const std::int64_t column = local - row * count;
    const double transpose =
        binding.candidate_periodic_response[response_begin + column * count + row];
    if (value != transpose) {
      atomicCAS(binding.periodic_system_errors + system, 0u, 3u);
    }
  }
}

__global__ void initialize_gfn2_refresh_sequence_kernel(std::uint32_t* sequence) {
  if (threadIdx.x == 0) *sequence = 1u;
}

/* Publish the common pre-factor gate into the eigensolver owner's canonical
 * activity mask. The overlap owner then commits factors only for these peers. */
__global__ void gate_gfn2_numerical_refresh_kernel(NumericalRefreshDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  /* Request-level semantic rejection preserves the prior canonical activity
   * and factor-cache transaction. Candidate diagnostics may still have been
   * overwritten earlier on the stream, but no persistent eligibility changes. */
  const Gfn2DeviceAdmission admission{binding.request_error, 1, binding.plan_token};
  if (!gfn2_request_admitted(admission)) return;
  const bool plan_healthy =
      atomicAdd(const_cast<std::uint32_t*>(binding.preprocessing_plan_error), 0u) == 0u &&
      (binding.d4_enabled == 0u ||
       atomicAdd(const_cast<std::uint32_t*>(binding.d4_device_error), 0u) == 0u) &&
      (binding.point_enabled == 0u ||
       atomicAdd(const_cast<std::uint32_t*>(binding.point_plan_error), 0u) == 0u) &&
      (binding.periodic_enabled == 0u || atomicAdd(binding.periodic_plan_error, 0u) == 0u) &&
      gfn2_request_admitted(admission) && atomicAdd(binding.field_plan_error, 0u) == 0u;
  const bool peer_healthy =
      binding.requested[system] == 1u && binding.preprocessing_published[system] == 1u &&
      (binding.d4_enabled == 0u || binding.d4_system_errors[system] == 0u) &&
      (binding.point_enabled == 0u || binding.point_system_errors[system] == 0u) &&
      (binding.periodic_enabled == 0u || binding.periodic_system_errors[system] == 0u) &&
      binding.field_system_errors[system] == 0u;
  if (threadIdx.x == 0) {
    const std::uint8_t eligible = plan_healthy && peer_healthy ? 1u : 0u;
    binding.eligible[system] = eligible;
    binding.factor_active[system] = eligible;
  }
}

__global__ void commit_gfn2_numerical_refresh_kernel(NumericalRefreshDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  const Gfn2DeviceAdmission admission{binding.request_error, 1, binding.plan_token};
  if (!gfn2_request_admitted(admission)) return;
  const std::uint64_t generation = *binding.geometry_epoch;
  const std::uint64_t previous_generation = binding.committed_generations[system];
  const bool publish = binding.eligible[system] == 1u && binding.factor_statuses[system] == 0u &&
                       binding.factor_generations[system] == generation;
  if (threadIdx.x == 0) {
    binding.eligible[system] = publish ? 1u : 0u;
    binding.refresh_predecessor_generations[system] = publish ? previous_generation : 0u;
    if (!publish) binding.warm_checkpoint_generations[system] = 0u;
  }
  if (!publish) return;

  if (threadIdx.x == 0) {
    binding.committed_field_attached[system] = binding.candidate_field_attached[system];
    binding.committed_field_vectors[3 * system] = binding.candidate_field_vectors[3 * system];
    binding.committed_field_vectors[3 * system + 1] =
        binding.candidate_field_vectors[3 * system + 1];
    binding.committed_field_vectors[3 * system + 2] =
        binding.candidate_field_vectors[3 * system + 2];
  }

  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  for (std::int64_t index = atom_begin * 3 + threadIdx.x; index < atom_end * 3;
       index += blockDim.x) {
    binding.committed_positions[index] = binding.candidate_positions[index];
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    binding.committed_field_atomic_potentials[atom] =
        binding.candidate_field_atomic_potentials[atom];
    binding.committed_field_dipole_potentials[3 * atom] =
        binding.candidate_field_dipole_potentials[3 * atom];
    binding.committed_field_dipole_potentials[3 * atom + 1] =
        binding.candidate_field_dipole_potentials[3 * atom + 1];
    binding.committed_field_dipole_potentials[3 * atom + 2] =
        binding.candidate_field_dipole_potentials[3 * atom + 2];
    binding.public_coordination[atom] = binding.candidate_coordination[atom];
    if (binding.d4_enabled != 0u) {
      binding.public_d4_coordination[atom] = binding.candidate_d4_coordination[atom];
    }
    if (binding.periodic_enabled != 0u) {
      binding.committed_periodic_shifts[atom] = binding.candidate_periodic_shifts[atom];
    }
  }

  const std::int64_t matrix_begin = binding.matrix_offsets[system];
  const std::int64_t matrix_end = binding.matrix_offsets[system + 1];
  for (std::int64_t index = matrix_begin + threadIdx.x; index < matrix_end; index += blockDim.x) {
    binding.public_overlap[index] = binding.candidate_overlap[index];
    binding.public_h0[index] = binding.candidate_h0[index];
  }
  /* Integral multipoles are global component-major arrays.  A peer owns its
   * matrix interval in every component plane; treating 3*M or 6*M as one
   * contiguous per-peer interval would publish bytes belonging to adjacent
   * peers and violate transactional rollback. */
  if (binding.multipoles_enabled != 0u) {
    for (std::int64_t matrix = matrix_begin + threadIdx.x; matrix < matrix_end;
         matrix += blockDim.x) {
      for (std::int64_t component = 0; component < 3; ++component) {
        const std::int64_t index = component * binding.total_matrices + matrix;
        binding.public_dipole[index] = binding.candidate_dipole[index];
      }
      for (std::int64_t component = 0; component < 6; ++component) {
        const std::int64_t index = component * binding.total_matrices + matrix;
        binding.public_quadrupole[index] = binding.candidate_quadrupole[index];
      }
    }
  }

  const std::int64_t pair_begin = binding.geometry_pair_offsets[system];
  const std::int64_t pair_end = binding.geometry_pair_offsets[system + 1];
  for (std::int64_t index = pair_begin * kGfn2GeometryPairDataElements + threadIdx.x;
       index < pair_end * kGfn2GeometryPairDataElements; index += blockDim.x) {
    binding.public_geometry_pairs[index] = binding.candidate_geometry_pairs[index];
  }
  if (binding.aes2_enabled != 0u) {
    for (std::int64_t index = pair_begin * kGfn2AES2PairDataElements + threadIdx.x;
         index < pair_end * kGfn2AES2PairDataElements; index += blockDim.x) {
      binding.public_aes2[index] = binding.candidate_aes2[index];
    }
  }
  const std::int64_t es2_begin = binding.es2_offsets[system];
  const std::int64_t es2_end = binding.es2_offsets[system + 1];
  for (std::int64_t index = es2_begin + threadIdx.x; index < es2_end; index += blockDim.x) {
    binding.public_es2[index] = binding.candidate_es2[index];
  }

  if (binding.point_enabled != 0u) {
    const std::int64_t point_begin = binding.point_offsets[system];
    const std::int64_t point_end = binding.point_offsets[system + 1];
    for (std::int64_t index = point_begin * 3 + threadIdx.x; index < point_end * 3;
         index += blockDim.x) {
      binding.committed_point_positions[index] = binding.candidate_point_positions[index];
    }
    for (std::int64_t point = point_begin + threadIdx.x; point < point_end; point += blockDim.x) {
      binding.committed_point_values[point] = binding.candidate_point_values[point];
      binding.committed_point_gammas[point] = binding.candidate_point_gammas[point];
    }
    const std::int64_t shell_begin = binding.shell_offsets[system];
    const std::int64_t shell_end = binding.shell_offsets[system + 1];
    for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
      binding.public_point_shell[shell] = binding.candidate_point_shell[shell];
    }
  }
  if (binding.periodic_enabled != 0u) {
    const std::int64_t response_begin = binding.response_offsets[system];
    const std::int64_t response_end = binding.response_offsets[system + 1];
    for (std::int64_t index = response_begin + threadIdx.x; index < response_end;
         index += blockDim.x) {
      binding.committed_periodic_response[index] = binding.candidate_periodic_response[index];
    }
  }
  if (threadIdx.x == 0) binding.committed_generations[system] = generation;
}

/*
 * Warm execution reuses the device wavefunction and published multipoles.
 * Same-epoch reuse also retains modified-Broyden history, while predecessor-
 * epoch migration starts a new mixer history window for the refreshed
 * operator. The driver-visible terminal trace belongs to one inference
 * attempt and must always be restarted or the bounded loop would treat the
 * prior converged state as inactive.
 */
struct WarmSccResetDeviceBinding {
  std::int64_t batch_size = 0;
  std::uint64_t plan_token = 0u;
  Gfn2GeometryEpochDevice geometry_epoch{};
  const std::uint8_t* eligible = nullptr;
  const std::uint64_t* committed_generations = nullptr;
  const std::uint64_t* refresh_predecessor_generations = nullptr;
  std::uint64_t* warm_checkpoint_generations = nullptr;
  const std::uint8_t* committed_field_attached = nullptr;
  const double* committed_field_vectors = nullptr;
  const std::uint8_t* checkpoint_field_attached = nullptr;
  const double* checkpoint_field_vectors = nullptr;
  std::uint32_t* request_error = nullptr;
  const std::uint32_t* interaction_error = nullptr;
  Gfn2SccMixerDeviceState mixer{};
  Gfn2SccDeviceState scc{};
};

static_assert(std::is_trivially_copyable_v<WarmSccResetDeviceBinding>);

__global__ void reset_gfn2_warm_scc_trace_kernel(WarmSccResetDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= binding.batch_size) return;
  const Gfn2DeviceAdmission admission{binding.interaction_error, 1, binding.plan_token};
  if (!gfn2_request_mutation_allowed(admission)) return;
  const std::uint64_t epoch = *binding.geometry_epoch.value;
  const std::uint64_t checkpoint = atomicAdd(
      reinterpret_cast<unsigned long long*>(binding.warm_checkpoint_generations + system), 0ULL);
  const std::uint64_t predecessor = binding.refresh_predecessor_generations[system];
  const bool same_field =
      binding.committed_field_attached[system] == binding.checkpoint_field_attached[system] &&
      binding.committed_field_vectors[3 * system] == binding.checkpoint_field_vectors[3 * system] &&
      binding.committed_field_vectors[3 * system + 1] ==
          binding.checkpoint_field_vectors[3 * system + 1] &&
      binding.committed_field_vectors[3 * system + 2] ==
          binding.checkpoint_field_vectors[3 * system + 2];
  const bool compatible = epoch != 0u && checkpoint != 0u && binding.eligible[system] == 1u &&
                          binding.committed_generations[system] == epoch &&
                          (checkpoint == epoch || checkpoint == predecessor) && same_field;
  if (!same_field) return;
  atomicExch(reinterpret_cast<unsigned long long*>(binding.warm_checkpoint_generations + system),
             0ULL);
  const bool migrated = compatible && checkpoint != epoch && checkpoint == predecessor;

  /* A geometry-epoch migration keeps the converged multipoles/wavefunction
   * but starts a new Broyden history window. The old finite-difference basis
   * belongs to the predecessor operator and can be substantially worse than
   * simple damping after even a small coordinate change. Setting iteration
   * zero makes subsequent slots overwrite old history before it can be read. */
  if (migrated) {
    binding.mixer.residual_rms[system] = 0.0;
    binding.mixer.residual_maximum[system] = 0.0;
    binding.mixer.iterations[system] = 0u;
    binding.mixer.system_statuses[system] = XTBLOOM_STATUS_SUCCESS;
    binding.mixer.residual_converged[system] = 0u;
  }

  /* iteration==0 deliberately seeds the first warm energy delta from zero,
   * matching fresh driver accounting while retaining the expensive electronic
   * checkpoint and, for same-epoch reuse, the mixer history. */
  binding.scc.free_energies[system] = 0.0;
  binding.scc.previous_free_energies[system] = 0.0;
  binding.scc.free_energy_changes[system] = 0.0;
  binding.scc.residual_rms[system] = 0.0;
  binding.scc.iterations[system] = 0u;
  binding.scc.converged[system] = 0u;
  binding.scc.system_statuses[system] = compatible || binding.eligible[system] == 0u
                                            ? XTBLOOM_STATUS_SUCCESS
                                            : XTBLOOM_STATUS_INTERNAL_ERROR;
}

struct WarmCheckpointPublicationDeviceBinding {
  std::int64_t batch_size = 0;
  Gfn2GeometryEpochDevice geometry_epoch{};
  const std::uint8_t* eligible = nullptr;
  const std::uint64_t* committed_generations = nullptr;
  const std::uint32_t* publication_plan_error = nullptr;
  const xtbloom_status_t* result_statuses = nullptr;
  const std::uint8_t* result_converged = nullptr;
  std::uint64_t* warm_checkpoint_generations = nullptr;
  std::uint32_t* batch_ready = nullptr;
  const std::uint8_t* committed_field_attached = nullptr;
  const double* committed_field_vectors = nullptr;
  std::uint8_t* checkpoint_field_attached = nullptr;
  double* checkpoint_field_vectors = nullptr;
  const std::uint32_t* request_error = nullptr;
};

static_assert(std::is_trivially_copyable_v<WarmCheckpointPublicationDeviceBinding>);

__global__ void publish_gfn2_warm_checkpoint_generation_kernel(
    WarmCheckpointPublicationDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= binding.batch_size) return;
  const Gfn2DeviceAdmission admission{binding.request_error, 1, 0u};
  if (!gfn2_request_mutation_allowed(admission)) return;
  const std::uint64_t epoch = *binding.geometry_epoch.value;
  const bool publish =
      atomicAdd(const_cast<std::uint32_t*>(binding.publication_plan_error), 0u) == 0u &&
      epoch != 0u && binding.eligible[system] == 1u &&
      binding.committed_generations[system] == epoch &&
      binding.result_statuses[system] == XTBLOOM_STATUS_SUCCESS &&
      binding.result_converged[system] == 1u;
  binding.warm_checkpoint_generations[system] = publish ? epoch : 0u;
  if (publish) {
    binding.checkpoint_field_attached[system] = binding.committed_field_attached[system];
    binding.checkpoint_field_vectors[3 * system] = binding.committed_field_vectors[3 * system];
    binding.checkpoint_field_vectors[3 * system + 1] =
        binding.committed_field_vectors[3 * system + 1];
    binding.checkpoint_field_vectors[3 * system + 2] =
        binding.committed_field_vectors[3 * system + 2];
  }
  if (!publish) atomicExch(binding.batch_ready, 0u);
}

__global__ void invalidate_gfn2_warm_checkpoint_if_admitted_kernel(
    std::uint64_t* generations, std::int64_t elements, const std::uint32_t* request_error) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const Gfn2DeviceAdmission admission{request_error, 1, 0u};
  if (gfn2_request_mutation_allowed(admission) && index < elements) generations[index] = 0u;
}

__global__ void initialize_gfn2_warm_checkpoint_batch_if_admitted_kernel(
    std::uint32_t* batch_ready, Gfn2DeviceAdmission admission) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && gfn2_request_mutation_allowed(admission)) {
    *batch_ready = UINT32_MAX;
  }
}

/*
 * Terminal analytic forces consume physical topology-major arrays, whereas
 * the converged SCC state is system-major and spin-major. This stable binding
 * projects the committed state once after the SCC loop so every downstream
 * force primitive can retain its established physical-offset contract.
 */
struct StationaryForceProjectionDeviceBinding {
  std::uint8_t enabled = 0u;
  std::uint8_t multipoles_enabled = 1u;
  const std::uint32_t* request_error = nullptr;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t total_spin_atoms = 0;
  std::int64_t total_spin_shells = 0;
  std::int64_t total_spin_matrix_elements = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* shell_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int32_t* spin_channels = nullptr;
  const std::int64_t* spin_atom_offsets = nullptr;
  const std::int64_t* spin_shell_offsets = nullptr;
  const std::int64_t* spin_matrix_offsets = nullptr;
  const std::int64_t* spin_coupling_offsets = nullptr;
  const double* spin_coupling_matrices = nullptr;

  const double* packed_density = nullptr;
  const double* packed_energy_weighted_density = nullptr;
  const double* packed_shell_charges = nullptr;
  const double* packed_atomic_charges = nullptr;
  const double* packed_atomic_dipoles = nullptr;
  const double* packed_atomic_quadrupoles = nullptr;

  double* total_density = nullptr;
  double* total_energy_weighted_density = nullptr;
  double* spin_density = nullptr;
  double* shell_charges = nullptr;
  double* atomic_charges = nullptr;
  double* atomic_dipoles = nullptr;
  double* atomic_quadrupoles = nullptr;
  double* spin_shell_potentials = nullptr;
};

static_assert(std::is_trivially_copyable_v<StationaryForceProjectionDeviceBinding>);

__global__ void project_gfn2_stationary_force_state_kernel(
    StationaryForceProjectionDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  const Gfn2DeviceAdmission admission{binding.request_error, 1, 0u};
  if (!gfn2_request_admitted(admission)) return;

  const std::int32_t channels = binding.spin_channels[system];
  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  const std::int64_t shell_begin = binding.shell_offsets[system];
  const std::int64_t shell_end = binding.shell_offsets[system + 1];
  const std::int64_t matrix_begin = binding.matrix_offsets[system];
  const std::int64_t matrix_end = binding.matrix_offsets[system + 1];
  const std::int64_t spin_atom_begin = binding.spin_atom_offsets[system];
  const std::int64_t spin_shell_begin = binding.spin_shell_offsets[system];
  const std::int64_t spin_matrix_begin = binding.spin_matrix_offsets[system];
  const std::int64_t physical_shells = shell_end - shell_begin;
  const std::int64_t physical_matrices = matrix_end - matrix_begin;

  for (std::int64_t local = threadIdx.x; local < physical_matrices; local += blockDim.x) {
    const double alpha_density = binding.packed_density[spin_matrix_begin + local];
    const double alpha_weighted = binding.packed_energy_weighted_density[spin_matrix_begin + local];
    if (channels == 1) {
      binding.total_density[matrix_begin + local] = alpha_density;
      binding.total_energy_weighted_density[matrix_begin + local] = alpha_weighted;
      binding.spin_density[matrix_begin + local] = 0.0;
    } else {
      const double beta_density =
          binding.packed_density[spin_matrix_begin + physical_matrices + local];
      const double beta_weighted =
          binding.packed_energy_weighted_density[spin_matrix_begin + physical_matrices + local];
      binding.total_density[matrix_begin + local] = alpha_density + beta_density;
      binding.total_energy_weighted_density[matrix_begin + local] = alpha_weighted + beta_weighted;
      binding.spin_density[matrix_begin + local] = alpha_density - beta_density;
    }
  }

  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const std::int64_t local = shell - shell_begin;
    binding.shell_charges[shell] = binding.packed_shell_charges[spin_shell_begin + local];
    if (channels == 1) binding.spin_shell_potentials[shell] = 0.0;
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t local_atom = atom - atom_begin;
    const std::int64_t spin_atom = spin_atom_begin + local_atom;
    binding.atomic_charges[atom] = binding.packed_atomic_charges[spin_atom];
    if (binding.multipoles_enabled != 0u) {
      for (std::int64_t component = 0; component < 3; ++component) {
        binding.atomic_dipoles[atom * 3 + component] =
            binding.packed_atomic_dipoles[spin_atom * 3 + component];
      }
      for (std::int64_t component = 0; component < 6; ++component) {
        binding.atomic_quadrupoles[atom * 6 + component] =
            binding.packed_atomic_quadrupoles[spin_atom * 6 + component];
      }
    }
  }

  if (channels == 2) {
    /* v_mag = W*m uses the committed raw magnetization shell population.
     * Coupling rows are atom-local and retain the CPU column accumulation
     * order, avoiding a different roundoff path in force parity checks. */
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::int64_t atom_shell_begin = binding.atom_shell_offsets[atom];
      const std::int64_t atom_shell_end = binding.atom_shell_offsets[atom + 1];
      const std::int64_t atom_shells = atom_shell_end - atom_shell_begin;
      const std::int64_t coupling_begin = binding.spin_coupling_offsets[atom];
      for (std::int64_t shell = atom_shell_begin + threadIdx.x; shell < atom_shell_end;
           shell += blockDim.x) {
        const std::int64_t row = shell - atom_shell_begin;
        double potential = 0.0;
        for (std::int64_t column = 0; column < atom_shells; ++column) {
          const std::int64_t magnetization_shell =
              spin_shell_begin + physical_shells + atom_shell_begin - shell_begin + column;
          potential += binding.spin_coupling_matrices[coupling_begin + row * atom_shells + column] *
                       binding.packed_shell_charges[magnetization_shell];
        }
        binding.spin_shell_potentials[shell] = potential;
      }
    }
  }
}

cudaError_t project_gfn2_stationary_force_state_cuda(
    const StationaryForceProjectionDeviceBinding& binding, cudaStream_t stream) noexcept {
  if (binding.enabled == 0u) return cudaSuccess;
  if (binding.enabled != 1u || binding.multipoles_enabled > 1u || binding.batch_size <= 0 ||
      binding.total_atoms <= 0 || binding.total_shells <= 0 || binding.total_matrix_elements <= 0 ||
      binding.total_spin_atoms < binding.total_atoms ||
      binding.total_spin_shells < binding.total_shells ||
      binding.total_spin_matrix_elements < binding.total_matrix_elements ||
      binding.atom_offsets == nullptr || binding.shell_offsets == nullptr ||
      binding.matrix_offsets == nullptr || binding.atom_shell_offsets == nullptr ||
      binding.spin_channels == nullptr || binding.spin_atom_offsets == nullptr ||
      binding.spin_shell_offsets == nullptr || binding.spin_matrix_offsets == nullptr ||
      binding.spin_coupling_offsets == nullptr || binding.spin_coupling_matrices == nullptr ||
      binding.packed_density == nullptr || binding.packed_energy_weighted_density == nullptr ||
      binding.packed_shell_charges == nullptr || binding.packed_atomic_charges == nullptr ||
      binding.total_density == nullptr || binding.total_energy_weighted_density == nullptr ||
      binding.spin_density == nullptr || binding.shell_charges == nullptr ||
      binding.atomic_charges == nullptr || binding.spin_shell_potentials == nullptr ||
      (binding.multipoles_enabled != 0u &&
       (binding.packed_atomic_dipoles == nullptr || binding.packed_atomic_quadrupoles == nullptr ||
        binding.atomic_dipoles == nullptr || binding.atomic_quadrupoles == nullptr)) ||
      binding.batch_size > static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max())) {
    return cudaErrorInvalidValue;
  }
  project_gfn2_stationary_force_state_kernel<<<static_cast<unsigned int>(binding.batch_size), 256,
                                               0, stream>>>(binding);
  return cudaPeekAtLastError();
}

struct TopologyKey {
  xtbloom_model_t model = XTBLOOM_MODEL_GFN2_XTB;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> point_offsets;
  std::vector<std::int64_t> response_offsets;
  std::uint32_t flags = 0u;
  std::int32_t maximum_iterations = 0;
  double charge_tolerance = 0.0;
  double energy_tolerance = 0.0;
  double electronic_temperature = 0.0;
  xtbloom_scc_mixer_t scc_mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
  std::int32_t scc_mixer_history = kDefaultMixerHistory;
  double scc_mixer_damping = kDefaultMixerDamping;
  xtbloom_determinism_t determinism = XTBLOOM_DETERMINISM_DEFAULT;
  bool periodic_enabled = false;
  /* Native cell identity is deliberately separate from the external
   * charge-response operator represented by periodic_enabled. */
  bool native_lattice_enabled = false;
  std::vector<double> cell_matrices;
  std::vector<std::int32_t> periodic_axes;

  std::uint64_t fingerprint() const noexcept {
    std::uint64_t hash = 0x4750555854424b59ULL;
    hash_append(hash, static_cast<std::uint32_t>(model));
    hash_append_vector(hash, atom_offsets);
    hash_append_vector(hash, atomic_numbers);
    hash_append_vector(hash, molecular_charges);
    hash_append_vector(hash, unpaired_electrons);
    hash_append_vector(hash, spin_channels);
    hash_append_vector(hash, point_offsets);
    hash_append_vector(hash, response_offsets);
    hash_append(hash, flags);
    hash_append(hash, static_cast<std::uint32_t>(maximum_iterations));
    hash_append(hash, double_bits(charge_tolerance));
    hash_append(hash, double_bits(energy_tolerance));
    hash_append(hash, double_bits(electronic_temperature));
    hash_append(hash, static_cast<std::uint32_t>(scc_mixer));
    hash_append(hash, static_cast<std::uint32_t>(scc_mixer_history));
    hash_append(hash, double_bits(scc_mixer_damping));
    hash_append(hash, static_cast<std::uint32_t>(determinism));
    hash_append(hash, periodic_enabled ? 1u : 0u);
    hash_append(hash, native_lattice_enabled ? 1u : 0u);
    hash_append_vector(hash, cell_matrices);
    hash_append_vector(hash, periodic_axes);
    return hash == 0u ? 1u : hash;
  }
};

/*
 * One stream-ordered comparison of the immutable fixed-plan descriptors.
 * HOST fields are compared by the submitting thread before launch; only
 * CUDA_DEVICE fields are populated here. The canonical device arrays are
 * plan-owned copies, so the kernel never waits for or dereferences host
 * storage and naturally follows preceding Torch work on the owner stream.
 */
struct FixedTopologyComparisonDeviceBinding {
  const std::int64_t* atom_offsets = nullptr;
  const std::int32_t* atomic_numbers = nullptr;
  const double* molecular_charges = nullptr;
  const std::int32_t* unpaired_electrons = nullptr;
  const std::int32_t* spin_channels = nullptr;
  const std::int64_t* point_offsets = nullptr;
  const std::int64_t* response_offsets = nullptr;
  const double* cell_matrices = nullptr;
  const std::int32_t* periodic_axes = nullptr;

  const std::int64_t* expected_atom_offsets = nullptr;
  const std::int32_t* expected_atomic_numbers = nullptr;
  const double* expected_molecular_charges = nullptr;
  const std::int32_t* expected_unpaired_electrons = nullptr;
  const std::int32_t* expected_spin_channels = nullptr;
  const std::int64_t* expected_point_offsets = nullptr;
  const std::int64_t* expected_response_offsets = nullptr;
  const double* expected_cell_matrices = nullptr;
  const std::int32_t* expected_periodic_axes = nullptr;

  std::int64_t atom_offset_elements = 0;
  std::int64_t atomic_number_elements = 0;
  std::int64_t batch_elements = 0;
  std::int64_t point_offset_elements = 0;
  std::int64_t response_offset_elements = 0;
  std::int64_t cell_elements = 0;
  std::int64_t periodic_axis_elements = 0;
  std::int64_t lattice_systems = 0;
  std::uint8_t reject_native_external_combinations = 0u;
  std::uint32_t* request_error = nullptr;
};

template <typename T>
__device__ bool fixed_topology_value_differs(const T* actual, const T* expected, std::int64_t index,
                                             std::int64_t elements) noexcept {
  return actual != nullptr && index < elements && actual[index] != expected[index];
}

__global__ void compare_fixed_topology_kernel(FixedTopologyComparisonDeviceBinding binding) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  std::int64_t maximum = binding.atom_offset_elements;
  if (binding.atomic_number_elements > maximum) maximum = binding.atomic_number_elements;
  if (binding.batch_elements > maximum) maximum = binding.batch_elements;
  if (binding.point_offset_elements > maximum) maximum = binding.point_offset_elements;
  if (binding.response_offset_elements > maximum) maximum = binding.response_offset_elements;
  if (binding.cell_elements > maximum) maximum = binding.cell_elements;
  if (binding.periodic_axis_elements > maximum) maximum = binding.periodic_axis_elements;
  for (std::int64_t element = index; element < maximum; element += stride) {
    const bool mismatch =
        fixed_topology_value_differs(binding.atom_offsets, binding.expected_atom_offsets, element,
                                     binding.atom_offset_elements) ||
        fixed_topology_value_differs(binding.atomic_numbers, binding.expected_atomic_numbers,
                                     element, binding.atomic_number_elements) ||
        fixed_topology_value_differs(binding.molecular_charges, binding.expected_molecular_charges,
                                     element, binding.batch_elements) ||
        fixed_topology_value_differs(binding.unpaired_electrons,
                                     binding.expected_unpaired_electrons, element,
                                     binding.batch_elements) ||
        fixed_topology_value_differs(binding.spin_channels, binding.expected_spin_channels, element,
                                     binding.batch_elements) ||
        fixed_topology_value_differs(binding.point_offsets, binding.expected_point_offsets, element,
                                     binding.point_offset_elements) ||
        fixed_topology_value_differs(binding.response_offsets, binding.expected_response_offsets,
                                     element, binding.response_offset_elements) ||
        fixed_topology_value_differs(binding.cell_matrices, binding.expected_cell_matrices, element,
                                     binding.cell_elements) ||
        fixed_topology_value_differs(binding.periodic_axes, binding.expected_periodic_axes, element,
                                     binding.periodic_axis_elements);
    if (mismatch) atomicMax(binding.request_error, kGfn2RequestErrorTopologyMismatch);
  }
}

__global__ void validate_lattice_request_kernel(FixedTopologyComparisonDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= binding.lattice_systems || binding.cell_matrices == nullptr ||
      binding.periodic_axes == nullptr) {
    return;
  }
  const std::int32_t mask = binding.periodic_axes[system];
  const double* cell = binding.cell_matrices + system * 9;
  if (mask == XTBLOOM_PERIODIC_AXES_NONE) {
    for (int element = 0; element < 9; ++element) {
      if (cell[element] != 0.0) {
        atomicMax(binding.request_error, kGfn2RequestErrorInvalid);
        return;
      }
    }
    return;
  }
  if ((mask & ~XTBLOOM_PERIODIC_AXES_XYZ) != 0) {
    atomicMax(binding.request_error, kGfn2RequestErrorInvalid);
    return;
  }
  if (mask != XTBLOOM_PERIODIC_AXES_XYZ) {
    atomicMax(binding.request_error, kGfn2RequestErrorNotSupported);
    return;
  }
  if (!gfn2::valid_lattice_cell_3d_binary64(cell)) {
    atomicMax(binding.request_error, kGfn2RequestErrorInvalid);
    return;
  }
  if (binding.reject_native_external_combinations != 0u) {
    atomicMax(binding.request_error, kGfn2RequestErrorNotSupported);
    return;
  }
  atomicMax(binding.request_error, kGfn2RequestErrorNotImplemented);
}

cudaError_t compare_fixed_topology_async(const FixedTopologyComparisonDeviceBinding& binding,
                                         cudaStream_t stream) noexcept {
  if (binding.request_error == nullptr || binding.atom_offset_elements <= 0 ||
      binding.atomic_number_elements <= 0 || binding.batch_elements <= 0 ||
      binding.point_offset_elements < 0 || binding.response_offset_elements < 0 ||
      binding.cell_elements < 0 || binding.periodic_axis_elements < 0 ||
      binding.expected_atom_offsets == nullptr || binding.expected_atomic_numbers == nullptr ||
      binding.expected_molecular_charges == nullptr ||
      binding.expected_unpaired_electrons == nullptr || binding.expected_spin_channels == nullptr ||
      (binding.point_offset_elements != 0 && binding.expected_point_offsets == nullptr) ||
      (binding.response_offset_elements != 0 && binding.expected_response_offsets == nullptr) ||
      (binding.cell_elements != 0 &&
       (binding.cell_matrices == nullptr || binding.expected_cell_matrices == nullptr)) ||
      (binding.periodic_axis_elements != 0 &&
       (binding.periodic_axes == nullptr || binding.expected_periodic_axes == nullptr))) {
    return cudaErrorInvalidValue;
  }
  std::int64_t maximum = binding.atom_offset_elements;
  maximum = std::max(maximum, binding.atomic_number_elements);
  maximum = std::max(maximum, binding.batch_elements);
  maximum = std::max(maximum, binding.point_offset_elements);
  maximum = std::max(maximum, binding.response_offset_elements);
  maximum = std::max(maximum, binding.cell_elements);
  maximum = std::max(maximum, binding.periodic_axis_elements);
  constexpr int kThreads = 256;
  const std::int64_t required_blocks = (maximum + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<std::int64_t>(required_blocks, 256));
  compare_fixed_topology_kernel<<<blocks, kThreads, 0, stream>>>(binding);
  return cudaPeekAtLastError();
}

cudaError_t validate_lattice_request_async(const FixedTopologyComparisonDeviceBinding& binding,
                                           cudaStream_t stream) noexcept {
  if (binding.request_error == nullptr || binding.lattice_systems < 0 ||
      (binding.lattice_systems != 0 &&
       (binding.cell_matrices == nullptr || binding.periodic_axes == nullptr))) {
    return cudaErrorInvalidValue;
  }
  if (binding.lattice_systems == 0) {
    return cudaSuccess;
  }
  constexpr int kThreads = 128;
  const int blocks = static_cast<int>((binding.lattice_systems + kThreads - 1) / kThreads);
  validate_lattice_request_kernel<<<blocks, kThreads, 0, stream>>>(binding);
  return cudaPeekAtLastError();
}

enum class TopologyMatch { kMatch, kMismatch, kInvalid };

bool valid_host_extent(const xtbloom_const_buffer_t& buffer, std::size_t required,
                       bool allow_absent = false) noexcept {
  if (required == 0u && buffer.data == nullptr && allow_absent) return true;
  return buffer.memory_space == XTBLOOM_MEMORY_HOST && buffer.reserved == 0u &&
         (required == 0u || buffer.data != nullptr) && buffer.size_bytes >= required;
}

template <typename T>
bool buffer_equals(const xtbloom_const_buffer_t& buffer, const std::vector<T>& expected) noexcept {
  const std::size_t bytes = expected.size() * sizeof(T);
  return valid_host_extent(buffer, bytes, expected.empty()) &&
         (bytes == 0u || std::memcmp(buffer.data, expected.data(), bytes) == 0);
}

bool double_buffer_equals(const xtbloom_const_buffer_t& buffer,
                          const std::vector<double>& expected) noexcept {
  const std::size_t bytes = expected.size() * sizeof(double);
  if (!valid_host_extent(buffer, bytes, expected.empty())) return false;
  const auto* source = static_cast<const std::byte*>(buffer.data);
  for (std::size_t index = 0; index < expected.size(); ++index) {
    double value = 0.0;
    std::memcpy(&value, source + index * sizeof(double), sizeof(double));
    if (value != expected[index]) return false;
  }
  return true;
}

template <typename T>
bool host_topology_buffer_equals(const xtbloom_const_buffer_t& buffer,
                                 const std::vector<T>& expected) noexcept {
  return buffer_equals(buffer, expected);
}

bool host_topology_buffer_equals(const xtbloom_const_buffer_t& buffer,
                                 const std::vector<double>& expected) noexcept {
  return double_buffer_equals(buffer, expected);
}

template <typename T>
bool validate_or_bind_fixed_topology_field(const char* name, const xtbloom_const_buffer_t& buffer,
                                           const std::vector<T>& expected, const T* expected_device,
                                           const T*& device_source, std::string& error) {
  device_source = nullptr;
  if (buffer.memory_space == XTBLOOM_MEMORY_HOST) {
    if (!host_topology_buffer_equals(buffer, expected)) {
      error = std::string(name) + " does not match the fixed CUDA plan topology";
      return false;
    }
    return true;
  }
  if (buffer.memory_space != XTBLOOM_MEMORY_CUDA_DEVICE ||
      (!expected.empty() && (buffer.data == nullptr || expected_device == nullptr))) {
    error = std::string(name) + " has no valid fixed CUDA plan comparison binding";
    return false;
  }
  device_source = static_cast<const T*>(buffer.data);
  return true;
}

bool spin_channel_buffer_equals(const xtbloom_batch_t& batch,
                                const std::vector<std::int32_t>& expected) noexcept {
  /* ABI-v1 and an empty ABI-v2 suffix both mean one restricted channel. */
  const bool supplied =
      batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
      (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);
  if (!supplied) {
    return std::all_of(expected.begin(), expected.end(),
                       [](std::int32_t channels) { return channels == 1; });
  }
  return buffer_equals(batch.spin_channels, expected);
}

bool finite_double_buffer(const xtbloom_const_buffer_t& buffer, std::int64_t elements,
                          bool positive = false, bool allow_absent = false) noexcept {
  std::size_t bytes = 0u;
  if (!checked_bytes(elements, sizeof(double), bytes) ||
      !valid_host_extent(buffer, bytes, allow_absent)) {
    return false;
  }
  const auto* source = static_cast<const std::byte*>(buffer.data);
  for (std::int64_t index = 0; index < elements; ++index) {
    double value = 0.0;
    std::memcpy(&value, source + static_cast<std::size_t>(index) * sizeof(double), sizeof(double));
    if (!std::isfinite(value) || (positive && value <= 0.0)) return false;
  }
  return true;
}

/* Allocation-free fixed-topology probe used before constructing any temporary
 * vectors. The public API performs structural validation before reaching this
 * owner; the checks repeated here keep the internal white-box entry point
 * fail-closed and ensure a reused call never hides malformed numerical views. */
TopologyMatch match_existing_topology(const xtbloom_batch_t& batch,
                                      const xtbloom_compute_options_t& options,
                                      const TopologyKey& key, std::string& error,
                                      bool validate_host_numerical = true) {
  const std::int64_t expected_batch = static_cast<std::int64_t>(key.molecular_charges.size());
  const std::int64_t expected_atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
  const std::int64_t expected_points = key.point_offsets.empty() ? 0 : key.point_offsets.back();
  if (batch.batch_size != expected_batch || batch.total_atoms != expected_atoms ||
      batch.total_point_charges != expected_points) {
    return TopologyMatch::kMismatch;
  }
  const bool periodic_enabled =
      batch.atomic_potential_shifts.data != nullptr || batch.total_charge_response_elements != 0 ||
      batch.charge_response_offsets.data != nullptr || batch.charge_response_matrix.data != nullptr;
  if (options.model != key.model || options.flags != key.flags ||
      options.max_scc_iterations != key.maximum_iterations ||
      options.charge_tolerance != key.charge_tolerance ||
      options.energy_tolerance != key.energy_tolerance ||
      options.electronic_temperature != key.electronic_temperature ||
      public_scc_mixer(options) != key.scc_mixer ||
      public_scc_mixer_history(options) != key.scc_mixer_history ||
      public_scc_mixer_damping(options) != key.scc_mixer_damping ||
      public_determinism(options) != key.determinism || periodic_enabled != key.periodic_enabled) {
    return TopologyMatch::kMismatch;
  }

  /* A native XYZ cell is part of the fixed topology.  An absent suffix is
   * equivalent to canonical all-NONE metadata only when the committed key has
   * no native periodic item; an active suffix is compared below even when all
   * of its masks are NONE so explicit and absent canonical forms remain
   * interchangeable. */
  const bool lattice_active = native_lattice_active(batch);
  if (key.native_lattice_enabled && !lattice_active) return TopologyMatch::kMismatch;
  if (lattice_active) {
    const std::size_t expected_cells = key.cell_matrices.size() * sizeof(double);
    const std::size_t expected_axes = key.periodic_axes.size() * sizeof(std::int32_t);
    if (!valid_host_extent(batch.cell_matrices, expected_cells) ||
        !valid_host_extent(batch.periodic_axes, expected_axes)) {
      error = "fixed-topology reuse received malformed native lattice descriptors";
      return TopologyMatch::kInvalid;
    }
    if (!double_buffer_equals(batch.cell_matrices, key.cell_matrices) ||
        !buffer_equals(batch.periodic_axes, key.periodic_axes)) {
      return TopologyMatch::kMismatch;
    }
  }

  if (!buffer_equals(batch.atom_offsets, key.atom_offsets) ||
      !buffer_equals(batch.atomic_numbers, key.atomic_numbers) ||
      !double_buffer_equals(batch.molecular_charges, key.molecular_charges) ||
      !buffer_equals(batch.unpaired_electrons, key.unpaired_electrons) ||
      !spin_channel_buffer_equals(batch, key.spin_channels)) {
    /* Distinguish a short/wrong-space descriptor from a legitimate topology
     * change so invalid reuse attempts cannot enter candidate construction. */
    std::size_t atom_offset_bytes = 0u;
    std::size_t atomic_number_bytes = 0u;
    std::size_t batch_double_bytes = 0u;
    std::size_t batch_integer_bytes = 0u;
    if (!checked_bytes(expected_batch + 1, sizeof(std::int64_t), atom_offset_bytes) ||
        !checked_bytes(expected_atoms, sizeof(std::int32_t), atomic_number_bytes) ||
        !checked_bytes(expected_batch, sizeof(double), batch_double_bytes) ||
        !checked_bytes(expected_batch, sizeof(std::int32_t), batch_integer_bytes) ||
        !valid_host_extent(batch.atom_offsets, atom_offset_bytes) ||
        !valid_host_extent(batch.atomic_numbers, atomic_number_bytes) ||
        !valid_host_extent(batch.molecular_charges, batch_double_bytes) ||
        !valid_host_extent(batch.unpaired_electrons, batch_integer_bytes)) {
      error = "fixed-topology reuse received a malformed host topology descriptor";
      return TopologyMatch::kInvalid;
    }
    const bool spin_supplied =
        batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
        (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);
    if (spin_supplied && !valid_host_extent(batch.spin_channels, batch_integer_bytes)) {
      error = "fixed-topology reuse received malformed host spin_channels";
      return TopologyMatch::kInvalid;
    }
    return TopologyMatch::kMismatch;
  }

  if (batch.point_charge_offsets.data != nullptr) {
    if (!buffer_equals(batch.point_charge_offsets, key.point_offsets)) {
      std::size_t bytes = 0u;
      if (!checked_bytes(expected_batch + 1, sizeof(std::int64_t), bytes) ||
          !valid_host_extent(batch.point_charge_offsets, bytes)) {
        error = "fixed-topology reuse received malformed point-charge offsets";
        return TopologyMatch::kInvalid;
      }
      return TopologyMatch::kMismatch;
    }
  } else if (expected_points != 0) {
    error = "fixed-topology reuse omitted required point-charge offsets";
    return TopologyMatch::kInvalid;
  }

  /* Response metadata is an all-or-nothing numerical view. Potential shifts
   * may enable the periodic component without a response matrix, but once any
   * response descriptor field is active it must retain the topology-derived
   * dense extent and canonical offsets. This mirrors public validation and
   * keeps the internal white-box entry point fail-closed as well. */
  const bool response_enabled =
      batch.total_charge_response_elements != 0 || batch.charge_response_offsets.data != nullptr ||
      batch.charge_response_offsets.size_bytes != 0u ||
      batch.charge_response_matrix.data != nullptr || batch.charge_response_matrix.size_bytes != 0u;
  if (response_enabled) {
    const std::int64_t expected_response_elements = key.response_offsets.back();
    if (batch.total_charge_response_elements != expected_response_elements ||
        batch.charge_response_offsets.data == nullptr ||
        batch.charge_response_matrix.data == nullptr ||
        !buffer_equals(batch.charge_response_offsets, key.response_offsets) ||
        (validate_host_numerical &&
         !finite_double_buffer(batch.charge_response_matrix, expected_response_elements))) {
      error = "fixed-topology reuse received incomplete or malformed dense charge-response data";
      return TopologyMatch::kInvalid;
    }
  }

  if (validate_host_numerical &&
      (!finite_double_buffer(batch.positions, expected_atoms * 3) ||
       !finite_double_buffer(batch.point_charge_positions, expected_points * 3, false, true) ||
       !finite_double_buffer(batch.point_charge_values, expected_points, false, true) ||
       !finite_double_buffer(batch.point_charge_gammas, expected_points, true, true) ||
       (batch.atomic_potential_shifts.data != nullptr &&
        !finite_double_buffer(batch.atomic_potential_shifts, expected_atoms)))) {
    error = "fixed-topology reuse received a malformed or nonfinite numerical host buffer";
    return TopologyMatch::kInvalid;
  }
  error.clear();
  return TopologyMatch::kMatch;
}

bool context_enqueue_host_topology_probe_available(const xtbloom_batch_t& batch) noexcept {
  const auto host_or_absent = [](const xtbloom_const_buffer_t& buffer) {
    return buffer.data == nullptr || buffer.memory_space == XTBLOOM_MEMORY_HOST;
  };
  return host_or_absent(batch.atom_offsets) && host_or_absent(batch.atomic_numbers) &&
         host_or_absent(batch.molecular_charges) && host_or_absent(batch.unpaired_electrons) &&
         (batch.struct_size < XTBLOOM_BATCH_V2_SIZE || host_or_absent(batch.spin_channels)) &&
         host_or_absent(batch.point_charge_offsets) &&
         host_or_absent(batch.charge_response_offsets) &&
         (batch.struct_size < XTBLOOM_BATCH_V4_SIZE ||
          (host_or_absent(batch.cell_matrices) && host_or_absent(batch.periodic_axes)));
}

bool context_enqueue_shape_policy_matches(const xtbloom_batch_t& batch,
                                          const xtbloom_compute_options_t& options,
                                          const TopologyKey& key) noexcept {
  const std::int64_t expected_batch = static_cast<std::int64_t>(key.molecular_charges.size());
  const std::int64_t expected_atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
  const std::int64_t expected_points = key.point_offsets.empty() ? 0 : key.point_offsets.back();
  const std::int64_t expected_response =
      key.response_offsets.empty() ? 0 : key.response_offsets.back();
  const bool response_active =
      batch.total_charge_response_elements != 0 || batch.charge_response_offsets.data != nullptr ||
      batch.charge_response_offsets.size_bytes != 0u ||
      batch.charge_response_matrix.data != nullptr || batch.charge_response_matrix.size_bytes != 0u;
  const bool periodic_enabled = batch.atomic_potential_shifts.data != nullptr ||
                                batch.atomic_potential_shifts.size_bytes != 0u || response_active;
  /* Device native-cell leaves are checked in stream order.  A committed native
   * key cannot be treated as molecular merely because the request omitted the
   * optional V4 suffix; the reverse transition remains admissible and is
   * classified by the fixed-topology comparison. */
  const bool lattice_active = native_lattice_active(batch);
  if (key.native_lattice_enabled && !lattice_active) return false;
  return batch.batch_size == expected_batch && batch.total_atoms == expected_atoms &&
         batch.total_point_charges == expected_points &&
         (!response_active || batch.total_charge_response_elements == expected_response) &&
         options.model == key.model && options.flags == key.flags &&
         options.max_scc_iterations == key.maximum_iterations &&
         options.charge_tolerance == key.charge_tolerance &&
         options.energy_tolerance == key.energy_tolerance &&
         options.electronic_temperature == key.electronic_temperature &&
         public_scc_mixer(options) == key.scc_mixer &&
         public_scc_mixer_history(options) == key.scc_mixer_history &&
         public_scc_mixer_damping(options) == key.scc_mixer_damping &&
         public_determinism(options) == key.determinism && periodic_enabled == key.periodic_enabled;
}

xtbloom_status_t validate_offsets(const char* name, const std::vector<std::int64_t>& offsets,
                                  std::int64_t batch_size, std::int64_t total,
                                  bool require_nonempty, std::string& error) {
  if (offsets.size() != static_cast<std::size_t>(batch_size + 1) || offsets.front() != 0 ||
      offsets.back() != total) {
    error = std::string(name) + " does not delimit the declared ragged batch";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t begin = offsets[static_cast<std::size_t>(system)];
    const std::int64_t end = offsets[static_cast<std::size_t>(system + 1)];
    if (end < begin || (require_nonempty && end == begin)) {
      error = std::string(name) + " is not monotone or contains an empty molecule";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

/* Populate the host-only preparation key with the same canonical native-cell
 * representation produced by CUDA topology staging.  prepare_host() reaches
 * this helper directly, so it cannot rely on the device validation kernel to
 * reject malformed cells or to normalize signed zero. */
xtbloom_status_t populate_host_native_lattice_key(const xtbloom_batch_t& batch, TopologyKey& key,
                                                  std::string& error) {
  std::int64_t cell_elements = 0;
  if (!checked_elements(batch.batch_size, 9, cell_elements) || batch.batch_size <= 0) {
    error = "native lattice extent overflows int64_t";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  key.cell_matrices.assign(static_cast<std::size_t>(cell_elements), 0.0);
  key.periodic_axes.assign(static_cast<std::size_t>(batch.batch_size), XTBLOOM_PERIODIC_AXES_NONE);
  key.native_lattice_enabled = false;
  if (!native_lattice_active(batch)) return XTBLOOM_STATUS_SUCCESS;

  xtbloom_status_t status = copy_host_buffer("cell_matrices", batch.cell_matrices, cell_elements,
                                             key.cell_matrices, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("periodic_axes", batch.periodic_axes, batch.batch_size,
                            key.periodic_axes, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  for (std::int64_t system = 0; system < batch.batch_size; ++system) {
    const std::size_t cell_offset = static_cast<std::size_t>(system) * 9u;
    const std::int32_t mask = key.periodic_axes[static_cast<std::size_t>(system)];
    double* const cell = key.cell_matrices.data() + cell_offset;
    if (mask == XTBLOOM_PERIODIC_AXES_NONE) {
      for (int element = 0; element < 9; ++element) {
        if (cell[element] != 0.0) {
          error = "a nonperiodic batch item must use an all-zero cell matrix";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        cell[element] = 0.0;
      }
      continue;
    }
    if ((mask & ~XTBLOOM_PERIODIC_AXES_XYZ) != 0) {
      error = "periodic_axes contains unknown mask bits";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (mask != XTBLOOM_PERIODIC_AXES_XYZ) {
      error = "one- and two-dimensional periodic axes are reserved but not supported";
      return XTBLOOM_STATUS_NOT_SUPPORTED;
    }
    if (!gfn2::valid_lattice_cell_3d_binary64(cell)) {
      error = "a periodic cell must be finite, right-handed, and nonsingular";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (int element = 0; element < 9; ++element) {
      if (cell[element] == 0.0) cell[element] = 0.0;
    }
    key.native_lattice_enabled = true;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t make_topology_key(const xtbloom_batch_t& batch,
                                   const xtbloom_compute_options_t& options, TopologyKey& key,
                                   std::vector<double>& positions,
                                   std::vector<double>& point_positions,
                                   std::vector<double>& point_values,
                                   std::vector<double>& point_gammas,
                                   std::vector<double>& periodic_shifts,
                                   std::vector<double>& periodic_response, std::string& error) {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_point_charges < 0 ||
      batch.total_charge_response_elements < 0 ||
      (options.model != XTBLOOM_MODEL_GFN1_XTB && options.model != XTBLOOM_MODEL_GFN2_XTB) ||
      options.max_scc_iterations <= 0 || !std::isfinite(options.charge_tolerance) ||
      options.charge_tolerance <= 0.0 || !std::isfinite(options.energy_tolerance) ||
      options.energy_tolerance <= 0.0 || !std::isfinite(options.electronic_temperature) ||
      options.electronic_temperature < 0.0) {
    error = "invalid CUDA xTB setup dimensions or compute policy";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  xtbloom_status_t status = validate_public_execution_policy(options, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  key.model = options.model;
  status = copy_host_buffer("atom_offsets", batch.atom_offsets, batch.batch_size + 1,
                            key.atom_offsets, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = validate_offsets("atom_offsets", key.atom_offsets, batch.batch_size, batch.total_atoms,
                            true, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("atomic_numbers", batch.atomic_numbers, batch.total_atoms,
                            key.atomic_numbers, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("positions", batch.positions, batch.total_atoms * 3, positions, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("molecular_charges", batch.molecular_charges, batch.batch_size,
                            key.molecular_charges, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("unpaired_electrons", batch.unpaired_electrons, batch.batch_size,
                            key.unpaired_electrons, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const bool spin_channels_present =
      batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
      (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);
  if (spin_channels_present) {
    status = copy_host_buffer("spin_channels", batch.spin_channels, batch.batch_size,
                              key.spin_channels, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    key.spin_channels.assign(static_cast<std::size_t>(batch.batch_size), 1);
  }

  for (std::int64_t atom = 0; atom < batch.total_atoms; ++atom) {
    const std::int32_t atomic_number = key.atomic_numbers[static_cast<std::size_t>(atom)];
    if (atomic_number <= 0 || atomic_number > 118) {
      error = "atomic_numbers contains an unsupported element";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t system = 0; system < batch.batch_size; ++system) {
    if (!std::isfinite(key.molecular_charges[static_cast<std::size_t>(system)])) {
      error = "molecular_charges contains a nonfinite value";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::int32_t channels = key.spin_channels[static_cast<std::size_t>(system)];
    if (channels != 1 && channels != 2) {
      error = "spin_channels values must be one or two";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (const double coordinate : positions) {
    if (!std::isfinite(coordinate)) {
      error = "positions contains a nonfinite value";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  if (batch.total_point_charges != 0 || batch.point_charge_offsets.data != nullptr) {
    status = copy_host_buffer("point_charge_offsets", batch.point_charge_offsets,
                              batch.batch_size + 1, key.point_offsets, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_offsets("point_charge_offsets", key.point_offsets, batch.batch_size,
                              batch.total_point_charges, false, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    key.point_offsets.assign(static_cast<std::size_t>(batch.batch_size + 1), 0);
  }
  status = copy_host_buffer("point_charge_positions", batch.point_charge_positions,
                            batch.total_point_charges * 3, point_positions, error, true);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("point_charge_values", batch.point_charge_values,
                            batch.total_point_charges, point_values, error, true);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = copy_host_buffer("point_charge_gammas", batch.point_charge_gammas,
                            batch.total_point_charges, point_gammas, error, true);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  for (const double value : point_positions) {
    if (!std::isfinite(value)) {
      error = "point_charge_positions contains a nonfinite value";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t point = 0; point < point_values.size(); ++point) {
    if (!std::isfinite(point_values[point]) || !std::isfinite(point_gammas[point]) ||
        point_gammas[point] <= 0.0) {
      error = "point charge values must be finite and gammas must be finite and positive";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  key.periodic_enabled =
      batch.atomic_potential_shifts.data != nullptr || batch.total_charge_response_elements != 0 ||
      batch.charge_response_offsets.data != nullptr || batch.charge_response_matrix.data != nullptr;
  if (batch.atomic_potential_shifts.data != nullptr) {
    status = copy_host_buffer("atomic_potential_shifts", batch.atomic_potential_shifts,
                              batch.total_atoms, periodic_shifts, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    periodic_shifts.assign(static_cast<std::size_t>(batch.total_atoms), 0.0);
  }

  std::vector<std::int64_t> expected_response_offsets(
      static_cast<std::size_t>(batch.batch_size + 1), 0);
  for (std::int64_t system = 0; system < batch.batch_size; ++system) {
    const std::int64_t atoms = key.atom_offsets[static_cast<std::size_t>(system + 1)] -
                               key.atom_offsets[static_cast<std::size_t>(system)];
    if (atoms > 0 && atoms > std::numeric_limits<std::int64_t>::max() / atoms) {
      error = "periodic response extent overflows int64_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::int64_t elements = atoms * atoms;
    const std::int64_t previous = expected_response_offsets[static_cast<std::size_t>(system)];
    if (elements > std::numeric_limits<std::int64_t>::max() - previous) {
      error = "periodic response prefix sum overflows int64_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    expected_response_offsets[static_cast<std::size_t>(system + 1)] = previous + elements;
  }
  if (batch.total_charge_response_elements != 0 || batch.charge_response_offsets.data != nullptr ||
      batch.charge_response_matrix.data != nullptr) {
    status = copy_host_buffer("charge_response_offsets", batch.charge_response_offsets,
                              batch.batch_size + 1, key.response_offsets, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    if (key.response_offsets != expected_response_offsets ||
        batch.total_charge_response_elements != expected_response_offsets.back()) {
      error = "charge_response_offsets does not match the dense per-system atom layout";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    status = copy_host_buffer("charge_response_matrix", batch.charge_response_matrix,
                              batch.total_charge_response_elements, periodic_response, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    key.response_offsets = expected_response_offsets;
    periodic_response.assign(static_cast<std::size_t>(expected_response_offsets.back()), 0.0);
  }
  for (const double value : periodic_shifts) {
    if (!std::isfinite(value)) {
      error = "atomic_potential_shifts contains a nonfinite value";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (const double value : periodic_response) {
    if (!std::isfinite(value)) {
      error = "charge_response_matrix contains a nonfinite value";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  status = populate_host_native_lattice_key(batch, key, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  key.flags = options.flags;
  key.maximum_iterations = options.max_scc_iterations;
  key.charge_tolerance = options.charge_tolerance;
  key.energy_tolerance = options.energy_tolerance;
  key.electronic_temperature = options.electronic_temperature;
  key.scc_mixer = public_scc_mixer(options);
  key.scc_mixer_history = public_scc_mixer_history(options);
  key.scc_mixer_damping = public_scc_mixer_damping(options);
  key.determinism = public_determinism(options);
  return XTBLOOM_STATUS_SUCCESS;
}

/*
 * Build topology-derived host plans without downloading caller numerical
 * buffers.  The deterministic seed is used only while constructing stable
 * owners, descriptors, arenas, and provider workspaces.  A public CUDA
 * transaction must refresh the resulting Prepared candidate from the real
 * caller buffers before it may replace the committed runtime.
 */
xtbloom_status_t make_topology_only_seed(
    const Gfn2CudaTopologyHostSnapshot& snapshot, const xtbloom_compute_options_t& options,
    TopologyKey& key, std::vector<double>& positions, std::vector<double>& point_positions,
    std::vector<double>& point_values, std::vector<double>& point_gammas,
    std::vector<double>& periodic_shifts, std::vector<double>& periodic_response,
    std::string& error) {
  if (snapshot.batch_size <= 0 || snapshot.total_atoms < snapshot.batch_size ||
      snapshot.total_point_charges < 0 ||
      snapshot.atom_offsets.size() != static_cast<std::size_t>(snapshot.batch_size + 1) ||
      snapshot.atomic_numbers.size() != static_cast<std::size_t>(snapshot.total_atoms) ||
      snapshot.molecular_charges.size() != static_cast<std::size_t>(snapshot.batch_size) ||
      snapshot.unpaired_electrons.size() != static_cast<std::size_t>(snapshot.batch_size) ||
      snapshot.spin_channels.size() != static_cast<std::size_t>(snapshot.batch_size) ||
      snapshot.point_charge_offsets.size() != static_cast<std::size_t>(snapshot.batch_size + 1) ||
      snapshot.charge_response_offsets.size() !=
          static_cast<std::size_t>(snapshot.batch_size + 1)) {
    error = "CUDA topology staging returned an inconsistent host snapshot";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  xtbloom_status_t status = validate_public_execution_policy(options, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  /* Device/mixed descriptors arrive through topology staging instead of
   * make_topology_key(). Preserve the selected model here as part of the
   * immutable cache identity so a GFN1 request can never reuse or construct
   * a GFN2 plan by default. */
  key.model = options.model;
  key.atom_offsets = snapshot.atom_offsets;
  key.atomic_numbers = snapshot.atomic_numbers;
  key.molecular_charges = snapshot.molecular_charges;
  key.unpaired_electrons = snapshot.unpaired_electrons;
  key.spin_channels = snapshot.spin_channels;
  key.point_offsets = snapshot.point_charge_offsets;
  key.response_offsets = snapshot.charge_response_offsets;
  key.flags = options.flags;
  key.maximum_iterations = options.max_scc_iterations;
  key.charge_tolerance = options.charge_tolerance;
  key.energy_tolerance = options.energy_tolerance;
  key.electronic_temperature = options.electronic_temperature;
  key.scc_mixer = public_scc_mixer(options);
  key.scc_mixer_history = public_scc_mixer_history(options);
  key.scc_mixer_damping = public_scc_mixer_damping(options);
  key.determinism = public_determinism(options);
  key.periodic_enabled = snapshot.periodic_enabled;
  key.native_lattice_enabled = snapshot.native_lattice_enabled;
  key.cell_matrices = snapshot.cell_matrices;
  key.periodic_axes = snapshot.periodic_axes;

  std::int64_t coordinate_elements = 0;
  std::int64_t point_coordinate_elements = 0;
  if (!checked_elements(snapshot.total_atoms, 3, coordinate_elements) ||
      !checked_elements(snapshot.total_point_charges, 3, point_coordinate_elements) ||
      snapshot.total_charge_response_elements < 0) {
    error = "CUDA topology-only seed extent overflows int64_t";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
  try {
    positions.assign(static_cast<std::size_t>(coordinate_elements), 0.0);
    for (std::int64_t system = 0; system < snapshot.batch_size; ++system) {
      const std::int64_t begin = snapshot.atom_offsets[static_cast<std::size_t>(system)];
      const std::int64_t end = snapshot.atom_offsets[static_cast<std::size_t>(system + 1)];
      for (std::int64_t atom = begin; atom < end; ++atom) {
        const std::int64_t local = atom - begin;
        positions[static_cast<std::size_t>(3 * atom)] = 8.0 * static_cast<double>(local);
      }
    }
    point_positions.assign(static_cast<std::size_t>(point_coordinate_elements), 0.0);
    point_values.assign(static_cast<std::size_t>(snapshot.total_point_charges), 0.0);
    point_gammas.assign(static_cast<std::size_t>(snapshot.total_point_charges), 1.0);
    periodic_shifts.assign(static_cast<std::size_t>(snapshot.total_atoms), 0.0);
    periodic_response.assign(static_cast<std::size_t>(snapshot.total_charge_response_elements),
                             0.0);
  } catch (const std::bad_alloc&) {
    error = "failed to allocate CUDA topology-only numerical seed";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

bool topology_snapshot_matches(const Gfn2CudaTopologyHostSnapshot& snapshot,
                               const xtbloom_compute_options_t& options,
                               const TopologyKey& key) noexcept {
  return options.model == key.model && snapshot.atom_offsets == key.atom_offsets &&
         snapshot.atomic_numbers == key.atomic_numbers &&
         snapshot.molecular_charges == key.molecular_charges &&
         snapshot.unpaired_electrons == key.unpaired_electrons &&
         snapshot.spin_channels == key.spin_channels &&
         snapshot.point_charge_offsets == key.point_offsets &&
         snapshot.charge_response_offsets == key.response_offsets &&
         snapshot.periodic_enabled == key.periodic_enabled && options.flags == key.flags &&
         snapshot.native_lattice_enabled == key.native_lattice_enabled &&
         snapshot.cell_matrices == key.cell_matrices &&
         snapshot.periodic_axes == key.periodic_axes &&
         options.max_scc_iterations == key.maximum_iterations &&
         options.charge_tolerance == key.charge_tolerance &&
         options.energy_tolerance == key.energy_tolerance &&
         options.electronic_temperature == key.electronic_temperature &&
         public_scc_mixer(options) == key.scc_mixer &&
         public_scc_mixer_history(options) == key.scc_mixer_history &&
         public_scc_mixer_damping(options) == key.scc_mixer_damping &&
         public_determinism(options) == key.determinism;
}

struct HostPlanByteBreakdown {
  std::size_t common_plan_vectors = 0u;
  std::size_t gfn1_plan_vectors = 0u;
  std::size_t common_model_plans = 0u;
  std::size_t gfn1_model_plans = 0u;
  std::size_t numerical_vectors = 0u;
  std::size_t gfn1_expanded_parameters = 0u;
  std::size_t gfn2_wavefunction_arena = 0u;
  std::size_t gfn1_wavefunction_arena = 0u;

  [[nodiscard]] std::size_t total() const noexcept {
    return common_plan_vectors + gfn1_plan_vectors + common_model_plans + gfn1_model_plans +
           numerical_vectors + gfn1_expanded_parameters + gfn2_wavefunction_arena +
           gfn1_wavefunction_arena;
  }
};

struct HostPlans {
  TopologyKey key;
  std::uint64_t fingerprint = 0u;
  std::uint64_t plan_token = 0u;
  std::uint64_t geometry_generation = kInitialGeometryGeneration;

  BasisPlan basis;
  IntegralPlan integrals;
  CoordinationPlan coordination;
  RepulsionPlan repulsion;
  H0Plan h0;
  Gfn2IntegralHostTaskDomains integral_task_domains;
  WavefunctionLayout wavefunction_layout;
  ES2Plan es2;
  ES3Plan es3;
  AES2Plan aes2;
  MullikenPlan mulliken;
  EigensolverPlan eigensolver;
  SccMixerPlan mixer;
  D4Plan d4;
  ExternalPointChargePlan external;
  PeriodicEmbeddingPlan periodic;
  SccDriverPlan driver;

  /* GFN1 owns scalar SCC plans that cannot be represented by the GFN2
   * multipole types. Common CUDA topology/linear-algebra descriptors are
   * still reused, while these plans remain the scientific source of truth. */
  gfn1::H0Plan gfn1_h0;
  gfn1::WavefunctionLayout gfn1_wavefunction_layout;
  gfn1::ES3Plan gfn1_es3;
  gfn1::SpinPopulationLayout gfn1_spin_layout;
  gfn1::SpinPolarizationPlan gfn1_spin;
  gfn1::MullikenPlan gfn1_mulliken;
  gfn1::SccMixerPlan gfn1_mixer;
  gfn1::ExternalPointChargePlan gfn1_external;
  gfn1::SccDriverPlan gfn1_driver;
  gfn1::RepulsionPlan gfn1_repulsion;
  gfn1::D3Plan gfn1_d3;
  gfn1::HalogenPlan gfn1_halogen;
  std::vector<std::uint8_t> gfn1_d3_reference_counts;
  std::vector<double> gfn1_d3_reference_cn;
  std::vector<double> gfn1_d3_reference_c6;
  std::vector<double> gfn1_d3_pair_rrij;
  std::vector<double> gfn1_d3_pair_damping_radii;
  std::vector<double> gfn1_halogen_scaled_radii;
  std::vector<double> gfn1_halogen_bond_strength;
  std::vector<std::uint8_t> gfn1_halogen_donor;
  std::vector<std::uint8_t> gfn1_halogen_acceptor;
  PinnedArena gfn1_wavefunction_storage;
  gfn1::WavefunctionView gfn1_wavefunction{};

  bool d4_enabled = false;
  bool periodic_enabled = false;
  bool gfn1_enabled = false;
  bool compact_integral_tasks = true;

  std::vector<double> positions;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response;

  std::vector<double> coordination_numbers;
  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<double> core_hamiltonian;
  std::vector<double> integral_workspace;

  std::vector<double> geometry_pair_data;
  std::vector<std::uint64_t> geometry_generations;

  std::vector<double> es2_matrix;
  std::vector<double> es2_matrix_scratch;
  std::vector<double> es2_shell_scratch;
  std::vector<double> es2_batch_scratch;
  std::vector<double> es2_gradient_scratch;
  ES2Workspace es2_workspace{};
  ES2GeometryCache es2_cache{};

  std::vector<double> aes2_pairs;
  std::vector<double> aes2_pair_scratch;
  std::vector<double> aes2_potential_scratch;
  std::vector<double> aes2_batch_scratch;
  std::vector<double> aes2_gradient_scratch;
  std::vector<double> aes2_coordination_scratch;
  AES2Workspace aes2_workspace{};
  AES2GeometryCache aes2_cache{};

  std::vector<Gfn2D4DeviceElementData> d4_elements;
  std::vector<Gfn2D4DeviceReferenceData> d4_references;
  std::vector<double> d4_coordination;

  std::vector<double> explicit_point_shell_potential;
  PinnedArena wavefunction_storage;
  WavefunctionView wavefunction{};

  [[nodiscard]] HostPlanByteBreakdown retained_host_byte_breakdown() const noexcept {
    const std::size_t common_plan_vectors =
        vector_bytes(key.atom_offsets) + vector_bytes(key.atomic_numbers) +
        vector_bytes(key.molecular_charges) + vector_bytes(key.unpaired_electrons) +
        vector_bytes(key.spin_channels) + vector_bytes(key.point_offsets) +
        vector_bytes(key.response_offsets) + vector_bytes(key.cell_matrices) +
        vector_bytes(key.periodic_axes) + common::basis_plan_resident_bytes(basis) +
        vector_bytes(integrals.matrix_offsets) + vector_bytes(coordination.atom_offsets) +
        vector_bytes(coordination.covalent_radius) + vector_bytes(repulsion.atom_offsets) +
        vector_bytes(repulsion.sqrt_alpha) + vector_bytes(repulsion.effective_charge) +
        vector_bytes(repulsion.light_element) + vector_bytes(h0.atom_offsets) +
        vector_bytes(h0.batch_shell_offsets) + vector_bytes(h0.batch_orbital_offsets) +
        vector_bytes(h0.matrix_offsets) + vector_bytes(h0.shell_pair_offsets) +
        vector_bytes(h0.atomic_radii) + vector_bytes(h0.shell_levels) +
        vector_bytes(h0.shell_coordination_scale) + vector_bytes(h0.shell_polynomial) +
        vector_bytes(h0.shell_pair_scale) + vector_bytes(integral_task_domains.forward_generic) +
        vector_bytes(integral_task_domains.forward_ss) +
        vector_bytes(integral_task_domains.h0_generic) + vector_bytes(integral_task_domains.h0_ss) +
        vector_bytes(integral_task_domains.force_generic) +
        vector_bytes(integral_task_domains.force_ss) +
        vector_bytes(integral_task_domains.accounting.primitive_signatures) +
        vector_bytes(es3.batch_shell_offsets) + vector_bytes(es3.shell_gamma3) +
        vector_bytes(external.atom_offsets) + vector_bytes(external.batch_shell_offsets) +
        vector_bytes(external.point_charge_offsets) + vector_bytes(external.shell_to_atom) +
        vector_bytes(external.shell_hardness) + vector_bytes(wavefunction_layout.atom_offsets) +
        vector_bytes(wavefunction_layout.batch_shell_offsets) +
        vector_bytes(wavefunction_layout.batch_orbital_offsets) +
        vector_bytes(wavefunction_layout.atomic_numbers) +
        vector_bytes(wavefunction_layout.molecular_charges) +
        vector_bytes(wavefunction_layout.unpaired_electrons) +
        vector_bytes(wavefunction_layout.spin_channels) +
        vector_bytes(wavefunction_layout.reference_atom_occupations) +
        vector_bytes(wavefunction_layout.reference_shell_occupations) +
        vector_bytes(wavefunction_layout.electron_counts) +
        vector_bytes(wavefunction_layout.alpha_electron_counts) +
        vector_bytes(wavefunction_layout.beta_electron_counts) +
        vector_bytes(wavefunction_layout.coefficients.system_offsets) +
        vector_bytes(wavefunction_layout.eigenvalues.system_offsets) +
        vector_bytes(wavefunction_layout.occupations.system_offsets) +
        vector_bytes(wavefunction_layout.density.system_offsets) +
        vector_bytes(wavefunction_layout.qsh.system_offsets) +
        vector_bytes(wavefunction_layout.qat.system_offsets) +
        vector_bytes(wavefunction_layout.dipole.system_offsets) +
        vector_bytes(wavefunction_layout.quadrupole.system_offsets) +
        vector_bytes(wavefunction_layout.energy_weighted_density.system_offsets);
    const std::size_t gfn1_plan_vectors =
        vector_bytes(gfn1_h0.atom_offsets) + vector_bytes(gfn1_h0.batch_shell_offsets) +
        vector_bytes(gfn1_h0.batch_orbital_offsets) + vector_bytes(gfn1_h0.matrix_offsets) +
        vector_bytes(gfn1_h0.shell_pair_offsets) + vector_bytes(gfn1_h0.atomic_radii) +
        vector_bytes(gfn1_h0.shell_levels) + vector_bytes(gfn1_h0.shell_coordination_scale) +
        vector_bytes(gfn1_h0.shell_polynomial) + vector_bytes(gfn1_h0.shell_pair_scale) +
        vector_bytes(gfn1_wavefunction_layout.atom_offsets) +
        vector_bytes(gfn1_wavefunction_layout.batch_shell_offsets) +
        vector_bytes(gfn1_wavefunction_layout.batch_orbital_offsets) +
        vector_bytes(gfn1_wavefunction_layout.atomic_numbers) +
        vector_bytes(gfn1_wavefunction_layout.molecular_charges) +
        vector_bytes(gfn1_wavefunction_layout.unpaired_electrons) +
        vector_bytes(gfn1_wavefunction_layout.spin_channels) +
        vector_bytes(gfn1_wavefunction_layout.reference_atom_occupations) +
        vector_bytes(gfn1_wavefunction_layout.reference_shell_occupations) +
        vector_bytes(gfn1_wavefunction_layout.electron_counts) +
        vector_bytes(gfn1_wavefunction_layout.alpha_electron_counts) +
        vector_bytes(gfn1_wavefunction_layout.beta_electron_counts) +
        vector_bytes(gfn1_wavefunction_layout.coefficients.system_offsets) +
        vector_bytes(gfn1_wavefunction_layout.eigenvalues.system_offsets) +
        vector_bytes(gfn1_wavefunction_layout.occupations.system_offsets) +
        vector_bytes(gfn1_wavefunction_layout.density.system_offsets) +
        vector_bytes(gfn1_wavefunction_layout.qsh.system_offsets) +
        vector_bytes(gfn1_wavefunction_layout.qat.system_offsets) +
        vector_bytes(gfn1_wavefunction_layout.energy_weighted_density.system_offsets) +
        vector_bytes(gfn1_es3.atom_offsets) + vector_bytes(gfn1_es3.atom_gamma3) +
        vector_bytes(gfn1_spin_layout.system_offsets) +
        vector_bytes(gfn1_spin_layout.spin_channels) + vector_bytes(gfn1_spin.atom_offsets) +
        vector_bytes(gfn1_spin.batch_shell_offsets) + vector_bytes(gfn1_spin.atom_shell_offsets) +
        vector_bytes(gfn1_spin.shell_population_offsets) + vector_bytes(gfn1_spin.spin_channels) +
        vector_bytes(gfn1_spin.coupling_offsets) + vector_bytes(gfn1_spin.coupling_matrices) +
        vector_bytes(gfn1_external.atom_offsets) + vector_bytes(gfn1_external.batch_shell_offsets) +
        vector_bytes(gfn1_external.atom_shell_offsets) +
        vector_bytes(gfn1_external.point_charge_offsets) +
        vector_bytes(gfn1_external.atom_to_batch) + vector_bytes(gfn1_external.point_to_batch) +
        vector_bytes(gfn1_external.shell_to_atom) + vector_bytes(gfn1_external.shell_hardness) +
        vector_bytes(gfn1_repulsion.atom_offsets) + vector_bytes(gfn1_repulsion.sqrt_alpha) +
        vector_bytes(gfn1_repulsion.effective_charge);
    const std::size_t common_model_plan_storage =
        es2.resident_bytes() + aes2.resident_bytes() + mulliken.resident_bytes() +
        eigensolver.resident_bytes() + mixer.resident_bytes() + d4.resident_bytes() +
        periodic.resident_bytes() + driver.resident_bytes();
    /* The GFN1 driver counts its copied scalar layouts but deliberately
     * excludes shared opaque subplans, so each owner below is counted once. */
    const std::size_t gfn1_model_plan_storage =
        gfn1_mulliken.resident_bytes() + gfn1_mixer.resident_bytes() +
        gfn1_driver.resident_bytes() + gfn1_d3.resident_bytes() + gfn1_halogen.resident_bytes();
    const std::size_t numerical_vectors =
        vector_bytes(positions) + vector_bytes(point_positions) + vector_bytes(point_values) +
        vector_bytes(point_gammas) + vector_bytes(periodic_shifts) +
        vector_bytes(periodic_response) + vector_bytes(coordination_numbers) +
        vector_bytes(overlap) + vector_bytes(dipole_integrals) +
        vector_bytes(quadrupole_integrals) + vector_bytes(core_hamiltonian) +
        vector_bytes(integral_workspace) + vector_bytes(geometry_pair_data) +
        vector_bytes(geometry_generations) + vector_bytes(es2_matrix) +
        vector_bytes(es2_matrix_scratch) + vector_bytes(es2_shell_scratch) +
        vector_bytes(es2_batch_scratch) + vector_bytes(es2_gradient_scratch) +
        vector_bytes(aes2_pairs) + vector_bytes(aes2_pair_scratch) +
        vector_bytes(aes2_potential_scratch) + vector_bytes(aes2_batch_scratch) +
        vector_bytes(aes2_gradient_scratch) + vector_bytes(aes2_coordination_scratch) +
        vector_bytes(d4_elements) + vector_bytes(d4_references) + vector_bytes(d4_coordination) +
        vector_bytes(explicit_point_shell_potential);
    const std::size_t gfn1_expanded_parameter_vectors =
        vector_bytes(gfn1_d3_reference_counts) + vector_bytes(gfn1_d3_reference_cn) +
        vector_bytes(gfn1_d3_reference_c6) + vector_bytes(gfn1_d3_pair_rrij) +
        vector_bytes(gfn1_d3_pair_damping_radii) + vector_bytes(gfn1_halogen_scaled_radii) +
        vector_bytes(gfn1_halogen_bond_strength) + vector_bytes(gfn1_halogen_donor) +
        vector_bytes(gfn1_halogen_acceptor);
    return {common_plan_vectors,
            gfn1_plan_vectors,
            common_model_plan_storage,
            gfn1_model_plan_storage,
            numerical_vectors,
            gfn1_expanded_parameter_vectors,
            wavefunction_storage.bytes(),
            gfn1_wavefunction_storage.bytes()};
  }

  [[nodiscard]] std::size_t retained_host_bytes() const noexcept {
    return retained_host_byte_breakdown().total();
  }

  xtbloom_status_t build(TopologyKey&& new_key, std::vector<double>&& new_positions,
                         std::vector<double>&& new_point_positions,
                         std::vector<double>&& new_point_values,
                         std::vector<double>&& new_point_gammas,
                         std::vector<double>&& new_periodic_shifts,
                         std::vector<double>&& new_periodic_response, std::uint64_t token,
                         std::string& error) {
    if (new_key.model == XTBLOOM_MODEL_GFN1_XTB) {
      return build_gfn1(std::move(new_key), std::move(new_positions),
                        std::move(new_point_positions), std::move(new_point_values),
                        std::move(new_point_gammas), std::move(new_periodic_shifts),
                        std::move(new_periodic_response), token, error);
    }
    return build_gfn2(std::move(new_key), std::move(new_positions), std::move(new_point_positions),
                      std::move(new_point_values), std::move(new_point_gammas),
                      std::move(new_periodic_shifts), std::move(new_periodic_response), token,
                      error);
  }

  xtbloom_status_t build_gfn1(TopologyKey&& new_key, std::vector<double>&& new_positions,
                              std::vector<double>&& new_point_positions,
                              std::vector<double>&& new_point_values,
                              std::vector<double>&& new_point_gammas,
                              std::vector<double>&& new_periodic_shifts,
                              std::vector<double>&& new_periodic_response, std::uint64_t token,
                              std::string& error);

  xtbloom_status_t build_gfn2(TopologyKey&& new_key, std::vector<double>&& new_positions,
                              std::vector<double>&& new_point_positions,
                              std::vector<double>&& new_point_values,
                              std::vector<double>&& new_point_gammas,
                              std::vector<double>&& new_periodic_shifts,
                              std::vector<double>&& new_periodic_response, std::uint64_t token,
                              std::string& error) {
    key = std::move(new_key);
    positions = std::move(new_positions);
    point_positions = std::move(new_point_positions);
    point_values = std::move(new_point_values);
    point_gammas = std::move(new_point_gammas);
    periodic_shifts = std::move(new_periodic_shifts);
    periodic_response = std::move(new_periodic_response);
    fingerprint = key.fingerprint();
    plan_token = token;
    gfn1_enabled = false;

    const std::int64_t batch = static_cast<std::int64_t>(key.molecular_charges.size());
    const std::int64_t atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
    const std::int64_t points = static_cast<std::int64_t>(point_values.size());
    xtbloom_status_t status = make_basis_plan(batch, atoms, key.atom_offsets.data(),
                                              key.atomic_numbers.data(), basis, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_integral_plan(basis, integrals, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_coordination_plan(batch, atoms, key.atom_offsets.data(),
                                    key.atomic_numbers.data(), coordination, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_repulsion_plan(batch, atoms, key.atom_offsets.data(), key.atomic_numbers.data(),
                                 repulsion, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_h0_plan(basis, integrals, key.atomic_numbers.data(), h0, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        make_gfn2_integral_task_domains(basis, h0.shell_pair_offsets, integral_task_domains, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = configured_integral_task_schedule(integral_task_domains.accounting,
                                               compact_integral_tasks, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_wavefunction_layout(basis, key.atomic_numbers.data(),
                                      key.molecular_charges.data(), key.unpaired_electrons.data(),
                                      key.spin_channels.data(), wavefunction_layout, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_es2_plan(basis, key.atomic_numbers.data(), es2, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_es3_plan(basis, key.atomic_numbers.data(), es3, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_aes2_plan(basis, key.atomic_numbers.data(), aes2, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_mulliken_plan(basis, integrals, wavefunction_layout, mulliken, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_eigensolver_plan(wavefunction_layout, eigensolver, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = make_scc_mixer_plan(wavefunction_layout, key.scc_mixer_history, key.scc_mixer_damping,
                                 key.charge_tolerance, key.charge_tolerance, mixer, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    d4_enabled = false;
    for (std::int64_t system = 0; system < batch; ++system) {
      d4_enabled = d4_enabled || key.atom_offsets[static_cast<std::size_t>(system + 1)] -
                                         key.atom_offsets[static_cast<std::size_t>(system)] >
                                     1;
    }
    if (d4_enabled) {
      status =
          make_d4_plan(batch, atoms, key.atom_offsets.data(), key.atomic_numbers.data(), d4, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }
    status = make_external_point_charge_plan(basis, key.atomic_numbers.data(), points,
                                             points == 0 ? nullptr : key.point_offsets.data(),
                                             external, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    periodic_enabled = key.periodic_enabled;
    if (periodic_enabled) {
      status = make_periodic_embedding_plan(batch, atoms, key.atom_offsets.data(), periodic, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }
    status =
        make_scc_driver_plan(wavefunction_layout, mulliken, es2, es3, aes2, eigensolver, mixer,
                             d4_enabled ? &d4 : nullptr, periodic_enabled ? &periodic : nullptr,
                             static_cast<std::uint64_t>(key.maximum_iterations),
                             key.electronic_temperature, key.energy_tolerance, driver, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    const std::size_t atom_count = static_cast<std::size_t>(atoms);
    const std::size_t shells = static_cast<std::size_t>(basis.total_shells);
    const std::size_t matrices = static_cast<std::size_t>(integrals.total_matrix_elements);
    const std::size_t integral_doubles =
        (integrals.workspace_size_bytes + sizeof(double) - 1u) / sizeof(double);
    coordination_numbers.resize(atom_count);
    overlap.resize(matrices);
    dipole_integrals.resize(3u * matrices);
    quadrupole_integrals.resize(6u * matrices);
    core_hamiltonian.resize(matrices);
    integral_workspace.resize(std::max<std::size_t>(integral_doubles, 1u));

    status = evaluate_coordination_cpu(coordination, positions.data(), coordination_numbers.data(),
                                       error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(),
                                  integral_workspace.data(),
                                  integral_workspace.size() * sizeof(double), error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = evaluate_multipole_cpu(basis, integrals, positions.data(), dipole_integrals.data(),
                                    quadrupole_integrals.data(), integral_workspace.data(),
                                    integral_workspace.size() * sizeof(double), error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = evaluate_h0_cpu(basis, integrals, h0, positions.data(), coordination_numbers.data(),
                             overlap.data(), core_hamiltonian.data(), error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    geometry_pair_data.assign(static_cast<std::size_t>(aes2.total_pairs()) *
                                  static_cast<std::size_t>(kGfn2GeometryPairDataElements),
                              0.0);
    geometry_generations.assign(static_cast<std::size_t>(batch), geometry_generation);

    es2_matrix.resize(static_cast<std::size_t>(es2.total_matrix_elements()));
    es2_matrix_scratch.resize(es2_matrix.size());
    es2_shell_scratch.resize(shells);
    es2_batch_scratch.resize(static_cast<std::size_t>(batch));
    es2_gradient_scratch.resize(3u * atom_count);
    es2_workspace = {es2_matrix_scratch.data(),   es2.total_matrix_elements(),
                     es2_shell_scratch.data(),    es2.total_shells(),
                     es2_batch_scratch.data(),    batch,
                     es2_gradient_scratch.data(), atoms * 3};
    status =
        update_es2_geometry_cache_cpu(es2, positions.data(), geometry_generation, es2_matrix.data(),
                                      es2_matrix.size(), es2_workspace, es2_cache, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    aes2_pairs.resize(static_cast<std::size_t>(aes2.pair_data_elements()));
    aes2_pair_scratch.resize(aes2_pairs.size());
    aes2_potential_scratch.resize(static_cast<std::size_t>(aes2.potential_scratch_elements()));
    aes2_batch_scratch.resize(static_cast<std::size_t>(batch));
    aes2_gradient_scratch.resize(3u * atom_count);
    aes2_coordination_scratch.resize(atom_count);
    aes2_workspace = {aes2_pair_scratch.data(),         aes2.pair_data_elements(),
                      aes2_potential_scratch.data(),    aes2.potential_scratch_elements(),
                      aes2_batch_scratch.data(),        batch,
                      aes2_gradient_scratch.data(),     atoms * 3,
                      aes2_coordination_scratch.data(), atoms};
    status = update_aes2_geometry_cache_cpu(aes2, positions.data(), coordination_numbers.data(),
                                            geometry_generation, aes2_pairs.data(),
                                            aes2_pairs.size(), aes2_workspace, aes2_cache, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    if (d4_enabled) {
      /* The setup owner needs only a stable initial CN image. The first CUDA
       * preprocessing transaction replaces these zeros before any D4
       * consumer is eligible; dense CPU D4 pair evaluation is intentionally
       * not part of cache construction anymore. */
      d4_coordination.assign(atom_count, 0.0);
      d4_elements.reserve(parameters::d4::kElements.size());
      for (const auto& element : parameters::d4::kElements) {
        d4_elements.push_back({element.reference_offset, element.reference_count,
                               element.covalent_radius, element.electronegativity,
                               element.effective_charge, element.hardness, element.r4r2});
      }
      d4_references.reserve(parameters::d4::kReferences.size());
      for (const auto& reference : parameters::d4::kReferences) {
        d4_references.push_back(
            {reference.coordination_number, reference.charge, reference.gaussian_count});
      }
    }

    explicit_point_shell_potential.resize(shells);
    status = evaluate_external_point_charge_potential_cpu(
        external, positions.data(), point_positions.empty() ? nullptr : point_positions.data(),
        point_values.empty() ? nullptr : point_values.data(),
        point_gammas.empty() ? nullptr : point_gammas.data(), explicit_point_shell_potential.data(),
        error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    if (wavefunction_storage.allocate(wavefunction_layout.workspace_size_bytes) != cudaSuccess) {
      error = "failed to allocate pinned host wavefunction initialization storage";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    status = bind_wavefunction_view(wavefunction_layout, wavefunction_storage.get(),
                                    wavefunction_storage.bytes(), wavefunction, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = initialize_sad_multipole_state(wavefunction_layout, wavefunction, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    return XTBLOOM_STATUS_SUCCESS;
  }

  Gfn2SccSetupInputSources input_sources() const noexcept {
    Gfn2SccSetupInputSources sources{};
    sources.basis = &basis;
    sources.integrals = &integrals;
    sources.h0_plan = &h0;
    sources.wavefunction = &wavefunction_layout;
    sources.es2 = &es2;
    sources.es3 = &es3;
    sources.aes2 = &aes2;
    sources.mulliken = &mulliken;
    sources.mixer = &mixer;
    sources.driver = &driver;
    sources.eigensolver_options.deterministic_debug =
        key.determinism == XTBLOOM_DETERMINISM_REPRODUCIBLE;
    sources.geometry_generation = geometry_generation;
    sources.atomic_numbers = setup_array(key.atomic_numbers);
    sources.positions = setup_array(positions);
    sources.covalent_radii = setup_array(coordination.covalent_radius);
    sources.h0 = setup_array(core_hamiltonian);
    sources.overlap = setup_array(overlap);
    sources.dipole_integrals = setup_array(dipole_integrals);
    sources.quadrupole_integrals = setup_array(quadrupole_integrals);
    sources.geometry_cache.pair_data = setup_array(geometry_pair_data);
    sources.geometry_cache.coordination_numbers = setup_array(coordination_numbers);
    sources.geometry_cache.system_generations = setup_array(geometry_generations);
    sources.es2_cache.coulomb_matrix = setup_array(es2_matrix);
    sources.aes2_cache.pair_data = setup_array(aes2_pairs);
    if (d4_enabled) {
      sources.d4.plan = &d4;
      sources.d4.elements = setup_array(d4_elements);
      sources.d4.references = setup_array(d4_references);
      sources.d4.reference_c6 = {parameters::d4::kReferenceC6.data(),
                                 static_cast<std::int64_t>(parameters::d4::kReferenceC6.size())};
      sources.d4.coordination_numbers = setup_array(d4_coordination);
    }
    if (!point_values.empty()) {
      sources.point_charges.plan = &external;
      sources.point_charges.positions = setup_array(point_positions);
      sources.point_charges.charges = setup_array(point_values);
      sources.point_charges.hardnesses = setup_array(point_gammas);
      sources.point_charges.shell_potential_cache = setup_array(explicit_point_shell_potential);
    }
    if (periodic_enabled) {
      sources.periodic.plan = &periodic;
      sources.periodic.shifts = setup_array(periodic_shifts);
      sources.periodic.response_matrices = setup_array(periodic_response);
    }
    return sources;
  }

  Gfn1SccSetupInputSources gfn1_input_sources() const noexcept {
    Gfn1SccSetupInputSources sources{};
    sources.basis = &basis;
    sources.integrals = &integrals;
    sources.h0_plan = &gfn1_h0;
    sources.wavefunction = &gfn1_wavefunction_layout;
    sources.es2 = &es2;
    sources.es3 = &gfn1_es3;
    sources.spin = &gfn1_spin;
    sources.mulliken = &gfn1_mulliken;
    sources.mixer = &gfn1_mixer;
    sources.driver = &gfn1_driver;
    sources.eigensolver_options.deterministic_debug =
        key.determinism == XTBLOOM_DETERMINISM_REPRODUCIBLE;
    sources.geometry_generation = geometry_generation;
    sources.atomic_numbers = setup_array(key.atomic_numbers);
    sources.positions = setup_array(positions);
    sources.covalent_radii = setup_array(coordination.covalent_radius);
    sources.h0 = setup_array(core_hamiltonian);
    sources.overlap = setup_array(overlap);
    sources.geometry_cache.pair_data = setup_array(geometry_pair_data);
    sources.geometry_cache.coordination_numbers = setup_array(coordination_numbers);
    sources.geometry_cache.system_generations = setup_array(geometry_generations);
    sources.es2_cache.coulomb_matrix = setup_array(es2_matrix);
    if (!point_values.empty()) {
      sources.point_charges.plan = &gfn1_external;
      sources.point_charges.positions = setup_array(point_positions);
      sources.point_charges.charges = setup_array(point_values);
      sources.point_charges.hardnesses = setup_array(point_gammas);
      sources.point_charges.shell_potential_cache = setup_array(explicit_point_shell_potential);
    }
    if (periodic_enabled) {
      sources.periodic.plan = &periodic;
      sources.periodic.shifts = setup_array(periodic_shifts);
      sources.periodic.response_matrices = setup_array(periodic_response);
    }
    return sources;
  }

  Gfn2SccIterationHostInitialization initialization() const noexcept {
    Gfn2SccIterationHostInitialization host{};
    host.mode = Gfn2SccIterationInitializationMode::kFresh;
    host.plan_token = plan_token;
    host.initialization_generation = kInitialStateGeneration;
    if (gfn1_enabled) {
      host.topology = {
          initialization_array(key.atom_offsets.data(),
                               static_cast<std::int64_t>(key.atom_offsets.size())),
          initialization_array(
              gfn1_wavefunction_layout.batch_shell_offsets.data(),
              static_cast<std::int64_t>(gfn1_wavefunction_layout.batch_shell_offsets.size())),
          plan_token};
      host.wavefunction.plan_token = plan_token;
      host.wavefunction.population = {
          initialization_array(gfn1_wavefunction.qsh, gfn1_wavefunction_layout.qsh.element_count),
          initialization_array(gfn1_wavefunction.qat, gfn1_wavefunction_layout.qat.element_count),
          {},
          {},
          plan_token};
      return host;
    }
    host.topology = {initialization_array(key.atom_offsets.data(),
                                          static_cast<std::int64_t>(key.atom_offsets.size())),
                     initialization_array(
                         wavefunction_layout.batch_shell_offsets.data(),
                         static_cast<std::int64_t>(wavefunction_layout.batch_shell_offsets.size())),
                     plan_token};
    host.wavefunction.plan_token = plan_token;
    host.wavefunction.population = {
        initialization_array(wavefunction.qsh, wavefunction_layout.qsh.element_count),
        initialization_array(wavefunction.qat, wavefunction_layout.qat.element_count),
        initialization_array(wavefunction.dipole, wavefunction_layout.dipole.element_count),
        initialization_array(wavefunction.quadrupole, wavefunction_layout.quadrupole.element_count),
        plan_token};
    return host;
  }
};

xtbloom_status_t HostPlans::build_gfn1(TopologyKey&& new_key, std::vector<double>&& new_positions,
                                       std::vector<double>&& new_point_positions,
                                       std::vector<double>&& new_point_values,
                                       std::vector<double>&& new_point_gammas,
                                       std::vector<double>&& new_periodic_shifts,
                                       std::vector<double>&& new_periodic_response,
                                       std::uint64_t token, std::string& error) {
  key = std::move(new_key);
  positions = std::move(new_positions);
  point_positions = std::move(new_point_positions);
  point_values = std::move(new_point_values);
  point_gammas = std::move(new_point_gammas);
  periodic_shifts = std::move(new_periodic_shifts);
  periodic_response = std::move(new_periodic_response);
  fingerprint = key.fingerprint();
  plan_token = token;
  gfn1_enabled = true;
  d4_enabled = false;

  const std::int64_t batch = static_cast<std::int64_t>(key.molecular_charges.size());
  const std::int64_t atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
  const std::int64_t points = static_cast<std::int64_t>(point_values.size());

  xtbloom_status_t status = gfn1::make_basis_plan(batch, atoms, key.atom_offsets.data(),
                                                  key.atomic_numbers.data(), basis, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn1::make_integral_plan(basis, integrals, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  gfn1::CoordinationPlan gfn1_coordination;
  status = gfn1::make_coordination_plan(batch, atoms, key.atom_offsets.data(),
                                        key.atomic_numbers.data(), gfn1_coordination, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  coordination.batch_size = gfn1_coordination.batch_size;
  coordination.total_atoms = gfn1_coordination.total_atoms;
  coordination.atom_offsets = gfn1_coordination.atom_offsets;
  coordination.covalent_radius = gfn1_coordination.covalent_radius;

  status = gfn1::make_repulsion_plan(batch, atoms, key.atom_offsets.data(),
                                     key.atomic_numbers.data(), gfn1_repulsion, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  repulsion = {};
  repulsion.batch_size = gfn1_repulsion.batch_size;
  repulsion.total_atoms = gfn1_repulsion.total_atoms;
  repulsion.atom_offsets = gfn1_repulsion.atom_offsets;
  repulsion.sqrt_alpha = gfn1_repulsion.sqrt_alpha;
  repulsion.effective_charge = gfn1_repulsion.effective_charge;
  repulsion.light_element.assign(static_cast<std::size_t>(atoms), 0u);

  status = gfn1::make_h0_plan(basis, integrals, key.atomic_numbers.data(), gfn1_h0, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  h0 = {};
  h0.batch_size = gfn1_h0.batch_size;
  h0.total_atoms = gfn1_h0.total_atoms;
  h0.total_shells = gfn1_h0.total_shells;
  h0.total_orbitals = gfn1_h0.total_orbitals;
  h0.total_matrix_elements = gfn1_h0.total_matrix_elements;
  h0.atom_offsets = gfn1_h0.atom_offsets;
  h0.batch_shell_offsets = gfn1_h0.batch_shell_offsets;
  h0.batch_orbital_offsets = gfn1_h0.batch_orbital_offsets;
  h0.matrix_offsets = gfn1_h0.matrix_offsets;
  h0.shell_pair_offsets = gfn1_h0.shell_pair_offsets;
  h0.atomic_radii = gfn1_h0.atomic_radii;
  h0.shell_levels = gfn1_h0.shell_levels;
  h0.shell_coordination_scale = gfn1_h0.shell_coordination_scale;
  h0.shell_polynomial = gfn1_h0.shell_polynomial;
  h0.shell_pair_scale = gfn1_h0.shell_pair_scale;
  status =
      make_gfn2_integral_task_domains(basis, h0.shell_pair_offsets, integral_task_domains, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = configured_integral_task_schedule(integral_task_domains.accounting,
                                             compact_integral_tasks, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  status = gfn1::make_wavefunction_layout(
      basis, key.atomic_numbers.data(), key.molecular_charges.data(), key.unpaired_electrons.data(),
      key.spin_channels.data(), gfn1_wavefunction_layout, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn1::make_es2_plan(basis, key.atomic_numbers.data(), es2, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn1::make_es3_plan(basis, key.atomic_numbers.data(), gfn1_es3, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  es3 = {};
  es3.batch_size = batch;
  es3.total_shells = basis.total_shells;
  es3.batch_shell_offsets = basis.batch_shell_offsets;
  es3.shell_gamma3.resize(static_cast<std::size_t>(basis.total_shells));
  for (std::int64_t shell = 0; shell < basis.total_shells; ++shell) {
    const std::int64_t atom = basis.shell_to_atom[static_cast<std::size_t>(shell)];
    es3.shell_gamma3[static_cast<std::size_t>(shell)] =
        gfn1_es3.atom_gamma3[static_cast<std::size_t>(atom)];
  }
  aes2 = {};

  status =
      gfn1::make_spin_population_layout(basis, key.spin_channels.data(), gfn1_spin_layout, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn1::make_spin_polarization_plan(basis, key.atomic_numbers.data(), gfn1_spin_layout,
                                             gfn1_spin, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status =
      gfn1::make_mulliken_plan(basis, integrals, gfn1_wavefunction_layout, gfn1_mulliken, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn2::make_eigensolver_plan(
      gfn1::make_eigensolver_wavefunction_layout(gfn1_wavefunction_layout), eigensolver, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn1::make_scc_mixer_plan(gfn1_wavefunction_layout, key.scc_mixer_history,
                                     key.scc_mixer_damping, key.charge_tolerance,
                                     key.charge_tolerance, gfn1_mixer, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn1::make_d3_plan(batch, atoms, key.atom_offsets.data(), key.atomic_numbers.data(),
                              gfn1_d3, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn1::make_halogen_plan(batch, atoms, key.atom_offsets.data(), key.atomic_numbers.data(),
                                   gfn1_halogen, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  const std::int64_t d3_pairs = gfn1_d3.total_pairs();
  gfn1_d3_reference_counts.assign(static_cast<std::size_t>(atoms), 0u);
  gfn1_d3_reference_cn.assign(static_cast<std::size_t>(atoms) * kGfn1D3MaximumReferences, 0.0);
  gfn1_d3_reference_c6.assign(static_cast<std::size_t>(d3_pairs) * kGfn1D3ReferencePairStride, 0.0);
  gfn1_d3_pair_rrij.assign(static_cast<std::size_t>(d3_pairs), 0.0);
  gfn1_d3_pair_damping_radii.assign(static_cast<std::size_t>(d3_pairs), 0.0);
  gfn1_halogen_scaled_radii.resize(static_cast<std::size_t>(atoms));
  gfn1_halogen_bond_strength.resize(static_cast<std::size_t>(atoms));
  gfn1_halogen_donor.resize(static_cast<std::size_t>(atoms));
  gfn1_halogen_acceptor.resize(static_cast<std::size_t>(atoms));
  for (std::int64_t atom = 0; atom < atoms; ++atom) {
    const std::uint32_t z =
        static_cast<std::uint32_t>(key.atomic_numbers[static_cast<std::size_t>(atom)]);
    const auto& d3_element = parameters::gfn1_d3::kElements[z - 1u];
    gfn1_d3_reference_counts[static_cast<std::size_t>(atom)] = d3_element.reference_count;
    const std::size_t ref_offset = static_cast<std::size_t>(atom) * kGfn1D3MaximumReferences;
    for (std::uint32_t ref = 0; ref < d3_element.reference_count; ++ref) {
      gfn1_d3_reference_cn[ref_offset + ref] = parameters::gfn1_d3::reference_cn(z, ref);
    }
    const auto* element = parameters::gfn1::find_element(z);
    if (element == nullptr) {
      error = "GFN1 CUDA correction setup found an unsupported atomic number";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    gfn1_halogen_scaled_radii[static_cast<std::size_t>(atom)] =
        parameters::gfn1::kGlobal.halogen_radius_scale * element->atomic_radius_bohr;
    gfn1_halogen_bond_strength[static_cast<std::size_t>(atom)] = element->xbond;
    gfn1_halogen_donor[static_cast<std::size_t>(atom)] =
        static_cast<std::uint8_t>(z == 17u || z == 35u || z == 53u || z == 85u);
    gfn1_halogen_acceptor[static_cast<std::size_t>(atom)] =
        static_cast<std::uint8_t>(z == 7u || z == 8u || z == 15u || z == 16u);
  }
  for (std::int64_t system = 0; system < batch; ++system) {
    const std::int64_t begin = key.atom_offsets[static_cast<std::size_t>(system)];
    const std::int64_t end = key.atom_offsets[static_cast<std::size_t>(system + 1)];
    for (std::int64_t second = begin + 1; second < end; ++second) {
      const std::int64_t local_second = second - begin;
      for (std::int64_t first = begin; first < second; ++first) {
        const std::int64_t local_first = first - begin;
        const std::int64_t pair = gfn1_d3.pair_offsets()[static_cast<std::size_t>(system)] +
                                  local_second * (local_second - 1) / 2 + local_first;
        const std::uint32_t z1 =
            static_cast<std::uint32_t>(key.atomic_numbers[static_cast<std::size_t>(first)]);
        const std::uint32_t z2 =
            static_cast<std::uint32_t>(key.atomic_numbers[static_cast<std::size_t>(second)]);
        const double rrij =
            3.0 * parameters::gfn1_d3::kR4R2[z1 - 1u] * parameters::gfn1_d3::kR4R2[z2 - 1u];
        gfn1_d3_pair_rrij[static_cast<std::size_t>(pair)] = rrij;
        gfn1_d3_pair_damping_radii[static_cast<std::size_t>(pair)] =
            parameters::gfn1::kGlobal.dispersion_a1 * std::sqrt(rrij) +
            parameters::gfn1::kGlobal.dispersion_a2;
        const std::size_t c6_offset = static_cast<std::size_t>(pair) * kGfn1D3ReferencePairStride;
        const std::uint32_t n1 = gfn1_d3_reference_counts[static_cast<std::size_t>(first)];
        const std::uint32_t n2 = gfn1_d3_reference_counts[static_cast<std::size_t>(second)];
        for (std::uint32_t i = 0; i < n1; ++i) {
          for (std::uint32_t j = 0; j < n2; ++j) {
            gfn1_d3_reference_c6[c6_offset + i * kGfn1D3MaximumReferences + j] =
                parameters::gfn1_d3::reference_c6(z1, i, z2, j);
          }
        }
      }
    }
  }

  status = gfn1::make_external_point_charge_plan(
      basis, es2, points, points == 0 ? nullptr : key.point_offsets.data(), gfn1_external, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  external = {};
  external.batch_size = gfn1_external.batch_size;
  external.total_atoms = gfn1_external.total_atoms;
  external.total_shells = gfn1_external.total_shells;
  external.total_point_charges = gfn1_external.total_point_charges;
  external.atom_offsets = gfn1_external.atom_offsets;
  external.batch_shell_offsets = gfn1_external.batch_shell_offsets;
  external.point_charge_offsets = gfn1_external.point_charge_offsets;
  external.shell_to_atom = gfn1_external.shell_to_atom;
  external.shell_hardness = gfn1_external.shell_hardness;

  periodic_enabled = key.periodic_enabled;
  if (periodic_enabled) {
    status =
        gfn2::make_periodic_embedding_plan(batch, atoms, key.atom_offsets.data(), periodic, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    periodic = {};
  }
  status = gfn1::make_scc_driver_plan(
      gfn1_wavefunction_layout, gfn1_mulliken, es2, gfn1_es3, gfn1_spin, eigensolver, gfn1_mixer,
      periodic_enabled ? &periodic : nullptr, static_cast<std::uint64_t>(key.maximum_iterations),
      key.electronic_temperature, key.energy_tolerance, gfn1_driver, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  const std::size_t atom_count = static_cast<std::size_t>(atoms);
  const std::size_t shell_count = static_cast<std::size_t>(basis.total_shells);
  const std::size_t matrix_count = static_cast<std::size_t>(integrals.total_matrix_elements);
  const std::size_t integral_doubles =
      (integrals.workspace_size_bytes + sizeof(double) - 1u) / sizeof(double);
  coordination_numbers.resize(atom_count);
  overlap.resize(matrix_count);
  dipole_integrals.clear();
  quadrupole_integrals.clear();
  core_hamiltonian.resize(matrix_count);
  integral_workspace.resize(std::max<std::size_t>(integral_doubles, 1u));

  status = gfn1::evaluate_coordination_cpu(gfn1_coordination, positions.data(),
                                           coordination_numbers.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn1::evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(),
                                      integral_workspace.data(),
                                      integral_workspace.size() * sizeof(double), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn1::evaluate_h0_cpu(basis, integrals, gfn1_h0, positions.data(),
                                 coordination_numbers.data(), overlap.data(),
                                 core_hamiltonian.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  geometry_pair_data.assign(static_cast<std::size_t>(gfn1_d3.total_pairs()) *
                                static_cast<std::size_t>(kGfn2GeometryPairDataElements),
                            0.0);
  geometry_generations.assign(static_cast<std::size_t>(batch), geometry_generation);

  es2_matrix.resize(static_cast<std::size_t>(es2.total_matrix_elements()));
  es2_matrix_scratch.resize(es2_matrix.size());
  es2_shell_scratch.resize(shell_count);
  es2_batch_scratch.resize(static_cast<std::size_t>(batch));
  es2_gradient_scratch.resize(3u * atom_count);
  es2_workspace = {es2_matrix_scratch.data(),   es2.total_matrix_elements(),
                   es2_shell_scratch.data(),    es2.total_shells(),
                   es2_batch_scratch.data(),    batch,
                   es2_gradient_scratch.data(), atoms * 3};
  status = gfn1::update_es2_geometry_cache_cpu(es2, positions.data(), geometry_generation,
                                               es2_matrix.data(), es2_matrix.size(), es2_workspace,
                                               es2_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  aes2_pairs.clear();
  aes2_pair_scratch.clear();
  aes2_potential_scratch.clear();
  aes2_batch_scratch.clear();
  aes2_gradient_scratch.clear();
  aes2_coordination_scratch.clear();
  d4_elements.clear();
  d4_references.clear();
  d4_coordination.clear();

  explicit_point_shell_potential.resize(shell_count);
  status = gfn1::evaluate_external_point_charge_potential_cpu(
      gfn1_external, positions.data(), point_positions.empty() ? nullptr : point_positions.data(),
      point_values.empty() ? nullptr : point_values.data(),
      point_gammas.empty() ? nullptr : point_gammas.data(), explicit_point_shell_potential.data(),
      error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  if (gfn1_wavefunction_storage.allocate(gfn1_wavefunction_layout.workspace_size_bytes) !=
      cudaSuccess) {
    error = "failed to allocate pinned host GFN1 wavefunction initialization storage";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
  status =
      gfn1::bind_wavefunction_view(gfn1_wavefunction_layout, gfn1_wavefunction_storage.get(),
                                   gfn1_wavefunction_storage.bytes(), gfn1_wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn1::initialize_sad_multipole_state(gfn1_wavefunction_layout, gfn1_wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  return XTBLOOM_STATUS_SUCCESS;
}

std::string cuda_error_message(const char* operation, cudaError_t status) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorName(status) << " ("
          << cudaGetErrorString(status) << ')';
  return message.str();
}

std::string setup_error_message(const char* operation, xtbloom_status_t status,
                                std::uint32_t error_code, std::uint32_t field, std::int64_t index) {
  std::ostringstream message;
  message << operation << " failed: status=" << status << " error=" << error_code
          << " field=" << field << " index=" << index;
  return message.str();
}

struct NumericalRefreshState {
  Gfn2PreprocessingDeviceBinding preprocessing{};
  NumericalRefreshDeviceBinding device{};
  /* Refresh reads preprocessing candidate positions; SCC/post/force/terminal
   * consumers read d4_cache with the exact committed-position identity. */
  Gfn2D4PairListDeviceCache d4_refresh_cache{};
  Gfn2D4PairListDeviceCache d4_cache{};
  Gfn2D4DeviceWorkspace d4_workspace{};
  Gfn2ExternalPointChargeDeviceBatch point_batch{};
  Gfn2ExternalPointChargeDeviceCache point_candidate{};
  Gfn2ExternalPointChargeDeviceWorkspace point_workspace{};
  Gfn2SccIterationDeviceActivity point_activity{};
  Gfn2ElectricFieldDeviceBatch field_batch{};
  Gfn2ElectricFieldDevicePotentials field_candidate_potentials{};
  Gfn2ElectricFieldDevicePotentials field_committed_potentials{};
  Gfn2ElectricFieldDevicePotentialView field_potential_view{};

  double* host_positions = nullptr;
  double* host_point_positions = nullptr;
  double* host_point_values = nullptr;
  double* host_point_gammas = nullptr;
  double* host_periodic_shifts = nullptr;
  double* host_periodic_response = nullptr;
  std::uint8_t* host_requested = nullptr;
  xtbloom_interaction_t* host_interaction_descriptors = nullptr;
  std::byte* host_interaction_payload = nullptr;

  /*
   * Host submissions first copy synchronously into this packed, pinned image.
   * The fixed device staging leaves above are then populated asynchronously,
   * so no queued CUDA work retains a caller-owned host pointer after return.
   */
  double* owned_host_positions = nullptr;
  double* owned_host_point_positions = nullptr;
  double* owned_host_point_values = nullptr;
  double* owned_host_point_gammas = nullptr;
  double* owned_host_periodic_shifts = nullptr;
  double* owned_host_periodic_response = nullptr;
  std::uint8_t* owned_host_requested = nullptr;
  xtbloom_interaction_t* owned_host_interaction_descriptors = nullptr;
  std::byte* owned_host_interaction_payload = nullptr;
  xtbloom_interaction_t* owned_host_interaction_descriptor_snapshot = nullptr;

  std::size_t interaction_descriptor_capacity_bytes = 0u;
  std::size_t interaction_payload_capacity_bytes = 0u;

  bool host_staging_poisoned = false;

  bool ready = false;
};

void project_interaction_staging(const InteractionStagingLayout& layout, void* device_arena,
                                 void* host_arena, NumericalRefreshState& state) noexcept {
  state.host_interaction_descriptors = reinterpret_cast<xtbloom_interaction_t*>(
      static_cast<std::byte*>(device_arena) + layout.descriptor_offset);
  state.host_interaction_payload = static_cast<std::byte*>(device_arena) + layout.payload_offset;
  state.owned_host_interaction_descriptors = reinterpret_cast<xtbloom_interaction_t*>(
      static_cast<std::byte*>(host_arena) + layout.descriptor_offset);
  state.owned_host_interaction_payload =
      static_cast<std::byte*>(host_arena) + layout.payload_offset;
  state.owned_host_interaction_descriptor_snapshot = reinterpret_cast<xtbloom_interaction_t*>(
      static_cast<std::byte*>(host_arena) + layout.descriptor_snapshot_offset);
  state.interaction_descriptor_capacity_bytes = layout.descriptor_capacity_bytes;
  state.interaction_payload_capacity_bytes = layout.payload_capacity_bytes;
}

/*
 * The host-owned pinned snapshot is a single-flight resource.  A host function
 * on a private completion stream releases it after that stream has waited on
 * the owner-stream H2D event.  The submission thread therefore needs only one
 * acquire load; it never queries CUDA progress or waits for either stream.
 *
 * This state is deliberately separate from NumericalRefreshState because the
 * latter is reset as an aggregate while constructing a fixed-topology owner.
 */
struct NumericalHostUploadCompletion {
  std::atomic<bool> pending{false};
};
static_assert(std::atomic<bool>::is_always_lock_free,
              "CUDA host-upload completion must stay allocation-free and nonblocking");

void CUDART_CB release_numerical_host_upload(void* user_data) noexcept {
  auto* const completion = static_cast<NumericalHostUploadCompletion*>(user_data);
  completion->pending.store(false, std::memory_order_release);
}

/*
 * Stable terminal descriptors and their context-owned result arena.  These
 * views are constructed once for a fixed topology and copied by value into
 * allocation-free stream submissions.  The result buffers are deliberately
 * internal: #125 later bridges them to host or device C-API destinations.
 */
struct InferenceState {
  Gfn2GeometryEpochConsumerDevice epoch_consumer{};

  Gfn2TerminalClassicalEnergyDevicePlan terminal_plan{};
  Gfn2TerminalClassicalEnergyDeviceActivity terminal_activity{};
  Gfn2TerminalClassicalEnergyDeviceResults terminal_results{};
  Gfn2TerminalClassicalEnergyDeviceWorkspace terminal_workspace{};
  Gfn2TerminalClassicalEnergyDeviceDiagnostics terminal_diagnostics{};
  /* GFN1 D3(BJ) and halogen energy are accumulated into the terminal
   * classical-energy slot after the transactional repulsion publication. The
   * final inference gate consumes the same terminal diagnostics, so a failed
   * correction peer can never publish a partial total energy. */
  Gfn1ClassicalCorrectionDevicePlan gfn1_correction_plan{};
  Gfn1ClassicalCorrectionDeviceWorkspace gfn1_correction_workspace{};

  Gfn2InferencePublicationDevicePlan publication_plan{};
  Gfn2InferencePublicationDeviceInput publication_input{};
  Gfn2InferencePublicationDeviceResults publication_results{};
  Gfn2InferencePublicationDeviceWorkspace publication_workspace{};
  Gfn2InferencePublicationDeviceDiagnostics publication_diagnostics{};

  /* Per-peer generation of the last successfully published SCC checkpoint. */
  std::uint64_t* warm_checkpoint_generations = nullptr;
  /* Device aggregate: nonzero only when every peer published a checkpoint. */
  std::uint32_t* warm_checkpoint_batch_ready = nullptr;
  /* Single-flight asynchronous request control and the checkpoint token moved
   * out of public storage at accepted admission. */
  std::uint32_t* request_start_mode = nullptr;
  std::uint64_t* admitted_warm_checkpoint_generations = nullptr;
  std::uint8_t* warm_checkpoint_field_attached = nullptr;
  double* warm_checkpoint_field_vectors = nullptr;
  bool ready = false;
  bool warm_checkpoint_ready = false;
};

/*
 * Fixed-topology public result staging. Every possible HOST route has a stable
 * pinned slice, while one device control record seals aggregate publication
 * before any CUDA caller destination is touched. Per-call route descriptors
 * borrow these addresses but never outlive the synchronous cache transaction.
 */
struct PublicResultState {
  double* energies = nullptr;
  double* qm_forces = nullptr;
  double* atomic_charges = nullptr;
  double* point_forces = nullptr;
  double* dipole_moments = nullptr;
  std::int32_t* iterations = nullptr;
  std::uint8_t* converged = nullptr;
  xtbloom_status_t* system_statuses = nullptr;
  Gfn2PublicResultBridgeControl* host_control = nullptr;
  /* Pinned mirror copied under the existing public completion event. */
  std::uint32_t* warm_checkpoint_ready = nullptr;
  /* The request flag and canonical topology arrays share the plan-owned
   * result arena. They are reset/compared on the owner stream before
   * inference, then consumed by the public-result preflight gate. */
  std::uint32_t* request_topology_error = nullptr;
  const std::int64_t* expected_atom_offsets = nullptr;
  const std::int32_t* expected_atomic_numbers = nullptr;
  const double* expected_molecular_charges = nullptr;
  const std::int32_t* expected_unpaired_electrons = nullptr;
  const std::int32_t* expected_spin_channels = nullptr;
  const std::int64_t* expected_point_offsets = nullptr;
  const std::int64_t* expected_response_offsets = nullptr;
  const double* expected_lattice_cells = nullptr;
  const std::int32_t* expected_periodic_axes = nullptr;
  double* staged_lattice_cells = nullptr;
  std::int32_t* staged_periodic_axes = nullptr;
  double* host_lattice_cells = nullptr;
  std::int32_t* host_periodic_axes = nullptr;
  Gfn2PublicResultBridgeDeviceStaging device_staging{};
  Gfn2PublicResultBridgeDeviceDiagnostics diagnostics{};
  std::uint32_t pending_result_flags = 0u;
  bool ready = false;
};

/*
 * The public bridge is intentionally represented as explicit host phases.
 * CUDA submission, completion observation, aggregate acceptance, and host
 * publication have different lifetime and failure boundaries; keeping those
 * boundaries visible lets the synchronous owner preserve today's ordering
 * while a later request owner can drive the same transaction incrementally.
 */
enum class PublicResultTransactionPhase : std::uint32_t {
  kEmpty = 0u,
  kPrepareSubmitted = 1u,
  kPrepareCompletionRecorded = 2u,
  kPrepareCompleted = 3u,
  kPrepareAccepted = 4u,
  kCommitSubmitted = 5u,
  kCommitCompletionRecorded = 6u,
  kCommitCompleted = 7u,
  kPublished = 8u,
};

/* Descriptor image retained across one public result transaction. All
 * referenced storage is owned by Prepared or borrowed from the caller until
 * synchronous return / asynchronous request completion. */
struct PublicResultTransaction {
  Gfn2PublicResultBridgeDevicePlan plan{};
  Gfn2PublicResultBridgeDeviceDestinations destinations{};
  PublicResultTransactionPhase phase = PublicResultTransactionPhase::kEmpty;
  bool prior_warm_checkpoint_ready = false;
};

class RequestExecutionGraphOwner {
 public:
  RequestExecutionGraphOwner() = default;
  ~RequestExecutionGraphOwner() { reset(); }
  RequestExecutionGraphOwner(const RequestExecutionGraphOwner&) = delete;
  RequestExecutionGraphOwner& operator=(const RequestExecutionGraphOwner&) = delete;

  void reset() noexcept {
    if (executable_ != nullptr) (void)cudaGraphExecDestroy(executable_);
    if (graph_ != nullptr) (void)cudaGraphDestroy(graph_);
    executable_ = nullptr;
    graph_ = nullptr;
    execution_condition_ = 0u;
    start_mode_condition_ = 0u;
  }

  [[nodiscard]] bool ready() const noexcept { return executable_ != nullptr; }
  [[nodiscard]] cudaGraph_t graph() const noexcept { return graph_; }
  [[nodiscard]] cudaGraphConditionalHandle execution_condition() const noexcept {
    return execution_condition_;
  }
  [[nodiscard]] cudaGraphConditionalHandle start_mode_condition() const noexcept {
    return start_mode_condition_;
  }

  cudaError_t create() noexcept {
    reset();
    cudaError_t status = cudaGraphCreate(&graph_, 0u);
    if (status == cudaSuccess) {
      status = cudaGraphConditionalHandleCreate(&execution_condition_, graph_, 0u,
                                                cudaGraphCondAssignDefault);
    }
    if (status == cudaSuccess) {
      status = cudaGraphConditionalHandleCreate(&start_mode_condition_, graph_, 0u,
                                                cudaGraphCondAssignDefault);
    }
    if (status != cudaSuccess) reset();
    return status;
  }

  cudaError_t add_if_node(cudaGraphNode_t dependency, cudaGraph_t& body) noexcept {
    body = nullptr;
    if (graph_ == nullptr || execution_condition_ == 0u || dependency == nullptr) {
      return cudaErrorInvalidValue;
    }
    cudaGraphNodeParams parameters{};
    parameters.type = cudaGraphNodeTypeConditional;
    parameters.conditional.handle = execution_condition_;
    parameters.conditional.type = cudaGraphCondTypeIf;
    parameters.conditional.size = 1u;
    cudaGraphNode_t node = nullptr;
    const cudaError_t status = cudaGraphAddNode(&node, graph_, &dependency, 1u, &parameters);
    if (status == cudaSuccess) body = parameters.conditional.phGraph_out[0];
    return status;
  }

  cudaError_t add_start_mode_switch(cudaGraph_t body, cudaGraphNode_t dependency,
                                    cudaGraph_t*& branches, cudaGraphNode_t& switch_node) noexcept {
    branches = nullptr;
    switch_node = nullptr;
    if (body == nullptr || start_mode_condition_ == 0u || dependency == nullptr) {
      return cudaErrorInvalidValue;
    }
    cudaGraphNodeParams parameters{};
    parameters.type = cudaGraphNodeTypeConditional;
    parameters.conditional.handle = start_mode_condition_;
    parameters.conditional.type = cudaGraphCondTypeSwitch;
    parameters.conditional.size = 2u;
    const cudaError_t status = cudaGraphAddNode(&switch_node, body, &dependency, 1u, &parameters);
    if (status == cudaSuccess) branches = parameters.conditional.phGraph_out;
    return status;
  }

  cudaError_t instantiate() noexcept {
    if (graph_ == nullptr || executable_ != nullptr) return cudaErrorInvalidValue;
    return cudaGraphInstantiate(&executable_, graph_, 0u);
  }

  cudaError_t upload(cudaStream_t stream) const noexcept {
    return executable_ == nullptr ? cudaErrorInvalidResourceHandle
                                  : cudaGraphUpload(executable_, stream);
  }

  cudaError_t launch(cudaStream_t stream) const noexcept {
    return executable_ == nullptr ? cudaErrorInvalidResourceHandle
                                  : cudaGraphLaunch(executable_, stream);
  }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  cudaGraphConditionalHandle execution_condition_ = 0u;
  cudaGraphConditionalHandle start_mode_condition_ = 0u;
};

}  // namespace

struct Gfn2CudaExecutionCache::Impl {
  struct EnergyForceBindings {
    Gfn2EnergyForceExecutionDevicePlan plan{};
    Gfn2EnergyForceExecutionDeviceInput input{};
    Gfn2EnergyForceExecutionDeviceResults results{};
    Gfn2EnergyForceExecutionDeviceIntermediates intermediates{};
    Gfn2EnergyForceExecutionDeviceWorkspace workspace{};
    Gfn2EnergyForceExecutionDeviceDiagnostics diagnostics{};
    StationaryForceProjectionDeviceBinding stationary_projection{};
  };

  struct Prepared {
    explicit Prepared(cudaStream_t owner_stream) noexcept : stream(owner_stream) {}
    ~Prepared() {
      /* Setup owners retain pinned provider images and the SCC initializer
       * retains its immutable device checkpoint. Numerical host-completion
       * functions also retain the address of this Prepared object's atomic
       * state. A failed candidate and a replaced cache therefore wait for the
       * owner and private completion streams before releasing any source image,
       * callback state, or arena referenced by queued work. CUDA does not
       * invoke a host function after a context failure; in that case pending
       * remains conservatively set and both failed synchronization attempts
       * still precede destruction. */
      if (submitted) {
        /* cudaStreamSynchronize(nullptr) waits for the selected legacy default
         * stream only; setup teardown never escalates to a device-wide fence. */
        (void)cudaStreamSynchronize(stream);
      }
      if (numerical_host_completion_stream.valid()) {
        (void)cudaStreamSynchronize(numerical_host_completion_stream.get());
      }
    }

    cudaStream_t stream = nullptr;
    bool submitted = false;
    HostPlans host;
    Gfn2SccSetupTopology topology_owner;
    Gfn2SccSetupInputs inputs_owner;
    Gfn2SccSetupEigensolver eigensolver_owner;
    Gfn2SccIterationInitializer initializer;
    DeviceArena topology_arena;
    DeviceArena input_arena;
    DeviceArena iteration_arena;
    DeviceArena eigensolver_setup_arena;
    PinnedArena provider_host_workspace;
    PinnedArena numerical_host_staging_arena;
    DeviceArena interaction_device_staging_arena;
    PinnedArena interaction_host_staging_arena;
    CudaStream numerical_host_completion_stream;
    CudaEvent numerical_host_upload_complete;
    CudaEvent numerical_host_release_complete;
    NumericalHostUploadCompletion numerical_host_upload_completion;
    /* Canonical shell-pair tasks are uploaded once and shared by numerical
     * refresh and force graphs; neither mutable arena duplicates them. */
    DeviceArena integral_task_arena;
    Gfn2IntegralDeviceTaskDomains integral_tasks{};
    DeviceArena force_immutable_arena;
    DeviceArena force_execution_arena;
    DeviceArena numerical_refresh_arena;
    DeviceArena inference_arena;
    DeviceArena public_result_device_arena;
    PinnedArena public_result_host_arena;
    PinnedArena candidate_validation_arena;
    CudaEvent public_result_completion_event;
    Gfn2RaggedTopologyView device_topology{};
    Gfn2WavefunctionLayoutView device_wavefunction{};
    Gfn2SccIterationDevicePlan plan_seed{};
    Gfn2SccIterationDeviceInput input_seed{};
    Gfn2SccIterationArenaRequirements iteration_requirements{};
    Gfn2SccIterationDeviceState state_seed{};
    Gfn2SccIterationDeviceWorkspace workspace_seed{};
    Gfn2SccIterationReportStorage report_storage{};
    Gfn2SccSetupEigensolverBinding eigensolver_binding{};
    Gfn2SccIterationInitializationReady ready{};
    Gfn2SccIterationBinding scc_binding{};
    /* Built once after setup validation and replayed by fresh/warm inference.
     * The owner retains a bounded fallback when conditional capture is not
     * supported by the selected CUDA/provider stack. */
    Gfn2SccLoopCudaGraphOwner scc_loop;
    /* Declared after scc_loop so the outer graph is destroyed first. Its
     * common numerical/SCC body retains the scc_loop root graph by reference;
     * only the small FRESH/WARM reset preludes are distinct. */
    RequestExecutionGraphOwner request_execution_graph;
    const std::int32_t* atomic_numbers = nullptr;
    const double* gfn1_repulsion_sqrt_alpha = nullptr;
    const double* gfn1_repulsion_effective_charge = nullptr;
    Gfn1ClassicalCorrectionDevicePlan gfn1_correction_plan{};
    EnergyForceBindings energy_force{};
    NumericalRefreshState numerical{};
    InferenceState inference{};
    PublicResultState public_result{};
    bool energy_force_smoke_ready = false;
  };

  static xtbloom_status_t validate_prepared_admission_aliases(const Prepared& candidate,
                                                              std::string& error) noexcept {
    const auto* const admission = candidate.public_result.request_topology_error;
    AddressRange admission_range{};
    AddressRange public_result_arena_range{};
    if (reinterpret_cast<std::uintptr_t>(admission) % alignof(std::uint32_t) != 0u ||
        !make_address_range(admission, sizeof(*admission), admission_range) ||
        !make_address_range(candidate.public_result_device_arena.get(),
                            candidate.public_result_device_arena.bytes(),
                            public_result_arena_range) ||
        !address_range_contains(public_result_arena_range, admission_range)) {
      error = "CUDA runtime admission scalar is outside its owned public-result range";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const auto reject_overlap = [&](const char* name, const void* pointer, std::int64_t elements,
                                    std::size_t element_size) -> bool {
      if (elements == 0) return false;
      std::size_t bytes = 0u;
      AddressRange writable{};
      if (!checked_bytes(elements, element_size, bytes)) {
        error =
            std::string("CUDA runtime admission audit found an invalid writable range for ") + name;
        return true;
      }
      /* This pass audits cross-subsystem aliasing, while the component
       * validators own required/optional pointer canonicalization. A null
       * later-bound view therefore contributes no range here; a non-null
       * malformed range remains an owner-level construction failure. */
      if (pointer == nullptr) return false;
      if (!make_address_range(pointer, bytes, writable)) {
        error =
            std::string("CUDA runtime admission audit found an invalid writable range for ") + name;
        return true;
      }
      if (!address_ranges_overlap(admission_range, writable)) return false;
      error = std::string("CUDA runtime admission scalar aliases writable ") + name;
      return true;
    };
    const auto reject_arena = [&](const char* name, const auto& arena) -> bool {
      AddressRange writable{};
      if (!make_address_range(arena.get(), arena.bytes(), writable)) return false;
      if (!address_ranges_overlap(admission_range, writable)) return false;
      error = std::string("CUDA runtime admission scalar aliases writable ") + name + " arena";
      return true;
    };

    /* Bulk restores and several component composers write complete arenas.
     * Leaf validators prove their internal projections; the owner-level audit
     * additionally keeps the call-wide gate outside every such destination. */
    if (reject_arena("SCC iteration", candidate.iteration_arena) ||
        reject_arena("eigensolver cache", candidate.eigensolver_setup_arena) ||
        reject_arena("numerical refresh", candidate.numerical_refresh_arena) ||
        reject_arena("interaction staging", candidate.interaction_device_staging_arena) ||
        reject_arena("energy/force execution", candidate.force_execution_arena) ||
        reject_arena("inference", candidate.inference_arena)) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const auto& numerical = candidate.numerical.device;
    const std::int64_t batch = numerical.batch_size;
    const std::int64_t atoms = numerical.total_atoms;
    const std::int64_t matrices = numerical.total_matrices;
    std::int64_t atom_coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t field_coordinates = 0;
    std::int64_t dipole_integrals = 0;
    std::int64_t quadrupole_integrals = 0;
    if (!checked_elements(atoms, 3, atom_coordinates) ||
        !checked_elements(numerical.total_point_charges, 3, point_coordinates) ||
        !checked_elements(batch, 3, field_coordinates) ||
        (numerical.multipoles_enabled != 0u && !checked_elements(matrices, 3, dipole_integrals)) ||
        (numerical.multipoles_enabled != 0u &&
         !checked_elements(matrices, 6, quadrupole_integrals))) {
      error = "CUDA runtime admission audit extent overflows int64_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
#define XTBLOOM_REJECT_WRITE(name, pointer, elements, type) \
  if (reject_overlap(name, pointer, elements, sizeof(type))) return XTBLOOM_STATUS_INVALID_ARGUMENT
    XTBLOOM_REJECT_WRITE("numerical requested mask", numerical.requested, batch, std::uint8_t);
    XTBLOOM_REJECT_WRITE("numerical eligibility mask", numerical.eligible, batch, std::uint8_t);
    XTBLOOM_REJECT_WRITE("overlap factor activity", numerical.factor_active, batch, std::uint8_t);
    XTBLOOM_REJECT_WRITE("committed generations", numerical.committed_generations, batch,
                         std::uint64_t);
    XTBLOOM_REJECT_WRITE("refresh predecessor generations",
                         numerical.refresh_predecessor_generations, batch, std::uint64_t);
    XTBLOOM_REJECT_WRITE("warm checkpoint generations", numerical.warm_checkpoint_generations,
                         batch, std::uint64_t);
    XTBLOOM_REJECT_WRITE("candidate positions", numerical.candidate_positions, atom_coordinates,
                         double);
    XTBLOOM_REJECT_WRITE("committed positions", numerical.committed_positions, atom_coordinates,
                         double);
    XTBLOOM_REJECT_WRITE("candidate point-charge positions", numerical.candidate_point_positions,
                         numerical.point_enabled != 0u ? point_coordinates : 0, double);
    XTBLOOM_REJECT_WRITE("candidate point-charge values", numerical.candidate_point_values,
                         numerical.point_enabled != 0u ? numerical.total_point_charges : 0, double);
    XTBLOOM_REJECT_WRITE("candidate point-charge gammas", numerical.candidate_point_gammas,
                         numerical.point_enabled != 0u ? numerical.total_point_charges : 0, double);
    XTBLOOM_REJECT_WRITE("committed point-charge positions", numerical.committed_point_positions,
                         numerical.point_enabled != 0u ? point_coordinates : 0, double);
    XTBLOOM_REJECT_WRITE("committed point-charge values", numerical.committed_point_values,
                         numerical.point_enabled != 0u ? numerical.total_point_charges : 0, double);
    XTBLOOM_REJECT_WRITE("committed point-charge gammas", numerical.committed_point_gammas,
                         numerical.point_enabled != 0u ? numerical.total_point_charges : 0, double);
    XTBLOOM_REJECT_WRITE("candidate periodic shifts", numerical.candidate_periodic_shifts,
                         numerical.periodic_enabled != 0u ? atoms : 0, double);
    XTBLOOM_REJECT_WRITE("candidate periodic response", numerical.candidate_periodic_response,
                         numerical.periodic_enabled != 0u ? numerical.total_response_elements : 0,
                         double);
    XTBLOOM_REJECT_WRITE("committed periodic shifts", numerical.committed_periodic_shifts,
                         numerical.periodic_enabled != 0u ? atoms : 0, double);
    XTBLOOM_REJECT_WRITE("committed periodic response", numerical.committed_periodic_response,
                         numerical.periodic_enabled != 0u ? numerical.total_response_elements : 0,
                         double);
    XTBLOOM_REJECT_WRITE("candidate field attachment", numerical.candidate_field_attached, batch,
                         std::uint8_t);
    XTBLOOM_REJECT_WRITE("candidate field vectors", numerical.candidate_field_vectors,
                         field_coordinates, double);
    XTBLOOM_REJECT_WRITE("candidate field atomic potentials",
                         numerical.candidate_field_atomic_potentials, atoms, double);
    XTBLOOM_REJECT_WRITE("candidate field dipole potentials",
                         numerical.candidate_field_dipole_potentials, atom_coordinates, double);
    XTBLOOM_REJECT_WRITE("committed field attachment", numerical.committed_field_attached, batch,
                         std::uint8_t);
    XTBLOOM_REJECT_WRITE("committed field vectors", numerical.committed_field_vectors,
                         field_coordinates, double);
    XTBLOOM_REJECT_WRITE("committed field atomic potentials",
                         numerical.committed_field_atomic_potentials, atoms, double);
    XTBLOOM_REJECT_WRITE("committed field dipole potentials",
                         numerical.committed_field_dipole_potentials, atom_coordinates, double);
    XTBLOOM_REJECT_WRITE("warm checkpoint field attachment",
                         numerical.warm_checkpoint_field_attached, batch, std::uint8_t);
    XTBLOOM_REJECT_WRITE("warm checkpoint field vectors", numerical.warm_checkpoint_field_vectors,
                         field_coordinates, double);
    XTBLOOM_REJECT_WRITE("interaction request diagnostic", numerical.interaction_request_error, 1,
                         std::uint32_t);
    XTBLOOM_REJECT_WRITE("field system diagnostics", numerical.field_system_errors, batch,
                         std::uint32_t);
    XTBLOOM_REJECT_WRITE("field plan diagnostic", numerical.field_plan_error, 1, std::uint32_t);
    XTBLOOM_REJECT_WRITE("periodic system diagnostics", numerical.periodic_system_errors,
                         numerical.periodic_enabled != 0u ? batch : 0, std::uint32_t);
    XTBLOOM_REJECT_WRITE("periodic plan diagnostic", numerical.periodic_plan_error,
                         numerical.periodic_enabled != 0u ? 1 : 0, std::uint32_t);
    XTBLOOM_REJECT_WRITE("public geometry pairs", numerical.public_geometry_pairs,
                         numerical.geometry_pair_elements, double);
    XTBLOOM_REJECT_WRITE("public coordination", numerical.public_coordination, atoms, double);
    XTBLOOM_REJECT_WRITE("public overlap", numerical.public_overlap, matrices, double);
    XTBLOOM_REJECT_WRITE("public dipole integrals", numerical.public_dipole, dipole_integrals,
                         double);
    XTBLOOM_REJECT_WRITE("public quadrupole integrals", numerical.public_quadrupole,
                         quadrupole_integrals, double);
    XTBLOOM_REJECT_WRITE("public H0", numerical.public_h0, matrices, double);
    XTBLOOM_REJECT_WRITE("public ES2 cache", numerical.public_es2, numerical.es2_elements, double);
    XTBLOOM_REJECT_WRITE("public AES2 cache", numerical.public_aes2, numerical.aes2_elements,
                         double);
    XTBLOOM_REJECT_WRITE("public D4 coordination", numerical.public_d4_coordination,
                         numerical.d4_enabled != 0u ? atoms : 0, double);
    XTBLOOM_REJECT_WRITE("public point-charge shell potential", numerical.public_point_shell,
                         numerical.point_enabled != 0u ? numerical.total_shells : 0, double);
    XTBLOOM_REJECT_WRITE("overlap factor generations", numerical.factor_generations, batch,
                         std::uint64_t);
    XTBLOOM_REJECT_WRITE("overlap factor statuses", numerical.factor_statuses, batch,
                         std::uint32_t);
    XTBLOOM_REJECT_WRITE("geometry epoch", numerical.geometry_epoch, 1, std::uint64_t);

    const auto& mixer = candidate.state_seed.mixer;
    XTBLOOM_REJECT_WRITE("warm mixer residual RMS", mixer.residual_rms, mixer.batch_elements,
                         double);
    XTBLOOM_REJECT_WRITE("warm mixer residual maximum", mixer.residual_maximum,
                         mixer.batch_elements, double);
    XTBLOOM_REJECT_WRITE("warm mixer iterations", mixer.iterations, mixer.batch_elements,
                         std::uint64_t);
    XTBLOOM_REJECT_WRITE("warm mixer statuses", mixer.system_statuses, mixer.batch_elements,
                         xtbloom_status_t);
    XTBLOOM_REJECT_WRITE("warm mixer convergence", mixer.residual_converged, mixer.batch_elements,
                         std::uint8_t);
    const auto& scc = candidate.state_seed.scc;
    XTBLOOM_REJECT_WRITE("SCC free energies", scc.free_energies, scc.batch_elements, double);
    XTBLOOM_REJECT_WRITE("SCC previous free energies", scc.previous_free_energies,
                         scc.batch_elements, double);
    XTBLOOM_REJECT_WRITE("SCC free-energy changes", scc.free_energy_changes, scc.batch_elements,
                         double);
    XTBLOOM_REJECT_WRITE("SCC residual RMS", scc.residual_rms, scc.batch_elements, double);
    XTBLOOM_REJECT_WRITE("SCC iterations", scc.iterations, scc.batch_elements, std::uint64_t);
    XTBLOOM_REJECT_WRITE("SCC statuses", scc.system_statuses, scc.batch_elements, xtbloom_status_t);
    XTBLOOM_REJECT_WRITE("SCC convergence", scc.converged, scc.batch_elements, std::uint8_t);
    XTBLOOM_REJECT_WRITE("SCC report system diagnostics", candidate.report_storage.system_errors,
                         candidate.report_storage.system_error_elements, std::uint32_t);
    XTBLOOM_REJECT_WRITE("SCC report device diagnostics", candidate.report_storage.device_errors,
                         candidate.report_storage.device_error_elements, std::uint32_t);
    XTBLOOM_REJECT_WRITE("SCC report sequence latches", candidate.report_storage.sequence_latches,
                         candidate.report_storage.sequence_latch_elements, std::uint32_t);

    const auto& projection = candidate.energy_force.stationary_projection;
    if (projection.enabled == 1u) {
      XTBLOOM_REJECT_WRITE("stationary total density", projection.total_density,
                           projection.total_matrix_elements, double);
      XTBLOOM_REJECT_WRITE("stationary energy-weighted density",
                           projection.total_energy_weighted_density,
                           projection.total_matrix_elements, double);
      XTBLOOM_REJECT_WRITE("stationary spin density", projection.spin_density,
                           projection.total_matrix_elements, double);
      XTBLOOM_REJECT_WRITE("stationary shell charges", projection.shell_charges,
                           projection.total_shells, double);
      XTBLOOM_REJECT_WRITE("stationary atomic charges", projection.atomic_charges,
                           projection.total_atoms, double);
      XTBLOOM_REJECT_WRITE("stationary atomic dipoles", projection.atomic_dipoles,
                           3 * projection.total_atoms, double);
      XTBLOOM_REJECT_WRITE("stationary atomic quadrupoles", projection.atomic_quadrupoles,
                           6 * projection.total_atoms, double);
      XTBLOOM_REJECT_WRITE("stationary spin-shell potentials", projection.spin_shell_potentials,
                           projection.total_shells, double);
    }

    const auto& inference = candidate.inference;
    XTBLOOM_REJECT_WRITE("warm checkpoint aggregate", inference.warm_checkpoint_batch_ready, 1,
                         std::uint32_t);
    XTBLOOM_REJECT_WRITE("warm checkpoint generation publication",
                         inference.warm_checkpoint_generations, batch, std::uint64_t);
    XTBLOOM_REJECT_WRITE("warm checkpoint attachment publication",
                         inference.warm_checkpoint_field_attached, batch, std::uint8_t);
    XTBLOOM_REJECT_WRITE("warm checkpoint field publication",
                         inference.warm_checkpoint_field_vectors, field_coordinates, double);
    XTBLOOM_REJECT_WRITE("inference energy results", inference.publication_results.energies,
                         inference.publication_results.energy_elements, double);
    XTBLOOM_REJECT_WRITE("inference force results", inference.publication_results.qm_forces,
                         inference.publication_results.qm_force_elements, double);
    XTBLOOM_REJECT_WRITE("inference charge results", inference.publication_results.atomic_charges,
                         inference.publication_results.atomic_charge_elements, double);
    XTBLOOM_REJECT_WRITE("inference point-force results",
                         inference.publication_results.point_forces,
                         inference.publication_results.point_force_elements, double);
    XTBLOOM_REJECT_WRITE("inference iteration results", inference.publication_results.iterations,
                         inference.publication_results.batch_elements, std::int32_t);
    XTBLOOM_REJECT_WRITE("inference convergence results", inference.publication_results.converged,
                         inference.publication_results.batch_elements, std::uint8_t);
    XTBLOOM_REJECT_WRITE("inference status results", inference.publication_results.system_statuses,
                         inference.publication_results.batch_elements, xtbloom_status_t);
    XTBLOOM_REJECT_WRITE("inference dipole results", inference.publication_results.dipole_moments,
                         inference.publication_results.dipole_moment_elements, double);
    XTBLOOM_REJECT_WRITE("inference epoch snapshot", inference.publication_workspace.epoch_snapshot,
                         inference.publication_workspace.epoch_snapshot_elements, std::uint64_t);
    XTBLOOM_REJECT_WRITE("inference system diagnostics",
                         inference.publication_diagnostics.system_errors,
                         inference.publication_diagnostics.system_error_elements, std::uint32_t);
    XTBLOOM_REJECT_WRITE("inference plan diagnostic", inference.publication_diagnostics.plan_error,
                         inference.publication_diagnostics.plan_error_elements, std::uint32_t);

    const auto& public_result = candidate.public_result;
    XTBLOOM_REJECT_WRITE("public-result energy staging", public_result.device_staging.energies,
                         public_result.device_staging.energy_elements, double);
    XTBLOOM_REJECT_WRITE("public-result force staging", public_result.device_staging.qm_forces,
                         public_result.device_staging.qm_force_elements, double);
    XTBLOOM_REJECT_WRITE("public-result charge staging",
                         public_result.device_staging.atomic_charges,
                         public_result.device_staging.atomic_charge_elements, double);
    XTBLOOM_REJECT_WRITE("public-result point-force staging",
                         public_result.device_staging.point_forces,
                         public_result.device_staging.point_force_elements, double);
    XTBLOOM_REJECT_WRITE("public-result iteration staging", public_result.device_staging.iterations,
                         public_result.device_staging.batch_elements, std::int32_t);
    XTBLOOM_REJECT_WRITE("public-result convergence staging",
                         public_result.device_staging.converged,
                         public_result.device_staging.batch_elements, std::uint8_t);
    XTBLOOM_REJECT_WRITE("public-result status staging",
                         public_result.device_staging.system_statuses,
                         public_result.device_staging.batch_elements, xtbloom_status_t);
    XTBLOOM_REJECT_WRITE("public-result dipole staging",
                         public_result.device_staging.dipole_moments,
                         public_result.device_staging.dipole_moment_elements, double);
    XTBLOOM_REJECT_WRITE("public-result diagnostics", public_result.diagnostics.control,
                         public_result.diagnostics.control_elements, Gfn2PublicResultBridgeControl);
#undef XTBLOOM_REJECT_WRITE
    return XTBLOOM_STATUS_SUCCESS;
  }

  /* A native periodic request cannot yet execute its Ewald/multipole kernels
   * on the device.  Keep the asynchronous public contract nevertheless: the
   * worker owns the validated CPU reference bridge, while query(false) only
   * observes this completion record and never performs a blocking CUDA or CPU
   * operation.  The request retains the caller's borrowed descriptors until
   * the worker has published all requested result buffers. */
  struct NativePeriodicAsyncState {
    ~NativePeriodicAsyncState() {
      if (worker.joinable()) worker.join();
    }

    std::mutex mutex;
    std::condition_variable condition;
    bool done = false;
    xtbloom_status_t status = XTBLOOM_STATUS_INTERNAL_ERROR;
    std::uint32_t result_flags = 0u;
    std::string error;
    std::thread worker;
  };

  struct ActiveRequest {
    std::uint64_t id = 0u;
    xtbloom_compute_options_t options{};
    xtbloom_batch_result_t result{};
    PublicResultTransaction transaction{};
    bool completion_ready = false;
    xtbloom_status_t completion_status = XTBLOOM_STATUS_SUCCESS;
    std::uint32_t result_flags = 0u;
    std::string completion_error;
    xtbloom_status_t deferred_status = XTBLOOM_STATUS_SUCCESS;
    std::string deferred_error;
    bool host_upload_release_ordered = false;
    std::shared_ptr<NativePeriodicAsyncState> native_periodic_async;
    /* A post-Graph host-side submission failure is already an accepted CUDA
     * request because the Graph may have consumed persistent state. If the
     * first exact-stream settlement attempt itself fails, keep the request
     * PENDING and retain this cache as its completion owner until a later
     * wait/destroy proves both the owner and private callback streams idle. */
    bool settlement_only_pending = false;
    /* Context enqueue builds a new Prepared owner transactionally and does not
     * publish it into the cache until caller-output commit is accepted. If a
     * post-launch failure cannot be settled immediately, retain that candidate
     * here so queued GPU work cannot outlive its arenas. The matching topology
     * staging candidate remains unpublished and is aborted only after both
     * execution streams are known idle. */
    std::unique_ptr<Prepared> pending_prepared;
    bool pending_topology_candidate = false;
  };

  /* Native XYZ periodic requests use the CPU periodic evaluator as a
   * correctness bridge until the CUDA Ewald/multipole kernels are available.
   * The cache is retained per CUDA context so repeated synchronous calls keep
   * the same SCC checkpoint and strict WARM semantics as the CPU backend. */
  std::unique_ptr<Gfn2CpuExecutionCache> native_periodic_cpu_cache;

  static bool request_semantic_rejection(const PublicResultState& state) noexcept {
    const auto aggregate =
        static_cast<Gfn2PublicResultBridgeError>(state.host_control->aggregate_error);
    return aggregate == Gfn2PublicResultBridgeError::kRequestTopologyMismatch ||
           aggregate == Gfn2PublicResultBridgeError::kRequestInvalidArgument ||
           aggregate == Gfn2PublicResultBridgeError::kRequestNotSupported ||
           aggregate == Gfn2PublicResultBridgeError::kRequestNotImplemented ||
           aggregate == Gfn2PublicResultBridgeError::kRequestWarmIncompatible;
  }

  Impl(std::int32_t selected_device, void* selected_stream) noexcept
      : device_id(selected_device),
        stream(reinterpret_cast<cudaStream_t>(selected_stream)),
        topology_staging(selected_device, selected_stream) {}

  ~Impl() {
    std::lock_guard<std::mutex> lock(mutex);
    /* CUDA current-device state is thread-local. Context teardown can run on a
     * different thread or after the caller selected another device. Select the
     * owner only for teardown and restore the caller's selection afterwards;
     * destructors cannot report a restoration failure, so both operations are
     * necessarily best effort. */
    int caller_device = -1;
    const bool caller_device_known = cudaGetDevice(&caller_device) == cudaSuccess;
    bool owner_device_selected = caller_device_known && caller_device == device_id;
    bool restore_caller_device = false;
    if (!owner_device_selected && cudaSetDevice(device_id) == cudaSuccess) {
      owner_device_selected = true;
      restore_caller_device = caller_device_known && caller_device != device_id;
    }

    /* One owner-stream fence settles both handle-owned work and the native
     * lattice D2H. A failed or wrong-device fence cannot authorize freeing the
     * pinned destination: detach it below so member destruction cannot issue
     * cudaFreeHost while DMA may still reference the image. */
    cudaError_t owner_stream_status = cudaSuccess;
    bool owner_stream_settlement_attempted = false;
    if (owner_device_selected && (handles_created || native_lattice_staging_pending)) {
      owner_stream_settlement_attempted = true;
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      if (native_lattice_staging_pending &&
          consume_execution_test_fault(
              Gfn2CudaExecutionTestFault::kNativeLatticeTeardownSettlement)) {
        g_native_lattice_teardown_faults.fetch_add(1u, std::memory_order_relaxed);
        owner_stream_status = cudaErrorUnknown;
      } else
#endif
      {
        owner_stream_status = cudaStreamSynchronize(stream);
      }
    }
    if (native_lattice_staging_pending) {
      const bool settled = owner_stream_settlement_attempted && owner_stream_status == cudaSuccess;
      if (settled) {
        native_lattice_staging_pending = false;
      } else {
        const std::size_t quarantined_bytes = native_lattice_host_arena.bytes();
        void* const quarantined = native_lattice_host_arena.release_without_free();
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
        if (quarantined != nullptr) {
          g_quarantined_native_lattice_arenas.fetch_add(1u, std::memory_order_relaxed);
          g_quarantined_native_lattice_bytes.fetch_add(
              static_cast<std::uint64_t>(quarantined_bytes), std::memory_order_relaxed);
        }
#else
        (void)quarantined;
        (void)quarantined_bytes;
#endif
      }
    }
    if (active_request.native_periodic_async != nullptr &&
        active_request.native_periodic_async->worker.joinable()) {
      /* Native periodic async bridge workers reference this implementation;
       * join them before member teardown begins. */
      active_request.native_periodic_async->worker.join();
    }
    if (prepared != nullptr) {
      prepared.reset();
    }
    if (active_request.pending_prepared != nullptr) {
      active_request.pending_prepared.reset();
    }
    if (active_request.pending_topology_candidate) {
      topology_staging.abort_candidate();
      active_request.pending_topology_candidate = false;
    }
    if (blas != nullptr) (void)cublasDestroy(blas);
    if (solver_jacobi != nullptr) (void)cusolverDnDestroySyevjInfo(solver_jacobi);
    if (solver_parameters != nullptr) (void)cusolverDnDestroyParams(solver_parameters);
    if (solver != nullptr) (void)cusolverDnDestroy(solver);
    if (restore_caller_device) (void)cudaSetDevice(caller_device);
  }

  /* Exception-only safety net for asynchronous admission while mutex is held.
   * Normal status failures use the diagnostic-preserving settlement helpers;
   * this no-throw path exists so an allocation or string exception can never
   * strand queued work after the public Request rolls back its reservation. */
  void settle_active_request_exception_noexcept_locked() noexcept {
    if (active_request.id == 0u) return;

    if (active_request.native_periodic_async != nullptr) {
      /* The worker does not take the cache mutex, so joining while this guard
       * still owns it is safe and guarantees no borrowed descriptor survives a
       * failed enqueue handoff. */
      if (active_request.native_periodic_async->worker.joinable()) {
        active_request.native_periodic_async->worker.join();
      }
      active_request = {};
      return;
    }

    int caller_device = -1;
    const bool caller_device_known = cudaGetDevice(&caller_device) == cudaSuccess;
    const bool owner_selected = (caller_device_known && caller_device == device_id) ||
                                cudaSetDevice(device_id) == cudaSuccess;
    bool settled = false;
    Prepared* const active_prepared = active_request.pending_prepared != nullptr
                                          ? active_request.pending_prepared.get()
                                          : prepared.get();
    if (owner_selected && active_prepared != nullptr) {
      auto& current = *active_prepared;
      cudaError_t owner_status = cudaSuccess;
      if (current.submitted) {
        owner_status = cudaStreamSynchronize(stream);
        if (owner_status == cudaSuccess) current.submitted = false;
      }
      cudaError_t host_status = cudaSuccess;
      if (current.numerical_host_completion_stream.valid()) {
        host_status = cudaStreamSynchronize(current.numerical_host_completion_stream.get());
        if (host_status == cudaSuccess) {
          current.numerical_host_upload_completion.pending.store(false, std::memory_order_release);
        }
      }
      settled = owner_status == cudaSuccess && host_status == cudaSuccess;
    } else if (owner_selected) {
      settled = true;
    }

    if (caller_device_known && caller_device != device_id) {
      (void)cudaSetDevice(caller_device);
    }
    if (settled) {
      if (active_request.pending_topology_candidate) {
        topology_staging.abort_candidate();
      }
      active_request = {};
    } else {
      request_poisoned = true;
      if (active_prepared != nullptr) active_prepared->numerical.host_staging_poisoned = true;
    }
  }

  xtbloom_status_t ensure_handles(std::string& error) {
    /* cudaSetDevice is intentionally repeated on every prepare. CUDA current
     * device selection is thread-local and is not preserved by this context. */
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice", cuda_status);
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    if (!ensure_cuda_gfn2_parameters(device_id, error)) {
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    if (handles_created) return XTBLOOM_STATUS_SUCCESS;

    /* Publish context handles only after the entire construction succeeds.
     * This keeps retries leak-free after a partial provider failure. */
    cusolverDnHandle_t candidate_solver = nullptr;
    cusolverDnParams_t candidate_parameters = nullptr;
    syevjInfo_t candidate_jacobi = nullptr;
    cublasHandle_t candidate_blas = nullptr;
    const auto destroy_candidate = [&]() noexcept {
      if (candidate_blas != nullptr) (void)cublasDestroy(candidate_blas);
      if (candidate_jacobi != nullptr) (void)cusolverDnDestroySyevjInfo(candidate_jacobi);
      if (candidate_parameters != nullptr) (void)cusolverDnDestroyParams(candidate_parameters);
      if (candidate_solver != nullptr) (void)cusolverDnDestroy(candidate_solver);
    };

    cusolverStatus_t solver_status = cusolverDnCreate(&candidate_solver);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      error = "cusolverDnCreate failed";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    solver_status = cusolverDnCreateParams(&candidate_parameters);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      destroy_candidate();
      error = "cusolverDnCreateParams failed";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    solver_status = cusolverDnCreateSyevjInfo(&candidate_jacobi);
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status =
          cusolverDnXsyevjSetTolerance(candidate_jacobi, std::numeric_limits<double>::epsilon());
    }
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status = cusolverDnXsyevjSetMaxSweeps(candidate_jacobi, 100);
    }
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status = cusolverDnXsyevjSetSortEig(candidate_jacobi, 1);
    }
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      destroy_candidate();
      error = "failed to configure the CUDA small-matrix Jacobi eigensolver";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    cublasStatus_t blas_status = cublasCreate(&candidate_blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
      destroy_candidate();
      error = "cublasCreate failed";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    solver_status = cusolverDnSetStream(candidate_solver, stream);
    blas_status = cublasSetStream(candidate_blas, stream);
    if (solver_status != CUSOLVER_STATUS_SUCCESS || blas_status != CUBLAS_STATUS_SUCCESS) {
      destroy_candidate();
      error = "failed to bind CUDA linear-algebra handles to the context stream";
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }
    solver = candidate_solver;
    solver_parameters = candidate_parameters;
    solver_jacobi = candidate_jacobi;
    blas = candidate_blas;
    handles_created = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t execute_native_periodic_cpu_bridge_locked(
      const xtbloom_batch_t& source, const xtbloom_compute_options_t& options,
      xtbloom_batch_result_t& destination, bool require_prepared_topology, std::string& error) {
    if (options.model != XTBLOOM_MODEL_GFN2_XTB) {
      error = "native periodic execution is released only for GFN2";
      return XTBLOOM_STATUS_NOT_IMPLEMENTED;
    }
    const cudaError_t fence_status = cudaStreamSynchronize(stream);
    if (fence_status != cudaSuccess) {
      error = cuda_error_message("CUDA native periodic ingress fence", fence_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    xtbloom_status_t status = validate_native_lattice_request_sync(source, error, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    /* Native periodic requests use the CPU evaluator only as their current
     * scientific implementation, but their immutable cell/mask identity is
     * still owned by the CUDA topology staging transaction. Stage before
     * running the bridge so device-resident descriptors are validated without
     * host dereference, and so a changed cell cannot silently reuse the prior
     * topology. A candidate is published only after the bridge has produced a
     * complete result; any failure aborts it and preserves the old topology. */
    const Gfn2CudaTopologyStagingDiagnostic topology_staged =
        topology_staging.stage_and_validate(source, error);
    if (!topology_staged.success()) return topology_staged.status;
    bool topology_candidate_pending =
        topology_staged.disposition == Gfn2CudaTopologyStageDisposition::kCandidate;
    const auto abort_topology_candidate = [&]() noexcept {
      if (topology_candidate_pending) {
        topology_staging.abort_candidate();
        topology_candidate_pending = false;
      }
    };

    NativePeriodicHostBridgeStorage staged;
    status = staged.stage_batch(source, stream, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      abort_topology_candidate();
      return status;
    }
    if ((options.flags & XTBLOOM_COMPUTE_STRAIN_DERIVATIVES) != 0u) {
      bool any_periodic = false;
      bool all_xyz = true;
      for (const std::int32_t mask : staged.periodic_axes) {
        any_periodic = any_periodic || mask == XTBLOOM_PERIODIC_AXES_XYZ;
        all_xyz = all_xyz && mask == XTBLOOM_PERIODIC_AXES_XYZ;
      }
      if (!all_xyz) {
        error = any_periodic ? "native-periodic strain derivatives require every batch item to use "
                               "native XYZ periodicity"
                             : "native-periodic strain derivatives require native XYZ periodic "
                               "descriptors";
        abort_topology_candidate();
        return any_periodic ? XTBLOOM_STATUS_NOT_SUPPORTED : XTBLOOM_STATUS_NOT_IMPLEMENTED;
      }
    }
    if (require_prepared_topology) {
      if (prepared == nullptr) {
        abort_topology_candidate();
        error = "CUDA native periodic plan has no prepared fixed-topology runtime";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      if (topology_candidate_pending) {
        abort_topology_candidate();
        error = "the batch topology does not match the fixed CUDA plan topology";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const TopologyMatch match =
          match_existing_topology(staged.batch, options, prepared->host.key, error);
      if (match == TopologyMatch::kInvalid) {
        abort_topology_candidate();
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (match != TopologyMatch::kMatch) {
        abort_topology_candidate();
        error = "the batch topology does not match the fixed CUDA plan topology";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }

    staged.bind_result(destination, options);
    try {
      if (native_periodic_cpu_cache == nullptr) {
        native_periodic_cpu_cache = std::make_unique<Gfn2CpuExecutionCache>(0);
      }
    } catch (const std::bad_alloc&) {
      abort_topology_candidate();
      error = "failed to allocate the native periodic CPU reference cache";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    status = execute_restricted_gfn2_cpu(*native_periodic_cpu_cache, staged.batch, options,
                                         staged.result, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      abort_topology_candidate();
      return status;
    }
    if (topology_candidate_pending) {
      const Gfn2CudaTopologyStagingDiagnostic prepared_commit =
          topology_staging.prepare_candidate_commit(error);
      if (!prepared_commit.success()) {
        abort_topology_candidate();
        return prepared_commit.status;
      }
      if (!topology_staging.candidate_publishable()) {
        abort_topology_candidate();
        error = "native periodic CUDA topology candidate is not publishable";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
    status = staged.publish_result(destination, stream, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      abort_topology_candidate();
      return status;
    }
    if (topology_candidate_pending && !topology_staging.publish_candidate()) {
      error = "native periodic CUDA topology publication invariant failed";
      /* Keep the staging owner reusable after an invariant failure.  The
       * caller-output commit has already completed, so this is reported as a
       * catastrophic internal error, but retaining the unpublished candidate
       * would poison every later request with kCandidatePending. */
      abort_topology_candidate();
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    destination.flags = staged.result.flags;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t enqueue_native_periodic_cpu_bridge_locked(
      const xtbloom_batch_t& source, const xtbloom_compute_options_t& options,
      const xtbloom_batch_result_t& destination, bool require_prepared_topology,
      NativePeriodicAsyncState& state, std::string& error) {
    /* Capture only descriptor values.  The public request contract keeps the
     * caller-owned arrays and result descriptor alive until request completion;
     * the worker therefore may safely consume their borrowed pointers after
     * this enqueue function returns. */
    const xtbloom_batch_t source_copy = source;
    const xtbloom_compute_options_t options_copy = options;
    const xtbloom_batch_result_t destination_copy = destination;
    xtbloom_batch_result_t* const destination_owner =
        const_cast<xtbloom_batch_result_t*>(&destination);
    try {
      state.worker = std::thread([this, &state, source_copy, options_copy, destination_copy,
                                  destination_owner, require_prepared_topology] {
        xtbloom_status_t status = XTBLOOM_STATUS_INTERNAL_ERROR;
        std::string worker_error;
        std::uint32_t worker_result_flags = 0u;
        int previous_device = -1;
        const bool previous_device_known = cudaGetDevice(&previous_device) == cudaSuccess;
        try {
          const cudaError_t select_status = cudaSetDevice(device_id);
          if (select_status == cudaSuccess) {
            xtbloom_batch_result_t worker_result = destination_copy;
            status = execute_native_periodic_cpu_bridge_locked(
                source_copy, options_copy, worker_result, require_prepared_topology, worker_error);
            if (status == XTBLOOM_STATUS_SUCCESS) {
              /* Result flags are the only descriptor field published by the
               * asynchronous backend.  Buffer bytes are committed by the bridge
               * before this store and are observed after the completion record. */
              destination_owner->flags = worker_result.flags;
              worker_result_flags = worker_result.flags;
            }
          } else {
            worker_error =
                cuda_error_message("CUDA native periodic worker device selection", select_status);
            status = XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
          }
        } catch (const std::bad_alloc&) {
          worker_error = "failed to allocate while executing the native periodic CUDA bridge";
          status = XTBLOOM_STATUS_ALLOCATION_FAILED;
        } catch (const std::exception& exception) {
          worker_error = exception.what();
          status = XTBLOOM_STATUS_INTERNAL_ERROR;
        } catch (...) {
          worker_error = "unknown exception while executing the native periodic CUDA bridge";
          status = XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        if (previous_device_known && previous_device != device_id) {
          const cudaError_t restore_status = cudaSetDevice(previous_device);
          if (restore_status != cudaSuccess && status == XTBLOOM_STATUS_SUCCESS) {
            worker_error = cuda_error_message("CUDA native periodic worker device restoration",
                                              restore_status);
            status = XTBLOOM_STATUS_INTERNAL_ERROR;
          }
        }
        {
          std::lock_guard<std::mutex> lock(state.mutex);
          state.status = status;
          state.error = std::move(worker_error);
          state.result_flags = status == XTBLOOM_STATUS_SUCCESS ? worker_result_flags : 0u;
          state.done = true;
        }
        state.condition.notify_all();
      });
    } catch (const std::system_error& exception) {
      error =
          std::string("failed to start native periodic CUDA bridge worker: ") + exception.what();
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    } catch (const std::bad_alloc&) {
      error = "failed to allocate native periodic CUDA bridge worker state";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t build_integral_task_domains(Prepared& candidate, std::string& error) {
    const auto& host = candidate.host.integral_task_domains;
    struct Offsets {
      std::size_t forward_generic = 0u;
      std::size_t forward_ss = 0u;
      std::size_t h0_generic = 0u;
      std::size_t h0_ss = 0u;
      std::size_t force_generic = 0u;
      std::size_t force_ss = 0u;
    } offset;
    ArenaLayout layout;
    offset.forward_generic = layout.append<Gfn2IntegralShellPairTask>(
        static_cast<std::int64_t>(host.forward_generic.size()));
    offset.forward_ss =
        layout.append<Gfn2IntegralShellPairTask>(static_cast<std::int64_t>(host.forward_ss.size()));
    offset.h0_generic =
        layout.append<Gfn2IntegralShellPairTask>(static_cast<std::int64_t>(host.h0_generic.size()));
    offset.h0_ss =
        layout.append<Gfn2IntegralShellPairTask>(static_cast<std::int64_t>(host.h0_ss.size()));
    offset.force_generic = layout.append<Gfn2IntegralShellPairTask>(
        static_cast<std::int64_t>(host.force_generic.size()));
    offset.force_ss =
        layout.append<Gfn2IntegralShellPairTask>(static_cast<std::int64_t>(host.force_ss.size()));
    if (!layout.valid()) {
      error = "CUDA integral task arena layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.integral_task_arena.allocate(layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA integral task arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    void* const arena = candidate.integral_task_arena.get();
    const auto upload = [&](std::size_t destination_offset, const auto& values) {
      if (values.empty()) return cudaSuccess;
      using Value = typename std::decay_t<decltype(values)>::value_type;
      return cudaMemcpyAsync(arena_pointer<Value>(arena, destination_offset), values.data(),
                             values.size() * sizeof(Value), cudaMemcpyHostToDevice, stream);
    };
    /* Any successful upload below can remain in flight if a later enqueue
     * fails. Mark the candidate before the first enqueue so its destructor
     * synchronizes the owner stream before releasing the shared arena. */
    candidate.submitted = true;
    cuda_status = upload(offset.forward_generic, host.forward_generic);
    if (cuda_status == cudaSuccess) cuda_status = upload(offset.forward_ss, host.forward_ss);
    if (cuda_status == cudaSuccess) cuda_status = upload(offset.h0_generic, host.h0_generic);
    if (cuda_status == cudaSuccess) cuda_status = upload(offset.h0_ss, host.h0_ss);
    if (cuda_status == cudaSuccess) cuda_status = upload(offset.force_generic, host.force_generic);
    if (cuda_status == cudaSuccess) cuda_status = upload(offset.force_ss, host.force_ss);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA integral task upload", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    candidate.integral_tasks = {
        static_cast<std::int64_t>(host.forward_generic.size()),
        static_cast<std::int64_t>(host.forward_ss.size()),
        static_cast<std::int64_t>(host.h0_generic.size()),
        static_cast<std::int64_t>(host.h0_ss.size()),
        static_cast<std::int64_t>(host.force_generic.size()),
        static_cast<std::int64_t>(host.force_ss.size()),
        arena_pointer_if<Gfn2IntegralShellPairTask>(
            arena, offset.forward_generic, static_cast<std::int64_t>(host.forward_generic.size())),
        arena_pointer_if<Gfn2IntegralShellPairTask>(
            arena, offset.forward_ss, static_cast<std::int64_t>(host.forward_ss.size())),
        arena_pointer_if<Gfn2IntegralShellPairTask>(
            arena, offset.h0_generic, static_cast<std::int64_t>(host.h0_generic.size())),
        arena_pointer_if<Gfn2IntegralShellPairTask>(arena, offset.h0_ss,
                                                    static_cast<std::int64_t>(host.h0_ss.size())),
        arena_pointer_if<Gfn2IntegralShellPairTask>(
            arena, offset.force_generic, static_cast<std::int64_t>(host.force_generic.size())),
        arena_pointer_if<Gfn2IntegralShellPairTask>(
            arena, offset.force_ss, static_cast<std::int64_t>(host.force_ss.size()))};
    return XTBLOOM_STATUS_SUCCESS;
  }

  static void bind_integral_task_domains(const Prepared& candidate,
                                         Gfn2IntegralDeviceBatch& batch) noexcept {
    batch.use_compact_tasks = candidate.host.compact_integral_tasks ? 1u : 0u;
    batch.forward_generic_task_count = candidate.integral_tasks.forward_generic_task_count;
    batch.forward_ss_task_count = candidate.integral_tasks.forward_ss_task_count;
    batch.h0_generic_task_count = candidate.integral_tasks.h0_generic_task_count;
    batch.h0_ss_task_count = candidate.integral_tasks.h0_ss_task_count;
    batch.force_generic_task_count = candidate.integral_tasks.force_generic_task_count;
    batch.force_ss_task_count = candidate.integral_tasks.force_ss_task_count;
    batch.forward_generic_tasks = candidate.integral_tasks.forward_generic_tasks;
    batch.forward_ss_tasks = candidate.integral_tasks.forward_ss_tasks;
    batch.h0_generic_tasks = candidate.integral_tasks.h0_generic_tasks;
    batch.h0_ss_tasks = candidate.integral_tasks.h0_ss_tasks;
    batch.force_generic_tasks = candidate.integral_tasks.force_generic_tasks;
    batch.force_ss_tasks = candidate.integral_tasks.force_ss_tasks;
  }

  xtbloom_status_t build_numerical_refresh_binding(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t shells = candidate.host.basis.total_shells;
    const std::int64_t orbitals = candidate.host.basis.total_orbitals;
    const std::int64_t primitives = candidate.host.basis.total_primitives;
    const std::int64_t matrices = candidate.host.integrals.total_matrix_elements;
    const std::int64_t points = candidate.host.external.total_point_charges;
    const std::int64_t response =
        static_cast<std::int64_t>(candidate.host.periodic_response.size());
    const std::int64_t geometry_pairs = candidate.plan_seed.geometry_batch.total_pairs;
    const std::int64_t es2_elements = candidate.plan_seed.es2_batch.total_matrix_elements;
    const bool multipoles_enabled = !candidate.host.gfn1_enabled;
    const bool aes2_enabled = !candidate.host.gfn1_enabled;
    const std::int64_t aes2_pairs = aes2_enabled ? candidate.plan_seed.aes2_batch.total_pairs : 0;
    const std::uint64_t token = candidate.host.plan_token;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t geometry_pair_elements = 0;
    std::int64_t aes2_elements = 0;
    std::int64_t dipole_elements = 0;
    std::int64_t quadrupole_elements = 0;
    /* Fixed-topology capacities for the optional sparse pair-list gate.  The
     * cell arrays use a per-system trailing slot so adjacent systems never
     * race on the exclusive-end sentinel. */
    std::int64_t maximum_system_atoms = 0;
    for (std::int64_t system = 0; system < batch; ++system) {
      maximum_system_atoms =
          std::max(maximum_system_atoms,
                   candidate.host.basis.atom_offsets[static_cast<std::size_t>(system + 1)] -
                       candidate.host.basis.atom_offsets[static_cast<std::size_t>(system)]);
    }
    /* Per-system sparse/dense dispatch decisions, uploaded once at setup. Each
     * peer independently crosses the measured 40-atom crossover, so a
     * heterogeneous batch no longer applies one strategy to every member. D4
     * still forces the leaf to exist for small dense peers; dense mode then
     * publishes the full triangle into the same committed fixed-capacity
     * transaction used by sparse peers. */
    std::vector<std::int32_t> pairlist_system_modes(static_cast<std::size_t>(batch));
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int64_t atoms_per_system =
          candidate.host.basis.atom_offsets[static_cast<std::size_t>(system + 1)] -
          candidate.host.basis.atom_offsets[static_cast<std::size_t>(system)];
      pairlist_system_modes[static_cast<std::size_t>(system)] = static_cast<std::int32_t>(
          xtbloom::detail::cuda::gfn2_pairlist_use_sparse_for(atoms_per_system)
              ? xtbloom::detail::cuda::Gfn2PairListMode::kSparse
              : xtbloom::detail::cuda::Gfn2PairListMode::kDense);
    }
    std::int64_t scaled_cells = 0;
    if (!checked_elements(maximum_system_atoms, 8, scaled_cells)) {
      error = "numerical refresh sparse cell capacity overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    std::int64_t sparse_cells_per_system = std::max<std::int64_t>(16, scaled_cells);
    std::int64_t sparse_neighbors_per_atom = maximum_system_atoms;
    std::int64_t sparse_pairs_per_system = 0;
    if (!checked_triangle(maximum_system_atoms, sparse_pairs_per_system)) {
      error = "numerical refresh sparse pair capacity overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    /* The committed pair-list schema requires a positive fixed slot capacity
     * even when every system is a singleton. Reserve one inert pair slot per
     * peer; builders still publish pair_count=0, so no synthetic pair becomes
     * visible to D4 or coordination consumers. */
    sparse_pairs_per_system = std::max<std::int64_t>(1, sparse_pairs_per_system);
    std::int64_t sparse_cell_capacity = 0;
    std::int64_t sparse_neighbor_capacity = 0;
    std::int64_t sparse_pair_capacity = 0;
    if (!checked_elements(batch, sparse_cells_per_system, sparse_cell_capacity) ||
        !checked_elements(atoms, sparse_neighbors_per_atom, sparse_neighbor_capacity) ||
        !checked_elements(batch, sparse_pairs_per_system, sparse_pair_capacity) ||
        sparse_cell_capacity > std::numeric_limits<std::int64_t>::max() - batch) {
      error = "numerical refresh sparse pair-list capacity overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(geometry_pairs, kGfn2GeometryPairDataElements, geometry_pair_elements) ||
        !checked_elements(aes2_pairs, kGfn2AES2PairDataElements, aes2_elements) ||
        (multipoles_enabled && !checked_elements(matrices, 3, dipole_elements)) ||
        (multipoles_enabled && !checked_elements(matrices, 6, quadrupole_elements))) {
      error = "numerical refresh element count overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }

    struct Offsets {
      std::size_t shell_pair_offsets = 0u;
      std::size_t shell_primitive_offsets = 0u;
      std::size_t angular_momenta = 0u;
      std::size_t primitive_exponents = 0u;
      std::size_t primitive_coefficients = 0u;
      std::size_t h0_atomic_radii = 0u;
      std::size_t h0_shell_levels = 0u;
      std::size_t h0_coordination_scale = 0u;
      std::size_t h0_polynomial = 0u;
      std::size_t h0_pair_scale = 0u;

      std::size_t committed_positions = 0u;
      std::size_t committed_point_positions = 0u;
      std::size_t committed_point_values = 0u;
      std::size_t committed_point_gammas = 0u;
      std::size_t committed_periodic_shifts = 0u;
      std::size_t committed_periodic_response = 0u;
      std::size_t committed_field_attached = 0u;
      std::size_t committed_field_vectors = 0u;
      std::size_t candidate_field_attached = 0u;
      std::size_t candidate_field_vectors = 0u;
      std::size_t committed_field_atomic_potentials = 0u;
      std::size_t committed_field_dipole_potentials = 0u;
      std::size_t candidate_field_atomic_potentials = 0u;
      std::size_t candidate_field_dipole_potentials = 0u;
      std::size_t field_system_errors = 0u;
      std::size_t field_plan_error = 0u;
      std::size_t interaction_request_error = 0u;
      std::size_t candidate_positions = 0u;
      std::size_t candidate_point_positions = 0u;
      std::size_t candidate_point_values = 0u;
      std::size_t candidate_point_gammas = 0u;
      std::size_t candidate_periodic_shifts = 0u;
      std::size_t candidate_periodic_response = 0u;
      std::size_t host_positions = 0u;
      std::size_t host_point_positions = 0u;
      std::size_t host_point_values = 0u;
      std::size_t host_point_gammas = 0u;
      std::size_t host_periodic_shifts = 0u;
      std::size_t host_periodic_response = 0u;
      std::size_t requested = 0u;
      std::size_t host_requested = 0u;
      std::size_t eligible = 0u;
      std::size_t committed_generations = 0u;
      std::size_t refresh_predecessor_generations = 0u;
      std::size_t geometry_epoch = 0u;

      std::size_t output_geometry_pairs = 0u;
      std::size_t output_coordination = 0u;
      std::size_t output_geometry_generations = 0u;
      std::size_t output_overlap = 0u;
      std::size_t output_dipole = 0u;
      std::size_t output_quadrupole = 0u;
      std::size_t output_h0 = 0u;
      std::size_t output_es2 = 0u;
      std::size_t output_aes2 = 0u;
      std::size_t output_operator_generations = 0u;
      std::size_t output_published = 0u;

      std::size_t positions_scratch = 0u;
      std::size_t geometry_candidate_pairs = 0u;
      std::size_t geometry_candidate_coordination = 0u;
      std::size_t geometry_candidate_generations = 0u;
      std::size_t geometry_pair_scratch = 0u;
      std::size_t geometry_coordination_scratch = 0u;
      std::size_t geometry_sequence = 0u;
      std::size_t overlap_candidate = 0u;
      std::size_t dipole_candidate = 0u;
      std::size_t quadrupole_candidate = 0u;
      std::size_t h0_candidate = 0u;
      std::size_t overlap_scratch = 0u;
      std::size_t dipole_scratch = 0u;
      std::size_t quadrupole_scratch = 0u;
      std::size_t h0_scratch = 0u;
      std::size_t integral_sequence = 0u;
      std::size_t es2_candidate = 0u;
      std::size_t es2_scratch = 0u;
      std::size_t aes2_candidate = 0u;
      std::size_t aes2_scratch = 0u;
      std::size_t geometry_system_errors = 0u;
      std::size_t geometry_device_error = 0u;
      std::size_t integral_system_errors = 0u;
      std::size_t integral_device_error = 0u;
      std::size_t es2_device_error = 0u;
      std::size_t aes2_system_errors = 0u;
      std::size_t aes2_device_error = 0u;
      std::size_t preprocessing_stages = 0u;
      std::size_t preprocessing_plan_error = 0u;
      std::size_t sparse_system_errors = 0u;
      std::size_t sparse_device_error = 0u;
      std::size_t sparse_coordination = 0u;
      std::size_t pairlist_pairs = 0u;
      std::size_t pairlist_offsets = 0u;
      std::size_t pairlist_pair_counts = 0u;
      std::size_t pairlist_neighbor_offsets = 0u;
      std::size_t pairlist_neighbor_counts = 0u;
      std::size_t pairlist_neighbors = 0u;
      std::size_t pairlist_generations = 0u;
      std::size_t pairlist_system_modes = 0u;
      std::size_t pairlist_meta = 0u;
      std::size_t pairlist_atom_cells = 0u;
      std::size_t pairlist_cell_counts = 0u;
      std::size_t pairlist_cell_offsets = 0u;
      std::size_t pairlist_cell_fill = 0u;
      std::size_t pairlist_cell_atoms = 0u;
      std::size_t pairlist_neighbor_cursor = 0u;
      std::size_t pairlist_neighbor_scratch = 0u;
      std::size_t pairlist_pair_cursor = 0u;
      std::size_t pairlist_sequence = 0u;

      /* Committed public pair-list view (preprocessing step 4).  Distinct
       * storage from the candidate so a failed peer can never expose a partial
       * candidate slice and consumers read only eligible, current peers. */
      std::size_t committed_pairs = 0u;
      std::size_t committed_pair_offsets = 0u;
      std::size_t committed_pair_counts = 0u;
      std::size_t committed_neighbor_offsets = 0u;
      std::size_t committed_neighbor_counts = 0u;
      std::size_t committed_neighbors = 0u;
      std::size_t committed_pair_generations = 0u;
      std::size_t committed_eligible_mask = 0u;

      std::size_t d4_coordination_scratch = 0u;
      std::size_t d4_system_errors = 0u;
      std::size_t d4_device_error = 0u;
      std::size_t point_candidate_shell = 0u;
      std::size_t point_shell_scratch = 0u;
      std::size_t point_system_errors = 0u;
      std::size_t point_plan_error = 0u;
      std::size_t point_sequence = 0u;
      std::size_t periodic_system_errors = 0u;
      std::size_t periodic_plan_error = 0u;
    } offset;

    struct HostStagingOffsets {
      std::size_t positions = 0u;
      std::size_t point_positions = 0u;
      std::size_t point_values = 0u;
      std::size_t point_gammas = 0u;
      std::size_t periodic_shifts = 0u;
      std::size_t periodic_response = 0u;
      std::size_t requested = 0u;
    } host_offset;

    ArenaLayout layout;
    offset.shell_pair_offsets = layout.append<std::int64_t>(
        static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()));
    offset.shell_primitive_offsets = layout.append<std::int64_t>(
        static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()));
    offset.angular_momenta = layout.append<std::uint8_t>(
        static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()));
    offset.primitive_exponents = layout.append<double>(primitives);
    offset.primitive_coefficients = layout.append<double>(primitives);
    offset.h0_atomic_radii = layout.append<double>(atoms);
    offset.h0_shell_levels = layout.append<double>(shells);
    offset.h0_coordination_scale = layout.append<double>(shells);
    offset.h0_polynomial = layout.append<double>(shells);
    offset.h0_pair_scale = layout.append<double>(candidate.host.h0.shell_pair_offsets.back());

    const auto append_numerical =
        [&](std::size_t& positions_offset, std::size_t& point_positions_offset,
            std::size_t& point_values_offset, std::size_t& point_gammas_offset,
            std::size_t& shifts_offset, std::size_t& response_offset) {
          positions_offset = layout.append<double>(coordinates);
          point_positions_offset = layout.append<double>(point_coordinates);
          point_values_offset = layout.append<double>(points);
          point_gammas_offset = layout.append<double>(points);
          shifts_offset = layout.append<double>(candidate.host.periodic_enabled ? atoms : 0);
          response_offset = layout.append<double>(candidate.host.periodic_enabled ? response : 0);
        };
    append_numerical(offset.committed_positions, offset.committed_point_positions,
                     offset.committed_point_values, offset.committed_point_gammas,
                     offset.committed_periodic_shifts, offset.committed_periodic_response);
    append_numerical(offset.candidate_positions, offset.candidate_point_positions,
                     offset.candidate_point_values, offset.candidate_point_gammas,
                     offset.candidate_periodic_shifts, offset.candidate_periodic_response);
    append_numerical(offset.host_positions, offset.host_point_positions, offset.host_point_values,
                     offset.host_point_gammas, offset.host_periodic_shifts,
                     offset.host_periodic_response);
    offset.committed_field_attached = layout.append<std::uint8_t>(batch);
    offset.committed_field_vectors = layout.append<double>(3 * batch);
    offset.candidate_field_attached = layout.append<std::uint8_t>(batch);
    offset.candidate_field_vectors = layout.append<double>(3 * batch);
    offset.committed_field_atomic_potentials = layout.append<double>(atoms);
    offset.committed_field_dipole_potentials = layout.append<double>(coordinates);
    offset.candidate_field_atomic_potentials = layout.append<double>(atoms);
    offset.candidate_field_dipole_potentials = layout.append<double>(coordinates);
    offset.field_system_errors = layout.append<std::uint32_t>(batch);
    offset.field_plan_error = layout.append<std::uint32_t>(1);
    offset.interaction_request_error = layout.append<std::uint32_t>(1);
    offset.requested = layout.append<std::uint8_t>(batch);
    offset.host_requested = layout.append<std::uint8_t>(batch);
    offset.eligible = layout.append<std::uint8_t>(batch);
    offset.committed_generations = layout.append<std::uint64_t>(batch);
    offset.refresh_predecessor_generations = layout.append<std::uint64_t>(batch);
    offset.geometry_epoch = layout.append<std::uint64_t>(1);

    offset.output_geometry_pairs = layout.append<double>(geometry_pair_elements);
    offset.output_coordination = layout.append<double>(atoms);
    offset.output_geometry_generations = layout.append<std::uint64_t>(batch);
    offset.output_overlap = layout.append<double>(matrices);
    offset.output_dipole = layout.append<double>(dipole_elements);
    offset.output_quadrupole = layout.append<double>(quadrupole_elements);
    offset.output_h0 = layout.append<double>(matrices);
    offset.output_es2 = layout.append<double>(es2_elements);
    offset.output_aes2 = layout.append<double>(aes2_elements);
    offset.output_operator_generations = layout.append<std::uint64_t>(batch);
    offset.output_published = layout.append<std::uint8_t>(batch);

    offset.positions_scratch = layout.append<double>(coordinates);
    offset.geometry_candidate_pairs = layout.append<double>(geometry_pair_elements);
    offset.geometry_candidate_coordination = layout.append<double>(atoms);
    offset.geometry_candidate_generations = layout.append<std::uint64_t>(batch);
    offset.geometry_pair_scratch = layout.append<double>(geometry_pair_elements);
    offset.geometry_coordination_scratch = layout.append<double>(atoms);
    offset.geometry_sequence = layout.append<std::uint32_t>(1);
    offset.overlap_candidate = layout.append<double>(matrices);
    offset.dipole_candidate = layout.append<double>(dipole_elements);
    offset.quadrupole_candidate = layout.append<double>(quadrupole_elements);
    offset.h0_candidate = layout.append<double>(matrices);
    offset.overlap_scratch = layout.append<double>(matrices);
    offset.dipole_scratch = layout.append<double>(dipole_elements);
    offset.quadrupole_scratch = layout.append<double>(quadrupole_elements);
    offset.h0_scratch = layout.append<double>(matrices);
    offset.integral_sequence = layout.append<std::uint32_t>(1);
    offset.es2_candidate = layout.append<double>(es2_elements);
    offset.es2_scratch = layout.append<double>(es2_elements);
    offset.aes2_candidate = layout.append<double>(aes2_elements);
    offset.aes2_scratch = layout.append<double>(aes2_elements);
    offset.geometry_system_errors = layout.append<std::uint32_t>(batch);
    offset.geometry_device_error = layout.append<std::uint32_t>(1);
    offset.integral_system_errors = layout.append<std::uint32_t>(batch);
    offset.integral_device_error = layout.append<std::uint32_t>(1);
    offset.es2_device_error = layout.append<std::uint32_t>(1);
    offset.aes2_system_errors = layout.append<std::uint32_t>(batch);
    offset.aes2_device_error = layout.append<std::uint32_t>(1);
    offset.preprocessing_stages = layout.append<std::uint32_t>(batch);
    offset.preprocessing_plan_error = layout.append<std::uint32_t>(1);

    /* Optional sparse pair-list gate.  The segments are always appended with
     * conservative fixed-topology capacities so a later dispatch toggle does
     * not require an arena rebuild; unused segments cost bytes only. */
    offset.sparse_system_errors = layout.append<std::uint32_t>(batch);
    offset.sparse_device_error = layout.append<std::uint32_t>(1);
    offset.sparse_coordination = layout.append<double>(atoms);
    offset.pairlist_pairs = layout.append<xtbloom::detail::Gfn2AtomPair>(sparse_pair_capacity);
    offset.pairlist_offsets = layout.append<std::int64_t>(batch + 1);
    offset.pairlist_pair_counts = layout.append<std::int64_t>(batch);
    offset.pairlist_neighbor_offsets = layout.append<std::int64_t>(atoms + 1);
    offset.pairlist_neighbor_counts = layout.append<std::int64_t>(atoms);
    offset.pairlist_neighbors = layout.append<std::int64_t>(sparse_neighbor_capacity);
    offset.pairlist_generations = layout.append<std::uint64_t>(batch);
    offset.pairlist_system_modes = layout.append<std::int32_t>(batch);
    offset.pairlist_meta = layout.append<xtbloom::detail::cuda::Gfn2PairListSystemMeta>(batch);
    offset.pairlist_atom_cells = layout.append<std::int64_t>(atoms);
    const std::int64_t sparse_cell_storage = sparse_cell_capacity + batch;
    offset.pairlist_cell_counts = layout.append<std::int64_t>(sparse_cell_storage);
    offset.pairlist_cell_offsets = layout.append<std::int64_t>(sparse_cell_storage);
    offset.pairlist_cell_fill = layout.append<std::int64_t>(sparse_cell_storage);
    offset.pairlist_cell_atoms = layout.append<std::int64_t>(atoms);
    offset.pairlist_neighbor_cursor = layout.append<std::int64_t>(atoms);
    offset.pairlist_neighbor_scratch = layout.append<std::int64_t>(sparse_neighbor_capacity);
    offset.pairlist_pair_cursor = layout.append<std::int64_t>(batch);
    offset.pairlist_sequence = layout.append<std::uint32_t>(1);

    /* Committed output pair-list storage: distinct fixed-capacity slices so a
     * failed peer's slice is never exposed and later peers never shift. */
    offset.committed_pairs = layout.append<xtbloom::detail::Gfn2AtomPair>(sparse_pair_capacity);
    offset.committed_pair_offsets = layout.append<std::int64_t>(batch + 1);
    offset.committed_pair_counts = layout.append<std::int64_t>(batch);
    offset.committed_neighbor_offsets = layout.append<std::int64_t>(atoms + 1);
    offset.committed_neighbor_counts = layout.append<std::int64_t>(atoms);
    offset.committed_neighbors = layout.append<std::int64_t>(sparse_neighbor_capacity);
    offset.committed_pair_generations = layout.append<std::uint64_t>(batch);
    offset.committed_eligible_mask = layout.append<std::uint8_t>(batch);

    if (candidate.host.d4_enabled) {
      offset.d4_coordination_scratch = layout.append<double>(atoms);
      offset.d4_system_errors = layout.append<std::uint32_t>(batch);
      offset.d4_device_error = layout.append<std::uint32_t>(1);
    }
    if (points != 0) {
      offset.point_candidate_shell = layout.append<double>(shells);
      offset.point_shell_scratch = layout.append<double>(shells);
      offset.point_system_errors = layout.append<std::uint32_t>(batch);
      offset.point_plan_error = layout.append<std::uint32_t>(1);
      offset.point_sequence = layout.append<std::uint32_t>(1);
    }
    if (candidate.host.periodic_enabled) {
      offset.periodic_system_errors = layout.append<std::uint32_t>(batch);
      offset.periodic_plan_error = layout.append<std::uint32_t>(1);
    }
    if (!layout.valid()) {
      error = "numerical refresh arena layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }

    ArenaLayout host_layout;
    host_offset.positions = host_layout.append<double>(coordinates);
    host_offset.point_positions = host_layout.append<double>(point_coordinates);
    host_offset.point_values = host_layout.append<double>(points);
    host_offset.point_gammas = host_layout.append<double>(points);
    host_offset.periodic_shifts =
        host_layout.append<double>(candidate.host.periodic_enabled ? atoms : 0);
    host_offset.periodic_response =
        host_layout.append<double>(candidate.host.periodic_enabled ? response : 0);
    host_offset.requested = host_layout.append<std::uint8_t>(batch);
    if (!host_layout.valid()) {
      error = "numerical host-staging arena layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }

    std::size_t interaction_descriptor_capacity = 0u;
    std::size_t interaction_payload_capacity = 0u;
    if (!checked_bytes(batch, sizeof(xtbloom_interaction_t), interaction_descriptor_capacity) ||
        !checked_bytes(batch, 32u, interaction_payload_capacity)) {
      error = "released interaction staging capacity overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    const InteractionStagingLayout interaction_layout = make_interaction_staging_layout(
        interaction_descriptor_capacity, interaction_payload_capacity);
    if (!interaction_layout.valid) {
      error = "released interaction staging layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }

    cudaError_t cuda_status = candidate.numerical_refresh_arena.allocate(layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_staging_arena.allocate(host_layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical pinned host-staging allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status =
        candidate.interaction_device_staging_arena.allocate(interaction_layout.arena_bytes);
    if (cuda_status == cudaSuccess) {
      cuda_status =
          candidate.interaction_host_staging_arena.allocate(interaction_layout.arena_bytes);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA released interaction staging allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_completion_stream.create(cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-completion stream creation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_upload_complete.create(cudaEventDisableTiming);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-upload event creation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_release_complete.create(cudaEventDisableTiming);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-release event creation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    void* const arena = candidate.numerical_refresh_arena.get();
    cuda_status = cudaMemsetAsync(arena, 0, candidate.numerical_refresh_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh arena initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = true;

    const auto upload = [&](std::size_t destination, const void* source, std::size_t bytes) {
      if (cuda_status == cudaSuccess && bytes != 0u) {
        cuda_status = cudaMemcpyAsync(static_cast<std::byte*>(arena) + destination, source, bytes,
                                      cudaMemcpyHostToDevice, stream);
      }
    };
    const auto upload_vector = [&](std::size_t destination, const auto& values) {
      using Value = typename std::decay_t<decltype(values)>::value_type;
      upload(destination, values.data(), values.size() * sizeof(Value));
    };
    upload_vector(offset.shell_pair_offsets, candidate.host.h0.shell_pair_offsets);
    upload_vector(offset.shell_primitive_offsets, candidate.host.basis.shell_primitive_offsets);
    upload_vector(offset.angular_momenta, candidate.host.basis.angular_momenta);
    upload_vector(offset.primitive_exponents, candidate.host.basis.primitive_exponents);
    upload_vector(offset.primitive_coefficients, candidate.host.basis.primitive_coefficients);
    upload_vector(offset.h0_atomic_radii, candidate.host.h0.atomic_radii);
    upload_vector(offset.h0_shell_levels, candidate.host.h0.shell_levels);
    upload_vector(offset.h0_coordination_scale, candidate.host.h0.shell_coordination_scale);
    upload_vector(offset.h0_polynomial, candidate.host.h0.shell_polynomial);
    upload_vector(offset.h0_pair_scale, candidate.host.h0.shell_pair_scale);
    upload_vector(offset.pairlist_system_modes, pairlist_system_modes);
    upload_vector(offset.committed_positions, candidate.host.positions);
    upload_vector(offset.committed_point_positions, candidate.host.point_positions);
    upload_vector(offset.committed_point_values, candidate.host.point_values);
    upload_vector(offset.committed_point_gammas, candidate.host.point_gammas);
    upload_vector(offset.committed_periodic_shifts, candidate.host.periodic_shifts);
    upload_vector(offset.committed_periodic_response, candidate.host.periodic_response);
    /* Refresh-owned committed caches start unpublished. prepare_host performs
     * the first real numerical transaction from the caller's geometry, while
     * topology-only plans defer that transaction to their first compute. This
     * prevents epoch 1 from advertising zero-initialized pair-list/D4 leaves. */
    std::vector<std::uint64_t> initial_generations(static_cast<std::size_t>(batch), 0u);
    upload_vector(offset.committed_generations, initial_generations);
    constexpr std::uint64_t kUnpublishedGeometryEpoch = 0u;
    upload(offset.geometry_epoch, &kUnpublishedGeometryEpoch, sizeof(kUnpublishedGeometryEpoch));
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(arena_pointer<std::uint8_t>(arena, offset.requested), 1,
                                    static_cast<std::size_t>(batch), stream);
    }
    if (cuda_status == cudaSuccess) {
      /* The dedicated eligibility vector begins unpublished and is set only
       * by the final refresh gate. It then survives SCC activity derivation so
       * terminal publication can distinguish a refresh-rejected peer after the
       * SCC ledger becomes inactive through convergence or failure. */
      cuda_status = cudaMemsetAsync(arena_pointer<std::uint8_t>(arena, offset.eligible), 0,
                                    static_cast<std::size_t>(batch), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(
          arena_pointer<std::uint64_t>(arena, offset.refresh_predecessor_generations), 0,
          static_cast<std::size_t>(batch) * sizeof(std::uint64_t), stream);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh setup upload", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    auto& numerical = candidate.numerical;
    numerical = {};
    numerical.host_positions = arena_pointer<double>(arena, offset.host_positions);
    numerical.host_point_positions =
        arena_pointer_if<double>(arena, offset.host_point_positions, point_coordinates);
    numerical.host_point_values = arena_pointer_if<double>(arena, offset.host_point_values, points);
    numerical.host_point_gammas = arena_pointer_if<double>(arena, offset.host_point_gammas, points);
    numerical.host_periodic_shifts = arena_pointer_if<double>(
        arena, offset.host_periodic_shifts, candidate.host.periodic_enabled ? atoms : 0);
    numerical.host_periodic_response = arena_pointer_if<double>(
        arena, offset.host_periodic_response, candidate.host.periodic_enabled ? response : 0);
    numerical.host_requested = arena_pointer<std::uint8_t>(arena, offset.host_requested);
    void* const host_arena = candidate.numerical_host_staging_arena.get();
    numerical.owned_host_positions = arena_pointer<double>(host_arena, host_offset.positions);
    numerical.owned_host_point_positions =
        arena_pointer_if<double>(host_arena, host_offset.point_positions, point_coordinates);
    numerical.owned_host_point_values =
        arena_pointer_if<double>(host_arena, host_offset.point_values, points);
    numerical.owned_host_point_gammas =
        arena_pointer_if<double>(host_arena, host_offset.point_gammas, points);
    numerical.owned_host_periodic_shifts = arena_pointer_if<double>(
        host_arena, host_offset.periodic_shifts, candidate.host.periodic_enabled ? atoms : 0);
    numerical.owned_host_periodic_response = arena_pointer_if<double>(
        host_arena, host_offset.periodic_response, candidate.host.periodic_enabled ? response : 0);
    numerical.owned_host_requested = arena_pointer<std::uint8_t>(host_arena, host_offset.requested);
    project_interaction_staging(interaction_layout,
                                candidate.interaction_device_staging_arena.get(),
                                candidate.interaction_host_staging_arena.get(), numerical);
    auto& binding = numerical.preprocessing;
    binding = {};
    binding.plan.abi_version = kGfn2PreprocessingAbiVersion;
    binding.plan.reserved = 0u;
    binding.plan.plan_token = token;
    binding.plan.geometry = candidate.plan_seed.geometry_batch;
    std::int64_t maximum_system_shells = 0;
    for (std::int64_t system = 0; system < batch; ++system) {
      maximum_system_shells =
          std::max(maximum_system_shells,
                   candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system + 1)] -
                       candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system)]);
    }
    binding.plan.integrals = {
        batch,
        atoms,
        shells,
        orbitals,
        primitives,
        matrices,
        candidate.host.h0.shell_pair_offsets.back(),
        maximum_system_shells,
        candidate.plan_seed.density_batch.contraction_tiles_per_channel,
        candidate.host.integrals.integral_cutoff,
        token,
        batch + 1,
        batch + 1,
        batch + 1,
        batch + 1,
        static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()),
        atoms + 1,
        shells + 1,
        static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()),
        shells,
        static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()),
        primitives,
        primitives,
        candidate.device_topology.atom_offsets,
        candidate.device_topology.batch_shell_offsets,
        candidate.device_topology.batch_orbital_offsets,
        candidate.device_topology.matrix_offsets,
        arena_pointer<std::int64_t>(arena, offset.shell_pair_offsets),
        candidate.device_topology.atom_shell_offsets,
        candidate.device_topology.shell_orbital_offsets,
        arena_pointer<std::int64_t>(arena, offset.shell_primitive_offsets),
        candidate.device_topology.shell_to_atom,
        arena_pointer<std::uint8_t>(arena, offset.angular_momenta),
        arena_pointer<double>(arena, offset.primitive_exponents),
        arena_pointer<double>(arena, offset.primitive_coefficients)};
    binding.plan.integrals.model =
        candidate.host.gfn1_enabled ? XtbModelFlavor::kGfn1 : XtbModelFlavor::kGfn2;
    bind_integral_task_domains(candidate, binding.plan.integrals);
    binding.plan.h0 = {atoms,
                       shells,
                       shells,
                       shells,
                       candidate.host.h0.shell_pair_offsets.back(),
                       token,
                       arena_pointer<double>(arena, offset.h0_atomic_radii),
                       arena_pointer<double>(arena, offset.h0_shell_levels),
                       arena_pointer<double>(arena, offset.h0_coordination_scale),
                       arena_pointer<double>(arena, offset.h0_polynomial),
                       arena_pointer<double>(arena, offset.h0_pair_scale)};
    binding.plan.es2 = candidate.plan_seed.es2_batch;
    binding.plan.aes2 = aes2_enabled ? candidate.plan_seed.aes2_batch : Gfn2AES2DeviceBatch{};
    binding.input = {arena_pointer<double>(arena, offset.candidate_positions), coordinates, token};
    binding.activity = {arena_pointer<std::uint8_t>(arena, offset.requested), batch,
                        arena_pointer<std::uint8_t>(arena, offset.output_published), batch, token};
    binding.output.geometry = {
        arena_pointer_if<double>(arena, offset.output_geometry_pairs, geometry_pair_elements),
        geometry_pair_elements,
        arena_pointer<double>(arena, offset.output_coordination),
        atoms,
        arena_pointer<std::uint64_t>(arena, offset.output_geometry_generations),
        batch,
        token};
    binding.output.overlap = arena_pointer<double>(arena, offset.output_overlap);
    binding.output.overlap_elements = matrices;
    binding.output.dipole_integrals =
        arena_pointer_if<double>(arena, offset.output_dipole, dipole_elements);
    binding.output.dipole_elements = dipole_elements;
    binding.output.quadrupole_integrals =
        arena_pointer_if<double>(arena, offset.output_quadrupole, quadrupole_elements);
    binding.output.quadrupole_elements = quadrupole_elements;
    binding.output.h0 = arena_pointer<double>(arena, offset.output_h0);
    binding.output.h0_elements = matrices;
    binding.output.es2 = {arena_pointer<double>(arena, offset.output_es2), es2_elements, 0u, token};
    binding.output.aes2 = aes2_enabled
                              ? Gfn2AES2DeviceCache{arena_pointer_if<double>(
                                                        arena, offset.output_aes2, aes2_elements),
                                                    aes2_elements, 0u, token}
                              : Gfn2AES2DeviceCache{};
    binding.output.operator_generations =
        arena_pointer<std::uint64_t>(arena, offset.output_operator_generations);
    binding.output.generation_elements = batch;
    binding.output.plan_token = token;
    const bool pairlist_enabled =
        !candidate.host.gfn1_enabled &&
        (candidate.host.d4_enabled ||
         xtbloom::detail::cuda::gfn2_pairlist_use_sparse_for(maximum_system_atoms));
    const double pairlist_builder_cutoff = candidate.host.d4_enabled
                                               ? kD4PairlistBuilderCutoffBohr
                                               : xtbloom::detail::cuda::kDefaultPairlistCutoffBohr;
    if (pairlist_enabled) {
      /* Capacities were provisioned once from fixed topology above. D4 makes
       * the leaf mandatory and widens only the physical builder superset to
       * 50 bohr. This source GFN2 coordination view remains canonical at 25
       * bohr; the D4 binding later projects its 30/50/25-bohr role views. */
      binding.plan.pairlist = {
          static_cast<std::int64_t>(batch),
          static_cast<std::int64_t>(atoms),
          static_cast<std::int64_t>(batch + 1),
          pairlist_builder_cutoff,
          sparse_cells_per_system,
          sparse_neighbors_per_atom,
          sparse_pairs_per_system,
          xtbloom::detail::cuda::Gfn2PairListMode::kSparse,
          token,
          candidate.device_topology.atom_offsets,
          xtbloom::detail::cuda::kGfn2PairListAllowDenseFallback,
          arena_pointer<std::int32_t>(arena, offset.pairlist_system_modes),
          batch,
      };
    }
    binding.diagnostics = {
        arena_pointer<std::uint32_t>(arena, offset.geometry_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.geometry_device_error),
        arena_pointer<std::uint32_t>(arena, offset.integral_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.integral_device_error),
        arena_pointer<std::uint32_t>(arena, offset.es2_device_error),
        aes2_enabled ? arena_pointer<std::uint32_t>(arena, offset.aes2_system_errors) : nullptr,
        aes2_enabled ? batch : 0,
        aes2_enabled ? arena_pointer<std::uint32_t>(arena, offset.aes2_device_error) : nullptr,
        arena_pointer<std::uint32_t>(arena, offset.preprocessing_stages),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.preprocessing_plan_error),
        pairlist_enabled ? arena_pointer<std::uint32_t>(arena, offset.sparse_system_errors)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::uint32_t>(arena, offset.sparse_device_error)
                         : nullptr,
        token};
    binding.workspace.positions_scratch = arena_pointer<double>(arena, offset.positions_scratch);
    binding.workspace.position_elements = coordinates;
    binding.workspace.geometry_candidate = {
        arena_pointer_if<double>(arena, offset.geometry_candidate_pairs, geometry_pair_elements),
        geometry_pair_elements,
        arena_pointer<double>(arena, offset.geometry_candidate_coordination),
        atoms,
        arena_pointer<std::uint64_t>(arena, offset.geometry_candidate_generations),
        batch,
        token};
    binding.workspace.geometry = {
        arena_pointer_if<double>(arena, offset.geometry_pair_scratch, geometry_pair_elements),
        geometry_pair_elements,
        arena_pointer<double>(arena, offset.geometry_coordination_scratch),
        atoms,
        nullptr,
        0,
        arena_pointer<std::uint32_t>(arena, offset.geometry_sequence),
        1,
        token};
    binding.workspace.sparse_coordination =
        pairlist_enabled ? arena_pointer<double>(arena, offset.sparse_coordination) : nullptr;
    binding.workspace.sparse_coordination_elements = pairlist_enabled ? atoms : 0;
    binding.workspace.pairlist_candidate = {
        pairlist_enabled
            ? arena_pointer<xtbloom::detail::Gfn2AtomPair>(arena, offset.pairlist_pairs)
            : nullptr,
        pairlist_enabled ? sparse_pair_capacity : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_offsets) : nullptr,
        pairlist_enabled ? batch + 1 : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_pair_counts)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_offsets)
                         : nullptr,
        pairlist_enabled ? atoms + 1 : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_counts)
                         : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbors) : nullptr,
        pairlist_enabled ? sparse_neighbor_capacity : 0,
        pairlist_enabled ? arena_pointer<std::uint64_t>(arena, offset.pairlist_generations)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? token : 0u};
    binding.workspace.pairlist = {
        pairlist_enabled ? arena_pointer<xtbloom::detail::cuda::Gfn2PairListSystemMeta>(
                               arena, offset.pairlist_meta)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_atom_cells) : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_counts)
                         : nullptr,
        pairlist_enabled ? sparse_cell_capacity + batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_offsets)
                         : nullptr,
        pairlist_enabled ? sparse_cell_capacity + batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_fill) : nullptr,
        pairlist_enabled ? sparse_cell_capacity + batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_atoms) : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_cursor)
                         : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_scratch)
                         : nullptr,
        pairlist_enabled ? sparse_neighbor_capacity : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_pair_cursor)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::uint32_t>(arena, offset.pairlist_sequence) : nullptr,
        pairlist_enabled ? 1 : 0,
        pairlist_enabled ? token : 0u};
    /* Committed pair-list consumer view published through the final per-system
     * gate (preprocessing step 4).  Uses dedicated output storage so a failed
     * peer never exposes a partial candidate slice; only eligible peers with a
     * current committed generation are consumable.  A disabled leaf keeps the
     * canonical empty viewed with a zero token. */
    binding.output.pairlist =
        pairlist_enabled
            ? xtbloom::detail::
                  Gfn2PairListConsumerView{xtbloom::detail::Gfn2PlanMemorySpace::kCudaDevice,
                                           xtbloom::detail::Gfn2PairListState::kCommitted,
                                           xtbloom::detail::Gfn2PairListRole::kCoordination,
                                           xtbloom::detail::Gfn2PairMapKind::kExplicit,
                                           token,
                                           xtbloom::detail::cuda::kDefaultPairlistCutoffBohr,
                                           pairlist_builder_cutoff,
                                           static_cast<std::int64_t>(batch),
                                           static_cast<std::int64_t>(atoms),
                                           sparse_pairs_per_system,
                                           sparse_neighbors_per_atom,
                                           batch + 1,
                                           atoms + 1,
                                           sparse_pair_capacity,
                                           sparse_neighbor_capacity,
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_pair_offsets),
                                           arena_pointer<xtbloom::detail::Gfn2AtomPair>(
                                               arena, offset.committed_pairs),
                                           static_cast<std::int64_t>(batch),
                                           static_cast<std::int64_t>(atoms),
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_pair_counts),
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_neighbor_counts),
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_neighbor_offsets),
                                           arena_pointer<std::int64_t>(arena,
                                                                       offset.committed_neighbors),
                                           static_cast<std::int64_t>(batch),
                                           static_cast<std::int64_t>(batch),
                                           0,
                                           arena_pointer<std::uint64_t>(
                                               arena, offset.committed_pair_generations),
                                           arena_pointer<std::uint8_t>(
                                               arena, offset.committed_eligible_mask),
                                           nullptr}
            : xtbloom::detail::Gfn2PairListConsumerView{};
    binding.workspace.overlap_candidate = arena_pointer<double>(arena, offset.overlap_candidate);
    binding.workspace.overlap_elements = matrices;
    binding.workspace.dipole_candidate =
        arena_pointer_if<double>(arena, offset.dipole_candidate, dipole_elements);
    binding.workspace.dipole_elements = dipole_elements;
    binding.workspace.quadrupole_candidate =
        arena_pointer_if<double>(arena, offset.quadrupole_candidate, quadrupole_elements);
    binding.workspace.quadrupole_elements = quadrupole_elements;
    binding.workspace.h0_candidate = arena_pointer<double>(arena, offset.h0_candidate);
    binding.workspace.h0_elements = matrices;
    binding.workspace.integrals = {
        arena_pointer<double>(arena, offset.overlap_scratch),
        matrices,
        arena_pointer_if<double>(arena, offset.dipole_scratch, dipole_elements),
        dipole_elements,
        arena_pointer_if<double>(arena, offset.quadrupole_scratch, quadrupole_elements),
        quadrupole_elements,
        arena_pointer<double>(arena, offset.h0_scratch),
        matrices,
        arena_pointer<std::uint32_t>(arena, offset.integral_sequence),
        1,
        token};
    binding.workspace.es2_candidate = {arena_pointer<double>(arena, offset.es2_candidate),
                                       es2_elements, 0u, token};
    binding.workspace.es2 = {arena_pointer<double>(arena, offset.es2_scratch),
                             es2_elements,
                             nullptr,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             0};
    binding.workspace.aes2_candidate =
        aes2_enabled ? Gfn2AES2DeviceCache{arena_pointer_if<double>(arena, offset.aes2_candidate,
                                                                    aes2_elements),
                                           aes2_elements, 0u, token}
                     : Gfn2AES2DeviceCache{};
    binding.workspace.aes2 =
        aes2_enabled ? Gfn2AES2DeviceWorkspace{arena_pointer_if<double>(arena, offset.aes2_scratch,
                                                                        aes2_elements),
                                               aes2_elements,
                                               nullptr,
                                               0,
                                               nullptr,
                                               0,
                                               nullptr,
                                               0,
                                               nullptr,
                                               0,
                                               nullptr,
                                               0}
                     : Gfn2AES2DeviceWorkspace{};
    binding.workspace.plan_token = token;
    binding.geometry_epoch = {arena_pointer<std::uint64_t>(arena, offset.geometry_epoch), 1, token};
    binding.plan_token = token;

    const auto seal = seal_gfn2_preprocessing_binding_cuda(binding);
    if (!seal.success()) {
      error = "CUDA runtime preprocessing binding rejected its fixed arena projection (error=" +
              std::to_string(static_cast<std::uint32_t>(seal.error)) +
              ", field=" + std::to_string(static_cast<std::uint32_t>(seal.field)) +
              ", index=" + std::to_string(seal.index) + ")";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    auto& device = numerical.device;
    device = {};
    device.batch_size = batch;
    device.total_atoms = atoms;
    device.total_shells = shells;
    device.total_matrices = matrices;
    device.total_point_charges = points;
    device.total_response_elements = response;
    device.geometry_pair_elements = geometry_pair_elements;
    device.es2_elements = es2_elements;
    device.aes2_elements = aes2_elements;
    device.multipoles_enabled = multipoles_enabled ? 1u : 0u;
    device.aes2_enabled = aes2_enabled ? 1u : 0u;
    device.d4_enabled = candidate.host.d4_enabled ? 1u : 0u;
    device.point_enabled = points != 0 ? 1u : 0u;
    device.periodic_enabled = candidate.host.periodic_enabled ? 1u : 0u;
    device.plan_token = token;
    device.atom_offsets = candidate.device_topology.atom_offsets;
    device.shell_offsets = candidate.device_topology.batch_shell_offsets;
    device.matrix_offsets = candidate.device_topology.matrix_offsets;
    device.geometry_pair_offsets = candidate.plan_seed.geometry_batch.pair_offsets;
    device.es2_offsets = candidate.plan_seed.es2_batch.matrix_offsets;
    device.point_offsets = candidate.plan_seed.explicit_point_charge_batch.point_charge_offsets;
    device.response_offsets = candidate.plan_seed.periodic_batch.matrix_offsets;
    device.requested = binding.activity.requested_mask;
    device.preprocessing_published = binding.activity.published_mask;
    device.eligible = arena_pointer<std::uint8_t>(arena, offset.eligible);
    device.factor_active = candidate.workspace_seed.ledger.active_mask;
    device.committed_generations =
        arena_pointer<std::uint64_t>(arena, offset.committed_generations);
    device.refresh_predecessor_generations =
        arena_pointer<std::uint64_t>(arena, offset.refresh_predecessor_generations);
    device.candidate_positions = arena_pointer<double>(arena, offset.candidate_positions);
    device.candidate_point_positions =
        arena_pointer_if<double>(arena, offset.candidate_point_positions, point_coordinates);
    device.candidate_point_values =
        arena_pointer_if<double>(arena, offset.candidate_point_values, points);
    device.candidate_point_gammas =
        arena_pointer_if<double>(arena, offset.candidate_point_gammas, points);
    device.candidate_periodic_shifts = arena_pointer_if<double>(
        arena, offset.candidate_periodic_shifts, candidate.host.periodic_enabled ? atoms : 0);
    device.candidate_periodic_response = arena_pointer_if<double>(
        arena, offset.candidate_periodic_response, candidate.host.periodic_enabled ? response : 0);
    device.committed_positions = arena_pointer<double>(arena, offset.committed_positions);
    device.committed_point_positions =
        arena_pointer_if<double>(arena, offset.committed_point_positions, point_coordinates);
    device.committed_point_values =
        arena_pointer_if<double>(arena, offset.committed_point_values, points);
    device.committed_point_gammas =
        arena_pointer_if<double>(arena, offset.committed_point_gammas, points);
    device.committed_periodic_shifts = arena_pointer_if<double>(
        arena, offset.committed_periodic_shifts, candidate.host.periodic_enabled ? atoms : 0);
    device.committed_periodic_response = arena_pointer_if<double>(
        arena, offset.committed_periodic_response, candidate.host.periodic_enabled ? response : 0);
    device.committed_field_attached =
        arena_pointer<std::uint8_t>(arena, offset.committed_field_attached);
    device.committed_field_vectors = arena_pointer<double>(arena, offset.committed_field_vectors);
    device.candidate_field_attached =
        arena_pointer<std::uint8_t>(arena, offset.candidate_field_attached);
    device.candidate_field_vectors = arena_pointer<double>(arena, offset.candidate_field_vectors);
    device.candidate_field_atomic_potentials =
        arena_pointer<double>(arena, offset.candidate_field_atomic_potentials);
    device.candidate_field_dipole_potentials =
        arena_pointer<double>(arena, offset.candidate_field_dipole_potentials);
    device.committed_field_atomic_potentials =
        arena_pointer<double>(arena, offset.committed_field_atomic_potentials);
    device.committed_field_dipole_potentials =
        arena_pointer<double>(arena, offset.committed_field_dipole_potentials);
    device.interaction_request_error =
        arena_pointer<std::uint32_t>(arena, offset.interaction_request_error);
    device.field_system_errors = arena_pointer<std::uint32_t>(arena, offset.field_system_errors);
    device.field_plan_error = arena_pointer<std::uint32_t>(arena, offset.field_plan_error);
    device.candidate_geometry_pairs = binding.output.geometry.pair_data;
    device.candidate_coordination = binding.output.geometry.coordination_numbers;
    device.candidate_overlap = binding.output.overlap;
    device.candidate_dipole = binding.output.dipole_integrals;
    device.candidate_quadrupole = binding.output.quadrupole_integrals;
    device.candidate_h0 = binding.output.h0;
    device.candidate_es2 = binding.output.es2.coulomb_matrix;
    device.candidate_aes2 = binding.output.aes2.pair_data;
    device.public_geometry_pairs =
        const_cast<double*>(candidate.plan_seed.geometry_cache.pair_data);
    device.public_coordination =
        const_cast<double*>(candidate.plan_seed.geometry_cache.coordination_numbers);
    device.public_overlap = const_cast<double*>(candidate.input_seed.hamiltonian.overlap);
    device.public_dipole =
        multipoles_enabled ? const_cast<double*>(candidate.input_seed.hamiltonian.dipole_integrals)
                           : nullptr;
    device.public_quadrupole =
        multipoles_enabled
            ? const_cast<double*>(candidate.input_seed.hamiltonian.quadrupole_integrals)
            : nullptr;
    device.public_h0 = const_cast<double*>(candidate.input_seed.hamiltonian.h0);
    device.public_es2 = candidate.plan_seed.es2_cache.coulomb_matrix;
    device.public_aes2 = aes2_enabled ? candidate.plan_seed.aes2_cache.pair_data : nullptr;
    device.preprocessing_plan_error = binding.diagnostics.plan_error;
    device.geometry_epoch = binding.geometry_epoch.value;

    if (candidate.host.d4_enabled) {
      numerical.d4_workspace.coordination_scratch =
          arena_pointer<double>(arena, offset.d4_coordination_scratch);
      numerical.d4_workspace.coordination_scratch_elements = atoms;
      numerical.d4_workspace.system_errors =
          arena_pointer<std::uint32_t>(arena, offset.d4_system_errors);
      numerical.d4_workspace.system_error_elements = batch;
      device.candidate_d4_coordination = numerical.d4_workspace.coordination_scratch;
      device.public_d4_coordination = candidate.plan_seed.d4_pairlist_cache.coordination_numbers;
      device.d4_system_errors = numerical.d4_workspace.system_errors;
      device.d4_device_error = arena_pointer<std::uint32_t>(arena, offset.d4_device_error);

      Gfn2PairListConsumerView d4_coordination_pairs{};
      Gfn2PairListConsumerView d4_two_body_pairs{};
      Gfn2PairListConsumerView d4_atm_pairs{};
      const auto d4_coordination_projection = project_gfn2_pair_list_role_binding(
          candidate.device_topology, binding.output.pairlist, Gfn2PairListRole::kD4Coordination,
          Gfn2PlanMemorySpace::kCudaDevice, d4_coordination_pairs);
      const auto d4_two_body_projection = project_gfn2_pair_list_role_binding(
          candidate.device_topology, binding.output.pairlist, Gfn2PairListRole::kD4TwoBody,
          Gfn2PlanMemorySpace::kCudaDevice, d4_two_body_pairs);
      const auto d4_atm_projection = project_gfn2_pair_list_role_binding(
          candidate.device_topology, binding.output.pairlist, Gfn2PairListRole::kD4Atm,
          Gfn2PlanMemorySpace::kCudaDevice, d4_atm_pairs);
      if (d4_coordination_projection.error != Gfn2PlanSchemaError::kSuccess ||
          d4_two_body_projection.error != Gfn2PlanSchemaError::kSuccess ||
          d4_atm_projection.error != Gfn2PlanSchemaError::kSuccess) {
        error = "CUDA runtime rejected a D4 role projection of the committed pair-list superset";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }

      /* The role views borrow one committed structural transaction. D4 CN is
       * built into scratch after that transaction commits, then the terminal
       * numerical gate alone copies it to the public outlet and advances the
       * common generation/eligibility authority. This keeps Graph replay on a
       * device epoch and prevents a D4-local failure from publishing stale CN. */
      numerical.d4_refresh_cache = {binding.workspace.positions_scratch,
                                    coordinates,
                                    device.public_d4_coordination,
                                    atoms,
                                    device.committed_generations,
                                    batch,
                                    device.eligible,
                                    batch,
                                    d4_coordination_pairs,
                                    d4_two_body_pairs,
                                    d4_atm_pairs,
                                    token};
      numerical.d4_cache = {device.committed_positions,
                            coordinates,
                            device.public_d4_coordination,
                            atoms,
                            device.committed_generations,
                            batch,
                            device.eligible,
                            batch,
                            d4_coordination_pairs,
                            d4_two_body_pairs,
                            d4_atm_pairs,
                            token};
      candidate.plan_seed.d4_pairlist_cache = numerical.d4_cache;
    }
    if (points != 0) {
      numerical.point_batch = candidate.plan_seed.explicit_point_charge_batch;
      numerical.point_batch.qm_positions = device.candidate_positions;
      numerical.point_batch.point_positions = device.candidate_point_positions;
      numerical.point_batch.point_charges = device.candidate_point_values;
      numerical.point_batch.point_hardnesses = device.candidate_point_gammas;
      numerical.point_candidate = {arena_pointer<double>(arena, offset.point_candidate_shell),
                                   shells, candidate.host.geometry_generation, token};
      numerical.point_workspace = {arena_pointer<double>(arena, offset.point_shell_scratch),
                                   shells};
      numerical.point_activity = {binding.activity.published_mask,
                                  arena_pointer<std::uint32_t>(arena, offset.point_sequence), batch,
                                  1, token};
      device.candidate_point_shell = numerical.point_candidate.shell_potentials;
      device.public_point_shell = candidate.plan_seed.explicit_point_charge_cache.shell_potentials;
      device.point_system_errors = arena_pointer<std::uint32_t>(arena, offset.point_system_errors);
      device.point_plan_error = arena_pointer<std::uint32_t>(arena, offset.point_plan_error);
    }
    if (candidate.host.periodic_enabled) {
      device.periodic_system_errors =
          arena_pointer<std::uint32_t>(arena, offset.periodic_system_errors);
      device.periodic_plan_error = arena_pointer<std::uint32_t>(arena, offset.periodic_plan_error);
    }
    numerical.field_batch = {batch, atoms, batch + 1, candidate.device_topology.atom_offsets,
                             token};
    numerical.field_candidate_potentials = {device.candidate_field_atomic_potentials, atoms,
                                            device.candidate_field_dipole_potentials, coordinates,
                                            token};
    numerical.field_committed_potentials = {device.committed_field_atomic_potentials, atoms,
                                            device.committed_field_dipole_potentials, coordinates,
                                            token};
    numerical.field_potential_view = {numerical.field_committed_potentials.atomic, atoms,
                                      numerical.field_committed_potentials.dipole, coordinates,
                                      token};

    /* Canonical runtime numerical pointers replace setup-only copies. */
    candidate.plan_seed.geometry_cache.geometry_generations = device.committed_generations;
    candidate.plan_seed.geometry_cache.generation_elements = batch;
    if (points != 0) {
      candidate.plan_seed.explicit_point_charge_batch.qm_positions = device.committed_positions;
      candidate.plan_seed.explicit_point_charge_batch.point_positions =
          device.committed_point_positions;
      candidate.plan_seed.explicit_point_charge_batch.point_charges = device.committed_point_values;
      candidate.plan_seed.explicit_point_charge_batch.point_hardnesses =
          device.committed_point_gammas;
    }
    if (candidate.host.periodic_enabled) {
      candidate.plan_seed.periodic_batch.shifts = device.committed_periodic_shifts;
      candidate.plan_seed.periodic_batch.response_matrices = device.committed_periodic_response;
    }

    std::vector<Gfn2SccCacheProvenanceBinding> provenance;
    provenance.reserve(
        static_cast<std::size_t>(candidate.plan_seed.provenance.cache_binding_count));
    const auto append_provenance = [&](Gfn2SccStageId stage) {
      Gfn2SccCacheProvenanceBinding record{};
      record.provenance = {Gfn2PlanMemorySpace::kCudaDevice,
                           Gfn2GenerationScope::kPerSystem,
                           token,
                           0u,
                           batch,
                           batch,
                           device.committed_generations};
      record.owner_stage = stage;
      provenance.push_back(record);
    };
    append_provenance(Gfn2SccStageId::kGeometry);
    append_provenance(Gfn2SccStageId::kES2Potential);
    if (aes2_enabled) append_provenance(Gfn2SccStageId::kAES2Potential);
    if (candidate.host.d4_enabled) append_provenance(Gfn2SccStageId::kD4Potential);
    if (points != 0) append_provenance(Gfn2SccStageId::kExplicitPointChargePotential);
    if (candidate.host.periodic_enabled) append_provenance(Gfn2SccStageId::kPeriodicPotential);
    if (static_cast<std::int64_t>(provenance.size()) !=
        candidate.plan_seed.provenance.cache_binding_count) {
      error = "runtime refresh provenance count disagrees with the setup owner";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaMemcpyAsync(
        const_cast<Gfn2SccCacheProvenanceBinding*>(candidate.plan_seed.provenance.cache_bindings),
        provenance.data(), provenance.size() * sizeof(provenance.front()), cudaMemcpyHostToDevice,
        stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA runtime provenance publication", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    numerical.ready = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t build_energy_force_bindings(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t shells = candidate.host.basis.total_shells;
    const std::int64_t orbitals = candidate.host.basis.total_orbitals;
    const std::int64_t primitives = candidate.host.basis.total_primitives;
    const std::int64_t matrices = candidate.host.integrals.total_matrix_elements;
    const std::int64_t points = candidate.host.external.total_point_charges;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t dipole_matrix_elements = 0;
    std::int64_t quadrupole_matrix_elements = 0;
    std::int64_t quadrupole_elements = 0;
    std::int64_t d4_weight_elements = 0;
    std::int64_t gfn1_weight_elements = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(matrices, 3, dipole_matrix_elements) ||
        !checked_elements(matrices, 6, quadrupole_matrix_elements) ||
        !checked_elements(atoms, 6, quadrupole_elements) ||
        !checked_elements(atoms, kGfn2D4MaximumReferences, d4_weight_elements) ||
        !checked_elements(atoms, kGfn1D3MaximumReferences, gfn1_weight_elements)) {
      error = "force descriptor element count overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    const bool force_mode = (candidate.host.key.flags &
                             (static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES) |
                              static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) |
                              static_cast<std::uint32_t>(XTBLOOM_COMPUTE_DIPOLE_MOMENTS))) != 0u;
    const bool explicit_points = points != 0;
    const bool d4_enabled = candidate.host.d4_enabled;
    const bool gfn1_enabled = candidate.host.gfn1_enabled;
    const bool periodic_enabled = candidate.host.periodic_enabled;
    const std::int64_t atomic_dipole_elements = gfn1_enabled ? 0 : coordinates;
    const std::int64_t atomic_quadrupole_elements = gfn1_enabled ? 0 : quadrupole_elements;
    const std::int64_t force_dipole_matrix_elements = gfn1_enabled ? 0 : dipole_matrix_elements;
    const std::int64_t force_quadrupole_matrix_elements =
        gfn1_enabled ? 0 : quadrupole_matrix_elements;
    const std::uint64_t token = candidate.host.plan_token;

    struct ImmutableOffsets {
      std::size_t atomic_numbers = 0u;
      std::size_t positions = 0u;
      std::size_t point_offsets = 0u;
      std::size_t shell_pair_offsets = 0u;
      std::size_t shell_primitive_offsets = 0u;
      std::size_t angular_momenta = 0u;
      std::size_t primitive_exponents = 0u;
      std::size_t primitive_coefficients = 0u;
      std::size_t h0_atomic_radii = 0u;
      std::size_t h0_shell_levels = 0u;
      std::size_t h0_coordination_scale = 0u;
      std::size_t h0_polynomial = 0u;
      std::size_t h0_pair_scale = 0u;
      std::size_t gfn1_repulsion_sqrt_alpha = 0u;
      std::size_t gfn1_repulsion_effective_charge = 0u;
      std::size_t gfn1_reference_counts = 0u;
      std::size_t gfn1_reference_cn = 0u;
      std::size_t gfn1_reference_c6 = 0u;
      std::size_t gfn1_pair_rrij = 0u;
      std::size_t gfn1_pair_damping_radii = 0u;
      std::size_t gfn1_halogen_scaled_radii = 0u;
      std::size_t gfn1_halogen_bond_strength = 0u;
      std::size_t gfn1_halogen_donor = 0u;
      std::size_t gfn1_halogen_acceptor = 0u;
    } immutable;

    ArenaLayout immutable_layout;
    if (force_mode) {
      immutable.atomic_numbers = immutable_layout.append<std::int32_t>(atoms);
      immutable.positions = immutable_layout.append<double>(explicit_points ? 0 : coordinates);
      /* A zero-point plan still needs a valid all-zero ragged offset leaf for
       * force composition. Real point-charge plans reuse the SCC input owner. */
      immutable.point_offsets =
          immutable_layout.append<std::int64_t>(explicit_points ? 0 : batch + 1);
      immutable.shell_pair_offsets = immutable_layout.append<std::int64_t>(
          static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()));
      immutable.shell_primitive_offsets = immutable_layout.append<std::int64_t>(
          static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()));
      immutable.angular_momenta = immutable_layout.append<std::uint8_t>(
          static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()));
      immutable.primitive_exponents = immutable_layout.append<double>(primitives);
      immutable.primitive_coefficients = immutable_layout.append<double>(primitives);
      immutable.h0_atomic_radii = immutable_layout.append<double>(atoms);
      immutable.h0_shell_levels = immutable_layout.append<double>(shells);
      immutable.h0_coordination_scale = immutable_layout.append<double>(shells);
      immutable.h0_polynomial = immutable_layout.append<double>(shells);
      immutable.h0_pair_scale =
          immutable_layout.append<double>(candidate.host.h0.shell_pair_offsets.back());
    }
    if (gfn1_enabled) {
      immutable.gfn1_repulsion_sqrt_alpha = immutable_layout.append<double>(atoms);
      immutable.gfn1_repulsion_effective_charge = immutable_layout.append<double>(atoms);
      immutable.gfn1_reference_counts = immutable_layout.append<std::uint8_t>(atoms);
      immutable.gfn1_reference_cn = immutable_layout.append<double>(gfn1_weight_elements);
      immutable.gfn1_reference_c6 = immutable_layout.append<double>(
          static_cast<std::int64_t>(candidate.host.gfn1_d3_reference_c6.size()));
      immutable.gfn1_pair_rrij =
          immutable_layout.append<double>(candidate.host.gfn1_d3.total_pairs());
      immutable.gfn1_pair_damping_radii =
          immutable_layout.append<double>(candidate.host.gfn1_d3.total_pairs());
      immutable.gfn1_halogen_scaled_radii = immutable_layout.append<double>(atoms);
      immutable.gfn1_halogen_bond_strength = immutable_layout.append<double>(atoms);
      immutable.gfn1_halogen_donor = immutable_layout.append<std::uint8_t>(atoms);
      immutable.gfn1_halogen_acceptor = immutable_layout.append<std::uint8_t>(atoms);
    }
    if (!immutable_layout.valid()) {
      error = "force immutable arena layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.force_immutable_arena.allocate(immutable_layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force immutable arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    const auto upload = [&](std::size_t offset, const void* source, std::size_t bytes) {
      if (bytes == 0u) return cudaSuccess;
      return cudaMemcpyAsync(
          static_cast<std::byte*>(candidate.force_immutable_arena.get()) + offset, source, bytes,
          cudaMemcpyHostToDevice, stream);
    };
    if (force_mode) {
      const auto upload_immutable = [&](std::size_t offset, const auto& source) {
        if (cuda_status != cudaSuccess) return;
        cuda_status = upload(offset, source.data(), source.size() * sizeof(source[0]));
      };
      upload_immutable(immutable.atomic_numbers, candidate.host.key.atomic_numbers);
      if (!explicit_points) {
        upload_immutable(immutable.positions, candidate.host.positions);
        upload_immutable(immutable.point_offsets, candidate.host.external.point_charge_offsets);
      }
      upload_immutable(immutable.shell_pair_offsets, candidate.host.h0.shell_pair_offsets);
      upload_immutable(immutable.shell_primitive_offsets,
                       candidate.host.basis.shell_primitive_offsets);
      upload_immutable(immutable.angular_momenta, candidate.host.basis.angular_momenta);
      upload_immutable(immutable.primitive_exponents, candidate.host.basis.primitive_exponents);
      upload_immutable(immutable.primitive_coefficients,
                       candidate.host.basis.primitive_coefficients);
      upload_immutable(immutable.h0_atomic_radii, candidate.host.h0.atomic_radii);
      upload_immutable(immutable.h0_shell_levels, candidate.host.h0.shell_levels);
      upload_immutable(immutable.h0_coordination_scale, candidate.host.h0.shell_coordination_scale);
      upload_immutable(immutable.h0_polynomial, candidate.host.h0.shell_polynomial);
      upload_immutable(immutable.h0_pair_scale, candidate.host.h0.shell_pair_scale);
    }
    if (gfn1_enabled) {
      const auto upload_immutable = [&](std::size_t offset, const auto& source) {
        if (cuda_status != cudaSuccess) return;
        cuda_status = upload(offset, source.data(), source.size() * sizeof(source[0]));
      };
      upload_immutable(immutable.gfn1_repulsion_sqrt_alpha,
                       candidate.host.gfn1_repulsion.sqrt_alpha);
      upload_immutable(immutable.gfn1_repulsion_effective_charge,
                       candidate.host.gfn1_repulsion.effective_charge);
      upload_immutable(immutable.gfn1_reference_counts, candidate.host.gfn1_d3_reference_counts);
      upload_immutable(immutable.gfn1_reference_cn, candidate.host.gfn1_d3_reference_cn);
      upload_immutable(immutable.gfn1_reference_c6, candidate.host.gfn1_d3_reference_c6);
      upload_immutable(immutable.gfn1_pair_rrij, candidate.host.gfn1_d3_pair_rrij);
      upload_immutable(immutable.gfn1_pair_damping_radii,
                       candidate.host.gfn1_d3_pair_damping_radii);
      upload_immutable(immutable.gfn1_halogen_scaled_radii,
                       candidate.host.gfn1_halogen_scaled_radii);
      upload_immutable(immutable.gfn1_halogen_bond_strength,
                       candidate.host.gfn1_halogen_bond_strength);
      upload_immutable(immutable.gfn1_halogen_donor, candidate.host.gfn1_halogen_donor);
      upload_immutable(immutable.gfn1_halogen_acceptor, candidate.host.gfn1_halogen_acceptor);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force immutable upload", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = candidate.submitted || force_mode;

    enum SequenceIndex : std::int64_t {
      kTotalSequence,
      kPostSequence,
      kPostCompositionSequence,
      kPostBridgeSequence,
      kPostPeriodicSequence,
      kH0Sequence,
      kHamiltonianSequence,
      kIntegralSequence,
      kCoordinationSequence,
      kClassicalSequence,
      kClassicalPrimitiveSequence,
      kExternalSequence,
      kCompositionSequence,
      kSequenceCount,
    };
    enum MaskIndex : std::int64_t {
      kRequestedMask,
      kEnergySuccessMask,
      kPostSuccessMask,
      kElectronicSuccessMask,
      kCoordinationSuccessMask,
      kClassicalSuccessMask,
      kExternalSuccessMask,
      kH0SuccessMask,
      kHamiltonianSuccessMask,
      kClassicalSelectedMask,
      kPostActiveMask,
      kMaskCount,
    };
    enum SystemErrorIndex : std::int64_t {
      kExecutionSystemError,
      kTotalSystemError,
      kPostStageSystemError,
      kPostSystemError,
      kH0SystemError,
      kHamiltonianSystemError,
      kIntegralSystemError,
      kCoordinationSystemError,
      kClassicalSystemError,
      kExternalSystemError,
      kCompositionSystemError,
      kClassicalPrimitiveSystemError,
      kSystemErrorCount,
    };
    enum DeviceErrorIndex : std::int64_t {
      kExecutionDeviceError,
      kTotalDeviceError,
      kPostStageDeviceError,
      kPostDeviceError,
      kH0DeviceError,
      kHamiltonianDeviceError,
      kIntegralDeviceError,
      kCoordinationDeviceError,
      kClassicalDeviceError,
      kExternalDeviceError,
      kCompositionPlanError,
      kClassicalPrimitiveDeviceError,
      kPostAes2PeerError,
      kDeviceErrorCount,
    };

    struct ExecutionOffsets {
      std::size_t repulsion = 0u;
      std::size_t d4_atm = 0u;
      std::size_t staged_energy = 0u;
      std::size_t public_energy = 0u;
      std::size_t sequences = 0u;
      std::size_t plan_failure = 0u;
      std::size_t masks = 0u;
      std::size_t system_errors = 0u;
      std::size_t device_errors = 0u;

      std::size_t public_qm_force = 0u;
      std::size_t public_point_force = 0u;
      std::size_t staged_qm_force = 0u;
      std::size_t staged_point_force = 0u;
      std::size_t stationary_density = 0u;
      std::size_t stationary_weighted_density = 0u;
      std::size_t stationary_spin_density = 0u;
      std::size_t stationary_shell_charges = 0u;
      std::size_t stationary_atomic_charges = 0u;
      std::size_t stationary_atomic_dipoles = 0u;
      std::size_t stationary_atomic_quadrupoles = 0u;
      std::size_t stationary_spin_shell_potential = 0u;
      std::size_t post_complete_shell = 0u;
      std::size_t post_complete_atom = 0u;
      std::size_t post_complete_dipole = 0u;
      std::size_t post_complete_quadrupole = 0u;
      std::size_t post_shell_scalar = 0u;
      std::size_t post_es2_shell = 0u;
      std::size_t post_es3_shell = 0u;
      std::size_t post_aes2_atom = 0u;
      std::size_t post_aes2_dipole = 0u;
      std::size_t post_aes2_quadrupole = 0u;
      std::size_t post_d4_atom = 0u;
      std::size_t post_periodic_atom = 0u;
      std::size_t post_staged_shell = 0u;
      std::size_t post_staged_atom = 0u;
      std::size_t post_staged_dipole = 0u;
      std::size_t post_staged_quadrupole = 0u;
      std::size_t post_staged_shell_scalar = 0u;
      std::size_t overlap_adjoint = 0u;
      std::size_t coordination_adjoint = 0u;
      std::size_t dipole_adjoint = 0u;
      std::size_t quadrupole_adjoint = 0u;
      std::size_t electronic_gradient = 0u;
      std::size_t classical_force = 0u;
      std::size_t explicit_qm_force = 0u;
      std::size_t explicit_point_force = 0u;

      std::size_t post_es2_shell_scratch = 0u;
      std::size_t post_aes2_potential_scratch = 0u;
      std::size_t post_d4_weights = 0u;
      std::size_t post_d4_weight_cn = 0u;
      std::size_t post_d4_weight_charge = 0u;
      std::size_t post_d4_atom_scratch = 0u;
      std::size_t post_d4_coordination = 0u;
      std::size_t post_d4_batch = 0u;
      std::size_t post_d4_gradient = 0u;
      std::size_t post_periodic_potential = 0u;
      std::size_t post_periodic_raw_response = 0u;
      std::size_t post_composition_shell = 0u;
      std::size_t post_composition_atom = 0u;
      std::size_t post_composition_dipole = 0u;
      std::size_t post_composition_quadrupole = 0u;
      std::size_t post_bridge_shell = 0u;
      std::size_t h0_overlap_scratch = 0u;
      std::size_t h0_coordination_scratch = 0u;
      std::size_t h0_gradient_scratch = 0u;
      std::size_t hamiltonian_overlap_scratch = 0u;
      std::size_t hamiltonian_dipole_scratch = 0u;
      std::size_t hamiltonian_quadrupole_scratch = 0u;
      std::size_t integral_gradient_scratch = 0u;
      std::size_t coordination_gradient_scratch = 0u;
      std::size_t classical_gradient_scratch = 0u;
      std::size_t classical_force_scratch = 0u;
      std::size_t sparse_gradient_scratch = 0u;
      std::size_t sparse_sequence = 0u;
      std::size_t classical_coordination_adjoint = 0u;
      std::size_t classical_aes2_gradient = 0u;
      std::size_t classical_aes2_coordination = 0u;
      std::size_t classical_d4_weights = 0u;
      std::size_t classical_d4_weight_cn = 0u;
      std::size_t classical_d4_weight_charge = 0u;
      std::size_t classical_d4_atom_scratch = 0u;
      std::size_t classical_d4_coordination = 0u;
      std::size_t classical_d4_batch = 0u;
      std::size_t classical_d4_gradient = 0u;
      std::size_t classical_geometry_gradient = 0u;
      std::size_t gfn1_weights = 0u;
      std::size_t gfn1_weight_cn_derivatives = 0u;
      std::size_t gfn1_coordination_adjoints = 0u;
      std::size_t gfn1_axis_neighbors = 0u;
      std::size_t gfn1_batch_scratch = 0u;
      std::size_t gfn1_gradient_scratch = 0u;
      std::size_t external_qm_scratch = 0u;
      std::size_t external_point_scratch = 0u;
      std::size_t composition_qm_scratch = 0u;
      std::size_t composition_point_scratch = 0u;
    } execution;

    ArenaLayout execution_layout;
    execution.repulsion = execution_layout.append<double>(batch);
    execution.d4_atm = execution_layout.append<double>(d4_enabled ? batch : 0);
    execution.staged_energy = execution_layout.append<double>(batch);
    execution.public_energy = execution_layout.append<double>(batch);
    execution.sequences = execution_layout.append<std::uint32_t>(
        force_mode ? static_cast<std::int64_t>(kSequenceCount) : 1);
    execution.plan_failure = execution_layout.append<std::uint32_t>(1);
    execution.masks = execution_layout.append<std::uint8_t>(force_mode ? kMaskCount * batch : 0);
    execution.system_errors =
        execution_layout.append<std::uint32_t>(force_mode ? kSystemErrorCount * batch : 2 * batch);
    execution.device_errors = execution_layout.append<std::uint32_t>(
        force_mode ? static_cast<std::int64_t>(kDeviceErrorCount) : 2);
    if (force_mode) {
      execution.public_qm_force = execution_layout.append<double>(coordinates);
      execution.public_point_force = execution_layout.append<double>(point_coordinates);
      execution.staged_qm_force = execution_layout.append<double>(coordinates);
      execution.staged_point_force = execution_layout.append<double>(point_coordinates);
      execution.stationary_density = execution_layout.append<double>(matrices);
      execution.stationary_weighted_density = execution_layout.append<double>(matrices);
      execution.stationary_spin_density = execution_layout.append<double>(matrices);
      execution.stationary_shell_charges = execution_layout.append<double>(shells);
      execution.stationary_atomic_charges = execution_layout.append<double>(atoms);
      execution.stationary_atomic_dipoles = execution_layout.append<double>(atomic_dipole_elements);
      execution.stationary_atomic_quadrupoles =
          execution_layout.append<double>(atomic_quadrupole_elements);
      execution.stationary_spin_shell_potential = execution_layout.append<double>(shells);
      execution.post_complete_shell = execution_layout.append<double>(shells);
      execution.post_complete_atom = execution_layout.append<double>(atoms);
      execution.post_complete_dipole = execution_layout.append<double>(atomic_dipole_elements);
      execution.post_complete_quadrupole =
          execution_layout.append<double>(atomic_quadrupole_elements);
      execution.post_shell_scalar = execution_layout.append<double>(shells);
      execution.post_es2_shell = execution_layout.append<double>(shells);
      execution.post_es3_shell = execution_layout.append<double>(shells);
      execution.post_aes2_atom = execution_layout.append<double>(gfn1_enabled ? 0 : atoms);
      execution.post_aes2_dipole = execution_layout.append<double>(atomic_dipole_elements);
      execution.post_aes2_quadrupole = execution_layout.append<double>(atomic_quadrupole_elements);
      execution.post_d4_atom = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.post_periodic_atom = execution_layout.append<double>(periodic_enabled ? atoms : 0);
      execution.post_staged_shell = execution_layout.append<double>(shells);
      execution.post_staged_atom = execution_layout.append<double>(atoms);
      execution.post_staged_dipole = execution_layout.append<double>(atomic_dipole_elements);
      execution.post_staged_quadrupole =
          execution_layout.append<double>(atomic_quadrupole_elements);
      execution.post_staged_shell_scalar = execution_layout.append<double>(shells);
      execution.overlap_adjoint = execution_layout.append<double>(matrices);
      execution.coordination_adjoint = execution_layout.append<double>(atoms);
      execution.dipole_adjoint = execution_layout.append<double>(force_dipole_matrix_elements);
      execution.quadrupole_adjoint =
          execution_layout.append<double>(force_quadrupole_matrix_elements);
      execution.electronic_gradient = execution_layout.append<double>(coordinates);
      execution.classical_force = execution_layout.append<double>(coordinates);
      execution.explicit_qm_force =
          execution_layout.append<double>(explicit_points ? coordinates : 0);
      execution.explicit_point_force =
          execution_layout.append<double>(explicit_points ? point_coordinates : 0);

      execution.post_es2_shell_scratch = execution_layout.append<double>(shells);
      execution.post_aes2_potential_scratch = execution_layout.append<double>(
          gfn1_enabled ? 0 : candidate.host.aes2.potential_scratch_elements());
      execution.post_d4_weights =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.post_d4_weight_cn =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.post_d4_weight_charge =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.post_d4_atom_scratch = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.post_d4_coordination = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.post_d4_batch = execution_layout.append<double>(d4_enabled ? batch : 0);
      execution.post_d4_gradient = execution_layout.append<double>(d4_enabled ? coordinates : 0);
      execution.post_periodic_potential =
          execution_layout.append<double>(periodic_enabled ? atoms : 0);
      execution.post_periodic_raw_response =
          execution_layout.append<double>(periodic_enabled ? atoms : 0);
      execution.post_composition_shell = execution_layout.append<double>(shells);
      execution.post_composition_atom = execution_layout.append<double>(atoms);
      execution.post_composition_dipole = execution_layout.append<double>(atomic_dipole_elements);
      execution.post_composition_quadrupole =
          execution_layout.append<double>(atomic_quadrupole_elements);
      execution.post_bridge_shell = execution_layout.append<double>(shells);
      execution.h0_overlap_scratch = execution_layout.append<double>(matrices);
      execution.h0_coordination_scratch = execution_layout.append<double>(atoms);
      execution.h0_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.hamiltonian_overlap_scratch = execution_layout.append<double>(matrices);
      execution.hamiltonian_dipole_scratch =
          execution_layout.append<double>(force_dipole_matrix_elements);
      execution.hamiltonian_quadrupole_scratch =
          execution_layout.append<double>(force_quadrupole_matrix_elements);
      execution.integral_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.coordination_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.classical_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.classical_force_scratch = execution_layout.append<double>(coordinates);
      execution.sparse_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.sparse_sequence = execution_layout.append<std::uint32_t>(1);
      execution.classical_coordination_adjoint = execution_layout.append<double>(atoms);
      execution.classical_aes2_gradient =
          execution_layout.append<double>(gfn1_enabled ? 0 : coordinates);
      execution.classical_aes2_coordination =
          execution_layout.append<double>(gfn1_enabled ? 0 : atoms);
      execution.classical_d4_weights =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.classical_d4_weight_cn =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.classical_d4_weight_charge =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.classical_d4_atom_scratch = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.classical_d4_coordination = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.classical_d4_batch = execution_layout.append<double>(d4_enabled ? batch : 0);
      execution.classical_d4_gradient =
          execution_layout.append<double>(d4_enabled ? coordinates : 0);
      execution.classical_geometry_gradient = execution_layout.append<double>(coordinates);
      execution.gfn1_weights =
          execution_layout.append<double>(gfn1_enabled ? gfn1_weight_elements : 0);
      execution.gfn1_weight_cn_derivatives =
          execution_layout.append<double>(gfn1_enabled ? gfn1_weight_elements : 0);
      execution.gfn1_coordination_adjoints =
          execution_layout.append<double>(gfn1_enabled ? atoms : 0);
      execution.gfn1_axis_neighbors =
          execution_layout.append<std::int64_t>(gfn1_enabled ? atoms : 0);
      execution.gfn1_batch_scratch = execution_layout.append<double>(gfn1_enabled ? batch : 0);
      execution.gfn1_gradient_scratch =
          execution_layout.append<double>(gfn1_enabled ? coordinates : 0);
      execution.external_qm_scratch =
          execution_layout.append<double>(explicit_points ? coordinates : 0);
      execution.external_point_scratch =
          execution_layout.append<double>(explicit_points ? point_coordinates : 0);
      execution.composition_qm_scratch = execution_layout.append<double>(coordinates);
      execution.composition_point_scratch = execution_layout.append<double>(point_coordinates);
    }
    if (!execution_layout.valid()) {
      error = "force execution arena layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.force_execution_arena.allocate(execution_layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force execution arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = cudaMemsetAsync(candidate.force_execution_arena.get(), 0,
                                  candidate.force_execution_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force execution arena initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    void* const execution_arena = candidate.force_execution_arena.get();
    void* const immutable_arena = candidate.force_immutable_arena.get();
    candidate.atomic_numbers =
        force_mode ? arena_pointer<std::int32_t>(immutable_arena, immutable.atomic_numbers)
                   : nullptr;
    if (gfn1_enabled) {
      candidate.gfn1_repulsion_sqrt_alpha =
          arena_pointer<double>(immutable_arena, immutable.gfn1_repulsion_sqrt_alpha);
      candidate.gfn1_repulsion_effective_charge =
          arena_pointer<double>(immutable_arena, immutable.gfn1_repulsion_effective_charge);
      const std::int64_t gfn1_pairs = candidate.host.gfn1_d3.total_pairs();
      candidate.gfn1_correction_plan = {
          batch,
          atoms,
          gfn1_pairs,
          token,
          candidate.device_topology.atom_offsets,
          candidate.plan_seed.geometry_batch.pair_offsets,
          candidate.plan_seed.geometry_batch.covalent_radii,
          arena_pointer<std::uint8_t>(immutable_arena, immutable.gfn1_reference_counts),
          arena_pointer<double>(immutable_arena, immutable.gfn1_reference_cn),
          arena_pointer_if<double>(immutable_arena, immutable.gfn1_reference_c6,
                                   gfn1_pairs * kGfn1D3ReferencePairStride),
          arena_pointer_if<double>(immutable_arena, immutable.gfn1_pair_rrij, gfn1_pairs),
          arena_pointer_if<double>(immutable_arena, immutable.gfn1_pair_damping_radii, gfn1_pairs),
          arena_pointer<double>(immutable_arena, immutable.gfn1_halogen_scaled_radii),
          arena_pointer<double>(immutable_arena, immutable.gfn1_halogen_bond_strength),
          arena_pointer<std::uint8_t>(immutable_arena, immutable.gfn1_halogen_donor),
          arena_pointer<std::uint8_t>(immutable_arena, immutable.gfn1_halogen_acceptor),
          XtbModelFlavor::kGfn1,
          batch + 1,
          batch + 1,
          atoms,
          atoms,
          gfn1_weight_elements,
          gfn1_pairs * kGfn1D3ReferencePairStride,
          gfn1_pairs,
          gfn1_pairs,
          atoms,
          atoms,
          atoms,
          atoms,
      };
    }
    auto* const repulsion = arena_pointer<double>(execution_arena, execution.repulsion);
    auto* const d4_atm =
        arena_pointer_if<double>(execution_arena, execution.d4_atm, d4_enabled ? batch : 0);
    auto* const staged_energy = arena_pointer<double>(execution_arena, execution.staged_energy);
    auto* const public_energy = arena_pointer<double>(execution_arena, execution.public_energy);
    auto* const sequence_pool = arena_pointer<std::uint32_t>(execution_arena, execution.sequences);
    auto* const plan_failure =
        arena_pointer<std::uint32_t>(execution_arena, execution.plan_failure);
    auto* const system_error_pool =
        arena_pointer<std::uint32_t>(execution_arena, execution.system_errors);
    auto* const device_error_pool =
        arena_pointer<std::uint32_t>(execution_arena, execution.device_errors);
    const auto system_errors = [&](SystemErrorIndex index) {
      return system_error_pool + static_cast<std::int64_t>(index) * batch;
    };
    const auto device_error = [&](DeviceErrorIndex index) {
      return device_error_pool + static_cast<std::int64_t>(index);
    };

    auto& binding = candidate.energy_force;
    binding = {};
    binding.plan.compute_forces = force_mode ? 1u : 0u;
    binding.plan.scc_potential_components = candidate.plan_seed.enabled_components;
    binding.plan.scc_energy_components = candidate.plan_seed.enabled_components;
    binding.plan.plan_token = token;
    const std::uint32_t total_components =
        d4_enabled ? static_cast<std::uint32_t>(Gfn2TotalEnergyComponent::kD4Atm) : 0u;
    binding.plan.total_energy_batch = {batch, total_components, token};

    binding.input.total_energy = {candidate.state_seed.scc.free_energies,
                                  batch,
                                  repulsion,
                                  batch,
                                  d4_atm,
                                  d4_enabled ? batch : 0,
                                  token};
    binding.input.scc_state = {candidate.state_seed.scc.system_statuses,
                               candidate.state_seed.scc.converged, batch, token};
    binding.input.plan_token = token;
    binding.results.energy = {public_energy, batch, token};
    binding.results.plan_token = token;
    binding.intermediates.energy = {staged_energy, batch, token};
    binding.intermediates.plan_token = token;
    binding.workspace.total_energy = {sequence_pool + kTotalSequence, 1, token};
    binding.workspace.plan_failure = plan_failure;
    binding.workspace.plan_failure_elements = 1;
    binding.workspace.plan_token = token;
    binding.diagnostics.execution_system_errors = system_errors(kExecutionSystemError);
    binding.diagnostics.execution_device_error = device_error(kExecutionDeviceError);
    binding.diagnostics.total_energy_system_errors = system_errors(kTotalSystemError);
    binding.diagnostics.total_energy_device_error = device_error(kTotalDeviceError);
    binding.diagnostics.batch_elements = batch;
    binding.diagnostics.plan_token = token;

    if (force_mode) {
      auto* const mask_pool = arena_pointer<std::uint8_t>(execution_arena, execution.masks);
      const auto mask = [&](MaskIndex index) {
        return mask_pool + static_cast<std::int64_t>(index) * batch;
      };
      cuda_status = cudaMemsetAsync(mask(kRequestedMask), 1,
                                    static_cast<std::size_t>(batch) * sizeof(std::uint8_t), stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("force request-mask initialization", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      candidate.submitted = true;

      const auto* const atomic_numbers = candidate.atomic_numbers;
      /* All force descriptors consume the same runtime-owned committed
       * coordinates, including zero-point batches. This is the address updated
       * transactionally by #122 after overlap factorization succeeds. */
      const auto* const positions = candidate.numerical.device.committed_positions;

      std::int64_t maximum_system_shells = 0;
      for (std::int64_t system = 0; system < batch; ++system) {
        maximum_system_shells = std::max(
            maximum_system_shells,
            candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system + 1)] -
                candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system)]);
      }
      binding.plan.integral_batch = {
          batch,
          atoms,
          shells,
          orbitals,
          primitives,
          matrices,
          candidate.host.h0.shell_pair_offsets.back(),
          maximum_system_shells,
          candidate.plan_seed.density_batch.contraction_tiles_per_channel,
          candidate.host.integrals.integral_cutoff,
          token,
          batch + 1,
          batch + 1,
          batch + 1,
          batch + 1,
          static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()),
          atoms + 1,
          shells + 1,
          static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()),
          shells,
          static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()),
          primitives,
          primitives,
          candidate.device_topology.atom_offsets,
          candidate.device_topology.batch_shell_offsets,
          candidate.device_topology.batch_orbital_offsets,
          candidate.device_topology.matrix_offsets,
          arena_pointer<std::int64_t>(immutable_arena, immutable.shell_pair_offsets),
          candidate.device_topology.atom_shell_offsets,
          candidate.device_topology.shell_orbital_offsets,
          arena_pointer<std::int64_t>(immutable_arena, immutable.shell_primitive_offsets),
          candidate.device_topology.shell_to_atom,
          arena_pointer<std::uint8_t>(immutable_arena, immutable.angular_momenta),
          arena_pointer<double>(immutable_arena, immutable.primitive_exponents),
          arena_pointer<double>(immutable_arena, immutable.primitive_coefficients)};
      binding.plan.integral_batch.model =
          gfn1_enabled ? XtbModelFlavor::kGfn1 : XtbModelFlavor::kGfn2;
      bind_integral_task_domains(candidate, binding.plan.integral_batch);
      binding.plan.h0_plan = {
          atoms,
          shells,
          shells,
          shells,
          candidate.host.h0.shell_pair_offsets.back(),
          token,
          arena_pointer<double>(immutable_arena, immutable.h0_atomic_radii),
          arena_pointer<double>(immutable_arena, immutable.h0_shell_levels),
          arena_pointer<double>(immutable_arena, immutable.h0_coordination_scale),
          arena_pointer<double>(immutable_arena, immutable.h0_polynomial),
          arena_pointer<double>(immutable_arena, immutable.h0_pair_scale)};
      binding.plan.hamiltonian_batch = candidate.plan_seed.hamiltonian_batch;
      binding.plan.coordination_batch = candidate.plan_seed.geometry_batch;
      binding.plan.coordination_cache = candidate.plan_seed.geometry_cache;
      binding.plan.geometry_generation = candidate.host.geometry_generation;

      Gfn2ExternalPointChargeDeviceBatch external_batch{};
      if (explicit_points) {
        external_batch = candidate.plan_seed.explicit_point_charge_batch;
      } else {
        const auto* const point_offsets =
            arena_pointer<std::int64_t>(immutable_arena, immutable.point_offsets);
        external_batch = {batch,
                          atoms,
                          shells,
                          0,
                          candidate.device_topology.atom_offsets,
                          candidate.device_topology.batch_shell_offsets,
                          point_offsets,
                          candidate.device_topology.shell_to_atom,
                          nullptr,
                          positions,
                          nullptr,
                          nullptr,
                          nullptr,
                          token};
      }
      binding.plan.external_point_charge_batch = external_batch;
      const Gfn2PeriodicEmbeddingDeviceBatch periodic_batch = candidate.plan_seed.periodic_batch;

      auto& post_plan = binding.plan.post_scc_potential_plan;
      post_plan.enabled_components = candidate.plan_seed.enabled_components;
      post_plan.geometry_generation = candidate.host.geometry_generation;
      post_plan.plan_token = token;
      post_plan.potential_batch = candidate.plan_seed.potential_batch;
      post_plan.scalar_bridge_batch = candidate.plan_seed.scalar_bridge_batch;
      post_plan.es2_batch = candidate.plan_seed.es2_batch;
      post_plan.es2_cache = candidate.plan_seed.es2_cache;
      post_plan.es3_batch = candidate.plan_seed.es3_batch;
      post_plan.aes2_batch = candidate.plan_seed.aes2_batch;
      post_plan.aes2_cache = candidate.plan_seed.aes2_cache;
      post_plan.d4_batch = candidate.plan_seed.d4_batch;
      post_plan.d4_parameters = candidate.plan_seed.d4_parameters;
      post_plan.d4_cache = candidate.numerical.d4_cache;
      post_plan.external_point_charge_batch = external_batch;
      post_plan.external_point_charge_cache = candidate.plan_seed.explicit_point_charge_cache;
      post_plan.periodic_batch = periodic_batch;

      std::uint32_t classical_components =
          static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kRepulsion) |
          static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kES2);
      if (!gfn1_enabled) {
        classical_components |= static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kAES2);
      }
      if (d4_enabled) {
        classical_components |=
            static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4TwoBody) |
            static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4ATM);
      }
      Gfn2D4DeviceBatch classical_d4_batch = candidate.plan_seed.d4_batch;
      if (d4_enabled) {
        /* The classical-force validator requires the repulsion and nested D4
         * views to share one canonical atomic-number leaf. The SCC setup owner
         * stores an equivalent copy in its input arena, but retaining both
         * addresses here would make the composed force plan cross-projection. */
        classical_d4_batch.atomic_numbers = atomic_numbers;
      }
      binding.plan.classical_plan = {batch,
                                     atoms,
                                     shells,
                                     classical_components,
                                     candidate.host.geometry_generation,
                                     token,
                                     candidate.device_topology.atom_offsets,
                                     atomic_numbers,
                                     candidate.plan_seed.geometry_batch,
                                     candidate.plan_seed.geometry_cache,
                                     candidate.plan_seed.es2_batch,
                                     candidate.plan_seed.es2_cache,
                                     candidate.plan_seed.aes2_batch,
                                     candidate.plan_seed.aes2_cache,
                                     classical_d4_batch,
                                     candidate.plan_seed.d4_parameters,
                                     candidate.numerical.d4_cache};
      binding.plan.classical_plan.model =
          gfn1_enabled ? XtbModelFlavor::kGfn1 : XtbModelFlavor::kGfn2;
      binding.plan.classical_plan.repulsion_sqrt_alpha = candidate.gfn1_repulsion_sqrt_alpha;
      binding.plan.classical_plan.repulsion_effective_charge =
          candidate.gfn1_repulsion_effective_charge;
      binding.plan.classical_plan.gfn1_correction = candidate.gfn1_correction_plan;
      binding.plan.classical_plan.repulsion_sqrt_alpha_elements = gfn1_enabled ? atoms : 0;
      binding.plan.classical_plan.repulsion_effective_charge_elements = gfn1_enabled ? atoms : 0;
      std::uint32_t composition_components =
          static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kElectronicGradient) |
          static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kClassicalForce);
      if (explicit_points) {
        composition_components |=
            static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kExplicitPointChargeForce);
      }
      binding.plan.force_composition_batch = {batch,
                                              atoms,
                                              points,
                                              batch + 1,
                                              batch + 1,
                                              candidate.device_topology.atom_offsets,
                                              external_batch.point_charge_offsets,
                                              composition_components,
                                              token};

      /* Step 5 sparse CN VJP leaf: borrow the committed pair-list consumer view
       * and the pair-list dispatch batch from the preprocessing binding.  The
       * final per-system gate already published committed generations and
       * eligibility, so the H0 CN VJP can parity-gate a sparse result against
       * the dense reference.  A disabled leaf keeps zero tokens and disables
       * the differential path. */
      binding.plan.pairlist_committed = candidate.numerical.preprocessing.output.pairlist;
      binding.plan.pairlist_batch = candidate.numerical.preprocessing.plan.pairlist;

      /* Project the committed spin-major state into stable physical arrays.
       * Every terminal force descriptor below consumes these buffers, never
       * SCC iteration scratch or the alpha channel by itself. */
      auto* const stationary_density =
          arena_pointer<double>(execution_arena, execution.stationary_density);
      auto* const stationary_weighted_density =
          arena_pointer<double>(execution_arena, execution.stationary_weighted_density);
      auto* const stationary_spin_density =
          arena_pointer<double>(execution_arena, execution.stationary_spin_density);
      auto* const stationary_shell_charges =
          arena_pointer<double>(execution_arena, execution.stationary_shell_charges);
      auto* const stationary_atomic_charges =
          arena_pointer<double>(execution_arena, execution.stationary_atomic_charges);
      auto* const stationary_atomic_dipoles = arena_pointer_if<double>(
          execution_arena, execution.stationary_atomic_dipoles, atomic_dipole_elements);
      auto* const stationary_atomic_quadrupoles = arena_pointer_if<double>(
          execution_arena, execution.stationary_atomic_quadrupoles, atomic_quadrupole_elements);
      auto* const stationary_spin_shell_potential =
          arena_pointer<double>(execution_arena, execution.stationary_spin_shell_potential);
      const Gfn2SccPotentialDeviceTopologyMultipoles physical{stationary_shell_charges,
                                                              shells,
                                                              stationary_atomic_charges,
                                                              atoms,
                                                              stationary_atomic_dipoles,
                                                              atomic_dipole_elements,
                                                              stationary_atomic_quadrupoles,
                                                              atomic_quadrupole_elements,
                                                              token};

      auto& projection = binding.stationary_projection;
      projection = {};
      projection.enabled = 1u;
      projection.multipoles_enabled = gfn1_enabled ? 0u : 1u;
      projection.batch_size = batch;
      projection.total_atoms = atoms;
      projection.total_shells = shells;
      projection.total_matrix_elements = matrices;
      projection.total_spin_atoms = candidate.device_wavefunction.total_spin_atoms;
      projection.total_spin_shells = candidate.device_wavefunction.total_spin_shells;
      projection.total_spin_matrix_elements =
          candidate.device_wavefunction.total_spin_matrix_elements;
      projection.atom_offsets = candidate.device_topology.atom_offsets;
      projection.shell_offsets = candidate.device_topology.batch_shell_offsets;
      projection.matrix_offsets = candidate.device_topology.matrix_offsets;
      projection.atom_shell_offsets = candidate.device_topology.atom_shell_offsets;
      projection.spin_channels = candidate.device_wavefunction.spin_channels;
      projection.spin_atom_offsets = candidate.device_wavefunction.spin_atom_offsets;
      projection.spin_shell_offsets = candidate.device_wavefunction.spin_shell_offsets;
      projection.spin_matrix_offsets = candidate.device_wavefunction.spin_matrix_offsets;
      projection.spin_coupling_offsets = candidate.plan_seed.spin_batch.coupling_offsets;
      projection.spin_coupling_matrices = candidate.plan_seed.spin_batch.coupling_matrices;
      projection.packed_density = candidate.state_seed.density.density;
      projection.packed_energy_weighted_density =
          candidate.state_seed.density.energy_weighted_density;
      projection.packed_shell_charges = candidate.state_seed.raw_population.qsh;
      projection.packed_atomic_charges = candidate.state_seed.raw_population.qat;
      projection.packed_atomic_dipoles = candidate.state_seed.raw_population.dipole;
      projection.packed_atomic_quadrupoles = candidate.state_seed.raw_population.quadrupole;
      projection.total_density = stationary_density;
      projection.total_energy_weighted_density = stationary_weighted_density;
      projection.spin_density = stationary_spin_density;
      projection.shell_charges = stationary_shell_charges;
      projection.atomic_charges = stationary_atomic_charges;
      projection.atomic_dipoles = stationary_atomic_dipoles;
      projection.atomic_quadrupoles = stationary_atomic_quadrupoles;
      projection.spin_shell_potentials = stationary_spin_shell_potential;

      binding.input.force_activity = {mask(kRequestedMask),
                                      candidate.state_seed.scc.system_statuses, batch, token};
      binding.input.post_scc_potential = {
          {mask(kRequestedMask), candidate.state_seed.scc.system_statuses, batch, token},
          physical.shell_charges,
          physical.shell_elements,
          physical.atomic_charges,
          physical.atom_elements,
          physical.atomic_dipoles,
          physical.dipole_elements,
          physical.atomic_quadrupoles,
          physical.quadrupole_elements,
          token};
      binding.input.post_scc_potential.electric_field_potentials =
          candidate.numerical.field_potential_view;
      if (gfn1_enabled) {
        binding.input.post_scc_potential.electric_field_potentials.dipole = nullptr;
        binding.input.post_scc_potential.electric_field_potentials.dipole_elements = 0;
      }
      binding.input.h0 = {positions,
                          coordinates,
                          candidate.plan_seed.geometry_cache.coordination_numbers,
                          atoms,
                          candidate.input_seed.hamiltonian.overlap,
                          matrices,
                          stationary_density,
                          matrices,
                          stationary_weighted_density,
                          matrices,
                          token};

      auto* const post_complete_shell =
          arena_pointer<double>(execution_arena, execution.post_complete_shell);
      auto* const post_complete_atom =
          arena_pointer<double>(execution_arena, execution.post_complete_atom);
      auto* const post_complete_dipole = arena_pointer_if<double>(
          execution_arena, execution.post_complete_dipole, atomic_dipole_elements);
      auto* const post_complete_quadrupole = arena_pointer_if<double>(
          execution_arena, execution.post_complete_quadrupole, atomic_quadrupole_elements);
      auto* const post_shell_scalar =
          arena_pointer<double>(execution_arena, execution.post_shell_scalar);
      binding.input.hamiltonian = {stationary_density,
                                   matrices,
                                   post_shell_scalar,
                                   shells,
                                   post_complete_dipole,
                                   atomic_dipole_elements,
                                   post_complete_quadrupole,
                                   atomic_quadrupole_elements,
                                   token};
      binding.input.hamiltonian.spin_density = stationary_spin_density;
      binding.input.hamiltonian.spin_density_elements = matrices;
      binding.input.hamiltonian.spin_shell_scalar_potentials = stationary_spin_shell_potential;
      binding.input.hamiltonian.spin_shell_scalar_elements = shells;
      binding.input.classical = {positions,
                                 coordinates,
                                 candidate.plan_seed.geometry_cache.coordination_numbers,
                                 atoms,
                                 physical.shell_charges,
                                 physical.shell_elements,
                                 physical.atomic_charges,
                                 physical.atom_elements,
                                 physical.atomic_dipoles,
                                 physical.dipole_elements,
                                 physical.atomic_quadrupoles,
                                 physical.quadrupole_elements,
                                 token};
      binding.input.external_shell_charges = physical.shell_charges;
      binding.input.external_shell_elements = explicit_points ? shells : 0;
      binding.input.electric_field = {
          candidate.numerical.device.committed_field_vectors,
          3 * batch,
          candidate.numerical.device.committed_positions,
          coordinates,
          token,
      };
      binding.input.raw_atomic_charges = physical.atomic_charges;
      binding.input.raw_atomic_charge_elements = atoms;

      auto* const public_qm_force =
          arena_pointer<double>(execution_arena, execution.public_qm_force);
      auto* const public_point_force = arena_pointer_if<double>(
          execution_arena, execution.public_point_force, point_coordinates);
      auto* const staged_qm_force =
          arena_pointer<double>(execution_arena, execution.staged_qm_force);
      auto* const staged_point_force = arena_pointer_if<double>(
          execution_arena, execution.staged_point_force, point_coordinates);
      binding.results.forces = {public_qm_force, coordinates, public_point_force, point_coordinates,
                                token};

      binding.intermediates.post_scc_potential = {
          {post_complete_shell, shells, post_complete_atom, atoms, post_complete_dipole,
           atomic_dipole_elements, post_complete_quadrupole, atomic_quadrupole_elements, token},
          post_shell_scalar,
          shells,
          token};
      auto& post_intermediates = binding.intermediates.post_scc_potential_intermediates;
      post_intermediates.es2_shell =
          arena_pointer<double>(execution_arena, execution.post_es2_shell);
      post_intermediates.es2_shell_elements = shells;
      post_intermediates.es3_shell =
          arena_pointer<double>(execution_arena, execution.post_es3_shell);
      post_intermediates.es3_shell_elements = shells;
      post_intermediates.aes2_atomic = arena_pointer_if<double>(
          execution_arena, execution.post_aes2_atom, gfn1_enabled ? 0 : atoms);
      post_intermediates.aes2_atomic_elements = gfn1_enabled ? 0 : atoms;
      post_intermediates.aes2_dipole = arena_pointer_if<double>(
          execution_arena, execution.post_aes2_dipole, atomic_dipole_elements);
      post_intermediates.aes2_dipole_elements = atomic_dipole_elements;
      post_intermediates.aes2_quadrupole = arena_pointer_if<double>(
          execution_arena, execution.post_aes2_quadrupole, atomic_quadrupole_elements);
      post_intermediates.aes2_quadrupole_elements = atomic_quadrupole_elements;
      post_intermediates.d4_atomic =
          arena_pointer_if<double>(execution_arena, execution.post_d4_atom, d4_enabled ? atoms : 0);
      post_intermediates.d4_atomic_elements = d4_enabled ? atoms : 0;
      post_intermediates.periodic_atomic = arena_pointer_if<double>(
          execution_arena, execution.post_periodic_atom, periodic_enabled ? atoms : 0);
      post_intermediates.periodic_atomic_elements = periodic_enabled ? atoms : 0;
      post_intermediates.complete = {
          arena_pointer<double>(execution_arena, execution.post_staged_shell),
          shells,
          arena_pointer<double>(execution_arena, execution.post_staged_atom),
          atoms,
          arena_pointer_if<double>(execution_arena, execution.post_staged_dipole,
                                   atomic_dipole_elements),
          atomic_dipole_elements,
          arena_pointer_if<double>(execution_arena, execution.post_staged_quadrupole,
                                   atomic_quadrupole_elements),
          atomic_quadrupole_elements,
          token};
      post_intermediates.shell_scalar =
          arena_pointer<double>(execution_arena, execution.post_staged_shell_scalar);
      post_intermediates.shell_scalar_elements = shells;
      post_intermediates.plan_token = token;

      auto* const overlap_adjoint =
          arena_pointer<double>(execution_arena, execution.overlap_adjoint);
      auto* const electronic_gradient =
          arena_pointer<double>(execution_arena, execution.electronic_gradient);
      binding.intermediates.h0 = {
          overlap_adjoint,
          matrices,
          arena_pointer<double>(execution_arena, execution.coordination_adjoint),
          atoms,
          electronic_gradient,
          coordinates,
          token};
      binding.intermediates.hamiltonian = {
          overlap_adjoint,
          matrices,
          arena_pointer_if<double>(execution_arena, execution.dipole_adjoint,
                                   force_dipole_matrix_elements),
          force_dipole_matrix_elements,
          arena_pointer_if<double>(execution_arena, execution.quadrupole_adjoint,
                                   force_quadrupole_matrix_elements),
          force_quadrupole_matrix_elements,
          token};
      binding.intermediates.classical = {
          arena_pointer<double>(execution_arena, execution.classical_force), coordinates, token};
      binding.intermediates.explicit_qm_forces = arena_pointer_if<double>(
          execution_arena, execution.explicit_qm_force, explicit_points ? coordinates : 0);
      binding.intermediates.explicit_qm_force_elements = explicit_points ? coordinates : 0;
      binding.intermediates.explicit_point_forces = arena_pointer_if<double>(
          execution_arena, execution.explicit_point_force, explicit_points ? point_coordinates : 0);
      binding.intermediates.explicit_point_force_elements = explicit_points ? point_coordinates : 0;
      binding.intermediates.forces = {staged_qm_force, coordinates, staged_point_force,
                                      point_coordinates, token};

      auto& post_workspace = binding.workspace.post_scc_potential;
      post_workspace.es2.shell_scratch =
          arena_pointer<double>(execution_arena, execution.post_es2_shell_scratch);
      post_workspace.es2.shell_elements = shells;
      if (!gfn1_enabled) {
        post_workspace.aes2.potential_scratch =
            arena_pointer<double>(execution_arena, execution.post_aes2_potential_scratch);
        post_workspace.aes2.potential_elements = candidate.host.aes2.potential_scratch_elements();
        post_workspace.aes2.scc_peer_error_scratch = device_error(kPostAes2PeerError);
        post_workspace.aes2.scc_peer_error_elements = 1;
      }
      if (d4_enabled) {
        post_workspace.d4 = {
            arena_pointer<double>(execution_arena, execution.post_d4_weights),
            arena_pointer<double>(execution_arena, execution.post_d4_weight_cn),
            arena_pointer<double>(execution_arena, execution.post_d4_weight_charge),
            d4_weight_elements,
            arena_pointer<double>(execution_arena, execution.post_d4_atom_scratch),
            arena_pointer<double>(execution_arena, execution.post_d4_coordination),
            atoms,
            arena_pointer<double>(execution_arena, execution.post_d4_batch),
            batch,
            arena_pointer<double>(execution_arena, execution.post_d4_gradient),
            coordinates,
            system_errors(kPostStageSystemError),
            batch};
      }
      if (periodic_enabled) {
        post_workspace.periodic = {
            arena_pointer<double>(execution_arena, execution.post_periodic_potential),
            arena_pointer<double>(execution_arena, execution.post_periodic_raw_response),
            sequence_pool + kPostPeriodicSequence,
            atoms,
            1,
            token};
      }
      post_workspace.composition = {
          arena_pointer<double>(execution_arena, execution.post_composition_shell),
          shells,
          arena_pointer<double>(execution_arena, execution.post_composition_atom),
          atoms,
          arena_pointer_if<double>(execution_arena, execution.post_composition_dipole,
                                   atomic_dipole_elements),
          atomic_dipole_elements,
          arena_pointer_if<double>(execution_arena, execution.post_composition_quadrupole,
                                   atomic_quadrupole_elements),
          atomic_quadrupole_elements,
          sequence_pool + kPostCompositionSequence,
          1,
          token};
      post_workspace.scalar_bridge = {
          arena_pointer<double>(execution_arena, execution.post_bridge_shell), shells,
          sequence_pool + kPostBridgeSequence, 1, token};
      post_workspace.active_mask = mask(kPostActiveMask);
      post_workspace.active_elements = batch;
      post_workspace.sequence_active = sequence_pool + kPostSequence;
      post_workspace.sequence_elements = 1;
      post_workspace.stage_system_errors = system_errors(kPostStageSystemError);
      post_workspace.stage_system_error_elements = batch;
      post_workspace.stage_device_error = device_error(kPostStageDeviceError);
      post_workspace.stage_device_error_elements = 1;
      post_workspace.plan_token = token;

      binding.workspace.h0 = {
          arena_pointer<double>(execution_arena, execution.h0_overlap_scratch),
          matrices,
          arena_pointer<double>(execution_arena, execution.h0_coordination_scratch),
          atoms,
          arena_pointer<double>(execution_arena, execution.h0_gradient_scratch),
          coordinates,
          sequence_pool + kH0Sequence,
          1,
          token};
      binding.workspace.hamiltonian = {
          arena_pointer<double>(execution_arena, execution.hamiltonian_overlap_scratch),
          matrices,
          arena_pointer_if<double>(execution_arena, execution.hamiltonian_dipole_scratch,
                                   force_dipole_matrix_elements),
          force_dipole_matrix_elements,
          arena_pointer_if<double>(execution_arena, execution.hamiltonian_quadrupole_scratch,
                                   force_quadrupole_matrix_elements),
          force_quadrupole_matrix_elements,
          sequence_pool + kHamiltonianSequence,
          1,
          token};
      binding.workspace.integral = {
          arena_pointer<double>(execution_arena, execution.integral_gradient_scratch), coordinates,
          sequence_pool + kIntegralSequence, 1, token};
      binding.workspace.electronic = {mask(kH0SuccessMask), mask(kHamiltonianSuccessMask), batch,
                                      token};
      binding.workspace.coordination = {
          nullptr,
          0,
          nullptr,
          0,
          arena_pointer<double>(execution_arena, execution.coordination_gradient_scratch),
          coordinates,
          sequence_pool + kCoordinationSequence,
          1,
          token};

      Gfn2AES2DeviceWorkspace classical_aes2{};
      /* Only the AES2 reverse pass is used here. Assign by name because the
       * update/potential scratch fields precede the gradient fields. */
      if (!gfn1_enabled) {
        classical_aes2.gradient_scratch =
            arena_pointer<double>(execution_arena, execution.classical_aes2_gradient);
        classical_aes2.gradient_elements = coordinates;
        classical_aes2.coordination_scratch =
            arena_pointer<double>(execution_arena, execution.classical_aes2_coordination);
        classical_aes2.coordination_elements = atoms;
      }
      Gfn2D4DeviceWorkspace classical_d4{};
      if (d4_enabled) {
        classical_d4 = {
            arena_pointer<double>(execution_arena, execution.classical_d4_weights),
            arena_pointer<double>(execution_arena, execution.classical_d4_weight_cn),
            arena_pointer<double>(execution_arena, execution.classical_d4_weight_charge),
            d4_weight_elements,
            arena_pointer<double>(execution_arena, execution.classical_d4_atom_scratch),
            arena_pointer<double>(execution_arena, execution.classical_d4_coordination),
            atoms,
            arena_pointer<double>(execution_arena, execution.classical_d4_batch),
            batch,
            arena_pointer<double>(execution_arena, execution.classical_d4_gradient),
            coordinates,
            system_errors(kClassicalPrimitiveSystemError),
            batch};
      }
      const Gfn2GeometryDeviceWorkspace classical_geometry{
          nullptr,
          0,
          nullptr,
          0,
          arena_pointer<double>(execution_arena, execution.classical_geometry_gradient),
          coordinates,
          sequence_pool + kClassicalPrimitiveSequence,
          1,
          token};
      binding.workspace.classical = {
          arena_pointer<double>(execution_arena, execution.classical_gradient_scratch),
          coordinates,
          arena_pointer<double>(execution_arena, execution.classical_force_scratch),
          coordinates,
          arena_pointer<double>(execution_arena, execution.classical_coordination_adjoint),
          atoms,
          mask(kClassicalSelectedMask),
          batch,
          system_errors(kClassicalPrimitiveSystemError),
          batch,
          device_error(kClassicalPrimitiveDeviceError),
          1,
          sequence_pool + kClassicalSequence,
          1,
          classical_aes2,
          classical_d4,
          classical_geometry,
          token,
          {}};
      if (gfn1_enabled) {
        binding.workspace.classical.gfn1_correction = {
            arena_pointer<double>(execution_arena, execution.gfn1_weights),
            arena_pointer<double>(execution_arena, execution.gfn1_weight_cn_derivatives),
            arena_pointer<double>(execution_arena, execution.gfn1_coordination_adjoints),
            arena_pointer<std::int64_t>(execution_arena, execution.gfn1_axis_neighbors),
            arena_pointer<double>(execution_arena, execution.gfn1_batch_scratch),
            arena_pointer<double>(execution_arena, execution.gfn1_gradient_scratch),
            token,
            gfn1_weight_elements,
            gfn1_weight_elements,
            atoms,
            atoms,
            batch,
            coordinates,
        };
      }
      binding.workspace.external_point_charge = {
          arena_pointer_if<double>(execution_arena, execution.external_qm_scratch,
                                   explicit_points ? coordinates : 0),
          explicit_points ? coordinates : 0,
          arena_pointer_if<double>(execution_arena, execution.external_point_scratch,
                                   explicit_points ? point_coordinates : 0),
          explicit_points ? point_coordinates : 0,
          sequence_pool + kExternalSequence,
          1,
          token};
      binding.workspace.force_composition = {
          arena_pointer<double>(execution_arena, execution.composition_qm_scratch),
          coordinates,
          arena_pointer_if<double>(execution_arena, execution.composition_point_scratch,
                                   point_coordinates),
          point_coordinates,
          sequence_pool + kCompositionSequence,
          1,
          token};
      binding.workspace.energy_success_mask = mask(kEnergySuccessMask);
      binding.workspace.post_scc_success_mask = mask(kPostSuccessMask);
      binding.workspace.electronic_success_mask = mask(kElectronicSuccessMask);
      binding.workspace.coordination_success_mask = mask(kCoordinationSuccessMask);
      binding.workspace.classical_success_mask = mask(kClassicalSuccessMask);
      binding.workspace.external_success_mask = mask(kExternalSuccessMask);
      binding.workspace.mask_elements = batch;
      /* Step 5 sparse CN VJP differential scratch and sequence. */
      binding.workspace.sparse_gradient_scratch =
          arena_pointer<double>(execution_arena, execution.sparse_gradient_scratch);
      binding.workspace.sparse_gradient_elements = coordinates;
      binding.workspace.sparse_sequence_active =
          arena_pointer<std::uint32_t>(execution_arena, execution.sparse_sequence);
      binding.workspace.sparse_sequence_elements = 1;

      binding.diagnostics.post_scc_potential = {system_errors(kPostSystemError),
                                                device_error(kPostDeviceError), batch, token};
      binding.diagnostics.electronic = {system_errors(kH0SystemError),
                                        device_error(kH0DeviceError),
                                        system_errors(kHamiltonianSystemError),
                                        device_error(kHamiltonianDeviceError),
                                        system_errors(kIntegralSystemError),
                                        device_error(kIntegralDeviceError),
                                        batch,
                                        token};
      binding.diagnostics.coordination_system_errors = system_errors(kCoordinationSystemError);
      binding.diagnostics.coordination_device_error = device_error(kCoordinationDeviceError);
      binding.diagnostics.classical_system_errors = system_errors(kClassicalSystemError);
      binding.diagnostics.classical_device_error = device_error(kClassicalDeviceError);
      binding.diagnostics.external_system_errors = system_errors(kExternalSystemError);
      binding.diagnostics.external_device_error = device_error(kExternalDeviceError);
      binding.diagnostics.force_composition_system_errors = system_errors(kCompositionSystemError);
      binding.diagnostics.force_composition_plan_error = device_error(kCompositionPlanError);

      /* The setup owner uploads host-generated geometry values, but the force
       * reverse passes compare their compact caches against CUDA arithmetic.
       * Build the geometry and CN-dependent AES2 caches once on the setup
       * stream; the numerical refresh transaction reuses these fixed public
       * addresses after geometry changes. */
      cuda_status = reset_gfn2_geometry_device_errors_cuda(
          batch, binding.diagnostics.coordination_system_errors,
          binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial geometry diagnostic reset", cuda_status);
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      cuda_status = update_gfn2_geometry_cache_cuda(
          binding.plan.coordination_batch, positions, candidate.host.geometry_generation,
          binding.plan.coordination_cache, candidate.workspace_seed.geometry_workspace,
          binding.diagnostics.coordination_system_errors,
          binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial geometry-cache construction", cuda_status);
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (!gfn1_enabled) {
        cuda_status = reset_gfn2_aes2_device_errors_cuda(
            batch, binding.diagnostics.coordination_system_errors,
            binding.diagnostics.coordination_device_error, stream);
        if (cuda_status != cudaSuccess) {
          error = cuda_error_message("CUDA initial AES2 diagnostic reset", cuda_status);
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        cuda_status = update_gfn2_aes2_geometry_cache_cuda(
            candidate.plan_seed.aes2_batch, positions,
            binding.plan.coordination_cache.coordination_numbers, candidate.plan_seed.aes2_cache,
            candidate.workspace_seed.aes2_workspace, binding.diagnostics.coordination_system_errors,
            binding.diagnostics.coordination_device_error, stream);
        if (cuda_status != cudaSuccess) {
          error = cuda_error_message("CUDA initial AES2-cache construction", cuda_status);
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }
    }

    /* Exercise the published descriptors, not merely their host validation.
     * All-bits-one is a NaN for the CUDA-supported IEEE-754 double format, so
     * a finite download proves that terminal publication actually ran. */
    std::size_t output_bytes = 0u;
    if (!checked_bytes(batch, sizeof(double), output_bytes)) {
      error = "energy smoke output extent overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = cudaMemsetAsync(binding.results.energy.total_energy, 0xff, output_bytes, stream);
    if (cuda_status == cudaSuccess && force_mode) {
      if (!checked_bytes(binding.results.forces.qm_force_elements, sizeof(double), output_bytes)) {
        error = "QM-force smoke output extent overflows size_t";
        return XTBLOOM_STATUS_ALLOCATION_FAILED;
      }
      cuda_status = cudaMemsetAsync(binding.results.forces.qm_forces, 0xff, output_bytes, stream);
      if (cuda_status == cudaSuccess && binding.results.forces.point_force_elements != 0) {
        if (!checked_bytes(binding.results.forces.point_force_elements, sizeof(double),
                           output_bytes)) {
          error = "point-force smoke output extent overflows size_t";
          return XTBLOOM_STATUS_ALLOCATION_FAILED;
        }
        cuda_status =
            cudaMemsetAsync(binding.results.forces.point_forces, 0xff, output_bytes, stream);
      }
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(candidate.state_seed.scc.free_energies, 0,
                                    static_cast<std::size_t>(batch) * sizeof(double), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          cudaMemsetAsync(candidate.state_seed.scc.system_statuses, 0,
                          static_cast<std::size_t>(batch) * sizeof(xtbloom_status_t), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(candidate.state_seed.scc.converged, 1,
                                    static_cast<std::size_t>(batch) * sizeof(std::uint8_t), stream);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA energy/force smoke initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (binding.stationary_projection.enabled == 1u) {
      cuda_status = project_gfn2_stationary_force_state_cuda(binding.stationary_projection, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA stationary force projection smoke", cuda_status);
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    /* The binding is constructed before the first device numerical refresh, so
     * neither the general committed pair list nor the D4 committed pair-list/CN
     * bundle is eligible yet. Run the validation smoke on a plan copy with those
     * refresh-owned consumers disabled. The dense coordination VJP and non-D4
     * leaves still exercise the complete publication chain; the production plan
     * retains every D4 component and exercises it after the first real refresh
     * has atomically published the pair lists, CN, generation, and eligibility. */
    Gfn2EnergyForceExecutionDevicePlan validation_plan = binding.plan;
    validation_plan.pairlist_committed.plan_token = 0u;
    validation_plan.pairlist_batch.plan_token = 0u;
    constexpr std::uint32_t kD4SccPotential =
        static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kD4TwoBody);
    constexpr std::uint32_t kD4SccEnergy =
        static_cast<std::uint32_t>(Gfn2SccClassicalEnergyComponent::kD4TwoBody);
    constexpr std::uint32_t kD4AtmEnergy =
        static_cast<std::uint32_t>(Gfn2TotalEnergyComponent::kD4Atm);
    constexpr std::uint32_t kD4ClassicalForces =
        static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4TwoBody) |
        static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4ATM);
    validation_plan.scc_potential_components &= ~kD4SccPotential;
    validation_plan.scc_energy_components &= ~kD4SccEnergy;
    validation_plan.post_scc_potential_plan.enabled_components &= ~kD4SccPotential;
    validation_plan.total_energy_batch.enabled_components &= ~kD4AtmEnergy;
    validation_plan.classical_plan.enabled_components &= ~kD4ClassicalForces;
    Gfn2EnergyForceExecutionDeviceInput validation_input = binding.input;
    validation_input.total_energy.d4_atm = nullptr;
    validation_input.total_energy.d4_atm_elements = 0;
    cuda_status = execute_gfn2_energy_force_cuda(validation_plan, validation_input, binding.results,
                                                 binding.intermediates, binding.workspace,
                                                 binding.diagnostics, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA energy/force composed smoke", cuda_status);
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    /* The smoke borrows the initialized SCC storage only long enough to drive
     * the terminal chain. Restore the immutable device checkpoint so the
     * published runtime still starts at iteration zero. */
    const auto restore_diagnostic = candidate.initializer.upload_async(
        candidate.iteration_arena.get(), candidate.iteration_arena.bytes(), candidate.ready,
        stream);
    if (!restore_diagnostic.success()) {
      error = setup_error_message("CUDA SCC fresh-state restoration", restore_diagnostic.status,
                                  static_cast<std::uint32_t>(restore_diagnostic.error),
                                  static_cast<std::uint32_t>(restore_diagnostic.field),
                                  restore_diagnostic.index);
      return restore_diagnostic.status;
    }
    candidate.submitted = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t build_inference_bindings(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t points = candidate.host.external.total_point_charges;
    const std::uint64_t token = candidate.host.plan_token;
    const std::uint32_t requested = candidate.host.key.flags;
    const bool d4_enabled = candidate.host.d4_enabled;
    const bool gfn1_enabled = candidate.host.gfn1_enabled;
    const bool energy_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    const bool dipoles_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_DIPOLE_MOMENTS)) != 0u;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t d4_weight_elements = 0;
    std::int64_t gfn1_weight_elements = 0;
    if (requested == 0u || !checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(atoms, kGfn2D4MaximumReferences, d4_weight_elements) ||
        !checked_elements(atoms, kGfn1D3MaximumReferences, gfn1_weight_elements)) {
      error = "inference result extent or requested-property mask is invalid";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    struct Offsets {
      std::size_t atomic_numbers = 0u;
      std::size_t terminal_repulsion_candidate = 0u;
      std::size_t terminal_d4_candidate = 0u;
      std::size_t terminal_d4_weights = 0u;
      std::size_t terminal_d4_weight_cn = 0u;
      std::size_t terminal_d4_weight_charge = 0u;
      std::size_t terminal_d4_atom_scratch = 0u;
      std::size_t terminal_d4_coordination_adjoints = 0u;
      std::size_t terminal_d4_batch_scratch = 0u;
      std::size_t terminal_d4_gradient_scratch = 0u;
      std::size_t terminal_epoch_snapshot = 0u;
      std::size_t terminal_repulsion_error = 0u;
      std::size_t terminal_d4_system_errors = 0u;
      std::size_t terminal_d4_device_error = 0u;
      std::size_t terminal_system_errors = 0u;
      std::size_t terminal_plan_error = 0u;
      std::size_t gfn1_weights = 0u;
      std::size_t gfn1_weight_cn_derivatives = 0u;
      std::size_t gfn1_coordination_adjoints = 0u;
      std::size_t gfn1_axis_neighbors = 0u;
      std::size_t gfn1_batch_scratch = 0u;
      std::size_t gfn1_gradient_scratch = 0u;

      std::size_t result_energies = 0u;
      std::size_t result_qm_forces = 0u;
      std::size_t result_atomic_charges = 0u;
      std::size_t result_point_forces = 0u;
      std::size_t result_dipole_moments = 0u;
      std::size_t result_iterations = 0u;
      std::size_t result_converged = 0u;
      std::size_t result_system_statuses = 0u;
      std::size_t publication_epoch_snapshot = 0u;
      std::size_t publication_system_errors = 0u;
      std::size_t publication_plan_error = 0u;
      std::size_t warm_checkpoint_generations = 0u;
      std::size_t warm_checkpoint_batch_ready = 0u;
      std::size_t request_start_mode = 0u;
      std::size_t admitted_warm_checkpoint_generations = 0u;
      std::size_t warm_checkpoint_field_attached = 0u;
      std::size_t warm_checkpoint_field_vectors = 0u;
    } offset;

    ArenaLayout layout;
    offset.atomic_numbers =
        layout.append<std::int32_t>(candidate.atomic_numbers == nullptr ? atoms : 0);
    offset.terminal_repulsion_candidate = layout.append<double>(batch);
    offset.terminal_d4_candidate = layout.append<double>(d4_enabled ? batch : 0);
    offset.terminal_d4_weights = layout.append<double>(d4_enabled ? d4_weight_elements : 0);
    offset.terminal_d4_weight_cn = layout.append<double>(d4_enabled ? d4_weight_elements : 0);
    offset.terminal_d4_weight_charge = layout.append<double>(d4_enabled ? d4_weight_elements : 0);
    offset.terminal_d4_atom_scratch = layout.append<double>(d4_enabled ? atoms : 0);
    offset.terminal_d4_coordination_adjoints = layout.append<double>(d4_enabled ? atoms : 0);
    offset.terminal_d4_batch_scratch = layout.append<double>(d4_enabled ? batch : 0);
    offset.terminal_d4_gradient_scratch = layout.append<double>(d4_enabled ? coordinates : 0);
    offset.terminal_epoch_snapshot = layout.append<std::uint64_t>(1);
    offset.terminal_repulsion_error = layout.append<std::uint32_t>(1);
    offset.terminal_d4_system_errors = layout.append<std::uint32_t>(d4_enabled ? batch : 0);
    offset.terminal_d4_device_error = layout.append<std::uint32_t>(d4_enabled ? 1 : 0);
    offset.terminal_system_errors = layout.append<std::uint32_t>(batch);
    offset.terminal_plan_error = layout.append<std::uint32_t>(1);
    offset.gfn1_weights = layout.append<double>(gfn1_enabled ? gfn1_weight_elements : 0);
    offset.gfn1_weight_cn_derivatives =
        layout.append<double>(gfn1_enabled ? gfn1_weight_elements : 0);
    offset.gfn1_coordination_adjoints = layout.append<double>(gfn1_enabled ? atoms : 0);
    offset.gfn1_axis_neighbors = layout.append<std::int64_t>(gfn1_enabled ? atoms : 0);
    offset.gfn1_batch_scratch = layout.append<double>(gfn1_enabled ? batch : 0);
    offset.gfn1_gradient_scratch = layout.append<double>(gfn1_enabled ? coordinates : 0);

    offset.result_energies = layout.append<double>(energy_requested ? batch : 0);
    offset.result_qm_forces = layout.append<double>(force_requested ? coordinates : 0);
    offset.result_atomic_charges =
        layout.append<double>((charges_requested || dipoles_requested) ? atoms : 0);
    offset.result_point_forces =
        layout.append<double>(point_forces_requested ? point_coordinates : 0);
    offset.result_dipole_moments = layout.append<double>(dipoles_requested ? 3 * batch : 0);
    offset.result_iterations = layout.append<std::int32_t>(batch);
    offset.result_converged = layout.append<std::uint8_t>(batch);
    offset.result_system_statuses = layout.append<xtbloom_status_t>(batch);
    offset.publication_epoch_snapshot = layout.append<std::uint64_t>(1);
    offset.publication_system_errors = layout.append<std::uint32_t>(batch);
    offset.publication_plan_error = layout.append<std::uint32_t>(1);
    offset.warm_checkpoint_generations = layout.append<std::uint64_t>(batch);
    offset.warm_checkpoint_batch_ready = layout.append<std::uint32_t>(1);
    offset.request_start_mode = layout.append<std::uint32_t>(1);
    offset.admitted_warm_checkpoint_generations = layout.append<std::uint64_t>(batch);
    offset.warm_checkpoint_field_attached = layout.append<std::uint8_t>(batch);
    offset.warm_checkpoint_field_vectors = layout.append<double>(3 * batch);
    if (!layout.valid()) {
      error = "inference arena layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }

    cudaError_t cuda_status = candidate.inference_arena.allocate(layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA inference arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    void* const arena = candidate.inference_arena.get();
    cuda_status = cudaMemsetAsync(arena, 0, candidate.inference_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA inference arena initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const std::int32_t* terminal_atomic_numbers = candidate.atomic_numbers;
    if (terminal_atomic_numbers == nullptr) {
      auto* const destination = arena_pointer<std::int32_t>(arena, offset.atomic_numbers);
      cuda_status = cudaMemcpyAsync(destination, candidate.host.key.atomic_numbers.data(),
                                    static_cast<std::size_t>(atoms) * sizeof(std::int32_t),
                                    cudaMemcpyHostToDevice, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA terminal atomic-number upload", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      terminal_atomic_numbers = destination;
    }
    candidate.submitted = true;

    auto& inference = candidate.inference;
    inference = {};
    inference.epoch_consumer = {
        candidate.numerical.preprocessing.geometry_epoch,
        candidate.numerical.device.committed_generations,
        candidate.numerical.device.eligible,
        batch,
        token,
    };

    Gfn2RepulsionDeviceBatch repulsion{
        batch,
        atoms,
        candidate.device_topology.atom_offsets,
        terminal_atomic_numbers,
        candidate.numerical.device.committed_positions,
    };
    if (gfn1_enabled) {
      repulsion.model = XtbModelFlavor::kGfn1;
      repulsion.sqrt_alpha = candidate.gfn1_repulsion_sqrt_alpha;
      repulsion.effective_charge = candidate.gfn1_repulsion_effective_charge;
      repulsion.sqrt_alpha_elements = atoms;
      repulsion.effective_charge_elements = atoms;
    }
    Gfn2D4DeviceBatch terminal_d4 = candidate.plan_seed.d4_batch;
    if (d4_enabled) terminal_d4.atomic_numbers = repulsion.atomic_numbers;
    inference.terminal_plan = {
        kGfn2TerminalClassicalEnergyAbiVersion,
        d4_enabled ? kGfn2TerminalClassicalEnergyAllComponents : 0u,
        token,
        repulsion,
        d4_enabled ? terminal_d4 : Gfn2D4DeviceBatch{},
        d4_enabled ? candidate.plan_seed.d4_parameters : Gfn2D4DeviceParameters{},
        d4_enabled ? candidate.numerical.d4_cache : Gfn2D4PairListDeviceCache{},
        candidate.numerical.preprocessing.geometry_epoch,
        candidate.numerical.device.committed_generations,
        batch,
    };
    inference.terminal_activity = {
        candidate.numerical.device.eligible, batch, token, {nullptr, 0, 0}};
    inference.terminal_results = {
        const_cast<double*>(candidate.energy_force.input.total_energy.repulsion),
        batch,
        d4_enabled ? const_cast<double*>(candidate.energy_force.input.total_energy.d4_atm)
                   : nullptr,
        d4_enabled ? batch : 0,
        token,
    };
    inference.terminal_workspace.repulsion_candidate =
        arena_pointer<double>(arena, offset.terminal_repulsion_candidate);
    inference.terminal_workspace.repulsion_elements = batch;
    inference.terminal_workspace.d4_atm_candidate =
        arena_pointer_if<double>(arena, offset.terminal_d4_candidate, d4_enabled ? batch : 0);
    inference.terminal_workspace.d4_atm_elements = d4_enabled ? batch : 0;
    if (d4_enabled) {
      inference.terminal_workspace.d4 = {
          arena_pointer<double>(arena, offset.terminal_d4_weights),
          arena_pointer<double>(arena, offset.terminal_d4_weight_cn),
          arena_pointer<double>(arena, offset.terminal_d4_weight_charge),
          d4_weight_elements,
          arena_pointer<double>(arena, offset.terminal_d4_atom_scratch),
          arena_pointer<double>(arena, offset.terminal_d4_coordination_adjoints),
          atoms,
          arena_pointer<double>(arena, offset.terminal_d4_batch_scratch),
          batch,
          arena_pointer<double>(arena, offset.terminal_d4_gradient_scratch),
          coordinates,
          arena_pointer<std::uint32_t>(arena, offset.terminal_d4_system_errors),
          batch,
      };
    }
    inference.terminal_workspace.epoch_snapshot =
        arena_pointer<std::uint64_t>(arena, offset.terminal_epoch_snapshot);
    inference.terminal_workspace.epoch_snapshot_elements = 1;
    inference.terminal_workspace.plan_token = token;
    inference.terminal_diagnostics = {
        arena_pointer<std::uint32_t>(arena, offset.terminal_repulsion_error),
        arena_pointer_if<std::uint32_t>(arena, offset.terminal_d4_system_errors,
                                        d4_enabled ? batch : 0),
        d4_enabled ? batch : 0,
        arena_pointer_if<std::uint32_t>(arena, offset.terminal_d4_device_error, d4_enabled ? 1 : 0),
        arena_pointer<std::uint32_t>(arena, offset.terminal_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.terminal_plan_error),
        1,
        token,
    };
    if (gfn1_enabled) {
      inference.gfn1_correction_plan = candidate.gfn1_correction_plan;
      inference.gfn1_correction_workspace = {
          arena_pointer<double>(arena, offset.gfn1_weights),
          arena_pointer<double>(arena, offset.gfn1_weight_cn_derivatives),
          arena_pointer<double>(arena, offset.gfn1_coordination_adjoints),
          arena_pointer<std::int64_t>(arena, offset.gfn1_axis_neighbors),
          arena_pointer<double>(arena, offset.gfn1_batch_scratch),
          arena_pointer<double>(arena, offset.gfn1_gradient_scratch),
          token,
          gfn1_weight_elements,
          gfn1_weight_elements,
          atoms,
          atoms,
          batch,
          coordinates,
      };
    }

    const auto input_if_requested = [](const double* pointer, bool enabled) {
      return enabled ? pointer : nullptr;
    };
    inference.publication_plan = {
        kGfn2InferencePublicationAbiVersion,
        requested,
        token,
        static_cast<std::uint64_t>(candidate.host.key.maximum_iterations),
        batch,
        atoms,
        points,
        candidate.device_topology.atom_offsets,
        points == 0 ? nullptr
                    : candidate.plan_seed.explicit_point_charge_batch.point_charge_offsets,
        candidate.numerical.preprocessing.geometry_epoch,
        candidate.numerical.device.committed_generations,
        batch,
    };
    inference.publication_input = {
        {nullptr, 0, 0},
        candidate.numerical.device.eligible,
        batch,
        candidate.state_seed.scc.iterations,
        candidate.state_seed.scc.converged,
        candidate.state_seed.scc.system_statuses,
        batch,
        input_if_requested(candidate.energy_force.results.energy.total_energy, energy_requested),
        energy_requested ? batch : 0,
        input_if_requested(candidate.energy_force.results.forces.qm_forces, force_requested),
        force_requested ? coordinates : 0,
        input_if_requested(candidate.energy_force.stationary_projection.enabled == 1u
                               ? candidate.energy_force.stationary_projection.atomic_charges
                               : candidate.workspace_seed.physical_topology.atomic_charges,
                           charges_requested || dipoles_requested),
        (charges_requested || dipoles_requested) ? atoms : 0,
        input_if_requested(candidate.energy_force.results.forces.point_forces,
                           point_forces_requested),
        point_forces_requested ? point_coordinates : 0,
        inference.terminal_diagnostics.system_errors,
        batch,
        inference.terminal_diagnostics.plan_error,
        candidate.energy_force.diagnostics.execution_system_errors,
        batch,
        candidate.energy_force.workspace.plan_failure,
        token,
        input_if_requested(candidate.numerical.device.committed_positions, dipoles_requested),
        dipoles_requested ? coordinates : 0,
        input_if_requested(candidate.energy_force.stationary_projection.atomic_dipoles,
                           dipoles_requested),
        dipoles_requested ? coordinates : 0,
    };
    inference.publication_results = {
        arena_pointer_if<double>(arena, offset.result_energies, energy_requested ? batch : 0),
        energy_requested ? batch : 0,
        arena_pointer_if<double>(arena, offset.result_qm_forces, force_requested ? coordinates : 0),
        force_requested ? coordinates : 0,
        arena_pointer_if<double>(arena, offset.result_atomic_charges,
                                 charges_requested ? atoms : 0),
        charges_requested ? atoms : 0,
        arena_pointer_if<double>(arena, offset.result_point_forces,
                                 point_forces_requested ? point_coordinates : 0),
        point_forces_requested ? point_coordinates : 0,
        arena_pointer<std::int32_t>(arena, offset.result_iterations),
        arena_pointer<std::uint8_t>(arena, offset.result_converged),
        arena_pointer<xtbloom_status_t>(arena, offset.result_system_statuses),
        batch,
        token,
        arena_pointer_if<double>(arena, offset.result_dipole_moments,
                                 dipoles_requested ? 3 * batch : 0),
        dipoles_requested ? 3 * batch : 0,
    };
    inference.publication_workspace = {
        arena_pointer<std::uint64_t>(arena, offset.publication_epoch_snapshot), 1, token};
    inference.publication_diagnostics = {
        arena_pointer<std::uint32_t>(arena, offset.publication_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.publication_plan_error),
        1,
        token,
    };
    inference.warm_checkpoint_generations =
        arena_pointer<std::uint64_t>(arena, offset.warm_checkpoint_generations);
    inference.warm_checkpoint_batch_ready =
        arena_pointer<std::uint32_t>(arena, offset.warm_checkpoint_batch_ready);
    inference.request_start_mode = arena_pointer<std::uint32_t>(arena, offset.request_start_mode);
    inference.admitted_warm_checkpoint_generations =
        arena_pointer<std::uint64_t>(arena, offset.admitted_warm_checkpoint_generations);
    inference.warm_checkpoint_field_attached =
        arena_pointer<std::uint8_t>(arena, offset.warm_checkpoint_field_attached);
    inference.warm_checkpoint_field_vectors =
        arena_pointer<double>(arena, offset.warm_checkpoint_field_vectors);
    inference.ready = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t build_public_result_state(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t points = candidate.host.external.total_point_charges;
    const std::uint32_t requested = candidate.host.key.flags;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t lattice_elements = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(batch, 9, lattice_elements)) {
      error = "public CUDA result staging extent overflows int64_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    const bool energy_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    const bool dipoles_requested =
        (requested & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_DIPOLE_MOMENTS)) != 0u;

    struct HostOffsets {
      std::size_t energies = 0u;
      std::size_t qm_forces = 0u;
      std::size_t atomic_charges = 0u;
      std::size_t point_forces = 0u;
      std::size_t dipole_moments = 0u;
      std::size_t iterations = 0u;
      std::size_t converged = 0u;
      std::size_t system_statuses = 0u;
      std::size_t control = 0u;
      std::size_t warm_checkpoint_ready = 0u;
      std::size_t lattice_cells = 0u;
      std::size_t periodic_axes = 0u;
    } host_offset;
    struct DeviceOffsets {
      std::size_t energies = 0u;
      std::size_t qm_forces = 0u;
      std::size_t atomic_charges = 0u;
      std::size_t point_forces = 0u;
      std::size_t dipole_moments = 0u;
      std::size_t iterations = 0u;
      std::size_t converged = 0u;
      std::size_t system_statuses = 0u;
      std::size_t control = 0u;
      std::size_t request_topology_error = 0u;
      std::size_t expected_atom_offsets = 0u;
      std::size_t expected_atomic_numbers = 0u;
      std::size_t expected_molecular_charges = 0u;
      std::size_t expected_unpaired_electrons = 0u;
      std::size_t expected_spin_channels = 0u;
      std::size_t expected_point_offsets = 0u;
      std::size_t expected_response_offsets = 0u;
      std::size_t expected_lattice_cells = 0u;
      std::size_t expected_periodic_axes = 0u;
      std::size_t lattice_cells = 0u;
      std::size_t periodic_axes = 0u;
    } device_offset;
    ArenaLayout host_layout;
    host_offset.energies = host_layout.append<double>(energy_requested ? batch : 0);
    host_offset.qm_forces = host_layout.append<double>(force_requested ? coordinates : 0);
    host_offset.atomic_charges = host_layout.append<double>(charges_requested ? atoms : 0);
    host_offset.point_forces =
        host_layout.append<double>(point_forces_requested ? point_coordinates : 0);
    host_offset.dipole_moments = host_layout.append<double>(dipoles_requested ? 3 * batch : 0);
    host_offset.iterations = host_layout.append<std::int32_t>(batch);
    host_offset.converged = host_layout.append<std::uint8_t>(batch);
    host_offset.system_statuses = host_layout.append<xtbloom_status_t>(batch);
    host_offset.control = host_layout.append<Gfn2PublicResultBridgeControl>(1);
    host_offset.warm_checkpoint_ready = host_layout.append<std::uint32_t>(1);
    host_offset.lattice_cells = host_layout.append<double>(lattice_elements);
    host_offset.periodic_axes = host_layout.append<std::int32_t>(batch);

    ArenaLayout device_layout;
    device_offset.energies = device_layout.append<double>(energy_requested ? batch : 0);
    device_offset.qm_forces = device_layout.append<double>(force_requested ? coordinates : 0);
    device_offset.atomic_charges = device_layout.append<double>(charges_requested ? atoms : 0);
    device_offset.point_forces =
        device_layout.append<double>(point_forces_requested ? point_coordinates : 0);
    device_offset.dipole_moments = device_layout.append<double>(dipoles_requested ? 3 * batch : 0);
    device_offset.iterations = device_layout.append<std::int32_t>(batch);
    device_offset.converged = device_layout.append<std::uint8_t>(batch);
    device_offset.system_statuses = device_layout.append<xtbloom_status_t>(batch);
    device_offset.control = device_layout.append<Gfn2PublicResultBridgeControl>(1);
    device_offset.request_topology_error = device_layout.append<std::uint32_t>(1);
    device_offset.expected_atom_offsets = device_layout.append<std::int64_t>(batch + 1);
    device_offset.expected_atomic_numbers = device_layout.append<std::int32_t>(atoms);
    device_offset.expected_molecular_charges = device_layout.append<double>(batch);
    device_offset.expected_unpaired_electrons = device_layout.append<std::int32_t>(batch);
    device_offset.expected_spin_channels = device_layout.append<std::int32_t>(batch);
    device_offset.expected_point_offsets = device_layout.append<std::int64_t>(batch + 1);
    device_offset.expected_response_offsets = device_layout.append<std::int64_t>(batch + 1);
    device_offset.expected_lattice_cells = device_layout.append<double>(lattice_elements);
    device_offset.expected_periodic_axes = device_layout.append<std::int32_t>(batch);
    device_offset.lattice_cells = device_layout.append<double>(lattice_elements);
    device_offset.periodic_axes = device_layout.append<std::int32_t>(batch);
    if (!host_layout.valid() || !device_layout.valid()) {
      error = "public CUDA result staging layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.public_result_host_arena.allocate(host_layout.bytes());
    if (cuda_status == cudaSuccess) {
      cuda_status = candidate.public_result_device_arena.allocate(device_layout.bytes());
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = candidate.public_result_completion_event.create(cudaEventDisableTiming);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA public result staging allocation", cuda_status);
      return cuda_status == cudaErrorMemoryAllocation ? XTBLOOM_STATUS_ALLOCATION_FAILED
                                                      : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaMemsetAsync(candidate.public_result_device_arena.get(), 0,
                                  candidate.public_result_device_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA public result control initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = true;

    void* const host_arena = candidate.public_result_host_arena.get();
    auto& state = candidate.public_result;
    state = {};
    state.energies =
        arena_pointer_if<double>(host_arena, host_offset.energies, energy_requested ? batch : 0);
    state.qm_forces = arena_pointer_if<double>(host_arena, host_offset.qm_forces,
                                               force_requested ? coordinates : 0);
    state.atomic_charges = arena_pointer_if<double>(host_arena, host_offset.atomic_charges,
                                                    charges_requested ? atoms : 0);
    state.point_forces = arena_pointer_if<double>(host_arena, host_offset.point_forces,
                                                  point_forces_requested ? point_coordinates : 0);
    state.dipole_moments = arena_pointer_if<double>(host_arena, host_offset.dipole_moments,
                                                    dipoles_requested ? 3 * batch : 0);
    state.iterations = arena_pointer<std::int32_t>(host_arena, host_offset.iterations);
    state.converged = arena_pointer<std::uint8_t>(host_arena, host_offset.converged);
    state.system_statuses =
        arena_pointer<xtbloom_status_t>(host_arena, host_offset.system_statuses);
    state.host_control =
        arena_pointer<Gfn2PublicResultBridgeControl>(host_arena, host_offset.control);
    state.warm_checkpoint_ready =
        arena_pointer<std::uint32_t>(host_arena, host_offset.warm_checkpoint_ready);
    void* const device_arena = candidate.public_result_device_arena.get();
    state.request_topology_error =
        arena_pointer<std::uint32_t>(device_arena, device_offset.request_topology_error);
    state.expected_atom_offsets =
        arena_pointer<std::int64_t>(device_arena, device_offset.expected_atom_offsets);
    state.expected_atomic_numbers =
        arena_pointer<std::int32_t>(device_arena, device_offset.expected_atomic_numbers);
    state.expected_molecular_charges =
        arena_pointer<double>(device_arena, device_offset.expected_molecular_charges);
    state.expected_unpaired_electrons =
        arena_pointer<std::int32_t>(device_arena, device_offset.expected_unpaired_electrons);
    state.expected_spin_channels =
        arena_pointer<std::int32_t>(device_arena, device_offset.expected_spin_channels);
    state.expected_point_offsets =
        arena_pointer<std::int64_t>(device_arena, device_offset.expected_point_offsets);
    state.expected_response_offsets =
        arena_pointer<std::int64_t>(device_arena, device_offset.expected_response_offsets);
    state.expected_lattice_cells =
        arena_pointer<double>(device_arena, device_offset.expected_lattice_cells);
    state.expected_periodic_axes =
        arena_pointer<std::int32_t>(device_arena, device_offset.expected_periodic_axes);
    state.staged_lattice_cells = arena_pointer<double>(device_arena, device_offset.lattice_cells);
    state.staged_periodic_axes =
        arena_pointer<std::int32_t>(device_arena, device_offset.periodic_axes);
    state.host_lattice_cells = arena_pointer<double>(host_arena, host_offset.lattice_cells);
    state.host_periodic_axes = arena_pointer<std::int32_t>(host_arena, host_offset.periodic_axes);
    state.device_staging = {
        arena_pointer_if<double>(device_arena, device_offset.energies,
                                 energy_requested ? batch : 0),
        energy_requested ? batch : 0,
        arena_pointer_if<double>(device_arena, device_offset.qm_forces,
                                 force_requested ? coordinates : 0),
        force_requested ? coordinates : 0,
        arena_pointer_if<double>(device_arena, device_offset.atomic_charges,
                                 charges_requested ? atoms : 0),
        charges_requested ? atoms : 0,
        arena_pointer_if<double>(device_arena, device_offset.point_forces,
                                 point_forces_requested ? point_coordinates : 0),
        point_forces_requested ? point_coordinates : 0,
        arena_pointer<std::int32_t>(device_arena, device_offset.iterations),
        arena_pointer<std::uint8_t>(device_arena, device_offset.converged),
        arena_pointer<xtbloom_status_t>(device_arena, device_offset.system_statuses),
        batch,
        candidate.host.plan_token,
        arena_pointer_if<double>(device_arena, device_offset.dipole_moments,
                                 dipoles_requested ? 3 * batch : 0),
        dipoles_requested ? 3 * batch : 0,
    };
    state.diagnostics = {
        arena_pointer<Gfn2PublicResultBridgeControl>(device_arena, device_offset.control),
        1,
        candidate.host.plan_token,
    };
    const auto upload = [&](std::size_t offset, const auto& values) {
      if (cuda_status != cudaSuccess || values.empty()) return;
      cuda_status =
          cudaMemcpyAsync(static_cast<std::byte*>(device_arena) + offset, values.data(),
                          values.size() * sizeof(values[0]), cudaMemcpyHostToDevice, stream);
    };
    upload(device_offset.expected_atom_offsets, candidate.host.key.atom_offsets);
    upload(device_offset.expected_atomic_numbers, candidate.host.key.atomic_numbers);
    upload(device_offset.expected_molecular_charges, candidate.host.key.molecular_charges);
    upload(device_offset.expected_unpaired_electrons, candidate.host.key.unpaired_electrons);
    upload(device_offset.expected_spin_channels, candidate.host.key.spin_channels);
    upload(device_offset.expected_point_offsets, candidate.host.key.point_offsets);
    upload(device_offset.expected_response_offsets, candidate.host.key.response_offsets);
    upload(device_offset.expected_lattice_cells, candidate.host.key.cell_matrices);
    upload(device_offset.expected_periodic_axes, candidate.host.key.periodic_axes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-topology comparison upload", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    state.ready = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t validate_candidate_setup(Prepared& candidate, std::string& error) {
    const auto& binding = candidate.energy_force;
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t setup_systems = candidate.eigensolver_binding.setup_system_error_elements;
    const std::int64_t factor_statuses = candidate.eigensolver_binding.cache.status_elements;
    const std::int64_t qm_force_elements =
        binding.plan.compute_forces == 1u ? binding.results.forces.qm_force_elements : 0;
    const std::int64_t point_force_elements =
        binding.plan.compute_forces == 1u ? binding.results.forces.point_force_elements : 0;
    struct Offsets {
      std::size_t setup_device_error = 0u;
      std::size_t setup_system_errors = 0u;
      std::size_t factor_statuses = 0u;
      std::size_t execution_system_errors = 0u;
      std::size_t execution_device_error = 0u;
      std::size_t plan_failure = 0u;
      std::size_t energies = 0u;
      std::size_t qm_forces = 0u;
      std::size_t point_forces = 0u;
      std::size_t converged = 0u;
    } offset;
    ArenaLayout layout;
    offset.setup_device_error = layout.append<std::uint32_t>(1);
    offset.setup_system_errors = layout.append<std::uint32_t>(setup_systems);
    offset.factor_statuses = layout.append<std::uint32_t>(factor_statuses);
    offset.execution_system_errors = layout.append<std::uint32_t>(batch);
    offset.execution_device_error = layout.append<std::uint32_t>(1);
    offset.plan_failure = layout.append<std::uint32_t>(1);
    offset.energies = layout.append<double>(batch);
    offset.qm_forces = layout.append<double>(qm_force_elements);
    offset.point_forces = layout.append<double>(point_force_elements);
    offset.converged = layout.append<std::uint8_t>(batch);
    if (!layout.valid()) {
      error = "CUDA candidate validation staging layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.candidate_validation_arena.allocate(layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA candidate validation staging allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    void* const arena = candidate.candidate_validation_arena.get();
    auto* const setup_device_error = arena_pointer<std::uint32_t>(arena, offset.setup_device_error);
    auto* const setup_system_errors =
        arena_pointer_if<std::uint32_t>(arena, offset.setup_system_errors, setup_systems);
    auto* const factor_status_values =
        arena_pointer_if<std::uint32_t>(arena, offset.factor_statuses, factor_statuses);
    auto* const execution_system_errors =
        arena_pointer<std::uint32_t>(arena, offset.execution_system_errors);
    auto* const execution_device_error =
        arena_pointer<std::uint32_t>(arena, offset.execution_device_error);
    auto* const plan_failure = arena_pointer<std::uint32_t>(arena, offset.plan_failure);
    auto* const energies = arena_pointer<double>(arena, offset.energies);
    auto* const qm_forces = arena_pointer_if<double>(arena, offset.qm_forces, qm_force_elements);
    auto* const point_forces =
        arena_pointer_if<double>(arena, offset.point_forces, point_force_elements);
    auto* const converged = arena_pointer<std::uint8_t>(arena, offset.converged);
    const auto enqueue = [&](void* destination, const void* source, std::int64_t elements,
                             std::size_t element_size) {
      return elements == 0 ? cudaSuccess
                           : cudaMemcpyAsync(destination, source,
                                             static_cast<std::size_t>(elements) * element_size,
                                             cudaMemcpyDeviceToHost, stream);
    };
    cuda_status = enqueue(setup_device_error, candidate.eigensolver_binding.setup_device_error, 1,
                          sizeof(std::uint32_t));
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(setup_system_errors, candidate.eigensolver_binding.setup_system_errors,
                            setup_systems, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          enqueue(factor_status_values, candidate.eigensolver_binding.cache.factor_statuses,
                  factor_statuses, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(execution_system_errors, binding.diagnostics.execution_system_errors,
                            batch, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(execution_device_error, binding.diagnostics.execution_device_error, 1,
                            sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(plan_failure, binding.workspace.plan_failure, 1, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(energies, binding.results.energy.total_energy, batch, sizeof(double));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          enqueue(qm_forces, binding.results.forces.qm_forces, qm_force_elements, sizeof(double));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(point_forces, binding.results.forces.point_forces, point_force_elements,
                            sizeof(double));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          enqueue(converged, candidate.state_seed.scc.converged, batch, sizeof(std::uint8_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaEventRecord(candidate.public_result_completion_event.get(), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaEventSynchronize(candidate.public_result_completion_event.get());
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA candidate validation completion", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = false;

    const auto any_nonzero = [](const std::uint32_t* values, std::int64_t elements) {
      return elements != 0 && std::any_of(values, values + elements,
                                          [](std::uint32_t value) { return value != 0u; });
    };
    if (*setup_device_error != 0u || any_nonzero(setup_system_errors, setup_systems) ||
        any_nonzero(factor_status_values, factor_statuses)) {
      error = "CUDA eigensolver setup reported an asynchronous factorization failure";
      return XTBLOOM_STATUS_EIGENSOLVER_FAILED;
    }

    const auto first_system_error =
        std::find_if(execution_system_errors, execution_system_errors + batch,
                     [](std::uint32_t value) { return value != 0u; });
    if (*execution_device_error != 0u || *plan_failure != 0u ||
        first_system_error != execution_system_errors + batch) {
      std::ostringstream message;
      message << "CUDA energy/force smoke reported device_error=" << *execution_device_error
              << " plan_failure=" << *plan_failure;
      if (first_system_error != execution_system_errors + batch) {
        message << " system=" << std::distance(execution_system_errors, first_system_error)
                << " system_error=" << *first_system_error;
      }
      error = message.str();
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const auto all_finite = [](const double* values, std::int64_t elements) {
      return elements == 0 || std::all_of(values, values + elements,
                                          [](double value) { return std::isfinite(value); });
    };
    if (!all_finite(energies, batch) || !all_finite(qm_forces, qm_force_elements) ||
        !all_finite(point_forces, point_force_elements)) {
      error = "CUDA energy/force smoke did not publish every requested output";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    if (std::any_of(converged, converged + batch, [](std::uint8_t value) { return value != 0u; })) {
      error = "CUDA energy/force smoke failed to restore the fresh SCC state";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    candidate.energy_force_smoke_ready = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t build_candidate(
      TopologyKey&& key, std::vector<double>&& positions, std::vector<double>&& point_positions,
      std::vector<double>&& point_values, std::vector<double>&& point_gammas,
      std::vector<double>&& periodic_shifts, std::vector<double>&& periodic_response,
      std::unique_ptr<Prepared>& output, std::string& error, bool build_request_graph = false) {
    auto candidate = std::make_unique<Prepared>(stream);
    const std::uint64_t fingerprint = key.fingerprint();
    std::uint64_t token = hash_mix(fingerprint ^ next_plan_token++ ^ 0x112112112ULL);
    if (token == 0u) token = next_plan_token++;
    xtbloom_status_t status = candidate->host.build(
        std::move(key), std::move(positions), std::move(point_positions), std::move(point_values),
        std::move(point_gammas), std::move(periodic_shifts), std::move(periodic_response), token,
        error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    status = build_integral_task_domains(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    auto topology_diagnostic =
        candidate->host.gfn1_enabled
            ? Gfn2SccSetupTopology::create(candidate->host.basis, candidate->host.integrals,
                                           candidate->host.gfn1_wavefunction_layout, token,
                                           candidate->topology_owner)
            : Gfn2SccSetupTopology::create(candidate->host.basis, candidate->host.integrals,
                                           candidate->host.wavefunction_layout, token,
                                           candidate->topology_owner);
    if (!topology_diagnostic.success()) {
      error = setup_error_message("CUDA topology owner construction", topology_diagnostic.status,
                                  static_cast<std::uint32_t>(topology_diagnostic.error),
                                  static_cast<std::uint32_t>(topology_diagnostic.field),
                                  topology_diagnostic.index);
      return topology_diagnostic.status;
    }
    cudaError_t cuda_status = candidate->topology_arena.allocate(
        candidate->topology_owner.requirements().immutable_device_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA topology arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    topology_diagnostic = candidate->topology_owner.bind_device_arena_and_upload_async(
        candidate->topology_arena.get(), candidate->topology_arena.bytes(),
        candidate->device_topology, candidate->device_wavefunction, stream);
    if (!topology_diagnostic.success()) {
      error = setup_error_message("CUDA topology upload", topology_diagnostic.status,
                                  static_cast<std::uint32_t>(topology_diagnostic.error),
                                  static_cast<std::uint32_t>(topology_diagnostic.field),
                                  topology_diagnostic.index);
      return topology_diagnostic.status;
    }
    candidate->submitted = true;

    auto input_diagnostic =
        candidate->host.gfn1_enabled
            ? Gfn2SccSetupInputs::create(candidate->host.gfn1_input_sources(),
                                         candidate->topology_owner.host_topology(), token,
                                         candidate->inputs_owner)
            : Gfn2SccSetupInputs::create(candidate->host.input_sources(),
                                         candidate->topology_owner.host_topology(), token,
                                         candidate->inputs_owner);
    if (!input_diagnostic.success()) {
      error = setup_error_message("CUDA input owner construction", input_diagnostic.status,
                                  static_cast<std::uint32_t>(input_diagnostic.error),
                                  static_cast<std::uint32_t>(input_diagnostic.field),
                                  input_diagnostic.index);
      return input_diagnostic.status;
    }
    cuda_status =
        candidate->input_arena.allocate(candidate->inputs_owner.requirements().device_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA input arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    input_diagnostic = candidate->inputs_owner.bind_device_arena_and_upload_async(
        candidate->device_topology, candidate->device_wavefunction, candidate->input_arena.get(),
        candidate->input_arena.bytes(), candidate->plan_seed, candidate->input_seed, stream);
    if (!input_diagnostic.success()) {
      error = setup_error_message("CUDA input upload", input_diagnostic.status,
                                  static_cast<std::uint32_t>(input_diagnostic.error),
                                  static_cast<std::uint32_t>(input_diagnostic.field),
                                  input_diagnostic.index);
      return input_diagnostic.status;
    }
    /* Uniform-field storage is part of every fixed CUDA plan so FRESH calls
     * can attach, change, or detach a field without changing the SCC arena.
     * Numerical refresh binds the address-stable values after that arena is
     * allocated; declaring the topology here lets the arena query reserve the
     * appended field energy/diagnostic storage. */
    candidate->plan_seed.electric_field_batch = {
        candidate->host.basis.batch_size,
        candidate->host.basis.total_atoms,
        candidate->host.basis.batch_size + 1,
        candidate->device_topology.atom_offsets,
        token,
    };

    Gfn2EigensolverOptions eigensolver_options = candidate->plan_seed.eigensolver_options;
    eigensolver_options.jacobi = solver_jacobi;
    auto eigensolver_diagnostic = Gfn2SccSetupEigensolver::create(
        candidate->topology_owner, candidate->host.overlap.data(),
        static_cast<std::int64_t>(candidate->host.overlap.size()),
        candidate->host.geometry_generation, token, solver, solver_parameters, blas,
        eigensolver_options, candidate->eigensolver_owner);
    if (!eigensolver_diagnostic.success()) {
      error = setup_error_message(
          "CUDA eigensolver owner construction", eigensolver_diagnostic.status,
          static_cast<std::uint32_t>(eigensolver_diagnostic.error),
          static_cast<std::uint32_t>(eigensolver_diagnostic.field), eigensolver_diagnostic.index);
      return eigensolver_diagnostic.status;
    }
    const auto& eigensolver_requirements = candidate->eigensolver_owner.requirements();
    const auto arena_diagnostic = query_gfn2_scc_iteration_arena_requirements_cuda(
        candidate->plan_seed, eigensolver_requirements.provider, candidate->iteration_requirements);
    if (!arena_diagnostic.success()) {
      error = "CUDA SCC iteration arena query rejected the composed production plan";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    cuda_status =
        candidate->iteration_arena.allocate(candidate->iteration_requirements.total_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA SCC iteration arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate->provider_host_workspace.allocate(
        eigensolver_requirements.provider.solver_host_workspace_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA SCC provider workspace allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    const auto bind_diagnostic = bind_gfn2_scc_iteration_arena_cuda(
        candidate->plan_seed, eigensolver_requirements.provider, candidate->iteration_requirements,
        candidate->iteration_arena.get(), candidate->iteration_arena.bytes(),
        candidate->provider_host_workspace.get(), candidate->provider_host_workspace.bytes(),
        candidate->state_seed, candidate->workspace_seed, candidate->report_storage);
    if (!bind_diagnostic.success()) {
      error = "CUDA SCC iteration arena binding rejected its own sealed requirements";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    status = build_numerical_refresh_binding(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    candidate->input_seed.electric_field = {
        candidate->numerical.device.committed_field_vectors,
        3 * candidate->host.basis.batch_size,
        candidate->numerical.device.committed_positions,
        3 * candidate->host.basis.total_atoms,
        token,
    };
    auto scc_field_potential_view = candidate->numerical.field_potential_view;
    if (candidate->host.gfn1_enabled) {
      scc_field_potential_view.dipole = nullptr;
      scc_field_potential_view.dipole_elements = 0;
    }
    candidate->input_seed.electric_field_potentials = scc_field_potential_view;
    candidate->plan_seed.classical_energy_batch.electric_field =
        candidate->plan_seed.electric_field_batch;
    /* Report projection rebuilds the composed input descriptors from the
     * seed while preserving runtime-owned numerical leaves. Keeping the field
     * diagnostics explicitly bound here makes the first captured iteration
     * and every replay use the arena's appended field slots. */
    candidate->input_seed.classical_energy.electric_field_multipoles = {
        candidate->workspace_seed.physical_topology.atomic_charges,
        candidate->workspace_seed.physical_topology.atom_elements,
        candidate->workspace_seed.physical_topology.atomic_dipoles,
        candidate->workspace_seed.physical_topology.dipole_elements,
        token,
    };
    candidate->input_seed.classical_energy.electric_field_potentials = scc_field_potential_view;
    candidate->input_seed.free_energy.electric_field =
        candidate->workspace_seed.staged_classical_energy.electric_field;
    candidate->input_seed.free_energy.electric_field_elements =
        candidate->workspace_seed.staged_classical_energy.electric_field_elements;
    cuda_status =
        candidate->eigensolver_setup_arena.allocate(eigensolver_requirements.setup_device_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA eigensolver setup arena allocation", cuda_status);
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    eigensolver_diagnostic = candidate->eigensolver_owner.bind_and_factor_overlap_async(
        candidate->device_topology, candidate->plan_seed, candidate->iteration_requirements,
        candidate->iteration_arena.get(), candidate->iteration_arena.bytes(),
        candidate->workspace_seed, candidate->provider_host_workspace.get(),
        candidate->provider_host_workspace.bytes(), candidate->eigensolver_setup_arena.get(),
        candidate->eigensolver_setup_arena.bytes(), candidate->eigensolver_binding, stream);
    if (!eigensolver_diagnostic.success()) {
      error = setup_error_message(
          "CUDA overlap factorization submission", eigensolver_diagnostic.status,
          static_cast<std::uint32_t>(eigensolver_diagnostic.error),
          static_cast<std::uint32_t>(eigensolver_diagnostic.field), eigensolver_diagnostic.index);
      return eigensolver_diagnostic.status;
    }
    candidate->plan_seed.eigensolver_batch = candidate->eigensolver_binding.batch;
    candidate->plan_seed.eigensolver_provider = candidate->eigensolver_binding.provider;
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
    if (consume_execution_test_fault(Gfn2CudaExecutionTestFault::kSccProviderUncapturedFallback)) {
      candidate->plan_seed.eigensolver_provider.capture_mode =
          Gfn2SccIterationProviderCaptureMode::kUncapturedSegmentRequired;
    }
#endif
    candidate->plan_seed.overlap_cache = candidate->eigensolver_binding.cache;
    candidate->plan_seed.eigensolver_options = candidate->eigensolver_binding.options;

    Gfn2SccIterationHostInitialization initialization = candidate->host.initialization();
    /* Mixed-spin fresh checkpoints must carry the same setup-sealed host
     * layout that produced the device plan; physical offsets alone cannot
     * reconstruct per-system spin-major extents. */
    initialization.wavefunction_layout = candidate->topology_owner.host_wavefunction_layout();
    auto initialization_diagnostic = Gfn2SccIterationInitializer::create(
        candidate->plan_seed, candidate->iteration_requirements, candidate->iteration_arena.get(),
        candidate->iteration_arena.bytes(), candidate->state_seed, candidate->workspace_seed,
        candidate->report_storage, initialization, candidate->initializer);
    if (!initialization_diagnostic.success()) {
      error =
          setup_error_message("CUDA SCC initializer construction", initialization_diagnostic.status,
                              static_cast<std::uint32_t>(initialization_diagnostic.error),
                              static_cast<std::uint32_t>(initialization_diagnostic.field),
                              initialization_diagnostic.index);
      return initialization_diagnostic.status;
    }
    initialization_diagnostic = candidate->initializer.upload_async(
        candidate->iteration_arena.get(), candidate->iteration_arena.bytes(), candidate->ready,
        stream);
    if (!initialization_diagnostic.success()) {
      error = setup_error_message("CUDA SCC initializer upload", initialization_diagnostic.status,
                                  static_cast<std::uint32_t>(initialization_diagnostic.error),
                                  static_cast<std::uint32_t>(initialization_diagnostic.field),
                                  initialization_diagnostic.index);
      return initialization_diagnostic.status;
    }
    const auto report_diagnostic = build_gfn2_scc_iteration_report_binding_cuda(
        candidate->report_storage, candidate->plan_seed, candidate->input_seed,
        candidate->state_seed, candidate->workspace_seed, candidate->scc_binding);
    if (report_diagnostic.error != Gfn2SccIterationBindingError::kSuccess) {
      error = setup_error_message("CUDA SCC report factory", XTBLOOM_STATUS_INVALID_ARGUMENT,
                                  static_cast<std::uint32_t>(report_diagnostic.error),
                                  static_cast<std::uint32_t>(report_diagnostic.field),
                                  report_diagnostic.index);
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    status = build_energy_force_bindings(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = build_inference_bindings(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    /* Numerical refresh is built first because inference consumes its epoch
     * descriptors. Close the reverse failure-invalidation and field-identity
     * edges only after the inference arena owns the stable checkpoint arrays. */
    candidate->numerical.device.warm_checkpoint_generations =
        candidate->inference.warm_checkpoint_generations;
    candidate->numerical.device.warm_checkpoint_field_attached =
        candidate->inference.warm_checkpoint_field_attached;
    candidate->numerical.device.warm_checkpoint_field_vectors =
        candidate->inference.warm_checkpoint_field_vectors;
    status = build_public_result_state(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    candidate->numerical.device.request_error = candidate->public_result.request_topology_error;
    candidate->numerical.preprocessing.admission = {candidate->public_result.request_topology_error,
                                                    1, candidate->host.plan_token};
    candidate->scc_binding.input.admission = {candidate->public_result.request_topology_error, 1,
                                              candidate->host.plan_token};
    candidate->energy_force.input.admission = {candidate->public_result.request_topology_error, 1,
                                               candidate->host.plan_token};
    candidate->energy_force.stationary_projection.request_error =
        candidate->public_result.request_topology_error;
    candidate->inference.terminal_activity.admission = {
        candidate->public_result.request_topology_error, 1, candidate->host.plan_token};
    candidate->inference.publication_input.admission = {
        candidate->public_result.request_topology_error, 1, candidate->host.plan_token};
#ifdef XTBLOOM_CUDA_TEST_HOOKS
    const auto alias_hook =
        static_cast<Gfn2CudaAdmissionAliasTestHook>(g_admission_alias_test_hook.exchange(
            static_cast<std::uint32_t>(Gfn2CudaAdmissionAliasTestHook::kNone),
            std::memory_order_relaxed));
    if (alias_hook == Gfn2CudaAdmissionAliasTestHook::kNumericalCandidatePositions) {
      candidate->numerical.device.candidate_positions =
          reinterpret_cast<double*>(candidate->public_result.request_topology_error);
    } else if (alias_hook == Gfn2CudaAdmissionAliasTestHook::kStationaryAtomicCharges) {
      candidate->energy_force.stationary_projection.atomic_charges =
          reinterpret_cast<double*>(candidate->public_result.request_topology_error);
    }
#endif
    const auto preprocessing_reseal =
        seal_gfn2_preprocessing_binding_cuda(candidate->numerical.preprocessing);
    if (!preprocessing_reseal.success()) {
      error = "CUDA runtime preprocessing admission binding is invalid";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    status = validate_prepared_admission_aliases(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    status = validate_candidate_setup(*candidate, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const Gfn2SccLoopGraphBuildResult loop_graph =
        candidate->scc_loop.build(candidate->scc_binding, candidate->inference.epoch_consumer);
    if (!loop_graph.success()) {
      std::ostringstream message;
      message << "CUDA SCC conditional Graph setup rejected the production binding: status="
              << static_cast<std::uint32_t>(loop_graph.status)
              << " iteration_status=" << static_cast<std::uint32_t>(loop_graph.iteration.status)
              << " stage=" << static_cast<std::uint32_t>(loop_graph.iteration.stage)
              << " cuda=" << static_cast<int>(loop_graph.cuda_status);
      error = message.str();
      return loop_graph.iteration.status == Gfn2SccIterationLaunchStatus::kInvalidBinding
                 ? XTBLOOM_STATUS_INVALID_ARGUMENT
                 : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    /* The setup owner factors its deterministic topology-only seed at
     * generation 1 so setup and graph validation can exercise a usable
     * overlap cache.  The externally visible numerical runtime, however,
     * starts unpublished at epoch 0.  Invalidate only the seed provenance
     * before publishing the candidate so the first real epoch-1 refresh must
     * refactor the caller geometry instead of mistaking the seed factor for a
     * cache hit. */
    cuda_status = cudaMemsetAsync(
        candidate->eigensolver_binding.cache.geometry_generations, 0,
        static_cast<std::size_t>(candidate->host.basis.batch_size) * sizeof(std::uint64_t), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA topology-only overlap cache invalidation", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    candidate->submitted = true;
    /* Native XYZ periodic plans currently execute through the validated CPU
     * periodic bridge.  That bridge publishes the ABI-v3 strain suffix on the
     * host and never enters the molecular CUDA request graph; capturing the
     * ordinary graph here would therefore feed the strain bit into the
     * device-only inference publication binding, which intentionally has no
     * strain storage and rejects the request.  Keep the fixed topology and
     * SCC setup intact, but defer graph construction for native-periodic plans
     * until CUDA-native Ewald/multipole publication is available. */
    if (build_request_graph && !candidate->host.key.native_lattice_enabled) {
      status = build_request_execution_graph(*candidate, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }
    output = std::move(candidate);
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t validate_public_request_pointers(const xtbloom_batch_t& batch,
                                                    const xtbloom_compute_options_t& options,
                                                    const xtbloom_batch_result_t* result,
                                                    std::string& error) {
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t cell_elements = 0;
    std::int64_t dipole_elements = 0;
    const bool lattice_active =
        batch.struct_size >= XTBLOOM_BATCH_V4_SIZE &&
        (batch.cell_matrices.data != nullptr || batch.cell_matrices.size_bytes != 0u ||
         batch.periodic_axes.data != nullptr || batch.periodic_axes.size_bytes != 0u);
    const bool interaction_suffix_present = batch.struct_size >= XTBLOOM_BATCH_V3_SIZE;
    const bool interaction_descriptors_active =
        interaction_suffix_present && (batch.interaction_descriptors.data != nullptr ||
                                       batch.interaction_descriptors.size_bytes != 0u);
    const bool interaction_payload_active =
        interaction_suffix_present &&
        (batch.interaction_payload.data != nullptr || batch.interaction_payload.size_bytes != 0u);
    if (!checked_elements(batch.total_atoms, 3, coordinates) ||
        !checked_elements(batch.total_point_charges, 3, point_coordinates) ||
        !checked_elements(batch.batch_size, 3, dipole_elements) ||
        (lattice_active && !checked_elements(batch.batch_size, 9, cell_elements))) {
      error = "CUDA public descriptor coordinate extent overflows int64_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const bool point_topology_active = batch.total_point_charges != 0 ||
                                       batch.point_charge_offsets.data != nullptr ||
                                       batch.point_charge_offsets.size_bytes != 0u;
    const bool shift_active = batch.atomic_potential_shifts.data != nullptr ||
                              batch.atomic_potential_shifts.size_bytes != 0u;
    const bool response_active = batch.total_charge_response_elements != 0 ||
                                 batch.charge_response_offsets.data != nullptr ||
                                 batch.charge_response_offsets.size_bytes != 0u ||
                                 batch.charge_response_matrix.data != nullptr ||
                                 batch.charge_response_matrix.size_bytes != 0u;
    CudaValidatedConstBuffer validated_const{};
    CudaValidatedBuffer validated_output{};
    const auto validate_const = [&](const char* name, const xtbloom_const_buffer_t& buffer,
                                    std::int64_t elements, std::size_t element_size,
                                    std::size_t alignment) -> xtbloom_status_t {
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, element_size, bytes)) {
        error = std::string(name) + " extent overflows size_t";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      return validate_cuda_const_buffer(device_id, name, buffer, bytes, alignment,
                                        CudaManagedMemoryPolicy::kReject, validated_const, error);
    };
    const auto validate_output = [&](const char* name, const xtbloom_buffer_t& buffer,
                                     std::int64_t elements, std::size_t element_size,
                                     std::size_t alignment) -> xtbloom_status_t {
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, element_size, bytes)) {
        error = std::string(name) + " extent overflows size_t";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      return validate_cuda_buffer(device_id, name, buffer, bytes, alignment,
                                  CudaManagedMemoryPolicy::kReject, validated_output, error);
    };

    xtbloom_status_t status =
        validate_const("atom_offsets", batch.atom_offsets, batch.batch_size + 1,
                       sizeof(std::int64_t), alignof(std::int64_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("atomic_numbers", batch.atomic_numbers, batch.total_atoms,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        validate_const("positions", batch.positions, coordinates, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("molecular_charges", batch.molecular_charges, batch.batch_size,
                            sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("unpaired_electrons", batch.unpaired_electrons, batch.batch_size,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const bool spin_present =
        batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
        (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);
    const xtbloom_const_buffer_t absent_spin{};
    const xtbloom_const_buffer_t& spin_buffer = spin_present ? batch.spin_channels : absent_spin;
    status = validate_const("spin_channels", spin_buffer, spin_present ? batch.batch_size : 0,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_offsets", batch.point_charge_offsets,
                            point_topology_active ? batch.batch_size + 1 : 0, sizeof(std::int64_t),
                            alignof(std::int64_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_positions", batch.point_charge_positions,
                            point_coordinates, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_values", batch.point_charge_values,
                            batch.total_point_charges, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_gammas", batch.point_charge_gammas,
                            batch.total_point_charges, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("atomic_potential_shifts", batch.atomic_potential_shifts,
                            shift_active ? batch.total_atoms : 0, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("charge_response_offsets", batch.charge_response_offsets,
                            response_active ? batch.batch_size + 1 : 0, sizeof(std::int64_t),
                            alignof(std::int64_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("charge_response_matrix", batch.charge_response_matrix,
                            response_active ? batch.total_charge_response_elements : 0,
                            sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const xtbloom_const_buffer_t absent_interaction{};
    const xtbloom_const_buffer_t& interaction_descriptor_buffer =
        interaction_descriptors_active ? batch.interaction_descriptors : absent_interaction;
    const xtbloom_const_buffer_t& interaction_payload_buffer =
        interaction_payload_active ? batch.interaction_payload : absent_interaction;
    /* HOST descriptors are a public byte store and are decoded with memcpy;
     * only device-resident descriptors require natural alignment for typed
     * device loads. */
    const std::size_t interaction_descriptor_alignment =
        interaction_descriptor_buffer.memory_space == XTBLOOM_MEMORY_HOST
            ? 1u
            : alignof(xtbloom_interaction_t);
    status = validate_const("interaction_descriptors", interaction_descriptor_buffer,
                            interaction_descriptors_active ? batch.total_interactions : 0,
                            sizeof(xtbloom_interaction_t), interaction_descriptor_alignment);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_cuda_const_buffer(
        device_id, "interaction_payload", interaction_payload_buffer,
        interaction_payload_active ? interaction_payload_buffer.size_bytes : 0u,
        alignof(std::int32_t), CudaManagedMemoryPolicy::kReject, validated_const, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const xtbloom_const_buffer_t absent_lattice{};
    const xtbloom_const_buffer_t& cell_buffer =
        lattice_active ? batch.cell_matrices : absent_lattice;
    const xtbloom_const_buffer_t& axes_buffer =
        lattice_active ? batch.periodic_axes : absent_lattice;
    status = validate_const("cell_matrices", cell_buffer, lattice_active ? cell_elements : 0,
                            sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_const("periodic_axes", axes_buffer, lattice_active ? batch.batch_size : 0,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    if (result == nullptr) return XTBLOOM_STATUS_SUCCESS;

    const bool energy_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    const bool dipoles_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_DIPOLE_MOMENTS)) != 0u;
    const bool strain_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_STRAIN_DERIVATIVES)) != 0u;
    std::int64_t strain_elements = 0;
    if (strain_requested && !checked_elements(batch.batch_size, 9, strain_elements)) {
      error = "strain_derivatives extent overflows int64_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    status = validate_output("energies", result->energies, energy_requested ? batch.batch_size : 0,
                             sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_output("forces", result->forces, force_requested ? coordinates : 0,
                             sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        validate_output("atomic_charges", result->atomic_charges,
                        charges_requested ? batch.total_atoms : 0, sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_output("point_charge_forces", result->point_charge_forces,
                             point_forces_requested ? point_coordinates : 0, sizeof(double),
                             alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const xtbloom_buffer_t absent_output{};
    const xtbloom_buffer_t& dipole_buffer = result->struct_size >= XTBLOOM_BATCH_RESULT_V2_SIZE
                                                ? result->dipole_moments
                                                : absent_output;
    status =
        validate_output("dipole_moments", dipole_buffer, dipoles_requested ? dipole_elements : 0,
                        sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const xtbloom_buffer_t& strain_buffer = result->struct_size >= XTBLOOM_BATCH_RESULT_V3_SIZE
                                                ? result->strain_derivatives
                                                : absent_output;
    status =
        validate_output("strain_derivatives", strain_buffer, strain_requested ? strain_elements : 0,
                        sizeof(double), alignof(double));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_output("scc_iterations", result->scc_iterations, batch.batch_size,
                             sizeof(std::int32_t), alignof(std::int32_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_output("scc_converged", result->scc_converged, batch.batch_size,
                             sizeof(std::uint8_t), alignof(std::uint8_t));
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    return validate_output("per_system_status", result->per_system_status, batch.batch_size,
                           sizeof(xtbloom_status_t), alignof(xtbloom_status_t));
  }

  xtbloom_status_t validate_native_lattice_request_sync(const xtbloom_batch_t& batch,
                                                        std::string& error,
                                                        bool allow_native_periodic = false) {
    if (native_lattice_staging_poisoned) {
      error = "CUDA native-cell staging is poisoned by an earlier stream-settlement failure";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (!native_lattice_active(batch)) {
      return XTBLOOM_STATUS_SUCCESS;
    }
    if (native_lattice_staging_pending) {
      const cudaError_t settle_status = cudaStreamSynchronize(stream);
      if (settle_status != cudaSuccess) {
        native_lattice_staging_poisoned = true;
        error = cuda_error_message("CUDA native-cell staging settlement", settle_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      native_lattice_staging_pending = false;
    }

    std::int64_t cell_elements = 0;
    if (!checked_elements(batch.batch_size, 9, cell_elements)) {
      error = "CUDA native-cell extent overflows int64_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    std::size_t cell_bytes = 0u;
    std::size_t axes_bytes = 0u;
    if (!checked_bytes(cell_elements, sizeof(double), cell_bytes) ||
        !checked_bytes(batch.batch_size, sizeof(std::int32_t), axes_bytes)) {
      error = "CUDA native-cell extent overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const bool cells_host = batch.cell_matrices.memory_space == XTBLOOM_MEMORY_HOST;
    const bool axes_host = batch.periodic_axes.memory_space == XTBLOOM_MEMORY_HOST;
    if (cells_host && axes_host) {
      return validate_host_native_lattice_request(batch, error, allow_native_periodic);
    }

    if (native_lattice_event.get() == nullptr) {
      const cudaError_t event_status = native_lattice_event.create(cudaEventDisableTiming);
      if (event_status != cudaSuccess) {
        error = cuda_error_message("CUDA native-cell validation event creation", event_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }

    ArenaLayout layout;
    const std::size_t cell_offset = layout.append<double>(cell_elements);
    const std::size_t axes_offset = layout.append<std::int32_t>(batch.batch_size);
    if (!layout.valid()) {
      error = "CUDA native-cell validation staging layout overflows size_t";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    if (native_lattice_host_arena.bytes() < layout.bytes()) {
      bool inject_allocation_failure = false;
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      inject_allocation_failure =
          consume_execution_test_fault(Gfn2CudaExecutionTestFault::kNativeLatticePinnedAllocation);
      if (inject_allocation_failure) {
        g_native_lattice_allocation_faults.fetch_add(1u, std::memory_order_relaxed);
      }
#endif
      const cudaError_t allocation_status =
          native_lattice_host_arena.allocate(layout.bytes(), inject_allocation_failure);
      if (allocation_status != cudaSuccess) {
        error =
            cuda_error_message("CUDA native-cell validation staging allocation", allocation_status);
        return XTBLOOM_STATUS_ALLOCATION_FAILED;
      }
    }
    void* const arena = native_lattice_host_arena.get();
    auto* const staged_cells = arena_pointer<double>(arena, cell_offset);
    auto* const staged_axes = arena_pointer<std::int32_t>(arena, axes_offset);
    if (cells_host) {
      std::memcpy(staged_cells, batch.cell_matrices.data, cell_bytes);
    } else {
      const cudaError_t copy_status = cudaMemcpyAsync(staged_cells, batch.cell_matrices.data,
                                                      cell_bytes, cudaMemcpyDeviceToHost, stream);
      if (copy_status != cudaSuccess) {
        error = cuda_error_message("CUDA cell-matrix validation copy", copy_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      native_lattice_staging_pending = true;
    }
    if (axes_host) {
      std::memcpy(staged_axes, batch.periodic_axes.data, axes_bytes);
    } else {
      const cudaError_t copy_status = cudaMemcpyAsync(staged_axes, batch.periodic_axes.data,
                                                      axes_bytes, cudaMemcpyDeviceToHost, stream);
      if (copy_status != cudaSuccess) {
        error = cuda_error_message("CUDA periodic-axis validation copy", copy_status);
        if (native_lattice_staging_pending) {
          const cudaError_t settle_status = cudaStreamSynchronize(stream);
          if (settle_status == cudaSuccess) {
            native_lattice_staging_pending = false;
          } else {
            native_lattice_staging_poisoned = true;
            error += "; additionally, " +
                     cuda_error_message("CUDA native-cell staging settlement", settle_status);
          }
        }
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      native_lattice_staging_pending = true;
    }
    const cudaError_t record_status = cudaEventRecord(native_lattice_event.get(), stream);
    if (record_status != cudaSuccess) {
      error = cuda_error_message("CUDA native-cell validation event", record_status);
      if (native_lattice_staging_pending) {
        const cudaError_t settle_status = cudaStreamSynchronize(stream);
        if (settle_status == cudaSuccess) {
          native_lattice_staging_pending = false;
        } else {
          native_lattice_staging_poisoned = true;
          error += "; additionally, " +
                   cuda_error_message("CUDA native-cell staging settlement", settle_status);
        }
      }
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    cudaError_t wait_status = cudaSuccess;
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
    if (consume_execution_test_fault(Gfn2CudaExecutionTestFault::kNativeLatticeCompletionWait)) {
      g_native_lattice_completion_faults.fetch_add(1u, std::memory_order_relaxed);
      wait_status = cudaErrorUnknown;
    } else
#endif
    {
      wait_status = cudaEventSynchronize(native_lattice_event.get());
    }
    if (wait_status != cudaSuccess) {
      native_lattice_staging_poisoned = true;
      error = cuda_error_message("CUDA native-cell validation completion", wait_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    native_lattice_staging_pending = false;

    xtbloom_batch_t staged_batch{};
    std::memcpy(&staged_batch, &batch,
                std::min<std::size_t>(batch.struct_size, sizeof(staged_batch)));
    staged_batch.cell_matrices = {staged_cells, cell_bytes, XTBLOOM_MEMORY_HOST, 0u};
    staged_batch.periodic_axes = {staged_axes, axes_bytes, XTBLOOM_MEMORY_HOST, 0u};
    return validate_host_native_lattice_request(staged_batch, error, allow_native_periodic);
  }

  xtbloom_status_t reset_request_topology_error_locked(Prepared& current, std::string& error) {
    if (!current.public_result.ready || current.public_result.request_topology_error == nullptr) {
      error = "CUDA fixed-topology request gate is not initialized";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const cudaError_t cuda_status = cudaMemsetAsync(current.public_result.request_topology_error, 0,
                                                    sizeof(std::uint32_t), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-topology request gate reset", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t enqueue_fixed_topology_validation_locked(
      Prepared& current, const xtbloom_batch_t& batch, const xtbloom_compute_options_t& options,
      std::string& error) {
    const TopologyKey& key = current.host.key;
    const std::int64_t expected_batch = static_cast<std::int64_t>(key.molecular_charges.size());
    const std::int64_t expected_atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
    const std::int64_t expected_points = key.point_offsets.empty() ? 0 : key.point_offsets.back();
    const std::int64_t expected_response =
        key.response_offsets.empty() ? 0 : key.response_offsets.back();
    const bool periodic_enabled = batch.atomic_potential_shifts.data != nullptr ||
                                  batch.atomic_potential_shifts.size_bytes != 0u ||
                                  batch.total_charge_response_elements != 0 ||
                                  batch.charge_response_offsets.data != nullptr ||
                                  batch.charge_response_offsets.size_bytes != 0u ||
                                  batch.charge_response_matrix.data != nullptr ||
                                  batch.charge_response_matrix.size_bytes != 0u;
    const bool lattice_active = native_lattice_active(batch);
    if (batch.batch_size != expected_batch || batch.total_atoms != expected_atoms ||
        batch.total_point_charges != expected_points || options.model != key.model ||
        options.flags != key.flags || options.max_scc_iterations != key.maximum_iterations ||
        options.charge_tolerance != key.charge_tolerance ||
        options.energy_tolerance != key.energy_tolerance ||
        options.electronic_temperature != key.electronic_temperature ||
        public_scc_mixer(options) != key.scc_mixer ||
        public_scc_mixer_history(options) != key.scc_mixer_history ||
        public_scc_mixer_damping(options) != key.scc_mixer_damping ||
        public_determinism(options) != key.determinism ||
        periodic_enabled != key.periodic_enabled) {
      error = "the batch or compute policy does not match the fixed CUDA plan topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (key.native_lattice_enabled && !lattice_active) {
      error = "the request omitted the native lattice required by the fixed CUDA plan topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const bool point_offsets_active = expected_points != 0 ||
                                      batch.point_charge_offsets.data != nullptr ||
                                      batch.point_charge_offsets.size_bytes != 0u;
    const bool response_active = batch.total_charge_response_elements != 0 ||
                                 batch.charge_response_offsets.data != nullptr ||
                                 batch.charge_response_offsets.size_bytes != 0u ||
                                 batch.charge_response_matrix.data != nullptr ||
                                 batch.charge_response_matrix.size_bytes != 0u;
    if (response_active && batch.total_charge_response_elements != expected_response) {
      error = "total_charge_response_elements does not match the fixed CUDA plan topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const auto& state = current.public_result;
    FixedTopologyComparisonDeviceBinding binding{};
    binding.expected_atom_offsets = state.expected_atom_offsets;
    binding.expected_atomic_numbers = state.expected_atomic_numbers;
    binding.expected_molecular_charges = state.expected_molecular_charges;
    binding.expected_unpaired_electrons = state.expected_unpaired_electrons;
    binding.expected_spin_channels = state.expected_spin_channels;
    binding.expected_point_offsets = state.expected_point_offsets;
    binding.expected_response_offsets = state.expected_response_offsets;
    binding.atom_offset_elements = expected_batch + 1;
    binding.atomic_number_elements = expected_atoms;
    binding.batch_elements = expected_batch;
    binding.point_offset_elements = point_offsets_active ? expected_batch + 1 : 0;
    binding.response_offset_elements = response_active ? expected_batch + 1 : 0;
    /* A molecular fixed plan may still be used by the asynchronous lattice
     * refusal matrix.  In that case the lattice validator owns the deferred
     * NOT_IMPLEMENTED/NOT_SUPPORTED/INVALID result; comparing the incoming
     * XYZ bytes against the plan's all-zero molecular snapshot here would
     * incorrectly turn the refusal into an immediate topology mismatch.  Once
     * a plan is genuinely native-periodic, its immutable cell and axis mask
     * remain part of the identity and must be compared before execution. */
    const bool compare_lattice_identity = lattice_active && key.native_lattice_enabled;
    if (compare_lattice_identity) {
      if (!checked_elements(expected_batch, 9, binding.cell_elements)) {
        error = "native lattice topology extent overflows int64_t";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      binding.periodic_axis_elements = expected_batch;
      binding.expected_cell_matrices = state.expected_lattice_cells;
      binding.expected_periodic_axes = state.expected_periodic_axes;
    }
    binding.request_error = state.request_topology_error;

    if (!validate_or_bind_fixed_topology_field("atom_offsets", batch.atom_offsets, key.atom_offsets,
                                               state.expected_atom_offsets, binding.atom_offsets,
                                               error) ||
        !validate_or_bind_fixed_topology_field("atomic_numbers", batch.atomic_numbers,
                                               key.atomic_numbers, state.expected_atomic_numbers,
                                               binding.atomic_numbers, error) ||
        !validate_or_bind_fixed_topology_field(
            "molecular_charges", batch.molecular_charges, key.molecular_charges,
            state.expected_molecular_charges, binding.molecular_charges, error) ||
        !validate_or_bind_fixed_topology_field(
            "unpaired_electrons", batch.unpaired_electrons, key.unpaired_electrons,
            state.expected_unpaired_electrons, binding.unpaired_electrons, error)) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const bool spin_supplied =
        batch.struct_size >= XTBLOOM_BATCH_V2_SIZE &&
        (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);
    if (spin_supplied) {
      if (!validate_or_bind_fixed_topology_field("spin_channels", batch.spin_channels,
                                                 key.spin_channels, state.expected_spin_channels,
                                                 binding.spin_channels, error)) {
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    } else if (!std::all_of(key.spin_channels.begin(), key.spin_channels.end(),
                            [](std::int32_t channels) { return channels == 1; })) {
      error = "spin_channels does not match the fixed CUDA plan topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (point_offsets_active &&
        !validate_or_bind_fixed_topology_field("point_charge_offsets", batch.point_charge_offsets,
                                               key.point_offsets, state.expected_point_offsets,
                                               binding.point_offsets, error)) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (response_active &&
        !validate_or_bind_fixed_topology_field(
            "charge_response_offsets", batch.charge_response_offsets, key.response_offsets,
            state.expected_response_offsets, binding.response_offsets, error)) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (compare_lattice_identity &&
        (!validate_or_bind_fixed_topology_field("cell_matrices", batch.cell_matrices,
                                                key.cell_matrices, state.expected_lattice_cells,
                                                binding.cell_matrices, error) ||
         !validate_or_bind_fixed_topology_field("periodic_axes", batch.periodic_axes,
                                                key.periodic_axes, state.expected_periodic_axes,
                                                binding.periodic_axes, error))) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    xtbloom_status_t status = reset_request_topology_error_locked(current, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    const cudaError_t cuda_status = compare_fixed_topology_async(binding, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-topology comparison submission", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t enqueue_native_lattice_validation_locked(Prepared& current,
                                                            const xtbloom_batch_t& batch,
                                                            std::string& error) {
    if (batch.struct_size < XTBLOOM_BATCH_V4_SIZE ||
        (batch.cell_matrices.data == nullptr && batch.cell_matrices.size_bytes == 0u &&
         batch.periodic_axes.data == nullptr && batch.periodic_axes.size_bytes == 0u)) {
      return XTBLOOM_STATUS_SUCCESS;
    }
    const auto& state = current.public_result;
    if (state.staged_lattice_cells == nullptr || state.staged_periodic_axes == nullptr ||
        state.host_lattice_cells == nullptr || state.host_periodic_axes == nullptr) {
      error = "CUDA fixed-plan lattice validation storage is not initialized";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    const auto stage = [&](const xtbloom_const_buffer_t& source, void* host_stage,
                           void* device_stage, std::size_t bytes) -> cudaError_t {
      if (source.memory_space == XTBLOOM_MEMORY_HOST) {
        std::memcpy(host_stage, source.data, bytes);
        return cudaMemcpyAsync(device_stage, host_stage, bytes, cudaMemcpyHostToDevice, stream);
      }
      return cudaMemcpyAsync(device_stage, source.data, bytes, cudaMemcpyDeviceToDevice, stream);
    };
    const std::size_t cell_bytes = static_cast<std::size_t>(batch.batch_size) * 9u * sizeof(double);
    const std::size_t axes_bytes =
        static_cast<std::size_t>(batch.batch_size) * sizeof(std::int32_t);
    cudaError_t cuda_status = stage(batch.cell_matrices, state.host_lattice_cells,
                                    state.staged_lattice_cells, cell_bytes);
    if (cuda_status == cudaSuccess) {
      cuda_status = stage(batch.periodic_axes, state.host_periodic_axes, state.staged_periodic_axes,
                          axes_bytes);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-plan lattice staging", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    FixedTopologyComparisonDeviceBinding binding{};
    binding.cell_matrices = state.staged_lattice_cells;
    binding.periodic_axes = state.staged_periodic_axes;
    binding.lattice_systems = batch.batch_size;
    binding.reject_native_external_combinations =
        (batch.total_point_charges != 0 || batch.atomic_potential_shifts.data != nullptr ||
         batch.atomic_potential_shifts.size_bytes != 0u ||
         batch.total_charge_response_elements != 0 ||
         batch.charge_response_offsets.data != nullptr ||
         batch.charge_response_offsets.size_bytes != 0u ||
         batch.charge_response_matrix.data != nullptr ||
         batch.charge_response_matrix.size_bytes != 0u ||
         (batch.struct_size >= XTBLOOM_BATCH_V3_SIZE && batch.total_interactions != 0))
            ? 1u
            : 0u;
    binding.request_error = state.request_topology_error;
    cuda_status = validate_lattice_request_async(binding, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-plan lattice validation submission", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t stage_numerical_ingress_locked(Prepared& current,
                                                  const Gfn2CudaNumericalInputView& input,
                                                  bool strict_warm,
                                                  bool allow_blocking_interaction_readback,
                                                  bool& host_upload_enqueued, std::string& error) {
    host_upload_enqueued = false;
    if (!current.numerical.ready) {
      error = "CUDA GFN2 numerical refresh requires a prepared fixed topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    auto& numerical = current.numerical;
    auto& device = numerical.device;
    if (device.refresh_predecessor_generations == nullptr ||
        device.warm_checkpoint_generations == nullptr) {
      error = "CUDA GFN2 numerical refresh has an incomplete warm-checkpoint binding";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    auto& preprocessing = numerical.preprocessing;
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for numerical refresh", cuda_status);
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }

    std::size_t requested_descriptor_bytes = 0u;
    if (input.total_interactions < 0 ||
        !checked_bytes(input.total_interactions, sizeof(xtbloom_interaction_t),
                       requested_descriptor_bytes)) {
      error = "interaction descriptor extent overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::size_t requested_payload_bytes =
        input.total_interactions == 0 ? 0u : input.interaction_payload.size_bytes;
    std::size_t released_descriptor_capacity = 0u;
    std::size_t released_payload_capacity = 0u;
    if (!checked_bytes(device.batch_size, sizeof(xtbloom_interaction_t),
                       released_descriptor_capacity) ||
        !checked_bytes(device.batch_size, 32u, released_payload_capacity) ||
        numerical.interaction_descriptor_capacity_bytes < released_descriptor_capacity ||
        numerical.interaction_payload_capacity_bytes < released_payload_capacity) {
      error = "CUDA released interaction staging capacity is incomplete";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    /* Validate the complete view before enqueueing any transfer. A synchronous
     * descriptor rejection must not leave an earlier asynchronous host read in
     * flight after the caller is told that the refresh did not start. */
    const auto validate_source = [&](const char* name, const xtbloom_const_buffer_t& buffer,
                                     std::int64_t elements, const double* device_stage,
                                     const double* owned_host_stage,
                                     bool allow_absent_zero = false) -> xtbloom_status_t {
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, sizeof(double), bytes)) {
        error = std::string(name) + " extent overflows size_t";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const bool supported_space = buffer.memory_space == XTBLOOM_MEMORY_HOST ||
                                   buffer.memory_space == XTBLOOM_MEMORY_CUDA_DEVICE;
      const bool absent = buffer.data == nullptr && buffer.size_bytes == 0u;
      if (elements == 0) {
        if (buffer.reserved != 0u || !supported_space) {
          error = std::string(name) + " has malformed empty-buffer metadata";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (allow_absent_zero && absent) {
        if (buffer.reserved != 0u || !supported_space || device_stage == nullptr) {
          error = std::string(name) + " has malformed absent-buffer metadata";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (buffer.reserved != 0u || buffer.data == nullptr || buffer.size_bytes < bytes ||
          !supported_space) {
        error = std::string(name) + " is not a sufficiently large host/CUDA buffer";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (buffer.memory_space == XTBLOOM_MEMORY_HOST &&
          (device_stage == nullptr || owned_host_stage == nullptr)) {
        error = std::string(name) + " has no runtime host-staging projection";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      return XTBLOOM_STATUS_SUCCESS;
    };

    xtbloom_status_t status =
        validate_source("positions", input.positions, device.total_atoms * 3,
                        numerical.host_positions, numerical.owned_host_positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    if (input.total_interactions < 0) {
      error = "total_interactions must be nonnegative";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    std::size_t interaction_descriptor_bytes = 0u;
    if (!checked_bytes(input.total_interactions, sizeof(xtbloom_interaction_t),
                       interaction_descriptor_bytes)) {
      error = "interaction descriptor extent overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const auto validate_byte_source = [&](const char* name, const xtbloom_const_buffer_t& buffer,
                                          std::size_t required, bool allow_absent) {
      const bool absent = buffer.data == nullptr && buffer.size_bytes == 0u;
      const bool supported_space = buffer.memory_space == XTBLOOM_MEMORY_HOST ||
                                   buffer.memory_space == XTBLOOM_MEMORY_CUDA_DEVICE;
      if ((allow_absent && absent && buffer.reserved == 0u && supported_space) ||
          (buffer.reserved == 0u && supported_space && buffer.data != nullptr &&
           buffer.size_bytes >= required)) {
        return XTBLOOM_STATUS_SUCCESS;
      }
      error = std::string(name) + " is not a sufficiently large host/CUDA byte buffer";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    };
    status = validate_byte_source("interaction_descriptors", input.interaction_descriptors,
                                  interaction_descriptor_bytes, input.total_interactions == 0);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_byte_source("interaction_payload", input.interaction_payload, 1u,
                                  input.total_interactions == 0);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_source("point_charge_positions", input.point_charge_positions,
                             device.total_point_charges * 3, numerical.host_point_positions,
                             numerical.owned_host_point_positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_source("point_charge_values", input.point_charge_values,
                             device.total_point_charges, numerical.host_point_values,
                             numerical.owned_host_point_values);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_source("point_charge_gammas", input.point_charge_gammas,
                             device.total_point_charges, numerical.host_point_gammas,
                             numerical.owned_host_point_gammas);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        validate_source("atomic_potential_shifts", input.atomic_potential_shifts,
                        device.periodic_enabled != 0u ? device.total_atoms : 0,
                        numerical.host_periodic_shifts, numerical.owned_host_periodic_shifts, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = validate_source("charge_response_matrix", input.charge_response_matrix,
                             device.periodic_enabled != 0u ? device.total_response_elements : 0,
                             numerical.host_periodic_response,
                             numerical.owned_host_periodic_response, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    std::size_t mask_bytes = 0u;
    if (!checked_bytes(device.batch_size, sizeof(std::uint8_t), mask_bytes)) {
      error = "numerical refresh activity extent overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    auto* const requested = const_cast<std::uint8_t*>(preprocessing.activity.requested_mask);
    const bool absent_mask =
        input.requested_mask.data == nullptr && input.requested_mask.size_bytes == 0u;
    const bool valid_mask_space = input.requested_mask.memory_space == XTBLOOM_MEMORY_HOST ||
                                  input.requested_mask.memory_space == XTBLOOM_MEMORY_CUDA_DEVICE;
    if ((absent_mask && (input.requested_mask.reserved != 0u || !valid_mask_space)) ||
        (!absent_mask &&
         (input.requested_mask.reserved != 0u || input.requested_mask.data == nullptr ||
          input.requested_mask.size_bytes < mask_bytes || !valid_mask_space))) {
      error = "requested_mask is not a sufficiently large host/CUDA uint8 buffer";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const auto is_host_source = [](const xtbloom_const_buffer_t& buffer,
                                   std::int64_t elements) noexcept {
      return elements != 0 && buffer.data != nullptr && buffer.memory_space == XTBLOOM_MEMORY_HOST;
    };
    const bool uses_host_staging =
        is_host_source(input.positions, device.total_atoms * 3) ||
        is_host_source(input.point_charge_positions, device.total_point_charges * 3) ||
        is_host_source(input.point_charge_values, device.total_point_charges) ||
        is_host_source(input.point_charge_gammas, device.total_point_charges) ||
        is_host_source(input.atomic_potential_shifts,
                       device.periodic_enabled != 0u ? device.total_atoms : 0) ||
        is_host_source(input.charge_response_matrix,
                       device.periodic_enabled != 0u ? device.total_response_elements : 0) ||
        (!absent_mask && input.requested_mask.memory_space == XTBLOOM_MEMORY_HOST) ||
        (input.total_interactions != 0 &&
         (input.interaction_descriptors.memory_space == XTBLOOM_MEMORY_HOST ||
          input.interaction_payload.memory_space == XTBLOOM_MEMORY_HOST));

    if (uses_host_staging) {
      cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
      cuda_status = cudaStreamIsCapturing(stream, &capture_status);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA numerical host-upload capture query", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      if (capture_status != cudaStreamCaptureStatusNone) {
        error = "CUDA Graph numerical refresh requires CUDA-device input buffers";
        return XTBLOOM_STATUS_NOT_SUPPORTED;
      }
      if (numerical.host_staging_poisoned) {
        error = "CUDA numerical host staging is unavailable after an earlier completion failure";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      if (current.numerical_host_upload_completion.pending.load(std::memory_order_acquire)) {
        error = "a previous CUDA numerical host upload is still in flight";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }

    /* HOST payload lifetime and fixed-plan workspace together require a
     * plan-owned image. Host descriptors can be compacted by system directly.
     * For device descriptors, a compact payload that fits the released arena
     * is copied at identical offsets and validated in stream order. Only the
     * synchronous entry points may use bounded descriptor readback for a
     * larger sparse caller view. */
    bool interaction_descriptors_prevalidated = false;
    bool interaction_payload_compacted = false;
    bool interaction_payload_staged_at_offsets = false;
    std::size_t interaction_payload_staging_displacement = 0u;
    bool prevalidated_reserved_interaction = false;
    std::uint32_t prevalidated_interaction_error = kGfn2RequestErrorNone;
    std::int64_t staged_interaction_count = input.total_interactions;
    std::size_t staged_interaction_descriptor_bytes = requested_descriptor_bytes;
    if (input.total_interactions != 0 &&
        (input.interaction_descriptors.memory_space == XTBLOOM_MEMORY_HOST ||
         input.interaction_payload.memory_space == XTBLOOM_MEMORY_HOST)) {
      const bool reverse_mixed =
          input.interaction_descriptors.memory_space == XTBLOOM_MEMORY_CUDA_DEVICE &&
          input.interaction_payload.memory_space == XTBLOOM_MEMORY_HOST;
      constexpr std::size_t kPayloadAlignmentSlack = alignof(double) - 1u;
      const bool compact_reverse_mixed =
          reverse_mixed &&
          released_payload_capacity <=
              std::numeric_limits<std::size_t>::max() - kPayloadAlignmentSlack &&
          requested_payload_bytes <= released_payload_capacity + kPayloadAlignmentSlack;
      if (compact_reverse_mixed) {
        /* Preserve descriptor offsets exactly and let the stream-ordered
         * admission kernel validate their device contents. Preserve the HOST
         * base modulo double alignment as well: ABI-valid blocks may be aligned
         * by a compensating offset even when the payload view base is not. */
        interaction_payload_staging_displacement =
            reinterpret_cast<std::uintptr_t>(input.interaction_payload.data) &
            (alignof(double) - 1u);
        std::memset(numerical.owned_host_interaction_payload, 0,
                    released_payload_capacity + 2u * kPayloadAlignmentSlack);
        std::memcpy(
            numerical.owned_host_interaction_payload + interaction_payload_staging_displacement,
            input.interaction_payload.data, requested_payload_bytes);
        interaction_payload_staged_at_offsets = true;
      } else {
        if (reverse_mixed && !allow_blocking_interaction_readback) {
          error =
              "asynchronous CUDA interaction staging requires a device descriptor/host payload "
              "view to fit the fixed released payload capacity";
          return XTBLOOM_STATUS_NOT_SUPPORTED;
        }
        interaction_descriptors_prevalidated = true;
        interaction_payload_compacted =
            input.interaction_payload.memory_space == XTBLOOM_MEMORY_HOST;
        std::memset(numerical.owned_host_interaction_payload, 0, released_payload_capacity);

        const std::size_t descriptor_capacity =
            numerical.interaction_descriptor_capacity_bytes / sizeof(xtbloom_interaction_t);
        if (descriptor_capacity == 0u) {
          error = "CUDA released interaction descriptor staging has zero capacity";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        std::int64_t field_count = 0;
        bool duplicate = false;
        const auto inspect = [&](const xtbloom_interaction_t& item) {
          if (prevalidated_interaction_error != kGfn2RequestErrorNone) return;
          const std::uint32_t type_mask = interaction_type_mask_host(item.type);
          if (item.flags != 0u || item.type == XTBLOOM_INTERACTION_NONE || type_mask == 0u ||
              item.system_index < 0 || item.system_index >= device.batch_size ||
              item.payload_offset > requested_payload_bytes ||
              item.payload_size > requested_payload_bytes - item.payload_offset) {
            prevalidated_interaction_error = kGfn2RequestErrorInvalid;
            return;
          }
          const std::uintptr_t payload_base =
              reinterpret_cast<std::uintptr_t>(input.interaction_payload.data);
          if (item.payload_offset > static_cast<std::uint64_t>(UINTPTR_MAX - payload_base)) {
            prevalidated_interaction_error = kGfn2RequestErrorInvalid;
            return;
          }
          const std::uintptr_t block_address =
              payload_base + static_cast<std::uintptr_t>(item.payload_offset);
          if (item.type == XTBLOOM_INTERACTION_ELECTRIC_FIELD) {
            if (item.payload_size != 32u || block_address % alignof(double) != 0u) {
              prevalidated_interaction_error = kGfn2RequestErrorInvalid;
              return;
            }
            if (interaction_payload_compacted) {
              std::int32_t version = 0;
              std::int32_t reserved = 0;
              std::array<double, 3> field{};
              const auto* block = reinterpret_cast<const std::byte*>(block_address);
              std::memcpy(&version, block, sizeof(version));
              std::memcpy(&reserved, block + sizeof(version), sizeof(reserved));
              std::memcpy(field.data(), block + 2u * sizeof(std::int32_t), sizeof(field));
              if (version != 1 || reserved != 0 ||
                  !std::all_of(field.begin(), field.end(),
                               [](double component) { return std::isfinite(component); })) {
                prevalidated_interaction_error = kGfn2RequestErrorInvalid;
                return;
              }
            }
          } else {
            if (item.payload_size < sizeof(std::int32_t) ||
                block_address % alignof(std::int32_t) != 0u) {
              prevalidated_interaction_error = kGfn2RequestErrorInvalid;
              return;
            }
            prevalidated_reserved_interaction = true;
          }

          auto* seen_masks =
              reinterpret_cast<std::uint32_t*>(numerical.owned_host_interaction_payload);
          std::uint32_t& seen = seen_masks[item.system_index];
          if ((seen & type_mask) != 0u) {
            duplicate = true;
            return;
          }
          seen |= type_mask;
          if (item.type == XTBLOOM_INTERACTION_ELECTRIC_FIELD) {
            if (field_count >= device.batch_size) {
              prevalidated_interaction_error = kGfn2RequestErrorInvalid;
              return;
            }
            numerical.owned_host_interaction_descriptors[field_count++] = item;
          }
        };

        if (input.interaction_descriptors.memory_space == XTBLOOM_MEMORY_HOST) {
          const auto* descriptor_bytes =
              static_cast<const std::byte*>(input.interaction_descriptors.data);
          for (std::int64_t index = 0; index < input.total_interactions; ++index) {
            xtbloom_interaction_t item{};
            std::memcpy(&item, descriptor_bytes + static_cast<std::size_t>(index) * sizeof(item),
                        sizeof(item));
            inspect(item);
            if (prevalidated_interaction_error != kGfn2RequestErrorNone) break;
          }
        } else {
          const auto* descriptor_bytes =
              static_cast<const std::byte*>(input.interaction_descriptors.data);
          std::int64_t offset = 0;
          while (offset < input.total_interactions &&
                 prevalidated_interaction_error == kGfn2RequestErrorNone) {
            const std::size_t count = std::min<std::size_t>(
                descriptor_capacity, static_cast<std::size_t>(input.total_interactions - offset));
            cuda_status = cudaMemcpyAsync(
                numerical.owned_host_interaction_descriptor_snapshot,
                descriptor_bytes + static_cast<std::size_t>(offset) * sizeof(xtbloom_interaction_t),
                count * sizeof(xtbloom_interaction_t), cudaMemcpyDeviceToHost, stream);
            if (cuda_status == cudaSuccess) {
              current.submitted = true;
              cuda_status = cudaStreamSynchronize(stream);
            }
            if (cuda_status != cudaSuccess) {
              error = cuda_error_message("CUDA interaction descriptor readback", cuda_status);
              return XTBLOOM_STATUS_INTERNAL_ERROR;
            }
            current.submitted = false;
            for (std::size_t index = 0; index < count; ++index) {
              inspect(numerical.owned_host_interaction_descriptor_snapshot[index]);
              if (prevalidated_interaction_error != kGfn2RequestErrorNone) break;
            }
            offset += static_cast<std::int64_t>(count);
          }
        }
        if (prevalidated_interaction_error == kGfn2RequestErrorNone && duplicate) {
          prevalidated_interaction_error = kGfn2RequestErrorInvalid;
        }

        staged_interaction_count =
            prevalidated_interaction_error == kGfn2RequestErrorNone ? field_count : 0;
        if (!checked_bytes(staged_interaction_count, sizeof(xtbloom_interaction_t),
                           staged_interaction_descriptor_bytes)) {
          error = "compacted interaction descriptor extent overflows size_t";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        if (interaction_payload_compacted &&
            prevalidated_interaction_error == kGfn2RequestErrorNone) {
          /* The first pass used this arena as one type bitmask per system.
           * Replacing it only after duplicate detection preserves exact
           * duplicate-versus-reserved error priority. */
          std::memset(numerical.owned_host_interaction_payload, 0, released_payload_capacity);
          for (std::int64_t index = 0; index < field_count; ++index) {
            const xtbloom_interaction_t& item = numerical.owned_host_interaction_descriptors[index];
            std::memcpy(
                numerical.owned_host_interaction_payload + 32u * item.system_index,
                static_cast<const std::byte*>(input.interaction_payload.data) + item.payload_offset,
                32u);
          }
        }
      }
    }

    const auto copy_to_owned_host = [&](const char* name, const xtbloom_const_buffer_t& buffer,
                                        std::int64_t elements,
                                        double* destination) -> xtbloom_status_t {
      if (elements == 0 || buffer.data == nullptr || buffer.memory_space != XTBLOOM_MEMORY_HOST) {
        return XTBLOOM_STATUS_SUCCESS;
      }
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, sizeof(double), bytes)) {
        error = std::string(name) + " extent changed after validation";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      std::memcpy(destination, buffer.data, bytes);
      return XTBLOOM_STATUS_SUCCESS;
    };
    status = copy_to_owned_host("positions", input.positions, device.total_atoms * 3,
                                numerical.owned_host_positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        copy_to_owned_host("point_charge_positions", input.point_charge_positions,
                           device.total_point_charges * 3, numerical.owned_host_point_positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_to_owned_host("point_charge_values", input.point_charge_values,
                                device.total_point_charges, numerical.owned_host_point_values);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_to_owned_host("point_charge_gammas", input.point_charge_gammas,
                                device.total_point_charges, numerical.owned_host_point_gammas);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_to_owned_host("atomic_potential_shifts", input.atomic_potential_shifts,
                                device.periodic_enabled != 0u ? device.total_atoms : 0,
                                numerical.owned_host_periodic_shifts);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = copy_to_owned_host("charge_response_matrix", input.charge_response_matrix,
                                device.periodic_enabled != 0u ? device.total_response_elements : 0,
                                numerical.owned_host_periodic_response);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    if (!absent_mask && input.requested_mask.memory_space == XTBLOOM_MEMORY_HOST) {
      std::memcpy(numerical.owned_host_requested, input.requested_mask.data, mask_bytes);
    }
    const auto seal_host_uploads = [&]() -> cudaError_t {
      if (!host_upload_enqueued) return cudaSuccess;
      auto& completion = current.numerical_host_upload_completion;
      /* Publish busy before recording completion: sufficiently short H2Ds may
       * let the private-stream callback run before this function returns. */
      completion.pending.store(true, std::memory_order_release);
      cudaError_t completion_status =
          cudaEventRecord(current.numerical_host_upload_complete.get(), stream);
      if (completion_status == cudaSuccess) {
        completion_status = cudaStreamWaitEvent(current.numerical_host_completion_stream.get(),
                                                current.numerical_host_upload_complete.get(), 0u);
      }
      if (completion_status == cudaSuccess) {
        completion_status = cudaLaunchHostFunc(current.numerical_host_completion_stream.get(),
                                               release_numerical_host_upload, &completion);
      }
      if (completion_status == cudaSuccess) {
        /* Request completion can wait on this event so the public event also
         * proves that the private callback no longer retains Prepared-owned
         * state. Ordinary query/wait then needs no second host stream fence. */
        completion_status = cudaEventRecord(current.numerical_host_release_complete.get(),
                                            current.numerical_host_completion_stream.get());
      }
      if (completion_status == cudaSuccess) {
        current.submitted = true;
      } else {
        /* Earlier H2Ds may already reference the pinned image.  Leave pending
         * set and poison host staging rather than making an unsafe retry. */
        numerical.host_staging_poisoned = true;
      }
      return completion_status;
    };

    const auto resolve_validated = [&](const char* name, const xtbloom_const_buffer_t& buffer,
                                       std::int64_t elements, double* device_stage,
                                       const double* owned_host_stage, const double*& source,
                                       bool allow_absent_zero = false) -> xtbloom_status_t {
      if (elements == 0) {
        source = nullptr;
        return XTBLOOM_STATUS_SUCCESS;
      }
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, sizeof(double), bytes)) {
        error = std::string(name) + " extent changed after validation";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      if (allow_absent_zero && buffer.data == nullptr && buffer.size_bytes == 0u) {
        cuda_status = cudaMemsetAsync(device_stage, 0, bytes, stream);
        if (cuda_status != cudaSuccess) {
          error = cuda_error_message(name, cuda_status);
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        current.submitted = true;
        source = device_stage;
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (buffer.memory_space == XTBLOOM_MEMORY_HOST) {
        cuda_status =
            cudaMemcpyAsync(device_stage, owned_host_stage, bytes, cudaMemcpyHostToDevice, stream);
        if (cuda_status != cudaSuccess) {
          const cudaError_t completion_status = seal_host_uploads();
          error = cuda_error_message(name, cuda_status);
          if (completion_status != cudaSuccess) {
            error += "; host-upload completion enqueue also failed";
          }
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        host_upload_enqueued = true;
        current.submitted = true;
        source = device_stage;
      } else {
        source = static_cast<const double*>(buffer.data);
      }
      return XTBLOOM_STATUS_SUCCESS;
    };

    NumericalRefreshDeviceSources sources{};
    status = resolve_validated("positions", input.positions, device.total_atoms * 3,
                               numerical.host_positions, numerical.owned_host_positions,
                               sources.positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    if (input.total_interactions != 0) {
      if (interaction_descriptors_prevalidated) {
        cuda_status = cudaMemcpyAsync(
            numerical.host_interaction_descriptors, numerical.owned_host_interaction_descriptors,
            staged_interaction_descriptor_bytes, cudaMemcpyHostToDevice, stream);
        if (cuda_status == cudaSuccess) {
          host_upload_enqueued = true;
          current.submitted = true;
          sources.interaction_descriptors = numerical.host_interaction_descriptors;
        }
      } else {
        sources.interaction_descriptors =
            static_cast<const xtbloom_interaction_t*>(input.interaction_descriptors.data);
      }
      if (cuda_status == cudaSuccess && interaction_payload_compacted) {
        cuda_status = cudaMemcpyAsync(numerical.host_interaction_payload,
                                      numerical.owned_host_interaction_payload,
                                      released_payload_capacity, cudaMemcpyHostToDevice, stream);
        if (cuda_status == cudaSuccess) {
          host_upload_enqueued = true;
          current.submitted = true;
          sources.interaction_payload = numerical.host_interaction_payload;
        }
      } else if (cuda_status == cudaSuccess && interaction_payload_staged_at_offsets) {
        cuda_status = cudaMemcpyAsync(
            numerical.host_interaction_payload + interaction_payload_staging_displacement,
            numerical.owned_host_interaction_payload + interaction_payload_staging_displacement,
            requested_payload_bytes, cudaMemcpyHostToDevice, stream);
        if (cuda_status == cudaSuccess) {
          host_upload_enqueued = true;
          current.submitted = true;
          sources.interaction_payload =
              numerical.host_interaction_payload + interaction_payload_staging_displacement;
        }
      } else if (cuda_status == cudaSuccess) {
        sources.interaction_payload = static_cast<const std::byte*>(input.interaction_payload.data);
      }
      if (cuda_status != cudaSuccess) {
        const cudaError_t completion_status = seal_host_uploads();
        error = cuda_error_message("CUDA interaction staging", cuda_status);
        if (completion_status != cudaSuccess)
          error += "; host-upload completion enqueue also failed";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
    sources.total_interactions =
        interaction_descriptors_prevalidated ? staged_interaction_count : input.total_interactions;
    sources.interaction_payload_address =
        reinterpret_cast<std::uintptr_t>(input.interaction_payload.data);
    sources.interaction_payload_bytes = requested_payload_bytes;
    sources.prevalidated_interaction_error = prevalidated_interaction_error;
    sources.interaction_payload_compacted_by_system = interaction_payload_compacted ? 1u : 0u;
    sources.interaction_payload_staged_at_offsets = interaction_payload_staged_at_offsets ? 1u : 0u;
    sources.prevalidated_reserved_interaction = prevalidated_reserved_interaction ? 1u : 0u;
    status = resolve_validated("point_charge_positions", input.point_charge_positions,
                               device.total_point_charges * 3, numerical.host_point_positions,
                               numerical.owned_host_point_positions, sources.point_positions);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = resolve_validated("point_charge_values", input.point_charge_values,
                               device.total_point_charges, numerical.host_point_values,
                               numerical.owned_host_point_values, sources.point_values);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = resolve_validated("point_charge_gammas", input.point_charge_gammas,
                               device.total_point_charges, numerical.host_point_gammas,
                               numerical.owned_host_point_gammas, sources.point_gammas);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = resolve_validated("atomic_potential_shifts", input.atomic_potential_shifts,
                               device.periodic_enabled != 0u ? device.total_atoms : 0,
                               numerical.host_periodic_shifts, numerical.owned_host_periodic_shifts,
                               sources.periodic_shifts, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status =
        resolve_validated("charge_response_matrix", input.charge_response_matrix,
                          device.periodic_enabled != 0u ? device.total_response_elements : 0,
                          numerical.host_periodic_response, numerical.owned_host_periodic_response,
                          sources.periodic_response, true);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    if (absent_mask) {
      cuda_status = cudaMemsetAsync(requested, 1, mask_bytes, stream);
    } else if (input.requested_mask.memory_space == XTBLOOM_MEMORY_HOST) {
      cuda_status = cudaMemcpyAsync(numerical.host_requested, numerical.owned_host_requested,
                                    mask_bytes, cudaMemcpyHostToDevice, stream);
      if (cuda_status == cudaSuccess) {
        host_upload_enqueued = true;
        current.submitted = true;
        cuda_status = cudaMemcpyAsync(requested, numerical.host_requested, mask_bytes,
                                      cudaMemcpyDeviceToDevice, stream);
      }
    } else {
      cuda_status = cudaMemcpyAsync(requested, input.requested_mask.data, mask_bytes,
                                    cudaMemcpyDeviceToDevice, stream);
    }
    if (cuda_status != cudaSuccess) {
      const cudaError_t completion_status = seal_host_uploads();
      error = cuda_error_message("CUDA numerical refresh activity staging", cuda_status);
      if (completion_status != cudaSuccess) {
        error += "; host-upload completion enqueue also failed";
      }
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    cuda_status = seal_host_uploads();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-upload completion enqueue", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cuda_status =
        cudaMemsetAsync(device.interaction_request_error, 0, sizeof(std::uint32_t), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA interaction request-gate reset", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    normalize_gfn2_interactions_kernel<<<1, 1, 0, stream>>>(device, sources);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA interaction semantic normalization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    merge_gfn2_request_error_kernel<<<1, 1, 0, stream>>>(
        device.interaction_request_error, current.public_result.request_topology_error);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA interaction request-gate publication", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (strict_warm) {
      constexpr int kWarmIdentityThreads = 256;
      const auto warm_identity_blocks = static_cast<unsigned int>(
          (static_cast<std::uint64_t>(device.batch_size) + kWarmIdentityThreads - 1u) /
          kWarmIdentityThreads);
      validate_gfn2_warm_field_identity_kernel<<<warm_identity_blocks, kWarmIdentityThreads, 0,
                                                 stream>>>(
          device, current.inference.warm_checkpoint_field_attached,
          current.inference.warm_checkpoint_field_vectors);
      cuda_status = cudaPeekAtLastError();
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA strict-WARM field admission", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
    stage_gfn2_numerical_inputs_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                         stream>>>(device, sources);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical input staging", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;

    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t execute_numerical_body_locked(Prepared& current, cudaStream_t execution_stream,
                                                 std::string& error) {
    if (!current.numerical.ready) {
      error = "CUDA GFN2 numerical refresh requires a prepared fixed topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    auto& numerical = current.numerical;
    auto& device = numerical.device;
    auto& preprocessing = numerical.preprocessing;
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for numerical refresh", cuda_status);
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }

    cuda_status = reset_gfn2_electric_field_device_errors_cuda(
        device.batch_size, device.field_system_errors, device.field_plan_error, execution_stream);
    if (cuda_status == cudaSuccess) {
      const Gfn2ElectricFieldDeviceInput field_input{
          device.candidate_field_vectors, 3 * device.batch_size, device.candidate_positions,
          3 * device.total_atoms, device.plan_token};
      cuda_status = refresh_gfn2_electric_field_potentials_cuda(
          numerical.field_batch, field_input, numerical.field_candidate_potentials,
          device.field_system_errors, device.field_plan_error, execution_stream);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA electric-field potential refresh", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    const auto preprocessing_launch =
        compose_gfn2_preprocessing_epoch_cuda(preprocessing, execution_stream);
    if (!preprocessing_launch.success()) {
      error = "CUDA numerical preprocessing composer rejected the runtime binding";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    if (device.d4_enabled != 0u) {
      cuda_status = reset_gfn2_d4_device_errors_cuda(
          device.batch_size, const_cast<std::uint32_t*>(device.d4_system_errors),
          const_cast<std::uint32_t*>(device.d4_device_error), execution_stream);
      if (cuda_status == cudaSuccess) {
        /* compose_gfn2_preprocessing_epoch_cuda commits the physical pair
         * superset earlier on this stream. The D4 refresh therefore observes
         * current role views without host polling, and its epoch overload is
         * safe under CUDA Graph replay. */
        cuda_status = update_gfn2_d4_pairlist_cache_cuda(
            current.plan_seed.d4_batch, current.plan_seed.d4_parameters,
            preprocessing.geometry_epoch, numerical.d4_refresh_cache, numerical.d4_workspace,
            const_cast<std::uint32_t*>(device.d4_device_error), execution_stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA D4 numerical refresh", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
    if (device.point_enabled != 0u) {
      cuda_status = reset_gfn2_external_point_charge_scc_errors_cuda(
          device.batch_size, const_cast<std::uint32_t*>(device.point_system_errors),
          const_cast<std::uint32_t*>(device.point_plan_error), execution_stream);
      if (cuda_status == cudaSuccess) {
        initialize_gfn2_refresh_sequence_kernel<<<1, 1, 0, execution_stream>>>(
            const_cast<std::uint32_t*>(numerical.point_activity.sequence_active));
        cuda_status = cudaPeekAtLastError();
      }
      if (cuda_status == cudaSuccess) {
        cuda_status = update_gfn2_external_point_charge_scc_potential_cache_cuda(
            numerical.point_batch, numerical.point_activity, numerical.point_candidate,
            kInitialGeometryGeneration, numerical.point_workspace,
            const_cast<std::uint32_t*>(device.point_system_errors),
            const_cast<std::uint32_t*>(device.point_plan_error), execution_stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA explicit-point-charge numerical refresh", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
    if (device.periodic_enabled != 0u) {
      cuda_status = cudaMemsetAsync(
          device.periodic_system_errors, 0,
          static_cast<std::size_t>(device.batch_size) * sizeof(std::uint32_t), execution_stream);
      if (cuda_status == cudaSuccess) {
        cuda_status =
            cudaMemsetAsync(device.periodic_plan_error, 0, sizeof(std::uint32_t), execution_stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA periodic refresh diagnostic reset", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      validate_gfn2_periodic_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                              execution_stream>>>(device);
      cuda_status = cudaPeekAtLastError();
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA periodic numerical validation", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }

    gate_gfn2_numerical_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 1, 0,
                                         execution_stream>>>(device);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical pre-factor publication gate", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const auto factor = current.eigensolver_owner.refactor_overlap_from_device_epoch_async(
        current.eigensolver_setup_arena.get(), current.eigensolver_setup_arena.bytes(),
        current.eigensolver_binding, preprocessing.output.overlap,
        preprocessing.output.overlap_elements, preprocessing.geometry_epoch,
        current.public_result.request_topology_error, execution_stream);
    if (!factor.success()) {
      error = setup_error_message("CUDA numerical overlap refactor", factor.status,
                                  static_cast<std::uint32_t>(factor.error),
                                  static_cast<std::uint32_t>(factor.field), factor.index);
      return factor.status;
    }
    device.factor_generations = current.eigensolver_binding.cache.geometry_generations;
    device.factor_statuses = current.eigensolver_binding.cache.factor_statuses;
    commit_gfn2_numerical_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                           execution_stream>>>(device);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical final publication", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t enqueue_inference_tail_locked(Prepared& current, bool capture_bounded_scc,
                                                 cudaStream_t execution_stream,
                                                 std::string& error) {
    auto& inference = current.inference;
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for CUDA GFN2 inference", cuda_status);
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }

    /* The synchronous path launches the production device-resident early-stop
     * Graph. The asynchronous request Graph instead captures the bounded
     * iteration DAG at setup. Nesting a device-launched/tail-launched SCC
     * Graph inside a conditional Graph did not provide a reliable completion
     * boundary on CUDA 12.9/Blackwell; the bounded body keeps one prebuilt
     * request-Graph launch per call and introduces no steady-state allocation,
     * host polling, transfer, or synchronization. */
    const Gfn2SccLoopLaunchResult loop =
        capture_bounded_scc ? launch_gfn2_restricted_scc_loop_cuda(
                                  current.scc_binding, inference.epoch_consumer, execution_stream)
                            : current.scc_loop.launch(execution_stream);
    if (!loop.success()) {
      std::ostringstream message;
      message << "CUDA SCC loop submission failed: mode="
              << static_cast<std::uint32_t>(loop.execution_mode)
              << " status=" << static_cast<std::uint32_t>(loop.iteration.status)
              << " stage=" << static_cast<std::uint32_t>(loop.iteration.stage)
              << " cuda=" << static_cast<int>(loop.iteration.cuda_status);
      error = message.str();
      if (loop.iteration.status == Gfn2SccIterationLaunchStatus::kInvalidBinding) {
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (loop.iteration.status == Gfn2SccIterationLaunchStatus::kCublasError ||
          loop.iteration.status == Gfn2SccIterationLaunchStatus::kCusolverError) {
        return XTBLOOM_STATUS_EIGENSOLVER_FAILED;
      }
      if (loop.iteration.status == Gfn2SccIterationLaunchStatus::kProviderCaptureUnsupported) {
        return XTBLOOM_STATUS_NOT_SUPPORTED;
      }
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cuda_status = evaluate_gfn2_terminal_classical_energy_cuda(
        inference.terminal_plan, inference.terminal_activity, inference.terminal_results,
        inference.terminal_workspace, inference.terminal_diagnostics, execution_stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA terminal classical energy", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (current.host.gfn1_enabled) {
      /* The terminal composer has already published the model-aware nuclear
       * repulsion candidate into private runtime storage. Add GFN1's remaining
       * charge-independent terms there; any peer error is recorded in the
       * terminal diagnostic array consumed by final public publication. */
      cuda_status = add_gfn1_classical_corrections_cuda(
          inference.gfn1_correction_plan, current.numerical.device.committed_positions,
          current.plan_seed.geometry_cache.coordination_numbers,
          inference.terminal_activity.requested_mask, inference.terminal_results.repulsion, nullptr,
          inference.gfn1_correction_workspace, inference.terminal_diagnostics.system_errors,
          inference.terminal_diagnostics.plan_error, execution_stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA GFN1 terminal classical corrections", cuda_status);
        return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                    : XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }

    if (current.energy_force.stationary_projection.enabled == 1u) {
      cuda_status = project_gfn2_stationary_force_state_cuda(
          current.energy_force.stationary_projection, execution_stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA stationary force projection", cuda_status);
        return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                    : XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }

    cuda_status = execute_gfn2_energy_force_cuda(
        current.energy_force.plan, current.energy_force.input, current.energy_force.results,
        current.energy_force.intermediates, current.energy_force.workspace,
        current.energy_force.diagnostics, inference.epoch_consumer, execution_stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA terminal energy/force execution", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cuda_status = publish_gfn2_inference_results_cuda(
        inference.publication_plan, inference.publication_input, inference.publication_results,
        inference.publication_workspace, inference.publication_diagnostics, execution_stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA internal inference publication", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    initialize_gfn2_warm_checkpoint_batch_if_admitted_kernel<<<1, 1, 0, execution_stream>>>(
        inference.warm_checkpoint_batch_ready,
        {current.public_result.request_topology_error, 1, current.host.plan_token});
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA warm-checkpoint aggregate initialization", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    const WarmCheckpointPublicationDeviceBinding checkpoint{
        current.host.basis.batch_size,
        inference.epoch_consumer.epoch,
        inference.epoch_consumer.eligible_mask,
        inference.epoch_consumer.committed_generations,
        inference.publication_diagnostics.plan_error,
        inference.publication_results.system_statuses,
        inference.publication_results.converged,
        inference.warm_checkpoint_generations,
        inference.warm_checkpoint_batch_ready,
        current.numerical.device.committed_field_attached,
        current.numerical.device.committed_field_vectors,
        inference.warm_checkpoint_field_attached,
        inference.warm_checkpoint_field_vectors,
        current.public_result.request_topology_error,
    };
    constexpr int kThreads = 256;
    const auto blocks = static_cast<unsigned int>(
        (static_cast<std::uint64_t>(checkpoint.batch_size) + kThreads - 1u) / kThreads);
    publish_gfn2_warm_checkpoint_generation_kernel<<<blocks, kThreads, 0, execution_stream>>>(
        checkpoint);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA warm-checkpoint publication", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    current.submitted = true;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t execute_inference_body_locked(Prepared& current, std::string& error) {
    return enqueue_inference_tail_locked(current, false, stream, error);
  }

  xtbloom_status_t build_request_execution_graph(Prepared& current, std::string& error) {
    if (!current.public_result.ready || current.public_result.request_topology_error == nullptr ||
        current.inference.warm_checkpoint_generations == nullptr ||
        current.inference.warm_checkpoint_batch_ready == nullptr ||
        current.inference.request_start_mode == nullptr ||
        current.inference.admitted_warm_checkpoint_generations == nullptr) {
      error = "CUDA asynchronous request graph has an incomplete fixed binding";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    /* A bounded SCC fallback remains a valid synchronous runtime. Async
     * enqueue reports that narrower capability gap without preventing plan
     * creation or synchronous plan execution on older CUDA/provider stacks. */
    if (!current.scc_loop.conditional_graph_ready()) {
      error.clear();
      return XTBLOOM_STATUS_SUCCESS;
    }

    auto& owner = current.request_execution_graph;
    if (owner.ready()) {
      error.clear();
      return XTBLOOM_STATUS_SUCCESS;
    }
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
    g_request_graph_build_attempts.fetch_add(1u, std::memory_order_relaxed);
    if (consume_execution_test_fault(Gfn2CudaExecutionTestFault::kRequestGraphCreate)) {
      error = "injected CUDA asynchronous request graph creation failure";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
#endif
    cudaError_t cuda_status = owner.create();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA asynchronous request graph creation", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cudaStream_t capture_stream = nullptr;
    cuda_status = cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      owner.reset();
      error = cuda_error_message("CUDA asynchronous request capture stream creation", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    bool capture_active = false;
    const auto finish_capture = [&]() noexcept {
      if (capture_active) {
        cudaGraph_t ended = nullptr;
        (void)cudaStreamEndCapture(capture_stream, &ended);
      }
      (void)cudaStreamDestroy(capture_stream);
      owner.reset();
    };

    cuda_status = cudaStreamBeginCaptureToGraph(capture_stream, owner.graph(), nullptr, nullptr, 0u,
                                                cudaStreamCaptureModeThreadLocal);
    if (cuda_status != cudaSuccess) {
      finish_capture();
      error = cuda_error_message("CUDA asynchronous request root capture", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    capture_active = true;

    const RequestAdmissionDeviceBinding admission_binding{
        current.host.basis.batch_size,
        current.public_result.request_topology_error,
        current.inference.request_start_mode,
        current.inference.warm_checkpoint_generations,
        current.inference.warm_checkpoint_batch_ready,
        current.inference.admitted_warm_checkpoint_generations,
        owner.execution_condition(),
        owner.start_mode_condition(),
    };
    constexpr int kAdmissionThreads = 256;
    const auto admission_blocks = static_cast<unsigned int>(
        (static_cast<std::uint64_t>(admission_binding.batch_size) + kAdmissionThreads - 1u) /
        kAdmissionThreads);
    consume_request_checkpoint_kernel<<<admission_blocks, kAdmissionThreads, 0, capture_stream>>>(
        admission_binding);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status == cudaSuccess) {
      set_request_execution_conditions_kernel<<<1, 1, 0, capture_stream>>>(admission_binding);
      cuda_status = cudaPeekAtLastError();
    }
    if (cuda_status != cudaSuccess) {
      finish_capture();
      error = cuda_error_message("CUDA asynchronous request admission capture", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaGraph_t captured_graph = nullptr;
    const cudaGraphNode_t* dependencies = nullptr;
    std::size_t dependency_count = 0u;
    cuda_status = cudaStreamGetCaptureInfo(capture_stream, &capture_status, nullptr,
                                           &captured_graph, &dependencies, &dependency_count);
    if (cuda_status != cudaSuccess || capture_status == cudaStreamCaptureStatusNone ||
        captured_graph != owner.graph() || dependencies == nullptr || dependency_count != 1u) {
      finish_capture();
      error = "CUDA asynchronous request capture has no admission dependency frontier";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    const cudaGraphNode_t admission = dependencies[0];
    cudaGraph_t body = nullptr;
    cuda_status = owner.add_if_node(admission, body);
    cudaGraph_t ended_graph = nullptr;
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaStreamEndCapture(capture_stream, &ended_graph);
      capture_active = false;
    }
    if (cuda_status != cudaSuccess || ended_graph != owner.graph() || body == nullptr) {
      finish_capture();
      error = cuda_error_message(
          "CUDA asynchronous request conditional graph",
          cuda_status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    /* Populate the accepted body once. Numerical refresh precedes a small
     * nested start-mode switch; the expensive SCC/terminal/publication tail is
     * shared by FRESH and WARM instead of being instantiated twice. */
    cudaGraph_t* start_mode_branches = nullptr;
    cuda_status = cudaStreamBeginCaptureToGraph(capture_stream, body, nullptr, nullptr, 0u,
                                                cudaStreamCaptureModeThreadLocal);
    if (cuda_status == cudaSuccess) {
      capture_active = true;
      std::string capture_error;
      xtbloom_status_t status =
          execute_numerical_body_locked(current, capture_stream, capture_error);
      if (status == XTBLOOM_STATUS_SUCCESS) {
        cudaStreamCaptureStatus body_capture_status = cudaStreamCaptureStatusNone;
        cudaGraph_t body_capture_graph = nullptr;
        const cudaGraphNode_t* body_dependencies = nullptr;
        std::size_t body_dependency_count = 0u;
        cuda_status = cudaStreamGetCaptureInfo(capture_stream, &body_capture_status, nullptr,
                                               &body_capture_graph, &body_dependencies,
                                               &body_dependency_count);
        if (cuda_status != cudaSuccess || body_capture_status == cudaStreamCaptureStatusNone ||
            body_capture_graph != body || body_dependencies == nullptr ||
            body_dependency_count != 1u) {
          status = XTBLOOM_STATUS_INTERNAL_ERROR;
          capture_error = "CUDA request body has no numerical dependency frontier";
        } else {
          cudaGraphNode_t switch_node = nullptr;
          cuda_status = owner.add_start_mode_switch(body, body_dependencies[0], start_mode_branches,
                                                    switch_node);
          if (cuda_status == cudaSuccess) {
            cuda_status = cudaStreamUpdateCaptureDependencies(capture_stream, &switch_node, 1u,
                                                              cudaStreamSetCaptureDependencies);
          }
          if (cuda_status != cudaSuccess || start_mode_branches == nullptr) {
            status = XTBLOOM_STATUS_INTERNAL_ERROR;
            capture_error = cuda_error_message("CUDA request start-mode switch", cuda_status);
          }
        }
      }
      if (status == XTBLOOM_STATUS_SUCCESS) {
        status = enqueue_inference_tail_locked(current, true, capture_stream, capture_error);
      }
      if (status != XTBLOOM_STATUS_SUCCESS) {
        finish_capture();
        error = capture_error.empty() ? "CUDA asynchronous request body capture failed"
                                      : std::move(capture_error);
        return status;
      }
      cuda_status = cudaStreamEndCapture(capture_stream, &ended_graph);
      capture_active = false;
    }
    if (cuda_status != cudaSuccess || ended_graph != body) {
      finish_capture();
      error = cuda_error_message(
          "CUDA asynchronous request body capture",
          cuda_status == cudaSuccess ? cudaErrorStreamCaptureInvalidated : cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    /* Branch 0 is FRESH: restore the immutable device checkpoint directly.
     * The initializer validated these fixed addresses during candidate setup,
     * and its owner outlives this executable. */
    cuda_status = cudaStreamBeginCaptureToGraph(capture_stream, start_mode_branches[0], nullptr,
                                                nullptr, 0u, cudaStreamCaptureModeThreadLocal);
    if (cuda_status == cudaSuccess) {
      capture_active = true;
      cuda_status = cudaMemcpyAsync(
          current.iteration_arena.get(), current.initializer.device_checkpoint(),
          current.initializer.image_bytes(), cudaMemcpyDeviceToDevice, capture_stream);
      cudaGraph_t fresh_graph = nullptr;
      if (cuda_status == cudaSuccess) {
        cuda_status = cudaStreamEndCapture(capture_stream, &fresh_graph);
        capture_active = false;
      }
      if (cuda_status == cudaSuccess && fresh_graph != start_mode_branches[0]) {
        cuda_status = cudaErrorStreamCaptureInvalidated;
      }
    }
    if (cuda_status != cudaSuccess) {
      finish_capture();
      error = cuda_error_message("CUDA request FRESH prelude capture", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    /* Branch 1 is WARM: validate the token moved into request scratch during
     * admission, then restart only the per-attempt trace/history state. */
    cuda_status = cudaStreamBeginCaptureToGraph(capture_stream, start_mode_branches[1], nullptr,
                                                nullptr, 0u, cudaStreamCaptureModeThreadLocal);
    if (cuda_status == cudaSuccess) {
      capture_active = true;
      const WarmSccResetDeviceBinding warm{
          current.host.basis.batch_size,
          current.host.plan_token,
          current.inference.epoch_consumer.epoch,
          current.inference.epoch_consumer.eligible_mask,
          current.inference.epoch_consumer.committed_generations,
          current.numerical.device.refresh_predecessor_generations,
          current.inference.admitted_warm_checkpoint_generations,
          current.numerical.device.committed_field_attached,
          current.numerical.device.committed_field_vectors,
          current.inference.warm_checkpoint_field_attached,
          current.inference.warm_checkpoint_field_vectors,
          current.public_result.request_topology_error,
          current.public_result.request_topology_error,
          current.state_seed.mixer,
          current.state_seed.scc,
      };
      constexpr int kWarmThreads = 256;
      const auto warm_blocks = static_cast<unsigned int>(
          (static_cast<std::uint64_t>(warm.batch_size) + kWarmThreads - 1u) / kWarmThreads);
      reset_gfn2_warm_scc_trace_kernel<<<warm_blocks, kWarmThreads, 0, capture_stream>>>(warm);
      cuda_status = cudaPeekAtLastError();
      cudaGraph_t warm_graph = nullptr;
      if (cuda_status == cudaSuccess) {
        cuda_status = cudaStreamEndCapture(capture_stream, &warm_graph);
        capture_active = false;
      }
      if (cuda_status == cudaSuccess && warm_graph != start_mode_branches[1]) {
        cuda_status = cudaErrorStreamCaptureInvalidated;
      }
    }
    if (cuda_status != cudaSuccess) {
      finish_capture();
      error = cuda_error_message("CUDA request WARM prelude capture", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

#if defined(XTBLOOM_CUDA_TEST_HOOKS)
    if (consume_execution_test_fault(Gfn2CudaExecutionTestFault::kRequestGraphInstantiate)) {
      cuda_status = cudaErrorUnknown;
    } else
#endif
    {
      cuda_status = owner.instantiate();
    }
    if (cuda_status == cudaSuccess) {
      /* Conditional Graph launch may otherwise perform a lazy upload on the
       * caller stream. That can make enqueue synchronously wait behind work
       * already ordered on the stream, violating the request API's admission
       * contract. Finish the one-time upload on the private setup stream. */
      cuda_status = owner.upload(capture_stream);
    }
    if (cuda_status == cudaSuccess) cuda_status = cudaStreamSynchronize(capture_stream);
    (void)cudaStreamDestroy(capture_stream);
    if (cuda_status != cudaSuccess) {
      owner.reset();
      error =
          cuda_error_message("CUDA asynchronous request graph instantiation/upload", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
    g_request_graph_build_successes.fetch_add(1u, std::memory_order_relaxed);
#endif
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t ensure_request_execution_graph_locked(Prepared& current, std::string& error) {
    return current.request_execution_graph.ready() ? XTBLOOM_STATUS_SUCCESS
                                                   : build_request_execution_graph(current, error);
  }

  xtbloom_status_t launch_request_execution_graph_locked(Prepared& current,
                                                         Gfn2CudaSccStartMode mode,
                                                         std::string& error) {
    if (mode != Gfn2CudaSccStartMode::kFresh && mode != Gfn2CudaSccStartMode::kWarm) {
      error = "CUDA asynchronous request received an unknown SCC start mode";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const RequestExecutionGraphOwner& owner = current.request_execution_graph;
    if (!owner.ready()) {
      error = "CUDA asynchronous request execution graph is not initialized";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    stage_request_start_mode_kernel<<<1, 1, 0, stream>>>(
        mode == Gfn2CudaSccStartMode::kWarm ? 1u : 0u, current.inference.request_start_mode);
    cudaError_t cuda_status = cudaPeekAtLastError();
    if (cuda_status == cudaSuccess) current.submitted = true;
    if (cuda_status == cudaSuccess) cuda_status = owner.launch(stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA asynchronous request graph launch", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.inference.warm_checkpoint_ready = false;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t refresh_numerical_locked(Prepared& current,
                                            const Gfn2CudaNumericalInputView& input,
                                            bool strict_warm,
                                            bool allow_blocking_interaction_readback,
                                            std::string& error) {
    bool host_upload_enqueued = false;
    xtbloom_status_t status = stage_numerical_ingress_locked(current, input, strict_warm,
                                                             allow_blocking_interaction_readback,
                                                             host_upload_enqueued, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = execute_numerical_body_locked(current, stream, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    if (host_upload_enqueued) {
      /* Make owner-stream completion prove that the private callback has
       * released the pinned snapshot. This preserves allocation-free,
       * no-polling steady state while preventing a completed refresh from
       * spuriously rejecting the next host-backed refresh. */
      const cudaError_t release_status =
          cudaStreamWaitEvent(stream, current.numerical_host_release_complete.get(), 0u);
      if (release_status != cudaSuccess) {
        current.numerical.host_staging_poisoned = true;
        error = cuda_error_message("CUDA numerical host-release tail ordering", release_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      current.submitted = true;
    }
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t execute_inference_locked(Prepared& current, Gfn2CudaSccStartMode mode,
                                            std::string& error) {
    if (!current.inference.ready || !current.numerical.ready) {
      error = "CUDA GFN2 inference requires a prepared numerical/runtime binding";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (current.numerical.device.refresh_predecessor_generations == nullptr ||
        current.inference.warm_checkpoint_generations == nullptr ||
        current.inference.warm_checkpoint_batch_ready == nullptr) {
      error = "CUDA GFN2 inference has an incomplete warm-checkpoint binding";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (mode != Gfn2CudaSccStartMode::kFresh && mode != Gfn2CudaSccStartMode::kWarm) {
      error = "CUDA GFN2 inference received an unknown SCC start mode";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    auto& inference = current.inference;
    if (mode == Gfn2CudaSccStartMode::kWarm && !inference.warm_checkpoint_ready) {
      error = "CUDA GFN2 warm inference requires a previously submitted checkpoint";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for CUDA GFN2 inference", cuda_status);
      return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    }

    if (mode == Gfn2CudaSccStartMode::kFresh) {
      /* A fresh attempt consumes every old checkpoint before any state image
       * is restored. If a later enqueue fails, no stale warm token can survive
       * and masquerade as the failed attempt's checkpoint. */
      const auto checkpoint_elements = current.host.basis.batch_size;
      constexpr int kInvalidateThreads = 256;
      const auto invalidate_blocks = static_cast<unsigned int>(
          (static_cast<std::uint64_t>(checkpoint_elements) + kInvalidateThreads - 1u) /
          kInvalidateThreads);
      invalidate_gfn2_warm_checkpoint_if_admitted_kernel<<<invalidate_blocks, kInvalidateThreads, 0,
                                                           stream>>>(
          inference.warm_checkpoint_generations, checkpoint_elements,
          current.public_result.request_topology_error);
      cuda_status = cudaPeekAtLastError();
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA fresh warm-checkpoint invalidation", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      current.submitted = true;
      inference.warm_checkpoint_ready = false;
      const auto diagnostic = current.initializer.upload_if_admitted_async(
          current.iteration_arena.get(), current.iteration_arena.bytes(), current.ready,
          current.public_result.request_topology_error, stream);
      if (!diagnostic.success()) {
        error = setup_error_message("CUDA SCC fresh-state restore", diagnostic.status,
                                    static_cast<std::uint32_t>(diagnostic.error),
                                    static_cast<std::uint32_t>(diagnostic.field), diagnostic.index);
        return diagnostic.status;
      }
      current.submitted = true;
    } else {
      const WarmSccResetDeviceBinding warm{
          current.host.basis.batch_size,
          current.host.plan_token,
          inference.epoch_consumer.epoch,
          inference.epoch_consumer.eligible_mask,
          inference.epoch_consumer.committed_generations,
          current.numerical.device.refresh_predecessor_generations,
          inference.warm_checkpoint_generations,
          current.numerical.device.committed_field_attached,
          current.numerical.device.committed_field_vectors,
          inference.warm_checkpoint_field_attached,
          inference.warm_checkpoint_field_vectors,
          current.public_result.request_topology_error,
          current.public_result.request_topology_error,
          current.state_seed.mixer,
          current.state_seed.scc,
      };
      constexpr int kThreads = 256;
      const auto blocks = static_cast<unsigned int>(
          (static_cast<std::uint64_t>(warm.batch_size) + kThreads - 1u) / kThreads);
      reset_gfn2_warm_scc_trace_kernel<<<blocks, kThreads, 0, stream>>>(warm);
      cuda_status = cudaPeekAtLastError();
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA warm SCC trace reset", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      current.submitted = true;
      inference.warm_checkpoint_ready = false;
    }

    const xtbloom_status_t body_status = execute_inference_body_locked(current, error);
    if (body_status != XTBLOOM_STATUS_SUCCESS) return body_status;
    inference.warm_checkpoint_ready = true;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  /*
   * A synchronous public call must not return while either the owner stream
   * can still read caller CUDA pointers or the private host-upload callback
   * can still access Prepared-owned lease state.  Failure paths use the same
   * disable-timing event as successful publication and fall back to an exact
   * owner-stream fence if recording that event itself fails.
   */
  xtbloom_status_t settle_public_submissions_locked(Prepared& current,
                                                    xtbloom_status_t primary_status,
                                                    std::string& error) {
    cudaError_t owner_status = cudaSuccess;
    if (current.submitted) {
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      if (consume_execution_test_fault(Gfn2CudaExecutionTestFault::kRequestSettlement)) {
        owner_status = cudaErrorUnknown;
      } else
#endif
      {
        owner_status = cudaEventRecord(current.public_result_completion_event.get(), stream);
        if (owner_status == cudaSuccess) {
          owner_status = cudaEventSynchronize(current.public_result_completion_event.get());
        }
        if (owner_status != cudaSuccess) {
          const cudaError_t fallback_status = cudaStreamSynchronize(stream);
          if (fallback_status == cudaSuccess) owner_status = cudaSuccess;
        }
        if (owner_status == cudaSuccess) current.submitted = false;
      }
    }

    cudaError_t host_status = cudaSuccess;
    if (current.numerical_host_completion_stream.valid()) {
      host_status = cudaStreamSynchronize(current.numerical_host_completion_stream.get());
      if (host_status == cudaSuccess) {
        /* The stream fence is after the release callback, so this store cannot
         * race a previous lease generation or clear a future one. */
        current.numerical_host_upload_completion.pending.store(false, std::memory_order_release);
      } else {
        current.numerical.host_staging_poisoned = true;
      }
    }
    if (owner_status == cudaSuccess && host_status == cudaSuccess) return primary_status;

    std::ostringstream message;
    if (!error.empty()) message << error << "; additionally, ";
    message << "CUDA public transaction settlement failed";
    if (owner_status != cudaSuccess) {
      message << " owner_stream=" << cudaGetErrorString(owner_status);
    }
    if (host_status != cudaSuccess) {
      message << " host_completion_stream=" << cudaGetErrorString(host_status);
    }
    error = message.str();
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  xtbloom_status_t enqueue_public_result_prepare_locked(
      Prepared& current, const xtbloom_batch_t& batch, const xtbloom_compute_options_t& options,
      xtbloom_batch_result_t& result, PublicResultTransaction& transaction, std::string& error) {
    const bool prior_warm_checkpoint_ready = transaction.prior_warm_checkpoint_ready;
    transaction = {};
    transaction.prior_warm_checkpoint_ready = prior_warm_checkpoint_ready;
    if (!current.public_result.ready || !current.inference.ready ||
        current.public_result_completion_event.get() == nullptr) {
      error = "CUDA public result bridge requires a complete prepared runtime";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const std::int64_t batch_size = current.host.basis.batch_size;
    const std::int64_t atoms = current.host.basis.total_atoms;
    const std::int64_t points = current.host.external.total_point_charges;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates)) {
      error = "CUDA public result extent overflows int64_t";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const std::uint64_t token = current.host.plan_token;
    const bool external_operator = batch.atomic_potential_shifts.data != nullptr ||
                                   batch.atomic_potential_shifts.size_bytes != 0u ||
                                   batch.charge_response_matrix.data != nullptr ||
                                   batch.charge_response_matrix.size_bytes != 0u;
    const std::uint32_t result_flags =
        (external_operator ? static_cast<std::uint32_t>(
                                 XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES)
                           : 0u) |
        (((options.flags & XTBLOOM_COMPUTE_DIPOLE_MOMENTS) != 0u)
             ? static_cast<std::uint32_t>(XTBLOOM_RESULT_DIPOLE_MOMENTS)
             : 0u);
    const Gfn2PublicResultBridgeDevicePlan plan{
        kGfn2PublicResultBridgeAbiVersion,
        options.flags,
        result_flags,
        0u,
        token,
        batch_size,
        atoms,
        points,
    };
    const auto& inference = current.inference;
    Gfn2PublicResultBridgeDeviceInput input{};
    input.energies = inference.publication_results.energies;
    input.energy_elements = inference.publication_results.energy_elements;
    input.qm_forces = inference.publication_results.qm_forces;
    input.qm_force_elements = inference.publication_results.qm_force_elements;
    input.atomic_charges = inference.publication_results.atomic_charges;
    input.atomic_charge_elements = inference.publication_results.atomic_charge_elements;
    input.point_forces = inference.publication_results.point_forces;
    input.point_force_elements = inference.publication_results.point_force_elements;
    input.iterations = inference.publication_results.iterations;
    input.converged = inference.publication_results.converged;
    input.system_statuses = inference.publication_results.system_statuses;
    input.batch_elements = inference.publication_results.batch_elements;
    input.publication_plan_error = inference.publication_diagnostics.plan_error;
    input.request_topology_error = current.public_result.request_topology_error;
    input.publication_epoch_snapshot = inference.publication_workspace.epoch_snapshot;
    input.current_geometry_epoch = current.numerical.preprocessing.geometry_epoch.value;
    input.plan_token = token;
    input.dipole_moments = inference.publication_results.dipole_moments;
    input.dipole_moment_elements = inference.publication_results.dipole_moment_elements;

    Gfn2PublicResultBridgeDeviceDestinations destinations{};
    Gfn2PublicResultBridgeHostStaging staging{};
    const auto bind = [](const xtbloom_buffer_t& output, std::int64_t elements, void* host_stage,
                         Gfn2PublicResultBridgeDestination& destination,
                         Gfn2PublicResultBridgeHostBuffer& host) {
      if (elements == 0) {
        destination = {};
        host = {};
        return;
      }
      const bool host_route = output.memory_space == XTBLOOM_MEMORY_HOST;
      destination.route =
          host_route ? Gfn2PublicResultRoute::kHost : Gfn2PublicResultRoute::kCudaDevice;
      destination.device_data = host_route ? nullptr : output.data;
      destination.elements = elements;
      host.data = host_route ? host_stage : nullptr;
      host.elements = host_route ? elements : 0;
    };
    const bool energy_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    const bool dipoles_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_DIPOLE_MOMENTS)) != 0u;
    const xtbloom_buffer_t absent_output{};
    const auto& dipole_output =
        result.struct_size >= XTBLOOM_BATCH_RESULT_V2_SIZE ? result.dipole_moments : absent_output;
    auto& public_state = current.public_result;
    bind(result.energies, energy_requested ? batch_size : 0, public_state.energies,
         destinations.energies, staging.energies);
    bind(result.forces, force_requested ? coordinates : 0, public_state.qm_forces,
         destinations.qm_forces, staging.qm_forces);
    bind(result.atomic_charges, charges_requested ? atoms : 0, public_state.atomic_charges,
         destinations.atomic_charges, staging.atomic_charges);
    bind(result.point_charge_forces, point_forces_requested ? point_coordinates : 0,
         public_state.point_forces, destinations.point_forces, staging.point_forces);
    bind(dipole_output, dipoles_requested ? 3 * batch_size : 0, public_state.dipole_moments,
         destinations.dipole_moments, staging.dipole_moments);
    bind(result.scc_iterations, batch_size, public_state.iterations, destinations.iterations,
         staging.iterations);
    bind(result.scc_converged, batch_size, public_state.converged, destinations.converged,
         staging.converged);
    bind(result.per_system_status, batch_size, public_state.system_statuses,
         destinations.system_statuses, staging.system_statuses);
    destinations.plan_token = token;
    staging.control = public_state.host_control;
    staging.control_elements = 1;
    staging.pending_result_flags = &public_state.pending_result_flags;
    staging.plan_token = token;

    cudaError_t cuda_status =
        prepare_gfn2_public_results_cuda(plan, input, public_state.device_staging, destinations,
                                         staging, public_state.diagnostics, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA public result bridge submission", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    cuda_status =
        cudaMemcpyAsync(public_state.warm_checkpoint_ready, inference.warm_checkpoint_batch_ready,
                        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA warm-checkpoint readiness download", cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    transaction.plan = plan;
    transaction.destinations = destinations;
    transaction.phase = PublicResultTransactionPhase::kPrepareSubmitted;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  /*
   * Append one completion marker after a submitted public bridge phase. The
   * commit phase retains the historical stream-fence fallback when recording
   * the event itself fails; prepare keeps its existing strict event behavior.
   */
  xtbloom_status_t record_public_result_completion_locked(
      Prepared& current, PublicResultTransaction& transaction,
      PublicResultTransactionPhase submitted_phase, PublicResultTransactionPhase recorded_phase,
      PublicResultTransactionPhase completed_phase, bool fallback_to_stream,
      const char* failure_context, std::string& error) {
    if (transaction.phase != submitted_phase ||
        current.public_result_completion_event.get() == nullptr) {
      error = "CUDA public result completion marker has an invalid transaction phase";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    const cudaError_t record_status =
        cudaEventRecord(current.public_result_completion_event.get(), stream);
    if (record_status == cudaSuccess) {
      transaction.phase = recorded_phase;
      return XTBLOOM_STATUS_SUCCESS;
    }
    if (fallback_to_stream && cudaStreamSynchronize(stream) == cudaSuccess) {
      current.submitted = false;
      transaction.phase = completed_phase;
      return XTBLOOM_STATUS_SUCCESS;
    }

    error = cuda_error_message(failure_context, record_status);
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  /* Completion observation never publishes host bytes or result.flags. */
  xtbloom_status_t wait_public_result_completion_locked(
      Prepared& current, PublicResultTransaction& transaction,
      PublicResultTransactionPhase recorded_phase, PublicResultTransactionPhase completed_phase,
      const char* failure_context, std::string& error) {
    if (transaction.phase == completed_phase) return XTBLOOM_STATUS_SUCCESS;
    if (transaction.phase != recorded_phase ||
        current.public_result_completion_event.get() == nullptr) {
      error = "CUDA public result completion wait has an invalid transaction phase";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    const cudaError_t cuda_status =
        cudaEventSynchronize(current.public_result_completion_event.get());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message(failure_context, cuda_status);
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    current.submitted = false;
    transaction.phase = completed_phase;
    return XTBLOOM_STATUS_SUCCESS;
  }

  /* Read the pinned aggregate only after prepare completion is observed. */
  xtbloom_status_t accept_public_result_prepare_locked(Prepared& current,
                                                       PublicResultTransaction& transaction,
                                                       std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kPrepareCompleted) {
      error = "CUDA public result prepare acceptance precedes completion";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const auto& public_state = current.public_result;
    const auto aggregate =
        static_cast<Gfn2PublicResultBridgeError>(public_state.host_control->aggregate_error);
    if (aggregate != Gfn2PublicResultBridgeError::kSuccess) {
      if (aggregate == Gfn2PublicResultBridgeError::kRequestTopologyMismatch) {
        error = "the batch topology does not match the fixed CUDA plan topology";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestInvalidArgument) {
        error = "CUDA stream-ordered request validation rejected an invalid descriptor";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotSupported) {
        error = "CUDA stream-ordered request validation rejected unsupported periodic axes";
        return XTBLOOM_STATUS_NOT_SUPPORTED;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotImplemented) {
        error = "the CUDA request uses a validated feature that is not implemented yet";
        return XTBLOOM_STATUS_NOT_IMPLEMENTED;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestWarmIncompatible) {
        error = "CUDA strict WARM SCC start requires an identical electric-field attachment";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      std::ostringstream message;
      message << "CUDA public result bridge rejected inference: aggregate_error="
              << static_cast<std::uint32_t>(aggregate) << " publication_plan_error="
              << public_state.host_control->internal_publication_plan_error
              << " publication_epoch=" << public_state.host_control->publication_epoch_snapshot
              << " current_epoch=" << public_state.host_control->current_geometry_epoch;
      error = message.str();
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    transaction.phase = PublicResultTransactionPhase::kPrepareAccepted;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  /* The asynchronous path submits the device-gated commit before returning,
   * so its host acceptance occurs only after that commit event completes. The
   * device kernel itself reads the same aggregate control and is a no-op on a
   * rejected prepare image. */
  xtbloom_status_t accept_public_result_after_commit_locked(Prepared& current,
                                                            PublicResultTransaction& transaction,
                                                            std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kCommitCompleted) {
      error = "CUDA public result acceptance precedes asynchronous commit completion";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const auto& public_state = current.public_result;
    const auto aggregate =
        static_cast<Gfn2PublicResultBridgeError>(public_state.host_control->aggregate_error);
    if (aggregate != Gfn2PublicResultBridgeError::kSuccess) {
      if (aggregate == Gfn2PublicResultBridgeError::kRequestTopologyMismatch) {
        error = "the batch topology does not match the fixed CUDA plan topology";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestInvalidArgument) {
        error = "CUDA stream-ordered request validation rejected an invalid descriptor";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotSupported) {
        error = "CUDA stream-ordered request validation rejected unsupported periodic axes";
        return XTBLOOM_STATUS_NOT_SUPPORTED;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotImplemented) {
        error = "the CUDA request uses a validated feature that is not implemented yet";
        return XTBLOOM_STATUS_NOT_IMPLEMENTED;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestWarmIncompatible) {
        error = "CUDA strict WARM SCC start requires an identical electric-field attachment";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      std::ostringstream message;
      message << "CUDA public result bridge rejected inference: aggregate_error="
              << static_cast<std::uint32_t>(aggregate) << " publication_plan_error="
              << public_state.host_control->internal_publication_plan_error
              << " publication_epoch=" << public_state.host_control->publication_epoch_snapshot
              << " current_epoch=" << public_state.host_control->current_geometry_epoch;
      error = message.str();
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  xtbloom_status_t enqueue_public_result_commit_locked(Prepared& current,
                                                       PublicResultTransaction& transaction,
                                                       bool allow_device_gated_prepare,
                                                       std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kPrepareAccepted &&
        !(allow_device_gated_prepare &&
          transaction.phase == PublicResultTransactionPhase::kPrepareSubmitted)) {
      error = "CUDA public result commit has no accepted prepared image";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const cudaError_t cuda_status = commit_gfn2_public_results_cuda(
        transaction.plan, current.public_result.device_staging, transaction.destinations,
        current.public_result.diagnostics, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA caller-device result commit submission", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? XTBLOOM_STATUS_INVALID_ARGUMENT
                                                  : XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    /* From this point onward caller CUDA bytes may be modified. No later
     * failure is recoverable by pretending that the transaction rolled back. */
    current.submitted = true;
    transaction.phase = PublicResultTransactionPhase::kCommitSubmitted;
    return XTBLOOM_STATUS_SUCCESS;
  }

  /* Host result publication is a separate, non-CUDA phase after commit wait. */
  xtbloom_status_t publish_public_results_locked(
      Prepared& current, const xtbloom_compute_options_t& options, xtbloom_batch_result_t& result,
      PublicResultTransaction& transaction, bool commit_host_outputs, bool publish_descriptor_flags,
      std::uint32_t& completed_result_flags, std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kCommitCompleted) {
      error = "CUDA public result host publication precedes commit completion";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (!commit_host_outputs) return XTBLOOM_STATUS_INTERNAL_ERROR;

    const std::int64_t batch_size = current.host.basis.batch_size;
    const std::int64_t atoms = current.host.basis.total_atoms;
    const std::int64_t points = current.host.external.total_point_charges;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates)) {
      error = "CUDA public result extent changed after commit acceptance";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const bool energy_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    const bool dipoles_requested =
        (options.flags & static_cast<std::uint32_t>(XTBLOOM_COMPUTE_DIPOLE_MOMENTS)) != 0u;
    const xtbloom_buffer_t absent_output{};
    const auto& dipole_output =
        result.struct_size >= XTBLOOM_BATCH_RESULT_V2_SIZE ? result.dipole_moments : absent_output;
    auto& public_state = current.public_result;
    const auto commit_host = [](const xtbloom_buffer_t& output, const void* staging_data,
                                std::int64_t elements, std::size_t element_size) {
      if (elements != 0 && output.memory_space == XTBLOOM_MEMORY_HOST) {
        std::memcpy(output.data, staging_data, static_cast<std::size_t>(elements) * element_size);
      }
    };
    commit_host(result.energies, public_state.energies, energy_requested ? batch_size : 0,
                sizeof(double));
    commit_host(result.forces, public_state.qm_forces, force_requested ? coordinates : 0,
                sizeof(double));
    commit_host(result.atomic_charges, public_state.atomic_charges, charges_requested ? atoms : 0,
                sizeof(double));
    commit_host(result.point_charge_forces, public_state.point_forces,
                point_forces_requested ? point_coordinates : 0, sizeof(double));
    commit_host(dipole_output, public_state.dipole_moments, dipoles_requested ? 3 * batch_size : 0,
                sizeof(double));
    commit_host(result.scc_iterations, public_state.iterations, batch_size, sizeof(std::int32_t));
    commit_host(result.scc_converged, public_state.converged, batch_size, sizeof(std::uint8_t));
    commit_host(result.per_system_status, public_state.system_statuses, batch_size,
                sizeof(xtbloom_status_t));
    completed_result_flags = public_state.pending_result_flags;
    if (publish_descriptor_flags) result.flags = completed_result_flags;
    current.inference.warm_checkpoint_ready = *public_state.warm_checkpoint_ready != 0u;
    transaction.phase = PublicResultTransactionPhase::kPublished;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }

  /* Complete the one request-owned publication transaction in a single place.
   * In particular, a deferred stream-ordered failure must not leave the host
   * checkpoint-ready bit resurrected by an otherwise successful D2H result
   * bridge. The device token was already consumed by FRESH/WARM admission, so
   * keeping the host bit false makes every later strict WARM reject safely. */
  void finalize_active_request_after_commit_locked(Prepared& current) {
    auto& active = active_request;
    if (active.deferred_status != XTBLOOM_STATUS_SUCCESS) {
      active.completion_status = active.deferred_status;
      active.result_flags = 0u;
      /* The device-gated commit is already ordered, so caller CUDA buffers may
       * have changed. Host publication remains controllable: close readiness
       * before diagnostic composition and do not copy staged host bytes for a
       * request whose accepted transaction already has a deferred failure. */
      current.inference.warm_checkpoint_ready = false;
      if (!active.completion_error.empty()) active.completion_error += "; additionally, ";
      active.completion_error += active.deferred_error;
      active.completion_ready = true;
      return;
    }

    xtbloom_status_t status = accept_public_result_after_commit_locked(current, active.transaction,
                                                                       active.completion_error);
    if (status == XTBLOOM_STATUS_SUCCESS) {
      status =
          publish_public_results_locked(current, active.options, active.result, active.transaction,
                                        true, false, active.result_flags, active.completion_error);
    }
    active.completion_status = status;
    if (status != XTBLOOM_STATUS_SUCCESS) {
      active.result_flags = 0u;
      current.inference.warm_checkpoint_ready = request_semantic_rejection(current.public_result)
                                                    ? active.transaction.prior_warm_checkpoint_ready
                                                    : false;
    }
    active.completion_ready = true;
  }

  Gfn2CudaExecutionIdentity snapshot() const noexcept {
    Gfn2CudaExecutionIdentity identity{};
    if (prepared == nullptr) return identity;
    const Prepared& current = *prepared;
    identity.topology_fingerprint = current.host.fingerprint;
    identity.plan_token = current.host.plan_token;
    identity.iteration_layout_fingerprint = current.iteration_requirements.layout_fingerprint;
    identity.enabled_component_mask = current.plan_seed.enabled_components;
    identity.scc_binding_ready = current.scc_binding.plan.plan_token == current.host.plan_token;
    identity.energy_force_binding_ready =
        current.energy_force.plan.plan_token == current.host.plan_token &&
        current.energy_force.input.plan_token == current.host.plan_token &&
        current.energy_force.results.plan_token == current.host.plan_token &&
        current.energy_force.intermediates.plan_token == current.host.plan_token &&
        current.energy_force.workspace.plan_token == current.host.plan_token &&
        current.energy_force.diagnostics.plan_token == current.host.plan_token;
    identity.force_mode_ready = current.energy_force.plan.compute_forces == 1u;
    identity.energy_force_smoke_ready = current.energy_force_smoke_ready ? 1u : 0u;
    identity.numerical_refresh_ready = current.numerical.ready ? 1u : 0u;
    identity.inference_ready =
        current.inference.ready &&
                current.inference.terminal_plan.plan_token == current.host.plan_token &&
                current.inference.publication_plan.plan_token == current.host.plan_token &&
                current.inference.publication_results.plan_token == current.host.plan_token
            ? 1u
            : 0u;
    identity.warm_checkpoint_ready = current.inference.warm_checkpoint_ready ? 1u : 0u;
    identity.scc_conditional_graph_ready = current.scc_loop.conditional_graph_ready() ? 1u : 0u;
    identity.scc_loop_fallback_reason =
        static_cast<std::uint32_t>(current.scc_loop.fallback_reason());
    identity.native_lattice_enabled = current.host.key.native_lattice_enabled ? 1u : 0u;
    identity.batch_size = current.host.basis.batch_size;
    identity.total_atoms = current.host.basis.total_atoms;
    identity.total_shells = current.host.basis.total_shells;
    identity.total_orbitals = current.host.basis.total_orbitals;
    identity.total_point_charges = current.host.external.total_point_charges;
    identity.native_lattice_systems = current.host.basis.batch_size;
    identity.solver_handle = opaque_address(solver);
    identity.solver_parameters = opaque_address(solver_parameters);
    identity.blas_handle = opaque_address(blas);
    identity.topology_owner = opaque_address(&current.topology_owner);
    identity.inputs_owner = opaque_address(&current.inputs_owner);
    identity.eigensolver_owner = opaque_address(&current.eigensolver_owner);
    identity.initializer_owner = opaque_address(&current.initializer);
    identity.scc_binding = opaque_address(&current.scc_binding);
    identity.scc_state_iterations = opaque_address(current.state_seed.scc.iterations);
    identity.scc_state_converged = opaque_address(current.state_seed.scc.converged);
    identity.scc_state_system_statuses = opaque_address(current.state_seed.scc.system_statuses);
    identity.scc_loop_owner = opaque_address(&current.scc_loop);
    identity.scc_loop_active_count =
        opaque_address(current.scc_loop.canonical_active_count_device());
    identity.scc_loop_numerical_body_count =
        opaque_address(current.scc_loop.numerical_body_count_device());
    identity.scc_loop_device_launch_error =
        opaque_address(current.scc_loop.device_launch_error_device());
    identity.energy_force_descriptors = opaque_address(&current.energy_force);
    identity.request_submissions = request_submissions;
    identity.request_active = active_request.id == 0u ? 0u : 1u;
    identity.topology_arena = opaque_address(current.topology_arena.get());
    identity.input_arena = opaque_address(current.input_arena.get());
    identity.iteration_arena = opaque_address(current.iteration_arena.get());
    identity.eigensolver_setup_arena = opaque_address(current.eigensolver_setup_arena.get());
    identity.provider_host_workspace = opaque_address(current.provider_host_workspace.get());
    identity.integral_task_arena = opaque_address(current.integral_task_arena.get());
    identity.force_immutable_arena = opaque_address(current.force_immutable_arena.get());
    identity.force_execution_arena = opaque_address(current.force_execution_arena.get());
    identity.numerical_refresh_arena = opaque_address(current.numerical_refresh_arena.get());
    identity.interaction_device_staging_arena =
        opaque_address(current.interaction_device_staging_arena.get());
    identity.interaction_host_staging_arena =
        opaque_address(current.interaction_host_staging_arena.get());
    identity.numerical_refresh_binding = opaque_address(&current.numerical.preprocessing);
    identity.numerical_epoch = opaque_address(current.numerical.preprocessing.geometry_epoch.value);
    identity.committed_generations = opaque_address(current.numerical.device.committed_generations);
    identity.numerical_eligible_mask = opaque_address(current.numerical.device.eligible);
    identity.overlap_factor_generations =
        opaque_address(current.eigensolver_binding.cache.geometry_generations);
    identity.overlap_factor_statuses =
        opaque_address(current.eigensolver_binding.cache.factor_statuses);
    const auto opaque_buffer = [](const auto* address, std::int64_t elements) noexcept {
      return Gfn2CudaOpaqueBufferIdentity{opaque_address(address), elements};
    };
    const auto& numerical = current.numerical.device;
    identity.committed_positions =
        opaque_buffer(numerical.committed_positions, numerical.total_atoms * 3);
    identity.committed_geometry_pairs =
        opaque_buffer(numerical.public_geometry_pairs, numerical.geometry_pair_elements);
    identity.committed_coordination_numbers =
        opaque_buffer(numerical.public_coordination, numerical.total_atoms);
    identity.committed_overlap = opaque_buffer(numerical.public_overlap, numerical.total_matrices);
    identity.committed_dipole_integrals =
        opaque_buffer(numerical.public_dipole, numerical.total_matrices * 3);
    identity.committed_quadrupole_integrals =
        opaque_buffer(numerical.public_quadrupole, numerical.total_matrices * 6);
    identity.committed_h0 = opaque_buffer(numerical.public_h0, numerical.total_matrices);
    identity.committed_es2 = opaque_buffer(numerical.public_es2, numerical.es2_elements);
    identity.committed_aes2 = opaque_buffer(numerical.public_aes2, numerical.aes2_elements);
    /* Production D4 consumes the committed physical pair-list superset and no
     * longer retains a dense five-double pair-value cache. Keep the diagnostic
     * leaf canonically empty so memory evidence can detect a regression. */
    identity.committed_d4_pairs = {};
    identity.committed_d4_coordination_numbers = opaque_buffer(
        numerical.public_d4_coordination, numerical.d4_enabled != 0u ? numerical.total_atoms : 0);
    identity.committed_point_charge_positions =
        opaque_buffer(numerical.committed_point_positions,
                      numerical.point_enabled != 0u ? numerical.total_point_charges * 3 : 0);
    identity.committed_point_charge_values =
        opaque_buffer(numerical.committed_point_values,
                      numerical.point_enabled != 0u ? numerical.total_point_charges : 0);
    identity.committed_point_charge_gammas =
        opaque_buffer(numerical.committed_point_gammas,
                      numerical.point_enabled != 0u ? numerical.total_point_charges : 0);
    identity.committed_point_charge_shell_potential = opaque_buffer(
        numerical.public_point_shell, numerical.point_enabled != 0u ? numerical.total_shells : 0);
    identity.committed_periodic_shifts =
        opaque_buffer(numerical.committed_periodic_shifts,
                      numerical.periodic_enabled != 0u ? numerical.total_atoms : 0);
    identity.committed_periodic_response =
        opaque_buffer(numerical.committed_periodic_response,
                      numerical.periodic_enabled != 0u ? numerical.total_response_elements : 0);
    identity.committed_native_cell_matrices =
        opaque_buffer(current.public_result.expected_lattice_cells,
                      static_cast<std::int64_t>(current.host.key.cell_matrices.size()));
    identity.committed_native_periodic_axes =
        opaque_buffer(current.public_result.expected_periodic_axes,
                      static_cast<std::int64_t>(current.host.key.periodic_axes.size()));
    identity.committed_generation_elements = numerical.batch_size;
    identity.numerical_eligible_elements = numerical.batch_size;
    identity.overlap_factor_generation_elements = numerical.batch_size;
    identity.overlap_factor_status_elements = numerical.batch_size;
    identity.inference_arena = opaque_address(current.inference_arena.get());
    identity.inference_epoch_consumer = opaque_address(&current.inference.epoch_consumer);
    identity.inference_results = opaque_address(&current.inference.publication_results);
    identity.inference_energies = opaque_address(current.inference.publication_results.energies);
    identity.inference_qm_forces = opaque_address(current.inference.publication_results.qm_forces);
    identity.inference_atomic_charges =
        opaque_address(current.inference.publication_results.atomic_charges);
    identity.inference_point_forces =
        opaque_address(current.inference.publication_results.point_forces);
    identity.inference_iterations =
        opaque_address(current.inference.publication_results.iterations);
    identity.inference_converged = opaque_address(current.inference.publication_results.converged);
    identity.inference_system_statuses =
        opaque_address(current.inference.publication_results.system_statuses);
    identity.inference_publication_epoch_snapshot =
        opaque_address(current.inference.publication_workspace.epoch_snapshot);
    identity.inference_publication_system_errors =
        opaque_address(current.inference.publication_diagnostics.system_errors);
    identity.inference_publication_plan_error =
        opaque_address(current.inference.publication_diagnostics.plan_error);
    identity.warm_checkpoint_generations =
        opaque_address(current.inference.warm_checkpoint_generations);
    identity.topology_arena_bytes = current.topology_arena.bytes();
    identity.input_arena_bytes = current.input_arena.bytes();
    identity.iteration_arena_bytes = current.iteration_arena.bytes();
    identity.eigensolver_setup_arena_bytes = current.eigensolver_setup_arena.bytes();
    identity.provider_host_workspace_bytes = current.provider_host_workspace.bytes();
    identity.integral_task_arena_bytes = current.integral_task_arena.bytes();
    identity.force_immutable_arena_bytes = current.force_immutable_arena.bytes();
    identity.force_execution_arena_bytes = current.force_execution_arena.bytes();
    identity.numerical_refresh_arena_bytes = current.numerical_refresh_arena.bytes();
    identity.inference_arena_bytes = current.inference_arena.bytes();
    identity.numerical_host_staging_arena_bytes = current.numerical_host_staging_arena.bytes();
    identity.numerical_host_staging_arena_bytes += current.interaction_host_staging_arena.bytes();
    identity.interaction_device_staging_arena_bytes =
        current.interaction_device_staging_arena.bytes();
    identity.interaction_descriptor_capacity_bytes =
        current.numerical.interaction_descriptor_capacity_bytes;
    identity.interaction_payload_capacity_bytes =
        current.numerical.interaction_payload_capacity_bytes;
    identity.public_result_device_arena_bytes = current.public_result_device_arena.bytes();
    identity.public_result_host_arena_bytes = current.public_result_host_arena.bytes();
    identity.candidate_validation_arena_bytes = current.candidate_validation_arena.bytes();
    identity.native_lattice_host_staging = opaque_address(native_lattice_host_arena.get());
    identity.native_lattice_host_staging_bytes = native_lattice_host_arena.bytes();
    const Gfn2CudaTopologyStagingWorkspaceBytes topology_workspace =
        topology_staging.workspace_bytes();
    identity.topology_staging_host_bytes = topology_workspace.host_bytes;
    identity.topology_staging_device_bytes = topology_workspace.device_bytes;
    /* Cache/prepared records contain the inline host plans, arena owners, and
     * descriptor metadata. Their nested heap payloads are counted below. */
    identity.runtime_owner_host_bytes = sizeof(Impl) + sizeof(current);
    const HostPlanByteBreakdown host_plan_bytes = current.host.retained_host_byte_breakdown();
    identity.host_plans_bytes = current.host.retained_host_bytes();
    identity.host_common_plan_vector_bytes = host_plan_bytes.common_plan_vectors;
    identity.host_gfn1_plan_vector_bytes = host_plan_bytes.gfn1_plan_vectors;
    identity.host_common_model_plan_bytes = host_plan_bytes.common_model_plans;
    identity.host_gfn1_model_plan_bytes = host_plan_bytes.gfn1_model_plans;
    identity.host_numerical_vector_bytes = host_plan_bytes.numerical_vectors;
    identity.host_gfn1_expanded_parameter_bytes = host_plan_bytes.gfn1_expanded_parameters;
    identity.host_gfn2_wavefunction_arena_bytes = host_plan_bytes.gfn2_wavefunction_arena;
    identity.host_gfn1_wavefunction_arena_bytes = host_plan_bytes.gfn1_wavefunction_arena;
    identity.topology_setup_host_bytes = current.topology_owner.retained_host_bytes();
    identity.inputs_setup_host_bytes = current.inputs_owner.retained_host_bytes();
    identity.eigensolver_setup_host_bytes = current.eigensolver_owner.retained_host_bytes();
    identity.initializer_host_bytes = current.initializer.retained_host_bytes();
    identity.initializer_device_checkpoint_bytes = current.initializer.image_bytes();
    identity.scc_loop_device_control_bytes = current.scc_loop.retained_device_bytes();
    identity.retained_host_workspace_bytes =
        identity.provider_host_workspace_bytes + identity.numerical_host_staging_arena_bytes +
        identity.public_result_host_arena_bytes + identity.candidate_validation_arena_bytes +
        identity.native_lattice_host_staging_bytes + identity.topology_staging_host_bytes +
        identity.runtime_owner_host_bytes + identity.host_plans_bytes +
        identity.topology_setup_host_bytes + identity.inputs_setup_host_bytes +
        identity.eigensolver_setup_host_bytes + identity.initializer_host_bytes;
    identity.retained_device_workspace_bytes =
        identity.topology_arena_bytes + identity.input_arena_bytes +
        identity.iteration_arena_bytes + identity.eigensolver_setup_arena_bytes +
        identity.integral_task_arena_bytes + identity.force_immutable_arena_bytes +
        identity.force_execution_arena_bytes + identity.numerical_refresh_arena_bytes +
        identity.inference_arena_bytes + identity.public_result_device_arena_bytes +
        identity.interaction_device_staging_arena_bytes + identity.topology_staging_device_bytes +
        identity.initializer_device_checkpoint_bytes + identity.scc_loop_device_control_bytes;
    return identity;
  }

  std::int32_t device_id = -1;
  cudaStream_t stream = nullptr;
  Gfn2CudaTopologyStaging topology_staging;
  cusolverDnHandle_t solver = nullptr;
  cusolverDnParams_t solver_parameters = nullptr;
  syevjInfo_t solver_jacobi = nullptr;
  cublasHandle_t blas = nullptr;
  bool handles_created = false;
  std::uint64_t next_plan_token = 1u;
  std::uint64_t next_request_id = 1u;
  std::uint64_t request_submissions = 0u;
  bool request_poisoned = false;
  PinnedArena native_lattice_host_arena;
  CudaEvent native_lattice_event;
  bool native_lattice_staging_pending = false;
  bool native_lattice_staging_poisoned = false;
  ActiveRequest active_request;
  std::unique_ptr<Prepared> prepared;
  mutable std::mutex mutex;
};

Gfn2CudaExecutionCache::Gfn2CudaExecutionCache(std::int32_t device_id, void* stream)
    : impl_(std::make_unique<Impl>(device_id, stream)) {}

Gfn2CudaExecutionCache::~Gfn2CudaExecutionCache() = default;

xtbloom_status_t execute_restricted_gfn2_cuda_impl(Gfn2CudaExecutionCache& cache,
                                                   const xtbloom_batch_t& batch,
                                                   const xtbloom_compute_options_t& options,
                                                   xtbloom_batch_result_t& result,
                                                   bool require_prepared_topology,
                                                   std::string& error) {
  if (cache.impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  auto& implementation = *cache.impl_;
  std::lock_guard<std::mutex> lock(implementation.mutex);
  if (implementation.request_poisoned) {
    error = "CUDA execution cache is poisoned by a failed request teardown";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  if (implementation.active_request.id != 0u) {
    error = "CUDA execution cache already has an active asynchronous request";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  ScopedCudaDevice device(implementation.device_id, error);
  if (!device.ok()) return device.status();
  const auto finish = [&](xtbloom_status_t status) -> xtbloom_status_t {
    std::string restore_error;
    const xtbloom_status_t restore_status = device.restore(restore_error);
    if (restore_status == XTBLOOM_STATUS_SUCCESS) return status;
    if (status != XTBLOOM_STATUS_SUCCESS && !error.empty()) {
      error += "; additionally, " + restore_error;
    } else {
      error = std::move(restore_error);
    }
    return restore_status;
  };

  /* Keep every transaction-owned CUDA object inside this scope.  It ends
   * before finish() restores the caller's current device, so failed candidate
   * teardown and host-callback settlement always run on the context device. */
  const xtbloom_status_t transaction_status = [&]() -> xtbloom_status_t {
    xtbloom_status_t status =
        validate_cuda_stream_owner(implementation.device_id, implementation.stream, true, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = implementation.validate_public_request_pointers(batch, options, &result, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = implementation.validate_native_lattice_request_sync(
        batch, error, native_lattice_active(batch) && options.model == XTBLOOM_MODEL_GFN2_XTB);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = implementation.ensure_handles(error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    /* Native XYZ periodicity is deliberately kept off the molecular CUDA
     * graph until the dedicated Ewald/multipole kernels land.  Route only
     * this explicitly identified case through the validated CPU periodic
     * reference bridge; molecular and external-operator requests continue
     * through the existing CUDA graph below. */
    if (native_lattice_active(batch) && options.model == XTBLOOM_MODEL_GFN2_XTB) {
      return implementation.execute_native_periodic_cpu_bridge_locked(
          batch, options, result, require_prepared_topology, error);
    }

    const Gfn2CudaTopologyStagingDiagnostic staged =
        implementation.topology_staging.stage_and_validate(batch, error);
    if (!staged.success()) return staged.status;
    bool topology_candidate_pending =
        staged.disposition == Gfn2CudaTopologyStageDisposition::kCandidate;
    const auto abort_topology_candidate = [&]() noexcept {
      if (topology_candidate_pending) {
        implementation.topology_staging.abort_candidate();
        topology_candidate_pending = false;
      }
    };
    if (require_prepared_topology && topology_candidate_pending) {
      abort_topology_candidate();
      error = "the batch topology does not match the fixed CUDA plan topology";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const Gfn2CudaTopologyHostSnapshot* topology =
        topology_candidate_pending ? implementation.topology_staging.candidate_snapshot()
                                   : implementation.topology_staging.committed_snapshot();
    if (topology == nullptr) {
      abort_topology_candidate();
      error = "CUDA topology staging did not expose the selected canonical snapshot";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    Gfn2CudaExecutionCache::Impl::Prepared* working = implementation.prepared.get();
    const bool reuse_runtime =
        working != nullptr && topology_snapshot_matches(*topology, options, working->host.key);
    const Gfn2CudaSccStartMode start_mode = public_scc_start_mode(options);
    if (start_mode == Gfn2CudaSccStartMode::kWarm) {
      if (!reuse_runtime) {
        abort_topology_candidate();
        error =
            "CUDA strict WARM SCC start requires the existing compatible fixed-topology runtime";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (!working->inference.warm_checkpoint_ready) {
        abort_topology_candidate();
        error = "CUDA strict WARM SCC start requires a preceding successful public checkpoint";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    std::unique_ptr<Gfn2CudaExecutionCache::Impl::Prepared> candidate;
    if (!reuse_runtime) {
      TopologyKey key;
      std::vector<double> seed_positions;
      std::vector<double> seed_point_positions;
      std::vector<double> seed_point_values;
      std::vector<double> seed_point_gammas;
      std::vector<double> seed_periodic_shifts;
      std::vector<double> seed_periodic_response;
      status = make_topology_only_seed(*topology, options, key, seed_positions,
                                       seed_point_positions, seed_point_values, seed_point_gammas,
                                       seed_periodic_shifts, seed_periodic_response, error);
      if (status == XTBLOOM_STATUS_SUCCESS) {
        try {
          status = implementation.build_candidate(
              std::move(key), std::move(seed_positions), std::move(seed_point_positions),
              std::move(seed_point_values), std::move(seed_point_gammas),
              std::move(seed_periodic_shifts), std::move(seed_periodic_response), candidate, error);
        } catch (const std::bad_alloc&) {
          error = "failed to allocate a CUDA GFN2 runtime candidate";
          status = XTBLOOM_STATUS_ALLOCATION_FAILED;
        } catch (const std::exception& exception) {
          error = exception.what();
          status = XTBLOOM_STATUS_INTERNAL_ERROR;
        } catch (...) {
          error = "unknown exception while constructing a CUDA GFN2 runtime candidate";
          status = XTBLOOM_STATUS_INTERNAL_ERROR;
        }
      }
      if (status != XTBLOOM_STATUS_SUCCESS) {
        abort_topology_candidate();
        return status;
      }
      working = candidate.get();
    }

    const auto fail_working_transaction = [&](xtbloom_status_t failure) {
      const xtbloom_status_t settled =
          implementation.settle_public_submissions_locked(*working, failure, error);
      abort_topology_candidate();
      return settled;
    };

    status = implementation.reset_request_topology_error_locked(*working, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);

    const bool prior_warm_checkpoint_ready = working->inference.warm_checkpoint_ready;
    Gfn2CudaNumericalInputView numerical{};
    numerical.positions = batch.positions;
    numerical.point_charge_positions = batch.point_charge_positions;
    numerical.point_charge_values = batch.point_charge_values;
    numerical.point_charge_gammas = batch.point_charge_gammas;
    numerical.atomic_potential_shifts = batch.atomic_potential_shifts;
    numerical.charge_response_matrix = batch.charge_response_matrix;
    if (batch.struct_size >= XTBLOOM_BATCH_V3_SIZE) {
      numerical.total_interactions = batch.total_interactions;
      numerical.interaction_descriptors = batch.interaction_descriptors;
      numerical.interaction_payload = batch.interaction_payload;
    }
    status = implementation.refresh_numerical_locked(
        *working, numerical, start_mode == Gfn2CudaSccStartMode::kWarm, true, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.execute_inference_locked(*working, start_mode, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);
    /* Public synchronous readiness is finalized only after the completion
     * event and aggregate bridge diagnostics are known to have succeeded. */
    PublicResultTransaction public_result_transaction;
    public_result_transaction.prior_warm_checkpoint_ready = prior_warm_checkpoint_ready;
    working->inference.warm_checkpoint_ready = false;
    status = implementation.enqueue_public_result_prepare_locked(*working, batch, options, result,
                                                                 public_result_transaction, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.record_public_result_completion_locked(
        *working, public_result_transaction, PublicResultTransactionPhase::kPrepareSubmitted,
        PublicResultTransactionPhase::kPrepareCompletionRecorded,
        PublicResultTransactionPhase::kPrepareCompleted, false, "CUDA public inference completion",
        error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.wait_public_result_completion_locked(
        *working, public_result_transaction,
        PublicResultTransactionPhase::kPrepareCompletionRecorded,
        PublicResultTransactionPhase::kPrepareCompleted, "CUDA public inference completion", error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.accept_public_result_prepare_locked(*working, public_result_transaction,
                                                                error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      if (Gfn2CudaExecutionCache::Impl::request_semantic_rejection(working->public_result)) {
        working->inference.warm_checkpoint_ready =
            public_result_transaction.prior_warm_checkpoint_ready;
      }
      return fail_working_transaction(status);
    }

    /* The completed prepare is accepted before the private host-upload stream
     * is settled and before any caller CUDA output is touched. */
    status = implementation.settle_public_submissions_locked(*working, status, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      abort_topology_candidate();
      return status;
    }
    if (topology_candidate_pending) {
      const Gfn2CudaTopologyStagingDiagnostic prepared_commit =
          implementation.topology_staging.prepare_candidate_commit(error);
      if (!prepared_commit.success()) return fail_working_transaction(prepared_commit.status);
      if (!implementation.topology_staging.candidate_publishable()) {
        error = "prepared CUDA topology candidate is not publishable";
        return fail_working_transaction(XTBLOOM_STATUS_INTERNAL_ERROR);
      }
    }

    status = implementation.enqueue_public_result_commit_locked(*working, public_result_transaction,
                                                                false, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail_working_transaction(status);

    /* A successful launch is the caller-device commit point. Ownership moves
     * must now follow even if stream completion later reports a hard fault;
     * claiming rollback could not restore caller CUDA bytes. */
    if (candidate != nullptr) implementation.prepared = std::move(candidate);
    bool ownership_published = true;
    if (topology_candidate_pending) {
      ownership_published = implementation.topology_staging.publish_candidate();
      topology_candidate_pending = false;
      if (!ownership_published) {
        error = "CUDA topology publication invariant failed after caller-device commit acceptance";
      }
    }
    constexpr const char* kCommitCompletionFailure =
        "CUDA caller-device result commit failed after its kernel was accepted; caller CUDA "
        "outputs may have been modified";
    status = implementation.record_public_result_completion_locked(
        *working, public_result_transaction, PublicResultTransactionPhase::kCommitSubmitted,
        PublicResultTransactionPhase::kCommitCompletionRecorded,
        PublicResultTransactionPhase::kCommitCompleted, true, kCommitCompletionFailure, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = implementation.wait_public_result_completion_locked(
        *working, public_result_transaction,
        PublicResultTransactionPhase::kCommitCompletionRecorded,
        PublicResultTransactionPhase::kCommitCompleted, kCommitCompletionFailure, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    std::uint32_t completed_result_flags = 0u;
    status = implementation.publish_public_results_locked(
        *working, options, result, public_result_transaction, ownership_published, true,
        completed_result_flags, error);
    return status;
  }();
  return finish(transaction_status);
}

xtbloom_status_t execute_restricted_gfn2_cuda(Gfn2CudaExecutionCache& cache,
                                              const xtbloom_batch_t& batch,
                                              const xtbloom_compute_options_t& options,
                                              xtbloom_batch_result_t& result, std::string& error) {
  return execute_restricted_gfn2_cuda_impl(cache, batch, options, result, false, error);
}

xtbloom_status_t execute_restricted_gfn2_cuda_plan(Gfn2CudaExecutionCache& cache,
                                                   const xtbloom_batch_t& batch,
                                                   const xtbloom_compute_options_t& options,
                                                   xtbloom_batch_result_t& result,
                                                   std::string& error) {
  return execute_restricted_gfn2_cuda_impl(cache, batch, options, result, true, error);
}

xtbloom_status_t enqueue_restricted_gfn2_cuda_impl(
    const std::shared_ptr<Gfn2CudaExecutionCache>& cache, const xtbloom_batch_t& batch,
    const xtbloom_compute_options_t& options, const xtbloom_batch_result_t& result,
    bool require_prepared_topology, RequestSubmission& submission, std::string& error) {
  submission = {};
  if (cache == nullptr || cache->impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  auto& implementation = *cache->impl_;
  xtbloom_status_t final_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  {
    std::lock_guard<std::mutex> lock(implementation.mutex);
    if (implementation.request_poisoned) {
      error = "CUDA execution cache is poisoned by a failed request teardown";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (implementation.active_request.id != 0u) {
      error = "CUDA execution cache already has an active asynchronous request";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    std::uint64_t request_id = implementation.next_request_id++;
    if (request_id == 0u) request_id = implementation.next_request_id++;
    implementation.active_request = {};
    implementation.active_request.id = request_id;
    struct ActiveRequestExceptionGuard {
      Gfn2CudaExecutionCache::Impl& implementation;
      bool armed = true;
      ~ActiveRequestExceptionGuard() {
        if (armed) implementation.settle_active_request_exception_noexcept_locked();
      }
      void dismiss() noexcept { armed = false; }
    } exception_guard{implementation};
    /* Valid short-prefix callers own no bytes beyond struct_size. Snapshot
     * only the validated prefix so asynchronous host publication never reads
     * optional suffix storage after enqueue returns. */
    std::memcpy(&implementation.active_request.options, &options,
                std::min<std::size_t>(options.struct_size, sizeof(xtbloom_compute_options_t)));
    std::memcpy(&implementation.active_request.result, &result,
                std::min<std::size_t>(result.struct_size, sizeof(xtbloom_batch_result_t)));
    ScopedCudaDevice device(implementation.device_id, error);
    if (!device.ok()) {
      implementation.active_request = {};
      return device.status();
    }

    const xtbloom_status_t transaction_status = [&]() -> xtbloom_status_t {
      xtbloom_status_t status =
          validate_cuda_stream_owner(implementation.device_id, implementation.stream, true, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      status = implementation.validate_public_request_pointers(batch, options, &result, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      status = validate_nonblocking_interaction_staging(batch, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      status = implementation.ensure_handles(error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;

      /* Native periodic GFN2 has a validated synchronous CPU reference bridge
       * while the device Ewald/multipole kernels are still being developed.
       * Run that bridge on a worker so context/plan enqueue remains genuinely
       * nonblocking.  A fixed molecular plan must continue through the normal
       * stream-ordered refusal path; only a native-periodic plan may reuse its
       * immutable topology during asynchronous execution. */
      const bool native_periodic_request =
          native_lattice_active(batch) && options.model == XTBLOOM_MODEL_GFN2_XTB;
      const bool native_periodic_bridge_allowed =
          native_periodic_request && (!require_prepared_topology ||
                                      (implementation.prepared != nullptr &&
                                       implementation.prepared->host.key.native_lattice_enabled));
      if (native_periodic_bridge_allowed) {
        std::shared_ptr<Gfn2CudaExecutionCache::Impl::NativePeriodicAsyncState> bridge_state;
        try {
          bridge_state = std::make_shared<Gfn2CudaExecutionCache::Impl::NativePeriodicAsyncState>();
        } catch (const std::bad_alloc&) {
          error = "failed to allocate native periodic CUDA bridge state";
          return XTBLOOM_STATUS_ALLOCATION_FAILED;
        }
        status = implementation.enqueue_native_periodic_cpu_bridge_locked(
            batch, options, result, require_prepared_topology, *bridge_state, error);
        if (status != XTBLOOM_STATUS_SUCCESS) return status;
        implementation.active_request.native_periodic_async = std::move(bridge_state);
        return XTBLOOM_STATUS_SUCCESS;
      }
      const Gfn2CudaSccStartMode start_mode = public_scc_start_mode(options);
      std::unique_ptr<Gfn2CudaExecutionCache::Impl::Prepared> candidate;
      bool topology_candidate_pending = false;
      const auto abort_topology_candidate = [&]() noexcept {
        if (topology_candidate_pending) {
          implementation.topology_staging.abort_candidate();
          topology_candidate_pending = false;
        }
      };
      Gfn2CudaExecutionCache::Impl::Prepared* working = implementation.prepared.get();
      if (start_mode == Gfn2CudaSccStartMode::kWarm) {
        if (working == nullptr) {
          error = "CUDA strict WARM SCC start requires an existing compatible prepared runtime";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        if (!working->inference.warm_checkpoint_ready) {
          error = "CUDA strict WARM SCC start requires a preceding successful public checkpoint";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }
      const auto ensure_selected_request_graph = [&]() -> xtbloom_status_t {
        if (working == nullptr) {
          error = "CUDA asynchronous request selected no prepared runtime";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        if (!working->scc_loop.conditional_graph_ready()) {
          error =
              "asynchronous CUDA execution is unavailable because the selected SCC "
              "provider requires the bounded uncaptured fallback";
          return XTBLOOM_STATUS_NOT_SUPPORTED;
        }
        return implementation.ensure_request_execution_graph_locked(*working, error);
      };
      if (require_prepared_topology) {
        if (working == nullptr) {
          error = "CUDA plan request has no prepared fixed-topology runtime";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        status = ensure_selected_request_graph();
        if (status != XTBLOOM_STATUS_SUCCESS) return status;
        status = implementation.enqueue_fixed_topology_validation_locked(*working, batch, options,
                                                                         error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return implementation.settle_public_submissions_locked(*working, status, error);
        }
      } else if (working != nullptr && context_enqueue_host_topology_probe_available(batch)) {
        /* Host topology can prove reuse or a legitimate topology replacement
         * without waiting on the owner stream, even when numerical leaves are
         * device-resident. Device topology still uses stream-ordered compare
         * or the bounded staging/setup admission below. */
        const TopologyMatch match =
            match_existing_topology(batch, options, working->host.key, error, false);
        if (match == TopologyMatch::kInvalid) return XTBLOOM_STATUS_INVALID_ARGUMENT;
        if (match == TopologyMatch::kMatch) {
          status = ensure_selected_request_graph();
          if (status != XTBLOOM_STATUS_SUCCESS) return status;
          status = implementation.enqueue_fixed_topology_validation_locked(*working, batch, options,
                                                                           error);
          if (status != XTBLOOM_STATUS_SUCCESS) {
            return implementation.settle_public_submissions_locked(*working, status, error);
          }
        } else {
          working = nullptr;
        }
      } else if (working != nullptr &&
                 context_enqueue_shape_policy_matches(batch, options, working->host.key)) {
        /* Device-resident topology bytes cannot be dereferenced at admission.
         * Reuse the prepared shape and compare the immutable key in stream
         * order. A mutation completes this request with INVALID_ARGUMENT; a
         * caller that intentionally changes device topology can use a changed
         * shape/policy or the synchronous convenience path to rebuild it. */
        status = ensure_selected_request_graph();
        if (status != XTBLOOM_STATUS_SUCCESS) return status;
        status = implementation.enqueue_fixed_topology_validation_locked(*working, batch, options,
                                                                         error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return implementation.settle_public_submissions_locked(*working, status, error);
        }
      } else {
        working = nullptr;
      }
      if (start_mode == Gfn2CudaSccStartMode::kWarm && working == nullptr) {
        error = "CUDA strict WARM SCC start requires the existing compatible topology and policy";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      if (!require_prepared_topology && working == nullptr) {
        const Gfn2CudaTopologyStagingDiagnostic staged =
            implementation.topology_staging.stage_and_validate(batch, error);
        if (!staged.success()) return staged.status;
        topology_candidate_pending =
            staged.disposition == Gfn2CudaTopologyStageDisposition::kCandidate;
        const Gfn2CudaTopologyHostSnapshot* topology =
            topology_candidate_pending ? implementation.topology_staging.candidate_snapshot()
                                       : implementation.topology_staging.committed_snapshot();
        if (topology == nullptr) {
          abort_topology_candidate();
          error = "CUDA context enqueue did not expose a canonical topology snapshot";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }

        /* Reaching this block means every reuse path above rejected and cleared
         * `working`, so the staged topology always needs a new candidate. */
        TopologyKey key;
        std::vector<double> seed_positions;
        std::vector<double> seed_point_positions;
        std::vector<double> seed_point_values;
        std::vector<double> seed_point_gammas;
        std::vector<double> seed_periodic_shifts;
        std::vector<double> seed_periodic_response;
        status = make_topology_only_seed(*topology, options, key, seed_positions,
                                         seed_point_positions, seed_point_values, seed_point_gammas,
                                         seed_periodic_shifts, seed_periodic_response, error);
        if (status == XTBLOOM_STATUS_SUCCESS) {
          try {
            status = implementation.build_candidate(
                std::move(key), std::move(seed_positions), std::move(seed_point_positions),
                std::move(seed_point_values), std::move(seed_point_gammas),
                std::move(seed_periodic_shifts), std::move(seed_periodic_response), candidate,
                error);
          } catch (const std::bad_alloc&) {
            error = "failed to allocate a CUDA context-enqueue runtime candidate";
            status = XTBLOOM_STATUS_ALLOCATION_FAILED;
          } catch (const std::exception& exception) {
            error = exception.what();
            status = XTBLOOM_STATUS_INTERNAL_ERROR;
          } catch (...) {
            error = "unknown exception while constructing a CUDA context-enqueue runtime";
            status = XTBLOOM_STATUS_INTERNAL_ERROR;
          }
        }
        if (status != XTBLOOM_STATUS_SUCCESS) {
          abort_topology_candidate();
          return status;
        }
        working = candidate.get();
        status = ensure_selected_request_graph();
        if (status != XTBLOOM_STATUS_SUCCESS) {
          abort_topology_candidate();
          return status;
        }
        if (topology_candidate_pending) {
          /* Seal the canonical topology before numerical inference is queued.
           * The bounded wait here belongs to topology/setup admission; placing
           * it after inference would accidentally turn a new-topology enqueue
           * into a synchronous compute. Ownership is still published only
           * after the caller-output commit has been accepted. */
          const Gfn2CudaTopologyStagingDiagnostic prepared_commit =
              implementation.topology_staging.prepare_candidate_commit(error);
          if (!prepared_commit.success()) {
            abort_topology_candidate();
            return prepared_commit.status;
          }
          if (!implementation.topology_staging.candidate_publishable()) {
            abort_topology_candidate();
            error = "prepared CUDA context-enqueue topology candidate is not publishable";
            return XTBLOOM_STATUS_INTERNAL_ERROR;
          }
        }
      }
      const auto fail_submitted = [&](xtbloom_status_t failure) {
        const xtbloom_status_t settled =
            implementation.settle_public_submissions_locked(*working, failure, error);
        abort_topology_candidate();
        return settled;
      };
      auto& active = implementation.active_request;
      active.transaction.prior_warm_checkpoint_ready = working->inference.warm_checkpoint_ready;
      /* Fixed-topology comparison above owns the one request-error reset. The
       * lattice gate appends a priority code with atomicMax so
       * a topology mismatch remains the most specific eventual diagnostic. */
      status = implementation.enqueue_native_lattice_validation_locked(*working, batch, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return fail_submitted(status);
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      if (consume_execution_test_fault(Gfn2CudaExecutionTestFault::kUnknownRequestValidationCode)) {
        constexpr std::uint32_t kUnknownRequestCode = 99u;
        const cudaError_t mark_status = cudaMemcpyAsync(
            working->public_result.request_topology_error, &kUnknownRequestCode,
            sizeof(kUnknownRequestCode), cudaMemcpyHostToDevice, implementation.stream);
        if (mark_status != cudaSuccess) {
          error = cuda_error_message("CUDA unknown request-code test injection", mark_status);
          return fail_submitted(XTBLOOM_STATUS_INTERNAL_ERROR);
        }
        working->submitted = true;
      }
#endif
      Gfn2CudaNumericalInputView numerical{};
      numerical.positions = batch.positions;
      numerical.point_charge_positions = batch.point_charge_positions;
      numerical.point_charge_values = batch.point_charge_values;
      numerical.point_charge_gammas = batch.point_charge_gammas;
      numerical.atomic_potential_shifts = batch.atomic_potential_shifts;
      numerical.charge_response_matrix = batch.charge_response_matrix;
      if (batch.struct_size >= XTBLOOM_BATCH_V3_SIZE) {
        numerical.total_interactions = batch.total_interactions;
        numerical.interaction_descriptors = batch.interaction_descriptors;
        numerical.interaction_payload = batch.interaction_payload;
      }
      bool host_upload_enqueued = false;
      status = implementation.stage_numerical_ingress_locked(
          *working, numerical, start_mode == Gfn2CudaSccStartMode::kWarm, false,
          host_upload_enqueued, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return fail_submitted(status);
      status = implementation.launch_request_execution_graph_locked(*working, start_mode, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return fail_submitted(status);
      working->inference.warm_checkpoint_ready = false;

      const auto accept_post_launch_failure = [&](xtbloom_status_t failure) {
        const xtbloom_status_t settled =
            implementation.settle_public_submissions_locked(*working, failure, error);
        const bool owner_idle = !working->submitted;
        const bool host_idle =
            !working->numerical_host_upload_completion.pending.load(std::memory_order_acquire);
        if (owner_idle && host_idle) {
          active.completion_ready = true;
          active.completion_status = settled;
          active.result_flags = 0u;
          active.completion_error = error;
          abort_topology_candidate();
        } else {
          /* The original failure remains the eventual compute status. The
           * settlement diagnostic describes why this enqueue must retain a
           * PENDING completion owner instead of publishing an unsafe inline
           * COMPLETE state. */
          active.settlement_only_pending = true;
          active.deferred_status = failure;
          active.deferred_error = error;
          if (candidate != nullptr) {
            active.pending_prepared = std::move(candidate);
            working = active.pending_prepared.get();
          }
          if (topology_candidate_pending) {
            active.pending_topology_candidate = true;
            topology_candidate_pending = false;
          }
        }
        return XTBLOOM_STATUS_SUCCESS;
      };
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      const bool prepare_and_settlement_failure = consume_execution_test_fault(
          Gfn2CudaExecutionTestFault::kRequestPrepareSubmissionAndSettlement);
      if (prepare_and_settlement_failure) {
        g_execution_test_fault.store(
            static_cast<std::uint32_t>(Gfn2CudaExecutionTestFault::kRequestSettlement),
            std::memory_order_release);
      }
      if (prepare_and_settlement_failure ||
          consume_execution_test_fault(Gfn2CudaExecutionTestFault::kRequestPrepareSubmission)) {
        error = "injected CUDA asynchronous result-prepare submission failure";
        return accept_post_launch_failure(XTBLOOM_STATUS_INTERNAL_ERROR);
      }
#endif
      status = implementation.enqueue_public_result_prepare_locked(
          *working, batch, options, active.result, active.transaction, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return accept_post_launch_failure(status);
      /* Commit is device-gated by the control record produced by prepare. It
       * is submitted now so downstream work on the same CUDA stream observes
       * real outputs without requiring a host query to advance execution. */
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
      const bool commit_and_settlement_failure = consume_execution_test_fault(
          Gfn2CudaExecutionTestFault::kRequestCommitSubmissionAndSettlement);
      if (commit_and_settlement_failure) {
        g_execution_test_fault.store(
            static_cast<std::uint32_t>(Gfn2CudaExecutionTestFault::kRequestSettlement),
            std::memory_order_release);
      }
      if (commit_and_settlement_failure ||
          consume_execution_test_fault(Gfn2CudaExecutionTestFault::kRequestCommitSubmission)) {
        error = "injected CUDA asynchronous result-commit submission failure";
        return accept_post_launch_failure(XTBLOOM_STATUS_INTERNAL_ERROR);
      }
#endif
      status = implementation.enqueue_public_result_commit_locked(*working, active.transaction,
                                                                  true, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return accept_post_launch_failure(status);
      if (candidate != nullptr) {
        implementation.prepared = std::move(candidate);
        working = implementation.prepared.get();
      }
      if (topology_candidate_pending) {
        if (!implementation.topology_staging.publish_candidate()) {
          /* Prepared ownership and caller-output commit are already accepted,
           * so rollback is impossible. Poison the cache rather than exposing
           * a new runtime key beside the old committed topology snapshot. */
          implementation.request_poisoned = true;
          working->numerical.host_staging_poisoned = true;
          active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
          active.deferred_error =
              "CUDA context-enqueue topology publication invariant failed after caller-output "
              "commit acceptance";
        }
        topology_candidate_pending = false;
      }
      ++implementation.request_submissions;
      if (host_upload_enqueued) {
        const cudaError_t release_status = cudaStreamWaitEvent(
            implementation.stream, working->numerical_host_release_complete.get(), 0u);
        if (release_status != cudaSuccess) {
          /* The commit is already accepted. Preserve a PENDING request and
           * let its exceptional settlement fence the two exact streams. */
          active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
          active.deferred_error = cuda_error_message(
              "CUDA numerical host-release ordering for request completion", release_status);
          return XTBLOOM_STATUS_SUCCESS;
        }
        active.host_upload_release_ordered = true;
      }
      status = implementation.record_public_result_completion_locked(
          *working, active.transaction, PublicResultTransactionPhase::kCommitSubmitted,
          PublicResultTransactionPhase::kCommitCompletionRecorded,
          PublicResultTransactionPhase::kCommitCompleted, true,
          "CUDA asynchronous caller-output completion", error);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        /* The commit was accepted, but neither event recording nor the
         * helper's exact-stream fallback settled it. Keep the request PENDING:
         * wait/destroy must synchronize the owner stream because the event
         * object may still name an older completed phase. */
        active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
        active.deferred_error = error;
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (active.transaction.phase == PublicResultTransactionPhase::kCommitCompleted) {
        const bool host_release_pending =
            working->numerical_host_upload_completion.pending.load(std::memory_order_acquire);
        if (host_release_pending && active.host_upload_release_ordered) {
          active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
          active.deferred_error =
              "CUDA numerical host release remained pending after "
              "owner-stream completion fallback";
          return XTBLOOM_STATUS_SUCCESS;
        }
        cudaError_t host_status = cudaSuccess;
        if (host_release_pending) {
          host_status = cudaStreamSynchronize(working->numerical_host_completion_stream.get());
        }
        if (host_status != cudaSuccess) {
          active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
          active.deferred_error = cuda_error_message(
              "CUDA numerical host snapshot settlement after event-record fallback", host_status);
          return XTBLOOM_STATUS_SUCCESS;
        }
        working->numerical_host_upload_completion.pending.store(false, std::memory_order_release);
        implementation.finalize_active_request_after_commit_locked(*working);
      }
      return XTBLOOM_STATUS_SUCCESS;
    }();

    std::string restore_error;
    const xtbloom_status_t restore_status = device.restore(restore_error);
    final_status = transaction_status;
    if (restore_status != XTBLOOM_STATUS_SUCCESS) {
      if (transaction_status == XTBLOOM_STATUS_SUCCESS) {
        auto& active = implementation.active_request;
        active.deferred_status = XTBLOOM_STATUS_INTERNAL_ERROR;
        /* Device restoration is part of the accepted request contract. Close
         * readiness before any potentially-throwing diagnostic allocation. */
        auto* const active_prepared = implementation.active_request.pending_prepared != nullptr
                                          ? implementation.active_request.pending_prepared.get()
                                          : implementation.prepared.get();
        if (active_prepared != nullptr) {
          active_prepared->inference.warm_checkpoint_ready = false;
        }
        if (!active.deferred_error.empty()) active.deferred_error += "; additionally, ";
        active.deferred_error += restore_error;
        if (active.completion_ready) {
          active.completion_status = XTBLOOM_STATUS_INTERNAL_ERROR;
          active.result_flags = 0u;
          if (!active.completion_error.empty()) active.completion_error += "; additionally, ";
          active.completion_error += active.deferred_error;
        }
        final_status = XTBLOOM_STATUS_SUCCESS;
      } else {
        final_status = restore_status;
        if (!error.empty() && error != restore_error) {
          error += "; additionally, " + restore_error;
        } else {
          error = std::move(restore_error);
        }
      }
    }
    if (final_status != XTBLOOM_STATUS_SUCCESS) {
      implementation.active_request = {};
    } else if (implementation.active_request.completion_ready) {
      submission.completed_inline = true;
      submission.completion_status = implementation.active_request.completion_status;
      submission.result_flags = implementation.active_request.result_flags;
      submission.completion_error = implementation.active_request.completion_error;
      implementation.active_request = {};
    }
    exception_guard.dismiss();
  }

  if (final_status != XTBLOOM_STATUS_SUCCESS) return final_status;
  if (!submission.completed_inline) {
    /* The plan already owns this shared_ptr/control block. Casting and copying
     * it into the request is allocation-free in steady state. */
    submission.pending = std::static_pointer_cast<RequestCompletion>(cache);
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t enqueue_restricted_gfn2_cuda_plan(
    const std::shared_ptr<Gfn2CudaExecutionCache>& cache, const xtbloom_batch_t& batch,
    const xtbloom_compute_options_t& options, const xtbloom_batch_result_t& result,
    RequestSubmission& submission, std::string& error) {
  return enqueue_restricted_gfn2_cuda_impl(cache, batch, options, result, true, submission, error);
}

xtbloom_status_t enqueue_restricted_gfn2_cuda(const std::shared_ptr<Gfn2CudaExecutionCache>& cache,
                                              const xtbloom_batch_t& batch,
                                              const xtbloom_compute_options_t& options,
                                              const xtbloom_batch_result_t& result,
                                              RequestSubmission& submission, std::string& error) {
  return enqueue_restricted_gfn2_cuda_impl(cache, batch, options, result, false, submission, error);
}

xtbloom_status_t Gfn2CudaExecutionCache::probe(bool wait,
                                               RequestCompletionResult& result) noexcept {
  result = {};
  if (impl_ == nullptr) {
    result.completion_error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& active = impl_->active_request;
    if (active.native_periodic_async != nullptr) {
      auto& bridge = *active.native_periodic_async;
      std::unique_lock<std::mutex> bridge_lock(bridge.mutex);
      if (!bridge.done) {
        if (!wait) {
          result.complete = false;
          return XTBLOOM_STATUS_SUCCESS;
        }
        bridge.condition.wait(bridge_lock, [&bridge] { return bridge.done; });
      }
      result.complete = true;
      result.completion_status = bridge.status;
      result.result_flags = bridge.result_flags;
      result.completion_error = bridge.error;
      return XTBLOOM_STATUS_SUCCESS;
    }
    Gfn2CudaExecutionCache::Impl::Prepared* current_owner =
        active.pending_prepared != nullptr ? active.pending_prepared.get() : impl_->prepared.get();
    if (active.id == 0u || current_owner == nullptr) {
      result.completion_error = "CUDA request completion does not own the active cache submission";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (active.completion_ready) {
      result.complete = true;
      result.completion_status = active.completion_status;
      result.result_flags = active.result_flags;
      result.completion_error = active.completion_error;
      return XTBLOOM_STATUS_SUCCESS;
    }

    std::string device_error;
    ScopedCudaDevice device(impl_->device_id, device_error);
    if (!device.ok()) {
      result.completion_error = std::move(device_error);
      return device.status();
    }
    auto& current = *current_owner;
    auto& transaction = active.transaction;

    if (active.settlement_only_pending) {
      bool incomplete = false;
      const auto observe_stream = [&](cudaStream_t stream, const char* operation,
                                      bool& submitted) -> xtbloom_status_t {
        if (!submitted) return XTBLOOM_STATUS_SUCCESS;
        const cudaError_t cuda_status =
            wait ? cudaStreamSynchronize(stream) : cudaStreamQuery(stream);
        if (!wait && cuda_status == cudaErrorNotReady) {
          incomplete = true;
          return XTBLOOM_STATUS_SUCCESS;
        }
        if (cuda_status != cudaSuccess) {
          result.completion_error = cuda_error_message(operation, cuda_status);
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        submitted = false;
        return XTBLOOM_STATUS_SUCCESS;
      };

      xtbloom_status_t status =
          observe_stream(impl_->stream, "CUDA post-launch request settlement", current.submitted);
      bool host_pending =
          current.numerical_host_upload_completion.pending.load(std::memory_order_acquire);
      if (status == XTBLOOM_STATUS_SUCCESS && !incomplete && host_pending) {
        status = observe_stream(current.numerical_host_completion_stream.get(),
                                "CUDA post-launch host snapshot settlement", host_pending);
        if (status == XTBLOOM_STATUS_SUCCESS && !host_pending) {
          current.numerical_host_upload_completion.pending.store(false, std::memory_order_release);
        }
      }
      if (status != XTBLOOM_STATUS_SUCCESS) {
        std::string ignored;
        (void)device.restore(ignored);
        return status;
      }
      if (!incomplete && !current.submitted &&
          !current.numerical_host_upload_completion.pending.load(std::memory_order_acquire)) {
        active.completion_ready = true;
        active.completion_status = active.deferred_status;
        active.result_flags = 0u;
        active.completion_error = active.deferred_error;
        active.settlement_only_pending = false;
        if (active.pending_topology_candidate) {
          impl_->topology_staging.abort_candidate();
          active.pending_topology_candidate = false;
        }
        active.pending_prepared.reset();
      }

      std::string restore_error;
      const xtbloom_status_t restore_status = device.restore(restore_error);
      if (restore_status != XTBLOOM_STATUS_SUCCESS) {
        result.completion_error = std::move(restore_error);
        return restore_status;
      }
      if (active.completion_ready) {
        result.complete = true;
        result.completion_status = active.completion_status;
        result.result_flags = active.result_flags;
        result.completion_error = active.completion_error;
      }
      return XTBLOOM_STATUS_SUCCESS;
    }

    bool incomplete = false;
    const auto observe_event = [&]() -> xtbloom_status_t {
      if (transaction.phase == PublicResultTransactionPhase::kCommitCompleted) {
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (transaction.phase == PublicResultTransactionPhase::kCommitSubmitted) {
        const cudaError_t stream_status =
            wait ? cudaStreamSynchronize(impl_->stream) : cudaStreamQuery(impl_->stream);
        if (!wait && stream_status == cudaErrorNotReady) {
          incomplete = true;
          return XTBLOOM_STATUS_SUCCESS;
        }
        if (stream_status != cudaSuccess) {
          result.completion_error = cuda_error_message(
              "CUDA asynchronous caller-output settlement after event-record failure",
              stream_status);
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        current.submitted = false;
        transaction.phase = PublicResultTransactionPhase::kCommitCompleted;
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (transaction.phase != PublicResultTransactionPhase::kCommitCompletionRecorded ||
          current.public_result_completion_event.get() == nullptr) {
        result.completion_error = "CUDA request has an invalid completion-event phase";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      const cudaError_t cuda_status =
          wait ? cudaEventSynchronize(current.public_result_completion_event.get())
               : cudaEventQuery(current.public_result_completion_event.get());
      if (!wait && cuda_status == cudaErrorNotReady) {
        incomplete = true;
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (cuda_status != cudaSuccess) {
        result.completion_error =
            cuda_error_message("CUDA asynchronous caller-output completion", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      current.submitted = false;
      transaction.phase = PublicResultTransactionPhase::kCommitCompleted;
      return XTBLOOM_STATUS_SUCCESS;
    };

    const auto observe_private_stream = [&]() -> xtbloom_status_t {
      if (!current.numerical_host_completion_stream.valid()) return XTBLOOM_STATUS_SUCCESS;
      if (!current.numerical_host_upload_completion.pending.load(std::memory_order_acquire)) {
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (active.host_upload_release_ordered) {
        result.completion_error =
            "CUDA numerical host release remained pending after request completion";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      const cudaError_t cuda_status =
          wait ? cudaStreamSynchronize(current.numerical_host_completion_stream.get())
               : cudaStreamQuery(current.numerical_host_completion_stream.get());
      if (!wait && cuda_status == cudaErrorNotReady) {
        incomplete = true;
        return XTBLOOM_STATUS_SUCCESS;
      }
      if (cuda_status != cudaSuccess) {
        result.completion_error =
            cuda_error_message("CUDA numerical host snapshot completion", cuda_status);
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      current.numerical_host_upload_completion.pending.store(false, std::memory_order_release);
      return XTBLOOM_STATUS_SUCCESS;
    };

    xtbloom_status_t status = observe_event();
    if (status != XTBLOOM_STATUS_SUCCESS) {
      std::string ignored;
      (void)device.restore(ignored);
      return status;
    }
    if (!incomplete) {
      status = observe_private_stream();
      if (status != XTBLOOM_STATUS_SUCCESS) {
        std::string ignored;
        (void)device.restore(ignored);
        return status;
      }
    }
    if (!incomplete) {
      impl_->finalize_active_request_after_commit_locked(current);
    }

    std::string restore_error;
    const xtbloom_status_t restore_status = device.restore(restore_error);
    if (restore_status != XTBLOOM_STATUS_SUCCESS) {
      /* Restoring the query thread's device is part of accessing the request,
       * not the submitted computation. Preserve an already finalized compute
       * snapshot so a later query can retrieve it, but report this call's
       * access failure through the API return channel. */
      result.completion_error = std::move(restore_error);
      return restore_status;
    }
    if (active.completion_ready) {
      result.complete = true;
      result.completion_status = active.completion_status;
      result.result_flags = active.result_flags;
      result.completion_error = active.completion_error;
    }
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    result.completion_error = "failed to allocate while probing CUDA request completion";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::exception& exception) {
    result.completion_error = exception.what();
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  } catch (...) {
    result.completion_error = "unknown exception while probing CUDA request completion";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
}

void Gfn2CudaExecutionCache::settle_noexcept() noexcept {
  if (impl_ == nullptr) return;
  try {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->active_request.id == 0u) return;
    if (impl_->active_request.native_periodic_async != nullptr) {
      if (impl_->active_request.native_periodic_async->worker.joinable()) {
        impl_->active_request.native_periodic_async->worker.join();
      }
      impl_->active_request = {};
      return;
    }
    if (impl_->active_request.completion_ready) {
      if (impl_->active_request.pending_topology_candidate) {
        impl_->topology_staging.abort_candidate();
      }
      impl_->active_request = {};
      return;
    }
    bool settled = false;
    Impl::Prepared* const current_owner = impl_->active_request.pending_prepared != nullptr
                                              ? impl_->active_request.pending_prepared.get()
                                              : impl_->prepared.get();
    if (current_owner != nullptr) {
      auto& current = *current_owner;
      std::string ignored;
      ScopedCudaDevice device(impl_->device_id, ignored);
      if (device.ok()) {
        cudaError_t owner_status = cudaSuccess;
        if (current.submitted) {
          owner_status = cudaErrorInvalidValue;
          if (impl_->active_request.transaction.phase ==
                  PublicResultTransactionPhase::kCommitCompletionRecorded &&
              current.public_result_completion_event.get() != nullptr) {
            owner_status = cudaEventSynchronize(current.public_result_completion_event.get());
            if (owner_status != cudaSuccess) owner_status = cudaStreamSynchronize(impl_->stream);
          } else {
            /* A failed record leaves the reusable event pointing at an older
             * phase; only the exact owner stream can settle this commit. */
            owner_status = cudaStreamSynchronize(impl_->stream);
          }
          if (owner_status == cudaSuccess) current.submitted = false;
        }
        cudaError_t host_status = cudaSuccess;
        if (current.numerical_host_completion_stream.valid() &&
            current.numerical_host_upload_completion.pending.load(std::memory_order_acquire)) {
          host_status = cudaStreamSynchronize(current.numerical_host_completion_stream.get());
          if (host_status == cudaSuccess) {
            current.numerical_host_upload_completion.pending.store(false,
                                                                   std::memory_order_release);
          }
        }
        settled = owner_status == cudaSuccess && host_status == cudaSuccess;
        (void)device.restore(ignored);
      }
    }
    if (settled) {
      if (impl_->active_request.pending_topology_candidate) {
        impl_->topology_staging.abort_candidate();
      }
      impl_->active_request = {};
    } else {
      impl_->request_poisoned = true;
      if (current_owner != nullptr) current_owner->numerical.host_staging_poisoned = true;
    }
  } catch (...) {
    /* Request destruction cannot report. Prepared teardown retains a second
     * exact-stream safety net if CUDA or allocation state is already broken. */
  }
}

xtbloom_status_t Gfn2CudaExecutionCache::prepare_host(const xtbloom_batch_t& batch,
                                                      const xtbloom_compute_options_t& options,
                                                      bool& reused, std::string& error) {
  reused = false;
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  xtbloom_status_t status = impl_->ensure_handles(error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  if (impl_->prepared != nullptr) {
    const TopologyMatch match =
        match_existing_topology(batch, options, impl_->prepared->host.key, error);
    if (match == TopologyMatch::kMatch) {
      status = impl_->reset_request_topology_error_locked(*impl_->prepared, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      Gfn2CudaNumericalInputView numerical{};
      numerical.positions = batch.positions;
      numerical.point_charge_positions = batch.point_charge_positions;
      numerical.point_charge_values = batch.point_charge_values;
      numerical.point_charge_gammas = batch.point_charge_gammas;
      numerical.atomic_potential_shifts = batch.atomic_potential_shifts;
      numerical.charge_response_matrix = batch.charge_response_matrix;
      if (batch.struct_size >= XTBLOOM_BATCH_V3_SIZE) {
        numerical.total_interactions = batch.total_interactions;
        numerical.interaction_descriptors = batch.interaction_descriptors;
        numerical.interaction_payload = batch.interaction_payload;
      }
      status = impl_->refresh_numerical_locked(*impl_->prepared, numerical, false, true, error);
      if (status == XTBLOOM_STATUS_SUCCESS) {
        /* prepare_host is the synchronous internal setup entry: callers may
         * immediately reuse or execute the prepared runtime after it returns.
         * Settle the owner-stream refresh and the pinned host-upload lease at
         * this boundary instead of exposing the asynchronous public API's
         * single-flight state to a subsequent prepare call. */
        status = impl_->settle_public_submissions_locked(*impl_->prepared, status, error);
      }
      reused = status == XTBLOOM_STATUS_SUCCESS;
      return status;
    }
    if (match == TopologyMatch::kInvalid) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  TopologyKey key;
  std::vector<double> positions;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response;
  std::unique_ptr<Impl::Prepared> candidate;
  try {
    /* Topology staging allocates host vectors too. Keep it in the same
     * exception-to-status boundary as candidate construction so no allocation
     * failure can escape toward the C ABI. */
    status = make_topology_key(batch, options, key, positions, point_positions, point_values,
                               point_gammas, periodic_shifts, periodic_response, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = impl_->build_candidate(std::move(key), std::move(positions),
                                    std::move(point_positions), std::move(point_values),
                                    std::move(point_gammas), std::move(periodic_shifts),
                                    std::move(periodic_response), candidate, error);
  } catch (const std::bad_alloc&) {
    error = "failed to allocate host metadata for the CUDA GFN2 runtime candidate";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::exception& exception) {
    error = exception.what();
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  } catch (...) {
    error = "unknown exception while constructing CUDA GFN2 runtime candidate";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  /* A direct host preparation promises an immediately executable numerical
   * runtime. Publish the initial geometry through the same transaction used
   * by reuse and public compute so pair-list/D4 leaves, overlap factors, the
   * common generation, and eligibility all become epoch 1 together. */
  Gfn2CudaNumericalInputView numerical{};
  numerical.positions = batch.positions;
  numerical.point_charge_positions = batch.point_charge_positions;
  numerical.point_charge_values = batch.point_charge_values;
  numerical.point_charge_gammas = batch.point_charge_gammas;
  numerical.atomic_potential_shifts = batch.atomic_potential_shifts;
  numerical.charge_response_matrix = batch.charge_response_matrix;
  if (batch.struct_size >= XTBLOOM_BATCH_V3_SIZE) {
    numerical.total_interactions = batch.total_interactions;
    numerical.interaction_descriptors = batch.interaction_descriptors;
    numerical.interaction_payload = batch.interaction_payload;
  }
  status = impl_->refresh_numerical_locked(*candidate, numerical, false, true, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = impl_->settle_public_submissions_locked(*candidate, status, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  impl_->prepared = std::move(candidate);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t Gfn2CudaExecutionCache::prepare_topology_only(
    const xtbloom_batch_t& batch, const xtbloom_compute_options_t& options, std::string& error) {
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  ScopedCudaDevice device(impl_->device_id, error);
  if (!device.ok()) return device.status();
  const auto finish = [&](xtbloom_status_t status) {
    std::string restore_error;
    const xtbloom_status_t restore_status = device.restore(restore_error);
    if (restore_status == XTBLOOM_STATUS_SUCCESS) return status;
    if (status != XTBLOOM_STATUS_SUCCESS && !error.empty()) {
      error += "; additionally, " + restore_error;
    } else {
      error = std::move(restore_error);
    }
    return restore_status;
  };

  const xtbloom_status_t setup_status = [&]() -> xtbloom_status_t {
    xtbloom_status_t status =
        validate_cuda_stream_owner(impl_->device_id, impl_->stream, true, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = impl_->validate_public_request_pointers(batch, options, nullptr, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = impl_->validate_native_lattice_request_sync(
        batch, error, native_lattice_active(batch) && options.model == XTBLOOM_MODEL_GFN2_XTB);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = impl_->ensure_handles(error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    const Gfn2CudaTopologyStagingDiagnostic staged =
        impl_->topology_staging.stage_and_validate(batch, error);
    if (!staged.success()) return staged.status;
    bool candidate_pending = staged.disposition == Gfn2CudaTopologyStageDisposition::kCandidate;
    const auto abort_candidate = [&]() noexcept {
      if (candidate_pending) {
        impl_->topology_staging.abort_candidate();
        candidate_pending = false;
      }
    };
    const Gfn2CudaTopologyHostSnapshot* topology =
        candidate_pending ? impl_->topology_staging.candidate_snapshot()
                          : impl_->topology_staging.committed_snapshot();
    if (topology == nullptr) {
      abort_candidate();
      error = "CUDA plan setup did not expose a canonical topology snapshot";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }

    if (impl_->prepared != nullptr &&
        topology_snapshot_matches(*topology, options, impl_->prepared->host.key)) {
      if (candidate_pending) {
        const Gfn2CudaTopologyStagingDiagnostic committed =
            impl_->topology_staging.commit_candidate(error);
        candidate_pending = false;
        if (!committed.success()) return committed.status;
      }
      error.clear();
      return XTBLOOM_STATUS_SUCCESS;
    }

    TopologyKey key;
    std::vector<double> positions;
    std::vector<double> point_positions;
    std::vector<double> point_values;
    std::vector<double> point_gammas;
    std::vector<double> periodic_shifts;
    std::vector<double> periodic_response;
    std::unique_ptr<Impl::Prepared> candidate;
    try {
      status =
          make_topology_only_seed(*topology, options, key, positions, point_positions, point_values,
                                  point_gammas, periodic_shifts, periodic_response, error);
      if (status == XTBLOOM_STATUS_SUCCESS) {
        status = impl_->build_candidate(std::move(key), std::move(positions),
                                        std::move(point_positions), std::move(point_values),
                                        std::move(point_gammas), std::move(periodic_shifts),
                                        std::move(periodic_response), candidate, error, true);
      }
    } catch (const std::bad_alloc&) {
      error = "failed to allocate host metadata for the CUDA GFN2 runtime candidate";
      status = XTBLOOM_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception& exception) {
      error = exception.what();
      status = XTBLOOM_STATUS_INTERNAL_ERROR;
    } catch (...) {
      error = "unknown exception while constructing CUDA GFN2 runtime candidate";
      status = XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    if (status != XTBLOOM_STATUS_SUCCESS) {
      abort_candidate();
      return status;
    }
    if (candidate_pending) {
      const Gfn2CudaTopologyStagingDiagnostic committed =
          impl_->topology_staging.commit_candidate(error);
      candidate_pending = false;
      if (!committed.success()) return committed.status;
    }
    impl_->prepared = std::move(candidate);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }();
  return finish(setup_status);
}

xtbloom_status_t Gfn2CudaExecutionCache::refresh_numerical_async(
    const Gfn2CudaNumericalInputView& input, std::string& error) {
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  xtbloom_status_t status = impl_->ensure_handles(error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (impl_->prepared == nullptr) {
    error = "CUDA GFN2 numerical refresh requires a prepared fixed topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return impl_->refresh_numerical_locked(*impl_->prepared, input, false, false, error);
}

xtbloom_status_t Gfn2CudaExecutionCache::execute_inference_async(Gfn2CudaSccStartMode mode,
                                                                 std::string& error) {
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  xtbloom_status_t status = impl_->ensure_handles(error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (impl_->prepared == nullptr) {
    error = "CUDA GFN2 inference requires a prepared numerical/runtime binding";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return impl_->execute_inference_locked(*impl_->prepared, mode, error);
}

bool Gfn2CudaExecutionCache::valid() const noexcept {
  if (impl_ == nullptr) return false;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->prepared != nullptr;
}

Gfn2CudaExecutionIdentity Gfn2CudaExecutionCache::identity() const noexcept {
  if (impl_ == nullptr) return {};
  std::lock_guard<std::mutex> lock(impl_->mutex);
  Gfn2CudaExecutionIdentity identity = impl_->snapshot();
  identity.request_completion_owner = opaque_address(this);
  return identity;
}

#if defined(XTBLOOM_CUDA_TEST_HOOKS)
xtbloom_status_t Gfn2CudaExecutionCache::validate_native_lattice_test_only(
    const xtbloom_batch_t& batch, std::string& error) {
  if (impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  ScopedCudaDevice device(impl_->device_id, error);
  if (!device.ok()) return device.status();
  const xtbloom_status_t status = impl_->validate_native_lattice_request_sync(batch, error);
  std::string restore_error;
  const xtbloom_status_t restore_status = device.restore(restore_error);
  if (restore_status == XTBLOOM_STATUS_SUCCESS) return status;
  if (status != XTBLOOM_STATUS_SUCCESS && !error.empty()) {
    error += "; additionally, " + restore_error;
  } else {
    error = std::move(restore_error);
  }
  return restore_status;
}

Gfn2CudaNativeLatticeTestIdentity Gfn2CudaExecutionCache::native_lattice_test_identity()
    const noexcept {
  if (impl_ == nullptr) return {};
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return {opaque_address(impl_->native_lattice_host_arena.get()),
          impl_->native_lattice_host_arena.bytes(), impl_->native_lattice_staging_pending,
          impl_->native_lattice_staging_poisoned};
}

void reset_gfn2_cuda_execution_test_state() noexcept {
  g_execution_test_fault.store(static_cast<std::uint32_t>(Gfn2CudaExecutionTestFault::kNone),
                               std::memory_order_release);
  g_native_lattice_allocation_faults.store(0u, std::memory_order_relaxed);
  g_native_lattice_completion_faults.store(0u, std::memory_order_relaxed);
  g_native_lattice_teardown_faults.store(0u, std::memory_order_relaxed);
  g_quarantined_native_lattice_arenas.store(0u, std::memory_order_relaxed);
  g_quarantined_native_lattice_bytes.store(0u, std::memory_order_relaxed);
  g_request_graph_build_attempts.store(0u, std::memory_order_relaxed);
  g_request_graph_build_successes.store(0u, std::memory_order_relaxed);
}

void arm_gfn2_cuda_execution_test_fault(Gfn2CudaExecutionTestFault fault) noexcept {
  g_execution_test_fault.store(static_cast<std::uint32_t>(fault), std::memory_order_release);
}

Gfn2CudaExecutionTestStats gfn2_cuda_execution_test_stats() noexcept {
  Gfn2CudaExecutionTestStats stats{};
  stats.native_lattice_allocation_faults =
      g_native_lattice_allocation_faults.load(std::memory_order_relaxed);
  stats.native_lattice_completion_faults =
      g_native_lattice_completion_faults.load(std::memory_order_relaxed);
  stats.native_lattice_teardown_faults =
      g_native_lattice_teardown_faults.load(std::memory_order_relaxed);
  stats.quarantined_native_lattice_arenas =
      g_quarantined_native_lattice_arenas.load(std::memory_order_relaxed);
  stats.quarantined_native_lattice_bytes =
      g_quarantined_native_lattice_bytes.load(std::memory_order_relaxed);
  stats.request_graph_build_attempts =
      g_request_graph_build_attempts.load(std::memory_order_relaxed);
  stats.request_graph_build_successes =
      g_request_graph_build_successes.load(std::memory_order_relaxed);
  return stats;
}

void set_gfn2_cuda_admission_alias_test_hook(Gfn2CudaAdmissionAliasTestHook hook) noexcept {
  g_admission_alias_test_hook.store(static_cast<std::uint32_t>(hook), std::memory_order_relaxed);
}
#endif

}  // namespace xtbloom::detail
