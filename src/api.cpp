#include <algorithm>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "model/common/basis.hpp"
#include "runtime/backend.hpp"
#include "runtime/gfn1_cpu_execution.hpp"
#include "runtime/gfn2_cpu_execution.hpp"
#include "runtime/model_plan.hpp"
#include "runtime/model_registry.hpp"
#include "xtbloom/xtbloom.h"
#if defined(XTBLOOM_HAS_CUDA)
#include "runtime/gfn2_cuda_execution.hpp"
#endif
#include "runtime/request.hpp"
#include "runtime/result_owner.hpp"
#include "runtime/validation.hpp"

struct xtbloom_context {
  xtbloom::detail::Context* implementation;
};

struct xtbloom_plan {
  xtbloom::detail::ModelPlan* implementation;
};

struct xtbloom_request {
  xtbloom::detail::Request* implementation;
};

struct xtbloom_result_owner {
  xtbloom::detail::ResultOwner* implementation;
};

namespace {

thread_local std::string last_error;

xtbloom_status_t fail(xtbloom_status_t status, std::string message) {
  last_error = std::move(message);
  return status;
}

template <typename T>
bool valid_header(const T* value, std::size_t minimum_size) {
  return value != nullptr && value->struct_size >= minimum_size &&
         value->api_version == XTBLOOM_API_VERSION;
}

template <typename Enum>
std::uint32_t raw_enum(const Enum& value) {
  static_assert(sizeof(Enum) == sizeof(std::uint32_t));
  std::uint32_t raw = 0;
  std::memcpy(&raw, &value, sizeof(raw));
  return raw;
}

template <typename T>
xtbloom_status_t initialize_structure(T* value, std::size_t caller_size, std::size_t minimum_size,
                                      const char* name) {
  if (value == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, std::string(name) + " is NULL");
  }
  if (caller_size < minimum_size || caller_size > std::numeric_limits<std::uint32_t>::max()) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                std::string(name) + " size is smaller than ABI v1 or exceeds uint32_t");
  }

  /* Never write beyond the layout known by this library version. */
  std::memset(value, 0, std::min(caller_size, sizeof(T)));
  value->struct_size = static_cast<std::uint32_t>(caller_size);
  value->api_version = XTBLOOM_API_VERSION;
  last_error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

/* Callback dimensions arrive through an internal C++ seam but ultimately
 * become pointer arithmetic in the C trampoline. Keep every multiplication
 * checked before converting to size_t or adding an offset, so malformed local
 * evaluator state cannot wrap an extent and expose unrelated memory. */
bool checked_product_i64(std::int64_t first, std::int64_t second, std::int64_t& result) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  result = first * second;
  return true;
}

bool checked_sum_i64(std::int64_t first, std::int64_t second, std::int64_t& result) noexcept {
  if (first < 0 || second < 0 || first > std::numeric_limits<std::int64_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

bool checked_product_size(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first != 0u && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

template <typename T>
bool vector_has_size(const std::vector<T>& values, std::int64_t expected) noexcept {
  return expected >= 0 && static_cast<std::uint64_t>(expected) == values.size();
}

bool valid_projection_basis(const xtbloom::detail::common::BasisPlan& basis,
                            std::int64_t atom_count, std::int64_t& orbitals,
                            std::int64_t& shells) noexcept {
  std::int64_t expected_shells = 0;
  std::int64_t expected_orbitals = 0;
  if (basis.batch_size != 1 || atom_count <= 0 || basis.total_atoms != atom_count ||
      !checked_product_i64(atom_count, 36, expected_shells) ||
      !checked_product_i64(atom_count, 108, expected_orbitals) ||
      basis.total_shells != expected_shells || basis.total_orbitals != expected_orbitals ||
      basis.total_shells <= 0 || basis.total_orbitals <= 0) {
    return false;
  }
  shells = basis.total_shells;
  orbitals = basis.total_orbitals;
  std::int64_t atom_offset_elements = 0;
  std::int64_t shell_offset_elements = 0;
  if (!checked_sum_i64(atom_count, 1, atom_offset_elements) ||
      !checked_sum_i64(shells, 1, shell_offset_elements) ||
      !vector_has_size(basis.atom_offsets, 2) ||
      !vector_has_size(basis.atom_shell_offsets, atom_offset_elements) ||
      !vector_has_size(basis.atom_orbital_offsets, atom_offset_elements) ||
      !vector_has_size(basis.shell_orbital_offsets, shell_offset_elements) ||
      !vector_has_size(basis.shell_to_atom, shells) ||
      !vector_has_size(basis.angular_momenta, shells)) {
    return false;
  }
  if (basis.atom_offsets.front() != 0 || basis.atom_offsets.back() != atom_count ||
      basis.shell_orbital_offsets.front() != 0 || basis.shell_orbital_offsets.back() != orbitals) {
    return false;
  }
  for (std::int64_t shell = 0; shell < shells; ++shell) {
    const auto index = static_cast<std::size_t>(shell);
    const std::int64_t begin = basis.shell_orbital_offsets[index];
    const std::int64_t end = basis.shell_orbital_offsets[index + 1u];
    if (begin < 0 || end <= begin || end > orbitals || basis.shell_to_atom[index] < 0 ||
        basis.shell_to_atom[index] >= atom_count || basis.angular_momenta[index] > 2u) {
      return false;
    }
  }
  return true;
}

xtbloom_status_t external_energy_callback_trampoline(
    void* opaque, const xtbloom::detail::gfn2::ExternalEnergyInput& input,
    xtbloom::detail::gfn2::ExternalEnergyOutput& output, std::string& error) {
  auto* context = static_cast<xtbloom::detail::Context*>(opaque);
  if (context == nullptr || context->external_energy_callback == nullptr ||
      input.wavefunction_layout == nullptr || input.wavefunction == nullptr ||
      input.integrals == nullptr || input.system < 0 ||
      input.system >= input.wavefunction_layout->batch_size) {
    error = "external energy callback context or SCC input is invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const auto& layout = *input.wavefunction_layout;
  const auto& wavefunction = *input.wavefunction;
  const auto& integrals = *input.integrals;
  const std::size_t system = static_cast<std::size_t>(input.system);
  if (system + 1u >= layout.atom_offsets.size() ||
      system + 1u >= layout.batch_orbital_offsets.size() ||
      system + 1u >= layout.batch_shell_offsets.size() || system >= layout.spin_channels.size() ||
      system + 1u >= layout.density.system_offsets.size()) {
    error = "external energy callback wavefunction partitions are incomplete";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::int64_t atom_begin = layout.atom_offsets[system];
  const std::int64_t atom_end = layout.atom_offsets[system + 1u];
  const std::int64_t orbital_begin = layout.batch_orbital_offsets[system];
  const std::int64_t orbital_end = layout.batch_orbital_offsets[system + 1u];
  const std::int64_t atom_count = atom_end - atom_begin;
  const std::int64_t nao = orbital_end - orbital_begin;
  const std::int64_t shell_begin = layout.batch_shell_offsets[system];
  const std::int64_t shell_end = layout.batch_shell_offsets[system + 1u];
  const std::int64_t shell_count = shell_end - shell_begin;
  const std::int32_t spin_channels = layout.spin_channels[system];
  const auto* projection_basis = input.projection_basis;
  std::int64_t projection_orbitals = 0;
  std::int64_t projection_shell_count = 0;
  const bool projection_valid = projection_basis != nullptr &&
                                valid_projection_basis(*projection_basis, atom_count,
                                                       projection_orbitals, projection_shell_count);
  std::int64_t positions_elements = 0;
  std::int64_t shell_offset_elements = 0;
  std::int64_t matrix_elements_i64 = 0;
  std::int64_t density_elements_i64 = 0;
  std::int64_t projection_matrix_elements = 0;
  std::int64_t projection_gradient_elements = 0;
  std::int64_t overlap_gradient_elements = 0;
  const bool dimensions_valid =
      checked_product_i64(3, atom_count, positions_elements) &&
      checked_sum_i64(shell_count, 1, shell_offset_elements) &&
      checked_product_i64(nao, nao, matrix_elements_i64) &&
      checked_product_i64(matrix_elements_i64, spin_channels, density_elements_i64) &&
      checked_product_i64(positions_elements, matrix_elements_i64, overlap_gradient_elements) &&
      projection_valid &&
      checked_product_i64(nao, projection_orbitals, projection_matrix_elements) &&
      checked_product_i64(3, atom_count, projection_gradient_elements) &&
      checked_product_i64(projection_gradient_elements, projection_matrix_elements,
                          projection_gradient_elements);
  if (atom_count <= 0 || nao <= 0 || (spin_channels != 1 && spin_channels != 2) ||
      wavefunction.density == nullptr || input.atomic_numbers == nullptr ||
      input.atomic_number_elements != atom_count || !dimensions_valid ||
      input.positions == nullptr || input.position_elements != positions_elements ||
      input.orbital_to_atom == nullptr || input.orbital_to_atom_elements != nao ||
      input.shell_orbital_offsets == nullptr ||
      input.shell_orbital_offset_elements != shell_offset_elements ||
      input.shell_to_atom == nullptr || input.shell_to_atom_elements != shell_count ||
      input.principal_quantum_numbers == nullptr ||
      input.principal_quantum_number_elements != shell_count || input.angular_momenta == nullptr ||
      input.angular_momentum_elements != shell_count || input.shell_index_begin != shell_begin ||
      input.shell_orbital_index_begin != orbital_begin || !projection_valid ||
      projection_shell_count <= 0 || input.projection_overlap == nullptr ||
      input.projection_overlap_elements != projection_matrix_elements ||
      (input.projection_overlap_gradient_elements != 0 &&
       input.projection_overlap_gradient == nullptr) ||
      (input.projection_overlap_gradient != nullptr &&
       input.projection_overlap_gradient_elements != projection_gradient_elements) ||
      input.mulliken == nullptr) {
    error = "external energy callback encountered an invalid wavefunction layout";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::size_t matrix_offset = 0u;
  for (std::size_t previous = 0u; previous < system; ++previous) {
    const std::int64_t previous_nao =
        layout.batch_orbital_offsets[previous + 1u] - layout.batch_orbital_offsets[previous];
    std::size_t previous_matrix_elements = 0u;
    if (previous_nao <= 0 ||
        !checked_product_size(static_cast<std::size_t>(previous_nao),
                              static_cast<std::size_t>(previous_nao), previous_matrix_elements) ||
        matrix_offset > std::numeric_limits<std::size_t>::max() - previous_matrix_elements) {
      error = "external energy callback matrix offset overflowed";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    matrix_offset += previous_matrix_elements;
  }
  const std::size_t matrix_elements = static_cast<std::size_t>(matrix_elements_i64);
  if (matrix_offset > std::numeric_limits<std::size_t>::max() - matrix_elements) {
    error = "external energy callback matrix extent overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  const std::int64_t density_begin_i64 = layout.density.system_offsets[system];
  const std::int64_t density_end_i64 = layout.density.system_offsets[system + 1u];
  if (density_begin_i64 < 0 || density_end_i64 < density_begin_i64 ||
      static_cast<std::uint64_t>(density_end_i64) > std::numeric_limits<std::size_t>::max()) {
    error = "external energy callback density offsets are invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t density_begin = static_cast<std::size_t>(density_begin_i64);
  const std::size_t density_end = static_cast<std::size_t>(density_end_i64);
  const std::size_t density_elements = density_end - density_begin;
  std::size_t expected_density_elements = 0u;
  std::int64_t matrix_end_i64 = 0;
  if (matrix_offset > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      matrix_elements > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      !checked_sum_i64(static_cast<std::int64_t>(matrix_offset),
                       static_cast<std::int64_t>(matrix_elements), matrix_end_i64) ||
      !checked_product_size(matrix_elements, static_cast<std::size_t>(spin_channels),
                            expected_density_elements) ||
      density_elements != expected_density_elements || integrals.overlap == nullptr ||
      integrals.matrix_elements < matrix_end_i64 ||
      (input.overlap_gradient_elements != 0 && input.overlap_gradient == nullptr) ||
      (input.overlap_gradient != nullptr &&
       input.overlap_gradient_elements != overlap_gradient_elements)) {
    error = "external energy callback density or overlap layout is inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  double callback_energy = 0.0;
  const xtbloom_status_t status = context->external_energy_callback(
      context->external_energy_opaque, input.system,
      static_cast<xtbloom_external_energy_phase_t>(input.phase), spin_channels, atom_count, nao,
      input.atomic_numbers, input.atomic_number_elements, input.positions, input.position_elements,
      input.orbital_to_atom, input.orbital_to_atom_elements, input.atom_index_begin, shell_count,
      input.shell_orbital_offsets, input.shell_orbital_offset_elements, input.shell_to_atom,
      input.shell_to_atom_elements, input.principal_quantum_numbers,
      input.principal_quantum_number_elements, input.angular_momenta,
      input.angular_momentum_elements, input.shell_index_begin, input.shell_orbital_index_begin,
      input.molecular_charge, input.unpaired_electrons, wavefunction.density + density_begin,
      static_cast<std::int64_t>(density_elements), integrals.overlap + matrix_offset,
      static_cast<std::int64_t>(matrix_elements), input.overlap_gradient,
      input.overlap_gradient_elements, projection_orbitals, projection_shell_count,
      projection_basis->shell_orbital_offsets.data(), projection_shell_count + 1,
      projection_basis->shell_to_atom.data(), projection_shell_count,
      projection_basis->angular_momenta.data(), projection_shell_count, input.projection_overlap,
      input.projection_overlap_elements, input.projection_overlap_gradient,
      input.projection_overlap_gradient_elements, output.hamiltonian, output.hamiltonian_elements,
      output.force, output.force_elements, &callback_energy);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    error = "external-energy callback returned a failure status";
    return status;
  }
  output.energy = callback_energy;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

class CompletionSettlementGuard {
 public:
  explicit CompletionSettlementGuard(
      std::shared_ptr<xtbloom::detail::RequestCompletion> completion) noexcept
      : completion_(std::move(completion)) {}
  ~CompletionSettlementGuard() {
    if (completion_ != nullptr) completion_->settle_noexcept();
  }
  CompletionSettlementGuard(const CompletionSettlementGuard&) = delete;
  CompletionSettlementGuard& operator=(const CompletionSettlementGuard&) = delete;

  void dismiss() noexcept { completion_.reset(); }

 private:
  std::shared_ptr<xtbloom::detail::RequestCompletion> completion_;
};

}  // namespace

extern "C" {

const char* xtbloom_version_string(void) { return XTBLOOM_VERSION_STRING; }

const char* xtbloom_status_string(xtbloom_status_t status) {
  switch (status) {
    case XTBLOOM_STATUS_SUCCESS:
      return "success";
    case XTBLOOM_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case XTBLOOM_STATUS_BACKEND_UNAVAILABLE:
      return "backend unavailable";
    case XTBLOOM_STATUS_NOT_SUPPORTED:
      return "not supported";
    case XTBLOOM_STATUS_ALLOCATION_FAILED:
      return "allocation failed";
    case XTBLOOM_STATUS_NOT_IMPLEMENTED:
      return "not implemented";
    case XTBLOOM_STATUS_INTERNAL_ERROR:
      return "internal error";
    case XTBLOOM_STATUS_SCC_NOT_CONVERGED:
      return "SCC not converged";
    case XTBLOOM_STATUS_EIGENSOLVER_FAILED:
      return "eigensolver failed";
  }
  return "unknown status";
}

const char* xtbloom_get_last_error(void) { return last_error.c_str(); }

xtbloom_status_t xtbloom_context_options_init(xtbloom_context_options_t* options,
                                              size_t struct_size) {
  const xtbloom_status_t status = initialize_structure(
      options, struct_size, XTBLOOM_CONTEXT_OPTIONS_V1_SIZE, "context options");
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  options->backend = XTBLOOM_BACKEND_AUTO;
  options->device_id = -1;
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t xtbloom_batch_init(xtbloom_batch_t* batch, size_t struct_size) {
  return initialize_structure(batch, struct_size, XTBLOOM_BATCH_V1_SIZE, "batch");
}

xtbloom_status_t xtbloom_compute_options_init(xtbloom_compute_options_t* options,
                                              size_t struct_size) {
  const xtbloom_status_t status = initialize_structure(
      options, struct_size, XTBLOOM_COMPUTE_OPTIONS_V1_SIZE, "compute options");
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  options->model = XTBLOOM_MODEL_GFN2_XTB;
  options->flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES;
  options->max_scc_iterations = 250;
  options->charge_tolerance = 1.0e-6;
  options->energy_tolerance = 1.0e-8;
  options->electronic_temperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;
  if (struct_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE) {
    options->scc_start_mode = XTBLOOM_SCC_START_FRESH;
  }
  if (struct_size >= XTBLOOM_COMPUTE_OPTIONS_V3_SIZE) {
    options->scc_mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
    options->scc_mixer_history = 8;
    options->scc_mixer_damping = 0.4;
    options->determinism = XTBLOOM_DETERMINISM_DEFAULT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t xtbloom_batch_result_init(xtbloom_batch_result_t* result, size_t struct_size) {
  return initialize_structure(result, struct_size, XTBLOOM_BATCH_RESULT_V1_SIZE, "batch result");
}

xtbloom_status_t xtbloom_workspace_query_init(xtbloom_workspace_query_t* query,
                                              size_t struct_size) {
  return initialize_structure(query, struct_size, XTBLOOM_WORKSPACE_QUERY_V1_SIZE,
                              "workspace query");
}

xtbloom_status_t xtbloom_request_info_init(xtbloom_request_info_t* info, size_t struct_size) {
  return initialize_structure(info, struct_size, XTBLOOM_REQUEST_INFO_V1_SIZE, "request info");
}

xtbloom_status_t xtbloom_context_create(const xtbloom_context_options_t* options,
                                        xtbloom_context_t** context) {
  if (context == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "context output pointer is NULL");
  }
  *context = nullptr;
  if (!valid_header(options, XTBLOOM_CONTEXT_OPTIONS_V1_SIZE)) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "context options are NULL, too small, or use an unsupported API version");
  }
  if (options->reserved != 0) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "context options reserved field must be zero");
  }
  const std::uint32_t backend = raw_enum(options->backend);
  if (backend > XTBLOOM_BACKEND_ROCM) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "context options contain an unknown backend");
  }

  try {
    xtbloom::detail::Context* implementation = nullptr;
    std::string error;
    const xtbloom_status_t status =
        xtbloom::detail::create_context(*options, implementation, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return fail(status, std::move(error));
    }

    xtbloom_context_t* wrapper = new (std::nothrow) xtbloom_context_t{implementation};
    if (wrapper == nullptr) {
      delete implementation;
      return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate a context handle");
    }
    *context = wrapper;
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, "unknown exception while creating context");
  }
}

void xtbloom_context_destroy(xtbloom_context_t* context) {
  if (context == nullptr) {
    return;
  }
  delete context->implementation;
  delete context;
}

xtbloom_backend_t xtbloom_context_get_backend(const xtbloom_context_t* context) {
  if (context == nullptr || context->implementation == nullptr) {
    last_error = "context is NULL";
    return XTBLOOM_BACKEND_AUTO;
  }
  last_error.clear();
  return context->implementation->backend;
}

int32_t xtbloom_context_get_device_id(const xtbloom_context_t* context) {
  if (context == nullptr || context->implementation == nullptr) {
    last_error = "context is NULL";
    return -1;
  }
  last_error.clear();
  return context->implementation->device_id;
}

xtbloom_status_t xtbloom_context_set_external_energy_callback(
    xtbloom_context_t* context, xtbloom_external_energy_callback_t callback, void* opaque) {
  if (context == nullptr || context->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "context is NULL");
  }
  try {
    /* Serialize callback installation with CPU compute transactions. The
     * native SCC workers retain the callback through an entire synchronous
     * call, so changing it concurrently would otherwise race the evaluator's
     * opaque state and violate the callback lifetime contract. */
    std::lock_guard<std::mutex> transaction_lock(context->implementation->cpu_transaction_mutex);
    const auto previous_callback = context->implementation->external_energy_callback;
    void* const previous_opaque = context->implementation->external_energy_opaque;
    context->implementation->external_energy_callback = callback;
    context->implementation->external_energy_opaque = opaque;
    std::string error;
    const xtbloom_status_t status = xtbloom::detail::set_external_energy_callback(
        *context->implementation,
        callback == nullptr ? nullptr : &external_energy_callback_trampoline,
        callback == nullptr ? nullptr : context->implementation, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      context->implementation->external_energy_callback = previous_callback;
      context->implementation->external_energy_opaque = previous_opaque;
      return fail(status, std::move(error));
    }
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                "unknown exception while configuring external energy callback");
  }
}

xtbloom_status_t xtbloom_context_set_external_energy_device_model(
    xtbloom_context_t* context, const xtbloom_external_energy_device_model_t* model) {
  if (context == nullptr || context->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "context is NULL");
  }
#if !defined(XTBLOOM_HAS_CUDA)
  (void)model;
  return fail(XTBLOOM_STATUS_NOT_SUPPORTED,
              "external energy native device evaluator requires a CUDA build");
#else
  if (context->implementation->backend != XTBLOOM_BACKEND_CUDA ||
      context->implementation->gfn2_cuda_execution_cache == nullptr) {
    return fail(XTBLOOM_STATUS_NOT_SUPPORTED,
                "external energy native device evaluator requires a CUDA GFN2 context");
  }
  try {
    std::string error;
    const xtbloom_status_t status =
        context->implementation->gfn2_cuda_execution_cache->set_external_energy_device_model(model,
                                                                                             error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail(status, std::move(error));
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED,
                "failed to allocate the external energy native device model");
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                "unknown exception while configuring the external energy native device model");
  }
#endif
}

xtbloom_status_t xtbloom_context_copy_external_energy_device_gradients(xtbloom_context_t* context,
                                                                       double* destination,
                                                                       int64_t elements) {
  if (context == nullptr || context->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "context is NULL");
  }
#if !defined(XTBLOOM_HAS_CUDA)
  (void)destination;
  (void)elements;
  return fail(XTBLOOM_STATUS_NOT_SUPPORTED,
              "external energy native device gradients require a CUDA build");
#else
  if (context->implementation->backend != XTBLOOM_BACKEND_CUDA ||
      context->implementation->gfn2_cuda_execution_cache == nullptr) {
    return fail(XTBLOOM_STATUS_NOT_SUPPORTED,
                "external energy native device gradients require a CUDA GFN2 context");
  }
  try {
    std::string error;
    const xtbloom_status_t status =
        context->implementation->gfn2_cuda_execution_cache->copy_external_energy_device_gradients(
            destination, elements, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return fail(status, std::move(error));
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                "unknown exception while copying external energy device gradients");
  }
#endif
}

xtbloom_status_t xtbloom_request_create(xtbloom_context_t* context, xtbloom_request_t** request) {
  if (request == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "request output pointer is NULL");
  }
  *request = nullptr;
  if (context == nullptr || context->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "context is NULL");
  }
  try {
    xtbloom::detail::Request* implementation =
        new (std::nothrow) xtbloom::detail::Request(*context->implementation);
    if (implementation == nullptr) {
      return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate a request implementation");
    }
    xtbloom_request_t* wrapper = new (std::nothrow) xtbloom_request_t{implementation};
    if (wrapper == nullptr) {
      delete implementation;
      return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate a request handle");
    }
    *request = wrapper;
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate a request");
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, "unknown exception while creating a request");
  }
}

namespace {

xtbloom_status_t validate_request_info(xtbloom_request_info_t* info) {
  if (!valid_header(info, XTBLOOM_REQUEST_INFO_V1_SIZE)) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "request info is NULL, too small, or uses an unsupported API version");
  }
  if (info->reserved != 0u) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "request info reserved field must be zero");
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

xtbloom_status_t xtbloom_request_query(xtbloom_request_t* request, xtbloom_request_info_t* info) {
  if (request == nullptr || request->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "request is NULL");
  }
  const xtbloom_status_t status = validate_request_info(info);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  try {
    std::string error;
    const xtbloom_status_t query_status = request->implementation->query(false, *info, error);
    if (query_status != XTBLOOM_STATUS_SUCCESS) {
      return fail(query_status, std::move(error));
    }
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED,
                "failed to allocate while querying a request completion");
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                "unknown exception while querying a request completion");
  }
}

xtbloom_status_t xtbloom_request_wait(xtbloom_request_t* request, xtbloom_request_info_t* info) {
  if (request == nullptr || request->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "request is NULL");
  }
  const xtbloom_status_t status = validate_request_info(info);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  try {
    std::string error;
    const xtbloom_status_t wait_status = request->implementation->query(true, *info, error);
    if (wait_status != XTBLOOM_STATUS_SUCCESS) {
      return fail(wait_status, std::move(error));
    }
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED,
                "failed to allocate while waiting for a request completion");
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                "unknown exception while waiting for a request completion");
  }
}

const char* xtbloom_request_get_error(const xtbloom_request_t* request) {
  if (request == nullptr || request->implementation == nullptr) {
    last_error = "request is NULL";
    return nullptr;
  }
  last_error.clear();
  return request->implementation->error();
}

void xtbloom_request_destroy(xtbloom_request_t* request) {
  if (request == nullptr) {
    return;
  }
  delete request->implementation;
  delete request;
  last_error.clear();
}

xtbloom_status_t xtbloom_compute_enqueue(xtbloom_context_t* context, const xtbloom_batch_t* batch,
                                         const xtbloom_compute_options_t* options,
                                         const xtbloom_batch_result_t* result,
                                         xtbloom_request_t* request) {
  if (context == nullptr || context->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "context is NULL");
  }
  if (request == nullptr || request->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "request is NULL");
  }
  if (request->implementation->context() != context->implementation) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "request was created by a different context");
  }
#if !defined(XTBLOOM_HAS_CUDA)
  (void)batch;
  (void)options;
  (void)result;
  /* A CPU-only library can create only CPU contexts, so this is the complete
   * public behavior rather than a fallback after an unreachable CUDA path. */
  return fail(XTBLOOM_STATUS_NOT_SUPPORTED,
              "asynchronous compute enqueue is not supported by the CPU backend");
#else
  if (context->implementation->backend == XTBLOOM_BACKEND_CPU) {
    /* Do not inspect descriptors or touch request/result state: callers may
     * probe capability with sentinels and then fall back to xtbloom_compute. */
    return fail(XTBLOOM_STATUS_NOT_SUPPORTED,
                "asynchronous compute enqueue is not supported by the CPU backend");
  }
  if (context->implementation->external_energy_callback != nullptr &&
      (context->implementation->gfn2_cuda_execution_cache == nullptr ||
       !context->implementation->gfn2_cuda_execution_cache
            ->external_energy_device_model_enabled())) {
    /* Host-staged external energy CUDA contexts intentionally use the synchronous CPU
     * SCC cache.  Borrowed callback state cannot safely cross the existing
     * CUDA request graph lifetime, so reject enqueue without inspecting any
     * descriptors or mutating request/result state. */
    return fail(XTBLOOM_STATUS_NOT_SUPPORTED,
                "asynchronous CUDA enqueue is not supported with the host-staged external energy "
                "correction callback");
  }

  if (batch == nullptr || options == nullptr || result == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "batch, compute options, or batch result is NULL");
  }

  bool reserved = false;
  try {
    std::string error;
    const xtbloom::detail::DescriptorValidationResult validation =
        xtbloom::detail::validate_compute_descriptor_structure_for_dispatch(
            context->implementation->backend, batch, options, result);
    if (!validation.ok()) {
      return fail(validation.status, validation.error);
    }
    xtbloom::detail::ModelBackendRoute model_route =
        xtbloom::detail::ModelBackendRoute::kUnavailable;
    const xtbloom_status_t model_status = xtbloom::detail::validate_model_dispatch(
        options->model, context->implementation->backend, error, &model_route);
    if (model_status != XTBLOOM_STATUS_SUCCESS) {
      return fail(model_status, std::move(error));
    }
    if (model_route != xtbloom::detail::ModelBackendRoute::kGfn1 &&
        model_route != xtbloom::detail::ModelBackendRoute::kGfn2) {
      return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                  "the registered model route has no asynchronous CUDA executor");
    }
    const xtbloom::detail::DescriptorValidationResult availability =
        xtbloom::detail::validate_compute_execution_availability(context->implementation->backend,
                                                                 *batch, *options);
    if (!availability.ok()) {
      return fail(availability.status, availability.error);
    }
    const xtbloom_status_t reserve_status =
        request->implementation->reserve_submission(*context->implementation, error);
    if (reserve_status != XTBLOOM_STATUS_SUCCESS) {
      return fail(reserve_status, std::move(error));
    }
    reserved = true;

    const std::shared_ptr<xtbloom::detail::Gfn2CudaExecutionCache>& cache =
        context->implementation->gfn2_cuda_execution_cache;
    if (cache == nullptr) {
      request->implementation->rollback_submission();
      return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                  "CUDA context does not own a GFN2 execution cache");
    }
    xtbloom::detail::RequestSubmission submission;
    const xtbloom_status_t enqueue_status = xtbloom::detail::enqueue_restricted_gfn2_cuda(
        cache, *batch, *options, *result, submission, error);
    if (enqueue_status != XTBLOOM_STATUS_SUCCESS) {
      request->implementation->rollback_submission();
      return fail(enqueue_status, std::move(error));
    }
    /* Publication is the ownership handoff from the API stack to the request.
     * If an invariant or mutex operation fails here, settle the already
     * accepted CUDA transaction instead of leaving borrowed buffers in use. */
    CompletionSettlementGuard completion_guard(submission.pending);
    const xtbloom_status_t publish_status =
        request->implementation->publish_submission(std::move(submission), error);
    if (publish_status != XTBLOOM_STATUS_SUCCESS) {
      request->implementation->rollback_submission();
      reserved = false;
      return fail(publish_status, std::move(error));
    }
    completion_guard.dismiss();
    reserved = false;
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    if (reserved) request->implementation->rollback_submission();
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED,
                "failed to allocate temporary storage while enqueueing CUDA GFN2 inference");
  } catch (const std::exception& exception) {
    if (reserved) request->implementation->rollback_submission();
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    if (reserved) request->implementation->rollback_submission();
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                "unknown exception while enqueueing CUDA GFN2 inference");
  }
#endif
}

xtbloom_status_t xtbloom_compute(xtbloom_context_t* context, const xtbloom_batch_t* batch,
                                 const xtbloom_compute_options_t* options,
                                 xtbloom_batch_result_t* result) {
  if (context == nullptr || context->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "context is NULL");
  }
  /* A private callback on a CUDA context selects the compatibility
   * host-staged path unless a native device model is installed.  The latter
   * keeps the normal CUDA graph route and executes the external evaluator in the
   * SCC device tail, so no callback or host staging is involved. */
#if defined(XTBLOOM_HAS_CUDA)
  const bool external_energy_device =
      context->implementation->backend == XTBLOOM_BACKEND_CUDA &&
      context->implementation->gfn2_cuda_execution_cache != nullptr &&
      context->implementation->gfn2_cuda_execution_cache->external_energy_device_model_enabled();
#else
  const bool external_energy_device = false;
#endif
  const bool external_energy_host_staged =
      context->implementation->backend == XTBLOOM_BACKEND_CUDA &&
      context->implementation->external_energy_callback != nullptr && !external_energy_device;
  const bool cuda_backend =
      context->implementation->backend == XTBLOOM_BACKEND_CUDA && !external_energy_host_staged;
  std::unique_lock<std::mutex> cpu_transaction;
  try {
    if (context->implementation->backend == XTBLOOM_BACKEND_CPU || external_energy_host_staged) {
      /* Validation, model dispatch, cache mutation, and publication are one
       * context transaction even when concurrent callers select different
       * model caches. Keep acquisition inside the C ABI exception boundary:
       * std::mutex::lock may report an operating-system failure by throwing. */
      cpu_transaction =
          std::unique_lock<std::mutex>(context->implementation->cpu_transaction_mutex);
    }
    xtbloom::detail::DescriptorValidationResult validation =
        cuda_backend ? xtbloom::detail::validate_compute_descriptor_structure_for_dispatch(
                           context->implementation->backend, batch, options, result)
                     : xtbloom::detail::validate_compute_descriptors_for_dispatch(
                           XTBLOOM_BACKEND_CPU, batch, options, result);
    if (!validation.ok()) {
      return fail(validation.status, std::move(validation.error));
    }

    /* CUDA completes pointer-attribute and topology semantic validation under
     * the cache transaction before accessing caller storage. CPU retains the
     * historical complete host validation sequence here. */
    (void)validation.pending_offset_checks;

    std::string route_error;
    xtbloom::detail::ModelBackendRoute model_route =
        xtbloom::detail::ModelBackendRoute::kUnavailable;
    const xtbloom_status_t model_status = xtbloom::detail::validate_model_dispatch(
        options->model, cuda_backend ? context->implementation->backend : XTBLOOM_BACKEND_CPU,
        route_error, &model_route);
    if (model_status != XTBLOOM_STATUS_SUCCESS) {
      return fail(model_status, std::move(route_error));
    }
    if (model_route != xtbloom::detail::ModelBackendRoute::kGfn1 &&
        model_route != xtbloom::detail::ModelBackendRoute::kGfn2) {
      return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                  "the registered model route has no synchronous executor");
    }
    if (external_energy_host_staged && options->model != XTBLOOM_MODEL_GFN2_XTB) {
      return fail(
          XTBLOOM_STATUS_NOT_SUPPORTED,
          "the external energy host-staged callback is available only for CUDA GFN2 requests");
    }
    const xtbloom::detail::DescriptorValidationResult availability =
        xtbloom::detail::validate_compute_execution_availability(
            cuda_backend ? context->implementation->backend : XTBLOOM_BACKEND_CPU, *batch,
            *options);
    if (!availability.ok()) {
      return fail(availability.status, std::move(availability.error));
    }
    if (!cuda_backend) {
      const xtbloom::detail::DescriptorValidationResult lattice_availability =
          xtbloom::detail::validate_host_lattice_execution_availability(*batch, options->model);
      if (!lattice_availability.ok()) {
        return fail(lattice_availability.status, std::move(lattice_availability.error));
      }
    }
  } catch (const std::bad_alloc&) {
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED,
                "failed to allocate temporary storage while validating or dispatching a compute "
                "request");
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                "unknown exception while validating or dispatching a compute request");
  }

  if (!cuda_backend) {
    try {
      if (options->model == XTBLOOM_MODEL_GFN1_XTB) {
        std::string error;
        const xtbloom_status_t cache_status =
            xtbloom::detail::ensure_gfn1_cpu_execution_cache(*context->implementation, error);
        if (cache_status != XTBLOOM_STATUS_SUCCESS) {
          return fail(cache_status, std::move(error));
        }
        const std::shared_ptr<xtbloom::detail::Gfn1CpuExecutionCache>& cache =
            context->implementation->gfn1_cpu_execution_cache;
        if (cache == nullptr) {
          return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                      "CPU context does not own a GFN1 execution cache");
        }
        const xtbloom_status_t status =
            xtbloom::detail::execute_gfn1_cpu(*cache, *batch, *options, *result, error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return fail(status, std::move(error));
        }
        last_error.clear();
        return XTBLOOM_STATUS_SUCCESS;
      }
      std::string error;
      const xtbloom_status_t cache_status =
          xtbloom::detail::ensure_gfn2_cpu_execution_cache(*context->implementation, error);
      if (cache_status != XTBLOOM_STATUS_SUCCESS) {
        return fail(cache_status, std::move(error));
      }
      const std::shared_ptr<xtbloom::detail::Gfn2CpuExecutionCache>& cache =
          context->implementation->gfn2_cpu_execution_cache;
      if (cache == nullptr) {
        return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                    "CPU context does not own a GFN2 execution cache");
      }
      const xtbloom_status_t status =
          xtbloom::detail::execute_restricted_gfn2_cpu(*cache, *batch, *options, *result, error);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        return fail(status, std::move(error));
      }
      last_error.clear();
      return XTBLOOM_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
      return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate CPU model execution state");
    } catch (const std::exception& exception) {
      return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
    } catch (...) {
      return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                  "unknown exception while executing CPU model inference");
    }
  }

#if defined(XTBLOOM_HAS_CUDA)
  try {
    const std::shared_ptr<xtbloom::detail::Gfn2CudaExecutionCache>& cache =
        context->implementation->gfn2_cuda_execution_cache;
    if (cache == nullptr) {
      return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                  "CUDA context does not own a GFN2 execution cache");
    }
    std::string error;
    const xtbloom_status_t status =
        xtbloom::detail::execute_restricted_gfn2_cuda(*cache, *batch, *options, *result, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return fail(status, std::move(error));
    }
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate CUDA GFN2 execution state");
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                "unknown exception while executing CUDA GFN2 inference");
  }
#else
  return fail(XTBLOOM_STATUS_BACKEND_UNAVAILABLE,
              "the xtbloom library was built without CUDA support");
#endif
}

xtbloom_status_t xtbloom_plan_create(xtbloom_context_t* context, const xtbloom_batch_t* batch,
                                     const xtbloom_compute_options_t* options,
                                     xtbloom_plan_t** plan) {
  if (plan == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "plan output pointer is NULL");
  }
  *plan = nullptr;
  if (context == nullptr || context->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "context is NULL");
  }
  if (batch == nullptr || options == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "batch or compute options is NULL");
  }
  try {
    std::unique_ptr<xtbloom::detail::ModelPlan> implementation(new (std::nothrow)
                                                                   xtbloom::detail::ModelPlan{});
    if (implementation == nullptr) {
      return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate a plan implementation");
    }
    std::string error;
    const xtbloom_status_t status =
        implementation->create(*context->implementation, *batch, *options, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      implementation->destroy();
      return fail(status, std::move(error));
    }

    xtbloom_plan_t* wrapper = new (std::nothrow) xtbloom_plan_t{implementation.get()};
    if (wrapper == nullptr) {
      implementation->destroy();
      return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate a plan handle");
    }
    implementation.release();
    *plan = wrapper;
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate a plan");
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, "unknown exception while creating a plan");
  }
}

void xtbloom_plan_destroy(xtbloom_plan_t* plan) {
  if (plan == nullptr) {
    return;
  }
  if (plan->implementation != nullptr) {
    plan->implementation->destroy();
    delete plan->implementation;
  }
  delete plan;
}

xtbloom_status_t xtbloom_plan_query_workspace(const xtbloom_plan_t* plan,
                                              xtbloom_workspace_query_t* query) {
  if (plan == nullptr || plan->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "plan is NULL");
  }
  if (!valid_header(query, XTBLOOM_WORKSPACE_QUERY_V1_SIZE)) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "workspace query is NULL, too small, or uses an unsupported API version");
  }
  try {
    std::string error;
    const xtbloom_status_t status = plan->implementation->query_workspace(
        static_cast<std::uint32_t>(query->compute_flags), *query, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return fail(status, std::move(error));
    }
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, "unknown exception while querying plan workspace");
  }
}

xtbloom_status_t xtbloom_plan_compute(xtbloom_plan_t* plan, const xtbloom_batch_t* batch,
                                      const xtbloom_compute_options_t* options,
                                      xtbloom_batch_result_t* result) {
  if (plan == nullptr || plan->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "plan is NULL");
  }
  if (batch == nullptr || options == nullptr || result == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "batch, compute options, or batch result is NULL");
  }
  try {
    std::string error;
    const xtbloom_status_t status = plan->implementation->compute(*batch, *options, *result, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return fail(status, std::move(error));
    }
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED,
                "failed to allocate temporary storage while executing a plan compute request");
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                "unknown exception while executing a plan compute request");
  }
}

xtbloom_status_t xtbloom_plan_compute_enqueue(xtbloom_plan_t* plan, const xtbloom_batch_t* batch,
                                              const xtbloom_compute_options_t* options,
                                              const xtbloom_batch_result_t* result,
                                              xtbloom_request_t* request) {
  if (plan == nullptr || plan->implementation == nullptr || !plan->implementation->valid()) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "plan is NULL");
  }
  if (request == nullptr || request->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "request is NULL");
  }
  xtbloom::detail::Context* context = plan->implementation->context();
  if (context == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "plan has no creating context");
  }
  if (request->implementation->context() != context) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "request and plan were created by different contexts");
  }
  if (context->backend == XTBLOOM_BACKEND_CPU) {
    /* Match the context enqueue capability probe: no descriptor validation,
     * request transition, result flag update, or caller-buffer write. */
    return fail(XTBLOOM_STATUS_NOT_SUPPORTED,
                "asynchronous plan enqueue is not supported by the CPU backend");
  }
  if (batch == nullptr || options == nullptr || result == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "batch, compute options, or batch result is NULL");
  }

  bool reserved = false;
  try {
    std::string error;
    const xtbloom_status_t reserve_status =
        request->implementation->reserve_submission(*context, error);
    if (reserve_status != XTBLOOM_STATUS_SUCCESS) {
      return fail(reserve_status, std::move(error));
    }
    reserved = true;
    xtbloom::detail::RequestSubmission submission;
    const xtbloom_status_t enqueue_status =
        plan->implementation->enqueue(*batch, *options, *result, submission, error);
    if (enqueue_status != XTBLOOM_STATUS_SUCCESS) {
      request->implementation->rollback_submission();
      return fail(enqueue_status, std::move(error));
    }
    /* Keep an allocation-free guard until the request state owns completion.
     * If mutex acquisition or an invariant check fails during publication,
     * the accepted cache transaction must still be settled. */
    CompletionSettlementGuard completion_guard(submission.pending);
    const xtbloom_status_t publish_status =
        request->implementation->publish_submission(std::move(submission), error);
    if (publish_status != XTBLOOM_STATUS_SUCCESS) {
      request->implementation->rollback_submission();
      reserved = false;
      return fail(publish_status, std::move(error));
    }
    completion_guard.dismiss();
    reserved = false;
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    if (reserved) request->implementation->rollback_submission();
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED,
                "failed to allocate CUDA plan request submission state");
  } catch (const std::exception& exception) {
    if (reserved) request->implementation->rollback_submission();
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    if (reserved) request->implementation->rollback_submission();
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                "unknown exception while enqueueing a CUDA plan request");
  }
}

namespace {

/*
 * The exported managed tensor is one allocation: the managed-tensor struct
 * followed by the copied shape storage. manager_ctx holds the arena owner
 * (retained once per export). The deleter is a plain native function that
 * importing frameworks may call from any thread after the Python wrapper is
 * gone, so it never touches Python state.
 */
void* dlpack_export_block(std::size_t managed_size, std::size_t ndim) {
  const std::size_t shape_bytes = ndim * sizeof(std::int64_t);
  if (shape_bytes > std::numeric_limits<std::size_t>::max() - managed_size) {
    return nullptr;
  }
  return ::operator new(managed_size + shape_bytes, std::nothrow);
}

/*
 * Both deleters receive the managed tensor whose manager_ctx is the public
 * wrapper handle. Releasing the wrapper may be the final reference, in which
 * case the wrapper must be freed here because the Python producer has already
 * closed: importing frameworks can call this deleter from any thread and long
 * after the producer is gone.
 */
xtbloom_result_owner_t* wrapper_from_manager_ctx(const void* manager_ctx) noexcept {
  return static_cast<xtbloom_result_owner_t*>(const_cast<void*>(manager_ctx));
}

void finish_dlpack_deletion(xtbloom_result_owner_t* wrapper, void* block) noexcept {
  xtbloom::detail::ResultOwner* implementation = wrapper->implementation;
  const bool final = implementation->release();
  ::operator delete(block);
  if (final) {
    delete wrapper;
  }
}

void legacy_dlpack_deleter(xtbloom::detail::DlpackManagedTensor* self) {
  if (self == nullptr) {
    return;
  }
  finish_dlpack_deletion(wrapper_from_manager_ctx(self->manager_ctx), self);
}

void versioned_dlpack_deleter(xtbloom::detail::DlpackManagedTensorVersioned* self) {
  if (self == nullptr) {
    return;
  }
  finish_dlpack_deletion(wrapper_from_manager_ctx(self->manager_ctx), self);
}

xtbloom_status_t populate_dlpack_view(xtbloom_result_owner_t* wrapper,
                                      xtbloom::detail::ResultOwner* owner,
                                      const xtbloom_dlpack_view_t* view, bool versioned,
                                      void** out_managed) {
  if (view == nullptr || out_managed == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "DLPack export view or output pointer is NULL");
  }
  if (!valid_header(view, XTBLOOM_DLPACK_VIEW_V1_SIZE)) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "DLPack view is NULL, too small, or uses an unsupported API version");
  }
  if (view->reserved != 0) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "DLPack view reserved field must be zero");
  }
  if (view->ndim < 0 || view->ndim > XTBLOOM_DLPACK_MAX_NDIM) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "DLPack view ndim must lie between 0 and XTBLOOM_DLPACK_MAX_NDIM");
  }
  if (view->dtype_lanes != 1) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "DLPack view lanes must be 1 (scalar descriptors only)");
  }
  const std::size_t dtype_size =
      xtbloom::detail::dlpack_dtype_size(view->dtype_code, view->dtype_bits);
  if (dtype_size == 0u) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "DLPack view uses an unsupported dtype code/bits combination");
  }
  if (view->ndim > 0 && view->shape == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "DLPack view with ndim > 0 requires a shape pointer");
  }

  /* Validate extents before touching the arena or the output pointer. */
  std::uint64_t element_count = 1u;
  const std::int64_t* shape = view->shape;
  for (std::int32_t index = 0; index < view->ndim; ++index) {
    if (shape[index] < 0) {
      return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "DLPack view has a negative shape extent");
    }
    if (shape[index] == 0) {
      /* Once one extent is zero the tensor has no elements; keep validating
       * the remaining extents without dividing by a zero element count. */
      element_count = 0u;
      continue;
    }
    if (element_count == 0u) {
      continue;
    }
    if (static_cast<std::uint64_t>(shape[index]) >
        std::numeric_limits<std::uint64_t>::max() / element_count) {
      return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "DLPack view shape overflows element count");
    }
    element_count *= static_cast<std::uint64_t>(shape[index]);
  }
  std::uint64_t payload_bytes = 0u;
  if (element_count != 0u) {
    if (element_count > std::numeric_limits<std::uint64_t>::max() / dtype_size) {
      return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "DLPack view payload size overflows uint64_t");
    }
    payload_bytes = element_count * dtype_size;
  }
  const std::uint64_t arena_size = static_cast<std::uint64_t>(owner->size_bytes());
  if (view->byte_offset > arena_size) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "DLPack view starts past the arena end");
  }
  if (payload_bytes > arena_size - view->byte_offset) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "DLPack view payload extends past the arena end");
  }
  if (payload_bytes != 0u) {
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(owner->data());
    const std::uintptr_t address = base + static_cast<std::uintptr_t>(view->byte_offset);
    if (address % dtype_size != 0u) {
      return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                  "DLPack view offset does not preserve the dtype alignment");
    }
  }

  const std::size_t managed_size = versioned ? sizeof(xtbloom::detail::DlpackManagedTensorVersioned)
                                             : sizeof(xtbloom::detail::DlpackManagedTensor);
  void* block = dlpack_export_block(managed_size, static_cast<std::size_t>(view->ndim));
  if (block == nullptr) {
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED,
                "failed to allocate a DLPack managed tensor export");
  }

  /* Transfer definitely: from here on the deleter owns the block and the
   * retained arena reference; a failure must not leave either behind. */
  std::int64_t* shape_storage =
      reinterpret_cast<std::int64_t*>(static_cast<unsigned char*>(block) + managed_size);
  void* data = payload_bytes == 0u ? nullptr
                                   : static_cast<unsigned char*>(owner->data()) + view->byte_offset;
  const std::int32_t device_id =
      owner->memory_space() == XTBLOOM_MEMORY_CUDA_DEVICE ? owner->device_id() : 0;
  const std::int32_t device_type = xtbloom::detail::dlpack_device_type(owner->memory_space());

  if (versioned) {
    xtbloom::detail::DlpackManagedTensorVersioned* managed =
        static_cast<xtbloom::detail::DlpackManagedTensorVersioned*>(block);
    managed->version_major = xtbloom::detail::kDlpackVersionMajor;
    managed->version_minor = xtbloom::detail::kDlpackVersionMinor;
    managed->manager_ctx = wrapper;
    managed->deleter = &versioned_dlpack_deleter;
    managed->flags = 0u;
    managed->dl_tensor.data = data;
    managed->dl_tensor.device.device_type = device_type;
    managed->dl_tensor.device.device_id = device_id;
    managed->dl_tensor.ndim = view->ndim;
    managed->dl_tensor.dtype.code = static_cast<std::uint8_t>(view->dtype_code);
    managed->dl_tensor.dtype.bits = static_cast<std::uint8_t>(view->dtype_bits);
    managed->dl_tensor.dtype.lanes = static_cast<std::uint16_t>(view->dtype_lanes);
    managed->dl_tensor.shape = shape_storage;
    managed->dl_tensor.strides = nullptr; /* compact row-major by convention */
    managed->dl_tensor.byte_offset = 0u;
    for (std::int32_t index = 0; index < view->ndim; ++index) {
      shape_storage[index] = shape[index];
    }
  } else {
    xtbloom::detail::DlpackManagedTensor* managed =
        static_cast<xtbloom::detail::DlpackManagedTensor*>(block);
    managed->manager_ctx = wrapper;
    managed->deleter = &legacy_dlpack_deleter;
    managed->dl_tensor.data = data;
    managed->dl_tensor.device.device_type = device_type;
    managed->dl_tensor.device.device_id = device_id;
    managed->dl_tensor.ndim = view->ndim;
    managed->dl_tensor.dtype.code = static_cast<std::uint8_t>(view->dtype_code);
    managed->dl_tensor.dtype.bits = static_cast<std::uint8_t>(view->dtype_bits);
    managed->dl_tensor.dtype.lanes = static_cast<std::uint16_t>(view->dtype_lanes);
    managed->dl_tensor.shape = shape_storage;
    managed->dl_tensor.strides = nullptr;
    managed->dl_tensor.byte_offset = 0u;
    for (std::int32_t index = 0; index < view->ndim; ++index) {
      shape_storage[index] = shape[index];
    }
  }

  owner->retain();
  *out_managed = block;
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

xtbloom_status_t xtbloom_result_owner_options_init(xtbloom_result_owner_options_t* options,
                                                   size_t struct_size) {
  const xtbloom_status_t status = initialize_structure(
      options, struct_size, XTBLOOM_RESULT_OWNER_OPTIONS_V1_SIZE, "result owner options");
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  options->memory_space = XTBLOOM_MEMORY_HOST;
  options->device_id = -1;
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t xtbloom_result_owner_create(const xtbloom_result_owner_options_t* options,
                                             xtbloom_result_owner_t** owner) {
  if (owner == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "result owner output pointer is NULL");
  }
  *owner = nullptr;
  if (!valid_header(options, XTBLOOM_RESULT_OWNER_OPTIONS_V1_SIZE)) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "result owner options are NULL, too small, or use an unsupported API version");
  }
  if (options->reserved != 0) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "result owner options reserved field must be zero");
  }
  if (options->memory_space != XTBLOOM_MEMORY_HOST &&
      options->memory_space != XTBLOOM_MEMORY_CUDA_DEVICE) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "result owner uses an unsupported memory space");
  }
  if ((options->memory_space == XTBLOOM_MEMORY_HOST && options->device_id != -1) ||
      (options->memory_space == XTBLOOM_MEMORY_CUDA_DEVICE && options->device_id < 0)) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "result owner device_id is inconsistent with its memory space");
  }
  if (options->size_bytes == 0u) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "result owner arena size must be nonzero");
  }
  if (options->size_bytes > std::numeric_limits<std::size_t>::max()) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "result owner arena size overflows size_t");
  }

  try {
    void* data = nullptr;
    std::string error;
    xtbloom_status_t status = XTBLOOM_STATUS_INVALID_ARGUMENT;
    if (options->memory_space == XTBLOOM_MEMORY_HOST) {
      status = xtbloom::detail::allocate_host_result_arena(
          static_cast<std::size_t>(options->size_bytes), &data, error);
    } else {
#if defined(XTBLOOM_HAS_CUDA)
      status = xtbloom::detail::allocate_cuda_result_arena(
          options->device_id, static_cast<std::size_t>(options->size_bytes), &data, error);
#else
      return fail(XTBLOOM_STATUS_BACKEND_UNAVAILABLE,
                  "the xtbloom library was built without CUDA support");
#endif
    }
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return fail(status, std::move(error));
    }

    xtbloom::detail::ResultOwner* implementation = new (std::nothrow)
        xtbloom::detail::ResultOwner(options->memory_space, options->device_id,
                                     static_cast<std::size_t>(options->size_bytes), data);
    if (implementation == nullptr) {
      if (options->memory_space == XTBLOOM_MEMORY_HOST) {
        xtbloom::detail::free_host_result_arena(data);
      } else {
#if defined(XTBLOOM_HAS_CUDA)
        xtbloom::detail::free_cuda_result_arena(options->device_id, data);
#endif
      }
      return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate a result owner handle");
    }
    xtbloom_result_owner_t* wrapper = new (std::nothrow) xtbloom_result_owner_t{implementation};
    if (wrapper == nullptr) {
      implementation->release();
      return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate a result owner wrapper");
    }
    *owner = wrapper;
    last_error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, "unknown exception while creating a result owner");
  }
}

xtbloom_status_t xtbloom_result_owner_buffer(const xtbloom_result_owner_t* owner,
                                             xtbloom_buffer_t* buffer) {
  if (owner == nullptr || owner->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "result owner is NULL");
  }
  if (buffer == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "result owner buffer output is NULL");
  }
  *buffer = {owner->implementation->data(), owner->implementation->size_bytes(),
             owner->implementation->memory_space(), 0u};
  last_error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

void xtbloom_result_owner_retain(xtbloom_result_owner_t* owner) {
  if (owner == nullptr || owner->implementation == nullptr) {
    last_error = "result owner is NULL";
    return;
  }
  owner->implementation->retain();
  last_error.clear();
}

void xtbloom_result_owner_release(xtbloom_result_owner_t* owner) {
  if (owner == nullptr || owner->implementation == nullptr) {
    /* The public release contract makes NULL a harmless no-op.  Preserve any
     * prior diagnostic so callers can still inspect the failing operation. */
    return;
  }
  xtbloom::detail::ResultOwner* implementation = owner->implementation;
  const bool final = implementation->release();
  last_error.clear();
  if (final) {
    /* ResultOwner::release() already destroyed the implementation. */
    delete owner;
  }
}

xtbloom_status_t xtbloom_result_owner_export_dltensor(const xtbloom_result_owner_t* owner,
                                                      const xtbloom_dlpack_view_t* view,
                                                      int version, void** out_managed) {
  /* On any failure *out_managed is set to NULL and no arena reference is
   * taken, so callers can treat a non-success status uniformly. */
  if (out_managed != nullptr) {
    *out_managed = nullptr;
  }
  if (owner == nullptr || owner->implementation == nullptr) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT, "result owner is NULL");
  }
  if (version != 0 && version != 1) {
    return fail(XTBLOOM_STATUS_INVALID_ARGUMENT,
                "DLPack export version must be 0 (legacy) or 1 (versioned)");
  }
  try {
    return populate_dlpack_view(const_cast<xtbloom_result_owner_t*>(owner), owner->implementation,
                                view, version != 0, out_managed);
  } catch (const std::bad_alloc&) {
    return fail(XTBLOOM_STATUS_ALLOCATION_FAILED, "failed to allocate a DLPack managed tensor");
  } catch (const std::exception& exception) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR, exception.what());
  } catch (...) {
    return fail(XTBLOOM_STATUS_INTERNAL_ERROR,
                "unknown exception while exporting a DLPack managed tensor");
  }
}

}  // extern "C"
