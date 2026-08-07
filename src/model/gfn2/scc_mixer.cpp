#include "model/gfn2/scc_mixer.hpp"
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace gpuxtb::detail::gfn2 {

struct SccMixerPlanData final {
  std::int64_t batch_size = 0;
  std::int64_t history_size = 0;
  std::int64_t total_vector_elements = 0;
  std::int64_t maximum_vector_elements = 0;
  double damping = 0.0;
  double rms_tolerance = 0.0;
  double maximum_tolerance = 0.0;

  std::size_t wavefunction_workspace_size_bytes = 0u;
  std::size_t qsh_offset_bytes = 0u;
  std::size_t dipole_offset_bytes = 0u;
  std::size_t quadrupole_offset_bytes = 0u;
  std::size_t state_size_bytes = 0u;
  std::size_t workspace_size_bytes = 0u;

  std::size_t current_input_offset_bytes = 0u;
  std::size_t previous_input_offset_bytes = 0u;
  std::size_t previous_residual_offset_bytes = 0u;
  std::size_t df_history_offset_bytes = 0u;
  std::size_t u_history_offset_bytes = 0u;
  std::size_t omega_offset_bytes = 0u;
  std::size_t residual_rms_offset_bytes = 0u;
  std::size_t residual_maximum_offset_bytes = 0u;
  std::size_t iteration_offset_bytes = 0u;
  std::size_t restart_count_offset_bytes = 0u;
  std::size_t system_status_offset_bytes = 0u;
  std::size_t initialized_offset_bytes = 0u;
  std::size_t converged_offset_bytes = 0u;

  std::size_t residual_scratch_offset_bytes = 0u;
  std::size_t mixed_scratch_offset_bytes = 0u;
  std::size_t delta_f_scratch_offset_bytes = 0u;
  std::size_t new_u_scratch_offset_bytes = 0u;
  std::size_t beta_scratch_offset_bytes = 0u;
  std::size_t coefficient_scratch_offset_bytes = 0u;
  std::size_t history_slot_scratch_offset_bytes = 0u;

  std::vector<std::int64_t> vector_offsets;
  std::vector<std::int64_t> history_offsets;
  std::vector<std::int64_t> qsh_system_offsets;
  std::vector<std::int64_t> dipole_system_offsets;
  std::vector<std::int64_t> quadrupole_system_offsets;
};

namespace {

static_assert(std::is_trivially_copyable_v<SccMixerState>);
static_assert(std::is_standard_layout_v<SccMixerState>);
static_assert(std::is_trivially_copyable_v<SccMixerWorkspace>);
static_assert(std::is_standard_layout_v<SccMixerWorkspace>);

constexpr double kOmegaZero = 0.01;
constexpr double kMinimumOmega = 1.0;
constexpr double kMaximumOmega = 100000.0;
constexpr double kOmegaFactor = 0.01;

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) {
  if (pointer == nullptr || bytes == 0u) {
    return false;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.begin < second.end && second.begin < first.end;
}

template <std::size_t N>
bool pairwise_disjoint(const std::array<AddressRange, N>& ranges) {
  for (std::size_t first = 0u; first < N; ++first) {
    for (std::size_t second = first + 1u; second < N; ++second) {
      if (ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

bool is_aligned(const void* pointer, std::size_t alignment) {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

bool checked_add_size(std::size_t increment, std::size_t& value) {
  if (value > std::numeric_limits<std::size_t>::max() - increment) {
    return false;
  }
  value += increment;
  return true;
}

bool checked_multiply_size(std::size_t first, std::size_t second, std::size_t& product) {
  if (first != 0u && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  product = first * second;
  return true;
}

bool checked_add_i64(std::int64_t increment, std::int64_t& value) {
  if (increment < 0 || value > std::numeric_limits<std::int64_t>::max() - increment) {
    return false;
  }
  value += increment;
  return true;
}

bool checked_multiply_i64(std::int64_t first, std::int64_t second, std::int64_t& product) {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  product = first * second;
  return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t& aligned) {
  const std::size_t remainder = value % alignment;
  const std::size_t padding = remainder == 0u ? 0u : alignment - remainder;
  aligned = value;
  return checked_add_size(padding, aligned);
}

bool append_segment(std::size_t bytes, std::size_t alignment, std::size_t& cursor,
                    std::size_t& offset) {
  if (!align_up(cursor, alignment, cursor)) {
    return false;
  }
  offset = cursor;
  return checked_add_size(bytes, cursor);
}

template <typename T>
T* offset_pointer(void* base, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

template <typename T>
const T* offset_pointer(const void* base, std::size_t offset) {
  return reinterpret_cast<const T*>(static_cast<const std::byte*>(base) + offset);
}

bool vector_range(const void* pointer, std::size_t count, std::size_t element_size,
                  AddressRange& range) {
  std::size_t bytes = 0u;
  return checked_multiply_size(count, element_size, bytes) && make_range(pointer, bytes, range);
}

bool overlaps_plan_storage(const SccMixerPlan& plan, const AddressRange& range) {
  const SccMixerPlanData* const data = plan.identity();
  if (data == nullptr) {
    return true;
  }
  AddressRange candidate;
  if (!make_range(data, sizeof(*data), candidate) || ranges_overlap(range, candidate)) {
    return true;
  }
  const std::array<const std::vector<std::int64_t>*, 5> vectors{
      {&data->vector_offsets, &data->history_offsets, &data->qsh_system_offsets,
       &data->dipole_system_offsets, &data->quadrupole_system_offsets}};
  for (const auto* vector : vectors) {
    if (!vector->empty() &&
        (!vector_range(vector->data(), vector->capacity(), sizeof(std::int64_t), candidate) ||
         ranges_overlap(range, candidate))) {
      return true;
    }
  }
  return false;
}

gpuxtb_status_t validate_plan(const SccMixerPlan& plan, std::string& error) {
  if (!plan.sealed()) {
    error = "SCC mixer plan is default-constructed, moved-from, or otherwise unsealed";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

bool exact_pointer(const void* base, std::size_t offset, const void* candidate) {
  return candidate == static_cast<const void*>(static_cast<const std::byte*>(base) + offset);
}

gpuxtb_status_t validate_state(const SccMixerPlan& plan, const SccMixerState& state,
                               std::string& error) {
  const SccMixerPlanData& data = *plan.identity();
  AddressRange range;
  if (!is_aligned(state.workspace_base, kSccMixerWorkspaceAlignment) ||
      state.workspace_size_bytes < data.state_size_bytes ||
      !make_range(state.workspace_base, data.state_size_bytes, range) ||
      state.plan_identity != &data ||
      !exact_pointer(state.workspace_base, data.current_input_offset_bytes, state.current_inputs) ||
      !exact_pointer(state.workspace_base, data.previous_input_offset_bytes,
                     state.previous_inputs) ||
      !exact_pointer(state.workspace_base, data.previous_residual_offset_bytes,
                     state.previous_residuals) ||
      !exact_pointer(state.workspace_base, data.df_history_offset_bytes, state.df_history) ||
      !exact_pointer(state.workspace_base, data.u_history_offset_bytes, state.u_history) ||
      !exact_pointer(state.workspace_base, data.omega_offset_bytes, state.omega) ||
      !exact_pointer(state.workspace_base, data.residual_rms_offset_bytes, state.residual_rms) ||
      !exact_pointer(state.workspace_base, data.residual_maximum_offset_bytes,
                     state.residual_maximum) ||
      !exact_pointer(state.workspace_base, data.iteration_offset_bytes, state.iterations) ||
      !exact_pointer(state.workspace_base, data.restart_count_offset_bytes, state.restart_counts) ||
      !exact_pointer(state.workspace_base, data.system_status_offset_bytes,
                     state.system_statuses) ||
      !exact_pointer(state.workspace_base, data.initialized_offset_bytes, state.initialized) ||
      !exact_pointer(state.workspace_base, data.converged_offset_bytes, state.converged)) {
    error = "SCC mixer state is malformed or belongs to a different plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_workspace(const SccMixerPlan& plan, const SccMixerWorkspace& workspace,
                                   std::string& error) {
  const SccMixerPlanData& data = *plan.identity();
  AddressRange range;
  if (!is_aligned(workspace.workspace_base, kSccMixerWorkspaceAlignment) ||
      workspace.workspace_size_bytes < data.workspace_size_bytes ||
      !make_range(workspace.workspace_base, data.workspace_size_bytes, range) ||
      workspace.plan_identity != &data ||
      !exact_pointer(workspace.workspace_base, data.residual_scratch_offset_bytes,
                     workspace.residual) ||
      !exact_pointer(workspace.workspace_base, data.mixed_scratch_offset_bytes, workspace.mixed) ||
      !exact_pointer(workspace.workspace_base, data.delta_f_scratch_offset_bytes,
                     workspace.delta_f) ||
      !exact_pointer(workspace.workspace_base, data.new_u_scratch_offset_bytes, workspace.new_u) ||
      !exact_pointer(workspace.workspace_base, data.beta_scratch_offset_bytes, workspace.beta) ||
      !exact_pointer(workspace.workspace_base, data.coefficient_scratch_offset_bytes,
                     workspace.coefficients) ||
      !exact_pointer(workspace.workspace_base, data.history_slot_scratch_offset_bytes,
                     workspace.history_slots)) {
    error = "SCC mixer scratch is malformed or belongs to a different plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_wavefunction(const SccMixerPlan& plan,
                                      const WavefunctionView& wavefunction, std::string& error) {
  const SccMixerPlanData& data = *plan.identity();
  AddressRange range;
  if (!is_aligned(wavefunction.workspace_base, kWavefunctionWorkspaceAlignment) ||
      wavefunction.workspace_size_bytes < data.wavefunction_workspace_size_bytes ||
      !make_range(wavefunction.workspace_base, data.wavefunction_workspace_size_bytes, range) ||
      !exact_pointer(wavefunction.workspace_base, data.qsh_offset_bytes, wavefunction.qsh) ||
      !exact_pointer(wavefunction.workspace_base, data.dipole_offset_bytes, wavefunction.dipole) ||
      !exact_pointer(wavefunction.workspace_base, data.quadrupole_offset_bytes,
                     wavefunction.quadrupole)) {
    error = "SCC mixer wavefunction is not the canonical binding sealed by its plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_active_ranges(const SccMixerPlan& plan,
                                       const WavefunctionView& wavefunction,
                                       const SccMixerState& state,
                                       const SccMixerWorkspace* workspace, std::string& error) {
  const SccMixerPlanData& data = *plan.identity();
  AddressRange wavefunction_range;
  AddressRange state_range;
  AddressRange plan_descriptor;
  AddressRange wavefunction_descriptor;
  AddressRange state_descriptor;
  AddressRange error_descriptor;
  if (!make_range(wavefunction.workspace_base, data.wavefunction_workspace_size_bytes,
                  wavefunction_range) ||
      !make_range(state.workspace_base, data.state_size_bytes, state_range) ||
      !make_range(&plan, sizeof(plan), plan_descriptor) ||
      !make_range(&wavefunction, sizeof(wavefunction), wavefunction_descriptor) ||
      !make_range(&state, sizeof(state), state_descriptor) ||
      !make_range(&error, sizeof(error), error_descriptor) ||
      ranges_overlap(wavefunction_range, state_range) ||
      overlaps_plan_storage(plan, wavefunction_range) || overlaps_plan_storage(plan, state_range)) {
    error = "SCC mixer wavefunction, state, plan, and descriptors must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::array<AddressRange, 4> controls{
      {plan_descriptor, wavefunction_descriptor, state_descriptor, error_descriptor}};
  for (const AddressRange& active : {wavefunction_range, state_range}) {
    for (const AddressRange& control : controls) {
      if (ranges_overlap(active, control)) {
        error = "SCC mixer numerical storage overlaps a control object";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  if (workspace != nullptr) {
    AddressRange workspace_range;
    AddressRange workspace_descriptor;
    if (!make_range(workspace->workspace_base, data.workspace_size_bytes, workspace_range) ||
        !make_range(workspace, sizeof(*workspace), workspace_descriptor) ||
        ranges_overlap(workspace_range, wavefunction_range) ||
        ranges_overlap(workspace_range, state_range) ||
        overlaps_plan_storage(plan, workspace_range) ||
        ranges_overlap(workspace_range, plan_descriptor) ||
        ranges_overlap(workspace_range, wavefunction_descriptor) ||
        ranges_overlap(workspace_range, state_descriptor) ||
        ranges_overlap(workspace_range, error_descriptor) ||
        ranges_overlap(wavefunction_range, workspace_descriptor) ||
        ranges_overlap(state_range, workspace_descriptor)) {
      error = "SCC mixer scratch overlaps active numerical or control storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_call(const SccMixerPlan& plan, const WavefunctionView& wavefunction,
                              const SccMixerState& state, const SccMixerWorkspace* workspace,
                              std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS ||
      (status = validate_state(plan, state, error)) != GPUXTB_STATUS_SUCCESS ||
      (workspace != nullptr &&
       (status = validate_workspace(plan, *workspace, error)) != GPUXTB_STATUS_SUCCESS) ||
      (status = validate_wavefunction(plan, wavefunction, error)) != GPUXTB_STATUS_SUCCESS ||
      (status = validate_active_ranges(plan, wavefunction, state, workspace, error)) !=
          GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_transaction(const SccMixerPlan& plan, const SccMixerState& source,
                                     const SccMixerState& staged, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS ||
      (status = validate_state(plan, source, error)) != GPUXTB_STATUS_SUCCESS ||
      (status = validate_state(plan, staged, error)) != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccMixerPlanData& data = *plan.identity();
  AddressRange source_range;
  AddressRange staged_range;
  AddressRange plan_descriptor;
  AddressRange source_descriptor;
  AddressRange staged_descriptor;
  AddressRange error_descriptor;
  if (!make_range(source.workspace_base, data.state_size_bytes, source_range) ||
      !make_range(staged.workspace_base, data.state_size_bytes, staged_range) ||
      !make_range(&plan, sizeof(plan), plan_descriptor) ||
      !make_range(&source, sizeof(source), source_descriptor) ||
      !make_range(&staged, sizeof(staged), staged_descriptor) ||
      !make_range(&error, sizeof(error), error_descriptor) ||
      ranges_overlap(source_range, staged_range) || overlaps_plan_storage(plan, source_range) ||
      overlaps_plan_storage(plan, staged_range)) {
    error = "SCC mixer transaction source and staged storage must be disjoint and unaliased";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::array<AddressRange, 4> controls{
      {plan_descriptor, source_descriptor, staged_descriptor, error_descriptor}};
  for (const AddressRange& active : {source_range, staged_range}) {
    for (const AddressRange& control : controls) {
      if (ranges_overlap(active, control)) {
        error = "SCC mixer transaction storage overlaps a control object";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

std::size_t system_index(std::int64_t system) { return static_cast<std::size_t>(system); }

std::size_t system_dimension(const SccMixerPlanData& data, std::size_t system) {
  return static_cast<std::size_t>(data.vector_offsets[system + 1u] - data.vector_offsets[system]);
}

std::size_t system_vector_offset(const SccMixerPlanData& data, std::size_t system) {
  return static_cast<std::size_t>(data.vector_offsets[system]);
}

std::size_t system_history_offset(const SccMixerPlanData& data, std::size_t system) {
  return static_cast<std::size_t>(data.history_offsets[system]);
}

template <typename Function>
void for_each_raw_component(const SccMixerPlanData& data, const WavefunctionView& wavefunction,
                            std::size_t system, Function&& function) {
  std::size_t packed = 0u;
  const std::array<const double*, 3> fields{
      {wavefunction.qsh, wavefunction.dipole, wavefunction.quadrupole}};
  const std::array<const std::vector<std::int64_t>*, 3> offsets{
      {&data.qsh_system_offsets, &data.dipole_system_offsets, &data.quadrupole_system_offsets}};
  for (std::size_t field = 0u; field < fields.size(); ++field) {
    const std::size_t begin = static_cast<std::size_t>((*offsets[field])[system]);
    const std::size_t end = static_cast<std::size_t>((*offsets[field])[system + 1u]);
    for (std::size_t source = begin; source < end; ++source, ++packed) {
      function(packed, fields[field][source]);
    }
  }
}

void publish_mixed_components(const SccMixerPlanData& data, const WavefunctionView& wavefunction,
                              std::size_t system, const double* mixed) {
  std::size_t packed = 0u;
  const std::array<double*, 3> fields{
      {wavefunction.qsh, wavefunction.dipole, wavefunction.quadrupole}};
  const std::array<const std::vector<std::int64_t>*, 3> offsets{
      {&data.qsh_system_offsets, &data.dipole_system_offsets, &data.quadrupole_system_offsets}};
  for (std::size_t field = 0u; field < fields.size(); ++field) {
    const std::size_t begin = static_cast<std::size_t>((*offsets[field])[system]);
    const std::size_t end = static_cast<std::size_t>((*offsets[field])[system + 1u]);
    for (std::size_t destination = begin; destination < end; ++destination, ++packed) {
      fields[field][destination] = mixed[packed];
    }
  }
}

bool raw_components_are_finite(const SccMixerPlanData& data, const WavefunctionView& wavefunction,
                               std::size_t system) {
  bool finite = true;
  for_each_raw_component(data, wavefunction, system, [&](std::size_t, double value) {
    finite = finite && std::isfinite(value);
  });
  return finite;
}

void copy_raw_components(const SccMixerPlanData& data, const WavefunctionView& wavefunction,
                         std::size_t system, double* destination) {
  for_each_raw_component(data, wavefunction, system,
                         [&](std::size_t packed, double value) { destination[packed] = value; });
}

/*
 * Copy every mixer record that belongs to exactly one system between two
 * full-layout bindings. The history buffers, omega tiles, and the batch-major
 * scalar records of other systems are never read or written, which is what
 * lets a per-system transaction cost scale with active-system history instead
 * of the total batch history.
 */
void copy_mixer_system_state(const SccMixerPlanData& data, std::size_t system,
                             const SccMixerState& source, const SccMixerState& destination) {
  const std::size_t dimension = system_dimension(data, system);
  const std::size_t vector_offset = system_vector_offset(data, system);
  const std::size_t history_offset = system_history_offset(data, system);
  const std::size_t memory = static_cast<std::size_t>(data.history_size);
  const std::size_t history_elements = dimension * memory;
  std::copy_n(source.current_inputs + vector_offset, dimension,
              destination.current_inputs + vector_offset);
  std::copy_n(source.previous_inputs + vector_offset, dimension,
              destination.previous_inputs + vector_offset);
  std::copy_n(source.previous_residuals + vector_offset, dimension,
              destination.previous_residuals + vector_offset);
  std::copy_n(source.df_history + history_offset, history_elements,
              destination.df_history + history_offset);
  std::copy_n(source.u_history + history_offset, history_elements,
              destination.u_history + history_offset);
  std::copy_n(source.omega + system * memory, memory, destination.omega + system * memory);
  destination.residual_rms[system] = source.residual_rms[system];
  destination.residual_maximum[system] = source.residual_maximum[system];
  destination.iterations[system] = source.iterations[system];
  destination.restart_counts[system] = source.restart_counts[system];
  destination.system_statuses[system] = source.system_statuses[system];
  destination.initialized[system] = source.initialized[system];
  destination.converged[system] = source.converged[system];
}

gpuxtb_status_t record_numeric_failure(const SccMixerState& state, std::size_t system,
                                       const char* message, std::string& error) {
  /* Preserve every numerical diagnostic and history field. A failed raw SCC
   * result is observable only through the per-system status and error text. */
  state.system_statuses[system] = GPUXTB_STATUS_INTERNAL_ERROR;
  error = message;
  return GPUXTB_STATUS_INTERNAL_ERROR;
}

bool dot_product(const double* first, const double* second, std::size_t count, double& result) {
  double sum = 0.0;
  for (std::size_t index = 0u; index < count; ++index) {
    sum += first[index] * second[index];
    if (!std::isfinite(sum)) {
      return false;
    }
  }
  result = sum;
  return true;
}

bool cholesky_solve(double* matrix, double* right_hand_side, std::size_t dimension) {
  /* beta is symmetric positive definite because omega0^2 is added to I. */
  for (std::size_t row = 0u; row < dimension; ++row) {
    for (std::size_t column = 0u; column <= row; ++column) {
      double value = matrix[row * dimension + column];
      for (std::size_t inner = 0u; inner < column; ++inner) {
        value -= matrix[row * dimension + inner] * matrix[column * dimension + inner];
      }
      if (!std::isfinite(value)) {
        return false;
      }
      if (row == column) {
        if (!(value > 0.0)) {
          return false;
        }
        matrix[row * dimension + column] = std::sqrt(value);
      } else {
        const double diagonal = matrix[column * dimension + column];
        value /= diagonal;
        if (!std::isfinite(value)) {
          return false;
        }
        matrix[row * dimension + column] = value;
      }
    }
  }

  for (std::size_t row = 0u; row < dimension; ++row) {
    double value = right_hand_side[row];
    for (std::size_t column = 0u; column < row; ++column) {
      value -= matrix[row * dimension + column] * right_hand_side[column];
    }
    value /= matrix[row * dimension + row];
    if (!std::isfinite(value)) {
      return false;
    }
    right_hand_side[row] = value;
  }
  for (std::size_t reverse = dimension; reverse > 0u; --reverse) {
    const std::size_t row = reverse - 1u;
    double value = right_hand_side[row];
    for (std::size_t column = row + 1u; column < dimension; ++column) {
      value -= matrix[column * dimension + row] * right_hand_side[column];
    }
    value /= matrix[row * dimension + row];
    if (!std::isfinite(value)) {
      return false;
    }
    right_hand_side[row] = value;
  }
  return true;
}

gpuxtb_status_t mix_system_unchecked(const SccMixerPlanData& data, std::size_t system,
                                     const WavefunctionView& wavefunction,
                                     const SccMixerState& state, const SccMixerWorkspace& workspace,
                                     std::string& error) {
  const std::size_t dimension = system_dimension(data, system);
  const std::size_t vector_offset = system_vector_offset(data, system);
  const std::size_t history_offset = system_history_offset(data, system);
  const std::size_t memory = static_cast<std::size_t>(data.history_size);
  const double* const current = state.current_inputs + vector_offset;
  const double* const previous = state.previous_inputs + vector_offset;
  const double* const previous_residual = state.previous_residuals + vector_offset;

  double residual_square = 0.0;
  double residual_maximum = 0.0;
  bool values_are_finite = true;
  for_each_raw_component(data, wavefunction, system, [&](std::size_t packed, double raw_value) {
    const double current_value = current[packed];
    const double residual = raw_value - current_value;
    workspace.residual[packed] = residual;
    values_are_finite = values_are_finite && std::isfinite(raw_value) &&
                        std::isfinite(current_value) && std::isfinite(residual);
    if (values_are_finite) {
      residual_square += residual * residual;
      residual_maximum = std::max(residual_maximum, std::abs(residual));
      values_are_finite = std::isfinite(residual_square);
    }
  });
  if (!values_are_finite) {
    return record_numeric_failure(state, system,
                                  "SCC mixer residual contains or produces NaN or infinity", error);
  }
  const double residual_norm = std::sqrt(residual_square);
  const double residual_rms = residual_norm / std::sqrt(static_cast<double>(dimension));
  if (!std::isfinite(residual_norm) || !std::isfinite(residual_rms)) {
    return record_numeric_failure(state, system, "SCC mixer residual norm is not finite", error);
  }

  const std::uint64_t old_iteration = state.iterations[system];
  if (old_iteration == std::numeric_limits<std::uint64_t>::max()) {
    return record_numeric_failure(state, system, "SCC mixer iteration counter cannot be advanced",
                                  error);
  }
  const std::uint64_t new_iteration = old_iteration + 1u;

  if (new_iteration == 1u) {
    for (std::size_t component = 0u; component < dimension; ++component) {
      const double value = current[component] + data.damping * workspace.residual[component];
      if (!std::isfinite(value)) {
        return record_numeric_failure(state, system,
                                      "SCC mixer damped startup produced NaN or infinity", error);
      }
      workspace.mixed[component] = value;
    }
  } else {
    double delta_f_square = 0.0;
    for (std::size_t component = 0u; component < dimension; ++component) {
      const double value = workspace.residual[component] - previous_residual[component];
      workspace.delta_f[component] = value;
      delta_f_square += value * value;
      if (!std::isfinite(value) || !std::isfinite(delta_f_square)) {
        return record_numeric_failure(state, system, "SCC mixer residual difference is not finite",
                                      error);
      }
    }
    const double delta_f_norm = std::sqrt(delta_f_square);
    const double denominator = std::max(delta_f_norm, std::numeric_limits<double>::epsilon());
    const double inverse = 1.0 / denominator;
    if (!std::isfinite(inverse)) {
      return record_numeric_failure(state, system, "SCC mixer residual normalization failed",
                                    error);
    }
    for (std::size_t component = 0u; component < dimension; ++component) {
      workspace.delta_f[component] *= inverse;
      const double value = data.damping * workspace.delta_f[component] +
                           inverse * (current[component] - previous[component]);
      if (!std::isfinite(workspace.delta_f[component]) || !std::isfinite(value)) {
        return record_numeric_failure(state, system,
                                      "SCC mixer Broyden update vector is not finite", error);
      }
      workspace.new_u[component] = value;
    }

    const double omega = residual_norm > kOmegaFactor / kMaximumOmega
                             ? std::max(kMinimumOmega, kOmegaFactor / residual_norm)
                             : kMaximumOmega;
    if (!std::isfinite(omega)) {
      return record_numeric_failure(state, system, "SCC mixer Broyden weight is not finite", error);
    }

    const std::size_t history_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(static_cast<std::uint64_t>(memory), old_iteration));
    const std::uint64_t first_iteration =
        old_iteration - static_cast<std::uint64_t>(history_count) + 1u;
    const std::size_t new_slot =
        static_cast<std::size_t>((old_iteration - 1u) % static_cast<std::uint64_t>(memory));
    for (std::size_t history = 0u; history < history_count; ++history) {
      const std::uint64_t represented_iteration =
          first_iteration + static_cast<std::uint64_t>(history);
      workspace.history_slots[history] = static_cast<std::int64_t>(
          (represented_iteration - 1u) % static_cast<std::uint64_t>(memory));
    }

    const auto df_vector = [&](std::size_t slot) {
      return slot == new_slot ? workspace.delta_f
                              : state.df_history + history_offset + slot * dimension;
    };
    const auto u_vector = [&](std::size_t slot) {
      return slot == new_slot ? workspace.new_u
                              : state.u_history + history_offset + slot * dimension;
    };
    const auto slot_omega = [&](std::size_t slot) {
      return slot == new_slot ? omega : state.omega[system * memory + slot];
    };

    for (std::size_t row = 0u; row < history_count; ++row) {
      const std::size_t row_slot = static_cast<std::size_t>(workspace.history_slots[row]);
      const double row_omega = slot_omega(row_slot);
      double coefficient_dot = 0.0;
      if (!std::isfinite(row_omega) ||
          !dot_product(df_vector(row_slot), workspace.residual, dimension, coefficient_dot)) {
        return record_numeric_failure(state, system, "SCC mixer Broyden coefficient is not finite",
                                      error);
      }
      workspace.coefficients[row] = row_omega * coefficient_dot;
      if (!std::isfinite(workspace.coefficients[row])) {
        return record_numeric_failure(state, system, "SCC mixer Broyden coefficient overflowed",
                                      error);
      }
      for (std::size_t column = 0u; column < history_count; ++column) {
        const std::size_t column_slot = static_cast<std::size_t>(workspace.history_slots[column]);
        const double column_omega = slot_omega(column_slot);
        double overlap = 0.0;
        if (!std::isfinite(column_omega) ||
            !dot_product(df_vector(row_slot), df_vector(column_slot), dimension, overlap)) {
          return record_numeric_failure(state, system,
                                        "SCC mixer Broyden history overlap is not finite", error);
        }
        double value = row_omega * column_omega * overlap;
        if (row == column) {
          value += kOmegaZero * kOmegaZero;
        }
        if (!std::isfinite(value)) {
          return record_numeric_failure(state, system, "SCC mixer Broyden matrix overflowed",
                                        error);
        }
        workspace.beta[row * history_count + column] = value;
      }
    }
    if (!cholesky_solve(workspace.beta, workspace.coefficients, history_count)) {
      return record_numeric_failure(state, system, "SCC mixer Broyden system is not usable", error);
    }

    for (std::size_t component = 0u; component < dimension; ++component) {
      double value = current[component] + data.damping * workspace.residual[component];
      for (std::size_t history = 0u; history < history_count; ++history) {
        const std::size_t slot = static_cast<std::size_t>(workspace.history_slots[history]);
        value -= slot_omega(slot) * workspace.coefficients[history] * u_vector(slot)[component];
      }
      if (!std::isfinite(value)) {
        return record_numeric_failure(state, system, "SCC mixer Broyden result is not finite",
                                      error);
      }
      workspace.mixed[component] = value;
    }

    std::copy_n(workspace.delta_f, dimension,
                state.df_history + history_offset + new_slot * dimension);
    std::copy_n(workspace.new_u, dimension,
                state.u_history + history_offset + new_slot * dimension);
    state.omega[system * memory + new_slot] = omega;
  }

  /* No operation below this point can fail: publish one complete system. */
  std::copy_n(current, dimension, state.previous_inputs + vector_offset);
  std::copy_n(workspace.residual, dimension, state.previous_residuals + vector_offset);
  std::copy_n(workspace.mixed, dimension, state.current_inputs + vector_offset);
  publish_mixed_components(data, wavefunction, system, workspace.mixed);
  state.residual_rms[system] = residual_rms;
  state.residual_maximum[system] = residual_maximum;
  state.iterations[system] = new_iteration;
  state.system_statuses[system] = GPUXTB_STATUS_SUCCESS;
  state.converged[system] =
      residual_rms < data.rms_tolerance && residual_maximum < data.maximum_tolerance ? 1u : 0u;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace

SccMixerPlan::SccMixerPlan(std::shared_ptr<const SccMixerPlanData> data) noexcept
    : data_(std::move(data)) {}

bool SccMixerPlan::sealed() const noexcept { return data_ != nullptr; }

std::int64_t SccMixerPlan::batch_size() const noexcept {
  return data_ == nullptr ? 0 : data_->batch_size;
}

std::int64_t SccMixerPlan::history_size() const noexcept {
  return data_ == nullptr ? 0 : data_->history_size;
}

std::int64_t SccMixerPlan::total_vector_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->total_vector_elements;
}

std::int64_t SccMixerPlan::maximum_vector_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->maximum_vector_elements;
}

double SccMixerPlan::damping() const noexcept { return data_ == nullptr ? 0.0 : data_->damping; }

double SccMixerPlan::rms_tolerance() const noexcept {
  return data_ == nullptr ? 0.0 : data_->rms_tolerance;
}

double SccMixerPlan::maximum_tolerance() const noexcept {
  return data_ == nullptr ? 0.0 : data_->maximum_tolerance;
}

std::size_t SccMixerPlan::state_size_bytes() const noexcept {
  return data_ == nullptr ? 0u : data_->state_size_bytes;
}

std::size_t SccMixerPlan::workspace_size_bytes() const noexcept {
  return data_ == nullptr ? 0u : data_->workspace_size_bytes;
}

std::size_t SccMixerPlan::resident_bytes() const noexcept {
  if (data_ == nullptr) {
    return 0u;
  }
  return sizeof(*data_) + data_->vector_offsets.capacity() * sizeof(std::int64_t) +
         data_->history_offsets.capacity() * sizeof(std::int64_t) +
         data_->qsh_system_offsets.capacity() * sizeof(std::int64_t) +
         data_->dipole_system_offsets.capacity() * sizeof(std::int64_t) +
         data_->quadrupole_system_offsets.capacity() * sizeof(std::int64_t);
}

const std::vector<std::int64_t>& SccMixerPlan::vector_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ == nullptr ? empty : data_->vector_offsets;
}

bool SccMixerPlan::matches_wavefunction_layout(const WavefunctionLayout& layout) const noexcept {
  return data_ != nullptr && data_->batch_size == layout.batch_size &&
         data_->wavefunction_workspace_size_bytes == layout.workspace_size_bytes &&
         data_->qsh_offset_bytes == layout.qsh.offset_bytes &&
         data_->dipole_offset_bytes == layout.dipole.offset_bytes &&
         data_->quadrupole_offset_bytes == layout.quadrupole.offset_bytes &&
         data_->qsh_system_offsets == layout.qsh.system_offsets &&
         data_->dipole_system_offsets == layout.dipole.system_offsets &&
         data_->quadrupole_system_offsets == layout.quadrupole.system_offsets;
}

bool SccMixerPlan::overlaps_storage(const void* data, std::size_t size_bytes) const noexcept {
  AddressRange range;
  return size_bytes != 0u && (data_ == nullptr || !make_range(data, size_bytes, range) ||
                              overlaps_plan_storage(*this, range));
}

const SccMixerPlanData* SccMixerPlan::identity() const noexcept { return data_.get(); }

gpuxtb_status_t make_scc_mixer_plan(const WavefunctionLayout& layout, std::int64_t history_size,
                                    double damping, double rms_tolerance, double maximum_tolerance,
                                    SccMixerPlan& plan, std::string& error) {
  WavefunctionWarmStartIdentity validated_layout;
  gpuxtb_status_t status =
      make_wavefunction_warm_start_identity(layout, 0u, validated_layout, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (history_size <= 0 || !std::isfinite(damping) || !(damping > 0.0) || damping > 1.0 ||
      !std::isfinite(rms_tolerance) || !(rms_tolerance > 0.0) ||
      !std::isfinite(maximum_tolerance) || !(maximum_tolerance > 0.0)) {
    error = "SCC mixer history, damping, and convergence tolerances are invalid";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  try {
    SccMixerPlanData created;
    created.batch_size = layout.batch_size;
    created.history_size = history_size;
    created.damping = damping;
    created.rms_tolerance = rms_tolerance;
    created.maximum_tolerance = maximum_tolerance;
    created.wavefunction_workspace_size_bytes = layout.workspace_size_bytes;
    created.qsh_offset_bytes = layout.qsh.offset_bytes;
    created.dipole_offset_bytes = layout.dipole.offset_bytes;
    created.quadrupole_offset_bytes = layout.quadrupole.offset_bytes;
    created.qsh_system_offsets = layout.qsh.system_offsets;
    created.dipole_system_offsets = layout.dipole.system_offsets;
    created.quadrupole_system_offsets = layout.quadrupole.system_offsets;

    const std::size_t batch = static_cast<std::size_t>(layout.batch_size);
    created.vector_offsets.assign(batch + 1u, 0);
    created.history_offsets.assign(batch + 1u, 0);
    for (std::size_t system = 0u; system < batch; ++system) {
      const std::int64_t qsh =
          layout.qsh.system_offsets[system + 1u] - layout.qsh.system_offsets[system];
      const std::int64_t dipole =
          layout.dipole.system_offsets[system + 1u] - layout.dipole.system_offsets[system];
      const std::int64_t quadrupole =
          layout.quadrupole.system_offsets[system + 1u] - layout.quadrupole.system_offsets[system];
      std::int64_t dimension = 0;
      std::int64_t history_elements = 0;
      if (!checked_add_i64(qsh, dimension) || !checked_add_i64(dipole, dimension) ||
          !checked_add_i64(quadrupole, dimension) || dimension <= 0 ||
          !checked_multiply_i64(dimension, history_size, history_elements) ||
          !checked_add_i64(dimension, created.vector_offsets[system + 1u]) ||
          !checked_add_i64(created.vector_offsets[system], created.vector_offsets[system + 1u]) ||
          !checked_add_i64(history_elements, created.history_offsets[system + 1u]) ||
          !checked_add_i64(created.history_offsets[system], created.history_offsets[system + 1u])) {
        error = "SCC mixer ragged vector or history dimensions overflow int64_t";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      created.maximum_vector_elements = std::max(created.maximum_vector_elements, dimension);
    }
    created.total_vector_elements = created.vector_offsets.back();

    const auto bytes_for = [&](std::int64_t count, std::size_t element_size, std::size_t& bytes) {
      return count >= 0 &&
             checked_multiply_size(static_cast<std::size_t>(count), element_size, bytes) &&
             static_cast<std::uint64_t>(count) <=
                 static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
    };
    std::int64_t omega_elements = 0;
    std::int64_t beta_elements = 0;
    if (!checked_multiply_i64(layout.batch_size, history_size, omega_elements) ||
        !checked_multiply_i64(history_size, history_size, beta_elements)) {
      error = "SCC mixer history dimensions overflow int64_t";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    const std::int64_t total_history_elements = created.history_offsets.back();
    std::size_t vector_bytes = 0u;
    std::size_t history_bytes = 0u;
    std::size_t omega_bytes = 0u;
    std::size_t batch_double_bytes = 0u;
    std::size_t batch_u64_bytes = 0u;
    std::size_t batch_status_bytes = 0u;
    std::size_t batch_byte_bytes = 0u;
    std::size_t maximum_vector_bytes = 0u;
    std::size_t beta_bytes = 0u;
    std::size_t coefficient_bytes = 0u;
    std::size_t slot_bytes = 0u;
    if (!bytes_for(created.total_vector_elements, sizeof(double), vector_bytes) ||
        !bytes_for(total_history_elements, sizeof(double), history_bytes) ||
        !bytes_for(omega_elements, sizeof(double), omega_bytes) ||
        !bytes_for(layout.batch_size, sizeof(double), batch_double_bytes) ||
        !bytes_for(layout.batch_size, sizeof(std::uint64_t), batch_u64_bytes) ||
        !bytes_for(layout.batch_size, sizeof(gpuxtb_status_t), batch_status_bytes) ||
        !bytes_for(layout.batch_size, sizeof(std::uint8_t), batch_byte_bytes) ||
        !bytes_for(created.maximum_vector_elements, sizeof(double), maximum_vector_bytes) ||
        !bytes_for(beta_elements, sizeof(double), beta_bytes) ||
        !bytes_for(history_size, sizeof(double), coefficient_bytes) ||
        !bytes_for(history_size, sizeof(std::int64_t), slot_bytes)) {
      error = "SCC mixer caller-owned storage exceeds addressable memory";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    std::size_t cursor = 0u;
    if (!append_segment(vector_bytes, alignof(double), cursor,
                        created.current_input_offset_bytes) ||
        !append_segment(vector_bytes, alignof(double), cursor,
                        created.previous_input_offset_bytes) ||
        !append_segment(vector_bytes, alignof(double), cursor,
                        created.previous_residual_offset_bytes) ||
        !append_segment(history_bytes, alignof(double), cursor, created.df_history_offset_bytes) ||
        !append_segment(history_bytes, alignof(double), cursor, created.u_history_offset_bytes) ||
        !append_segment(omega_bytes, alignof(double), cursor, created.omega_offset_bytes) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.residual_rms_offset_bytes) ||
        !append_segment(batch_double_bytes, alignof(double), cursor,
                        created.residual_maximum_offset_bytes) ||
        !append_segment(batch_u64_bytes, alignof(std::uint64_t), cursor,
                        created.iteration_offset_bytes) ||
        !append_segment(batch_u64_bytes, alignof(std::uint64_t), cursor,
                        created.restart_count_offset_bytes) ||
        !append_segment(batch_status_bytes, alignof(gpuxtb_status_t), cursor,
                        created.system_status_offset_bytes) ||
        !append_segment(batch_byte_bytes, alignof(std::uint8_t), cursor,
                        created.initialized_offset_bytes) ||
        !append_segment(batch_byte_bytes, alignof(std::uint8_t), cursor,
                        created.converged_offset_bytes) ||
        !align_up(cursor, kSccMixerWorkspaceAlignment, created.state_size_bytes)) {
      error = "SCC mixer persistent state packing overflows size_t";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    cursor = 0u;
    if (!append_segment(maximum_vector_bytes, alignof(double), cursor,
                        created.residual_scratch_offset_bytes) ||
        !append_segment(maximum_vector_bytes, alignof(double), cursor,
                        created.mixed_scratch_offset_bytes) ||
        !append_segment(maximum_vector_bytes, alignof(double), cursor,
                        created.delta_f_scratch_offset_bytes) ||
        !append_segment(maximum_vector_bytes, alignof(double), cursor,
                        created.new_u_scratch_offset_bytes) ||
        !append_segment(beta_bytes, alignof(double), cursor, created.beta_scratch_offset_bytes) ||
        !append_segment(coefficient_bytes, alignof(double), cursor,
                        created.coefficient_scratch_offset_bytes) ||
        !append_segment(slot_bytes, alignof(std::int64_t), cursor,
                        created.history_slot_scratch_offset_bytes) ||
        !align_up(cursor, kSccMixerWorkspaceAlignment, created.workspace_size_bytes)) {
      error = "SCC mixer scratch packing overflows size_t";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    auto sealed = std::make_shared<const SccMixerPlanData>(std::move(created));
    plan = SccMixerPlan(std::move(sealed));
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate SCC mixer plan metadata";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

gpuxtb_status_t bind_scc_mixer_state(const SccMixerPlan& plan, void* workspace,
                                     std::size_t workspace_size, SccMixerState& state,
                                     std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccMixerPlanData& data = *plan.identity();
  AddressRange workspace_range;
  AddressRange plan_descriptor;
  AddressRange state_descriptor;
  AddressRange error_descriptor;
  if (!is_aligned(workspace, kSccMixerWorkspaceAlignment) ||
      workspace_size < data.state_size_bytes ||
      !make_range(workspace, data.state_size_bytes, workspace_range) ||
      !make_range(&plan, sizeof(plan), plan_descriptor) ||
      !make_range(&state, sizeof(state), state_descriptor) ||
      !make_range(&error, sizeof(error), error_descriptor) ||
      overlaps_plan_storage(plan, workspace_range) ||
      ranges_overlap(workspace_range, plan_descriptor) ||
      ranges_overlap(workspace_range, state_descriptor) ||
      ranges_overlap(workspace_range, error_descriptor)) {
    error = "SCC mixer persistent state storage is invalid or overlaps control storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  SccMixerState created;
  created.workspace_base = workspace;
  created.workspace_size_bytes = workspace_size;
  created.current_inputs = offset_pointer<double>(workspace, data.current_input_offset_bytes);
  created.previous_inputs = offset_pointer<double>(workspace, data.previous_input_offset_bytes);
  created.previous_residuals =
      offset_pointer<double>(workspace, data.previous_residual_offset_bytes);
  created.df_history = offset_pointer<double>(workspace, data.df_history_offset_bytes);
  created.u_history = offset_pointer<double>(workspace, data.u_history_offset_bytes);
  created.omega = offset_pointer<double>(workspace, data.omega_offset_bytes);
  created.residual_rms = offset_pointer<double>(workspace, data.residual_rms_offset_bytes);
  created.residual_maximum = offset_pointer<double>(workspace, data.residual_maximum_offset_bytes);
  created.iterations = offset_pointer<std::uint64_t>(workspace, data.iteration_offset_bytes);
  created.restart_counts =
      offset_pointer<std::uint64_t>(workspace, data.restart_count_offset_bytes);
  created.system_statuses =
      offset_pointer<gpuxtb_status_t>(workspace, data.system_status_offset_bytes);
  created.initialized = offset_pointer<std::uint8_t>(workspace, data.initialized_offset_bytes);
  created.converged = offset_pointer<std::uint8_t>(workspace, data.converged_offset_bytes);
  created.plan_identity = &data;

  std::memset(workspace, 0, data.state_size_bytes);
  std::fill_n(created.system_statuses, static_cast<std::size_t>(data.batch_size),
              GPUXTB_STATUS_INVALID_ARGUMENT);
  state = created;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t bind_scc_mixer_workspace(const SccMixerPlan& plan, void* workspace,
                                         std::size_t workspace_size, SccMixerWorkspace& view,
                                         std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccMixerPlanData& data = *plan.identity();
  AddressRange workspace_range;
  AddressRange plan_descriptor;
  AddressRange view_descriptor;
  AddressRange error_descriptor;
  if (!is_aligned(workspace, kSccMixerWorkspaceAlignment) ||
      workspace_size < data.workspace_size_bytes ||
      !make_range(workspace, data.workspace_size_bytes, workspace_range) ||
      !make_range(&plan, sizeof(plan), plan_descriptor) ||
      !make_range(&view, sizeof(view), view_descriptor) ||
      !make_range(&error, sizeof(error), error_descriptor) ||
      overlaps_plan_storage(plan, workspace_range) ||
      ranges_overlap(workspace_range, plan_descriptor) ||
      ranges_overlap(workspace_range, view_descriptor) ||
      ranges_overlap(workspace_range, error_descriptor)) {
    error = "SCC mixer scratch storage is invalid or overlaps control storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  SccMixerWorkspace created;
  created.workspace_base = workspace;
  created.workspace_size_bytes = workspace_size;
  created.residual = offset_pointer<double>(workspace, data.residual_scratch_offset_bytes);
  created.mixed = offset_pointer<double>(workspace, data.mixed_scratch_offset_bytes);
  created.delta_f = offset_pointer<double>(workspace, data.delta_f_scratch_offset_bytes);
  created.new_u = offset_pointer<double>(workspace, data.new_u_scratch_offset_bytes);
  created.beta = offset_pointer<double>(workspace, data.beta_scratch_offset_bytes);
  created.coefficients = offset_pointer<double>(workspace, data.coefficient_scratch_offset_bytes);
  created.history_slots =
      offset_pointer<std::int64_t>(workspace, data.history_slot_scratch_offset_bytes);
  created.plan_identity = &data;
  view = created;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t initialize_scc_mixer_state_cpu(const SccMixerPlan& plan,
                                               const WavefunctionView& wavefunction,
                                               const SccMixerState& state, std::string& error) {
  gpuxtb_status_t status = validate_call(plan, wavefunction, state, nullptr, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccMixerPlanData& data = *plan.identity();
  const std::size_t batch = static_cast<std::size_t>(data.batch_size);
  for (std::size_t system = 0u; system < batch; ++system) {
    if (!raw_components_are_finite(data, wavefunction, system)) {
      error = "SCC mixer initial wavefunction contains NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  std::memset(state.workspace_base, 0, data.state_size_bytes);
  for (std::size_t system = 0u; system < batch; ++system) {
    copy_raw_components(data, wavefunction, system,
                        state.current_inputs + system_vector_offset(data, system));
    state.initialized[system] = 1u;
    state.system_statuses[system] = GPUXTB_STATUS_SUCCESS;
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t restart_scc_mixer_system_cpu(const SccMixerPlan& plan, std::int64_t system,
                                             const WavefunctionView& wavefunction,
                                             const SccMixerState& state, std::string& error) {
  gpuxtb_status_t status = validate_call(plan, wavefunction, state, nullptr, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccMixerPlanData& data = *plan.identity();
  if (system < 0 || system >= data.batch_size) {
    error = "SCC mixer restart requires a valid system index";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t index = system_index(system);
  if (state.initialized[index] != 1u) {
    error = "SCC mixer system must be initialized before it can be restarted";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (state.restart_counts[index] == std::numeric_limits<std::uint64_t>::max()) {
    error = "SCC mixer restart counter cannot be advanced";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (!raw_components_are_finite(data, wavefunction, index)) {
    error = "SCC mixer restart wavefunction contains NaN or infinity";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t dimension = system_dimension(data, index);
  const std::size_t vector_offset = system_vector_offset(data, index);
  const std::size_t history_offset = system_history_offset(data, index);
  const std::size_t history_elements = dimension * static_cast<std::size_t>(data.history_size);
  copy_raw_components(data, wavefunction, index, state.current_inputs + vector_offset);
  std::fill_n(state.previous_inputs + vector_offset, dimension, 0.0);
  std::fill_n(state.previous_residuals + vector_offset, dimension, 0.0);
  std::fill_n(state.df_history + history_offset, history_elements, 0.0);
  std::fill_n(state.u_history + history_offset, history_elements, 0.0);
  std::fill_n(state.omega + index * static_cast<std::size_t>(data.history_size),
              static_cast<std::size_t>(data.history_size), 0.0);
  state.residual_rms[index] = 0.0;
  state.residual_maximum[index] = 0.0;
  state.iterations[index] = 0u;
  ++state.restart_counts[index];
  state.system_statuses[index] = GPUXTB_STATUS_SUCCESS;
  state.converged[index] = 0u;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t mix_scc_broyden_system_cpu(const SccMixerPlan& plan, std::int64_t system,
                                           const WavefunctionView& wavefunction,
                                           const SccMixerState& state,
                                           const SccMixerWorkspace& workspace, std::string& error) {
  gpuxtb_status_t status = validate_call(plan, wavefunction, state, &workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccMixerPlanData& data = *plan.identity();
  if (system < 0 || system >= data.batch_size) {
    error = "SCC mixer worker requires a valid system index";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t index = system_index(system);
  if (state.initialized[index] != 1u) {
    error = "SCC mixer worker requires initialized per-system state";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return mix_system_unchecked(data, index, wavefunction, state, workspace, error);
}

gpuxtb_status_t mix_scc_broyden_batch_cpu(const SccMixerPlan& plan,
                                          const WavefunctionView& wavefunction,
                                          const SccMixerState& state,
                                          const SccMixerWorkspace& workspace, std::string& error) {
  gpuxtb_status_t status = validate_call(plan, wavefunction, state, &workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccMixerPlanData& data = *plan.identity();
  const std::size_t batch = static_cast<std::size_t>(data.batch_size);
  for (std::size_t system = 0u; system < batch; ++system) {
    if (state.initialized[system] != 1u) {
      error = "SCC mixer batch requires every system state to be initialized";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  gpuxtb_status_t first_failure = GPUXTB_STATUS_SUCCESS;
  std::string first_error;
  for (std::size_t system = 0u; system < batch; ++system) {
    status = mix_system_unchecked(data, system, wavefunction, state, workspace, error);
    if (status != GPUXTB_STATUS_SUCCESS && first_failure == GPUXTB_STATUS_SUCCESS) {
      first_failure = status;
      first_error = error;
    }
  }
  if (first_failure != GPUXTB_STATUS_SUCCESS) {
    error = std::move(first_error);
    return first_failure;
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t prepare_scc_mixer_system_transaction_cpu(const SccMixerPlan& plan,
                                                         std::int64_t system,
                                                         const SccMixerState& source,
                                                         const SccMixerState& staged,
                                                         std::string& error) {
  gpuxtb_status_t status = validate_transaction(plan, source, staged, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccMixerPlanData& data = *plan.identity();
  if (system < 0 || system >= data.batch_size) {
    error = "SCC mixer transaction requires a valid system index";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t index = system_index(system);
  if (source.initialized[index] != 1u) {
    error = "SCC mixer transaction source system must be initialized";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  copy_mixer_system_state(data, index, source, staged);
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t commit_scc_mixer_system_transaction_cpu(const SccMixerPlan& plan,
                                                        std::int64_t system,
                                                        const SccMixerState& staged,
                                                        const SccMixerState& destination,
                                                        std::string& error) {
  gpuxtb_status_t status = validate_transaction(plan, staged, destination, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const SccMixerPlanData& data = *plan.identity();
  if (system < 0 || system >= data.batch_size) {
    error = "SCC mixer transaction requires a valid system index";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t index = system_index(system);
  copy_mixer_system_state(data, index, staged, destination);
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail::gfn2
