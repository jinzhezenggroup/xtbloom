#include "model/gfn2/periodic_embedding.hpp"
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace gpuxtb::detail::gfn2 {

struct PeriodicEmbeddingPlanData final {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t maximum_atoms = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> matrix_offsets;
};

namespace {

static_assert(std::is_trivially_copyable_v<PeriodicEmbeddingView>);
static_assert(std::is_standard_layout_v<PeriodicEmbeddingView>);
static_assert(std::is_trivially_copyable_v<PeriodicEmbeddingWorkspace>);
static_assert(std::is_standard_layout_v<PeriodicEmbeddingWorkspace>);

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
  bool active = false;
};

bool representable_as_size(std::int64_t value) {
  return value >= 0 &&
         static_cast<std::uint64_t>(value) <=
             static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) &&
         static_cast<std::uint64_t>(value) <=
             static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

bool count_fits_storage(std::int64_t count, std::size_t element_size, bool add_sentinel = false) {
  if (!representable_as_size(count)) {
    return false;
  }
  const auto value = static_cast<std::uint64_t>(count);
  const auto extra = add_sentinel ? std::uint64_t{1} : std::uint64_t{0};
  if (value > std::numeric_limits<std::uint64_t>::max() - extra) {
    return false;
  }
  const auto length = value + extra;
  return length <=
             static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / element_size &&
         length <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

bool checked_square(std::int64_t value, std::int64_t& square) {
  if (value < 0 || (value != 0 && value > std::numeric_limits<std::int64_t>::max() / value)) {
    return false;
  }
  square = value * value;
  return true;
}

bool checked_add(std::int64_t increment, std::int64_t& total) {
  if (increment < 0 || total > std::numeric_limits<std::int64_t>::max() - increment) {
    return false;
  }
  total += increment;
  return true;
}

bool checked_bytes(std::int64_t count, std::size_t element_size, std::size_t& bytes) {
  if (!representable_as_size(count) ||
      static_cast<std::uint64_t>(count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) {
  if (bytes == 0u) {
    range = {};
    return true;
  }
  if (pointer == nullptr) {
    return false;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  range = {begin, begin + bytes, true};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.active && second.active && first.begin < second.end && second.begin < first.end;
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

template <typename T>
bool required_pointer(const T* pointer, std::int64_t required) {
  return required == 0 || is_aligned(pointer, alignof(T));
}

bool range_for_count(const void* pointer, std::int64_t count, std::size_t element_size,
                     AddressRange& range) {
  std::size_t bytes = 0u;
  return checked_bytes(count, element_size, bytes) && make_range(pointer, bytes, range);
}

std::array<AddressRange, 4> plan_ranges(const PeriodicEmbeddingPlan& plan) {
  AddressRange descriptor;
  AddressRange data;
  AddressRange atom_offsets;
  AddressRange matrix_offsets;
  /* Capacity, rather than logical size, is the complete allocation owned by
   * the immutable plan. Callers must not hide numerical buffers in spare
   * vector backing storage outside the active metadata prefix. */
  const auto atom_bytes = plan.atom_offsets().capacity() * sizeof(std::int64_t);
  const auto matrix_bytes = plan.matrix_offsets().capacity() * sizeof(std::int64_t);
  if (!make_range(&plan, sizeof(plan), descriptor) ||
      !make_range(plan.identity(), sizeof(PeriodicEmbeddingPlanData), data) ||
      !make_range(plan.atom_offsets().data(), atom_bytes, atom_offsets) ||
      !make_range(plan.matrix_offsets().data(), matrix_bytes, matrix_offsets)) {
    return {};
  }
  return {{descriptor, data, atom_offsets, matrix_offsets}};
}

bool overlaps_plan_storage(const PeriodicEmbeddingPlan& plan, const AddressRange& range) {
  for (const AddressRange& candidate : plan_ranges(plan)) {
    if (ranges_overlap(range, candidate)) {
      return true;
    }
  }
  return false;
}

gpuxtb_status_t validate_plan(const PeriodicEmbeddingPlan& plan, std::string& error) {
  if (!plan.sealed() || plan.batch_size() <= 0 || plan.total_atoms() < 0 ||
      plan.total_matrix_elements() < 0 || plan.maximum_atoms() < 0 ||
      plan.batch_size() == std::numeric_limits<std::int64_t>::max() ||
      !representable_as_size(plan.batch_size()) || !representable_as_size(plan.total_atoms()) ||
      !representable_as_size(plan.total_matrix_elements()) ||
      !representable_as_size(plan.maximum_atoms())) {
    error = "periodic embedding plan is unsealed or has invalid dimensions";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch_count = static_cast<std::size_t>(plan.batch_size());
  if (plan.atom_offsets().size() != batch_count + 1u ||
      plan.matrix_offsets().size() != batch_count + 1u || plan.atom_offsets().front() != 0 ||
      plan.atom_offsets().back() != plan.total_atoms() || plan.matrix_offsets().front() != 0 ||
      plan.matrix_offsets().back() != plan.total_matrix_elements()) {
    error = "periodic embedding plan storage is internally inconsistent";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  std::int64_t expected_matrix_offset = 0;
  std::int64_t expected_maximum = 0;
  for (std::size_t system = 0u; system < batch_count; ++system) {
    const std::int64_t begin = plan.atom_offsets()[system];
    const std::int64_t end = plan.atom_offsets()[system + 1u];
    std::int64_t matrix_elements = 0;
    if (begin < 0 || begin > end || end > plan.total_atoms() ||
        plan.matrix_offsets()[system] != expected_matrix_offset ||
        !checked_square(end - begin, matrix_elements) ||
        !checked_add(matrix_elements, expected_matrix_offset)) {
      error = "periodic embedding plan offsets are not valid ragged partitions";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    expected_maximum = std::max(expected_maximum, end - begin);
  }
  if (expected_matrix_offset != plan.total_matrix_elements() ||
      expected_maximum != plan.maximum_atoms()) {
    error = "periodic embedding plan extents disagree with its offsets";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_view_shape(const PeriodicEmbeddingPlan& plan,
                                    const PeriodicEmbeddingView& view, std::string& error) {
  if (view.plan_identity != plan.identity() || view.shift_elements < plan.total_atoms() ||
      view.response_elements < plan.total_matrix_elements() ||
      view.charge_elements < plan.total_atoms() || view.potential_elements < plan.total_atoms() ||
      view.energy_elements < plan.batch_size() || view.status_elements < plan.batch_size() ||
      !required_pointer(view.shifts, plan.total_atoms()) ||
      !required_pointer(view.response_matrices, plan.total_matrix_elements()) ||
      !required_pointer(view.atomic_charges, plan.total_atoms()) ||
      !required_pointer(view.atomic_potentials, plan.total_atoms()) ||
      !required_pointer(view.energies, plan.batch_size()) ||
      !required_pointer(view.system_statuses, plan.batch_size())) {
    error = "periodic embedding view is malformed, undersized, or belongs to another plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_workspace_shape(const PeriodicEmbeddingPlan& plan,
                                         const PeriodicEmbeddingWorkspace& workspace,
                                         std::string& error) {
  if (workspace.plan_identity != plan.identity() ||
      workspace.potential_elements < plan.maximum_atoms() ||
      !required_pointer(workspace.potential_scratch, plan.maximum_atoms())) {
    error = "periodic embedding workspace is malformed, undersized, or belongs to another plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_view_binding_ranges(const PeriodicEmbeddingPlan& plan,
                                             const PeriodicEmbeddingView& view,
                                             const PeriodicEmbeddingView& destination,
                                             std::string& error) {
  std::array<AddressRange, 7> active;
  if (!range_for_count(view.shifts, plan.total_atoms(), sizeof(double), active[0]) ||
      !range_for_count(view.response_matrices, plan.total_matrix_elements(), sizeof(double),
                       active[1]) ||
      !range_for_count(view.atomic_charges, plan.total_atoms(), sizeof(double), active[2]) ||
      !range_for_count(view.atomic_potentials, plan.total_atoms(), sizeof(double), active[3]) ||
      !range_for_count(view.energies, plan.batch_size(), sizeof(double), active[4]) ||
      !range_for_count(view.system_statuses, plan.batch_size(), sizeof(gpuxtb_status_t),
                       active[5]) ||
      !make_range(&error, sizeof(error), active[6])) {
    error = "periodic embedding view ranges exceed addressable host storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (!pairwise_disjoint(active)) {
    error = "periodic embedding view buffers and error storage must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  AddressRange destination_range;
  if (!make_range(&destination, sizeof(destination), destination_range)) {
    error = "periodic embedding view descriptor range is invalid";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (const AddressRange& range : active) {
    if (overlaps_plan_storage(plan, range) || ranges_overlap(range, destination_range)) {
      error = "periodic embedding view buffers must not overlap plan or descriptor storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_workspace_binding_range(const PeriodicEmbeddingPlan& plan,
                                                 const PeriodicEmbeddingWorkspace& workspace,
                                                 const PeriodicEmbeddingWorkspace& destination,
                                                 std::string& error) {
  AddressRange scratch;
  AddressRange error_range;
  AddressRange destination_range;
  if (!range_for_count(workspace.potential_scratch, plan.maximum_atoms(), sizeof(double),
                       scratch) ||
      !make_range(&error, sizeof(error), error_range) ||
      !make_range(&destination, sizeof(destination), destination_range)) {
    error = "periodic embedding workspace ranges exceed addressable host storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (ranges_overlap(scratch, error_range) || overlaps_plan_storage(plan, scratch) ||
      ranges_overlap(scratch, destination_range)) {
    error = "periodic embedding scratch must not overlap plan, descriptor, or error storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_active_ranges(const PeriodicEmbeddingPlan& plan,
                                       const PeriodicEmbeddingView& view,
                                       const PeriodicEmbeddingWorkspace& workspace,
                                       std::string& error) {
  std::array<AddressRange, 8> active;
  if (!range_for_count(view.shifts, plan.total_atoms(), sizeof(double), active[0]) ||
      !range_for_count(view.response_matrices, plan.total_matrix_elements(), sizeof(double),
                       active[1]) ||
      !range_for_count(view.atomic_charges, plan.total_atoms(), sizeof(double), active[2]) ||
      !range_for_count(view.atomic_potentials, plan.total_atoms(), sizeof(double), active[3]) ||
      !range_for_count(view.energies, plan.batch_size(), sizeof(double), active[4]) ||
      !range_for_count(view.system_statuses, plan.batch_size(), sizeof(gpuxtb_status_t),
                       active[5]) ||
      !range_for_count(workspace.potential_scratch, plan.maximum_atoms(), sizeof(double),
                       active[6]) ||
      !make_range(&error, sizeof(error), active[7])) {
    error = "periodic embedding active ranges exceed addressable host storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (!pairwise_disjoint(active)) {
    error = "periodic embedding inputs, outputs, statuses, scratch, and error must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  AddressRange view_descriptor;
  AddressRange workspace_descriptor;
  if (!make_range(&view, sizeof(view), view_descriptor) ||
      !make_range(&workspace, sizeof(workspace), workspace_descriptor)) {
    error = "periodic embedding descriptor ranges are invalid";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (const AddressRange& range : active) {
    if (overlaps_plan_storage(plan, range) || ranges_overlap(range, view_descriptor) ||
        ranges_overlap(range, workspace_descriptor)) {
      error = "periodic embedding buffers must not overlap plan or descriptor storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_call(const PeriodicEmbeddingPlan& plan, const PeriodicEmbeddingView& view,
                              const PeriodicEmbeddingWorkspace& workspace, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_view_shape(plan, view, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_workspace_shape(plan, workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  return validate_active_ranges(plan, view, workspace, error);
}

bool evaluate_system_unchecked(const PeriodicEmbeddingPlan& plan, std::size_t system,
                               const PeriodicEmbeddingView& view,
                               const PeriodicEmbeddingWorkspace& workspace) {
  const std::int64_t atom_begin = plan.atom_offsets()[system];
  const std::int64_t atom_end = plan.atom_offsets()[system + 1u];
  const std::int64_t atom_count = atom_end - atom_begin;
  if (atom_count == 0) {
    view.energies[system] = 0.0;
    view.system_statuses[system] = GPUXTB_STATUS_SUCCESS;
    return true;
  }
  const std::int64_t matrix_begin = plan.matrix_offsets()[system];
  const double* const shifts = view.shifts + static_cast<std::size_t>(atom_begin);
  const double* const charges = view.atomic_charges + static_cast<std::size_t>(atom_begin);
  const double* const matrix = view.response_matrices + static_cast<std::size_t>(matrix_begin);

  for (std::int64_t atom = 0; atom < atom_count; ++atom) {
    if (!std::isfinite(shifts[atom]) || !std::isfinite(charges[atom])) {
      return false;
    }
    workspace.potential_scratch[atom] = 0.0;
  }

  /*
   * Traverse the symmetric operator by triangles so each stored element is
   * read once: the upper value performs both symmetric products and the lower
   * value verifies the host contract. Exact symmetry keeps v equal to dE/dq
   * without silently changing a host-generated response operator.
   */
  for (std::int64_t row = 0; row < atom_count; ++row) {
    const double diagonal = matrix[row * atom_count + row];
    if (!std::isfinite(diagonal)) {
      return false;
    }
    const double diagonal_response =
        std::fma(diagonal, charges[row], workspace.potential_scratch[row]);
    if (!std::isfinite(diagonal_response)) {
      return false;
    }
    workspace.potential_scratch[row] = diagonal_response;

    for (std::int64_t column = row + 1; column < atom_count; ++column) {
      const double upper = matrix[row * atom_count + column];
      const double lower = matrix[column * atom_count + row];
      if (!std::isfinite(upper) || !std::isfinite(lower) || upper != lower) {
        return false;
      }
      const double row_response =
          std::fma(upper, charges[column], workspace.potential_scratch[row]);
      const double column_response =
          std::fma(upper, charges[row], workspace.potential_scratch[column]);
      if (!std::isfinite(row_response) || !std::isfinite(column_response)) {
        return false;
      }
      workspace.potential_scratch[row] = row_response;
      workspace.potential_scratch[column] = column_response;
    }
  }

  double linear_energy = 0.0;
  double quadratic_energy = 0.0;
  for (std::int64_t atom = 0; atom < atom_count; ++atom) {
    const double response = workspace.potential_scratch[atom];
    const double potential = shifts[atom] + response;
    linear_energy = std::fma(charges[atom], shifts[atom], linear_energy);
    quadratic_energy = std::fma(charges[atom], response, quadratic_energy);
    if (!std::isfinite(potential) || !std::isfinite(linear_energy) ||
        !std::isfinite(quadratic_energy)) {
      return false;
    }
    workspace.potential_scratch[atom] = potential;
  }
  const double energy = std::fma(0.5, quadratic_energy, linear_energy);
  if (!std::isfinite(energy)) {
    return false;
  }

  if (atom_count != 0) {
    std::memcpy(view.atomic_potentials + static_cast<std::size_t>(atom_begin),
                workspace.potential_scratch, static_cast<std::size_t>(atom_count) * sizeof(double));
  }
  view.energies[system] = energy;
  view.system_statuses[system] = GPUXTB_STATUS_SUCCESS;
  return true;
}

}  // namespace

namespace {

const std::vector<std::int64_t> kEmptyInt64Vector;

}  // namespace

PeriodicEmbeddingPlan::PeriodicEmbeddingPlan(
    std::shared_ptr<const PeriodicEmbeddingPlanData> data) noexcept
    : data_(std::move(data)) {}

bool PeriodicEmbeddingPlan::sealed() const noexcept { return data_ != nullptr; }

std::int64_t PeriodicEmbeddingPlan::batch_size() const noexcept {
  return data_ == nullptr ? 0 : data_->batch_size;
}

std::int64_t PeriodicEmbeddingPlan::total_atoms() const noexcept {
  return data_ == nullptr ? 0 : data_->total_atoms;
}

std::int64_t PeriodicEmbeddingPlan::total_matrix_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->total_matrix_elements;
}

std::int64_t PeriodicEmbeddingPlan::maximum_atoms() const noexcept {
  return data_ == nullptr ? 0 : data_->maximum_atoms;
}

std::size_t PeriodicEmbeddingPlan::resident_bytes() const noexcept {
  if (data_ == nullptr) {
    return 0u;
  }
  return sizeof(*data_) + data_->atom_offsets.capacity() * sizeof(std::int64_t) +
         data_->matrix_offsets.capacity() * sizeof(std::int64_t);
}

const std::vector<std::int64_t>& PeriodicEmbeddingPlan::atom_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->atom_offsets;
}

const std::vector<std::int64_t>& PeriodicEmbeddingPlan::matrix_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->matrix_offsets;
}

bool PeriodicEmbeddingPlan::overlaps_storage(const void* data,
                                             std::size_t size_bytes) const noexcept {
  AddressRange range;
  return size_bytes != 0u && (data_ == nullptr || !make_range(data, size_bytes, range) ||
                              overlaps_plan_storage(*this, range));
}

const PeriodicEmbeddingPlanData* PeriodicEmbeddingPlan::identity() const noexcept {
  return data_.get();
}

gpuxtb_status_t make_periodic_embedding_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                             const std::int64_t* atom_offsets,
                                             PeriodicEmbeddingPlan& plan, std::string& error) {
  if (batch_size <= 0 || total_atoms < 0 ||
      batch_size == std::numeric_limits<std::int64_t>::max() ||
      !count_fits_storage(batch_size, sizeof(std::int64_t), true) ||
      !count_fits_storage(total_atoms, sizeof(double)) || atom_offsets == nullptr ||
      !is_aligned(atom_offsets, alignof(std::int64_t))) {
    error = "periodic embedding requires positive batch size and valid atom offsets";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  try {
    auto created = std::make_shared<PeriodicEmbeddingPlanData>();
    created->batch_size = batch_size;
    created->total_atoms = total_atoms;
    const std::size_t offset_count = static_cast<std::size_t>(batch_size) + 1u;
    created->atom_offsets.assign(atom_offsets, atom_offsets + offset_count);
    created->matrix_offsets.resize(offset_count, 0);
    if (created->atom_offsets.front() != 0 || created->atom_offsets.back() != total_atoms) {
      error = "periodic embedding atom offsets must span exactly total_atoms";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    for (std::size_t system = 0u; system < static_cast<std::size_t>(batch_size); ++system) {
      const std::int64_t begin = created->atom_offsets[system];
      const std::int64_t end = created->atom_offsets[system + 1u];
      std::int64_t matrix_elements = 0;
      if (begin < 0 || begin > end || end > total_atoms ||
          !checked_square(end - begin, matrix_elements) ||
          !checked_add(matrix_elements, created->total_matrix_elements)) {
        error = "periodic embedding atom partition or dense matrix extent overflows";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      created->maximum_atoms = std::max(created->maximum_atoms, end - begin);
      created->matrix_offsets[system + 1u] = created->total_matrix_elements;
    }
    if (!count_fits_storage(created->total_matrix_elements, sizeof(double)) ||
        !count_fits_storage(created->maximum_atoms, sizeof(double))) {
      error = "periodic embedding plan dimensions exceed host storage limits";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    PeriodicEmbeddingPlan completed(std::move(created));
    const gpuxtb_status_t status = validate_plan(completed, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
    plan = std::move(completed);
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate periodic embedding plan";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "periodic embedding plan dimensions exceed host container limits";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

gpuxtb_status_t bind_periodic_embedding_view(
    const PeriodicEmbeddingPlan& plan, const double* shifts, std::size_t shift_elements,
    const double* response_matrices, std::size_t response_elements, const double* atomic_charges,
    std::size_t charge_elements, double* atomic_potentials, std::size_t potential_elements,
    double* energies, std::size_t energy_elements, gpuxtb_status_t* system_statuses,
    std::size_t status_elements, PeriodicEmbeddingView& view, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const auto fits_i64 = [](std::size_t value) {
    return value <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
  };
  if (!fits_i64(shift_elements) || !fits_i64(response_elements) || !fits_i64(charge_elements) ||
      !fits_i64(potential_elements) || !fits_i64(energy_elements) || !fits_i64(status_elements)) {
    error = "periodic embedding view counts exceed signed backend dimensions";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  PeriodicEmbeddingView completed{shifts,
                                  static_cast<std::int64_t>(shift_elements),
                                  response_matrices,
                                  static_cast<std::int64_t>(response_elements),
                                  atomic_charges,
                                  static_cast<std::int64_t>(charge_elements),
                                  atomic_potentials,
                                  static_cast<std::int64_t>(potential_elements),
                                  energies,
                                  static_cast<std::int64_t>(energy_elements),
                                  system_statuses,
                                  static_cast<std::int64_t>(status_elements),
                                  plan.identity()};
  status = validate_view_shape(plan, completed, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_view_binding_ranges(plan, completed, view, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  view = completed;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t bind_periodic_embedding_workspace(const PeriodicEmbeddingPlan& plan,
                                                  double* potential_scratch,
                                                  std::size_t potential_elements,
                                                  PeriodicEmbeddingWorkspace& workspace,
                                                  std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (potential_elements > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error = "periodic embedding workspace count exceeds signed backend dimensions";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  PeriodicEmbeddingWorkspace completed{
      potential_scratch, static_cast<std::int64_t>(potential_elements), plan.identity()};
  status = validate_workspace_shape(plan, completed, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_workspace_binding_range(plan, completed, workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  workspace = completed;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t evaluate_periodic_embedding_system_cpu(const PeriodicEmbeddingPlan& plan,
                                                       std::int64_t system,
                                                       const PeriodicEmbeddingView& view,
                                                       const PeriodicEmbeddingWorkspace& workspace,
                                                       std::string& error) {
  const gpuxtb_status_t status = validate_call(plan, view, workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (system < 0 || system >= plan.batch_size()) {
    error = "periodic embedding system index is out of range";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t system_index = static_cast<std::size_t>(system);
  if (!evaluate_system_unchecked(plan, system_index, view, workspace)) {
    view.system_statuses[system_index] = GPUXTB_STATUS_INTERNAL_ERROR;
    error = "periodic embedding system contains invalid numerical data or overflowed";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t evaluate_periodic_embedding_batch_cpu(const PeriodicEmbeddingPlan& plan,
                                                      const PeriodicEmbeddingView& view,
                                                      const PeriodicEmbeddingWorkspace& workspace,
                                                      std::string& error) {
  const gpuxtb_status_t status = validate_call(plan, view, workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  bool failed = false;
  for (std::size_t system = 0u; system < static_cast<std::size_t>(plan.batch_size()); ++system) {
    if (!evaluate_system_unchecked(plan, system, view, workspace)) {
      view.system_statuses[system] = GPUXTB_STATUS_INTERNAL_ERROR;
      failed = true;
    }
  }
  if (failed) {
    error = "periodic embedding failed numerically for at least one system";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail::gfn2
