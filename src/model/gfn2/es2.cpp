#include "model/gfn2/es2.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#include "data/parameters/gfn2.hpp"

namespace gpuxtb::detail::gfn2 {

struct ES2PlanData final {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_matrix_elements = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<double> shell_hardness;
};

namespace {

static_assert(parameters::gfn2::kGlobal.charge_average == 0u,
              "GFN2 ES2 requires arithmetic shell-hardness averaging");
static_assert(parameters::gfn2::kGlobal.charge_gexp == 2.0,
              "GFN2 ES2 implementation specializes the gexp=2 kernel");
static_assert(std::is_trivially_copyable_v<ES2GeometryCache>);
static_assert(std::is_standard_layout_v<ES2GeometryCache>);
static_assert(std::is_trivially_copyable_v<ES2Workspace>);
static_assert(std::is_standard_layout_v<ES2Workspace>);

bool count_fits_storage(std::int64_t count, std::size_t element_size, bool add_sentinel = false) {
  if (count < 0) {
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

bool checked_add(std::int64_t increment, std::int64_t& total) {
  if (increment < 0 || total > std::numeric_limits<std::int64_t>::max() - increment) {
    return false;
  }
  total += increment;
  return true;
}

bool checked_square(std::int64_t value, std::int64_t& square) {
  if (value < 0 || (value != 0 && value > std::numeric_limits<std::int64_t>::max() / value)) {
    return false;
  }
  square = value * value;
  return true;
}

bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                    std::size_t second_bytes) {
  if (first_bytes == 0u || second_bytes == 0u) {
    return false;
  }
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  const auto first_end = first_begin + first_bytes;
  const auto second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

bool is_aligned_double(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(double) == 0u;
}

gpuxtb_status_t validate_plan(const ES2Plan& plan, std::string& error) {
  if (!plan.sealed()) {
    error = "ES2 plan is default-constructed, moved-from, or otherwise unsealed";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_cache(const ES2Plan& plan, const ES2GeometryCache& cache,
                               std::string& error) {
  if (cache.coulomb_matrix == nullptr || cache.matrix_elements != plan.total_matrix_elements() ||
      !is_aligned_double(cache.coulomb_matrix) || cache.plan_identity != plan.identity()) {
    error = "ES2 geometry cache is missing or belongs to a different plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_scratch(double* pointer, std::int64_t available, std::int64_t required,
                                 const char* message, std::string& error) {
  if (pointer == nullptr || available < required || !is_aligned_double(pointer)) {
    error = message;
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_positions_pointer(const double* positions, std::string& error) {
  if (positions == nullptr || !is_aligned_double(positions)) {
    error = "ES2 positions must not be NULL or misaligned";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_position_values(const ES2Plan& plan, const double* positions,
                                         std::string& error) {
  const std::size_t atom_count = static_cast<std::size_t>(plan.total_atoms());
  for (std::size_t coordinate = 0; coordinate < atom_count * 3u; ++coordinate) {
    if (!std::isfinite(positions[coordinate])) {
      error = "ES2 positions contain NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_shell_charge_pointer(const double* shell_charges, std::string& error) {
  if (shell_charges == nullptr || !is_aligned_double(shell_charges)) {
    error = "ES2 shell charges must not be NULL or misaligned";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_shell_charge_values(const ES2Plan& plan, const double* shell_charges,
                                             std::string& error) {
  for (std::int64_t shell = 0; shell < plan.total_shells(); ++shell) {
    if (!std::isfinite(shell_charges[static_cast<std::size_t>(shell)])) {
      error = "ES2 shell charges contain NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

struct MemoryRange {
  const void* data = nullptr;
  std::size_t size_bytes = 0;
};

std::array<MemoryRange, 8> plan_storage_ranges(const ES2Plan& plan) {
  return {{{&plan, sizeof(plan)},
           {plan.identity(), sizeof(ES2PlanData)},
           {plan.atom_offsets().data(), plan.atom_offsets().capacity() * sizeof(std::int64_t)},
           {plan.batch_shell_offsets().data(),
            plan.batch_shell_offsets().capacity() * sizeof(std::int64_t)},
           {plan.atom_shell_offsets().data(),
            plan.atom_shell_offsets().capacity() * sizeof(std::int64_t)},
           {plan.matrix_offsets().data(), plan.matrix_offsets().capacity() * sizeof(std::int64_t)},
           {plan.shell_to_atom().data(), plan.shell_to_atom().capacity() * sizeof(std::int64_t)},
           {plan.shell_hardness().data(), plan.shell_hardness().capacity() * sizeof(double)}}};
}

bool overlaps_plan_storage(const ES2Plan& plan, const void* data, std::size_t size_bytes) {
  for (const MemoryRange& range : plan_storage_ranges(plan)) {
    if (ranges_overlap(data, size_bytes, range.data, range.size_bytes)) {
      return true;
    }
  }
  return false;
}

bool overlaps_control_storage(const ES2Plan& plan, const ES2GeometryCache& cache,
                              const ES2Workspace& workspace, const void* data,
                              std::size_t size_bytes) {
  /*
   * These descriptor objects remain live throughout every operation. Treat
   * them like immutable plan storage so a reinterpreted caller buffer cannot
   * overwrite a pointer/count that a later loop iteration will dereference.
   */
  return overlaps_plan_storage(plan, data, size_bytes) ||
         ranges_overlap(data, size_bytes, &cache, sizeof(cache)) ||
         ranges_overlap(data, size_bytes, &workspace, sizeof(workspace));
}

std::size_t matrix_index(const ES2Plan& plan, std::size_t batch, std::int64_t row_shell,
                         std::int64_t column_shell) {
  const std::int64_t shell_begin = plan.batch_shell_offsets()[batch];
  const std::int64_t molecule_shells = plan.batch_shell_offsets()[batch + 1u] - shell_begin;
  return static_cast<std::size_t>(plan.matrix_offsets()[batch] +
                                  (row_shell - shell_begin) * molecule_shells +
                                  (column_shell - shell_begin));
}

double arithmetic_hardness(double first_hardness, double second_hardness) {
  const double sum = first_hardness + second_hardness;
  if (std::isfinite(sum)) {
    /* Add-before-half preserves positive subnormal averages. */
    return 0.5 * sum;
  }
  /* Half-before-add is only needed when the positive finite sum overflows. */
  return 0.5 * first_hardness + 0.5 * second_hardness;
}

bool softened_kernel(double dx, double dy, double dz, double first_hardness, double second_hardness,
                     double& value) {
  const double average_hardness = arithmetic_hardness(first_hardness, second_hardness);
  const double inverse_average_hardness = 1.0 / average_hardness;
  const double softened_distance =
      std::hypot(std::hypot(dx, dy), std::hypot(dz, inverse_average_hardness));
  value = 1.0 / softened_distance;
  return average_hardness > 0.0 && std::isfinite(average_hardness) &&
         std::isfinite(inverse_average_hardness) && softened_distance > 0.0 &&
         std::isfinite(softened_distance) && value > 0.0 && std::isfinite(value);
}

bool calculate_potential_row(const ES2Plan& plan, const ES2GeometryCache& cache,
                             const double* shell_charges, std::size_t batch, std::int64_t row_shell,
                             double& potential) {
  const std::int64_t shell_begin = plan.batch_shell_offsets()[batch];
  const std::int64_t shell_end = plan.batch_shell_offsets()[batch + 1u];
  potential = 0.0;
  for (std::int64_t column_shell = shell_begin; column_shell < shell_end; ++column_shell) {
    const double kernel = cache.coulomb_matrix[matrix_index(plan, batch, row_shell, column_shell)];
    const double contribution = kernel * shell_charges[static_cast<std::size_t>(column_shell)];
    const double updated = potential + contribution;
    if (!(kernel > 0.0) || !std::isfinite(kernel) || !std::isfinite(contribution) ||
        !std::isfinite(updated)) {
      return false;
    }
    potential = updated;
  }
  return true;
}

bool calculate_batch_energy(const ES2Plan& plan, const ES2GeometryCache& cache,
                            const double* shell_charges, std::size_t batch, double& energy) {
  const std::int64_t shell_begin = plan.batch_shell_offsets()[batch];
  const std::int64_t shell_end = plan.batch_shell_offsets()[batch + 1u];
  energy = 0.0;
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
    double potential = 0.0;
    if (!calculate_potential_row(plan, cache, shell_charges, batch, shell, potential)) {
      return false;
    }
    const double contribution = 0.5 * shell_charges[static_cast<std::size_t>(shell)] * potential;
    const double updated = energy + contribution;
    if (!std::isfinite(contribution) || !std::isfinite(updated)) {
      return false;
    }
    energy = updated;
  }
  return true;
}

}  // namespace

namespace {

const std::vector<std::int64_t> kEmptyInt64Vector;
const std::vector<double> kEmptyDoubleVector;

}  // namespace

ES2Plan::ES2Plan(std::shared_ptr<const ES2PlanData> data) noexcept : data_(std::move(data)) {}

bool ES2Plan::sealed() const noexcept { return data_ != nullptr; }

std::int64_t ES2Plan::batch_size() const noexcept {
  return data_ == nullptr ? 0 : data_->batch_size;
}

std::int64_t ES2Plan::total_atoms() const noexcept {
  return data_ == nullptr ? 0 : data_->total_atoms;
}

std::int64_t ES2Plan::total_shells() const noexcept {
  return data_ == nullptr ? 0 : data_->total_shells;
}

std::int64_t ES2Plan::total_matrix_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->total_matrix_elements;
}

const std::vector<std::int64_t>& ES2Plan::atom_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->atom_offsets;
}

const std::vector<std::int64_t>& ES2Plan::batch_shell_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->batch_shell_offsets;
}

const std::vector<std::int64_t>& ES2Plan::atom_shell_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->atom_shell_offsets;
}

const std::vector<std::int64_t>& ES2Plan::matrix_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->matrix_offsets;
}

const std::vector<std::int64_t>& ES2Plan::shell_to_atom() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->shell_to_atom;
}

const std::vector<double>& ES2Plan::shell_hardness() const noexcept {
  return data_ == nullptr ? kEmptyDoubleVector : data_->shell_hardness;
}

bool ES2Plan::overlaps_storage(const void* data, std::size_t size_bytes) const noexcept {
  return size_bytes != 0u && (data_ == nullptr || overlaps_plan_storage(*this, data, size_bytes));
}

const ES2PlanData* ES2Plan::identity() const noexcept { return data_.get(); }

gpuxtb_status_t make_es2_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                              ES2Plan& plan, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      !count_fits_storage(basis.batch_size, sizeof(std::int64_t), true) ||
      !count_fits_storage(basis.total_atoms, sizeof(std::int64_t), true) ||
      !count_fits_storage(basis.total_shells, sizeof(std::int64_t)) || atomic_numbers == nullptr) {
    error = "ES2 plan requires a representable basis and atomic numbers";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch_count = static_cast<std::size_t>(basis.batch_size);
  const std::size_t atom_count = static_cast<std::size_t>(basis.total_atoms);
  const std::size_t shell_count = static_cast<std::size_t>(basis.total_shells);
  if (basis.atom_offsets.size() != batch_count + 1u ||
      basis.batch_shell_offsets.size() != batch_count + 1u ||
      basis.atom_shell_offsets.size() != atom_count + 1u ||
      basis.shell_to_atom.size() != shell_count ||
      basis.principal_quantum_numbers.size() != shell_count ||
      basis.angular_momenta.size() != shell_count || basis.slater_exponents.size() != shell_count ||
      basis.atom_offsets.front() != 0 || basis.atom_offsets.back() != basis.total_atoms ||
      basis.batch_shell_offsets.front() != 0 ||
      basis.batch_shell_offsets.back() != basis.total_shells ||
      basis.atom_shell_offsets.front() != 0 ||
      basis.atom_shell_offsets.back() != basis.total_shells) {
    error = "ES2 plan received an inconsistent basis plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  std::int64_t total_matrix_elements = 0;
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const std::int64_t atom_begin = basis.atom_offsets[batch];
    const std::int64_t atom_end = basis.atom_offsets[batch + 1u];
    const std::int64_t shell_begin = basis.batch_shell_offsets[batch];
    const std::int64_t shell_end = basis.batch_shell_offsets[batch + 1u];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > basis.total_atoms ||
        shell_begin < 0 || shell_begin > shell_end || shell_end > basis.total_shells ||
        shell_begin != basis.atom_shell_offsets[static_cast<std::size_t>(atom_begin)] ||
        shell_end != basis.atom_shell_offsets[static_cast<std::size_t>(atom_end)]) {
      error = "ES2 basis offsets are not valid ragged partitions";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    std::int64_t matrix_elements = 0;
    if (!checked_square(shell_end - shell_begin, matrix_elements) ||
        !checked_add(matrix_elements, total_matrix_elements)) {
      error = "ES2 ragged Coulomb matrices exceed representable storage";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  if (total_matrix_elements <= 0 || !count_fits_storage(total_matrix_elements, sizeof(double))) {
    error = "ES2 Coulomb matrix storage is not representable";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  try {
    ES2PlanData created;
    created.batch_size = basis.batch_size;
    created.total_atoms = basis.total_atoms;
    created.total_shells = basis.total_shells;
    created.total_matrix_elements = total_matrix_elements;
    created.atom_offsets = basis.atom_offsets;
    created.batch_shell_offsets = basis.batch_shell_offsets;
    created.atom_shell_offsets = basis.atom_shell_offsets;
    created.shell_to_atom = basis.shell_to_atom;
    created.matrix_offsets.resize(batch_count + 1u);
    created.shell_hardness.resize(shell_count);

    std::int64_t matrix_offset = 0;
    for (std::size_t batch = 0; batch < batch_count; ++batch) {
      created.matrix_offsets[batch] = matrix_offset;
      const std::int64_t molecule_shells =
          basis.batch_shell_offsets[batch + 1u] - basis.batch_shell_offsets[batch];
      matrix_offset += molecule_shells * molecule_shells;
    }
    created.matrix_offsets[batch_count] = matrix_offset;

    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::size_t atom_index = static_cast<std::size_t>(atom);
      const std::int32_t atomic_number = atomic_numbers[atom_index];
      const auto* element =
          parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number || !(element->gam > 0.0) ||
          !std::isfinite(element->gam)) {
        error = "ES2 plan contains an unsupported element or invalid hardness";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }

      const std::int64_t shell_begin = basis.atom_shell_offsets[atom_index];
      const std::int64_t shell_end = basis.atom_shell_offsets[atom_index + 1u];
      const std::size_t parameter_begin = element->shell_offset;
      if (shell_begin < 0 || shell_begin >= shell_end || shell_end > basis.total_shells ||
          shell_end - shell_begin != element->shell_count ||
          parameter_begin > parameters::gfn2::kShells.size() ||
          element->shell_count > parameters::gfn2::kShells.size() - parameter_begin) {
        error = "ES2 element list does not match the basis shell layout";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }

      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        const std::size_t shell_index = static_cast<std::size_t>(shell);
        const std::size_t local_shell = static_cast<std::size_t>(shell - shell_begin);
        const auto& shell_parameters = parameters::gfn2::kShells[parameter_begin + local_shell];
        if (basis.shell_to_atom[shell_index] != atom ||
            basis.principal_quantum_numbers[shell_index] !=
                shell_parameters.principal_quantum_number ||
            basis.angular_momenta[shell_index] != shell_parameters.angular_momentum ||
            basis.slater_exponents[shell_index] != shell_parameters.slater ||
            !(shell_parameters.shell_hubbard_scale > 0.0) ||
            !std::isfinite(shell_parameters.shell_hubbard_scale)) {
          error = "ES2 element list does not match the basis shell metadata";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        const double hardness = element->gam * shell_parameters.shell_hubbard_scale;
        if (!(hardness > 0.0) || !std::isfinite(hardness)) {
          error = "ES2 generated shell hardness is invalid";
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
        created.shell_hardness[shell_index] = hardness;
      }
    }

    auto sealed = std::make_shared<const ES2PlanData>(std::move(created));
    plan = ES2Plan(std::move(sealed));
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN2 ES2 plan";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

gpuxtb_status_t update_es2_geometry_cache_cpu(const ES2Plan& plan, const double* positions,
                                              std::uint64_t geometry_generation,
                                              double* matrix_storage,
                                              std::size_t matrix_storage_elements,
                                              const ES2Workspace& workspace,
                                              ES2GeometryCache& cache, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_positions_pointer(positions, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (matrix_storage == nullptr ||
      matrix_storage_elements < static_cast<std::size_t>(plan.total_matrix_elements()) ||
      !is_aligned_double(matrix_storage)) {
    error = "ES2 matrix storage is NULL, too small, or misaligned";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_scratch(workspace.matrix_scratch, workspace.matrix_elements,
                            plan.total_matrix_elements(),
                            "ES2 matrix scratch is NULL, too small, or misaligned", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const std::size_t position_bytes =
      static_cast<std::size_t>(plan.total_atoms()) * 3u * sizeof(double);
  const std::size_t matrix_bytes =
      static_cast<std::size_t>(plan.total_matrix_elements()) * sizeof(double);
  if (cache.coulomb_matrix != nullptr) {
    status = validate_cache(plan, cache, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
  }
  if (overlaps_control_storage(plan, cache, workspace, positions, position_bytes) ||
      overlaps_control_storage(plan, cache, workspace, matrix_storage, matrix_bytes) ||
      overlaps_control_storage(plan, cache, workspace, workspace.matrix_scratch, matrix_bytes) ||
      (cache.coulomb_matrix != nullptr &&
       overlaps_control_storage(plan, cache, workspace, cache.coulomb_matrix, matrix_bytes)) ||
      ranges_overlap(positions, position_bytes, matrix_storage, matrix_bytes) ||
      ranges_overlap(positions, position_bytes, workspace.matrix_scratch, matrix_bytes) ||
      ranges_overlap(matrix_storage, matrix_bytes, workspace.matrix_scratch, matrix_bytes)) {
    error =
        "ES2 positions, matrix output, matrix scratch, cache, workspace, and plan storage must "
        "not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (cache.coulomb_matrix != nullptr) {
    if (ranges_overlap(cache.coulomb_matrix, matrix_bytes, workspace.matrix_scratch,
                       matrix_bytes)) {
      error = "ES2 active cache and matrix scratch must not overlap";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  status = validate_position_values(plan, positions, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    const std::int64_t atom_begin = plan.atom_offsets()[batch_index];
    const std::int64_t atom_end = plan.atom_offsets()[batch_index + 1u];
    for (std::int64_t first = atom_begin; first < atom_end; ++first) {
      const std::size_t first_index = static_cast<std::size_t>(first);
      const std::int64_t first_shell_begin = plan.atom_shell_offsets()[first_index];
      const std::int64_t first_shell_end = plan.atom_shell_offsets()[first_index + 1u];
      for (std::int64_t first_shell = first_shell_begin; first_shell < first_shell_end;
           ++first_shell) {
        for (std::int64_t second_shell = first_shell_begin; second_shell < first_shell_end;
             ++second_shell) {
          const double average_hardness =
              arithmetic_hardness(plan.shell_hardness()[static_cast<std::size_t>(first_shell)],
                                  plan.shell_hardness()[static_cast<std::size_t>(second_shell)]);
          const double inverse_average_hardness = 1.0 / average_hardness;
          if (!(average_hardness > 0.0) || !std::isfinite(average_hardness) ||
              !std::isfinite(inverse_average_hardness)) {
            error = "ES2 onsite hardness arithmetic exceeded floating-point range";
            return GPUXTB_STATUS_INVALID_ARGUMENT;
          }
          workspace.matrix_scratch[matrix_index(plan, batch_index, first_shell, second_shell)] =
              average_hardness;
        }
      }
      for (std::int64_t second = atom_begin; second < first; ++second) {
        const std::size_t second_index = static_cast<std::size_t>(second);
        const double dx = positions[first_index * 3u] - positions[second_index * 3u];
        const double dy = positions[first_index * 3u + 1u] - positions[second_index * 3u + 1u];
        const double dz = positions[first_index * 3u + 2u] - positions[second_index * 3u + 2u];
        if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
          error = "ES2 coordinate differences overflow floating-point range";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        const std::int64_t second_shell_begin = plan.atom_shell_offsets()[second_index];
        const std::int64_t second_shell_end = plan.atom_shell_offsets()[second_index + 1u];
        for (std::int64_t first_shell = first_shell_begin; first_shell < first_shell_end;
             ++first_shell) {
          for (std::int64_t second_shell = second_shell_begin; second_shell < second_shell_end;
               ++second_shell) {
            double kernel = 0.0;
            if (!softened_kernel(
                    dx, dy, dz, plan.shell_hardness()[static_cast<std::size_t>(first_shell)],
                    plan.shell_hardness()[static_cast<std::size_t>(second_shell)], kernel)) {
              error = "ES2 Coulomb-kernel arithmetic exceeded floating-point range";
              return GPUXTB_STATUS_INVALID_ARGUMENT;
            }
            workspace.matrix_scratch[matrix_index(plan, batch_index, first_shell, second_shell)] =
                kernel;
            workspace.matrix_scratch[matrix_index(plan, batch_index, second_shell, first_shell)] =
                kernel;
          }
        }
      }
    }
  }
  std::copy_n(workspace.matrix_scratch, static_cast<std::size_t>(plan.total_matrix_elements()),
              matrix_storage);
  cache.coulomb_matrix = matrix_storage;
  cache.matrix_elements = plan.total_matrix_elements();
  cache.geometry_generation = geometry_generation;
  cache.plan_identity = plan.identity();
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t evaluate_es2_potential_cpu(const ES2Plan& plan, const ES2GeometryCache& cache,
                                           const double* shell_charges, double* shell_potentials,
                                           const ES2Workspace& workspace, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_cache(plan, cache, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_shell_charge_pointer(shell_charges, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (shell_potentials == nullptr || !is_aligned_double(shell_potentials)) {
    error = "ES2 shell potential output must not be NULL or misaligned";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_scratch(workspace.shell_scratch, workspace.shell_elements, plan.total_shells(),
                            "ES2 shell scratch is NULL, too small, or misaligned", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const std::size_t shell_bytes = static_cast<std::size_t>(plan.total_shells()) * sizeof(double);
  const std::size_t matrix_bytes =
      static_cast<std::size_t>(plan.total_matrix_elements()) * sizeof(double);
  if (overlaps_control_storage(plan, cache, workspace, shell_charges, shell_bytes) ||
      overlaps_control_storage(plan, cache, workspace, cache.coulomb_matrix, matrix_bytes) ||
      overlaps_control_storage(plan, cache, workspace, shell_potentials, shell_bytes) ||
      overlaps_control_storage(plan, cache, workspace, workspace.shell_scratch, shell_bytes) ||
      ranges_overlap(shell_charges, shell_bytes, shell_potentials, shell_bytes) ||
      ranges_overlap(cache.coulomb_matrix, matrix_bytes, shell_potentials, shell_bytes) ||
      ranges_overlap(shell_charges, shell_bytes, workspace.shell_scratch, shell_bytes) ||
      ranges_overlap(cache.coulomb_matrix, matrix_bytes, workspace.shell_scratch, shell_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, workspace.shell_scratch, shell_bytes)) {
    error =
        "ES2 potential inputs, output, cache, shell scratch, descriptors, and plan storage must "
        "not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_shell_charge_values(plan, shell_charges, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    for (std::int64_t shell = plan.batch_shell_offsets()[batch_index];
         shell < plan.batch_shell_offsets()[batch_index + 1u]; ++shell) {
      double potential = 0.0;
      if (!calculate_potential_row(plan, cache, shell_charges, batch_index, shell, potential)) {
        error = "ES2 potential arithmetic exceeded floating-point range";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      workspace.shell_scratch[static_cast<std::size_t>(shell)] = potential;
    }
  }
  std::copy_n(workspace.shell_scratch, static_cast<std::size_t>(plan.total_shells()),
              shell_potentials);
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_es2_energy_cpu(const ES2Plan& plan, const ES2GeometryCache& cache,
                                   const double* shell_charges, double* energies,
                                   const ES2Workspace& workspace, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_cache(plan, cache, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_shell_charge_pointer(shell_charges, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (energies == nullptr || !is_aligned_double(energies)) {
    error = "ES2 energy output must not be NULL or misaligned";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_scratch(workspace.batch_scratch, workspace.batch_elements, plan.batch_size(),
                            "ES2 batch scratch is NULL, too small, or misaligned", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const std::size_t shell_bytes = static_cast<std::size_t>(plan.total_shells()) * sizeof(double);
  const std::size_t matrix_bytes =
      static_cast<std::size_t>(plan.total_matrix_elements()) * sizeof(double);
  const std::size_t energy_bytes = static_cast<std::size_t>(plan.batch_size()) * sizeof(double);
  if (overlaps_control_storage(plan, cache, workspace, shell_charges, shell_bytes) ||
      overlaps_control_storage(plan, cache, workspace, cache.coulomb_matrix, matrix_bytes) ||
      overlaps_control_storage(plan, cache, workspace, energies, energy_bytes) ||
      overlaps_control_storage(plan, cache, workspace, workspace.batch_scratch, energy_bytes) ||
      ranges_overlap(shell_charges, shell_bytes, energies, energy_bytes) ||
      ranges_overlap(cache.coulomb_matrix, matrix_bytes, energies, energy_bytes) ||
      ranges_overlap(shell_charges, shell_bytes, workspace.batch_scratch, energy_bytes) ||
      ranges_overlap(cache.coulomb_matrix, matrix_bytes, workspace.batch_scratch, energy_bytes) ||
      ranges_overlap(energies, energy_bytes, workspace.batch_scratch, energy_bytes)) {
    error =
        "ES2 energy inputs, output, cache, batch scratch, descriptors, and plan storage must not "
        "overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_shell_charge_values(plan, shell_charges, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    if (!calculate_batch_energy(plan, cache, shell_charges, batch_index,
                                workspace.batch_scratch[batch_index])) {
      error = "ES2 energy arithmetic exceeded floating-point range";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    if (!std::isfinite(energies[batch_index]) ||
        !std::isfinite(energies[batch_index] + workspace.batch_scratch[batch_index])) {
      error = "ES2 accumulated energy exceeded floating-point range";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    energies[batch_index] += workspace.batch_scratch[batch_index];
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_es2_energy_system_cpu(const ES2Plan& plan, const ES2GeometryCache& cache,
                                          std::int64_t system, const double* shell_charges,
                                          double& accumulated_energy, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_cache(plan, cache, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (system < 0 || system >= plan.batch_size()) {
    error = "ES2 energy system index is out of range";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_shell_charge_pointer(shell_charges, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  const std::size_t shell_bytes = static_cast<std::size_t>(plan.total_shells()) * sizeof(double);
  const std::size_t matrix_bytes =
      static_cast<std::size_t>(plan.total_matrix_elements()) * sizeof(double);
  const void* energy_pointer = &accumulated_energy;
  if (overlaps_plan_storage(plan, shell_charges, shell_bytes) ||
      overlaps_plan_storage(plan, cache.coulomb_matrix, matrix_bytes) ||
      overlaps_plan_storage(plan, energy_pointer, sizeof(double)) ||
      ranges_overlap(shell_charges, shell_bytes, &cache, sizeof(cache)) ||
      ranges_overlap(cache.coulomb_matrix, matrix_bytes, &cache, sizeof(cache)) ||
      ranges_overlap(energy_pointer, sizeof(double), &cache, sizeof(cache)) ||
      ranges_overlap(energy_pointer, sizeof(double), shell_charges, shell_bytes) ||
      ranges_overlap(energy_pointer, sizeof(double), cache.coulomb_matrix, matrix_bytes) ||
      ranges_overlap(energy_pointer, sizeof(double), &error, sizeof(error))) {
    error = "ES2 one-system energy inputs, output, cache, error, and plan storage must not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t system_index = static_cast<std::size_t>(system);
  const std::int64_t shell_begin = plan.batch_shell_offsets()[system_index];
  const std::int64_t shell_end = plan.batch_shell_offsets()[system_index + 1u];
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
    if (!std::isfinite(shell_charges[static_cast<std::size_t>(shell)])) {
      error = "ES2 target-system shell charges contain NaN or infinity";
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
  }

  double contribution = 0.0;
  if (!calculate_batch_energy(plan, cache, shell_charges, system_index, contribution) ||
      !std::isfinite(accumulated_energy) || !std::isfinite(accumulated_energy + contribution)) {
    error = "ES2 target-system energy arithmetic exceeded floating-point range";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  accumulated_energy += contribution;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_es2_gradient_cpu(const ES2Plan& plan, const ES2GeometryCache& cache,
                                     const double* positions, std::uint64_t geometry_generation,
                                     const double* shell_charges, double* gradients,
                                     const ES2Workspace& workspace, std::string& error) {
  gpuxtb_status_t status = validate_plan(plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_cache(plan, cache, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (cache.geometry_generation != geometry_generation) {
    error = "ES2 gradient positions do not match the cached geometry generation";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_positions_pointer(positions, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_shell_charge_pointer(shell_charges, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (gradients == nullptr || !is_aligned_double(gradients)) {
    error = "ES2 gradient output must not be NULL or misaligned";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::int64_t gradient_elements = plan.total_atoms() * 3;
  status =
      validate_scratch(workspace.gradient_scratch, workspace.gradient_elements, gradient_elements,
                       "ES2 gradient scratch is NULL, too small, or misaligned", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const std::size_t position_bytes =
      static_cast<std::size_t>(plan.total_atoms()) * 3u * sizeof(double);
  const std::size_t shell_bytes = static_cast<std::size_t>(plan.total_shells()) * sizeof(double);
  const std::size_t matrix_bytes =
      static_cast<std::size_t>(plan.total_matrix_elements()) * sizeof(double);
  if (overlaps_control_storage(plan, cache, workspace, positions, position_bytes) ||
      overlaps_control_storage(plan, cache, workspace, shell_charges, shell_bytes) ||
      overlaps_control_storage(plan, cache, workspace, cache.coulomb_matrix, matrix_bytes) ||
      overlaps_control_storage(plan, cache, workspace, gradients, position_bytes) ||
      overlaps_control_storage(plan, cache, workspace, workspace.gradient_scratch,
                               position_bytes) ||
      ranges_overlap(positions, position_bytes, gradients, position_bytes) ||
      ranges_overlap(shell_charges, shell_bytes, gradients, position_bytes) ||
      ranges_overlap(cache.coulomb_matrix, matrix_bytes, gradients, position_bytes) ||
      ranges_overlap(positions, position_bytes, workspace.gradient_scratch, position_bytes) ||
      ranges_overlap(shell_charges, shell_bytes, workspace.gradient_scratch, position_bytes) ||
      ranges_overlap(cache.coulomb_matrix, matrix_bytes, workspace.gradient_scratch,
                     position_bytes) ||
      ranges_overlap(gradients, position_bytes, workspace.gradient_scratch, position_bytes)) {
    error =
        "ES2 gradient inputs, output, cache, gradient scratch, descriptors, and plan storage must "
        "not overlap";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  status = validate_position_values(plan, positions, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_shell_charge_values(plan, shell_charges, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  std::fill_n(workspace.gradient_scratch, static_cast<std::size_t>(gradient_elements), 0.0);
  for (std::int64_t batch = 0; batch < plan.batch_size(); ++batch) {
    const std::size_t batch_index = static_cast<std::size_t>(batch);
    const std::int64_t atom_begin = plan.atom_offsets()[batch_index];
    const std::int64_t atom_end = plan.atom_offsets()[batch_index + 1u];
    for (std::int64_t first = atom_begin; first < atom_end; ++first) {
      const std::size_t first_index = static_cast<std::size_t>(first);
      const std::int64_t first_shell_begin = plan.atom_shell_offsets()[first_index];
      const std::int64_t first_shell_end = plan.atom_shell_offsets()[first_index + 1u];
      for (std::int64_t second = atom_begin; second < first; ++second) {
        const std::size_t second_index = static_cast<std::size_t>(second);
        const double displacement[3]{
            positions[first_index * 3u] - positions[second_index * 3u],
            positions[first_index * 3u + 1u] - positions[second_index * 3u + 1u],
            positions[first_index * 3u + 2u] - positions[second_index * 3u + 2u],
        };
        if (!std::isfinite(displacement[0]) || !std::isfinite(displacement[1]) ||
            !std::isfinite(displacement[2])) {
          error = "ES2 coordinate differences overflow floating-point range";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        const std::int64_t second_shell_begin = plan.atom_shell_offsets()[second_index];
        const std::int64_t second_shell_end = plan.atom_shell_offsets()[second_index + 1u];
        double weighted_kernel_derivative = 0.0;
        for (std::int64_t first_shell = first_shell_begin; first_shell < first_shell_end;
             ++first_shell) {
          for (std::int64_t second_shell = second_shell_begin; second_shell < second_shell_end;
               ++second_shell) {
            const double kernel =
                cache.coulomb_matrix[matrix_index(plan, batch_index, first_shell, second_shell)];
            double shell_contribution =
                shell_charges[static_cast<std::size_t>(first_shell)] * kernel;
            shell_contribution *= shell_charges[static_cast<std::size_t>(second_shell)];
            shell_contribution *= kernel;
            shell_contribution *= kernel;
            const double updated = weighted_kernel_derivative + shell_contribution;
            if (!(kernel > 0.0) || !std::isfinite(kernel) || !std::isfinite(shell_contribution) ||
                !std::isfinite(updated)) {
              error = "ES2 coordinate VJP arithmetic exceeded floating-point range";
              return GPUXTB_STATUS_INVALID_ARGUMENT;
            }
            weighted_kernel_derivative = updated;
          }
        }
        for (std::size_t axis = 0; axis < 3u; ++axis) {
          const double pair_contribution = -weighted_kernel_derivative * displacement[axis];
          const std::size_t first_coordinate = first_index * 3u + axis;
          const std::size_t second_coordinate = second_index * 3u + axis;
          const double updated_first =
              workspace.gradient_scratch[first_coordinate] + pair_contribution;
          const double updated_second =
              workspace.gradient_scratch[second_coordinate] - pair_contribution;
          if (!std::isfinite(pair_contribution) || !std::isfinite(updated_first) ||
              !std::isfinite(updated_second)) {
            error = "ES2 coordinate VJP arithmetic exceeded floating-point range";
            return GPUXTB_STATUS_INVALID_ARGUMENT;
          }
          workspace.gradient_scratch[first_coordinate] = updated_first;
          workspace.gradient_scratch[second_coordinate] = updated_second;
        }
      }
    }
  }

  for (std::size_t coordinate = 0; coordinate < static_cast<std::size_t>(gradient_elements);
       ++coordinate) {
    if (!std::isfinite(gradients[coordinate]) ||
        !std::isfinite(gradients[coordinate] + workspace.gradient_scratch[coordinate])) {
      error = "ES2 accumulated gradient exceeded floating-point range";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t coordinate = 0; coordinate < static_cast<std::size_t>(gradient_elements);
       ++coordinate) {
    gradients[coordinate] += workspace.gradient_scratch[coordinate];
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail::gfn2
