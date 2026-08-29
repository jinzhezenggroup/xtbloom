// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/periodic_topology.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace xtbloom::detail::gfn2 {

struct PeriodicShortRangePlanData {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<Lattice3D> lattices;
  std::array<std::vector<std::int64_t>, 3> translation_offsets;
  std::array<std::vector<LatticeTranslation>, 3> translations;
  std::size_t workspace_size_bytes = 0u;
  std::size_t wrapped_positions_offset = 0u;
  std::size_t atom_scratch_offset = 0u;
  std::size_t secondary_atom_scratch_offset = 0u;
  std::size_t gradient_scratch_offset = 0u;
  std::size_t strain_scratch_offset = 0u;
  std::size_t batch_scratch_offset = 0u;
};

namespace {

constexpr std::array<double, 3> kCutoffs{
    kPeriodicShortRangeCutoffBohr,
    kPeriodicD4CoordinationCutoffBohr,
    kPeriodicD4TwoBodyCutoffBohr,
};

bool valid_count(std::int64_t value) noexcept {
  return value >= 0 && static_cast<std::uint64_t>(value) <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

bool checked_multiply(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first != 0u && second > std::numeric_limits<std::size_t>::max() / first) return false;
  result = first * second;
  return true;
}

bool checked_add(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first > std::numeric_limits<std::size_t>::max() - second) return false;
  result = first + second;
  return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t& result) noexcept {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) return false;
  const std::size_t mask = alignment - 1u;
  if (value > std::numeric_limits<std::size_t>::max() - mask) return false;
  result = (value + mask) & ~mask;
  return true;
}

bool append_doubles(std::size_t elements, std::size_t& cursor, std::size_t& offset) noexcept {
  std::size_t bytes = 0u;
  if (!checked_multiply(elements, sizeof(double), bytes) ||
      !align_up(cursor, alignof(double), offset)) {
    return false;
  }
  return checked_add(offset, bytes, cursor);
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) noexcept {
  if (bytes != 0u && pointer == nullptr) return false;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
  range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

template <typename T>
bool overlaps_vector(const AddressRange& active, const std::vector<T>& values) noexcept {
  std::size_t bytes = 0u;
  AddressRange storage;
  return checked_multiply(values.capacity(), sizeof(T), bytes) &&
         make_range(values.data(), bytes, storage) && ranges_overlap(active, storage);
}

template <typename T>
T* offset_pointer(void* base, std::size_t offset) noexcept {
  return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

std::size_t cutoff_index(PeriodicTranslationCutoff cutoff) noexcept {
  switch (cutoff) {
    case PeriodicTranslationCutoff::kShortRange25:
      return 0u;
    case PeriodicTranslationCutoff::kD4Coordination30:
      return 1u;
    case PeriodicTranslationCutoff::kD4TwoBody50:
      return 2u;
  }
  return 3u;
}

xtbloom_status_t validate_plan(const PeriodicShortRangePlan& plan, std::string& error) {
  if (!plan.sealed() || plan.batch_size() <= 0 || plan.total_atoms() <= 0 ||
      plan.atom_offsets().size() != static_cast<std::size_t>(plan.batch_size() + 1) ||
      plan.atom_offsets().front() != 0 || plan.atom_offsets().back() != plan.total_atoms()) {
    error = "periodic short-range plan is incomplete or internally inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_workspace_layout(const PeriodicShortRangePlan& plan,
                                           const PeriodicShortRangeWorkspace& workspace,
                                           std::string& error) {
  const PeriodicShortRangePlanData& data = *plan.identity();
  if (workspace.plan_identity != plan.identity() || workspace.workspace_base == nullptr ||
      reinterpret_cast<std::uintptr_t>(workspace.workspace_base) %
              kPeriodicShortRangeWorkspaceAlignment !=
          0u ||
      workspace.workspace_size_bytes < plan.workspace_size_bytes() ||
      workspace.wrapped_positions !=
          offset_pointer<double>(workspace.workspace_base, data.wrapped_positions_offset) ||
      workspace.atom_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.atom_scratch_offset) ||
      workspace.secondary_atom_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.secondary_atom_scratch_offset) ||
      workspace.gradient_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.gradient_scratch_offset) ||
      workspace.strain_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.strain_scratch_offset) ||
      workspace.batch_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.batch_scratch_offset) ||
      workspace.wrapped_position_elements != plan.total_atoms() * 3 ||
      workspace.atom_elements != plan.total_atoms() ||
      workspace.gradient_elements != plan.total_atoms() * 3 ||
      workspace.strain_elements != plan.batch_size() * 9 ||
      workspace.batch_elements != plan.batch_size() ||
      plan.overlaps_storage(workspace.workspace_base, plan.workspace_size_bytes())) {
    error = "periodic short-range workspace is incomplete or belongs to another plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

PeriodicShortRangePlan::PeriodicShortRangePlan(
    std::shared_ptr<const PeriodicShortRangePlanData> data) noexcept
    : data_(std::move(data)) {}

bool PeriodicShortRangePlan::sealed() const noexcept { return static_cast<bool>(data_); }
std::int64_t PeriodicShortRangePlan::batch_size() const noexcept {
  return data_ ? data_->batch_size : 0;
}
std::int64_t PeriodicShortRangePlan::total_atoms() const noexcept {
  return data_ ? data_->total_atoms : 0;
}
const std::vector<std::int64_t>& PeriodicShortRangePlan::atom_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ ? data_->atom_offsets : empty;
}
const Lattice3D& PeriodicShortRangePlan::lattice(std::int64_t system) const noexcept {
  static const Lattice3D empty;
  if (!data_ || system < 0 || system >= data_->batch_size) return empty;
  return data_->lattices[static_cast<std::size_t>(system)];
}
LatticeTranslationView PeriodicShortRangePlan::translations(
    std::int64_t system, PeriodicTranslationCutoff cutoff) const noexcept {
  const std::size_t kind = cutoff_index(cutoff);
  if (!data_ || kind >= data_->translations.size() || system < 0 || system >= data_->batch_size) {
    return {};
  }
  const auto& offsets = data_->translation_offsets[kind];
  const std::int64_t begin = offsets[static_cast<std::size_t>(system)];
  const std::int64_t end = offsets[static_cast<std::size_t>(system + 1)];
  const auto& values = data_->translations[kind];
  return {values.data() + static_cast<std::size_t>(begin), end - begin};
}
std::size_t PeriodicShortRangePlan::workspace_size_bytes() const noexcept {
  return data_ ? data_->workspace_size_bytes : 0u;
}

bool PeriodicShortRangePlan::overlaps_storage(const void* data, std::size_t size_bytes) const noexcept {
  if (size_bytes == 0u) return false;
  AddressRange active;
  AddressRange descriptor;
  AddressRange plan_data;
  if (!data_ || !make_range(data, size_bytes, active) ||
      !make_range(this, sizeof(*this), descriptor) ||
      !make_range(data_.get(), sizeof(*data_), plan_data)) {
    return true;
  }
  if (ranges_overlap(active, descriptor) || ranges_overlap(active, plan_data) ||
      overlaps_vector(active, data_->atom_offsets) || overlaps_vector(active, data_->lattices)) {
    return true;
  }
  for (const auto& offsets : data_->translation_offsets) {
    if (overlaps_vector(active, offsets)) return true;
  }
  for (const auto& translations : data_->translations) {
    if (overlaps_vector(active, translations)) return true;
  }
  return false;
}

const PeriodicShortRangePlanData* PeriodicShortRangePlan::identity() const noexcept {
  return data_.get();
}

xtbloom_status_t make_periodic_short_range_plan(
    std::int64_t batch_size, std::int64_t total_atoms, const std::int64_t* atom_offsets,
    const double* cell_matrices, PeriodicShortRangePlan& plan, std::string& error) {
  if (batch_size <= 0 || total_atoms <= 0 || !valid_count(batch_size) ||
      !valid_count(total_atoms) || atom_offsets == nullptr || cell_matrices == nullptr ||
      static_cast<std::uint64_t>(batch_size) >=
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      static_cast<std::uint64_t>(batch_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / 9u) ||
      atom_offsets[0] != 0 || atom_offsets[batch_size] != total_atoms) {
    error = "periodic short-range plan requires a valid positive ragged batch and cells";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    auto created = std::make_shared<PeriodicShortRangePlanData>();
    created->batch_size = batch_size;
    created->total_atoms = total_atoms;
    created->atom_offsets.assign(atom_offsets, atom_offsets + batch_size + 1);
    created->lattices.resize(static_cast<std::size_t>(batch_size));
    for (auto& offsets : created->translation_offsets) {
      offsets.resize(static_cast<std::size_t>(batch_size + 1), 0);
    }

    for (std::int64_t system = 0; system < batch_size; ++system) {
      const std::int64_t begin = atom_offsets[system];
      const std::int64_t end = atom_offsets[system + 1];
      if (begin < 0 || begin > end || end > total_atoms) {
        error = "periodic short-range atom offsets are not a valid ragged partition";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      std::string local_error;
      xtbloom_status_t status = make_lattice_3d(
          cell_matrices + static_cast<std::size_t>(system) * 9u,
          created->lattices[static_cast<std::size_t>(system)], local_error);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        error = "periodic short-range cell " + std::to_string(system) + ": " + local_error;
        return status;
      }
      for (std::size_t kind = 0; kind < kCutoffs.size(); ++kind) {
        std::vector<LatticeTranslation> local;
        status = make_lattice_translations(created->lattices[static_cast<std::size_t>(system)],
                                           kCutoffs[kind], LatticeOriginPolicy::kInclude, local,
                                           local_error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          error = "periodic short-range translation set " + std::to_string(kind) +
                  " for system " + std::to_string(system) + ": " + local_error;
          return status;
        }
        auto& values = created->translations[kind];
        if (local.size() > values.max_size() - values.size() ||
            local.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) -
                               values.size()) {
          error = "periodic short-range translation count overflows";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        values.insert(values.end(), local.begin(), local.end());
        created->translation_offsets[kind][static_cast<std::size_t>(system + 1)] =
            static_cast<std::int64_t>(values.size());
      }
    }

    const std::size_t atom_count = static_cast<std::size_t>(total_atoms);
    const std::size_t batch_count = static_cast<std::size_t>(batch_size);
    std::size_t wrapped_count = 0u;
    std::size_t gradient_count = 0u;
    std::size_t strain_count = 0u;
    if (!checked_multiply(atom_count, 3u, wrapped_count) ||
        !checked_multiply(atom_count, 3u, gradient_count) ||
        !checked_multiply(batch_count, 9u, strain_count)) {
      error = "periodic short-range workspace element count overflows";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    std::size_t cursor = 0u;
    if (!append_doubles(wrapped_count, cursor, created->wrapped_positions_offset) ||
        !append_doubles(atom_count, cursor, created->atom_scratch_offset) ||
        !append_doubles(atom_count, cursor, created->secondary_atom_scratch_offset) ||
        !append_doubles(gradient_count, cursor, created->gradient_scratch_offset) ||
        !append_doubles(strain_count, cursor, created->strain_scratch_offset) ||
        !append_doubles(batch_count, cursor, created->batch_scratch_offset) ||
        !align_up(cursor, kPeriodicShortRangeWorkspaceAlignment, created->workspace_size_bytes)) {
      error = "periodic short-range workspace byte count overflows";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    plan = PeriodicShortRangePlan(std::move(created));
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate periodic short-range plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t bind_periodic_short_range_workspace(const PeriodicShortRangePlan& plan,
                                                     void* workspace,
                                                     std::size_t workspace_size,
                                                     PeriodicShortRangeWorkspace& view,
                                                     std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  AddressRange workspace_range;
  AddressRange view_range;
  AddressRange error_range;
  if (workspace == nullptr ||
      reinterpret_cast<std::uintptr_t>(workspace) % kPeriodicShortRangeWorkspaceAlignment != 0u ||
      workspace_size < plan.workspace_size_bytes() ||
      !make_range(workspace, plan.workspace_size_bytes(), workspace_range) ||
      !make_range(&view, sizeof(view), view_range) || !make_range(&error, sizeof(error), error_range) ||
      ranges_overlap(workspace_range, view_range) || ranges_overlap(workspace_range, error_range) ||
      ranges_overlap(view_range, error_range) ||
      plan.overlaps_storage(workspace, plan.workspace_size_bytes())) {
    error =
        "periodic short-range workspace must be aligned, sufficiently large, and disjoint from "
        "plan and descriptor storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const PeriodicShortRangePlanData& data = *plan.identity();
  PeriodicShortRangeWorkspace bound;
  bound.workspace_base = workspace;
  bound.workspace_size_bytes = workspace_size;
  bound.wrapped_positions = offset_pointer<double>(workspace, data.wrapped_positions_offset);
  bound.wrapped_position_elements = plan.total_atoms() * 3;
  bound.atom_scratch = offset_pointer<double>(workspace, data.atom_scratch_offset);
  bound.secondary_atom_scratch =
      offset_pointer<double>(workspace, data.secondary_atom_scratch_offset);
  bound.atom_elements = plan.total_atoms();
  bound.gradient_scratch = offset_pointer<double>(workspace, data.gradient_scratch_offset);
  bound.gradient_elements = plan.total_atoms() * 3;
  bound.strain_scratch = offset_pointer<double>(workspace, data.strain_scratch_offset);
  bound.strain_elements = plan.batch_size() * 9;
  bound.batch_scratch = offset_pointer<double>(workspace, data.batch_scratch_offset);
  bound.batch_elements = plan.batch_size();
  bound.plan_identity = plan.identity();
  view = bound;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_periodic_short_range_workspace(
    const PeriodicShortRangePlan& plan, const PeriodicShortRangeWorkspace& workspace,
    std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  return validate_workspace_layout(plan, workspace, error);
}

xtbloom_status_t update_periodic_short_range_geometry_cpu(
    const PeriodicShortRangePlan& plan, const double* positions,
    std::uint64_t geometry_generation, const PeriodicShortRangeWorkspace& workspace,
    PeriodicShortRangeGeometry& geometry, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = validate_workspace_layout(plan, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (positions == nullptr || geometry_generation == 0u) {
    error = "periodic short-range geometry requires positions and a nonzero generation";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  /* First pass proves that publication to the workspace cannot fail halfway. */
  std::array<double, 3> wrapped{};
  for (std::int64_t system = 0; system < plan.batch_size(); ++system) {
    const std::int64_t begin = plan.atom_offsets()[static_cast<std::size_t>(system)];
    const std::int64_t end = plan.atom_offsets()[static_cast<std::size_t>(system + 1)];
    for (std::int64_t atom = begin; atom < end; ++atom) {
      std::string local_error;
      status = wrap_cartesian(plan.lattice(system), positions + static_cast<std::size_t>(atom) * 3u,
                              wrapped.data(), local_error);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        error = "periodic short-range position " + std::to_string(atom) + ": " + local_error;
        return status;
      }
    }
  }
  for (std::int64_t system = 0; system < plan.batch_size(); ++system) {
    const std::int64_t begin = plan.atom_offsets()[static_cast<std::size_t>(system)];
    const std::int64_t end = plan.atom_offsets()[static_cast<std::size_t>(system + 1)];
    for (std::int64_t atom = begin; atom < end; ++atom) {
      std::string local_error;
      status = wrap_cartesian(
          plan.lattice(system), positions + static_cast<std::size_t>(atom) * 3u,
          workspace.wrapped_positions + static_cast<std::size_t>(atom) * 3u, local_error);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        error = "periodic short-range wrapping changed after validation: " + local_error;
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
  }

  PeriodicShortRangeGeometry prepared;
  prepared.wrapped_positions = workspace.wrapped_positions;
  prepared.wrapped_position_elements = plan.total_atoms() * 3;
  prepared.geometry_generation = geometry_generation;
  prepared.plan_identity = plan.identity();
  geometry = prepared;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn2
