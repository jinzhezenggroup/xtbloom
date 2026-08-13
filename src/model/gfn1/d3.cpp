// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/d3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "data/parameters/gfn1.hpp"
#include "data/parameters/gfn1_d3.hpp"
#include "model/gfn1/coordination.hpp"

namespace xtbloom::detail::gfn1 {

struct D3PlanData {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_pairs = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> pair_offsets;
  std::vector<std::uint8_t> atomic_numbers;
  std::vector<double> pair_rrij;
  std::vector<double> pair_damping_radii;
  CoordinationPlan coordination_plan;
  std::size_t workspace_size_bytes = 0u;
  std::size_t weight_offset = 0u;
  std::size_t weight_cn_offset = 0u;
  std::size_t coordination_adjoint_offset = 0u;
  std::size_t batch_scratch_offset = 0u;
  std::size_t gradient_scratch_offset = 0u;
};

namespace {

constexpr double kReferenceWeightFactor = 4.0;
constexpr double kTwoBodyCutoff = 50.0;
constexpr double kTwoBodySwitchWidth = 0.05;
constexpr double kMinimumDistanceSquared = std::numeric_limits<double>::epsilon();

static_assert(parameters::gfn1::kGlobal.dispersion_model == 1u,
              "the generated GFN1 parameters must select D3");
static_assert(parameters::gfn1::kGlobal.dispersion_s9 == 0.0,
              "GFN1 D3 must not allocate or evaluate an ATM term");
static_assert(parameters::gfn1_d3::kElementCount == parameters::gfn1::kElementCount,
              "GFN1 and its D3 tables must cover the same element domain");
static_assert(kD3MaximumReferences == 7u,
              "simple-dftd3 v1.4.0 uses at most seven reference states");

constexpr bool d3_reference_layout_fits_workspace() {
  for (std::size_t element_index = 0u; element_index < parameters::gfn1_d3::kElements.size();
       ++element_index) {
    const auto& element = parameters::gfn1_d3::kElements[element_index];
    if (element.reference_count == 0u || element.reference_count > kD3MaximumReferences) {
      return false;
    }
    const auto atomic_number = static_cast<std::uint32_t>(element_index + 1u);
    for (std::uint32_t first = 0u; first < element.reference_count; ++first) {
      for (std::uint32_t second = first + 1u; second < element.reference_count; ++second) {
        if (parameters::gfn1_d3::reference_cn(atomic_number, first) ==
            parameters::gfn1_d3::reference_cn(atomic_number, second)) {
          return false;
        }
      }
    }
  }
  return true;
}

static_assert(d3_reference_layout_fits_workspace(),
              "GFN1 D3 reference slices must fit the workspace stride and have unique CNs");

bool checked_add_size(std::size_t first, std::size_t second, std::size_t& result) {
  if (first > std::numeric_limits<std::size_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

bool checked_multiply_size(std::size_t first, std::size_t second, std::size_t& result) {
  if (first != 0u && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) {
  if (bytes != 0u && pointer == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.begin < second.end && second.begin < first.end;
}

bool aligned(const void* pointer, std::size_t alignment) {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

template <typename T>
T* offset_pointer(void* base, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
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

template <typename T>
bool overlaps_vector(const AddressRange& active, const std::vector<T>& values) {
  std::size_t bytes = 0u;
  AddressRange storage;
  return checked_multiply_size(values.capacity(), sizeof(T), bytes) &&
         make_range(values.data(), bytes, storage) && ranges_overlap(active, storage);
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t& result) {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return false;
  }
  const std::size_t mask = alignment - 1u;
  if (value > std::numeric_limits<std::size_t>::max() - mask) {
    return false;
  }
  result = (value + mask) & ~mask;
  return true;
}

bool append_segment(std::size_t bytes, std::size_t& cursor, std::size_t& offset) {
  return align_up(cursor, alignof(double), offset) && checked_add_size(offset, bytes, cursor);
}

bool valid_count(std::int64_t value) {
  return value >= 0 && static_cast<std::uint64_t>(value) <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

bool finite_values(const double* values, std::size_t count) {
  if (values == nullptr) {
    return false;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

xtbloom_status_t validate_plan(const D3Plan& plan, std::string& error) {
  if (!plan.sealed() || plan.batch_size() <= 0 || plan.total_atoms() <= 0 ||
      plan.total_pairs() < 0) {
    error = "GFN1 D3 plan is not sealed or has invalid extents";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_workspace(const D3Plan& plan, const D3Workspace& workspace,
                                    std::string& error) {
  const D3PlanData& data = *plan.identity();
  if (workspace.plan_identity != plan.identity() ||
      !aligned(workspace.workspace_base, kD3WorkspaceAlignment) ||
      workspace.workspace_size_bytes < plan.workspace_size_bytes() ||
      workspace.weights != offset_pointer<double>(workspace.workspace_base, data.weight_offset) ||
      workspace.weight_cn_derivatives !=
          offset_pointer<double>(workspace.workspace_base, data.weight_cn_offset) ||
      workspace.coordination_adjoints !=
          offset_pointer<double>(workspace.workspace_base, data.coordination_adjoint_offset) ||
      workspace.batch_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.batch_scratch_offset) ||
      workspace.gradient_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.gradient_scratch_offset) ||
      workspace.weight_elements !=
          plan.total_atoms() * static_cast<std::int64_t>(kD3MaximumReferences) ||
      workspace.atom_elements != plan.total_atoms() ||
      workspace.batch_elements != plan.batch_size() ||
      workspace.gradient_elements != plan.total_atoms() * 3) {
    error = "GFN1 D3 workspace is incomplete or belongs to another plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

template <std::size_t N, std::size_t M>
bool valid_call_storage(const D3Plan& plan, const D3Workspace& workspace,
                        const std::array<AddressRange, N>& numerical,
                        const std::array<AddressRange, M>& controls) {
  AddressRange workspace_range;
  if (!make_range(workspace.workspace_base, plan.workspace_size_bytes(), workspace_range) ||
      plan.overlaps_storage(workspace.workspace_base, plan.workspace_size_bytes()) ||
      !pairwise_disjoint(numerical) || !pairwise_disjoint(controls)) {
    return false;
  }
  for (const AddressRange& range : numerical) {
    const std::size_t bytes = static_cast<std::size_t>(range.end - range.begin);
    if (ranges_overlap(range, workspace_range) ||
        plan.overlaps_storage(reinterpret_cast<const void*>(range.begin), bytes)) {
      return false;
    }
    for (const AddressRange& control : controls) {
      if (ranges_overlap(range, control)) {
        return false;
      }
    }
  }
  for (const AddressRange& control : controls) {
    if (ranges_overlap(workspace_range, control)) {
      return false;
    }
  }
  return true;
}

std::size_t pair_index(const D3PlanData& data, std::int64_t batch, std::int64_t first,
                       std::int64_t second) {
  const std::int64_t begin = data.atom_offsets[static_cast<std::size_t>(batch)];
  const std::int64_t local_first = first - begin;
  const std::int64_t local_second = second - begin;
  return static_cast<std::size_t>(data.pair_offsets[static_cast<std::size_t>(batch)] +
                                  local_second * (local_second - 1) / 2 + local_first);
}

void prepare_weight_slice(const D3PlanData& data, const double* coordination,
                          const D3Workspace& workspace) {
  std::fill_n(workspace.weights, static_cast<std::size_t>(workspace.weight_elements), 0.0);
  std::fill_n(workspace.weight_cn_derivatives, static_cast<std::size_t>(workspace.weight_elements),
              0.0);
  for (std::int64_t atom = 0; atom < data.total_atoms; ++atom) {
    const std::size_t atom_index = static_cast<std::size_t>(atom);
    const std::uint32_t atomic_number = data.atomic_numbers[atom_index];
    const auto& element = parameters::gfn1_d3::kElements[atomic_number - 1u];
    const std::size_t weight_begin = atom_index * kD3MaximumReferences;
    double norm = 0.0;
    double derivative_norm = 0.0;
    for (std::size_t reference = 0u; reference < element.reference_count; ++reference) {
      const double reference_cn =
          parameters::gfn1_d3::reference_cn(atomic_number, static_cast<std::uint32_t>(reference));
      const double delta = reference_cn - coordination[atom_index];
      const double unnormalized = std::exp(-kReferenceWeightFactor * delta * delta);
      workspace.weights[weight_begin + reference] = unnormalized;
      norm += unnormalized;
      derivative_norm += 2.0 * kReferenceWeightFactor * delta * unnormalized;
    }
    const double inverse_norm = 1.0 / norm;
    double maximum_cn = -std::numeric_limits<double>::infinity();
    for (std::size_t reference = 0u; reference < element.reference_count; ++reference) {
      maximum_cn = std::max(maximum_cn, parameters::gfn1_d3::reference_cn(
                                            atomic_number, static_cast<std::uint32_t>(reference)));
    }
    for (std::size_t reference = 0u; reference < element.reference_count; ++reference) {
      const double reference_cn =
          parameters::gfn1_d3::reference_cn(atomic_number, static_cast<std::uint32_t>(reference));
      const double delta = reference_cn - coordination[atom_index];
      const double unnormalized = workspace.weights[weight_begin + reference];
      double weight = unnormalized * inverse_norm;
      if (!std::isfinite(weight)) {
        /* Preserve simple-dftd3 v1.4.0's maxval-equality fallback exactly. */
        weight = reference_cn == maximum_cn ? 1.0 : 0.0;
      }
      double derivative = 2.0 * kReferenceWeightFactor * delta * unnormalized * inverse_norm -
                          unnormalized * derivative_norm * inverse_norm * inverse_norm;
      if (!std::isfinite(derivative)) {
        derivative = 0.0;
      }
      workspace.weights[weight_begin + reference] = weight;
      workspace.weight_cn_derivatives[weight_begin + reference] = derivative;
    }
  }
}

struct PairCoefficient {
  double c6 = 0.0;
  double first_cn = 0.0;
  double second_cn = 0.0;
};

PairCoefficient pair_coefficient(const D3PlanData& data, std::int64_t first, std::int64_t second,
                                 const D3Workspace& workspace) {
  PairCoefficient result;
  const std::size_t first_index = static_cast<std::size_t>(first);
  const std::size_t second_index = static_cast<std::size_t>(second);
  const std::uint32_t first_atomic_number = data.atomic_numbers[first_index];
  const std::uint32_t second_atomic_number = data.atomic_numbers[second_index];
  const auto& first_element = parameters::gfn1_d3::kElements[first_atomic_number - 1u];
  const auto& second_element = parameters::gfn1_d3::kElements[second_atomic_number - 1u];
  const std::size_t first_weight = first_index * kD3MaximumReferences;
  const std::size_t second_weight = second_index * kD3MaximumReferences;
  for (std::size_t first_reference = 0u; first_reference < first_element.reference_count;
       ++first_reference) {
    const double first_value = workspace.weights[first_weight + first_reference];
    const double first_derivative = workspace.weight_cn_derivatives[first_weight + first_reference];
    for (std::size_t second_reference = 0u; second_reference < second_element.reference_count;
         ++second_reference) {
      const double second_value = workspace.weights[second_weight + second_reference];
      const double second_derivative =
          workspace.weight_cn_derivatives[second_weight + second_reference];
      const double reference_c6 = parameters::gfn1_d3::reference_c6(
          first_atomic_number, static_cast<std::uint32_t>(first_reference), second_atomic_number,
          static_cast<std::uint32_t>(second_reference));
      result.c6 += first_value * second_value * reference_c6;
      result.first_cn += first_derivative * second_value * reference_c6;
      result.second_cn += first_value * second_derivative * reference_c6;
    }
  }
  return result;
}

struct SmoothCutoff {
  double value = 0.0;
  double derivative = 0.0;
};

SmoothCutoff smooth_cutoff(double distance) {
  const double inner = kTwoBodyCutoff - kTwoBodySwitchWidth;
  if (distance <= inner) {
    return {1.0, 0.0};
  }
  if (distance >= kTwoBodyCutoff) {
    return {0.0, 0.0};
  }
  const double x = (kTwoBodyCutoff - distance) / kTwoBodySwitchWidth;
  return {x * x * x * (10.0 + x * (-15.0 + 6.0 * x)),
          -30.0 * x * x * (1.0 - x) * (1.0 - x) / kTwoBodySwitchWidth};
}

struct PairContribution {
  double energy = 0.0;
  double first_cn = 0.0;
  double second_cn = 0.0;
  double gradient_scale = 0.0;
};

xtbloom_status_t evaluate_pair(const D3PlanData& data, std::size_t packed_pair,
                               double distance_squared, const PairCoefficient& coefficient,
                               PairContribution& contribution, std::string& error) {
  const double distance = std::sqrt(distance_squared);
  const double rrij = data.pair_rrij[packed_pair];
  const double damping_radius = data.pair_damping_radii[packed_pair];
  const double damping_radius2 = damping_radius * damping_radius;
  const double damping_radius4 = damping_radius2 * damping_radius2;
  const double damping_radius6 = damping_radius4 * damping_radius2;
  const double damping_radius8 = damping_radius4 * damping_radius4;
  const double r4 = distance_squared * distance_squared;
  const double t6 = 1.0 / (r4 * distance_squared + damping_radius6);
  const double t8 = 1.0 / (r4 * r4 + damping_radius8);
  const double phi = parameters::gfn1::kGlobal.dispersion_s6 * t6 +
                     parameters::gfn1::kGlobal.dispersion_s8 * rrij * t8;
  const double derivative_over_distance =
      parameters::gfn1::kGlobal.dispersion_s6 * (-6.0 * r4 * t6 * t6) +
      parameters::gfn1::kGlobal.dispersion_s8 * rrij * (-8.0 * r4 * distance_squared * t8 * t8);
  const SmoothCutoff cutoff = smooth_cutoff(distance);
  const double damping = cutoff.value * phi;
  contribution.energy = -coefficient.c6 * damping;
  contribution.first_cn = -coefficient.first_cn * damping;
  contribution.second_cn = -coefficient.second_cn * damping;
  contribution.gradient_scale = -coefficient.c6 * (cutoff.value * derivative_over_distance +
                                                   cutoff.derivative * phi / distance);
  if (!std::isfinite(coefficient.c6) || !std::isfinite(coefficient.first_cn) ||
      !std::isfinite(coefficient.second_cn) || !std::isfinite(contribution.energy) ||
      !std::isfinite(contribution.first_cn) || !std::isfinite(contribution.second_cn) ||
      !std::isfinite(contribution.gradient_scale)) {
    error = "GFN1 D3 pair arithmetic exceeded floating-point range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

D3Plan::D3Plan(std::shared_ptr<const D3PlanData> data) noexcept : data_(std::move(data)) {}

bool D3Plan::sealed() const noexcept { return static_cast<bool>(data_); }

std::int64_t D3Plan::batch_size() const noexcept { return data_ ? data_->batch_size : 0; }

std::int64_t D3Plan::total_atoms() const noexcept { return data_ ? data_->total_atoms : 0; }

std::int64_t D3Plan::total_pairs() const noexcept { return data_ ? data_->total_pairs : 0; }

const std::vector<std::int64_t>& D3Plan::atom_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ ? data_->atom_offsets : empty;
}

const std::vector<std::int64_t>& D3Plan::pair_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ ? data_->pair_offsets : empty;
}

bool D3Plan::matches_atomic_numbers(const std::int32_t* atomic_numbers) const noexcept {
  if (!data_ || atomic_numbers == nullptr) {
    return false;
  }
  for (std::int64_t atom = 0; atom < data_->total_atoms; ++atom) {
    if (atomic_numbers[atom] != data_->atomic_numbers[static_cast<std::size_t>(atom)]) {
      return false;
    }
  }
  return true;
}

bool D3Plan::overlaps_storage(const void* data, std::size_t size_bytes) const noexcept {
  if (size_bytes == 0u) {
    return false;
  }
  AddressRange active;
  AddressRange descriptor;
  AddressRange plan_data;
  if (data_ == nullptr || !make_range(data, size_bytes, active) ||
      !make_range(this, sizeof(*this), descriptor) ||
      !make_range(data_.get(), sizeof(*data_), plan_data)) {
    return true;
  }
  if (ranges_overlap(active, descriptor) || ranges_overlap(active, plan_data)) {
    return true;
  }
  return overlaps_vector(active, data_->atom_offsets) ||
         overlaps_vector(active, data_->pair_offsets) ||
         overlaps_vector(active, data_->atomic_numbers) ||
         overlaps_vector(active, data_->pair_rrij) ||
         overlaps_vector(active, data_->pair_damping_radii) ||
         overlaps_vector(active, data_->coordination_plan.atom_offsets) ||
         overlaps_vector(active, data_->coordination_plan.covalent_radius);
}

std::size_t D3Plan::workspace_size_bytes() const noexcept {
  return data_ ? data_->workspace_size_bytes : 0u;
}

std::size_t D3Plan::resident_bytes() const noexcept {
  if (!data_) {
    return 0u;
  }
  return sizeof(*data_) + data_->atom_offsets.capacity() * sizeof(std::int64_t) +
         data_->pair_offsets.capacity() * sizeof(std::int64_t) +
         data_->atomic_numbers.capacity() * sizeof(std::uint8_t) +
         data_->pair_rrij.capacity() * sizeof(double) +
         data_->pair_damping_radii.capacity() * sizeof(double) +
         data_->coordination_plan.atom_offsets.capacity() * sizeof(std::int64_t) +
         data_->coordination_plan.covalent_radius.capacity() * sizeof(double);
}

const CoordinationPlan& D3Plan::coordination_plan() const noexcept {
  static const CoordinationPlan empty;
  return data_ ? data_->coordination_plan : empty;
}

const D3PlanData* D3Plan::identity() const noexcept { return data_.get(); }

xtbloom_status_t make_d3_plan(std::int64_t batch_size, std::int64_t total_atoms,
                              const std::int64_t* atom_offsets, const std::int32_t* atomic_numbers,
                              D3Plan& plan, std::string& error) {
  if (batch_size <= 0 || total_atoms <= 0 || !valid_count(batch_size) ||
      !valid_count(total_atoms) || atom_offsets == nullptr || atomic_numbers == nullptr ||
      atom_offsets[0] != 0 || atom_offsets[batch_size] != total_atoms) {
    error = "GFN1 D3 plan requires a valid positive ragged batch";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto created = std::make_shared<D3PlanData>();
    created->batch_size = batch_size;
    created->total_atoms = total_atoms;
    created->atom_offsets.assign(atom_offsets, atom_offsets + batch_size + 1);
    created->pair_offsets.resize(static_cast<std::size_t>(batch_size + 1), 0);
    created->atomic_numbers.resize(static_cast<std::size_t>(total_atoms));
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      const std::int64_t begin = atom_offsets[batch];
      const std::int64_t end = atom_offsets[batch + 1];
      if (begin < 0 || begin > end || end > total_atoms) {
        error = "GFN1 D3 atom offsets are not a valid ragged partition";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const std::uint64_t count = static_cast<std::uint64_t>(end - begin);
      if (count > 0u &&
          count - 1u >
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / count) {
        error = "GFN1 D3 pair count overflows";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const std::uint64_t pairs = count * (count - 1u) / 2u;
      const std::int64_t previous = created->pair_offsets[static_cast<std::size_t>(batch)];
      if (pairs > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() - previous)) {
        error = "GFN1 D3 total pair count overflows";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      created->pair_offsets[static_cast<std::size_t>(batch + 1)] =
          previous + static_cast<std::int64_t>(pairs);
    }
    created->total_pairs = created->pair_offsets.back();
    for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
      const std::int32_t atomic_number = atomic_numbers[atom];
      if (atomic_number <= 0 ||
          atomic_number > static_cast<std::int32_t>(parameters::gfn1_d3::kElementCount)) {
        error = "GFN1 D3 plan contains an unsupported atomic number";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      created->atomic_numbers[static_cast<std::size_t>(atom)] =
          static_cast<std::uint8_t>(atomic_number);
    }
    xtbloom_status_t status = make_coordination_plan(
        batch_size, total_atoms, atom_offsets, atomic_numbers, created->coordination_plan, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }

    const std::size_t pair_count = static_cast<std::size_t>(created->total_pairs);
    created->pair_rrij.resize(pair_count);
    created->pair_damping_radii.resize(pair_count);
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      const std::int64_t begin = atom_offsets[batch];
      const std::int64_t end = atom_offsets[batch + 1];
      for (std::int64_t second = begin + 1; second < end; ++second) {
        for (std::int64_t first = begin; first < second; ++first) {
          const std::size_t packed_pair = pair_index(*created, batch, first, second);
          const std::uint32_t first_atomic_number =
              created->atomic_numbers[static_cast<std::size_t>(first)];
          const std::uint32_t second_atomic_number =
              created->atomic_numbers[static_cast<std::size_t>(second)];
          const double rrij = 3.0 * parameters::gfn1_d3::kR4R2[first_atomic_number - 1u] *
                              parameters::gfn1_d3::kR4R2[second_atomic_number - 1u];
          created->pair_rrij[packed_pair] = rrij;
          created->pair_damping_radii[packed_pair] =
              parameters::gfn1::kGlobal.dispersion_a1 * std::sqrt(rrij) +
              parameters::gfn1::kGlobal.dispersion_a2;
        }
      }
    }

    const std::size_t atom_count = static_cast<std::size_t>(total_atoms);
    const std::size_t batch_count = static_cast<std::size_t>(batch_size);
    std::size_t weight_elements = 0u;
    std::size_t gradient_elements = 0u;
    if (!checked_multiply_size(atom_count, kD3MaximumReferences, weight_elements) ||
        !checked_multiply_size(atom_count, 3u, gradient_elements)) {
      error = "GFN1 D3 workspace element count overflows";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    std::size_t cursor = 0u;
    auto append_doubles = [&](std::size_t elements, std::size_t& offset) {
      std::size_t bytes = 0u;
      return checked_multiply_size(elements, sizeof(double), bytes) &&
             append_segment(bytes, cursor, offset);
    };
    if (!append_doubles(weight_elements, created->weight_offset) ||
        !append_doubles(weight_elements, created->weight_cn_offset) ||
        !append_doubles(atom_count, created->coordination_adjoint_offset) ||
        !append_doubles(batch_count, created->batch_scratch_offset) ||
        !append_doubles(gradient_elements, created->gradient_scratch_offset) ||
        !align_up(cursor, kD3WorkspaceAlignment, created->workspace_size_bytes)) {
      error = "GFN1 D3 workspace byte count overflows";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    plan = D3Plan(std::move(created));
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 D3 plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN1 D3 plan dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t bind_d3_workspace(const D3Plan& plan, void* workspace, std::size_t workspace_size,
                                   D3Workspace& view, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  AddressRange workspace_range;
  AddressRange view_range;
  AddressRange error_range;
  if (!aligned(workspace, kD3WorkspaceAlignment) || workspace_size < plan.workspace_size_bytes() ||
      !make_range(workspace, plan.workspace_size_bytes(), workspace_range) ||
      !make_range(&view, sizeof(view), view_range) ||
      !make_range(&error, sizeof(error), error_range) ||
      ranges_overlap(workspace_range, view_range) || ranges_overlap(workspace_range, error_range) ||
      ranges_overlap(view_range, error_range) ||
      plan.overlaps_storage(workspace, plan.workspace_size_bytes())) {
    error = "GFN1 D3 workspace must be sufficiently large and 64-byte aligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const D3PlanData& data = *plan.identity();
  D3Workspace bound;
  bound.workspace_base = workspace;
  bound.workspace_size_bytes = workspace_size;
  bound.weights = offset_pointer<double>(workspace, data.weight_offset);
  bound.weight_cn_derivatives = offset_pointer<double>(workspace, data.weight_cn_offset);
  bound.weight_elements = data.total_atoms * static_cast<std::int64_t>(kD3MaximumReferences);
  bound.coordination_adjoints = offset_pointer<double>(workspace, data.coordination_adjoint_offset);
  bound.atom_elements = data.total_atoms;
  bound.batch_scratch = offset_pointer<double>(workspace, data.batch_scratch_offset);
  bound.batch_elements = data.batch_size;
  bound.gradient_scratch = offset_pointer<double>(workspace, data.gradient_scratch_offset);
  bound.gradient_elements = data.total_atoms * 3;
  bound.plan_identity = plan.identity();
  view = bound;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_d3_cpu(const D3Plan& plan, const double* positions,
                            const double* coordination_numbers, double* energies, double* gradients,
                            const D3Workspace& workspace, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = validate_workspace(plan, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  const D3PlanData& data = *plan.identity();
  const std::size_t atom_count = static_cast<std::size_t>(data.total_atoms);
  const std::size_t batch_count = static_cast<std::size_t>(data.batch_size);
  const std::size_t coordinate_count = atom_count * 3u;
  const bool derivatives = gradients != nullptr;
  if (!aligned(positions, alignof(double)) || !aligned(coordination_numbers, alignof(double)) ||
      !aligned(energies, alignof(double)) ||
      (derivatives && !aligned(gradients, alignof(double))) ||
      !finite_values(positions, coordinate_count) ||
      !finite_values(coordination_numbers, atom_count) || !finite_values(energies, batch_count) ||
      (derivatives && !finite_values(gradients, coordinate_count))) {
    error = "GFN1 D3 evaluation requires aligned, finite inputs and accumulators";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::size_t position_bytes = 0u;
  std::size_t atom_bytes = 0u;
  std::size_t batch_bytes = 0u;
  std::size_t gradient_bytes = 0u;
  std::array<AddressRange, 4> numerical{};
  std::array<AddressRange, 4> controls{};
  if (!checked_multiply_size(coordinate_count, sizeof(double), position_bytes) ||
      !checked_multiply_size(atom_count, sizeof(double), atom_bytes) ||
      !checked_multiply_size(batch_count, sizeof(double), batch_bytes) ||
      !checked_multiply_size(derivatives ? coordinate_count : 0u, sizeof(double), gradient_bytes) ||
      !make_range(positions, position_bytes, numerical[0]) ||
      !make_range(coordination_numbers, atom_bytes, numerical[1]) ||
      !make_range(energies, batch_bytes, numerical[2]) ||
      !make_range(gradients, gradient_bytes, numerical[3]) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&workspace, sizeof(workspace), controls[1]) ||
      !make_range(&error, sizeof(error), controls[2]) ||
      !make_range(plan.identity(), sizeof(D3PlanData), controls[3]) ||
      !valid_call_storage(plan, workspace, numerical, controls)) {
    error = "GFN1 D3 buffers overlap numerical, plan, workspace, or descriptor storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  prepare_weight_slice(data, coordination_numbers, workspace);
  if (!finite_values(workspace.weights, static_cast<std::size_t>(workspace.weight_elements)) ||
      !finite_values(workspace.weight_cn_derivatives,
                     static_cast<std::size_t>(workspace.weight_elements))) {
    error = "GFN1 D3 reference interpolation exceeded floating-point range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::fill_n(workspace.batch_scratch, batch_count, 0.0);
  if (derivatives) {
    std::fill_n(workspace.coordination_adjoints, atom_count, 0.0);
    std::fill_n(workspace.gradient_scratch, coordinate_count, 0.0);
  }

  const double cutoff_squared = kTwoBodyCutoff * kTwoBodyCutoff;
  for (std::int64_t batch = 0; batch < data.batch_size; ++batch) {
    const std::int64_t begin = data.atom_offsets[static_cast<std::size_t>(batch)];
    const std::int64_t end = data.atom_offsets[static_cast<std::size_t>(batch + 1)];
    for (std::int64_t second = begin + 1; second < end; ++second) {
      const std::size_t second_index = static_cast<std::size_t>(second);
      for (std::int64_t first = begin; first < second; ++first) {
        const std::size_t first_index = static_cast<std::size_t>(first);
        const double dx = positions[first_index * 3u] - positions[second_index * 3u];
        const double dy = positions[first_index * 3u + 1u] - positions[second_index * 3u + 1u];
        const double dz = positions[first_index * 3u + 2u] - positions[second_index * 3u + 2u];
        const double distance_squared = dx * dx + dy * dy + dz * dz;
        if (!std::isfinite(distance_squared)) {
          error = "GFN1 D3 coordinate differences overflow floating-point range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        if (distance_squared < kMinimumDistanceSquared || distance_squared > cutoff_squared) {
          continue;
        }
        const PairCoefficient coefficient = pair_coefficient(data, first, second, workspace);
        PairContribution contribution;
        status = evaluate_pair(data, pair_index(data, batch, first, second), distance_squared,
                               coefficient, contribution, error);
        if (status != XTBLOOM_STATUS_SUCCESS) {
          return status;
        }
        workspace.batch_scratch[static_cast<std::size_t>(batch)] += contribution.energy;
        if (derivatives) {
          workspace.coordination_adjoints[first_index] += contribution.first_cn;
          workspace.coordination_adjoints[second_index] += contribution.second_cn;
          const double components[3]{dx, dy, dz};
          for (std::size_t axis = 0u; axis < 3u; ++axis) {
            const double increment = contribution.gradient_scale * components[axis];
            workspace.gradient_scratch[first_index * 3u + axis] += increment;
            workspace.gradient_scratch[second_index * 3u + axis] -= increment;
          }
        }
      }
    }
  }
  if (!finite_values(workspace.batch_scratch, batch_count) ||
      (derivatives && (!finite_values(workspace.coordination_adjoints, atom_count) ||
                       !finite_values(workspace.gradient_scratch, coordinate_count)))) {
    error = "GFN1 D3 accumulation exceeded floating-point range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  if (derivatives) {
    status = add_coordination_gradient_cpu(data.coordination_plan, positions,
                                           workspace.coordination_adjoints,
                                           workspace.gradient_scratch, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
  }

  for (std::size_t batch = 0u; batch < batch_count; ++batch) {
    if (!std::isfinite(energies[batch] + workspace.batch_scratch[batch])) {
      error = "GFN1 D3 energy publication would exceed floating-point range";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  if (derivatives) {
    for (std::size_t coordinate = 0u; coordinate < coordinate_count; ++coordinate) {
      if (!std::isfinite(gradients[coordinate] + workspace.gradient_scratch[coordinate])) {
        error = "GFN1 D3 gradient publication would exceed floating-point range";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
  }
  for (std::size_t batch = 0u; batch < batch_count; ++batch) {
    energies[batch] += workspace.batch_scratch[batch];
  }
  if (derivatives) {
    for (std::size_t coordinate = 0u; coordinate < coordinate_count; ++coordinate) {
      gradients[coordinate] += workspace.gradient_scratch[coordinate];
    }
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
