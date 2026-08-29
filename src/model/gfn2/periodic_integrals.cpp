// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/periodic_integrals.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "model/common/integrals.hpp"

namespace xtbloom::detail::gfn2 {
namespace {

constexpr std::size_t kDipoleComponents = common::kIntegralDipoleComponents;
constexpr std::size_t kQuadrupoleComponents = common::kIntegralQuadrupoleComponents;
constexpr std::size_t kMultipoleComponents = common::kIntegralMultipoleComponents;
constexpr double kMinimumImageDistanceSquared = std::numeric_limits<double>::epsilon();

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

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

bool append_bytes(std::size_t bytes, std::size_t alignment, std::size_t& cursor,
                  std::size_t& offset) noexcept {
  if (!align_up(cursor, alignment, offset)) return false;
  return checked_add(offset, bytes, cursor);
}

bool append_doubles(std::size_t elements, std::size_t& cursor, std::size_t& offset) noexcept {
  std::size_t bytes = 0u;
  return checked_multiply(elements, sizeof(double), bytes) &&
         append_bytes(bytes, alignof(double), cursor, offset);
}

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) noexcept {
  if (bytes != 0u && pointer == nullptr) return false;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;
  range = {begin, begin + bytes};
  return true;
}

bool make_double_range(const double* pointer, std::size_t elements, AddressRange& range) noexcept {
  std::size_t bytes = 0u;
  return checked_multiply(elements, sizeof(double), bytes) && make_range(pointer, bytes, range);
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

bool finite_values(const double* values, std::size_t count) noexcept {
  if (values == nullptr) return false;
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) return false;
  }
  return true;
}

std::size_t spherical_count(std::uint8_t angular_momentum) noexcept {
  return 2u * static_cast<std::size_t>(angular_momentum) + 1u;
}

bool overlaps_basis_storage(const AddressRange& active, const BasisPlan& basis) noexcept {
  AddressRange descriptor;
  if (!make_range(&basis, sizeof(basis), descriptor) || ranges_overlap(active, descriptor)) {
    return true;
  }
  return overlaps_vector(active, basis.atom_offsets) ||
         overlaps_vector(active, basis.batch_shell_offsets) ||
         overlaps_vector(active, basis.batch_orbital_offsets) ||
         overlaps_vector(active, basis.batch_cartesian_orbital_offsets) ||
         overlaps_vector(active, basis.batch_primitive_offsets) ||
         overlaps_vector(active, basis.atom_shell_offsets) ||
         overlaps_vector(active, basis.atom_orbital_offsets) ||
         overlaps_vector(active, basis.atom_cartesian_orbital_offsets) ||
         overlaps_vector(active, basis.atom_primitive_offsets) ||
         overlaps_vector(active, basis.shell_orbital_offsets) ||
         overlaps_vector(active, basis.shell_cartesian_orbital_offsets) ||
         overlaps_vector(active, basis.shell_primitive_offsets) ||
         overlaps_vector(active, basis.shell_to_atom) ||
         overlaps_vector(active, basis.principal_quantum_numbers) ||
         overlaps_vector(active, basis.angular_momenta) ||
         overlaps_vector(active, basis.shell_is_valence) ||
         overlaps_vector(active, basis.slater_exponents) ||
         overlaps_vector(active, basis.primitive_exponents) ||
         overlaps_vector(active, basis.primitive_coefficients);
}

bool overlaps_integral_storage(const AddressRange& active, const IntegralPlan& integrals) noexcept {
  AddressRange descriptor;
  return !make_range(&integrals, sizeof(integrals), descriptor) ||
         ranges_overlap(active, descriptor) || overlaps_vector(active, integrals.matrix_offsets);
}

bool overlaps_h0_storage(const AddressRange& active, const H0Plan& h0) noexcept {
  AddressRange descriptor;
  if (!make_range(&h0, sizeof(h0), descriptor) || ranges_overlap(active, descriptor)) return true;
  return overlaps_vector(active, h0.atom_offsets) ||
         overlaps_vector(active, h0.batch_shell_offsets) ||
         overlaps_vector(active, h0.batch_orbital_offsets) ||
         overlaps_vector(active, h0.matrix_offsets) ||
         overlaps_vector(active, h0.shell_pair_offsets) ||
         overlaps_vector(active, h0.atomic_radii) || overlaps_vector(active, h0.shell_levels) ||
         overlaps_vector(active, h0.shell_coordination_scale) ||
         overlaps_vector(active, h0.shell_polynomial) ||
         overlaps_vector(active, h0.shell_pair_scale);
}

bool overlaps_model_storage(const AddressRange& active, const BasisPlan& basis,
                            const IntegralPlan& integrals, const H0Plan& h0,
                            const PeriodicIntegralPlan& periodic,
                            const PeriodicShortRangePlan& topology) noexcept {
  return overlaps_basis_storage(active, basis) || overlaps_integral_storage(active, integrals) ||
         overlaps_h0_storage(active, h0) ||
         periodic.overlaps_storage(reinterpret_cast<const void*>(active.begin),
                                   active.end - active.begin) ||
         topology.overlaps_storage(reinterpret_cast<const void*>(active.begin),
                                   active.end - active.begin);
}

bool valid_plan_relationship(const BasisPlan& basis, const IntegralPlan& integrals,
                             const PeriodicIntegralPlan& periodic,
                             const PeriodicShortRangePlan& topology) noexcept {
  const PeriodicIntegralPlanData* data = periodic.data();
  return data != nullptr && periodic.sealed() && topology.sealed() &&
         data->batch_size == basis.batch_size && data->total_atoms == basis.total_atoms &&
         data->total_shells == basis.total_shells &&
         data->total_matrix_elements == integrals.total_matrix_elements &&
         data->integral_cutoff == integrals.integral_cutoff &&
         data->minimum_primitive_exponent == basis.minimum_primitive_exponent &&
         data->atom_offsets == basis.atom_offsets &&
         data->matrix_offsets == integrals.matrix_offsets &&
         data->topology_identity == topology.identity() &&
         topology.batch_size() == basis.batch_size && topology.total_atoms() == basis.total_atoms &&
         topology.atom_offsets() == basis.atom_offsets &&
         data->translation_offsets.size() == static_cast<std::size_t>(basis.batch_size + 1) &&
         data->translation_offsets.front() == 0 &&
         data->translation_offsets.back() == static_cast<std::int64_t>(data->translations.size()) &&
         data->realspace_cutoffs.size() == static_cast<std::size_t>(basis.batch_size) &&
         data->workspace_size_bytes >= kPeriodicIntegralWorkspaceAlignment;
}

xtbloom_status_t validate_context(const BasisPlan& basis, const IntegralPlan& integrals,
                                  const H0Plan& h0, const PeriodicIntegralPlan& periodic,
                                  const PeriodicShortRangePlan& topology,
                                  const PeriodicShortRangeGeometry& geometry,
                                  const PeriodicShortRangeWorkspace& geometry_workspace,
                                  const double* coordination_numbers, void* workspace,
                                  std::size_t workspace_size, std::string& error) {
  xtbloom_status_t status = common::validate_integral_plan(basis, integrals, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = validate_h0_plan(basis, integrals, h0, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  /* The image radius is derived from this aggregate. Recompute it after the
   * general basis validation so a mutated value object cannot retain a stale
   * minimum and silently omit diffuse-image contributions. */
  const double actual_minimum =
      *std::min_element(basis.primitive_exponents.begin(), basis.primitive_exponents.end());
  if (actual_minimum != basis.minimum_primitive_exponent) {
    error = "periodic integral basis minimum exponent is inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!valid_plan_relationship(basis, integrals, periodic, topology) ||
      periodic.data()->topology_identity == nullptr ||
      periodic.data()->batch_size != basis.batch_size ||
      periodic.data()->total_atoms != basis.total_atoms ||
      periodic.data()->total_shells != basis.total_shells ||
      periodic.data()->total_matrix_elements != integrals.total_matrix_elements ||
      periodic.data()->integral_cutoff != integrals.integral_cutoff ||
      periodic.data()->minimum_primitive_exponent != basis.minimum_primitive_exponent ||
      periodic.data()->atom_offsets != basis.atom_offsets ||
      periodic.data()->matrix_offsets != integrals.matrix_offsets ||
      geometry.plan_identity != periodic.data()->topology_identity ||
      geometry_workspace.plan_identity != periodic.data()->topology_identity ||
      geometry.geometry_generation == 0u || geometry.wrapped_positions == nullptr ||
      geometry.wrapped_positions != geometry_workspace.wrapped_positions ||
      geometry.wrapped_position_elements != basis.total_atoms * 3 ||
      geometry_workspace.wrapped_position_elements != basis.total_atoms * 3) {
    error = "periodic integral plan, topology, or geometry identity is inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  status = validate_periodic_short_range_workspace(topology, geometry_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (workspace == nullptr ||
      reinterpret_cast<std::uintptr_t>(workspace) % kPeriodicIntegralWorkspaceAlignment != 0u ||
      workspace_size < periodic.workspace_size_bytes() ||
      periodic.overlaps_storage(workspace, periodic.workspace_size_bytes())) {
    error = "periodic integral workspace is too small, misaligned, or aliases its plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t atom_count = static_cast<std::size_t>(basis.total_atoms);
  if (!finite_values(geometry.wrapped_positions, atom_count * 3u)) {
    error = "periodic integral wrapped positions contain NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!finite_values(coordination_numbers, atom_count)) {
    error = "periodic H0 coordination numbers contain NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

std::int64_t absolute_index(std::int64_t value) noexcept { return value < 0 ? -value : value; }

bool lexicographic_translation_position(const std::array<std::int64_t, 3>& repeat,
                                        const std::array<std::int64_t, 3>& index,
                                        std::size_t& position) noexcept {
  const std::uint64_t width_y = static_cast<std::uint64_t>(repeat[1]) * 2u + 1u;
  const std::uint64_t width_z = static_cast<std::uint64_t>(repeat[2]) * 2u + 1u;
  const std::uint64_t first = static_cast<std::uint64_t>(index[0] + repeat[0]);
  const std::uint64_t second = static_cast<std::uint64_t>(index[1] + repeat[1]);
  const std::uint64_t third = static_cast<std::uint64_t>(index[2] + repeat[2]);
  if (first > std::numeric_limits<std::uint64_t>::max() / width_y) return false;
  std::uint64_t linear = first * width_y + second;
  if (linear > std::numeric_limits<std::uint64_t>::max() / width_z) return false;
  linear = linear * width_z + third;
  const std::uint64_t origin =
      (static_cast<std::uint64_t>(repeat[0]) * width_y + static_cast<std::uint64_t>(repeat[1])) *
          width_z +
      static_cast<std::uint64_t>(repeat[2]);
  const std::uint64_t stored = linear == origin ? 0u : (linear < origin ? linear + 1u : linear);
  if (stored > std::numeric_limits<std::size_t>::max()) return false;
  position = static_cast<std::size_t>(stored);
  return true;
}

xtbloom_status_t make_tblite_ordered_translations(const Lattice3D& lattice, double cutoff,
                                                  std::vector<LatticeTranslation>& ordered,
                                                  std::string& error) {
  std::vector<LatticeTranslation> lexicographic;
  xtbloom_status_t status = make_lattice_translations(
      lattice, cutoff, LatticeOriginPolicy::kInclude, lexicographic, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (lexicographic.empty()) {
    error = "periodic integral lattice generator returned no origin";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  std::array<std::int64_t, 3> repeat{};
  for (const LatticeTranslation& translation : lexicographic) {
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      repeat[axis] = std::max(repeat[axis], absolute_index(translation.index[axis]));
    }
  }

  try {
    std::vector<LatticeTranslation> created;
    created.reserve(lexicographic.size());
    for (std::int64_t nx = 0; nx <= repeat[0]; ++nx) {
      for (std::int64_t ny = 0; ny <= repeat[1]; ++ny) {
        for (std::int64_t nz = 0; nz <= repeat[2]; ++nz) {
          const std::array<std::int64_t, 2> signs_x{1, -1};
          const std::array<std::int64_t, 2> signs_y{1, -1};
          const std::array<std::int64_t, 2> signs_z{1, -1};
          const std::size_t count_x = nx == 0 ? 1u : 2u;
          const std::size_t count_y = ny == 0 ? 1u : 2u;
          const std::size_t count_z = nz == 0 ? 1u : 2u;
          for (std::size_t sx = 0; sx < count_x; ++sx) {
            for (std::size_t sy = 0; sy < count_y; ++sy) {
              for (std::size_t sz = 0; sz < count_z; ++sz) {
                const std::array<std::int64_t, 3> index{nx * signs_x[sx], ny * signs_y[sy],
                                                        nz * signs_z[sz]};
                std::size_t source = 0u;
                if (!lexicographic_translation_position(repeat, index, source) ||
                    source >= lexicographic.size() || lexicographic[source].index != index) {
                  error = "periodic integral translation reordering lost an image";
                  return XTBLOOM_STATUS_INTERNAL_ERROR;
                }
                created.push_back(lexicographic[source]);
              }
            }
          }
        }
      }
    }
    if (created.size() != lexicographic.size()) {
      error = "periodic integral translation ordering produced an inconsistent count";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    ordered = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate tblite-ordered periodic integral translations";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

enum class ImageGeometryStatus {
  kInside,
  kOutside,
};

ImageGeometryStatus image_geometry(const double* ket, const double* bra,
                                   const LatticeTranslation& translation, double cutoff,
                                   double cutoff_squared, std::array<double, 3>& vector,
                                   double& distance_squared) noexcept {
  using lattice_binary64_detail::absolute;
  using lattice_binary64_detail::finite;
  using lattice_binary64_detail::rounded_add;
  using lattice_binary64_detail::rounded_multiply;
  using lattice_binary64_detail::rounded_subtract;

  for (std::size_t axis = 0; axis < 3u; ++axis) {
    vector[axis] =
        rounded_subtract(rounded_subtract(ket[axis], bra[axis]), translation.cartesian[axis]);
    if (!finite(vector[axis]) || absolute(vector[axis]) > cutoff) {
      return ImageGeometryStatus::kOutside;
    }
  }
  distance_squared = rounded_add(
      rounded_add(rounded_multiply(vector[0], vector[0]), rounded_multiply(vector[1], vector[1])),
      rounded_multiply(vector[2], vector[2]));
  /* Match tblite's adjacency filter exactly. The epsilon exclusion applies
   * to every pair/image, while central onsite blocks are evaluated once by
   * the separate onsite traversal below. */
  if (!finite(distance_squared) || distance_squared > cutoff_squared ||
      distance_squared < kMinimumImageDistanceSquared) {
    return ImageGeometryStatus::kOutside;
  }
  return ImageGeometryStatus::kInside;
}

template <typename PairOperation, typename OnsiteOperation>
xtbloom_status_t for_each_periodic_pair(const BasisPlan& basis,
                                        const PeriodicIntegralPlan& periodic,
                                        const PeriodicShortRangeGeometry& geometry,
                                        PairOperation&& pair_operation,
                                        OnsiteOperation&& onsite_operation, std::string& error) {
  for (std::int64_t system = 0; system < basis.batch_size; ++system) {
    const std::size_t system_index = static_cast<std::size_t>(system);
    const std::int64_t atom_begin = basis.atom_offsets[system_index];
    const std::int64_t atom_end = basis.atom_offsets[system_index + 1u];
    const PeriodicIntegralTranslationView translations = periodic.translations(system);
    const double cutoff = periodic.realspace_cutoff(system);
    const double cutoff_squared = cutoff * cutoff;
    if (translations.data == nullptr || translations.size <= 0 || !(cutoff > 0.0) ||
        !std::isfinite(cutoff_squared)) {
      error = "periodic integral translation topology is incomplete";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t ket_atom = atom_begin; ket_atom < atom_end; ++ket_atom) {
      const double* ket = geometry.wrapped_positions + static_cast<std::size_t>(ket_atom) * 3u;
      for (std::int64_t bra_atom = atom_begin; bra_atom <= ket_atom; ++bra_atom) {
        const double* bra = geometry.wrapped_positions + static_cast<std::size_t>(bra_atom) * 3u;
        for (std::int64_t image = 0; image < translations.size; ++image) {
          std::array<double, 3> vector{};
          double distance_squared = 0.0;
          if (image_geometry(ket, bra, translations.data[image], cutoff, cutoff_squared, vector,
                             distance_squared) == ImageGeometryStatus::kOutside) {
            continue;
          }
          const xtbloom_status_t status =
              pair_operation(system, ket_atom, bra_atom, vector, distance_squared, error);
          if (status != XTBLOOM_STATUS_SUCCESS) return status;
        }
      }
      const xtbloom_status_t status = onsite_operation(system, ket_atom, error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

template <typename Operation>
void for_each_shell_pair(const BasisPlan& basis, std::int64_t ket_atom, std::int64_t bra_atom,
                         Operation&& operation) {
  const std::size_t ket_index = static_cast<std::size_t>(ket_atom);
  const std::size_t bra_index = static_cast<std::size_t>(bra_atom);
  for (std::int64_t ket_shell = basis.atom_shell_offsets[ket_index];
       ket_shell < basis.atom_shell_offsets[ket_index + 1u]; ++ket_shell) {
    for (std::int64_t bra_shell = basis.atom_shell_offsets[bra_index];
         bra_shell < basis.atom_shell_offsets[bra_index + 1u]; ++bra_shell) {
      operation(static_cast<std::size_t>(bra_shell), static_cast<std::size_t>(ket_shell));
    }
  }
}

struct MatrixBlock {
  std::size_t matrix_offset = 0u;
  std::size_t orbital_begin = 0u;
  std::size_t orbital_count = 0u;
  std::size_t bra_orbital = 0u;
  std::size_t ket_orbital = 0u;
  std::size_t bra_count = 0u;
  std::size_t ket_count = 0u;
};

MatrixBlock matrix_block(const BasisPlan& basis, const IntegralPlan& integrals, std::int64_t system,
                         std::size_t bra_shell, std::size_t ket_shell) noexcept {
  const std::size_t system_index = static_cast<std::size_t>(system);
  MatrixBlock block;
  block.matrix_offset = static_cast<std::size_t>(integrals.matrix_offsets[system_index]);
  block.orbital_begin = static_cast<std::size_t>(basis.batch_orbital_offsets[system_index]);
  block.orbital_count = static_cast<std::size_t>(basis.batch_orbital_offsets[system_index + 1u] -
                                                 basis.batch_orbital_offsets[system_index]);
  block.bra_orbital = static_cast<std::size_t>(basis.shell_orbital_offsets[bra_shell]);
  block.ket_orbital = static_cast<std::size_t>(basis.shell_orbital_offsets[ket_shell]);
  block.bra_count = spherical_count(basis.angular_momenta[bra_shell]);
  block.ket_count = spherical_count(basis.angular_momenta[ket_shell]);
  return block;
}

std::size_t forward_matrix_index(const MatrixBlock& block, std::size_t bra_ao,
                                 std::size_t ket_ao) noexcept {
  return block.matrix_offset +
         (block.bra_orbital - block.orbital_begin + bra_ao) * block.orbital_count +
         block.ket_orbital - block.orbital_begin + ket_ao;
}

std::size_t reverse_matrix_index(const MatrixBlock& block, std::size_t bra_ao,
                                 std::size_t ket_ao) noexcept {
  return block.matrix_offset +
         (block.ket_orbital - block.orbital_begin + ket_ao) * block.orbital_count +
         block.bra_orbital - block.orbital_begin + bra_ao;
}

void shifted_multipoles(const std::array<double, 3>& vector, double overlap,
                        const std::array<double, kDipoleComponents>& dipole,
                        const std::array<double, kQuadrupoleComponents>& quadrupole,
                        std::array<double, kDipoleComponents>& shifted_dipole,
                        std::array<double, kQuadrupoleComponents>& shifted_quadrupole) noexcept {
  for (std::size_t component = 0; component < kDipoleComponents; ++component) {
    shifted_dipole[component] = dipole[component] + vector[component] * overlap;
  }
  std::array<double, kQuadrupoleComponents> shift{};
  shift[0] = 2.0 * vector[0] * dipole[0] + vector[0] * vector[0] * overlap;
  shift[1] = vector[0] * dipole[1] + vector[1] * dipole[0] + vector[0] * vector[1] * overlap;
  shift[2] = 2.0 * vector[1] * dipole[1] + vector[1] * vector[1] * overlap;
  shift[3] = vector[0] * dipole[2] + vector[2] * dipole[0] + vector[0] * vector[2] * overlap;
  shift[4] = vector[1] * dipole[2] + vector[2] * dipole[1] + vector[1] * vector[2] * overlap;
  shift[5] = 2.0 * vector[2] * dipole[2] + vector[2] * vector[2] * overlap;
  const double trace = 0.5 * (shift[0] + shift[2] + shift[5]);
  for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
    const bool diagonal = component == 0u || component == 2u || component == 5u;
    shifted_quadrupole[component] =
        quadrupole[component] + 1.5 * shift[component] - (diagonal ? trace : 0.0);
  }
}

void add_multipole_shift_pullback(
    const std::array<double, 3>& vector, double overlap,
    const std::array<double, kDipoleComponents>& dipole,
    const std::array<double, kDipoleComponents>& reverse_dipole_adjoint,
    const std::array<double, kQuadrupoleComponents>& reverse_quadrupole_adjoint,
    double& overlap_adjoint, std::array<double, kDipoleComponents>& dipole_adjoint,
    std::array<double, 3>& vector_adjoint) noexcept {
  for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
    overlap_adjoint += reverse_dipole_adjoint[coordinate] * vector[coordinate];
    dipole_adjoint[coordinate] += reverse_dipole_adjoint[coordinate];
    vector_adjoint[coordinate] += reverse_dipole_adjoint[coordinate] * overlap;
  }

  const double diagonal_sum =
      reverse_quadrupole_adjoint[0] + reverse_quadrupole_adjoint[2] + reverse_quadrupole_adjoint[5];
  const std::array<double, kQuadrupoleComponents> raw{
      1.5 * reverse_quadrupole_adjoint[0] - 0.5 * diagonal_sum,
      1.5 * reverse_quadrupole_adjoint[1],
      1.5 * reverse_quadrupole_adjoint[2] - 0.5 * diagonal_sum,
      1.5 * reverse_quadrupole_adjoint[3],
      1.5 * reverse_quadrupole_adjoint[4],
      1.5 * reverse_quadrupole_adjoint[5] - 0.5 * diagonal_sum,
  };
  const double x = vector[0];
  const double y = vector[1];
  const double z = vector[2];
  const double dx = dipole[0];
  const double dy = dipole[1];
  const double dz = dipole[2];

  overlap_adjoint += raw[0] * x * x + raw[1] * x * y + raw[2] * y * y + raw[3] * x * z +
                     raw[4] * y * z + raw[5] * z * z;
  dipole_adjoint[0] += 2.0 * raw[0] * x + raw[1] * y + raw[3] * z;
  dipole_adjoint[1] += raw[1] * x + 2.0 * raw[2] * y + raw[4] * z;
  dipole_adjoint[2] += raw[3] * x + raw[4] * y + 2.0 * raw[5] * z;
  vector_adjoint[0] += raw[0] * (2.0 * dx + 2.0 * x * overlap) + raw[1] * (dy + y * overlap) +
                       raw[3] * (dz + z * overlap);
  vector_adjoint[1] += raw[1] * (dx + x * overlap) + raw[2] * (2.0 * dy + 2.0 * y * overlap) +
                       raw[4] * (dz + z * overlap);
  vector_adjoint[2] += raw[3] * (dx + x * overlap) + raw[4] * (dy + y * overlap) +
                       raw[5] * (2.0 * dz + 2.0 * z * overlap);
}

struct H0ImageFactor {
  double factor = 0.0;
  double average_level = 0.0;
  double spatial_scale = 0.0;
  double spatial_scale_derivative = 0.0;
  double distance = 0.0;
};

H0ImageFactor h0_image_factor(const BasisPlan& basis, const H0Plan& h0,
                              const double* coordination_numbers, std::int64_t system,
                              std::int64_t ket_atom, std::int64_t bra_atom, std::size_t ket_shell,
                              std::size_t bra_shell, double distance_squared) noexcept {
  const std::size_t ket_atom_index = static_cast<std::size_t>(ket_atom);
  const std::size_t bra_atom_index = static_cast<std::size_t>(bra_atom);
  const double ket_level = h0.shell_levels[ket_shell] - h0.shell_coordination_scale[ket_shell] *
                                                            coordination_numbers[ket_atom_index];
  const double bra_level = h0.shell_levels[bra_shell] - h0.shell_coordination_scale[bra_shell] *
                                                            coordination_numbers[bra_atom_index];
  const double distance = std::sqrt(distance_squared);
  const double reduced =
      std::sqrt(distance / (h0.atomic_radii[ket_atom_index] + h0.atomic_radii[bra_atom_index]));
  const double ket_polynomial = 1.0 + h0.shell_polynomial[ket_shell] * reduced;
  const double bra_polynomial = 1.0 + h0.shell_polynomial[bra_shell] * reduced;
  const std::size_t system_index = static_cast<std::size_t>(system);
  const std::int64_t shell_begin = basis.batch_shell_offsets[system_index];
  const std::int64_t shell_count = basis.batch_shell_offsets[system_index + 1u] - shell_begin;
  const std::int64_t pair_begin = h0.shell_pair_offsets[system_index];
  const std::int64_t local_bra = static_cast<std::int64_t>(bra_shell) - shell_begin;
  const std::int64_t local_ket = static_cast<std::int64_t>(ket_shell) - shell_begin;
  const double pair_scale = h0.shell_pair_scale[static_cast<std::size_t>(
      pair_begin + local_bra * shell_count + local_ket)];
  H0ImageFactor result;
  result.average_level = 0.5 * (ket_level + bra_level);
  result.spatial_scale = pair_scale * ket_polynomial * bra_polynomial;
  result.spatial_scale_derivative = pair_scale *
                                    (h0.shell_polynomial[ket_shell] * bra_polynomial +
                                     h0.shell_polynomial[bra_shell] * ket_polynomial) *
                                    reduced / (2.0 * distance);
  result.factor = result.average_level * result.spatial_scale;
  result.distance = distance;
  return result;
}

double h0_onsite_factor(const H0Plan& h0, const double* coordination_numbers, std::int64_t atom,
                        std::size_t ket_shell, std::size_t bra_shell) noexcept {
  const std::size_t atom_index = static_cast<std::size_t>(atom);
  const double ket_level = h0.shell_levels[ket_shell] - h0.shell_coordination_scale[ket_shell] *
                                                            coordination_numbers[atom_index];
  const double bra_level = h0.shell_levels[bra_shell] - h0.shell_coordination_scale[bra_shell] *
                                                            coordination_numbers[atom_index];
  return 0.5 * (ket_level + bra_level);
}

bool valid_buffer_set(const BasisPlan& basis, const IntegralPlan& integrals, const H0Plan& h0,
                      const PeriodicIntegralPlan& periodic, const PeriodicShortRangePlan& topology,
                      const PeriodicShortRangeWorkspace& geometry_workspace,
                      const PeriodicShortRangeGeometry& geometry, void* workspace,
                      std::size_t workspace_size, const AddressRange* numerical,
                      std::size_t numerical_count) noexcept {
  AddressRange integral_workspace;
  AddressRange geometry_storage;
  AddressRange geometry_descriptor;
  AddressRange workspace_descriptor;
  if (!make_range(workspace, workspace_size, integral_workspace) ||
      !make_range(geometry_workspace.workspace_base, geometry_workspace.workspace_size_bytes,
                  geometry_storage) ||
      !make_range(&geometry, sizeof(geometry), geometry_descriptor) ||
      !make_range(&geometry_workspace, sizeof(geometry_workspace), workspace_descriptor) ||
      ranges_overlap(integral_workspace, geometry_storage) ||
      ranges_overlap(integral_workspace, geometry_descriptor) ||
      ranges_overlap(integral_workspace, workspace_descriptor) ||
      overlaps_model_storage(integral_workspace, basis, integrals, h0, periodic, topology)) {
    return false;
  }
  for (std::size_t first = 0; first < numerical_count; ++first) {
    if (ranges_overlap(numerical[first], integral_workspace) ||
        ranges_overlap(numerical[first], geometry_storage) ||
        ranges_overlap(numerical[first], geometry_descriptor) ||
        ranges_overlap(numerical[first], workspace_descriptor) ||
        overlaps_basis_storage(numerical[first], basis) ||
        overlaps_integral_storage(numerical[first], integrals) ||
        overlaps_h0_storage(numerical[first], h0) ||
        periodic.overlaps_storage(reinterpret_cast<const void*>(numerical[first].begin),
                                  numerical[first].end - numerical[first].begin) ||
        topology.overlaps_storage(reinterpret_cast<const void*>(numerical[first].begin),
                                  numerical[first].end - numerical[first].begin)) {
      return false;
    }
    for (std::size_t second = first + 1u; second < numerical_count; ++second) {
      if (ranges_overlap(numerical[first], numerical[second])) return false;
    }
  }
  return true;
}

}  // namespace

bool PeriodicIntegralPlan::overlaps_storage(const void* pointer, std::size_t bytes) const noexcept {
  if (bytes == 0u) return false;
  AddressRange active;
  AddressRange descriptor;
  AddressRange plan_data;
  if (data_ == nullptr || !make_range(pointer, bytes, active) ||
      !make_range(this, sizeof(*this), descriptor) ||
      !make_range(data_.get(), sizeof(*data_), plan_data)) {
    return true;
  }
  return ranges_overlap(active, descriptor) || ranges_overlap(active, plan_data) ||
         overlaps_vector(active, data_->atom_offsets) ||
         overlaps_vector(active, data_->matrix_offsets) ||
         overlaps_vector(active, data_->translation_offsets) ||
         overlaps_vector(active, data_->realspace_cutoffs) ||
         overlaps_vector(active, data_->translations);
}

xtbloom_status_t make_periodic_integral_plan(const BasisPlan& basis, const IntegralPlan& integrals,
                                             const PeriodicShortRangePlan& periodic,
                                             PeriodicIntegralPlan& plan, std::string& error) {
  xtbloom_status_t status = common::validate_integral_plan(basis, integrals, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (!periodic.sealed() || periodic.batch_size() != basis.batch_size ||
      periodic.total_atoms() != basis.total_atoms ||
      periodic.atom_offsets() != basis.atom_offsets || !(basis.minimum_primitive_exponent > 0.0) ||
      !std::isfinite(basis.minimum_primitive_exponent)) {
    error = "periodic integral plan requires matching topology and a finite minimum exponent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const double actual_minimum =
      *std::min_element(basis.primitive_exponents.begin(), basis.primitive_exponents.end());
  if (actual_minimum != basis.minimum_primitive_exponent) {
    error = "periodic integral basis minimum exponent is inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  /* Pinned tblite caps the exponent-derived image search at 40 bohr. This is
   * a compatibility rule, not a Wigner--Seitz or performance-only cutoff. */
  const double cutoff =
      std::min(std::sqrt(2.0 * integrals.integral_cutoff / basis.minimum_primitive_exponent),
               kPeriodicIntegralMaximumCutoffBohr);
  if (!(cutoff > 0.0) || !std::isfinite(cutoff)) {
    error = "periodic integral real-space cutoff is invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    auto created = std::make_shared<PeriodicIntegralPlanData>();
    created->batch_size = basis.batch_size;
    created->total_atoms = basis.total_atoms;
    created->total_shells = basis.total_shells;
    created->total_matrix_elements = integrals.total_matrix_elements;
    created->integral_cutoff = integrals.integral_cutoff;
    created->minimum_primitive_exponent = basis.minimum_primitive_exponent;
    created->atom_offsets = basis.atom_offsets;
    created->matrix_offsets = integrals.matrix_offsets;
    created->translation_offsets.resize(static_cast<std::size_t>(basis.batch_size + 1), 0);
    created->realspace_cutoffs.resize(static_cast<std::size_t>(basis.batch_size), cutoff);
    created->topology_identity = periodic.identity();

    for (std::int64_t system = 0; system < basis.batch_size; ++system) {
      std::vector<LatticeTranslation> local;
      std::string local_error;
      status =
          make_tblite_ordered_translations(periodic.lattice(system), cutoff, local, local_error);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        error = "periodic integral translations for system " + std::to_string(system) + ": " +
                local_error;
        return status;
      }
      if (local.size() > created->translations.max_size() - created->translations.size() ||
          local.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) -
                             created->translations.size()) {
        error = "periodic integral translation count overflows";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      created->translations.insert(created->translations.end(), local.begin(), local.end());
      created->translation_offsets[static_cast<std::size_t>(system + 1)] =
          static_cast<std::int64_t>(created->translations.size());
    }

    const std::size_t matrix_elements = static_cast<std::size_t>(integrals.total_matrix_elements);
    const std::size_t atom_elements = static_cast<std::size_t>(basis.total_atoms);
    const std::size_t batch_elements = static_cast<std::size_t>(basis.batch_size);
    std::size_t dipole_elements = 0u;
    std::size_t quadrupole_elements = 0u;
    std::size_t gradient_elements = 0u;
    std::size_t strain_elements = 0u;
    if (!checked_multiply(matrix_elements, kDipoleComponents, dipole_elements) ||
        !checked_multiply(matrix_elements, kQuadrupoleComponents, quadrupole_elements) ||
        !checked_multiply(atom_elements, 3u, gradient_elements) ||
        !checked_multiply(batch_elements, 9u, strain_elements)) {
      error = "periodic integral workspace element count overflows";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    std::size_t cursor = 0u;
    if (!append_bytes(sizeof(common::IntegralWorkspace), alignof(common::IntegralWorkspace), cursor,
                      created->integral_workspace_offset) ||
        !append_doubles(matrix_elements, cursor, created->overlap_scratch_offset) ||
        !append_doubles(dipole_elements, cursor, created->dipole_scratch_offset) ||
        !append_doubles(quadrupole_elements, cursor, created->quadrupole_scratch_offset) ||
        !append_doubles(matrix_elements, cursor, created->h0_scratch_offset) ||
        !append_doubles(atom_elements, cursor, created->coordination_scratch_offset) ||
        !append_doubles(gradient_elements, cursor, created->gradient_scratch_offset) ||
        !append_doubles(strain_elements, cursor, created->strain_scratch_offset) ||
        !align_up(cursor, kPeriodicIntegralWorkspaceAlignment, created->workspace_size_bytes)) {
      error = "periodic integral workspace byte count overflows";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    plan = PeriodicIntegralPlan(std::move(created));
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate periodic integral plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t evaluate_periodic_integrals_h0_cpu(
    const BasisPlan& basis, const IntegralPlan& integrals, const H0Plan& h0,
    const PeriodicIntegralPlan& periodic, const PeriodicShortRangePlan& topology,
    const PeriodicShortRangeGeometry& geometry,
    const PeriodicShortRangeWorkspace& geometry_workspace, const double* coordination_numbers,
    double* overlap, double* dipole, double* quadrupole, double* hamiltonian, void* workspace,
    std::size_t workspace_size, std::string& error) {
  xtbloom_status_t status =
      validate_context(basis, integrals, h0, periodic, topology, geometry, geometry_workspace,
                       coordination_numbers, workspace, workspace_size, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  const std::size_t matrix_elements = static_cast<std::size_t>(integrals.total_matrix_elements);
  const std::size_t atom_elements = static_cast<std::size_t>(basis.total_atoms);
  std::size_t dipole_elements = 0u;
  std::size_t quadrupole_elements = 0u;
  if (!checked_multiply(matrix_elements, kDipoleComponents, dipole_elements) ||
      !checked_multiply(matrix_elements, kQuadrupoleComponents, quadrupole_elements)) {
    error = "periodic integral output dimensions overflow";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::array<AddressRange, 5> numerical{};
  if (!make_double_range(coordination_numbers, atom_elements, numerical[0]) ||
      !make_double_range(overlap, matrix_elements, numerical[1]) ||
      !make_double_range(dipole, dipole_elements, numerical[2]) ||
      !make_double_range(quadrupole, quadrupole_elements, numerical[3]) ||
      !make_double_range(hamiltonian, matrix_elements, numerical[4]) ||
      !valid_buffer_set(basis, integrals, h0, periodic, topology, geometry_workspace, geometry,
                        workspace, workspace_size, numerical.data(), numerical.size())) {
    error = "periodic integral inputs, outputs, workspace, and plans must be disjoint";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const PeriodicIntegralPlanData& data = *periodic.data();
  auto& integral_scratch =
      *offset_pointer<common::IntegralWorkspace>(workspace, data.integral_workspace_offset);
  double* overlap_scratch = offset_pointer<double>(workspace, data.overlap_scratch_offset);
  double* dipole_scratch = offset_pointer<double>(workspace, data.dipole_scratch_offset);
  double* quadrupole_scratch = offset_pointer<double>(workspace, data.quadrupole_scratch_offset);
  double* h0_scratch = offset_pointer<double>(workspace, data.h0_scratch_offset);
  std::fill_n(overlap_scratch, matrix_elements, 0.0);
  std::fill_n(dipole_scratch, dipole_elements, 0.0);
  std::fill_n(quadrupole_scratch, quadrupole_elements, 0.0);
  std::fill_n(h0_scratch, matrix_elements, 0.0);

  status = for_each_periodic_pair(
      basis, periodic, geometry,
      [&](std::int64_t system, std::int64_t ket_atom, std::int64_t bra_atom,
          const std::array<double, 3>& vector, double distance_squared, std::string&) {
        const bool publish_reverse = ket_atom != bra_atom;
        for_each_shell_pair(
            basis, ket_atom, bra_atom, [&](std::size_t bra_shell, std::size_t ket_shell) {
              common::compute_shell_pair_cpu(basis, bra_shell, ket_shell, vector.data(),
                                             integrals.integral_cutoff, false, true,
                                             integral_scratch);
              const MatrixBlock block =
                  matrix_block(basis, integrals, system, bra_shell, ket_shell);
              const std::size_t block_size = block.bra_count * block.ket_count;
              const H0ImageFactor h0_factor =
                  h0_image_factor(basis, h0, coordination_numbers, system, ket_atom, bra_atom,
                                  ket_shell, bra_shell, distance_squared);
              for (std::size_t bra_ao = 0; bra_ao < block.bra_count; ++bra_ao) {
                for (std::size_t ket_ao = 0; ket_ao < block.ket_count; ++ket_ao) {
                  const std::size_t block_index = bra_ao * block.ket_count + ket_ao;
                  const std::size_t forward = forward_matrix_index(block, bra_ao, ket_ao);
                  const double local_overlap = integral_scratch.spherical[block_index];
                  overlap_scratch[forward] += local_overlap;
                  h0_scratch[forward] += local_overlap * h0_factor.factor;

                  std::array<double, kDipoleComponents> local_dipole{};
                  for (std::size_t component = 0; component < kDipoleComponents; ++component) {
                    local_dipole[component] =
                        integral_scratch.spherical_multipole[component * block_size + block_index];
                    dipole_scratch[component * matrix_elements + forward] +=
                        local_dipole[component];
                  }
                  std::array<double, kQuadrupoleComponents> local_quadrupole{};
                  for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
                    local_quadrupole[component] =
                        integral_scratch
                            .spherical_multipole[(component + kDipoleComponents) * block_size +
                                                 block_index];
                    quadrupole_scratch[component * matrix_elements + forward] +=
                        local_quadrupole[component];
                  }

                  if (!publish_reverse) continue;
                  const std::size_t reverse = reverse_matrix_index(block, bra_ao, ket_ao);
                  overlap_scratch[reverse] += local_overlap;
                  h0_scratch[reverse] += local_overlap * h0_factor.factor;
                  std::array<double, kDipoleComponents> shifted_dipole{};
                  std::array<double, kQuadrupoleComponents> shifted_quadrupole{};
                  shifted_multipoles(vector, local_overlap, local_dipole, local_quadrupole,
                                     shifted_dipole, shifted_quadrupole);
                  for (std::size_t component = 0; component < kDipoleComponents; ++component) {
                    dipole_scratch[component * matrix_elements + reverse] +=
                        shifted_dipole[component];
                  }
                  for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
                    quadrupole_scratch[component * matrix_elements + reverse] +=
                        shifted_quadrupole[component];
                  }
                }
              }
            });
        return XTBLOOM_STATUS_SUCCESS;
      },
      [&](std::int64_t system, std::int64_t atom, std::string&) {
        const std::array<double, 3> zero{};
        for_each_shell_pair(basis, atom, atom, [&](std::size_t bra_shell, std::size_t ket_shell) {
          common::compute_shell_pair_cpu(basis, bra_shell, ket_shell, zero.data(),
                                         integrals.integral_cutoff, false, true, integral_scratch);
          const MatrixBlock block = matrix_block(basis, integrals, system, bra_shell, ket_shell);
          const std::size_t block_size = block.bra_count * block.ket_count;
          const double factor =
              h0_onsite_factor(h0, coordination_numbers, atom, ket_shell, bra_shell);
          for (std::size_t bra_ao = 0; bra_ao < block.bra_count; ++bra_ao) {
            for (std::size_t ket_ao = 0; ket_ao < block.ket_count; ++ket_ao) {
              const std::size_t block_index = bra_ao * block.ket_count + ket_ao;
              const std::size_t output = forward_matrix_index(block, bra_ao, ket_ao);
              const double local_overlap = integral_scratch.spherical[block_index];
              overlap_scratch[output] += local_overlap;
              h0_scratch[output] += local_overlap * factor;
              for (std::size_t component = 0; component < kDipoleComponents; ++component) {
                dipole_scratch[component * matrix_elements + output] +=
                    integral_scratch.spherical_multipole[component * block_size + block_index];
              }
              for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
                quadrupole_scratch[component * matrix_elements + output] +=
                    integral_scratch
                        .spherical_multipole[(component + kDipoleComponents) * block_size +
                                             block_index];
              }
            }
          }
        });
        return XTBLOOM_STATUS_SUCCESS;
      },
      error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  if (!finite_values(overlap_scratch, matrix_elements) ||
      !finite_values(dipole_scratch, dipole_elements) ||
      !finite_values(quadrupole_scratch, quadrupole_elements) ||
      !finite_values(h0_scratch, matrix_elements)) {
    error = "periodic integral or H0 evaluation overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::copy_n(overlap_scratch, matrix_elements, overlap);
  std::copy_n(dipole_scratch, dipole_elements, dipole);
  std::copy_n(quadrupole_scratch, quadrupole_elements, quadrupole);
  std::copy_n(h0_scratch, matrix_elements, hamiltonian);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_periodic_integrals_h0_vjp_cpu(
    const BasisPlan& basis, const IntegralPlan& integrals, const H0Plan& h0,
    const PeriodicIntegralPlan& periodic, const PeriodicShortRangePlan& topology,
    const PeriodicShortRangeGeometry& geometry,
    const PeriodicShortRangeWorkspace& geometry_workspace, const double* coordination_numbers,
    const double* dE_doverlap, const double* dE_ddipole, const double* dE_dquadrupole,
    const double* dE_dhamiltonian, double* dE_dcn, double* gradients, double* strain_derivatives,
    void* workspace, std::size_t workspace_size, std::string& error, CpuIsa cpu_isa) {
  xtbloom_status_t status =
      validate_context(basis, integrals, h0, periodic, topology, geometry, geometry_workspace,
                       coordination_numbers, workspace, workspace_size, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  const std::size_t matrix_elements = static_cast<std::size_t>(integrals.total_matrix_elements);
  const std::size_t atom_elements = static_cast<std::size_t>(basis.total_atoms);
  const std::size_t gradient_elements = atom_elements * 3u;
  const std::size_t strain_elements = static_cast<std::size_t>(basis.batch_size) * 9u;
  std::size_t dipole_elements = 0u;
  std::size_t quadrupole_elements = 0u;
  if (!checked_multiply(matrix_elements, kDipoleComponents, dipole_elements) ||
      !checked_multiply(matrix_elements, kQuadrupoleComponents, quadrupole_elements)) {
    error = "periodic integral derivative dimensions overflow";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  std::array<AddressRange, 8> numerical{};
  if (!make_double_range(coordination_numbers, atom_elements, numerical[0]) ||
      !make_double_range(dE_doverlap, matrix_elements, numerical[1]) ||
      !make_double_range(dE_ddipole, dipole_elements, numerical[2]) ||
      !make_double_range(dE_dquadrupole, quadrupole_elements, numerical[3]) ||
      !make_double_range(dE_dhamiltonian, matrix_elements, numerical[4]) ||
      !make_double_range(dE_dcn, atom_elements, numerical[5]) ||
      !make_double_range(gradients, gradient_elements, numerical[6]) ||
      !make_double_range(strain_derivatives, strain_elements, numerical[7]) ||
      !valid_buffer_set(basis, integrals, h0, periodic, topology, geometry_workspace, geometry,
                        workspace, workspace_size, numerical.data(), numerical.size()) ||
      !finite_values(dE_doverlap, matrix_elements) || !finite_values(dE_ddipole, dipole_elements) ||
      !finite_values(dE_dquadrupole, quadrupole_elements) ||
      !finite_values(dE_dhamiltonian, matrix_elements) || !finite_values(dE_dcn, atom_elements) ||
      !finite_values(gradients, gradient_elements) ||
      !finite_values(strain_derivatives, strain_elements)) {
    error = "periodic integral derivatives, accumulators, workspace, and plans are invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const PeriodicIntegralPlanData& data = *periodic.data();
  auto& integral_scratch =
      *offset_pointer<common::IntegralWorkspace>(workspace, data.integral_workspace_offset);
  double* coordination_scratch =
      offset_pointer<double>(workspace, data.coordination_scratch_offset);
  double* gradient_scratch = offset_pointer<double>(workspace, data.gradient_scratch_offset);
  double* strain_scratch = offset_pointer<double>(workspace, data.strain_scratch_offset);
  std::fill_n(coordination_scratch, atom_elements, 0.0);
  std::fill_n(gradient_scratch, gradient_elements, 0.0);
  std::fill_n(strain_scratch, strain_elements, 0.0);
  const common::IntegralKernelTable& kernels = common::integral_kernels_for_cpu_isa(cpu_isa);

  status = for_each_periodic_pair(
      basis, periodic, geometry,
      [&](std::int64_t system, std::int64_t ket_atom, std::int64_t bra_atom,
          const std::array<double, 3>& vector, double distance_squared, std::string&) {
        const bool publish_reverse = ket_atom != bra_atom;
        std::array<double, 3> pair_gradient{};
        for_each_shell_pair(
            basis, ket_atom, bra_atom, [&](std::size_t bra_shell, std::size_t ket_shell) {
              kernels.multipole_gradient_shell_pair(basis, bra_shell, ket_shell, vector.data(),
                                                    integrals.integral_cutoff, &integral_scratch);
              const MatrixBlock block =
                  matrix_block(basis, integrals, system, bra_shell, ket_shell);
              const std::size_t block_size = block.bra_count * block.ket_count;
              const H0ImageFactor factor =
                  h0_image_factor(basis, h0, coordination_numbers, system, ket_atom, bra_atom,
                                  ket_shell, bra_shell, distance_squared);
              double block_hamiltonian_weight = 0.0;

              for (std::size_t bra_ao = 0; bra_ao < block.bra_count; ++bra_ao) {
                for (std::size_t ket_ao = 0; ket_ao < block.ket_count; ++ket_ao) {
                  const std::size_t block_index = bra_ao * block.ket_count + ket_ao;
                  const std::size_t forward = forward_matrix_index(block, bra_ao, ket_ao);
                  const std::size_t reverse = reverse_matrix_index(block, bra_ao, ket_ao);
                  const double local_overlap = integral_scratch.spherical[block_index];
                  double overlap_adjoint =
                      dE_doverlap[forward] + dE_dhamiltonian[forward] * factor.factor;
                  block_hamiltonian_weight += dE_dhamiltonian[forward] * local_overlap;

                  std::array<double, kDipoleComponents> local_dipole{};
                  std::array<double, kDipoleComponents> dipole_adjoint{};
                  std::array<double, kDipoleComponents> reverse_dipole_adjoint{};
                  for (std::size_t component = 0; component < kDipoleComponents; ++component) {
                    local_dipole[component] =
                        integral_scratch.spherical_multipole[component * block_size + block_index];
                    dipole_adjoint[component] = dE_ddipole[component * matrix_elements + forward];
                    if (publish_reverse) {
                      reverse_dipole_adjoint[component] =
                          dE_ddipole[component * matrix_elements + reverse];
                    }
                  }

                  std::array<double, kQuadrupoleComponents> quadrupole_adjoint{};
                  std::array<double, kQuadrupoleComponents> reverse_quadrupole_adjoint{};
                  for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
                    quadrupole_adjoint[component] =
                        dE_dquadrupole[component * matrix_elements + forward];
                    if (publish_reverse) {
                      reverse_quadrupole_adjoint[component] =
                          dE_dquadrupole[component * matrix_elements + reverse];
                      quadrupole_adjoint[component] += reverse_quadrupole_adjoint[component];
                    }
                  }

                  std::array<double, 3> explicit_vector_adjoint{};
                  if (publish_reverse) {
                    overlap_adjoint +=
                        dE_doverlap[reverse] + dE_dhamiltonian[reverse] * factor.factor;
                    block_hamiltonian_weight += dE_dhamiltonian[reverse] * local_overlap;
                    add_multipole_shift_pullback(vector, local_overlap, local_dipole,
                                                 reverse_dipole_adjoint, reverse_quadrupole_adjoint,
                                                 overlap_adjoint, dipole_adjoint,
                                                 explicit_vector_adjoint);
                  }

                  for (std::size_t axis = 0; axis < 3u; ++axis) {
                    double derivative =
                        explicit_vector_adjoint[axis] +
                        overlap_adjoint *
                            integral_scratch.spherical_gradient[axis * block_size + block_index];
                    const double* multipole_gradient =
                        integral_scratch.spherical_multipole_gradient.data() +
                        axis * kMultipoleComponents * block_size;
                    for (std::size_t component = 0; component < kDipoleComponents; ++component) {
                      derivative += dipole_adjoint[component] *
                                    multipole_gradient[component * block_size + block_index];
                    }
                    for (std::size_t component = 0; component < kQuadrupoleComponents;
                         ++component) {
                      derivative +=
                          quadrupole_adjoint[component] *
                          multipole_gradient[(component + kDipoleComponents) * block_size +
                                             block_index];
                    }
                    pair_gradient[axis] += derivative;
                  }
                }
              }

              const double level_weight = 0.5 * block_hamiltonian_weight * factor.spatial_scale;
              coordination_scratch[static_cast<std::size_t>(ket_atom)] -=
                  h0.shell_coordination_scale[ket_shell] * level_weight;
              coordination_scratch[static_cast<std::size_t>(bra_atom)] -=
                  h0.shell_coordination_scale[bra_shell] * level_weight;
              const double radial_scale = block_hamiltonian_weight * factor.average_level *
                                          factor.spatial_scale_derivative / factor.distance;
              for (std::size_t axis = 0; axis < 3u; ++axis) {
                pair_gradient[axis] += radial_scale * vector[axis];
              }
            });

        if (ket_atom != bra_atom) {
          for (std::size_t axis = 0; axis < 3u; ++axis) {
            gradient_scratch[static_cast<std::size_t>(ket_atom) * 3u + axis] += pair_gradient[axis];
            gradient_scratch[static_cast<std::size_t>(bra_atom) * 3u + axis] -= pair_gradient[axis];
          }
        }
        double* system_strain = strain_scratch + static_cast<std::size_t>(system) * 9u;
        for (std::size_t row = 0; row < 3u; ++row) {
          for (std::size_t column = 0; column < 3u; ++column) {
            system_strain[row * 3u + column] += pair_gradient[row] * vector[column];
          }
        }
        return XTBLOOM_STATUS_SUCCESS;
      },
      [&](std::int64_t system, std::int64_t atom, std::string&) {
        const std::array<double, 3> zero{};
        for_each_shell_pair(basis, atom, atom, [&](std::size_t bra_shell, std::size_t ket_shell) {
          common::compute_shell_pair_cpu(basis, bra_shell, ket_shell, zero.data(),
                                         integrals.integral_cutoff, false, false, integral_scratch);
          const MatrixBlock block = matrix_block(basis, integrals, system, bra_shell, ket_shell);
          double block_hamiltonian_weight = 0.0;
          for (std::size_t bra_ao = 0; bra_ao < block.bra_count; ++bra_ao) {
            for (std::size_t ket_ao = 0; ket_ao < block.ket_count; ++ket_ao) {
              const std::size_t block_index = bra_ao * block.ket_count + ket_ao;
              const std::size_t output = forward_matrix_index(block, bra_ao, ket_ao);
              block_hamiltonian_weight +=
                  dE_dhamiltonian[output] * integral_scratch.spherical[block_index];
            }
          }
          const double level_weight = 0.5 * block_hamiltonian_weight;
          coordination_scratch[static_cast<std::size_t>(atom)] -=
              (h0.shell_coordination_scale[ket_shell] + h0.shell_coordination_scale[bra_shell]) *
              level_weight;
        });
        return XTBLOOM_STATUS_SUCCESS;
      },
      error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  if (!finite_values(coordination_scratch, atom_elements) ||
      !finite_values(gradient_scratch, gradient_elements) ||
      !finite_values(strain_scratch, strain_elements)) {
    error = "periodic integral or H0 derivative overflowed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  for (std::size_t index = 0; index < atom_elements; ++index) {
    if (!std::isfinite(dE_dcn[index] + coordination_scratch[index])) {
      error = "periodic H0 coordination derivative publication would overflow";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  for (std::size_t index = 0; index < gradient_elements; ++index) {
    if (!std::isfinite(gradients[index] + gradient_scratch[index])) {
      error = "periodic integral Cartesian derivative publication would overflow";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  for (std::size_t index = 0; index < strain_elements; ++index) {
    if (!std::isfinite(strain_derivatives[index] + strain_scratch[index])) {
      error = "periodic integral strain derivative publication would overflow";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  for (std::size_t index = 0; index < atom_elements; ++index) {
    dE_dcn[index] += coordination_scratch[index];
  }
  for (std::size_t index = 0; index < gradient_elements; ++index) {
    gradients[index] += gradient_scratch[index];
  }
  for (std::size_t index = 0; index < strain_elements; ++index) {
    strain_derivatives[index] += strain_scratch[index];
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn2
