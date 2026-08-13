// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/mulliken.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace xtbloom::detail::gfn1 {

struct MullikenPlanData final {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::int64_t matrix_elements = 0;
  std::int64_t density_elements = 0;
  std::int64_t shell_population_elements = 0;
  std::int64_t atom_population_elements = 0;
  std::int64_t population_scratch_elements = 0;
  std::int64_t hamiltonian_scratch_elements = 0;

  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> batch_orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> shell_orbital_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::int64_t> orbital_to_shell;
  std::vector<std::int32_t> spin_channels;
  std::vector<double> reference_shell_occupations;
  std::vector<std::int64_t> density_offsets;
  std::vector<std::int64_t> shell_population_offsets;
  std::vector<std::int64_t> atom_population_offsets;
};

namespace {

static_assert(std::is_trivially_copyable_v<MullikenIntegralView>);
static_assert(std::is_trivially_copyable_v<MullikenDensityView>);
static_assert(std::is_trivially_copyable_v<MullikenPopulationView>);
static_assert(std::is_trivially_copyable_v<MullikenPotentialView>);
static_assert(std::is_trivially_copyable_v<MullikenHamiltonianView>);
static_assert(std::is_trivially_copyable_v<MullikenWorkspace>);

const std::vector<std::int64_t> kEmptyInt64;
const std::vector<std::int32_t> kEmptyInt32;
const std::vector<double> kEmptyDouble;

bool checked_add(std::int64_t value, std::int64_t& total) noexcept {
  if (value < 0 || total > std::numeric_limits<std::int64_t>::max() - value) {
    return false;
  }
  total += value;
  return true;
}

bool bytes_for(std::int64_t elements, std::size_t& bytes) noexcept {
  if (elements < 0 ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(double))) {
    return false;
  }
  bytes = static_cast<std::size_t>(elements) * sizeof(double);
  return true;
}

bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                    std::size_t second_bytes) noexcept {
  if (first_bytes == 0u || second_bytes == 0u) return false;
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  return first_begin < second_begin + second_bytes && second_begin < first_begin + first_bytes;
}

struct MemoryRange {
  const void* data = nullptr;
  std::size_t size_bytes = 0u;
};

template <std::size_t N>
bool pairwise_disjoint(const std::array<MemoryRange, N>& ranges) noexcept {
  for (std::size_t first = 0u; first < N; ++first) {
    for (std::size_t second = first + 1u; second < N; ++second) {
      if (ranges_overlap(ranges[first].data, ranges[first].size_bytes, ranges[second].data,
                         ranges[second].size_bytes)) {
        return false;
      }
    }
  }
  return true;
}

std::array<MemoryRange, 15> plan_storage_ranges(const MullikenPlan& plan) {
  const MullikenPlanData* data = plan.identity();
  return {{{&plan, sizeof(plan)},
           {data, sizeof(MullikenPlanData)},
           {data->atom_offsets.data(), data->atom_offsets.capacity() * sizeof(std::int64_t)},
           {data->batch_shell_offsets.data(),
            data->batch_shell_offsets.capacity() * sizeof(std::int64_t)},
           {data->batch_orbital_offsets.data(),
            data->batch_orbital_offsets.capacity() * sizeof(std::int64_t)},
           {data->matrix_offsets.data(), data->matrix_offsets.capacity() * sizeof(std::int64_t)},
           {data->shell_orbital_offsets.data(),
            data->shell_orbital_offsets.capacity() * sizeof(std::int64_t)},
           {data->shell_to_atom.data(), data->shell_to_atom.capacity() * sizeof(std::int64_t)},
           {data->orbital_to_shell.data(),
            data->orbital_to_shell.capacity() * sizeof(std::int64_t)},
           {data->spin_channels.data(), data->spin_channels.capacity() * sizeof(std::int32_t)},
           {data->reference_shell_occupations.data(),
            data->reference_shell_occupations.capacity() * sizeof(double)},
           {data->density_offsets.data(),
            data->density_offsets.capacity() * sizeof(std::int64_t)},
           {data->shell_population_offsets.data(),
            data->shell_population_offsets.capacity() * sizeof(std::int64_t)},
           {data->atom_population_offsets.data(),
            data->atom_population_offsets.capacity() * sizeof(std::int64_t)},
           {nullptr, 0u}}};
}

bool overlaps_plan_storage(const MullikenPlan& plan, const MemoryRange& candidate) {
  for (const MemoryRange& range : plan_storage_ranges(plan)) {
    if (ranges_overlap(candidate.data, candidate.size_bytes, range.data, range.size_bytes)) {
      return true;
    }
  }
  return false;
}

bool finite_add_product(double first, double second, double& accumulator) noexcept {
  if (!std::isfinite(first) || !std::isfinite(second)) {
    return false;
  }
  const double updated = std::fma(first, second, accumulator);
  if (!std::isfinite(updated)) {
    return false;
  }
  accumulator = updated;
  return true;
}

bool valid_workspace(const MullikenWorkspace& workspace, std::int64_t required) noexcept {
  return required > 0 && workspace.scratch != nullptr && workspace.elements >= required &&
         reinterpret_cast<std::uintptr_t>(workspace.scratch) % alignof(double) == 0u;
}

bool valid_view_identity(const MullikenPlan& plan, const MullikenPlanData* identity) noexcept {
  return plan.sealed() && identity == plan.identity();
}

xtbloom_status_t validate_plan(const MullikenPlan& plan, std::string& error) {
  if (!plan.sealed()) {
    error = "GFN1 Mulliken plan is not sealed";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_common_views(const MullikenPlan& plan,
                                       const MullikenIntegralView& integrals,
                                       std::string& error) {
  if (!valid_view_identity(plan, integrals.plan_identity) || integrals.overlap == nullptr ||
      integrals.elements != plan.matrix_elements() ||
      reinterpret_cast<std::uintptr_t>(integrals.overlap) % alignof(double) != 0u) {
    error = "GFN1 Mulliken overlap view is not an exact binding of the plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

void population_system_unchecked(const MullikenPlanData& data, std::size_t system,
                                 const double* overlap, const double* density,
                                 double* qsh_scratch, double* qat_scratch,
                                 std::string& error, xtbloom_status_t& status) {
  const std::int64_t atom_begin = data.atom_offsets[system];
  const std::int64_t atom_end = data.atom_offsets[system + 1u];
  const std::int64_t shell_begin = data.batch_shell_offsets[system];
  const std::int64_t shell_end = data.batch_shell_offsets[system + 1u];
  const std::int64_t orbital_begin = data.batch_orbital_offsets[system];
  const std::int64_t orbital_end = data.batch_orbital_offsets[system + 1u];
  const std::int64_t atoms = atom_end - atom_begin;
  const std::int64_t shells = shell_end - shell_begin;
  const std::int64_t orbitals = orbital_end - orbital_begin;
  const std::int32_t nspin = data.spin_channels[system];
  const std::int64_t matrix_base = data.matrix_offsets[system];
  const std::int64_t density_base = data.density_offsets[system];
  const std::int64_t qsh_base = data.shell_population_offsets[system];
  const std::int64_t qat_base = data.atom_population_offsets[system];

  std::fill_n(qsh_scratch + qsh_base, static_cast<std::size_t>(nspin) * shells, 0.0);
  std::fill_n(qat_scratch + qat_base, static_cast<std::size_t>(nspin) * atoms, 0.0);

  for (std::int32_t spin = 0; spin < nspin; ++spin) {
    const std::int64_t spin_matrix = density_base + spin * orbitals * orbitals;
    for (std::int64_t ket = 0; ket < orbitals; ++ket) {
      const std::int64_t global_orbital = orbital_begin + ket;
      const std::int64_t local_shell =
          data.orbital_to_shell[static_cast<std::size_t>(global_orbital)] - shell_begin;
      double& shell_population = qsh_scratch[qsh_base + spin * shells + local_shell];
      for (std::int64_t bra = 0; bra < orbitals; ++bra) {
        const std::int64_t matrix = matrix_base + bra * orbitals + ket;
        const std::int64_t density_index = spin_matrix + bra * orbitals + ket;
        if (!finite_add_product(-density[density_index], overlap[matrix], shell_population)) {
          error = "GFN1 Mulliken population input is non-finite or contraction overflowed";
          status = XTBLOOM_STATUS_INTERNAL_ERROR;
          return;
        }
      }
    }
  }

  for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
    const std::size_t charge_index = static_cast<std::size_t>(qsh_base + local_shell);
    const double reference =
        data.reference_shell_occupations[static_cast<std::size_t>(shell_begin + local_shell)];
    if (nspin == 1) {
      const double charge = qsh_scratch[charge_index] + reference;
      if (!std::isfinite(charge)) {
        error = "GFN1 Mulliken reference-charge addition overflowed";
        status = XTBLOOM_STATUS_INTERNAL_ERROR;
        return;
      }
      qsh_scratch[charge_index] = charge;
    } else {
      const std::size_t magnetization_index =
          static_cast<std::size_t>(qsh_base + shells + local_shell);
      const double alpha_contraction = qsh_scratch[charge_index];
      const double beta_contraction = qsh_scratch[magnetization_index];
      const double charge = alpha_contraction + beta_contraction + reference;
      /* Since each contraction already carries the leading minus sign, this
       * alpha-minus-beta value equals the established N_beta-N_alpha shell
       * magnetization used by tblite/xTB. */
      const double magnetization = alpha_contraction - beta_contraction;
      if (!std::isfinite(charge) || !std::isfinite(magnetization)) {
        error = "GFN1 Mulliken spin conversion overflowed";
        status = XTBLOOM_STATUS_INTERNAL_ERROR;
        return;
      }
      qsh_scratch[charge_index] = charge;
      qsh_scratch[magnetization_index] = magnetization;
    }
  }

  for (std::int32_t channel = 0; channel < nspin; ++channel) {
    for (std::int64_t local_shell = 0; local_shell < shells; ++local_shell) {
      const std::int64_t shell = shell_begin + local_shell;
      const std::int64_t local_atom = data.shell_to_atom[static_cast<std::size_t>(shell)] - atom_begin;
      double& atom_population = qat_scratch[qat_base + channel * atoms + local_atom];
      const double updated = atom_population + qsh_scratch[qsh_base + channel * shells + local_shell];
      if (!std::isfinite(updated)) {
        error = "GFN1 Mulliken shell-to-atom reduction overflowed";
        status = XTBLOOM_STATUS_INTERNAL_ERROR;
        return;
      }
      atom_population = updated;
    }
  }
  status = XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_hamiltonian_system_unchecked(
    const MullikenPlanData& data, std::size_t system, const double* overlap,
    const double* potential, double* matrix, double* converted_potential,
    std::string& error) {
  const std::int64_t orbital_begin = data.batch_orbital_offsets[system];
  const std::int64_t orbital_end = data.batch_orbital_offsets[system + 1u];
  const std::int64_t shell_begin = data.batch_shell_offsets[system];
  const std::int64_t shell_end = data.batch_shell_offsets[system + 1u];
  const std::int64_t orbitals = orbital_end - orbital_begin;
  const std::int64_t shells = shell_end - shell_begin;
  const std::int32_t nspin = data.spin_channels[system];
  const std::int64_t overlap_begin = data.matrix_offsets[system];
  const std::int64_t matrix_begin = data.density_offsets[system];
  const std::int64_t qsh_begin = data.shell_population_offsets[system];

  if (nspin == 1) {
    for (std::int64_t shell = 0; shell < shells; ++shell) {
      const double value = potential[qsh_begin + shell];
      if (!std::isfinite(value)) {
        error = "GFN1 Mulliken shell potential contains NaN or infinity";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      converted_potential[qsh_begin + shell] = value;
    }
  } else {
    for (std::int64_t shell = 0; shell < shells; ++shell) {
      const double charge = potential[qsh_begin + shell];
      const double magnetization = potential[qsh_begin + shells + shell];
      /* Keep the primitive in tblite's charge/magnetization convention. The
       * SCC driver applies tblite's later unrestricted-Hamiltonian factor
       * without doubling the already full-scale H0 channel copies. */
      const double alpha = 0.5 * (charge + magnetization);
      const double beta = 0.5 * (charge - magnetization);
      if (!std::isfinite(alpha) || !std::isfinite(beta)) {
        error = "GFN1 Mulliken potential spin conversion overflowed";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      converted_potential[qsh_begin + shell] = alpha;
      converted_potential[qsh_begin + shells + shell] = beta;
    }
  }

  for (std::int32_t spin = 0; spin < nspin; ++spin) {
    const std::int64_t spin_matrix = matrix_begin + spin * orbitals * orbitals;
    const std::int64_t spin_potential = qsh_begin + spin * shells;
    for (std::int64_t row = 0; row < orbitals; ++row) {
      const std::int64_t row_shell =
          data.orbital_to_shell[static_cast<std::size_t>(orbital_begin + row)] - shell_begin;
      for (std::int64_t column = row; column < orbitals; ++column) {
        const std::int64_t column_shell =
            data.orbital_to_shell[static_cast<std::size_t>(orbital_begin + column)] - shell_begin;
        const std::int64_t overlap_index = overlap_begin + row * orbitals + column;
        const double overlap_value = overlap[overlap_index];
        const double row_potential = converted_potential[spin_potential + row_shell];
        const double column_potential = converted_potential[spin_potential + column_shell];
        const double shift = -0.5 * overlap_value * (row_potential + column_potential);
        if (!std::isfinite(overlap_value) || !std::isfinite(shift)) {
          error = "GFN1 Mulliken scalar Hamiltonian assembly contains invalid data or overflowed";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        const std::int64_t forward = spin_matrix + row * orbitals + column;
        const std::int64_t reverse = spin_matrix + column * orbitals + row;
        const double updated = matrix[forward] + shift;
        if (!std::isfinite(matrix[forward]) || !std::isfinite(updated)) {
          error = "GFN1 Mulliken Hamiltonian accumulation contains invalid data or overflowed";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        matrix[forward] = updated;
        if (reverse != forward) matrix[reverse] = updated;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_population_call(const MullikenPlan& plan,
                                          const MullikenIntegralView& integrals,
                                          const MullikenDensityView& density,
                                          const MullikenPopulationView& population,
                                          const MullikenWorkspace& workspace,
                                          std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = validate_common_views(plan, integrals, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (!valid_view_identity(plan, density.plan_identity) || density.density == nullptr ||
      density.elements != plan.density_elements() ||
      !valid_view_identity(plan, population.plan_identity) || population.qsh == nullptr ||
      population.qat == nullptr || population.qsh_elements != plan.shell_population_elements() ||
      population.qat_elements != plan.atom_population_elements() ||
      !valid_workspace(workspace, plan.population_scratch_elements())) {
    error = "GFN1 Mulliken population views or workspace are not exact plan bindings";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::size_t matrix_bytes = 0u;
  std::size_t density_bytes = 0u;
  std::size_t qsh_bytes = 0u;
  std::size_t qat_bytes = 0u;
  std::size_t scratch_bytes = 0u;
  const std::array<MemoryRange, 6> descriptors{{{&integrals, sizeof(integrals)},
                                                {&density, sizeof(density)},
                                                {&population, sizeof(population)},
                                                {&workspace, sizeof(workspace)},
                                                {&error, sizeof(error)},
                                                {&plan, sizeof(plan)}}};
  if (!bytes_for(plan.matrix_elements(), matrix_bytes) ||
      !bytes_for(plan.density_elements(), density_bytes) ||
      !bytes_for(plan.shell_population_elements(), qsh_bytes) ||
      !bytes_for(plan.atom_population_elements(), qat_bytes) ||
      !bytes_for(workspace.elements, scratch_bytes)) {
    error = "GFN1 Mulliken population extents are not representable";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::array<MemoryRange, 5> numerical{{{integrals.overlap, matrix_bytes},
                                              {density.density, density_bytes},
                                              {population.qsh, qsh_bytes},
                                              {population.qat, qat_bytes},
                                              {workspace.scratch, scratch_bytes}}};
  if (!pairwise_disjoint(numerical)) {
    error = "GFN1 Mulliken population inputs, outputs, and scratch must be disjoint";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (const MemoryRange& range : numerical) {
    if (overlaps_plan_storage(plan, range)) {
      error = "GFN1 Mulliken population buffers must not overlap plan storage";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (const MemoryRange& descriptor : descriptors) {
      if (ranges_overlap(range.data, range.size_bytes, descriptor.data, descriptor.size_bytes)) {
        error = "GFN1 Mulliken population buffers must not overlap descriptors or diagnostics";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_hamiltonian_call(const MullikenPlan& plan,
                                           const MullikenIntegralView& integrals,
                                           const MullikenPotentialView& potential,
                                           const MullikenHamiltonianView& hamiltonian,
                                           const MullikenWorkspace& workspace,
                                           std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = validate_common_views(plan, integrals, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (!valid_view_identity(plan, potential.plan_identity) || potential.vsh == nullptr ||
      potential.elements != plan.shell_population_elements() ||
      !valid_view_identity(plan, hamiltonian.plan_identity) || hamiltonian.matrix == nullptr ||
      hamiltonian.elements != plan.density_elements() ||
      !valid_workspace(workspace, plan.hamiltonian_scratch_elements())) {
    error = "GFN1 Mulliken Hamiltonian views or workspace are not exact plan bindings";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::size_t matrix_bytes = 0u;
  std::size_t potential_bytes = 0u;
  std::size_t hamiltonian_bytes = 0u;
  std::size_t scratch_bytes = 0u;
  if (!bytes_for(plan.matrix_elements(), matrix_bytes) ||
      !bytes_for(plan.shell_population_elements(), potential_bytes) ||
      !bytes_for(plan.density_elements(), hamiltonian_bytes) ||
      !bytes_for(workspace.elements, scratch_bytes)) {
    error = "GFN1 Mulliken Hamiltonian extents are not representable";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::array<MemoryRange, 4> numerical{{{integrals.overlap, matrix_bytes},
                                              {potential.vsh, potential_bytes},
                                              {hamiltonian.matrix, hamiltonian_bytes},
                                              {workspace.scratch, scratch_bytes}}};
  if (!pairwise_disjoint(numerical)) {
    error = "GFN1 Mulliken Hamiltonian inputs, output, and scratch must be disjoint";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::array<MemoryRange, 6> descriptors{{{&integrals, sizeof(integrals)},
                                                {&potential, sizeof(potential)},
                                                {&hamiltonian, sizeof(hamiltonian)},
                                                {&workspace, sizeof(workspace)},
                                                {&error, sizeof(error)},
                                                {&plan, sizeof(plan)}}};
  for (const MemoryRange& range : numerical) {
    if (overlaps_plan_storage(plan, range)) {
      error = "GFN1 Mulliken Hamiltonian buffers must not overlap plan storage";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (const MemoryRange& descriptor : descriptors) {
      if (ranges_overlap(range.data, range.size_bytes, descriptor.data, descriptor.size_bytes)) {
        error = "GFN1 Mulliken Hamiltonian buffers must not overlap descriptors or diagnostics";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

MullikenPlan::MullikenPlan(std::shared_ptr<const MullikenPlanData> data) noexcept
    : data_(std::move(data)) {}

bool MullikenPlan::sealed() const noexcept { return data_ != nullptr; }
std::int64_t MullikenPlan::batch_size() const noexcept { return data_ ? data_->batch_size : 0; }
std::int64_t MullikenPlan::total_atoms() const noexcept { return data_ ? data_->total_atoms : 0; }
std::int64_t MullikenPlan::total_shells() const noexcept { return data_ ? data_->total_shells : 0; }
std::int64_t MullikenPlan::total_orbitals() const noexcept { return data_ ? data_->total_orbitals : 0; }
std::int64_t MullikenPlan::matrix_elements() const noexcept { return data_ ? data_->matrix_elements : 0; }
std::int64_t MullikenPlan::density_elements() const noexcept { return data_ ? data_->density_elements : 0; }
std::int64_t MullikenPlan::shell_population_elements() const noexcept { return data_ ? data_->shell_population_elements : 0; }
std::int64_t MullikenPlan::atom_population_elements() const noexcept { return data_ ? data_->atom_population_elements : 0; }
std::int64_t MullikenPlan::population_scratch_elements() const noexcept { return data_ ? data_->population_scratch_elements : 0; }
std::int64_t MullikenPlan::hamiltonian_scratch_elements() const noexcept { return data_ ? data_->hamiltonian_scratch_elements : 0; }

std::size_t MullikenPlan::resident_bytes() const noexcept {
  if (!data_) return 0u;
  const auto bytes = [](const auto& values) { return values.capacity() * sizeof(values[0]); };
  return sizeof(MullikenPlanData) + bytes(data_->atom_offsets) + bytes(data_->batch_shell_offsets) +
         bytes(data_->batch_orbital_offsets) + bytes(data_->matrix_offsets) +
         bytes(data_->shell_orbital_offsets) + bytes(data_->shell_to_atom) +
         bytes(data_->orbital_to_shell) + bytes(data_->spin_channels) +
         bytes(data_->reference_shell_occupations) + bytes(data_->density_offsets) +
         bytes(data_->shell_population_offsets) + bytes(data_->atom_population_offsets);
}

bool MullikenPlan::overlaps_storage(const void* data, std::size_t size_bytes) const noexcept {
  if (!data_ || size_bytes == 0u) return false;
  return overlaps_plan_storage(*this, MemoryRange{data, size_bytes});
}

const std::vector<std::int64_t>& MullikenPlan::atom_offsets() const noexcept { return data_ ? data_->atom_offsets : kEmptyInt64; }
const std::vector<std::int64_t>& MullikenPlan::batch_shell_offsets() const noexcept { return data_ ? data_->batch_shell_offsets : kEmptyInt64; }
const std::vector<std::int64_t>& MullikenPlan::batch_orbital_offsets() const noexcept { return data_ ? data_->batch_orbital_offsets : kEmptyInt64; }
const std::vector<std::int64_t>& MullikenPlan::matrix_offsets() const noexcept { return data_ ? data_->matrix_offsets : kEmptyInt64; }
const std::vector<std::int64_t>& MullikenPlan::shell_orbital_offsets() const noexcept { return data_ ? data_->shell_orbital_offsets : kEmptyInt64; }
const std::vector<std::int64_t>& MullikenPlan::shell_to_atom() const noexcept { return data_ ? data_->shell_to_atom : kEmptyInt64; }
const std::vector<std::int64_t>& MullikenPlan::orbital_to_shell() const noexcept { return data_ ? data_->orbital_to_shell : kEmptyInt64; }
const std::vector<std::int32_t>& MullikenPlan::spin_channels() const noexcept { return data_ ? data_->spin_channels : kEmptyInt32; }
const std::vector<double>& MullikenPlan::reference_shell_occupations() const noexcept { return data_ ? data_->reference_shell_occupations : kEmptyDouble; }
const MullikenPlanData* MullikenPlan::identity() const noexcept { return data_.get(); }

xtbloom_status_t make_mulliken_plan(const BasisPlan& basis, const IntegralPlan& integrals,
                                    const WavefunctionLayout& wavefunction, MullikenPlan& plan,
                                    std::string& error) {
  const bool topology_valid =
      basis.batch_size > 0 && basis.total_atoms > 0 && basis.total_shells > 0 &&
      basis.total_orbitals > 0 && integrals.batch_size == basis.batch_size &&
      wavefunction.batch_size == basis.batch_size && wavefunction.total_atoms == basis.total_atoms &&
      wavefunction.total_shells == basis.total_shells &&
      wavefunction.total_orbitals == basis.total_orbitals &&
      basis.atom_offsets == wavefunction.atom_offsets &&
      basis.batch_shell_offsets == wavefunction.batch_shell_offsets &&
      basis.batch_orbital_offsets == wavefunction.batch_orbital_offsets &&
      basis.batch_size + 1 == static_cast<std::int64_t>(integrals.matrix_offsets.size()) &&
      wavefunction.reference_shell_occupations.size() ==
          static_cast<std::size_t>(basis.total_shells) &&
      wavefunction.spin_channels.size() == static_cast<std::size_t>(basis.batch_size);
  if (!topology_valid) {
    error = "GFN1 Mulliken inputs do not describe one exact ragged topology";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    MullikenPlanData created;
    created.batch_size = basis.batch_size;
    created.total_atoms = basis.total_atoms;
    created.total_shells = basis.total_shells;
    created.total_orbitals = basis.total_orbitals;
    created.matrix_elements = integrals.total_matrix_elements;
    created.density_elements = wavefunction.density.element_count;
    created.shell_population_elements = wavefunction.qsh.element_count;
    created.atom_population_elements = wavefunction.qat.element_count;
    created.atom_offsets = basis.atom_offsets;
    created.batch_shell_offsets = basis.batch_shell_offsets;
    created.batch_orbital_offsets = basis.batch_orbital_offsets;
    created.matrix_offsets = integrals.matrix_offsets;
    created.shell_orbital_offsets = basis.shell_orbital_offsets;
    created.shell_to_atom = basis.shell_to_atom;
    created.spin_channels = wavefunction.spin_channels;
    created.reference_shell_occupations = wavefunction.reference_shell_occupations;
    created.density_offsets = wavefunction.density.system_offsets;
    created.shell_population_offsets = wavefunction.qsh.system_offsets;
    created.atom_population_offsets = wavefunction.qat.system_offsets;
    created.orbital_to_shell.resize(static_cast<std::size_t>(basis.total_orbitals));
    for (std::int64_t shell = 0; shell < basis.total_shells; ++shell) {
      const std::int64_t begin = basis.shell_orbital_offsets[static_cast<std::size_t>(shell)];
      const std::int64_t end = basis.shell_orbital_offsets[static_cast<std::size_t>(shell + 1)];
      if (begin < 0 || begin >= end || end > basis.total_orbitals) {
        error = "GFN1 Mulliken basis has an invalid shell orbital partition";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      std::fill(created.orbital_to_shell.begin() + begin,
                created.orbital_to_shell.begin() + end, shell);
    }
    created.population_scratch_elements = created.shell_population_elements;
    if (!checked_add(created.atom_population_elements, created.population_scratch_elements)) {
      error = "GFN1 Mulliken population workspace size overflowed";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    created.hamiltonian_scratch_elements = created.density_elements;
    if (!checked_add(created.shell_population_elements, created.hamiltonian_scratch_elements)) {
      error = "GFN1 Mulliken Hamiltonian workspace size overflowed";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    auto sealed = std::make_shared<const MullikenPlanData>(std::move(created));
    plan = MullikenPlan(std::move(sealed));
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 Mulliken plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t evaluate_mulliken_population_system_cpu(
    const MullikenPlan& plan, const MullikenIntegralView& integrals,
    const MullikenDensityView& density, const MullikenPopulationView& population,
    std::int64_t system, const MullikenWorkspace& workspace, std::string& error) {
  xtbloom_status_t status =
      validate_population_call(plan, integrals, density, population, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (system < 0 || system >= plan.batch_size()) {
    error = "GFN1 Mulliken target system is out of range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const MullikenPlanData& data = *plan.identity();
  double* qsh_scratch = workspace.scratch;
  double* qat_scratch = qsh_scratch + data.shell_population_elements;
  population_system_unchecked(data, static_cast<std::size_t>(system), integrals.overlap,
                              density.density, qsh_scratch, qat_scratch, error, status);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const std::size_t index = static_cast<std::size_t>(system);
  const std::int64_t qsh_begin = data.shell_population_offsets[index];
  const std::int64_t qsh_end = data.shell_population_offsets[index + 1u];
  const std::int64_t qat_begin = data.atom_population_offsets[index];
  const std::int64_t qat_end = data.atom_population_offsets[index + 1u];
  std::copy(qsh_scratch + qsh_begin, qsh_scratch + qsh_end, population.qsh + qsh_begin);
  std::copy(qat_scratch + qat_begin, qat_scratch + qat_end, population.qat + qat_begin);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t evaluate_mulliken_population_cpu(const MullikenPlan& plan,
                                                  const MullikenIntegralView& integrals,
                                                  const MullikenDensityView& density,
                                                  const MullikenPopulationView& population,
                                                  const MullikenWorkspace& workspace,
                                                  std::string& error) {
  xtbloom_status_t status =
      validate_population_call(plan, integrals, density, population, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  const MullikenPlanData& data = *plan.identity();
  double* qsh_scratch = workspace.scratch;
  double* qat_scratch = qsh_scratch + data.shell_population_elements;
  for (std::size_t system = 0; system < static_cast<std::size_t>(data.batch_size); ++system) {
    population_system_unchecked(data, system, integrals.overlap, density.density, qsh_scratch,
                                qat_scratch, error, status);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  std::copy_n(qsh_scratch, static_cast<std::size_t>(data.shell_population_elements), population.qsh);
  std::copy_n(qat_scratch, static_cast<std::size_t>(data.atom_population_elements), population.qat);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_mulliken_hamiltonian_system_cpu(
    const MullikenPlan& plan, const MullikenIntegralView& integrals,
    const MullikenPotentialView& potential, const MullikenHamiltonianView& hamiltonian,
    std::int64_t system, const MullikenWorkspace& workspace, std::string& error) {
  xtbloom_status_t status =
      validate_hamiltonian_call(plan, integrals, potential, hamiltonian, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (system < 0 || system >= plan.batch_size()) {
    error = "GFN1 Mulliken target system is out of range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const MullikenPlanData& data = *plan.identity();
  const std::size_t index = static_cast<std::size_t>(system);
  const std::int64_t hamiltonian_begin = data.density_offsets[index];
  const std::int64_t hamiltonian_end = data.density_offsets[index + 1u];
  double* matrix_scratch = workspace.scratch;
  double* potential_scratch = matrix_scratch + data.density_elements;

  std::copy(hamiltonian.matrix + hamiltonian_begin, hamiltonian.matrix + hamiltonian_end,
            matrix_scratch + hamiltonian_begin);
  status = add_hamiltonian_system_unchecked(data, index, integrals.overlap, potential.vsh,
                                            matrix_scratch, potential_scratch, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  std::copy_n(matrix_scratch + hamiltonian_begin,
              static_cast<std::size_t>(hamiltonian_end - hamiltonian_begin),
              hamiltonian.matrix + hamiltonian_begin);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_mulliken_hamiltonian_cpu(const MullikenPlan& plan,
                                              const MullikenIntegralView& integrals,
                                              const MullikenPotentialView& potential,
                                              const MullikenHamiltonianView& hamiltonian,
                                              const MullikenWorkspace& workspace,
                                              std::string& error) {
  xtbloom_status_t status =
      validate_hamiltonian_call(plan, integrals, potential, hamiltonian, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  /* Stage the complete batch and publish only after every peer succeeds. */
  std::copy_n(hamiltonian.matrix, static_cast<std::size_t>(hamiltonian.elements),
              workspace.scratch);
  for (std::int64_t system = 0; system < plan.batch_size(); ++system) {
    status = add_hamiltonian_system_unchecked(
        *plan.identity(), static_cast<std::size_t>(system), integrals.overlap, potential.vsh,
        workspace.scratch, workspace.scratch + plan.density_elements(), error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  std::copy_n(workspace.scratch, static_cast<std::size_t>(hamiltonian.elements),
              hamiltonian.matrix);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
