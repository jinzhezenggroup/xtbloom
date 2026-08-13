// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/spin.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "data/parameters/gfn1.hpp"
#include "data/parameters/tblite_spin.hpp"

namespace xtbloom::detail::gfn1 {
namespace {

bool representable(std::int64_t value) {
  return value >= 0 && static_cast<std::uint64_t>(value) <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
}

bool checked_add(std::int64_t value, std::int64_t& total) {
  if (value < 0 || total > std::numeric_limits<std::int64_t>::max() - value) {
    return false;
  }
  total += value;
  return true;
}

bool count_bytes(std::int64_t count, std::size_t element_size, std::size_t& bytes) {
  if (!representable(count) ||
      static_cast<std::uint64_t>(count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                    std::size_t second_bytes) {
  if (first_bytes == 0u || second_bytes == 0u) {
    return false;
  }
  const std::uintptr_t first_begin = reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  return first_begin < second_begin + second_bytes && second_begin < first_begin + first_bytes;
}

std::size_t coupling_index(std::uint8_t first, std::uint8_t second) {
  if (first > second) {
    std::swap(first, second);
  }
  if (first == 0u) {
    return second == 0u ? 0u : (second == 1u ? 1u : 3u);
  }
  if (first == 1u) {
    return second == 1u ? 2u : 4u;
  }
  return 5u;
}

xtbloom_status_t validate_view(SpinPolarizationView view, std::string& error) {
  if (view.batch_size <= 0 || view.total_atoms <= 0 || view.total_shells <= 0 ||
      view.shell_population_elements <= 0 || !representable(view.batch_size) ||
      !representable(view.total_atoms) || !representable(view.total_shells) ||
      !representable(view.shell_population_elements) ||
      view.batch_size == std::numeric_limits<std::int64_t>::max() ||
      view.total_atoms == std::numeric_limits<std::int64_t>::max() ||
      view.atom_offset_count != view.batch_size + 1 ||
      view.batch_shell_offset_count != view.batch_size + 1 ||
      view.atom_shell_offset_count != view.total_atoms + 1 ||
      view.shell_population_offset_count != view.batch_size + 1 ||
      view.spin_channel_count != view.batch_size ||
      view.coupling_offset_count != view.total_atoms + 1 || view.coupling_matrix_count <= 0 ||
      view.atom_offsets == nullptr || view.batch_shell_offsets == nullptr ||
      view.atom_shell_offsets == nullptr || view.shell_population_offsets == nullptr ||
      view.spin_channels == nullptr || view.coupling_offsets == nullptr ||
      view.coupling_matrices == nullptr) {
    error = "GFN1 spin-polarization view is incomplete or has unrepresentable dimensions";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (view.atom_offsets[0] != 0 || view.atom_offsets[view.batch_size] != view.total_atoms ||
      view.batch_shell_offsets[0] != 0 ||
      view.batch_shell_offsets[view.batch_size] != view.total_shells ||
      view.atom_shell_offsets[0] != 0 ||
      view.atom_shell_offsets[view.total_atoms] != view.total_shells ||
      view.shell_population_offsets[0] != 0 ||
      view.shell_population_offsets[view.batch_size] != view.shell_population_elements ||
      view.coupling_offsets[0] != 0 ||
      view.coupling_offsets[view.total_atoms] != view.coupling_matrix_count) {
    error = "GFN1 spin-polarization offsets do not span their packed fields";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    const std::int64_t atom_begin = view.atom_offsets[system];
    const std::int64_t atom_end = view.atom_offsets[system + 1];
    const std::int64_t shell_begin = view.batch_shell_offsets[system];
    const std::int64_t shell_end = view.batch_shell_offsets[system + 1];
    const std::int64_t population_begin = view.shell_population_offsets[system];
    const std::int64_t population_end = view.shell_population_offsets[system + 1];
    const std::int32_t channels = view.spin_channels[system];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > view.total_atoms ||
        shell_begin < 0 || shell_begin > shell_end || shell_end > view.total_shells ||
        population_begin < 0 || population_begin > population_end ||
        population_end > view.shell_population_elements || (channels != 1 && channels != 2) ||
        population_end - population_begin != (shell_end - shell_begin) * channels ||
        view.atom_shell_offsets[atom_begin] != shell_begin ||
        view.atom_shell_offsets[atom_end] != shell_end) {
      error = "GFN1 spin-polarization view has an invalid ragged system partition";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t atom = 0; atom < view.total_atoms; ++atom) {
    const std::int64_t shell_begin = view.atom_shell_offsets[atom];
    const std::int64_t shell_end = view.atom_shell_offsets[atom + 1];
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t matrix_begin = view.coupling_offsets[atom];
    const std::int64_t matrix_end = view.coupling_offsets[atom + 1];
    if (shell_begin < 0 || shell_begin >= shell_end || shell_end > view.total_shells || shells > 3 ||
        matrix_begin < 0 || matrix_begin > matrix_end || matrix_end > view.coupling_matrix_count ||
        matrix_end - matrix_begin != shells * shells) {
      error = "GFN1 spin-polarization view has an invalid atom-local coupling partition";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t matrix = matrix_begin; matrix < matrix_end; ++matrix) {
      if (!std::isfinite(view.coupling_matrices[matrix])) {
        error = "GFN1 spin-polarization view contains a non-finite coupling";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

bool evaluate_unrestricted_system(SpinPolarizationView view, std::int64_t system,
                                  const double* shell_populations, double* potentials,
                                  double& energy) {
  const std::int64_t atom_begin = view.atom_offsets[system];
  const std::int64_t atom_end = view.atom_offsets[system + 1];
  const std::int64_t system_shell_begin = view.batch_shell_offsets[system];
  const std::int64_t system_shell_end = view.batch_shell_offsets[system + 1];
  const std::int64_t system_shells = system_shell_end - system_shell_begin;
  const std::int64_t magnetization_base = view.shell_population_offsets[system] + system_shells;
  energy = 0.0;
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    const std::int64_t shell_begin = view.atom_shell_offsets[atom];
    const std::int64_t shell_end = view.atom_shell_offsets[atom + 1];
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t matrix_begin = view.coupling_offsets[atom];
    for (std::int64_t row = 0; row < shells; ++row) {
      double potential = 0.0;
      for (std::int64_t column = 0; column < shells; ++column) {
        const std::int64_t population =
            magnetization_base + shell_begin - system_shell_begin + column;
        potential = std::fma(view.coupling_matrices[matrix_begin + row * shells + column],
                             shell_populations[population], potential);
      }
      if (!std::isfinite(potential)) {
        return false;
      }
      const std::int64_t population =
          magnetization_base + shell_begin - system_shell_begin + row;
      energy = std::fma(0.5 * shell_populations[population], potential, energy);
      if (!std::isfinite(energy)) {
        return false;
      }
      if (potentials != nullptr) {
        potentials[population] = potential;
      }
    }
  }
  return true;
}

}  // namespace

xtbloom_status_t make_spin_population_layout(const BasisPlan& basis,
                                             const std::int32_t* spin_channels,
                                             SpinPopulationLayout& layout, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_shells <= 0 || spin_channels == nullptr ||
      !representable(basis.batch_size) || !representable(basis.total_shells) ||
      basis.batch_shell_offsets.size() != static_cast<std::size_t>(basis.batch_size) + 1u ||
      basis.batch_shell_offsets.front() != 0 ||
      basis.batch_shell_offsets.back() != basis.total_shells) {
    error = "GFN1 spin population layout requires one complete basis and channel list";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  try {
    SpinPopulationLayout created;
    created.batch_size = basis.batch_size;
    created.total_shells = basis.total_shells;
    created.system_offsets.resize(static_cast<std::size_t>(basis.batch_size) + 1u, 0);
    created.spin_channels.assign(spin_channels, spin_channels + basis.batch_size);
    std::int64_t elements = 0;
    for (std::int64_t system = 0; system < basis.batch_size; ++system) {
      const std::int64_t shell_begin = basis.batch_shell_offsets[static_cast<std::size_t>(system)];
      const std::int64_t shell_end =
          basis.batch_shell_offsets[static_cast<std::size_t>(system) + 1u];
      const std::int32_t channels = spin_channels[system];
      const std::int64_t shells = shell_end - shell_begin;
      if (shell_begin < 0 || shell_begin > shell_end || shell_end > basis.total_shells ||
          (channels != 1 && channels != 2) ||
          (shells != 0 && shells > std::numeric_limits<std::int64_t>::max() / channels) ||
          !checked_add(shells * channels, elements)) {
        error = "GFN1 spin population layout has an invalid ragged system partition";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      created.system_offsets[static_cast<std::size_t>(system) + 1u] = elements;
    }
    created.element_count = elements;
    layout = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 spin population layout";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN1 spin population dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t make_spin_polarization_plan(const BasisPlan& basis,
                                             const std::int32_t* atomic_numbers,
                                             const SpinPopulationLayout& populations,
                                             SpinPolarizationPlan& plan, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      atomic_numbers == nullptr || populations.batch_size != basis.batch_size ||
      populations.total_shells != basis.total_shells || populations.element_count <= 0 ||
      basis.atom_offsets.size() != static_cast<std::size_t>(basis.batch_size) + 1u ||
      basis.batch_shell_offsets.size() != static_cast<std::size_t>(basis.batch_size) + 1u ||
      basis.atom_shell_offsets.size() != static_cast<std::size_t>(basis.total_atoms) + 1u ||
      basis.shell_to_atom.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.principal_quantum_numbers.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.angular_momenta.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.slater_exponents.size() != static_cast<std::size_t>(basis.total_shells) ||
      populations.system_offsets.size() != static_cast<std::size_t>(basis.batch_size) + 1u ||
      populations.spin_channels.size() != static_cast<std::size_t>(basis.batch_size)) {
    error = "GFN1 spin plan requires one complete matching basis, element list, and population layout";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  try {
    SpinPolarizationPlan created;
    created.batch_size = basis.batch_size;
    created.total_atoms = basis.total_atoms;
    created.total_shells = basis.total_shells;
    created.shell_population_elements = populations.element_count;
    created.atom_offsets = basis.atom_offsets;
    created.batch_shell_offsets = basis.batch_shell_offsets;
    created.atom_shell_offsets = basis.atom_shell_offsets;
    created.shell_population_offsets = populations.system_offsets;
    created.spin_channels = populations.spin_channels;
    created.coupling_offsets.resize(static_cast<std::size_t>(basis.total_atoms) + 1u, 0);

    std::int64_t coupling_count = 0;
    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::int64_t shell_begin = basis.atom_shell_offsets[static_cast<std::size_t>(atom)];
      const std::int64_t shell_end = basis.atom_shell_offsets[static_cast<std::size_t>(atom) + 1u];
      const std::int64_t shells = shell_end - shell_begin;
      if (shell_begin < 0 || shell_begin >= shell_end || shell_end > basis.total_shells ||
          shells > 3 || shells > std::numeric_limits<std::int64_t>::max() / shells ||
          !checked_add(shells * shells, coupling_count)) {
        error = "GFN1 spin basis has an unsupported atom-local shell partition";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      created.coupling_offsets[static_cast<std::size_t>(atom) + 1u] = coupling_count;
    }
    if (!representable(coupling_count)) {
      error = "GFN1 spin coupling dimensions exceed host container limits";
      return XTBLOOM_STATUS_ALLOCATION_FAILED;
    }
    created.coupling_matrices.resize(static_cast<std::size_t>(coupling_count));
    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::int32_t atomic_number = atomic_numbers[static_cast<std::size_t>(atom)];
      if (atomic_number <= 0 ||
          static_cast<std::size_t>(atomic_number) > parameters::tblite::kSpinConstants.size()) {
        error = "GFN1 spin plan contains an unsupported element";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const std::int64_t shell_begin = basis.atom_shell_offsets[static_cast<std::size_t>(atom)];
      const std::int64_t shell_end = basis.atom_shell_offsets[static_cast<std::size_t>(atom) + 1u];
      const std::int64_t shells = shell_end - shell_begin;
      const auto* element =
          parameters::gfn1::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || shells != element->shell_count) {
        error = "GFN1 spin element list does not match the basis shell layout";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        const std::size_t shell_index = static_cast<std::size_t>(shell);
        const auto& parameter = parameters::gfn1::kShells[
            element->shell_offset + static_cast<std::size_t>(shell - shell_begin)];
        if (basis.shell_to_atom[shell_index] != atom ||
            basis.principal_quantum_numbers[shell_index] != parameter.principal_quantum_number ||
            basis.angular_momenta[shell_index] != parameter.angular_momentum ||
            basis.slater_exponents[shell_index] != parameter.slater) {
          error = "GFN1 spin element list does not match the basis shell metadata";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }
      const std::int64_t matrix_begin = created.coupling_offsets[static_cast<std::size_t>(atom)];
      for (std::int64_t row = 0; row < shells; ++row) {
        const std::uint8_t row_l = basis.angular_momenta[static_cast<std::size_t>(shell_begin + row)];
        if (row_l > 2u) {
          error = "GFN1 spin plan supports only s, p, and d shells";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        for (std::int64_t column = 0; column < shells; ++column) {
          const std::uint8_t column_l =
              basis.angular_momenta[static_cast<std::size_t>(shell_begin + column)];
          if (column_l > 2u) {
            error = "GFN1 spin plan supports only s, p, and d shells";
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
          created.coupling_matrices[static_cast<std::size_t>(matrix_begin + row * shells + column)] =
              parameters::tblite::kSpinConstants[static_cast<std::size_t>(atomic_number - 1)]
                                                [coupling_index(row_l, column_l)];
        }
      }
    }
    SpinPolarizationView view = make_spin_polarization_view(created);
    if (validate_view(view, error) != XTBLOOM_STATUS_SUCCESS) {
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    plan = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 spin-polarization plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN1 spin-polarization dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

SpinPolarizationView make_spin_polarization_view(const SpinPolarizationPlan& plan) noexcept {
  const auto count = [](std::size_t value) noexcept {
    return value <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())
               ? static_cast<std::int64_t>(value)
               : std::int64_t{-1};
  };
  return SpinPolarizationView{
      plan.batch_size,
      plan.total_atoms,
      plan.total_shells,
      plan.shell_population_elements,
      count(plan.atom_offsets.size()),
      count(plan.batch_shell_offsets.size()),
      count(plan.atom_shell_offsets.size()),
      count(plan.shell_population_offsets.size()),
      count(plan.spin_channels.size()),
      count(plan.coupling_offsets.size()),
      count(plan.coupling_matrices.size()),
      plan.atom_offsets.data(),
      plan.batch_shell_offsets.data(),
      plan.atom_shell_offsets.data(),
      plan.shell_population_offsets.data(),
      plan.spin_channels.data(),
      plan.coupling_offsets.data(),
      plan.coupling_matrices.data(),
  };
}

xtbloom_status_t evaluate_spin_polarization_cpu(SpinPolarizationView view,
                                                const double* shell_populations,
                                                double* spin_energies,
                                                double* shell_potentials,
                                                std::string& error) {
  xtbloom_status_t status = validate_view(view, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (shell_populations == nullptr || spin_energies == nullptr || shell_potentials == nullptr) {
    error = "GFN1 spin populations and outputs must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::size_t population_bytes = 0u;
  std::size_t energy_bytes = 0u;
  if (!count_bytes(view.shell_population_elements, sizeof(double), population_bytes) ||
      !count_bytes(view.batch_size, sizeof(double), energy_bytes) ||
      ranges_overlap(shell_populations, population_bytes, spin_energies, energy_bytes) ||
      ranges_overlap(shell_populations, population_bytes, shell_potentials, population_bytes) ||
      ranges_overlap(spin_energies, energy_bytes, shell_potentials, population_bytes)) {
    error = "GFN1 spin outputs must be mutually disjoint from their inputs";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t element = 0; element < view.shell_population_elements; ++element) {
    if (!std::isfinite(shell_populations[element])) {
      error = "GFN1 spin populations contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    if (view.spin_channels[system] == 2) {
      double energy = 0.0;
      if (!evaluate_unrestricted_system(view, system, shell_populations, nullptr, energy)) {
        error = "GFN1 spin energy or potential exceeded floating-point range";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
    }
  }
  std::fill_n(spin_energies, static_cast<std::size_t>(view.batch_size), 0.0);
  std::fill_n(shell_potentials, static_cast<std::size_t>(view.shell_population_elements), 0.0);
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    if (view.spin_channels[system] == 2) {
      (void)evaluate_unrestricted_system(view, system, shell_populations, shell_potentials,
                                         spin_energies[system]);
    }
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t evaluate_spin_polarization_system_cpu(
    SpinPolarizationView view, std::int64_t system, const double* shell_populations,
    double& spin_energy, double* shell_potentials, std::string& error) {
  xtbloom_status_t status = validate_view(view, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (system < 0 || system >= view.batch_size || shell_populations == nullptr ||
      shell_potentials == nullptr) {
    error = "GFN1 spin one-system inputs and outputs are invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::size_t population_bytes = 0u;
  if (!count_bytes(view.shell_population_elements, sizeof(double), population_bytes) ||
      ranges_overlap(shell_populations, population_bytes, shell_potentials, population_bytes)) {
    error = "GFN1 spin one-system potential output must not overlap populations";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::int64_t population_begin = view.shell_population_offsets[system];
  const std::int64_t population_end = view.shell_population_offsets[system + 1];
  for (std::int64_t element = population_begin; element < population_end; ++element) {
    if (!std::isfinite(shell_populations[element])) {
      error = "GFN1 spin target populations contain NaN or infinity";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  double energy = 0.0;
  if (view.spin_channels[system] == 2 &&
      !evaluate_unrestricted_system(view, system, shell_populations, shell_potentials, energy)) {
    error = "GFN1 spin target potential exceeded floating-point range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  spin_energy = energy;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_spin_polarization_energy_system_cpu(
    SpinPolarizationView view, std::int64_t system, const double* shell_populations,
    double& accumulated_energy, std::string& error) {
  xtbloom_status_t status = validate_view(view, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (system < 0 || system >= view.batch_size || shell_populations == nullptr) {
    error = "GFN1 spin energy system index or populations are invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!std::isfinite(accumulated_energy)) {
    error = "GFN1 spin accumulated energy is not finite";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  const std::int64_t population_begin = view.shell_population_offsets[system];
  const std::int64_t population_end = view.shell_population_offsets[system + 1];
  for (std::int64_t element = population_begin; element < population_end; ++element) {
    if (!std::isfinite(shell_populations[element])) {
      error = "GFN1 spin target populations contain NaN or infinity";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  double energy = 0.0;
  if (view.spin_channels[system] == 2 &&
      !evaluate_unrestricted_system(view, system, shell_populations, nullptr, energy)) {
    error = "GFN1 spin target energy exceeded floating-point range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  if (!std::isfinite(accumulated_energy + energy)) {
    error = "GFN1 spin accumulated energy exceeded floating-point range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  accumulated_energy += energy;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
