// SPDX-License-Identifier: GPL-3.0-or-later
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/spin.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "data/parameters/tblite_spin.hpp"

namespace gpuxtb::detail::gfn2 {
namespace {

bool representable_as_size(std::int64_t value) {
  return value >= 0 && static_cast<std::uint64_t>(value) <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
}

bool count_bytes(std::int64_t count, std::size_t element_size, std::size_t& bytes) {
  if (count < 0 ||
      static_cast<std::uint64_t>(count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
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

gpuxtb_status_t validate_view(SpinPolarizationView view, std::string& error) {
  if (view.batch_size <= 0 || view.total_atoms <= 0 || view.total_shells <= 0 ||
      view.shell_population_elements <= 0 || !representable_as_size(view.batch_size) ||
      !representable_as_size(view.total_atoms) || !representable_as_size(view.total_shells) ||
      !representable_as_size(view.shell_population_elements) ||
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
    error = "spin-polarization view is incomplete or has unrepresentable dimensions";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
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
    error = "spin-polarization view offsets do not span their packed fields";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    const std::int64_t atom_begin = view.atom_offsets[system];
    const std::int64_t atom_end = view.atom_offsets[system + 1];
    const std::int64_t shell_begin = view.batch_shell_offsets[system];
    const std::int64_t shell_end = view.batch_shell_offsets[system + 1];
    const std::int64_t population_begin = view.shell_population_offsets[system];
    const std::int64_t population_end = view.shell_population_offsets[system + 1];
    const std::int32_t channels = view.spin_channels[system];
    if (atom_begin < 0 || atom_begin >= atom_end || atom_end > view.total_atoms ||
        shell_begin < 0 || shell_begin >= shell_end || shell_end > view.total_shells ||
        population_begin < 0 || population_begin >= population_end ||
        population_end > view.shell_population_elements || (channels != 1 && channels != 2) ||
        population_end - population_begin != (shell_end - shell_begin) * channels ||
        view.atom_shell_offsets[atom_begin] != shell_begin ||
        view.atom_shell_offsets[atom_end] != shell_end) {
      error = "spin-polarization view has an invalid ragged system partition";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t atom = 0; atom < view.total_atoms; ++atom) {
    const std::int64_t shell_begin = view.atom_shell_offsets[atom];
    const std::int64_t shell_end = view.atom_shell_offsets[atom + 1];
    const std::int64_t matrix_begin = view.coupling_offsets[atom];
    const std::int64_t matrix_end = view.coupling_offsets[atom + 1];
    const std::int64_t shells = shell_end - shell_begin;
    if (shell_begin < 0 || shell_begin >= shell_end || shell_end > view.total_shells ||
        matrix_begin < 0 || matrix_begin > matrix_end || matrix_end > view.coupling_matrix_count ||
        shells > 3 || matrix_end - matrix_begin != shells * shells) {
      error = "spin-polarization view has an invalid atom-local coupling partition";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t matrix = matrix_begin; matrix < matrix_end; ++matrix) {
      if (!std::isfinite(view.coupling_matrices[matrix])) {
        error = "spin-polarization view contains a non-finite coupling";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return GPUXTB_STATUS_SUCCESS;
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
      const std::int64_t population = magnetization_base + shell_begin - system_shell_begin + row;
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

gpuxtb_status_t make_spin_polarization_plan(const BasisPlan& basis,
                                            const WavefunctionLayout& wavefunction,
                                            SpinPolarizationPlan& plan, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      wavefunction.batch_size != basis.batch_size ||
      wavefunction.total_atoms != basis.total_atoms ||
      wavefunction.total_shells != basis.total_shells ||
      basis.atom_offsets != wavefunction.atom_offsets ||
      basis.batch_shell_offsets != wavefunction.batch_shell_offsets ||
      wavefunction.atomic_numbers.size() != static_cast<std::size_t>(basis.total_atoms) ||
      basis.atom_shell_offsets.size() != static_cast<std::size_t>(basis.total_atoms) + 1u ||
      basis.angular_momenta.size() != static_cast<std::size_t>(basis.total_shells) ||
      wavefunction.qsh.system_offsets.size() != static_cast<std::size_t>(basis.batch_size) + 1u ||
      wavefunction.spin_channels.size() != static_cast<std::size_t>(basis.batch_size)) {
    error = "spin-polarization plan requires one complete matching basis and wavefunction layout";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  try {
    SpinPolarizationPlan created;
    created.batch_size = basis.batch_size;
    created.total_atoms = basis.total_atoms;
    created.total_shells = basis.total_shells;
    created.shell_population_elements = wavefunction.qsh.element_count;
    created.atom_offsets = basis.atom_offsets;
    created.batch_shell_offsets = basis.batch_shell_offsets;
    created.atom_shell_offsets = basis.atom_shell_offsets;
    created.shell_population_offsets = wavefunction.qsh.system_offsets;
    created.spin_channels = wavefunction.spin_channels;
    created.coupling_offsets.resize(static_cast<std::size_t>(basis.total_atoms) + 1u, 0);

    std::int64_t coupling_count = 0;
    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::int64_t shell_begin = basis.atom_shell_offsets[static_cast<std::size_t>(atom)];
      const std::int64_t shell_end = basis.atom_shell_offsets[static_cast<std::size_t>(atom) + 1u];
      const std::int64_t shells = shell_end - shell_begin;
      if (shell_begin < 0 || shell_begin >= shell_end || shell_end > basis.total_shells ||
          shells > 3 ||
          coupling_count > std::numeric_limits<std::int64_t>::max() - shells * shells) {
        error = "spin-polarization basis has an unsupported atom-local shell partition";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      coupling_count += shells * shells;
      created.coupling_offsets[static_cast<std::size_t>(atom) + 1u] = coupling_count;
    }
    if (!representable_as_size(coupling_count)) {
      error = "spin-polarization coupling dimensions exceed host container limits";
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    created.coupling_matrices.resize(static_cast<std::size_t>(coupling_count));

    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::int32_t atomic_number =
          wavefunction.atomic_numbers[static_cast<std::size_t>(atom)];
      if (atomic_number <= 0 ||
          static_cast<std::size_t>(atomic_number) > parameters::tblite::kSpinConstants.size()) {
        error = "spin-polarization plan contains an unsupported element";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      const std::int64_t shell_begin = basis.atom_shell_offsets[static_cast<std::size_t>(atom)];
      const std::int64_t shell_end = basis.atom_shell_offsets[static_cast<std::size_t>(atom) + 1u];
      const std::int64_t shells = shell_end - shell_begin;
      const std::int64_t matrix_begin = created.coupling_offsets[static_cast<std::size_t>(atom)];
      for (std::int64_t row = 0; row < shells; ++row) {
        const std::uint8_t row_l =
            basis.angular_momenta[static_cast<std::size_t>(shell_begin + row)];
        if (row_l > 2u) {
          error = "spin-polarization plan supports only GFN2 s, p, and d shells";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        for (std::int64_t column = 0; column < shells; ++column) {
          const std::uint8_t column_l =
              basis.angular_momenta[static_cast<std::size_t>(shell_begin + column)];
          if (column_l > 2u) {
            error = "spin-polarization plan supports only GFN2 s, p, and d shells";
            return GPUXTB_STATUS_INVALID_ARGUMENT;
          }
          created
              .coupling_matrices[static_cast<std::size_t>(matrix_begin + row * shells + column)] =
              parameters::tblite::kSpinConstants[static_cast<std::size_t>(atomic_number - 1)]
                                                [coupling_index(row_l, column_l)];
        }
      }
    }

    const SpinPolarizationView view = make_spin_polarization_view(created);
    if (validate_view(view, error) != GPUXTB_STATUS_SUCCESS) {
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    plan = std::move(created);
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN2 spin-polarization plan";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN2 spin-polarization plan dimensions exceed host container limits";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
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

gpuxtb_status_t evaluate_spin_polarization_cpu(SpinPolarizationView view,
                                               const double* shell_populations,
                                               double* spin_energies, double* shell_potentials,
                                               std::string& error) {
  gpuxtb_status_t status = validate_view(view, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (shell_populations == nullptr || spin_energies == nullptr || shell_potentials == nullptr) {
    error = "spin-polarization populations and outputs must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  std::size_t population_bytes = 0u;
  std::size_t energy_bytes = 0u;
  if (!count_bytes(view.shell_population_elements, sizeof(double), population_bytes) ||
      !count_bytes(view.batch_size, sizeof(double), energy_bytes) ||
      ranges_overlap(shell_populations, population_bytes, spin_energies, energy_bytes) ||
      ranges_overlap(shell_populations, population_bytes, shell_potentials, population_bytes) ||
      ranges_overlap(spin_energies, energy_bytes, shell_potentials, population_bytes)) {
    error = "spin-polarization outputs must be mutually disjoint from their inputs";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t element = 0; element < view.shell_population_elements; ++element) {
    if (!std::isfinite(shell_populations[element])) {
      error = "spin-polarization shell populations contain NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    if (view.spin_channels[system] == 2) {
      double energy = 0.0;
      if (!evaluate_unrestricted_system(view, system, shell_populations, nullptr, energy)) {
        error = "spin-polarization energy or potential exceeded floating-point range";
        return GPUXTB_STATUS_INTERNAL_ERROR;
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
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t evaluate_spin_polarization_system_cpu(SpinPolarizationView view,
                                                      std::int64_t system,
                                                      const double* shell_populations,
                                                      double& spin_energy, double* shell_potentials,
                                                      std::string& error) {
  gpuxtb_status_t status = validate_view(view, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (system < 0 || system >= view.batch_size || shell_populations == nullptr ||
      shell_potentials == nullptr ||
      reinterpret_cast<std::uintptr_t>(shell_populations) % alignof(double) != 0u ||
      reinterpret_cast<std::uintptr_t>(shell_potentials) % alignof(double) != 0u ||
      reinterpret_cast<std::uintptr_t>(&spin_energy) % alignof(double) != 0u) {
    error = "spin-polarization one-system inputs and outputs must not be NULL or misaligned";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (ranges_overlap(shell_populations,
                     static_cast<std::size_t>(view.shell_population_elements) * sizeof(double),
                     shell_potentials,
                     static_cast<std::size_t>(view.shell_population_elements) * sizeof(double))) {
    error = "spin-polarization one-system potentials must not overlap their inputs";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::int64_t population_begin = view.shell_population_offsets[system];
  const std::int64_t population_end = view.shell_population_offsets[system + 1];
  for (std::int64_t element = population_begin; element < population_end; ++element) {
    if (!std::isfinite(shell_populations[element])) {
      error = "spin-polarization target populations contain NaN or infinity";
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
  }
  double energy = 0.0;
  if (view.spin_channels[system] == 2) {
    if (!evaluate_unrestricted_system(view, system, shell_populations, shell_potentials, energy)) {
      error = "spin-polarization target potential exceeded floating-point range";
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
  }
  if (!std::isfinite(energy)) {
    error = "spin-polarization target energy is not finite";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  spin_energy = energy;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_spin_polarization_energy_system_cpu(SpinPolarizationView view,
                                                        std::int64_t system,
                                                        const double* shell_populations,
                                                        double& accumulated_energy,
                                                        std::string& error) {
  gpuxtb_status_t status = validate_view(view, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (system < 0 || system >= view.batch_size || shell_populations == nullptr) {
    error = "spin-polarization energy system index or populations are invalid";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (!std::isfinite(accumulated_energy)) {
    error = "spin-polarization accumulated energy is not finite";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  const std::int64_t population_begin = view.shell_population_offsets[system];
  const std::int64_t population_end = view.shell_population_offsets[system + 1];
  for (std::int64_t element = population_begin; element < population_end; ++element) {
    if (!std::isfinite(shell_populations[element])) {
      error = "spin-polarization target populations contain NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  double energy = 0.0;
  if (view.spin_channels[system] == 2 &&
      !evaluate_unrestricted_system(view, system, shell_populations, nullptr, energy)) {
    error = "spin-polarization target energy exceeded floating-point range";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  const double updated = accumulated_energy + energy;
  if (!std::isfinite(updated)) {
    error = "spin-polarization accumulated energy exceeded floating-point range";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  accumulated_energy = updated;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail::gfn2
