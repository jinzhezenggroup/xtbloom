#ifndef XTBLOOM_MODEL_GFN2_MULLIKEN_KERNELS_IMPL_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_MULLIKEN_KERNELS_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "model/gfn2/mulliken_kernels.hpp"

namespace xtbloom::detail::gfn2::kernel_implementation {

static bool add_product(double left, double right, double& accumulator) noexcept {
  /* std::fma is part of the existing numerical and overflow-classification
   * contract. The AVX2/FMA TU turns it into hardware FMA; baseline remains safe
   * on the wheel's oldest supported x86-64 CPUs. */
  const double updated = std::fma(left, right, accumulator);
  if (!std::isfinite(updated)) {
    return false;
  }
  accumulator = updated;
  return true;
}

static void population_record_failure(MullikenPopulationTask& task,
                                      std::uint64_t candidate) noexcept {
  std::uint64_t current = task.failure.load(std::memory_order_relaxed);
  while ((current == 0u || candidate < current) &&
         !task.failure.compare_exchange_weak(current, candidate, std::memory_order_relaxed,
                                             std::memory_order_relaxed)) {
  }
}

static void population_fail(MullikenPopulationTask& task, int code,
                            std::uint64_t position) noexcept {
  population_record_failure(task, (position << 16u) | static_cast<std::uint32_t>(code));
}

static void population_chunk(void* opaque, std::size_t chunk) noexcept {
  MullikenPopulationTask& task = *static_cast<MullikenPopulationTask*>(opaque);
  /* Every chunk must inspect its own range even after a peer reports failure.
   * Each chunk returns at its first failure; the atomic minimum reconstructs
   * the first serially encountered site independently of worker scheduling. */
  const std::int64_t per_chunk = (task.atoms + task.chunk_count - 1) / task.chunk_count;
  const std::int64_t local_atom_begin =
      std::min<std::int64_t>(static_cast<std::int64_t>(chunk) * per_chunk, task.atoms);
  const std::int64_t local_atom_end =
      std::min<std::int64_t>(local_atom_begin + per_chunk, task.atoms);
  const std::int64_t* orbitals_begin = task.orbital_to_atom + task.orbital_begin;
  const std::int64_t* orbitals_end = orbitals_begin + task.orbitals;
  const std::int64_t local_ket_begin =
      std::lower_bound(orbitals_begin, orbitals_end, task.atom_begin + local_atom_begin) -
      orbitals_begin;
  const std::int64_t local_ket_end =
      std::lower_bound(orbitals_begin, orbitals_end, task.atom_begin + local_atom_end) -
      orbitals_begin;
  for (std::int32_t spin = 0; spin < task.nspin; ++spin) {
    const std::int64_t spin_matrix_base = task.density_base + spin * task.orbitals * task.orbitals;
    for (std::int64_t local_ket = local_ket_begin; local_ket < local_ket_end; ++local_ket) {
      const std::int64_t ket = task.orbital_begin + local_ket;
      const std::int64_t local_shell =
          task.orbital_to_shell[static_cast<std::size_t>(ket)] - task.shell_begin;
      const std::int64_t local_atom =
          task.orbital_to_atom[static_cast<std::size_t>(ket)] - task.atom_begin;
      double& shell_charge = task.qsh_scratch[static_cast<std::size_t>(
          task.qsh_base + spin * task.shells + local_shell)];
      for (std::int64_t local_bra = 0; local_bra < task.orbitals; ++local_bra) {
        const std::int64_t matrix_index = task.matrix_base + local_bra * task.orbitals + local_ket;
        const std::int64_t density_index = spin_matrix_base + local_bra * task.orbitals + local_ket;
        const std::uint64_t element_position =
            static_cast<std::uint64_t>(
                ((spin * task.orbitals + local_ket) * task.orbitals + local_bra)) *
            11u;
        const double density_value = task.density[static_cast<std::size_t>(density_index)];
        const double overlap_value = task.overlap[static_cast<std::size_t>(matrix_index)];
        if (!std::isfinite(density_value) || !std::isfinite(overlap_value)) {
          population_fail(task, 1, element_position);
          return;
        }
        if (!add_product(-density_value, overlap_value, shell_charge)) {
          population_fail(task, 4, element_position + 1u);
          return;
        }
        for (std::int64_t component = 0; component < 3; ++component) {
          double& value = task.dipole_scratch[static_cast<std::size_t>(
              task.dipole_base + (spin * task.atoms + local_atom) * 3 + component)];
          const double integral = task.dipole_integrals[static_cast<std::size_t>(
              component * task.matrix_elements + matrix_index)];
          if (!std::isfinite(integral)) {
            population_fail(task, 2, element_position + 2u + static_cast<std::uint64_t>(component));
            return;
          }
          if (!add_product(-density_value, integral, value)) {
            population_fail(task, 5, element_position + 2u + static_cast<std::uint64_t>(component));
            return;
          }
        }
        for (std::int64_t component = 0; component < 6; ++component) {
          double& value = task.quadrupole_scratch[static_cast<std::size_t>(
              task.quadrupole_base + (spin * task.atoms + local_atom) * 6 + component)];
          const double integral = task.quadrupole_integrals[static_cast<std::size_t>(
              component * task.matrix_elements + matrix_index)];
          if (!std::isfinite(integral)) {
            population_fail(task, 3, element_position + 5u + static_cast<std::uint64_t>(component));
            return;
          }
          if (!add_product(-density_value, integral, value)) {
            population_fail(task, 6, element_position + 5u + static_cast<std::uint64_t>(component));
            return;
          }
        }
      }
    }
  }
}

static void hamiltonian_record_failure(MullikenHamiltonianTask& task,
                                       std::uint64_t candidate) noexcept {
  std::uint64_t current = task.failure.load(std::memory_order_relaxed);
  while ((current == 0u || candidate < current) &&
         !task.failure.compare_exchange_weak(current, candidate, std::memory_order_relaxed,
                                             std::memory_order_relaxed)) {
  }
}

static void hamiltonian_fail(MullikenHamiltonianTask& task, int code,
                             std::uint64_t position) noexcept {
  hamiltonian_record_failure(task, (position << 16u) | static_cast<std::uint32_t>(code));
}

static void hamiltonian_chunk(void* opaque, std::size_t chunk) noexcept {
  MullikenHamiltonianTask& task = *static_cast<MullikenHamiltonianTask*>(opaque);
  const std::int64_t per_chunk = (task.orbitals + task.chunk_count - 1) / task.chunk_count;
  const std::int64_t local_row_begin =
      std::min<std::int64_t>(static_cast<std::int64_t>(chunk) * per_chunk, task.orbitals);
  const std::int64_t local_row_end =
      std::min<std::int64_t>(local_row_begin + per_chunk, task.orbitals);
  for (std::int32_t spin = 0; spin < task.nspin; ++spin) {
    const std::int64_t spin_matrix_base =
        task.hamiltonian_base + spin * task.orbitals * task.orbitals;
    for (std::int64_t local_row = local_row_begin; local_row < local_row_end; ++local_row) {
      const std::int64_t row = task.orbital_begin + local_row;
      const std::int64_t row_shell = task.orbital_to_shell[static_cast<std::size_t>(row)];
      const std::int64_t row_atom = task.orbital_to_atom[static_cast<std::size_t>(row)];
      const std::int64_t local_row_shell = row_shell - task.shell_begin;
      const std::int64_t local_row_atom = row_atom - task.atom_begin;
      const double row_vat = task.vat_scratch[static_cast<std::size_t>(
          task.vat_base + spin * task.atoms + local_row_atom)];
      const double row_vsh = task.vsh_scratch[static_cast<std::size_t>(
          task.vsh_base + spin * task.shells + local_row_shell)];

      for (std::int64_t local_column = local_row; local_column < task.orbitals; ++local_column) {
        const std::int64_t column = task.orbital_begin + local_column;
        const std::int64_t column_shell = task.orbital_to_shell[static_cast<std::size_t>(column)];
        const std::int64_t column_atom = task.orbital_to_atom[static_cast<std::size_t>(column)];
        const std::int64_t local_column_shell = column_shell - task.shell_begin;
        const std::int64_t local_column_atom = column_atom - task.atom_begin;
        const double column_vat = task.vat_scratch[static_cast<std::size_t>(
            task.vat_base + spin * task.atoms + local_column_atom)];
        const double column_vsh = task.vsh_scratch[static_cast<std::size_t>(
            task.vsh_base + spin * task.shells + local_column_shell)];

        const std::int64_t forward_matrix =
            task.matrix_base + local_row * task.orbitals + local_column;
        const std::int64_t reverse_matrix =
            task.matrix_base + local_column * task.orbitals + local_row;
        const std::int64_t forward_hamiltonian =
            spin_matrix_base + local_row * task.orbitals + local_column;
        const std::int64_t reverse_hamiltonian =
            spin_matrix_base + local_column * task.orbitals + local_row;
        double shift = 0.0;
        const double overlap = task.overlap[static_cast<std::size_t>(forward_matrix)];
        const double reverse_overlap = task.overlap[static_cast<std::size_t>(reverse_matrix)];
        const double half_overlap = -0.5 * overlap;
        const std::uint64_t element_position =
            static_cast<std::uint64_t>(
                ((spin * task.orbitals + local_row) * task.orbitals + local_column)) *
            12u;
        if (!std::isfinite(overlap) || !std::isfinite(reverse_overlap)) {
          hamiltonian_fail(task, 1, element_position);
          return;
        }
        if (!std::isfinite(half_overlap) || !add_product(half_overlap, row_vat, shift) ||
            !add_product(half_overlap, row_vsh, shift) ||
            !add_product(half_overlap, column_vat, shift) ||
            !add_product(half_overlap, column_vsh, shift)) {
          hamiltonian_fail(task, 4, element_position + 1u);
          return;
        }

        for (std::int64_t component = 0; component < 3; ++component) {
          const double row_potential = task.dipole_scratch[static_cast<std::size_t>(
              task.dipole_base + (spin * task.atoms + local_row_atom) * 3 + component)];
          const double column_potential = task.dipole_scratch[static_cast<std::size_t>(
              task.dipole_base + (spin * task.atoms + local_column_atom) * 3 + component)];
          const double forward_integral =
              -0.5 * task.dipole_integrals[static_cast<std::size_t>(
                         component * task.matrix_elements + forward_matrix)];
          const double reverse_integral =
              -0.5 * task.dipole_integrals[static_cast<std::size_t>(
                         component * task.matrix_elements + reverse_matrix)];
          if (!std::isfinite(forward_integral) || !std::isfinite(reverse_integral)) {
            hamiltonian_fail(task, 2,
                             element_position + 2u + static_cast<std::uint64_t>(component));
            return;
          }
          if (!std::isfinite(row_potential) || !std::isfinite(column_potential) ||
              !add_product(forward_integral, column_potential, shift) ||
              !add_product(reverse_integral, row_potential, shift)) {
            hamiltonian_fail(task, 5,
                             element_position + 2u + static_cast<std::uint64_t>(component));
            return;
          }
        }

        for (std::int64_t component = 0; component < 6; ++component) {
          const double row_potential = task.quadrupole_scratch[static_cast<std::size_t>(
              task.quadrupole_base + (spin * task.atoms + local_row_atom) * 6 + component)];
          const double column_potential = task.quadrupole_scratch[static_cast<std::size_t>(
              task.quadrupole_base + (spin * task.atoms + local_column_atom) * 6 + component)];
          const double forward_integral =
              -0.5 * task.quadrupole_integrals[static_cast<std::size_t>(
                         component * task.matrix_elements + forward_matrix)];
          const double reverse_integral =
              -0.5 * task.quadrupole_integrals[static_cast<std::size_t>(
                         component * task.matrix_elements + reverse_matrix)];
          if (!std::isfinite(forward_integral) || !std::isfinite(reverse_integral)) {
            hamiltonian_fail(task, 3,
                             element_position + 5u + static_cast<std::uint64_t>(component));
            return;
          }
          if (!std::isfinite(row_potential) || !std::isfinite(column_potential) ||
              !add_product(forward_integral, column_potential, shift) ||
              !add_product(reverse_integral, row_potential, shift)) {
            hamiltonian_fail(task, 6,
                             element_position + 5u + static_cast<std::uint64_t>(component));
            return;
          }
        }
        const double forward_value =
            task.hamiltonian_scratch[static_cast<std::size_t>(forward_hamiltonian)] + shift;
        if (!std::isfinite(forward_value)) {
          hamiltonian_fail(task, 7, element_position + 11u);
          return;
        }
        task.hamiltonian_scratch[static_cast<std::size_t>(forward_hamiltonian)] = forward_value;
        if (forward_hamiltonian != reverse_hamiltonian) {
          const double reverse_value =
              task.hamiltonian_scratch[static_cast<std::size_t>(reverse_hamiltonian)] + shift;
          if (!std::isfinite(reverse_value)) {
            hamiltonian_fail(task, 7, element_position + 11u);
            return;
          }
          task.hamiltonian_scratch[static_cast<std::size_t>(reverse_hamiltonian)] = reverse_value;
        }
      }
    }
  }
}

}  // namespace xtbloom::detail::gfn2::kernel_implementation

#endif  // XTBLOOM_MODEL_GFN2_MULLIKEN_KERNELS_IMPL_HPP
