// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/es3.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>

#include "data/parameters/gfn1.hpp"

namespace xtbloom::detail::gfn1 {
namespace {

static_assert(!parameters::gfn1::kGlobal.thirdorder_shell_resolved,
              "GFN1 ES3 must remain atom resolved");

bool representable(std::int64_t value) {
  return value >= 0 && static_cast<std::uint64_t>(value) <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
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

bool aligned_double(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(double) == 0u;
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

xtbloom_status_t validate_view(ES3View view, std::string& error) {
  if (view.batch_size <= 0 || view.total_atoms <= 0 || !representable(view.batch_size) ||
      !representable(view.total_atoms) ||
      view.batch_size == std::numeric_limits<std::int64_t>::max() ||
      view.atom_offset_count != view.batch_size + 1 || view.atom_gamma3_count != view.total_atoms ||
      view.atom_offsets == nullptr || view.atom_gamma3 == nullptr) {
    error = "GFN1 ES3 view is incomplete or has unrepresentable dimensions";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (view.atom_offsets[0] != 0 || view.atom_offsets[view.batch_size] != view.total_atoms) {
    error = "GFN1 ES3 atom offsets do not span the packed atoms";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    if (view.atom_offsets[system] < 0 ||
        view.atom_offsets[system] > view.atom_offsets[system + 1] ||
        view.atom_offsets[system + 1] > view.total_atoms) {
      error = "GFN1 ES3 atom offsets are not a valid ragged partition";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t atom = 0; atom < view.total_atoms; ++atom) {
    if (!std::isfinite(view.atom_gamma3[atom])) {
      error = "GFN1 ES3 view contains a non-finite Hubbard derivative";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_system(ES3View view, std::int64_t system, std::int64_t& atom_begin,
                                 std::int64_t& atom_end, std::string& error) {
  const xtbloom_status_t status = validate_view(view, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (system < 0 || system >= view.batch_size) {
    error = "GFN1 ES3 system index is out of range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  atom_begin = view.atom_offsets[system];
  atom_end = view.atom_offsets[system + 1];
  return XTBLOOM_STATUS_SUCCESS;
}

bool atom_potential(double gamma3, double charge, double& potential) {
  potential = gamma3 * charge * charge;
  return std::isfinite(charge) && std::isfinite(potential);
}

bool system_energy(ES3View view, std::int64_t atom_begin, std::int64_t atom_end,
                   const double* atomic_charges, double& energy) {
  energy = 0.0;
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    const double charge = atomic_charges[atom];
    const double contribution = view.atom_gamma3[atom] * charge * charge * charge / 3.0;
    const double updated = energy + contribution;
    if (!std::isfinite(charge) || !std::isfinite(contribution) || !std::isfinite(updated)) {
      return false;
    }
    energy = updated;
  }
  return true;
}

}  // namespace

xtbloom_status_t make_es3_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                               ES3Plan& plan, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      atomic_numbers == nullptr || !representable(basis.batch_size) ||
      !representable(basis.total_atoms) ||
      basis.atom_offsets.size() != static_cast<std::size_t>(basis.batch_size) + 1u ||
      basis.atom_shell_offsets.size() != static_cast<std::size_t>(basis.total_atoms) + 1u ||
      basis.shell_to_atom.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.principal_quantum_numbers.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.angular_momenta.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.slater_exponents.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.atom_offsets.front() != 0 || basis.atom_offsets.back() != basis.total_atoms) {
    error = "GFN1 ES3 plan requires one complete representable basis and element list";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t system = 0; system < basis.batch_size; ++system) {
    if (basis.atom_offsets[static_cast<std::size_t>(system)] < 0 ||
        basis.atom_offsets[static_cast<std::size_t>(system)] >
            basis.atom_offsets[static_cast<std::size_t>(system) + 1u] ||
        basis.atom_offsets[static_cast<std::size_t>(system) + 1u] > basis.total_atoms) {
      error = "GFN1 ES3 basis offsets are not a valid ragged partition";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  try {
    ES3Plan created;
    created.batch_size = basis.batch_size;
    created.total_atoms = basis.total_atoms;
    created.atom_offsets = basis.atom_offsets;
    created.atom_gamma3.resize(static_cast<std::size_t>(basis.total_atoms));
    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::int32_t atomic_number = atomic_numbers[static_cast<std::size_t>(atom)];
      const auto* element =
          parameters::gfn1::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number ||
          !std::isfinite(element->gam3)) {
        error = "GFN1 ES3 plan contains an unsupported element or invalid Hubbard derivative";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const std::int64_t shell_begin = basis.atom_shell_offsets[static_cast<std::size_t>(atom)];
      const std::int64_t shell_end = basis.atom_shell_offsets[static_cast<std::size_t>(atom) + 1u];
      const std::size_t parameter_begin = element->shell_offset;
      if (shell_begin < 0 || shell_begin >= shell_end || shell_end > basis.total_shells ||
          shell_end - shell_begin != element->shell_count ||
          parameter_begin > parameters::gfn1::kShells.size() ||
          element->shell_count > parameters::gfn1::kShells.size() - parameter_begin) {
        error = "GFN1 ES3 element list does not match the basis shell layout";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        const std::size_t shell_index = static_cast<std::size_t>(shell);
        const auto& parameter =
            parameters::gfn1::kShells[parameter_begin +
                                      static_cast<std::size_t>(shell - shell_begin)];
        if (basis.shell_to_atom[shell_index] != atom ||
            basis.principal_quantum_numbers[shell_index] != parameter.principal_quantum_number ||
            basis.angular_momenta[shell_index] != parameter.angular_momentum ||
            basis.slater_exponents[shell_index] != parameter.slater) {
          error = "GFN1 ES3 element list does not match the basis shell metadata";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }
      created.atom_gamma3[static_cast<std::size_t>(atom)] = element->gam3;
    }
    plan = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 ES3 plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN1 ES3 dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

ES3View make_es3_view(const ES3Plan& plan) noexcept {
  const auto count = [](std::size_t value) noexcept {
    return value <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())
               ? static_cast<std::int64_t>(value)
               : std::int64_t{-1};
  };
  return ES3View{plan.batch_size,
                 plan.total_atoms,
                 count(plan.atom_offsets.size()),
                 count(plan.atom_gamma3.size()),
                 plan.atom_offsets.data(),
                 plan.atom_gamma3.data()};
}

xtbloom_status_t evaluate_es3_potential_cpu(ES3View view, const double* atomic_charges,
                                            double* atomic_potentials, std::string& error) {
  xtbloom_status_t status = validate_view(view, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (atomic_charges == nullptr || atomic_potentials == nullptr ||
      !aligned_double(atomic_charges) || !aligned_double(atomic_potentials)) {
    error = "GFN1 ES3 atomic charges and potentials must not be NULL or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::size_t atom_bytes = 0u;
  if (!count_bytes(view.total_atoms, sizeof(double), atom_bytes) ||
      ranges_overlap(atomic_charges, atom_bytes, atomic_potentials, atom_bytes)) {
    error = "GFN1 ES3 atomic potential output must not overlap its input";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t atom = 0; atom < view.total_atoms; ++atom) {
    double potential = 0.0;
    if (!atom_potential(view.atom_gamma3[atom], atomic_charges[atom], potential)) {
      error = "GFN1 ES3 potential arithmetic contains invalid data or exceeded range";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  for (std::int64_t atom = 0; atom < view.total_atoms; ++atom) {
    atomic_potentials[atom] = view.atom_gamma3[atom] * atomic_charges[atom] * atomic_charges[atom];
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_es3_energy_cpu(ES3View view, const double* atomic_charges, double* energies,
                                    std::string& error) {
  xtbloom_status_t status = validate_view(view, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (atomic_charges == nullptr || energies == nullptr || !aligned_double(atomic_charges) ||
      !aligned_double(energies)) {
    error = "GFN1 ES3 atomic charges and energy output must not be NULL or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::size_t atom_bytes = 0u;
  std::size_t energy_bytes = 0u;
  if (!count_bytes(view.total_atoms, sizeof(double), atom_bytes) ||
      !count_bytes(view.batch_size, sizeof(double), energy_bytes) ||
      ranges_overlap(atomic_charges, atom_bytes, energies, energy_bytes)) {
    error = "GFN1 ES3 energy output must not overlap its input";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    double contribution = 0.0;
    if (!std::isfinite(energies[system]) ||
        !system_energy(view, view.atom_offsets[system], view.atom_offsets[system + 1],
                       atomic_charges, contribution) ||
        !std::isfinite(energies[system] + contribution)) {
      error = "GFN1 ES3 energy arithmetic contains invalid data or exceeded range";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    double contribution = 0.0;
    (void)system_energy(view, view.atom_offsets[system], view.atom_offsets[system + 1],
                        atomic_charges, contribution);
    energies[system] += contribution;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t evaluate_es3_potential_system_cpu(ES3View view, std::int64_t system,
                                                   const double* atomic_charges,
                                                   double* atomic_potentials, std::string& error) {
  std::int64_t atom_begin = 0;
  std::int64_t atom_end = 0;
  xtbloom_status_t status = validate_system(view, system, atom_begin, atom_end, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (atomic_charges == nullptr || atomic_potentials == nullptr ||
      !aligned_double(atomic_charges) || !aligned_double(atomic_potentials)) {
    error = "GFN1 ES3 one-system charges and potentials must not be NULL or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  std::size_t atom_bytes = 0u;
  if (!count_bytes(view.total_atoms, sizeof(double), atom_bytes) ||
      ranges_overlap(atomic_charges, atom_bytes, atomic_potentials, atom_bytes)) {
    error = "GFN1 ES3 one-system potential output must not overlap its input";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    double potential = 0.0;
    if (!atom_potential(view.atom_gamma3[atom], atomic_charges[atom], potential)) {
      error = "GFN1 ES3 target-system potential contains invalid data or exceeded range";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    atomic_potentials[atom] = potential;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_es3_energy_system_cpu(ES3View view, std::int64_t system,
                                           const double* atomic_charges, double& accumulated_energy,
                                           std::string& error) {
  std::int64_t atom_begin = 0;
  std::int64_t atom_end = 0;
  xtbloom_status_t status = validate_system(view, system, atom_begin, atom_end, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (atomic_charges == nullptr || !aligned_double(atomic_charges)) {
    error = "GFN1 ES3 one-system charges must not be NULL or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!std::isfinite(accumulated_energy)) {
    error = "GFN1 ES3 accumulated energy is not finite";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  double contribution = 0.0;
  if (!system_energy(view, atom_begin, atom_end, atomic_charges, contribution) ||
      !std::isfinite(accumulated_energy + contribution)) {
    error = "GFN1 ES3 target-system energy contains invalid data or exceeded range";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  accumulated_energy += contribution;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
