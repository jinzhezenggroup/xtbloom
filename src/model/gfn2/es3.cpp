#include "model/gfn2/es3.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "data/parameters/gfn2.hpp"

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

bool is_aligned(const void* pointer, std::size_t alignment) {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

gpuxtb_status_t validate_basis(const BasisPlan& basis, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      !representable_as_size(basis.batch_size) || !representable_as_size(basis.total_atoms) ||
      !representable_as_size(basis.total_shells) ||
      static_cast<std::uint64_t>(basis.batch_size) >=
          static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    error = "ES3 requires a positive, representable basis plan";
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
    error = "ES3 basis plan is incomplete or internally inconsistent";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const std::int64_t atom_begin = basis.atom_offsets[batch];
    const std::int64_t atom_end = basis.atom_offsets[batch + 1u];
    const std::int64_t shell_begin = basis.batch_shell_offsets[batch];
    const std::int64_t shell_end = basis.batch_shell_offsets[batch + 1u];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > basis.total_atoms ||
        shell_begin < 0 || shell_begin > shell_end || shell_end > basis.total_shells ||
        shell_begin != basis.atom_shell_offsets[static_cast<std::size_t>(atom_begin)] ||
        shell_end != basis.atom_shell_offsets[static_cast<std::size_t>(atom_end)]) {
      error = "ES3 basis offsets are not valid ragged partitions";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    const std::int64_t shell_begin = basis.atom_shell_offsets[atom];
    const std::int64_t shell_end = basis.atom_shell_offsets[atom + 1u];
    if (shell_begin < 0 || shell_begin >= shell_end || shell_end > basis.total_shells) {
      error = "ES3 atom-to-shell offsets are invalid";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::size_t shell_index = static_cast<std::size_t>(shell);
      if (basis.shell_to_atom[shell_index] != static_cast<std::int64_t>(atom) ||
          basis.angular_momenta[shell_index] > 2u || !(basis.slater_exponents[shell_index] > 0.0) ||
          !std::isfinite(basis.slater_exponents[shell_index])) {
        error = "ES3 shell metadata is invalid";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_view(ES3View view, std::string& error) {
  std::size_t offset_bytes = 0;
  std::size_t shell_bytes = 0;
  if (view.batch_size <= 0 || view.total_shells <= 0 || !representable_as_size(view.batch_size) ||
      !representable_as_size(view.total_shells) ||
      view.batch_size == std::numeric_limits<std::int64_t>::max() ||
      view.batch_shell_offset_count != view.batch_size + 1 ||
      view.shell_gamma3_count != view.total_shells ||
      !count_bytes(view.batch_shell_offset_count, sizeof(std::int64_t), offset_bytes) ||
      !count_bytes(view.shell_gamma3_count, sizeof(double), shell_bytes) ||
      view.batch_shell_offsets == nullptr || view.shell_gamma3 == nullptr) {
    error = "ES3 view is incomplete or has unrepresentable dimensions";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (view.batch_shell_offsets[0] != 0 ||
      view.batch_shell_offsets[view.batch_size] != view.total_shells) {
    error = "ES3 view offsets do not span the stored shells";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < view.batch_size; ++batch) {
    const std::int64_t begin = view.batch_shell_offsets[batch];
    const std::int64_t end = view.batch_shell_offsets[batch + 1];
    if (begin < 0 || begin > end || end > view.total_shells) {
      error = "ES3 view offsets are not a valid ragged partition";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t shell = 0; shell < view.total_shells; ++shell) {
    if (!std::isfinite(view.shell_gamma3[shell])) {
      error = "ES3 view contains a non-finite shell Gamma3";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t validate_system_view(ES3View view, std::int64_t system, std::int64_t& shell_begin,
                                     std::int64_t& shell_end, std::string& error) {
  std::size_t offset_bytes = 0;
  std::size_t shell_bytes = 0;
  if (view.batch_size <= 0 || view.total_shells <= 0 || !representable_as_size(view.batch_size) ||
      !representable_as_size(view.total_shells) ||
      view.batch_size == std::numeric_limits<std::int64_t>::max() ||
      view.batch_shell_offset_count != view.batch_size + 1 ||
      view.shell_gamma3_count != view.total_shells ||
      !count_bytes(view.batch_shell_offset_count, sizeof(std::int64_t), offset_bytes) ||
      !count_bytes(view.shell_gamma3_count, sizeof(double), shell_bytes) ||
      !is_aligned(view.batch_shell_offsets, alignof(std::int64_t)) ||
      !is_aligned(view.shell_gamma3, alignof(double))) {
    error = "ES3 view is incomplete, misaligned, or has unrepresentable dimensions";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (system < 0 || system >= view.batch_size) {
    error = "ES3 energy system index is out of range";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (view.batch_shell_offsets[0] != 0 ||
      view.batch_shell_offsets[view.batch_size] != view.total_shells) {
    error = "ES3 view offsets do not span the stored shells";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  shell_begin = view.batch_shell_offsets[system];
  shell_end = view.batch_shell_offsets[system + 1];
  if (shell_begin < 0 || shell_begin > shell_end || shell_end > view.total_shells) {
    error = "ES3 target-system offsets are not a valid packed slice";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return GPUXTB_STATUS_SUCCESS;
}

/*
 * Evaluate in double for ordinary inputs, matching the operation order of the
 * reference formula. If an intermediate overflows, underflows, or loses range
 * while the mathematically final double remains representable, retry in the
 * wider host reference type.
 */
bool shell_potential(double gamma3, double charge, double& result) {
  if (charge == 0.0 || gamma3 == 0.0) {
    result = 0.0;
    return true;
  }
  const double square = charge * charge;
  result = square * gamma3;
  if (std::isnormal(square) && std::isnormal(gamma3) && std::isnormal(result)) {
    return true;
  }
  const long double wide = static_cast<long double>(charge) * static_cast<long double>(charge) *
                           static_cast<long double>(gamma3);
  if (!std::isfinite(wide) ||
      std::abs(wide) > static_cast<long double>(std::numeric_limits<double>::max())) {
    return false;
  }
  result = static_cast<double>(wide);
  return std::isfinite(result);
}

bool shell_energy(double gamma3, double charge, double& result) {
  if (charge == 0.0 || gamma3 == 0.0) {
    result = 0.0;
    return true;
  }
  const double square = charge * charge;
  const double cube = square * charge;
  const double scaled = cube * gamma3;
  result = scaled / 3.0;
  if (std::isnormal(square) && std::isnormal(cube) && std::isnormal(gamma3) &&
      std::isnormal(scaled) && std::isnormal(result)) {
    return true;
  }
  const long double wide = static_cast<long double>(charge) * static_cast<long double>(charge) *
                           static_cast<long double>(charge) * static_cast<long double>(gamma3) /
                           3.0L;
  if (!std::isfinite(wide) ||
      std::abs(wide) > static_cast<long double>(std::numeric_limits<double>::max())) {
    return false;
  }
  result = static_cast<double>(wide);
  return std::isfinite(result);
}

gpuxtb_status_t validate_charges(ES3View view, const double* shell_charges, std::string& error) {
  if (shell_charges == nullptr) {
    error = "ES3 shell charges must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t shell = 0; shell < view.total_shells; ++shell) {
    if (!std::isfinite(shell_charges[shell])) {
      error = "ES3 shell charges contain NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace

gpuxtb_status_t make_es3_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                              ES3Plan& plan, std::string& error) {
  gpuxtb_status_t status = validate_basis(basis, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (atomic_numbers == nullptr) {
    error = "ES3 atomic numbers must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  try {
    ES3Plan created;
    created.batch_size = basis.batch_size;
    created.total_shells = basis.total_shells;
    created.batch_shell_offsets = basis.batch_shell_offsets;
    created.shell_gamma3.resize(static_cast<std::size_t>(basis.total_shells));

    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::size_t atom_index = static_cast<std::size_t>(atom);
      const std::int32_t atomic_number = atomic_numbers[atom_index];
      const auto* element =
          parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number ||
          !std::isfinite(element->gam3)) {
        error = "ES3 plan contains an unsupported element or invalid gam3 parameter";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }

      const std::int64_t shell_begin = basis.atom_shell_offsets[atom_index];
      const std::int64_t shell_end = basis.atom_shell_offsets[atom_index + 1u];
      if (shell_end - shell_begin != element->shell_count ||
          element->shell_offset > parameters::gfn2::kShells.size() ||
          element->shell_count > parameters::gfn2::kShells.size() - element->shell_offset) {
        error = "ES3 atomic numbers do not match the supplied basis shell layout";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }

      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        const std::size_t shell_index = static_cast<std::size_t>(shell);
        const std::size_t local_shell = static_cast<std::size_t>(shell - shell_begin);
        const auto& parameter = parameters::gfn2::kShells[element->shell_offset + local_shell];
        const std::uint8_t angular_momentum = basis.angular_momenta[shell_index];
        if (basis.shell_to_atom[shell_index] != atom ||
            basis.principal_quantum_numbers[shell_index] != parameter.principal_quantum_number ||
            angular_momentum != parameter.angular_momentum ||
            basis.slater_exponents[shell_index] != parameter.slater || angular_momentum > 2u) {
          error = "ES3 atomic numbers do not match the supplied basis shell metadata";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        const double scale = parameters::gfn2::kGlobal.thirdorder_shell_scale[angular_momentum];
        const double gamma3 = element->gam3 * scale;
        if (!(scale > 0.0) || !std::isfinite(scale) || !std::isfinite(gamma3)) {
          error = "ES3 generated shell Gamma3 parameter is invalid";
          return GPUXTB_STATUS_INTERNAL_ERROR;
        }
        created.shell_gamma3[shell_index] = gamma3;
      }
    }

    plan = std::move(created);
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN2 ES3 plan";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN2 ES3 plan dimensions exceed host container limits";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

ES3View make_es3_view(const ES3Plan& plan) noexcept {
  const auto checked_count = [](std::size_t count) noexcept {
    return count <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())
               ? static_cast<std::int64_t>(count)
               : std::int64_t{-1};
  };
  return ES3View{plan.batch_size,
                 plan.total_shells,
                 checked_count(plan.batch_shell_offsets.size()),
                 checked_count(plan.shell_gamma3.size()),
                 plan.batch_shell_offsets.data(),
                 plan.shell_gamma3.data()};
}

gpuxtb_status_t evaluate_es3_potential_cpu(ES3View view, const double* shell_charges,
                                           double* shell_potentials, std::string& error) {
  gpuxtb_status_t status = validate_view(view, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_charges(view, shell_charges, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (shell_potentials == nullptr) {
    error = "ES3 shell potential output must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  std::size_t shell_bytes = 0;
  std::size_t offset_bytes = 0;
  if (!count_bytes(view.total_shells, sizeof(double), shell_bytes) ||
      !count_bytes(view.batch_shell_offset_count, sizeof(std::int64_t), offset_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, shell_charges, shell_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, view.shell_gamma3, shell_bytes) ||
      ranges_overlap(shell_potentials, shell_bytes, view.batch_shell_offsets, offset_bytes)) {
    error = "ES3 shell potential output must not overlap its inputs";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  /* Preflight every result before overwriting any caller-owned output. */
  for (std::int64_t shell = 0; shell < view.total_shells; ++shell) {
    double potential = 0.0;
    if (!shell_potential(view.shell_gamma3[shell], shell_charges[shell], potential)) {
      error = "ES3 shell potential arithmetic exceeded floating-point range";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t shell = 0; shell < view.total_shells; ++shell) {
    double potential = 0.0;
    (void)shell_potential(view.shell_gamma3[shell], shell_charges[shell], potential);
    shell_potentials[shell] = potential;
  }

  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_es3_energy_cpu(ES3View view, const double* shell_charges, double* energies,
                                   std::string& error) {
  gpuxtb_status_t status = validate_view(view, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_charges(view, shell_charges, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (energies == nullptr) {
    error = "ES3 energy output must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  std::size_t energy_bytes = 0;
  std::size_t shell_bytes = 0;
  std::size_t offset_bytes = 0;
  if (!count_bytes(view.batch_size, sizeof(double), energy_bytes) ||
      !count_bytes(view.total_shells, sizeof(double), shell_bytes) ||
      !count_bytes(view.batch_shell_offset_count, sizeof(std::int64_t), offset_bytes) ||
      ranges_overlap(energies, energy_bytes, shell_charges, shell_bytes) ||
      ranges_overlap(energies, energy_bytes, view.shell_gamma3, shell_bytes) ||
      ranges_overlap(energies, energy_bytes, view.batch_shell_offsets, offset_bytes)) {
    error = "ES3 energy output must not overlap its inputs";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < view.batch_size; ++batch) {
    if (!std::isfinite(energies[batch])) {
      error = "ES3 input energies contain NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  /* Preflight all batches so later overflow cannot partially update output. */
  for (std::int64_t batch = 0; batch < view.batch_size; ++batch) {
    double energy = energies[batch];
    for (std::int64_t shell = view.batch_shell_offsets[batch];
         shell < view.batch_shell_offsets[batch + 1]; ++shell) {
      double contribution = 0.0;
      if (!shell_energy(view.shell_gamma3[shell], shell_charges[shell], contribution)) {
        error = "ES3 shell energy arithmetic exceeded floating-point range";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      const double updated = energy + contribution;
      if (!std::isfinite(updated)) {
        error = "ES3 accumulated energy exceeded floating-point range";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      energy = updated;
    }
  }

  for (std::int64_t batch = 0; batch < view.batch_size; ++batch) {
    double energy = energies[batch];
    for (std::int64_t shell = view.batch_shell_offsets[batch];
         shell < view.batch_shell_offsets[batch + 1]; ++shell) {
      double contribution = 0.0;
      (void)shell_energy(view.shell_gamma3[shell], shell_charges[shell], contribution);
      energy += contribution;
    }
    energies[batch] = energy;
  }

  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_es3_energy_system_cpu(ES3View view, std::int64_t system,
                                          const double* shell_charges, double& accumulated_energy,
                                          std::string& error) {
  std::int64_t shell_begin = 0;
  std::int64_t shell_end = 0;
  gpuxtb_status_t status = validate_system_view(view, system, shell_begin, shell_end, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (!is_aligned(shell_charges, alignof(double))) {
    error = "ES3 shell charges must not be NULL or misaligned";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  std::size_t shell_bytes = 0;
  std::size_t offset_bytes = 0;
  if (!count_bytes(view.total_shells, sizeof(double), shell_bytes) ||
      !count_bytes(view.batch_shell_offset_count, sizeof(std::int64_t), offset_bytes) ||
      ranges_overlap(&accumulated_energy, sizeof(double), shell_charges, shell_bytes) ||
      ranges_overlap(&accumulated_energy, sizeof(double), view.shell_gamma3, shell_bytes) ||
      ranges_overlap(&accumulated_energy, sizeof(double), view.batch_shell_offsets, offset_bytes) ||
      ranges_overlap(&accumulated_energy, sizeof(double), &error, sizeof(error))) {
    error = "ES3 one-system energy output must not overlap inputs or error storage";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  double energy = accumulated_energy;
  if (!std::isfinite(energy)) {
    error = "ES3 target-system accumulated energy contains NaN or infinity";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
    const double gamma3 = view.shell_gamma3[shell];
    const double charge = shell_charges[shell];
    double contribution = 0.0;
    if (!std::isfinite(gamma3) || !std::isfinite(charge) ||
        !shell_energy(gamma3, charge, contribution) || !std::isfinite(energy + contribution)) {
      error = "ES3 target-system energy contains invalid numerical data or overflowed";
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    energy += contribution;
  }
  accumulated_energy = energy;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail::gfn2
