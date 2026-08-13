// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/h0.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "data/parameters/gfn1.hpp"

namespace xtbloom::detail::gfn1 {
namespace {

/* GFN1 retains tblite/xTB's historical parameter-file conversion. */
constexpr double kElectronvoltToHartree = 1.0 / 27.21138505;
constexpr double kDerivativeDistanceSquaredCutoff = std::numeric_limits<double>::epsilon();

bool representable_as_size(std::int64_t value) {
  return value >= 0 && static_cast<std::uint64_t>(value) <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
}

bool checked_square(std::int64_t value, std::int64_t& square) {
  if (value < 0 || (value > 0 && value > std::numeric_limits<std::int64_t>::max() / value)) {
    return false;
  }
  square = value * value;
  return true;
}

const parameters::gfn1::ShellParameters* element_shells(
    const parameters::gfn1::ElementParameters& element) {
  const std::size_t begin = element.shell_offset;
  const std::size_t count = element.shell_count;
  if (begin > parameters::gfn1::kShells.size() ||
      count > parameters::gfn1::kShells.size() - begin) {
    return nullptr;
  }
  return parameters::gfn1::kShells.data() + begin;
}

xtbloom_status_t validate_basis_and_integrals(const BasisPlan& basis, const IntegralPlan& integrals,
                                              std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      basis.total_orbitals <= 0 || !representable_as_size(basis.batch_size) ||
      !representable_as_size(basis.total_atoms) || !representable_as_size(basis.total_shells) ||
      !representable_as_size(basis.total_orbitals) ||
      static_cast<std::uint64_t>(basis.total_atoms) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / 3u) ||
      integrals.total_matrix_elements < 0 ||
      !representable_as_size(integrals.total_matrix_elements)) {
    error = "GFN1 H0 requires a positive, representable basis plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const auto batch_count = static_cast<std::size_t>(basis.batch_size);
  const auto atom_count = static_cast<std::size_t>(basis.total_atoms);
  const auto shell_count = static_cast<std::size_t>(basis.total_shells);
  if (basis.atom_offsets.size() != batch_count + 1u ||
      basis.batch_shell_offsets.size() != batch_count + 1u ||
      basis.batch_orbital_offsets.size() != batch_count + 1u ||
      basis.atom_shell_offsets.size() != atom_count + 1u ||
      basis.shell_orbital_offsets.size() != shell_count + 1u ||
      basis.shell_to_atom.size() != shell_count ||
      basis.principal_quantum_numbers.size() != shell_count ||
      basis.angular_momenta.size() != shell_count || basis.slater_exponents.size() != shell_count ||
      basis.shell_is_valence.size() != shell_count) {
    error = "GFN1 H0 basis plan is incomplete or internally inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (basis.atom_offsets.front() != 0 || basis.atom_offsets.back() != basis.total_atoms ||
      basis.batch_shell_offsets.front() != 0 ||
      basis.batch_shell_offsets.back() != basis.total_shells ||
      basis.batch_orbital_offsets.front() != 0 ||
      basis.batch_orbital_offsets.back() != basis.total_orbitals ||
      basis.atom_shell_offsets.front() != 0 ||
      basis.atom_shell_offsets.back() != basis.total_shells ||
      basis.shell_orbital_offsets.front() != 0 ||
      basis.shell_orbital_offsets.back() != basis.total_orbitals) {
    error = "GFN1 H0 basis offsets do not span the stored dimensions";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (integrals.batch_size != basis.batch_size ||
      integrals.matrix_offsets.size() != batch_count + 1u ||
      integrals.matrix_offsets.front() != 0 ||
      integrals.matrix_offsets.back() != integrals.total_matrix_elements) {
    error = "GFN1 H0 integral plan is incompatible with the basis plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const std::int64_t atom_begin = basis.atom_offsets[batch];
    const std::int64_t atom_end = basis.atom_offsets[batch + 1u];
    const std::int64_t shell_begin = basis.batch_shell_offsets[batch];
    const std::int64_t shell_end = basis.batch_shell_offsets[batch + 1u];
    const std::int64_t orbital_begin = basis.batch_orbital_offsets[batch];
    const std::int64_t orbital_end = basis.batch_orbital_offsets[batch + 1u];
    std::int64_t matrix_size = 0;
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > basis.total_atoms ||
        shell_begin < 0 || shell_begin > shell_end || shell_end > basis.total_shells ||
        orbital_begin < 0 || orbital_begin > orbital_end || orbital_end > basis.total_orbitals ||
        basis.atom_shell_offsets[static_cast<std::size_t>(atom_begin)] != shell_begin ||
        basis.atom_shell_offsets[static_cast<std::size_t>(atom_end)] != shell_end ||
        basis.shell_orbital_offsets[static_cast<std::size_t>(shell_begin)] != orbital_begin ||
        basis.shell_orbital_offsets[static_cast<std::size_t>(shell_end)] != orbital_end ||
        !checked_square(orbital_end - orbital_begin, matrix_size) ||
        integrals.matrix_offsets[batch] < 0 ||
        integrals.matrix_offsets[batch] > integrals.matrix_offsets[batch + 1u] ||
        integrals.matrix_offsets[batch + 1u] > integrals.total_matrix_elements ||
        integrals.matrix_offsets[batch + 1u] - integrals.matrix_offsets[batch] != matrix_size) {
      error = "GFN1 H0 ragged basis or matrix offsets are invalid";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    const std::int64_t begin = basis.atom_shell_offsets[atom];
    const std::int64_t end = basis.atom_shell_offsets[atom + 1u];
    if (begin < 0 || begin >= end || end > basis.total_shells) {
      error = "GFN1 H0 atom-to-shell offsets are invalid";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t shell = begin; shell < end; ++shell) {
      if (basis.shell_to_atom[static_cast<std::size_t>(shell)] != static_cast<std::int64_t>(atom)) {
        error = "GFN1 H0 shell ownership is inconsistent with atom-to-shell offsets";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  for (std::size_t shell = 0; shell < shell_count; ++shell) {
    if (basis.shell_to_atom[shell] < 0 || basis.shell_to_atom[shell] >= basis.total_atoms ||
        basis.shell_orbital_offsets[shell] < 0 ||
        basis.shell_orbital_offsets[shell] > basis.shell_orbital_offsets[shell + 1u] ||
        basis.shell_orbital_offsets[shell + 1u] > basis.total_orbitals ||
        basis.angular_momenta[shell] > 2u || !(basis.slater_exponents[shell] > 0.0) ||
        !std::isfinite(basis.slater_exponents[shell]) || basis.shell_is_valence[shell] > 1u) {
      error = "GFN1 H0 shell metadata is invalid";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_plan(const BasisPlan& basis, const IntegralPlan& integrals,
                               const H0Plan& plan, std::string& error) {
  xtbloom_status_t status = validate_basis_and_integrals(basis, integrals, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (plan.batch_size != basis.batch_size || plan.total_atoms != basis.total_atoms ||
      plan.total_shells != basis.total_shells || plan.total_orbitals != basis.total_orbitals ||
      plan.total_matrix_elements != integrals.total_matrix_elements ||
      plan.atom_offsets != basis.atom_offsets ||
      plan.batch_shell_offsets != basis.batch_shell_offsets ||
      plan.batch_orbital_offsets != basis.batch_orbital_offsets ||
      plan.matrix_offsets != integrals.matrix_offsets) {
    error = "GFN1 H0 plan is incompatible with the supplied basis or integral plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const auto batch_count = static_cast<std::size_t>(plan.batch_size);
  const auto atom_count = static_cast<std::size_t>(plan.total_atoms);
  const auto shell_count = static_cast<std::size_t>(plan.total_shells);
  if (plan.shell_pair_offsets.size() != batch_count + 1u ||
      plan.atomic_radii.size() != atom_count || plan.shell_levels.size() != shell_count ||
      plan.shell_coordination_scale.size() != shell_count ||
      plan.shell_polynomial.size() != shell_count || plan.shell_pair_offsets.front() != 0 ||
      !representable_as_size(plan.shell_pair_offsets.back()) ||
      static_cast<std::size_t>(plan.shell_pair_offsets.back()) != plan.shell_pair_scale.size()) {
    error = "GFN1 H0 plan is incomplete or internally inconsistent";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const std::int64_t shells =
        plan.batch_shell_offsets[batch + 1u] - plan.batch_shell_offsets[batch];
    std::int64_t expected = 0;
    if (!checked_square(shells, expected) || plan.shell_pair_offsets[batch] < 0 ||
        plan.shell_pair_offsets[batch + 1u] - plan.shell_pair_offsets[batch] != expected) {
      error = "GFN1 H0 shell-pair offsets do not match the ragged shell dimensions";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (double radius : plan.atomic_radii) {
    if (!(radius > 0.0) || !std::isfinite(radius)) {
      error = "GFN1 H0 plan contains an invalid atomic radius";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t shell = 0; shell < shell_count; ++shell) {
    if (!std::isfinite(plan.shell_levels[shell]) ||
        !std::isfinite(plan.shell_coordination_scale[shell]) ||
        !std::isfinite(plan.shell_polynomial[shell])) {
      error = "GFN1 H0 plan contains an invalid shell parameter";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (double scale : plan.shell_pair_scale) {
    if (!(scale > 0.0) || !std::isfinite(scale)) {
      error = "GFN1 H0 plan contains an invalid shell-pair scale";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_evaluation_inputs(const BasisPlan& basis, const IntegralPlan& integrals,
                                            const H0Plan& plan, const double* positions,
                                            const double* coordination_numbers,
                                            const double* overlap, std::string& error) {
  xtbloom_status_t status = validate_plan(basis, integrals, plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (positions == nullptr || coordination_numbers == nullptr || overlap == nullptr) {
    error = "GFN1 H0 positions, coordination numbers, and overlap must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const auto atom_count = static_cast<std::size_t>(plan.total_atoms);
  for (std::size_t coordinate = 0; coordinate < atom_count * 3u; ++coordinate) {
    if (!std::isfinite(positions[coordinate])) {
      error = "GFN1 H0 positions contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    if (!std::isfinite(coordination_numbers[atom])) {
      error = "GFN1 H0 coordination numbers contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t element = 0; element < plan.total_matrix_elements; ++element) {
    if (!std::isfinite(overlap[element])) {
      error = "GFN1 H0 overlap contains NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

xtbloom_status_t make_h0_plan(const BasisPlan& basis, const IntegralPlan& integrals,
                              const std::int32_t* atomic_numbers, H0Plan& plan,
                              std::string& error) {
  xtbloom_status_t status = validate_basis_and_integrals(basis, integrals, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (atomic_numbers == nullptr) {
    error = "GFN1 H0 atomic numbers must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    H0Plan created;
    created.batch_size = basis.batch_size;
    created.total_atoms = basis.total_atoms;
    created.total_shells = basis.total_shells;
    created.total_orbitals = basis.total_orbitals;
    created.total_matrix_elements = integrals.total_matrix_elements;
    created.atom_offsets = basis.atom_offsets;
    created.batch_shell_offsets = basis.batch_shell_offsets;
    created.batch_orbital_offsets = basis.batch_orbital_offsets;
    created.matrix_offsets = integrals.matrix_offsets;
    created.atomic_radii.resize(static_cast<std::size_t>(basis.total_atoms));
    created.shell_levels.resize(static_cast<std::size_t>(basis.total_shells));
    created.shell_coordination_scale.resize(static_cast<std::size_t>(basis.total_shells));
    created.shell_polynomial.resize(static_cast<std::size_t>(basis.total_shells));

    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::int32_t atomic_number = atomic_numbers[atom];
      const auto* element =
          parameters::gfn1::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number ||
          !(element->atomic_radius_bohr > 0.0) || !std::isfinite(element->atomic_radius_bohr)) {
        error = "GFN1 H0 plan contains an unsupported element or invalid radius";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const auto* shells = element_shells(*element);
      const auto atom_index = static_cast<std::size_t>(atom);
      const std::int64_t shell_begin = basis.atom_shell_offsets[atom_index];
      const std::int64_t shell_end = basis.atom_shell_offsets[atom_index + 1u];
      if (shells == nullptr || shell_end - shell_begin != element->shell_count) {
        error = "GFN1 H0 atomic numbers do not match the supplied basis plan";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      created.atomic_radii[atom_index] = element->atomic_radius_bohr;

      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        const std::size_t local_shell = static_cast<std::size_t>(shell - shell_begin);
        const auto& parameter = shells[local_shell];
        const auto shell_index = static_cast<std::size_t>(shell);
        if (parameter.angular_momentum != basis.angular_momenta[shell_index] ||
            parameter.principal_quantum_number != basis.principal_quantum_numbers[shell_index] ||
            parameter.slater != basis.slater_exponents[shell_index] ||
            static_cast<std::uint8_t>(parameter.is_valence) !=
                basis.shell_is_valence[shell_index]) {
          error = "GFN1 H0 shell parameters do not match the supplied basis plan";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        created.shell_levels[shell_index] = parameter.level_electronvolt * kElectronvoltToHartree;
        created.shell_coordination_scale[shell_index] =
            parameter.coordination_number_scale_electronvolt * kElectronvoltToHartree;
        created.shell_polynomial[shell_index] = parameter.shell_polynomial;
      }
    }

    created.shell_pair_offsets.resize(static_cast<std::size_t>(basis.batch_size) + 1u);
    created.shell_pair_offsets[0] = 0;
    for (std::int64_t batch = 0; batch < basis.batch_size; ++batch) {
      const std::int64_t shell_count =
          basis.batch_shell_offsets[static_cast<std::size_t>(batch + 1)] -
          basis.batch_shell_offsets[static_cast<std::size_t>(batch)];
      std::int64_t pair_count = 0;
      if (!checked_square(shell_count, pair_count) ||
          created.shell_pair_offsets[static_cast<std::size_t>(batch)] >
              std::numeric_limits<std::int64_t>::max() - pair_count) {
        error = "GFN1 H0 shell-pair storage size overflows int64";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      created.shell_pair_offsets[static_cast<std::size_t>(batch + 1)] =
          created.shell_pair_offsets[static_cast<std::size_t>(batch)] + pair_count;
    }
    if (!representable_as_size(created.shell_pair_offsets.back())) {
      error = "GFN1 H0 shell-pair storage is not representable on this platform";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    created.shell_pair_scale.resize(static_cast<std::size_t>(created.shell_pair_offsets.back()));

    for (std::int64_t batch = 0; batch < basis.batch_size; ++batch) {
      const std::int64_t shell_begin = basis.batch_shell_offsets[static_cast<std::size_t>(batch)];
      const std::int64_t shell_end = basis.batch_shell_offsets[static_cast<std::size_t>(batch + 1)];
      const std::int64_t molecule_shells = shell_end - shell_begin;
      const std::int64_t pair_begin = created.shell_pair_offsets[static_cast<std::size_t>(batch)];
      for (std::int64_t first = shell_begin; first < shell_end; ++first) {
        const auto first_index = static_cast<std::size_t>(first);
        const auto first_atom = static_cast<std::size_t>(basis.shell_to_atom[first_index]);
        const std::int32_t first_number = atomic_numbers[first_atom];
        const auto* first_element =
            parameters::gfn1::find_element(static_cast<std::uint32_t>(first_number));
        for (std::int64_t second = shell_begin; second < shell_end; ++second) {
          const auto second_index = static_cast<std::size_t>(second);
          const auto second_atom = static_cast<std::size_t>(basis.shell_to_atom[second_index]);
          const std::int32_t second_number = atomic_numbers[second_atom];
          const auto* second_element =
              parameters::gfn1::find_element(static_cast<std::uint32_t>(second_number));

          double scale = parameters::gfn1::kGlobal.hamiltonian_kpol;
          const bool first_valence = basis.shell_is_valence[first_index] != 0u;
          const bool second_valence = basis.shell_is_valence[second_index] != 0u;
          if (first_valence && second_valence) {
            const std::size_t angular_index =
                static_cast<std::size_t>(basis.angular_momenta[first_index]) * 3u +
                static_cast<std::size_t>(basis.angular_momenta[second_index]);
            const double electronegativity_difference =
                first_element->electronegativity - second_element->electronegativity;
            const double electronegativity_scale =
                1.0 + parameters::gfn1::kGlobal.hamiltonian_enscale * electronegativity_difference *
                          electronegativity_difference;
            scale = parameters::gfn1::kGlobal.shell_pair_scale[angular_index] *
                    parameters::gfn1::pair_scale(static_cast<std::uint32_t>(first_number),
                                                 static_cast<std::uint32_t>(second_number)) *
                    electronegativity_scale;
          } else if (first_valence) {
            const std::size_t diagonal =
                static_cast<std::size_t>(basis.angular_momenta[first_index]) * 3u +
                static_cast<std::size_t>(basis.angular_momenta[first_index]);
            scale = 0.5 * (parameters::gfn1::kGlobal.shell_pair_scale[diagonal] +
                           parameters::gfn1::kGlobal.hamiltonian_kpol);
          } else if (second_valence) {
            const std::size_t diagonal =
                static_cast<std::size_t>(basis.angular_momenta[second_index]) * 3u +
                static_cast<std::size_t>(basis.angular_momenta[second_index]);
            scale = 0.5 * (parameters::gfn1::kGlobal.shell_pair_scale[diagonal] +
                           parameters::gfn1::kGlobal.hamiltonian_kpol);
          }

          const std::int64_t local_first = first - shell_begin;
          const std::int64_t local_second = second - shell_begin;
          created.shell_pair_scale[static_cast<std::size_t>(
              pair_begin + local_first * molecule_shells + local_second)] = scale;
        }
      }
    }

    plan = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 H0 plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN1 H0 plan dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t evaluate_h0_cpu(const BasisPlan& basis, const IntegralPlan& integrals,
                                 const H0Plan& plan, const double* positions,
                                 const double* coordination_numbers, const double* overlap,
                                 double* hamiltonian, std::string& error) {
  xtbloom_status_t status = validate_evaluation_inputs(basis, integrals, plan, positions,
                                                       coordination_numbers, overlap, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (hamiltonian == nullptr) {
    error = "GFN1 H0 Hamiltonian output must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const auto batch_index = static_cast<std::size_t>(batch);
    const std::int64_t shell_begin = plan.batch_shell_offsets[batch_index];
    const std::int64_t shell_end = plan.batch_shell_offsets[batch_index + 1u];
    const std::int64_t molecule_shells = shell_end - shell_begin;
    const std::int64_t orbital_begin = plan.batch_orbital_offsets[batch_index];
    const std::int64_t orbital_end = plan.batch_orbital_offsets[batch_index + 1u];
    const std::int64_t molecule_orbitals = orbital_end - orbital_begin;
    const std::int64_t matrix_begin = plan.matrix_offsets[batch_index];
    const std::int64_t pair_begin = plan.shell_pair_offsets[batch_index];

    for (std::int64_t first = shell_begin; first < shell_end; ++first) {
      const auto first_index = static_cast<std::size_t>(first);
      const std::int64_t first_atom = basis.shell_to_atom[first_index];
      const auto first_atom_index = static_cast<std::size_t>(first_atom);
      const double first_level =
          plan.shell_levels[first_index] -
          plan.shell_coordination_scale[first_index] * coordination_numbers[first_atom];
      const std::int64_t first_orbital_begin = basis.shell_orbital_offsets[first_index];
      const std::int64_t first_orbital_end = basis.shell_orbital_offsets[first_index + 1u];

      for (std::int64_t second = shell_begin; second < shell_end; ++second) {
        const auto second_index = static_cast<std::size_t>(second);
        const std::int64_t second_atom = basis.shell_to_atom[second_index];
        const auto second_atom_index = static_cast<std::size_t>(second_atom);
        const double second_level =
            plan.shell_levels[second_index] -
            plan.shell_coordination_scale[second_index] * coordination_numbers[second_atom];
        double spatial_scale = 1.0;
        if (first_atom != second_atom) {
          const double dx = positions[first_atom_index * 3u] - positions[second_atom_index * 3u];
          const double dy =
              positions[first_atom_index * 3u + 1u] - positions[second_atom_index * 3u + 1u];
          const double dz =
              positions[first_atom_index * 3u + 2u] - positions[second_atom_index * 3u + 2u];
          const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
          const double reduced_distance =
              std::sqrt(distance / (plan.atomic_radii[first_atom_index] +
                                    plan.atomic_radii[second_atom_index]));
          const double polynomial = (1.0 + plan.shell_polynomial[first_index] * reduced_distance) *
                                    (1.0 + plan.shell_polynomial[second_index] * reduced_distance);
          const std::int64_t local_first = first - shell_begin;
          const std::int64_t local_second = second - shell_begin;
          spatial_scale = plan.shell_pair_scale[static_cast<std::size_t>(
                              pair_begin + local_first * molecule_shells + local_second)] *
                          polynomial;
        }
        const double factor = 0.5 * (first_level + second_level) * spatial_scale;
        const std::int64_t second_orbital_begin = basis.shell_orbital_offsets[second_index];
        const std::int64_t second_orbital_end = basis.shell_orbital_offsets[second_index + 1u];
        for (std::int64_t first_orbital = first_orbital_begin; first_orbital < first_orbital_end;
             ++first_orbital) {
          const std::int64_t row = first_orbital - orbital_begin;
          for (std::int64_t second_orbital = second_orbital_begin;
               second_orbital < second_orbital_end; ++second_orbital) {
            const std::int64_t column = second_orbital - orbital_begin;
            const std::int64_t matrix_index = matrix_begin + row * molecule_orbitals + column;
            hamiltonian[matrix_index] = overlap[matrix_index] * factor;
          }
        }
      }
    }
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t add_h0_vjp_cpu(const BasisPlan& basis, const IntegralPlan& integrals,
                                const H0Plan& plan, const double* positions,
                                const double* coordination_numbers, const double* overlap,
                                const double* dE_dhamiltonian, double* dE_doverlap, double* dE_dcn,
                                double* gradients, std::string& error) {
  xtbloom_status_t status = validate_evaluation_inputs(basis, integrals, plan, positions,
                                                       coordination_numbers, overlap, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (dE_dhamiltonian == nullptr || dE_doverlap == nullptr || dE_dcn == nullptr ||
      gradients == nullptr) {
    error = "GFN1 H0 VJP inputs and outputs must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t element = 0; element < plan.total_matrix_elements; ++element) {
    if (!std::isfinite(dE_dhamiltonian[element])) {
      error = "GFN1 H0 Hamiltonian derivatives contain NaN or infinity";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  /*
   * tblite skips H0 pair derivatives at r^2 <= epsilon(1.0). Validate all
   * other pair distances before touching caller adjoints so an overflowing
   * coordinate difference cannot leave a partially accumulated VJP.
   */
  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const auto batch_index = static_cast<std::size_t>(batch);
    const std::int64_t atom_begin = plan.atom_offsets[batch_index];
    const std::int64_t atom_end = plan.atom_offsets[batch_index + 1u];
    for (std::int64_t first = atom_begin; first < atom_end; ++first) {
      const auto first_index = static_cast<std::size_t>(first);
      for (std::int64_t second = atom_begin; second < first; ++second) {
        const auto second_index = static_cast<std::size_t>(second);
        const double dx = positions[first_index * 3u] - positions[second_index * 3u];
        const double dy = positions[first_index * 3u + 1u] - positions[second_index * 3u + 1u];
        const double dz = positions[first_index * 3u + 2u] - positions[second_index * 3u + 2u];
        const double distance_squared = dx * dx + dy * dy + dz * dz;
        if (!std::isfinite(distance_squared)) {
          error = "GFN1 H0 coordinate differences overflow floating-point range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
      }
    }
  }

  for (std::int64_t batch = 0; batch < plan.batch_size; ++batch) {
    const auto batch_index = static_cast<std::size_t>(batch);
    const std::int64_t shell_begin = plan.batch_shell_offsets[batch_index];
    const std::int64_t shell_end = plan.batch_shell_offsets[batch_index + 1u];
    const std::int64_t molecule_shells = shell_end - shell_begin;
    const std::int64_t orbital_begin = plan.batch_orbital_offsets[batch_index];
    const std::int64_t orbital_end = plan.batch_orbital_offsets[batch_index + 1u];
    const std::int64_t molecule_orbitals = orbital_end - orbital_begin;
    const std::int64_t matrix_begin = plan.matrix_offsets[batch_index];
    const std::int64_t pair_begin = plan.shell_pair_offsets[batch_index];

    for (std::int64_t first = shell_begin; first < shell_end; ++first) {
      const auto first_index = static_cast<std::size_t>(first);
      const std::int64_t first_atom = basis.shell_to_atom[first_index];
      const auto first_atom_index = static_cast<std::size_t>(first_atom);
      const double first_level =
          plan.shell_levels[first_index] -
          plan.shell_coordination_scale[first_index] * coordination_numbers[first_atom];
      const std::int64_t first_orbital_begin = basis.shell_orbital_offsets[first_index];
      const std::int64_t first_orbital_end = basis.shell_orbital_offsets[first_index + 1u];

      for (std::int64_t second = shell_begin; second < shell_end; ++second) {
        const auto second_index = static_cast<std::size_t>(second);
        const std::int64_t second_atom = basis.shell_to_atom[second_index];
        const auto second_atom_index = static_cast<std::size_t>(second_atom);
        const double second_level =
            plan.shell_levels[second_index] -
            plan.shell_coordination_scale[second_index] * coordination_numbers[second_atom];
        const double average_level = 0.5 * (first_level + second_level);
        double spatial_scale = 1.0;
        double spatial_scale_derivative = 0.0;
        double dx = 0.0;
        double dy = 0.0;
        double dz = 0.0;
        double distance = 0.0;
        if (first_atom != second_atom) {
          dx = positions[first_atom_index * 3u] - positions[second_atom_index * 3u];
          dy = positions[first_atom_index * 3u + 1u] - positions[second_atom_index * 3u + 1u];
          dz = positions[first_atom_index * 3u + 2u] - positions[second_atom_index * 3u + 2u];
          const double distance_squared = dx * dx + dy * dy + dz * dz;
          if (distance_squared <= kDerivativeDistanceSquaredCutoff) {
            /* The reference omits every H0 contribution to this pair's VJP. */
            continue;
          }
          distance = std::sqrt(distance_squared);
          const double reduced_distance =
              std::sqrt(distance / (plan.atomic_radii[first_atom_index] +
                                    plan.atomic_radii[second_atom_index]));
          const double first_polynomial =
              1.0 + plan.shell_polynomial[first_index] * reduced_distance;
          const double second_polynomial =
              1.0 + plan.shell_polynomial[second_index] * reduced_distance;
          const std::int64_t local_first = first - shell_begin;
          const std::int64_t local_second = second - shell_begin;
          const double pair_scale = plan.shell_pair_scale[static_cast<std::size_t>(
              pair_begin + local_first * molecule_shells + local_second)];
          spatial_scale = pair_scale * first_polynomial * second_polynomial;
          const double polynomial_derivative =
              (plan.shell_polynomial[first_index] * second_polynomial +
               plan.shell_polynomial[second_index] * first_polynomial) *
              reduced_distance / (2.0 * distance);
          spatial_scale_derivative = pair_scale * polynomial_derivative;
        }

        double block_weight = 0.0;
        const double factor = average_level * spatial_scale;
        const std::int64_t second_orbital_begin = basis.shell_orbital_offsets[second_index];
        const std::int64_t second_orbital_end = basis.shell_orbital_offsets[second_index + 1u];
        for (std::int64_t first_orbital = first_orbital_begin; first_orbital < first_orbital_end;
             ++first_orbital) {
          const std::int64_t row = first_orbital - orbital_begin;
          for (std::int64_t second_orbital = second_orbital_begin;
               second_orbital < second_orbital_end; ++second_orbital) {
            const std::int64_t column = second_orbital - orbital_begin;
            const std::int64_t matrix_index = matrix_begin + row * molecule_orbitals + column;
            const double adjoint = dE_dhamiltonian[matrix_index];
            dE_doverlap[matrix_index] += adjoint * factor;
            block_weight += adjoint * overlap[matrix_index];
          }
        }

        const double level_weight = 0.5 * block_weight * spatial_scale;
        dE_dcn[first_atom] -= plan.shell_coordination_scale[first_index] * level_weight;
        dE_dcn[second_atom] -= plan.shell_coordination_scale[second_index] * level_weight;

        if (first_atom != second_atom) {
          const double radial_derivative = block_weight * average_level * spatial_scale_derivative;
          const double coordinate_scale = radial_derivative / distance;
          const double gx = coordinate_scale * dx;
          const double gy = coordinate_scale * dy;
          const double gz = coordinate_scale * dz;
          gradients[first_atom_index * 3u] += gx;
          gradients[first_atom_index * 3u + 1u] += gy;
          gradients[first_atom_index * 3u + 2u] += gz;
          gradients[second_atom_index * 3u] -= gx;
          gradients[second_atom_index * 3u + 1u] -= gy;
          gradients[second_atom_index * 3u + 2u] -= gz;
        }
      }
    }
  }

  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn1
