// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/basis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "data/parameters/gfn1.hpp"
#include "model/common/sto.hpp"

namespace xtbloom::detail::gfn1 {
namespace {

bool count_fits_vector(std::int64_t count, std::size_t element_size, bool add_sentinel = false) {
  if (count < 0) {
    return false;
  }
  const auto value = static_cast<std::uint64_t>(count);
  const auto extra = add_sentinel ? std::uint64_t{1} : std::uint64_t{0};
  if (value > std::numeric_limits<std::uint64_t>::max() - extra) {
    return false;
  }
  const auto length = value + extra;
  return length <=
             static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / element_size &&
         length <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

bool checked_add(std::int64_t increment, std::int64_t& total) {
  if (increment < 0 || total > std::numeric_limits<std::int64_t>::max() - increment) {
    return false;
  }
  total += increment;
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

bool validate_shell(const parameters::gfn1::ShellParameters& shell) {
  const double* alpha = nullptr;
  const double* coeff = nullptr;
  return shell.angular_momentum <= 4u && shell.gaussian_count >= 1u && shell.gaussian_count <= 6u &&
         shell.slater > 0.0 && std::isfinite(shell.slater) &&
         common::sto_table(shell.principal_quantum_number, shell.angular_momentum,
                           shell.gaussian_count, alpha, coeff);
}

}  // namespace

xtbloom_status_t make_basis_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                 const std::int64_t* atom_offsets,
                                 const std::int32_t* atomic_numbers, BasisPlan& plan,
                                 std::string& error) {
  if (batch_size <= 0 || total_atoms <= 0 ||
      !count_fits_vector(batch_size, sizeof(std::int64_t), true) ||
      !count_fits_vector(total_atoms, sizeof(std::int64_t), true)) {
    error = "basis plan requires positive, representable batch and atom counts";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets == nullptr || atomic_numbers == nullptr) {
    error = "basis plan offsets and atomic numbers must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets[0] != 0 || atom_offsets[batch_size] != total_atoms) {
    error = "basis plan offsets must start at zero and end at total_atoms";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    if (atom_offsets[batch] < 0 || atom_offsets[batch] > atom_offsets[batch + 1] ||
        atom_offsets[batch + 1] > total_atoms) {
      error = "basis plan offsets must be a monotone ragged partition";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  try {
    BasisPlan created;
    created.batch_size = batch_size;
    created.total_atoms = total_atoms;
    created.atom_offsets.assign(atom_offsets, atom_offsets + batch_size + 1);
    created.atom_shell_offsets.resize(static_cast<std::size_t>(total_atoms + 1));
    created.atom_orbital_offsets.resize(static_cast<std::size_t>(total_atoms + 1));
    created.atom_cartesian_orbital_offsets.resize(static_cast<std::size_t>(total_atoms + 1));
    created.atom_primitive_offsets.resize(static_cast<std::size_t>(total_atoms + 1));

    for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
      const std::size_t atom_index = static_cast<std::size_t>(atom);
      created.atom_shell_offsets[atom_index] = created.total_shells;
      created.atom_orbital_offsets[atom_index] = created.total_orbitals;
      created.atom_cartesian_orbital_offsets[atom_index] = created.total_cartesian_orbitals;
      created.atom_primitive_offsets[atom_index] = created.total_primitives;

      const std::int32_t atomic_number = atomic_numbers[atom];
      const auto* element =
          parameters::gfn1::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number) {
        error = "basis plan contains an unsupported atomic number";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const auto* shells = element_shells(*element);
      if (shells == nullptr || element->shell_count == 0u) {
        error = "basis plan contains inconsistent generated element parameters";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }

      std::array<std::uint8_t, 5> first_count{};
      for (std::size_t local_shell = 0; local_shell < element->shell_count; ++local_shell) {
        const auto& shell = shells[local_shell];
        if (!validate_shell(shell)) {
          error = "basis plan contains an unsupported generated STO-nG shell";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        const std::int64_t spherical = 2 * static_cast<std::int64_t>(shell.angular_momentum) + 1;
        const std::int64_t cartesian = (static_cast<std::int64_t>(shell.angular_momentum) + 1) *
                                       (static_cast<std::int64_t>(shell.angular_momentum) + 2) / 2;
        std::int64_t primitives = shell.gaussian_count;
        const std::size_t angular_momentum = shell.angular_momentum;
        if (first_count[angular_momentum] != 0u) {
          primitives += first_count[angular_momentum];
        } else {
          first_count[angular_momentum] = shell.gaussian_count;
        }
        if (primitives > static_cast<std::int64_t>(common::kMaximumContractedPrimitives) ||
            !checked_add(1, created.total_shells) ||
            !checked_add(spherical, created.total_orbitals) ||
            !checked_add(cartesian, created.total_cartesian_orbitals) ||
            !checked_add(primitives, created.total_primitives)) {
          error = "basis plan dimensions overflow the supported index range";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        created.maximum_angular_momentum =
            std::max(created.maximum_angular_momentum, shell.angular_momentum);
      }
    }
    const std::size_t atom_count = static_cast<std::size_t>(total_atoms);
    created.atom_shell_offsets[atom_count] = created.total_shells;
    created.atom_orbital_offsets[atom_count] = created.total_orbitals;
    created.atom_cartesian_orbital_offsets[atom_count] = created.total_cartesian_orbitals;
    created.atom_primitive_offsets[atom_count] = created.total_primitives;

    if (!count_fits_vector(created.total_shells, sizeof(std::int64_t), true) ||
        !count_fits_vector(created.total_orbitals, sizeof(double)) ||
        !count_fits_vector(created.total_cartesian_orbitals, sizeof(double)) ||
        !count_fits_vector(created.total_primitives, sizeof(double))) {
      error = "basis plan dimensions are not representable by host containers";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    const std::size_t shell_count = static_cast<std::size_t>(created.total_shells);
    const std::size_t primitive_count = static_cast<std::size_t>(created.total_primitives);
    created.shell_orbital_offsets.resize(shell_count + 1);
    created.shell_cartesian_orbital_offsets.resize(shell_count + 1);
    created.shell_primitive_offsets.resize(shell_count + 1);
    created.shell_to_atom.resize(shell_count);
    created.principal_quantum_numbers.resize(shell_count);
    created.angular_momenta.resize(shell_count);
    created.shell_is_valence.resize(shell_count);
    created.slater_exponents.resize(shell_count);
    created.primitive_exponents.resize(primitive_count);
    created.primitive_coefficients.resize(primitive_count);

    std::int64_t shell_index = 0;
    std::int64_t orbital_index = 0;
    std::int64_t cartesian_index = 0;
    std::int64_t primitive_index = 0;
    double minimum_alpha = std::numeric_limits<double>::infinity();
    for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
      const auto* element =
          parameters::gfn1::find_element(static_cast<std::uint32_t>(atomic_numbers[atom]));
      const auto* shells = element_shells(*element);
      std::array<std::int64_t, 5> first_shell;
      first_shell.fill(-1);

      for (std::size_t local_shell = 0; local_shell < element->shell_count; ++local_shell) {
        const auto& shell = shells[local_shell];
        const std::size_t current_shell = static_cast<std::size_t>(shell_index);
        const std::size_t current_primitive = static_cast<std::size_t>(primitive_index);
        created.shell_orbital_offsets[current_shell] = orbital_index;
        created.shell_cartesian_orbital_offsets[current_shell] = cartesian_index;
        created.shell_primitive_offsets[current_shell] = primitive_index;
        created.shell_to_atom[current_shell] = atom;
        created.principal_quantum_numbers[current_shell] = shell.principal_quantum_number;
        created.angular_momenta[current_shell] = shell.angular_momentum;
        created.shell_is_valence[current_shell] = shell.is_valence ? 1u : 0u;
        created.slater_exponents[current_shell] = shell.slater;

        double* alpha = created.primitive_exponents.data() + current_primitive;
        double* coeff = created.primitive_coefficients.data() + current_primitive;
        common::expand_sto_shell(shell.principal_quantum_number, shell.angular_momentum,
                                 shell.gaussian_count, shell.slater, alpha, coeff);
        std::size_t actual_count = shell.gaussian_count;

        const std::size_t l = shell.angular_momentum;
        if (first_shell[l] >= 0) {
          const std::size_t first = static_cast<std::size_t>(first_shell[l]);
          const std::size_t first_begin =
              static_cast<std::size_t>(created.shell_primitive_offsets[first]);
          const std::size_t first_count = static_cast<std::size_t>(
              created.shell_primitive_offsets[first + 1] - created.shell_primitive_offsets[first]);
          common::orthogonalize_to_first(created.primitive_exponents.data() + first_begin,
                                         created.primitive_coefficients.data() + first_begin,
                                         first_count, alpha, coeff, shell.gaussian_count);
          actual_count += first_count;
        } else {
          first_shell[l] = shell_index;
        }
        for (std::size_t primitive = 0; primitive < actual_count; ++primitive) {
          minimum_alpha = std::min(minimum_alpha, alpha[primitive]);
        }

        ++shell_index;
        orbital_index += 2 * static_cast<std::int64_t>(shell.angular_momentum) + 1;
        cartesian_index += (static_cast<std::int64_t>(shell.angular_momentum) + 1) *
                           (static_cast<std::int64_t>(shell.angular_momentum) + 2) / 2;
        primitive_index += static_cast<std::int64_t>(actual_count);
        created.shell_primitive_offsets[static_cast<std::size_t>(shell_index)] = primitive_index;
      }
    }
    created.shell_orbital_offsets[shell_count] = orbital_index;
    created.shell_cartesian_orbital_offsets[shell_count] = cartesian_index;
    created.shell_primitive_offsets[shell_count] = primitive_index;
    created.minimum_primitive_exponent = minimum_alpha;

    const std::size_t batch_count = static_cast<std::size_t>(batch_size);
    created.batch_shell_offsets.resize(batch_count + 1);
    created.batch_orbital_offsets.resize(batch_count + 1);
    created.batch_cartesian_orbital_offsets.resize(batch_count + 1);
    created.batch_primitive_offsets.resize(batch_count + 1);
    for (std::int64_t batch = 0; batch <= batch_size; ++batch) {
      const std::size_t batch_index = static_cast<std::size_t>(batch);
      const std::size_t atom_index = static_cast<std::size_t>(atom_offsets[batch]);
      created.batch_shell_offsets[batch_index] = created.atom_shell_offsets[atom_index];
      created.batch_orbital_offsets[batch_index] = created.atom_orbital_offsets[atom_index];
      created.batch_cartesian_orbital_offsets[batch_index] =
          created.atom_cartesian_orbital_offsets[atom_index];
      created.batch_primitive_offsets[batch_index] = created.atom_primitive_offsets[atom_index];
    }

    plan = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 basis plan";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN1 basis plan dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

}  // namespace xtbloom::detail::gfn1
