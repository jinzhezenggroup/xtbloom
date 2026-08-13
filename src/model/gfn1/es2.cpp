// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn1/es2.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#include "data/parameters/gfn1.hpp"

namespace xtbloom::detail::gfn1 {
namespace {

static_assert(parameters::gfn1::kGlobal.charge_average == 1u,
              "GFN1 ES2 requires harmonic shell-hardness averaging");
static_assert(parameters::gfn1::kGlobal.charge_gexp == 2.0,
              "the shared ES2 kernel specializes the GFN1 gexp=2 form");

bool representable_count(std::int64_t value, std::size_t element_size, bool add_sentinel = false) {
  if (value < 0) {
    return false;
  }
  const std::uint64_t count = static_cast<std::uint64_t>(value);
  const std::uint64_t extra = add_sentinel ? 1u : 0u;
  return count <= std::numeric_limits<std::uint64_t>::max() - extra &&
         count + extra <= std::numeric_limits<std::size_t>::max() / element_size;
}

}  // namespace

xtbloom_status_t make_es2_plan(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                               ES2Plan& plan, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      !representable_count(basis.total_atoms, sizeof(std::int64_t), true) ||
      !representable_count(basis.total_shells, sizeof(double)) || atomic_numbers == nullptr ||
      basis.atom_shell_offsets.size() != static_cast<std::size_t>(basis.total_atoms) + 1u ||
      basis.shell_to_atom.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.principal_quantum_numbers.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.angular_momenta.size() != static_cast<std::size_t>(basis.total_shells) ||
      basis.slater_exponents.size() != static_cast<std::size_t>(basis.total_shells)) {
    error = "GFN1 ES2 plan requires one complete representable basis and element list";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    std::vector<double> hardness(static_cast<std::size_t>(basis.total_shells));
    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::size_t atom_index = static_cast<std::size_t>(atom);
      const std::int32_t atomic_number = atomic_numbers[atom_index];
      const auto* element =
          parameters::gfn1::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number || !(element->gam > 0.0) ||
          !std::isfinite(element->gam)) {
        error = "GFN1 ES2 plan contains an unsupported element or invalid atomic hardness";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      const std::int64_t shell_begin = basis.atom_shell_offsets[atom_index];
      const std::int64_t shell_end = basis.atom_shell_offsets[atom_index + 1u];
      const std::size_t parameter_begin = element->shell_offset;
      if (shell_begin < 0 || shell_begin >= shell_end || shell_end > basis.total_shells ||
          shell_end - shell_begin != element->shell_count ||
          parameter_begin > parameters::gfn1::kShells.size() ||
          element->shell_count > parameters::gfn1::kShells.size() - parameter_begin) {
        error = "GFN1 ES2 element list does not match the basis shell layout";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        const std::size_t shell_index = static_cast<std::size_t>(shell);
        const std::size_t local_shell = static_cast<std::size_t>(shell - shell_begin);
        const auto& parameter = parameters::gfn1::kShells[parameter_begin + local_shell];
        if (basis.shell_to_atom[shell_index] != atom ||
            basis.principal_quantum_numbers[shell_index] != parameter.principal_quantum_number ||
            basis.angular_momenta[shell_index] != parameter.angular_momentum ||
            basis.slater_exponents[shell_index] != parameter.slater ||
            !(parameter.shell_hubbard_scale > 0.0) ||
            !std::isfinite(parameter.shell_hubbard_scale)) {
          error = "GFN1 ES2 element list does not match the basis shell metadata";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        const double shell_hardness = element->gam * parameter.shell_hubbard_scale;
        if (!(shell_hardness > 0.0) || !std::isfinite(shell_hardness)) {
          error = "GFN1 ES2 generated shell hardness is invalid";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        hardness[shell_index] = shell_hardness;
      }
    }
    return gfn2::make_es2_plan_from_shell_hardness(basis, gfn2::ES2HardnessAverage::kHarmonic,
                                                   hardness.data(), basis.total_shells, plan,
                                                   error);
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN1 ES2 parameter expansion";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN1 ES2 parameter dimensions exceed host container limits";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

}  // namespace xtbloom::detail::gfn1
