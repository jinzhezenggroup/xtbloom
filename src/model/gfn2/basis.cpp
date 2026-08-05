// SPDX-License-Identifier: GPL-3.0-or-later

#include "model/gfn2/basis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "data/parameters/gfn2.hpp"

namespace gpuxtb::detail::gfn2 {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoOverPi = 2.0 / kPi;
constexpr std::array<double, 5> kDoubleFactorial{1.0, 1.0, 3.0, 15.0, 105.0};
constexpr std::size_t kMaximumContractedPrimitives = 12;

/*
 * Stewart STO-3G and STO-4G tables copied from tblite_basis_slater
 * (LGPL-3.0-or-later) at the revision pinned by
 * data/parameters/manifest.json. Rows follow tblite's ityp order: 1s..5s,
 * 2p..5p, 3d..5d, 4f..5f, 5g.
 */
constexpr std::array<std::array<double, 3>, 15> kAlpha3{{
    {{2.227660584e+0, 4.057711562e-1, 1.098175104e-1}},
    {{2.581578398e+0, 1.567622104e-1, 6.018332272e-2}},
    {{5.641487709e-1, 6.924421391e-2, 3.269529097e-2}},
    {{2.267938753e-1, 4.448178019e-2, 2.195294664e-2}},
    {{1.080198458e-1, 4.408119382e-2, 2.610811810e-2}},
    {{9.192379002e-1, 2.359194503e-1, 8.009805746e-2}},
    {{2.692880368e+0, 1.489359592e-1, 5.739585040e-2}},
    {{4.859692220e-1, 7.430216918e-2, 3.653340923e-2}},
    {{2.127482317e-1, 4.729648620e-2, 2.604865324e-2}},
    {{5.229112225e-1, 1.639595876e-1, 6.386630021e-2}},
    {{1.777717219e-1, 8.040647350e-2, 3.949855551e-2}},
    {{4.913352950e-1, 7.329090601e-2, 3.594209290e-2}},
    {{3.483826963e-1, 1.249380537e-1, 5.349995725e-2}},
    {{1.649233885e-1, 7.487066646e-2, 3.735787219e-2}},
    {{2.545432122e-1, 1.006544376e-1, 4.624463922e-2}},
}};

constexpr std::array<std::array<double, 3>, 15> kCoeff3{{
    {{1.543289673e-1, 5.353281423e-1, 4.446345422e-1}},
    {{-5.994474934e-2, 5.960385398e-1, 4.581786291e-1}},
    {{-1.782577972e-1, 8.612761663e-1, 2.261841969e-1}},
    {{-3.349048323e-1, 1.056744667e+0, 1.256661680e-1}},
    {{-6.617401158e-1, 7.467595004e-1, 7.146490945e-1}},
    {{1.623948553e-1, 5.661708862e-1, 4.223071752e-1}},
    {{-1.061945788e-2, 5.218564264e-1, 5.450015143e-1}},
    {{-6.147823411e-2, 6.604172234e-1, 3.932639495e-1}},
    {{-1.389529695e-1, 8.076691064e-1, 2.726029342e-1}},
    {{1.686596060e-1, 5.847984817e-1, 4.056779523e-1}},
    {{2.308552718e-1, 6.042409177e-1, 2.595768926e-1}},
    {{-2.010175008e-2, 5.899370608e-1, 4.658445960e-1}},
    {{1.737856685e-1, 5.973380628e-1, 3.929395614e-1}},
    {{1.909729355e-1, 6.146060459e-1, 3.059611271e-1}},
    {{1.780980905e-1, 6.063757846e-1, 3.828552923e-1}},
}};

constexpr std::array<std::array<double, 4>, 15> kAlpha4{{
    {{5.216844534e+0, 9.546182760e-1, 2.652034102e-1, 8.801862774e-2}},
    {{1.161525551e+1, 2.000243111e+0, 1.607280687e-1, 6.125744532e-2}},
    {{1.513265591e+0, 4.262497508e-1, 7.643320863e-2, 3.760545063e-2}},
    {{3.242212833e-1, 1.663217177e-1, 5.081097451e-2, 2.829066600e-2}},
    {{8.602284252e-1, 1.189050200e-1, 3.446076176e-2, 1.974798796e-2}},
    {{1.798260992e+0, 4.662622228e-1, 1.643718620e-1, 6.543927065e-2}},
    {{1.853180239e+0, 1.915075719e-1, 8.655487938e-2, 4.184253862e-2}},
    {{1.492607880e+0, 4.327619272e-1, 7.553156064e-2, 3.706272183e-2}},
    {{3.962838833e-1, 1.838858552e-1, 4.943555157e-2, 2.750222273e-2}},
    {{9.185846715e-1, 2.920461109e-1, 1.187568890e-1, 5.286755896e-2}},
    {{1.995825422e+0, 1.823461280e-1, 8.197240896e-2, 4.000634951e-2}},
    {{4.230617826e-1, 8.293863702e-2, 4.590326388e-2, 2.628744797e-2}},
    {{5.691670217e-1, 2.074585819e-1, 9.298346885e-2, 4.473508853e-2}},
    {{2.017831152e-1, 1.001952178e-1, 5.447006630e-2, 3.037569283e-2}},
    {{3.945205573e-1, 1.588100623e-1, 7.646521729e-2, 3.898703611e-2}},
}};

constexpr std::array<std::array<double, 4>, 15> kCoeff4{{
    {{5.675242080e-2, 2.601413550e-1, 5.328461143e-1, 2.916254405e-1}},
    {{-1.198411747e-2, -5.472052539e-2, 5.805587176e-1, 4.770079976e-1}},
    {{-3.295496352e-2, -1.724516959e-1, 7.518511194e-1, 3.589627317e-1}},
    {{-1.120682822e-1, -2.845426863e-1, 8.909873788e-1, 3.517811205e-1}},
    {{1.103657561e-2, -5.606519023e-1, 1.179429987e+0, 1.734974376e-1}},
    {{5.713170255e-2, 2.857455515e-1, 5.517873105e-1, 2.632314924e-1}},
    {{-1.434249391e-2, 2.755177589e-1, 5.846750879e-1, 2.144986514e-1}},
    {{-6.035216774e-3, -6.013310874e-2, 6.451518200e-1, 4.117923820e-1}},
    {{-1.801459207e-2, -1.360777372e-1, 7.533973719e-1, 3.409304859e-1}},
    {{5.799057705e-2, 3.045581349e-1, 5.601358038e-1, 2.432423313e-1}},
    {{-2.816702620e-3, 2.177095871e-1, 6.058047348e-1, 2.717811257e-1}},
    {{-2.421626009e-2, 3.937644956e-1, 5.489520286e-1, 1.190436963e-1}},
    {{5.902730589e-2, 3.191828952e-1, 5.639423893e-1, 2.284796537e-1}},
    {{9.174268830e-2, 4.023496947e-1, 4.937432100e-1, 1.254001522e-1}},
    {{6.010484250e-2, 3.309738329e-1, 5.655207585e-1, 2.171122608e-1}},
}};

constexpr std::array<double, 6> kAlpha6s{{5.800292686e-1, 2.718262251e-1, 7.938523262e-2,
                                          4.975088254e-2, 2.983643556e-2, 1.886067216e-2}};
constexpr std::array<double, 6> kCoeff6s{{4.554359511e-3, 5.286443143e-2, -7.561016358e-1,
                                          -2.269803820e-1, 1.332494651e+0, 3.622518293e-1}};
constexpr std::array<double, 6> kAlpha6p{{6.696537714e-1, 1.395089793e-1, 8.163894960e-2,
                                          4.586329272e-2, 2.961305556e-2, 1.882221321e-2}};
constexpr std::array<double, 6> kCoeff6p{{2.782723680e-3, -1.282887780e-1, -2.266255943e-1,
                                          4.682259383e-1, 6.752048848e-1, 1.091534212e-1}};

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

const parameters::gfn2::ShellParameters* element_shells(
    const parameters::gfn2::ElementParameters& element) {
  const std::size_t begin = element.shell_offset;
  const std::size_t count = element.shell_count;
  if (begin > parameters::gfn2::kShells.size() ||
      count > parameters::gfn2::kShells.size() - begin) {
    return nullptr;
  }
  return parameters::gfn2::kShells.data() + begin;
}

bool base_sto_table(std::uint8_t n, std::uint8_t l, std::uint8_t ng, const double*& alpha,
                    const double*& coeff) {
  if (n == 6u && ng == 6u) {
    if (l == 0u) {
      alpha = kAlpha6s.data();
      coeff = kCoeff6s.data();
      return true;
    }
    if (l == 1u) {
      alpha = kAlpha6p.data();
      coeff = kCoeff6p.data();
      return true;
    }
    return false;
  }
  if (n == 0u || n > 5u || l >= n || l > 4u) {
    return false;
  }

  std::size_t type = 0;
  if (l == 0u) {
    type = static_cast<std::size_t>(n - 1u);
  } else if (l == 1u) {
    type = static_cast<std::size_t>(n + 3u);
  } else if (l == 2u) {
    type = static_cast<std::size_t>(n + 6u);
  } else if (l == 3u) {
    type = static_cast<std::size_t>(n + 8u);
  } else {
    type = static_cast<std::size_t>(n + 9u);
  }
  if (type >= kAlpha3.size()) {
    return false;
  }
  if (ng == 3u) {
    alpha = kAlpha3[type].data();
    coeff = kCoeff3[type].data();
    return true;
  }
  if (ng == 4u) {
    alpha = kAlpha4[type].data();
    coeff = kCoeff4[type].data();
    return true;
  }
  return false;
}

bool validate_shell(const parameters::gfn2::ShellParameters& shell) {
  const double* alpha = nullptr;
  const double* coeff = nullptr;
  return shell.angular_momentum <= 4u && shell.gaussian_count >= 1u && shell.gaussian_count <= 6u &&
         shell.slater > 0.0 && std::isfinite(shell.slater) &&
         base_sto_table(shell.principal_quantum_number, shell.angular_momentum,
                        shell.gaussian_count, alpha, coeff);
}

void expand_shell(const parameters::gfn2::ShellParameters& shell, double* alpha, double* coeff) {
  const double* base_alpha = nullptr;
  const double* base_coeff = nullptr;
  (void)base_sto_table(shell.principal_quantum_number, shell.angular_momentum, shell.gaussian_count,
                       base_alpha, base_coeff);

  const double zeta_squared = shell.slater * shell.slater;
  const std::size_t l = shell.angular_momentum;
  for (std::size_t primitive = 0; primitive < shell.gaussian_count; ++primitive) {
    alpha[primitive] = base_alpha[primitive] * zeta_squared;
    const double normalization =
        std::pow(kTwoOverPi * alpha[primitive], 0.75) *
        std::pow(std::sqrt(4.0 * alpha[primitive]), static_cast<double>(l)) /
        std::sqrt(kDoubleFactorial[l]);
    coeff[primitive] = base_coeff[primitive] * normalization;
  }
}

void orthogonalize_to_first(const double* first_alpha, const double* first_coeff,
                            std::size_t first_count, double* alpha, double* coeff,
                            std::size_t base_count) {
  double overlap = 0.0;
  for (std::size_t first = 0; first < first_count; ++first) {
    for (std::size_t second = 0; second < base_count; ++second) {
      const double exponent_sum = first_alpha[first] + alpha[second];
      const double primitive_overlap = std::pow(std::sqrt(kPi / exponent_sum), 3.0);
      overlap += first_coeff[first] * coeff[second] * primitive_overlap;
    }
  }
  for (std::size_t primitive = 0; primitive < first_count; ++primitive) {
    alpha[base_count + primitive] = first_alpha[primitive];
    coeff[base_count + primitive] = -overlap * first_coeff[primitive];
  }

  const std::size_t count = base_count + first_count;
  double norm_squared = 0.0;
  for (std::size_t first = 0; first < count; ++first) {
    for (std::size_t second = 0; second < count; ++second) {
      const double exponent_sum = alpha[first] + alpha[second];
      const double primitive_overlap = std::pow(std::sqrt(kPi / exponent_sum), 3.0);
      norm_squared += coeff[first] * coeff[second] * primitive_overlap;
    }
  }
  const double inverse_norm = 1.0 / std::sqrt(norm_squared);
  for (std::size_t primitive = 0; primitive < count; ++primitive) {
    coeff[primitive] *= inverse_norm;
  }
}

}  // namespace

gpuxtb_status_t make_basis_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                const std::int64_t* atom_offsets,
                                const std::int32_t* atomic_numbers, BasisPlan& plan,
                                std::string& error) {
  if (batch_size <= 0 || total_atoms <= 0 ||
      !count_fits_vector(batch_size, sizeof(std::int64_t), true) ||
      !count_fits_vector(total_atoms, sizeof(std::int64_t), true)) {
    error = "basis plan requires positive, representable batch and atom counts";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets == nullptr || atomic_numbers == nullptr) {
    error = "basis plan offsets and atomic numbers must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (atom_offsets[0] != 0 || atom_offsets[batch_size] != total_atoms) {
    error = "basis plan offsets must start at zero and end at total_atoms";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    if (atom_offsets[batch] < 0 || atom_offsets[batch] > atom_offsets[batch + 1] ||
        atom_offsets[batch + 1] > total_atoms) {
      error = "basis plan offsets must be a monotone ragged partition";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
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
          parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number) {
        error = "basis plan contains an unsupported atomic number";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      const auto* shells = element_shells(*element);
      if (shells == nullptr || element->shell_count == 0u) {
        error = "basis plan contains inconsistent generated element parameters";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }

      std::array<std::uint8_t, 5> first_count{};
      for (std::size_t local_shell = 0; local_shell < element->shell_count; ++local_shell) {
        const auto& shell = shells[local_shell];
        if (!validate_shell(shell)) {
          error = "basis plan contains an unsupported generated STO-nG shell";
          return GPUXTB_STATUS_INTERNAL_ERROR;
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
        if (primitives > static_cast<std::int64_t>(kMaximumContractedPrimitives) ||
            !checked_add(1, created.total_shells) ||
            !checked_add(spherical, created.total_orbitals) ||
            !checked_add(cartesian, created.total_cartesian_orbitals) ||
            !checked_add(primitives, created.total_primitives)) {
          error = "basis plan dimensions overflow the supported index range";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
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
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    const std::size_t shell_count = static_cast<std::size_t>(created.total_shells);
    const std::size_t primitive_count = static_cast<std::size_t>(created.total_primitives);
    created.shell_orbital_offsets.resize(shell_count + 1);
    created.shell_cartesian_orbital_offsets.resize(shell_count + 1);
    created.shell_primitive_offsets.resize(shell_count + 1);
    created.shell_to_atom.resize(shell_count);
    created.principal_quantum_numbers.resize(shell_count);
    created.angular_momenta.resize(shell_count);
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
          parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_numbers[atom]));
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
        created.slater_exponents[current_shell] = shell.slater;

        double* alpha = created.primitive_exponents.data() + current_primitive;
        double* coeff = created.primitive_coefficients.data() + current_primitive;
        expand_shell(shell, alpha, coeff);
        std::size_t actual_count = shell.gaussian_count;

        const std::size_t l = shell.angular_momentum;
        if (first_shell[l] >= 0) {
          const std::size_t first = static_cast<std::size_t>(first_shell[l]);
          const std::size_t first_begin =
              static_cast<std::size_t>(created.shell_primitive_offsets[first]);
          const std::size_t first_count = static_cast<std::size_t>(
              created.shell_primitive_offsets[first + 1] - created.shell_primitive_offsets[first]);
          orthogonalize_to_first(created.primitive_exponents.data() + first_begin,
                                 created.primitive_coefficients.data() + first_begin, first_count,
                                 alpha, coeff, shell.gaussian_count);
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
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN2 basis plan";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN2 basis plan dimensions exceed host container limits";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

}  // namespace gpuxtb::detail::gfn2
