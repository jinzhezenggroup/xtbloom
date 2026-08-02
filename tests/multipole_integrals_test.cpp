#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "model/gfn2/integrals.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

constexpr std::size_t kDipoleComponents = 3;
constexpr std::size_t kQuadrupoleComponents = 6;

bool near(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

struct Evaluation {
  gpuxtb::detail::gfn2::BasisPlan basis;
  gpuxtb::detail::gfn2::IntegralPlan integrals;
  std::vector<double> workspace;
  std::vector<double> overlap;
  std::vector<double> dipole;
  std::vector<double> quadrupole;
};

bool evaluate(std::int64_t batch_size, const std::vector<std::int64_t>& atom_offsets,
              const std::vector<std::int32_t>& atomic_numbers, const std::vector<double>& positions,
              Evaluation& evaluation, std::string& error) {
  if (gpuxtb::detail::gfn2::make_basis_plan(
          batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
          atomic_numbers.data(), evaluation.basis, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  if (gpuxtb::detail::gfn2::make_integral_plan(evaluation.basis, evaluation.integrals, error) !=
      GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  evaluation.workspace.resize((evaluation.integrals.workspace_size_bytes + sizeof(double) - 1u) /
                              sizeof(double));
  const std::size_t matrix_elements =
      static_cast<std::size_t>(evaluation.integrals.total_matrix_elements);
  evaluation.overlap.resize(matrix_elements);
  evaluation.dipole.resize(kDipoleComponents * matrix_elements);
  evaluation.quadrupole.resize(kQuadrupoleComponents * matrix_elements);
  if (gpuxtb::detail::gfn2::evaluate_overlap_cpu(
          evaluation.basis, evaluation.integrals, positions.data(), evaluation.overlap.data(),
          evaluation.workspace.data(), evaluation.workspace.size() * sizeof(double),
          error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  return gpuxtb::detail::gfn2::evaluate_multipole_cpu(
             evaluation.basis, evaluation.integrals, positions.data(), evaluation.dipole.data(),
             evaluation.quadrupole.data(), evaluation.workspace.data(),
             evaluation.workspace.size() * sizeof(double), error) == GPUXTB_STATUS_SUCCESS;
}

struct ReferenceSample {
  std::size_t bra_shell;
  std::size_t ket_shell;
  std::size_t bra_ao;
  std::size_t ket_ao;
  std::array<double, kDipoleComponents> dipole;
  std::array<double, kQuadrupoleComponents> quadrupole;
};

int test_tblite_si_reference() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{14, 14};
  const std::vector<double> positions{0.0, 0.0, 0.0, 1.1, -0.7, 2.3};
  Evaluation evaluation;
  std::string error;
  CHECK(evaluate(1, atom_offsets, atomic_numbers, positions, evaluation, error));

  /*
   * Independent golden from tblite 0.7.0 at revision
   * fa8a4416e8fe093d0075bc10ac875494c2a449a9. The exported
   * tblite_integral_multipole:multipole_cgto routine was compiled with GCC
   * 14.3.0 and fed tblite_basis_slater's normalized GFN2 Si 3s/3p/3d
   * contractions. vec=(1.1,-0.7,2.3) bohr is ket minus bra, exactly matching
   * the local-ket-origin convention exercised here. One AO pair from each of
   * the nine s/p/d shell combinations fixes every tensor component.
   */
  constexpr std::array<ReferenceSample, 9> reference{{
      {0,
       0,
       0,
       0,
       {-3.00397989080880445e-1, 1.91162356687832991e-1, -6.28104886260022699e-1},
       {-2.26913971146729621e-1, -1.56003355163376656e-1, -3.72787238312484392e-1,
        5.12582452679666067e-1, -3.26188833523423871e-1, 5.99701209459214013e-1}},
      {1,
       0,
       1,
       0,
       {-2.87034365716915263e-1, 1.82658232728946046e-1, -3.26010042227728170e-2},
       {4.91952317270226036e-1, -1.76671833728098387e-1, 3.26752680537458651e-1,
        4.32176355970956133e-2, -2.75021317436062937e-2, -8.18704997807684798e-1}},
      {2,
       0,
       3,
       0,
       {4.70943545841780442e-1, 8.31504217422929653e-2, 1.45172730370284244e-2},
       {-4.37168247576510682e-1, 2.42722249703251652e-1, 4.87044692904094800e-1,
        1.14880320139409270e-1, -1.78686386985728828e-3, -4.98764453275841319e-2}},
      {0,
       1,
       0,
       1,
       {2.03291826574488882e-1, -1.29367526001947453e-1, 9.92626488750163016e-1},
       {6.85542035638492142e-1, 8.87421676285506944e-2, 7.68521724849604349e-1,
        -6.90782780311507083e-1, 4.39589042016413523e-1, -1.45406376048809638e0}},
      {1,
       1,
       2,
       1,
       {-3.30437229426779500e-1, -8.71711601655566770e-2, 5.09968452009154527e-1},
       {9.08659326017443369e-1, -1.89781390896452407e-1, 1.39546293583380887e-1,
        1.09763828008977105e0, 2.70076207068814744e-1, -1.04820561960082426e0}},
      {2,
       1,
       4,
       2,
       {2.90474230733686722e-1, -5.90917377621156311e-3, -2.10945898942491245e-1},
       {-2.59652519499131151e-1, 4.46876966641527318e-1, 1.11512719457499743e-1,
        -4.45110571373256880e-1, 1.01883555629649677e-2, 1.48139800041631520e-1}},
      {0,
       2,
       0,
       3,
       {-7.14102464540552595e-1, 7.15870719751074552e-2, -5.22940466679915450e-1},
       {2.52353135588085031e-1, -3.69887142189545104e-1, -7.90901358317703806e-1,
        2.60248732331541266e0, -2.64005398872857844e-1, 5.38548222729618775e-1}},
      {1,
       2,
       0,
       3,
       {1.98464142513550934e-1, 2.77402307323064701e-1, 1.78566177215461280e-1},
       {-1.69795409302650691e-1, -1.04320371211189844e0, 5.12597629915014674e-1,
        -6.52757970852293701e-1, -9.55938390947724481e-1, -3.42802220612363873e-1}},
      {2,
       2,
       1,
       3,
       {1.02829435175688742e-3, -1.40052448266430943e-2, 1.47941346357884085e-2},
       {-9.00752321045828058e-2, 2.19552789759313249e-1, -1.06169265720478301e-1,
        -5.50626016015657999e-2, 1.20385067590523587e-1, 1.96244497825061093e-1}},
  }};
  constexpr std::array<std::size_t, 3> shell_offsets{0, 1, 4};
  constexpr std::size_t nao = 18;
  constexpr std::size_t atom_orbitals = 9;
  const std::size_t matrix_elements = evaluation.overlap.size();
  for (const ReferenceSample& sample : reference) {
    const std::size_t row = shell_offsets[sample.bra_shell] + sample.bra_ao;
    const std::size_t column = atom_orbitals + shell_offsets[sample.ket_shell] + sample.ket_ao;
    const std::size_t matrix_index = row * nao + column;
    for (std::size_t component = 0; component < kDipoleComponents; ++component) {
      CHECK(near(evaluation.dipole[component * matrix_elements + matrix_index],
                 sample.dipole[component], 1.5e-14));
    }
    for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
      CHECK(near(evaluation.quadrupole[component * matrix_elements + matrix_index],
                 sample.quadrupole[component], 2.0e-14));
    }
  }
  return 0;
}

int test_origin_translation_identities_and_trace() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{14, 14};
  const std::vector<double> positions{0.0, 0.0, 0.0, 1.1, -0.7, 2.3};
  Evaluation evaluation;
  std::string error;
  CHECK(evaluate(1, atom_offsets, atomic_numbers, positions, evaluation, error));

  constexpr std::array<double, 3> displacement{1.1, -0.7, 2.3};
  constexpr std::size_t nao = 18;
  const std::size_t matrix_elements = evaluation.overlap.size();
  for (std::size_t bra = 0; bra < 9; ++bra) {
    for (std::size_t ket = 9; ket < 18; ++ket) {
      const std::size_t forward = bra * nao + ket;
      const std::size_t reverse = ket * nao + bra;
      const double overlap = evaluation.overlap[forward];
      std::array<double, 3> dipole{};
      for (std::size_t component = 0; component < 3; ++component) {
        dipole[component] = evaluation.dipole[component * matrix_elements + forward];
        const double expected = dipole[component] + displacement[component] * overlap;
        CHECK(near(evaluation.dipole[component * matrix_elements + reverse], expected, 3.0e-15));
      }
      std::array<double, 6> shift{};
      shift[0] = 2.0 * displacement[0] * dipole[0] + displacement[0] * displacement[0] * overlap;
      shift[1] = displacement[0] * dipole[1] + displacement[1] * dipole[0] +
                 displacement[0] * displacement[1] * overlap;
      shift[2] = 2.0 * displacement[1] * dipole[1] + displacement[1] * displacement[1] * overlap;
      shift[3] = displacement[0] * dipole[2] + displacement[2] * dipole[0] +
                 displacement[0] * displacement[2] * overlap;
      shift[4] = displacement[1] * dipole[2] + displacement[2] * dipole[1] +
                 displacement[1] * displacement[2] * overlap;
      shift[5] = 2.0 * displacement[2] * dipole[2] + displacement[2] * displacement[2] * overlap;
      const double trace = 0.5 * (shift[0] + shift[2] + shift[5]);
      for (std::size_t component = 0; component < 6; ++component) {
        const bool diagonal = component == 0u || component == 2u || component == 5u;
        const double expected = evaluation.quadrupole[component * matrix_elements + forward] +
                                1.5 * shift[component] - (diagonal ? trace : 0.0);
        CHECK(
            near(evaluation.quadrupole[component * matrix_elements + reverse], expected, 5.0e-15));
      }
    }
  }

  for (std::size_t element = 0; element < matrix_elements; ++element) {
    const double trace = evaluation.quadrupole[element] +
                         evaluation.quadrupole[2u * matrix_elements + element] +
                         evaluation.quadrupole[5u * matrix_elements + element];
    CHECK(near(trace, 0.0, 1.0e-15));
  }

  /* On one atom both AO origins coincide, so every component is symmetric. */
  for (std::size_t atom = 0; atom < 2; ++atom) {
    const std::size_t begin = atom * 9u;
    for (std::size_t row = begin; row < begin + 9u; ++row) {
      for (std::size_t column = begin; column < begin + 9u; ++column) {
        for (std::size_t component = 0; component < kDipoleComponents; ++component) {
          CHECK(near(evaluation.dipole[component * matrix_elements + row * nao + column],
                     evaluation.dipole[component * matrix_elements + column * nao + row], 2.0e-16));
        }
        for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
          CHECK(near(evaluation.quadrupole[component * matrix_elements + row * nao + column],
                     evaluation.quadrupole[component * matrix_elements + column * nao + row],
                     3.0e-16));
        }
      }
    }
  }
  return 0;
}

int test_translation_and_rotation_covariance() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{14, 14};
  const std::vector<double> positions{0.0, 0.0, 0.0, 1.1, -0.7, 2.3};
  Evaluation original;
  std::string error;
  CHECK(evaluate(1, atom_offsets, atomic_numbers, positions, original, error));

  std::vector<double> translated = positions;
  for (std::size_t atom = 0; atom < 2; ++atom) {
    translated[atom * 3u] += 4.25;
    translated[atom * 3u + 1u] -= 1.75;
    translated[atom * 3u + 2u] += 0.625;
  }
  Evaluation shifted;
  CHECK(evaluate(1, atom_offsets, atomic_numbers, translated, shifted, error));
  for (std::size_t element = 0; element < original.dipole.size(); ++element) {
    CHECK(near(shifted.dipole[element], original.dipole[element], 2.0e-15));
  }
  for (std::size_t element = 0; element < original.quadrupole.size(); ++element) {
    CHECK(near(shifted.quadrupole[element], original.quadrupole[element], 5.0e-15));
  }

  std::vector<double> rotated = positions;
  for (std::size_t atom = 0; atom < 2; ++atom) {
    const double x = positions[atom * 3u];
    const double y = positions[atom * 3u + 1u];
    rotated[atom * 3u] = -y;
    rotated[atom * 3u + 1u] = x;
  }
  Evaluation turned;
  CHECK(evaluate(1, atom_offsets, atomic_numbers, rotated, turned, error));

  constexpr std::array<double, 81> ao_rotation{
      1, 0,  0, 0, 0,  0,  0, 0, 0,  // s
      0, 0,  0, 1, 0,  0,  0, 0, 0,  // y -> x
      0, 0,  1, 0, 0,  0,  0, 0, 0,  // z -> z
      0, -1, 0, 0, 0,  0,  0, 0, 0,  // x -> -y
      0, 0,  0, 0, -1, 0,  0, 0, 0,  // xy -> -xy
      0, 0,  0, 0, 0,  0,  0, 1, 0,  // yz -> xz
      0, 0,  0, 0, 0,  0,  1, 0, 0,  // z2 -> z2
      0, 0,  0, 0, 0,  -1, 0, 0, 0,  // xz -> -yz
      0, 0,  0, 0, 0,  0,  0, 0, -1  // x2-y2 -> -(x2-y2)
  };
  constexpr std::array<double, 9> vector_rotation{0, -1, 0, 1, 0, 0, 0, 0, 1};
  constexpr std::array<double, 36> tensor_rotation{
      0, 0, 1, 0, 0,  0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, -1, 0, 0, 0,  0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
  };
  constexpr std::size_t atom_orbitals = 9;
  constexpr std::size_t nao = 18;
  constexpr std::size_t matrix_elements = nao * nao;
  for (std::size_t row = 0; row < atom_orbitals; ++row) {
    for (std::size_t column = 0; column < atom_orbitals; ++column) {
      for (std::size_t output_component = 0; output_component < 3; ++output_component) {
        double expected = 0.0;
        for (std::size_t input_component = 0; input_component < 3; ++input_component) {
          double rotated_ao = 0.0;
          for (std::size_t inner_row = 0; inner_row < atom_orbitals; ++inner_row) {
            for (std::size_t inner_column = 0; inner_column < atom_orbitals; ++inner_column) {
              rotated_ao += ao_rotation[row * atom_orbitals + inner_row] *
                            original.dipole[input_component * matrix_elements + inner_row * nao +
                                            atom_orbitals + inner_column] *
                            ao_rotation[column * atom_orbitals + inner_column];
            }
          }
          expected += vector_rotation[output_component * 3u + input_component] * rotated_ao;
        }
        CHECK(near(
            turned.dipole[output_component * matrix_elements + row * nao + atom_orbitals + column],
            expected, 1.5e-14));
      }
      for (std::size_t output_component = 0; output_component < 6; ++output_component) {
        double expected = 0.0;
        for (std::size_t input_component = 0; input_component < 6; ++input_component) {
          double rotated_ao = 0.0;
          for (std::size_t inner_row = 0; inner_row < atom_orbitals; ++inner_row) {
            for (std::size_t inner_column = 0; inner_column < atom_orbitals; ++inner_column) {
              rotated_ao += ao_rotation[row * atom_orbitals + inner_row] *
                            original.quadrupole[input_component * matrix_elements +
                                                inner_row * nao + atom_orbitals + inner_column] *
                            ao_rotation[column * atom_orbitals + inner_column];
            }
          }
          expected += tensor_rotation[output_component * 6u + input_component] * rotated_ao;
        }
        CHECK(near(turned.quadrupole[output_component * matrix_elements + row * nao +
                                     atom_orbitals + column],
                   expected, 3.0e-14));
      }
    }
  }
  return 0;
}

int test_ragged_batch_equals_sequential() {
  const std::vector<std::int64_t> offsets{0, 2, 2, 3, 5};
  const std::vector<std::int32_t> atomic_numbers{14, 14, 14, 8, 1};
  const std::vector<double> positions{
      0.0, 0.0, 0.0, 1.1, -0.7, 2.3, -0.2, 0.4, 1.5, 2.0, -1.0, 0.5, 3.2, -0.8, 0.1,
  };
  Evaluation batch;
  std::string error;
  CHECK(evaluate(4, offsets, atomic_numbers, positions, batch, error));

  const std::size_t batch_matrix_elements = batch.overlap.size();
  for (std::size_t molecule = 0; molecule < 4; ++molecule) {
    const std::int64_t atom_begin = offsets[molecule];
    const std::int64_t atom_end = offsets[molecule + 1u];
    if (atom_begin == atom_end) {
      continue;
    }
    const std::vector<std::int64_t> sequential_offsets{0, atom_end - atom_begin};
    const std::vector<std::int32_t> sequential_numbers(atomic_numbers.begin() + atom_begin,
                                                       atomic_numbers.begin() + atom_end);
    const std::vector<double> sequential_positions(positions.begin() + atom_begin * 3,
                                                   positions.begin() + atom_end * 3);
    Evaluation sequential;
    CHECK(evaluate(1, sequential_offsets, sequential_numbers, sequential_positions, sequential,
                   error));
    const std::size_t packed_begin =
        static_cast<std::size_t>(batch.integrals.matrix_offsets[molecule]);
    const std::size_t sequential_elements = sequential.overlap.size();
    for (std::size_t component = 0; component < kDipoleComponents; ++component) {
      for (std::size_t element = 0; element < sequential_elements; ++element) {
        CHECK(batch.dipole[component * batch_matrix_elements + packed_begin + element] ==
              sequential.dipole[component * sequential_elements + element]);
      }
    }
    for (std::size_t component = 0; component < kQuadrupoleComponents; ++component) {
      for (std::size_t element = 0; element < sequential_elements; ++element) {
        CHECK(batch.quadrupole[component * batch_matrix_elements + packed_begin + element] ==
              sequential.quadrupole[component * sequential_elements + element]);
      }
    }
  }
  return 0;
}

int test_validation_preserves_outputs() {
  constexpr std::array<std::int64_t, 2> offsets{0, 1};
  constexpr std::array<std::int32_t, 1> atomic_numbers{14};
  gpuxtb::detail::gfn2::BasisPlan basis;
  gpuxtb::detail::gfn2::IntegralPlan plan;
  std::string error;
  CHECK(gpuxtb::detail::gfn2::make_basis_plan(1, 1, offsets.data(), atomic_numbers.data(), basis,
                                              error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::make_integral_plan(basis, plan, error) == GPUXTB_STATUS_SUCCESS);
  std::array<double, 3> positions{};
  const std::size_t matrix_elements = static_cast<std::size_t>(plan.total_matrix_elements);
  std::vector<double> dipole(kDipoleComponents * matrix_elements, 3.0);
  std::vector<double> quadrupole(kQuadrupoleComponents * matrix_elements, 4.0);
  std::vector<double> workspace((plan.workspace_size_bytes + sizeof(double) - 1u) / sizeof(double));

  CHECK(gpuxtb::detail::gfn2::evaluate_multipole_cpu(
            basis, plan, positions.data(), dipole.data(), quadrupole.data(), workspace.data(),
            plan.workspace_size_bytes - 1u, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(dipole.begin(), dipole.end(), [](double value) { return value == 3.0; }));
  CHECK(
      std::all_of(quadrupole.begin(), quadrupole.end(), [](double value) { return value == 4.0; }));

  positions[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(gpuxtb::detail::gfn2::evaluate_multipole_cpu(
            basis, plan, positions.data(), dipole.data(), quadrupole.data(), workspace.data(),
            workspace.size() * sizeof(double), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  positions[0] = 0.0;
  CHECK(gpuxtb::detail::gfn2::evaluate_multipole_cpu(
            basis, plan, positions.data(), nullptr, quadrupole.data(), workspace.data(),
            workspace.size() * sizeof(double), error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(
      std::all_of(quadrupole.begin(), quadrupole.end(), [](double value) { return value == 4.0; }));

  constexpr std::array<std::int64_t, 2> pair_offsets{0, 2};
  constexpr std::array<std::int32_t, 2> pair_atomic_numbers{1, 1};
  gpuxtb::detail::gfn2::BasisPlan pair_basis;
  gpuxtb::detail::gfn2::IntegralPlan pair_plan;
  CHECK(gpuxtb::detail::gfn2::make_basis_plan(1, 2, pair_offsets.data(), pair_atomic_numbers.data(),
                                              pair_basis, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::make_integral_plan(pair_basis, pair_plan, error) ==
        GPUXTB_STATUS_SUCCESS);
  const double maximum = std::numeric_limits<double>::max();
  const std::array<double, 6> extreme_positions{maximum, 0.0, 0.0, -maximum, 0.0, 0.0};
  const std::size_t pair_matrix_elements =
      static_cast<std::size_t>(pair_plan.total_matrix_elements);
  std::vector<double> pair_dipole(kDipoleComponents * pair_matrix_elements, 7.0);
  std::vector<double> pair_quadrupole(kQuadrupoleComponents * pair_matrix_elements, 8.0);
  std::vector<double> pair_workspace((pair_plan.workspace_size_bytes + sizeof(double) - 1u) /
                                     sizeof(double));
  CHECK(gpuxtb::detail::gfn2::evaluate_multipole_cpu(
            pair_basis, pair_plan, extreme_positions.data(), pair_dipole.data(),
            pair_quadrupole.data(), pair_workspace.data(), pair_workspace.size() * sizeof(double),
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(pair_dipole.begin(), pair_dipole.end(),
                    [](double value) { return value == 7.0; }));
  CHECK(std::all_of(pair_quadrupole.begin(), pair_quadrupole.end(),
                    [](double value) { return value == 8.0; }));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_tblite_si_reference(); status != 0) {
    return status;
  }
  if (const int status = test_origin_translation_identities_and_trace(); status != 0) {
    return status;
  }
  if (const int status = test_translation_and_rotation_covariance(); status != 0) {
    return status;
  }
  if (const int status = test_ragged_batch_equals_sequential(); status != 0) {
    return status;
  }
  return test_validation_preserves_outputs();
}
