// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "model/gfn1/basis.hpp"
#include "model/gfn1/h0.hpp"
#include "model/gfn1/integrals.hpp"
#include "model/gfn1/scc_driver.hpp"

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

namespace {

using namespace xtbloom::detail::gfn1;

class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t requested) {
    size_ = (std::max<std::size_t>(requested, 1u) + 63u) & ~std::size_t{63u};
    data_ = std::aligned_alloc(64u, size_);
    if (data_ != nullptr) std::memset(data_, 0, size_);
  }
  ~AlignedBuffer() { std::free(data_); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0u;
};

using LapackInt = xtbloom::detail::gfn2::LapackInt;
using CpuLinearAlgebraBackend = xtbloom::detail::gfn2::CpuLinearAlgebraBackend;

std::atomic<int> diagonalization_calls{0};
std::atomic<int> failed_diagonalization_call{-1};

LapackInt tiny_dpotrf(LapackInt, char, LapackInt n, double* matrix, LapackInt) {
  for (LapackInt column = 0; column < n; ++column) {
    for (LapackInt row = column; row < n; ++row) {
      double value = matrix[column * n + row];
      for (LapackInt inner = 0; inner < column; ++inner) {
        value -= matrix[inner * n + row] * matrix[inner * n + column];
      }
      if (row == column) {
        if (!(value > 0.0) || !std::isfinite(value)) return column + 1;
        matrix[column * n + column] = std::sqrt(value);
      } else {
        matrix[column * n + row] = value / matrix[column * n + column];
      }
    }
  }
  return 0;
}

LapackInt tiny_dpocon(LapackInt, char, LapackInt n, const double*, LapackInt, double,
                      double* reciprocal_condition, double*, LapackInt*) {
  if (n <= 0) return -3;
  *reciprocal_condition = 1.0;
  return 0;
}

LapackInt tiny_dsyevd(LapackInt, char, char, LapackInt n, double* matrix, LapackInt,
                      double* eigenvalues, double*, LapackInt, LapackInt*, LapackInt) {
  const int call = diagonalization_calls.fetch_add(1, std::memory_order_relaxed);
  if (call == failed_diagonalization_call.load(std::memory_order_relaxed)) return 1;
  if (n <= 0 || n > 32) return -4;

  std::array<double, 32u * 32u> values{};
  std::array<double, 32u * 32u> vectors{};
  for (LapackInt row = 0; row < n; ++row) {
    vectors[static_cast<std::size_t>(row * n + row)] = 1.0;
    for (LapackInt column = 0; column < n; ++column) {
      values[static_cast<std::size_t>(row * n + column)] = matrix[column * n + row];
    }
  }
  bool converged = false;
  for (int sweep = 0; sweep < 128 * n * n; ++sweep) {
    LapackInt p = 0;
    LapackInt q = 0;
    double largest = 0.0;
    for (LapackInt row = 0; row < n; ++row) {
      for (LapackInt column = row + 1; column < n; ++column) {
        const double magnitude = std::abs(values[static_cast<std::size_t>(row * n + column)]);
        if (magnitude > largest) {
          largest = magnitude;
          p = row;
          q = column;
        }
      }
    }
    if (largest <= 1.0e-14) {
      converged = true;
      break;
    }
    const double app = values[static_cast<std::size_t>(p * n + p)];
    const double aqq = values[static_cast<std::size_t>(q * n + q)];
    const double apq = values[static_cast<std::size_t>(p * n + q)];
    const double tau = (aqq - app) / (2.0 * apq);
    const double tangent = std::copysign(1.0, tau) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
    const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
    const double sine = tangent * cosine;
    for (LapackInt index = 0; index < n; ++index) {
      if (index != p && index != q) {
        const double aip = values[static_cast<std::size_t>(index * n + p)];
        const double aiq = values[static_cast<std::size_t>(index * n + q)];
        const double updated_p = cosine * aip - sine * aiq;
        const double updated_q = sine * aip + cosine * aiq;
        values[static_cast<std::size_t>(index * n + p)] = updated_p;
        values[static_cast<std::size_t>(p * n + index)] = updated_p;
        values[static_cast<std::size_t>(index * n + q)] = updated_q;
        values[static_cast<std::size_t>(q * n + index)] = updated_q;
      }
      const double vip = vectors[static_cast<std::size_t>(index * n + p)];
      const double viq = vectors[static_cast<std::size_t>(index * n + q)];
      vectors[static_cast<std::size_t>(index * n + p)] = cosine * vip - sine * viq;
      vectors[static_cast<std::size_t>(index * n + q)] = sine * vip + cosine * viq;
    }
    values[static_cast<std::size_t>(p * n + p)] =
        cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
    values[static_cast<std::size_t>(q * n + q)] =
        sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
    values[static_cast<std::size_t>(p * n + q)] = 0.0;
    values[static_cast<std::size_t>(q * n + p)] = 0.0;
  }
  if (!converged) return 1;

  std::array<LapackInt, 32> order{};
  for (LapackInt index = 0; index < n; ++index) order[static_cast<std::size_t>(index)] = index;
  std::sort(order.begin(), order.begin() + n, [&](LapackInt first, LapackInt second) {
    return values[static_cast<std::size_t>(first * n + first)] <
           values[static_cast<std::size_t>(second * n + second)];
  });
  for (LapackInt column = 0; column < n; ++column) {
    const LapackInt source = order[static_cast<std::size_t>(column)];
    eigenvalues[column] = values[static_cast<std::size_t>(source * n + source)];
    for (LapackInt row = 0; row < n; ++row) {
      matrix[column * n + row] = vectors[static_cast<std::size_t>(row * n + source)];
    }
  }
  return 0;
}

void tiny_dtrsm(int layout, int side, int triangle, int transpose, int diagonal, LapackInt rows,
                LapackInt columns, double alpha, const double* triangular,
                LapackInt leading_triangular, double* rhs, LapackInt leading_rhs) {
  constexpr int kColMajor = 102;
  constexpr int kLeft = 141;
  constexpr int kRight = 142;
  constexpr int kLower = 122;
  constexpr int kNoTrans = 111;
  constexpr int kTrans = 112;
  constexpr int kNonUnit = 131;
  if (layout != kColMajor || (side != kLeft && side != kRight) || triangle != kLower ||
      (transpose != kNoTrans && transpose != kTrans) || diagonal != kNonUnit || rows < 0 ||
      columns < 0 || leading_rhs < std::max<LapackInt>(1, rows) ||
      leading_triangular < std::max<LapackInt>(1, side == kLeft ? rows : columns)) {
    std::abort();
  }
  if (side == kLeft) {
    for (LapackInt column = 0; column < columns; ++column) {
      if (transpose == kNoTrans) {
        for (LapackInt row = 0; row < rows; ++row) {
          double value = alpha * rhs[column * leading_rhs + row];
          for (LapackInt inner = 0; inner < row; ++inner) {
            value -=
                triangular[inner * leading_triangular + row] * rhs[column * leading_rhs + inner];
          }
          rhs[column * leading_rhs + row] = value / triangular[row * leading_triangular + row];
        }
      } else {
        for (LapackInt row = rows; row-- > 0;) {
          double value = alpha * rhs[column * leading_rhs + row];
          for (LapackInt inner = row + 1; inner < rows; ++inner) {
            value -=
                triangular[row * leading_triangular + inner] * rhs[column * leading_rhs + inner];
          }
          rhs[column * leading_rhs + row] = value / triangular[row * leading_triangular + row];
        }
      }
    }
  } else {
    if (transpose != kTrans) std::abort();
    for (LapackInt row = 0; row < rows; ++row) {
      for (LapackInt column = 0; column < columns; ++column) {
        double value = alpha * rhs[column * leading_rhs + row];
        for (LapackInt inner = 0; inner < column; ++inner) {
          value -= rhs[inner * leading_rhs + row] * triangular[inner * leading_triangular + column];
        }
        rhs[column * leading_rhs + row] = value / triangular[column * leading_triangular + column];
      }
    }
  }
}

void tiny_dgemm(int layout, int transpose_left, int transpose_right, LapackInt rows,
                LapackInt columns, LapackInt inner, double alpha, const double* left,
                LapackInt leading_left, const double* right, LapackInt leading_right, double beta,
                double* result, LapackInt leading_result) {
  constexpr int kColMajor = 102;
  constexpr int kNoTrans = 111;
  constexpr int kTrans = 112;
  if (layout != kColMajor || transpose_left != kNoTrans || transpose_right != kTrans || rows < 0 ||
      columns < 0 || inner < 0 || leading_left < std::max<LapackInt>(1, rows) ||
      leading_right < std::max<LapackInt>(1, columns) ||
      leading_result < std::max<LapackInt>(1, rows)) {
    std::abort();
  }
  for (LapackInt column = 0; column < columns; ++column) {
    for (LapackInt row = 0; row < rows; ++row) {
      double value = 0.0;
      for (LapackInt k = 0; k < inner; ++k) {
        value += left[k * leading_left + row] * right[k * leading_right + column];
      }
      result[column * leading_result + row] =
          alpha * value + beta * result[column * leading_result + row];
    }
  }
}

const CpuLinearAlgebraBackend& backend() {
  static const CpuLinearAlgebraBackend value = [] {
    CpuLinearAlgebraBackend result;
    std::string error;
    if (xtbloom::detail::gfn2::make_internal_test_lp64_backend(
            &tiny_dpotrf, &tiny_dpocon, &tiny_dsyevd, &tiny_dtrsm, &tiny_dgemm, nullptr, result,
            error) != XTBLOOM_STATUS_SUCCESS) {
      std::abort();
    }
    return result;
  }();
  return value;
}

struct FixtureInput {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
};

FixtureInput mixed_input() {
  return {{0, 2, 3},  {1, 1, 1}, {0.0, 0.0, -0.7, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0},
          {0.0, 0.0}, {0, 1},    {1, 2}};
}

FixtureInput two_restricted_input() {
  return {{0, 2, 4},  {1, 1, 1, 1}, {0.0, 0.0, -0.7, 0.0, 0.0, 0.7, 0.0, 0.0, -0.8, 0.0, 0.0, 0.8},
          {0.0, 0.0}, {0, 0},       {1, 1}};
}

struct Fixture {
  FixtureInput input;
  BasisPlan basis;
  IntegralPlan integrals;
  H0Plan h0_plan;
  WavefunctionLayout wavefunction_layout;
  MullikenPlan mulliken;
  ES2Plan es2;
  ES3Plan es3;
  SpinPopulationLayout spin_layout;
  SpinPolarizationPlan spin;
  EigensolverPlan eigensolver;
  SccMixerPlan mixer;
  PeriodicEmbeddingPlan periodic;
  SccDriverPlan driver;

  std::vector<double> coordination;
  std::vector<double> overlap;
  std::vector<double> h0;
  std::vector<double> es2_matrix;
  std::vector<double> es2_matrix_scratch;
  ES2GeometryCache es2_cache;
  std::vector<double> point_potential;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response;

  std::unique_ptr<AlignedBuffer> integral_workspace;
  std::unique_ptr<AlignedBuffer> wavefunction_storage;
  std::unique_ptr<AlignedBuffer> overlap_cache_storage;
  std::unique_ptr<AlignedBuffer> eigensolver_workspace_storage;
  std::unique_ptr<AlignedBuffer> mixer_state_storage;
  std::unique_ptr<AlignedBuffer> driver_state_storage;
  std::unique_ptr<AlignedBuffer> driver_workspace_storage;

  WavefunctionView wavefunction;
  EigensolverOverlapCache overlap_cache;
  EigensolverWorkspace eigensolver_workspace;
  SccMixerState mixer_state;
  SccDriverState driver_state;
  SccDriverWorkspace driver_workspace;
  SccDriverGeometryView geometry;

  bool initialize(FixtureInput definition, std::uint64_t maximum_iterations,
                  double electronic_temperature, double mixer_tolerance, double energy_tolerance,
                  bool enable_periodic, std::string& error) {
    input = std::move(definition);
    const std::int64_t batch = static_cast<std::int64_t>(input.atom_offsets.size() - 1u);
    const std::int64_t atoms = static_cast<std::int64_t>(input.atomic_numbers.size());
    if (make_basis_plan(batch, atoms, input.atom_offsets.data(), input.atomic_numbers.data(), basis,
                        error) != XTBLOOM_STATUS_SUCCESS ||
        make_integral_plan(basis, integrals, error) != XTBLOOM_STATUS_SUCCESS ||
        make_h0_plan(basis, integrals, input.atomic_numbers.data(), h0_plan, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_wavefunction_layout(basis, input.atomic_numbers.data(), input.molecular_charges.data(),
                                 input.unpaired_electrons.data(), input.spin_channels.data(),
                                 wavefunction_layout, error) != XTBLOOM_STATUS_SUCCESS ||
        make_mulliken_plan(basis, integrals, wavefunction_layout, mulliken, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn1::make_es2_plan(basis, input.atomic_numbers.data(), es2, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_es3_plan(basis, input.atomic_numbers.data(), es3, error) != XTBLOOM_STATUS_SUCCESS ||
        make_spin_population_layout(basis, input.spin_channels.data(), spin_layout, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_spin_polarization_plan(basis, input.atomic_numbers.data(), spin_layout, spin, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn2::make_eigensolver_plan(
            make_eigensolver_wavefunction_layout(wavefunction_layout), eigensolver, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        make_scc_mixer_plan(wavefunction_layout, 3, 0.4, mixer_tolerance, mixer_tolerance, mixer,
                            error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    if (enable_periodic &&
        xtbloom::detail::gfn2::make_periodic_embedding_plan(
            batch, atoms, input.atom_offsets.data(), periodic, error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    if (make_scc_driver_plan(wavefunction_layout, mulliken, es2, es3, spin, eigensolver, mixer,
                             enable_periodic ? &periodic : nullptr, maximum_iterations,
                             electronic_temperature, energy_tolerance, driver,
                             error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    coordination.assign(static_cast<std::size_t>(atoms), 0.0);
    overlap.resize(static_cast<std::size_t>(integrals.total_matrix_elements));
    h0.resize(overlap.size());
    integral_workspace = std::make_unique<AlignedBuffer>(integrals.workspace_size_bytes);
    if (integral_workspace->data() == nullptr ||
        evaluate_overlap_cpu(basis, integrals, input.positions.data(), overlap.data(),
                             integral_workspace->data(), integral_workspace->size(),
                             error) != XTBLOOM_STATUS_SUCCESS ||
        evaluate_h0_cpu(basis, integrals, h0_plan, input.positions.data(), coordination.data(),
                        overlap.data(), h0.data(), error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    es2_matrix.resize(static_cast<std::size_t>(es2.total_matrix_elements()));
    es2_matrix_scratch.resize(es2_matrix.size());
    ES2Workspace es2_update;
    es2_update.matrix_scratch = es2_matrix_scratch.data();
    es2_update.matrix_elements = es2.total_matrix_elements();
    if (update_es2_geometry_cache_cpu(es2, input.positions.data(), 1u, es2_matrix.data(),
                                      es2_matrix.size(), es2_update, es2_cache,
                                      error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    wavefunction_storage =
        std::make_unique<AlignedBuffer>(wavefunction_layout.workspace_size_bytes);
    overlap_cache_storage = std::make_unique<AlignedBuffer>(eigensolver.overlap_cache_size_bytes());
    eigensolver_workspace_storage =
        std::make_unique<AlignedBuffer>(eigensolver.workspace_size_bytes());
    mixer_state_storage = std::make_unique<AlignedBuffer>(mixer.state_size_bytes());
    driver_state_storage = std::make_unique<AlignedBuffer>(driver.state_size_bytes());
    driver_workspace_storage = std::make_unique<AlignedBuffer>(driver.workspace_size_bytes());
    if (bind_wavefunction_view(wavefunction_layout, wavefunction_storage->data(),
                               wavefunction_storage->size(), wavefunction,
                               error) != XTBLOOM_STATUS_SUCCESS ||
        initialize_sad_multipole_state(wavefunction_layout, wavefunction, error) !=
            XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn2::bind_eigensolver_overlap_cache(
            eigensolver, overlap_cache_storage->data(), overlap_cache_storage->size(),
            overlap_cache, error) != XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn2::bind_eigensolver_workspace(
            eigensolver, eigensolver_workspace_storage->data(),
            eigensolver_workspace_storage->size(), eigensolver_workspace,
            error) != XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::gfn2::factor_overlap_cpu(eigensolver, overlap.data(), 1u, backend(),
                                                  eigensolver_workspace, overlap_cache,
                                                  error) != XTBLOOM_STATUS_SUCCESS ||
        bind_scc_mixer_state(mixer, mixer_state_storage->data(), mixer_state_storage->size(),
                             mixer_state, error) != XTBLOOM_STATUS_SUCCESS ||
        bind_scc_driver_state(driver, driver_state_storage->data(), driver_state_storage->size(),
                              driver_state, error) != XTBLOOM_STATUS_SUCCESS ||
        bind_scc_driver_workspace(driver, driver_workspace_storage->data(),
                                  driver_workspace_storage->size(), driver_workspace,
                                  error) != XTBLOOM_STATUS_SUCCESS ||
        initialize_scc_driver_state_cpu(driver, wavefunction, mixer_state, driver_state, error) !=
            XTBLOOM_STATUS_SUCCESS) {
      return false;
    }

    point_potential.assign(static_cast<std::size_t>(wavefunction_layout.total_shells), 0.0);
    geometry.h0 = h0.data();
    geometry.h0_elements = integrals.total_matrix_elements;
    geometry.integrals = {overlap.data(), integrals.total_matrix_elements, mulliken.identity()};
    geometry.es2_cache = es2_cache;
    geometry.geometry_generation = 1u;
    if (enable_periodic) {
      periodic_shifts.assign(static_cast<std::size_t>(atoms), 0.0);
      periodic_response.assign(static_cast<std::size_t>(periodic.total_matrix_elements()), 0.0);
      geometry.periodic_shifts = periodic_shifts.data();
      geometry.periodic_shift_elements = atoms;
      geometry.periodic_response_matrices = periodic_response.data();
      geometry.periodic_response_elements = periodic.total_matrix_elements();
      geometry.periodic_embedding_generation = 1u;
      geometry.periodic_plan_identity = periodic.identity();
    }
    return true;
  }

  xtbloom_status_t iterate(std::string& error) {
    return iterate_scc_driver_batch_cpu(driver, geometry, backend(), overlap_cache, wavefunction,
                                        mixer_state, driver_state, driver_workspace, error);
  }
};

bool near(double actual, double expected, double tolerance = 2.0e-12) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <=
             tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

std::vector<std::byte> snapshot(const AlignedBuffer& buffer) {
  std::vector<std::byte> bytes(buffer.size());
  std::memcpy(bytes.data(), buffer.data(), buffer.size());
  return bytes;
}

bool equals_snapshot(const AlignedBuffer& buffer, const std::vector<std::byte>& bytes) {
  return bytes.size() == buffer.size() &&
         std::memcmp(buffer.data(), bytes.data(), buffer.size()) == 0;
}

int test_restricted_unrestricted_free_energy_periodic_and_point_potential() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(mixed_input(), 1u, 0.02, 1.0e100, 1.0e100, true, error));
  for (std::size_t shell = 0; shell < fixture.point_potential.size(); ++shell) {
    fixture.point_potential[shell] = 0.001 * static_cast<double>(shell + 1u);
  }
  fixture.geometry.explicit_point_charge_shell_potential = fixture.point_potential.data();
  fixture.geometry.explicit_point_charge_shell_elements =
      static_cast<std::int64_t>(fixture.point_potential.size());
  constexpr std::array<double, 3> shifts{0.02, -0.01, 0.03};
  constexpr std::array<double, 5> response{0.04, 0.01, 0.01, -0.02, 0.05};
  CHECK(fixture.periodic_shifts.size() == shifts.size());
  CHECK(fixture.periodic_response.size() == response.size());
  std::copy(shifts.begin(), shifts.end(), fixture.periodic_shifts.begin());
  std::copy(response.begin(), response.end(), fixture.periodic_response.begin());

  diagonalization_calls.store(0, std::memory_order_relaxed);
  failed_diagonalization_call.store(-1, std::memory_order_relaxed);
  CHECK(fixture.iterate(error) == XTBLOOM_STATUS_SUCCESS);
  for (std::size_t system = 0; system < 2u; ++system) {
    CHECK(fixture.driver_state.system_statuses[system] == XTBLOOM_STATUS_SUCCESS);
    CHECK(fixture.driver_state.iterations[system] == 1u);
    CHECK(fixture.mixer_state.iterations[system] == 1u);
    CHECK(fixture.driver_state.converged[system] == 1u);
    CHECK(fixture.mixer_state.converged[system] == 1u);
    CHECK(near(fixture.driver_state.free_energies[system],
               fixture.driver_state.internal_energies[system] -
                   0.02 * fixture.driver_state.entropies[system]));
    CHECK(fixture.driver_state.entropies[system] >= 0.0);
    const double sum =
        fixture.driver_state.core_energies[system] + fixture.driver_state.es2_energies[system] +
        fixture.driver_state.es3_energies[system] + fixture.driver_state.spin_energies[system] +
        fixture.driver_state.explicit_point_charge_energies[system] +
        fixture.driver_state.periodic_embedding_energies[system];
    CHECK(near(fixture.driver_state.internal_energies[system], sum));

    const std::int64_t shell_begin = fixture.wavefunction_layout.batch_shell_offsets[system];
    const std::int64_t shell_end = fixture.wavefunction_layout.batch_shell_offsets[system + 1u];
    const std::int64_t qsh_begin = fixture.wavefunction_layout.qsh.system_offsets[system];
    double point_energy = 0.0;
    for (std::int64_t local = 0; local < shell_end - shell_begin; ++local) {
      point_energy = std::fma(
          fixture.driver_workspace.raw_qsh[qsh_begin + local],
          fixture.point_potential[static_cast<std::size_t>(shell_begin + local)], point_energy);
    }
    CHECK(near(fixture.driver_state.explicit_point_charge_energies[system], point_energy));
  }

  /* The unrestricted Hamiltonian must preserve full-scale H0 while doubling
   * the half-valued Mulliken shift, matching tblite's later unrestricted
   * eigensolver factor without doubling the core operator. */
  constexpr std::size_t unrestricted_system = 1u;
  const std::int64_t matrix_begin = fixture.mulliken.matrix_offsets()[unrestricted_system];
  const std::int64_t matrix_end = fixture.mulliken.matrix_offsets()[unrestricted_system + 1u];
  const std::int64_t matrix_count = matrix_end - matrix_begin;
  const std::int64_t density_begin =
      fixture.wavefunction_layout.density.system_offsets[unrestricted_system];
  std::vector<double> expected(
      static_cast<std::size_t>(fixture.wavefunction_layout.density.element_count), 0.0);
  for (std::int32_t spin = 0; spin < 2; ++spin) {
    std::copy_n(fixture.h0.data() + matrix_begin, static_cast<std::size_t>(matrix_count),
                expected.data() + density_begin + spin * matrix_count);
  }
  std::vector<double> mulliken_scratch(
      static_cast<std::size_t>(fixture.mulliken.hamiltonian_scratch_elements()));
  const MullikenWorkspace workspace{mulliken_scratch.data(),
                                    fixture.mulliken.hamiltonian_scratch_elements()};
  const MullikenPotentialView potential{fixture.driver_workspace.shell_potentials,
                                        fixture.wavefunction_layout.qsh.element_count,
                                        fixture.mulliken.identity()};
  const MullikenHamiltonianView hamiltonian{expected.data(),
                                            fixture.wavefunction_layout.density.element_count,
                                            fixture.mulliken.identity()};
  CHECK(add_mulliken_hamiltonian_system_cpu(fixture.mulliken, fixture.geometry.integrals, potential,
                                            hamiltonian, unrestricted_system, workspace,
                                            error) == XTBLOOM_STATUS_SUCCESS);
  for (std::int32_t spin = 0; spin < 2; ++spin) {
    for (std::int64_t matrix = 0; matrix < matrix_count; ++matrix) {
      const std::size_t index =
          static_cast<std::size_t>(density_begin + spin * matrix_count + matrix);
      const double h0 = fixture.h0[static_cast<std::size_t>(matrix_begin + matrix)];
      expected[index] = std::fma(2.0, expected[index] - h0, h0);
      CHECK(near(fixture.driver_workspace.hamiltonian[index], expected[index], 5.0e-13));
    }
  }
  return 0;
}

int test_max_iterations_and_exact_restart_population() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(mixed_input(), 1u, 0.0, 1.0e-30, 1.0e-30, false, error));
  CHECK(fixture.iterate(error) == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(fixture.driver_state.system_statuses[1] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(fixture.driver_state.iterations[0] == 1u && fixture.driver_state.iterations[1] == 1u);
  CHECK(fixture.mixer_state.iterations[0] == 1u && fixture.mixer_state.iterations[1] == 1u);
  CHECK(fixture.iterate(error) == XTBLOOM_STATUS_SUCCESS);
  /* A terminal data failure is explicitly restarted from the public raw
   * population. Restart deliberately accepts the failure-status mismatch and
   * resets both the driver and mixer as one transaction. */
  CHECK(restart_scc_driver_system_cpu(fixture.driver, 1, fixture.wavefunction, fixture.mixer_state,
                                      fixture.driver_state, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.iterations[1] == 0u && fixture.mixer_state.iterations[1] == 0u);
  CHECK(fixture.driver_state.converged[1] == 0u && fixture.mixer_state.converged[1] == 0u);
  const std::int64_t begin = fixture.wavefunction_layout.qsh.system_offsets[1];
  const std::int64_t end = fixture.wavefunction_layout.qsh.system_offsets[2];
  const std::int64_t vector_begin = fixture.mixer.vector_offsets()[1];
  for (std::int64_t element = 0; element < end - begin; ++element) {
    CHECK(fixture.mixer_state.current_inputs[vector_begin + element] ==
          fixture.wavefunction.qsh[begin + element]);
  }
  return 0;
}

int test_eigensolver_failure_preserves_successful_peer() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(two_restricted_input(), 1u, 0.0, 1.0e100, 1.0e100, false, error));
  const auto wavefunction_before = snapshot(*fixture.wavefunction_storage);
  diagonalization_calls.store(0, std::memory_order_relaxed);
  failed_diagonalization_call.store(0, std::memory_order_relaxed);
  CHECK(fixture.iterate(error) == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  failed_diagonalization_call.store(-1, std::memory_order_relaxed);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  CHECK(fixture.driver_state.system_statuses[1] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.iterations[0] == 1u && fixture.driver_state.iterations[1] == 1u);
  CHECK(std::isnan(fixture.driver_state.free_energies[0]));
  CHECK(std::isfinite(fixture.driver_state.free_energies[1]));

  const auto* before = reinterpret_cast<const double*>(
      wavefunction_before.data() + fixture.wavefunction_layout.density.offset_bytes);
  const std::int64_t first_end = fixture.wavefunction_layout.density.system_offsets[1];
  CHECK(std::equal(fixture.wavefunction.density, fixture.wavefunction.density + first_end, before));
  const std::int64_t second_begin = first_end;
  const std::int64_t second_end = fixture.wavefunction_layout.density.system_offsets[2];
  CHECK(!std::equal(fixture.wavefunction.density + second_begin,
                    fixture.wavefunction.density + second_end, before + second_begin));
  return 0;
}

int test_mixer_and_nonfinite_history_failures_are_transactional() {
  Fixture mixer_failure;
  std::string error;
  CHECK(mixer_failure.initialize(two_restricted_input(), 2u, 0.0, 1.0e-30, 1.0e-30, false, error));
  const std::int64_t first_vector_end = mixer_failure.mixer.vector_offsets()[1];
  std::fill_n(mixer_failure.mixer_state.current_inputs, static_cast<std::size_t>(first_vector_end),
              std::numeric_limits<double>::max());
  const auto wavefunction_before = snapshot(*mixer_failure.wavefunction_storage);
  CHECK(mixer_failure.iterate(error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(mixer_failure.driver_state.system_statuses[0] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(mixer_failure.mixer_state.system_statuses[0] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(mixer_failure.driver_state.system_statuses[1] == XTBLOOM_STATUS_SUCCESS);
  CHECK(mixer_failure.mixer_state.iterations[0] == 0u);
  CHECK(mixer_failure.mixer_state.iterations[1] == 1u);
  const auto failed_field_unchanged = [&](const WavefunctionFieldLayout& layout,
                                          const double* current) {
    const auto* before =
        reinterpret_cast<const double*>(wavefunction_before.data() + layout.offset_bytes);
    const std::int64_t end = layout.system_offsets[1];
    return std::equal(current, current + end, before);
  };
  CHECK(failed_field_unchanged(mixer_failure.wavefunction_layout.coefficients,
                               mixer_failure.wavefunction.coefficients));
  CHECK(failed_field_unchanged(mixer_failure.wavefunction_layout.eigenvalues,
                               mixer_failure.wavefunction.eigenvalues));
  CHECK(failed_field_unchanged(mixer_failure.wavefunction_layout.occupations,
                               mixer_failure.wavefunction.occupations));
  CHECK(failed_field_unchanged(mixer_failure.wavefunction_layout.density,
                               mixer_failure.wavefunction.density));
  CHECK(failed_field_unchanged(mixer_failure.wavefunction_layout.qsh,
                               mixer_failure.wavefunction.qsh));
  CHECK(failed_field_unchanged(mixer_failure.wavefunction_layout.qat,
                               mixer_failure.wavefunction.qat));
  CHECK(failed_field_unchanged(mixer_failure.wavefunction_layout.energy_weighted_density,
                               mixer_failure.wavefunction.energy_weighted_density));

  Fixture history_failure;
  CHECK(
      history_failure.initialize(two_restricted_input(), 3u, 0.0, 1.0e-30, 1.0e-30, false, error));
  CHECK(history_failure.iterate(error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(history_failure.driver_state.iterations[0] == 1u);
  history_failure.driver_state.free_energies[0] = std::numeric_limits<double>::quiet_NaN();
  const auto history_wavefunction_before = snapshot(*history_failure.wavefunction_storage);
  const std::uint64_t peer_iteration_before = history_failure.driver_state.iterations[1];
  CHECK(history_failure.iterate(error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(history_failure.driver_state.system_statuses[0] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(history_failure.mixer_state.system_statuses[0] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(history_failure.mixer_state.iterations[0] == 1u);
  CHECK(history_failure.driver_state.iterations[0] == 2u);
  CHECK(history_failure.driver_state.iterations[1] == peer_iteration_before + 1u);
  const auto* history_before =
      reinterpret_cast<const double*>(history_wavefunction_before.data() +
                                      history_failure.wavefunction_layout.density.offset_bytes);
  const std::int64_t system0_density_end =
      history_failure.wavefunction_layout.density.system_offsets[1];
  CHECK(std::equal(history_failure.wavefunction.density,
                   history_failure.wavefunction.density + system0_density_end, history_before));
  return 0;
}

int test_binding_state_agreement_and_component_sealing() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(mixed_input(), 3u, 0.0, 1.0e-10, 1.0e-10, false, error));
  const auto state_before = snapshot(*fixture.driver_state_storage);
  const auto wavefunction_before = snapshot(*fixture.wavefunction_storage);
  const auto mixer_before = snapshot(*fixture.mixer_state_storage);

  SccDriverState bad_state = fixture.driver_state;
  ++bad_state.free_energies;
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, bad_state, fixture.driver_workspace,
                                     error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(equals_snapshot(*fixture.driver_state_storage, state_before));
  CHECK(equals_snapshot(*fixture.wavefunction_storage, wavefunction_before));
  CHECK(equals_snapshot(*fixture.mixer_state_storage, mixer_before));

  SccDriverWorkspace bad_workspace = fixture.driver_workspace;
  ++bad_workspace.eigensolver_workspace.coefficients;
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state, bad_workspace,
                                     error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  fixture.driver_state.converged[0] = 1u;
  CHECK(fixture.iterate(error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  fixture.driver_state.converged[0] = 0u;
  ++fixture.driver_state.iterations[0];
  CHECK(fixture.iterate(error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  fixture.driver_state.iterations[0] = fixture.mixer_state.iterations[0];

  ES3Plan forged_es3 = fixture.es3;
  ++forged_es3.total_atoms;
  SccDriverPlan sentinel = fixture.driver;
  CHECK(make_scc_driver_plan(fixture.wavefunction_layout, fixture.mulliken, fixture.es2, forged_es3,
                             fixture.spin, fixture.eigensolver, fixture.mixer, nullptr, 3u, 0.0,
                             1.0e-10, sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(sentinel.identity() == fixture.driver.identity());

  SpinPolarizationPlan forged_spin = fixture.spin;
  forged_spin.spin_channels[0] = 2;
  CHECK(make_scc_driver_plan(fixture.wavefunction_layout, fixture.mulliken, fixture.es2,
                             fixture.es3, forged_spin, fixture.eigensolver, fixture.mixer, nullptr,
                             3u, 0.0, 1.0e-10, sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(sentinel.identity() == fixture.driver.identity());
  return 0;
}

int test_stationary_projection_mixed_spin_and_transaction() {
  Fixture fixture;
  std::string error;
  CHECK(fixture.initialize(mixed_input(), 3u, 0.0, 1.0e-10, 1.0e-10, false, error));

  for (std::int64_t element = 0; element < fixture.wavefunction_layout.density.element_count;
       ++element) {
    fixture.wavefunction.density[element] = 0.01 * static_cast<double>(element + 1);
    fixture.wavefunction.energy_weighted_density[element] =
        -0.02 * static_cast<double>(element + 1);
  }
  for (std::int64_t element = 0; element < fixture.wavefunction_layout.qsh.element_count;
       ++element) {
    fixture.wavefunction.qsh[element] = 0.1 * static_cast<double>(element + 1);
    fixture.driver_workspace.shell_potentials[element] = -0.3 * static_cast<double>(element + 1);
  }
  for (std::int64_t element = 0; element < fixture.wavefunction_layout.qat.element_count;
       ++element) {
    fixture.wavefunction.qat[element] = 0.2 * static_cast<double>(element + 1);
  }

  std::int64_t matrices = 0;
  for (std::int64_t system = 0; system < fixture.wavefunction_layout.batch_size; ++system) {
    const std::int64_t orbitals =
        fixture.wavefunction_layout.batch_orbital_offsets[static_cast<std::size_t>(system) + 1u] -
        fixture.wavefunction_layout.batch_orbital_offsets[static_cast<std::size_t>(system)];
    matrices += orbitals * orbitals;
  }
  std::vector<double> density(static_cast<std::size_t>(matrices), 91.0);
  std::vector<double> weighted(static_cast<std::size_t>(matrices), 92.0);
  std::vector<double> spin_density(static_cast<std::size_t>(matrices), 93.0);
  std::vector<double> shell_charges(
      static_cast<std::size_t>(fixture.wavefunction_layout.total_shells), 94.0);
  std::vector<double> atomic_charges(
      static_cast<std::size_t>(fixture.wavefunction_layout.total_atoms), 95.0);
  std::vector<double> scalar_potentials(shell_charges.size(), 96.0);
  std::vector<double> spin_potentials(shell_charges.size(), 97.0);
  const SccStationaryProjection projection{density.data(),
                                           weighted.data(),
                                           spin_density.data(),
                                           matrices,
                                           shell_charges.data(),
                                           atomic_charges.data(),
                                           scalar_potentials.data(),
                                           spin_potentials.data(),
                                           fixture.wavefunction_layout.total_shells,
                                           fixture.wavefunction_layout.total_atoms};
  CHECK(project_scc_stationary_state_cpu(fixture.wavefunction_layout, fixture.wavefunction,
                                         fixture.driver_workspace.shell_potentials,
                                         fixture.wavefunction_layout.qsh.element_count, projection,
                                         error) == XTBLOOM_STATUS_SUCCESS);

  std::int64_t compact_matrix = 0;
  for (std::int64_t system = 0; system < fixture.wavefunction_layout.batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    const std::int64_t orbitals = fixture.wavefunction_layout.batch_orbital_offsets[index + 1u] -
                                  fixture.wavefunction_layout.batch_orbital_offsets[index];
    const std::int64_t count = orbitals * orbitals;
    const std::int64_t packed = fixture.wavefunction_layout.density.system_offsets[index];
    for (std::int64_t element = 0; element < count; ++element) {
      const double alpha = fixture.wavefunction.density[packed + element];
      const double alpha_w = fixture.wavefunction.energy_weighted_density[packed + element];
      if (fixture.wavefunction_layout.spin_channels[index] == 1) {
        CHECK(near(density[compact_matrix + element], alpha));
        CHECK(near(weighted[compact_matrix + element], alpha_w));
        CHECK(spin_density[compact_matrix + element] == 0.0);
      } else {
        const double beta = fixture.wavefunction.density[packed + count + element];
        const double beta_w =
            fixture.wavefunction.energy_weighted_density[packed + count + element];
        CHECK(near(density[compact_matrix + element], alpha + beta));
        CHECK(near(weighted[compact_matrix + element], alpha_w + beta_w));
        CHECK(near(spin_density[compact_matrix + element], alpha - beta));
      }
    }
    const std::int64_t shell_begin = fixture.wavefunction_layout.batch_shell_offsets[index];
    const std::int64_t shells =
        fixture.wavefunction_layout.batch_shell_offsets[index + 1u] - shell_begin;
    const std::int64_t qsh = fixture.wavefunction_layout.qsh.system_offsets[index];
    for (std::int64_t shell = 0; shell < shells; ++shell) {
      CHECK(near(shell_charges[shell_begin + shell], fixture.wavefunction.qsh[qsh + shell]));
      CHECK(near(scalar_potentials[shell_begin + shell],
                 fixture.driver_workspace.shell_potentials[qsh + shell]));
      const double expected_spin =
          fixture.wavefunction_layout.spin_channels[index] == 2
              ? fixture.driver_workspace.shell_potentials[qsh + shells + shell]
              : 0.0;
      CHECK(near(spin_potentials[shell_begin + shell], expected_spin));
    }
    const std::int64_t atom_begin = fixture.wavefunction_layout.atom_offsets[index];
    const std::int64_t atoms = fixture.wavefunction_layout.atom_offsets[index + 1u] - atom_begin;
    const std::int64_t qat = fixture.wavefunction_layout.qat.system_offsets[index];
    for (std::int64_t atom = 0; atom < atoms; ++atom) {
      CHECK(near(atomic_charges[atom_begin + atom], fixture.wavefunction.qat[qat + atom]));
    }
    compact_matrix += count;
  }

  const std::vector<double> density_before = density;
  const std::vector<double> weighted_before = weighted;
  const std::vector<double> spin_before = spin_density;
  const std::vector<double> shell_before = shell_charges;
  const std::vector<double> atom_before = atomic_charges;
  const std::vector<double> scalar_before = scalar_potentials;
  const std::vector<double> spin_potential_before = spin_potentials;
  const std::int64_t unrestricted_begin = fixture.wavefunction_layout.density.system_offsets[1];
  const double saved = fixture.wavefunction.density[unrestricted_begin];
  fixture.wavefunction.density[unrestricted_begin] = std::numeric_limits<double>::max();
  fixture.wavefunction
      .density[unrestricted_begin +
               (fixture.wavefunction_layout.density.system_offsets[2] - unrestricted_begin) / 2] =
      std::numeric_limits<double>::max();
  CHECK(project_scc_stationary_state_cpu(fixture.wavefunction_layout, fixture.wavefunction,
                                         fixture.driver_workspace.shell_potentials,
                                         fixture.wavefunction_layout.qsh.element_count, projection,
                                         error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(density == density_before && weighted == weighted_before && spin_density == spin_before &&
        shell_charges == shell_before && atomic_charges == atom_before &&
        scalar_potentials == scalar_before && spin_potentials == spin_potential_before);
  fixture.wavefunction.density[unrestricted_begin] = saved;
  return 0;
}

}  // namespace

int main() {
  if (const int result = test_restricted_unrestricted_free_energy_periodic_and_point_potential();
      result != 0) {
    return result;
  }
  if (const int result = test_max_iterations_and_exact_restart_population(); result != 0) {
    return result;
  }
  if (const int result = test_eigensolver_failure_preserves_successful_peer(); result != 0) {
    return result;
  }
  if (const int result = test_mixer_and_nonfinite_history_failures_are_transactional();
      result != 0) {
    return result;
  }
  if (const int result = test_binding_state_agreement_and_component_sealing(); result != 0) {
    return result;
  }
  return test_stationary_projection_mixed_spin_and_transaction();
}
