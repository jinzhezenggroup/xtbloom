#include "model/gfn2/scc_driver.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"

namespace allocation_test {
std::atomic<std::size_t> count{0u};
std::atomic<bool> enabled{false};
}  // namespace allocation_test

void* operator new(std::size_t size) {
  if (allocation_test::enabled.load(std::memory_order_relaxed)) {
    allocation_test::count.fetch_add(1u, std::memory_order_relaxed);
  }
  if (void* pointer = std::malloc(size == 0u ? 1u : size); pointer != nullptr) {
    return pointer;
  }
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete[](pointer); }

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using namespace xtbloom::detail::gfn2;

struct AlignedBuffer {
  void* data = nullptr;
  std::size_t size = 0u;

  explicit AlignedBuffer(std::size_t requested) {
    size = (std::max<std::size_t>(requested, 1u) + 63u) & ~std::size_t{63u};
    data = std::aligned_alloc(64u, size);
    if (data != nullptr) {
      std::memset(data, 0, size);
    }
  }
  ~AlignedBuffer() { std::free(data); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

std::atomic<int> diagonalizations{0};

LapackInt tiny_dpotrf(LapackInt, char, LapackInt n, double* matrix, LapackInt) {
  for (LapackInt column = 0; column < n; ++column) {
    for (LapackInt row = column; row < n; ++row) {
      double value = matrix[column * n + row];
      for (LapackInt inner = 0; inner < column; ++inner) {
        value -= matrix[inner * n + row] * matrix[inner * n + column];
      }
      if (row == column) {
        if (!(value > 0.0) || !std::isfinite(value)) {
          return column + 1;
        }
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
  if (n <= 0) {
    return -3;
  }
  *reciprocal_condition = 1.0;
  return 0;
}

LapackInt tiny_dsyevd(LapackInt, char, char, LapackInt n, double* matrix, LapackInt,
                      double* eigenvalues, double*, LapackInt, LapackInt*, LapackInt) {
  if (n <= 0 || n > 16) {
    return -4;
  }
  diagonalizations.fetch_add(1, std::memory_order_relaxed);
  std::array<double, 16u * 16u> values{};
  std::array<double, 16u * 16u> vectors{};
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
  if (!converged) {
    return 1;
  }
  std::array<LapackInt, 16> order{};
  for (LapackInt index = 0; index < n; ++index) {
    order[static_cast<std::size_t>(index)] = index;
  }
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

void tiny_dtrsm(int, int side, int, int transpose, int, LapackInt rows, LapackInt columns,
                double alpha, const double* triangular, LapackInt, double* rhs, LapackInt) {
  constexpr int kLeft = 141;
  constexpr int kNoTrans = 111;
  if (side == kLeft) {
    for (LapackInt column = 0; column < columns; ++column) {
      if (transpose == kNoTrans) {
        for (LapackInt row = 0; row < rows; ++row) {
          double value = alpha * rhs[column * rows + row];
          for (LapackInt inner = 0; inner < row; ++inner) {
            value -= triangular[inner * rows + row] * rhs[column * rows + inner];
          }
          rhs[column * rows + row] = value / triangular[row * rows + row];
        }
      } else {
        for (LapackInt row = rows; row-- > 0;) {
          double value = alpha * rhs[column * rows + row];
          for (LapackInt inner = row + 1; inner < rows; ++inner) {
            value -= triangular[row * rows + inner] * rhs[column * rows + inner];
          }
          rhs[column * rows + row] = value / triangular[row * rows + row];
        }
      }
    }
  } else {
    /* The eigensolver uses only right-side L^T solves. Each matrix row then
     * reduces to an independent forward substitution with L. */
    for (LapackInt row = 0; row < rows; ++row) {
      for (LapackInt column = 0; column < columns; ++column) {
        double value = alpha * rhs[column * rows + row];
        for (LapackInt inner = 0; inner < column; ++inner) {
          value -= rhs[inner * rows + row] * triangular[inner * columns + column];
        }
        rhs[column * rows + row] = value / triangular[column * columns + column];
      }
    }
  }
}

void tiny_dgemm(int, int, int, LapackInt rows, LapackInt columns, LapackInt inner, double alpha,
                const double* left, LapackInt leading_left, const double* right,
                LapackInt leading_right, double beta, double* result, LapackInt leading_result) {
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
    if (make_internal_test_lp64_backend(&tiny_dpotrf, &tiny_dpocon, &tiny_dsyevd, &tiny_dtrsm,
                                        &tiny_dgemm, nullptr, result,
                                        error) != XTBLOOM_STATUS_SUCCESS) {
      std::abort();
    }
    return result;
  }();
  return value;
}

struct Fixture {
  std::int64_t batch_size = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> charges;
  std::vector<std::int32_t> unpaired;
  std::vector<std::int32_t> spins;
  std::vector<double> coordination;

  BasisPlan basis;
  IntegralPlan integral_plan;
  H0Plan h0_plan;
  WavefunctionLayout wavefunction_layout;
  ES2Plan es2_plan;
  ES3Plan es3_plan;
  AES2Plan aes2_plan;
  MullikenPlan mulliken_plan;
  EigensolverPlan eigensolver_plan;
  SccMixerPlan mixer_plan;
  D4Plan d4_plan;
  PeriodicEmbeddingPlan periodic_plan;
  SccDriverPlan driver_plan;

  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response;

  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<double> h0;
  std::unique_ptr<AlignedBuffer> integral_scratch;

  std::unique_ptr<AlignedBuffer> es2_storage;
  std::unique_ptr<AlignedBuffer> es2_scratch_storage;
  ES2GeometryCache es2_cache;
  ES2Workspace es2_scratch;
  std::unique_ptr<AlignedBuffer> aes2_storage;
  std::unique_ptr<AlignedBuffer> aes2_scratch_storage;
  AES2GeometryCache aes2_cache;
  AES2Workspace aes2_scratch;
  std::vector<double> d4_pair_data;
  std::vector<double> d4_coordination;
  D4GeometryCache d4_cache;

  std::unique_ptr<AlignedBuffer> wavefunction_storage;
  WavefunctionView wavefunction;
  std::unique_ptr<AlignedBuffer> overlap_cache_storage;
  EigensolverOverlapCache overlap_cache;
  std::unique_ptr<AlignedBuffer> eigensolver_scratch_storage;
  EigensolverWorkspace eigensolver_scratch;
  std::unique_ptr<AlignedBuffer> mixer_state_storage;
  SccMixerState mixer_state;
  std::unique_ptr<AlignedBuffer> driver_state_storage;
  SccDriverState driver_state;
  std::unique_ptr<AlignedBuffer> driver_scratch_storage;
  SccDriverWorkspace driver_scratch;
  SccDriverGeometryView geometry;
};

struct ComponentPlans {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  BasisPlan basis;
  IntegralPlan integrals;
  WavefunctionLayout wavefunction;
  MullikenPlan mulliken;
  ES2Plan es2;
  ES3Plan es3;
  AES2Plan aes2;
  EigensolverPlan eigensolver;
  SccMixerPlan mixer;
};

struct FixtureTopology {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
};

std::size_t append_test_segment(std::size_t cursor, std::size_t bytes, std::size_t alignment) {
  const std::size_t aligned = (cursor + alignment - 1u) & ~(alignment - 1u);
  return aligned + bytes;
}

std::size_t expected_disabled_state_size(const Fixture& fixture) {
  const std::size_t batch = static_cast<std::size_t>(fixture.batch_size);
  std::size_t cursor = 0u;
  /* Five legacy thermodynamic traces plus the seven complete-energy traces
   * that remain when optional D4 and periodic embedding are disabled. */
  for (int field = 0; field < 12; ++field) {
    cursor = append_test_segment(cursor, batch * sizeof(double), alignof(double));
  }
  cursor = append_test_segment(cursor, batch * sizeof(std::uint64_t), alignof(std::uint64_t));
  cursor = append_test_segment(cursor, batch * sizeof(xtbloom_status_t), alignof(xtbloom_status_t));
  cursor = append_test_segment(cursor, batch * sizeof(std::uint8_t), alignof(std::uint8_t));
  cursor = append_test_segment(cursor, batch * sizeof(std::uint8_t), alignof(std::uint8_t));
  return append_test_segment(cursor, 0u, kSccDriverWorkspaceAlignment);
}

std::size_t expected_disabled_workspace_size(const Fixture& fixture) {
  const WavefunctionLayout& wavefunction = fixture.wavefunction_layout;
  const std::size_t batch = static_cast<std::size_t>(fixture.batch_size);
  const auto doubles = [](std::int64_t count) {
    return static_cast<std::size_t>(count) * sizeof(double);
  };
  std::size_t cursor = 0u;
  cursor = append_test_segment(cursor, wavefunction.workspace_size_bytes,
                               kWavefunctionWorkspaceAlignment);
  cursor =
      append_test_segment(cursor, doubles(wavefunction.density.element_count), alignof(double));
  cursor = append_test_segment(cursor, doubles(wavefunction.total_shells), alignof(double));
  cursor = append_test_segment(cursor, doubles(wavefunction.total_atoms), alignof(double));
  cursor = append_test_segment(cursor, 3u * doubles(wavefunction.total_atoms), alignof(double));
  cursor = append_test_segment(cursor, 6u * doubles(wavefunction.total_atoms), alignof(double));
  cursor = append_test_segment(cursor, doubles(wavefunction.total_shells), alignof(double));
  cursor = append_test_segment(cursor, doubles(wavefunction.total_atoms), alignof(double));
  cursor = append_test_segment(cursor, 3u * doubles(wavefunction.total_atoms), alignof(double));
  cursor = append_test_segment(cursor, 6u * doubles(wavefunction.total_atoms), alignof(double));
  cursor = append_test_segment(cursor, doubles(wavefunction.qat.element_count), alignof(double));
  cursor = append_test_segment(cursor, doubles(wavefunction.qsh.element_count), alignof(double));
  cursor = append_test_segment(cursor, doubles(wavefunction.dipole.element_count), alignof(double));
  cursor =
      append_test_segment(cursor, doubles(wavefunction.quadrupole.element_count), alignof(double));
  cursor = append_test_segment(cursor, doubles(wavefunction.qsh.element_count), alignof(double));
  cursor = append_test_segment(cursor, doubles(wavefunction.qsh.element_count), alignof(double));
  cursor = append_test_segment(cursor, doubles(wavefunction.qat.element_count), alignof(double));
  cursor = append_test_segment(cursor, doubles(wavefunction.dipole.element_count), alignof(double));
  cursor =
      append_test_segment(cursor, doubles(wavefunction.quadrupole.element_count), alignof(double));
  /* core, ES2, ES3, AES2, spin, explicit-PC, internal, and complete free energy */
  for (int field = 0; field < 8; ++field) {
    cursor = append_test_segment(cursor, batch * sizeof(double), alignof(double));
  }
  cursor = append_test_segment(cursor, doubles(wavefunction.total_shells), alignof(double));
  cursor = append_test_segment(cursor, doubles(fixture.aes2_plan.potential_scratch_elements()),
                               alignof(double));
  cursor =
      append_test_segment(cursor,
                          doubles(std::max(fixture.mulliken_plan.population_scratch_elements(),
                                           fixture.mulliken_plan.hamiltonian_scratch_elements())),
                          alignof(double));
  cursor = append_test_segment(cursor, fixture.eigensolver_plan.worker_workspace_size_bytes(),
                               kEigensolverWorkspaceAlignment);
  cursor = append_test_segment(cursor, fixture.mixer_plan.state_size_bytes(),
                               kSccMixerWorkspaceAlignment);
  cursor = append_test_segment(cursor, fixture.mixer_plan.workspace_size_bytes(),
                               kSccMixerWorkspaceAlignment);
  cursor = append_test_segment(cursor, batch * sizeof(xtbloom_status_t), alignof(xtbloom_status_t));
  cursor = append_test_segment(cursor, 2u * batch * sizeof(double), alignof(double));
  for (int field = 0; field < 3; ++field) {
    cursor = append_test_segment(cursor, batch * sizeof(double), alignof(double));
  }
  cursor = append_test_segment(cursor, batch * sizeof(std::uint8_t), alignof(std::uint8_t));
  return append_test_segment(cursor, 0u, kSccDriverWorkspaceAlignment);
}

/* One system's complete mixer history snapshot for byte-equality checks. */
struct MixerSystemSnapshot {
  std::vector<double> current_inputs;
  std::vector<double> previous_inputs;
  std::vector<double> previous_residuals;
  std::vector<double> df_history;
  std::vector<double> u_history;
  std::vector<double> omega;
  double residual_rms = 0.0;
  double residual_maximum = 0.0;
  std::uint64_t iterations = 0u;
  std::uint64_t restart_counts = 0u;
  xtbloom_status_t system_status = XTBLOOM_STATUS_SUCCESS;
  std::uint8_t initialized = 0u;
  std::uint8_t converged = 0u;
};

MixerSystemSnapshot snapshot_mixer_system(const Fixture& fixture, std::size_t system) {
  const std::vector<std::int64_t>& offsets = fixture.mixer_plan.vector_offsets();
  const std::size_t memory = static_cast<std::size_t>(fixture.mixer_plan.history_size());
  const std::size_t dimension = static_cast<std::size_t>(offsets[system + 1u] - offsets[system]);
  const std::size_t vector_offset = static_cast<std::size_t>(offsets[system]);
  std::size_t history_offset = 0u;
  for (std::size_t previous = 0u; previous < system; ++previous) {
    history_offset += static_cast<std::size_t>(offsets[previous + 1u] - offsets[previous]);
  }
  history_offset *= memory;
  MixerSystemSnapshot snapshot;
  snapshot.current_inputs.assign(fixture.mixer_state.current_inputs + vector_offset,
                                 fixture.mixer_state.current_inputs + vector_offset + dimension);
  snapshot.previous_inputs.assign(fixture.mixer_state.previous_inputs + vector_offset,
                                  fixture.mixer_state.previous_inputs + vector_offset + dimension);
  snapshot.previous_residuals.assign(
      fixture.mixer_state.previous_residuals + vector_offset,
      fixture.mixer_state.previous_residuals + vector_offset + dimension);
  snapshot.df_history.assign(fixture.mixer_state.df_history + history_offset,
                             fixture.mixer_state.df_history + history_offset + dimension * memory);
  snapshot.u_history.assign(fixture.mixer_state.u_history + history_offset,
                            fixture.mixer_state.u_history + history_offset + dimension * memory);
  snapshot.omega.assign(fixture.mixer_state.omega + system * memory,
                        fixture.mixer_state.omega + (system + 1u) * memory);
  snapshot.residual_rms = fixture.mixer_state.residual_rms[system];
  snapshot.residual_maximum = fixture.mixer_state.residual_maximum[system];
  snapshot.iterations = fixture.mixer_state.iterations[system];
  snapshot.restart_counts = fixture.mixer_state.restart_counts[system];
  snapshot.system_status = fixture.mixer_state.system_statuses[system];
  snapshot.initialized = fixture.mixer_state.initialized[system];
  snapshot.converged = fixture.mixer_state.converged[system];
  return snapshot;
}

bool mixer_snapshots_equal(const MixerSystemSnapshot& first, const MixerSystemSnapshot& second) {
  return first.current_inputs == second.current_inputs &&
         first.previous_inputs == second.previous_inputs &&
         first.previous_residuals == second.previous_residuals &&
         first.df_history == second.df_history && first.u_history == second.u_history &&
         first.omega == second.omega && first.residual_rms == second.residual_rms &&
         first.residual_maximum == second.residual_maximum &&
         first.iterations == second.iterations && first.restart_counts == second.restart_counts &&
         first.system_status == second.system_status && first.initialized == second.initialized &&
         first.converged == second.converged;
}

bool make_component_plans(std::vector<std::int32_t> atomic_numbers, double molecular_charge,
                          std::int32_t unpaired_electrons, ComponentPlans& plans,
                          std::string& error) {
  plans.atomic_numbers = std::move(atomic_numbers);
  plans.atom_offsets = {0, static_cast<std::int64_t>(plans.atomic_numbers.size())};
  plans.molecular_charges = {molecular_charge};
  plans.unpaired_electrons = {unpaired_electrons};
  plans.spin_channels = {1};
  return make_basis_plan(1, plans.atom_offsets.back(), plans.atom_offsets.data(),
                         plans.atomic_numbers.data(), plans.basis,
                         error) == XTBLOOM_STATUS_SUCCESS &&
         make_integral_plan(plans.basis, plans.integrals, error) == XTBLOOM_STATUS_SUCCESS &&
         make_wavefunction_layout(plans.basis, plans.atomic_numbers.data(),
                                  plans.molecular_charges.data(), plans.unpaired_electrons.data(),
                                  plans.spin_channels.data(), plans.wavefunction,
                                  error) == XTBLOOM_STATUS_SUCCESS &&
         make_mulliken_plan(plans.basis, plans.integrals, plans.wavefunction, plans.mulliken,
                            error) == XTBLOOM_STATUS_SUCCESS &&
         make_es2_plan(plans.basis, plans.atomic_numbers.data(), plans.es2, error) ==
             XTBLOOM_STATUS_SUCCESS &&
         make_es3_plan(plans.basis, plans.atomic_numbers.data(), plans.es3, error) ==
             XTBLOOM_STATUS_SUCCESS &&
         make_aes2_plan(plans.basis, plans.atomic_numbers.data(), plans.aes2, error) ==
             XTBLOOM_STATUS_SUCCESS &&
         make_eigensolver_plan(plans.wavefunction, plans.eigensolver, error) ==
             XTBLOOM_STATUS_SUCCESS &&
         make_scc_mixer_plan(plans.wavefunction, 3, 0.4, 1.0e-10, 1.0e-10, plans.mixer, error) ==
             XTBLOOM_STATUS_SUCCESS;
}

bool make_fixture(std::int64_t batch_size, Fixture& fixture, std::string& error,
                  std::uint64_t maximum_iterations = 5u, double mixer_tolerance = 1.0e-10,
                  bool enable_periodic_embedding = false, const FixtureTopology* topology = nullptr,
                  bool use_nullable_driver_overload = false, bool enable_d4 = false,
                  double energy_tolerance = 1.0e100, double electronic_temperature = 0.0,
                  bool use_compatibility_driver_overload = false) {
  fixture.batch_size = batch_size;
  if (topology == nullptr) {
    fixture.atom_offsets.resize(static_cast<std::size_t>(batch_size) + 1u);
    for (std::int64_t system = 0; system <= batch_size; ++system) {
      fixture.atom_offsets[static_cast<std::size_t>(system)] = system;
    }
    fixture.atomic_numbers.assign(static_cast<std::size_t>(batch_size), 1);
    fixture.positions.assign(static_cast<std::size_t>(3 * batch_size), 0.0);
    fixture.charges.assign(static_cast<std::size_t>(batch_size), 1.0);  // isolated H+
  } else {
    if (topology->atom_offsets.size() != static_cast<std::size_t>(batch_size) + 1u ||
        topology->atom_offsets.front() != 0 || topology->atom_offsets.back() <= 0 ||
        topology->atomic_numbers.size() !=
            static_cast<std::size_t>(topology->atom_offsets.back()) ||
        topology->positions.size() != 3u * topology->atomic_numbers.size() ||
        topology->molecular_charges.size() != static_cast<std::size_t>(batch_size) ||
        (!topology->unpaired_electrons.empty() &&
         topology->unpaired_electrons.size() != static_cast<std::size_t>(batch_size)) ||
        (!topology->spin_channels.empty() &&
         topology->spin_channels.size() != static_cast<std::size_t>(batch_size))) {
      error = "test fixture topology is malformed";
      return false;
    }
    fixture.atom_offsets = topology->atom_offsets;
    fixture.atomic_numbers = topology->atomic_numbers;
    fixture.positions = topology->positions;
    fixture.charges = topology->molecular_charges;
  }
  const std::int64_t total_atoms = fixture.atom_offsets.back();
  fixture.unpaired.assign(static_cast<std::size_t>(batch_size), 0);
  fixture.spins.assign(static_cast<std::size_t>(batch_size), 1);
  if (topology != nullptr && !topology->unpaired_electrons.empty()) {
    fixture.unpaired = topology->unpaired_electrons;
  }
  if (topology != nullptr && !topology->spin_channels.empty()) {
    fixture.spins = topology->spin_channels;
  }
  fixture.coordination.assign(static_cast<std::size_t>(total_atoms), 0.0);

  if (make_basis_plan(batch_size, total_atoms, fixture.atom_offsets.data(),
                      fixture.atomic_numbers.data(), fixture.basis,
                      error) != XTBLOOM_STATUS_SUCCESS ||
      make_integral_plan(fixture.basis, fixture.integral_plan, error) != XTBLOOM_STATUS_SUCCESS ||
      make_h0_plan(fixture.basis, fixture.integral_plan, fixture.atomic_numbers.data(),
                   fixture.h0_plan, error) != XTBLOOM_STATUS_SUCCESS ||
      make_wavefunction_layout(fixture.basis, fixture.atomic_numbers.data(), fixture.charges.data(),
                               fixture.unpaired.data(), fixture.spins.data(),
                               fixture.wavefunction_layout, error) != XTBLOOM_STATUS_SUCCESS ||
      make_es2_plan(fixture.basis, fixture.atomic_numbers.data(), fixture.es2_plan, error) !=
          XTBLOOM_STATUS_SUCCESS ||
      make_es3_plan(fixture.basis, fixture.atomic_numbers.data(), fixture.es3_plan, error) !=
          XTBLOOM_STATUS_SUCCESS ||
      make_aes2_plan(fixture.basis, fixture.atomic_numbers.data(), fixture.aes2_plan, error) !=
          XTBLOOM_STATUS_SUCCESS ||
      make_mulliken_plan(fixture.basis, fixture.integral_plan, fixture.wavefunction_layout,
                         fixture.mulliken_plan, error) != XTBLOOM_STATUS_SUCCESS ||
      make_eigensolver_plan(fixture.wavefunction_layout, fixture.eigensolver_plan, error) !=
          XTBLOOM_STATUS_SUCCESS ||
      make_scc_mixer_plan(fixture.wavefunction_layout, 3, 0.4, mixer_tolerance, mixer_tolerance,
                          fixture.mixer_plan, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  if (enable_periodic_embedding &&
      make_periodic_embedding_plan(batch_size, total_atoms, fixture.atom_offsets.data(),
                                   fixture.periodic_plan, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  if (enable_d4 && make_d4_plan(batch_size, total_atoms, fixture.atom_offsets.data(),
                                fixture.atomic_numbers.data(), fixture.d4_plan,
                                error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  xtbloom_status_t driver_status = XTBLOOM_STATUS_SUCCESS;
  if (!use_compatibility_driver_overload) {
    driver_status = make_scc_driver_plan(
        fixture.wavefunction_layout, fixture.mulliken_plan, fixture.es2_plan, fixture.es3_plan,
        fixture.aes2_plan, fixture.eigensolver_plan, fixture.mixer_plan,
        enable_d4 ? &fixture.d4_plan : nullptr,
        enable_periodic_embedding ? &fixture.periodic_plan : nullptr, maximum_iterations,
        electronic_temperature, energy_tolerance, fixture.driver_plan, error);
  } else if (enable_d4) {
    driver_status = make_scc_driver_plan(
        fixture.wavefunction_layout, fixture.mulliken_plan, fixture.es2_plan, fixture.es3_plan,
        fixture.aes2_plan, fixture.eigensolver_plan, fixture.mixer_plan, &fixture.d4_plan,
        enable_periodic_embedding ? &fixture.periodic_plan : nullptr, maximum_iterations,
        electronic_temperature, fixture.driver_plan, error);
  } else if (use_nullable_driver_overload || enable_periodic_embedding) {
    driver_status = make_scc_driver_plan(
        fixture.wavefunction_layout, fixture.mulliken_plan, fixture.es2_plan, fixture.es3_plan,
        fixture.aes2_plan, fixture.eigensolver_plan, fixture.mixer_plan,
        enable_periodic_embedding ? &fixture.periodic_plan : nullptr, maximum_iterations,
        electronic_temperature, fixture.driver_plan, error);
  } else {
    driver_status = make_scc_driver_plan(
        fixture.wavefunction_layout, fixture.mulliken_plan, fixture.es2_plan, fixture.es3_plan,
        fixture.aes2_plan, fixture.eigensolver_plan, fixture.mixer_plan, maximum_iterations,
        electronic_temperature, fixture.driver_plan, error);
  }
  if (driver_status != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  if (enable_periodic_embedding) {
    fixture.periodic_shifts.assign(static_cast<std::size_t>(total_atoms), 0.0);
    fixture.periodic_response.assign(
        static_cast<std::size_t>(fixture.periodic_plan.total_matrix_elements()), 0.0);
  }

  fixture.overlap.resize(static_cast<std::size_t>(fixture.integral_plan.total_matrix_elements));
  fixture.dipole_integrals.resize(
      static_cast<std::size_t>(3 * fixture.integral_plan.total_matrix_elements));
  fixture.quadrupole_integrals.resize(
      static_cast<std::size_t>(6 * fixture.integral_plan.total_matrix_elements));
  fixture.h0.resize(static_cast<std::size_t>(fixture.integral_plan.total_matrix_elements));
  fixture.integral_scratch =
      std::make_unique<AlignedBuffer>(fixture.integral_plan.workspace_size_bytes);
  if (evaluate_overlap_cpu(fixture.basis, fixture.integral_plan, fixture.positions.data(),
                           fixture.overlap.data(), fixture.integral_scratch->data,
                           fixture.integral_scratch->size, error) != XTBLOOM_STATUS_SUCCESS ||
      evaluate_multipole_cpu(fixture.basis, fixture.integral_plan, fixture.positions.data(),
                             fixture.dipole_integrals.data(), fixture.quadrupole_integrals.data(),
                             fixture.integral_scratch->data, fixture.integral_scratch->size,
                             error) != XTBLOOM_STATUS_SUCCESS ||
      evaluate_h0_cpu(fixture.basis, fixture.integral_plan, fixture.h0_plan,
                      fixture.positions.data(), fixture.coordination.data(), fixture.overlap.data(),
                      fixture.h0.data(), error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }

  fixture.es2_storage = std::make_unique<AlignedBuffer>(
      static_cast<std::size_t>(fixture.es2_plan.total_matrix_elements()) * sizeof(double));
  fixture.es2_scratch_storage = std::make_unique<AlignedBuffer>(
      static_cast<std::size_t>(fixture.es2_plan.total_matrix_elements()) * sizeof(double));
  fixture.es2_scratch.matrix_scratch = static_cast<double*>(fixture.es2_scratch_storage->data);
  fixture.es2_scratch.matrix_elements = fixture.es2_plan.total_matrix_elements();
  if (update_es2_geometry_cache_cpu(
          fixture.es2_plan, fixture.positions.data(), 1u,
          static_cast<double*>(fixture.es2_storage->data),
          static_cast<std::size_t>(fixture.es2_plan.total_matrix_elements()), fixture.es2_scratch,
          fixture.es2_cache, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }

  fixture.aes2_storage = std::make_unique<AlignedBuffer>(
      static_cast<std::size_t>(fixture.aes2_plan.pair_data_elements()) * sizeof(double));
  fixture.aes2_scratch_storage = std::make_unique<AlignedBuffer>(
      static_cast<std::size_t>(fixture.aes2_plan.pair_data_elements()) * sizeof(double));
  fixture.aes2_scratch.pair_scratch = static_cast<double*>(fixture.aes2_scratch_storage->data);
  fixture.aes2_scratch.pair_elements = fixture.aes2_plan.pair_data_elements();
  if (update_aes2_geometry_cache_cpu(
          fixture.aes2_plan, fixture.positions.data(), fixture.coordination.data(), 1u,
          static_cast<double*>(fixture.aes2_storage->data),
          static_cast<std::size_t>(fixture.aes2_plan.pair_data_elements()), fixture.aes2_scratch,
          fixture.aes2_cache, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }

  fixture.wavefunction_storage =
      std::make_unique<AlignedBuffer>(fixture.wavefunction_layout.workspace_size_bytes);
  fixture.overlap_cache_storage =
      std::make_unique<AlignedBuffer>(fixture.eigensolver_plan.overlap_cache_size_bytes());
  fixture.eigensolver_scratch_storage =
      std::make_unique<AlignedBuffer>(fixture.eigensolver_plan.workspace_size_bytes());
  fixture.mixer_state_storage =
      std::make_unique<AlignedBuffer>(fixture.mixer_plan.state_size_bytes());
  fixture.driver_state_storage =
      std::make_unique<AlignedBuffer>(fixture.driver_plan.state_size_bytes());
  fixture.driver_scratch_storage =
      std::make_unique<AlignedBuffer>(fixture.driver_plan.workspace_size_bytes());
  if (bind_wavefunction_view(fixture.wavefunction_layout, fixture.wavefunction_storage->data,
                             fixture.wavefunction_storage->size, fixture.wavefunction,
                             error) != XTBLOOM_STATUS_SUCCESS ||
      initialize_sad_multipole_state(fixture.wavefunction_layout, fixture.wavefunction, error) !=
          XTBLOOM_STATUS_SUCCESS ||
      bind_eigensolver_overlap_cache(fixture.eigensolver_plan, fixture.overlap_cache_storage->data,
                                     fixture.overlap_cache_storage->size, fixture.overlap_cache,
                                     error) != XTBLOOM_STATUS_SUCCESS ||
      bind_eigensolver_workspace(fixture.eigensolver_plan,
                                 fixture.eigensolver_scratch_storage->data,
                                 fixture.eigensolver_scratch_storage->size,
                                 fixture.eigensolver_scratch, error) != XTBLOOM_STATUS_SUCCESS ||
      factor_overlap_cpu(fixture.eigensolver_plan, fixture.overlap.data(), 1u, backend(),
                         fixture.eigensolver_scratch, fixture.overlap_cache,
                         error) != XTBLOOM_STATUS_SUCCESS ||
      bind_scc_mixer_state(fixture.mixer_plan, fixture.mixer_state_storage->data,
                           fixture.mixer_state_storage->size, fixture.mixer_state,
                           error) != XTBLOOM_STATUS_SUCCESS ||
      bind_scc_driver_state(fixture.driver_plan, fixture.driver_state_storage->data,
                            fixture.driver_state_storage->size, fixture.driver_state,
                            error) != XTBLOOM_STATUS_SUCCESS ||
      bind_scc_driver_workspace(fixture.driver_plan, fixture.driver_scratch_storage->data,
                                fixture.driver_scratch_storage->size, fixture.driver_scratch,
                                error) != XTBLOOM_STATUS_SUCCESS ||
      initialize_scc_driver_state_cpu(fixture.driver_plan, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }

  fixture.geometry.h0 = fixture.h0.data();
  fixture.geometry.h0_elements = fixture.integral_plan.total_matrix_elements;
  fixture.geometry.integrals = {
      fixture.overlap.data(), fixture.dipole_integrals.data(), fixture.quadrupole_integrals.data(),
      fixture.integral_plan.total_matrix_elements, fixture.mulliken_plan.identity()};
  fixture.geometry.es2_cache = fixture.es2_cache;
  fixture.geometry.aes2_cache = fixture.aes2_cache;
  fixture.geometry.geometry_generation = 1u;
  if (enable_d4) {
    const std::size_t pair_elements =
        static_cast<std::size_t>(fixture.d4_plan.total_pairs()) * kD4PairDataElements;
    fixture.d4_pair_data.assign(std::max<std::size_t>(pair_elements, 1u), 0.0);
    fixture.d4_coordination.assign(static_cast<std::size_t>(total_atoms), 0.0);
    if (update_d4_geometry_cache_cpu(fixture.d4_plan, fixture.positions.data(),
                                     fixture.geometry.geometry_generation,
                                     fixture.d4_pair_data.data(), fixture.d4_pair_data.size(),
                                     fixture.d4_coordination.data(), fixture.d4_coordination.size(),
                                     fixture.driver_scratch.d4_workspace, fixture.d4_cache,
                                     error) != XTBLOOM_STATUS_SUCCESS) {
      return false;
    }
    fixture.geometry.d4_cache = fixture.d4_cache;
  }
  if (enable_periodic_embedding) {
    fixture.geometry.periodic_shifts = fixture.periodic_shifts.data();
    fixture.geometry.periodic_shift_elements = total_atoms;
    fixture.geometry.periodic_response_matrices = fixture.periodic_response.data();
    fixture.geometry.periodic_response_elements = fixture.periodic_plan.total_matrix_elements();
    fixture.geometry.periodic_embedding_generation = 1u;
    fixture.geometry.periodic_plan_identity = fixture.periodic_plan.identity();
  }
  return true;
}

int test_complete_energy_components_free_energy_and_restart() {
  const FixtureTopology topology{{0, 2}, {1, 1}, {-0.7, 0.0, 0.0, 0.7, 0.0, 0.0}, {0.0}, {}, {}};
  Fixture fixture;
  std::string error;
  error.reserve(256u);
  constexpr double temperature = 0.5;
  CHECK(make_fixture(1, fixture, error, 1u, 1.0e100, true, &topology, false, true, 1.0e100,
                     temperature));
  CHECK(fixture.driver_plan.energy_tolerance() == 1.0e100);

  fixture.periodic_shifts = {0.03, -0.02};
  fixture.periodic_response = {0.04, 0.01, 0.01, -0.03};
  fixture.geometry.periodic_shifts = fixture.periodic_shifts.data();
  fixture.geometry.periodic_response_matrices = fixture.periodic_response.data();
  std::vector<double> explicit_pc(
      static_cast<std::size_t>(fixture.wavefunction_layout.total_shells));
  for (std::size_t shell = 0; shell < explicit_pc.size(); ++shell) {
    explicit_pc[shell] = 0.015 * static_cast<double>(shell + 1u);
  }
  fixture.geometry.explicit_point_charge_shell_potential = explicit_pc.data();
  fixture.geometry.explicit_point_charge_shell_elements =
      static_cast<std::int64_t>(explicit_pc.size());

  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.converged[0] == 1u);

  const std::array<double, 8> components{{
      fixture.driver_state.core_energies[0],
      fixture.driver_state.es2_energies[0],
      fixture.driver_state.es3_energies[0],
      fixture.driver_state.aes2_energies[0],
      fixture.driver_state.spin_energies[0],
      fixture.driver_state.d4_two_body_energies[0],
      fixture.driver_state.explicit_point_charge_energies[0],
      fixture.driver_state.periodic_embedding_energies[0],
  }};
  double component_sum = 0.0;
  for (const double component : components) {
    CHECK(std::isfinite(component));
    component_sum += component;
  }
  const double scale = std::max(1.0, std::abs(component_sum));
  CHECK(std::abs(fixture.driver_state.internal_energies[0] - component_sum) < 1.0e-13 * scale);

  /* The core trace is Tr(P H0), including both off-diagonal density entries;
   * a diagonal-only contraction would silently lose covalent contributions. */
  const std::int64_t orbitals = fixture.wavefunction_layout.total_orbitals;
  double expected_core = 0.0;
  bool has_off_diagonal_density = false;
  for (std::int64_t row = 0; row < orbitals; ++row) {
    for (std::int64_t column = 0; column < orbitals; ++column) {
      const std::size_t matrix = static_cast<std::size_t>(row * orbitals + column);
      expected_core =
          std::fma(fixture.h0[matrix], fixture.wavefunction.density[matrix], expected_core);
      if (row != column && std::abs(fixture.wavefunction.density[matrix]) > 1.0e-8) {
        has_off_diagonal_density = true;
      }
    }
  }
  CHECK(has_off_diagonal_density);
  CHECK(std::abs(fixture.driver_state.core_energies[0] - expected_core) <
        1.0e-13 * std::max(1.0, std::abs(expected_core)));

  const std::int64_t shell_begin = fixture.wavefunction_layout.batch_shell_offsets[0];
  const std::int64_t shell_end = fixture.wavefunction_layout.batch_shell_offsets[1];
  const std::int64_t qsh_base = fixture.wavefunction_layout.qsh.system_offsets[0];
  double expected_pc = 0.0;
  for (std::int64_t local_shell = 0; local_shell < shell_end - shell_begin; ++local_shell) {
    expected_pc =
        std::fma(fixture.driver_scratch.raw_qsh[static_cast<std::size_t>(qsh_base + local_shell)],
                 explicit_pc[static_cast<std::size_t>(shell_begin + local_shell)], expected_pc);
  }
  CHECK(std::abs(fixture.driver_state.explicit_point_charge_energies[0] - expected_pc) < 1.0e-14);

  CHECK(fixture.driver_state.entropies[0] > 0.0);
  const double expected_free =
      fixture.driver_state.internal_energies[0] - temperature * fixture.driver_state.entropies[0];
  CHECK(std::abs(fixture.driver_state.free_energies[0] - expected_free) <
        1.0e-13 * std::max(1.0, std::abs(expected_free)));
  CHECK(std::abs(fixture.driver_state.free_energies[0] - fixture.driver_state.band_energies[0]) >
        1.0e-8);

  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 0, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == XTBLOOM_STATUS_SUCCESS);
  for (const double* trace :
       {fixture.driver_state.free_energies, fixture.driver_state.previous_free_energies,
        fixture.driver_state.free_energy_changes, fixture.driver_state.entropies,
        fixture.driver_state.band_energies, fixture.driver_state.core_energies,
        fixture.driver_state.es2_energies, fixture.driver_state.es3_energies,
        fixture.driver_state.aes2_energies, fixture.driver_state.spin_energies,
        fixture.driver_state.d4_two_body_energies,
        fixture.driver_state.explicit_point_charge_energies,
        fixture.driver_state.periodic_embedding_energies, fixture.driver_state.internal_energies}) {
    CHECK(std::isnan(trace[0]));
  }
  CHECK(fixture.driver_state.iterations[0] == 0u);
  CHECK(fixture.driver_state.converged[0] == 0u);
  return 0;
}

int test_energy_and_residual_convergence_gates_are_independent_and_strict() {
  std::string error;
  const auto prepare_nonstationary_h_plus = [&](Fixture& fixture) {
    fixture.wavefunction.qsh[0] = 0.0;
    fixture.wavefunction.qat[0] = 0.0;
    return restart_scc_driver_system_cpu(fixture.driver_plan, 0, fixture.wavefunction,
                                         fixture.mixer_state, fixture.driver_state,
                                         error) == XTBLOOM_STATUS_SUCCESS;
  };

  Fixture energy_only;
  CHECK(make_fixture(1, energy_only, error, 1u, 0.1, false, nullptr, false, false, 1.0e100));
  CHECK(prepare_nonstationary_h_plus(energy_only));
  CHECK(iterate_scc_driver_batch_cpu(
            energy_only.driver_plan, energy_only.geometry, backend(), energy_only.overlap_cache,
            energy_only.wavefunction, energy_only.mixer_state, energy_only.driver_state,
            energy_only.driver_scratch, error) == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(std::abs(energy_only.driver_state.free_energy_changes[0]) <
        energy_only.driver_plan.energy_tolerance());
  CHECK(energy_only.mixer_state.residual_rms[0] >= energy_only.mixer_plan.rms_tolerance());

  Fixture residual_only;
  CHECK(make_fixture(1, residual_only, error, 1u, 2.0, false, nullptr, false, false, 1.0e-300));
  CHECK(prepare_nonstationary_h_plus(residual_only));
  CHECK(iterate_scc_driver_batch_cpu(residual_only.driver_plan, residual_only.geometry, backend(),
                                     residual_only.overlap_cache, residual_only.wavefunction,
                                     residual_only.mixer_state, residual_only.driver_state,
                                     residual_only.driver_scratch,
                                     error) == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(residual_only.mixer_state.residual_rms[0] < residual_only.mixer_plan.rms_tolerance());
  CHECK(std::abs(residual_only.driver_state.free_energy_changes[0]) >=
        residual_only.driver_plan.energy_tolerance());

  Fixture both;
  CHECK(make_fixture(1, both, error, 1u, 2.0, false, nullptr, false, false, 1.0e100));
  CHECK(prepare_nonstationary_h_plus(both));
  CHECK(iterate_scc_driver_batch_cpu(both.driver_plan, both.geometry, backend(), both.overlap_cache,
                                     both.wavefunction, both.mixer_state, both.driver_state,
                                     both.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(both.driver_state.converged[0] == 1u);
  const double boundary = std::abs(both.driver_state.free_energy_changes[0]);
  CHECK(boundary > 0.0);

  Fixture strict_boundary;
  CHECK(make_fixture(1, strict_boundary, error, 1u, 2.0, false, nullptr, false, false, boundary));
  CHECK(prepare_nonstationary_h_plus(strict_boundary));
  CHECK(iterate_scc_driver_batch_cpu(strict_boundary.driver_plan, strict_boundary.geometry,
                                     backend(), strict_boundary.overlap_cache,
                                     strict_boundary.wavefunction, strict_boundary.mixer_state,
                                     strict_boundary.driver_state, strict_boundary.driver_scratch,
                                     error) == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(std::abs(strict_boundary.driver_state.free_energy_changes[0]) == boundary);
  CHECK(strict_boundary.mixer_state.residual_rms[0] < strict_boundary.mixer_plan.rms_tolerance());
  CHECK(strict_boundary.driver_state.converged[0] == 0u);
  return 0;
}

int test_complete_energy_failure_isolated_from_ragged_peer() {
  const FixtureTopology topology{{0, 1, 2},  {1, 2}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                 {1.0, 2.0}, {},     {}};
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(2, fixture, error, 1u, 1.0e100, false, &topology));
  std::vector<double> explicit_pc(
      static_cast<std::size_t>(fixture.wavefunction_layout.total_shells), 0.0);
  const std::int64_t failed_shell_begin = fixture.wavefunction_layout.batch_shell_offsets[1];
  const std::int64_t failed_shell_end = fixture.wavefunction_layout.batch_shell_offsets[2];
  for (std::int64_t shell = failed_shell_begin; shell < failed_shell_end; ++shell) {
    /* The Hamiltonian shift remains finite, while the He++ raw shell charge
     * doubles this value during the post-eigensolve explicit-PC contraction. */
    explicit_pc[static_cast<std::size_t>(shell)] = 0.75 * std::numeric_limits<double>::max();
  }
  fixture.geometry.explicit_point_charge_shell_potential = explicit_pc.data();
  fixture.geometry.explicit_point_charge_shell_elements =
      static_cast<std::int64_t>(explicit_pc.size());
  const std::int64_t failed_qsh_base = fixture.wavefunction_layout.qsh.system_offsets[1];
  const double failed_qsh_before =
      fixture.wavefunction.qsh[static_cast<std::size_t>(failed_qsh_base)];

  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
            fixture.wavefunction, fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.iterations[0] == 1u);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(std::isfinite(fixture.driver_state.internal_energies[0]));
  CHECK(fixture.driver_state.system_statuses[1] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.driver_state.iterations[1] == 1u);
  CHECK(std::isnan(fixture.driver_state.internal_energies[1]));
  CHECK(std::isnan(fixture.driver_state.free_energies[1]));
  CHECK(fixture.wavefunction.qsh[static_cast<std::size_t>(failed_qsh_base)] == failed_qsh_before);
  return 0;
}

int test_preparation_numerical_failure_isolated_from_ragged_peer() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(2, fixture, error));
  std::vector<double> explicit_pc(
      static_cast<std::size_t>(fixture.wavefunction_layout.total_shells), 0.0);
  const std::int64_t failed_shell = fixture.wavefunction_layout.batch_shell_offsets[1];
  explicit_pc[static_cast<std::size_t>(failed_shell)] = std::numeric_limits<double>::quiet_NaN();
  fixture.geometry.explicit_point_charge_shell_potential = explicit_pc.data();
  fixture.geometry.explicit_point_charge_shell_elements =
      static_cast<std::int64_t>(explicit_pc.size());
  const double failed_qsh_before =
      fixture.wavefunction
          .qsh[static_cast<std::size_t>(fixture.wavefunction_layout.qsh.system_offsets[1])];
  diagonalizations.store(0, std::memory_order_relaxed);

  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
            fixture.wavefunction, fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == 1);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.iterations[0] == 1u);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(fixture.driver_state.system_statuses[1] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.driver_state.iterations[1] == 0u);
  CHECK(fixture.wavefunction
            .qsh[static_cast<std::size_t>(fixture.wavefunction_layout.qsh.system_offsets[1])] ==
        failed_qsh_before);

  /* H0 is likewise target numerical data after the complete geometry view has
   * passed structural validation, so poisoning one system must not roll back
   * the healthy peer. */
  Fixture h0_failure;
  CHECK(make_fixture(2, h0_failure, error));
  const std::int64_t failed_matrix = h0_failure.mulliken_plan.matrix_offsets()[1];
  h0_failure.h0[static_cast<std::size_t>(failed_matrix)] = std::numeric_limits<double>::quiet_NaN();
  diagonalizations.store(0, std::memory_order_relaxed);
  CHECK(iterate_scc_driver_batch_cpu(
            h0_failure.driver_plan, h0_failure.geometry, backend(), h0_failure.overlap_cache,
            h0_failure.wavefunction, h0_failure.mixer_state, h0_failure.driver_state,
            h0_failure.driver_scratch, error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == 1);
  CHECK(h0_failure.driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(h0_failure.driver_state.iterations[0] == 1u);
  CHECK(h0_failure.driver_state.system_statuses[1] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(h0_failure.driver_state.iterations[1] == 0u);
  return 0;
}

int test_ragged_failure_isolation_restart_and_skip() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(2, fixture, error));
  fixture.overlap_cache.system_statuses[1] = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
  diagonalizations.store(0, std::memory_order_relaxed);
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
            fixture.wavefunction, fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == 1);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(fixture.driver_state.iterations[0] == 1u);
  CHECK(fixture.driver_state.system_statuses[1] == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  CHECK(fixture.driver_state.iterations[1] == 1u);
  CHECK(std::isnan(fixture.driver_state.free_energies[1]));

  const double first_qsh = fixture.wavefunction.qsh[0];
  const int calls_before_skip = diagonalizations.load(std::memory_order_relaxed);
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == calls_before_skip);
  CHECK(fixture.wavefunction.qsh[0] == first_qsh);

  fixture.overlap_cache.system_statuses[1] = XTBLOOM_STATUS_SUCCESS;
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 1, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == calls_before_skip + 1);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(fixture.driver_state.converged[1] == 1u);
  CHECK(fixture.driver_state.iterations[1] == 1u);
  CHECK(std::abs(fixture.wavefunction.qsh[1] - 1.0) < 1.0e-14);
  CHECK(std::abs(fixture.wavefunction.qat[1] - 1.0) < 1.0e-14);
  return 0;
}

int test_converged_system_skips_classical_and_mulliken_arithmetic() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(2, fixture, error));
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(fixture.driver_state.converged[1] == 1u);

  /* Restart system 1 so it becomes active again while system 0 remains
   * converged, then poison the converged system's public multipoles. The
   * active-only driver must never read the converged peer's data: a classic
   * whole-batch gather would reject the NaN before any system ran. */
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 1, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == XTBLOOM_STATUS_SUCCESS);
  const std::int64_t qsh_begin = fixture.wavefunction_layout.qsh.system_offsets[0];
  const std::int64_t qsh_end = fixture.wavefunction_layout.qsh.system_offsets[1];
  const std::int64_t qat_begin = fixture.wavefunction_layout.qat.system_offsets[0];
  const std::int64_t qat_end = fixture.wavefunction_layout.qat.system_offsets[1];
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::int64_t element = qsh_begin; element < qsh_end; ++element) {
    fixture.wavefunction.qsh[static_cast<std::size_t>(element)] = nan;
  }
  for (std::int64_t element = qat_begin; element < qat_end; ++element) {
    fixture.wavefunction.qat[static_cast<std::size_t>(element)] = nan;
  }

  const int diagonalizations_before = diagonalizations.load(std::memory_order_relaxed);
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == diagonalizations_before + 1);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(fixture.driver_state.converged[1] == 1u);
  CHECK(std::isnan(fixture.wavefunction.qsh[static_cast<std::size_t>(qsh_begin)]));
  CHECK(std::isnan(fixture.wavefunction.qat[static_cast<std::size_t>(qat_begin)]));
  return 0;
}

int test_structural_failure_atomicity_and_zero_allocation() {
  Fixture fixture;
  std::string error;
  error.reserve(256u);
  CHECK(make_fixture(1, fixture, error));
  const double before_qsh = fixture.wavefunction.qsh[0];
  const xtbloom_status_t before_status = fixture.driver_state.system_statuses[0];
  SccDriverGeometryView malformed = fixture.geometry;
  malformed.h0_elements = 0;
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, malformed, backend(), fixture.overlap_cache, fixture.wavefunction,
            fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.wavefunction.qsh[0] == before_qsh);
  CHECK(fixture.driver_state.system_statuses[0] == before_status);
  CHECK(fixture.driver_state.iterations[0] == 0u);

  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t status = iterate_scc_driver_batch_cpu(
      fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache, fixture.wavefunction,
      fixture.mixer_state, fixture.driver_state, fixture.driver_scratch, error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  CHECK(fixture.driver_state.converged[0] == 1u);

  /* A correctly bound driver descriptor may still overlap a separately bound
   * mixer descriptor. Restart must reject that relationship before either
   * workspace or the wavefunction is modified. */
  SccDriverState overlapping_state;
  CHECK(bind_scc_driver_state(fixture.driver_plan, fixture.mixer_state_storage->data,
                              fixture.mixer_state_storage->size, overlapping_state,
                              error) == XTBLOOM_STATUS_SUCCESS);
  std::vector<unsigned char> before_overlap(fixture.mixer_state_storage->size);
  std::memcpy(before_overlap.data(), fixture.mixer_state_storage->data,
              fixture.mixer_state_storage->size);
  const double qsh_before_restart = fixture.wavefunction.qsh[0];
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 0, fixture.wavefunction,
                                      fixture.mixer_state, overlapping_state,
                                      error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::memcmp(before_overlap.data(), fixture.mixer_state_storage->data,
                    fixture.mixer_state_storage->size) == 0);
  CHECK(fixture.wavefunction.qsh[0] == qsh_before_restart);
  return 0;
}

int test_mixed_restricted_unrestricted_hydrogen_batch() {
  const FixtureTopology topology{{0, 1, 2},  {1, 1}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                 {1.0, 0.0}, {0, 1}, {1, 2}};
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(2, fixture, error, 1u, 1.0e100, false, &topology, false, false, 1.0e100));

  const std::int64_t open_shell_qsh = fixture.wavefunction_layout.qsh.system_offsets[1];
  CHECK(fixture.wavefunction_layout.qsh.system_offsets[2] - open_shell_qsh == 2);
  CHECK(fixture.wavefunction.qsh[open_shell_qsh] == 0.0);
  fixture.wavefunction.qsh[open_shell_qsh + 1] = -1.0;
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 1, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == XTBLOOM_STATUS_SUCCESS);

  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.system_statuses[1] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.spin_energies[0] == 0.0);
  CHECK(std::abs(fixture.driver_state.spin_energies[1] + 0.0358125) < 2.0e-14);

  const std::int64_t h0_index = fixture.mulliken_plan.matrix_offsets()[1];
  const std::int64_t hamiltonian_base = fixture.wavefunction_layout.density.system_offsets[1];
  const double h0 = fixture.h0[static_cast<std::size_t>(h0_index)];
  const double overlap = fixture.overlap[static_cast<std::size_t>(h0_index)];
  CHECK(std::abs(fixture.driver_scratch.hamiltonian[hamiltonian_base] - (h0 - overlap * 0.071625)) <
        2.0e-15);
  CHECK(std::abs(fixture.driver_scratch.hamiltonian[hamiltonian_base + 1] -
                 (h0 + overlap * 0.071625)) < 2.0e-15);
  CHECK(std::abs(fixture.wavefunction.qsh[open_shell_qsh]) < 2.0e-13);
  CHECK(std::abs(fixture.wavefunction.qsh[open_shell_qsh + 1] + 1.0) < 2.0e-13);
  return 0;
}

int test_max_iteration_status_counts_attempts() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(1, fixture, error, 1u));
  fixture.wavefunction.qsh[0] = 0.0;
  fixture.wavefunction.qat[0] = 0.0;
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 0, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
            fixture.wavefunction, fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  CHECK(fixture.driver_state.iterations[0] == 1u);
  CHECK(fixture.mixer_state.iterations[0] == 1u);
  CHECK(fixture.driver_state.converged[0] == 0u);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_SCC_NOT_CONVERGED);
  return 0;
}

int test_mixer_failure_preserves_public_history_and_counts_attempt() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(1, fixture, error));
  const double qsh_before = fixture.wavefunction.qsh[0];
  const double current_input_before = fixture.mixer_state.current_inputs[0];
  const double previous_input_before = fixture.mixer_state.previous_inputs[0];
  fixture.mixer_state.iterations[0] = std::numeric_limits<std::uint64_t>::max();
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
            fixture.wavefunction, fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.driver_state.iterations[0] == 1u);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.mixer_state.iterations[0] == std::numeric_limits<std::uint64_t>::max());
  CHECK(fixture.mixer_state.current_inputs[0] == current_input_before);
  CHECK(fixture.mixer_state.previous_inputs[0] == previous_input_before);
  CHECK(fixture.wavefunction.qsh[0] == qsh_before);
  CHECK(std::isnan(fixture.driver_state.free_energies[0]));
  return 0;
}

int test_ragged_mixer_failure_isolated_from_peer_commit() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(2, fixture, error));
  /* A converged-peer reference: system 0 converges on the first attempt and
   * then stays inactive, so it must never be copied by a later transaction
   * that only touches the restarted system 1. */
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(fixture.driver_state.converged[1] == 1u);

  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 1, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == XTBLOOM_STATUS_SUCCESS);
  const MixerSystemSnapshot peer_before = snapshot_mixer_system(fixture, 0u);
  const double peer_qsh_before = fixture.wavefunction.qsh[0];
  CHECK(fixture.driver_state.iterations[0] == 1u);
  const std::uint64_t peer_driver_iterations_before = fixture.driver_state.iterations[0];

  /* Force system 1's mixer transition to fail while system 0 remains a
   * converged, inactive peer. */
  fixture.mixer_state.iterations[1] = std::numeric_limits<std::uint64_t>::max();
  const MixerSystemSnapshot failed_before = snapshot_mixer_system(fixture, 1u);
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
            fixture.wavefunction, fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(fixture.driver_state.system_statuses[1] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.driver_state.iterations[1] == 1u);
  CHECK(fixture.driver_state.iterations[0] == peer_driver_iterations_before);
  /* The failed system's own public history is byte-identical except for its
   * published status, and the inactive peer's history is byte-identical
   * including its status. */
  const MixerSystemSnapshot failed_after = snapshot_mixer_system(fixture, 1u);
  CHECK(failed_after.iterations == failed_before.iterations);
  CHECK(failed_after.restart_counts == failed_before.restart_counts);
  CHECK(failed_after.current_inputs == failed_before.current_inputs);
  CHECK(failed_after.previous_inputs == failed_before.previous_inputs);
  CHECK(failed_after.previous_residuals == failed_before.previous_residuals);
  CHECK(failed_after.df_history == failed_before.df_history);
  CHECK(failed_after.u_history == failed_before.u_history);
  CHECK(failed_after.omega == failed_before.omega);
  CHECK(failed_after.residual_rms == failed_before.residual_rms);
  CHECK(failed_after.residual_maximum == failed_before.residual_maximum);
  CHECK(failed_after.initialized == failed_before.initialized);
  CHECK(failed_after.converged == failed_before.converged);
  CHECK(failed_after.system_status == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(mixer_snapshots_equal(peer_before, snapshot_mixer_system(fixture, 0u)));
  CHECK(fixture.wavefunction.qsh[0] == peer_qsh_before);

  /* Restart the failed system from its unchanged public state and let the
   * active-only path advance it while the converged peer is skipped. */
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 1, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.iterations[1] == 1u);
  CHECK(fixture.driver_state.converged[1] == 1u);
  CHECK(mixer_snapshots_equal(peer_before, snapshot_mixer_system(fixture, 0u)));
  return 0;
}

int test_inactive_peer_history_untouched_by_active_only_commit() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(3, fixture, error, 5u));
  /* Run to full convergence so systems 0 and 1 are inactive peers. */
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(fixture.driver_state.converged[1] == 1u);
  CHECK(fixture.driver_state.converged[2] == 1u);

  /* System 2 becomes the only active member while systems 0 and 1 stay
   * converged. Its commit must copy only its own history: the inactive peers'
   * mixer histories (including their statuses) must remain byte-identical. */
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 2, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == XTBLOOM_STATUS_SUCCESS);
  const MixerSystemSnapshot first = snapshot_mixer_system(fixture, 0u);
  const MixerSystemSnapshot second = snapshot_mixer_system(fixture, 1u);
  const double active_qsh_before = fixture.wavefunction.qsh[0];
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(mixer_snapshots_equal(first, snapshot_mixer_system(fixture, 0u)));
  CHECK(mixer_snapshots_equal(second, snapshot_mixer_system(fixture, 1u)));
  CHECK(fixture.driver_state.converged[2] == 1u);
  CHECK(fixture.mixer_state.iterations[2] == 1u);
  CHECK(fixture.wavefunction.qsh[0] == active_qsh_before);
  return 0;
}

int test_energy_history_failure_isolated_from_peer_commit() {
  /* Two H2 systems with 2 electrons: with a strict complete free-energy
   * tolerance every member stays active after the first iteration (the
   * first energy change |E1 - 0| is far above 1e-14), so old_iteration is 1
   * on the next call. */
  const FixtureTopology topology{
      {0, 2, 4},  {1, 1, 1, 1}, {0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0},
      {0.0, 0.0}, {},           {}};
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(2, fixture, error, 5u, 1.0e-10, false, &topology, false, false, 1.0e-14));
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.iterations[0] == 1u);
  CHECK(fixture.driver_state.iterations[1] == 1u);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.system_statuses[1] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.converged[0] == 0u);
  CHECK(fixture.driver_state.converged[1] == 0u);
  CHECK(std::isfinite(fixture.driver_state.free_energies[0]));
  CHECK(std::isfinite(fixture.driver_state.free_energies[1]));

  /* Corrupt system 0's committed energy history directly: its next transition
   * is a per-system data-level failure that must discard its staged mixer
   * transaction while system 1 still commits. */
  fixture.driver_state.free_energies[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(fixture.mixer_state.iterations[1] == 1u);
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
            fixture.wavefunction, fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(std::isnan(fixture.driver_state.free_energies[0]));
  /* The failed system's public mixer history is not advanced: the attempt was
   * counted only in the driver trace, mirroring the mix-failure contract. */
  CHECK(fixture.mixer_state.iterations[0] == 1u);
  CHECK(fixture.driver_state.iterations[0] == 2u);
  /* The peer committed normally. */
  CHECK(fixture.driver_state.system_statuses[1] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.iterations[1] == 2u);
  CHECK(fixture.mixer_state.iterations[1] == 2u);
  CHECK(std::isfinite(fixture.driver_state.free_energies[1]));
  return 0;
}

int test_converged_wavefunction_publishes_raw_mulliken_multipoles() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(1, fixture, error, 5u, 2.0));
  fixture.wavefunction.qsh[0] = 0.0;
  fixture.wavefunction.qat[0] = 0.0;
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 0, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(std::abs(fixture.mixer_state.current_inputs[0] - 0.4) < 1.0e-14);
  CHECK(std::abs(fixture.wavefunction.qsh[0] - 1.0) < 1.0e-14);
  CHECK(std::abs(fixture.wavefunction.qat[0] - 1.0) < 1.0e-14);
  return 0;
}

int test_cached_explicit_point_charge_potential_enters_hamiltonian() {
  Fixture reference;
  Fixture embedded;
  std::string error;
  CHECK(make_fixture(1, reference, error));
  CHECK(make_fixture(1, embedded, error));
  const std::array<double, 1> point_charge_shell_potential{{0.25}};
  embedded.geometry.explicit_point_charge_shell_potential = point_charge_shell_potential.data();
  embedded.geometry.explicit_point_charge_shell_elements = 1;

  CHECK(iterate_scc_driver_batch_cpu(reference.driver_plan, reference.geometry, backend(),
                                     reference.overlap_cache, reference.wavefunction,
                                     reference.mixer_state, reference.driver_state,
                                     reference.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(iterate_scc_driver_batch_cpu(embedded.driver_plan, embedded.geometry, backend(),
                                     embedded.overlap_cache, embedded.wavefunction,
                                     embedded.mixer_state, embedded.driver_state,
                                     embedded.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::abs((embedded.wavefunction.eigenvalues[0] - reference.wavefunction.eigenvalues[0]) +
                 point_charge_shell_potential[0]) < 1.0e-14);
  return 0;
}

int test_disabled_layout_and_bitwise_compatibility() {
  Fixture legacy_overload;
  Fixture nullable_overload;
  Fixture enabled;
  std::string error;
  CHECK(make_fixture(1, legacy_overload, error, 5u, 1.0e-10, false, nullptr, false, false,
                     kDefaultSccEnergyTolerance, 0.0, true));
  CHECK(make_fixture(1, nullable_overload, error, 5u, 1.0e-10, false, nullptr, true, false,
                     kDefaultSccEnergyTolerance, 0.0, true));
  CHECK(make_fixture(1, enabled, error, 5u, 1.0e-10, true));

  CHECK(legacy_overload.driver_plan.state_size_bytes() ==
        expected_disabled_state_size(legacy_overload));
  CHECK(legacy_overload.driver_plan.workspace_size_bytes() ==
        expected_disabled_workspace_size(legacy_overload));
  CHECK(nullable_overload.driver_plan.state_size_bytes() ==
        legacy_overload.driver_plan.state_size_bytes());
  CHECK(nullable_overload.driver_plan.workspace_size_bytes() ==
        legacy_overload.driver_plan.workspace_size_bytes());
  CHECK(enabled.driver_plan.state_size_bytes() >= legacy_overload.driver_plan.state_size_bytes());
  CHECK(enabled.driver_plan.workspace_size_bytes() >=
        legacy_overload.driver_plan.workspace_size_bytes());
  CHECK(enabled.driver_state.periodic_embedding_energies != nullptr);
  CHECK(enabled.driver_scratch.periodic_atomic_potentials != nullptr);
  CHECK(enabled.driver_scratch.periodic_embedding_energies != nullptr);
  CHECK(enabled.driver_scratch.periodic_system_statuses != nullptr);
  CHECK(enabled.driver_scratch.periodic_embedding_workspace.potential_scratch != nullptr);

  for (Fixture* fixture : {&legacy_overload, &nullable_overload}) {
    CHECK(!fixture->driver_plan.d4_enabled());
    CHECK(fixture->driver_state.d4_two_body_energies == nullptr);
    CHECK(fixture->driver_scratch.d4_atomic_potentials == nullptr);
    CHECK(fixture->driver_scratch.d4_two_body_energies == nullptr);
    CHECK(fixture->driver_scratch.d4_workspace.workspace_base == nullptr);
    CHECK(fixture->driver_state.periodic_embedding_energies == nullptr);
    CHECK(fixture->driver_scratch.periodic_atomic_potentials == nullptr);
    CHECK(fixture->driver_scratch.periodic_embedding_energies == nullptr);
    CHECK(fixture->driver_scratch.periodic_system_statuses == nullptr);
    CHECK(fixture->driver_scratch.periodic_embedding_workspace.potential_scratch == nullptr);
    CHECK(fixture->driver_scratch.periodic_embedding_workspace.potential_elements == 0);
    CHECK(fixture->driver_scratch.periodic_embedding_workspace.plan_identity == nullptr);
  }

  const std::array<double, 1> point_charge_shell_potential{{0.375}};
  legacy_overload.geometry.explicit_point_charge_shell_potential =
      point_charge_shell_potential.data();
  legacy_overload.geometry.explicit_point_charge_shell_elements = 1;
  nullable_overload.geometry.explicit_point_charge_shell_potential =
      point_charge_shell_potential.data();
  nullable_overload.geometry.explicit_point_charge_shell_elements = 1;
  CHECK(iterate_scc_driver_batch_cpu(legacy_overload.driver_plan, legacy_overload.geometry,
                                     backend(), legacy_overload.overlap_cache,
                                     legacy_overload.wavefunction, legacy_overload.mixer_state,
                                     legacy_overload.driver_state, legacy_overload.driver_scratch,
                                     error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(iterate_scc_driver_batch_cpu(
            nullable_overload.driver_plan, nullable_overload.geometry, backend(),
            nullable_overload.overlap_cache, nullable_overload.wavefunction,
            nullable_overload.mixer_state, nullable_overload.driver_state,
            nullable_overload.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::memcmp(legacy_overload.wavefunction_storage->data,
                    nullable_overload.wavefunction_storage->data,
                    legacy_overload.wavefunction_layout.workspace_size_bytes) == 0);
  CHECK(std::memcmp(legacy_overload.mixer_state_storage->data,
                    nullable_overload.mixer_state_storage->data,
                    legacy_overload.mixer_plan.state_size_bytes()) == 0);
  CHECK(std::memcmp(legacy_overload.driver_state_storage->data,
                    nullable_overload.driver_state_storage->data,
                    legacy_overload.driver_plan.state_size_bytes()) == 0);
  /* These hexadecimal values were frozen from the pre-periodic SCC driver at
   * ed7e355. They independently guard the legacy ES2 -> ES3 -> explicit-PC
   * addition order instead of only comparing two overloads of today's code. */
  CHECK(legacy_overload.driver_scratch.shell_potentials[0] == 0x1.b8b6f9fcb0c02p-1);
  CHECK(legacy_overload.wavefunction.eigenvalues[0] == -0x1.4116c650fd6a8p+0);
  return 0;
}

int test_optional_d4_potential_energy_restart_and_zero_allocation() {
  const FixtureTopology topology{{0, 2}, {6, 8}, {-1.1, 0.0, 0.0, 1.1, 0.0, 0.0}, {0.0}, {}, {}};
  Fixture reference;
  Fixture enabled;
  Fixture combined;
  std::string error;
  error.reserve(256u);
  CHECK(make_fixture(1, reference, error, 5u, 1.0e-10, false, &topology));
  CHECK(make_fixture(1, enabled, error, 5u, 1.0e-10, false, &topology, false, true));
  CHECK(make_fixture(1, combined, error, 5u, 1.0e-10, true, &topology, false, true));
  CHECK(!reference.driver_plan.d4_enabled());
  CHECK(enabled.driver_plan.d4_enabled());
  CHECK(combined.driver_plan.d4_enabled());
  CHECK(combined.driver_plan.periodic_embedding_enabled());
  CHECK(combined.driver_state.d4_two_body_energies != nullptr);
  CHECK(combined.driver_state.periodic_embedding_energies != nullptr);
  CHECK(enabled.driver_state.d4_two_body_energies != nullptr);
  CHECK(enabled.driver_scratch.d4_atomic_potentials != nullptr);
  CHECK(enabled.driver_scratch.d4_two_body_energies != nullptr);
  CHECK(enabled.driver_scratch.d4_workspace.plan_identity == enabled.d4_plan.identity());

  /* D4 must consume the charge channel reconstructed from the current mixed
   * shell charges. Public qat is a derived field and can legitimately be
   * stale between caller-side edits and the next driver iteration. */
  std::array<double, 2> mixed_atomic_charges{};
  const std::int64_t shell_begin = enabled.wavefunction_layout.batch_shell_offsets[0];
  const std::int64_t shell_end = enabled.wavefunction_layout.batch_shell_offsets[1];
  const std::int64_t qsh_base = enabled.wavefunction_layout.qsh.system_offsets[0];
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
    const std::int64_t atom =
        enabled.mulliken_plan.shell_to_atom()[static_cast<std::size_t>(shell)];
    mixed_atomic_charges[static_cast<std::size_t>(atom)] +=
        enabled.wavefunction.qsh[static_cast<std::size_t>(qsh_base + shell - shell_begin)];
  }
  enabled.wavefunction.qat[0] = 91.0;
  enabled.wavefunction.qat[1] = -73.0;

  std::array<double, 1> expected_energy{};
  std::array<double, 2> expected_potential{};
  CHECK(evaluate_d4_two_body_cpu(enabled.d4_plan, enabled.d4_cache, mixed_atomic_charges.data(),
                                 expected_energy.data(), expected_potential.data(),
                                 enabled.driver_scratch.d4_workspace,
                                 error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::abs(expected_potential[0]) + std::abs(expected_potential[1]) > 1.0e-12);

  CHECK(iterate_scc_driver_batch_cpu(reference.driver_plan, reference.geometry, backend(),
                                     reference.overlap_cache, reference.wavefunction,
                                     reference.mixer_state, reference.driver_state,
                                     reference.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t enabled_status = iterate_scc_driver_batch_cpu(
      enabled.driver_plan, enabled.geometry, backend(), enabled.overlap_cache, enabled.wavefunction,
      enabled.mixer_state, enabled.driver_state, enabled.driver_scratch, error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(enabled_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  double expected_raw_energy = 0.0;
  CHECK(evaluate_d4_two_body_system_cpu(enabled.d4_plan, enabled.d4_cache, 0,
                                        enabled.driver_scratch.raw_qat, expected_raw_energy,
                                        nullptr, enabled.driver_scratch.d4_workspace,
                                        error) == XTBLOOM_STATUS_SUCCESS);
  /* The Hamiltonian uses the mixed input potential above, but the published
   * complete SCC energy must be evaluated from density-derived raw charges. */
  CHECK(std::abs(enabled.driver_state.d4_two_body_energies[0] - expected_raw_energy) < 1.0e-14);
  CHECK(std::abs(expected_raw_energy - expected_energy[0]) > 1.0e-16);
  for (std::size_t atom = 0u; atom < expected_potential.size(); ++atom) {
    CHECK(std::abs(enabled.driver_scratch.d4_atomic_potentials[atom] - expected_potential[atom]) <
          1.0e-14);
    CHECK(std::abs((enabled.driver_scratch.atomic_potentials[atom] -
                    reference.driver_scratch.atomic_potentials[atom]) -
                   expected_potential[atom]) < 1.0e-14);
  }
  const std::int64_t orbitals = enabled.wavefunction_layout.total_orbitals;
  for (std::int64_t row = 0; row < orbitals; ++row) {
    const auto row_upper = std::upper_bound(enabled.basis.atom_orbital_offsets.begin(),
                                            enabled.basis.atom_orbital_offsets.end(), row);
    const std::size_t row_atom =
        static_cast<std::size_t>(row_upper - enabled.basis.atom_orbital_offsets.begin() - 1);
    for (std::int64_t column = 0; column < orbitals; ++column) {
      const auto column_upper = std::upper_bound(enabled.basis.atom_orbital_offsets.begin(),
                                                 enabled.basis.atom_orbital_offsets.end(), column);
      const std::size_t column_atom =
          static_cast<std::size_t>(column_upper - enabled.basis.atom_orbital_offsets.begin() - 1);
      const std::size_t matrix = static_cast<std::size_t>(row * orbitals + column);
      const double expected_shift =
          -0.5 * enabled.overlap[matrix] *
          (expected_potential[row_atom] + expected_potential[column_atom]);
      CHECK(std::abs((enabled.driver_scratch.hamiltonian[matrix] -
                      reference.driver_scratch.hamiltonian[matrix]) -
                     expected_shift) < 1.0e-13);
    }
  }

  CHECK(restart_scc_driver_system_cpu(enabled.driver_plan, 0, enabled.wavefunction,
                                      enabled.mixer_state, enabled.driver_state,
                                      error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::isnan(enabled.driver_state.d4_two_body_energies[0]));
  const double restarted_qsh = enabled.wavefunction.qsh[0];
  SccDriverGeometryView stale = enabled.geometry;
  ++stale.d4_cache.geometry_generation;
  CHECK(iterate_scc_driver_batch_cpu(enabled.driver_plan, stale, backend(), enabled.overlap_cache,
                                     enabled.wavefunction, enabled.mixer_state,
                                     enabled.driver_state, enabled.driver_scratch,
                                     error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(enabled.driver_state.iterations[0] == 0u);
  CHECK(std::isnan(enabled.driver_state.d4_two_body_energies[0]));
  CHECK(enabled.wavefunction.qsh[0] == restarted_qsh);

  D4Plan same_topology_other_d4;
  CHECK(make_d4_plan(1, 2, enabled.atom_offsets.data(), enabled.atomic_numbers.data(),
                     same_topology_other_d4, error) == XTBLOOM_STATUS_SUCCESS);
  SccDriverGeometryView wrong_identity = enabled.geometry;
  wrong_identity.d4_cache.plan_identity = same_topology_other_d4.identity();
  CHECK(iterate_scc_driver_batch_cpu(
            enabled.driver_plan, wrong_identity, backend(), enabled.overlap_cache,
            enabled.wavefunction, enabled.mixer_state, enabled.driver_state, enabled.driver_scratch,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(enabled.driver_state.iterations[0] == 0u);
  CHECK(std::isnan(enabled.driver_state.d4_two_body_energies[0]));

  SccDriverGeometryView aliased_plan_storage = enabled.geometry;
  aliased_plan_storage.d4_cache.coordination_numbers =
      reinterpret_cast<double*>(const_cast<std::int64_t*>(enabled.d4_plan.atom_offsets().data()));
  CHECK(iterate_scc_driver_batch_cpu(
            enabled.driver_plan, aliased_plan_storage, backend(), enabled.overlap_cache,
            enabled.wavefunction, enabled.mixer_state, enabled.driver_state, enabled.driver_scratch,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(enabled.driver_state.iterations[0] == 0u);
  CHECK(std::isnan(enabled.driver_state.d4_two_body_energies[0]));

  SccDriverWorkspace malformed_workspace = enabled.driver_scratch;
  ++malformed_workspace.d4_atomic_potentials;
  CHECK(iterate_scc_driver_batch_cpu(enabled.driver_plan, enabled.geometry, backend(),
                                     enabled.overlap_cache, enabled.wavefunction,
                                     enabled.mixer_state, enabled.driver_state, malformed_workspace,
                                     error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(enabled.driver_state.iterations[0] == 0u);
  malformed_workspace = enabled.driver_scratch;
  ++malformed_workspace.d4_workspace.weights;
  CHECK(iterate_scc_driver_batch_cpu(enabled.driver_plan, enabled.geometry, backend(),
                                     enabled.overlap_cache, enabled.wavefunction,
                                     enabled.mixer_state, enabled.driver_state, malformed_workspace,
                                     error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(enabled.driver_state.iterations[0] == 0u);

  SccDriverGeometryView unexpected_d4 = reference.geometry;
  unexpected_d4.d4_cache = enabled.d4_cache;
  CHECK(iterate_scc_driver_batch_cpu(
            reference.driver_plan, unexpected_d4, backend(), reference.overlap_cache,
            reference.wavefunction, reference.mixer_state, reference.driver_state,
            reference.driver_scratch, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  Fixture skipped;
  CHECK(make_fixture(1, skipped, error, 5u, 1.0e-10, false, nullptr, false, true));
  CHECK(iterate_scc_driver_batch_cpu(skipped.driver_plan, skipped.geometry, backend(),
                                     skipped.overlap_cache, skipped.wavefunction,
                                     skipped.mixer_state, skipped.driver_state,
                                     skipped.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(skipped.driver_state.converged[0] == 1u);
  const double converged_d4_energy = skipped.driver_state.d4_two_body_energies[0];
  skipped.driver_scratch.d4_two_body_energies[0] = 123.0;
  skipped.driver_scratch.d4_atomic_potentials[0] = 456.0;
  const int calls_before_skip = diagonalizations.load(std::memory_order_relaxed);
  CHECK(iterate_scc_driver_batch_cpu(skipped.driver_plan, skipped.geometry, backend(),
                                     skipped.overlap_cache, skipped.wavefunction,
                                     skipped.mixer_state, skipped.driver_state,
                                     skipped.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == calls_before_skip);
  CHECK(skipped.driver_state.d4_two_body_energies[0] == converged_d4_energy);
  CHECK(skipped.driver_scratch.d4_two_body_energies[0] == 123.0);
  CHECK(skipped.driver_scratch.d4_atomic_potentials[0] == 456.0);

  Fixture isolated_failure;
  CHECK(make_fixture(2, isolated_failure, error, 5u, 1.0e-10, false, nullptr, false, true));
  isolated_failure.overlap_cache.system_statuses[1] = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
  CHECK(iterate_scc_driver_batch_cpu(isolated_failure.driver_plan, isolated_failure.geometry,
                                     backend(), isolated_failure.overlap_cache,
                                     isolated_failure.wavefunction, isolated_failure.mixer_state,
                                     isolated_failure.driver_state, isolated_failure.driver_scratch,
                                     error) == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  CHECK(isolated_failure.driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(isolated_failure.driver_state.iterations[0] == 1u);
  CHECK(isolated_failure.driver_state.d4_two_body_energies[0] == 0.0);
  CHECK(isolated_failure.driver_state.system_statuses[1] == XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  CHECK(isolated_failure.driver_state.iterations[1] == 1u);
  CHECK(std::isnan(isolated_failure.driver_state.d4_two_body_energies[1]));
  return 0;
}

int test_d4_system_potential_uses_full_output_layout() {
  const FixtureTopology topology{
      {0, 2, 4},  {1, 1, 1, 1}, {-1.1, 0.0, 0.0, 1.1, 0.0, 0.0, 3.0, 0.0, 0.0, 5.2, 0.0, 0.0},
      {0.0, 0.0}, {},           {}};
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(2, fixture, error, 1u, 1.0e100, false, &topology, false, true, 1.0e100));

  /* The one-system D4 API receives full-layout output storage and applies the
   * target atom offset itself. Re-evaluate the second member into an
   * independent buffer to guard the driver's call site against double-offset
   * writes that would corrupt or overrun the first/second member boundary. */
  std::vector<double> mixed_charges(static_cast<std::size_t>(fixture.d4_plan.total_atoms()), 0.0);
  for (std::size_t system = 0; system < 2u; ++system) {
    const std::int64_t shell_begin = fixture.wavefunction_layout.batch_shell_offsets[system];
    const std::int64_t shell_end = fixture.wavefunction_layout.batch_shell_offsets[system + 1u];
    const std::int64_t qsh_base = fixture.wavefunction_layout.qsh.system_offsets[system];
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::int64_t atom =
          fixture.mulliken_plan.shell_to_atom()[static_cast<std::size_t>(shell)];
      mixed_charges[static_cast<std::size_t>(atom)] +=
          fixture.wavefunction.qsh[static_cast<std::size_t>(qsh_base + shell - shell_begin)];
    }
  }
  const std::int64_t atom_begin = fixture.d4_plan.atom_offsets()[1];
  const std::int64_t atom_end = fixture.d4_plan.atom_offsets()[2];
  std::vector<double> expected(static_cast<std::size_t>(fixture.d4_plan.total_atoms()), 0.0);
  double expected_energy = 0.0;
  CHECK(evaluate_d4_two_body_system_cpu(
            fixture.d4_plan, fixture.d4_cache, 1, mixed_charges.data(), expected_energy,
            expected.data(), fixture.driver_scratch.d4_workspace, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::any_of(expected.begin() + atom_begin, expected.begin() + atom_end,
                    [](double value) { return std::abs(value) > 1.0e-18; }));

  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    CHECK(fixture.driver_scratch.d4_atomic_potentials[static_cast<std::size_t>(atom)] ==
          expected[static_cast<std::size_t>(atom)]);
  }
  return 0;
}

int test_d4_atm_is_not_part_of_scc() {
  const FixtureTopology topology{
      {0, 3}, {6, 6, 6}, {0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.5, 2.598076211353316, 0.0},
      {0.0},  {},        {}};
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(1, fixture, error, 5u, 1.0e-10, false, &topology, false, true));

  std::array<double, 1> two_body{};
  std::array<double, 3> potential{};
  std::array<double, 1> atm{};
  CHECK(evaluate_d4_two_body_cpu(fixture.d4_plan, fixture.d4_cache, fixture.wavefunction.qat,
                                 two_body.data(), potential.data(),
                                 fixture.driver_scratch.d4_workspace,
                                 error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(evaluate_d4_atm_cpu(fixture.d4_plan, fixture.d4_cache, atm.data(),
                            fixture.driver_scratch.d4_workspace, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::abs(atm[0]) > 1.0e-18);

  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  const double stored = fixture.driver_state.d4_two_body_energies[0];
  CHECK(std::abs(stored - two_body[0]) < 1.0e-6 * std::abs(atm[0]));
  CHECK(std::abs(stored - (two_body[0] + atm[0])) > 0.5 * std::abs(atm[0]));
  return 0;
}

int test_optional_periodic_embedding_and_explicit_point_charge_composition() {
  Fixture reference;
  Fixture embedded;
  std::string error;
  error.reserve(256u);
  CHECK(make_fixture(1, reference, error));
  CHECK(make_fixture(1, embedded, error, 5u, 1.0e-10, true));
  CHECK(!reference.driver_plan.periodic_embedding_enabled());
  CHECK(embedded.driver_plan.periodic_embedding_enabled());
  CHECK(reference.driver_state.periodic_embedding_energies == nullptr);
  CHECK(reference.driver_scratch.periodic_atomic_potentials == nullptr);
  CHECK(reference.driver_scratch.periodic_embedding_energies == nullptr);
  CHECK(reference.driver_scratch.periodic_system_statuses == nullptr);
  CHECK(reference.driver_scratch.periodic_embedding_workspace.potential_scratch == nullptr);

  embedded.periodic_shifts[0] = 0.25;
  embedded.periodic_response[0] = 0.5;
  embedded.geometry.periodic_embedding_generation = 73u;
  const std::array<double, 1> point_charge_shell_potential{{0.125}};
  embedded.geometry.explicit_point_charge_shell_potential = point_charge_shell_potential.data();
  embedded.geometry.explicit_point_charge_shell_elements = 1;

  CHECK(iterate_scc_driver_batch_cpu(reference.driver_plan, reference.geometry, backend(),
                                     reference.overlap_cache, reference.wavefunction,
                                     reference.mixer_state, reference.driver_state,
                                     reference.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t status = iterate_scc_driver_batch_cpu(
      embedded.driver_plan, embedded.geometry, backend(), embedded.overlap_cache,
      embedded.wavefunction, embedded.mixer_state, embedded.driver_state, embedded.driver_scratch,
      error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  CHECK(status == XTBLOOM_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);

  const double periodic_potential = 0.25 + 0.5 * 1.0;
  CHECK(std::abs((embedded.wavefunction.eigenvalues[0] - reference.wavefunction.eigenvalues[0]) +
                 point_charge_shell_potential[0] + periodic_potential) < 1.0e-14);
  CHECK(std::abs(embedded.driver_state.periodic_embedding_energies[0] - 0.5) < 1.0e-14);
  return 0;
}

int test_ragged_dense_periodic_response_publishes_raw_energy() {
  const FixtureTopology topology{
      {0, 2, 3}, {6, 1, 1}, {0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 1.0}, {}, {}};
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(2, fixture, error, 5u, 1.0e6, true, &topology));
  CHECK(fixture.wavefunction_layout.batch_shell_offsets[1] -
            fixture.wavefunction_layout.batch_shell_offsets[0] >
        2);
  CHECK(fixture.periodic_plan.matrix_offsets() == std::vector<std::int64_t>({0, 4, 5}));
  CHECK(std::abs(fixture.wavefunction.qat[0] - 0.5) < 1.0e-14);
  CHECK(std::abs(fixture.wavefunction.qat[1] - 0.5) < 1.0e-14);
  CHECK(std::abs(fixture.wavefunction.qat[2] - 1.0) < 1.0e-14);

  /* The periodic driver must rebuild atomic charges from mixed shell charges;
   * stale public qat values are not valid SCC inputs for b + A q. */
  fixture.wavefunction.qat[0] = -7.0;
  fixture.wavefunction.qat[1] = 11.0;
  fixture.wavefunction.qat[2] = -13.0;

  fixture.periodic_shifts = {0.1, -0.2, 0.3};
  fixture.periodic_response = {0.4, 0.15, 0.15, -0.1, 0.25};
  fixture.geometry.periodic_shifts = fixture.periodic_shifts.data();
  fixture.geometry.periodic_response_matrices = fixture.periodic_response.data();
  fixture.geometry.periodic_embedding_generation = 999u;
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);

  /* Hamiltonian assembly consumed the initial mixed charges. The complete
   * energy pass intentionally leaves the diagnostic potential rebuilt from
   * the density-derived raw charges used for the published energy trace. */
  const double q0 = fixture.driver_scratch.raw_qat[0];
  const double q1 = fixture.driver_scratch.raw_qat[1];
  const double q2 = fixture.driver_scratch.raw_qat[2];
  const double potential0 = 0.1 + 0.4 * q0 + 0.15 * q1;
  const double potential1 = -0.2 + 0.15 * q0 - 0.1 * q1;
  const double potential2 = 0.3 + 0.25 * q2;
  CHECK(std::abs(fixture.driver_scratch.periodic_atomic_potentials[0] - potential0) < 1.0e-14);
  CHECK(std::abs(fixture.driver_scratch.periodic_atomic_potentials[1] - potential1) < 1.0e-14);
  CHECK(std::abs(fixture.driver_scratch.periodic_atomic_potentials[2] - potential2) < 1.0e-14);
  const double energy0 = 0.5 * (0.1 + potential0) * q0 + 0.5 * (-0.2 + potential1) * q1;
  const double energy1 = 0.5 * (0.3 + potential2) * q2;
  CHECK(std::abs(fixture.driver_state.periodic_embedding_energies[0] - energy0) < 1.0e-14);
  CHECK(std::abs(fixture.driver_state.periodic_embedding_energies[1] - energy1) < 1.0e-14);
  CHECK(fixture.driver_scratch.periodic_system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_scratch.periodic_system_statuses[1] == XTBLOOM_STATUS_SUCCESS);
  return 0;
}

int test_periodic_numerical_failure_is_isolated_before_eigensolve() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(2, fixture, error, 5u, 1.0e-10, true));
  fixture.periodic_shifts[1] = std::numeric_limits<double>::quiet_NaN();
  const double failed_qsh_before = fixture.wavefunction.qsh[1];
  diagonalizations.store(0, std::memory_order_relaxed);

  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
            fixture.wavefunction, fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == 1);
  CHECK(fixture.driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.driver_state.iterations[0] == 1u);
  CHECK(fixture.driver_state.converged[0] == 1u);
  CHECK(fixture.driver_state.periodic_embedding_energies[0] == 0.0);
  CHECK(fixture.driver_state.system_statuses[1] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.driver_state.iterations[1] == 0u);
  CHECK(std::isnan(fixture.driver_state.periodic_embedding_energies[1]));
  CHECK(fixture.wavefunction.qsh[1] == failed_qsh_before);

  fixture.periodic_shifts[1] = 0.0;
  CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 1, fixture.wavefunction,
                                      fixture.mixer_state, fixture.driver_state,
                                      error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state,
                                     fixture.driver_scratch, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(diagonalizations.load(std::memory_order_relaxed) == 2);
  CHECK(fixture.driver_state.iterations[1] == 1u);
  return 0;
}

int test_periodic_plan_geometry_workspace_provenance_and_aliases() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture(1, fixture, error, 5u, 1.0e-10, true));
  const double qsh_before = fixture.wavefunction.qsh[0];
  const xtbloom_status_t state_status_before = fixture.driver_state.system_statuses[0];

  PeriodicEmbeddingPlan same_topology_other_identity;
  CHECK(make_periodic_embedding_plan(1, 1, fixture.atom_offsets.data(),
                                     same_topology_other_identity,
                                     error) == XTBLOOM_STATUS_SUCCESS);
  SccDriverGeometryView malformed = fixture.geometry;
  malformed.periodic_plan_identity = same_topology_other_identity.identity();
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, malformed, backend(), fixture.overlap_cache, fixture.wavefunction,
            fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.wavefunction.qsh[0] == qsh_before);
  CHECK(fixture.driver_state.system_statuses[0] == state_status_before);

  malformed = fixture.geometry;
  malformed.periodic_embedding_generation = 0u;
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, malformed, backend(), fixture.overlap_cache, fixture.wavefunction,
            fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  malformed = fixture.geometry;
  malformed.periodic_shifts = fixture.driver_state.free_energies;
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, malformed, backend(), fixture.overlap_cache, fixture.wavefunction,
            fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  malformed = fixture.geometry;
  malformed.periodic_shifts =
      reinterpret_cast<const double*>(fixture.periodic_plan.atom_offsets().data());
  CHECK(iterate_scc_driver_batch_cpu(
            fixture.driver_plan, malformed, backend(), fixture.overlap_cache, fixture.wavefunction,
            fixture.mixer_state, fixture.driver_state, fixture.driver_scratch,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  SccDriverWorkspace malformed_workspace = fixture.driver_scratch;
  malformed_workspace.periodic_embedding_workspace.plan_identity =
      same_topology_other_identity.identity();
  CHECK(iterate_scc_driver_batch_cpu(fixture.driver_plan, fixture.geometry, backend(),
                                     fixture.overlap_cache, fixture.wavefunction,
                                     fixture.mixer_state, fixture.driver_state, malformed_workspace,
                                     error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  Fixture disabled;
  CHECK(make_fixture(1, disabled, error));
  SccDriverGeometryView unexpected = disabled.geometry;
  unexpected.periodic_shifts = fixture.periodic_shifts.data();
  unexpected.periodic_shift_elements = 1;
  unexpected.periodic_response_matrices = fixture.periodic_response.data();
  unexpected.periodic_response_elements = 1;
  unexpected.periodic_embedding_generation = unexpected.geometry_generation;
  unexpected.periodic_plan_identity = fixture.periodic_plan.identity();
  CHECK(iterate_scc_driver_batch_cpu(
            disabled.driver_plan, unexpected, backend(), disabled.overlap_cache,
            disabled.wavefunction, disabled.mixer_state, disabled.driver_state,
            disabled.driver_scratch, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  const std::array<std::int64_t, 2> empty_offsets{{0, 0}};
  PeriodicEmbeddingPlan wrong_topology;
  CHECK(make_periodic_embedding_plan(1, 0, empty_offsets.data(), wrong_topology, error) ==
        XTBLOOM_STATUS_SUCCESS);
  SccDriverPlan sentinel = fixture.driver_plan;
  const SccDriverPlanData* const sentinel_identity = sentinel.identity();
  CHECK(make_scc_driver_plan(fixture.wavefunction_layout, fixture.mulliken_plan, fixture.es2_plan,
                             fixture.es3_plan, fixture.aes2_plan, fixture.eigensolver_plan,
                             fixture.mixer_plan, &wrong_topology, 5u, 0.0, sentinel,
                             error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(sentinel.identity() == sentinel_identity);
  return 0;
}

int test_component_chemistry_and_layout_mismatches_are_rejected() {
  std::string error;
  ComponentPlans ch;
  ComponentPlans hc;
  CHECK(make_component_plans({6, 1}, 1.0, 0, ch, error));
  CHECK(make_component_plans({1, 6}, 1.0, 0, hc, error));
  CHECK(ch.wavefunction.total_atoms == hc.wavefunction.total_atoms);
  CHECK(ch.wavefunction.total_shells == hc.wavefunction.total_shells);
  CHECK(ch.wavefunction.total_orbitals == hc.wavefunction.total_orbitals);
  CHECK(ch.wavefunction.electron_counts == hc.wavefunction.electron_counts);

  SccDriverPlan sentinel;
  CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, ch.es2, ch.es3, ch.aes2, ch.eigensolver,
                             ch.mixer, 5u, 0.0, sentinel, error) == XTBLOOM_STATUS_SUCCESS);
  const SccDriverPlanData* const sentinel_identity = sentinel.identity();
  SccDriverPlan output = sentinel;

  for (const double invalid_tolerance : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                         std::numeric_limits<double>::quiet_NaN()}) {
    CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, ch.es2, ch.es3, ch.aes2,
                               ch.eigensolver, ch.mixer, nullptr, nullptr, 5u, 0.0,
                               invalid_tolerance, output,
                               error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(output.identity() == sentinel_identity);
  }

  CHECK(make_scc_driver_plan(ch.wavefunction, hc.mulliken, ch.es2, ch.es3, ch.aes2, ch.eigensolver,
                             ch.mixer, 5u, 0.0, output, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);
  CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, hc.es2, ch.es3, ch.aes2, ch.eigensolver,
                             ch.mixer, 5u, 0.0, output, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);
  CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, ch.es2, hc.es3, ch.aes2, ch.eigensolver,
                             ch.mixer, 5u, 0.0, output, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);
  CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, ch.es2, ch.es3, hc.aes2, ch.eigensolver,
                             ch.mixer, 5u, 0.0, output, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);

  D4Plan wrong_d4_chemistry;
  CHECK(make_d4_plan(1, static_cast<std::int64_t>(hc.atomic_numbers.size()), hc.atom_offsets.data(),
                     hc.atomic_numbers.data(), wrong_d4_chemistry,
                     error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, ch.es2, ch.es3, ch.aes2, ch.eigensolver,
                             ch.mixer, &wrong_d4_chemistry, nullptr, 5u, 0.0, output,
                             error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);

  ES3Plan modified_es3 = ch.es3;
  modified_es3.shell_gamma3[0] += 1.0;
  CHECK(make_scc_driver_plan(ch.wavefunction, ch.mulliken, ch.es2, modified_es3, ch.aes2,
                             ch.eigensolver, ch.mixer, 5u, 0.0, output,
                             error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);

  ComponentPlans h_plus;
  ComponentPlans neutral_h;
  CHECK(make_component_plans({1}, 1.0, 0, h_plus, error));
  CHECK(make_component_plans({1}, 0.0, 1, neutral_h, error));
  CHECK(make_scc_driver_plan(h_plus.wavefunction, h_plus.mulliken, h_plus.es2, h_plus.es3,
                             h_plus.aes2, neutral_h.eigensolver, h_plus.mixer, 5u, 0.0, output,
                             error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == sentinel_identity);

  ComponentPlans five_neon;
  ComponentPlans six_hydrogen;
  CHECK(make_component_plans({10, 10, 10, 10, 10}, 0.0, 0, five_neon, error));
  CHECK(make_component_plans({1, 1, 1, 1, 1, 1}, 0.0, 0, six_hydrogen, error));
  CHECK(five_neon.mixer.total_vector_elements() == six_hydrogen.mixer.total_vector_elements());
  SccDriverPlan neon_driver;
  CHECK(make_scc_driver_plan(five_neon.wavefunction, five_neon.mulliken, five_neon.es2,
                             five_neon.es3, five_neon.aes2, five_neon.eigensolver, five_neon.mixer,
                             5u, 0.0, neon_driver, error) == XTBLOOM_STATUS_SUCCESS);
  output = neon_driver;
  const SccDriverPlanData* const neon_identity = neon_driver.identity();
  CHECK(make_scc_driver_plan(five_neon.wavefunction, five_neon.mulliken, five_neon.es2,
                             five_neon.es3, five_neon.aes2, five_neon.eigensolver,
                             six_hydrogen.mixer, 5u, 0.0, output,
                             error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(output.identity() == neon_identity);
  return 0;
}

int test_control_descriptors_cannot_alias_numerical_storage() {
  std::string error;
  {
    Fixture fixture;
    CHECK(make_fixture(1, fixture, error));
    auto* aliased_state =
        ::new (fixture.wavefunction_storage->data) SccDriverState(fixture.driver_state);
    std::vector<unsigned char> mixer_before(fixture.mixer_state_storage->size);
    std::vector<unsigned char> state_before(fixture.driver_state_storage->size);
    std::memcpy(mixer_before.data(), fixture.mixer_state_storage->data, mixer_before.size());
    std::memcpy(state_before.data(), fixture.driver_state_storage->data, state_before.size());
    CHECK(initialize_scc_driver_state_cpu(fixture.driver_plan, fixture.wavefunction,
                                          fixture.mixer_state, *aliased_state,
                                          error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::memcmp(mixer_before.data(), fixture.mixer_state_storage->data,
                      mixer_before.size()) == 0);
    CHECK(std::memcmp(state_before.data(), fixture.driver_state_storage->data,
                      state_before.size()) == 0);
  }
  {
    Fixture fixture;
    CHECK(make_fixture(1, fixture, error));
    auto* aliased_state =
        ::new (fixture.mixer_state_storage->data) SccDriverState(fixture.driver_state);
    std::vector<unsigned char> mixer_before(fixture.mixer_state_storage->size);
    std::vector<unsigned char> wavefunction_before(fixture.wavefunction_storage->size);
    std::memcpy(mixer_before.data(), fixture.mixer_state_storage->data, mixer_before.size());
    std::memcpy(wavefunction_before.data(), fixture.wavefunction_storage->data,
                wavefunction_before.size());
    CHECK(restart_scc_driver_system_cpu(fixture.driver_plan, 0, fixture.wavefunction,
                                        fixture.mixer_state, *aliased_state,
                                        error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::memcmp(mixer_before.data(), fixture.mixer_state_storage->data,
                      mixer_before.size()) == 0);
    CHECK(std::memcmp(wavefunction_before.data(), fixture.wavefunction_storage->data,
                      wavefunction_before.size()) == 0);
  }
  {
    Fixture fixture;
    CHECK(make_fixture(1, fixture, error));
    auto* aliased_wavefunction =
        ::new (fixture.driver_scratch_storage->data) WavefunctionView(fixture.wavefunction);
    std::vector<unsigned char> scratch_before(fixture.driver_scratch_storage->size);
    std::vector<unsigned char> state_before(fixture.driver_state_storage->size);
    std::memcpy(scratch_before.data(), fixture.driver_scratch_storage->data, scratch_before.size());
    std::memcpy(state_before.data(), fixture.driver_state_storage->data, state_before.size());
    CHECK(iterate_scc_driver_batch_cpu(
              fixture.driver_plan, fixture.geometry, backend(), fixture.overlap_cache,
              *aliased_wavefunction, fixture.mixer_state, fixture.driver_state,
              fixture.driver_scratch, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(std::memcmp(scratch_before.data(), fixture.driver_scratch_storage->data,
                      scratch_before.size()) == 0);
    CHECK(std::memcmp(state_before.data(), fixture.driver_state_storage->data,
                      state_before.size()) == 0);
  }
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_complete_energy_components_free_energy_and_restart(); status != 0) {
    return status;
  }
  if (const int status = test_energy_and_residual_convergence_gates_are_independent_and_strict();
      status != 0) {
    return status;
  }
  if (const int status = test_complete_energy_failure_isolated_from_ragged_peer(); status != 0) {
    return status;
  }
  if (const int status = test_preparation_numerical_failure_isolated_from_ragged_peer();
      status != 0) {
    return status;
  }
  if (const int status = test_component_chemistry_and_layout_mismatches_are_rejected();
      status != 0) {
    return status;
  }
  if (const int status = test_control_descriptors_cannot_alias_numerical_storage(); status != 0) {
    return status;
  }
  if (const int status = test_mixed_restricted_unrestricted_hydrogen_batch(); status != 0) {
    return status;
  }
  if (const int status = test_max_iteration_status_counts_attempts(); status != 0) {
    return status;
  }
  if (const int status = test_mixer_failure_preserves_public_history_and_counts_attempt();
      status != 0) {
    return status;
  }
  if (const int status = test_ragged_mixer_failure_isolated_from_peer_commit(); status != 0) {
    return status;
  }
  if (const int status = test_energy_history_failure_isolated_from_peer_commit(); status != 0) {
    return status;
  }
  if (const int status = test_inactive_peer_history_untouched_by_active_only_commit();
      status != 0) {
    return status;
  }
  if (const int status = test_converged_wavefunction_publishes_raw_mulliken_multipoles();
      status != 0) {
    return status;
  }
  if (const int status = test_cached_explicit_point_charge_potential_enters_hamiltonian();
      status != 0) {
    return status;
  }
  if (const int status = test_disabled_layout_and_bitwise_compatibility(); status != 0) {
    return status;
  }
  if (const int status = test_optional_d4_potential_energy_restart_and_zero_allocation();
      status != 0) {
    return status;
  }
  if (const int status = test_d4_system_potential_uses_full_output_layout(); status != 0) {
    return status;
  }
  if (const int status = test_d4_atm_is_not_part_of_scc(); status != 0) {
    return status;
  }
  if (const int status = test_optional_periodic_embedding_and_explicit_point_charge_composition();
      status != 0) {
    return status;
  }
  if (const int status = test_ragged_dense_periodic_response_publishes_raw_energy(); status != 0) {
    return status;
  }
  if (const int status = test_periodic_numerical_failure_is_isolated_before_eigensolve();
      status != 0) {
    return status;
  }
  if (const int status = test_periodic_plan_geometry_workspace_provenance_and_aliases();
      status != 0) {
    return status;
  }
  if (const int status = test_ragged_failure_isolation_restart_and_skip(); status != 0) {
    return status;
  }
  if (const int status = test_converged_system_skips_classical_and_mulliken_arithmetic();
      status != 0) {
    return status;
  }
  return test_structural_failure_atomicity_and_zero_allocation();
}
