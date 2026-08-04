#include "tests/support/gfn2_scc_test_case.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace gpuxtb::test::gfn2 {
namespace {

using namespace gpuxtb::detail::gfn2;

constexpr std::size_t kHostAlignment = 64u;

/* Own one zero-initialized allocation satisfying every current host binding. */
class AlignedBuffer {
 public:
  AlignedBuffer() noexcept = default;
  ~AlignedBuffer() { std::free(data_); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  [[nodiscard]] bool allocate(std::size_t requested) noexcept {
    if (data_ != nullptr || requested > std::numeric_limits<std::size_t>::max() - 63u) {
      return false;
    }
    size_ = (std::max<std::size_t>(requested, 1u) + 63u) & ~std::size_t{63u};
    data_ = std::aligned_alloc(kHostAlignment, size_);
    if (data_ == nullptr) {
      size_ = 0u;
      return false;
    }
    std::memset(data_, 0, size_);
    return true;
  }

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0u;
};

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

/* Jacobi is sufficient for the fixture's at-most-six-orbital molecules. */
LapackInt tiny_dsyevd(LapackInt, char, char, LapackInt n, double* matrix, LapackInt,
                      double* eigenvalues, double*, LapackInt, LapackInt*, LapackInt) {
  if (n <= 0 || n > 16) {
    return -4;
  }
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
    /* The eigensolver only requests right-side L^T solves in this path. */
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

gpuxtb_status_t allocate(AlignedBuffer& buffer, std::size_t bytes, const char* purpose,
                         std::string& error) {
  if (buffer.allocate(bytes)) {
    return GPUXTB_STATUS_SUCCESS;
  }
  error = std::string("failed to allocate host SCC fixture ") + purpose;
  return GPUXTB_STATUS_ALLOCATION_FAILED;
}

std::vector<std::byte> copy_bytes(const AlignedBuffer& buffer) {
  std::vector<std::byte> copy(buffer.size());
  std::memcpy(copy.data(), buffer.data(), buffer.size());
  return copy;
}

}  // namespace

struct HostSccCase::Impl {
  HostSccCaseOptions options;
  std::int64_t batch_size = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  std::vector<double> coordination_numbers;

  BasisPlan basis;
  IntegralPlan integrals;
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
  ExternalPointChargePlan point_charge_plan;
  SccDriverPlan driver_plan;

  std::vector<std::int64_t> point_charge_offsets;
  std::vector<double> point_charge_positions;
  std::vector<double> point_charge_charges;
  std::vector<double> point_charge_hardnesses;
  std::vector<double> explicit_point_charge_shell_potential;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response_matrices;

  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<double> h0;
  AlignedBuffer integral_scratch;

  AlignedBuffer es2_storage;
  AlignedBuffer es2_scratch_storage;
  ES2GeometryCache es2_cache;
  ES2Workspace es2_scratch;
  AlignedBuffer aes2_storage;
  AlignedBuffer aes2_scratch_storage;
  AES2GeometryCache aes2_cache;
  AES2Workspace aes2_scratch;
  std::vector<double> d4_pair_data;
  std::vector<double> d4_coordination;
  D4GeometryCache d4_cache;

  AlignedBuffer wavefunction_storage;
  WavefunctionView wavefunction;
  AlignedBuffer overlap_cache_storage;
  EigensolverOverlapCache overlap_cache;
  AlignedBuffer eigensolver_scratch_storage;
  EigensolverWorkspace eigensolver_scratch;
  AlignedBuffer mixer_state_storage;
  SccMixerState mixer_state;
  AlignedBuffer driver_state_storage;
  SccDriverState driver_state;
  AlignedBuffer driver_workspace_storage;
  SccDriverWorkspace driver_workspace;
  SccDriverGeometryView geometry;
  CpuLinearAlgebraBackend cpu_backend;

  gpuxtb_status_t build(std::string& error);
  bool append_system(SmallSystemKind kind, std::int64_t system);
};

bool HostSccCase::Impl::append_system(SmallSystemKind kind, std::int64_t system) {
  const double shift = 8.0 * static_cast<double>(system);
  const auto atom = [&](std::int32_t atomic_number, double x, double y, double z) {
    atomic_numbers.push_back(atomic_number);
    positions.insert(positions.end(), {x + shift, y, z});
  };

  switch (kind) {
    case SmallSystemKind::kH2:
      atom(1, -0.7, 0.0, 0.0);
      atom(1, 0.7, 0.0, 0.0);
      break;
    case SmallSystemKind::kHe:
      atom(2, 0.0, 0.0, 0.0);
      break;
    case SmallSystemKind::kLiH:
      atom(3, -1.5, 0.0, 0.0);
      atom(1, 1.5, 0.0, 0.0);
      break;
    case SmallSystemKind::kCH2:
      atom(6, 0.0, 0.0, 0.0);
      atom(1, 1.6, 0.0, 1.0);
      atom(1, -1.6, 0.0, 1.0);
      break;
    default:
      return false;
  }
  atom_offsets.push_back(static_cast<std::int64_t>(atomic_numbers.size()));
  const std::size_t electronic_index = static_cast<std::size_t>(system);
  molecular_charges.push_back(
      options.molecular_charges.empty() ? 0.0 : options.molecular_charges[electronic_index]);
  unpaired_electrons.push_back(
      options.unpaired_electrons.empty() ? 0 : options.unpaired_electrons[electronic_index]);
  spin_channels.push_back(options.spin_channels.empty() ? 1
                                                        : options.spin_channels[electronic_index]);
  return true;
}

gpuxtb_status_t HostSccCase::Impl::build(std::string& error) {
  error.clear();
  if (options.systems.empty()) {
    error = "host SCC fixture requires at least one system";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (options.systems.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    error = "host SCC fixture batch size exceeds int64 range";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (options.geometry_generation == 0u) {
    error = "host SCC fixture geometry generation must be nonzero";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const auto valid_optional_batch = [&](std::size_t elements) {
    return elements == 0u || elements == options.systems.size();
  };
  if (!valid_optional_batch(options.molecular_charges.size()) ||
      !valid_optional_batch(options.unpaired_electrons.size()) ||
      !valid_optional_batch(options.spin_channels.size())) {
    error = "host SCC fixture electronic vectors must be empty or match systems.size()";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  batch_size = static_cast<std::int64_t>(options.systems.size());
  atom_offsets.reserve(options.systems.size() + 1u);
  atom_offsets.push_back(0);
  for (std::size_t system = 0; system < options.systems.size(); ++system) {
    if (!append_system(options.systems[system], static_cast<std::int64_t>(system))) {
      error = "host SCC fixture contains an unknown small-system kind";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  const std::int64_t total_atoms = atom_offsets.back();
  coordination_numbers.assign(static_cast<std::size_t>(total_atoms), 0.0);

  gpuxtb_status_t status = make_basis_plan(batch_size, total_atoms, atom_offsets.data(),
                                           atomic_numbers.data(), basis, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = make_integral_plan(basis, integrals, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = make_h0_plan(basis, integrals, atomic_numbers.data(), h0_plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = make_wavefunction_layout(basis, atomic_numbers.data(), molecular_charges.data(),
                                    unpaired_electrons.data(), spin_channels.data(),
                                    wavefunction_layout, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = make_es2_plan(basis, atomic_numbers.data(), es2_plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = make_es3_plan(basis, atomic_numbers.data(), es3_plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = make_aes2_plan(basis, atomic_numbers.data(), aes2_plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = make_mulliken_plan(basis, integrals, wavefunction_layout, mulliken_plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = make_eigensolver_plan(wavefunction_layout, eigensolver_plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = make_scc_mixer_plan(wavefunction_layout, options.mixer_history, options.mixer_damping,
                               options.residual_tolerance, options.residual_tolerance, mixer_plan,
                               error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  if (options.enable_d4) {
    status = make_d4_plan(batch_size, total_atoms, atom_offsets.data(), atomic_numbers.data(),
                          d4_plan, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
  }
  if (options.enable_periodic_embedding) {
    status = make_periodic_embedding_plan(batch_size, total_atoms, atom_offsets.data(),
                                          periodic_plan, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
  }
  if (options.enable_explicit_point_charges) {
    point_charge_offsets.resize(static_cast<std::size_t>(batch_size) + 1u);
    for (std::int64_t system = 0; system <= batch_size; ++system) {
      point_charge_offsets[static_cast<std::size_t>(system)] = system;
    }
    point_charge_positions.reserve(3u * static_cast<std::size_t>(batch_size));
    point_charge_charges.reserve(static_cast<std::size_t>(batch_size));
    point_charge_hardnesses.assign(static_cast<std::size_t>(batch_size), 1.5);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      const std::int64_t first_atom = atom_offsets[static_cast<std::size_t>(system)];
      const std::size_t xyz = 3u * static_cast<std::size_t>(first_atom);
      point_charge_positions.insert(
          point_charge_positions.end(),
          {positions[xyz] + 2.2, positions[xyz + 1u] + 0.4, positions[xyz + 2u] - 0.3});
      point_charge_charges.push_back(system % 2 == 0 ? 0.05 : -0.04);
    }
    status = make_external_point_charge_plan(basis, atomic_numbers.data(), batch_size,
                                             point_charge_offsets.data(), point_charge_plan, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
  }

  status = make_scc_driver_plan(
      wavefunction_layout, mulliken_plan, es2_plan, es3_plan, aes2_plan, eigensolver_plan,
      mixer_plan, options.enable_d4 ? &d4_plan : nullptr,
      options.enable_periodic_embedding ? &periodic_plan : nullptr, options.maximum_iterations,
      options.electronic_temperature, options.energy_tolerance, driver_plan, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  if (options.enable_periodic_embedding) {
    periodic_shifts.resize(static_cast<std::size_t>(total_atoms));
    for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
      const double magnitude = 0.004 * static_cast<double>(1 + atom % 3);
      periodic_shifts[static_cast<std::size_t>(atom)] = atom % 2 == 0 ? magnitude : -magnitude;
    }
    periodic_response_matrices.assign(
        static_cast<std::size_t>(periodic_plan.total_matrix_elements()), 0.0);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      const std::int64_t atoms = atom_offsets[static_cast<std::size_t>(system + 1)] -
                                 atom_offsets[static_cast<std::size_t>(system)];
      const std::int64_t matrix_base =
          periodic_plan.matrix_offsets()[static_cast<std::size_t>(system)];
      for (std::int64_t row = 0; row < atoms; ++row) {
        for (std::int64_t column = 0; column < atoms; ++column) {
          const std::int64_t separation = row >= column ? row - column : column - row;
          const double value = row == column ? 0.02 + 0.001 * static_cast<double>(row)
                                             : 0.002 / static_cast<double>(1 + separation);
          const std::int64_t index = matrix_base + row * atoms + column;
          periodic_response_matrices[static_cast<std::size_t>(index)] = value;
        }
      }
    }
  }

  const std::size_t matrix_elements = static_cast<std::size_t>(integrals.total_matrix_elements);
  overlap.resize(matrix_elements);
  dipole_integrals.resize(3u * matrix_elements);
  quadrupole_integrals.resize(6u * matrix_elements);
  h0.resize(matrix_elements);
  status = allocate(integral_scratch, integrals.workspace_size_bytes, "integral workspace", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(),
                                integral_scratch.data(), integral_scratch.size(), error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = evaluate_multipole_cpu(basis, integrals, positions.data(), dipole_integrals.data(),
                                  quadrupole_integrals.data(), integral_scratch.data(),
                                  integral_scratch.size(), error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = evaluate_h0_cpu(basis, integrals, h0_plan, positions.data(), coordination_numbers.data(),
                           overlap.data(), h0.data(), error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  const std::size_t es2_elements = static_cast<std::size_t>(es2_plan.total_matrix_elements());
  status = allocate(es2_storage, es2_elements * sizeof(double), "ES2 cache", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(es2_scratch_storage, es2_elements * sizeof(double), "ES2 scratch", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  es2_scratch.matrix_scratch = static_cast<double*>(es2_scratch_storage.data());
  es2_scratch.matrix_elements = es2_plan.total_matrix_elements();
  status = update_es2_geometry_cache_cpu(es2_plan, positions.data(), options.geometry_generation,
                                         static_cast<double*>(es2_storage.data()), es2_elements,
                                         es2_scratch, es2_cache, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  const std::size_t aes2_elements = static_cast<std::size_t>(aes2_plan.pair_data_elements());
  status = allocate(aes2_storage, aes2_elements * sizeof(double), "AES2 cache", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(aes2_scratch_storage, aes2_elements * sizeof(double), "AES2 scratch", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  aes2_scratch.pair_scratch = static_cast<double*>(aes2_scratch_storage.data());
  aes2_scratch.pair_elements = aes2_plan.pair_data_elements();
  status = update_aes2_geometry_cache_cpu(
      aes2_plan, positions.data(), coordination_numbers.data(), options.geometry_generation,
      static_cast<double*>(aes2_storage.data()), aes2_elements, aes2_scratch, aes2_cache, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  if (options.enable_explicit_point_charges) {
    explicit_point_charge_shell_potential.resize(
        static_cast<std::size_t>(wavefunction_layout.total_shells));
    status = evaluate_external_point_charge_potential_cpu(
        point_charge_plan, positions.data(), point_charge_positions.data(),
        point_charge_charges.data(), point_charge_hardnesses.data(),
        explicit_point_charge_shell_potential.data(), error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
  }

  status = make_internal_test_lp64_backend(&tiny_dpotrf, &tiny_dpocon, &tiny_dsyevd, &tiny_dtrsm,
                                           &tiny_dgemm, nullptr, cpu_backend, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(wavefunction_storage, wavefunction_layout.workspace_size_bytes, "wavefunction",
                    error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(overlap_cache_storage, eigensolver_plan.overlap_cache_size_bytes(),
                    "overlap cache", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(eigensolver_scratch_storage, eigensolver_plan.workspace_size_bytes(),
                    "eigensolver workspace", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(mixer_state_storage, mixer_plan.state_size_bytes(), "mixer state", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(driver_state_storage, driver_plan.state_size_bytes(), "driver state", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = allocate(driver_workspace_storage, driver_plan.workspace_size_bytes(),
                    "driver workspace", error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  status = bind_wavefunction_view(wavefunction_layout, wavefunction_storage.data(),
                                  wavefunction_storage.size(), wavefunction, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = initialize_sad_multipole_state(wavefunction_layout, wavefunction, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = bind_eigensolver_overlap_cache(eigensolver_plan, overlap_cache_storage.data(),
                                          overlap_cache_storage.size(), overlap_cache, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status =
      bind_eigensolver_workspace(eigensolver_plan, eigensolver_scratch_storage.data(),
                                 eigensolver_scratch_storage.size(), eigensolver_scratch, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = factor_overlap_cpu(eigensolver_plan, overlap.data(), options.geometry_generation,
                              cpu_backend, eigensolver_scratch, overlap_cache, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = bind_scc_mixer_state(mixer_plan, mixer_state_storage.data(), mixer_state_storage.size(),
                                mixer_state, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = bind_scc_driver_state(driver_plan, driver_state_storage.data(),
                                 driver_state_storage.size(), driver_state, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status = bind_scc_driver_workspace(driver_plan, driver_workspace_storage.data(),
                                     driver_workspace_storage.size(), driver_workspace, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  status =
      initialize_scc_driver_state_cpu(driver_plan, wavefunction, mixer_state, driver_state, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }

  geometry.h0 = h0.data();
  geometry.h0_elements = integrals.total_matrix_elements;
  geometry.integrals = {overlap.data(), dipole_integrals.data(), quadrupole_integrals.data(),
                        integrals.total_matrix_elements, mulliken_plan.identity()};
  geometry.es2_cache = es2_cache;
  geometry.aes2_cache = aes2_cache;
  geometry.geometry_generation = options.geometry_generation;

  if (options.enable_explicit_point_charges) {
    geometry.explicit_point_charge_shell_potential = explicit_point_charge_shell_potential.data();
    geometry.explicit_point_charge_shell_elements = wavefunction_layout.total_shells;
  }
  if (options.enable_d4) {
    const std::size_t pair_elements =
        static_cast<std::size_t>(d4_plan.total_pairs()) * kD4PairDataElements;
    d4_pair_data.assign(std::max<std::size_t>(pair_elements, 1u), 0.0);
    d4_coordination.assign(static_cast<std::size_t>(total_atoms), 0.0);
    status = update_d4_geometry_cache_cpu(d4_plan, positions.data(), options.geometry_generation,
                                          d4_pair_data.data(), d4_pair_data.size(),
                                          d4_coordination.data(), d4_coordination.size(),
                                          driver_workspace.d4_workspace, d4_cache, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
    geometry.d4_cache = d4_cache;
  }
  if (options.enable_periodic_embedding) {
    geometry.periodic_shifts = periodic_shifts.data();
    geometry.periodic_shift_elements = total_atoms;
    geometry.periodic_response_matrices = periodic_response_matrices.data();
    geometry.periodic_response_elements = periodic_plan.total_matrix_elements();
    geometry.periodic_embedding_generation = options.geometry_generation;
    geometry.periodic_plan_identity = periodic_plan.identity();
  }
  return GPUXTB_STATUS_SUCCESS;
}

HostSccCase::HostSccCase() noexcept = default;
HostSccCase::~HostSccCase() = default;
HostSccCase::HostSccCase(HostSccCase&&) noexcept = default;
HostSccCase& HostSccCase::operator=(HostSccCase&&) noexcept = default;

gpuxtb_status_t HostSccCase::create(const HostSccCaseOptions& options, HostSccCase& output,
                                    std::string& error) {
  try {
    auto candidate = std::make_unique<Impl>();
    candidate->options = options;
    const gpuxtb_status_t status = candidate->build(error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
    output.impl_ = std::move(candidate);
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate host SCC fixture metadata";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

bool HostSccCase::valid() const noexcept { return impl_ != nullptr; }

gpuxtb_status_t HostSccCase::run_one_iteration(std::string& error) {
  if (!valid()) {
    error = "host SCC fixture is not initialized";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return iterate_scc_driver_batch_cpu(impl_->driver_plan, impl_->geometry, impl_->cpu_backend,
                                      impl_->overlap_cache, impl_->wavefunction, impl_->mixer_state,
                                      impl_->driver_state, impl_->driver_workspace, error);
}

HostSccCheckpoint HostSccCase::checkpoint() const {
  HostSccCheckpoint result;
  if (!valid()) {
    return result;
  }
  result.wavefunction = copy_bytes(impl_->wavefunction_storage);
  result.mixer_state = copy_bytes(impl_->mixer_state_storage);
  result.driver_state = copy_bytes(impl_->driver_state_storage);
  result.driver_workspace = copy_bytes(impl_->driver_workspace_storage);
  return result;
}

gpuxtb_status_t HostSccCase::restore(const HostSccCheckpoint& checkpoint, std::string& error) {
  if (!valid()) {
    error = "host SCC fixture is not initialized";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (checkpoint.wavefunction.size() != impl_->wavefunction_storage.size() ||
      checkpoint.mixer_state.size() != impl_->mixer_state_storage.size() ||
      checkpoint.driver_state.size() != impl_->driver_state_storage.size() ||
      checkpoint.driver_workspace.size() != impl_->driver_workspace_storage.size()) {
    error = "host SCC checkpoint extents do not match this fixture";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  /* Extents are validated as one transaction before any destination changes. */
  std::memcpy(impl_->wavefunction_storage.data(), checkpoint.wavefunction.data(),
              checkpoint.wavefunction.size());
  std::memcpy(impl_->mixer_state_storage.data(), checkpoint.mixer_state.data(),
              checkpoint.mixer_state.size());
  std::memcpy(impl_->driver_state_storage.data(), checkpoint.driver_state.data(),
              checkpoint.driver_state.size());
  std::memcpy(impl_->driver_workspace_storage.data(), checkpoint.driver_workspace.data(),
              checkpoint.driver_workspace.size());
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

const HostSccCaseOptions& HostSccCase::options() const noexcept { return impl_->options; }
std::int64_t HostSccCase::batch_size() const noexcept { return impl_->batch_size; }
std::int64_t HostSccCase::total_atoms() const noexcept { return impl_->atom_offsets.back(); }
const std::vector<std::int64_t>& HostSccCase::atom_offsets() const noexcept {
  return impl_->atom_offsets;
}
const std::vector<std::int32_t>& HostSccCase::atomic_numbers() const noexcept {
  return impl_->atomic_numbers;
}
const std::vector<double>& HostSccCase::positions() const noexcept { return impl_->positions; }
const std::vector<double>& HostSccCase::molecular_charges() const noexcept {
  return impl_->molecular_charges;
}
const std::vector<std::int32_t>& HostSccCase::unpaired_electrons() const noexcept {
  return impl_->unpaired_electrons;
}
const std::vector<std::int32_t>& HostSccCase::spin_channels() const noexcept {
  return impl_->spin_channels;
}
const std::vector<double>& HostSccCase::coordination_numbers() const noexcept {
  return impl_->coordination_numbers;
}
const std::vector<std::int64_t>& HostSccCase::point_charge_offsets() const noexcept {
  return impl_->point_charge_offsets;
}
const std::vector<double>& HostSccCase::point_charge_positions() const noexcept {
  return impl_->point_charge_positions;
}
const std::vector<double>& HostSccCase::point_charge_charges() const noexcept {
  return impl_->point_charge_charges;
}
const std::vector<double>& HostSccCase::point_charge_hardnesses() const noexcept {
  return impl_->point_charge_hardnesses;
}
const std::vector<double>& HostSccCase::explicit_point_charge_shell_potential() const noexcept {
  return impl_->explicit_point_charge_shell_potential;
}
std::vector<double>& HostSccCase::explicit_point_charge_shell_potential() noexcept {
  return impl_->explicit_point_charge_shell_potential;
}
const std::vector<double>& HostSccCase::periodic_shifts() const noexcept {
  return impl_->periodic_shifts;
}
std::vector<double>& HostSccCase::periodic_shifts() noexcept { return impl_->periodic_shifts; }
const std::vector<double>& HostSccCase::periodic_response_matrices() const noexcept {
  return impl_->periodic_response_matrices;
}
std::vector<double>& HostSccCase::periodic_response_matrices() noexcept {
  return impl_->periodic_response_matrices;
}
const std::vector<double>& HostSccCase::overlap() const noexcept { return impl_->overlap; }
std::vector<double>& HostSccCase::overlap() noexcept { return impl_->overlap; }
const std::vector<double>& HostSccCase::dipole_integrals() const noexcept {
  return impl_->dipole_integrals;
}
std::vector<double>& HostSccCase::dipole_integrals() noexcept { return impl_->dipole_integrals; }
const std::vector<double>& HostSccCase::quadrupole_integrals() const noexcept {
  return impl_->quadrupole_integrals;
}
std::vector<double>& HostSccCase::quadrupole_integrals() noexcept {
  return impl_->quadrupole_integrals;
}
const std::vector<double>& HostSccCase::h0() const noexcept { return impl_->h0; }
std::vector<double>& HostSccCase::h0() noexcept { return impl_->h0; }

const BasisPlan& HostSccCase::basis_plan() const noexcept { return impl_->basis; }
const IntegralPlan& HostSccCase::integral_plan() const noexcept { return impl_->integrals; }
const H0Plan& HostSccCase::h0_plan() const noexcept { return impl_->h0_plan; }
const WavefunctionLayout& HostSccCase::wavefunction_layout() const noexcept {
  return impl_->wavefunction_layout;
}
const ES2Plan& HostSccCase::es2_plan() const noexcept { return impl_->es2_plan; }
const ES3Plan& HostSccCase::es3_plan() const noexcept { return impl_->es3_plan; }
const AES2Plan& HostSccCase::aes2_plan() const noexcept { return impl_->aes2_plan; }
const MullikenPlan& HostSccCase::mulliken_plan() const noexcept { return impl_->mulliken_plan; }
const EigensolverPlan& HostSccCase::eigensolver_plan() const noexcept {
  return impl_->eigensolver_plan;
}
const SccMixerPlan& HostSccCase::mixer_plan() const noexcept { return impl_->mixer_plan; }
const D4Plan* HostSccCase::d4_plan() const noexcept {
  return impl_->options.enable_d4 ? &impl_->d4_plan : nullptr;
}
const PeriodicEmbeddingPlan* HostSccCase::periodic_plan() const noexcept {
  return impl_->options.enable_periodic_embedding ? &impl_->periodic_plan : nullptr;
}
const ExternalPointChargePlan* HostSccCase::point_charge_plan() const noexcept {
  return impl_->options.enable_explicit_point_charges ? &impl_->point_charge_plan : nullptr;
}
const SccDriverPlan& HostSccCase::driver_plan() const noexcept { return impl_->driver_plan; }
const ES2GeometryCache& HostSccCase::es2_cache() const noexcept { return impl_->es2_cache; }
ES2GeometryCache& HostSccCase::es2_cache() noexcept { return impl_->es2_cache; }
const AES2GeometryCache& HostSccCase::aes2_cache() const noexcept { return impl_->aes2_cache; }
AES2GeometryCache& HostSccCase::aes2_cache() noexcept { return impl_->aes2_cache; }
const D4GeometryCache* HostSccCase::d4_cache() const noexcept {
  return impl_->options.enable_d4 ? &impl_->d4_cache : nullptr;
}
D4GeometryCache* HostSccCase::d4_cache() noexcept {
  return impl_->options.enable_d4 ? &impl_->d4_cache : nullptr;
}
const EigensolverOverlapCache& HostSccCase::overlap_cache() const noexcept {
  return impl_->overlap_cache;
}
const WavefunctionView& HostSccCase::wavefunction() const noexcept { return impl_->wavefunction; }
WavefunctionView& HostSccCase::wavefunction() noexcept { return impl_->wavefunction; }
const SccMixerState& HostSccCase::mixer_state() const noexcept { return impl_->mixer_state; }
SccMixerState& HostSccCase::mixer_state() noexcept { return impl_->mixer_state; }
const SccDriverState& HostSccCase::driver_state() const noexcept { return impl_->driver_state; }
SccDriverState& HostSccCase::driver_state() noexcept { return impl_->driver_state; }
const SccDriverWorkspace& HostSccCase::driver_workspace() const noexcept {
  return impl_->driver_workspace;
}
SccDriverWorkspace& HostSccCase::driver_workspace() noexcept { return impl_->driver_workspace; }
const SccDriverGeometryView& HostSccCase::geometry() const noexcept { return impl_->geometry; }
SccDriverGeometryView& HostSccCase::geometry() noexcept { return impl_->geometry; }
const CpuLinearAlgebraBackend& HostSccCase::cpu_backend() const noexcept {
  return impl_->cpu_backend;
}

}  // namespace gpuxtb::test::gfn2
