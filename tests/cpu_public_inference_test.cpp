#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "gpuxtb/gpuxtb.h"

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      std::cerr << "CHECK failed at line " << __LINE__ << ": " << #condition \
                << "; last error: " << gpuxtb_get_last_error() << '\n';      \
      return __LINE__;                                                       \
    }                                                                        \
  } while (false)

namespace {

template <typename T>
gpuxtb_const_buffer_t input_buffer(const std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), GPUXTB_MEMORY_HOST,
          0};
}

template <typename T>
gpuxtb_buffer_t output_buffer(std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), GPUXTB_MEMORY_HOST,
          0};
}

bool near(double lhs, double rhs, double tolerance) {
  const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
  return std::abs(lhs - rhs) <= tolerance * scale;
}

struct ContextDeleter {
  void operator()(gpuxtb_context_t* context) const noexcept { gpuxtb_context_destroy(context); }
};

using ContextHandle = std::unique_ptr<gpuxtb_context_t, ContextDeleter>;

struct PublicBatch {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;

  std::vector<std::int64_t> point_offsets;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> periodic_shifts;
  std::vector<std::int64_t> response_offsets;
  std::vector<double> response_matrix;

  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<std::int32_t> statuses;

  gpuxtb_batch_t batch{};
  gpuxtb_compute_options_t options{};
  gpuxtb_batch_result_t result{};

  void bind(std::uint32_t flags) {
    gpuxtb_batch_init(&batch, sizeof(batch));
    gpuxtb_compute_options_init(&options, sizeof(options));
    gpuxtb_batch_result_init(&result, sizeof(result));
    options.flags = flags;

    batch.batch_size = static_cast<std::int64_t>(atom_offsets.size() - 1u);
    batch.total_atoms = static_cast<std::int64_t>(atomic_numbers.size());
    batch.total_point_charges = static_cast<std::int64_t>(point_values.size());
    batch.total_charge_response_elements = static_cast<std::int64_t>(response_matrix.size());
    batch.atom_offsets = input_buffer(atom_offsets);
    batch.atomic_numbers = input_buffer(atomic_numbers);
    batch.positions = input_buffer(positions);
    batch.molecular_charges = input_buffer(molecular_charges);
    batch.unpaired_electrons = input_buffer(unpaired_electrons);
    batch.point_charge_offsets = input_buffer(point_offsets);
    batch.point_charge_positions = input_buffer(point_positions);
    batch.point_charge_values = input_buffer(point_values);
    batch.point_charge_gammas = input_buffer(point_gammas);
    batch.atomic_potential_shifts = input_buffer(periodic_shifts);
    batch.charge_response_offsets = input_buffer(response_offsets);
    batch.charge_response_matrix = input_buffer(response_matrix);

    const std::size_t systems = static_cast<std::size_t>(batch.batch_size);
    energies.assign(systems, -71.0);
    forces.assign(3u * atomic_numbers.size(), -72.0);
    atomic_charges.assign(atomic_numbers.size(), -73.0);
    point_forces.assign(3u * point_values.size(), -74.0);
    iterations.assign(systems, -75);
    converged.assign(systems, 76u);
    statuses.assign(systems, -77);
    result.flags = UINT32_C(0x5a5a5a5a);
    result.energies = output_buffer(energies);
    result.forces = output_buffer(forces);
    result.atomic_charges = output_buffer(atomic_charges);
    result.point_charge_forces = output_buffer(point_forces);
    result.scc_iterations = output_buffer(iterations);
    result.scc_converged = output_buffer(converged);
    result.per_system_status = output_buffer(statuses);
  }
};

PublicBatch make_h2_he_batch() {
  PublicBatch request;
  request.atom_offsets = {0, 2, 3};
  request.atomic_numbers = {1, 1, 2};
  request.positions = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0, 7.0, 0.0, 0.0};
  request.molecular_charges = {0.0, 0.0};
  request.unpaired_electrons = {0, 0};
  return request;
}

PublicBatch slice_system(const PublicBatch& source, std::size_t system) {
  PublicBatch result;
  const std::int64_t atom_begin = source.atom_offsets[system];
  const std::int64_t atom_end = source.atom_offsets[system + 1u];
  result.atom_offsets = {0, atom_end - atom_begin};
  result.atomic_numbers.assign(source.atomic_numbers.begin() + atom_begin,
                               source.atomic_numbers.begin() + atom_end);
  result.positions.assign(source.positions.begin() + 3 * atom_begin,
                          source.positions.begin() + 3 * atom_end);
  result.molecular_charges = {source.molecular_charges[system]};
  result.unpaired_electrons = {source.unpaired_electrons[system]};
  return result;
}

ContextHandle make_cpu_context() {
  gpuxtb_context_options_t options{};
  if (gpuxtb_context_options_init(&options, sizeof(options)) != GPUXTB_STATUS_SUCCESS) {
    return {};
  }
  options.backend = GPUXTB_BACKEND_CPU;
  gpuxtb_context_t* raw_context = nullptr;
  const gpuxtb_status_t status = gpuxtb_context_create(&options, &raw_context);
  ContextHandle context(raw_context);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return {};
  }
  return context;
}

int test_batch_matches_sequential_and_reuses_context() {
  ContextHandle context = make_cpu_context();
  CHECK(context != nullptr);
  PublicBatch batch = make_h2_he_batch();
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  batch.bind(flags);
  CHECK(gpuxtb_compute(context.get(), &batch.batch, &batch.options, &batch.result) ==
        GPUXTB_STATUS_SUCCESS);
  for (std::size_t system = 0; system < 2u; ++system) {
    CHECK(batch.statuses[system] == GPUXTB_STATUS_SUCCESS);
    CHECK(batch.converged[system] == 1u);
    CHECK(batch.iterations[system] > 0);
    CHECK(std::isfinite(batch.energies[system]));
  }
  CHECK(std::all_of(batch.forces.begin(), batch.forces.end(),
                    [](double value) { return std::isfinite(value); }));
  CHECK(std::all_of(batch.atomic_charges.begin(), batch.atomic_charges.end(),
                    [](double value) { return std::isfinite(value); }));

  for (std::size_t system = 0; system < 2u; ++system) {
    PublicBatch sequential = slice_system(batch, system);
    sequential.bind(flags);
    CHECK(gpuxtb_compute(context.get(), &sequential.batch, &sequential.options,
                         &sequential.result) == GPUXTB_STATUS_SUCCESS);
    CHECK(sequential.statuses[0] == GPUXTB_STATUS_SUCCESS);
    CHECK(near(batch.energies[system], sequential.energies[0], 2.0e-12));
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1u];
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::size_t local = static_cast<std::size_t>(atom - atom_begin);
      CHECK(near(batch.atomic_charges[static_cast<std::size_t>(atom)],
                 sequential.atomic_charges[local], 2.0e-11));
      for (std::size_t axis = 0; axis < 3u; ++axis) {
        CHECK(near(batch.forces[3u * static_cast<std::size_t>(atom) + axis],
                   sequential.forces[3u * local + axis], 3.0e-9));
      }
    }
  }

  /* Re-enter the original topology with changed geometry to exercise plan reuse. */
  batch.positions[0] -= 0.01;
  batch.positions[3] += 0.01;
  batch.bind(flags);
  CHECK(gpuxtb_compute(context.get(), &batch.batch, &batch.options, &batch.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(batch.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(std::isfinite(batch.energies[0]));
  return 0;
}

int test_energy_only_and_invalid_call_transactionality() {
  ContextHandle context = make_cpu_context();
  CHECK(context != nullptr);
  PublicBatch request = make_h2_he_batch();
  request.bind(GPUXTB_COMPUTE_ENERGY);
  request.result.forces = {};
  request.result.atomic_charges = {};
  request.result.point_charge_forces = {};
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(std::all_of(request.energies.begin(), request.energies.end(),
                    [](double value) { return std::isfinite(value); }));

  const std::vector<double> energies_before = request.energies;
  const std::vector<std::int32_t> iterations_before = request.iterations;
  const std::vector<std::uint8_t> converged_before = request.converged;
  const std::vector<std::int32_t> statuses_before = request.statuses;
  const std::uint32_t flags_before = request.result.flags;
  --request.batch.positions.size_bytes;
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(request.energies == energies_before);
  CHECK(request.iterations == iterations_before);
  CHECK(request.converged == converged_before);
  CHECK(request.statuses == statuses_before);
  CHECK(request.result.flags == flags_before);

  ++request.batch.positions.size_bytes;
  request.positions[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(request.energies == energies_before);
  CHECK(request.iterations == iterations_before);
  CHECK(request.converged == converged_before);
  CHECK(request.statuses == statuses_before);
  CHECK(request.result.flags == flags_before);
  return 0;
}

int test_point_charges_and_periodic_operator() {
  ContextHandle context = make_cpu_context();
  CHECK(context != nullptr);
  PublicBatch request;
  request.atom_offsets = {0, 2};
  request.atomic_numbers = {1, 1};
  request.positions = {-0.71, 0.0, 0.0, 0.71, 0.0, 0.0};
  request.molecular_charges = {0.0};
  request.unpaired_electrons = {0};
  request.point_offsets = {0, 1};
  request.point_positions = {0.2, 1.8, -0.7};
  request.point_values = {0.25};
  request.point_gammas = {0.82};
  request.periodic_shifts = {0.003, -0.002};
  request.response_offsets = {0, 4};
  request.response_matrix = {0.02, 0.001, 0.001, 0.018};
  const std::uint32_t flags = GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES |
                              GPUXTB_COMPUTE_ATOMIC_CHARGES | GPUXTB_COMPUTE_POINT_CHARGE_FORCES;
  request.bind(flags);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(std::isfinite(request.energies[0]));
  CHECK(std::all_of(request.point_forces.begin(), request.point_forces.end(),
                    [](double value) { return std::isfinite(value); }));
  CHECK((request.result.flags & GPUXTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES) != 0u);

  request.bind(GPUXTB_COMPUTE_POINT_CHARGE_FORCES);
  request.result.energies = {};
  request.result.forces = {};
  request.result.atomic_charges = {};
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(std::all_of(request.point_forces.begin(), request.point_forces.end(),
                    [](double value) { return std::isfinite(value); }));
  return 0;
}

int test_one_member_numerical_failure_isolated() {
  ContextHandle context = make_cpu_context();
  CHECK(context != nullptr);
  PublicBatch request;
  request.atom_offsets = {0, 2, 4};
  request.atomic_numbers = {1, 1, 1, 1};
  request.positions = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1e-6, 0.0, 0.0};
  request.molecular_charges = {0.0, 0.0};
  request.unpaired_electrons = {0, 0};
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  request.bind(flags);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(request.converged[0] == 1u);
  CHECK(std::isfinite(request.energies[0]));
  CHECK(request.statuses[1] == GPUXTB_STATUS_EIGENSOLVER_FAILED ||
        request.statuses[1] == GPUXTB_STATUS_SCC_NOT_CONVERGED);
  CHECK(request.converged[1] == 0u);
  CHECK(std::isnan(request.energies[1]));
  for (std::size_t coordinate = 6u; coordinate < 12u; ++coordinate) {
    CHECK(std::isnan(request.forces[coordinate]));
  }
  for (std::size_t atom = 2u; atom < 4u; ++atom) {
    CHECK(std::isnan(request.atomic_charges[atom]));
  }
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_batch_matches_sequential_and_reuses_context(); line != 0) {
    return line;
  }
  if (const int line = test_energy_only_and_invalid_call_transactionality(); line != 0) {
    return line;
  }
  if (const int line = test_point_charges_and_periodic_operator(); line != 0) {
    return line;
  }
  if (const int line = test_one_member_numerical_failure_isolated(); line != 0) {
    return line;
  }
  return 0;
}
