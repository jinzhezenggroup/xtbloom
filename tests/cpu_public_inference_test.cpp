#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <vector>

#include "gpuxtb/gpuxtb.h"

namespace allocation_test {
std::atomic<std::size_t> count{0u};
std::atomic<bool> enabled{false};
}  // namespace allocation_test

/* Interpose C++ allocations from the public runtime so the steady-state test
 * can prove that context-owned request/result staging retains its capacity. */
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
  std::vector<std::int32_t> spin_channels;

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
    batch.spin_channels = input_buffer(spin_channels);

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

PublicBatch make_repeated_h2_he_batch(std::size_t copies) {
  PublicBatch request;
  request.atom_offsets.push_back(0);
  for (std::size_t copy = 0u; copy < copies; ++copy) {
    const double displacement = 0.001 * static_cast<double>(copy);
    request.atomic_numbers.insert(request.atomic_numbers.end(), {1, 1});
    request.positions.insert(request.positions.end(),
                             {-0.70 - displacement, 0.0, 0.0, 0.70 + displacement, 0.0, 0.0});
    request.molecular_charges.push_back(0.0);
    request.unpaired_electrons.push_back(0);
    request.atom_offsets.push_back(static_cast<std::int64_t>(request.atomic_numbers.size()));

    request.atomic_numbers.push_back(2);
    request.positions.insert(request.positions.end(),
                             {7.0 + 0.01 * static_cast<double>(copy), 0.0, 0.0});
    request.molecular_charges.push_back(0.0);
    request.unpaired_electrons.push_back(0);
    request.atom_offsets.push_back(static_cast<std::int64_t>(request.atomic_numbers.size()));
  }
  return request;
}

PublicBatch make_repeated_qmmm_batch(std::size_t systems) {
  PublicBatch request;
  request.atom_offsets.push_back(0);
  request.point_offsets.push_back(0);
  for (std::size_t system = 0u; system < systems; ++system) {
    const double displacement = 0.002 * static_cast<double>(system);
    request.atomic_numbers.insert(request.atomic_numbers.end(), {1, 1});
    request.positions.insert(request.positions.end(),
                             {-0.71 - displacement, 0.0, 0.0, 0.71 + displacement, 0.0, 0.0});
    request.molecular_charges.push_back(0.0);
    request.unpaired_electrons.push_back(0);
    request.atom_offsets.push_back(static_cast<std::int64_t>(request.atomic_numbers.size()));
    request.point_positions.insert(request.point_positions.end(), {0.2, 1.8 + displacement, -0.7});
    request.point_values.push_back(0.25 - 0.001 * static_cast<double>(system));
    request.point_gammas.push_back(0.82);
    request.point_offsets.push_back(static_cast<std::int64_t>(request.point_values.size()));
  }
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
  if (!source.spin_channels.empty()) {
    result.spin_channels = {source.spin_channels[system]};
  }
  return result;
}

ContextHandle make_cpu_context(std::int32_t cpu_threads = 0) {
  gpuxtb_context_options_t options{};
  if (gpuxtb_context_options_init(&options, sizeof(options)) != GPUXTB_STATUS_SUCCESS) {
    return {};
  }
  options.backend = GPUXTB_BACKEND_CPU;
  options.cpu_threads = cpu_threads;
  gpuxtb_context_t* raw_context = nullptr;
  const gpuxtb_status_t status = gpuxtb_context_create(&options, &raw_context);
  ContextHandle context(raw_context);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return {};
  }
  return context;
}

int test_explicit_thread_counts_are_deterministic() {
  constexpr std::int32_t kParallelThreads = 4;
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  ContextHandle serial_context = make_cpu_context(1);
  ContextHandle parallel_context = make_cpu_context(kParallelThreads);
  CHECK(serial_context != nullptr);
  CHECK(parallel_context != nullptr);

  PublicBatch serial = make_repeated_h2_he_batch(8u);
  PublicBatch parallel = make_repeated_h2_he_batch(8u);
  serial.bind(flags);
  parallel.bind(flags);
  CHECK(gpuxtb_compute(serial_context.get(), &serial.batch, &serial.options, &serial.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_compute(parallel_context.get(), &parallel.batch, &parallel.options,
                       &parallel.result) == GPUXTB_STATUS_SUCCESS);

  /* Systems have disjoint state and the outer scheduler never changes a
   * system's arithmetic order, so thread count must not change any bit. */
  CHECK(parallel.energies == serial.energies);
  CHECK(parallel.forces == serial.forces);
  CHECK(parallel.atomic_charges == serial.atomic_charges);
  CHECK(parallel.iterations == serial.iterations);
  CHECK(parallel.converged == serial.converged);
  CHECK(parallel.statuses == serial.statuses);

  const std::vector<double> reference_energies = parallel.energies;
  const std::vector<double> reference_forces = parallel.forces;
  const std::vector<double> reference_charges = parallel.atomic_charges;
  const std::vector<std::int32_t> reference_iterations = parallel.iterations;
  for (int repetition = 0; repetition < 4; ++repetition) {
    CHECK(gpuxtb_compute(parallel_context.get(), &parallel.batch, &parallel.options,
                         &parallel.result) == GPUXTB_STATUS_SUCCESS);
    CHECK(parallel.energies == reference_energies);
    CHECK(parallel.forces == reference_forces);
    CHECK(parallel.atomic_charges == reference_charges);
    CHECK(parallel.iterations == reference_iterations);
    CHECK(parallel.converged == serial.converged);
    CHECK(parallel.statuses == serial.statuses);
  }
  return 0;
}

int test_parallel_qmmm_matches_serial() {
  const std::uint32_t flags = GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES |
                              GPUXTB_COMPUTE_ATOMIC_CHARGES | GPUXTB_COMPUTE_POINT_CHARGE_FORCES;
  ContextHandle serial_context = make_cpu_context(1);
  ContextHandle parallel_context = make_cpu_context(4);
  CHECK(serial_context != nullptr);
  CHECK(parallel_context != nullptr);

  PublicBatch serial = make_repeated_qmmm_batch(8u);
  PublicBatch parallel = make_repeated_qmmm_batch(8u);
  serial.bind(flags);
  parallel.bind(flags);
  CHECK(gpuxtb_compute(serial_context.get(), &serial.batch, &serial.options, &serial.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_compute(parallel_context.get(), &parallel.batch, &parallel.options,
                       &parallel.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(parallel.energies == serial.energies);
  CHECK(parallel.forces == serial.forces);
  CHECK(parallel.atomic_charges == serial.atomic_charges);
  CHECK(parallel.point_forces == serial.point_forces);
  CHECK(parallel.iterations == serial.iterations);
  CHECK(parallel.converged == serial.converged);
  CHECK(parallel.statuses == serial.statuses);
  return 0;
}

int test_concurrent_calls_on_one_context_are_serialized_transactions() {
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  ContextHandle reference_context = make_cpu_context(1);
  ContextHandle shared_context = make_cpu_context(4);
  CHECK(reference_context != nullptr);
  CHECK(shared_context != nullptr);

  PublicBatch reference_first = make_repeated_h2_he_batch(4u);
  PublicBatch reference_second = make_repeated_h2_he_batch(4u);
  reference_second.positions[0] -= 0.013;
  reference_second.positions[3] += 0.013;
  reference_first.bind(flags);
  reference_second.bind(flags);
  CHECK(gpuxtb_compute(reference_context.get(), &reference_first.batch, &reference_first.options,
                       &reference_first.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_compute(reference_context.get(), &reference_second.batch, &reference_second.options,
                       &reference_second.result) == GPUXTB_STATUS_SUCCESS);

  PublicBatch actual_first = make_repeated_h2_he_batch(4u);
  PublicBatch actual_second = make_repeated_h2_he_batch(4u);
  actual_second.positions[0] -= 0.013;
  actual_second.positions[3] += 0.013;
  actual_first.bind(flags);
  actual_second.bind(flags);

  for (int repetition = 0; repetition < 8; ++repetition) {
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    gpuxtb_status_t first_status = GPUXTB_STATUS_INTERNAL_ERROR;
    gpuxtb_status_t second_status = GPUXTB_STATUS_INTERNAL_ERROR;
    std::thread first([&] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      first_status = gpuxtb_compute(shared_context.get(), &actual_first.batch,
                                    &actual_first.options, &actual_first.result);
    });
    std::thread second([&] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      second_status = gpuxtb_compute(shared_context.get(), &actual_second.batch,
                                     &actual_second.options, &actual_second.result);
    });
    while (ready.load(std::memory_order_acquire) != 2) {
      std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    first.join();
    second.join();

    CHECK(first_status == GPUXTB_STATUS_SUCCESS);
    CHECK(second_status == GPUXTB_STATUS_SUCCESS);
    CHECK(actual_first.energies == reference_first.energies);
    CHECK(actual_first.forces == reference_first.forces);
    CHECK(actual_first.atomic_charges == reference_first.atomic_charges);
    CHECK(actual_first.iterations == reference_first.iterations);
    CHECK(actual_first.converged == reference_first.converged);
    CHECK(actual_first.statuses == reference_first.statuses);
    CHECK(actual_first.result.flags == reference_first.result.flags);
    CHECK(actual_second.energies == reference_second.energies);
    CHECK(actual_second.forces == reference_second.forces);
    CHECK(actual_second.atomic_charges == reference_second.atomic_charges);
    CHECK(actual_second.iterations == reference_second.iterations);
    CHECK(actual_second.converged == reference_second.converged);
    CHECK(actual_second.statuses == reference_second.statuses);
    CHECK(actual_second.result.flags == reference_second.result.flags);
  }
  return 0;
}

int test_steady_state_reuses_transaction_staging() {
  ContextHandle context = make_cpu_context(4);
  CHECK(context != nullptr);
  PublicBatch request = make_repeated_qmmm_batch(8u);
  const std::uint32_t flags = GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES |
                              GPUXTB_COMPUTE_ATOMIC_CHARGES | GPUXTB_COMPUTE_POINT_CHARGE_FORCES;
  request.bind(flags);
  /* The first two calls grow every context-owned vector and warm the provider. */
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);

  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_release);
  const gpuxtb_status_t status =
      gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result);
  allocation_test::enabled.store(false, std::memory_order_release);
  CHECK(status == GPUXTB_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  return 0;
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

int test_worker_call_failure_is_atomic_and_recoverable() {
  ContextHandle context = make_cpu_context(4);
  CHECK(context != nullptr);
  PublicBatch request = make_repeated_h2_he_batch(2u);
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  /* System two is an H2 molecule. Its finite but enormous separation passes
   * descriptor validation and is rejected by a worker's integral range check
   * while peer systems may already have completed. */
  const std::size_t hostile_coordinate = 3u * 4u;
  const double original_coordinate = request.positions[hostile_coordinate];
  request.positions[hostile_coordinate] = 1.0e200;
  request.bind(flags);
  const std::vector<double> expected_energies = request.energies;
  const std::vector<double> expected_forces = request.forces;
  const std::vector<double> expected_charges = request.atomic_charges;
  const std::vector<std::int32_t> expected_iterations = request.iterations;
  const std::vector<std::uint8_t> expected_converged = request.converged;
  const std::vector<std::int32_t> expected_statuses = request.statuses;
  const std::uint32_t expected_flags = request.result.flags;
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::strstr(gpuxtb_get_last_error(), "positions are too large") != nullptr);
  CHECK(request.energies == expected_energies);
  CHECK(request.forces == expected_forces);
  CHECK(request.atomic_charges == expected_charges);
  CHECK(request.iterations == expected_iterations);
  CHECK(request.converged == expected_converged);
  CHECK(request.statuses == expected_statuses);
  CHECK(request.result.flags == expected_flags);

  request.positions[hostile_coordinate] = original_coordinate;
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(std::all_of(request.statuses.begin(), request.statuses.end(),
                    [](std::int32_t status) { return status == GPUXTB_STATUS_SUCCESS; }));
  return 0;
}

int test_public_unrestricted_energy_forces() {
  ContextHandle context = make_cpu_context();
  CHECK(context != nullptr);
  PublicBatch request;
  request.atom_offsets = {0, 1, 2};
  request.atomic_numbers = {1, 1};
  request.positions = {0.0, 0.0, 0.0, 4.0, 0.0, 0.0};
  request.molecular_charges = {1.0, 0.0};
  request.unpaired_electrons = {0, 1};
  request.spin_channels = {1, 2};
  request.bind(GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_ATOMIC_CHARGES);
  CHECK(request.batch.struct_size >= GPUXTB_BATCH_V2_SIZE);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses ==
        std::vector<std::int32_t>({GPUXTB_STATUS_SUCCESS, GPUXTB_STATUS_SUCCESS}));
  CHECK(request.converged == std::vector<std::uint8_t>({1u, 1u}));
  CHECK(std::isfinite(request.energies[0]) && std::isfinite(request.energies[1]));
  CHECK(std::abs(request.atomic_charges[0] - 1.0) < 2.0e-12);
  CHECK(std::abs(request.atomic_charges[1]) < 2.0e-12);

  request.bind(GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses ==
        std::vector<std::int32_t>({GPUXTB_STATUS_SUCCESS, GPUXTB_STATUS_SUCCESS}));
  CHECK(std::all_of(request.forces.begin(), request.forces.end(),
                    [](double value) { return std::isfinite(value); }));

  request.spin_channels[1] = 3;
  request.bind(GPUXTB_COMPUTE_ENERGY);
  request.result.flags = UINT32_C(0xa55aa55a);
  const std::vector<double> invalid_energies = request.energies;
  const std::vector<double> invalid_forces = request.forces;
  const std::vector<double> invalid_charges = request.atomic_charges;
  const std::vector<std::int32_t> invalid_iterations = request.iterations;
  const std::vector<std::uint8_t> invalid_converged = request.converged;
  const std::vector<std::int32_t> invalid_statuses = request.statuses;
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(request.energies == invalid_energies);
  CHECK(request.forces == invalid_forces);
  CHECK(request.atomic_charges == invalid_charges);
  CHECK(request.iterations == invalid_iterations);
  CHECK(request.converged == invalid_converged);
  CHECK(request.statuses == invalid_statuses);
  CHECK(request.result.flags == UINT32_C(0xa55aa55a));

  /* An ABI-v1 caller has no spin_channels suffix and remains restricted. */
  request.spin_channels.clear();
  request.unpaired_electrons = {0, 1};
  request.bind(GPUXTB_COMPUTE_ENERGY);
  request.batch.struct_size = GPUXTB_BATCH_V1_SIZE;
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  return 0;
}

int test_mixed_restricted_unrestricted_batch_matches_serial() {
  ContextHandle parallel_context = make_cpu_context(4);
  ContextHandle serial_context = make_cpu_context(1);
  CHECK(parallel_context != nullptr);
  CHECK(serial_context != nullptr);

  PublicBatch mixed;
  mixed.atom_offsets = {0, 2, 4};
  mixed.atomic_numbers = {1, 1, 8, 1};
  mixed.positions = {-0.71, 0.0, 0.0, 0.71, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8, 0.0, 0.0};
  mixed.molecular_charges = {0.0, 0.0};
  mixed.unpaired_electrons = {0, 1};
  mixed.spin_channels = {1, 2};
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  mixed.bind(flags);
  CHECK(gpuxtb_compute(parallel_context.get(), &mixed.batch, &mixed.options, &mixed.result) ==
        GPUXTB_STATUS_SUCCESS);

  for (std::size_t system = 0; system < 2u; ++system) {
    PublicBatch sequential = slice_system(mixed, system);
    sequential.bind(flags);
    CHECK(gpuxtb_compute(serial_context.get(), &sequential.batch, &sequential.options,
                         &sequential.result) == GPUXTB_STATUS_SUCCESS);
    CHECK(mixed.energies[system] == sequential.energies[0]);
    CHECK(mixed.iterations[system] == sequential.iterations[0]);
    CHECK(mixed.converged[system] == sequential.converged[0]);
    CHECK(mixed.statuses[system] == sequential.statuses[0]);
    const std::size_t atom_begin = static_cast<std::size_t>(mixed.atom_offsets[system]);
    const std::size_t atom_end = static_cast<std::size_t>(mixed.atom_offsets[system + 1u]);
    CHECK(std::equal(sequential.forces.begin(), sequential.forces.end(),
                     mixed.forces.begin() + 3u * atom_begin, mixed.forces.begin() + 3u * atom_end));
    CHECK(std::equal(sequential.atomic_charges.begin(), sequential.atomic_charges.end(),
                     mixed.atomic_charges.begin() + atom_begin,
                     mixed.atomic_charges.begin() + atom_end));
  }
  return 0;
}

int test_unrestricted_alias_validation_is_atomic_and_recoverable() {
  ContextHandle context = make_cpu_context(4);
  CHECK(context != nullptr);
  PublicBatch request;
  request.atom_offsets = {0, 2, 4};
  request.atomic_numbers = {1, 1, 8, 1};
  request.positions = {-0.71, 0.0, 0.0, 0.71, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8, 0.0, 0.0};
  request.molecular_charges = {0.0, 0.0};
  request.unpaired_electrons = {0, 1};
  request.spin_channels = {1, 2};
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  request.bind(flags);
  request.result.flags = UINT32_C(0x5aa55aa5);
  const std::vector<double> positions_before = request.positions;
  const std::vector<std::int32_t> spin_channels_before = request.spin_channels;
  const std::vector<double> energies_before = request.energies;
  const std::vector<double> forces_before = request.forces;
  const std::vector<double> charges_before = request.atomic_charges;
  const std::vector<std::int32_t> iterations_before = request.iterations;
  const std::vector<std::uint8_t> converged_before = request.converged;
  const std::vector<std::int32_t> statuses_before = request.statuses;

  request.result.forces = {request.positions.data(), request.positions.size() * sizeof(double),
                           GPUXTB_MEMORY_HOST, 0};
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::strstr(gpuxtb_get_last_error(), "aliases") != nullptr);
  CHECK(request.positions == positions_before);
  CHECK(request.spin_channels == spin_channels_before);
  CHECK(request.energies == energies_before);
  CHECK(request.forces == forces_before);
  CHECK(request.atomic_charges == charges_before);
  CHECK(request.iterations == iterations_before);
  CHECK(request.converged == converged_before);
  CHECK(request.statuses == statuses_before);
  CHECK(request.result.flags == UINT32_C(0x5aa55aa5));

  request.result.forces = output_buffer(request.forces);
  request.result.scc_iterations = {request.spin_channels.data(),
                                   request.spin_channels.size() * sizeof(std::int32_t),
                                   GPUXTB_MEMORY_HOST, 0};
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::strstr(gpuxtb_get_last_error(), "spin_channels aliases scc_iterations") != nullptr);
  CHECK(request.positions == positions_before);
  CHECK(request.spin_channels == spin_channels_before);
  CHECK(request.energies == energies_before);
  CHECK(request.forces == forces_before);
  CHECK(request.atomic_charges == charges_before);
  CHECK(request.iterations == iterations_before);
  CHECK(request.converged == converged_before);
  CHECK(request.statuses == statuses_before);
  CHECK(request.result.flags == UINT32_C(0x5aa55aa5));

  request.result.scc_iterations = output_buffer(request.iterations);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(std::all_of(request.statuses.begin(), request.statuses.end(),
                    [](std::int32_t status) { return status == GPUXTB_STATUS_SUCCESS; }));
  return 0;
}

int test_oh_radical_unrestricted_force_matches_energy_finite_difference() {
  ContextHandle context = make_cpu_context(1);
  CHECK(context != nullptr);

  PublicBatch request;
  request.atom_offsets = {0, 2};
  request.atomic_numbers = {8, 1};
  request.positions = {0.0, 0.0, 0.0, 1.8, 0.0, 0.0};
  request.molecular_charges = {0.0};
  request.unpaired_electrons = {1};
  request.spin_channels = {2};
  request.bind(GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses == std::vector<std::int32_t>({GPUXTB_STATUS_SUCCESS}));
  CHECK(request.converged == std::vector<std::uint8_t>({1u}));
  CHECK(std::isfinite(request.energies[0]));
  CHECK(std::all_of(request.forces.begin(), request.forces.end(),
                    [](double value) { return std::isfinite(value); }));
  CHECK(std::all_of(request.atomic_charges.begin(), request.atomic_charges.end(),
                    [](double value) { return std::isfinite(value); }));

  /* Spin polarization has no explicit coordinate derivative, but its Mulliken
   * magnetization response contributes through the overlap derivative. */
  constexpr double step = 1.0e-4;
  const double analytic_hydrogen_force = request.forces[3];
  request.positions[3] += step;
  request.bind(GPUXTB_COMPUTE_ENERGY);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  const double energy_plus = request.energies[0];
  request.positions[3] -= 2.0 * step;
  request.bind(GPUXTB_COMPUTE_ENERGY);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  const double energy_minus = request.energies[0];
  const double finite_difference_force = -(energy_plus - energy_minus) / (2.0 * step);
  CHECK(near(analytic_hydrogen_force, finite_difference_force, 2.0e-5));

  request.positions[3] += step;
  request.bind(GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  for (std::size_t axis = 0u; axis < 3u; ++axis) {
    CHECK(std::abs(request.forces[axis] + request.forces[3u + axis]) < 2.0e-10);
  }
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
  ContextHandle context = make_cpu_context(4);
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
  if (const int line = test_explicit_thread_counts_are_deterministic(); line != 0) {
    return line;
  }
  if (const int line = test_parallel_qmmm_matches_serial(); line != 0) {
    return line;
  }
  if (const int line = test_concurrent_calls_on_one_context_are_serialized_transactions();
      line != 0) {
    return line;
  }
  if (const int line = test_steady_state_reuses_transaction_staging(); line != 0) {
    return line;
  }
  if (const int line = test_batch_matches_sequential_and_reuses_context(); line != 0) {
    return line;
  }
  if (const int line = test_energy_only_and_invalid_call_transactionality(); line != 0) {
    return line;
  }
  if (const int line = test_worker_call_failure_is_atomic_and_recoverable(); line != 0) {
    return line;
  }
  if (const int line = test_public_unrestricted_energy_forces(); line != 0) {
    return line;
  }
  if (const int line = test_mixed_restricted_unrestricted_batch_matches_serial(); line != 0) {
    return line;
  }
  if (const int line = test_unrestricted_alias_validation_is_atomic_and_recoverable(); line != 0) {
    return line;
  }
  if (const int line = test_oh_radical_unrestricted_force_matches_energy_finite_difference();
      line != 0) {
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
