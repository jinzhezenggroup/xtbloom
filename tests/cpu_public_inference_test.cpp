#include <algorithm>
#include <array>
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

  /* ABI-v3 uniform external electric fields (one attachment per system when
   * non-empty). A zero vector is still an explicit attachment for WARM
   * identity; an empty vector leaves the interaction suffix unbound. */
  std::vector<std::array<double, 3>> fields;

  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<double> dipole_moments;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<std::int32_t> statuses;

  gpuxtb_batch_t batch{};
  gpuxtb_compute_options_t options{};
  gpuxtb_batch_result_t result{};

  /* Bind one uniform electric field attachment per system as a released
   * block_version-1 payload (32 bytes: version, reserved, three doubles). */
  void bind_fields(std::vector<std::uint8_t>* payload_storage,
                   std::vector<gpuxtb_interaction_t>* descriptor_storage) {
    if (fields.empty()) {
      return;
    }
    const std::size_t systems = static_cast<std::size_t>(batch.batch_size);
    payload_storage->clear();
    descriptor_storage->clear();
    descriptor_storage->reserve(systems);
    std::size_t cursor = 0u;
    for (std::size_t system = 0u; system < systems; ++system) {
      std::array<std::uint8_t, 32> block{};
      std::int32_t version = 1;
      std::int32_t reserved = 0;
      std::memcpy(block.data(), &version, sizeof(version));
      std::memcpy(block.data() + 4, &reserved, sizeof(reserved));
      std::memcpy(block.data() + 8, fields[system].data(), sizeof(fields[system]));
      payload_storage->insert(payload_storage->end(), block.begin(), block.end());
      gpuxtb_interaction_t interaction{};
      interaction.type = GPUXTB_INTERACTION_ELECTRIC_FIELD;
      interaction.system_index = static_cast<std::int64_t>(system);
      interaction.payload_offset = cursor;
      interaction.payload_size = sizeof(block);
      descriptor_storage->push_back(interaction);
      cursor += sizeof(block);
    }
    batch.total_interactions = static_cast<std::int64_t>(descriptor_storage->size());
    batch.interaction_descriptors = input_buffer(*descriptor_storage);
    batch.interaction_payload = input_buffer(*payload_storage);
  }

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
    dipole_moments.assign(3u * systems, -78.0);
    iterations.assign(systems, -75);
    converged.assign(systems, 76u);
    statuses.assign(systems, -77);
    result.flags = UINT32_C(0x5a5a5a5a);
    result.energies = output_buffer(energies);
    result.forces = output_buffer(forces);
    result.atomic_charges = output_buffer(atomic_charges);
    result.point_charge_forces = output_buffer(point_forces);
    result.dipole_moments = output_buffer(dipole_moments);
    result.scc_iterations = output_buffer(iterations);
    result.scc_converged = output_buffer(converged);
    result.per_system_status = output_buffer(statuses);
  }
};

template <typename T>
bool same_bytes(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  return lhs.size() == rhs.size() &&
         (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) == 0);
}

/* Snapshot every caller-owned output so rejected WARM identities can be
 * checked against the public no-publication contract byte for byte. */
struct PublicOutputImage {
  explicit PublicOutputImage(const PublicBatch& request)
      : energies(request.energies),
        forces(request.forces),
        atomic_charges(request.atomic_charges),
        point_forces(request.point_forces),
        iterations(request.iterations),
        converged(request.converged),
        statuses(request.statuses),
        flags(request.result.flags) {}

  bool matches(const PublicBatch& request) const {
    return same_bytes(energies, request.energies) && same_bytes(forces, request.forces) &&
           same_bytes(atomic_charges, request.atomic_charges) &&
           same_bytes(point_forces, request.point_forces) &&
           same_bytes(iterations, request.iterations) && same_bytes(converged, request.converged) &&
           same_bytes(statuses, request.statuses) && flags == request.result.flags;
  }

  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<std::int32_t> statuses;
  std::uint32_t flags = 0u;
};

bool warm_rejection_is_atomic(gpuxtb_context_t* context, PublicBatch& request) {
  const PublicOutputImage before(request);
  return gpuxtb_compute(context, &request.batch, &request.options, &request.result) ==
             GPUXTB_STATUS_INVALID_ARGUMENT &&
         before.matches(request);
}

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

  /* Transition the same contexts to batch one, where the parallel context
   * dispatches Mulliken phases over otherwise-idle workers. Repeated calls
   * must remain bit-identical to the one-thread public path. */
  PublicBatch serial_single = make_repeated_h2_he_batch(1u);
  PublicBatch parallel_single = make_repeated_h2_he_batch(1u);
  serial_single.bind(flags);
  parallel_single.bind(flags);
  CHECK(gpuxtb_compute(serial_context.get(), &serial_single.batch, &serial_single.options,
                       &serial_single.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_compute(parallel_context.get(), &parallel_single.batch, &parallel_single.options,
                       &parallel_single.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(parallel_single.energies == serial_single.energies);
  CHECK(parallel_single.forces == serial_single.forces);
  CHECK(parallel_single.atomic_charges == serial_single.atomic_charges);
  CHECK(parallel_single.iterations == serial_single.iterations);
  CHECK(parallel_single.converged == serial_single.converged);
  CHECK(parallel_single.statuses == serial_single.statuses);

  for (int repetition = 0; repetition < 4; ++repetition) {
    CHECK(gpuxtb_compute(parallel_context.get(), &parallel_single.batch, &parallel_single.options,
                         &parallel_single.result) == GPUXTB_STATUS_SUCCESS);
    CHECK(parallel_single.energies == serial_single.energies);
    CHECK(parallel_single.forces == serial_single.forces);
    CHECK(parallel_single.atomic_charges == serial_single.atomic_charges);
    CHECK(parallel_single.iterations == serial_single.iterations);
    CHECK(parallel_single.converged == serial_single.converged);
    CHECK(parallel_single.statuses == serial_single.statuses);
  }

  /* Returning to the multi-system outer scheduler must restore the original
   * batch result without retaining the intra-system executor assignment. */
  CHECK(gpuxtb_compute(parallel_context.get(), &parallel.batch, &parallel.options,
                       &parallel.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(parallel.energies == reference_energies);
  CHECK(parallel.forces == reference_forces);
  CHECK(parallel.atomic_charges == reference_charges);
  CHECK(parallel.iterations == reference_iterations);
  CHECK(parallel.converged == serial.converged);
  CHECK(parallel.statuses == serial.statuses);
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

  /* Strict WARM uses the retained wavefunction image without introducing a
   * separate steady-state allocation path. Warm twice before measuring so the
   * test isolates the stable checkpoint-consumption cycle. */
  request.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);

  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_release);
  const gpuxtb_status_t warm_status =
      gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result);
  allocation_test::enabled.store(false, std::memory_order_release);
  CHECK(warm_status == GPUXTB_STATUS_SUCCESS);
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

/* tblite 0.7.0 CLI energy golden for water (h2o.xyz at 0.0001 accuracy) with
 * --efield 0.0514221,0.1028442,-0.0771332 V/A, which is
 * (0.001, 0.002, -0.0015) atomic units. The tblite analytic field gradient is
 * intentionally not an oracle: it applies +E per atom rather than the
 * derivative +q_i E. Public energy finite differences below gate forces. */
int test_electric_field_golden_energy_and_scc_state() {
  ContextHandle context = make_cpu_context();
  CHECK(context != nullptr);

  PublicBatch request;
  request.atom_offsets = {0, 3};
  request.atomic_numbers = {8, 1, 1};
  const double ang2bohr = 1.8897261246257702;
  const double xyz[3][3] = {
      {0.0, 0.0, -0.2358784530},
      {0.0, 1.4270063049, 1.0081495306},
      {0.0, -1.4270063049, 1.0081495306},
  };
  request.positions.clear();
  for (int atom = 0; atom < 3; ++atom) {
    for (int component = 0; component < 3; ++component) {
      request.positions.push_back(xyz[atom][component] * ang2bohr);
    }
  }
  request.molecular_charges = {0.0};
  request.unpaired_electrons = {0};
  const std::uint32_t flags = GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES |
                              GPUXTB_COMPUTE_ATOMIC_CHARGES | GPUXTB_COMPUTE_DIPOLE_MOMENTS;
  request.bind(flags);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(request.converged[0] == 1u);

  /* No-field golden (tblite 0.7.0): total energy -4.7666769284984083 Eh and
   * force (negative gradient) O=(~0,~0,0.09395990), H=(~0,-0.05609130,
   * -0.04697995). */
  CHECK(near(request.energies[0], -4.7666769284984, 1.0e-10));
  CHECK(near(request.forces[2], 0.0939599, 1.0e-7));
  CHECK(near(request.forces[4], -0.0560913, 1.0e-7));
  CHECK(near(request.forces[5], -0.04697995, 1.0e-7));
  CHECK(near(request.atomic_charges[0], -0.412317, 1.0e-5));
  CHECK(near(request.atomic_charges[1], 0.206158, 1.0e-5));

  /* The pinned field energy remains independent oracle evidence. */
  request.fields = {{0.001, 0.002, -0.0015}};
  std::vector<std::uint8_t> payload_storage;
  std::vector<gpuxtb_interaction_t> descriptor_storage;
  request.bind_fields(&payload_storage, &descriptor_storage);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(request.converged[0] == 1u);
  CHECK(near(request.energies[0], -4.7652477392228, 1.0e-7));
  /* Mulliken charges conserve the molecular charge and respond to the field:
   * the oxygen gains charge (becomes more positive) as the field pulls
   * electron density toward it. The pinned CLI golden does not publish
   * fielded charges, so only these independent physical checks apply here. */
  CHECK(std::all_of(request.atomic_charges.begin(), request.atomic_charges.end(),
                    [](double value) { return std::isfinite(value); }));
  CHECK(std::abs(request.atomic_charges[0] + request.atomic_charges[1] +
                 request.atomic_charges[2]) < 1.0e-12);
  CHECK(request.atomic_charges[0] > -0.412317); /* more positive than no-field */
  CHECK((request.result.flags & GPUXTB_RESULT_DIPOLE_MOMENTS) != 0u);
  CHECK(std::isfinite(request.dipole_moments[0]) && std::isfinite(request.dipole_moments[1]) &&
        std::isfinite(request.dipole_moments[2]));
  return 0;
}

/* The analytic force must be the negative derivative of the reported field
 * energy. Multiple central-difference steps distinguish the correct explicit
 * +q_i E term from the nonvariational +E-per-atom tblite gradient. */
int test_electric_field_force_matches_energy_finite_difference() {
  ContextHandle context = make_cpu_context();
  CHECK(context != nullptr);

  PublicBatch request;
  request.atom_offsets = {0, 3};
  request.atomic_numbers = {8, 1, 1};
  request.positions = {0.0,          0.0,           -0.7357858611, 1.4418315287, 0.0,
                       0.3678929305, -1.4418315287, 0.0,           0.3678929305};
  request.molecular_charges = {0.0};
  request.unpaired_electrons = {0};
  request.fields = {{0.003, -0.004, 0.005}};
  std::vector<std::uint8_t> payload_storage;
  std::vector<gpuxtb_interaction_t> descriptor_storage;
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;

  request.bind(flags);
  request.options.max_scc_iterations = 500;
  request.options.charge_tolerance = 1.0e-10;
  request.options.energy_tolerance = 1.0e-12;
  request.bind_fields(&payload_storage, &descriptor_storage);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);
  const std::vector<double> analytic_forces = request.forces;
  const std::vector<double> reference_positions = request.positions;
  for (std::size_t axis = 0u; axis < 3u; ++axis) {
    double net_force = 0.0;
    for (std::size_t atom = 0u; atom < 3u; ++atom) {
      net_force += analytic_forces[3u * atom + axis];
    }
    CHECK(std::abs(net_force) < 1.0e-9);
  }

  for (const double step : {2.0e-3, 1.0e-3, 5.0e-4}) {
    for (std::size_t coordinate = 0u; coordinate < reference_positions.size(); ++coordinate) {
      request.positions = reference_positions;
      request.positions[coordinate] += step;
      request.bind(GPUXTB_COMPUTE_ENERGY);
      request.options.max_scc_iterations = 500;
      request.options.charge_tolerance = 1.0e-10;
      request.options.energy_tolerance = 1.0e-12;
      request.bind_fields(&payload_storage, &descriptor_storage);
      CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
            GPUXTB_STATUS_SUCCESS);
      CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);
      const double energy_plus = request.energies[0];

      request.positions = reference_positions;
      request.positions[coordinate] -= step;
      request.bind(GPUXTB_COMPUTE_ENERGY);
      request.options.max_scc_iterations = 500;
      request.options.charge_tolerance = 1.0e-10;
      request.options.energy_tolerance = 1.0e-12;
      request.bind_fields(&payload_storage, &descriptor_storage);
      CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
            GPUXTB_STATUS_SUCCESS);
      CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);
      const double energy_minus = request.energies[0];
      const double numerical_force = -(energy_plus - energy_minus) / (2.0 * step);
      CHECK(std::abs(numerical_force - analytic_forces[coordinate]) < 1.0e-5);
    }
  }
  return 0;
}

int test_electric_field_dipole_publication() {
  ContextHandle context = make_cpu_context();
  CHECK(context != nullptr);

  PublicBatch request;
  request.atom_offsets = {0, 2};
  request.atomic_numbers = {1, 1};
  request.positions = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0};
  request.molecular_charges = {0.0};
  request.unpaired_electrons = {0};
  request.fields = {{0.001, 0.0, 0.0}};
  std::vector<std::uint8_t> payload_storage;
  std::vector<gpuxtb_interaction_t> descriptor_storage;
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_DIPOLE_MOMENTS;
  request.bind(flags);
  request.bind_fields(&payload_storage, &descriptor_storage);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(request.converged[0] == 1u);
  CHECK((request.result.flags & GPUXTB_RESULT_DIPOLE_MOMENTS) != 0u);
  CHECK(std::all_of(request.dipole_moments.begin(), request.dipole_moments.end(),
                    [](double value) { return std::isfinite(value); }));

  /* Without the flag the outlet is not touched. */
  request.bind(GPUXTB_COMPUTE_ENERGY);
  request.bind_fields(&payload_storage, &descriptor_storage);
  std::vector<double> sentinel(request.dipole_moments.size(), -71.0);
  request.dipole_moments = sentinel;
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK((request.result.flags & GPUXTB_RESULT_DIPOLE_MOMENTS) == 0u);
  CHECK(same_bytes(sentinel, request.dipole_moments));
  return 0;
}

/* The electric field is part of the WARM identity: a WARM call whose field
 * differs from the previously converged compatible call is rejected strictly
 * before any caller output is modified. */
int test_electric_field_warm_identity_is_strict() {
  ContextHandle context = make_cpu_context();
  CHECK(context != nullptr);
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;

  PublicBatch h2;
  h2.atom_offsets = {0, 2};
  h2.atomic_numbers = {1, 1};
  h2.positions = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0};
  h2.molecular_charges = {0.0};
  h2.unpaired_electrons = {0};
  h2.fields = {{0.002, 0.0, 0.0}};
  std::vector<std::uint8_t> payload_storage;
  std::vector<gpuxtb_interaction_t> descriptor_storage;

  /* First-call WARM (with any field) has no compatible converged identity. */
  h2.bind(flags);
  h2.bind_fields(&payload_storage, &descriptor_storage);
  h2.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), h2));
  CHECK(std::strstr(gpuxtb_get_last_error(), "WARM") != nullptr);

  /* Converge FRESH with the field, then WARM with the same field succeeds. */
  h2.bind(flags);
  h2.bind_fields(&payload_storage, &descriptor_storage);
  CHECK(gpuxtb_compute(context.get(), &h2.batch, &h2.options, &h2.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(h2.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(h2.converged[0] == 1u);
  const double field_energy = h2.energies[0];

  h2.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(context.get(), &h2.batch, &h2.options, &h2.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(h2.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(h2.converged[0] == 1u);
  CHECK(field_energy == h2.energies[0]);

  /* A WARM call with a different field is a changed compute policy: strict
   * WARM rejects it without touching any output byte. */
  h2.fields = {{0.003, 0.0, 0.0}};
  h2.bind(flags);
  h2.bind_fields(&payload_storage, &descriptor_storage);
  h2.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), h2));
  CHECK(std::strstr(gpuxtb_get_last_error(), "identical") != nullptr);

  /* Attachment presence is identity even when every component is zero. A
   * field-free WARM call must not consume an explicit-zero checkpoint. */
  ContextHandle zero_context = make_cpu_context();
  CHECK(zero_context != nullptr);
  PublicBatch explicit_zero = h2;
  explicit_zero.fields = {{0.0, 0.0, 0.0}};
  explicit_zero.bind(flags);
  explicit_zero.bind_fields(&payload_storage, &descriptor_storage);
  CHECK(gpuxtb_compute(zero_context.get(), &explicit_zero.batch, &explicit_zero.options,
                       &explicit_zero.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(explicit_zero.statuses[0] == GPUXTB_STATUS_SUCCESS);

  explicit_zero.fields.clear();
  explicit_zero.bind(flags);
  explicit_zero.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(zero_context.get(), explicit_zero));

  /* The failed mismatched request leaves the explicit-zero checkpoint valid. */
  explicit_zero.fields = {{0.0, 0.0, 0.0}};
  explicit_zero.bind(flags);
  explicit_zero.bind_fields(&payload_storage, &descriptor_storage);
  explicit_zero.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(zero_context.get(), &explicit_zero.batch, &explicit_zero.options,
                       &explicit_zero.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(explicit_zero.statuses[0] == GPUXTB_STATUS_SUCCESS);
  return 0;
}

/* With a uniform field, a rigid translation changes energy by -Q(E.delta) and
 * total force by +Q E. A neutral system therefore retains its energy, atomic
 * forces, and zero net force under translation. */
int test_electric_field_translation_invariance() {
  ContextHandle context = make_cpu_context();
  CHECK(context != nullptr);

  PublicBatch request;
  request.atom_offsets = {0, 2};
  request.atomic_numbers = {1, 1};
  request.positions = {-0.70, 0.0, 0.1, 0.70, 0.0, -0.1};
  request.molecular_charges = {0.0};
  request.unpaired_electrons = {0};
  request.fields = {{0.003, -0.004, 0.005}};
  std::vector<std::uint8_t> payload_storage;
  std::vector<gpuxtb_interaction_t> descriptor_storage;
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  request.bind(flags);
  request.bind_fields(&payload_storage, &descriptor_storage);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  const double reference_energy = request.energies[0];
  const std::vector<double> reference_forces = request.forces;
  const std::vector<double> translated = {-0.2, 0.5, 1.1, 1.2, 0.5, 0.9};
  request.positions = translated;
  request.bind(flags);
  request.bind_fields(&payload_storage, &descriptor_storage);
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(near(request.energies[0], reference_energy, 1.0e-10));
  for (std::size_t component = 0u; component < reference_forces.size(); ++component) {
    CHECK(near(request.forces[component], reference_forces[component], 1.0e-9));
  }
  for (std::size_t axis = 0u; axis < 3u; ++axis) {
    CHECK(std::abs(request.forces[axis] + request.forces[3u + axis]) < 2.0e-10);
  }
  return 0;
}

int test_public_warm_start_transactions() {
  ContextHandle context = make_cpu_context();
  CHECK(context != nullptr);
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;

  PublicBatch h2;
  h2.atom_offsets = {0, 2};
  h2.atomic_numbers = {1, 1};
  h2.positions = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0};
  h2.molecular_charges = {0.0};
  h2.unpaired_electrons = {0};
  h2.bind(flags);

  /* First-call WARM has no fully converged compatible identity: it is rejected
   * strictly without changing any output byte or result flag. */
  h2.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), h2));
  CHECK(std::strstr(gpuxtb_get_last_error(), "WARM") != nullptr);

  /* A fully converged FRESH run publishes the checkpoint. */
  h2.bind(flags);
  CHECK(gpuxtb_compute(context.get(), &h2.batch, &h2.options, &h2.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(h2.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(h2.converged[0] == 1u);
  CHECK(h2.iterations[0] > 0);
  const double fresh_energy = h2.energies[0];
  const std::int32_t fresh_iterations = h2.iterations[0];
  const std::vector<double> fresh_charges = h2.atomic_charges;
  const std::vector<double> fresh_forces = h2.forces;

  /* Same-geometry WARM consumes the checkpoint and reconverges without doing
   * more work than the fresh run, with the same physics to SCC tolerance. */
  h2.bind(flags);
  h2.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(context.get(), &h2.batch, &h2.options, &h2.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(h2.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(h2.converged[0] == 1u);
  CHECK(h2.iterations[0] <= fresh_iterations);
  CHECK(std::abs(h2.energies[0] - fresh_energy) <= h2.options.energy_tolerance);
  for (std::size_t atom = 0u; atom < 2u; ++atom) {
    CHECK(near(h2.atomic_charges[atom], fresh_charges[atom], 1.0e-5));
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
      CHECK(near(h2.forces[3u * atom + axis], fresh_forces[3u * atom + axis], 1.0e-3));
    }
  }

  /* Changed-geometry WARM reuses the converged electronic state as the initial
   * SCC guess for the new coordinates. Publish a fresh reference on an
   * isolated context to prove the warm trajectory cannot exceed it. */
  ContextHandle fresh_reference = make_cpu_context();
  CHECK(fresh_reference != nullptr);
  PublicBatch perturbed = h2;
  perturbed.bind(flags);
  perturbed.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  perturbed.positions[0] += 0.002;
  perturbed.positions[3] -= 0.002;
  perturbed.bind(flags);

  PublicBatch reference = perturbed;
  reference.positions = perturbed.positions;
  reference.bind(flags);
  CHECK(gpuxtb_compute(fresh_reference.get(), &reference.batch, &reference.options,
                       &reference.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(reference.statuses[0] == GPUXTB_STATUS_SUCCESS);
  const std::int32_t reference_iterations = reference.iterations[0];

  perturbed.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(context.get(), &perturbed.batch, &perturbed.options, &perturbed.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(perturbed.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(perturbed.converged[0] == 1u);
  CHECK(perturbed.iterations[0] <= reference_iterations);
  CHECK(std::abs(perturbed.energies[0] - reference.energies[0]) <=
        perturbed.options.energy_tolerance);
  for (std::size_t atom = 0u; atom < 2u; ++atom) {
    CHECK(near(perturbed.atomic_charges[atom], reference.atomic_charges[atom], 1.0e-5));
  }

  /* Every CUDA compute-options key field is also part of the CPU identity.
   * Reject each individually and retain the old H2 checkpoint throughout. */
  PublicBatch changed_flags = h2;
  changed_flags.bind(GPUXTB_COMPUTE_ENERGY);
  changed_flags.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), changed_flags));

  PublicBatch changed_charge = h2;
  changed_charge.molecular_charges[0] = 2.0;
  changed_charge.bind(flags);
  changed_charge.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), changed_charge));

  PublicBatch changed_unpaired = h2;
  changed_unpaired.unpaired_electrons[0] = 2;
  changed_unpaired.bind(flags);
  changed_unpaired.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), changed_unpaired));

  PublicBatch changed_spin = h2;
  changed_spin.spin_channels = {2};
  changed_spin.bind(flags);
  changed_spin.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), changed_spin));

  PublicBatch changed_iterations = h2;
  changed_iterations.bind(flags);
  ++changed_iterations.options.max_scc_iterations;
  changed_iterations.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), changed_iterations));

  PublicBatch changed_charge_tolerance = h2;
  changed_charge_tolerance.bind(flags);
  changed_charge_tolerance.options.charge_tolerance =
      std::nextafter(changed_charge_tolerance.options.charge_tolerance, 1.0);
  changed_charge_tolerance.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), changed_charge_tolerance));

  PublicBatch changed_energy_tolerance = h2;
  changed_energy_tolerance.bind(flags);
  changed_energy_tolerance.options.energy_tolerance =
      std::nextafter(changed_energy_tolerance.options.energy_tolerance, 1.0);
  changed_energy_tolerance.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), changed_energy_tolerance));

  PublicBatch changed_temperature = h2;
  changed_temperature.bind(flags);
  changed_temperature.options.electronic_temperature = 0.02;
  changed_temperature.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), changed_temperature));

  h2.bind(flags);
  h2.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(context.get(), &h2.batch, &h2.options, &h2.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(h2.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(std::abs(h2.energies[0] - fresh_energy) <= h2.options.energy_tolerance);

  /* Candidate construction is transactional. A FRESH unsupported topology
   * fails while building ensure_systems, but the prior H2 systems/checkpoint
   * remain available to the next compatible strict WARM request. */
  PublicBatch unsupported = h2;
  unsupported.atomic_numbers[0] = 0;
  unsupported.bind(flags);
  const PublicOutputImage unsupported_before(unsupported);
  CHECK(gpuxtb_compute(context.get(), &unsupported.batch, &unsupported.options,
                       &unsupported.result) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::strstr(gpuxtb_get_last_error(), "unsupported atomic number") != nullptr);
  CHECK(unsupported_before.matches(unsupported));

  h2.bind(flags);
  h2.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(context.get(), &h2.batch, &h2.options, &h2.result) == GPUXTB_STATUS_SUCCESS);

  /* A changed topology is rejected by WARM but can establish its own identity
   * through FRESH. The subsequent WARM proves changed-topology recovery. */
  PublicBatch methane;
  methane.atom_offsets = {0, 5};
  methane.atomic_numbers = {6, 1, 1, 1, 1};
  methane.positions = {0.0,   0.0,   0.0,  1.09,  1.09,  1.09,  1.09, -1.09,
                       -1.09, -1.09, 1.09, -1.09, -1.09, -1.09, 1.09};
  methane.molecular_charges = {0.0};
  methane.unpaired_electrons = {0};
  methane.bind(flags);
  methane.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(context.get(), methane));

  methane.bind(flags);
  CHECK(gpuxtb_compute(context.get(), &methane.batch, &methane.options, &methane.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(methane.statuses[0] == GPUXTB_STATUS_SUCCESS);
  methane.bind(flags);
  methane.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(context.get(), &methane.batch, &methane.options, &methane.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(methane.statuses[0] == GPUXTB_STATUS_SUCCESS);

  /* A FRESH call that does not converge (one SCC iteration) must not advertise
   * a checkpoint; the next strict WARM is rejected atomically. */
  ContextHandle nonconverged_context = make_cpu_context();
  CHECK(nonconverged_context != nullptr);
  PublicBatch forced = h2;
  forced.bind(flags);
  forced.options.max_scc_iterations = 1;
  CHECK(gpuxtb_compute(nonconverged_context.get(), &forced.batch, &forced.options,
                       &forced.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(forced.statuses[0] == GPUXTB_STATUS_SCC_NOT_CONVERGED);
  CHECK(forced.converged[0] == 0u);
  CHECK(std::isnan(forced.energies[0]));
  forced.bind(flags);
  forced.options.max_scc_iterations = 1;
  forced.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(nonconverged_context.get(), forced));

  /* Ragged restricted/unrestricted systems share the same whole-batch WARM
   * transaction and retain independent per-system electronic checkpoints. */
  ContextHandle mixed_context = make_cpu_context(2);
  CHECK(mixed_context != nullptr);
  PublicBatch mixed;
  mixed.atom_offsets = {0, 2, 4};
  mixed.atomic_numbers = {1, 1, 8, 1};
  mixed.positions = {-0.71, 0.0, 0.0, 0.71, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8, 0.0, 0.0};
  mixed.molecular_charges = {0.0, 0.0};
  mixed.unpaired_electrons = {0, 1};
  mixed.spin_channels = {1, 2};
  mixed.bind(flags);
  CHECK(gpuxtb_compute(mixed_context.get(), &mixed.batch, &mixed.options, &mixed.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(mixed.statuses ==
        std::vector<std::int32_t>({GPUXTB_STATUS_SUCCESS, GPUXTB_STATUS_SUCCESS}));
  mixed.bind(flags);
  mixed.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(mixed_context.get(), &mixed.batch, &mixed.options, &mixed.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(mixed.statuses ==
        std::vector<std::int32_t>({GPUXTB_STATUS_SUCCESS, GPUXTB_STATUS_SUCCESS}));

  /* Point-charge values/coordinates and periodic operators are numerical
   * inputs, while their ragged/enablement structure is identity. */
  ContextHandle embedded_context = make_cpu_context();
  CHECK(embedded_context != nullptr);
  PublicBatch embedded;
  embedded.atom_offsets = {0, 2};
  embedded.atomic_numbers = {1, 1};
  embedded.positions = {-0.71, 0.0, 0.0, 0.71, 0.0, 0.0};
  embedded.molecular_charges = {0.0};
  embedded.unpaired_electrons = {0};
  embedded.point_offsets = {0, 1};
  embedded.point_positions = {0.2, 1.8, -0.7};
  embedded.point_values = {0.25};
  embedded.point_gammas = {0.82};
  embedded.periodic_shifts = {0.003, -0.002};
  embedded.response_offsets = {0, 4};
  embedded.response_matrix = {0.02, 0.001, 0.001, 0.018};
  const std::uint32_t embedded_flags = flags | GPUXTB_COMPUTE_POINT_CHARGE_FORCES;
  embedded.bind(embedded_flags);
  CHECK(gpuxtb_compute(embedded_context.get(), &embedded.batch, &embedded.options,
                       &embedded.result) == GPUXTB_STATUS_SUCCESS);
  embedded.point_positions[1] += 0.01;
  embedded.point_values[0] -= 0.002;
  embedded.periodic_shifts[0] += 0.0001;
  embedded.response_matrix[0] += 0.0002;
  embedded.bind(embedded_flags);
  embedded.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(embedded_context.get(), &embedded.batch, &embedded.options,
                       &embedded.result) == GPUXTB_STATUS_SUCCESS);
  CHECK((embedded.result.flags & GPUXTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES) != 0u);

  PublicBatch changed_points = embedded;
  changed_points.point_offsets = {0, 2};
  changed_points.point_positions.insert(changed_points.point_positions.end(), {1.2, -0.4, 0.3});
  changed_points.point_values.push_back(-0.1);
  changed_points.point_gammas.push_back(0.9);
  changed_points.bind(embedded_flags);
  changed_points.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(embedded_context.get(), changed_points));

  PublicBatch disabled_periodic = embedded;
  disabled_periodic.periodic_shifts.clear();
  disabled_periodic.response_offsets.clear();
  disabled_periodic.response_matrix.clear();
  disabled_periodic.bind(embedded_flags);
  disabled_periodic.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(embedded_context.get(), disabled_periodic));

  embedded.bind(embedded_flags);
  embedded.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(embedded_context.get(), &embedded.batch, &embedded.options,
                       &embedded.result) == GPUXTB_STATUS_SUCCESS);

  /* One data-level peer failure revokes whole-batch readiness even though the
   * successful peer remains independently published. */
  ContextHandle peer_context = make_cpu_context(2);
  CHECK(peer_context != nullptr);
  PublicBatch peers;
  peers.atom_offsets = {0, 2, 4};
  peers.atomic_numbers = {1, 1, 1, 1};
  peers.positions = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0, 3.30, 0.0, 0.0, 4.70, 0.0, 0.0};
  peers.molecular_charges = {0.0, 0.0};
  peers.unpaired_electrons = {0, 0};
  peers.bind(flags);
  CHECK(gpuxtb_compute(peer_context.get(), &peers.batch, &peers.options, &peers.result) ==
        GPUXTB_STATUS_SUCCESS);
  peers.positions[9] = peers.positions[6] + 1.1e-6;
  peers.bind(flags);
  peers.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(gpuxtb_compute(peer_context.get(), &peers.batch, &peers.options, &peers.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(peers.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(peers.statuses[1] == GPUXTB_STATUS_EIGENSOLVER_FAILED);
  peers.positions[9] = 4.70;
  peers.bind(flags);
  peers.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(warm_rejection_is_atomic(peer_context.get(), peers));
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

int test_degenerate_occupation_representability_is_publicly_successful() {
  ContextHandle context = make_cpu_context(1);
  CHECK(context != nullptr);

  PublicBatch request;
  request.atom_offsets = {0, 2, 5};
  request.atomic_numbers = {1, 1, 1, 1, 1};
  request.positions = {
      -0.70, 0.0, 0.0, 0.70, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e20, 0.0, 0.0, 2.0e20, 0.0, 0.0,
  };
  /* The fractional charge gives the three-atom system nextafter(6, 0)
   * electrons, hence nextafter(3, 0) electrons in each restricted spin
   * channel. At 1e20 bohr separation all three one-center Hamiltonian blocks
   * are bitwise identical while inter-center interactions safely tend to
   * zero, producing the public form of issue #31's representability corner. */
  request.molecular_charges = {0.0, 3.0 - 2.0 * std::nextafter(3.0, 0.0)};
  request.unpaired_electrons = {0, 0};
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  request.bind(flags);
  CHECK(request.options.electronic_temperature > 0.0);
  CHECK(0.5 * (3.0 - request.molecular_charges[1]) == std::nextafter(3.0, 0.0));
  CHECK(gpuxtb_compute(context.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);

  /* The ordinary H2 peer proves that the unusual fractional-charge system
   * publishes independently without disturbing another batch member. */
  CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(request.converged[0] == 1u);
  CHECK(std::isfinite(request.energies[0]));
  CHECK(request.statuses[1] == GPUXTB_STATUS_SUCCESS);
  CHECK(request.converged[1] == 1u);
  CHECK(request.iterations[1] > 0 && request.iterations[1] <= request.options.max_scc_iterations);
  CHECK(near(request.energies[1], -1.8322400836158348, 2.0e-12));
  for (std::size_t atom = 2u; atom < 5u; ++atom) {
    CHECK(request.atomic_charges[atom] == -1.0);
  }
  CHECK(std::all_of(request.forces.begin(), request.forces.end(),
                    [](double value) { return std::isfinite(value); }));
  return 0;
}

struct PlanDeleter {
  void operator()(gpuxtb_plan_t* plan) const noexcept { gpuxtb_plan_destroy(plan); }
};

using PlanHandle = std::unique_ptr<gpuxtb_plan_t, PlanDeleter>;

int test_plan_creation_model_and_abi_prefix_contracts() {
  const std::uint32_t flags = GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES;
  ContextHandle context = make_cpu_context(1);
  CHECK(context != nullptr);

  PublicBatch request = make_repeated_h2_he_batch(2u);
  request.bind(flags);

  /* This allocation contains only the ABI-v1 prefix. Plan normalization must
   * not copy/read the ABI-v2 suffix that an older caller does not own. */
  void* short_storage = ::operator new(GPUXTB_COMPUTE_OPTIONS_V1_SIZE);
  auto* short_options = static_cast<gpuxtb_compute_options_t*>(short_storage);
  CHECK(gpuxtb_compute_options_init(short_options, GPUXTB_COMPUTE_OPTIONS_V1_SIZE) ==
        GPUXTB_STATUS_SUCCESS);
  short_options->flags = flags;
  gpuxtb_plan_t* raw_short_plan = nullptr;
  CHECK(gpuxtb_plan_create(context.get(), &request.batch, short_options, &raw_short_plan) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(raw_short_plan != nullptr);
  gpuxtb_plan_destroy(raw_short_plan);
  ::operator delete(short_storage);

  /* GFN1 is a reserved ABI value. Plan setup rejects it consistently on CPU
   * before it creates a GFN2 cache that would fail only at execution time. */
  gpuxtb_compute_options_t gfn1 = request.options;
  gfn1.model = GPUXTB_MODEL_GFN1_XTB;
  gpuxtb_plan_t* raw_gfn1_plan = reinterpret_cast<gpuxtb_plan_t*>(UINTPTR_MAX);
  CHECK(gpuxtb_plan_create(context.get(), &request.batch, &gfn1, &raw_gfn1_plan) ==
        GPUXTB_STATUS_NOT_SUPPORTED);
  CHECK(raw_gfn1_plan == nullptr);
  return 0;
}

int test_plan_create_query_workspace_and_reuse() {
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  ContextHandle context = make_cpu_context(4);
  CHECK(context != nullptr);

  PublicBatch request = make_repeated_h2_he_batch(8u);
  request.bind(flags);
  gpuxtb_plan_t* raw_plan = nullptr;
  CHECK(gpuxtb_plan_create(context.get(), &request.batch, &request.options, &raw_plan) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(raw_plan != nullptr);
  CHECK(gpuxtb_context_get_backend(context.get()) == GPUXTB_BACKEND_CPU);
  PlanHandle plan(raw_plan);

  /* Workspace query: CPU host workspace with the requested properties must be
   * nonzero and well-aligned; a CPU plan has no device workspace. */
  gpuxtb_workspace_query_t query{};
  CHECK(gpuxtb_workspace_query_init(&query, sizeof(query)) == GPUXTB_STATUS_SUCCESS);
  query.compute_flags = flags;
  CHECK(gpuxtb_plan_query_workspace(plan.get(), &query) == GPUXTB_STATUS_SUCCESS);
  CHECK(query.host_required_bytes > 0u);
  CHECK(query.host_required_alignment >= 8u);
  CHECK(query.device_required_bytes == 0u);
  CHECK(query.device_required_alignment == 1u);

  gpuxtb_workspace_query_t invalid_query = query;
  invalid_query.compute_flags = 0u;
  invalid_query.host_required_bytes = UINT64_C(0x1122334455667788);
  CHECK(gpuxtb_plan_query_workspace(plan.get(), &invalid_query) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(invalid_query.host_required_bytes == UINT64_C(0x1122334455667788));
  invalid_query = query;
  invalid_query.reserved = 1u;
  CHECK(gpuxtb_plan_query_workspace(plan.get(), &invalid_query) == GPUXTB_STATUS_INVALID_ARGUMENT);
  invalid_query = query;
  invalid_query.compute_flags = GPUXTB_COMPUTE_ENERGY;
  CHECK(gpuxtb_plan_query_workspace(plan.get(), &invalid_query) == GPUXTB_STATUS_INVALID_ARGUMENT);

  /* Requested properties change the reported host workspace: forces and
   * charges keep additional output staging alive. */
  PublicBatch energy_request = request;
  energy_request.bind(GPUXTB_COMPUTE_ENERGY);
  gpuxtb_plan_t* raw_energy_plan = nullptr;
  CHECK(gpuxtb_plan_create(context.get(), &energy_request.batch, &energy_request.options,
                           &raw_energy_plan) == GPUXTB_STATUS_SUCCESS);
  PlanHandle energy_plan(raw_energy_plan);
  gpuxtb_workspace_query_t energy_query{};
  CHECK(gpuxtb_workspace_query_init(&energy_query, sizeof(energy_query)) == GPUXTB_STATUS_SUCCESS);
  energy_query.compute_flags = GPUXTB_COMPUTE_ENERGY;
  CHECK(gpuxtb_plan_query_workspace(energy_plan.get(), &energy_query) == GPUXTB_STATUS_SUCCESS);
  CHECK(energy_query.host_required_bytes > 0u);
  CHECK(query.host_required_bytes > energy_query.host_required_bytes);

  /* CPU staging canonicalizes a missing zero-point-charge offset vector to
   * the same all-zero cache image as an explicit vector. The latter plan owns
   * one additional topology snapshot, which must still appear in the public
   * aggregate even though the execution-cache reservation is otherwise equal. */
  PublicBatch explicit_zero_point_offsets = request;
  explicit_zero_point_offsets.point_offsets.assign(
      static_cast<std::size_t>(explicit_zero_point_offsets.batch.batch_size) + 1u, 0);
  explicit_zero_point_offsets.bind(flags);
  gpuxtb_plan_t* raw_explicit_offsets_plan = nullptr;
  CHECK(gpuxtb_plan_create(context.get(), &explicit_zero_point_offsets.batch,
                           &explicit_zero_point_offsets.options,
                           &raw_explicit_offsets_plan) == GPUXTB_STATUS_SUCCESS);
  PlanHandle explicit_offsets_plan(raw_explicit_offsets_plan);
  gpuxtb_workspace_query_t explicit_offsets_query{};
  CHECK(gpuxtb_workspace_query_init(&explicit_offsets_query, sizeof(explicit_offsets_query)) ==
        GPUXTB_STATUS_SUCCESS);
  explicit_offsets_query.compute_flags = flags;
  CHECK(gpuxtb_plan_query_workspace(explicit_offsets_plan.get(), &explicit_offsets_query) ==
        GPUXTB_STATUS_SUCCESS);
  const std::uint64_t point_offset_snapshot_bytes =
      explicit_zero_point_offsets.point_offsets.size() * sizeof(std::int64_t);
  CHECK(explicit_offsets_query.host_required_bytes >=
        query.host_required_bytes + point_offset_snapshot_bytes);

  /* Creating a second plan on the same context is independent. */
  gpuxtb_plan_t* second_plan = nullptr;
  CHECK(gpuxtb_plan_create(context.get(), &request.batch, &request.options, &second_plan) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(second_plan != nullptr);
  gpuxtb_plan_destroy(second_plan);

  /* Plan compute reproduces the exact convenience-path results. */
  PublicBatch via_compute = request;
  via_compute.bind(flags);
  CHECK(gpuxtb_compute(context.get(), &via_compute.batch, &via_compute.options,
                       &via_compute.result) == GPUXTB_STATUS_SUCCESS);

  request.bind(flags);
  CHECK(gpuxtb_plan_compute(plan.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses == via_compute.statuses);
  CHECK(request.converged == via_compute.converged);
  CHECK(request.iterations == via_compute.iterations);
  CHECK(request.energies == via_compute.energies);
  CHECK(request.forces == via_compute.forces);
  CHECK(request.atomic_charges == via_compute.atomic_charges);
  CHECK(request.result.flags == via_compute.result.flags);
  return 0;
}

int test_plan_fixed_topology_zero_steady_state_allocations() {
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  ContextHandle context = make_cpu_context(1);
  CHECK(context != nullptr);

  PublicBatch request = make_repeated_h2_he_batch(4u);
  request.bind(flags);
  gpuxtb_plan_t* raw_plan = nullptr;
  CHECK(gpuxtb_plan_create(context.get(), &request.batch, &request.options, &raw_plan) ==
        GPUXTB_STATUS_SUCCESS);
  PlanHandle plan(raw_plan);

  PublicBatch other;
  other.atom_offsets = {0, 5};
  other.atomic_numbers = {6, 1, 1, 1, 1};
  other.positions = {0.0,   0.0,   0.0,  1.09,  1.09,  1.09,  1.09, -1.09,
                     -1.09, -1.09, 1.09, -1.09, -1.09, -1.09, 1.09};
  other.molecular_charges = {0.0};
  other.unpaired_electrons = {0};
  other.bind(flags);
  gpuxtb_plan_t* raw_other_plan = nullptr;
  CHECK(gpuxtb_plan_create(context.get(), &other.batch, &other.options, &raw_other_plan) ==
        GPUXTB_STATUS_SUCCESS);
  PlanHandle other_plan(raw_other_plan);

  /* Plan creation pre-warms every plan-owned vector. The first call, already
   * using changed geometry, must therefore perform no host allocations. */
  request.positions[0] -= 0.011;
  request.positions[3] += 0.011;
  request.bind(flags);

  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_release);
  const gpuxtb_status_t status =
      gpuxtb_plan_compute(plan.get(), &request.batch, &request.options, &request.result);
  allocation_test::enabled.store(false, std::memory_order_release);
  CHECK(status == GPUXTB_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(std::isfinite(request.energies[0]));
  const std::vector<double> no_field_energies = request.energies;
  const std::vector<double> no_field_forces = request.forces;

  gpuxtb_workspace_query_t workspace_before{};
  CHECK(gpuxtb_workspace_query_init(&workspace_before, sizeof(workspace_before)) ==
        GPUXTB_STATUS_SUCCESS);
  workspace_before.compute_flags = flags;
  CHECK(gpuxtb_plan_query_workspace(plan.get(), &workspace_before) == GPUXTB_STATUS_SUCCESS);

  /* Field presence and values are numerical plan inputs. Every system owns
   * topology-sized field storage from plan creation, so adding and changing a
   * FRESH field must update in place without rebuilding or allocating. */
  request.fields.assign(static_cast<std::size_t>(request.batch.batch_size),
                        std::array<double, 3>{0.001, -0.002, 0.003});
  std::vector<std::uint8_t> payload_storage;
  std::vector<gpuxtb_interaction_t> descriptor_storage;
  request.bind(flags);
  request.bind_fields(&payload_storage, &descriptor_storage);
  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_release);
  const gpuxtb_status_t field_status =
      gpuxtb_plan_compute(plan.get(), &request.batch, &request.options, &request.result);
  allocation_test::enabled.store(false, std::memory_order_release);
  CHECK(field_status == GPUXTB_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  const std::vector<double> first_field_energies = request.energies;
  const std::vector<double> first_field_forces = request.forces;

  ContextHandle reference_context = make_cpu_context(1);
  CHECK(reference_context != nullptr);
  PublicBatch reference = request;
  std::vector<std::uint8_t> reference_payload;
  std::vector<gpuxtb_interaction_t> reference_descriptors;
  reference.bind(flags);
  reference.bind_fields(&reference_payload, &reference_descriptors);
  CHECK(gpuxtb_compute(reference_context.get(), &reference.batch, &reference.options,
                       &reference.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(first_field_energies == reference.energies);
  CHECK(first_field_forces == reference.forces);

  for (std::array<double, 3>& field : request.fields) {
    field = {-0.002, 0.0015, 0.0005};
  }
  request.bind(flags);
  request.bind_fields(&payload_storage, &descriptor_storage);
  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_release);
  const gpuxtb_status_t changed_field_status =
      gpuxtb_plan_compute(plan.get(), &request.batch, &request.options, &request.result);
  allocation_test::enabled.store(false, std::memory_order_release);
  CHECK(changed_field_status == GPUXTB_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  CHECK(request.energies != first_field_energies);

  reference.fields = request.fields;
  reference.bind(flags);
  reference.bind_fields(&reference_payload, &reference_descriptors);
  CHECK(gpuxtb_compute(reference_context.get(), &reference.batch, &reference.options,
                       &reference.result) == GPUXTB_STATUS_SUCCESS);
  CHECK(request.energies == reference.energies);
  CHECK(request.forces == reference.forces);

  /* Detaching the field is the same allocation-free numerical update and must
   * restore the field-free result for the unchanged geometry. */
  request.fields.clear();
  request.bind(flags);
  allocation_test::count.store(0u, std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_release);
  const gpuxtb_status_t detached_field_status =
      gpuxtb_plan_compute(plan.get(), &request.batch, &request.options, &request.result);
  allocation_test::enabled.store(false, std::memory_order_release);
  CHECK(detached_field_status == GPUXTB_STATUS_SUCCESS);
  CHECK(allocation_test::count.load(std::memory_order_relaxed) == 0u);
  CHECK(request.energies == no_field_energies);
  CHECK(request.forces == no_field_forces);

  gpuxtb_workspace_query_t workspace_after{};
  CHECK(gpuxtb_workspace_query_init(&workspace_after, sizeof(workspace_after)) ==
        GPUXTB_STATUS_SUCCESS);
  workspace_after.compute_flags = flags;
  CHECK(gpuxtb_plan_query_workspace(plan.get(), &workspace_after) == GPUXTB_STATUS_SUCCESS);
  CHECK(workspace_after.host_required_bytes == workspace_before.host_required_bytes);
  return 0;
}

int test_plan_topology_mismatch_fails_before_output_mutation() {
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  ContextHandle context = make_cpu_context(1);
  CHECK(context != nullptr);

  PublicBatch request = make_repeated_h2_he_batch(2u);
  request.bind(flags);
  gpuxtb_plan_t* null_batch_plan = reinterpret_cast<gpuxtb_plan_t*>(UINTPTR_MAX);
  CHECK(gpuxtb_plan_create(context.get(), nullptr, &request.options, &null_batch_plan) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(null_batch_plan == nullptr);
  gpuxtb_plan_t* null_options_plan = reinterpret_cast<gpuxtb_plan_t*>(UINTPTR_MAX);
  CHECK(gpuxtb_plan_create(context.get(), &request.batch, nullptr, &null_options_plan) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(null_options_plan == nullptr);
  gpuxtb_plan_t* raw_plan = nullptr;
  CHECK(gpuxtb_plan_create(context.get(), &request.batch, &request.options, &raw_plan) ==
        GPUXTB_STATUS_SUCCESS);
  PlanHandle plan(raw_plan);

  /* A different topology (methane) against the H2/He plan is a corrupted
   * request: it must fail before any caller output is modified. */
  PublicBatch other;
  other.atom_offsets = {0, 5};
  other.atomic_numbers = {6, 1, 1, 1, 1};
  other.positions = {0.0,   0.0,   0.0,  1.09,  1.09,  1.09,  1.09, -1.09,
                     -1.09, -1.09, 1.09, -1.09, -1.09, -1.09, 1.09};
  other.molecular_charges = {0.0};
  other.unpaired_electrons = {0};
  other.bind(flags);
  const std::vector<double> energies_before = other.energies;
  const std::vector<double> forces_before = other.forces;
  const std::vector<double> charges_before = other.atomic_charges;
  const std::vector<std::int32_t> iterations_before = other.iterations;
  const std::vector<std::uint8_t> converged_before = other.converged;
  const std::vector<std::int32_t> statuses_before = other.statuses;
  const std::uint32_t flags_before = other.result.flags;
  CHECK(gpuxtb_plan_compute(plan.get(), &other.batch, &other.options, &other.result) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::strstr(gpuxtb_get_last_error(), "topology") != nullptr);
  CHECK(other.energies == energies_before);
  CHECK(other.forces == forces_before);
  CHECK(other.atomic_charges == charges_before);
  CHECK(other.iterations == iterations_before);
  CHECK(other.converged == converged_before);
  CHECK(other.statuses == statuses_before);
  CHECK(other.result.flags == flags_before);

  /* The correct topology still computes after the failed attempt. */
  request.bind(flags);
  CHECK(gpuxtb_plan_compute(plan.get(), &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_SUCCESS);
  CHECK(request.statuses[0] == GPUXTB_STATUS_SUCCESS);

  /* A null plan handle is rejected without touching outputs. */
  request.result.flags = UINT32_C(0x12345678);
  CHECK(gpuxtb_plan_compute(nullptr, &request.batch, &request.options, &request.result) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(request.result.flags == UINT32_C(0x12345678));
  CHECK(gpuxtb_plan_compute(plan.get(), nullptr, &request.options, &request.result) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(request.result.flags == UINT32_C(0x12345678));
  CHECK(gpuxtb_plan_compute(plan.get(), &request.batch, nullptr, &request.result) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(request.result.flags == UINT32_C(0x12345678));
  CHECK(gpuxtb_plan_compute(plan.get(), &request.batch, &request.options, nullptr) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);

  PublicBatch mismatched_policy = request;
  mismatched_policy.bind(GPUXTB_COMPUTE_ENERGY);
  mismatched_policy.result.flags = UINT32_C(0x87654321);
  CHECK(gpuxtb_plan_compute(plan.get(), &mismatched_policy.batch, &mismatched_policy.options,
                            &mismatched_policy.result) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(mismatched_policy.result.flags == UINT32_C(0x87654321));
  return 0;
}

int test_plan_multi_threaded_reuse() {
  const std::uint32_t flags =
      GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;
  ContextHandle context = make_cpu_context(4);
  CHECK(context != nullptr);

  PublicBatch request = make_repeated_h2_he_batch(4u);
  request.bind(flags);
  gpuxtb_plan_t* raw_plan = nullptr;
  CHECK(gpuxtb_plan_create(context.get(), &request.batch, &request.options, &raw_plan) ==
        GPUXTB_STATUS_SUCCESS);
  PlanHandle plan(raw_plan);

  const std::size_t workers = 4u;
  std::vector<gpuxtb_status_t> statuses(workers, GPUXTB_STATUS_INTERNAL_ERROR);
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (std::size_t worker = 0u; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      PublicBatch batch(request);
      batch.positions[1] += 0.0001 * static_cast<double>(worker);
      batch.bind(flags);
      statuses[worker] =
          gpuxtb_plan_compute(plan.get(), &batch.batch, &batch.options, &batch.result);
      if (statuses[worker] != GPUXTB_STATUS_SUCCESS) {
        std::cerr << "worker " << worker << " failed: " << gpuxtb_get_last_error() << '\n';
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  for (gpuxtb_status_t status : statuses) {
    if (status != GPUXTB_STATUS_SUCCESS) {
      std::cerr << "plan multi-threaded worker failed with " << status << ": "
                << gpuxtb_get_last_error() << '\n';
    }
    CHECK(status == GPUXTB_STATUS_SUCCESS);
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
  if (const int line = test_public_warm_start_transactions(); line != 0) {
    return line;
  }
  if (const int line = test_point_charges_and_periodic_operator(); line != 0) {
    return line;
  }
  if (const int line = test_one_member_numerical_failure_isolated(); line != 0) {
    return line;
  }
  if (const int line = test_degenerate_occupation_representability_is_publicly_successful();
      line != 0) {
    return line;
  }
  if (const int line = test_plan_creation_model_and_abi_prefix_contracts(); line != 0) {
    return line;
  }
  if (const int line = test_plan_create_query_workspace_and_reuse(); line != 0) {
    return line;
  }
  if (const int line = test_plan_fixed_topology_zero_steady_state_allocations(); line != 0) {
    return line;
  }
  if (const int line = test_plan_topology_mismatch_fails_before_output_mutation(); line != 0) {
    return line;
  }
  if (const int line = test_plan_multi_threaded_reuse(); line != 0) {
    return line;
  }
  if (const int line = test_electric_field_golden_energy_and_scc_state(); line != 0) {
    return line;
  }
  if (const int line = test_electric_field_force_matches_energy_finite_difference(); line != 0) {
    return line;
  }
  if (const int line = test_electric_field_dipole_publication(); line != 0) {
    return line;
  }
  if (const int line = test_electric_field_warm_identity_is_strict(); line != 0) {
    return line;
  }
  if (const int line = test_electric_field_translation_invariance(); line != 0) {
    return line;
  }
  return 0;
}
