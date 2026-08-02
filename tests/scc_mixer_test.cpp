#include "model/gfn2/scc_mixer.hpp"

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
#include <thread>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "model/gfn2/wavefunction.hpp"

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

using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::SccMixerPlan;
using gpuxtb::detail::gfn2::SccMixerState;
using gpuxtb::detail::gfn2::SccMixerWorkspace;
using gpuxtb::detail::gfn2::WavefunctionLayout;
using gpuxtb::detail::gfn2::WavefunctionView;

class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t bytes) : bytes_(bytes) {
    data_ = std::aligned_alloc(gpuxtb::detail::gfn2::kSccMixerWorkspaceAlignment, bytes_);
    if (data_ != nullptr) {
      std::memset(data_, 0, bytes_);
    }
  }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  ~AlignedBuffer() { std::free(data_); }

  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0u;
};

struct Fixture {
  BasisPlan basis;
  WavefunctionLayout layout;
  SccMixerPlan plan;
  std::unique_ptr<AlignedBuffer> wavefunction_storage;
  std::unique_ptr<AlignedBuffer> state_storage;
  std::unique_ptr<AlignedBuffer> scratch_storage;
  WavefunctionView wavefunction;
  SccMixerState state;
  SccMixerWorkspace scratch;
};

bool near(double first, double second, double tolerance = 2.0e-12) {
  return std::abs(first - second) <= tolerance;
}

bool make_fixture(const std::vector<std::int64_t>& atom_offsets,
                  const std::vector<std::int32_t>& atomic_numbers,
                  const std::vector<double>& charges, const std::vector<std::int32_t>& unpaired,
                  const std::vector<std::int32_t>& spin_channels, std::int64_t history_size,
                  Fixture& fixture, std::string& error) {
  const std::int64_t batch_size = static_cast<std::int64_t>(charges.size());
  if (gpuxtb::detail::gfn2::make_basis_plan(
          batch_size, static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
          atomic_numbers.data(), fixture.basis, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_wavefunction_layout(
          fixture.basis, atomic_numbers.data(), charges.data(), unpaired.data(),
          spin_channels.data(), fixture.layout, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::make_scc_mixer_plan(fixture.layout, history_size, 0.4, 1.0e-8, 2.0e-8,
                                                fixture.plan, error) != GPUXTB_STATUS_SUCCESS) {
    return false;
  }

  fixture.wavefunction_storage =
      std::make_unique<AlignedBuffer>(fixture.layout.workspace_size_bytes);
  fixture.state_storage = std::make_unique<AlignedBuffer>(fixture.plan.state_size_bytes());
  fixture.scratch_storage = std::make_unique<AlignedBuffer>(fixture.plan.workspace_size_bytes());
  if (fixture.wavefunction_storage->data() == nullptr || fixture.state_storage->data() == nullptr ||
      fixture.scratch_storage->data() == nullptr) {
    error = "test fixture allocation failed";
    return false;
  }
  return gpuxtb::detail::gfn2::bind_wavefunction_view(
             fixture.layout, fixture.wavefunction_storage->data(),
             fixture.wavefunction_storage->size(), fixture.wavefunction,
             error) == GPUXTB_STATUS_SUCCESS &&
         gpuxtb::detail::gfn2::bind_scc_mixer_state(fixture.plan, fixture.state_storage->data(),
                                                    fixture.state_storage->size(), fixture.state,
                                                    error) == GPUXTB_STATUS_SUCCESS &&
         gpuxtb::detail::gfn2::bind_scc_mixer_workspace(
             fixture.plan, fixture.scratch_storage->data(), fixture.scratch_storage->size(),
             fixture.scratch, error) == GPUXTB_STATUS_SUCCESS;
}

std::size_t field_begin(const gpuxtb::detail::gfn2::WavefunctionFieldLayout& field,
                        std::size_t system) {
  return static_cast<std::size_t>(field.system_offsets[system]);
}

std::size_t field_end(const gpuxtb::detail::gfn2::WavefunctionFieldLayout& field,
                      std::size_t system) {
  return static_cast<std::size_t>(field.system_offsets[system + 1u]);
}

std::vector<double> get_system_vector(const Fixture& fixture, std::size_t system) {
  std::vector<double> result;
  const std::size_t dimension = static_cast<std::size_t>(
      fixture.plan.vector_offsets()[system + 1u] - fixture.plan.vector_offsets()[system]);
  result.reserve(dimension);
  const std::array<const double*, 3> fields{
      {fixture.wavefunction.qsh, fixture.wavefunction.dipole, fixture.wavefunction.quadrupole}};
  const std::array<const gpuxtb::detail::gfn2::WavefunctionFieldLayout*, 3> layouts{
      {&fixture.layout.qsh, &fixture.layout.dipole, &fixture.layout.quadrupole}};
  for (std::size_t field = 0u; field < fields.size(); ++field) {
    result.insert(result.end(), fields[field] + field_begin(*layouts[field], system),
                  fields[field] + field_end(*layouts[field], system));
  }
  return result;
}

void set_system_vector(const Fixture& fixture, std::size_t system,
                       const std::vector<double>& values) {
  std::size_t packed = 0u;
  const std::array<double*, 3> fields{
      {fixture.wavefunction.qsh, fixture.wavefunction.dipole, fixture.wavefunction.quadrupole}};
  const std::array<const gpuxtb::detail::gfn2::WavefunctionFieldLayout*, 3> layouts{
      {&fixture.layout.qsh, &fixture.layout.dipole, &fixture.layout.quadrupole}};
  for (std::size_t field = 0u; field < fields.size(); ++field) {
    for (std::size_t destination = field_begin(*layouts[field], system);
         destination < field_end(*layouts[field], system); ++destination, ++packed) {
      fields[field][destination] = values[packed];
    }
  }
}

void set_initial_vectors(Fixture& fixture) {
  const std::size_t batch = static_cast<std::size_t>(fixture.plan.batch_size());
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::size_t dimension = static_cast<std::size_t>(
        fixture.plan.vector_offsets()[system + 1u] - fixture.plan.vector_offsets()[system]);
    std::vector<double> values(dimension);
    for (std::size_t component = 0u; component < dimension; ++component) {
      values[component] =
          0.03 * static_cast<double>(system + 1u) + 0.002 * static_cast<double>(component + 1u);
    }
    set_system_vector(fixture, system, values);
  }
}

std::vector<double> residual_for(std::size_t system, std::size_t iteration, std::size_t dimension) {
  std::vector<double> residual(dimension);
  for (std::size_t component = 0u; component < dimension; ++component) {
    const double index = static_cast<double>(component + 1u);
    const double alternating = component % 2u == 0u ? 1.0 : -1.0;
    const double phase = static_cast<double>(component % 3u) - 1.0;
    residual[component] = 0.001 * static_cast<double>(system + 1u) * index +
                          0.0004 * static_cast<double>(iteration + 1u) * alternating +
                          0.00015 * static_cast<double>(iteration) * phase;
  }
  return residual;
}

void install_raw_output(Fixture& fixture, std::size_t system, const std::vector<double>& residual) {
  const std::size_t offset = static_cast<std::size_t>(fixture.plan.vector_offsets()[system]);
  std::vector<double> raw(residual.size());
  for (std::size_t component = 0u; component < residual.size(); ++component) {
    raw[component] = fixture.state.current_inputs[offset + component] + residual[component];
  }
  set_system_vector(fixture, system, raw);
}

bool equal_double_arrays(const double* first, const double* second, std::size_t count,
                         double tolerance = 0.0) {
  for (std::size_t index = 0u; index < count; ++index) {
    if (tolerance == 0.0 ? first[index] != second[index]
                         : !near(first[index], second[index], tolerance)) {
      return false;
    }
  }
  return true;
}

bool states_equal(const Fixture& first, const Fixture& second) {
  const std::size_t total = static_cast<std::size_t>(first.plan.total_vector_elements());
  const std::size_t batch = static_cast<std::size_t>(first.plan.batch_size());
  const std::size_t memory = static_cast<std::size_t>(first.plan.history_size());
  return equal_double_arrays(first.state.current_inputs, second.state.current_inputs, total) &&
         equal_double_arrays(first.state.previous_inputs, second.state.previous_inputs, total) &&
         equal_double_arrays(first.state.previous_residuals, second.state.previous_residuals,
                             total) &&
         equal_double_arrays(first.state.df_history, second.state.df_history, total * memory) &&
         equal_double_arrays(first.state.u_history, second.state.u_history, total * memory) &&
         equal_double_arrays(first.state.omega, second.state.omega, batch * memory) &&
         equal_double_arrays(first.state.residual_rms, second.state.residual_rms, batch) &&
         equal_double_arrays(first.state.residual_maximum, second.state.residual_maximum, batch) &&
         std::equal(first.state.iterations, first.state.iterations + batch,
                    second.state.iterations) &&
         std::equal(first.state.restart_counts, first.state.restart_counts + batch,
                    second.state.restart_counts) &&
         std::equal(first.state.system_statuses, first.state.system_statuses + batch,
                    second.state.system_statuses) &&
         std::equal(first.state.initialized, first.state.initialized + batch,
                    second.state.initialized) &&
         std::equal(first.state.converged, first.state.converged + batch, second.state.converged);
}

int test_tblite_trace_and_ring_history() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 1}, {1}, {1.0}, {0}, {1}, 3, fixture, error));
  CHECK(fixture.plan.maximum_vector_elements() == 10);
  std::vector<double> initial(10u);
  for (std::size_t component = 0u; component < initial.size(); ++component) {
    initial[component] = 0.05 * static_cast<double>(component + 1u);
  }
  set_system_vector(fixture, 0u, initial);
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(
            fixture.plan, fixture.wavefunction, fixture.state, error) == GPUXTB_STATUS_SUCCESS);

  const std::array<std::array<double, 10>, 6> expected{{
      {{0.054000000000000006, 0.10800000000000001, 0.16200000000000003, 0.21600000000000003,
        0.27000000000000002, 0.32400000000000007, 0.37800000000000006, 0.43200000000000005,
        0.48599999999999999, 0.54000000000000004}},
      {{0.07443626619586563, 0.13215195095875032, 0.21216174429894297, 0.2698774290618276,
        0.34988722240202019, 0.40760290716490499, 0.48761270050509758, 0.54532838526798222,
        0.62533817860817464, 0.68305386337105956}},
      {{0.066832349364537913, 0.12290542441617094, 0.19484999757093724, 0.2492065487259027,
        0.32115112188066897, 0.37722419693230208, 0.44745224619040069, 0.50352532124203375,
        0.57546989439679985, 0.62982644555176548}},
      {{0.07300609353444773, 0.13277696660815397, 0.21207132312613922, 0.26973110801614386,
        0.34902546453412908, 0.40879633760784156, 0.48597960594212641, 0.5457504790158334,
        0.62504483553381984, 0.6827046204238254}},
      {{0.069231460211817933, 0.12746580757264292, 0.2024803995714857, 0.25892250940685241,
        0.33330790842881808, 0.39154225578964652, 0.46476461026303173, 0.52299895762385717,
        0.59738435664582334, 0.65382646648119069}},
      {{0.070322609766037644, 0.1288936126215946, 0.20541245939033212, 0.26207982769422339,
        0.33789437634569014, 0.39646537920125086, 0.47108059141832354, 0.52965159427388098,
        0.60546614292534839, 0.66213351122924025}},
  }};

  std::array<std::vector<double>, 6> residuals;
  for (std::size_t iteration = 0u; iteration < residuals.size(); ++iteration) {
    residuals[iteration].resize(10u);
    for (std::size_t component = 0u; component < 10u; ++component) {
      const double index = static_cast<double>(component + 1u);
      if (iteration == 0u) {
        residuals[iteration][component] = 0.01 * index;
      } else if (iteration == 1u) {
        residuals[iteration][component] = 0.008 * index + (component % 2u == 0u ? 0.003 : -0.003);
      } else if (iteration == 2u) {
        residuals[iteration][component] =
            -0.004 * index + 0.002 * (static_cast<double>(component % 3u) - 1.0);
      } else if (iteration == 3u) {
        residuals[iteration][component] = 0.003 * index - (component % 2u == 0u ? 0.0015 : -0.0015);
      } else if (iteration == 4u) {
        residuals[iteration][component] =
            -0.002 * index + 0.0007 * (static_cast<double>(component % 4u) - 1.5);
      } else {
        residuals[iteration][component] = 0.001 * index + (component % 2u == 0u ? 0.0002 : -0.0002);
      }
    }
    install_raw_output(fixture, 0u, residuals[iteration]);
    CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_batch_cpu(fixture.plan, fixture.wavefunction,
                                                          fixture.state, fixture.scratch,
                                                          error) == GPUXTB_STATUS_SUCCESS);
    const std::vector<double> mixed = get_system_vector(fixture, 0u);
    for (std::size_t component = 0u; component < expected[iteration].size(); ++component) {
      CHECK(near(mixed[component], expected[iteration][component], 8.0e-13));
    }
    CHECK(fixture.state.iterations[0] == iteration + 1u);
  }

  /* After six iterations with memory three, slots contain dF4, dF5, dF3. */
  const std::array<std::size_t, 3> represented_differences{{4u, 5u, 3u}};
  for (std::size_t slot = 0u; slot < represented_differences.size(); ++slot) {
    const std::size_t current_residual = represented_differences[slot];
    double norm_square = 0.0;
    for (std::size_t component = 0u; component < 10u; ++component) {
      const double difference =
          residuals[current_residual][component] - residuals[current_residual - 1u][component];
      norm_square += difference * difference;
    }
    const double norm = std::sqrt(norm_square);
    for (std::size_t component = 0u; component < 10u; ++component) {
      const double expected_df =
          (residuals[current_residual][component] - residuals[current_residual - 1u][component]) /
          norm;
      CHECK(near(fixture.state.df_history[slot * 10u + component], expected_df, 2.0e-14));
    }
    CHECK(fixture.state.omega[slot] == 1.0);
  }
  return 0;
}

int test_batch_equals_sequential_and_parallel_workers() {
  const std::vector<std::int64_t> atom_offsets{0, 1, 3};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 1};
  const std::vector<double> charges{1.0, 0.0};
  const std::vector<std::int32_t> unpaired{0, 1};
  const std::vector<std::int32_t> spin_channels{1, 2};
  Fixture batch;
  Fixture sequential;
  std::string error;
  CHECK(make_fixture(atom_offsets, atomic_numbers, charges, unpaired, spin_channels, 4, batch,
                     error));
  CHECK(make_fixture(atom_offsets, atomic_numbers, charges, unpaired, spin_channels, 4, sequential,
                     error));
  set_initial_vectors(batch);
  set_initial_vectors(sequential);
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(
            batch.plan, batch.wavefunction, batch.state, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(
            sequential.plan, sequential.wavefunction, sequential.state, error) ==
        GPUXTB_STATUS_SUCCESS);

  const std::size_t systems = static_cast<std::size_t>(batch.plan.batch_size());
  for (std::size_t iteration = 0u; iteration < 7u; ++iteration) {
    for (std::size_t system = 0u; system < systems; ++system) {
      const std::size_t dimension = static_cast<std::size_t>(
          batch.plan.vector_offsets()[system + 1u] - batch.plan.vector_offsets()[system]);
      const std::vector<double> residual = residual_for(system, iteration, dimension);
      install_raw_output(batch, system, residual);
      install_raw_output(sequential, system, residual);
    }
    CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_batch_cpu(batch.plan, batch.wavefunction,
                                                          batch.state, batch.scratch,
                                                          error) == GPUXTB_STATUS_SUCCESS);
    for (std::size_t system = 0u; system < systems; ++system) {
      CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_system_cpu(
                sequential.plan, static_cast<std::int64_t>(system), sequential.wavefunction,
                sequential.state, sequential.scratch, error) == GPUXTB_STATUS_SUCCESS);
      CHECK(get_system_vector(batch, system) == get_system_vector(sequential, system));
    }
    CHECK(states_equal(batch, sequential));
  }

  Fixture parallel;
  Fixture serial_workers;
  CHECK(make_fixture(atom_offsets, atomic_numbers, charges, unpaired, spin_channels, 4, parallel,
                     error));
  CHECK(make_fixture(atom_offsets, atomic_numbers, charges, unpaired, spin_channels, 4,
                     serial_workers, error));
  set_initial_vectors(parallel);
  set_initial_vectors(serial_workers);
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(
            parallel.plan, parallel.wavefunction, parallel.state, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(
            serial_workers.plan, serial_workers.wavefunction, serial_workers.state, error) ==
        GPUXTB_STATUS_SUCCESS);
  for (std::size_t system = 0u; system < systems; ++system) {
    const std::size_t dimension = static_cast<std::size_t>(
        parallel.plan.vector_offsets()[system + 1u] - parallel.plan.vector_offsets()[system]);
    const std::vector<double> residual = residual_for(system, 0u, dimension);
    install_raw_output(parallel, system, residual);
    install_raw_output(serial_workers, system, residual);
  }
  /* Warm up both fixtures so the concurrent calls below exercise the shared
   * Broyden-history path rather than only the first-step damping path. */
  for (std::size_t system = 0u; system < systems; ++system) {
    CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_system_cpu(
              parallel.plan, static_cast<std::int64_t>(system), parallel.wavefunction,
              parallel.state, parallel.scratch, error) == GPUXTB_STATUS_SUCCESS);
    CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_system_cpu(
              serial_workers.plan, static_cast<std::int64_t>(system), serial_workers.wavefunction,
              serial_workers.state, serial_workers.scratch, error) == GPUXTB_STATUS_SUCCESS);
  }
  for (std::size_t system = 0u; system < systems; ++system) {
    const std::size_t dimension = static_cast<std::size_t>(
        parallel.plan.vector_offsets()[system + 1u] - parallel.plan.vector_offsets()[system]);
    const std::vector<double> residual = residual_for(system, 1u, dimension);
    install_raw_output(parallel, system, residual);
    install_raw_output(serial_workers, system, residual);
  }
  AlignedBuffer second_scratch_storage(parallel.plan.workspace_size_bytes());
  SccMixerWorkspace second_scratch;
  CHECK(gpuxtb::detail::gfn2::bind_scc_mixer_workspace(
            parallel.plan, second_scratch_storage.data(), second_scratch_storage.size(),
            second_scratch, error) == GPUXTB_STATUS_SUCCESS);
  gpuxtb_status_t first_status = GPUXTB_STATUS_INTERNAL_ERROR;
  gpuxtb_status_t second_status = GPUXTB_STATUS_INTERNAL_ERROR;
  std::string first_error;
  std::string second_error;
  std::thread first_worker([&] {
    first_status = gpuxtb::detail::gfn2::mix_scc_broyden_system_cpu(
        parallel.plan, 0, parallel.wavefunction, parallel.state, parallel.scratch, first_error);
  });
  std::thread second_worker([&] {
    second_status = gpuxtb::detail::gfn2::mix_scc_broyden_system_cpu(
        parallel.plan, 1, parallel.wavefunction, parallel.state, second_scratch, second_error);
  });
  first_worker.join();
  second_worker.join();
  CHECK(first_status == GPUXTB_STATUS_SUCCESS);
  CHECK(second_status == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_system_cpu(
            serial_workers.plan, 0, serial_workers.wavefunction, serial_workers.state,
            serial_workers.scratch, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_system_cpu(
            serial_workers.plan, 1, serial_workers.wavefunction, serial_workers.state,
            serial_workers.scratch, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(states_equal(parallel, serial_workers));
  CHECK(get_system_vector(parallel, 0u) == get_system_vector(serial_workers, 0u));
  CHECK(get_system_vector(parallel, 1u) == get_system_vector(serial_workers, 1u));
  return 0;
}

int test_nonfinite_isolation_failure_atomicity_and_restart() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 1, 3}, {1, 8, 1}, {1.0, 0.0}, {0, 1}, {1, 2}, 3, fixture, error));
  set_initial_vectors(fixture);
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(
            fixture.plan, fixture.wavefunction, fixture.state, error) == GPUXTB_STATUS_SUCCESS);
  for (std::size_t system = 0u; system < 2u; ++system) {
    const std::size_t dimension = static_cast<std::size_t>(
        fixture.plan.vector_offsets()[system + 1u] - fixture.plan.vector_offsets()[system]);
    install_raw_output(fixture, system, residual_for(system, 0u, dimension));
  }
  CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_batch_cpu(fixture.plan, fixture.wavefunction,
                                                        fixture.state, fixture.scratch,
                                                        error) == GPUXTB_STATUS_SUCCESS);

  /* Establish a converged state before injecting a numerical failure. The
   * failure must not partially clear convergence or any other diagnostics. */
  install_raw_output(fixture, 0u, std::vector<double>(10u, 0.0));
  const std::size_t second_dimension =
      static_cast<std::size_t>(fixture.plan.vector_offsets()[2] - fixture.plan.vector_offsets()[1]);
  install_raw_output(fixture, 1u, residual_for(1u, 1u, second_dimension));
  CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_batch_cpu(fixture.plan, fixture.wavefunction,
                                                        fixture.state, fixture.scratch,
                                                        error) == GPUXTB_STATUS_SUCCESS);
  CHECK(fixture.state.converged[0] == 1u);

  const std::size_t first_offset = static_cast<std::size_t>(fixture.plan.vector_offsets()[0]);
  const std::size_t first_dimension =
      static_cast<std::size_t>(fixture.plan.vector_offsets()[1] - fixture.plan.vector_offsets()[0]);
  const std::size_t first_history =
      first_dimension * static_cast<std::size_t>(fixture.plan.history_size());
  const std::vector<double> current_before(
      fixture.state.current_inputs + first_offset,
      fixture.state.current_inputs + first_offset + first_dimension);
  const std::vector<double> previous_before(
      fixture.state.previous_inputs + first_offset,
      fixture.state.previous_inputs + first_offset + first_dimension);
  const std::vector<double> residual_before(
      fixture.state.previous_residuals + first_offset,
      fixture.state.previous_residuals + first_offset + first_dimension);
  const std::vector<double> df_before(fixture.state.df_history,
                                      fixture.state.df_history + first_history);
  const std::vector<double> u_before(fixture.state.u_history,
                                     fixture.state.u_history + first_history);
  const std::vector<double> omega_before(
      fixture.state.omega,
      fixture.state.omega + static_cast<std::size_t>(fixture.plan.history_size()));
  const std::uint64_t first_iteration_before = fixture.state.iterations[0];
  const std::uint64_t second_iteration_before = fixture.state.iterations[1];
  const std::uint64_t first_restart_before = fixture.state.restart_counts[0];
  const double first_rms_before = fixture.state.residual_rms[0];
  const double first_maximum_before = fixture.state.residual_maximum[0];
  const std::uint8_t first_initialized_before = fixture.state.initialized[0];
  const std::uint8_t first_converged_before = fixture.state.converged[0];

  std::vector<double> bad_raw = current_before;
  bad_raw[0] = std::numeric_limits<double>::quiet_NaN();
  set_system_vector(fixture, 0u, bad_raw);
  install_raw_output(fixture, 1u, residual_for(1u, 2u, second_dimension));
  CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_batch_cpu(fixture.plan, fixture.wavefunction,
                                                        fixture.state, fixture.scratch,
                                                        error) == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(std::isnan(get_system_vector(fixture, 0u)[0]));
  CHECK(std::equal(current_before.begin(), current_before.end(),
                   fixture.state.current_inputs + first_offset));
  CHECK(std::equal(previous_before.begin(), previous_before.end(),
                   fixture.state.previous_inputs + first_offset));
  CHECK(std::equal(residual_before.begin(), residual_before.end(),
                   fixture.state.previous_residuals + first_offset));
  CHECK(std::equal(df_before.begin(), df_before.end(), fixture.state.df_history));
  CHECK(std::equal(u_before.begin(), u_before.end(), fixture.state.u_history));
  CHECK(std::equal(omega_before.begin(), omega_before.end(), fixture.state.omega));
  CHECK(fixture.state.iterations[0] == first_iteration_before);
  CHECK(fixture.state.restart_counts[0] == first_restart_before);
  CHECK(fixture.state.residual_rms[0] == first_rms_before);
  CHECK(fixture.state.residual_maximum[0] == first_maximum_before);
  CHECK(fixture.state.initialized[0] == first_initialized_before);
  CHECK(fixture.state.converged[0] == first_converged_before);
  CHECK(fixture.state.system_statuses[0] == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(fixture.state.iterations[1] == second_iteration_before + 1u);
  CHECK(fixture.state.system_statuses[1] == GPUXTB_STATUS_SUCCESS);

  const std::vector<double> second_current_before(
      fixture.state.current_inputs + fixture.plan.vector_offsets()[1],
      fixture.state.current_inputs + fixture.plan.vector_offsets()[2]);
  const std::uint64_t second_restart_before = fixture.state.restart_counts[1];
  std::vector<double> restart(first_dimension);
  for (std::size_t component = 0u; component < first_dimension; ++component) {
    restart[component] = -0.02 * static_cast<double>(component + 1u);
  }
  set_system_vector(fixture, 0u, restart);
  CHECK(gpuxtb::detail::gfn2::restart_scc_mixer_system_cpu(
            fixture.plan, 0, fixture.wavefunction, fixture.state, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(fixture.state.restart_counts[0] == 1u);
  CHECK(fixture.state.iterations[0] == 0u);
  CHECK(fixture.state.system_statuses[0] == GPUXTB_STATUS_SUCCESS);
  CHECK(std::equal(restart.begin(), restart.end(), fixture.state.current_inputs));
  CHECK(std::all_of(fixture.state.df_history, fixture.state.df_history + first_history,
                    [](double value) { return value == 0.0; }));
  CHECK(fixture.state.restart_counts[1] == second_restart_before);
  CHECK(std::equal(second_current_before.begin(), second_current_before.end(),
                   fixture.state.current_inputs + fixture.plan.vector_offsets()[1]));

  const std::vector<std::byte> state_snapshot(
      static_cast<const std::byte*>(fixture.state_storage->data()),
      static_cast<const std::byte*>(fixture.state_storage->data()) +
          fixture.plan.state_size_bytes());
  const std::vector<double> wavefunction_snapshot(
      static_cast<const double*>(fixture.wavefunction_storage->data()),
      static_cast<const double*>(fixture.wavefunction_storage->data()) +
          fixture.layout.workspace_size_bytes / sizeof(double));
  WavefunctionView malformed = fixture.wavefunction;
  malformed.qsh = malformed.dipole;
  CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_batch_cpu(fixture.plan, malformed, fixture.state,
                                                        fixture.scratch,
                                                        error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(std::memcmp(state_snapshot.data(), fixture.state_storage->data(),
                    fixture.plan.state_size_bytes()) == 0);
  CHECK(std::equal(wavefunction_snapshot.begin(), wavefunction_snapshot.end(),
                   static_cast<const double*>(fixture.wavefunction_storage->data())));
  return 0;
}

int test_initialization_atomicity_validation_and_zero_allocation() {
  Fixture fixture;
  std::string error;
  CHECK(make_fixture({0, 1, 3}, {1, 8, 1}, {1.0, 0.0}, {0, 1}, {1, 2}, 3, fixture, error));
  set_initial_vectors(fixture);
  std::vector<double> second = get_system_vector(fixture, 1u);
  second.back() = std::numeric_limits<double>::infinity();
  set_system_vector(fixture, 1u, second);
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(fixture.plan, fixture.wavefunction,
                                                             fixture.state, error) ==
        GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.state.initialized[0] == 0u);
  CHECK(fixture.state.initialized[1] == 0u);
  CHECK(std::all_of(fixture.state.current_inputs,
                    fixture.state.current_inputs + fixture.plan.total_vector_elements(),
                    [](double value) { return value == 0.0; }));

  set_initial_vectors(fixture);
  CHECK(gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(
            fixture.plan, fixture.wavefunction, fixture.state, error) == GPUXTB_STATUS_SUCCESS);
  const auto* const identity = fixture.plan.identity();
  SccMixerPlan preserved = fixture.plan;
  CHECK(gpuxtb::detail::gfn2::make_scc_mixer_plan(fixture.layout, 0, 0.4, 1.0e-8, 1.0e-8, preserved,
                                                  error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(preserved.identity() == identity);

  AlignedBuffer undersized_state(fixture.plan.state_size_bytes());
  SccMixerState rejected_state = fixture.state;
  CHECK(gpuxtb::detail::gfn2::bind_scc_mixer_state(
            fixture.plan, undersized_state.data(), fixture.plan.state_size_bytes() - 1u,
            rejected_state, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(rejected_state.workspace_base == fixture.state.workspace_base);
  SccMixerWorkspace rejected_workspace = fixture.scratch;
  CHECK(gpuxtb::detail::gfn2::bind_scc_mixer_workspace(
            fixture.plan, static_cast<std::byte*>(fixture.scratch_storage->data()) + 1,
            fixture.scratch_storage->size() - 1u, rejected_workspace,
            error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(rejected_workspace.workspace_base == fixture.scratch.workspace_base);

  const std::size_t systems = static_cast<std::size_t>(fixture.plan.batch_size());
  for (std::size_t system = 0u; system < systems; ++system) {
    const std::size_t dimension = static_cast<std::size_t>(
        fixture.plan.vector_offsets()[system + 1u] - fixture.plan.vector_offsets()[system]);
    install_raw_output(fixture, system, residual_for(system, 0u, dimension));
  }
  error.reserve(256u);
  const std::size_t before = allocation_test::count.load(std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const gpuxtb_status_t status = gpuxtb::detail::gfn2::mix_scc_broyden_batch_cpu(
      fixture.plan, fixture.wavefunction, fixture.state, fixture.scratch, error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  const std::size_t after = allocation_test::count.load(std::memory_order_relaxed);
  CHECK(status == GPUXTB_STATUS_SUCCESS);
  CHECK(after == before);

  for (std::size_t system = 0u; system < systems; ++system) {
    const std::size_t dimension = static_cast<std::size_t>(
        fixture.plan.vector_offsets()[system + 1u] - fixture.plan.vector_offsets()[system]);
    install_raw_output(fixture, system, std::vector<double>(dimension, 0.0));
  }
  CHECK(gpuxtb::detail::gfn2::mix_scc_broyden_batch_cpu(fixture.plan, fixture.wavefunction,
                                                        fixture.state, fixture.scratch,
                                                        error) == GPUXTB_STATUS_SUCCESS);
  CHECK(fixture.state.converged[0] == 1u);
  CHECK(fixture.state.converged[1] == 1u);
  CHECK(fixture.state.residual_rms[0] == 0.0);
  CHECK(fixture.state.residual_maximum[1] == 0.0);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_tblite_trace_and_ring_history(); status != 0) {
    return status;
  }
  if (const int status = test_batch_equals_sequential_and_parallel_workers(); status != 0) {
    return status;
  }
  if (const int status = test_nonfinite_isolation_failure_atomicity_and_restart(); status != 0) {
    return status;
  }
  if (const int status = test_initialization_atomicity_validation_and_zero_allocation();
      status != 0) {
    return status;
  }
  return 0;
}
