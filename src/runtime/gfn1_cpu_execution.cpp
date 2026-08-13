// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "runtime/gfn1_cpu_execution.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#if defined(_WIN32)
#include <malloc.h>
#endif

#include "model/gfn1/basis.hpp"
#include "model/gfn1/d3.hpp"
#include "model/gfn1/es2.hpp"
#include "model/gfn1/es3.hpp"
#include "model/gfn1/external_point_charges.hpp"
#include "model/gfn1/force.hpp"
#include "model/gfn1/h0.hpp"
#include "model/gfn1/halogen.hpp"
#include "model/gfn1/integrals.hpp"
#include "model/gfn1/mulliken.hpp"
#include "model/gfn1/repulsion.hpp"
#include "model/gfn1/scc_driver.hpp"
#include "model/gfn1/scc_mixer.hpp"
#include "model/gfn1/spin.hpp"
#include "model/gfn1/wavefunction.hpp"

namespace xtbloom::detail {
namespace {

using namespace xtbloom::detail::gfn1;

constexpr std::size_t kHostAlignment = 64u;
constexpr std::int32_t kDefaultMixerHistory = 8;
constexpr double kDefaultMixerDamping = 0.4;
constexpr std::size_t kMaximumAutomaticCpuThreads = 64u;
constexpr std::uint32_t kSupportedFlags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                                          XTBLOOM_COMPUTE_ATOMIC_CHARGES |
                                          XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;

std::size_t process_cpu_count() noexcept {
#if defined(__linux__)
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
    std::size_t count = 0u;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &affinity)) ++count;
    }
    if (count != 0u) return count;
  }
#endif
  const unsigned int hardware = std::thread::hardware_concurrency();
  return hardware == 0u ? 1u : static_cast<std::size_t>(hardware);
}

std::size_t resolve_cpu_threads(std::int32_t requested) noexcept {
  const std::size_t available = process_cpu_count();
  if (requested == 0) {
    return std::max<std::size_t>(1u, std::min(available, kMaximumAutomaticCpuThreads));
  }
  return std::max<std::size_t>(
      1u, std::min<std::size_t>(available, static_cast<std::size_t>(requested)));
}

/* Each model cache owns its worker pool because the provider cleanup callback
 * is tied to that cache's separately loaded backend namespace. Sharing a pool
 * would let the second model overwrite the first model's TLS cleanup owner. */
class CpuWorkerPool final {
 public:
  using Task = void (*)(void*, std::size_t) noexcept;
  using ThreadCleanup = void (*)(void*) noexcept;

  explicit CpuWorkerPool(std::size_t concurrency)
      : concurrency_(std::max<std::size_t>(1u, concurrency)) {
    workers_.reserve(concurrency_ - 1u);
    for (std::size_t worker = 0u; worker + 1u < concurrency_; ++worker) {
      try {
        workers_.emplace_back([this] { worker_loop(); });
      } catch (const std::system_error&) {
        break;
      }
    }
    concurrency_ = workers_.size() + 1u;
  }

  ~CpuWorkerPool() { stop_and_join(); }
  CpuWorkerPool(const CpuWorkerPool&) = delete;
  CpuWorkerPool& operator=(const CpuWorkerPool&) = delete;

  void parallel_for(std::size_t task_count, void* context, Task task) noexcept {
    if (task_count == 0u) return;
    if (workers_.empty() || task_count == 1u) {
      for (std::size_t index = 0u; index < task_count; ++index) task(context, index);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      task_ = task;
      task_context_ = context;
      task_count_ = task_count;
      next_task_.store(0u, std::memory_order_relaxed);
      completed_workers_ = 0u;
      ++generation_;
    }
    work_available_.notify_all();
    drain_tasks(task, context, task_count);
    std::unique_lock<std::mutex> lock(mutex_);
    work_complete_.wait(lock, [this] { return completed_workers_ == workers_.size(); });
    task_ = nullptr;
    task_context_ = nullptr;
    task_count_ = 0u;
  }

  [[nodiscard]] std::size_t resident_bytes() const noexcept {
    return workers_.capacity() * sizeof(std::thread);
  }

  void set_thread_cleanup(void* context, ThreadCleanup cleanup) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    thread_cleanup_context_ = context;
    thread_cleanup_ = cleanup;
  }

 private:
  void drain_tasks(Task task, void* context, std::size_t task_count) noexcept {
    for (;;) {
      const std::size_t index = next_task_.fetch_add(1u, std::memory_order_relaxed);
      if (index >= task_count) return;
      task(context, index);
    }
  }

  void worker_loop() noexcept {
    std::uint64_t observed_generation = 0u;
    for (;;) {
      Task task = nullptr;
      void* context = nullptr;
      std::size_t task_count = 0u;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_available_.wait(lock, [this, observed_generation] {
          return stopping_ || generation_ != observed_generation;
        });
        if (stopping_) {
          void* cleanup_context = thread_cleanup_context_;
          ThreadCleanup cleanup = thread_cleanup_;
          lock.unlock();
          if (cleanup != nullptr) cleanup(cleanup_context);
          return;
        }
        observed_generation = generation_;
        task = task_;
        context = task_context_;
        task_count = task_count_;
      }
      drain_tasks(task, context, task_count);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++completed_workers_;
      }
      work_complete_.notify_one();
    }
  }

  void stop_and_join() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    work_available_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
  }

  std::size_t concurrency_ = 1u;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_available_;
  std::condition_variable work_complete_;
  std::atomic<std::size_t> next_task_{0u};
  Task task_ = nullptr;
  void* task_context_ = nullptr;
  std::size_t task_count_ = 0u;
  std::size_t completed_workers_ = 0u;
  std::uint64_t generation_ = 0u;
  bool stopping_ = false;
  void* thread_cleanup_context_ = nullptr;
  ThreadCleanup thread_cleanup_ = nullptr;
};

void* host_aligned_allocate(std::size_t alignment, std::size_t size) {
#if defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  return std::aligned_alloc(alignment, size);
#endif
}

void host_aligned_free(void* pointer) {
#if defined(_WIN32)
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}

class AlignedBuffer {
 public:
  AlignedBuffer() = default;
  ~AlignedBuffer() { reset(); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  bool allocate(std::size_t requested) {
    reset();
    const std::size_t useful = std::max<std::size_t>(requested, 1u);
    if (useful > std::numeric_limits<std::size_t>::max() - (kHostAlignment - 1u)) return false;
    size_ = (useful + kHostAlignment - 1u) & ~(kHostAlignment - 1u);
    data_ = host_aligned_allocate(kHostAlignment, size_);
    if (data_ == nullptr) {
      size_ = 0u;
      return false;
    }
    std::memset(data_, 0, size_);
    return true;
  }

  void reset() noexcept {
    host_aligned_free(data_);
    data_ = nullptr;
    size_ = 0u;
  }

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0u;
};

xtbloom_status_t allocate(AlignedBuffer& buffer, std::size_t bytes, const char* purpose,
                          std::string& error) {
  if (buffer.allocate(bytes)) return XTBLOOM_STATUS_SUCCESS;
  error = std::string("failed to allocate CPU GFN1 ") + purpose;
  return XTBLOOM_STATUS_ALLOCATION_FAILED;
}

template <typename T>
void copy_from_buffer(const xtbloom_const_buffer_t& source, std::size_t count,
                      std::vector<T>& destination) {
  destination.resize(count);
  if (count != 0u) std::memcpy(destination.data(), source.data, count * sizeof(T));
}

template <typename T>
void publish_to_buffer(const std::vector<T>& source, xtbloom_buffer_t& destination) {
  if (!source.empty()) std::memcpy(destination.data, source.data(), source.size() * sizeof(T));
}

template <typename T>
std::size_t vector_bytes(const std::vector<T>& values) noexcept {
  return values.capacity() * sizeof(T);
}

bool finite_values(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

struct HostRequest {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_point_charges = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> point_offsets;
  std::vector<double> point_positions;
  std::vector<double> point_charges;
  std::vector<double> point_hardnesses;
  std::vector<double> periodic_shifts;
  std::vector<std::int64_t> response_offsets;
  std::vector<double> response_matrices;
  bool shifts_enabled = false;
  bool response_enabled = false;
};

void stage_request(const xtbloom_batch_t& batch, HostRequest& request) {
  request.batch_size = batch.batch_size;
  request.total_atoms = batch.total_atoms;
  request.total_point_charges = batch.total_point_charges;
  copy_from_buffer(batch.atom_offsets, static_cast<std::size_t>(batch.batch_size) + 1u,
                   request.atom_offsets);
  copy_from_buffer(batch.atomic_numbers, static_cast<std::size_t>(batch.total_atoms),
                   request.atomic_numbers);
  copy_from_buffer(batch.positions, 3u * static_cast<std::size_t>(batch.total_atoms),
                   request.positions);
  copy_from_buffer(batch.molecular_charges, static_cast<std::size_t>(batch.batch_size),
                   request.molecular_charges);
  copy_from_buffer(batch.unpaired_electrons, static_cast<std::size_t>(batch.batch_size),
                   request.unpaired_electrons);
  if (batch.struct_size >= XTBLOOM_BATCH_V2_SIZE && batch.spin_channels.data != nullptr) {
    copy_from_buffer(batch.spin_channels, static_cast<std::size_t>(batch.batch_size),
                     request.spin_channels);
  } else {
    request.spin_channels.assign(static_cast<std::size_t>(batch.batch_size), 1);
  }

  if (batch.total_point_charges != 0) {
    copy_from_buffer(batch.point_charge_offsets, static_cast<std::size_t>(batch.batch_size) + 1u,
                     request.point_offsets);
    copy_from_buffer(batch.point_charge_positions,
                     3u * static_cast<std::size_t>(batch.total_point_charges),
                     request.point_positions);
    copy_from_buffer(batch.point_charge_values, static_cast<std::size_t>(batch.total_point_charges),
                     request.point_charges);
    copy_from_buffer(batch.point_charge_gammas, static_cast<std::size_t>(batch.total_point_charges),
                     request.point_hardnesses);
  } else {
    request.point_offsets.assign(static_cast<std::size_t>(batch.batch_size) + 1u, 0);
    request.point_positions.clear();
    request.point_charges.clear();
    request.point_hardnesses.clear();
  }

  request.shifts_enabled = batch.atomic_potential_shifts.data != nullptr;
  if (request.shifts_enabled) {
    copy_from_buffer(batch.atomic_potential_shifts, static_cast<std::size_t>(batch.total_atoms),
                     request.periodic_shifts);
  } else {
    request.periodic_shifts.clear();
  }
  request.response_enabled = batch.total_charge_response_elements != 0;
  if (request.response_enabled) {
    copy_from_buffer(batch.charge_response_offsets, static_cast<std::size_t>(batch.batch_size) + 1u,
                     request.response_offsets);
    copy_from_buffer(batch.charge_response_matrix,
                     static_cast<std::size_t>(batch.total_charge_response_elements),
                     request.response_matrices);
  } else {
    request.response_offsets.clear();
    request.response_matrices.clear();
  }
}

xtbloom_status_t validate_hidden_request(const xtbloom_batch_t& batch,
                                         const xtbloom_compute_options_t& options,
                                         const HostRequest& request, std::string& error) {
  if (options.model != XTBLOOM_MODEL_GFN1_XTB || options.flags == 0u ||
      (options.flags & ~kSupportedFlags) != 0u || options.max_scc_iterations <= 0 ||
      !std::isfinite(options.charge_tolerance) || options.charge_tolerance <= 0.0 ||
      !std::isfinite(options.energy_tolerance) || options.energy_tolerance <= 0.0 ||
      !std::isfinite(options.electronic_temperature) || options.electronic_temperature < 0.0) {
    error = "internal CPU GFN1 options contain an unsupported output or invalid SCC policy";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE &&
      options.scc_start_mode != XTBLOOM_SCC_START_FRESH &&
      options.scc_start_mode != XTBLOOM_SCC_START_WARM) {
    error = "internal CPU GFN1 SCC start mode is invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V3_SIZE &&
      (options.scc_mixer != XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN || options.scc_mixer_history < 1 ||
       options.scc_mixer_history > 64 || !std::isfinite(options.scc_mixer_damping) ||
       options.scc_mixer_damping <= 0.0 || options.scc_mixer_damping > 1.0 ||
       (options.determinism != XTBLOOM_DETERMINISM_DEFAULT &&
        options.determinism != XTBLOOM_DETERMINISM_REPRODUCIBLE))) {
    error = "internal CPU GFN1 mixer or determinism policy is invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (batch.struct_size >= XTBLOOM_BATCH_V3_SIZE && batch.total_interactions != 0) {
    error = "internal CPU GFN1 execution does not implement interaction attachments";
    return XTBLOOM_STATUS_NOT_IMPLEMENTED;
  }
  if (!finite_values(request.positions) || !finite_values(request.molecular_charges) ||
      !finite_values(request.point_positions) || !finite_values(request.point_charges) ||
      !finite_values(request.periodic_shifts) || !finite_values(request.response_matrices)) {
    error = "internal CPU GFN1 numerical inputs contain NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!std::all_of(request.point_hardnesses.begin(), request.point_hardnesses.end(),
                   [](double value) { return std::isfinite(value) && value > 0.0; })) {
    error = "internal CPU GFN1 point-charge gammas must be finite and positive";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (request.response_enabled) {
    if (request.response_offsets.size() != static_cast<std::size_t>(request.batch_size) + 1u ||
        request.response_offsets.front() != 0 ||
        request.response_offsets.back() !=
            static_cast<std::int64_t>(request.response_matrices.size())) {
      error = "internal CPU GFN1 charge-response offsets are malformed";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t system = 0; system < request.batch_size; ++system) {
      const std::size_t index = static_cast<std::size_t>(system);
      const std::int64_t atoms = request.atom_offsets[index + 1u] - request.atom_offsets[index];
      const std::int64_t begin = request.response_offsets[index];
      const std::int64_t end = request.response_offsets[index + 1u];
      if (atoms < 0 || (atoms != 0 && atoms > std::numeric_limits<std::int64_t>::max() / atoms) ||
          begin < 0 || end < begin || end - begin != atoms * atoms ||
          end > static_cast<std::int64_t>(request.response_matrices.size())) {
        error =
            "internal CPU GFN1 charge-response segments must contain exactly atoms*atoms values";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      for (std::int64_t row = 0; row < atoms; ++row) {
        for (std::int64_t column = row + 1; column < atoms; ++column) {
          if (request.response_matrices[static_cast<std::size_t>(begin + row * atoms + column)] !=
              request.response_matrices[static_cast<std::size_t>(begin + column * atoms + row)]) {
            error = "internal CPU GFN1 charge-response matrices must be exactly symmetric";
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
        }
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

struct NormalizedPolicy {
  xtbloom_scc_mixer_t mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
  std::int32_t history = kDefaultMixerHistory;
  double damping = kDefaultMixerDamping;
  xtbloom_determinism_t determinism = XTBLOOM_DETERMINISM_DEFAULT;
};

NormalizedPolicy normalize_policy(const xtbloom_compute_options_t& options) noexcept {
  NormalizedPolicy policy;
  if (options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V3_SIZE) {
    policy.mixer = options.scc_mixer;
    policy.history = options.scc_mixer_history;
    policy.damping = options.scc_mixer_damping;
    policy.determinism = options.determinism;
  }
  return policy;
}

struct SystemKey {
  std::vector<std::int32_t> atomic_numbers;
  double molecular_charge = 0.0;
  std::int32_t unpaired_electrons = 0;
  std::int32_t spin_channels = 1;
  std::int64_t point_count = 0;
  bool periodic_enabled = false;
  std::uint32_t compute_flags = 0u;
  std::int32_t maximum_iterations = 0;
  double charge_tolerance = 0.0;
  double energy_tolerance = 0.0;
  double electronic_temperature = 0.0;
  xtbloom_scc_mixer_t mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
  std::int32_t mixer_history = kDefaultMixerHistory;
  double mixer_damping = kDefaultMixerDamping;
  xtbloom_determinism_t determinism = XTBLOOM_DETERMINISM_DEFAULT;

  friend bool operator==(const SystemKey& first, const SystemKey& second) {
    return first.atomic_numbers == second.atomic_numbers &&
           first.molecular_charge == second.molecular_charge &&
           first.unpaired_electrons == second.unpaired_electrons &&
           first.spin_channels == second.spin_channels && first.point_count == second.point_count &&
           first.periodic_enabled == second.periodic_enabled &&
           first.compute_flags == second.compute_flags &&
           first.maximum_iterations == second.maximum_iterations &&
           first.charge_tolerance == second.charge_tolerance &&
           first.energy_tolerance == second.energy_tolerance &&
           first.electronic_temperature == second.electronic_temperature &&
           first.mixer == second.mixer && first.mixer_history == second.mixer_history &&
           first.mixer_damping == second.mixer_damping && first.determinism == second.determinism;
  }
};

void make_system_keys(const HostRequest& request, const xtbloom_compute_options_t& options,
                      std::vector<SystemKey>& keys) {
  keys.resize(static_cast<std::size_t>(request.batch_size));
  const NormalizedPolicy policy = normalize_policy(options);
  const bool periodic = request.shifts_enabled || request.response_enabled;
  for (std::int64_t system = 0; system < request.batch_size; ++system) {
    const std::size_t index = static_cast<std::size_t>(system);
    const std::int64_t atom_begin = request.atom_offsets[index];
    const std::int64_t atom_end = request.atom_offsets[index + 1u];
    const std::int64_t point_begin = request.point_offsets[index];
    const std::int64_t point_end = request.point_offsets[index + 1u];
    SystemKey& key = keys[index];
    key.atomic_numbers.assign(request.atomic_numbers.begin() + atom_begin,
                              request.atomic_numbers.begin() + atom_end);
    key.molecular_charge = request.molecular_charges[index];
    key.unpaired_electrons = request.unpaired_electrons[index];
    key.spin_channels = request.spin_channels[index];
    key.point_count = point_end - point_begin;
    key.periodic_enabled = periodic;
    key.compute_flags = options.flags;
    key.maximum_iterations = options.max_scc_iterations;
    key.charge_tolerance = options.charge_tolerance;
    key.energy_tolerance = options.energy_tolerance;
    key.electronic_temperature = options.electronic_temperature;
    key.mixer = policy.mixer;
    key.mixer_history = policy.history;
    key.mixer_damping = policy.damping;
    key.determinism = policy.determinism;
  }
}

struct SystemOutput {
  xtbloom_status_t status = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
  std::int32_t iterations = 0;
  std::uint8_t converged = 0u;
  double energy = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;

  void reset() noexcept {
    status = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
    iterations = 0;
    converged = 0u;
    energy = std::numeric_limits<double>::quiet_NaN();
    forces.clear();
    atomic_charges.clear();
    point_forces.clear();
  }
};

struct SystemExecution {
  explicit SystemExecution(SystemKey value) : key(std::move(value)) {}

  SystemKey key;
  std::vector<std::int64_t> atom_offsets{0, 0};
  std::vector<std::int64_t> point_offsets{0, 0};
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;

  BasisPlan basis;
  IntegralPlan integrals;
  RepulsionPlan repulsion;
  H0Plan h0;
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
  D3Plan d3;
  HalogenPlan halogen;
  ExternalPointChargePlan external;

  std::vector<double> positions;
  std::vector<double> point_positions;
  std::vector<double> point_charges;
  std::vector<double> point_hardnesses;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response;
  std::vector<double> coordination;
  std::vector<double> overlap;
  std::vector<double> hamiltonian;
  std::vector<double> es2_matrix;
  std::vector<double> es2_matrix_scratch;
  std::vector<double> point_shell_potential;

  AlignedBuffer integral_storage;
  AlignedBuffer wavefunction_storage;
  AlignedBuffer checkpoint_storage;
  AlignedBuffer overlap_cache_storage;
  AlignedBuffer eigensolver_workspace_storage;
  AlignedBuffer mixer_state_storage;
  AlignedBuffer driver_state_storage;
  AlignedBuffer driver_workspace_storage;
  AlignedBuffer d3_workspace_storage;
  AlignedBuffer halogen_workspace_storage;

  WavefunctionView wavefunction;
  EigensolverOverlapCache overlap_cache;
  EigensolverWorkspace eigensolver_workspace;
  SccMixerState mixer_state;
  SccDriverState driver_state;
  SccDriverWorkspace driver_workspace;
  D3Workspace d3_workspace;
  HalogenWorkspace halogen_workspace;
  ES2GeometryCache es2_cache;
  SccDriverGeometryView geometry;

  std::vector<double> stationary_density;
  std::vector<double> stationary_weighted_density;
  std::vector<double> stationary_spin_density;
  std::vector<double> stationary_shell_charges;
  std::vector<double> stationary_atomic_charges;
  std::vector<double> stationary_scalar_potential;
  std::vector<double> stationary_spin_potential;

  std::vector<double> energy_scratch;
  std::vector<double> component_energy_scratch;
  std::vector<double> total_gradient;
  std::vector<double> component_gradient;
  std::vector<double> component_staging;
  std::vector<double> force_scratch;
  std::vector<double> point_force_scratch;
  std::vector<double> overlap_adjoint;
  std::vector<double> coordination_adjoint;
  std::vector<double> es2_gradient_scratch;
  ForceWorkspace force_workspace;

  std::uint64_t geometry_generation = 0u;
  bool checkpoint_valid = false;

  xtbloom_status_t build(std::string& error);
  xtbloom_status_t infer(const CpuLinearAlgebraBackend& backend, const double* input_positions,
                         const double* input_point_positions, const double* input_point_charges,
                         const double* input_point_hardnesses, const double* input_shifts,
                         const double* input_response, bool warm_start, SystemOutput& output,
                         std::string& error);
  void promote_checkpoint() noexcept {
    std::memcpy(checkpoint_storage.data(), wavefunction_storage.data(),
                wavefunction_storage.size());
    checkpoint_valid = true;
  }
  void invalidate_checkpoint() noexcept { checkpoint_valid = false; }
  [[nodiscard]] std::size_t resident_bytes() const noexcept;
};

xtbloom_status_t SystemExecution::build(std::string& error) {
  const std::int64_t atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
  atom_offsets[1] = atoms;
  point_offsets[1] = key.point_count;
  molecular_charges = {key.molecular_charge};
  unpaired_electrons = {key.unpaired_electrons};
  spin_channels = {key.spin_channels};

  xtbloom_status_t status =
      make_basis_plan(1, atoms, atom_offsets.data(), key.atomic_numbers.data(), basis, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if ((status = make_integral_plan(basis, integrals, error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = make_repulsion_plan(1, atoms, atom_offsets.data(), key.atomic_numbers.data(),
                                    repulsion, error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = make_h0_plan(basis, integrals, key.atomic_numbers.data(), h0, error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = make_wavefunction_layout(basis, key.atomic_numbers.data(), molecular_charges.data(),
                                         unpaired_electrons.data(), spin_channels.data(),
                                         wavefunction_layout, error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = make_mulliken_plan(basis, integrals, wavefunction_layout, mulliken, error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = gfn1::make_es2_plan(basis, key.atomic_numbers.data(), es2, error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = make_es3_plan(basis, key.atomic_numbers.data(), es3, error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = make_spin_population_layout(basis, spin_channels.data(), spin_layout, error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = make_spin_polarization_plan(basis, key.atomic_numbers.data(), spin_layout, spin,
                                            error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = gfn2::make_eigensolver_plan(
           make_eigensolver_wavefunction_layout(wavefunction_layout), eigensolver, error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = make_scc_mixer_plan(wavefunction_layout, key.mixer_history, key.mixer_damping,
                                    key.charge_tolerance, key.charge_tolerance, mixer, error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = make_d3_plan(1, atoms, atom_offsets.data(), key.atomic_numbers.data(), d3,
                             error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = make_halogen_plan(1, atoms, atom_offsets.data(), key.atomic_numbers.data(), halogen,
                                  error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = make_external_point_charge_plan(
           basis, es2, key.point_count, key.point_count == 0 ? nullptr : point_offsets.data(),
           external, error)) != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (key.periodic_enabled &&
      (status = gfn2::make_periodic_embedding_plan(1, atoms, atom_offsets.data(), periodic,
                                                   error)) != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = make_scc_driver_plan(wavefunction_layout, mulliken, es2, es3, spin, eigensolver, mixer,
                                key.periodic_enabled ? &periodic : nullptr,
                                static_cast<std::uint64_t>(key.maximum_iterations),
                                key.electronic_temperature, key.energy_tolerance, driver, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  const std::size_t atom_count = static_cast<std::size_t>(atoms);
  const std::size_t point_count = static_cast<std::size_t>(key.point_count);
  const std::size_t shell_count = static_cast<std::size_t>(basis.total_shells);
  const std::size_t matrix_count = static_cast<std::size_t>(integrals.total_matrix_elements);
  positions.resize(3u * atom_count);
  point_positions.resize(3u * point_count);
  point_charges.resize(point_count);
  point_hardnesses.resize(point_count);
  periodic_shifts.assign(atom_count, 0.0);
  periodic_response.assign(atom_count * atom_count, 0.0);
  coordination.resize(atom_count);
  overlap.resize(matrix_count);
  hamiltonian.resize(matrix_count);
  es2_matrix.resize(static_cast<std::size_t>(es2.total_matrix_elements()));
  es2_matrix_scratch.resize(es2_matrix.size());
  point_shell_potential.resize(shell_count);

  if ((status = allocate(integral_storage, integrals.workspace_size_bytes, "integral workspace",
                         error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = allocate(wavefunction_storage, wavefunction_layout.workspace_size_bytes,
                         "wavefunction", error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = allocate(checkpoint_storage, wavefunction_layout.workspace_size_bytes,
                         "warm checkpoint", error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = allocate(overlap_cache_storage, eigensolver.overlap_cache_size_bytes(),
                         "overlap cache", error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = allocate(eigensolver_workspace_storage, eigensolver.workspace_size_bytes(),
                         "eigensolver workspace", error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = allocate(mixer_state_storage, mixer.state_size_bytes(), "mixer state", error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = allocate(driver_state_storage, driver.state_size_bytes(), "driver state", error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = allocate(driver_workspace_storage, driver.workspace_size_bytes(),
                         "driver workspace", error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = allocate(d3_workspace_storage, d3.workspace_size_bytes(), "D3 workspace", error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = allocate(halogen_workspace_storage, halogen.workspace_size_bytes(),
                         "halogen workspace", error)) != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if ((status = bind_wavefunction_view(wavefunction_layout, wavefunction_storage.data(),
                                       wavefunction_storage.size(), wavefunction, error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = gfn2::bind_eigensolver_overlap_cache(eigensolver, overlap_cache_storage.data(),
                                                     overlap_cache_storage.size(), overlap_cache,
                                                     error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = gfn2::bind_eigensolver_workspace(
           eigensolver, eigensolver_workspace_storage.data(), eigensolver_workspace_storage.size(),
           eigensolver_workspace, error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = bind_scc_mixer_state(mixer, mixer_state_storage.data(), mixer_state_storage.size(),
                                     mixer_state, error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = bind_scc_driver_state(driver, driver_state_storage.data(),
                                      driver_state_storage.size(), driver_state, error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = bind_scc_driver_workspace(driver, driver_workspace_storage.data(),
                                          driver_workspace_storage.size(), driver_workspace,
                                          error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = bind_d3_workspace(d3, d3_workspace_storage.data(), d3_workspace_storage.size(),
                                  d3_workspace, error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = bind_halogen_workspace(halogen, halogen_workspace_storage.data(),
                                       halogen_workspace_storage.size(), halogen_workspace,
                                       error)) != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }

  stationary_density.resize(matrix_count);
  stationary_weighted_density.resize(matrix_count);
  stationary_spin_density.resize(key.spin_channels == 2 ? matrix_count : 0u);
  stationary_shell_charges.resize(shell_count);
  stationary_atomic_charges.resize(atom_count);
  stationary_scalar_potential.resize(shell_count);
  stationary_spin_potential.resize(key.spin_channels == 2 ? shell_count : 0u);
  energy_scratch.resize(1u);
  component_energy_scratch.resize(1u);
  total_gradient.resize(3u * atom_count);
  component_gradient.resize(3u * atom_count);
  component_staging.resize(18u * atom_count);
  force_scratch.resize(3u * atom_count);
  point_force_scratch.resize(3u * point_count);
  overlap_adjoint.resize(matrix_count);
  coordination_adjoint.resize(atom_count);
  es2_gradient_scratch.resize(3u * atom_count);
  force_workspace = {energy_scratch.data(),
                     component_energy_scratch.data(),
                     1,
                     total_gradient.data(),
                     component_gradient.data(),
                     component_staging.data(),
                     static_cast<std::int64_t>(component_staging.size()),
                     force_scratch.data(),
                     atoms * 3,
                     overlap_adjoint.data(),
                     static_cast<std::int64_t>(overlap_adjoint.size()),
                     coordination_adjoint.data(),
                     atoms,
                     point_force_scratch.empty() ? nullptr : point_force_scratch.data(),
                     static_cast<std::int64_t>(point_force_scratch.size()),
                     integral_storage.data(),
                     integral_storage.size(),
                     {nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0},
                     d3_workspace,
                     halogen_workspace};
  force_workspace.es2_workspace.gradient_scratch = es2_gradient_scratch.data();
  force_workspace.es2_workspace.gradient_elements = atoms * 3;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t SystemExecution::infer(const CpuLinearAlgebraBackend& backend,
                                        const double* input_positions,
                                        const double* input_point_positions,
                                        const double* input_point_charges,
                                        const double* input_point_hardnesses,
                                        const double* input_shifts, const double* input_response,
                                        bool warm_start, SystemOutput& output, std::string& error) {
  std::copy_n(input_positions, positions.size(), positions.data());
  if (!point_positions.empty()) {
    std::copy_n(input_point_positions, point_positions.size(), point_positions.data());
    std::copy_n(input_point_charges, point_charges.size(), point_charges.data());
    std::copy_n(input_point_hardnesses, point_hardnesses.size(), point_hardnesses.data());
  }
  if (key.periodic_enabled) {
    if (input_shifts == nullptr) {
      std::fill(periodic_shifts.begin(), periodic_shifts.end(), 0.0);
    } else {
      std::copy_n(input_shifts, periodic_shifts.size(), periodic_shifts.data());
    }
    if (input_response == nullptr) {
      std::fill(periodic_response.begin(), periodic_response.end(), 0.0);
    } else {
      std::copy_n(input_response, periodic_response.size(), periodic_response.data());
    }
  }

  ++geometry_generation;
  if (geometry_generation == 0u) geometry_generation = 1u;
  xtbloom_status_t status = evaluate_coordination_cpu(d3.coordination_plan(), positions.data(),
                                                      coordination.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if ((status = evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(),
                                     integral_storage.data(), integral_storage.size(), error)) !=
          XTBLOOM_STATUS_SUCCESS ||
      (status = evaluate_h0_cpu(basis, integrals, h0, positions.data(), coordination.data(),
                                overlap.data(), hamiltonian.data(), error)) !=
          XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  ES2Workspace es2_update;
  es2_update.matrix_scratch = es2_matrix_scratch.data();
  es2_update.matrix_elements = es2.total_matrix_elements();
  status =
      update_es2_geometry_cache_cpu(es2, positions.data(), geometry_generation, es2_matrix.data(),
                                    es2_matrix.size(), es2_update, es2_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = evaluate_external_point_charge_potential_cpu(
      external, positions.data(), point_positions.empty() ? nullptr : point_positions.data(),
      point_charges.empty() ? nullptr : point_charges.data(),
      point_hardnesses.empty() ? nullptr : point_hardnesses.data(), point_shell_potential.data(),
      error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = gfn2::factor_overlap_cpu(eigensolver, overlap.data(), geometry_generation, backend,
                                    eigensolver_workspace, overlap_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  if (warm_start) {
    if (!checkpoint_valid) {
      error = "internal CPU GFN1 WARM execution found no per-system checkpoint";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(wavefunction_storage.data(), checkpoint_storage.data(),
                wavefunction_storage.size());
  } else {
    status = initialize_sad_multipole_state(wavefunction_layout, wavefunction, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  status = initialize_scc_driver_state_cpu(driver, wavefunction, mixer_state, driver_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  geometry = {};
  geometry.h0 = hamiltonian.data();
  geometry.h0_elements = integrals.total_matrix_elements;
  geometry.integrals = {overlap.data(), integrals.total_matrix_elements, mulliken.identity()};
  geometry.es2_cache = es2_cache;
  geometry.geometry_generation = geometry_generation;
  if (key.point_count != 0) {
    geometry.explicit_point_charge_shell_potential = point_shell_potential.data();
    geometry.explicit_point_charge_shell_elements = wavefunction_layout.total_shells;
  }
  if (key.periodic_enabled) {
    geometry.periodic_shifts = periodic_shifts.data();
    geometry.periodic_shift_elements = static_cast<std::int64_t>(periodic_shifts.size());
    geometry.periodic_response_matrices = periodic_response.data();
    geometry.periodic_response_elements = static_cast<std::int64_t>(periodic_response.size());
    geometry.periodic_embedding_generation = geometry_generation;
    geometry.periodic_plan_identity = periodic.identity();
  }

  while (driver_state.converged[0] == 0u &&
         driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS) {
    status = iterate_scc_driver_batch_cpu(driver, geometry, backend, overlap_cache, wavefunction,
                                          mixer_state, driver_state, driver_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) break;
  }
  output.iterations = static_cast<std::int32_t>(std::min<std::uint64_t>(
      driver_state.iterations[0],
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())));
  if (status != XTBLOOM_STATUS_SUCCESS || driver_state.converged[0] == 0u) {
    output.status = status == XTBLOOM_STATUS_SUCCESS ? driver_state.system_statuses[0] : status;
    return output.status;
  }

  /* Rebuild the exact converged charge, spin, point, and periodic potentials
   * without advancing SCC. The model-owned seam is also responsible for ES3
   * atom-to-shell broadcasting, so the executor cannot drift from iteration
   * assembly when a new scalar term is added. */
  status = rebuild_scc_stationary_potentials_cpu(driver, geometry, wavefunction, driver_workspace,
                                                 error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  const SccStationaryProjection projection{
      stationary_density.data(),
      stationary_weighted_density.data(),
      stationary_spin_density.empty() ? nullptr : stationary_spin_density.data(),
      integrals.total_matrix_elements,
      stationary_shell_charges.data(),
      stationary_atomic_charges.data(),
      stationary_scalar_potential.data(),
      stationary_spin_potential.empty() ? nullptr : stationary_spin_potential.data(),
      basis.total_shells,
      basis.total_atoms};
  status = project_scc_stationary_state_cpu(
      wavefunction_layout, wavefunction, driver_workspace.shell_potentials,
      wavefunction_layout.qsh.element_count, projection, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  output.atomic_charges.assign(stationary_atomic_charges.begin(), stationary_atomic_charges.end());
  const bool need_energy_or_force =
      (key.compute_flags & (XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                            XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
  if (need_energy_or_force) {
    const bool need_qm_force = (key.compute_flags & XTBLOOM_COMPUTE_FORCES) != 0u;
    const bool need_point_force =
        (key.compute_flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0u && key.point_count != 0;
    const bool compose_force = need_qm_force || need_point_force;
    output.forces.assign(compose_force ? positions.size() : 0u, 0.0);
    output.point_forces.assign(need_point_force ? point_positions.size() : 0u, 0.0);
    const StationaryInput input{
        key.atomic_numbers.data(),
        positions.data(),
        coordination.data(),
        geometry_generation,
        overlap.data(),
        stationary_density.data(),
        stationary_weighted_density.data(),
        stationary_shell_charges.data(),
        stationary_scalar_potential.data(),
        driver_state.free_energies,
        stationary_spin_density.empty() ? nullptr : stationary_spin_density.data(),
        stationary_spin_potential.empty() ? nullptr : stationary_spin_potential.data(),
        point_positions.empty() ? nullptr : point_positions.data(),
        point_charges.empty() ? nullptr : point_charges.data(),
        point_hardnesses.empty() ? nullptr : point_hardnesses.data()};
    status = evaluate_gfn1_energy_forces_cpu(
        basis, integrals, d3.coordination_plan(), repulsion, h0, mulliken, es2, es2_cache, d3,
        halogen, key.point_count == 0 ? nullptr : &external, input, &output.energy,
        compose_force ? output.forces.data() : nullptr,
        need_point_force ? output.point_forces.data() : nullptr, {}, force_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  output.status = XTBLOOM_STATUS_SUCCESS;
  output.converged = 1u;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

std::size_t SystemExecution::resident_bytes() const noexcept {
  const std::size_t small_vectors = vector_bytes(key.atomic_numbers) + vector_bytes(atom_offsets) +
                                    vector_bytes(point_offsets) + vector_bytes(molecular_charges) +
                                    vector_bytes(unpaired_electrons) + vector_bytes(spin_channels);
  const std::size_t direct_plan_vectors =
      common::basis_plan_resident_bytes(basis) + vector_bytes(integrals.matrix_offsets) +
      vector_bytes(repulsion.atom_offsets) + vector_bytes(repulsion.sqrt_alpha) +
      vector_bytes(repulsion.effective_charge) + vector_bytes(h0.atom_offsets) +
      vector_bytes(h0.batch_shell_offsets) + vector_bytes(h0.batch_orbital_offsets) +
      vector_bytes(h0.matrix_offsets) + vector_bytes(h0.shell_pair_offsets) +
      vector_bytes(h0.atomic_radii) + vector_bytes(h0.shell_levels) +
      vector_bytes(h0.shell_coordination_scale) + vector_bytes(h0.shell_polynomial) +
      vector_bytes(h0.shell_pair_scale) + vector_bytes(es3.atom_offsets) +
      vector_bytes(es3.atom_gamma3) + vector_bytes(spin_layout.system_offsets) +
      vector_bytes(spin_layout.spin_channels) + vector_bytes(spin.atom_offsets) +
      vector_bytes(spin.batch_shell_offsets) + vector_bytes(spin.atom_shell_offsets) +
      vector_bytes(spin.shell_population_offsets) + vector_bytes(spin.spin_channels) +
      vector_bytes(spin.coupling_offsets) + vector_bytes(spin.coupling_matrices) +
      vector_bytes(external.atom_offsets) + vector_bytes(external.batch_shell_offsets) +
      vector_bytes(external.atom_shell_offsets) + vector_bytes(external.point_charge_offsets) +
      vector_bytes(external.atom_to_batch) + vector_bytes(external.point_to_batch) +
      vector_bytes(external.shell_to_atom) + vector_bytes(external.shell_hardness);
  const std::size_t wavefunction_plan_vectors =
      vector_bytes(wavefunction_layout.atom_offsets) +
      vector_bytes(wavefunction_layout.batch_shell_offsets) +
      vector_bytes(wavefunction_layout.batch_orbital_offsets) +
      vector_bytes(wavefunction_layout.atomic_numbers) +
      vector_bytes(wavefunction_layout.molecular_charges) +
      vector_bytes(wavefunction_layout.unpaired_electrons) +
      vector_bytes(wavefunction_layout.spin_channels) +
      vector_bytes(wavefunction_layout.reference_atom_occupations) +
      vector_bytes(wavefunction_layout.reference_shell_occupations) +
      vector_bytes(wavefunction_layout.electron_counts) +
      vector_bytes(wavefunction_layout.alpha_electron_counts) +
      vector_bytes(wavefunction_layout.beta_electron_counts) +
      vector_bytes(wavefunction_layout.coefficients.system_offsets) +
      vector_bytes(wavefunction_layout.eigenvalues.system_offsets) +
      vector_bytes(wavefunction_layout.occupations.system_offsets) +
      vector_bytes(wavefunction_layout.density.system_offsets) +
      vector_bytes(wavefunction_layout.qsh.system_offsets) +
      vector_bytes(wavefunction_layout.qat.system_offsets) +
      vector_bytes(wavefunction_layout.energy_weighted_density.system_offsets);
  const std::size_t opaque_plan_storage =
      d3.resident_bytes() + halogen.resident_bytes() + es2.resident_bytes() +
      eigensolver.resident_bytes() + mulliken.resident_bytes() + mixer.resident_bytes() +
      driver.resident_bytes() + (periodic.sealed() ? periodic.resident_bytes() : 0u);
  const std::vector<const std::vector<double>*> doubles{&positions,
                                                        &point_positions,
                                                        &point_charges,
                                                        &point_hardnesses,
                                                        &periodic_shifts,
                                                        &periodic_response,
                                                        &coordination,
                                                        &overlap,
                                                        &hamiltonian,
                                                        &es2_matrix,
                                                        &es2_matrix_scratch,
                                                        &point_shell_potential,
                                                        &stationary_density,
                                                        &stationary_weighted_density,
                                                        &stationary_spin_density,
                                                        &stationary_shell_charges,
                                                        &stationary_atomic_charges,
                                                        &stationary_scalar_potential,
                                                        &stationary_spin_potential,
                                                        &energy_scratch,
                                                        &component_energy_scratch,
                                                        &total_gradient,
                                                        &component_gradient,
                                                        &component_staging,
                                                        &force_scratch,
                                                        &point_force_scratch,
                                                        &overlap_adjoint,
                                                        &coordination_adjoint,
                                                        &es2_gradient_scratch};
  std::size_t planar_vectors = 0u;
  for (const auto* values : doubles) planar_vectors += vector_bytes(*values);
  const std::size_t aligned_buffers =
      integral_storage.size() + wavefunction_storage.size() + checkpoint_storage.size() +
      overlap_cache_storage.size() + eigensolver_workspace_storage.size() +
      mixer_state_storage.size() + driver_state_storage.size() + driver_workspace_storage.size() +
      d3_workspace_storage.size() + halogen_workspace_storage.size();
  return small_vectors + direct_plan_vectors + wavefunction_plan_vectors + opaque_plan_storage +
         planar_vectors + aligned_buffers;
}

}  // namespace

struct Gfn1CpuExecutionCache::Impl {
  enum class TaskFailure : std::uint8_t { kNone, kAllocation, kException, kUnknown };

  explicit Impl(std::int32_t requested_threads)
      : cpu_threads(resolve_cpu_threads(requested_threads)), workers(cpu_threads) {}

  ~Impl() {
    /* Release owner-thread state before the backend namespace is destroyed.
     * Persistent workers execute the matching callback from worker_loop. */
    backend.release_thread_resources();
  }

  std::mutex mutex;
  CpuLinearAlgebraBackend backend;
  bool backend_initialized = false;
  HostRequest request;
  std::vector<SystemKey> keys;
  std::vector<SystemKey> requested_keys;
  std::vector<std::unique_ptr<SystemExecution>> systems;
  std::vector<SystemOutput> outputs;
  std::vector<std::string> system_errors;
  std::vector<xtbloom_status_t> inference_statuses;
  std::vector<TaskFailure> task_failures;
  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<std::int32_t> statuses;
  bool systems_ready_for_warm = false;
  const std::size_t cpu_threads;
  CpuWorkerPool workers;

  static void release_backend_thread_resources(void* backend_context) noexcept {
    static_cast<CpuLinearAlgebraBackend*>(backend_context)->release_thread_resources();
  }

  xtbloom_status_t ensure_backend(std::string& error) {
    if (backend_initialized) return XTBLOOM_STATUS_SUCCESS;
    const xtbloom_status_t status = gfn2::make_mkl_rt_lp64_backend(backend, error);
    if (status == XTBLOOM_STATUS_SUCCESS) {
      if (backend.production_mkl_isolated()) {
        workers.set_thread_cleanup(&backend, &release_backend_thread_resources);
      }
      backend_initialized = true;
    }
    return status;
  }

  xtbloom_status_t ensure_systems(const std::vector<SystemKey>& requested, std::string& error) {
    if (requested == keys) return XTBLOOM_STATUS_SUCCESS;
    std::vector<std::unique_ptr<SystemExecution>> candidate;
    candidate.reserve(requested.size());
    for (const SystemKey& key : requested) {
      auto system = std::make_unique<SystemExecution>(key);
      const xtbloom_status_t status = system->build(error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
      candidate.push_back(std::move(system));
    }
    systems = std::move(candidate);
    keys = requested;
    systems_ready_for_warm = false;
    return XTBLOOM_STATUS_SUCCESS;
  }

  void prepare_staging(std::uint32_t flags) {
    const std::size_t batch = static_cast<std::size_t>(request.batch_size);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    outputs.resize(batch);
    for (std::size_t index = 0u; index < batch; ++index) {
      const std::size_t atoms =
          static_cast<std::size_t>(request.atom_offsets[index + 1u] - request.atom_offsets[index]);
      const std::size_t points = static_cast<std::size_t>(request.point_offsets[index + 1u] -
                                                          request.point_offsets[index]);
      outputs[index].forces.reserve(3u * atoms);
      outputs[index].atomic_charges.reserve(atoms);
      outputs[index].point_forces.reserve(3u * points);
    }
    system_errors.resize(batch);
    inference_statuses.assign(batch, XTBLOOM_STATUS_INTERNAL_ERROR);
    task_failures.assign(batch, TaskFailure::kNone);
    energies.assign((flags & XTBLOOM_COMPUTE_ENERGY) != 0u ? batch : 0u, nan);
    forces.assign((flags & XTBLOOM_COMPUTE_FORCES) != 0u
                      ? 3u * static_cast<std::size_t>(request.total_atoms)
                      : 0u,
                  nan);
    atomic_charges.assign((flags & XTBLOOM_COMPUTE_ATOMIC_CHARGES) != 0u
                              ? static_cast<std::size_t>(request.total_atoms)
                              : 0u,
                          nan);
    point_forces.assign((flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0u
                            ? 3u * static_cast<std::size_t>(request.total_point_charges)
                            : 0u,
                        nan);
    iterations.assign(batch, 0);
    converged.assign(batch, 0u);
    statuses.assign(batch, XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  }

  struct InferenceJob {
    Impl& owner;
    const xtbloom_compute_options_t& options;
  };

  static void infer_system(void* opaque_job, std::size_t index) noexcept {
    InferenceJob& job = *static_cast<InferenceJob*>(opaque_job);
    Impl& owner = job.owner;
    const HostRequest& request = owner.request;
    SystemOutput& output = owner.outputs[index];
    std::string& system_error = owner.system_errors[index];
    output.reset();
    system_error.clear();
    const std::int64_t atom_begin = request.atom_offsets[index];
    const std::int64_t point_begin = request.point_offsets[index];
    const std::int64_t points = request.point_offsets[index + 1u] - point_begin;
    const double* shifts =
        request.shifts_enabled ? request.periodic_shifts.data() + atom_begin : nullptr;
    const double* response = request.response_enabled ? request.response_matrices.data() +
                                                            request.response_offsets[index]
                                                      : nullptr;
    try {
      const bool warm = job.options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE &&
                        job.options.scc_start_mode == XTBLOOM_SCC_START_WARM;
      owner.inference_statuses[index] = owner.systems[index]->infer(
          owner.backend, request.positions.data() + 3 * atom_begin,
          points == 0 ? nullptr : request.point_positions.data() + 3 * point_begin,
          points == 0 ? nullptr : request.point_charges.data() + point_begin,
          points == 0 ? nullptr : request.point_hardnesses.data() + point_begin, shifts, response,
          warm, output, system_error);
    } catch (const std::bad_alloc&) {
      owner.inference_statuses[index] = XTBLOOM_STATUS_ALLOCATION_FAILED;
      owner.task_failures[index] = TaskFailure::kAllocation;
      system_error.clear();
    } catch (const std::exception& exception) {
      owner.inference_statuses[index] = XTBLOOM_STATUS_INTERNAL_ERROR;
      owner.task_failures[index] = TaskFailure::kException;
      try {
        system_error = exception.what();
      } catch (...) {
        system_error.clear();
      }
    } catch (...) {
      owner.inference_statuses[index] = XTBLOOM_STATUS_INTERNAL_ERROR;
      owner.task_failures[index] = TaskFailure::kUnknown;
      system_error.clear();
    }
  }
};

Gfn1CpuExecutionCache::Gfn1CpuExecutionCache(std::int32_t cpu_threads)
    : impl_(std::make_unique<Impl>(cpu_threads)) {}
Gfn1CpuExecutionCache::~Gfn1CpuExecutionCache() = default;

xtbloom_status_t set_gfn1_cpu_linear_algebra_backend_for_testing(
    Gfn1CpuExecutionCache& cache, const gfn2::CpuLinearAlgebraBackend& backend,
    std::string& error) {
  std::lock_guard<std::mutex> lock(cache.impl_->mutex);
  if (cache.impl_->backend_initialized || !backend.ready() || backend.production()) {
    error = "internal CPU GFN1 test backend must be verified and installed before use";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  cache.impl_->backend = backend;
  cache.impl_->backend_initialized = true;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t prepare_gfn1_cpu(Gfn1CpuExecutionCache& cache, const xtbloom_batch_t& batch,
                                  const xtbloom_compute_options_t& options, bool& reused,
                                  std::string& error) {
  reused = false;
  try {
    std::lock_guard<std::mutex> lock(cache.impl_->mutex);
    stage_request(batch, cache.impl_->request);
    xtbloom_status_t status = validate_hidden_request(batch, options, cache.impl_->request, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    make_system_keys(cache.impl_->request, options, cache.impl_->requested_keys);
    reused = cache.impl_->requested_keys == cache.impl_->keys;
    if ((status = cache.impl_->ensure_backend(error)) != XTBLOOM_STATUS_SUCCESS ||
        (status = cache.impl_->ensure_systems(cache.impl_->requested_keys, error)) !=
            XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
    cache.impl_->prepare_staging(options.flags);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate internal CPU GFN1 plan state";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t execute_gfn1_cpu(Gfn1CpuExecutionCache& cache, const xtbloom_batch_t& batch,
                                  const xtbloom_compute_options_t& options,
                                  xtbloom_batch_result_t& result, std::string& error) {
  try {
    std::lock_guard<std::mutex> lock(cache.impl_->mutex);
    Gfn1CpuExecutionCache::Impl& implementation = *cache.impl_;
    stage_request(batch, implementation.request);
    xtbloom_status_t status =
        validate_hidden_request(batch, options, implementation.request, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    make_system_keys(implementation.request, options, implementation.requested_keys);
    const bool warm = options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE &&
                      options.scc_start_mode == XTBLOOM_SCC_START_WARM;
    if (warm && (implementation.requested_keys != implementation.keys ||
                 !implementation.systems_ready_for_warm)) {
      error = "internal CPU GFN1 WARM start requires one compatible fully composed batch";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if ((status = implementation.ensure_backend(error)) != XTBLOOM_STATUS_SUCCESS) return status;
    implementation.prepare_staging(options.flags);
    if ((status = implementation.ensure_systems(implementation.requested_keys, error)) !=
        XTBLOOM_STATUS_SUCCESS) {
      return status;
    }

    implementation.systems_ready_for_warm = false;
    Gfn1CpuExecutionCache::Impl::InferenceJob job{implementation, options};
    implementation.workers.parallel_for(static_cast<std::size_t>(implementation.request.batch_size),
                                        &job, &Gfn1CpuExecutionCache::Impl::infer_system);
    bool all_composed = true;
    for (std::int64_t system = 0; system < implementation.request.batch_size; ++system) {
      const std::size_t index = static_cast<std::size_t>(system);
      SystemOutput& output = implementation.outputs[index];
      const std::int64_t atom_begin = implementation.request.atom_offsets[index];
      const std::int64_t point_begin = implementation.request.point_offsets[index];
      const std::int64_t points = implementation.request.point_offsets[index + 1u] - point_begin;
      status = implementation.inference_statuses[index];
      implementation.iterations[index] = output.iterations;
      if (status == XTBLOOM_STATUS_SCC_NOT_CONVERGED ||
          status == XTBLOOM_STATUS_EIGENSOLVER_FAILED) {
        implementation.statuses[index] = status;
        implementation.systems[index]->invalidate_checkpoint();
        all_composed = false;
        continue;
      }
      if (status != XTBLOOM_STATUS_SUCCESS) {
        for (auto& retained : implementation.systems) retained->invalidate_checkpoint();
        if (!implementation.system_errors[index].empty()) {
          error = implementation.system_errors[index];
        } else if (implementation.task_failures[index] ==
                   Gfn1CpuExecutionCache::Impl::TaskFailure::kAllocation) {
          error = "failed to allocate CPU GFN1 per-system inference state";
        } else if (implementation.task_failures[index] ==
                   Gfn1CpuExecutionCache::Impl::TaskFailure::kUnknown) {
          error = "unknown exception while executing a CPU GFN1 batch member";
        } else {
          error = "CPU GFN1 batch member failed without a diagnostic";
        }
        return status;
      }
      implementation.statuses[index] = XTBLOOM_STATUS_SUCCESS;
      implementation.converged[index] = 1u;
      if ((options.flags & XTBLOOM_COMPUTE_ENERGY) != 0u)
        implementation.energies[index] = output.energy;
      if ((options.flags & XTBLOOM_COMPUTE_FORCES) != 0u)
        std::copy(output.forces.begin(), output.forces.end(),
                  implementation.forces.begin() + 3 * atom_begin);
      if ((options.flags & XTBLOOM_COMPUTE_ATOMIC_CHARGES) != 0u)
        std::copy(output.atomic_charges.begin(), output.atomic_charges.end(),
                  implementation.atomic_charges.begin() + atom_begin);
      if ((options.flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0u && points != 0)
        std::copy(output.point_forces.begin(), output.point_forces.end(),
                  implementation.point_forces.begin() + 3 * point_begin);
    }

    if (all_composed) {
      for (auto& system : implementation.systems) system->promote_checkpoint();
      implementation.systems_ready_for_warm = true;
    } else {
      for (auto& system : implementation.systems) system->invalidate_checkpoint();
    }
    if ((options.flags & XTBLOOM_COMPUTE_ENERGY) != 0u)
      publish_to_buffer(implementation.energies, result.energies);
    if ((options.flags & XTBLOOM_COMPUTE_FORCES) != 0u)
      publish_to_buffer(implementation.forces, result.forces);
    if ((options.flags & XTBLOOM_COMPUTE_ATOMIC_CHARGES) != 0u)
      publish_to_buffer(implementation.atomic_charges, result.atomic_charges);
    if ((options.flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0u)
      publish_to_buffer(implementation.point_forces, result.point_charge_forces);
    publish_to_buffer(implementation.iterations, result.scc_iterations);
    publish_to_buffer(implementation.converged, result.scc_converged);
    publish_to_buffer(implementation.statuses, result.per_system_status);
    result.flags =
        (implementation.request.shifts_enabled || implementation.request.response_enabled)
            ? XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES
            : 0u;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate internal CPU GFN1 execution staging";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

std::size_t persistent_workspace_bytes_gfn1_cpu(Gfn1CpuExecutionCache& cache) noexcept {
  std::lock_guard<std::mutex> lock(cache.impl_->mutex);
  const Gfn1CpuExecutionCache::Impl& implementation = *cache.impl_;
  std::size_t total = sizeof(Gfn1CpuExecutionCache::Impl) + vector_bytes(implementation.systems);
  for (const auto& system : implementation.systems)
    total += sizeof(SystemExecution) + system->resident_bytes();
  const auto keys_bytes = [](const std::vector<SystemKey>& keys) {
    std::size_t bytes = vector_bytes(keys);
    for (const auto& key : keys) bytes += vector_bytes(key.atomic_numbers);
    return bytes;
  };
  const auto outputs_bytes = [](const std::vector<SystemOutput>& outputs) {
    std::size_t bytes = vector_bytes(outputs);
    for (const SystemOutput& output : outputs) {
      bytes += vector_bytes(output.forces) + vector_bytes(output.atomic_charges) +
               vector_bytes(output.point_forces);
    }
    return bytes;
  };
  total += keys_bytes(implementation.keys) + keys_bytes(implementation.requested_keys) +
           outputs_bytes(implementation.outputs) + vector_bytes(implementation.energies) +
           vector_bytes(implementation.forces) + vector_bytes(implementation.atomic_charges) +
           vector_bytes(implementation.point_forces) + vector_bytes(implementation.iterations) +
           vector_bytes(implementation.converged) + vector_bytes(implementation.statuses) +
           vector_bytes(implementation.system_errors) +
           vector_bytes(implementation.inference_statuses) +
           vector_bytes(implementation.task_failures) + implementation.workers.resident_bytes();
  const HostRequest& request = implementation.request;
  total += vector_bytes(request.atom_offsets) + vector_bytes(request.atomic_numbers) +
           vector_bytes(request.positions) + vector_bytes(request.molecular_charges) +
           vector_bytes(request.unpaired_electrons) + vector_bytes(request.spin_channels) +
           vector_bytes(request.point_offsets) + vector_bytes(request.point_positions) +
           vector_bytes(request.point_charges) + vector_bytes(request.point_hardnesses) +
           vector_bytes(request.periodic_shifts) + vector_bytes(request.response_offsets) +
           vector_bytes(request.response_matrices);
  return total;
}

}  // namespace xtbloom::detail
