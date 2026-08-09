#include "runtime/gfn2_cpu_execution.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
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
#include <string>
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

#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/eigensolver.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/es3.hpp"
#include "model/gfn2/external_point_charges.hpp"
#include "model/gfn2/force.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/periodic_embedding.hpp"
#include "model/gfn2/repulsion.hpp"
#include "model/gfn2/scc_driver.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/spin.hpp"
#include "model/gfn2/wavefunction.hpp"

namespace xtbloom::detail {
namespace {

using namespace xtbloom::detail::gfn2;

constexpr std::size_t kHostAlignment = 64u;
constexpr std::int64_t kMixerHistory = 8;
constexpr double kMixerDamping = 0.4;
constexpr std::size_t kMaximumAutomaticCpuThreads = 64u;

/* ``std::aligned_alloc`` is not provided by MSVC; wrap the platform primitive
 * so AlignedBuffer can allocate and free without per-platform guards. */
void* host_aligned_allocate(std::size_t alignment, std::size_t size) {
#if defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  return std::aligned_alloc(alignment, size);
#endif
}

void host_aligned_free(void* ptr) {
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

std::size_t process_cpu_count() noexcept {
#if defined(__linux__)
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
    std::size_t count = 0u;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &affinity)) {
        ++count;
      }
    }
    if (count != 0u) {
      return count;
    }
  }
#endif
  const unsigned int hardware = std::thread::hardware_concurrency();
  return hardware == 0u ? 1u : static_cast<std::size_t>(hardware);
}

std::size_t resolve_cpu_threads(std::int32_t requested) noexcept {
  const std::size_t available = process_cpu_count();
  if (requested == 0) {
    return std::max<std::size_t>(1u, std::min<std::size_t>(available, kMaximumAutomaticCpuThreads));
  }
  /* A context thread count is a parallelism ceiling, not permission to
   * oversubscribe the process affinity mask. This also bounds hostile but
   * otherwise ABI-valid int32 values without adding a new public failure. */
  return std::max<std::size_t>(
      1u, std::min<std::size_t>(available, static_cast<std::size_t>(requested)));
}

/*
 * Context-owned fixed worker pool for independent systems in one public batch.
 * The calling thread participates, so N requested CPU threads require only
 * N-1 persistent background workers and preserve the batch-one latency path.
 */
class CpuWorkerPool final {
 public:
  using Task = void (*)(void*, std::size_t) noexcept;

  explicit CpuWorkerPool(std::size_t concurrency)
      : concurrency_(std::max<std::size_t>(1u, concurrency)) {
    workers_.reserve(concurrency_ - 1u);
    for (std::size_t worker = 0u; worker + 1u < concurrency_; ++worker) {
      try {
        workers_.emplace_back([this] { worker_loop(); });
      } catch (const std::system_error&) {
        /* Resource limits can be lower than the affinity mask (for example,
         * a host-wide process/thread limit). Retain the workers already
         * created and continue with the smaller fixed pool instead of making
         * an otherwise valid CPU context unusable. */
        break;
      }
    }
    concurrency_ = workers_.size() + 1u;
  }

  ~CpuWorkerPool() { stop_and_join(); }

  CpuWorkerPool(const CpuWorkerPool&) = delete;
  CpuWorkerPool& operator=(const CpuWorkerPool&) = delete;

  void parallel_for(std::size_t task_count, void* context, Task task) noexcept {
    if (task_count == 0u) {
      return;
    }
    if (workers_.empty() || task_count == 1u) {
      for (std::size_t task_index = 0u; task_index < task_count; ++task_index) {
        task(context, task_index);
      }
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

  /* Actual worker count including the calling thread. This can be smaller
   * than the requested thread count when OS resource limits truncated worker
   * creation in the constructor, and it is what chunk sizing must honor. */
  [[nodiscard]] std::size_t concurrency() const noexcept { return concurrency_; }

  [[nodiscard]] std::size_t resident_bytes() const noexcept {
    return workers_.capacity() * sizeof(std::thread);
  }

 private:
  void drain_tasks(Task task, void* context, std::size_t task_count) noexcept {
    for (;;) {
      const std::size_t task_index = next_task_.fetch_add(1u, std::memory_order_relaxed);
      if (task_index >= task_count) {
        return;
      }
      task(context, task_index);
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
      if (worker.joinable()) {
        worker.join();
      }
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
};

class AlignedBuffer {
 public:
  AlignedBuffer() noexcept = default;
  ~AlignedBuffer() { host_aligned_free(data_); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0u)) {}

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      host_aligned_free(data_);
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0u);
    }
    return *this;
  }

  [[nodiscard]] bool allocate(std::size_t requested) noexcept {
    if (data_ != nullptr ||
        requested > std::numeric_limits<std::size_t>::max() - (kHostAlignment - 1u)) {
      return false;
    }
    const std::size_t useful = std::max<std::size_t>(requested, 1u);
    size_ = (useful + kHostAlignment - 1u) & ~(kHostAlignment - 1u);
    data_ = host_aligned_allocate(kHostAlignment, size_);
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

xtbloom_status_t allocate(AlignedBuffer& buffer, std::size_t bytes, const char* purpose,
                          std::string& error) {
  if (buffer.allocate(bytes)) {
    return XTBLOOM_STATUS_SUCCESS;
  }
  error = std::string("failed to allocate CPU GFN2 ") + purpose;
  return XTBLOOM_STATUS_ALLOCATION_FAILED;
}

template <typename T>
void copy_from_c_buffer(const xtbloom_const_buffer_t& source, std::size_t count,
                        std::vector<T>& destination) {
  destination.resize(count);
  if (count != 0u) {
    std::memcpy(destination.data(), source.data, count * sizeof(T));
  }
}

template <typename T>
void publish_to_c_buffer(const std::vector<T>& source, xtbloom_buffer_t& destination) {
  if (!source.empty()) {
    std::memcpy(destination.data, source.data(), source.size() * sizeof(T));
  }
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
  /* Per-system uniform electric field in atomic units (Hartree per elementary
   * charge per bohr), plus a distinct attachment-presence bit. The ABI permits
   * an explicit zero-valued field, which is physically a no-op but remains a
   * different WARM interaction identity from no attachment. */
  std::vector<std::array<double, 3>> field_by_system;
  std::vector<std::uint8_t> field_attached_by_system;
  bool shifts_enabled = false;
  bool response_enabled = false;
};

void stage_electric_fields(const xtbloom_batch_t& batch, HostRequest& request);

void stage_request(const xtbloom_batch_t& batch, HostRequest& request) {
  request.batch_size = batch.batch_size;
  request.total_atoms = batch.total_atoms;
  request.total_point_charges = batch.total_point_charges;
  copy_from_c_buffer(batch.atom_offsets, static_cast<std::size_t>(batch.batch_size) + 1u,
                     request.atom_offsets);
  copy_from_c_buffer(batch.atomic_numbers, static_cast<std::size_t>(batch.total_atoms),
                     request.atomic_numbers);
  copy_from_c_buffer(batch.positions, 3u * static_cast<std::size_t>(batch.total_atoms),
                     request.positions);
  copy_from_c_buffer(batch.molecular_charges, static_cast<std::size_t>(batch.batch_size),
                     request.molecular_charges);
  copy_from_c_buffer(batch.unpaired_electrons, static_cast<std::size_t>(batch.batch_size),
                     request.unpaired_electrons);
  const bool spin_channels_present =
      batch.struct_size >= XTBLOOM_BATCH_V2_SIZE && batch.spin_channels.data != nullptr;
  if (spin_channels_present) {
    copy_from_c_buffer(batch.spin_channels, static_cast<std::size_t>(batch.batch_size),
                       request.spin_channels);
  } else {
    request.spin_channels.assign(static_cast<std::size_t>(batch.batch_size), 1);
  }

  if (batch.total_point_charges != 0) {
    copy_from_c_buffer(batch.point_charge_offsets, static_cast<std::size_t>(batch.batch_size) + 1u,
                       request.point_offsets);
    copy_from_c_buffer(batch.point_charge_positions,
                       3u * static_cast<std::size_t>(batch.total_point_charges),
                       request.point_positions);
    copy_from_c_buffer(batch.point_charge_values,
                       static_cast<std::size_t>(batch.total_point_charges), request.point_charges);
    copy_from_c_buffer(batch.point_charge_gammas,
                       static_cast<std::size_t>(batch.total_point_charges),
                       request.point_hardnesses);
  } else {
    request.point_offsets.assign(static_cast<std::size_t>(batch.batch_size) + 1u, 0);
    request.point_positions.clear();
    request.point_charges.clear();
    request.point_hardnesses.clear();
  }

  request.shifts_enabled = batch.atomic_potential_shifts.data != nullptr &&
                           batch.atomic_potential_shifts.size_bytes != 0u;
  if (request.shifts_enabled) {
    copy_from_c_buffer(batch.atomic_potential_shifts, static_cast<std::size_t>(batch.total_atoms),
                       request.periodic_shifts);
  } else {
    request.periodic_shifts.clear();
  }
  request.response_enabled = batch.total_charge_response_elements != 0;
  if (request.response_enabled) {
    copy_from_c_buffer(batch.charge_response_offsets,
                       static_cast<std::size_t>(batch.batch_size) + 1u, request.response_offsets);
    copy_from_c_buffer(batch.charge_response_matrix,
                       static_cast<std::size_t>(batch.total_charge_response_elements),
                       request.response_matrices);
  } else {
    request.response_offsets.clear();
    request.response_matrices.clear();
  }

  request.field_by_system.assign(static_cast<std::size_t>(batch.batch_size),
                                 std::array<double, 3>{0.0, 0.0, 0.0});
  request.field_attached_by_system.assign(static_cast<std::size_t>(batch.batch_size), 0u);
  if (batch.struct_size >= XTBLOOM_BATCH_V3_SIZE && batch.total_interactions != 0) {
    stage_electric_fields(batch, request);
  }
}

/*
 * Stage ABI-v3 electric-field attachments into the per-system field vectors.
 *
 * The structural host semantics have already validated the descriptor and
 * payload bytes (matching block versions, alignment, finite values, duplicate
 * rejection), so this pass only relocates the released 32-byte payload block
 * into the per-system field vector. Only the electric-field tag is released;
 * every other tag is refused before execution by the validation layer.
 */
void stage_electric_fields(const xtbloom_batch_t& batch, HostRequest& request) {
  const unsigned char* descriptors =
      static_cast<const unsigned char*>(batch.interaction_descriptors.data);
  const unsigned char* payload = static_cast<const unsigned char*>(batch.interaction_payload.data);
  for (std::int64_t index = 0; index < batch.total_interactions; ++index) {
    const unsigned char* descriptor =
        descriptors + static_cast<std::size_t>(index) * sizeof(xtbloom_interaction_t);
    std::int32_t type = 0;
    std::int64_t system_index = 0;
    std::uint64_t payload_offset = 0u;
    std::memcpy(&type, descriptor + offsetof(xtbloom_interaction_t, type), sizeof(type));
    std::memcpy(&system_index, descriptor + offsetof(xtbloom_interaction_t, system_index),
                sizeof(system_index));
    std::memcpy(&payload_offset, descriptor + offsetof(xtbloom_interaction_t, payload_offset),
                sizeof(payload_offset));
    if (type != XTBLOOM_INTERACTION_ELECTRIC_FIELD) {
      /* Validation refuses every other tag before execution; keep this branch
       * defensive and unreachable. */
      continue;
    }
    double field[3] = {0.0, 0.0, 0.0};
    std::memcpy(field,
                payload + static_cast<std::size_t>(payload_offset) + 2u * sizeof(std::int32_t),
                sizeof(field));
    request.field_by_system[static_cast<std::size_t>(system_index)] = {field[0], field[1],
                                                                       field[2]};
    request.field_attached_by_system[static_cast<std::size_t>(system_index)] = 1u;
  }
}

bool all_finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

xtbloom_status_t validate_host_numerics(const HostRequest& request, std::string& error) {
  if (!all_finite(request.positions)) {
    error = "positions contain NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!all_finite(request.molecular_charges)) {
    error = "molecular_charges contain NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!all_finite(request.point_positions) || !all_finite(request.point_charges)) {
    error = "external point-charge positions or values contain NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!std::all_of(request.point_hardnesses.begin(), request.point_hardnesses.end(),
                   [](double value) { return std::isfinite(value) && value > 0.0; })) {
    error = "point_charge_gammas must be finite and positive";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!all_finite(request.periodic_shifts) || !all_finite(request.response_matrices)) {
    error = "periodic b/A inputs contain NaN or infinity";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (request.response_enabled) {
    for (std::int64_t system = 0; system < request.batch_size; ++system) {
      const std::size_t index = static_cast<std::size_t>(system);
      const std::int64_t atoms = request.atom_offsets[index + 1u] - request.atom_offsets[index];
      const std::int64_t begin = request.response_offsets[index];
      for (std::int64_t row = 0; row < atoms; ++row) {
        for (std::int64_t column = row + 1; column < atoms; ++column) {
          const double upper =
              request.response_matrices[static_cast<std::size_t>(begin + row * atoms + column)];
          const double lower =
              request.response_matrices[static_cast<std::size_t>(begin + column * atoms + row)];
          if (upper != lower) {
            error = "charge_response_matrix must be exactly symmetric per system";
            return XTBLOOM_STATUS_INVALID_ARGUMENT;
          }
        }
      }
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

struct SystemKey {
  std::vector<std::int32_t> atomic_numbers;
  double molecular_charge = 0.0;
  std::int32_t unpaired_electrons = 0;
  std::int32_t spin_channels = 1;
  std::int64_t point_count = 0;
  bool periodic_enabled = false;
  /* Uniform external electric field in atomic units. Presence is recorded
   * separately because an explicit zero field remains part of the interaction
   * set and therefore differs from no attachment for strict WARM identity. */
  bool field_attached = false;
  std::array<double, 3> field{0.0, 0.0, 0.0};
  std::uint32_t compute_flags = 0u;
  std::int32_t maximum_iterations = 0;
  double charge_tolerance = 0.0;
  double energy_tolerance = 0.0;
  double electronic_temperature = 0.0;

  friend bool operator==(const SystemKey& lhs, const SystemKey& rhs) {
    return lhs.atomic_numbers == rhs.atomic_numbers &&
           lhs.molecular_charge == rhs.molecular_charge &&
           lhs.unpaired_electrons == rhs.unpaired_electrons &&
           lhs.spin_channels == rhs.spin_channels && lhs.point_count == rhs.point_count &&
           lhs.periodic_enabled == rhs.periodic_enabled &&
           lhs.field_attached == rhs.field_attached && lhs.field == rhs.field &&
           lhs.compute_flags == rhs.compute_flags &&
           lhs.maximum_iterations == rhs.maximum_iterations &&
           lhs.charge_tolerance == rhs.charge_tolerance &&
           lhs.energy_tolerance == rhs.energy_tolerance &&
           lhs.electronic_temperature == rhs.electronic_temperature;
  }
};

void make_system_keys(const HostRequest& request, const xtbloom_compute_options_t& options,
                      std::vector<SystemKey>& keys) {
  keys.resize(static_cast<std::size_t>(request.batch_size));
  const bool periodic_enabled = request.shifts_enabled || request.response_enabled;
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
    key.periodic_enabled = periodic_enabled;
    key.field_attached = request.field_attached_by_system[index] != 0u;
    key.field = request.field_by_system[index];
    key.compute_flags = options.flags;
    key.maximum_iterations = options.max_scc_iterations;
    key.charge_tolerance = options.charge_tolerance;
    key.energy_tolerance = options.energy_tolerance;
    key.electronic_temperature = options.electronic_temperature;
  }
}

/* Field presence and values change only numerical inputs. Every CPU system
 * preallocates field_vat/field_vdp for its full atom count, so these members
 * are excluded from the identity that decides whether rebuilding would be
 * necessary. The exact SystemKey equality above remains the strict WARM gate. */
bool same_prepared_layout(const SystemKey& lhs, const SystemKey& rhs) {
  return lhs.atomic_numbers == rhs.atomic_numbers && lhs.molecular_charge == rhs.molecular_charge &&
         lhs.unpaired_electrons == rhs.unpaired_electrons &&
         lhs.spin_channels == rhs.spin_channels && lhs.point_count == rhs.point_count &&
         lhs.periodic_enabled == rhs.periodic_enabled && lhs.compute_flags == rhs.compute_flags &&
         lhs.maximum_iterations == rhs.maximum_iterations &&
         lhs.charge_tolerance == rhs.charge_tolerance &&
         lhs.energy_tolerance == rhs.energy_tolerance &&
         lhs.electronic_temperature == rhs.electronic_temperature;
}

struct SystemOutput {
  xtbloom_status_t status = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
  std::int32_t iterations = 0;
  std::uint8_t converged = 0u;
  double energy = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  /* Per-system molecular dipole moment (three doubles, atomic units),
   * published when XTBLOOM_COMPUTE_DIPOLE_MOMENTS is requested. */
  std::array<double, 3> dipole_moments{0.0, 0.0, 0.0};

  void reset() noexcept {
    status = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
    iterations = 0;
    converged = 0u;
    energy = std::numeric_limits<double>::quiet_NaN();
    /* clear retains the topology-sized capacity for steady-state calls. */
    forces.clear();
    atomic_charges.clear();
    point_forces.clear();
    dipole_moments = {0.0, 0.0, 0.0};
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
  CoordinationPlan coordination;
  RepulsionPlan repulsion;
  H0Plan h0;
  WavefunctionLayout wavefunction_layout;
  ES2Plan es2;
  ES3Plan es3;
  AES2Plan aes2;
  MullikenPlan mulliken;
  EigensolverPlan eigensolver;
  SccMixerPlan mixer;
  SpinPolarizationPlan spin;
  D4Plan d4;
  bool d4_enabled = false;
  ExternalPointChargePlan external;
  PeriodicEmbeddingPlan periodic;
  SccDriverPlan driver;

  /* Optional intra-system parallel dispatch for a single-system batch. Set by
   * the execution cache only when batch_size == 1, so the calling thread is
   * the sole pool user and the idle background workers can safely execute the
   * per-iteration chunked phases without reentrant pool use. A null executor
   * keeps the serial path exactly. */
  SccParallelExecutor parallel_executor;

  std::vector<double> positions;
  std::vector<double> point_positions;
  std::vector<double> point_charges;
  std::vector<double> point_hardnesses;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response;

  std::vector<double> coordination_numbers;
  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<double> core_hamiltonian;
  AlignedBuffer integral_workspace;

  std::vector<double> es2_matrix;
  std::vector<double> es2_matrix_scratch;
  std::vector<double> es2_shell_scratch;
  std::vector<double> es2_batch_scratch;
  std::vector<double> es2_gradient_scratch;
  ES2Workspace es2_workspace;
  ES2GeometryCache es2_cache;

  std::vector<double> aes2_pairs;
  std::vector<double> aes2_pair_scratch;
  std::vector<double> aes2_potential_scratch;
  std::vector<double> aes2_batch_scratch;
  std::vector<double> aes2_gradient_scratch;
  std::vector<double> aes2_coordination_scratch;
  AES2Workspace aes2_workspace;
  AES2GeometryCache aes2_cache;

  AlignedBuffer d4_workspace_storage;
  D4Workspace d4_workspace;
  std::vector<double> d4_pairs;
  std::vector<double> d4_coordination;
  D4GeometryCache d4_cache;

  std::vector<double> explicit_point_shell_potential;

  AlignedBuffer wavefunction_storage;
  WavefunctionView wavefunction;
  AlignedBuffer overlap_cache_storage;
  EigensolverOverlapCache overlap_cache;
  AlignedBuffer eigensolver_workspace_storage;
  EigensolverWorkspace eigensolver_workspace;
  AlignedBuffer mixer_state_storage;
  SccMixerState mixer_state;
  AlignedBuffer driver_state_storage;
  SccDriverState driver_state;
  AlignedBuffer driver_workspace_storage;
  SccDriverWorkspace driver_workspace;
  SccDriverGeometryView geometry;

  std::vector<double> component_shell_potential;
  std::vector<double> scalar_shell_potential;
  std::vector<double> atomic_potential;
  std::vector<double> d4_atomic_potential;
  std::vector<double> periodic_atomic_potential;
  std::vector<double> dipole_potential;
  std::vector<double> quadrupole_potential;
  std::vector<double> periodic_energy;
  std::vector<xtbloom_status_t> periodic_status;

  /* Uniform external electric field in atomic units. Presence is distinct from
   * the three values so an explicit zero block remains visible to WARM policy.
   * field_vat holds the per-atom scalar potential -E . r and field_vdp the
   * per-atom dipolar potential -E, both recomputed when positions change. */
  bool field_attached = false;
  std::array<double, 3> field{0.0, 0.0, 0.0};
  std::vector<double> field_vat;
  std::vector<double> field_vdp;

  std::vector<double> energy_scratch;
  std::vector<double> component_energy_scratch;
  std::vector<double> total_gradient;
  std::vector<double> component_gradient;
  std::vector<double> force_scratch;
  std::vector<double> point_force_scratch;
  std::vector<double> overlap_adjoint;
  std::vector<double> dipole_adjoint;
  std::vector<double> quadrupole_adjoint;
  std::vector<double> coordination_adjoint;
  std::vector<double> stationary_density;
  std::vector<double> stationary_energy_weighted_density;
  std::vector<double> stationary_spin_density;
  std::vector<double> packed_spin_shell_potential;
  std::vector<double> stationary_spin_shell_potential;
  std::vector<double> spin_energy_scratch;
  RestrictedGfn2ForceWorkspace force_workspace;

  std::uint64_t geometry_generation = 0u;

  /* Strict warm-start checkpoint. A clone of the fully converged wavefunction
   * (coefficients, eigenvalues, occupations, densities, and the qsh/qat/dipole/
   * quadrupole multipoles) taken after a successful inference so the next WARM
   * call can seed SCC from the converged electronic state instead of the SAD
   * guess. The clone is independent of the live wavefunction buffer, which the
   * next FRESH call rewrites with the SAD state. Warm consumption is gated by
   * the whole-batch identity check in the execution cache, so the flag only
   * fails closed against a stale or partial state within the same identity. */
  AlignedBuffer warm_checkpoint_wavefunction_storage;
  bool warm_checkpoint_valid = false;

  xtbloom_status_t build(std::string& error);
  xtbloom_status_t infer(const CpuLinearAlgebraBackend& backend, const double* input_positions,
                         const double* input_point_positions, const double* input_point_charges,
                         const double* input_point_hardnesses, const double* input_shifts,
                         const double* input_response, std::uint32_t compute_flags, bool warm_start,
                         SystemOutput& output, std::string& error);

  /* Steady-state host reservation: every persistent buffer this system owns.
   * Capacities are reported because repeated inference reuses them; the value
   * is topology- and spin-dependent but independent of the requested property
   * flags (all scratch is preallocated up front). */
  std::size_t resident_bytes() const noexcept;

  /* Update only numerical field state. This is noexcept and allocation-free,
   * which lets a fixed plan accept FRESH field changes without rebuilding its
   * topology-sized SystemExecution. */
  void set_field(bool attached, const std::array<double, 3>& value) noexcept {
    key.field_attached = attached;
    key.field = value;
    field_attached = attached;
    field = value;
  }

 private:
  xtbloom_status_t refresh_geometry(const CpuLinearAlgebraBackend& backend, bool warm_start,
                                    std::string& error);
  xtbloom_status_t run_scc(const CpuLinearAlgebraBackend& backend, std::string& error);
  xtbloom_status_t restore_warm_checkpoint(std::string& error);
  xtbloom_status_t refresh_stationary_potentials(std::string& error);
};

xtbloom_status_t SystemExecution::build(std::string& error) {
  const std::int64_t atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
  if (atoms <= 0) {
    error = "restricted CPU GFN2 system has no atoms";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  atom_offsets[1] = atoms;
  /* A one-atom D4 plan has no physical pair or ATM contribution. Disabling
   * that optional component also preserves canonical zero-length bindings. */
  d4_enabled = atoms > 1;
  point_offsets[1] = key.point_count;
  molecular_charges = {key.molecular_charge};
  unpaired_electrons = {key.unpaired_electrons};
  spin_channels = {key.spin_channels};

  xtbloom_status_t status =
      make_basis_plan(1, atoms, atom_offsets.data(), key.atomic_numbers.data(), basis, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_integral_plan(basis, integrals, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_coordination_plan(1, atoms, atom_offsets.data(), key.atomic_numbers.data(),
                                  coordination, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_repulsion_plan(1, atoms, atom_offsets.data(), key.atomic_numbers.data(), repulsion,
                               error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_h0_plan(basis, integrals, key.atomic_numbers.data(), h0, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_wavefunction_layout(basis, key.atomic_numbers.data(), molecular_charges.data(),
                                    unpaired_electrons.data(), spin_channels.data(),
                                    wavefunction_layout, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_es2_plan(basis, key.atomic_numbers.data(), es2, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_es3_plan(basis, key.atomic_numbers.data(), es3, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_aes2_plan(basis, key.atomic_numbers.data(), aes2, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_mulliken_plan(basis, integrals, wavefunction_layout, mulliken, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_eigensolver_plan(wavefunction_layout, eigensolver, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_scc_mixer_plan(wavefunction_layout, kMixerHistory, kMixerDamping,
                               key.charge_tolerance, key.charge_tolerance, mixer, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = make_spin_polarization_plan(basis, wavefunction_layout, spin, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (d4_enabled) {
    status = make_d4_plan(1, atoms, atom_offsets.data(), key.atomic_numbers.data(), d4, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  status = make_external_point_charge_plan(basis, key.atomic_numbers.data(), key.point_count,
                                           key.point_count == 0 ? nullptr : point_offsets.data(),
                                           external, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (key.periodic_enabled) {
    status = make_periodic_embedding_plan(1, atoms, atom_offsets.data(), periodic, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  status =
      make_scc_driver_plan(wavefunction_layout, mulliken, es2, es3, aes2, eigensolver, mixer,
                           d4_enabled ? &d4 : nullptr, key.periodic_enabled ? &periodic : nullptr,
                           static_cast<std::uint64_t>(key.maximum_iterations),
                           key.electronic_temperature, key.energy_tolerance, driver, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  const std::size_t atom_count = static_cast<std::size_t>(atoms);
  const std::size_t point_count = static_cast<std::size_t>(key.point_count);
  const std::size_t shells = static_cast<std::size_t>(basis.total_shells);
  const std::size_t matrix = static_cast<std::size_t>(integrals.total_matrix_elements);
  positions.resize(3u * atom_count);
  point_positions.resize(3u * point_count);
  point_charges.resize(point_count);
  point_hardnesses.resize(point_count);
  periodic_shifts.assign(atom_count, 0.0);
  periodic_response.assign(atom_count * atom_count, 0.0);
  coordination_numbers.resize(atom_count);
  overlap.resize(matrix);
  dipole_integrals.resize(3u * matrix);
  quadrupole_integrals.resize(6u * matrix);
  core_hamiltonian.resize(matrix);
  field_attached = key.field_attached;
  field = key.field;
  field_vat.resize(atom_count);
  field_vdp.resize(3u * atom_count);
  status =
      allocate(integral_workspace, integrals.workspace_size_bytes, "integral workspace", error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  es2_matrix.resize(static_cast<std::size_t>(es2.total_matrix_elements()));
  es2_matrix_scratch.resize(es2_matrix.size());
  es2_shell_scratch.resize(shells);
  es2_batch_scratch.resize(1u);
  es2_gradient_scratch.resize(3u * atom_count);
  es2_workspace = {es2_matrix_scratch.data(),   es2.total_matrix_elements(),
                   es2_shell_scratch.data(),    es2.total_shells(),
                   es2_batch_scratch.data(),    1,
                   es2_gradient_scratch.data(), atoms * 3};

  aes2_pairs.resize(static_cast<std::size_t>(aes2.pair_data_elements()));
  aes2_pair_scratch.resize(aes2_pairs.size());
  aes2_potential_scratch.resize(static_cast<std::size_t>(aes2.potential_scratch_elements()));
  aes2_batch_scratch.resize(1u);
  aes2_gradient_scratch.resize(3u * atom_count);
  aes2_coordination_scratch.resize(atom_count);
  aes2_workspace = {aes2_pair_scratch.data(),         aes2.pair_data_elements(),
                    aes2_potential_scratch.data(),    aes2.potential_scratch_elements(),
                    aes2_batch_scratch.data(),        1,
                    aes2_gradient_scratch.data(),     atoms * 3,
                    aes2_coordination_scratch.data(), atoms};

  if (d4_enabled) {
    status = allocate(d4_workspace_storage, d4.workspace_size_bytes(), "D4 workspace", error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    status = bind_d4_workspace(d4, d4_workspace_storage.data(), d4_workspace_storage.size(),
                               d4_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
    d4_pairs.resize(static_cast<std::size_t>(d4.total_pairs()) * kD4PairDataElements);
    d4_coordination.resize(atom_count);
  }
  explicit_point_shell_potential.resize(shells);

  status = allocate(wavefunction_storage, wavefunction_layout.workspace_size_bytes,
                    "wavefunction state", error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = allocate(warm_checkpoint_wavefunction_storage, wavefunction_layout.workspace_size_bytes,
                    "warm-start wavefunction checkpoint", error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = allocate(overlap_cache_storage, eigensolver.overlap_cache_size_bytes(), "overlap cache",
                    error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = allocate(eigensolver_workspace_storage, eigensolver.workspace_size_bytes(),
                    "eigensolver workspace", error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = allocate(mixer_state_storage, mixer.state_size_bytes(), "SCC mixer state", error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = allocate(driver_state_storage, driver.state_size_bytes(), "SCC driver state", error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = allocate(driver_workspace_storage, driver.workspace_size_bytes(), "SCC driver workspace",
                    error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  status = bind_wavefunction_view(wavefunction_layout, wavefunction_storage.data(),
                                  wavefunction_storage.size(), wavefunction, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = bind_eigensolver_overlap_cache(eigensolver, overlap_cache_storage.data(),
                                          overlap_cache_storage.size(), overlap_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = bind_eigensolver_workspace(eigensolver, eigensolver_workspace_storage.data(),
                                      eigensolver_workspace_storage.size(), eigensolver_workspace,
                                      error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = bind_scc_mixer_state(mixer, mixer_state_storage.data(), mixer_state_storage.size(),
                                mixer_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = bind_scc_driver_state(driver, driver_state_storage.data(), driver_state_storage.size(),
                                 driver_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = bind_scc_driver_workspace(driver, driver_workspace_storage.data(),
                                     driver_workspace_storage.size(), driver_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  component_shell_potential.resize(shells);
  scalar_shell_potential.resize(shells);
  atomic_potential.resize(atom_count);
  d4_atomic_potential.resize(atom_count);
  periodic_atomic_potential.resize(atom_count);
  dipole_potential.resize(3u * atom_count);
  quadrupole_potential.resize(6u * atom_count);
  periodic_energy.resize(1u);
  periodic_status.resize(1u);

  energy_scratch.resize(1u);
  component_energy_scratch.resize(1u);
  total_gradient.resize(3u * atom_count);
  component_gradient.resize(3u * atom_count);
  force_scratch.resize(3u * atom_count);
  point_force_scratch.resize(3u * point_count);
  overlap_adjoint.resize(matrix);
  dipole_adjoint.resize(3u * matrix);
  quadrupole_adjoint.resize(6u * matrix);
  coordination_adjoint.resize(atom_count);
  stationary_density.resize(matrix);
  stationary_energy_weighted_density.resize(matrix);
  stationary_spin_density.resize(key.spin_channels == 2 ? matrix : 0u);
  packed_spin_shell_potential.resize(
      key.spin_channels == 2 ? static_cast<std::size_t>(wavefunction_layout.qsh.element_count)
                             : 0u);
  stationary_spin_shell_potential.resize(key.spin_channels == 2 ? shells : 0u);
  spin_energy_scratch.resize(key.spin_channels == 2 ? 1u : 0u);
  force_workspace = {
      energy_scratch.data(),
      component_energy_scratch.data(),
      1,
      total_gradient.data(),
      component_gradient.data(),
      force_scratch.data(),
      atoms * 3,
      overlap_adjoint.data(),
      static_cast<std::int64_t>(overlap_adjoint.size()),
      dipole_adjoint.data(),
      static_cast<std::int64_t>(dipole_adjoint.size()),
      quadrupole_adjoint.data(),
      static_cast<std::int64_t>(quadrupole_adjoint.size()),
      coordination_adjoint.data(),
      atoms,
      point_force_scratch.empty() ? nullptr : point_force_scratch.data(),
      static_cast<std::int64_t>(point_force_scratch.size()),
      integral_workspace.data(),
      integral_workspace.size(),
      es2_workspace,
      aes2_workspace,
      d4_workspace,
  };
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

namespace {

template <typename T>
std::size_t vector_bytes(const std::vector<T>& values) noexcept {
  return values.capacity() * sizeof(T);
}

std::size_t sum_double_vectors(std::initializer_list<const std::vector<double>*> vectors) noexcept {
  std::size_t total = 0u;
  for (const std::vector<double>* vector : vectors) {
    total += vector_bytes(*vector);
  }
  return total;
}

}  // namespace

std::size_t SystemExecution::resident_bytes() const noexcept {
  const std::size_t small_vectors = vector_bytes(key.atomic_numbers) + vector_bytes(atom_offsets) +
                                    vector_bytes(point_offsets) + vector_bytes(molecular_charges) +
                                    vector_bytes(unpaired_electrons) + vector_bytes(spin_channels) +
                                    vector_bytes(periodic_status);
  const std::size_t basis_plan_vectors =
      vector_bytes(basis.atom_offsets) + vector_bytes(basis.batch_shell_offsets) +
      vector_bytes(basis.batch_orbital_offsets) +
      vector_bytes(basis.batch_cartesian_orbital_offsets) +
      vector_bytes(basis.batch_primitive_offsets) + vector_bytes(basis.atom_shell_offsets) +
      vector_bytes(basis.atom_orbital_offsets) +
      vector_bytes(basis.atom_cartesian_orbital_offsets) +
      vector_bytes(basis.atom_primitive_offsets) + vector_bytes(basis.shell_orbital_offsets) +
      vector_bytes(basis.shell_cartesian_orbital_offsets) +
      vector_bytes(basis.shell_primitive_offsets) + vector_bytes(basis.shell_to_atom) +
      vector_bytes(basis.principal_quantum_numbers) + vector_bytes(basis.angular_momenta) +
      vector_bytes(basis.slater_exponents) + vector_bytes(basis.primitive_exponents) +
      vector_bytes(basis.primitive_coefficients);
  const std::size_t direct_plan_vectors =
      basis_plan_vectors + vector_bytes(integrals.matrix_offsets) +
      vector_bytes(coordination.atom_offsets) + vector_bytes(coordination.covalent_radius) +
      vector_bytes(repulsion.atom_offsets) + vector_bytes(repulsion.sqrt_alpha) +
      vector_bytes(repulsion.effective_charge) + vector_bytes(repulsion.light_element) +
      vector_bytes(h0.atom_offsets) + vector_bytes(h0.batch_shell_offsets) +
      vector_bytes(h0.batch_orbital_offsets) + vector_bytes(h0.matrix_offsets) +
      vector_bytes(h0.shell_pair_offsets) + vector_bytes(h0.atomic_radii) +
      vector_bytes(h0.shell_levels) + vector_bytes(h0.shell_coordination_scale) +
      vector_bytes(h0.shell_polynomial) + vector_bytes(h0.shell_pair_scale) +
      vector_bytes(es3.batch_shell_offsets) + vector_bytes(es3.shell_gamma3) +
      vector_bytes(spin.atom_offsets) + vector_bytes(spin.batch_shell_offsets) +
      vector_bytes(spin.atom_shell_offsets) + vector_bytes(spin.shell_population_offsets) +
      vector_bytes(spin.spin_channels) + vector_bytes(spin.coupling_offsets) +
      vector_bytes(spin.coupling_matrices) + vector_bytes(external.atom_offsets) +
      vector_bytes(external.batch_shell_offsets) + vector_bytes(external.point_charge_offsets) +
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
      vector_bytes(wavefunction_layout.dipole.system_offsets) +
      vector_bytes(wavefunction_layout.quadrupole.system_offsets) +
      vector_bytes(wavefunction_layout.energy_weighted_density.system_offsets);
  const std::size_t opaque_plan_storage = es2.resident_bytes() + aes2.resident_bytes() +
                                          mulliken.resident_bytes() + eigensolver.resident_bytes() +
                                          mixer.resident_bytes() + d4.resident_bytes() +
                                          periodic.resident_bytes() + driver.resident_bytes();
  const std::size_t planar_vectors = sum_double_vectors({
      &positions,
      &point_positions,
      &point_charges,
      &point_hardnesses,
      &periodic_shifts,
      &periodic_response,
      &coordination_numbers,
      &overlap,
      &dipole_integrals,
      &quadrupole_integrals,
      &core_hamiltonian,
      &field_vat,
      &field_vdp,
      &es2_matrix,
      &es2_matrix_scratch,
      &es2_shell_scratch,
      &es2_batch_scratch,
      &es2_gradient_scratch,
      &aes2_pairs,
      &aes2_pair_scratch,
      &aes2_potential_scratch,
      &aes2_batch_scratch,
      &aes2_gradient_scratch,
      &aes2_coordination_scratch,
      &d4_pairs,
      &d4_coordination,
      &explicit_point_shell_potential,
      &component_shell_potential,
      &scalar_shell_potential,
      &atomic_potential,
      &d4_atomic_potential,
      &periodic_atomic_potential,
      &dipole_potential,
      &quadrupole_potential,
      &periodic_energy,
      &energy_scratch,
      &component_energy_scratch,
      &total_gradient,
      &component_gradient,
      &force_scratch,
      &point_force_scratch,
      &overlap_adjoint,
      &dipole_adjoint,
      &quadrupole_adjoint,
      &coordination_adjoint,
      &stationary_density,
      &stationary_energy_weighted_density,
      &stationary_spin_density,
      &packed_spin_shell_potential,
      &stationary_spin_shell_potential,
      &spin_energy_scratch,
  });
  const std::size_t aligned_buffers =
      integral_workspace.size() + d4_workspace_storage.size() + wavefunction_storage.size() +
      warm_checkpoint_wavefunction_storage.size() + overlap_cache_storage.size() +
      eigensolver_workspace_storage.size() + mixer_state_storage.size() +
      driver_state_storage.size() + driver_workspace_storage.size();
  return small_vectors + direct_plan_vectors + wavefunction_plan_vectors + opaque_plan_storage +
         planar_vectors + aligned_buffers;
}

xtbloom_status_t SystemExecution::refresh_geometry(const CpuLinearAlgebraBackend& backend,
                                                   bool warm_start, std::string& error) {
  ++geometry_generation;
  if (geometry_generation == 0u) {
    geometry_generation = 1u;
  }
  xtbloom_status_t status =
      evaluate_coordination_cpu(coordination, positions.data(), coordination_numbers.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(),
                                integral_workspace.data(), integral_workspace.size(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = evaluate_multipole_cpu(basis, integrals, positions.data(), dipole_integrals.data(),
                                  quadrupole_integrals.data(), integral_workspace.data(),
                                  integral_workspace.size(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = evaluate_h0_cpu(basis, integrals, h0, positions.data(), coordination_numbers.data(),
                           overlap.data(), core_hamiltonian.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status =
      update_es2_geometry_cache_cpu(es2, positions.data(), geometry_generation, es2_matrix.data(),
                                    es2_matrix.size(), es2_workspace, es2_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = update_aes2_geometry_cache_cpu(aes2, positions.data(), coordination_numbers.data(),
                                          geometry_generation, aes2_pairs.data(), aes2_pairs.size(),
                                          aes2_workspace, aes2_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (d4_enabled) {
    status = update_d4_geometry_cache_cpu(d4, positions.data(), geometry_generation,
                                          d4_pairs.data(), d4_pairs.size(), d4_coordination.data(),
                                          d4_coordination.size(), d4_workspace, d4_cache, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  status = evaluate_external_point_charge_potential_cpu(
      external, positions.data(), point_positions.empty() ? nullptr : point_positions.data(),
      point_charges.empty() ? nullptr : point_charges.data(),
      point_hardnesses.empty() ? nullptr : point_hardnesses.data(),
      explicit_point_shell_potential.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = factor_overlap_cpu(eigensolver, overlap.data(), geometry_generation, backend,
                              eigensolver_workspace, overlap_cache, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  if (warm_start) {
    status = restore_warm_checkpoint(error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    status = initialize_sad_multipole_state(wavefunction_layout, wavefunction, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  }
  status = initialize_scc_driver_state_cpu(driver, wavefunction, mixer_state, driver_state, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  geometry = {};
  geometry.h0 = core_hamiltonian.data();
  geometry.h0_elements = integrals.total_matrix_elements;
  geometry.integrals = {overlap.data(), dipole_integrals.data(), quadrupole_integrals.data(),
                        integrals.total_matrix_elements, mulliken.identity()};
  geometry.es2_cache = es2_cache;
  geometry.aes2_cache = aes2_cache;
  if (d4_enabled) {
    geometry.d4_cache = d4_cache;
  }
  geometry.geometry_generation = geometry_generation;
  if (key.point_count != 0) {
    geometry.explicit_point_charge_shell_potential = explicit_point_shell_potential.data();
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
  if (field_attached) {
    /* vat_i = -E . r_i and vdp_alpha = -E_alpha, matching the released
     * electric-field block contract and the tblite field potential. The field
     * is uniform, so the dipolar potential is identical on every atom. */
    for (std::size_t atom = 0; atom < field_vat.size(); ++atom) {
      const double* r = positions.data() + 3u * atom;
      field_vat[atom] = -(field[0] * r[0] + field[1] * r[1] + field[2] * r[2]);
      field_vdp[3u * atom + 0u] = -field[0];
      field_vdp[3u * atom + 1u] = -field[1];
      field_vdp[3u * atom + 2u] = -field[2];
    }
    geometry.field_atomic_potential = field_vat.data();
    geometry.field_atomic_potential_elements = static_cast<std::int64_t>(field_vat.size());
    geometry.field_dipole_potential = field_vdp.data();
    geometry.field_dipole_potential_elements = static_cast<std::int64_t>(field_vdp.size());
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t SystemExecution::restore_warm_checkpoint(std::string& error) {
  /* The whole-batch identity gate in the execution cache runs before this
   * system is dispatched, so a missing checkpoint here is an internal
   * inconsistency rather than a caller policy error. Fail closed. */
  if (!warm_checkpoint_valid || warm_checkpoint_wavefunction_storage.data() == nullptr) {
    error = "CPU WARM SCC start found no compatible converged checkpoint in the retained system";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  std::memcpy(wavefunction_storage.data(), warm_checkpoint_wavefunction_storage.data(),
              wavefunction_storage.size());
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t SystemExecution::run_scc(const CpuLinearAlgebraBackend& backend,
                                          std::string& error) {
  while (driver_state.converged[0] == 0u &&
         driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS) {
    const xtbloom_status_t status = iterate_scc_driver_batch_cpu(
        driver, geometry, backend, overlap_cache, wavefunction, mixer_state, driver_state,
        driver_workspace, error,
        scc_parallel_enabled(parallel_executor) ? &parallel_executor : nullptr);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
  }
  return driver_state.converged[0] != 0u ? XTBLOOM_STATUS_SUCCESS : driver_state.system_statuses[0];
}

xtbloom_status_t SystemExecution::refresh_stationary_potentials(std::string& error) {
  xtbloom_status_t status = evaluate_es2_potential_cpu(
      es2, es2_cache, wavefunction.qsh, component_shell_potential.data(), es2_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  scalar_shell_potential = component_shell_potential;
  status = evaluate_es3_potential_cpu(make_es3_view(es3), wavefunction.qsh,
                                      component_shell_potential.data(), error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  for (std::size_t shell = 0; shell < scalar_shell_potential.size(); ++shell) {
    scalar_shell_potential[shell] += component_shell_potential[shell];
    if (key.point_count != 0) {
      scalar_shell_potential[shell] += explicit_point_shell_potential[shell];
    }
  }

  if (key.spin_channels == 2) {
    xtbloom_status_t spin_status = evaluate_spin_polarization_cpu(
        make_spin_polarization_view(spin), wavefunction.qsh, spin_energy_scratch.data(),
        packed_spin_shell_potential.data(), error);
    if (spin_status != XTBLOOM_STATUS_SUCCESS) return spin_status;
    const std::size_t shell_count = scalar_shell_potential.size();
    std::copy_n(packed_spin_shell_potential.data() + shell_count, shell_count,
                stationary_spin_shell_potential.data());
  }

  status = evaluate_aes2_potential_cpu(aes2, aes2_cache, wavefunction.qat, wavefunction.dipole,
                                       wavefunction.quadrupole, atomic_potential.data(),
                                       dipole_potential.data(), quadrupole_potential.data(),
                                       aes2_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (d4_enabled) {
    status = evaluate_d4_two_body_cpu(d4, d4_cache, wavefunction.qat, energy_scratch.data(),
                                      d4_atomic_potential.data(), d4_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    std::fill(d4_atomic_potential.begin(), d4_atomic_potential.end(), 0.0);
  }

  if (key.periodic_enabled) {
    const PeriodicEmbeddingView view{
        periodic_shifts.data(),
        static_cast<std::int64_t>(periodic_shifts.size()),
        periodic_response.data(),
        static_cast<std::int64_t>(periodic_response.size()),
        wavefunction.qat,
        wavefunction_layout.total_atoms,
        periodic_atomic_potential.data(),
        wavefunction_layout.total_atoms,
        periodic_energy.data(),
        1,
        periodic_status.data(),
        1,
        periodic.identity(),
    };
    status = evaluate_periodic_embedding_batch_cpu(
        periodic, view, driver_workspace.periodic_embedding_workspace, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;
  } else {
    std::fill(periodic_atomic_potential.begin(), periodic_atomic_potential.end(), 0.0);
  }

  for (std::size_t atom = 0; atom < atomic_potential.size(); ++atom) {
    atomic_potential[atom] += d4_atomic_potential[atom] + periodic_atomic_potential[atom];
  }
  if (field_attached) {
    /* Mirror the SCC driver injection: the field's scalar potential -E . r
     * contributes on the charge-channel atom potential (mapped onto shells by
     * the loop below) and its dipolar potential -E on the charge-channel
     * dipole potential, so the stationary integral adjoints retain the field's
     * density-response terms. The remaining explicit coordinate derivative,
     * +q_i E in the public force convention, is added at the force boundary. */
    for (std::size_t atom = 0; atom < atomic_potential.size(); ++atom) {
      atomic_potential[atom] += field_vat[atom];
    }
    for (std::size_t component = 0; component < dipole_potential.size(); ++component) {
      dipole_potential[component] += field_vdp[component];
    }
  }
  for (std::size_t shell = 0; shell < scalar_shell_potential.size(); ++shell) {
    const std::size_t atom = static_cast<std::size_t>(basis.shell_to_atom[shell]);
    scalar_shell_potential[shell] += atomic_potential[atom];
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t SystemExecution::infer(
    const CpuLinearAlgebraBackend& backend, const double* input_positions,
    const double* input_point_positions, const double* input_point_charges,
    const double* input_point_hardnesses, const double* input_shifts, const double* input_response,
    std::uint32_t compute_flags, bool warm_start, SystemOutput& output, std::string& error) {
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

  xtbloom_status_t status = refresh_geometry(backend, warm_start, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  status = run_scc(backend, error);
  output.iterations = static_cast<std::int32_t>(std::min<std::uint64_t>(
      driver_state.iterations[0],
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())));
  if (status != XTBLOOM_STATUS_SUCCESS) {
    /* A failed or non-converged SCC run leaves the live wavefunction in a
     * partial mixed state; revoke any retained checkpoint so a later strict
     * WARM request cannot consume it. */
    warm_checkpoint_valid = false;
    output.status = status;
    return status;
  }

  /* The converged electronic state is the warm checkpoint for the next WARM
   * call (consumed regardless of the next geometry, but only within the same
   * topology/options identity enforced by the execution cache). */
  std::memcpy(warm_checkpoint_wavefunction_storage.data(), wavefunction_storage.data(),
              wavefunction_storage.size());
  warm_checkpoint_valid = true;

  output.status = XTBLOOM_STATUS_SUCCESS;
  output.converged = 1u;
  output.atomic_charges.assign(wavefunction.qat,
                               wavefunction.qat + wavefunction_layout.total_atoms);

  if ((compute_flags & XTBLOOM_COMPUTE_DIPOLE_MOMENTS) != 0u) {
    /* Molecular dipole moment in the tblite molmom convention:
     * sum_i (r_i * q_i + d_i) over the charge-channel SCC multipoles. The
     * dipole population field stores [channel, atom, 3]; channel zero is the
     * charge channel in both restricted and unrestricted layouts. */
    const std::int64_t dipole_stride = 3 * wavefunction_layout.total_atoms;
    std::array<double, 3> moment{0.0, 0.0, 0.0};
    for (std::int64_t atom = 0; atom < wavefunction_layout.total_atoms; ++atom) {
      const double charge = wavefunction.qat[atom];
      for (std::int64_t component = 0; component < 3; ++component) {
        moment[static_cast<std::size_t>(component)] +=
            positions[static_cast<std::size_t>(3 * atom + component)] * charge +
            wavefunction.dipole[static_cast<std::size_t>(atom * 3 + component)];
      }
    }
    output.dipole_moments = moment;
  }

  const bool need_energy_or_force =
      (compute_flags & (XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                        XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
  if (!need_energy_or_force) {
    return XTBLOOM_STATUS_SUCCESS;
  }

  status = refresh_stationary_potentials(error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;

  const std::size_t matrix_elements = stationary_density.size();
  if (key.spin_channels == 1) {
    std::copy_n(wavefunction.density, matrix_elements, stationary_density.data());
    std::copy_n(wavefunction.energy_weighted_density, matrix_elements,
                stationary_energy_weighted_density.data());
  } else {
    const double* alpha_density = wavefunction.density;
    const double* beta_density = wavefunction.density + matrix_elements;
    const double* alpha_weighted = wavefunction.energy_weighted_density;
    const double* beta_weighted = wavefunction.energy_weighted_density + matrix_elements;
    for (std::size_t element = 0u; element < matrix_elements; ++element) {
      const double total_density = alpha_density[element] + beta_density[element];
      const double spin_density = alpha_density[element] - beta_density[element];
      const double total_weighted = alpha_weighted[element] + beta_weighted[element];
      if (!std::isfinite(total_density) || !std::isfinite(spin_density) ||
          !std::isfinite(total_weighted)) {
        error = "unrestricted stationary density reduction overflowed";
        return XTBLOOM_STATUS_INTERNAL_ERROR;
      }
      stationary_density[element] = total_density;
      stationary_spin_density[element] = spin_density;
      stationary_energy_weighted_density[element] = total_weighted;
    }
  }

  const bool need_qm_forces = (compute_flags & XTBLOOM_COMPUTE_FORCES) != 0u;
  const bool need_point_forces =
      (compute_flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0u && key.point_count != 0;
  const bool compose_qm_forces = need_qm_forces || need_point_forces;
  /* The stationary composer validates a QM-force sink whenever any force is
   * requested. Point-only public calls therefore use an unpublished QM sink. */
  output.forces.assign(compose_qm_forces ? positions.size() : 0u, 0.0);
  output.point_forces.assign(need_point_forces ? point_positions.size() : 0u, 0.0);
  const RestrictedGfn2StationaryInput input{
      positions.data(),
      coordination_numbers.data(),
      geometry_generation,
      overlap.data(),
      stationary_density.data(),
      stationary_energy_weighted_density.data(),
      wavefunction.qsh,
      wavefunction.qat,
      wavefunction.dipole,
      wavefunction.quadrupole,
      scalar_shell_potential.data(),
      dipole_potential.data(),
      quadrupole_potential.data(),
      driver_state.free_energies,
      point_positions.empty() ? nullptr : point_positions.data(),
      point_charges.empty() ? nullptr : point_charges.data(),
      point_hardnesses.empty() ? nullptr : point_hardnesses.data(),
      key.spin_channels == 2 ? stationary_spin_density.data() : nullptr,
      key.spin_channels == 2 ? stationary_spin_shell_potential.data() : nullptr,
  };
  RestrictedGfn2ForceWorkspace composer_workspace = force_workspace;
  if (!d4_enabled) {
    composer_workspace.component_energy_scratch = nullptr;
    composer_workspace.d4_workspace = {};
  }
  status = evaluate_restricted_gfn2_energy_forces_cpu(
      basis, integrals, coordination, repulsion, h0, mulliken, es2, es2_cache, aes2, aes2_cache,
      d4_enabled ? &d4 : nullptr, d4_enabled ? &d4_cache : nullptr,
      key.point_count == 0 ? nullptr : &external, input, &output.energy,
      compose_qm_forces ? output.forces.data() : nullptr,
      need_point_forces ? output.point_forces.data() : nullptr, {}, composer_workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) return status;
  if (compose_qm_forces && field_attached) {
    /* The stationary composer already carries the response of the converged
     * density and atomic multipoles through the injected field potentials.
     * The remaining explicit derivative of -sum_i q_i E.r_i is +q_i E in the
     * public F=-dE/dR convention. */
    for (std::size_t atom = 0; atom < field_vat.size(); ++atom) {
      const double charge = wavefunction.qat[atom];
      output.forces[3u * atom + 0u] = std::fma(charge, field[0], output.forces[3u * atom + 0u]);
      output.forces[3u * atom + 1u] = std::fma(charge, field[1], output.forces[3u * atom + 1u]);
      output.forces[3u * atom + 2u] = std::fma(charge, field[2], output.forces[3u * atom + 2u]);
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

struct Gfn2CpuExecutionCache::Impl {
  enum class TaskFailure : std::uint8_t { kNone, kAllocation, kException, kUnknown };

  explicit Impl(std::int32_t requested_threads)
      : cpu_threads(resolve_cpu_threads(requested_threads)), workers(cpu_threads) {}

  std::mutex mutex;
  CpuLinearAlgebraBackend backend;
  bool backend_initialized = false;
  std::vector<SystemKey> keys;
  std::vector<std::unique_ptr<SystemExecution>> systems;
  /* True only when the most recent executed batch had every member converge.
   * The strict WARM gate in execute_restricted_gfn2_cpu refuses to start from a
   * checkpoint unless this identity already produced a fully converged result. */
  bool systems_ready_for_warm = false;

  /* The remaining members are context-owned transaction staging. Their
   * capacities survive repeated calls with the same or smaller topology. */
  HostRequest request;
  std::vector<SystemKey> requested_keys;
  std::vector<SystemOutput> outputs;
  std::vector<std::string> system_errors;
  std::vector<xtbloom_status_t> inference_statuses;
  std::vector<TaskFailure> task_failures;
  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<double> point_forces;
  std::vector<double> dipole_moments;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<std::int32_t> system_statuses;

  const std::size_t cpu_threads;
  CpuWorkerPool workers;

  /* Adapter from the SCC driver's chunked executor to this context's fixed
   * worker pool. Used only for batch==1, when the pool is idle. */
  static void dispatch_scc_chunks(void* pool_context, std::size_t chunk_count,
                                  void (*body)(void*, std::size_t) noexcept,
                                  void* body_context) noexcept {
    static_cast<CpuWorkerPool*>(pool_context)->parallel_for(chunk_count, body_context, body);
  }

  xtbloom_status_t ensure_backend(std::string& error) {
    if (backend_initialized) {
      return XTBLOOM_STATUS_SUCCESS;
    }
    const xtbloom_status_t status = make_mkl_rt_lp64_backend(backend, error);
    if (status == XTBLOOM_STATUS_SUCCESS) {
      backend_initialized = true;
    }
    return status;
  }

  xtbloom_status_t ensure_systems(const std::vector<SystemKey>& requested, std::string& error) {
    if (requested == keys) {
      return XTBLOOM_STATUS_SUCCESS;
    }
    const bool reusable_layout = requested.size() == keys.size() &&
                                 requested.size() == systems.size() &&
                                 std::equal(requested.begin(), requested.end(), keys.begin(),
                                            [](const SystemKey& next, const SystemKey& current) {
                                              return same_prepared_layout(next, current);
                                            });
    if (reusable_layout) {
      /* Only field presence/value changed. Update the preallocated numerical
       * storage in place and keep the exact key for subsequent WARM checks. */
      for (std::size_t index = 0u; index < requested.size(); ++index) {
        systems[index]->set_field(requested[index].field_attached, requested[index].field);
        keys[index].field_attached = requested[index].field_attached;
        keys[index].field = requested[index].field;
      }
      return XTBLOOM_STATUS_SUCCESS;
    }
    std::vector<std::unique_ptr<SystemExecution>> candidate;
    candidate.reserve(requested.size());
    /* Intra-system phase parallelism is valid only when a single system runs
     * in this batch: the outer parallel_for then short-circuits to the calling
     * thread and the pool is idle, so dispatching per-iteration chunks to it is
     * not reentrant. For larger batches the pool is already executing the outer
     * per-system tasks and must not be re-entered from inside them. A pool that
     * ended up with a single worker (including cpu_threads=1 and truncated
     * thread creation) gets no executor, preserving the exact serial path. */
    const bool intra_system_parallel = requested.size() == 1u && workers.concurrency() > 1u;
    for (const SystemKey& key : requested) {
      auto system = std::make_unique<SystemExecution>(key);
      if (intra_system_parallel) {
        system->parallel_executor.pool_context = &workers;
        system->parallel_executor.worker_count = workers.concurrency();
        system->parallel_executor.dispatch_chunks = &dispatch_scc_chunks;
      }
      const xtbloom_status_t status = system->build(error);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        return status;
      }
      candidate.push_back(std::move(system));
    }
    systems = std::move(candidate);
    keys = requested;
    return XTBLOOM_STATUS_SUCCESS;
  }

  void prepare_staging(std::uint32_t flags) {
    const std::size_t batch_size = static_cast<std::size_t>(request.batch_size);
    const std::size_t atom_count = static_cast<std::size_t>(request.total_atoms);
    const std::size_t point_count = static_cast<std::size_t>(request.total_point_charges);
    const double nan = std::numeric_limits<double>::quiet_NaN();

    outputs.resize(batch_size);
    /* Plan creation calls prepare_staging before the first inference. Reserve
     * the per-system publication vectors here so the first plan call has the
     * same allocation-free behavior as later calls. */
    for (std::size_t index = 0u; index < batch_size; ++index) {
      const std::int64_t atom_begin = request.atom_offsets[index];
      const std::int64_t atom_end = request.atom_offsets[index + 1u];
      const std::int64_t point_begin = request.point_offsets[index];
      const std::int64_t point_end = request.point_offsets[index + 1u];
      const std::size_t atoms = static_cast<std::size_t>(atom_end - atom_begin);
      const std::size_t points = static_cast<std::size_t>(point_end - point_begin);
      outputs[index].forces.reserve(3u * atoms);
      outputs[index].atomic_charges.reserve(atoms);
      outputs[index].point_forces.reserve(3u * points);
    }
    system_errors.resize(batch_size);
    inference_statuses.assign(batch_size, XTBLOOM_STATUS_INTERNAL_ERROR);
    task_failures.assign(batch_size, TaskFailure::kNone);
    iterations.assign(batch_size, 0);
    converged.assign(batch_size, 0u);
    system_statuses.assign(batch_size, XTBLOOM_STATUS_EIGENSOLVER_FAILED);

    if ((flags & XTBLOOM_COMPUTE_ENERGY) != 0u) {
      energies.assign(batch_size, nan);
    } else {
      energies.clear();
    }
    if ((flags & XTBLOOM_COMPUTE_FORCES) != 0u) {
      forces.assign(3u * atom_count, nan);
    } else {
      forces.clear();
    }
    if ((flags & XTBLOOM_COMPUTE_ATOMIC_CHARGES) != 0u) {
      atomic_charges.assign(atom_count, nan);
    } else {
      atomic_charges.clear();
    }
    if ((flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0u) {
      point_forces.assign(3u * point_count, nan);
    } else {
      point_forces.clear();
    }
    if ((flags & XTBLOOM_COMPUTE_DIPOLE_MOMENTS) != 0u) {
      dipole_moments.assign(3u * batch_size, nan);
    } else {
      dipole_moments.clear();
    }
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
    const std::int64_t point_end = request.point_offsets[index + 1u];
    const std::int64_t points = point_end - point_begin;
    const double* shifts =
        request.shifts_enabled ? request.periodic_shifts.data() + atom_begin : nullptr;
    const double* response = request.response_enabled ? request.response_matrices.data() +
                                                            request.response_offsets[index]
                                                      : nullptr;

    try {
      const bool warm_start = job.options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE &&
                              job.options.scc_start_mode == XTBLOOM_SCC_START_WARM;
      owner.inference_statuses[index] = owner.systems[index]->infer(
          owner.backend, request.positions.data() + 3 * atom_begin,
          points == 0 ? nullptr : request.point_positions.data() + 3 * point_begin,
          points == 0 ? nullptr : request.point_charges.data() + point_begin,
          points == 0 ? nullptr : request.point_hardnesses.data() + point_begin, shifts, response,
          job.options.flags, warm_start, output, system_error);
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

Gfn2CpuExecutionCache::Gfn2CpuExecutionCache(std::int32_t cpu_threads)
    : impl_(std::make_unique<Impl>(cpu_threads)) {}
Gfn2CpuExecutionCache::~Gfn2CpuExecutionCache() = default;

xtbloom_status_t execute_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                             const xtbloom_batch_t& batch,
                                             const xtbloom_compute_options_t& options,
                                             xtbloom_batch_result_t& result, std::string& error) {
  try {
    std::lock_guard<std::mutex> lock(cache.impl_->mutex);
    Gfn2CpuExecutionCache::Impl& implementation = *cache.impl_;
    stage_request(batch, implementation.request);
    xtbloom_status_t status = validate_host_numerics(implementation.request, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
    make_system_keys(implementation.request, options, implementation.requested_keys);

    const bool warm_requested = options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE &&
                                options.scc_start_mode == XTBLOOM_SCC_START_WARM;
    if (warm_requested) {
      /* Strict WARM: refuse without changing any caller output when this
       * context has no fully converged compatible identity to consume. The
       * identity covers the complete SystemKey set (topology plus compute
       * policy, tolerances, and electronic temperature), exactly matching the
       * CUDA warm-start contract. */
      if (implementation.requested_keys != implementation.keys ||
          !implementation.systems_ready_for_warm) {
        error =
            "CPU WARM SCC start requires a previous fully converged call with identical "
            "topology and compute options";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    status = implementation.ensure_backend(error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }

    /* Keep the currently published checkpoint consumable until every
     * pre-execution staging/setup step succeeds. prepare_staging only mutates
     * transaction scratch, while ensure_systems builds a different identity
     * into a local candidate before replacing the retained systems. A failure
     * in either step therefore leaves both the old systems and their
     * whole-batch readiness token intact for a later compatible WARM call. */
    implementation.prepare_staging(options.flags);
    status = implementation.ensure_systems(implementation.requested_keys, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }

    /* Numerical execution is the predecessor attempt for strict WARM
     * semantics. From this point onward, a failed or non-converged call must
     * not expose the older checkpoint as though the attempt never occurred. */
    implementation.systems_ready_for_warm = false;
    Gfn2CpuExecutionCache::Impl::InferenceJob job{implementation, options};
    implementation.workers.parallel_for(static_cast<std::size_t>(implementation.request.batch_size),
                                        &job, &Gfn2CpuExecutionCache::Impl::infer_system);

    const HostRequest& request = implementation.request;
    bool all_converged = true;
    for (std::int64_t system = 0; system < request.batch_size; ++system) {
      const std::size_t index = static_cast<std::size_t>(system);
      const std::int64_t atom_begin = request.atom_offsets[index];
      const std::int64_t point_begin = request.point_offsets[index];
      const std::int64_t point_end = request.point_offsets[index + 1u];
      const std::int64_t points = point_end - point_begin;
      SystemOutput& output = implementation.outputs[index];
      status = implementation.inference_statuses[index];
      implementation.iterations[index] = output.iterations;
      if (status != XTBLOOM_STATUS_SUCCESS) {
        if (status == XTBLOOM_STATUS_SCC_NOT_CONVERGED ||
            status == XTBLOOM_STATUS_EIGENSOLVER_FAILED) {
          implementation.system_statuses[index] = status;
          all_converged = false;
          continue;
        }
        if (!implementation.system_errors[index].empty()) {
          error = implementation.system_errors[index];
        } else if (implementation.task_failures[index] ==
                   Gfn2CpuExecutionCache::Impl::TaskFailure::kAllocation) {
          error = "failed to allocate CPU GFN2 per-system inference state";
        } else if (implementation.task_failures[index] ==
                   Gfn2CpuExecutionCache::Impl::TaskFailure::kUnknown) {
          error = "unknown exception while executing a CPU GFN2 batch member";
        } else {
          error = "CPU GFN2 batch member failed without a diagnostic";
        }
        return status;
      }

      implementation.system_statuses[index] = XTBLOOM_STATUS_SUCCESS;
      implementation.converged[index] = 1u;
      if ((options.flags & XTBLOOM_COMPUTE_ENERGY) != 0u) {
        implementation.energies[index] = output.energy;
      }
      if ((options.flags & XTBLOOM_COMPUTE_FORCES) != 0u) {
        std::copy(output.forces.begin(), output.forces.end(),
                  implementation.forces.begin() + 3 * atom_begin);
      }
      if ((options.flags & XTBLOOM_COMPUTE_ATOMIC_CHARGES) != 0u) {
        std::copy(output.atomic_charges.begin(), output.atomic_charges.end(),
                  implementation.atomic_charges.begin() + atom_begin);
      }
      if ((options.flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0u && points != 0) {
        std::copy(output.point_forces.begin(), output.point_forces.end(),
                  implementation.point_forces.begin() + 3 * point_begin);
      }
      if ((options.flags & XTBLOOM_COMPUTE_DIPOLE_MOMENTS) != 0u) {
        std::copy_n(output.dipole_moments.begin(), 3,
                    implementation.dipole_moments.begin() + 3 * index);
      }
    }

    if ((options.flags & XTBLOOM_COMPUTE_ENERGY) != 0u) {
      publish_to_c_buffer(implementation.energies, result.energies);
    }
    if ((options.flags & XTBLOOM_COMPUTE_FORCES) != 0u) {
      publish_to_c_buffer(implementation.forces, result.forces);
    }
    if ((options.flags & XTBLOOM_COMPUTE_ATOMIC_CHARGES) != 0u) {
      publish_to_c_buffer(implementation.atomic_charges, result.atomic_charges);
    }
    if ((options.flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0u) {
      publish_to_c_buffer(implementation.point_forces, result.point_charge_forces);
    }
    if ((options.flags & XTBLOOM_COMPUTE_DIPOLE_MOMENTS) != 0u) {
      publish_to_c_buffer(implementation.dipole_moments, result.dipole_moments);
    }
    publish_to_c_buffer(implementation.iterations, result.scc_iterations);
    publish_to_c_buffer(implementation.converged, result.scc_converged);
    publish_to_c_buffer(implementation.system_statuses, result.per_system_status);
    result.flags =
        static_cast<std::uint32_t>((request.shifts_enabled || request.response_enabled)
                                       ? XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES
                                       : 0u) |
        static_cast<std::uint32_t>((options.flags & XTBLOOM_COMPUTE_DIPOLE_MOMENTS) != 0u
                                       ? XTBLOOM_RESULT_DIPOLE_MOMENTS
                                       : 0u);
    implementation.systems_ready_for_warm = all_converged;
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate CPU GFN2 execution staging";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t prepare_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache,
                                             const xtbloom_batch_t& batch,
                                             const xtbloom_compute_options_t& options, bool& reused,
                                             std::string& error) {
  reused = false;
  try {
    std::lock_guard<std::mutex> lock(cache.impl_->mutex);
    Gfn2CpuExecutionCache::Impl& implementation = *cache.impl_;
    stage_request(batch, implementation.request);
    xtbloom_status_t status = validate_host_numerics(implementation.request, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
    make_system_keys(implementation.request, options, implementation.requested_keys);
    reused = implementation.requested_keys == implementation.keys;
    status = implementation.ensure_backend(error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
    /* Plan setup is the allocation-permitted path: build every per-system
     * execution object now so a subsequent xtbloom_plan_compute performs no
     * steady-state allocation. Only a rebuild invalidates a retained warm
     * checkpoint; a reused identity keeps its checkpoint consumable. */
    if (!reused) {
      implementation.systems_ready_for_warm = false;
    }
    status = implementation.ensure_systems(implementation.requested_keys, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      return status;
    }
    implementation.prepare_staging(options.flags);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate CPU GFN2 plan setup state";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

std::size_t persistent_workspace_bytes_restricted_gfn2_cpu(Gfn2CpuExecutionCache& cache) noexcept {
  std::lock_guard<std::mutex> lock(cache.impl_->mutex);
  const Gfn2CpuExecutionCache::Impl& implementation = *cache.impl_;
  /* Count every retained allocation owned by the plan cache. The object sizes
   * cover inline metadata; vector capacities cover their heap reservations. */
  std::size_t total = sizeof(Gfn2CpuExecutionCache::Impl) +
                      implementation.workers.resident_bytes() +
                      vector_bytes(implementation.systems);
  for (const std::unique_ptr<SystemExecution>& system : implementation.systems) {
    total += sizeof(SystemExecution) + system->resident_bytes();
  }

  const auto key_vector_bytes = [](const std::vector<SystemKey>& keys) noexcept {
    std::size_t bytes = vector_bytes(keys);
    for (const SystemKey& key : keys) bytes += vector_bytes(key.atomic_numbers);
    return bytes;
  };
  const auto output_vector_bytes = [](const std::vector<SystemOutput>& outputs) noexcept {
    std::size_t bytes = vector_bytes(outputs);
    for (const SystemOutput& output : outputs) {
      bytes += vector_bytes(output.forces) + vector_bytes(output.atomic_charges) +
               vector_bytes(output.point_forces);
    }
    return bytes;
  };
  const auto string_vector_bytes = [](const std::vector<std::string>& strings) noexcept {
    std::size_t bytes = vector_bytes(strings);
    /* capacity()+1 is a conservative reservation for implementations whose
     * empty diagnostic uses small-string storage inside the vector element. */
    for (const std::string& value : strings) bytes += value.capacity() + 1u;
    return bytes;
  };

  /* Batch-level topology/policy keys and transaction staging all survive
   * steady-state calls and must be represented by the workspace query. */
  total += key_vector_bytes(implementation.keys) + key_vector_bytes(implementation.requested_keys) +
           output_vector_bytes(implementation.outputs) +
           string_vector_bytes(implementation.system_errors) +
           vector_bytes(implementation.inference_statuses) +
           vector_bytes(implementation.task_failures) + vector_bytes(implementation.energies) +
           vector_bytes(implementation.forces) + vector_bytes(implementation.atomic_charges) +
           vector_bytes(implementation.point_forces) + vector_bytes(implementation.dipole_moments) +
           vector_bytes(implementation.iterations) + vector_bytes(implementation.converged) +
           vector_bytes(implementation.system_statuses);

  const HostRequest& request = implementation.request;
  total += vector_bytes(request.atom_offsets) + vector_bytes(request.atomic_numbers) +
           vector_bytes(request.positions) + vector_bytes(request.molecular_charges) +
           vector_bytes(request.unpaired_electrons) + vector_bytes(request.spin_channels) +
           vector_bytes(request.point_offsets) + vector_bytes(request.point_positions) +
           vector_bytes(request.point_charges) + vector_bytes(request.point_hardnesses) +
           vector_bytes(request.periodic_shifts) + vector_bytes(request.response_offsets) +
           vector_bytes(request.response_matrices) + vector_bytes(request.field_by_system) +
           vector_bytes(request.field_attached_by_system);
  return total;
}

}  // namespace xtbloom::detail
