#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_scc_publication.cuh"

namespace {

using namespace xtbloom::detail::cuda;

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

#define CUDA_CHECK(expression)                                                \
  do {                                                                        \
    const cudaError_t cuda_status_ = (expression);                            \
    if (cuda_status_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "CUDA failure at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(cuda_status_));                         \
      return __LINE__;                                                        \
    }                                                                         \
  } while (false)

constexpr std::uint64_t kPlanToken = 0x5055424c49534831ULL;
constexpr double kPublicSentinel = -7319.25;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t elements) { require(allocate(elements)); }
  ~DeviceBuffer() { cudaFree(pointer_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)), elements_(other.elements_) {
    other.elements_ = 0u;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      cudaFree(pointer_);
      pointer_ = std::exchange(other.pointer_, nullptr);
      elements_ = other.elements_;
      other.elements_ = 0u;
    }
    return *this;
  }

  cudaError_t allocate(std::size_t elements) {
    cudaFree(pointer_);
    pointer_ = nullptr;
    elements_ = elements;
    return elements == 0u ? cudaSuccess
                          : cudaMalloc(reinterpret_cast<void**>(&pointer_), elements * sizeof(T));
  }

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream = nullptr) const {
    if (values.size() != elements_) {
      return cudaErrorInvalidValue;
    }
    return elements_ == 0u ? cudaSuccess
                           : cudaMemcpyAsync(pointer_, values.data(), elements_ * sizeof(T),
                                             cudaMemcpyHostToDevice, stream);
  }

  cudaError_t download(std::vector<T>& values, cudaStream_t stream = nullptr) const {
    values.resize(elements_);
    return elements_ == 0u ? cudaSuccess
                           : cudaMemcpyAsync(values.data(), pointer_, elements_ * sizeof(T),
                                             cudaMemcpyDeviceToHost, stream);
  }

  T* get() const { return pointer_; }
  std::size_t size() const { return elements_; }

 private:
  static void require(cudaError_t status) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA allocation failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }

  T* pointer_ = nullptr;
  std::size_t elements_ = 0u;
};

template <typename T>
struct Arena {
  DeviceBuffer<T> device;
  std::vector<T> host;
  std::size_t cursor = 0u;

  void allocate(std::size_t elements, T value = T{}) {
    if (device.allocate(elements) != cudaSuccess) {
      std::abort();
    }
    host.assign(elements, value);
    cursor = 0u;
  }

  T* take(std::size_t elements) {
    if (elements > host.size() - cursor) {
      std::fprintf(stderr, "test arena capacity exceeded\n");
      std::abort();
    }
    T* result = device.get() + cursor;
    cursor += elements;
    return result;
  }

  std::size_t offset(const T* pointer) const {
    return static_cast<std::size_t>(pointer - device.get());
  }

  T& at(T* pointer, std::size_t index) { return host[offset(pointer) + index]; }
  T& at(const T* pointer, std::size_t index) { return host[offset(pointer) + index]; }
  const T& at(const T* pointer, std::size_t index) const { return host[offset(pointer) + index]; }

  cudaError_t upload(cudaStream_t stream = nullptr) const { return device.upload(host, stream); }
  cudaError_t download(std::vector<T>& values, cudaStream_t stream = nullptr) const {
    return device.download(values, stream);
  }
};

struct PublicSnapshot {
  std::vector<double> doubles;
  std::vector<std::uint64_t> uint64s;
  std::vector<xtbloom_status_t> statuses;
  std::vector<std::uint8_t> bytes;
};

template <typename T>
bool byte_equal(const T& first, const T& second) {
  return std::memcmp(&first, &second, sizeof(T)) == 0;
}

template <typename T>
bool vector_byte_equal(const std::vector<T>& first, const std::vector<T>& second) {
  return first.size() == second.size() &&
         (first.empty() || std::memcmp(first.data(), second.data(), first.size() * sizeof(T)) == 0);
}

template <typename T>
bool public_range_unchanged(const Arena<T>& arena, const std::vector<T>& before,
                            const std::vector<T>& after, const T* pointer, std::int64_t begin,
                            std::int64_t end) {
  const std::size_t base = arena.offset(pointer);
  for (std::int64_t index = begin; index < end; ++index) {
    const std::size_t absolute = base + static_cast<std::size_t>(index);
    if (!byte_equal(before[absolute], after[absolute])) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool staged_matches_public(const Arena<T>& staged_arena, const T* staged_pointer,
                           const Arena<T>& public_arena, const T* public_pointer,
                           const std::vector<T>& public_after, std::int64_t begin,
                           std::int64_t end) {
  const std::size_t staged_base = staged_arena.offset(staged_pointer);
  const std::size_t public_base = public_arena.offset(public_pointer);
  for (std::int64_t index = begin; index < end; ++index) {
    if (!byte_equal(staged_arena.host[staged_base + static_cast<std::size_t>(index)],
                    public_after[public_base + static_cast<std::size_t>(index)])) {
      return false;
    }
  }
  return true;
}

struct Fixture {
  explicit Fixture(std::int64_t requested_batch, bool complex_case = false,
                   bool mixed_spin_case = false)
      : batch(requested_batch), complex(complex_case), mixed_spin(mixed_spin_case) {
    build_topology();
    allocate_storage();
    bind_descriptors();
    initialize_values();
  }

  std::int64_t batch = 0;
  bool complex = false;
  bool mixed_spin = false;
  std::int64_t atoms = 0;
  std::int64_t shells = 0;
  std::int64_t orbitals = 0;
  std::int64_t matrices = 0;
  std::int64_t spin_channels_total = 0;
  std::int64_t spin_atoms = 0;
  std::int64_t spin_shells = 0;
  std::int64_t spin_orbitals = 0;
  std::int64_t spin_matrices = 0;
  std::int64_t vector_elements = 0;
  std::int64_t history_elements = 0;
  std::int64_t omega_elements = 0;
  static constexpr std::int64_t history_size = 2;

  std::vector<std::int64_t> atom_offsets_host;
  std::vector<std::int64_t> shell_offsets_host;
  std::vector<std::int64_t> orbital_offsets_host;
  std::vector<std::int64_t> matrix_offsets_host;
  std::vector<std::int64_t> shell_to_atom_host;
  std::vector<std::int32_t> spin_channels_host;
  std::vector<std::int64_t> spin_channel_offsets_host;
  std::vector<std::int64_t> spin_atom_offsets_host;
  std::vector<std::int64_t> spin_shell_offsets_host;
  std::vector<std::int64_t> spin_orbital_offsets_host;
  std::vector<std::int64_t> spin_matrix_offsets_host;
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> shell_offsets;
  DeviceBuffer<std::int64_t> orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<std::int32_t> spin_channels;
  DeviceBuffer<std::int64_t> spin_channel_offsets;
  DeviceBuffer<std::int64_t> spin_atom_offsets;
  DeviceBuffer<std::int64_t> spin_shell_offsets;
  DeviceBuffer<std::int64_t> spin_orbital_offsets;
  DeviceBuffer<std::int64_t> spin_matrix_offsets;

  Arena<double> staged_d;
  Arena<std::uint64_t> staged_u64;
  Arena<xtbloom_status_t> staged_status;
  Arena<std::uint8_t> staged_u8;
  Arena<double> public_d;
  Arena<std::uint64_t> public_u64;
  Arena<xtbloom_status_t> public_status;
  Arena<std::uint8_t> public_u8;

  DeviceBuffer<std::uint8_t> active_mask;
  DeviceBuffer<xtbloom_status_t> pending_statuses;
  DeviceBuffer<std::uint64_t> failure_records;
  DeviceBuffer<std::uint64_t> plan_failure;
  DeviceBuffer<std::uint32_t> canonical_sequence;
  std::vector<std::uint8_t> active_host;
  std::vector<xtbloom_status_t> pending_host;
  std::vector<std::uint64_t> failure_host;
  std::vector<std::uint64_t> plan_failure_host;
  std::vector<std::uint32_t> canonical_sequence_host;

  DeviceBuffer<double> mixed_qat;
  DeviceBuffer<double> old_energies;
  DeviceBuffer<double> energy_changes;
  DeviceBuffer<std::uint64_t> next_iterations;
  DeviceBuffer<xtbloom_status_t> next_statuses;
  DeviceBuffer<std::uint8_t> next_converged;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;
  DeviceBuffer<std::uint32_t> stage_sequence;

  Gfn2SccPublicationDevicePlan plan{};
  Gfn2SccIterationDeviceLedger ledger{};
  Gfn2SccIterationDeviceActivity activity{};
  Gfn2SccPublicationDeviceStagedState staged{};
  Gfn2SccPublicationDevicePublicState public_state{};
  Gfn2SccPublicationDeviceWorkspace workspace{};
  Gfn2SccStageDeviceReport report{};

  void build_topology() {
    atom_offsets_host.assign(static_cast<std::size_t>(batch + 1), 0);
    shell_offsets_host.assign(static_cast<std::size_t>(batch + 1), 0);
    orbital_offsets_host.assign(static_cast<std::size_t>(batch + 1), 0);
    matrix_offsets_host.assign(static_cast<std::size_t>(batch + 1), 0);
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int64_t system_atoms = 1 + (system & 1);
      const std::int64_t system_shells = system_atoms + 1;
      const std::int64_t system_orbitals = 1 + (system & 1);
      atom_offsets_host[static_cast<std::size_t>(system + 1)] =
          atom_offsets_host[static_cast<std::size_t>(system)] + system_atoms;
      shell_offsets_host[static_cast<std::size_t>(system + 1)] =
          shell_offsets_host[static_cast<std::size_t>(system)] + system_shells;
      orbital_offsets_host[static_cast<std::size_t>(system + 1)] =
          orbital_offsets_host[static_cast<std::size_t>(system)] + system_orbitals;
      matrix_offsets_host[static_cast<std::size_t>(system + 1)] =
          matrix_offsets_host[static_cast<std::size_t>(system)] + system_orbitals * system_orbitals;
    }
    atoms = atom_offsets_host.back();
    shells = shell_offsets_host.back();
    orbitals = orbital_offsets_host.back();
    matrices = matrix_offsets_host.back();
    spin_channels_host.assign(static_cast<std::size_t>(batch), 1);
    spin_channel_offsets_host.assign(static_cast<std::size_t>(batch + 1), 0);
    spin_atom_offsets_host.assign(static_cast<std::size_t>(batch + 1), 0);
    spin_shell_offsets_host.assign(static_cast<std::size_t>(batch + 1), 0);
    spin_orbital_offsets_host.assign(static_cast<std::size_t>(batch + 1), 0);
    spin_matrix_offsets_host.assign(static_cast<std::size_t>(batch + 1), 0);
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int32_t channels = mixed_spin && (system & 1) == 0 ? 2 : 1;
      spin_channels_host[static_cast<std::size_t>(system)] = channels;
      spin_channel_offsets_host[static_cast<std::size_t>(system + 1)] =
          spin_channel_offsets_host[static_cast<std::size_t>(system)] + channels;
      spin_atom_offsets_host[static_cast<std::size_t>(system + 1)] =
          spin_atom_offsets_host[static_cast<std::size_t>(system)] +
          channels * (atom_offsets_host[static_cast<std::size_t>(system + 1)] -
                      atom_offsets_host[static_cast<std::size_t>(system)]);
      spin_shell_offsets_host[static_cast<std::size_t>(system + 1)] =
          spin_shell_offsets_host[static_cast<std::size_t>(system)] +
          channels * (shell_offsets_host[static_cast<std::size_t>(system + 1)] -
                      shell_offsets_host[static_cast<std::size_t>(system)]);
      spin_orbital_offsets_host[static_cast<std::size_t>(system + 1)] =
          spin_orbital_offsets_host[static_cast<std::size_t>(system)] +
          channels * (orbital_offsets_host[static_cast<std::size_t>(system + 1)] -
                      orbital_offsets_host[static_cast<std::size_t>(system)]);
      spin_matrix_offsets_host[static_cast<std::size_t>(system + 1)] =
          spin_matrix_offsets_host[static_cast<std::size_t>(system)] +
          channels * (matrix_offsets_host[static_cast<std::size_t>(system + 1)] -
                      matrix_offsets_host[static_cast<std::size_t>(system)]);
    }
    spin_channels_total = spin_channel_offsets_host.back();
    spin_atoms = spin_atom_offsets_host.back();
    spin_shells = spin_shell_offsets_host.back();
    spin_orbitals = spin_orbital_offsets_host.back();
    spin_matrices = spin_matrix_offsets_host.back();
    vector_elements = spin_shells + 9 * spin_atoms;
    history_elements = vector_elements * history_size;
    omega_elements = batch * history_size;

    shell_to_atom_host.resize(static_cast<std::size_t>(shells));
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int64_t atom_begin = atom_offsets_host[static_cast<std::size_t>(system)];
      const std::int64_t atom_end = atom_offsets_host[static_cast<std::size_t>(system + 1)];
      const std::int64_t shell_begin = shell_offsets_host[static_cast<std::size_t>(system)];
      const std::int64_t shell_end = shell_offsets_host[static_cast<std::size_t>(system + 1)];
      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        shell_to_atom_host[static_cast<std::size_t>(shell)] =
            shell - shell_begin < 2 ? atom_begin : atom_end - 1;
      }
    }
  }

  void allocate_storage() {
    const std::size_t scale =
        static_cast<std::size_t>(batch + spin_atoms + spin_shells + spin_orbitals + spin_matrices +
                                 vector_elements + history_elements + omega_elements);
    const std::size_t double_capacity = 4096u + 64u * scale;
    staged_d.allocate(double_capacity);
    public_d.allocate(double_capacity, kPublicSentinel);
    staged_u64.allocate(256u + 16u * static_cast<std::size_t>(batch), 0u);
    public_u64.allocate(256u + 16u * static_cast<std::size_t>(batch), 19u);
    staged_status.allocate(128u + 8u * static_cast<std::size_t>(batch), XTBLOOM_STATUS_SUCCESS);
    public_status.allocate(128u + 8u * static_cast<std::size_t>(batch), XTBLOOM_STATUS_SUCCESS);
    staged_u8.allocate(128u + 8u * static_cast<std::size_t>(batch), 1u);
    public_u8.allocate(128u + 8u * static_cast<std::size_t>(batch), 0u);

    atom_offsets.allocate(atom_offsets_host.size());
    shell_offsets.allocate(shell_offsets_host.size());
    orbital_offsets.allocate(orbital_offsets_host.size());
    matrix_offsets.allocate(matrix_offsets_host.size());
    shell_to_atom.allocate(shell_to_atom_host.size());
    spin_channels.allocate(spin_channels_host.size());
    spin_channel_offsets.allocate(spin_channel_offsets_host.size());
    spin_atom_offsets.allocate(spin_atom_offsets_host.size());
    spin_shell_offsets.allocate(spin_shell_offsets_host.size());
    spin_orbital_offsets.allocate(spin_orbital_offsets_host.size());
    spin_matrix_offsets.allocate(spin_matrix_offsets_host.size());
    active_mask.allocate(static_cast<std::size_t>(batch));
    pending_statuses.allocate(static_cast<std::size_t>(batch));
    failure_records.allocate(static_cast<std::size_t>(batch));
    plan_failure.allocate(1u);
    canonical_sequence.allocate(1u);
    mixed_qat.allocate(static_cast<std::size_t>(spin_atoms));
    old_energies.allocate(static_cast<std::size_t>(batch));
    energy_changes.allocate(static_cast<std::size_t>(batch));
    next_iterations.allocate(static_cast<std::size_t>(batch));
    next_statuses.allocate(static_cast<std::size_t>(batch));
    next_converged.allocate(static_cast<std::size_t>(batch));
    system_errors.allocate(static_cast<std::size_t>(batch));
    device_error.allocate(1u);
    stage_sequence.allocate(1u);
  }

  void bind_wavefunction(Gfn2SccPublicationDeviceWavefunction& value, Arena<double>& arena) {
    value.eigenpairs = {arena.take(static_cast<std::size_t>(spin_orbitals)), spin_orbitals,
                        arena.take(static_cast<std::size_t>(spin_matrices)), spin_matrices,
                        kPlanToken};
    value.occupations = {arena.take(static_cast<std::size_t>(2 * orbitals)),
                         2 * orbitals,
                         arena.take(static_cast<std::size_t>(2 * batch)),
                         2 * batch,
                         arena.take(static_cast<std::size_t>(2 * batch)),
                         2 * batch,
                         arena.take(static_cast<std::size_t>(batch)),
                         batch,
                         kPlanToken};
    value.density.density = arena.take(static_cast<std::size_t>(spin_matrices));
    value.density.density_elements = spin_matrices;
    value.density.energy_weighted_density = arena.take(static_cast<std::size_t>(spin_matrices));
    value.density.weighted_density_elements = spin_matrices;
    value.density.band_energies = arena.take(static_cast<std::size_t>(batch));
    value.density.band_energy_elements = batch;
    value.density.occupation_sums = arena.take(static_cast<std::size_t>(batch));
    value.density.occupation_sum_elements = batch;
    value.density.density_traces = arena.take(static_cast<std::size_t>(batch));
    value.density.density_trace_elements = batch;
    value.density.weighted_density_traces = arena.take(static_cast<std::size_t>(batch));
    value.density.weighted_density_trace_elements = batch;
    value.density.channel_band_energies = arena.take(static_cast<std::size_t>(spin_channels_total));
    value.density.channel_band_energy_elements = spin_channels_total;
    value.density.channel_occupation_sums =
        arena.take(static_cast<std::size_t>(spin_channels_total));
    value.density.channel_occupation_sum_elements = spin_channels_total;
    value.density.channel_density_traces =
        arena.take(static_cast<std::size_t>(spin_channels_total));
    value.density.channel_density_trace_elements = spin_channels_total;
    value.density.channel_weighted_density_traces =
        arena.take(static_cast<std::size_t>(spin_channels_total));
    value.density.channel_weighted_density_trace_elements = spin_channels_total;
    value.density.plan_token = kPlanToken;
    value.population = {arena.take(static_cast<std::size_t>(spin_shells)),
                        spin_shells,
                        arena.take(static_cast<std::size_t>(spin_atoms)),
                        spin_atoms,
                        arena.take(static_cast<std::size_t>(3 * spin_atoms)),
                        3 * spin_atoms,
                        arena.take(static_cast<std::size_t>(6 * spin_atoms)),
                        6 * spin_atoms,
                        kPlanToken};
    value.plan_token = kPlanToken;
  }

  void bind_energy(Gfn2SccPublicationDeviceEnergyTrace& value, Arena<double>& arena,
                   double* entropy_alias = nullptr) {
    auto& free = value.free_energy;
    free.core = arena.take(static_cast<std::size_t>(batch));
    free.core_elements = batch;
    free.es2 = arena.take(static_cast<std::size_t>(batch));
    free.es2_elements = batch;
    free.es3 = arena.take(static_cast<std::size_t>(batch));
    free.es3_elements = batch;
    free.aes2 = arena.take(static_cast<std::size_t>(batch));
    free.aes2_elements = batch;
    free.spin = arena.take(static_cast<std::size_t>(batch));
    free.spin_elements = batch;
    free.d4_two_body = arena.take(static_cast<std::size_t>(batch));
    free.d4_two_body_elements = batch;
    free.explicit_point_charge = arena.take(static_cast<std::size_t>(batch));
    free.explicit_point_charge_elements = batch;
    free.periodic_embedding = arena.take(static_cast<std::size_t>(batch));
    free.periodic_embedding_elements = batch;
    free.entropy =
        entropy_alias == nullptr ? arena.take(static_cast<std::size_t>(batch)) : entropy_alias;
    free.entropy_elements = batch;
    free.internal_energy = arena.take(static_cast<std::size_t>(batch));
    free.internal_energy_elements = batch;
    free.free_energy = arena.take(static_cast<std::size_t>(batch));
    free.free_energy_elements = batch;
    free.plan_token = kPlanToken;
    value.spin_energies = free.spin;
    value.spin_energy_elements = batch;
    value.classical = {free.es2,
                       batch,
                       free.es3,
                       batch,
                       free.aes2,
                       batch,
                       free.d4_two_body,
                       batch,
                       free.explicit_point_charge,
                       batch,
                       free.periodic_embedding,
                       batch,
                       arena.take(static_cast<std::size_t>(batch)),
                       batch,
                       kPlanToken};
    value.plan_token = kPlanToken;
  }

  void bind_mixer(Gfn2SccMixerDeviceState& value, Arena<double>& doubles,
                  Arena<std::uint64_t>& uint64s, Arena<xtbloom_status_t>& statuses,
                  Arena<std::uint8_t>& bytes) {
    value.current_inputs = doubles.take(static_cast<std::size_t>(vector_elements));
    value.previous_inputs = doubles.take(static_cast<std::size_t>(vector_elements));
    value.previous_residuals = doubles.take(static_cast<std::size_t>(vector_elements));
    value.df_history = doubles.take(static_cast<std::size_t>(history_elements));
    value.u_history = doubles.take(static_cast<std::size_t>(history_elements));
    value.omega = doubles.take(static_cast<std::size_t>(omega_elements));
    value.residual_rms = doubles.take(static_cast<std::size_t>(batch));
    value.residual_maximum = doubles.take(static_cast<std::size_t>(batch));
    value.iterations = uint64s.take(static_cast<std::size_t>(batch));
    value.restart_counts = uint64s.take(static_cast<std::size_t>(batch));
    value.system_statuses = statuses.take(static_cast<std::size_t>(batch));
    value.initialized = bytes.take(static_cast<std::size_t>(batch));
    value.residual_converged = bytes.take(static_cast<std::size_t>(batch));
    value.total_vector_elements = vector_elements;
    value.history_elements = history_elements;
    value.omega_elements = omega_elements;
    value.batch_elements = batch;
    value.plan_token = kPlanToken;
  }

  Gfn2SccDeviceMultipoles bind_multipoles(Arena<double>& arena) {
    return {arena.take(static_cast<std::size_t>(spin_shells)),
            spin_shells,
            arena.take(static_cast<std::size_t>(3 * spin_atoms)),
            3 * spin_atoms,
            arena.take(static_cast<std::size_t>(6 * spin_atoms)),
            6 * spin_atoms,
            kPlanToken};
  }

  void bind_descriptors() {
    plan.batch_size = batch;
    plan.total_atoms = atoms;
    plan.total_shells = shells;
    plan.total_orbitals = orbitals;
    plan.total_matrix_elements = matrices;
    plan.total_mixer_vector_elements = vector_elements;
    plan.history_size = history_size;
    plan.atom_offset_count = batch + 1;
    plan.shell_offset_count = batch + 1;
    plan.orbital_offset_count = batch + 1;
    plan.matrix_offset_count = batch + 1;
    plan.shell_to_atom_count = shells;
    plan.atom_offsets = atom_offsets.get();
    plan.shell_offsets = shell_offsets.get();
    plan.orbital_offsets = orbital_offsets.get();
    plan.matrix_offsets = matrix_offsets.get();
    plan.shell_to_atom = shell_to_atom.get();
    plan.wavefunction_layout.memory_space = xtbloom::detail::Gfn2PlanMemorySpace::kCudaDevice;
    plan.wavefunction_layout.plan_token = kPlanToken;
    plan.wavefunction_layout.layout_fingerprint = 0x51cc0deULL;
    plan.wavefunction_layout.batch_size = batch;
    plan.wavefunction_layout.total_spin_channels = spin_channels_total;
    plan.wavefunction_layout.total_spin_orbitals = spin_orbitals;
    plan.wavefunction_layout.total_spin_matrix_elements = spin_matrices;
    plan.wavefunction_layout.total_spin_shells = spin_shells;
    plan.wavefunction_layout.total_spin_atoms = spin_atoms;
    plan.wavefunction_layout.spin_channel_count = batch;
    plan.wavefunction_layout.spin_channel_offset_count = batch + 1;
    plan.wavefunction_layout.spin_orbital_offset_count = batch + 1;
    plan.wavefunction_layout.spin_matrix_offset_count = batch + 1;
    plan.wavefunction_layout.spin_shell_offset_count = batch + 1;
    plan.wavefunction_layout.spin_atom_offset_count = batch + 1;
    plan.wavefunction_layout.spin_channels = spin_channels.get();
    plan.wavefunction_layout.spin_channel_offsets = spin_channel_offsets.get();
    plan.wavefunction_layout.spin_orbital_offsets = spin_orbital_offsets.get();
    plan.wavefunction_layout.spin_matrix_offsets = spin_matrix_offsets.get();
    plan.wavefunction_layout.spin_shell_offsets = spin_shell_offsets.get();
    plan.wavefunction_layout.spin_atom_offsets = spin_atom_offsets.get();
    plan.maximum_iterations = 5u;
    plan.residual_rms_tolerance = 0.1;
    plan.energy_tolerance = 0.1;
    plan.plan_token = kPlanToken;

    ledger = {active_mask.get(),
              pending_statuses.get(),
              failure_records.get(),
              plan_failure.get(),
              canonical_sequence.get(),
              batch,
              1,
              kPlanToken};
    activity = {active_mask.get(), canonical_sequence.get(), batch, 1, kPlanToken};

    bind_wavefunction(staged.wavefunction, staged_d);
    bind_energy(staged.energy, staged_d, staged.wavefunction.occupations.entropies);
    bind_mixer(staged.mixer, staged_d, staged_u64, staged_status, staged_u8);
    const Gfn2SccDeviceMultipoles next = bind_multipoles(staged_d);
    staged.next_mixed = {next.shell_charges,
                         next.shell_elements,
                         next.atomic_dipoles,
                         next.dipole_elements,
                         next.atomic_quadrupoles,
                         next.quadrupole_elements,
                         kPlanToken};
    staged.plan_token = kPlanToken;

    bind_wavefunction(public_state.wavefunction, public_d);
    bind_energy(public_state.energy, public_d);
    bind_mixer(public_state.mixer, public_d, public_u64, public_status, public_u8);
    public_state.published = {public_state.wavefunction.population.qsh,
                              spin_shells,
                              public_state.wavefunction.population.dipole,
                              3 * spin_atoms,
                              public_state.wavefunction.population.quadrupole,
                              6 * spin_atoms,
                              kPlanToken};
    public_state.scc.current_inputs = bind_multipoles(public_d);
    public_state.scc.free_energies = public_d.take(static_cast<std::size_t>(batch));
    public_state.scc.previous_free_energies = public_d.take(static_cast<std::size_t>(batch));
    public_state.scc.free_energy_changes = public_d.take(static_cast<std::size_t>(batch));
    public_state.scc.residual_rms = public_d.take(static_cast<std::size_t>(batch));
    public_state.scc.iterations = public_u64.take(static_cast<std::size_t>(batch));
    public_state.scc.system_statuses = public_status.take(static_cast<std::size_t>(batch));
    public_state.scc.converged = public_u8.take(static_cast<std::size_t>(batch));
    public_state.scc.batch_elements = batch;
    public_state.scc.plan_token = kPlanToken;
    public_state.plan_token = kPlanToken;

    workspace = {mixed_qat.get(),
                 spin_atoms,
                 old_energies.get(),
                 energy_changes.get(),
                 next_iterations.get(),
                 next_statuses.get(),
                 next_converged.get(),
                 batch,
                 system_errors.get(),
                 batch,
                 device_error.get(),
                 1,
                 stage_sequence.get(),
                 1,
                 kPlanToken};
    report.stage = Gfn2SccStageId::kStatePublication;
    report.system_code_format = Gfn2SccStageCodeFormat::kUint32Error;
    report.system_codes = system_errors.get();
    report.system_code_elements = batch;
    report.device_error = device_error.get();
    report.device_error_elements = 1;
    report.stage_sequence_active = stage_sequence.get();
    report.stage_sequence_elements = 1;
    report.peer_error_mask = kGfn2SccPublicationPeerErrorMask;
    report.peer_failure_status = XTBLOOM_STATUS_INTERNAL_ERROR;
    report.plan_token = kPlanToken;
    report.device_code_role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
  }

  void initialize_values() {
    for (std::size_t index = 0; index < staged_d.host.size(); ++index) {
      staged_d.host[index] = 0.125 + 0.00025 * static_cast<double>(index);
    }
    std::fill(public_d.host.begin(), public_d.host.end(), kPublicSentinel);
    std::fill(staged_u64.host.begin(), staged_u64.host.end(), 7u);
    std::fill(public_u64.host.begin(), public_u64.host.end(), 19u);
    std::fill(staged_status.host.begin(), staged_status.host.end(), XTBLOOM_STATUS_SUCCESS);
    std::fill(public_status.host.begin(), public_status.host.end(), XTBLOOM_STATUS_SUCCESS);
    std::fill(staged_u8.host.begin(), staged_u8.host.end(), 1u);
    std::fill(public_u8.host.begin(), public_u8.host.end(), 0u);

    for (std::int64_t shell = 0; shell < spin_shells; ++shell) {
      staged_d.at(staged.wavefunction.population.qsh, static_cast<std::size_t>(shell)) =
          0.01 * static_cast<double>(shell + 1);
      staged_d.at(staged.next_mixed.shell_charges, static_cast<std::size_t>(shell)) =
          0.02 * static_cast<double>(shell + 1);
    }
    for (std::int64_t atom = 0; atom < spin_atoms; ++atom) {
      staged_d.at(staged.wavefunction.population.qat, static_cast<std::size_t>(atom)) =
          -0.03 * static_cast<double>(atom + 1);
    }
    for (std::int64_t element = 0; element < 3 * spin_atoms; ++element) {
      staged_d.at(staged.wavefunction.population.dipole, static_cast<std::size_t>(element)) =
          0.04 * static_cast<double>(element + 1);
      staged_d.at(staged.next_mixed.atomic_dipoles, static_cast<std::size_t>(element)) =
          0.05 * static_cast<double>(element + 1);
    }
    for (std::int64_t element = 0; element < 6 * spin_atoms; ++element) {
      staged_d.at(staged.wavefunction.population.quadrupole, static_cast<std::size_t>(element)) =
          0.006 * static_cast<double>(element + 1);
      staged_d.at(staged.next_mixed.atomic_quadrupoles, static_cast<std::size_t>(element)) =
          0.007 * static_cast<double>(element + 1);
    }

    active_host.assign(static_cast<std::size_t>(batch), 1u);
    pending_host.assign(static_cast<std::size_t>(batch), XTBLOOM_STATUS_SUCCESS);
    failure_host.assign(static_cast<std::size_t>(batch), 0u);
    plan_failure_host.assign(1u, 0u);
    canonical_sequence_host.assign(1u, 1u);
    for (std::int64_t system = 0; system < batch; ++system) {
      public_u64.at(public_state.scc.iterations, static_cast<std::size_t>(system)) = 0u;
      public_status.at(public_state.scc.system_statuses, static_cast<std::size_t>(system)) =
          XTBLOOM_STATUS_SUCCESS;
      public_u8.at(public_state.scc.converged, static_cast<std::size_t>(system)) = 0u;
      public_d.at(public_state.scc.free_energies, static_cast<std::size_t>(system)) =
          0.5 + static_cast<double>(system);
      staged_d.at(staged.energy.free_energy.free_energy, static_cast<std::size_t>(system)) =
          0.25 + 0.01 * static_cast<double>(system);
      staged_d.at(staged.mixer.residual_rms, static_cast<std::size_t>(system)) = 0.2;
      staged_status.at(staged.mixer.system_statuses, static_cast<std::size_t>(system)) =
          XTBLOOM_STATUS_SUCCESS;
      staged_u8.at(staged.mixer.initialized, static_cast<std::size_t>(system)) = 1u;
    }

    if (complex && batch >= 6) {
      public_u64.at(public_state.scc.iterations, 1u) = 2u;
      public_d.at(public_state.scc.free_energies, 1u) = 1.0;
      staged_d.at(staged.energy.free_energy.free_energy, 1u) = 1.05;
      staged_d.at(staged.mixer.residual_rms, 1u) = 0.05;

      public_u64.at(public_state.scc.iterations, 2u) = 4u;
      public_d.at(public_state.scc.free_energies, 2u) = 2.0;
      staged_d.at(staged.energy.free_energy.free_energy, 2u) = 2.2;

      active_host[3] = 0u;
      public_u8.at(public_state.scc.converged, 3u) = 1u;
      const std::int64_t inactive_shell = shell_offsets_host[3];
      staged_d.at(staged.next_mixed.shell_charges, static_cast<std::size_t>(inactive_shell)) =
          std::numeric_limits<double>::quiet_NaN();
      staged_d.at(staged.wavefunction.population.qsh, static_cast<std::size_t>(inactive_shell)) =
          std::numeric_limits<double>::quiet_NaN();
      staged_d.at(staged.mixer.residual_rms, 3u) = std::numeric_limits<double>::quiet_NaN();
      staged_d.at(staged.energy.free_energy.free_energy, 3u) =
          std::numeric_limits<double>::quiet_NaN();

      public_u64.at(public_state.scc.iterations, 4u) = 1u;
      public_d.at(public_state.scc.free_energies, 4u) = 4.0;
      staged_d.at(staged.energy.free_energy.free_energy, 4u) = 4.2;
      const std::int64_t failed_shell = shell_offsets_host[4];
      staged_d.at(staged.next_mixed.shell_charges, static_cast<std::size_t>(failed_shell)) =
          std::numeric_limits<double>::quiet_NaN();

      public_u64.at(public_state.scc.iterations, 5u) = 1u;
      public_d.at(public_state.scc.free_energies, 5u) = 5.0;
      staged_d.at(staged.energy.free_energy.free_energy, 5u) = 5.2;
    }
  }

  cudaError_t upload(cudaStream_t stream = nullptr) const {
    cudaError_t status = atom_offsets.upload(atom_offsets_host, stream);
    if (status != cudaSuccess) return status;
    status = shell_offsets.upload(shell_offsets_host, stream);
    if (status != cudaSuccess) return status;
    status = orbital_offsets.upload(orbital_offsets_host, stream);
    if (status != cudaSuccess) return status;
    status = matrix_offsets.upload(matrix_offsets_host, stream);
    if (status != cudaSuccess) return status;
    status = shell_to_atom.upload(shell_to_atom_host, stream);
    if (status != cudaSuccess) return status;
    status = spin_channels.upload(spin_channels_host, stream);
    if (status != cudaSuccess) return status;
    status = spin_channel_offsets.upload(spin_channel_offsets_host, stream);
    if (status != cudaSuccess) return status;
    status = spin_atom_offsets.upload(spin_atom_offsets_host, stream);
    if (status != cudaSuccess) return status;
    status = spin_shell_offsets.upload(spin_shell_offsets_host, stream);
    if (status != cudaSuccess) return status;
    status = spin_orbital_offsets.upload(spin_orbital_offsets_host, stream);
    if (status != cudaSuccess) return status;
    status = spin_matrix_offsets.upload(spin_matrix_offsets_host, stream);
    if (status != cudaSuccess) return status;
    status = staged_d.upload(stream);
    if (status != cudaSuccess) return status;
    status = staged_u64.upload(stream);
    if (status != cudaSuccess) return status;
    status = staged_status.upload(stream);
    if (status != cudaSuccess) return status;
    status = staged_u8.upload(stream);
    if (status != cudaSuccess) return status;
    status = public_d.upload(stream);
    if (status != cudaSuccess) return status;
    status = public_u64.upload(stream);
    if (status != cudaSuccess) return status;
    status = public_status.upload(stream);
    if (status != cudaSuccess) return status;
    status = public_u8.upload(stream);
    if (status != cudaSuccess) return status;
    status = active_mask.upload(active_host, stream);
    if (status != cudaSuccess) return status;
    status = pending_statuses.upload(pending_host, stream);
    if (status != cudaSuccess) return status;
    status = failure_records.upload(failure_host, stream);
    if (status != cudaSuccess) return status;
    status = plan_failure.upload(plan_failure_host, stream);
    if (status != cudaSuccess) return status;
    return canonical_sequence.upload(canonical_sequence_host, stream);
  }

  cudaError_t snapshot(PublicSnapshot& result, cudaStream_t stream = nullptr) const {
    cudaError_t status = public_d.download(result.doubles, stream);
    if (status != cudaSuccess) return status;
    status = public_u64.download(result.uint64s, stream);
    if (status != cudaSuccess) return status;
    status = public_status.download(result.statuses, stream);
    if (status != cudaSuccess) return status;
    return public_u8.download(result.bytes, stream);
  }

  cudaError_t enqueue(cudaStream_t stream = nullptr) const {
    cudaError_t status = reset_gfn2_scc_publication_errors_cuda(plan, workspace, stream);
    if (status != cudaSuccess) return status;
    status = preflight_gfn2_scc_publication_cuda(plan, activity, ledger, staged, public_state,
                                                 workspace, stream);
    if (status != cudaSuccess) return status;
    status = normalize_gfn2_scc_stage_cuda(report, ledger, stream);
    if (status != cudaSuccess) return status;
    return commit_gfn2_scc_publication_cuda(plan, activity, ledger, staged, public_state, workspace,
                                            stream);
  }
};

bool snapshot_equal(const PublicSnapshot& first, const PublicSnapshot& second) {
  return vector_byte_equal(first.doubles, second.doubles) &&
         vector_byte_equal(first.uint64s, second.uint64s) &&
         vector_byte_equal(first.statuses, second.statuses) &&
         vector_byte_equal(first.bytes, second.bytes);
}

int test_ragged_transaction_and_peer_failure() {
  Fixture fixture(6, true);
  CUDA_CHECK(fixture.upload());
  PublicSnapshot before;
  CUDA_CHECK(fixture.snapshot(before));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(fixture.enqueue());

  PublicSnapshot after;
  CUDA_CHECK(fixture.snapshot(after));
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  std::vector<std::uint32_t> stage_sequence;
  std::vector<std::uint8_t> active;
  std::vector<xtbloom_status_t> pending;
  std::vector<std::uint64_t> failures;
  CUDA_CHECK(fixture.system_errors.download(system_errors));
  CUDA_CHECK(fixture.device_error.download(device_error));
  CUDA_CHECK(fixture.stage_sequence.download(stage_sequence));
  CUDA_CHECK(fixture.active_mask.download(active));
  CUDA_CHECK(fixture.pending_statuses.download(pending));
  CUDA_CHECK(fixture.failure_records.download(failures));
  CUDA_CHECK(cudaDeviceSynchronize());

  CHECK(device_error[0] == 0u);
  CHECK(stage_sequence[0] == 1u);
  CHECK(system_errors[4] ==
        static_cast<std::uint32_t>(Gfn2SccPublicationDeviceError::kNonfiniteNextMixedMultipole));
  CHECK(active[4] == 0u);
  CHECK(pending[4] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(gfn2_scc_failure_stage(failures[4]) == Gfn2SccStageId::kStatePublication);
  CHECK(gfn2_scc_failure_code(failures[4]) == 4u);

  const auto& plan = fixture.plan;
  const auto check_wavefunction = [&](std::int64_t system) {
    const std::int64_t orbital_begin = fixture.orbital_offsets_host[system];
    const std::int64_t orbital_end = fixture.orbital_offsets_host[system + 1];
    const std::int64_t matrix_begin = fixture.matrix_offsets_host[system];
    const std::int64_t matrix_end = fixture.matrix_offsets_host[system + 1];
    return staged_matches_public(
               fixture.staged_d, fixture.staged.wavefunction.eigenpairs.eigenvalues,
               fixture.public_d, fixture.public_state.wavefunction.eigenpairs.eigenvalues,
               after.doubles, orbital_begin, orbital_end) &&
           staged_matches_public(
               fixture.staged_d, fixture.staged.wavefunction.eigenpairs.coefficients,
               fixture.public_d, fixture.public_state.wavefunction.eigenpairs.coefficients,
               after.doubles, matrix_begin, matrix_end) &&
           staged_matches_public(
               fixture.staged_d, fixture.staged.wavefunction.occupations.occupations,
               fixture.public_d, fixture.public_state.wavefunction.occupations.occupations,
               after.doubles, 2 * orbital_begin, 2 * orbital_end) &&
           staged_matches_public(fixture.staged_d, fixture.staged.wavefunction.density.density,
                                 fixture.public_d,
                                 fixture.public_state.wavefunction.density.density, after.doubles,
                                 matrix_begin, matrix_end) &&
           staged_matches_public(
               fixture.staged_d, fixture.staged.wavefunction.density.energy_weighted_density,
               fixture.public_d, fixture.public_state.wavefunction.density.energy_weighted_density,
               after.doubles, matrix_begin, matrix_end);
  };
  CHECK(check_wavefunction(0));
  CHECK(check_wavefunction(1));
  CHECK(check_wavefunction(2));
  CHECK(check_wavefunction(5));

  const auto check_population = [&](std::int64_t system, bool converged) {
    const std::int64_t atom_begin = fixture.atom_offsets_host[system];
    const std::int64_t atom_end = fixture.atom_offsets_host[system + 1];
    const std::int64_t shell_begin = fixture.shell_offsets_host[system];
    const std::int64_t shell_end = fixture.shell_offsets_host[system + 1];
    const double* expected_qsh = converged ? fixture.staged.wavefunction.population.qsh
                                           : fixture.staged.next_mixed.shell_charges;
    const double* expected_dipole = converged ? fixture.staged.wavefunction.population.dipole
                                              : fixture.staged.next_mixed.atomic_dipoles;
    const double* expected_quadrupole = converged
                                            ? fixture.staged.wavefunction.population.quadrupole
                                            : fixture.staged.next_mixed.atomic_quadrupoles;
    if (!staged_matches_public(fixture.staged_d, expected_qsh, fixture.public_d,
                               fixture.public_state.published.shell_charges, after.doubles,
                               shell_begin, shell_end) ||
        !staged_matches_public(fixture.staged_d, expected_dipole, fixture.public_d,
                               fixture.public_state.published.atomic_dipoles, after.doubles,
                               3 * atom_begin, 3 * atom_end) ||
        !staged_matches_public(fixture.staged_d, expected_quadrupole, fixture.public_d,
                               fixture.public_state.published.atomic_quadrupoles, after.doubles,
                               6 * atom_begin, 6 * atom_end)) {
      return false;
    }
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      double expected_qat = fixture.staged_d.at(fixture.staged.wavefunction.population.qat,
                                                static_cast<std::size_t>(atom));
      if (!converged) {
        expected_qat = 0.0;
        for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
          if (fixture.shell_to_atom_host[static_cast<std::size_t>(shell)] == atom) {
            expected_qat += fixture.staged_d.at(fixture.staged.next_mixed.shell_charges,
                                                static_cast<std::size_t>(shell));
          }
        }
      }
      const std::size_t public_index =
          fixture.public_d.offset(fixture.public_state.wavefunction.population.qat) +
          static_cast<std::size_t>(atom);
      if (!byte_equal(expected_qat, after.doubles[public_index])) {
        return false;
      }
    }
    return true;
  };
  CHECK(check_population(0, false));
  CHECK(check_population(1, true));
  CHECK(check_population(2, false));
  CHECK(check_population(5, false));

  const auto public_iteration = [&](std::int64_t system) {
    return after.uint64s[fixture.public_u64.offset(fixture.public_state.scc.iterations) +
                         static_cast<std::size_t>(system)];
  };
  const auto public_system_status = [&](std::int64_t system) {
    return after.statuses[fixture.public_status.offset(fixture.public_state.scc.system_statuses) +
                          static_cast<std::size_t>(system)];
  };
  const auto public_converged = [&](std::int64_t system) {
    return after.bytes[fixture.public_u8.offset(fixture.public_state.scc.converged) +
                       static_cast<std::size_t>(system)];
  };
  CHECK(public_iteration(0) == 1u && public_system_status(0) == XTBLOOM_STATUS_SUCCESS &&
        public_converged(0) == 0u);
  CHECK(public_iteration(1) == 3u && public_system_status(1) == XTBLOOM_STATUS_SUCCESS &&
        public_converged(1) == 1u);
  CHECK(public_iteration(2) == 5u && public_system_status(2) == XTBLOOM_STATUS_SCC_NOT_CONVERGED &&
        public_converged(2) == 0u);
  CHECK(public_iteration(5) == 2u && public_system_status(5) == XTBLOOM_STATUS_SUCCESS);

  const std::int64_t inactive_atom_begin = fixture.atom_offsets_host[3];
  const std::int64_t inactive_atom_end = fixture.atom_offsets_host[4];
  const std::int64_t inactive_shell_begin = fixture.shell_offsets_host[3];
  const std::int64_t inactive_shell_end = fixture.shell_offsets_host[4];
  CHECK(public_range_unchanged(fixture.public_d, before.doubles, after.doubles,
                               fixture.public_state.published.shell_charges, inactive_shell_begin,
                               inactive_shell_end));
  CHECK(public_range_unchanged(fixture.public_d, before.doubles, after.doubles,
                               fixture.public_state.published.atomic_dipoles,
                               3 * inactive_atom_begin, 3 * inactive_atom_end));
  CHECK(public_range_unchanged(fixture.public_u64, before.uint64s, after.uint64s,
                               fixture.public_state.scc.iterations, 3, 4));
  CHECK(public_range_unchanged(fixture.public_status, before.statuses, after.statuses,
                               fixture.public_state.scc.system_statuses, 3, 4));
  CHECK(public_range_unchanged(fixture.public_u8, before.bytes, after.bytes,
                               fixture.public_state.scc.converged, 3, 4));

  const std::int64_t failed_atom_begin = fixture.atom_offsets_host[4];
  const std::int64_t failed_atom_end = fixture.atom_offsets_host[5];
  const std::int64_t failed_shell_begin = fixture.shell_offsets_host[4];
  const std::int64_t failed_shell_end = fixture.shell_offsets_host[5];
  CHECK(public_range_unchanged(fixture.public_d, before.doubles, after.doubles,
                               fixture.public_state.published.shell_charges, failed_shell_begin,
                               failed_shell_end));
  CHECK(public_range_unchanged(fixture.public_d, before.doubles, after.doubles,
                               fixture.public_state.published.atomic_dipoles, 3 * failed_atom_begin,
                               3 * failed_atom_end));
  CHECK(public_range_unchanged(fixture.public_d, before.doubles, after.doubles,
                               fixture.public_state.wavefunction.eigenpairs.eigenvalues,
                               fixture.orbital_offsets_host[4], fixture.orbital_offsets_host[5]));
  CHECK(public_range_unchanged(fixture.public_d, before.doubles, after.doubles,
                               fixture.public_state.mixer.current_inputs,
                               fixture.shell_offsets_host[4] + 9 * failed_atom_begin,
                               fixture.shell_offsets_host[5] + 9 * failed_atom_end));
  CHECK(public_iteration(4) == 2u);
  CHECK(public_system_status(4) == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(public_converged(4) == 0u);
  const std::vector<double*> nan_fields{
      fixture.public_state.energy.free_energy.core,
      fixture.public_state.energy.free_energy.es2,
      fixture.public_state.energy.free_energy.es3,
      fixture.public_state.energy.free_energy.aes2,
      fixture.public_state.energy.spin_energies,
      fixture.public_state.energy.free_energy.d4_two_body,
      fixture.public_state.energy.free_energy.explicit_point_charge,
      fixture.public_state.energy.free_energy.periodic_embedding,
      fixture.public_state.energy.free_energy.entropy,
      fixture.public_state.energy.free_energy.internal_energy,
      fixture.public_state.energy.free_energy.free_energy,
      fixture.public_state.energy.classical.classical_total,
      fixture.public_state.wavefunction.density.band_energies,
      fixture.public_state.scc.previous_free_energies,
      fixture.public_state.scc.free_energies,
      fixture.public_state.scc.free_energy_changes,
      fixture.public_state.scc.residual_rms};
  for (const double* field : nan_fields) {
    CHECK(std::isnan(after.doubles[fixture.public_d.offset(field) + 4u]));
  }
  (void)plan;
  return 0;
}

int test_publication_error_domain() {
  for (std::uint32_t expected = 3u; expected <= 8u; ++expected) {
    Fixture fixture(1);
    if (expected == 3u) {
      fixture.staged_d.at(fixture.staged.next_mixed.shell_charges, 0u) =
          std::numeric_limits<double>::max();
      fixture.staged_d.at(fixture.staged.next_mixed.shell_charges, 1u) =
          std::numeric_limits<double>::max();
    } else if (expected == 4u) {
      fixture.staged_d.at(fixture.staged.next_mixed.shell_charges, 0u) =
          std::numeric_limits<double>::quiet_NaN();
    } else if (expected == 5u) {
      fixture.staged_d.at(fixture.staged.wavefunction.population.qsh, 0u) =
          std::numeric_limits<double>::quiet_NaN();
    } else if (expected == 6u) {
      fixture.staged_d.at(fixture.staged.mixer.residual_rms, 0u) =
          std::numeric_limits<double>::quiet_NaN();
    } else if (expected == 7u) {
      fixture.staged_d.at(fixture.staged.energy.free_energy.free_energy, 0u) =
          std::numeric_limits<double>::quiet_NaN();
    } else {
      fixture.public_u64.at(fixture.public_state.scc.iterations, 0u) = 1u;
      fixture.public_d.at(fixture.public_state.scc.free_energies, 0u) =
          -std::numeric_limits<double>::max();
      fixture.staged_d.at(fixture.staged.energy.free_energy.free_energy, 0u) =
          std::numeric_limits<double>::max();
    }
    CUDA_CHECK(fixture.upload());
    CUDA_CHECK(reset_gfn2_scc_publication_errors_cuda(fixture.plan, fixture.workspace));
    CUDA_CHECK(preflight_gfn2_scc_publication_cuda(fixture.plan, fixture.activity, fixture.ledger,
                                                   fixture.staged, fixture.public_state,
                                                   fixture.workspace));
    std::vector<std::uint32_t> errors;
    std::vector<std::uint32_t> device;
    std::vector<std::uint32_t> sequence;
    CUDA_CHECK(fixture.system_errors.download(errors));
    CUDA_CHECK(fixture.device_error.download(device));
    CUDA_CHECK(fixture.stage_sequence.download(sequence));
    CUDA_CHECK(cudaDeviceSynchronize());
    if (expected == 3u || expected == 8u) {
      CHECK(errors[0] == 0u);
      CHECK(device[0] == expected);
      CHECK(sequence[0] == 0u);
    } else {
      CHECK(errors[0] == expected);
      CHECK(device[0] == 0u);
      CHECK(sequence[0] == 1u);
    }
  }
  return 0;
}

int test_plan_fail_closed() {
  for (int mode = 0; mode < 3; ++mode) {
    Fixture fixture(3);
    if (mode == 0) {
      fixture.shell_offsets_host[1] = -1;
    } else if (mode == 1) {
      fixture.public_u8.at(fixture.public_state.scc.converged, 0u) = 1u;
    } else {
      fixture.public_u64.at(fixture.public_state.scc.iterations, 0u) = 1u;
      fixture.public_d.at(fixture.public_state.scc.free_energies, 0u) =
          std::numeric_limits<double>::quiet_NaN();
    }
    CUDA_CHECK(fixture.upload());
    PublicSnapshot before;
    CUDA_CHECK(fixture.snapshot(before));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(fixture.enqueue());
    PublicSnapshot after;
    CUDA_CHECK(fixture.snapshot(after));
    std::vector<std::uint32_t> device;
    std::vector<std::uint32_t> stage_sequence;
    std::vector<std::uint32_t> canonical_sequence;
    CUDA_CHECK(fixture.device_error.download(device));
    CUDA_CHECK(fixture.stage_sequence.download(stage_sequence));
    CUDA_CHECK(fixture.canonical_sequence.download(canonical_sequence));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(device[0] ==
          static_cast<std::uint32_t>(mode == 0 ? Gfn2SccPublicationDeviceError::kInvalidOffsets
                                               : Gfn2SccPublicationDeviceError::kInvalidState));
    CHECK(stage_sequence[0] == 0u);
    CHECK(canonical_sequence[0] == 0u);
    CHECK(snapshot_equal(before, after));
  }
  return 0;
}

int test_strict_convergence_and_iteration_zero_seed() {
  for (int mode = 0; mode < 3; ++mode) {
    Fixture fixture(1);
    if (mode == 0) {
      /* Iteration zero must use an exact old-energy seed of zero without
       * touching the poisoned persistent free-energy slot. */
      fixture.public_d.at(fixture.public_state.scc.free_energies, 0u) =
          std::numeric_limits<double>::quiet_NaN();
      fixture.staged_d.at(fixture.staged.energy.free_energy.free_energy, 0u) = 0.05;
      fixture.staged_d.at(fixture.staged.mixer.residual_rms, 0u) = 0.05;
    } else if (mode == 1) {
      fixture.staged_d.at(fixture.staged.energy.free_energy.free_energy, 0u) = 0.05;
      fixture.staged_d.at(fixture.staged.mixer.residual_rms, 0u) =
          fixture.plan.residual_rms_tolerance;
    } else {
      fixture.staged_d.at(fixture.staged.energy.free_energy.free_energy, 0u) =
          fixture.plan.energy_tolerance;
      fixture.staged_d.at(fixture.staged.mixer.residual_rms, 0u) = 0.05;
    }
    CUDA_CHECK(fixture.upload());
    CUDA_CHECK(fixture.enqueue());
    PublicSnapshot after;
    CUDA_CHECK(fixture.snapshot(after));
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::size_t converged = fixture.public_u8.offset(fixture.public_state.scc.converged);
    const std::size_t old_energy =
        fixture.public_d.offset(fixture.public_state.scc.previous_free_energies);
    const std::size_t energy_change =
        fixture.public_d.offset(fixture.public_state.scc.free_energy_changes);
    CHECK(after.bytes[converged] == (mode == 0 ? 1u : 0u));
    CHECK(after.doubles[old_energy] == 0.0);
    CHECK(after.doubles[energy_change] ==
          fixture.staged_d.at(fixture.staged.energy.free_energy.free_energy, 0u));
  }
  return 0;
}

int test_canonical_prior_failure_publication() {
  for (const Gfn2SccStageId stage : {Gfn2SccStageId::kPeriodicPotential, Gfn2SccStageId::kMixer}) {
    Fixture fixture(1);
    fixture.active_host[0] = 0u;
    fixture.pending_host[0] = XTBLOOM_STATUS_INTERNAL_ERROR;
    fixture.failure_host[0] = gfn2_scc_stage_failure_record(stage, 5u);
    CUDA_CHECK(fixture.upload());
    PublicSnapshot before;
    CUDA_CHECK(fixture.snapshot(before));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(fixture.enqueue());
    PublicSnapshot after;
    CUDA_CHECK(fixture.snapshot(after));
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::size_t iteration = fixture.public_u64.offset(fixture.public_state.scc.iterations);
    const std::size_t scc_status =
        fixture.public_status.offset(fixture.public_state.scc.system_statuses);
    const std::size_t mixer_status =
        fixture.public_status.offset(fixture.public_state.mixer.system_statuses);
    CHECK(after.statuses[scc_status] == XTBLOOM_STATUS_INTERNAL_ERROR);
    CHECK(after.uint64s[iteration] == (stage == Gfn2SccStageId::kMixer ? 1u : 0u));
    CHECK(after.statuses[mixer_status] == (stage == Gfn2SccStageId::kMixer
                                               ? XTBLOOM_STATUS_INTERNAL_ERROR
                                               : before.statuses[mixer_status]));
    CHECK(public_range_unchanged(fixture.public_d, before.doubles, after.doubles,
                                 fixture.public_state.mixer.current_inputs, 0,
                                 fixture.vector_elements));
    CHECK(public_range_unchanged(fixture.public_d, before.doubles, after.doubles,
                                 fixture.public_state.mixer.df_history, 0,
                                 fixture.history_elements));
    CHECK(std::isnan(after.doubles[fixture.public_d.offset(
        fixture.public_state.wavefunction.density.band_energies)]));
    CHECK(std::isnan(after.doubles[fixture.public_d.offset(
        fixture.public_state.energy.free_energy.free_energy)]));
  }
  return 0;
}

int test_mixed_spin_transaction() {
  Fixture fixture(4, false, true);
  CHECK(fixture.public_state.energy.spin_energies == fixture.public_state.energy.free_energy.spin);
  CHECK(fixture.staged.energy.spin_energies == fixture.staged.energy.free_energy.spin);
  CUDA_CHECK(fixture.upload());
  CUDA_CHECK(fixture.enqueue());
  PublicSnapshot after;
  CUDA_CHECK(fixture.snapshot(after));
  CUDA_CHECK(cudaDeviceSynchronize());

  for (std::int64_t system = 0; system < fixture.batch; ++system) {
    const std::int64_t orbital_begin = fixture.spin_orbital_offsets_host[system];
    const std::int64_t orbital_end = fixture.spin_orbital_offsets_host[system + 1];
    const std::int64_t matrix_begin = fixture.spin_matrix_offsets_host[system];
    const std::int64_t matrix_end = fixture.spin_matrix_offsets_host[system + 1];
    const std::int64_t channel_begin = fixture.spin_channel_offsets_host[system];
    const std::int64_t channel_end = fixture.spin_channel_offsets_host[system + 1];
    const std::int64_t shell_begin = fixture.spin_shell_offsets_host[system];
    const std::int64_t shell_end = fixture.spin_shell_offsets_host[system + 1];
    const std::int64_t atom_begin = fixture.spin_atom_offsets_host[system];
    const std::int64_t atom_end = fixture.spin_atom_offsets_host[system + 1];
    const std::int64_t vector_begin = shell_begin + 9 * atom_begin;
    const std::int64_t vector_end = shell_end + 9 * atom_end;

    CHECK(staged_matches_public(
        fixture.staged_d, fixture.staged.wavefunction.eigenpairs.eigenvalues, fixture.public_d,
        fixture.public_state.wavefunction.eigenpairs.eigenvalues, after.doubles, orbital_begin,
        orbital_end));
    CHECK(staged_matches_public(
        fixture.staged_d, fixture.staged.wavefunction.eigenpairs.coefficients, fixture.public_d,
        fixture.public_state.wavefunction.eigenpairs.coefficients, after.doubles, matrix_begin,
        matrix_end));
    CHECK(staged_matches_public(fixture.staged_d, fixture.staged.wavefunction.density.density,
                                fixture.public_d, fixture.public_state.wavefunction.density.density,
                                after.doubles, matrix_begin, matrix_end));
    CHECK(staged_matches_public(
        fixture.staged_d, fixture.staged.wavefunction.density.channel_band_energies,
        fixture.public_d, fixture.public_state.wavefunction.density.channel_band_energies,
        after.doubles, channel_begin, channel_end));
    CHECK(staged_matches_public(fixture.staged_d, fixture.staged.next_mixed.shell_charges,
                                fixture.public_d, fixture.public_state.published.shell_charges,
                                after.doubles, shell_begin, shell_end));
    CHECK(staged_matches_public(fixture.staged_d, fixture.staged.mixer.current_inputs,
                                fixture.public_d, fixture.public_state.mixer.current_inputs,
                                after.doubles, vector_begin, vector_end));

    const std::int64_t physical_shell_begin = fixture.shell_offsets_host[system];
    const std::int64_t physical_shells =
        fixture.shell_offsets_host[system + 1] - physical_shell_begin;
    const std::int64_t physical_atom_begin = fixture.atom_offsets_host[system];
    const std::int64_t physical_atoms = fixture.atom_offsets_host[system + 1] - physical_atom_begin;
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::int64_t local_atom = atom - atom_begin;
      const std::int64_t channel = local_atom / physical_atoms;
      const std::int64_t physical_atom = physical_atom_begin + local_atom % physical_atoms;
      double expected = 0.0;
      for (std::int64_t local_shell = 0; local_shell < physical_shells; ++local_shell) {
        const std::int64_t physical_shell = physical_shell_begin + local_shell;
        if (fixture.shell_to_atom_host[static_cast<std::size_t>(physical_shell)] == physical_atom) {
          expected += fixture.staged_d.at(
              fixture.staged.next_mixed.shell_charges,
              static_cast<std::size_t>(shell_begin + channel * physical_shells + local_shell));
        }
      }
      const std::size_t output =
          fixture.public_d.offset(fixture.public_state.wavefunction.population.qat) +
          static_cast<std::size_t>(atom);
      CHECK(byte_equal(expected, after.doubles[output]));
    }

    const std::size_t spin_energy =
        fixture.public_d.offset(fixture.public_state.energy.spin_energies) +
        static_cast<std::size_t>(system);
    CHECK(byte_equal(
        fixture.staged_d.at(fixture.staged.energy.spin_energies, static_cast<std::size_t>(system)),
        after.doubles[spin_energy]));
  }
  return 0;
}

int test_batch_sizes() {
  for (const bool mixed_spin : {false, true}) {
    for (const std::int64_t batch : {1, 8, 32, 128}) {
      Fixture fixture(batch, false, mixed_spin);
      CUDA_CHECK(fixture.upload());
      CUDA_CHECK(fixture.enqueue());
      PublicSnapshot after;
      std::vector<std::uint32_t> errors;
      CUDA_CHECK(fixture.snapshot(after));
      CUDA_CHECK(fixture.system_errors.download(errors));
      CUDA_CHECK(cudaDeviceSynchronize());
      CHECK(std::all_of(errors.begin(), errors.end(),
                        [](std::uint32_t value) { return value == 0u; }));
      for (std::int64_t system = 0; system < batch; ++system) {
        CHECK(after.uint64s[fixture.public_u64.offset(fixture.public_state.scc.iterations) +
                            static_cast<std::size_t>(system)] == 1u);
        CHECK(
            after.statuses[fixture.public_status.offset(fixture.public_state.scc.system_statuses) +
                           static_cast<std::size_t>(system)] == XTBLOOM_STATUS_SUCCESS);
      }
    }
  }
  return 0;
}

int test_custom_stream_graph_replay() {
  Fixture fixture(1);
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(fixture.upload(stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(fixture.enqueue(stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u));

  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  PublicSnapshot first;
  CUDA_CHECK(fixture.snapshot(first, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(first.uint64s[fixture.public_u64.offset(fixture.public_state.scc.iterations)] == 1u);
  CHECK(first.statuses[fixture.public_status.offset(fixture.public_state.scc.system_statuses)] ==
        XTBLOOM_STATUS_SUCCESS);

  const double healthy = fixture.staged_d.at(fixture.staged.next_mixed.shell_charges, 0u);
  fixture.staged_d.at(fixture.staged.next_mixed.shell_charges, 0u) =
      std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(fixture.upload(stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  PublicSnapshot failed;
  CUDA_CHECK(fixture.snapshot(failed, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(failed.statuses[fixture.public_status.offset(fixture.public_state.scc.system_statuses)] ==
        XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(failed.uint64s[fixture.public_u64.offset(fixture.public_state.scc.iterations)] == 1u);
  CHECK(
      std::isnan(failed.doubles[fixture.public_d.offset(fixture.public_state.scc.free_energies)]));

  fixture.staged_d.at(fixture.staged.next_mixed.shell_charges, 0u) = healthy;
  CUDA_CHECK(fixture.upload(stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  PublicSnapshot recovered;
  CUDA_CHECK(fixture.snapshot(recovered, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(
      recovered.statuses[fixture.public_status.offset(fixture.public_state.scc.system_statuses)] ==
      XTBLOOM_STATUS_SUCCESS);
  CHECK(recovered.uint64s[fixture.public_u64.offset(fixture.public_state.scc.iterations)] == 1u);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  int status = test_ragged_transaction_and_peer_failure();
  if (status != 0) return status;
  status = test_publication_error_domain();
  if (status != 0) return status;
  status = test_plan_fail_closed();
  if (status != 0) return status;
  status = test_canonical_prior_failure_publication();
  if (status != 0) return status;
  status = test_strict_convergence_and_iteration_zero_seed();
  if (status != 0) return status;
  status = test_mixed_spin_transaction();
  if (status != 0) return status;
  status = test_batch_sizes();
  if (status != 0) return status;
  status = test_custom_stream_graph_replay();
  if (status != 0) return status;
  std::puts("cuda_scc_publication_test: PASS");
  return 0;
}
