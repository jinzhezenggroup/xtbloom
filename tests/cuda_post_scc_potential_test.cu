#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "backends/cuda/gfn2_post_scc_potential.cuh"

#define CHECK(condition)                                                                 \
  do {                                                                                   \
    if (!(condition)) {                                                                  \
      std::fprintf(stderr, "post-SCC potential check failed at line %d: %s\n", __LINE__, \
                   #condition);                                                          \
      return __LINE__;                                                                   \
    }                                                                                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using namespace gpuxtb::detail;
using namespace gpuxtb::detail::cuda;

constexpr std::uint64_t kPlanToken = 0x109109109ULL;
constexpr std::uint64_t kGeometryGeneration = 109u;
constexpr double kSentinel = -109109.0;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream) {
    elements_ = values.size();
    if (elements_ == 0u) {
      return cudaSuccess;
    }
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&data_), elements_ * sizeof(T));
    if (status == cudaSuccess) {
      status = cudaMemcpyAsync(data_, values.data(), elements_ * sizeof(T), cudaMemcpyHostToDevice,
                               stream);
    }
    return status;
  }

  cudaError_t overwrite(const std::vector<T>& values, cudaStream_t stream) {
    return values.size() != elements_ ? cudaErrorInvalidValue
                                      : cudaMemcpyAsync(data_, values.data(), elements_ * sizeof(T),
                                                        cudaMemcpyHostToDevice, stream);
  }

  cudaError_t download(std::vector<T>& values, cudaStream_t stream) const {
    values.resize(elements_);
    return elements_ == 0u ? cudaSuccess
                           : cudaMemcpyAsync(values.data(), data_, elements_ * sizeof(T),
                                             cudaMemcpyDeviceToHost, stream);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return elements_; }

 private:
  T* data_ = nullptr;
  std::size_t elements_ = 0u;
};

bool close(double first, double second, double tolerance = 3.0e-13) noexcept {
  return std::abs(first - second) <= tolerance * std::max({1.0, std::abs(first), std::abs(second)});
}

struct Fixture {
  std::int64_t batch_size = 0;
  std::int64_t atoms = 0;
  std::int64_t shells = 0;

  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> shell_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> pair_offsets;
  std::vector<std::int64_t> dipole_offsets;
  std::vector<std::int64_t> quadrupole_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<double> shell_hardness;
  std::vector<double> es2_matrix;
  std::vector<double> gamma3;
  std::vector<double> dipole_kernel;
  std::vector<double> quadrupole_kernel;
  std::vector<double> multipole_radius;
  std::vector<double> multipole_valence;
  std::vector<double> pair_data;
  std::vector<double> explicit_shell_potential;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response;
  std::vector<double> raw_shell;
  std::vector<double> raw_atomic;
  std::vector<double> raw_dipole;
  std::vector<double> raw_quadrupole;
  std::vector<std::uint8_t> requested;
  std::vector<gpuxtb_status_t> statuses;

  DeviceBuffer<std::int64_t> d_atom_offsets;
  DeviceBuffer<std::int64_t> d_shell_offsets;
  DeviceBuffer<std::int64_t> d_qsh_offsets;
  DeviceBuffer<std::int64_t> d_atom_shell_offsets;
  DeviceBuffer<std::int64_t> d_matrix_offsets;
  DeviceBuffer<std::int64_t> d_pair_offsets;
  DeviceBuffer<std::int64_t> d_dipole_offsets;
  DeviceBuffer<std::int64_t> d_quadrupole_offsets;
  DeviceBuffer<std::int64_t> d_shell_to_atom;
  DeviceBuffer<double> d_shell_hardness;
  DeviceBuffer<double> d_es2_matrix;
  DeviceBuffer<double> d_gamma3;
  DeviceBuffer<double> d_dipole_kernel;
  DeviceBuffer<double> d_quadrupole_kernel;
  DeviceBuffer<double> d_multipole_radius;
  DeviceBuffer<double> d_multipole_valence;
  DeviceBuffer<double> d_pair_data;
  DeviceBuffer<double> d_explicit_shell_potential;
  DeviceBuffer<double> d_periodic_shifts;
  DeviceBuffer<double> d_periodic_response;
  DeviceBuffer<double> d_raw_shell;
  DeviceBuffer<double> d_raw_atomic;
  DeviceBuffer<double> d_raw_dipole;
  DeviceBuffer<double> d_raw_quadrupole;
  DeviceBuffer<std::uint8_t> d_requested;
  DeviceBuffer<gpuxtb_status_t> d_statuses;
  DeviceBuffer<std::uint64_t> d_geometry_epoch;
  DeviceBuffer<std::uint64_t> d_committed_generations;
  DeviceBuffer<std::uint8_t> d_eligible;

  DeviceBuffer<double> d_es2_shell;
  DeviceBuffer<double> d_es3_shell;
  DeviceBuffer<double> d_aes2_atomic;
  DeviceBuffer<double> d_aes2_dipole;
  DeviceBuffer<double> d_aes2_quadrupole;
  DeviceBuffer<double> d_periodic_atomic;
  DeviceBuffer<double> d_staged_shell;
  DeviceBuffer<double> d_staged_atomic;
  DeviceBuffer<double> d_staged_dipole;
  DeviceBuffer<double> d_staged_quadrupole;
  DeviceBuffer<double> d_staged_shell_scalar;
  DeviceBuffer<double> d_result_shell;
  DeviceBuffer<double> d_result_atomic;
  DeviceBuffer<double> d_result_dipole;
  DeviceBuffer<double> d_result_quadrupole;
  DeviceBuffer<double> d_result_shell_scalar;

  DeviceBuffer<double> d_es2_shell_scratch;
  DeviceBuffer<double> d_aes2_potential_scratch;
  DeviceBuffer<std::uint32_t> d_aes2_peer_scratch;
  DeviceBuffer<double> d_periodic_scratch;
  DeviceBuffer<std::uint32_t> d_periodic_sequence;
  DeviceBuffer<double> d_compose_shell_scratch;
  DeviceBuffer<double> d_compose_atom_scratch;
  DeviceBuffer<double> d_compose_dipole_scratch;
  DeviceBuffer<double> d_compose_quadrupole_scratch;
  DeviceBuffer<std::uint32_t> d_compose_sequence;
  DeviceBuffer<double> d_bridge_scratch;
  DeviceBuffer<std::uint32_t> d_bridge_sequence;
  DeviceBuffer<std::uint8_t> d_active;
  DeviceBuffer<std::uint32_t> d_sequence;
  DeviceBuffer<std::uint32_t> d_stage_system_errors;
  DeviceBuffer<std::uint32_t> d_stage_device_error;
  DeviceBuffer<std::uint32_t> d_system_errors;
  DeviceBuffer<std::uint32_t> d_device_error;

  cudaError_t initialize(std::int64_t count, cudaStream_t stream) {
    batch_size = count;
    atoms = 2 * count;
    shells = atoms;
    atom_offsets.resize(static_cast<std::size_t>(count + 1));
    shell_offsets.resize(static_cast<std::size_t>(count + 1));
    matrix_offsets.resize(static_cast<std::size_t>(count + 1));
    pair_offsets.resize(static_cast<std::size_t>(count + 1));
    dipole_offsets.resize(static_cast<std::size_t>(count + 1));
    quadrupole_offsets.resize(static_cast<std::size_t>(count + 1));
    atom_shell_offsets.resize(static_cast<std::size_t>(atoms + 1));
    shell_to_atom.resize(static_cast<std::size_t>(shells));
    shell_hardness.resize(static_cast<std::size_t>(shells));
    es2_matrix.resize(static_cast<std::size_t>(4 * count));
    gamma3.resize(static_cast<std::size_t>(shells));
    dipole_kernel.assign(static_cast<std::size_t>(atoms), 0.25);
    quadrupole_kernel.assign(static_cast<std::size_t>(atoms), 0.08);
    multipole_radius.assign(static_cast<std::size_t>(atoms), 1.0);
    multipole_valence.assign(static_cast<std::size_t>(atoms), 1.0);
    pair_data.resize(static_cast<std::size_t>(5 * count));
    explicit_shell_potential.resize(static_cast<std::size_t>(shells));
    periodic_shifts.resize(static_cast<std::size_t>(atoms));
    periodic_response.resize(static_cast<std::size_t>(4 * count));
    raw_shell.resize(static_cast<std::size_t>(shells));
    raw_atomic.resize(static_cast<std::size_t>(atoms));
    raw_dipole.assign(static_cast<std::size_t>(3 * atoms), 0.0);
    raw_quadrupole.assign(static_cast<std::size_t>(6 * atoms), 0.0);
    requested.assign(static_cast<std::size_t>(count), 1u);
    statuses.assign(static_cast<std::size_t>(count), GPUXTB_STATUS_SUCCESS);

    for (std::int64_t system = 0; system < count; ++system) {
      atom_offsets[static_cast<std::size_t>(system)] = 2 * system;
      shell_offsets[static_cast<std::size_t>(system)] = 2 * system;
      matrix_offsets[static_cast<std::size_t>(system)] = 4 * system;
      pair_offsets[static_cast<std::size_t>(system)] = system;
      dipole_offsets[static_cast<std::size_t>(system)] = 6 * system;
      quadrupole_offsets[static_cast<std::size_t>(system)] = 12 * system;
      const std::size_t matrix = static_cast<std::size_t>(4 * system);
      es2_matrix[matrix] = 1.2;
      es2_matrix[matrix + 1] = 0.55;
      es2_matrix[matrix + 2] = 0.55;
      es2_matrix[matrix + 3] = 1.8;
      const std::size_t pair = static_cast<std::size_t>(5 * system);
      pair_data[pair] = 1.0;
      pair_data[pair + 1] = 0.0;
      pair_data[pair + 2] = 0.0;
      pair_data[pair + 3] = 0.3;
      pair_data[pair + 4] = 0.2;
      periodic_response[matrix] = 0.20;
      periodic_response[matrix + 1] = 0.05;
      periodic_response[matrix + 2] = 0.05;
      periodic_response[matrix + 3] = 0.30;
      for (std::int64_t local = 0; local < 2; ++local) {
        const std::int64_t atom = 2 * system + local;
        const std::size_t index = static_cast<std::size_t>(atom);
        atom_shell_offsets[index] = atom;
        shell_to_atom[index] = atom;
        shell_hardness[index] = local == 0 ? 1.2 : 1.8;
        gamma3[index] = local == 0 ? 0.10 : 0.15;
        explicit_shell_potential[index] = 0.006 * static_cast<double>(atom + 1);
        periodic_shifts[index] = -0.004 * static_cast<double>(atom + 1);
        raw_shell[index] = 0.03 * static_cast<double>(atom + 1);
        raw_atomic[index] = -0.02 * static_cast<double>(atom + 2);
        raw_dipole[static_cast<std::size_t>(3 * atom)] = 0.01 * static_cast<double>(atom + 1);
        raw_quadrupole[static_cast<std::size_t>(6 * atom)] = 0.003 * static_cast<double>(atom + 1);
      }
    }
    atom_offsets.back() = atoms;
    shell_offsets.back() = shells;
    matrix_offsets.back() = 4 * count;
    pair_offsets.back() = count;
    dipole_offsets.back() = 3 * atoms;
    quadrupole_offsets.back() = 6 * atoms;
    atom_shell_offsets.back() = shells;

    cudaError_t status = cudaSuccess;
#define UPLOAD(name)                        \
  if (status == cudaSuccess) {              \
    status = d_##name.upload(name, stream); \
  }
    UPLOAD(atom_offsets)
    UPLOAD(shell_offsets)
    if (status == cudaSuccess) {
      status = d_qsh_offsets.upload(shell_offsets, stream);
    }
    UPLOAD(atom_shell_offsets)
    UPLOAD(matrix_offsets)
    UPLOAD(pair_offsets)
    UPLOAD(dipole_offsets)
    UPLOAD(quadrupole_offsets)
    UPLOAD(shell_to_atom)
    UPLOAD(shell_hardness)
    UPLOAD(es2_matrix)
    UPLOAD(gamma3)
    UPLOAD(dipole_kernel)
    UPLOAD(quadrupole_kernel)
    UPLOAD(multipole_radius)
    UPLOAD(multipole_valence)
    UPLOAD(pair_data)
    UPLOAD(explicit_shell_potential)
    UPLOAD(periodic_shifts)
    UPLOAD(periodic_response)
    UPLOAD(raw_shell)
    UPLOAD(raw_atomic)
    UPLOAD(raw_dipole)
    UPLOAD(raw_quadrupole)
    UPLOAD(requested)
    UPLOAD(statuses)
#undef UPLOAD
    if (status == cudaSuccess) {
      status = d_geometry_epoch.upload({kGeometryGeneration}, stream);
    }
    if (status == cudaSuccess) {
      status = d_committed_generations.upload(
          std::vector<std::uint64_t>(static_cast<std::size_t>(count), kGeometryGeneration),
          stream);
    }
    if (status == cudaSuccess) {
      status = d_eligible.upload(
          std::vector<std::uint8_t>(static_cast<std::size_t>(count), 1u), stream);
    }
    const auto doubles = [&](DeviceBuffer<double>& buffer, std::int64_t elements,
                             double value = 0.0) {
      return status == cudaSuccess
                 ? buffer.upload(std::vector<double>(static_cast<std::size_t>(elements), value),
                                 stream)
                 : status;
    };
    status = doubles(d_es2_shell, shells);
    status = doubles(d_es3_shell, shells);
    status = doubles(d_aes2_atomic, atoms);
    status = doubles(d_aes2_dipole, 3 * atoms);
    status = doubles(d_aes2_quadrupole, 6 * atoms);
    status = doubles(d_periodic_atomic, atoms);
    status = doubles(d_staged_shell, shells);
    status = doubles(d_staged_atomic, atoms);
    status = doubles(d_staged_dipole, 3 * atoms);
    status = doubles(d_staged_quadrupole, 6 * atoms);
    status = doubles(d_staged_shell_scalar, shells);
    status = doubles(d_result_shell, shells, kSentinel);
    status = doubles(d_result_atomic, atoms, kSentinel);
    status = doubles(d_result_dipole, 3 * atoms, kSentinel);
    status = doubles(d_result_quadrupole, 6 * atoms, kSentinel);
    status = doubles(d_result_shell_scalar, shells, kSentinel);
    status = doubles(d_es2_shell_scratch, shells);
    status = doubles(d_aes2_potential_scratch, 10 * atoms);
    status = doubles(d_periodic_scratch, atoms);
    status = doubles(d_compose_shell_scratch, shells);
    status = doubles(d_compose_atom_scratch, atoms);
    status = doubles(d_compose_dipole_scratch, 3 * atoms);
    status = doubles(d_compose_quadrupole_scratch, 6 * atoms);
    status = doubles(d_bridge_scratch, shells);
    if (status == cudaSuccess) {
      status = d_aes2_peer_scratch.upload({0u}, stream);
    }
    if (status == cudaSuccess) {
      status = d_periodic_sequence.upload({0u}, stream);
    }
    if (status == cudaSuccess) {
      status = d_compose_sequence.upload({0u}, stream);
    }
    if (status == cudaSuccess) {
      status = d_bridge_sequence.upload({0u}, stream);
    }
    if (status == cudaSuccess) {
      status = d_active.upload(std::vector<std::uint8_t>(static_cast<std::size_t>(count)), stream);
    }
    if (status == cudaSuccess) {
      status = d_sequence.upload({0u}, stream);
    }
    if (status == cudaSuccess) {
      status = d_stage_system_errors.upload(
          std::vector<std::uint32_t>(static_cast<std::size_t>(count)), stream);
    }
    if (status == cudaSuccess) {
      status = d_stage_device_error.upload({0u}, stream);
    }
    if (status == cudaSuccess) {
      status = d_system_errors.upload(std::vector<std::uint32_t>(static_cast<std::size_t>(count)),
                                      stream);
    }
    if (status == cudaSuccess) {
      status = d_device_error.upload({0u}, stream);
    }
    return status;
  }

  Gfn2PostSccPotentialDevicePlan plan() const {
    Gfn2PostSccPotentialDevicePlan value{};
    value.enabled_components =
        static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
        static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
        static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2) |
        static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kExplicitPointCharge) |
        static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kPeriodicEmbedding);
    value.geometry_generation = kGeometryGeneration;
    value.plan_token = kPlanToken;

    auto& batch = value.potential_batch;
    batch.batch_size = batch_size;
    batch.total_atoms = atoms;
    batch.total_shells = shells;
    batch.plan_token = kPlanToken;
    batch.atom_offset_count = batch_size + 1;
    batch.batch_shell_offset_count = batch_size + 1;
    batch.qsh_offset_count = batch_size + 1;
    batch.qat_offset_count = batch_size + 1;
    batch.dipole_offset_count = batch_size + 1;
    batch.quadrupole_offset_count = batch_size + 1;
    batch.shell_to_atom_count = shells;
    batch.atom_offsets = d_atom_offsets.get();
    batch.batch_shell_offsets = d_shell_offsets.get();
    batch.qsh_offsets = d_qsh_offsets.get();
    batch.qat_offsets = d_atom_offsets.get();
    batch.dipole_offsets = d_dipole_offsets.get();
    batch.quadrupole_offsets = d_quadrupole_offsets.get();
    batch.shell_to_atom = d_shell_to_atom.get();

    auto& topology = value.scalar_bridge_batch.topology;
    topology.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    topology.pair_map_kind = Gfn2PairMapKind::kNone;
    topology.plan_token = kPlanToken;
    topology.batch_size = batch_size;
    topology.total_atoms = atoms;
    topology.total_shells = shells;
    topology.atom_offset_count = batch_size + 1;
    topology.batch_shell_offset_count = batch_size + 1;
    topology.shell_to_atom_count = shells;
    topology.atom_offsets = d_atom_offsets.get();
    topology.batch_shell_offsets = d_shell_offsets.get();
    topology.shell_to_atom = d_shell_to_atom.get();
    value.scalar_bridge_batch.qsh_offset_count = batch_size + 1;
    value.scalar_bridge_batch.qat_offset_count = batch_size + 1;
    value.scalar_bridge_batch.qsh_offsets = d_qsh_offsets.get();
    value.scalar_bridge_batch.qat_offsets = d_atom_offsets.get();

    value.es2_batch.batch_size = batch_size;
    value.es2_batch.total_atoms = atoms;
    value.es2_batch.total_shells = shells;
    value.es2_batch.total_matrix_elements = 4 * batch_size;
    value.es2_batch.plan_token = kPlanToken;
    value.es2_batch.atom_offset_count = batch_size + 1;
    value.es2_batch.batch_shell_offset_count = batch_size + 1;
    value.es2_batch.atom_shell_offset_count = atoms + 1;
    value.es2_batch.matrix_offset_count = batch_size + 1;
    value.es2_batch.shell_to_atom_count = shells;
    value.es2_batch.shell_hardness_count = shells;
    value.es2_batch.atom_offsets = d_atom_offsets.get();
    value.es2_batch.batch_shell_offsets = d_shell_offsets.get();
    value.es2_batch.atom_shell_offsets = d_atom_shell_offsets.get();
    value.es2_batch.matrix_offsets = d_matrix_offsets.get();
    value.es2_batch.shell_to_atom = d_shell_to_atom.get();
    value.es2_batch.shell_hardness = d_shell_hardness.get();
    value.es2_cache = {const_cast<double*>(d_es2_matrix.get()), 4 * batch_size, kGeometryGeneration,
                       kPlanToken};

    value.es3_batch = {batch_size,     shells,    batch_size + 1, shells, d_shell_offsets.get(),
                       d_gamma3.get(), kPlanToken};

    value.aes2_batch.batch_size = batch_size;
    value.aes2_batch.total_atoms = atoms;
    value.aes2_batch.total_pairs = batch_size;
    value.aes2_batch.plan_token = kPlanToken;
    value.aes2_batch.atom_offset_count = batch_size + 1;
    value.aes2_batch.pair_offset_count = batch_size + 1;
    value.aes2_batch.dipole_kernel_count = atoms;
    value.aes2_batch.quadrupole_kernel_count = atoms;
    value.aes2_batch.multipole_radius_count = atoms;
    value.aes2_batch.multipole_valence_cn_count = atoms;
    value.aes2_batch.atom_offsets = d_atom_offsets.get();
    value.aes2_batch.pair_offsets = d_pair_offsets.get();
    value.aes2_batch.dipole_kernel = d_dipole_kernel.get();
    value.aes2_batch.quadrupole_kernel = d_quadrupole_kernel.get();
    value.aes2_batch.multipole_radius = d_multipole_radius.get();
    value.aes2_batch.multipole_valence_cn = d_multipole_valence.get();
    value.aes2_cache = {const_cast<double*>(d_pair_data.get()), 5 * batch_size, kGeometryGeneration,
                        kPlanToken};

    value.external_point_charge_batch.batch_size = batch_size;
    value.external_point_charge_batch.total_atoms = atoms;
    value.external_point_charge_batch.total_shells = shells;
    value.external_point_charge_batch.total_point_charges = 0;
    value.external_point_charge_batch.plan_token = kPlanToken;
    value.external_point_charge_cache = {const_cast<double*>(d_explicit_shell_potential.get()),
                                         shells, kGeometryGeneration, kPlanToken};

    value.periodic_batch.batch_size = batch_size;
    value.periodic_batch.total_atoms = atoms;
    value.periodic_batch.total_matrix_elements = 4 * batch_size;
    value.periodic_batch.atom_offset_count = batch_size + 1;
    value.periodic_batch.matrix_offset_count = batch_size + 1;
    value.periodic_batch.shift_elements = atoms;
    value.periodic_batch.response_elements = 4 * batch_size;
    value.periodic_batch.plan_token = kPlanToken;
    value.periodic_batch.atom_offsets = d_atom_offsets.get();
    value.periodic_batch.matrix_offsets = d_matrix_offsets.get();
    value.periodic_batch.shifts = d_periodic_shifts.get();
    value.periodic_batch.response_matrices = d_periodic_response.get();
    value.periodic_batch.geometry_generation = kGeometryGeneration;
    return value;
  }

  Gfn2PostSccPotentialDeviceInput input() const {
    Gfn2PostSccPotentialDeviceInput value{};
    value.activity.requested_mask = d_requested.get();
    value.activity.system_statuses = d_statuses.get();
    value.activity.batch_elements = batch_size;
    value.activity.plan_token = kPlanToken;
    value.raw_shell_charges = d_raw_shell.get();
    value.shell_elements = shells;
    value.raw_atomic_charges = d_raw_atomic.get();
    value.atom_elements = atoms;
    value.raw_atomic_dipoles = d_raw_dipole.get();
    value.dipole_elements = 3 * atoms;
    value.raw_atomic_quadrupoles = d_raw_quadrupole.get();
    value.quadrupole_elements = 6 * atoms;
    value.plan_token = kPlanToken;
    return value;
  }

  Gfn2PostSccPotentialDeviceResults results() {
    Gfn2PostSccPotentialDeviceResults value{};
    value.complete.shell = d_result_shell.get();
    value.complete.shell_elements = shells;
    value.complete.atomic = d_result_atomic.get();
    value.complete.atom_elements = atoms;
    value.complete.dipole = d_result_dipole.get();
    value.complete.dipole_elements = 3 * atoms;
    value.complete.quadrupole = d_result_quadrupole.get();
    value.complete.quadrupole_elements = 6 * atoms;
    value.complete.plan_token = kPlanToken;
    value.shell_scalar = d_result_shell_scalar.get();
    value.shell_scalar_elements = shells;
    value.plan_token = kPlanToken;
    return value;
  }

  Gfn2PostSccPotentialDeviceIntermediates intermediates() {
    Gfn2PostSccPotentialDeviceIntermediates value{};
    value.es2_shell = d_es2_shell.get();
    value.es2_shell_elements = shells;
    value.es3_shell = d_es3_shell.get();
    value.es3_shell_elements = shells;
    value.aes2_atomic = d_aes2_atomic.get();
    value.aes2_atomic_elements = atoms;
    value.aes2_dipole = d_aes2_dipole.get();
    value.aes2_dipole_elements = 3 * atoms;
    value.aes2_quadrupole = d_aes2_quadrupole.get();
    value.aes2_quadrupole_elements = 6 * atoms;
    value.periodic_atomic = d_periodic_atomic.get();
    value.periodic_atomic_elements = atoms;
    value.complete = {d_staged_shell.get(),
                      shells,
                      d_staged_atomic.get(),
                      atoms,
                      d_staged_dipole.get(),
                      3 * atoms,
                      d_staged_quadrupole.get(),
                      6 * atoms,
                      kPlanToken};
    value.shell_scalar = d_staged_shell_scalar.get();
    value.shell_scalar_elements = shells;
    value.plan_token = kPlanToken;
    return value;
  }

  Gfn2PostSccPotentialDeviceWorkspace workspace() {
    Gfn2PostSccPotentialDeviceWorkspace value{};
    value.es2.shell_scratch = d_es2_shell_scratch.get();
    value.es2.shell_elements = shells;
    value.aes2.potential_scratch = d_aes2_potential_scratch.get();
    value.aes2.potential_elements = 10 * atoms;
    value.aes2.scc_peer_error_scratch = d_aes2_peer_scratch.get();
    value.aes2.scc_peer_error_elements = 1;
    value.periodic.potential_scratch = d_periodic_scratch.get();
    value.periodic.sequence_active = d_periodic_sequence.get();
    value.periodic.atom_elements = atoms;
    value.periodic.sequence_elements = 1;
    value.periodic.plan_token = kPlanToken;
    value.composition.shell_scratch = d_compose_shell_scratch.get();
    value.composition.shell_elements = shells;
    value.composition.atom_scratch = d_compose_atom_scratch.get();
    value.composition.atom_elements = atoms;
    value.composition.dipole_scratch = d_compose_dipole_scratch.get();
    value.composition.dipole_elements = 3 * atoms;
    value.composition.quadrupole_scratch = d_compose_quadrupole_scratch.get();
    value.composition.quadrupole_elements = 6 * atoms;
    value.composition.sequence_active = d_compose_sequence.get();
    value.composition.sequence_elements = 1;
    value.composition.plan_token = kPlanToken;
    value.scalar_bridge.shell_scratch = d_bridge_scratch.get();
    value.scalar_bridge.shell_elements = shells;
    value.scalar_bridge.sequence_active = d_bridge_sequence.get();
    value.scalar_bridge.sequence_elements = 1;
    value.scalar_bridge.plan_token = kPlanToken;
    value.active_mask = d_active.get();
    value.active_elements = batch_size;
    value.sequence_active = d_sequence.get();
    value.sequence_elements = 1;
    value.stage_system_errors = d_stage_system_errors.get();
    value.stage_system_error_elements = batch_size;
    value.stage_device_error = d_stage_device_error.get();
    value.stage_device_error_elements = 1;
    value.plan_token = kPlanToken;
    return value;
  }

  Gfn2PostSccPotentialDeviceDiagnostics diagnostics() {
    return {d_system_errors.get(), d_device_error.get(), batch_size, kPlanToken};
  }

  Gfn2GeometryEpochConsumerDevice geometry_consumer() {
    return {{d_geometry_epoch.get(), 1, kPlanToken}, d_committed_generations.get(),
            d_eligible.get(), batch_size, kPlanToken};
  }

  double expected_shell(std::int64_t system, std::int64_t local,
                        const std::vector<double>& charges) const {
    const std::size_t shell = static_cast<std::size_t>(2 * system + local);
    const std::size_t matrix = static_cast<std::size_t>(4 * system);
    const double q0 = charges[static_cast<std::size_t>(2 * system)];
    const double q1 = charges[static_cast<std::size_t>(2 * system + 1)];
    const double es2 = local == 0 ? es2_matrix[matrix] * q0 + es2_matrix[matrix + 1] * q1
                                  : es2_matrix[matrix + 2] * q0 + es2_matrix[matrix + 3] * q1;
    return es2 + gamma3[shell] * charges[shell] * charges[shell] + explicit_shell_potential[shell];
  }
};

int run_batch(std::int64_t batch_size) {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Fixture fixture;
  CUDA_CHECK(fixture.initialize(batch_size, stream));

  std::vector<double> last_mixed(static_cast<std::size_t>(fixture.shells));
  for (std::int64_t shell = 0; shell < fixture.shells; ++shell) {
    last_mixed[static_cast<std::size_t>(shell)] = -0.11 * static_cast<double>(shell + 1);
  }

  if (batch_size == 8) {
    fixture.requested[1] = 0u;
    fixture.statuses[2] = GPUXTB_STATUS_INTERNAL_ERROR;
    fixture.requested[3] = 2u;
    fixture.raw_shell[2] = std::numeric_limits<double>::quiet_NaN();
    fixture.raw_shell[4] = std::numeric_limits<double>::infinity();
    fixture.raw_shell[6] = std::numeric_limits<double>::quiet_NaN();
    fixture.raw_shell[8] = std::numeric_limits<double>::quiet_NaN();
    CUDA_CHECK(fixture.d_requested.overwrite(fixture.requested, stream));
    CUDA_CHECK(fixture.d_statuses.overwrite(fixture.statuses, stream));
    CUDA_CHECK(fixture.d_raw_shell.overwrite(fixture.raw_shell, stream));
  }

  auto plan = fixture.plan();
  auto input = fixture.input();
  auto results = fixture.results();
  auto intermediates = fixture.intermediates();
  auto workspace = fixture.workspace();
  auto diagnostics = fixture.diagnostics();
  CUDA_CHECK(refresh_gfn2_post_scc_potentials_cuda(plan, input, results, intermediates, workspace,
                                                   diagnostics, stream));

  std::vector<double> shell;
  std::vector<double> atomic;
  std::vector<double> shell_scalar;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CUDA_CHECK(fixture.d_result_shell.download(shell, stream));
  CUDA_CHECK(fixture.d_result_atomic.download(atomic, stream));
  CUDA_CHECK(fixture.d_result_shell_scalar.download(shell_scalar, stream));
  CUDA_CHECK(fixture.d_system_errors.download(system_errors, stream));
  CUDA_CHECK(fixture.d_device_error.download(device_error, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(device_error[0] == 0u);

  for (std::int64_t system = 0; system < batch_size; ++system) {
    const bool suppressed = batch_size == 8 && system >= 1 && system <= 4;
    for (std::int64_t local = 0; local < 2; ++local) {
      const std::size_t index = static_cast<std::size_t>(2 * system + local);
      if (suppressed) {
        CHECK(shell[index] == kSentinel);
        CHECK(atomic[index] == kSentinel);
        CHECK(shell_scalar[index] == kSentinel);
      } else {
        const double expected = fixture.expected_shell(system, local, fixture.raw_shell);
        CHECK(close(shell[index], expected));
        CHECK(!close(shell[index], fixture.expected_shell(system, local, last_mixed), 1.0e-10));
        CHECK(std::isfinite(atomic[index]));
        CHECK(close(shell_scalar[index], shell[index] + atomic[index]));
      }
    }
  }
  if (batch_size == 8) {
    CHECK(system_errors[1] == 0u);
    CHECK(system_errors[2] == 0u);
    CHECK(gfn2_post_scc_potential_error_stage(system_errors[3]) ==
          Gfn2PostSccPotentialStage::kActivity);
    CHECK(gfn2_post_scc_potential_error_stage(system_errors[4]) == Gfn2PostSccPotentialStage::kES2);

    Gfn2PostSccPotentialDeviceResults aliased = results;
    aliased.complete.shell = intermediates.complete.shell;
    CHECK(refresh_gfn2_post_scc_potentials_cuda(plan, input, aliased, intermediates, workspace,
                                                diagnostics, stream) == cudaErrorInvalidValue);
    aliased = results;
    aliased.complete.atomic = const_cast<double*>(input.raw_atomic_charges);
    CHECK(refresh_gfn2_post_scc_potentials_cuda(plan, input, aliased, intermediates, workspace,
                                                diagnostics, stream) == cudaErrorInvalidValue);
    aliased = results;
    aliased.complete.shell = plan.external_point_charge_cache.shell_potentials;
    CHECK(refresh_gfn2_post_scc_potentials_cuda(plan, input, aliased, intermediates, workspace,
                                                diagnostics, stream) == cudaErrorInvalidValue);

    /* D4 is disabled in this fixture. Its plan, intermediate, and workspace
     * descriptors must remain completely uninspected, even when they contain
     * values that would be invalid or alias public output if D4 were active. */
    Gfn2PostSccPotentialDevicePlan ignored_d4_plan = plan;
    ignored_d4_plan.d4_batch.plan_token = 0u;
    ignored_d4_plan.d4_cache.geometry_generation = 0u;
    Gfn2PostSccPotentialDeviceIntermediates ignored_d4_intermediates = intermediates;
    ignored_d4_intermediates.d4_atomic = results.complete.atomic;
    ignored_d4_intermediates.d4_atomic_elements = results.complete.atom_elements;
    Gfn2PostSccPotentialDeviceWorkspace ignored_d4_workspace = workspace;
    ignored_d4_workspace.d4.weights = results.complete.shell;
    ignored_d4_workspace.d4.weight_charge_derivatives = results.complete.shell;
    ignored_d4_workspace.d4.weight_elements = results.complete.shell_elements;
    ignored_d4_workspace.d4.atom_scratch = results.complete.atomic;
    ignored_d4_workspace.d4.atom_elements = results.complete.atom_elements;
    ignored_d4_workspace.d4.system_errors = nullptr;
    ignored_d4_workspace.d4.system_error_elements = -1;
    CHECK(refresh_gfn2_post_scc_potentials_cuda(ignored_d4_plan, input, results,
                                                ignored_d4_intermediates, ignored_d4_workspace,
                                                diagnostics, stream) == cudaSuccess);

    Gfn2PostSccPotentialDevicePlan stale = plan;
    stale.es2_cache.geometry_generation = kGeometryGeneration + 1u;
    CHECK(refresh_gfn2_post_scc_potentials_cuda(stale, input, results, intermediates, workspace,
                                                diagnostics, stream) == cudaErrorInvalidValue);

    /* A nonzero field-offset base is outside the current SCC composer schema.
     * It must fail asynchronously before any member publishes final bytes. */
    std::vector<std::int64_t> invalid_qsh_offsets = fixture.shell_offsets;
    invalid_qsh_offsets[0] = 1;
    CUDA_CHECK(fixture.d_qsh_offsets.overwrite(invalid_qsh_offsets, stream));
    CUDA_CHECK(fixture.d_result_shell.overwrite(
        std::vector<double>(static_cast<std::size_t>(fixture.shells), kSentinel), stream));
    CUDA_CHECK(fixture.d_result_atomic.overwrite(
        std::vector<double>(static_cast<std::size_t>(fixture.atoms), kSentinel), stream));
    CUDA_CHECK(fixture.d_result_dipole.overwrite(
        std::vector<double>(static_cast<std::size_t>(3 * fixture.atoms), kSentinel), stream));
    CUDA_CHECK(fixture.d_result_quadrupole.overwrite(
        std::vector<double>(static_cast<std::size_t>(6 * fixture.atoms), kSentinel), stream));
    CUDA_CHECK(fixture.d_result_shell_scalar.overwrite(
        std::vector<double>(static_cast<std::size_t>(fixture.shells), kSentinel), stream));
    CUDA_CHECK(refresh_gfn2_post_scc_potentials_cuda(plan, input, results, intermediates, workspace,
                                                     diagnostics, stream));
    CUDA_CHECK(fixture.d_result_shell.download(shell, stream));
    CUDA_CHECK(fixture.d_result_atomic.download(atomic, stream));
    CUDA_CHECK(fixture.d_device_error.download(device_error, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(gfn2_post_scc_potential_error_stage(device_error[0]) ==
          Gfn2PostSccPotentialStage::kComposition);
    CHECK(std::all_of(shell.begin(), shell.end(), [](double value) { return value == kSentinel; }));
    CHECK(
        std::all_of(atomic.begin(), atomic.end(), [](double value) { return value == kSentinel; }));
    CUDA_CHECK(fixture.d_qsh_offsets.overwrite(fixture.shell_offsets, stream));

    fixture.requested.assign(8u, 1u);
    fixture.statuses.assign(8u, GPUXTB_STATUS_SUCCESS);
    for (std::int64_t index = 0; index < fixture.shells; ++index) {
      fixture.raw_shell[static_cast<std::size_t>(index)] = 0.025 * static_cast<double>(index + 1);
    }
    CUDA_CHECK(fixture.d_requested.overwrite(fixture.requested, stream));
    CUDA_CHECK(fixture.d_statuses.overwrite(fixture.statuses, stream));
    CUDA_CHECK(fixture.d_raw_shell.overwrite(fixture.raw_shell, stream));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CUDA_CHECK(refresh_gfn2_post_scc_potentials_cuda(plan, input, results, intermediates, workspace,
                                                     diagnostics, stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    fixture.raw_shell[0] = -0.47;
    fixture.raw_shell[1] = 0.31;
    fixture.raw_atomic[0] = 0.52;
    CUDA_CHECK(fixture.d_raw_shell.overwrite(fixture.raw_shell, stream));
    CUDA_CHECK(fixture.d_raw_atomic.overwrite(fixture.raw_atomic, stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(fixture.d_result_shell.download(shell, stream));
    CUDA_CHECK(fixture.d_result_atomic.download(atomic, stream));
    CUDA_CHECK(fixture.d_result_shell_scalar.download(shell_scalar, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(close(shell[0], fixture.expected_shell(0, 0, fixture.raw_shell)));
    CHECK(close(shell[1], fixture.expected_shell(0, 1, fixture.raw_shell)));
    CHECK(close(shell_scalar[0], shell[0] + atomic[0]));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_device_epoch_graph_replay() {
  constexpr std::int64_t batch_size = 8;
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Fixture fixture;
  CUDA_CHECK(fixture.initialize(batch_size, stream));
  auto plan = fixture.plan();
  auto input = fixture.input();
  auto results = fixture.results();
  auto intermediates = fixture.intermediates();
  auto workspace = fixture.workspace();
  auto diagnostics = fixture.diagnostics();
  const auto consumer = fixture.geometry_consumer();

  const auto reset_public = [&]() {
    cudaError_t status = fixture.d_result_shell.overwrite(
        std::vector<double>(static_cast<std::size_t>(fixture.shells), kSentinel), stream);
    if (status == cudaSuccess) {
      status = fixture.d_result_atomic.overwrite(
          std::vector<double>(static_cast<std::size_t>(fixture.atoms), kSentinel), stream);
    }
    if (status == cudaSuccess) {
      status = fixture.d_result_dipole.overwrite(
          std::vector<double>(static_cast<std::size_t>(3 * fixture.atoms), kSentinel), stream);
    }
    if (status == cudaSuccess) {
      status = fixture.d_result_quadrupole.overwrite(
          std::vector<double>(static_cast<std::size_t>(6 * fixture.atoms), kSentinel), stream);
    }
    if (status == cudaSuccess) {
      status = fixture.d_result_shell_scalar.overwrite(
          std::vector<double>(static_cast<std::size_t>(fixture.shells), kSentinel), stream);
    }
    return status;
  };
  const auto download_public = [&](std::vector<double>& shell, std::vector<double>& atomic,
                                   std::vector<double>& scalar,
                                   std::vector<std::uint32_t>& system_errors,
                                   std::vector<std::uint32_t>& device_error) {
    cudaError_t status = fixture.d_result_shell.download(shell, stream);
    if (status == cudaSuccess) status = fixture.d_result_atomic.download(atomic, stream);
    if (status == cudaSuccess) status = fixture.d_result_shell_scalar.download(scalar, stream);
    if (status == cudaSuccess) status = fixture.d_system_errors.download(system_errors, stream);
    if (status == cudaSuccess) status = fixture.d_device_error.download(device_error, stream);
    return status;
  };

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(refresh_gfn2_post_scc_potentials_cuda(plan, input, results, intermediates, workspace,
                                                   diagnostics, consumer, stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u));

  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  std::vector<double> shell;
  std::vector<double> atomic;
  std::vector<double> scalar;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CUDA_CHECK(download_public(shell, atomic, scalar, system_errors, device_error));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(device_error[0] == 0u);
  CHECK(std::all_of(system_errors.begin(), system_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  for (std::int64_t system = 0; system < batch_size; ++system) {
    for (std::int64_t local = 0; local < 2; ++local) {
      const std::size_t index = static_cast<std::size_t>(2 * system + local);
      CHECK(close(shell[index], fixture.expected_shell(system, local, fixture.raw_shell)));
      CHECK(close(scalar[index], shell[index] + atomic[index]));
    }
  }

  const std::uint64_t next_epoch = kGeometryGeneration + 1u;
  std::vector<std::uint64_t> committed(static_cast<std::size_t>(batch_size), next_epoch);
  std::vector<std::uint8_t> eligible(static_cast<std::size_t>(batch_size), 1u);
  committed[1] = kGeometryGeneration;
  eligible[2] = 0u;
  CUDA_CHECK(fixture.d_geometry_epoch.overwrite({next_epoch}, stream));
  CUDA_CHECK(fixture.d_committed_generations.overwrite(committed, stream));
  CUDA_CHECK(fixture.d_eligible.overwrite(eligible, stream));
  CUDA_CHECK(reset_public());
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(download_public(shell, atomic, scalar, system_errors, device_error));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(device_error[0] == 0u);
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const bool suppressed = system == 1 || system == 2;
    if (suppressed) {
      CHECK(system_errors[static_cast<std::size_t>(system)] != 0u);
      CHECK(gfn2_post_scc_potential_error_stage(
                system_errors[static_cast<std::size_t>(system)]) ==
            Gfn2PostSccPotentialStage::kActivity);
    }
    for (std::int64_t local = 0; local < 2; ++local) {
      const std::size_t index = static_cast<std::size_t>(2 * system + local);
      if (suppressed) {
        CHECK(shell[index] == kSentinel);
        CHECK(atomic[index] == kSentinel);
        CHECK(scalar[index] == kSentinel);
      } else {
        CHECK(close(shell[index], fixture.expected_shell(system, local, fixture.raw_shell)));
      }
    }
  }

  committed.assign(static_cast<std::size_t>(batch_size), next_epoch);
  eligible.assign(static_cast<std::size_t>(batch_size), 1u);
  fixture.raw_shell[0] = -0.41;
  fixture.raw_shell[1] = 0.29;
  CUDA_CHECK(fixture.d_committed_generations.overwrite(committed, stream));
  CUDA_CHECK(fixture.d_eligible.overwrite(eligible, stream));
  CUDA_CHECK(fixture.d_raw_shell.overwrite(fixture.raw_shell, stream));
  CUDA_CHECK(reset_public());
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(download_public(shell, atomic, scalar, system_errors, device_error));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(device_error[0] == 0u);
  CHECK(std::all_of(system_errors.begin(), system_errors.end(),
                    [](std::uint32_t value) { return value == 0u; }));
  CHECK(close(shell[0], fixture.expected_shell(0, 0, fixture.raw_shell)));
  CHECK(close(shell[1], fixture.expected_shell(0, 1, fixture.raw_shell)));

  eligible[3] = 2u;
  CUDA_CHECK(fixture.d_eligible.overwrite(eligible, stream));
  CUDA_CHECK(reset_public());
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(download_public(shell, atomic, scalar, system_errors, device_error));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(gfn2_post_scc_potential_error_stage(device_error[0]) ==
        Gfn2PostSccPotentialStage::kActivity);
  CHECK(std::all_of(shell.begin(), shell.end(), [](double value) { return value == kSentinel; }));

  Gfn2GeometryEpochConsumerDevice cross_plan = consumer;
  cross_plan.plan_token ^= 1u;
  CHECK(refresh_gfn2_post_scc_potentials_cuda(plan, input, results, intermediates, workspace,
                                              diagnostics, cross_plan, stream) ==
        cudaErrorInvalidValue);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    if (const int line = run_batch(batch_size); line != 0) {
      return line;
    }
  }
  return test_device_epoch_graph_replay();
}
