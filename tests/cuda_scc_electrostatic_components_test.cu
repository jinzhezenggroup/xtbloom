#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_aes2.cuh"
#include "backends/cuda/gfn2_es2.cuh"
#include "backends/cuda/gfn2_es3.cuh"
#include "backends/cuda/gfn2_external_point_charges.cuh"

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::evaluate_gfn2_aes2_scc_energy_cuda;
using xtbloom::detail::cuda::evaluate_gfn2_aes2_scc_potential_cuda;
using xtbloom::detail::cuda::evaluate_gfn2_es2_scc_energy_cuda;
using xtbloom::detail::cuda::evaluate_gfn2_es2_scc_potential_cuda;
using xtbloom::detail::cuda::evaluate_gfn2_es3_scc_energy_cuda;
using xtbloom::detail::cuda::evaluate_gfn2_es3_scc_potential_cuda;
using xtbloom::detail::cuda::evaluate_gfn2_external_point_charge_scc_energy_cuda;
using xtbloom::detail::cuda::Gfn2AES2DeviceBatch;
using xtbloom::detail::cuda::Gfn2AES2DeviceCache;
using xtbloom::detail::cuda::Gfn2AES2DeviceWorkspace;
using xtbloom::detail::cuda::Gfn2ES2DeviceBatch;
using xtbloom::detail::cuda::Gfn2ES2DeviceCache;
using xtbloom::detail::cuda::Gfn2ES2DeviceError;
using xtbloom::detail::cuda::Gfn2ES2DeviceWorkspace;
using xtbloom::detail::cuda::Gfn2ES3DeviceBatch;
using xtbloom::detail::cuda::Gfn2ES3DeviceError;
using xtbloom::detail::cuda::Gfn2ExternalPointChargeDeviceBatch;
using xtbloom::detail::cuda::Gfn2ExternalPointChargeDeviceCache;
using xtbloom::detail::cuda::Gfn2ExternalPointChargeDeviceWorkspace;
using xtbloom::detail::cuda::Gfn2SccIterationDeviceActivity;
using xtbloom::detail::cuda::reset_gfn2_aes2_device_errors_cuda;
using xtbloom::detail::cuda::reset_gfn2_es2_scc_errors_cuda;
using xtbloom::detail::cuda::reset_gfn2_es3_scc_errors_cuda;
using xtbloom::detail::cuda::reset_gfn2_external_point_charge_scc_errors_cuda;
using xtbloom::detail::cuda::update_gfn2_external_point_charge_scc_potential_cache_cuda;

constexpr std::uint64_t kPlanToken = 0x92e520260803ULL;
constexpr std::uint64_t kGeometryGeneration = 17u;
constexpr double kSentinel = -912345.0;

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
    if (data_ != nullptr) {
      const cudaError_t status = cudaFree(data_);
      if (status != cudaSuccess) {
        return status;
      }
      data_ = nullptr;
    }
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

bool close(double actual, double expected) {
  return std::abs(actual - expected) <= 2.0e-14 * std::max(1.0, std::abs(expected));
}

struct Es2Fixture {
  std::int64_t batch_size = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> shell_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<double> shell_hardness;
  std::vector<double> gamma;

  DeviceBuffer<std::int64_t> d_atom_offsets;
  DeviceBuffer<std::int64_t> d_shell_offsets;
  DeviceBuffer<std::int64_t> d_atom_shell_offsets;
  DeviceBuffer<std::int64_t> d_matrix_offsets;
  DeviceBuffer<std::int64_t> d_shell_to_atom;
  DeviceBuffer<double> d_shell_hardness;
  DeviceBuffer<double> d_gamma;
  DeviceBuffer<double> d_shell_scratch;
  DeviceBuffer<double> d_batch_scratch;
  DeviceBuffer<std::uint8_t> d_active;
  DeviceBuffer<std::uint32_t> d_sequence;
  DeviceBuffer<std::uint32_t> d_system_errors;
  DeviceBuffer<std::uint32_t> d_plan_error;

  cudaError_t initialize(std::int64_t count, cudaStream_t stream) {
    batch_size = count;
    atom_offsets.resize(static_cast<std::size_t>(count + 1));
    shell_offsets.resize(static_cast<std::size_t>(count + 1));
    matrix_offsets.resize(static_cast<std::size_t>(count + 1));
    atom_shell_offsets.resize(static_cast<std::size_t>(2 * count + 1));
    shell_to_atom.resize(static_cast<std::size_t>(2 * count));
    shell_hardness.resize(static_cast<std::size_t>(2 * count));
    gamma.resize(static_cast<std::size_t>(4 * count));
    for (std::int64_t system = 0; system < count; ++system) {
      atom_offsets[static_cast<std::size_t>(system)] = 2 * system;
      shell_offsets[static_cast<std::size_t>(system)] = 2 * system;
      matrix_offsets[static_cast<std::size_t>(system)] = 4 * system;
      for (std::int64_t local = 0; local < 2; ++local) {
        const std::int64_t shell = 2 * system + local;
        atom_shell_offsets[static_cast<std::size_t>(shell)] = shell;
        shell_to_atom[static_cast<std::size_t>(shell)] = shell;
        shell_hardness[static_cast<std::size_t>(shell)] = local == 0 ? 1.2 : 1.8;
      }
      const double average = 1.5;
      const double off_diagonal = 1.0 / std::hypot(1.25, 1.0 / average);
      gamma[static_cast<std::size_t>(4 * system)] = 1.2;
      gamma[static_cast<std::size_t>(4 * system + 1)] = off_diagonal;
      gamma[static_cast<std::size_t>(4 * system + 2)] = off_diagonal;
      gamma[static_cast<std::size_t>(4 * system + 3)] = 1.8;
    }
    atom_offsets.back() = 2 * count;
    shell_offsets.back() = 2 * count;
    matrix_offsets.back() = 4 * count;
    atom_shell_offsets.back() = 2 * count;

    cudaError_t status = d_atom_offsets.upload(atom_offsets, stream);
#define UPLOAD(name)                        \
  if (status == cudaSuccess) {              \
    status = d_##name.upload(name, stream); \
  }
    UPLOAD(shell_offsets)
    UPLOAD(atom_shell_offsets)
    UPLOAD(matrix_offsets)
    UPLOAD(shell_to_atom)
    UPLOAD(shell_hardness)
    UPLOAD(gamma)
#undef UPLOAD
    if (status == cudaSuccess) {
      status =
          d_shell_scratch.upload(std::vector<double>(static_cast<std::size_t>(2 * count)), stream);
    }
    if (status == cudaSuccess) {
      status = d_batch_scratch.upload(std::vector<double>(static_cast<std::size_t>(count)), stream);
    }
    if (status == cudaSuccess) {
      status =
          d_active.upload(std::vector<std::uint8_t>(static_cast<std::size_t>(count), 1u), stream);
    }
    if (status == cudaSuccess) {
      status = d_sequence.upload({1u}, stream);
    }
    if (status == cudaSuccess) {
      status = d_system_errors.upload(std::vector<std::uint32_t>(static_cast<std::size_t>(count)),
                                      stream);
    }
    if (status == cudaSuccess) {
      status = d_plan_error.upload({0u}, stream);
    }
    return status;
  }

  Gfn2ES2DeviceBatch batch() const {
    Gfn2ES2DeviceBatch value{};
    value.batch_size = batch_size;
    value.total_atoms = 2 * batch_size;
    value.total_shells = 2 * batch_size;
    value.total_matrix_elements = 4 * batch_size;
    value.plan_token = kPlanToken;
    value.atom_offset_count = batch_size + 1;
    value.batch_shell_offset_count = batch_size + 1;
    value.atom_shell_offset_count = 2 * batch_size + 1;
    value.matrix_offset_count = batch_size + 1;
    value.shell_to_atom_count = 2 * batch_size;
    value.shell_hardness_count = 2 * batch_size;
    value.atom_offsets = d_atom_offsets.get();
    value.batch_shell_offsets = d_shell_offsets.get();
    value.atom_shell_offsets = d_atom_shell_offsets.get();
    value.matrix_offsets = d_matrix_offsets.get();
    value.shell_to_atom = d_shell_to_atom.get();
    value.shell_hardness = d_shell_hardness.get();
    return value;
  }

  Gfn2ES2DeviceCache cache(std::uint64_t generation = kGeometryGeneration) {
    return {d_gamma.get(), 4 * batch_size, generation, kPlanToken};
  }

  Gfn2ES2DeviceWorkspace workspace() {
    Gfn2ES2DeviceWorkspace value{};
    value.shell_scratch = d_shell_scratch.get();
    value.shell_elements = 2 * batch_size;
    value.batch_scratch = d_batch_scratch.get();
    value.batch_elements = batch_size;
    return value;
  }

  Gfn2SccIterationDeviceActivity activity() const {
    return {d_active.get(), d_sequence.get(), batch_size, 1, kPlanToken};
  }
};

void expected_system(const Es2Fixture& fixture, std::int64_t system, const std::vector<double>& q,
                     double* first_potential, double* second_potential, double* energy) {
  const std::size_t shell = static_cast<std::size_t>(2 * system);
  const std::size_t matrix = static_cast<std::size_t>(4 * system);
  *first_potential = fixture.gamma[matrix] * q[shell] + fixture.gamma[matrix + 1] * q[shell + 1];
  *second_potential =
      fixture.gamma[matrix + 2] * q[shell] + fixture.gamma[matrix + 3] * q[shell + 1];
  *energy = 0.5 * (q[shell] * *first_potential + q[shell + 1] * *second_potential);
}

int test_batches_and_graph() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Es2Fixture fixture;
    CUDA_CHECK(fixture.initialize(batch_size, stream));
    std::vector<double> mixed(static_cast<std::size_t>(2 * batch_size));
    std::vector<double> raw(static_cast<std::size_t>(2 * batch_size));
    for (std::int64_t shell = 0; shell < 2 * batch_size; ++shell) {
      mixed[static_cast<std::size_t>(shell)] = 0.03 * static_cast<double>(shell + 1);
      raw[static_cast<std::size_t>(shell)] = -0.02 * static_cast<double>(shell + 2);
    }
    DeviceBuffer<double> d_mixed;
    DeviceBuffer<double> d_raw;
    DeviceBuffer<double> d_potential;
    DeviceBuffer<double> d_energy;
    CUDA_CHECK(d_mixed.upload(mixed, stream));
    CUDA_CHECK(d_raw.upload(raw, stream));
    CUDA_CHECK(d_potential.upload(
        std::vector<double>(static_cast<std::size_t>(2 * batch_size), kSentinel), stream));
    CUDA_CHECK(d_energy.upload(std::vector<double>(static_cast<std::size_t>(batch_size),
                                                   std::numeric_limits<double>::quiet_NaN()),
                               stream));
    CUDA_CHECK(reset_gfn2_es2_scc_errors_cuda(batch_size, fixture.d_system_errors.get(),
                                              fixture.d_plan_error.get(), stream));
    CUDA_CHECK(evaluate_gfn2_es2_scc_potential_cuda(
        fixture.batch(), fixture.cache(), kGeometryGeneration, fixture.activity(), d_mixed.get(),
        d_potential.get(), fixture.workspace(), fixture.d_system_errors.get(),
        fixture.d_plan_error.get(), stream));
    CUDA_CHECK(evaluate_gfn2_es2_scc_energy_cuda(
        fixture.batch(), fixture.cache(), kGeometryGeneration, fixture.activity(), d_raw.get(),
        d_energy.get(), fixture.workspace(), fixture.d_system_errors.get(),
        fixture.d_plan_error.get(), stream));
    std::vector<double> potential;
    std::vector<double> energy;
    std::vector<std::uint32_t> errors;
    std::vector<std::uint32_t> plan_error;
    CUDA_CHECK(d_potential.download(potential, stream));
    CUDA_CHECK(d_energy.download(energy, stream));
    CUDA_CHECK(fixture.d_system_errors.download(errors, stream));
    CUDA_CHECK(fixture.d_plan_error.download(plan_error, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(plan_error[0] == 0u);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      double first = 0.0;
      double second = 0.0;
      double expected_energy = 0.0;
      expected_system(fixture, system, mixed, &first, &second, &expected_energy);
      CHECK(close(potential[static_cast<std::size_t>(2 * system)], first));
      CHECK(close(potential[static_cast<std::size_t>(2 * system + 1)], second));
      expected_system(fixture, system, raw, &first, &second, &expected_energy);
      CHECK(close(energy[static_cast<std::size_t>(system)], expected_energy));
      CHECK(errors[static_cast<std::size_t>(system)] == 0u);
    }

    if (batch_size == 8) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t executable = nullptr;
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      CUDA_CHECK(reset_gfn2_es2_scc_errors_cuda(batch_size, fixture.d_system_errors.get(),
                                                fixture.d_plan_error.get(), stream));
      CUDA_CHECK(evaluate_gfn2_es2_scc_potential_cuda(
          fixture.batch(), fixture.cache(), kGeometryGeneration, fixture.activity(), d_mixed.get(),
          d_potential.get(), fixture.workspace(), fixture.d_system_errors.get(),
          fixture.d_plan_error.get(), stream));
      CUDA_CHECK(evaluate_gfn2_es2_scc_energy_cuda(
          fixture.batch(), fixture.cache(), kGeometryGeneration, fixture.activity(), d_raw.get(),
          d_energy.get(), fixture.workspace(), fixture.d_system_errors.get(),
          fixture.d_plan_error.get(), stream));
      CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
      mixed[0] = 0.71;
      mixed[1] = -0.19;
      raw[0] = -0.43;
      raw[1] = 0.27;
      CUDA_CHECK(cudaMemcpyAsync(d_mixed.get(), mixed.data(), mixed.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_raw.get(), raw.data(), raw.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaGraphLaunch(executable, stream));
      CUDA_CHECK(d_potential.download(potential, stream));
      CUDA_CHECK(d_energy.download(energy, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      double first = 0.0;
      double second = 0.0;
      double expected_energy = 0.0;
      expected_system(fixture, 0, mixed, &first, &second, &expected_energy);
      CHECK(close(potential[0], first) && close(potential[1], second));
      expected_system(fixture, 0, raw, &first, &second, &expected_energy);
      CHECK(close(energy[0], expected_energy));
      CUDA_CHECK(cudaGraphExecDestroy(executable));
      CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}

int test_es2_tree_overflow_is_not_published() {
  constexpr std::int64_t kBatchSize = 2;
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  Es2Fixture fixture;
  CUDA_CHECK(fixture.initialize(kBatchSize, stream));

  const double lane_kernel = 0.1875 * std::numeric_limits<double>::max();
  std::fill_n(fixture.gamma.begin(), 4, lane_kernel);
  const std::vector<double> raw{2.0, 2.0, 0.2, -0.15};
  DeviceBuffer<double> d_raw;
  DeviceBuffer<double> d_energy;
  CUDA_CHECK(fixture.d_gamma.upload(fixture.gamma, stream));
  CUDA_CHECK(d_raw.upload(raw, stream));
  CUDA_CHECK(d_energy.upload(std::vector<double>(2u, kSentinel), stream));
  CUDA_CHECK(reset_gfn2_es2_scc_errors_cuda(kBatchSize, fixture.d_system_errors.get(),
                                            fixture.d_plan_error.get(), stream));
  CUDA_CHECK(evaluate_gfn2_es2_scc_energy_cuda(
      fixture.batch(), fixture.cache(), kGeometryGeneration, fixture.activity(), d_raw.get(),
      d_energy.get(), fixture.workspace(), fixture.d_system_errors.get(),
      fixture.d_plan_error.get(), stream));

  std::vector<double> energy;
  std::vector<std::uint32_t> errors;
  std::vector<std::uint32_t> plan_error;
  CUDA_CHECK(d_energy.download(energy, stream));
  CUDA_CHECK(fixture.d_system_errors.download(errors, stream));
  CUDA_CHECK(fixture.d_plan_error.download(plan_error, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  /* Both row energies are finite 0.75 * DBL_MAX values. Their tree sum alone
   * overflows, so the failed system must retain its caller-visible sentinel. */
  CHECK(energy[0] == kSentinel);
  CHECK(errors[0] == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kNonfiniteEnergyArithmetic));
  CHECK(plan_error[0] == 0u);
  double first = 0.0;
  double second = 0.0;
  double expected_energy = 0.0;
  expected_system(fixture, 1, raw, &first, &second, &expected_energy);
  CHECK(close(energy[1], expected_energy) && errors[1] == 0u);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_activity_peer_isolation_and_provenance() {
  constexpr std::int64_t kBatchSize = 8;
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  Es2Fixture fixture;
  CUDA_CHECK(fixture.initialize(kBatchSize, stream));
  std::vector<std::uint8_t> active(static_cast<std::size_t>(kBatchSize), 1u);
  active[1] = 0u;
  std::vector<double> mixed(static_cast<std::size_t>(2 * kBatchSize), 0.2);
  std::vector<double> raw(static_cast<std::size_t>(2 * kBatchSize), -0.15);
  mixed[2] = std::numeric_limits<double>::quiet_NaN();
  mixed[6] = std::numeric_limits<double>::infinity();
  raw[2] = std::numeric_limits<double>::quiet_NaN();
  raw[8] = std::numeric_limits<double>::infinity();
  DeviceBuffer<double> d_mixed;
  DeviceBuffer<double> d_raw;
  DeviceBuffer<double> d_potential;
  DeviceBuffer<double> d_energy;
  CUDA_CHECK(fixture.d_active.upload(active, stream));
  CUDA_CHECK(d_mixed.upload(mixed, stream));
  CUDA_CHECK(d_raw.upload(raw, stream));
  CUDA_CHECK(d_potential.upload(std::vector<double>(16u, kSentinel), stream));
  CUDA_CHECK(d_energy.upload(std::vector<double>(8u, kSentinel), stream));
  CUDA_CHECK(reset_gfn2_es2_scc_errors_cuda(kBatchSize, fixture.d_system_errors.get(),
                                            fixture.d_plan_error.get(), stream));
  CUDA_CHECK(evaluate_gfn2_es2_scc_potential_cuda(
      fixture.batch(), fixture.cache(), kGeometryGeneration, fixture.activity(), d_mixed.get(),
      d_potential.get(), fixture.workspace(), fixture.d_system_errors.get(),
      fixture.d_plan_error.get(), stream));
  CUDA_CHECK(evaluate_gfn2_es2_scc_energy_cuda(
      fixture.batch(), fixture.cache(), kGeometryGeneration, fixture.activity(), d_raw.get(),
      d_energy.get(), fixture.workspace(), fixture.d_system_errors.get(),
      fixture.d_plan_error.get(), stream));
  std::vector<double> potential;
  std::vector<double> energy;
  std::vector<std::uint32_t> errors;
  std::vector<std::uint32_t> plan_error;
  CUDA_CHECK(d_potential.download(potential, stream));
  CUDA_CHECK(d_energy.download(energy, stream));
  CUDA_CHECK(fixture.d_system_errors.download(errors, stream));
  CUDA_CHECK(fixture.d_plan_error.download(plan_error, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(plan_error[0] == 0u);
  CHECK(potential[2] == kSentinel && potential[3] == kSentinel);
  CHECK(energy[1] == kSentinel);
  CHECK(errors[1] == 0u);
  CHECK(errors[3] == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kNonfiniteShellCharge));
  CHECK(errors[4] == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kNonfiniteShellCharge));
  CHECK(potential[6] == kSentinel && potential[7] == kSentinel && energy[4] == kSentinel);
  CHECK(potential[0] != kSentinel && energy[0] != kSentinel);

  /* An active peer must not authorize reads of an inactive member's topology. */
  active.assign(8u, 0u);
  active[0] = 1u;
  const std::int64_t poisoned_offset = -1;
  CUDA_CHECK(fixture.d_active.upload(active, stream));
  CUDA_CHECK(cudaMemcpyAsync(fixture.d_shell_offsets.get() + 4, &poisoned_offset,
                             sizeof(poisoned_offset), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(fixture.d_atom_shell_offsets.get() + 8, &poisoned_offset,
                             sizeof(poisoned_offset), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(reset_gfn2_es2_scc_errors_cuda(kBatchSize, fixture.d_system_errors.get(),
                                            fixture.d_plan_error.get(), stream));
  CUDA_CHECK(evaluate_gfn2_es2_scc_potential_cuda(
      fixture.batch(), fixture.cache(), kGeometryGeneration, fixture.activity(), d_mixed.get(),
      d_potential.get(), fixture.workspace(), fixture.d_system_errors.get(),
      fixture.d_plan_error.get(), stream));
  CUDA_CHECK(evaluate_gfn2_es2_scc_energy_cuda(
      fixture.batch(), fixture.cache(), kGeometryGeneration, fixture.activity(), d_raw.get(),
      d_energy.get(), fixture.workspace(), fixture.d_system_errors.get(),
      fixture.d_plan_error.get(), stream));
  CUDA_CHECK(fixture.d_plan_error.download(plan_error, stream));
  CUDA_CHECK(d_energy.download(energy, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(plan_error[0] == 0u && energy[0] != kSentinel);
  CUDA_CHECK(cudaMemcpyAsync(fixture.d_shell_offsets.get(), fixture.shell_offsets.data(),
                             fixture.shell_offsets.size() * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(fixture.d_atom_shell_offsets.get(), fixture.atom_shell_offsets.data(),
                             fixture.atom_shell_offsets.size() * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice, stream));

  /* A closed sequence and an all-inactive stage do not inspect malformed plan/cache data. */
  const std::int64_t malformed_offset = -1;
  CUDA_CHECK(cudaMemcpyAsync(fixture.d_shell_offsets.get(), &malformed_offset,
                             sizeof(malformed_offset), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(fixture.d_sequence.upload({0u}, stream));
  CUDA_CHECK(reset_gfn2_es2_scc_errors_cuda(kBatchSize, fixture.d_system_errors.get(),
                                            fixture.d_plan_error.get(), stream));
  CUDA_CHECK(evaluate_gfn2_es2_scc_potential_cuda(
      fixture.batch(), fixture.cache(kGeometryGeneration + 1u), kGeometryGeneration,
      fixture.activity(), d_mixed.get(), d_potential.get(), fixture.workspace(),
      fixture.d_system_errors.get(), fixture.d_plan_error.get(), stream));
  CUDA_CHECK(fixture.d_plan_error.download(plan_error, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(plan_error[0] == 0u);
  CUDA_CHECK(cudaMemcpyAsync(fixture.d_shell_offsets.get(), fixture.shell_offsets.data(),
                             fixture.shell_offsets.size() * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(fixture.d_sequence.upload({1u}, stream));
  CUDA_CHECK(fixture.d_active.upload(std::vector<std::uint8_t>(8u, 0u), stream));
  CUDA_CHECK(evaluate_gfn2_es2_scc_energy_cuda(
      fixture.batch(), fixture.cache(kGeometryGeneration + 1u), kGeometryGeneration,
      fixture.activity(), d_raw.get(), d_energy.get(), fixture.workspace(),
      fixture.d_system_errors.get(), fixture.d_plan_error.get(), stream));
  CUDA_CHECK(fixture.d_plan_error.download(plan_error, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(plan_error[0] == 0u);

  active.assign(8u, 0u);
  active[0] = 1u;
  CUDA_CHECK(fixture.d_active.upload(active, stream));
  CUDA_CHECK(evaluate_gfn2_es2_scc_energy_cuda(
      fixture.batch(), fixture.cache(kGeometryGeneration + 1u), kGeometryGeneration,
      fixture.activity(), d_raw.get(), d_energy.get(), fixture.workspace(),
      fixture.d_system_errors.get(), fixture.d_plan_error.get(), stream));
  CUDA_CHECK(fixture.d_plan_error.download(plan_error, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(plan_error[0] == static_cast<std::uint32_t>(Gfn2ES2DeviceError::kInvalidCacheMatrix));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_es3_batches_inactive_and_graph() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<std::int64_t> offsets(static_cast<std::size_t>(batch_size + 1));
    std::vector<double> gamma(static_cast<std::size_t>(2 * batch_size));
    std::vector<double> mixed(static_cast<std::size_t>(2 * batch_size));
    std::vector<double> raw(static_cast<std::size_t>(2 * batch_size));
    for (std::int64_t system = 0; system <= batch_size; ++system) {
      offsets[static_cast<std::size_t>(system)] = 2 * system;
    }
    for (std::int64_t shell = 0; shell < 2 * batch_size; ++shell) {
      gamma[static_cast<std::size_t>(shell)] = shell % 2 == 0 ? 0.4 : -0.15;
      mixed[static_cast<std::size_t>(shell)] = 0.02 * static_cast<double>(shell + 1);
      raw[static_cast<std::size_t>(shell)] = -0.03 * static_cast<double>(shell + 2);
    }
    std::vector<std::uint8_t> active(static_cast<std::size_t>(batch_size), 1u);
    if (batch_size == 8) {
      active[1] = 0u;
      mixed[2] = std::numeric_limits<double>::quiet_NaN();
      raw[2] = std::numeric_limits<double>::infinity();
    }
    DeviceBuffer<std::int64_t> d_offsets;
    DeviceBuffer<double> d_gamma;
    DeviceBuffer<double> d_mixed;
    DeviceBuffer<double> d_raw;
    DeviceBuffer<double> d_potential;
    DeviceBuffer<double> d_energy;
    DeviceBuffer<std::uint8_t> d_active;
    DeviceBuffer<std::uint32_t> d_sequence;
    DeviceBuffer<std::uint32_t> d_errors;
    DeviceBuffer<std::uint32_t> d_plan;
    CUDA_CHECK(d_offsets.upload(offsets, stream));
    CUDA_CHECK(d_gamma.upload(gamma, stream));
    CUDA_CHECK(d_mixed.upload(mixed, stream));
    CUDA_CHECK(d_raw.upload(raw, stream));
    CUDA_CHECK(d_potential.upload(
        std::vector<double>(static_cast<std::size_t>(2 * batch_size), kSentinel), stream));
    CUDA_CHECK(d_energy.upload(std::vector<double>(static_cast<std::size_t>(batch_size),
                                                   std::numeric_limits<double>::quiet_NaN()),
                               stream));
    CUDA_CHECK(d_active.upload(active, stream));
    CUDA_CHECK(d_sequence.upload({1u}, stream));
    CUDA_CHECK(
        d_errors.upload(std::vector<std::uint32_t>(static_cast<std::size_t>(batch_size)), stream));
    CUDA_CHECK(d_plan.upload({0u}, stream));
    Gfn2ES3DeviceBatch batch{};
    batch.batch_size = batch_size;
    batch.total_shells = 2 * batch_size;
    batch.batch_shell_offset_count = batch_size + 1;
    batch.shell_gamma3_count = 2 * batch_size;
    batch.batch_shell_offsets = d_offsets.get();
    batch.shell_gamma3 = d_gamma.get();
    batch.plan_token = kPlanToken;
    const Gfn2SccIterationDeviceActivity activity_view{d_active.get(), d_sequence.get(), batch_size,
                                                       1, kPlanToken};
    CUDA_CHECK(reset_gfn2_es3_scc_errors_cuda(batch_size, d_errors.get(), d_plan.get(), stream));
    CUDA_CHECK(evaluate_gfn2_es3_scc_potential_cuda(batch, activity_view, d_mixed.get(),
                                                    d_potential.get(), d_errors.get(), d_plan.get(),
                                                    stream));
    CUDA_CHECK(evaluate_gfn2_es3_scc_energy_cuda(batch, activity_view, d_raw.get(), d_energy.get(),
                                                 d_errors.get(), d_plan.get(), stream));
    std::vector<double> potential;
    std::vector<double> energy;
    std::vector<std::uint32_t> errors;
    std::vector<std::uint32_t> plan;
    CUDA_CHECK(d_potential.download(potential, stream));
    CUDA_CHECK(d_energy.download(energy, stream));
    CUDA_CHECK(d_errors.download(errors, stream));
    CUDA_CHECK(d_plan.download(plan, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(plan[0] == 0u);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      if (batch_size == 8 && system == 1) {
        CHECK(potential[2] == kSentinel && potential[3] == kSentinel);
        CHECK(std::isnan(energy[1]));
        CHECK(errors[1] == 0u);
        continue;
      }
      double expected_energy = 0.0;
      for (std::int64_t local = 0; local < 2; ++local) {
        const std::size_t shell = static_cast<std::size_t>(2 * system + local);
        CHECK(close(potential[shell], gamma[shell] * mixed[shell] * mixed[shell]));
        expected_energy += gamma[shell] * raw[shell] * raw[shell] * raw[shell] / 3.0;
      }
      CHECK(close(energy[static_cast<std::size_t>(system)], expected_energy));
      CHECK(errors[static_cast<std::size_t>(system)] == 0u);
    }
    if (batch_size == 8) {
      /* Immutable topology is plan-wide even when only one peer is active.
       * Poisoning an inactive peer must fail the complete ES3 transaction
       * before the healthy peer can republish. */
      active.assign(8u, 0u);
      active[0] = 1u;
      const std::int64_t poisoned_offset = -1;
      CUDA_CHECK(cudaMemcpyAsync(d_active.get(), active.data(),
                                 active.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(d_offsets.get() + 4, &poisoned_offset, sizeof(poisoned_offset),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(reset_gfn2_es3_scc_errors_cuda(batch_size, d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(evaluate_gfn2_es3_scc_potential_cuda(batch, activity_view, d_mixed.get(),
                                                      d_potential.get(), d_errors.get(),
                                                      d_plan.get(), stream));
      CUDA_CHECK(evaluate_gfn2_es3_scc_energy_cuda(
          batch, activity_view, d_raw.get(), d_energy.get(), d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(d_plan.download(plan, stream));
      CUDA_CHECK(d_energy.download(energy, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(plan[0] == static_cast<std::uint32_t>(Gfn2ES3DeviceError::kInvalidOffsets));
      CHECK(close(energy[0], gamma[0] * raw[0] * raw[0] * raw[0] / 3.0 +
                                 gamma[1] * raw[1] * raw[1] * raw[1] / 3.0));
      CUDA_CHECK(cudaMemcpyAsync(d_offsets.get(), offsets.data(),
                                 offsets.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice,
                                 stream));

      /* Every typed pointer consumed by the SCC path is rejected when misaligned. */
      DeviceBuffer<std::uint8_t> d_misaligned;
      CUDA_CHECK(d_misaligned.upload(std::vector<std::uint8_t>(128u), stream));
      auto* const misaligned = d_misaligned.get() + 1;
      Gfn2ES3DeviceBatch malformed_batch = batch;
      malformed_batch.batch_shell_offsets = reinterpret_cast<const std::int64_t*>(misaligned);
      CHECK(evaluate_gfn2_es3_scc_potential_cuda(malformed_batch, activity_view, d_mixed.get(),
                                                 d_potential.get(), d_errors.get(), d_plan.get(),
                                                 stream) == cudaErrorInvalidValue);
      malformed_batch = batch;
      malformed_batch.shell_gamma3 = reinterpret_cast<const double*>(misaligned);
      CHECK(evaluate_gfn2_es3_scc_potential_cuda(malformed_batch, activity_view, d_mixed.get(),
                                                 d_potential.get(), d_errors.get(), d_plan.get(),
                                                 stream) == cudaErrorInvalidValue);
      Gfn2SccIterationDeviceActivity malformed_activity = activity_view;
      malformed_activity.sequence_active = reinterpret_cast<const std::uint32_t*>(misaligned);
      CHECK(evaluate_gfn2_es3_scc_potential_cuda(batch, malformed_activity, d_mixed.get(),
                                                 d_potential.get(), d_errors.get(), d_plan.get(),
                                                 stream) == cudaErrorInvalidValue);
      CHECK(evaluate_gfn2_es3_scc_potential_cuda(
                batch, activity_view, reinterpret_cast<const double*>(misaligned),
                d_potential.get(), d_errors.get(), d_plan.get(), stream) == cudaErrorInvalidValue);
      CHECK(evaluate_gfn2_es3_scc_potential_cuda(
                batch, activity_view, d_mixed.get(), reinterpret_cast<double*>(misaligned),
                d_errors.get(), d_plan.get(), stream) == cudaErrorInvalidValue);
      CHECK(evaluate_gfn2_es3_scc_potential_cuda(batch, activity_view, d_mixed.get(),
                                                 d_potential.get(),
                                                 reinterpret_cast<std::uint32_t*>(misaligned),
                                                 d_plan.get(), stream) == cudaErrorInvalidValue);
      CHECK(evaluate_gfn2_es3_scc_potential_cuda(
                batch, activity_view, d_mixed.get(), d_potential.get(), d_errors.get(),
                reinterpret_cast<std::uint32_t*>(misaligned), stream) == cudaErrorInvalidValue);
      CHECK(reset_gfn2_es3_scc_errors_cuda(batch_size, reinterpret_cast<std::uint32_t*>(misaligned),
                                           d_plan.get(), stream) == cudaErrorInvalidValue);

      cudaGraph_t graph = nullptr;
      cudaGraphExec_t executable = nullptr;
      active.assign(8u, 1u);
      mixed[2] = 0.11;
      raw[2] = -0.07;
      CUDA_CHECK(cudaMemcpyAsync(d_active.get(), active.data(),
                                 active.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(d_mixed.upload(mixed, stream));
      CUDA_CHECK(d_raw.upload(raw, stream));
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      CUDA_CHECK(reset_gfn2_es3_scc_errors_cuda(batch_size, d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(evaluate_gfn2_es3_scc_potential_cuda(batch, activity_view, d_mixed.get(),
                                                      d_potential.get(), d_errors.get(),
                                                      d_plan.get(), stream));
      CUDA_CHECK(evaluate_gfn2_es3_scc_energy_cuda(
          batch, activity_view, d_raw.get(), d_energy.get(), d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
      mixed[0] = 0.63;
      raw[0] = -0.41;
      CUDA_CHECK(cudaMemcpyAsync(d_mixed.get(), mixed.data(), mixed.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_raw.get(), raw.data(), raw.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaGraphLaunch(executable, stream));
      CUDA_CHECK(d_potential.download(potential, stream));
      CUDA_CHECK(d_energy.download(energy, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(close(potential[0], gamma[0] * mixed[0] * mixed[0]));
      CHECK(close(energy[0], gamma[0] * raw[0] * raw[0] * raw[0] / 3.0 +
                                 gamma[1] * raw[1] * raw[1] * raw[1] / 3.0));
      CUDA_CHECK(cudaGraphExecDestroy(executable));
      CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}

int test_aes2_batches_inactive_and_graph() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const std::int64_t atoms = 2 * batch_size;
    std::vector<std::int64_t> atom_offsets(static_cast<std::size_t>(batch_size + 1));
    std::vector<std::int64_t> pair_offsets(static_cast<std::size_t>(batch_size + 1));
    for (std::int64_t system = 0; system <= batch_size; ++system) {
      atom_offsets[static_cast<std::size_t>(system)] = 2 * system;
      pair_offsets[static_cast<std::size_t>(system)] = system;
    }
    std::vector<double> dipole_kernel(static_cast<std::size_t>(atoms), 0.25);
    std::vector<double> quadrupole_kernel(static_cast<std::size_t>(atoms), 0.08);
    std::vector<double> radius(static_cast<std::size_t>(atoms), 1.0);
    std::vector<double> valence(static_cast<std::size_t>(atoms), 1.0);
    std::vector<double> pair_data(static_cast<std::size_t>(5 * batch_size));
    for (std::int64_t pair = 0; pair < batch_size; ++pair) {
      const std::size_t base = static_cast<std::size_t>(5 * pair);
      pair_data[base] = 1.0;
      pair_data[base + 1] = 0.0;
      pair_data[base + 2] = 0.0;
      pair_data[base + 3] = 0.3;
      pair_data[base + 4] = 0.2;
    }
    std::vector<double> mixed_q(static_cast<std::size_t>(atoms));
    std::vector<double> mixed_d(static_cast<std::size_t>(3 * atoms), 0.0);
    std::vector<double> mixed_Q(static_cast<std::size_t>(6 * atoms), 0.0);
    std::vector<double> raw_q(static_cast<std::size_t>(atoms));
    std::vector<double> raw_d(static_cast<std::size_t>(3 * atoms), 0.0);
    std::vector<double> raw_Q(static_cast<std::size_t>(6 * atoms), 0.0);
    for (std::int64_t atom = 0; atom < atoms; ++atom) {
      mixed_q[static_cast<std::size_t>(atom)] = 0.04 * static_cast<double>(atom + 1);
      raw_q[static_cast<std::size_t>(atom)] = -0.03 * static_cast<double>(atom + 2);
      mixed_d[static_cast<std::size_t>(3 * atom)] = 0.02 * static_cast<double>(atom + 1);
      raw_d[static_cast<std::size_t>(3 * atom)] = -0.015 * static_cast<double>(atom + 1);
    }
    DeviceBuffer<std::int64_t> d_atom_offsets;
    DeviceBuffer<std::int64_t> d_pair_offsets;
    DeviceBuffer<double> d_dipole_kernel;
    DeviceBuffer<double> d_quadrupole_kernel;
    DeviceBuffer<double> d_radius;
    DeviceBuffer<double> d_valence;
    DeviceBuffer<double> d_pair_data;
    DeviceBuffer<double> d_mixed_q;
    DeviceBuffer<double> d_mixed_d;
    DeviceBuffer<double> d_mixed_Q;
    DeviceBuffer<double> d_raw_q;
    DeviceBuffer<double> d_raw_d;
    DeviceBuffer<double> d_raw_Q;
    DeviceBuffer<double> d_charge_potential;
    DeviceBuffer<double> d_dipole_potential;
    DeviceBuffer<double> d_quadrupole_potential;
    DeviceBuffer<double> d_energy;
    DeviceBuffer<double> d_potential_scratch;
    DeviceBuffer<double> d_batch_scratch;
    DeviceBuffer<std::uint32_t> d_peer_scratch;
    DeviceBuffer<std::uint8_t> d_active;
    DeviceBuffer<std::uint32_t> d_sequence;
    DeviceBuffer<std::uint32_t> d_errors;
    DeviceBuffer<std::uint32_t> d_plan;
#define UPLOAD_BUFFER(device, host) CUDA_CHECK(device.upload(host, stream))
    UPLOAD_BUFFER(d_atom_offsets, atom_offsets);
    UPLOAD_BUFFER(d_pair_offsets, pair_offsets);
    UPLOAD_BUFFER(d_dipole_kernel, dipole_kernel);
    UPLOAD_BUFFER(d_quadrupole_kernel, quadrupole_kernel);
    UPLOAD_BUFFER(d_radius, radius);
    UPLOAD_BUFFER(d_valence, valence);
    UPLOAD_BUFFER(d_pair_data, pair_data);
    UPLOAD_BUFFER(d_mixed_q, mixed_q);
    UPLOAD_BUFFER(d_mixed_d, mixed_d);
    UPLOAD_BUFFER(d_mixed_Q, mixed_Q);
    UPLOAD_BUFFER(d_raw_q, raw_q);
    UPLOAD_BUFFER(d_raw_d, raw_d);
    UPLOAD_BUFFER(d_raw_Q, raw_Q);
#undef UPLOAD_BUFFER
    CUDA_CHECK(d_charge_potential.upload(
        std::vector<double>(static_cast<std::size_t>(atoms), kSentinel), stream));
    CUDA_CHECK(d_dipole_potential.upload(
        std::vector<double>(static_cast<std::size_t>(3 * atoms), kSentinel), stream));
    CUDA_CHECK(d_quadrupole_potential.upload(
        std::vector<double>(static_cast<std::size_t>(6 * atoms), kSentinel), stream));
    CUDA_CHECK(d_energy.upload(std::vector<double>(static_cast<std::size_t>(batch_size),
                                                   std::numeric_limits<double>::quiet_NaN()),
                               stream));
    CUDA_CHECK(d_potential_scratch.upload(std::vector<double>(static_cast<std::size_t>(10 * atoms)),
                                          stream));
    CUDA_CHECK(
        d_batch_scratch.upload(std::vector<double>(static_cast<std::size_t>(batch_size)), stream));
    CUDA_CHECK(d_peer_scratch.upload({0u}, stream));
    CUDA_CHECK(d_active.upload(std::vector<std::uint8_t>(static_cast<std::size_t>(batch_size), 1u),
                               stream));
    CUDA_CHECK(d_sequence.upload({1u}, stream));
    CUDA_CHECK(
        d_errors.upload(std::vector<std::uint32_t>(static_cast<std::size_t>(batch_size)), stream));
    CUDA_CHECK(d_plan.upload({0u}, stream));

    Gfn2AES2DeviceBatch batch{};
    batch.batch_size = batch_size;
    batch.total_atoms = atoms;
    batch.total_pairs = batch_size;
    batch.plan_token = kPlanToken;
    batch.atom_offset_count = batch_size + 1;
    batch.pair_offset_count = batch_size + 1;
    batch.dipole_kernel_count = atoms;
    batch.quadrupole_kernel_count = atoms;
    batch.multipole_radius_count = atoms;
    batch.multipole_valence_cn_count = atoms;
    batch.atom_offsets = d_atom_offsets.get();
    batch.pair_offsets = d_pair_offsets.get();
    batch.dipole_kernel = d_dipole_kernel.get();
    batch.quadrupole_kernel = d_quadrupole_kernel.get();
    batch.multipole_radius = d_radius.get();
    batch.multipole_valence_cn = d_valence.get();
    const Gfn2AES2DeviceCache cache{d_pair_data.get(), 5 * batch_size, kGeometryGeneration,
                                    kPlanToken};
    Gfn2AES2DeviceWorkspace workspace{};
    workspace.potential_scratch = d_potential_scratch.get();
    workspace.potential_elements = 10 * atoms;
    workspace.batch_scratch = d_batch_scratch.get();
    workspace.batch_elements = batch_size;
    workspace.scc_peer_error_scratch = d_peer_scratch.get();
    workspace.scc_peer_error_elements = 1;
    const Gfn2SccIterationDeviceActivity activity{d_active.get(), d_sequence.get(), batch_size, 1,
                                                  kPlanToken};
    CUDA_CHECK(
        reset_gfn2_aes2_device_errors_cuda(batch_size, d_errors.get(), d_plan.get(), stream));
    CUDA_CHECK(evaluate_gfn2_aes2_scc_potential_cuda(
        batch, cache, kGeometryGeneration, activity, d_mixed_q.get(), d_mixed_d.get(),
        d_mixed_Q.get(), d_charge_potential.get(), d_dipole_potential.get(),
        d_quadrupole_potential.get(), workspace, d_errors.get(), d_plan.get(), stream));
    CUDA_CHECK(evaluate_gfn2_aes2_scc_energy_cuda(
        batch, cache, kGeometryGeneration, activity, d_raw_q.get(), d_raw_d.get(), d_raw_Q.get(),
        d_energy.get(), workspace, d_errors.get(), d_plan.get(), stream));
    std::vector<double> charge_potential;
    std::vector<double> dipole_potential;
    std::vector<double> energy;
    std::vector<std::uint32_t> errors;
    std::vector<std::uint32_t> plan;
    CUDA_CHECK(d_charge_potential.download(charge_potential, stream));
    CUDA_CHECK(d_dipole_potential.download(dipole_potential, stream));
    CUDA_CHECK(d_energy.download(energy, stream));
    CUDA_CHECK(d_errors.download(errors, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::int64_t system = 0; system < batch_size; ++system) {
      const std::int64_t first = 2 * system;
      const std::int64_t second = first + 1;
      const double q_first = raw_q[static_cast<std::size_t>(first)];
      const double q_second = raw_q[static_cast<std::size_t>(second)];
      const double d_first = raw_d[static_cast<std::size_t>(3 * first)];
      const double d_second = raw_d[static_cast<std::size_t>(3 * second)];
      const double onsite = 0.25 * (d_first * d_first + d_second * d_second);
      const double pair =
          0.3 * (q_first * d_second - q_second * d_first) - 0.4 * d_first * d_second;
      CHECK(close(energy[static_cast<std::size_t>(system)], onsite + pair));
      CHECK(std::isfinite(charge_potential[static_cast<std::size_t>(first)]));
      CHECK(std::isfinite(dipole_potential[static_cast<std::size_t>(3 * first)]));
      CHECK(errors[static_cast<std::size_t>(system)] == 0u);
    }
    if (batch_size == 8) {
      /* Rejected host metadata must not reach the total_atoms*10 element-count expression. */
      Gfn2AES2DeviceBatch overflowing_batch = batch;
      overflowing_batch.total_atoms = std::numeric_limits<std::int64_t>::max() /
                                          xtbloom::detail::cuda::kGfn2AES2PotentialElementsPerAtom +
                                      1;
      overflowing_batch.dipole_kernel_count = overflowing_batch.total_atoms;
      overflowing_batch.quadrupole_kernel_count = overflowing_batch.total_atoms;
      overflowing_batch.multipole_radius_count = overflowing_batch.total_atoms;
      overflowing_batch.multipole_valence_cn_count = overflowing_batch.total_atoms;
      CHECK(evaluate_gfn2_aes2_scc_potential_cuda(
                overflowing_batch, cache, kGeometryGeneration, activity, d_mixed_q.get(),
                d_mixed_d.get(), d_mixed_Q.get(), d_charge_potential.get(),
                d_dipole_potential.get(), d_quadrupole_potential.get(), workspace, d_errors.get(),
                d_plan.get(), stream) == cudaErrorInvalidValue);

      std::vector<std::uint8_t> active(8u, 0u);
      active[0] = 1u;
      const std::int64_t poisoned_offset = -1;
      CUDA_CHECK(cudaMemcpyAsync(d_active.get(), active.data(),
                                 active.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(d_atom_offsets.get() + 4, &poisoned_offset,
                                 sizeof(poisoned_offset), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_pair_offsets.get() + 4, &poisoned_offset,
                                 sizeof(poisoned_offset), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(
          reset_gfn2_aes2_device_errors_cuda(batch_size, d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(evaluate_gfn2_aes2_scc_potential_cuda(
          batch, cache, kGeometryGeneration, activity, d_mixed_q.get(), d_mixed_d.get(),
          d_mixed_Q.get(), d_charge_potential.get(), d_dipole_potential.get(),
          d_quadrupole_potential.get(), workspace, d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(evaluate_gfn2_aes2_scc_energy_cuda(
          batch, cache, kGeometryGeneration, activity, d_raw_q.get(), d_raw_d.get(), d_raw_Q.get(),
          d_energy.get(), workspace, d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(d_plan.download(plan, stream));
      CUDA_CHECK(d_energy.download(energy, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(plan[0] == 0u);
      CHECK(std::isfinite(energy[0]));
      CUDA_CHECK(cudaMemcpyAsync(d_atom_offsets.get(), atom_offsets.data(),
                                 atom_offsets.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(d_pair_offsets.get(), pair_offsets.data(),
                                 pair_offsets.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice,
                                 stream));

      active.assign(8u, 1u);
      active[1] = 0u;
      mixed_q[2] = std::numeric_limits<double>::quiet_NaN();
      raw_d[6] = std::numeric_limits<double>::infinity();
      CUDA_CHECK(cudaMemcpyAsync(d_active.get(), active.data(), active.size(),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_mixed_q.get(), mixed_q.data(), mixed_q.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_raw_d.get(), raw_d.data(), raw_d.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemsetAsync(d_errors.get(), 0, 8u * sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(d_plan.get(), 0, sizeof(std::uint32_t), stream));
      CUDA_CHECK(evaluate_gfn2_aes2_scc_potential_cuda(
          batch, cache, kGeometryGeneration, activity, d_mixed_q.get(), d_mixed_d.get(),
          d_mixed_Q.get(), d_charge_potential.get(), d_dipole_potential.get(),
          d_quadrupole_potential.get(), workspace, d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(evaluate_gfn2_aes2_scc_energy_cuda(
          batch, cache, kGeometryGeneration, activity, d_raw_q.get(), d_raw_d.get(), d_raw_Q.get(),
          d_energy.get(), workspace, d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(d_errors.download(errors, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(errors[1] == 0u);

      active[1] = 1u;
      mixed_q[2] = 0.12;
      raw_d[6] = -0.045;
      CUDA_CHECK(cudaMemcpyAsync(d_active.get(), active.data(), active.size(),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_mixed_q.get(), mixed_q.data(), mixed_q.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_raw_d.get(), raw_d.data(), raw_d.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t executable = nullptr;
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      CUDA_CHECK(
          reset_gfn2_aes2_device_errors_cuda(batch_size, d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(evaluate_gfn2_aes2_scc_potential_cuda(
          batch, cache, kGeometryGeneration, activity, d_mixed_q.get(), d_mixed_d.get(),
          d_mixed_Q.get(), d_charge_potential.get(), d_dipole_potential.get(),
          d_quadrupole_potential.get(), workspace, d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(evaluate_gfn2_aes2_scc_energy_cuda(
          batch, cache, kGeometryGeneration, activity, d_raw_q.get(), d_raw_d.get(), d_raw_Q.get(),
          d_energy.get(), workspace, d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
      const double old_energy = energy[0];
      mixed_q[0] = 0.91;
      raw_q[0] = -0.73;
      raw_d[0] = 0.44;
      CUDA_CHECK(cudaMemcpyAsync(d_mixed_q.get(), mixed_q.data(), mixed_q.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_raw_q.get(), raw_q.data(), raw_q.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_raw_d.get(), raw_d.data(), raw_d.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaGraphLaunch(executable, stream));
      CUDA_CHECK(d_charge_potential.download(charge_potential, stream));
      CUDA_CHECK(d_energy.download(energy, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(std::isfinite(charge_potential[0]) && std::isfinite(energy[0]));
      CHECK(!close(energy[0], old_energy));
      CUDA_CHECK(cudaGraphExecDestroy(executable));
      CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}

int test_point_charge_batches_inactive_and_graph() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<std::int64_t> offsets(static_cast<std::size_t>(batch_size + 1));
    std::vector<std::int64_t> shell_to_atom(static_cast<std::size_t>(batch_size));
    std::vector<double> shell_hardness(static_cast<std::size_t>(batch_size), 1.4);
    std::vector<double> qm_positions(static_cast<std::size_t>(3 * batch_size), 0.0);
    std::vector<double> point_positions(static_cast<std::size_t>(3 * batch_size), 0.0);
    std::vector<double> point_charges(static_cast<std::size_t>(batch_size), 0.5);
    std::vector<double> point_hardness(static_cast<std::size_t>(batch_size), 1.6);
    std::vector<double> raw(static_cast<std::size_t>(batch_size));
    for (std::int64_t system = 0; system <= batch_size; ++system) {
      offsets[static_cast<std::size_t>(system)] = system;
    }
    for (std::int64_t system = 0; system < batch_size; ++system) {
      shell_to_atom[static_cast<std::size_t>(system)] = system;
      qm_positions[static_cast<std::size_t>(3 * system)] = 2.0 * static_cast<double>(system);
      point_positions[static_cast<std::size_t>(3 * system)] =
          2.0 * static_cast<double>(system) + 1.0;
      raw[static_cast<std::size_t>(system)] = -0.1 * static_cast<double>(system + 1);
    }
    DeviceBuffer<std::int64_t> d_atom_offsets;
    DeviceBuffer<std::int64_t> d_shell_offsets;
    DeviceBuffer<std::int64_t> d_point_offsets;
    DeviceBuffer<std::int64_t> d_shell_to_atom;
    DeviceBuffer<double> d_shell_hardness;
    DeviceBuffer<double> d_qm_positions;
    DeviceBuffer<double> d_point_positions;
    DeviceBuffer<double> d_point_charges;
    DeviceBuffer<double> d_point_hardness;
    DeviceBuffer<double> d_raw;
    DeviceBuffer<double> d_cache;
    DeviceBuffer<double> d_scratch;
    DeviceBuffer<double> d_energy;
    DeviceBuffer<std::uint8_t> d_active;
    DeviceBuffer<std::uint32_t> d_sequence;
    DeviceBuffer<std::uint32_t> d_errors;
    DeviceBuffer<std::uint32_t> d_plan;
    CUDA_CHECK(d_atom_offsets.upload(offsets, stream));
    CUDA_CHECK(d_shell_offsets.upload(offsets, stream));
    CUDA_CHECK(d_point_offsets.upload(offsets, stream));
    CUDA_CHECK(d_shell_to_atom.upload(shell_to_atom, stream));
    CUDA_CHECK(d_shell_hardness.upload(shell_hardness, stream));
    CUDA_CHECK(d_qm_positions.upload(qm_positions, stream));
    CUDA_CHECK(d_point_positions.upload(point_positions, stream));
    CUDA_CHECK(d_point_charges.upload(point_charges, stream));
    CUDA_CHECK(d_point_hardness.upload(point_hardness, stream));
    CUDA_CHECK(d_raw.upload(raw, stream));
    CUDA_CHECK(d_cache.upload(std::vector<double>(static_cast<std::size_t>(batch_size), kSentinel),
                              stream));
    CUDA_CHECK(d_scratch.upload(std::vector<double>(static_cast<std::size_t>(batch_size)), stream));
    CUDA_CHECK(d_energy.upload(std::vector<double>(static_cast<std::size_t>(batch_size), kSentinel),
                               stream));
    CUDA_CHECK(d_active.upload(std::vector<std::uint8_t>(static_cast<std::size_t>(batch_size), 1u),
                               stream));
    CUDA_CHECK(d_sequence.upload({1u}, stream));
    CUDA_CHECK(
        d_errors.upload(std::vector<std::uint32_t>(static_cast<std::size_t>(batch_size)), stream));
    CUDA_CHECK(d_plan.upload({0u}, stream));
    Gfn2ExternalPointChargeDeviceBatch batch{};
    batch.batch_size = batch_size;
    batch.total_atoms = batch_size;
    batch.total_shells = batch_size;
    batch.total_point_charges = batch_size;
    batch.atom_offsets = d_atom_offsets.get();
    batch.batch_shell_offsets = d_shell_offsets.get();
    batch.point_charge_offsets = d_point_offsets.get();
    batch.shell_to_atom = d_shell_to_atom.get();
    batch.shell_hardness = d_shell_hardness.get();
    batch.qm_positions = d_qm_positions.get();
    batch.point_positions = d_point_positions.get();
    batch.point_charges = d_point_charges.get();
    batch.point_hardnesses = d_point_hardness.get();
    batch.plan_token = kPlanToken;
    const Gfn2ExternalPointChargeDeviceCache cache{d_cache.get(), batch_size, kGeometryGeneration,
                                                   kPlanToken};
    const Gfn2ExternalPointChargeDeviceWorkspace workspace{d_scratch.get(), batch_size};
    const Gfn2SccIterationDeviceActivity activity{d_active.get(), d_sequence.get(), batch_size, 1,
                                                  kPlanToken};
    CUDA_CHECK(reset_gfn2_external_point_charge_scc_errors_cuda(batch_size, d_errors.get(),
                                                                d_plan.get(), stream));
    CUDA_CHECK(update_gfn2_external_point_charge_scc_potential_cache_cuda(
        batch, activity, cache, kGeometryGeneration, workspace, d_errors.get(), d_plan.get(),
        stream));
    CUDA_CHECK(evaluate_gfn2_external_point_charge_scc_energy_cuda(
        batch, activity, cache, kGeometryGeneration, d_raw.get(), d_energy.get(), d_errors.get(),
        d_plan.get(), stream));
    std::vector<double> potentials;
    std::vector<double> energies;
    std::vector<std::uint32_t> errors;
    std::vector<std::uint32_t> plan;
    CUDA_CHECK(d_cache.download(potentials, stream));
    CUDA_CHECK(d_energy.download(energies, stream));
    CUDA_CHECK(d_errors.download(errors, stream));
    CUDA_CHECK(d_plan.download(plan, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double distance = std::hypot(1.0, 2.0 / 3.0);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      const double potential = point_charges[static_cast<std::size_t>(system)] / distance;
      CHECK(close(potentials[static_cast<std::size_t>(system)], potential));
      CHECK(close(energies[static_cast<std::size_t>(system)],
                  raw[static_cast<std::size_t>(system)] * potential));
      CHECK(errors[static_cast<std::size_t>(system)] == 0u);
    }
    CHECK(plan[0] == 0u);

    if (batch_size == 8) {
      auto update_status = [&](const Gfn2ExternalPointChargeDeviceBatch& selected_batch,
                               const Gfn2SccIterationDeviceActivity& selected_activity,
                               const Gfn2ExternalPointChargeDeviceCache& selected_cache,
                               std::uint32_t* selected_system_errors,
                               std::uint32_t* selected_plan_error) {
        return update_gfn2_external_point_charge_scc_potential_cache_cuda(
            selected_batch, selected_activity, selected_cache, kGeometryGeneration, workspace,
            selected_system_errors, selected_plan_error, stream);
      };
      auto energy_status = [&](const Gfn2ExternalPointChargeDeviceBatch& selected_batch,
                               const Gfn2SccIterationDeviceActivity& selected_activity,
                               const Gfn2ExternalPointChargeDeviceCache& selected_cache,
                               const double* selected_raw, double* selected_energy,
                               std::uint32_t* selected_system_errors,
                               std::uint32_t* selected_plan_error) {
        return evaluate_gfn2_external_point_charge_scc_energy_cuda(
            selected_batch, selected_activity, selected_cache, kGeometryGeneration, selected_raw,
            selected_energy, selected_system_errors, selected_plan_error, stream);
      };

      std::vector<std::uint8_t> active(8u, 0u);
      active[0] = 1u;
      const std::int64_t poisoned_offset = -1;
      CUDA_CHECK(cudaMemcpyAsync(d_active.get(), active.data(), active.size(),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_atom_offsets.get() + 4, &poisoned_offset,
                                 sizeof(poisoned_offset), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_shell_offsets.get() + 4, &poisoned_offset,
                                 sizeof(poisoned_offset), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_point_offsets.get() + 4, &poisoned_offset,
                                 sizeof(poisoned_offset), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(reset_gfn2_external_point_charge_scc_errors_cuda(batch_size, d_errors.get(),
                                                                  d_plan.get(), stream));
      CUDA_CHECK(update_status(batch, activity, cache, d_errors.get(), d_plan.get()));
      CUDA_CHECK(energy_status(batch, activity, cache, d_raw.get(), d_energy.get(), d_errors.get(),
                               d_plan.get()));
      CUDA_CHECK(d_plan.download(plan, stream));
      CUDA_CHECK(d_energy.download(energies, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(plan[0] == 0u);
      CHECK(close(energies[0], raw[0] * point_charges[0] / distance));
      CUDA_CHECK(cudaMemcpyAsync(d_atom_offsets.get(), offsets.data(),
                                 offsets.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(d_shell_offsets.get(), offsets.data(),
                                 offsets.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(d_point_offsets.get(), offsets.data(),
                                 offsets.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice,
                                 stream));

      /* Host-visible binding faults must fail synchronously without a kernel launch. */
      Gfn2ExternalPointChargeDeviceBatch zero_token_batch = batch;
      zero_token_batch.plan_token = 0u;
      Gfn2SccIterationDeviceActivity zero_token_activity = activity;
      zero_token_activity.plan_token = 0u;
      CHECK(update_status(zero_token_batch, zero_token_activity, cache, d_errors.get(),
                          d_plan.get()) == cudaErrorInvalidValue);

      auto rejects_control_aliases = [&](const void* source) {
        Gfn2SccIterationDeviceActivity selected_activity = activity;
        selected_activity.active_mask = static_cast<const std::uint8_t*>(source);
        if (update_status(batch, selected_activity, cache, d_errors.get(), d_plan.get()) !=
            cudaErrorInvalidValue) {
          return false;
        }
        selected_activity = activity;
        selected_activity.sequence_active = static_cast<const std::uint32_t*>(source);
        if (update_status(batch, selected_activity, cache, d_errors.get(), d_plan.get()) !=
            cudaErrorInvalidValue) {
          return false;
        }
        auto* const writable = static_cast<std::uint32_t*>(const_cast<void*>(source));
        if (update_status(batch, activity, cache, writable, d_plan.get()) !=
                cudaErrorInvalidValue ||
            update_status(batch, activity, cache, d_errors.get(), writable) !=
                cudaErrorInvalidValue) {
          return false;
        }
        return true;
      };
      CHECK(rejects_control_aliases(batch.atom_offsets));
      CHECK(rejects_control_aliases(batch.qm_positions));

      DeviceBuffer<std::uint8_t> d_misaligned;
      CUDA_CHECK(d_misaligned.upload(std::vector<std::uint8_t>(256u), stream));
      auto* const misaligned = d_misaligned.get() + 1;
      Gfn2ExternalPointChargeDeviceBatch malformed_batch = batch;
      malformed_batch.atom_offsets = reinterpret_cast<const std::int64_t*>(misaligned);
      CHECK(update_status(malformed_batch, activity, cache, d_errors.get(), d_plan.get()) ==
            cudaErrorInvalidValue);
      malformed_batch = batch;
      malformed_batch.shell_to_atom = reinterpret_cast<const std::int64_t*>(misaligned);
      CHECK(update_status(malformed_batch, activity, cache, d_errors.get(), d_plan.get()) ==
            cudaErrorInvalidValue);
      malformed_batch = batch;
      malformed_batch.shell_hardness = reinterpret_cast<const double*>(misaligned);
      CHECK(update_status(malformed_batch, activity, cache, d_errors.get(), d_plan.get()) ==
            cudaErrorInvalidValue);
      malformed_batch = batch;
      malformed_batch.qm_positions = reinterpret_cast<const double*>(misaligned);
      CHECK(update_status(malformed_batch, activity, cache, d_errors.get(), d_plan.get()) ==
            cudaErrorInvalidValue);
      malformed_batch = batch;
      malformed_batch.point_positions = reinterpret_cast<const double*>(misaligned);
      CHECK(update_status(malformed_batch, activity, cache, d_errors.get(), d_plan.get()) ==
            cudaErrorInvalidValue);
      Gfn2ExternalPointChargeDeviceCache malformed_cache = cache;
      malformed_cache.shell_potentials = reinterpret_cast<double*>(misaligned);
      CHECK(update_status(batch, activity, malformed_cache, d_errors.get(), d_plan.get()) ==
            cudaErrorInvalidValue);
      Gfn2SccIterationDeviceActivity malformed_activity = activity;
      malformed_activity.sequence_active = reinterpret_cast<const std::uint32_t*>(misaligned);
      CHECK(update_status(batch, malformed_activity, cache, d_errors.get(), d_plan.get()) ==
            cudaErrorInvalidValue);
      CHECK(update_status(batch, activity, cache, reinterpret_cast<std::uint32_t*>(misaligned),
                          d_plan.get()) == cudaErrorInvalidValue);
      CHECK(update_status(batch, activity, cache, d_errors.get(),
                          reinterpret_cast<std::uint32_t*>(misaligned)) == cudaErrorInvalidValue);
      CHECK(energy_status(batch, activity, cache, reinterpret_cast<const double*>(misaligned),
                          d_energy.get(), d_errors.get(), d_plan.get()) == cudaErrorInvalidValue);
      CHECK(energy_status(batch, activity, cache, d_raw.get(),
                          reinterpret_cast<double*>(misaligned), d_errors.get(),
                          d_plan.get()) == cudaErrorInvalidValue);

      active.assign(8u, 1u);
      active[1] = 0u;
      qm_positions[3] = std::numeric_limits<double>::quiet_NaN();
      point_charges[1] = std::numeric_limits<double>::infinity();
      raw[1] = std::numeric_limits<double>::quiet_NaN();
      CUDA_CHECK(cudaMemcpyAsync(d_active.get(), active.data(), active.size(),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_qm_positions.get(), qm_positions.data(),
                                 qm_positions.size() * sizeof(double), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(d_point_charges.get(), point_charges.data(),
                                 point_charges.size() * sizeof(double), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(d_raw.get(), raw.data(), raw.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemsetAsync(d_errors.get(), 0, 8u * sizeof(std::uint32_t), stream));
      CUDA_CHECK(cudaMemsetAsync(d_plan.get(), 0, sizeof(std::uint32_t), stream));
      const double sentinel = kSentinel;
      CUDA_CHECK(cudaMemcpyAsync(d_cache.get() + 1, &sentinel, sizeof(sentinel),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_energy.get() + 1, &sentinel, sizeof(sentinel),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(update_gfn2_external_point_charge_scc_potential_cache_cuda(
          batch, activity, cache, kGeometryGeneration, workspace, d_errors.get(), d_plan.get(),
          stream));
      CUDA_CHECK(evaluate_gfn2_external_point_charge_scc_energy_cuda(
          batch, activity, cache, kGeometryGeneration, d_raw.get(), d_energy.get(), d_errors.get(),
          d_plan.get(), stream));
      CUDA_CHECK(d_cache.download(potentials, stream));
      CUDA_CHECK(d_energy.download(energies, stream));
      CUDA_CHECK(d_errors.download(errors, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(potentials[1] == kSentinel && energies[1] == kSentinel && errors[1] == 0u);

      /* Restore the active member and prove changed-input graph replay uses the rebuilt cache. */
      active[1] = 1u;
      qm_positions[3] = 2.0;
      point_charges[1] = 0.5;
      raw[1] = -0.2;
      CUDA_CHECK(cudaMemcpyAsync(d_active.get(), active.data(), active.size(),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_qm_positions.get(), qm_positions.data(),
                                 qm_positions.size() * sizeof(double), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(d_point_charges.get(), point_charges.data(),
                                 point_charges.size() * sizeof(double), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(d_raw.get(), raw.data(), raw.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t executable = nullptr;
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      CUDA_CHECK(reset_gfn2_external_point_charge_scc_errors_cuda(batch_size, d_errors.get(),
                                                                  d_plan.get(), stream));
      CUDA_CHECK(update_gfn2_external_point_charge_scc_potential_cache_cuda(
          batch, activity, cache, kGeometryGeneration, workspace, d_errors.get(), d_plan.get(),
          stream));
      CUDA_CHECK(evaluate_gfn2_external_point_charge_scc_energy_cuda(
          batch, activity, cache, kGeometryGeneration, d_raw.get(), d_energy.get(), d_errors.get(),
          d_plan.get(), stream));
      CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
      point_charges[0] = -0.35;
      raw[0] = 0.27;
      CUDA_CHECK(cudaMemcpyAsync(d_point_charges.get(), point_charges.data(),
                                 point_charges.size() * sizeof(double), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(d_raw.get(), raw.data(), raw.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaGraphLaunch(executable, stream));
      CUDA_CHECK(d_cache.download(potentials, stream));
      CUDA_CHECK(d_energy.download(energies, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(close(potentials[0], point_charges[0] / distance));
      CHECK(close(energies[0], raw[0] * point_charges[0] / distance));
      CUDA_CHECK(cudaGraphExecDestroy(executable));
      CUDA_CHECK(cudaGraphDestroy(graph));

      active.assign(8u, 0u);
      CUDA_CHECK(cudaMemcpyAsync(d_active.get(), active.data(), active.size(),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemsetAsync(d_plan.get(), 0, sizeof(std::uint32_t), stream));
      CUDA_CHECK(evaluate_gfn2_external_point_charge_scc_energy_cuda(
          batch, activity, cache, kGeometryGeneration + 1u, d_raw.get(), d_energy.get(),
          d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(d_plan.download(plan, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(plan[0] == 0u);
      active[0] = 1u;
      CUDA_CHECK(cudaMemcpyAsync(d_active.get(), active.data(), active.size(),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(evaluate_gfn2_external_point_charge_scc_energy_cuda(
          batch, activity, cache, kGeometryGeneration + 1u, d_raw.get(), d_energy.get(),
          d_errors.get(), d_plan.get(), stream));
      CUDA_CHECK(d_plan.download(plan, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(plan[0] ==
            static_cast<std::uint32_t>(
                xtbloom::detail::cuda::Gfn2ExternalPointChargeDeviceError::kCacheMismatch));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}

}  // namespace

int main() {
  if (const int line = test_batches_and_graph(); line != 0) {
    return line;
  }
  if (const int line = test_es2_tree_overflow_is_not_published(); line != 0) {
    return line;
  }
  if (const int line = test_activity_peer_isolation_and_provenance(); line != 0) {
    return line;
  }
  if (const int line = test_es3_batches_inactive_and_graph(); line != 0) {
    return line;
  }
  if (const int line = test_aes2_batches_inactive_and_graph(); line != 0) {
    return line;
  }
  return test_point_charge_batches_inactive_and_graph();
}
