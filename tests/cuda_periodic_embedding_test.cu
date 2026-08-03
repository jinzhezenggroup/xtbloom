#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "backends/cuda/gfn2_periodic_embedding.cuh"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::cuda::Gfn2PeriodicEmbeddingDeviceBatch;
using gpuxtb::detail::cuda::Gfn2PeriodicEmbeddingDeviceError;
using gpuxtb::detail::cuda::Gfn2PeriodicEmbeddingDeviceWorkspace;

constexpr std::uint64_t kPlanToken = 0x8f51ce009fa78123ULL;
constexpr double kPotentialSentinel = -913.25;
constexpr double kEnergySentinel = 417.5;
constexpr gpuxtb_status_t kStatusSentinel = GPUXTB_STATUS_EIGENSOLVER_FAILED;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  cudaError_t allocate(std::size_t count) {
    if (count == 0u) {
      return cudaErrorInvalidValue;
    }
    count_ = count;
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t copy_from(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    if (source == nullptr || count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t copy_to(T* destination, std::size_t count, cudaStream_t stream = nullptr) const {
    if (destination == nullptr || count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

template <typename T>
cudaError_t allocate_and_copy(DeviceBuffer<T>& destination, const std::vector<T>& source,
                              cudaStream_t stream) {
  cudaError_t status = destination.allocate(source.size());
  return status == cudaSuccess ? destination.copy_from(source.data(), source.size(), stream)
                               : status;
}

struct HostCase {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<double> shifts;
  std::vector<double> matrices;
  std::vector<double> mixed_charges;
  std::vector<double> raw_charges;
  std::vector<double> expected_potentials;
  std::vector<double> expected_energies;

  std::size_t batch_size() const { return atom_offsets.size() - 1u; }
  std::size_t total_atoms() const { return shifts.size(); }
};

/* CPU-order reference with distinct mixed and raw atomic charge vectors. */
void evaluate_reference(HostCase& data) {
  data.expected_potentials.assign(data.total_atoms(), kPotentialSentinel);
  data.expected_energies.assign(data.batch_size(), kEnergySentinel);
  for (std::size_t system = 0u; system < data.batch_size(); ++system) {
    const std::int64_t atom_begin = data.atom_offsets[system];
    const std::int64_t atom_end = data.atom_offsets[system + 1u];
    const std::int64_t atom_count = atom_end - atom_begin;
    if (atom_count == 0) {
      data.expected_energies[system] = 0.0;
      continue;
    }
    const double* matrix = data.matrices.data() + data.matrix_offsets[system];
    double linear = 0.0;
    double quadratic = 0.0;
    for (std::int64_t row = 0; row < atom_count; ++row) {
      double mixed_response = 0.0;
      double raw_response = 0.0;
      for (std::int64_t column = 0; column < atom_count; ++column) {
        const double element =
            column < row ? matrix[column * atom_count + row] : matrix[row * atom_count + column];
        mixed_response =
            std::fma(element, data.mixed_charges[static_cast<std::size_t>(atom_begin + column)],
                     mixed_response);
        raw_response = std::fma(
            element, data.raw_charges[static_cast<std::size_t>(atom_begin + column)], raw_response);
      }
      const std::size_t atom = static_cast<std::size_t>(atom_begin + row);
      data.expected_potentials[atom] = data.shifts[atom] + mixed_response;
      linear = std::fma(data.raw_charges[atom], data.shifts[atom], linear);
      quadratic = std::fma(data.raw_charges[atom], raw_response, quadratic);
    }
    data.expected_energies[system] = std::fma(0.5, quadratic, linear);
  }
}

HostCase make_case(std::size_t batch_size) {
  HostCase data;
  data.atom_offsets.resize(batch_size + 1u, 0);
  data.matrix_offsets.resize(batch_size + 1u, 0);
  std::int64_t atoms = 0;
  std::int64_t matrix_elements = 0;
  for (std::size_t system = 0u; system < batch_size; ++system) {
    data.atom_offsets[system] = atoms;
    data.matrix_offsets[system] = matrix_elements;
    /* Batch one remains nonempty; larger batches deliberately include empties. */
    const std::int64_t atom_count = batch_size == 1u ? 3 : static_cast<std::int64_t>(system % 5u);
    atoms += atom_count;
    matrix_elements += atom_count * atom_count;
  }
  data.atom_offsets[batch_size] = atoms;
  data.matrix_offsets[batch_size] = matrix_elements;

  data.shifts.resize(static_cast<std::size_t>(atoms));
  data.mixed_charges.resize(static_cast<std::size_t>(atoms));
  data.raw_charges.resize(static_cast<std::size_t>(atoms));
  data.matrices.resize(static_cast<std::size_t>(matrix_elements));
  for (std::int64_t atom = 0; atom < atoms; ++atom) {
    const double sign = atom % 2 == 0 ? 1.0 : -1.0;
    data.shifts[static_cast<std::size_t>(atom)] =
        sign * (0.013 + 0.002 * static_cast<double>(atom % 11));
    data.mixed_charges[static_cast<std::size_t>(atom)] =
        sign * (0.071 + 0.009 * static_cast<double>(atom % 7));
    data.raw_charges[static_cast<std::size_t>(atom)] =
        -sign * (0.037 + 0.011 * static_cast<double>(atom % 13));
  }
  for (std::size_t system = 0u; system < batch_size; ++system) {
    const std::int64_t atom_count = data.atom_offsets[system + 1u] - data.atom_offsets[system];
    const std::int64_t matrix_begin = data.matrix_offsets[system];
    for (std::int64_t row = 0; row < atom_count; ++row) {
      for (std::int64_t column = row; column < atom_count; ++column) {
        double value = row == column ? 0.4 + 0.03 * static_cast<double>((system + row) % 5u)
                                     : 0.015 * static_cast<double>(1 + row + column);
        if ((system + static_cast<std::size_t>(row + column)) % 2u != 0u) {
          value = -value;
        }
        data.matrices[static_cast<std::size_t>(matrix_begin + row * atom_count + column)] = value;
        data.matrices[static_cast<std::size_t>(matrix_begin + column * atom_count + row)] = value;
      }
    }
  }
  /* The host contract treats opposite signed zero entries as symmetric. */
  for (std::size_t system = 0u; system < batch_size; ++system) {
    const std::int64_t atom_count = data.atom_offsets[system + 1u] - data.atom_offsets[system];
    if (atom_count >= 2) {
      const std::size_t matrix_begin = static_cast<std::size_t>(data.matrix_offsets[system]);
      data.matrices[matrix_begin + 1u] = 0.0;
      data.matrices[matrix_begin + static_cast<std::size_t>(atom_count)] = -0.0;
      break;
    }
  }
  evaluate_reference(data);
  return data;
}

struct DeviceFixture {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<double> shifts;
  DeviceBuffer<double> matrices;
  DeviceBuffer<double> mixed;
  DeviceBuffer<double> raw;
  DeviceBuffer<double> potentials;
  DeviceBuffer<double> energies;
  DeviceBuffer<gpuxtb_status_t> statuses;
  DeviceBuffer<double> potential_scratch;
  DeviceBuffer<double> raw_response_scratch;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> error;

  cudaError_t initialize(const HostCase& host, cudaStream_t stream) {
    const std::vector<double> potential_seed(host.total_atoms(), kPotentialSentinel);
    const std::vector<double> energy_seed(host.batch_size(), kEnergySentinel);
    const std::vector<gpuxtb_status_t> status_seed(host.batch_size(), kStatusSentinel);
    cudaError_t status = allocate_and_copy(atom_offsets, host.atom_offsets, stream);
    if (status == cudaSuccess) {
      status = allocate_and_copy(matrix_offsets, host.matrix_offsets, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(shifts, host.shifts, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(matrices, host.matrices, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(mixed, host.mixed_charges, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(raw, host.raw_charges, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(potentials, potential_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(energies, energy_seed, stream);
    }
    if (status == cudaSuccess) {
      status = allocate_and_copy(statuses, status_seed, stream);
    }
    if (status == cudaSuccess) {
      status = potential_scratch.allocate(std::max<std::size_t>(host.total_atoms(), 1u));
    }
    if (status == cudaSuccess) {
      status = raw_response_scratch.allocate(std::max<std::size_t>(host.total_atoms(), 1u));
    }
    if (status == cudaSuccess) {
      status = sequence_active.allocate(1u);
    }
    if (status == cudaSuccess) {
      status = error.allocate(1u);
    }
    return status;
  }

  Gfn2PeriodicEmbeddingDeviceBatch batch(const HostCase& host) const {
    return Gfn2PeriodicEmbeddingDeviceBatch{
        static_cast<std::int64_t>(host.batch_size()),
        static_cast<std::int64_t>(host.total_atoms()),
        static_cast<std::int64_t>(host.matrices.size()),
        static_cast<std::int64_t>(host.atom_offsets.size()),
        static_cast<std::int64_t>(host.matrix_offsets.size()),
        static_cast<std::int64_t>(host.shifts.size()),
        static_cast<std::int64_t>(host.matrices.size()),
        kPlanToken,
        atom_offsets.get(),
        matrix_offsets.get(),
        shifts.get(),
        matrices.get(),
    };
  }

  Gfn2PeriodicEmbeddingDeviceWorkspace workspace(const HostCase& host) {
    return Gfn2PeriodicEmbeddingDeviceWorkspace{potential_scratch.get(),
                                                raw_response_scratch.get(),
                                                sequence_active.get(),
                                                static_cast<std::int64_t>(host.total_atoms()),
                                                1,
                                                kPlanToken};
  }

  cudaError_t reset_outputs(const HostCase& host, cudaStream_t stream) {
    const std::vector<double> potential_seed(host.total_atoms(), kPotentialSentinel);
    const std::vector<double> energy_seed(host.batch_size(), kEnergySentinel);
    const std::vector<gpuxtb_status_t> status_seed(host.batch_size(), kStatusSentinel);
    cudaError_t status = potentials.copy_from(potential_seed.data(), potential_seed.size(), stream);
    if (status == cudaSuccess) {
      status = energies.copy_from(energy_seed.data(), energy_seed.size(), stream);
    }
    return status == cudaSuccess
               ? statuses.copy_from(status_seed.data(), status_seed.size(), stream)
               : status;
  }
};

struct HostResults {
  std::vector<double> potentials;
  std::vector<double> energies;
  std::vector<gpuxtb_status_t> statuses;
  std::uint32_t error = 99u;
};

cudaError_t copy_results(const HostCase& host, const DeviceFixture& device, HostResults& result,
                         cudaStream_t stream) {
  result.potentials.resize(host.total_atoms());
  result.energies.resize(host.batch_size());
  result.statuses.resize(host.batch_size());
  cudaError_t status =
      device.potentials.copy_to(result.potentials.data(), result.potentials.size(), stream);
  if (status == cudaSuccess) {
    status = device.energies.copy_to(result.energies.data(), result.energies.size(), stream);
  }
  if (status == cudaSuccess) {
    status = device.statuses.copy_to(result.statuses.data(), result.statuses.size(), stream);
  }
  return status == cudaSuccess ? device.error.copy_to(&result.error, 1u, stream) : status;
}

int compare_success(const HostCase& host, const HostResults& actual) {
  CHECK(actual.error == static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kSuccess));
  CHECK(actual.potentials == host.expected_potentials);
  CHECK(actual.energies == host.expected_energies);
  CHECK(std::all_of(actual.statuses.begin(), actual.statuses.end(),
                    [](gpuxtb_status_t status) { return status == GPUXTB_STATUS_SUCCESS; }));
  return 0;
}

int test_batch_sizes_custom_stream_and_distinct_charge_semantics() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    HostCase host = make_case(batch_size);
    CHECK(host.mixed_charges != host.raw_charges);
    DeviceFixture device;
    CUDA_CHECK(device.initialize(host, stream));
    CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_device_error_cuda(
        device.error.get(), stream));
    CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_cuda(
        device.batch(host), device.mixed.get(), device.raw.get(), device.potentials.get(),
        device.energies.get(), device.statuses.get(), device.workspace(host), device.error.get(),
        stream));
    HostResults actual;
    CUDA_CHECK(copy_results(host, device, actual, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int comparison = compare_success(host, actual);
    CHECK(comparison == 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_peer_failure_isolation_and_transactional_publication() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  HostCase host = make_case(8u);
  const std::size_t failed_system = 3u;
  const std::size_t failed_atom = static_cast<std::size_t>(host.atom_offsets[failed_system]);
  host.raw_charges[failed_atom] = std::numeric_limits<double>::quiet_NaN();

  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_device_error_cuda(
      device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_cuda(
      device.batch(host), device.mixed.get(), device.raw.get(), device.potentials.get(),
      device.energies.get(), device.statuses.get(), device.workspace(host), device.error.get(),
      stream));
  HostResults actual;
  CUDA_CHECK(copy_results(host, device, actual, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(actual.error ==
        static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kNonfiniteRawCharge));

  HostCase healthy_reference = host;
  healthy_reference.raw_charges[failed_atom] = 0.123;
  evaluate_reference(healthy_reference);
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::size_t begin = static_cast<std::size_t>(host.atom_offsets[system]);
    const std::size_t end = static_cast<std::size_t>(host.atom_offsets[system + 1u]);
    if (system == failed_system) {
      CHECK(actual.statuses[system] == GPUXTB_STATUS_INTERNAL_ERROR);
      CHECK(actual.energies[system] == kEnergySentinel);
      for (std::size_t atom = begin; atom < end; ++atom) {
        CHECK(actual.potentials[atom] == kPotentialSentinel);
      }
    } else {
      CHECK(actual.statuses[system] == GPUXTB_STATUS_SUCCESS);
      CHECK(actual.energies[system] == healthy_reference.expected_energies[system]);
      for (std::size_t atom = begin; atom < end; ++atom) {
        CHECK(actual.potentials[atom] == healthy_reference.expected_potentials[atom]);
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_nonsymmetric_matrix_failure_is_isolated() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  HostCase host = make_case(8u);
  const std::size_t failed_system = 4u;
  const std::int64_t atom_count =
      host.atom_offsets[failed_system + 1u] - host.atom_offsets[failed_system];
  CHECK(atom_count >= 2);
  const std::size_t matrix_begin = static_cast<std::size_t>(host.matrix_offsets[failed_system]);
  host.matrices[matrix_begin + 1u] = 0.25;
  host.matrices[matrix_begin + static_cast<std::size_t>(atom_count)] = -0.25;

  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_device_error_cuda(
      device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_cuda(
      device.batch(host), device.mixed.get(), device.raw.get(), device.potentials.get(),
      device.energies.get(), device.statuses.get(), device.workspace(host), device.error.get(),
      stream));
  HostResults actual;
  CUDA_CHECK(copy_results(host, device, actual, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(actual.error ==
        static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kNonsymmetricResponseMatrix));

  HostCase healthy_reference = host;
  healthy_reference.matrices[matrix_begin + static_cast<std::size_t>(atom_count)] = 0.25;
  evaluate_reference(healthy_reference);
  for (std::size_t system = 0u; system < host.batch_size(); ++system) {
    const std::size_t begin = static_cast<std::size_t>(host.atom_offsets[system]);
    const std::size_t end = static_cast<std::size_t>(host.atom_offsets[system + 1u]);
    if (system == failed_system) {
      CHECK(actual.statuses[system] == GPUXTB_STATUS_INTERNAL_ERROR);
      CHECK(actual.energies[system] == kEnergySentinel);
      for (std::size_t atom = begin; atom < end; ++atom) {
        CHECK(actual.potentials[atom] == kPotentialSentinel);
      }
    } else {
      CHECK(actual.statuses[system] == GPUXTB_STATUS_SUCCESS);
      CHECK(actual.energies[system] == healthy_reference.expected_energies[system]);
      for (std::size_t atom = begin; atom < end; ++atom) {
        CHECK(actual.potentials[atom] == healthy_reference.expected_potentials[atom]);
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_all_empty_batch_accepts_null_numerical_ranges() {
  constexpr std::size_t batch_size = 8u;
  const std::vector<std::int64_t> offsets(batch_size + 1u, 0);
  std::vector<double> energy_seed(batch_size, kEnergySentinel);
  std::vector<gpuxtb_status_t> status_seed(batch_size, kStatusSentinel);
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<double> energies;
  DeviceBuffer<gpuxtb_status_t> statuses;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> error;
  CUDA_CHECK(atom_offsets.allocate(offsets.size()));
  CUDA_CHECK(matrix_offsets.allocate(offsets.size()));
  CUDA_CHECK(energies.allocate(batch_size));
  CUDA_CHECK(statuses.allocate(batch_size));
  CUDA_CHECK(sequence_active.allocate(1u));
  CUDA_CHECK(error.allocate(1u));
  CUDA_CHECK(atom_offsets.copy_from(offsets.data(), offsets.size()));
  CUDA_CHECK(matrix_offsets.copy_from(offsets.data(), offsets.size()));
  CUDA_CHECK(energies.copy_from(energy_seed.data(), energy_seed.size()));
  CUDA_CHECK(statuses.copy_from(status_seed.data(), status_seed.size()));

  const Gfn2PeriodicEmbeddingDeviceBatch batch{
      static_cast<std::int64_t>(batch_size),
      0,
      0,
      static_cast<std::int64_t>(offsets.size()),
      static_cast<std::int64_t>(offsets.size()),
      0,
      0,
      kPlanToken,
      atom_offsets.get(),
      matrix_offsets.get(),
      nullptr,
      nullptr,
  };
  const Gfn2PeriodicEmbeddingDeviceWorkspace workspace{nullptr, nullptr, sequence_active.get(),
                                                       0,       1,       kPlanToken};
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_device_error_cuda(error.get()));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_cuda(
      batch, nullptr, nullptr, nullptr, energies.get(), statuses.get(), workspace, error.get()));
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(energies.copy_to(energy_seed.data(), energy_seed.size()));
  CUDA_CHECK(statuses.copy_to(status_seed.data(), status_seed.size()));
  CUDA_CHECK(error.copy_to(&semantic_error, 1u));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  CHECK(std::all_of(energy_seed.begin(), energy_seed.end(),
                    [](double value) { return value == 0.0; }));
  CHECK(std::all_of(status_seed.begin(), status_seed.end(),
                    [](gpuxtb_status_t status) { return status == GPUXTB_STATUS_SUCCESS; }));
  return 0;
}

int test_topology_failure_and_sticky_upstream_are_whole_call_atomic() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  HostCase host = make_case(8u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));

  std::vector<std::int64_t> invalid_offsets = host.matrix_offsets;
  ++invalid_offsets[2];
  CUDA_CHECK(
      device.matrix_offsets.copy_from(invalid_offsets.data(), invalid_offsets.size(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_device_error_cuda(
      device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_cuda(
      device.batch(host), device.mixed.get(), device.raw.get(), device.potentials.get(),
      device.energies.get(), device.statuses.get(), device.workspace(host), device.error.get(),
      stream));
  HostResults actual;
  CUDA_CHECK(copy_results(host, device, actual, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(actual.error ==
        static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kInvalidOffsets));
  CHECK(std::all_of(actual.potentials.begin(), actual.potentials.end(),
                    [](double value) { return value == kPotentialSentinel; }));
  CHECK(std::all_of(actual.energies.begin(), actual.energies.end(),
                    [](double value) { return value == kEnergySentinel; }));
  CHECK(std::all_of(actual.statuses.begin(), actual.statuses.end(),
                    [](gpuxtb_status_t status) { return status == kStatusSentinel; }));

  CUDA_CHECK(device.matrix_offsets.copy_from(host.matrix_offsets.data(), host.matrix_offsets.size(),
                                             stream));
  std::vector<std::int64_t> extreme_offsets = host.atom_offsets;
  extreme_offsets[1] = std::numeric_limits<std::int64_t>::min();
  CUDA_CHECK(device.atom_offsets.copy_from(extreme_offsets.data(), extreme_offsets.size(), stream));
  CUDA_CHECK(device.reset_outputs(host, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_device_error_cuda(
      device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_cuda(
      device.batch(host), device.mixed.get(), device.raw.get(), device.potentials.get(),
      device.energies.get(), device.statuses.get(), device.workspace(host), device.error.get(),
      stream));
  CUDA_CHECK(copy_results(host, device, actual, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(actual.error ==
        static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kInvalidOffsets));
  CHECK(std::all_of(actual.potentials.begin(), actual.potentials.end(),
                    [](double value) { return value == kPotentialSentinel; }));
  CHECK(std::all_of(actual.energies.begin(), actual.energies.end(),
                    [](double value) { return value == kEnergySentinel; }));
  CHECK(std::all_of(actual.statuses.begin(), actual.statuses.end(),
                    [](gpuxtb_status_t status) { return status == kStatusSentinel; }));

  CUDA_CHECK(
      device.atom_offsets.copy_from(host.atom_offsets.data(), host.atom_offsets.size(), stream));
  CUDA_CHECK(device.reset_outputs(host, stream));
  const std::uint32_t upstream =
      static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kNonfiniteShift);
  CUDA_CHECK(device.error.copy_from(&upstream, 1u, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_cuda(
      device.batch(host), device.mixed.get(), device.raw.get(), device.potentials.get(),
      device.energies.get(), device.statuses.get(), device.workspace(host), device.error.get(),
      stream));
  CUDA_CHECK(copy_results(host, device, actual, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(actual.error == upstream);
  CHECK(std::all_of(actual.potentials.begin(), actual.potentials.end(),
                    [](double value) { return value == kPotentialSentinel; }));
  CHECK(std::all_of(actual.energies.begin(), actual.energies.end(),
                    [](double value) { return value == kEnergySentinel; }));
  CHECK(std::all_of(actual.statuses.begin(), actual.statuses.end(),
                    [](gpuxtb_status_t status) { return status == kStatusSentinel; }));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_host_validation_and_caller_owned_workspace() {
  HostCase host = make_case(8u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, nullptr));
  const Gfn2PeriodicEmbeddingDeviceBatch batch = device.batch(host);
  Gfn2PeriodicEmbeddingDeviceWorkspace workspace = device.workspace(host);

  --workspace.atom_elements;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_cuda(
            batch, device.mixed.get(), device.raw.get(), device.potentials.get(),
            device.energies.get(), device.statuses.get(), workspace,
            device.error.get()) == cudaErrorInvalidValue);
  workspace = device.workspace(host);
  ++workspace.plan_token;
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_cuda(
            batch, device.mixed.get(), device.raw.get(), device.potentials.get(),
            device.energies.get(), device.statuses.get(), workspace,
            device.error.get()) == cudaErrorInvalidValue);
  workspace = device.workspace(host);
  CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_cuda(
            batch, device.mixed.get(), device.raw.get(), device.mixed.get(), device.energies.get(),
            device.statuses.get(), workspace, device.error.get()) == cudaErrorInvalidValue);
  CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_device_error_cuda(nullptr) ==
        cudaErrorInvalidValue);
  return 0;
}

int test_cuda_graph_capture_and_replay() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  HostCase host = make_case(32u);
  DeviceFixture device;
  CUDA_CHECK(device.initialize(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_device_error_cuda(
      device.error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_cuda(
      device.batch(host), device.mixed.get(), device.raw.get(), device.potentials.get(),
      device.energies.get(), device.statuses.get(), device.workspace(host), device.error.get(),
      stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0u));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  HostResults actual;
  CUDA_CHECK(copy_results(host, device, actual, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const int comparison = compare_success(host, actual);
  CHECK(comparison == 0);
  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_batch_sizes_custom_stream_and_distinct_charge_semantics();
      status != 0) {
    return status;
  }
  if (const int status = test_peer_failure_isolation_and_transactional_publication(); status != 0) {
    return status;
  }
  if (const int status = test_nonsymmetric_matrix_failure_is_isolated(); status != 0) {
    return status;
  }
  if (const int status = test_all_empty_batch_accepts_null_numerical_ranges(); status != 0) {
    return status;
  }
  if (const int status = test_topology_failure_and_sticky_upstream_are_whole_call_atomic();
      status != 0) {
    return status;
  }
  if (const int status = test_host_validation_and_caller_owned_workspace(); status != 0) {
    return status;
  }
  return test_cuda_graph_capture_and_replay();
}
