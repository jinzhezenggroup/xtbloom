#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "backends/cuda/gfn2_d4.cuh"
#include "backends/cuda/gfn2_periodic_embedding.cuh"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::cuda::Gfn2D4DeviceBatch;
using gpuxtb::detail::cuda::Gfn2D4DeviceCache;
using gpuxtb::detail::cuda::Gfn2D4DeviceElementData;
using gpuxtb::detail::cuda::Gfn2D4DeviceError;
using gpuxtb::detail::cuda::Gfn2D4DeviceParameters;
using gpuxtb::detail::cuda::Gfn2D4DeviceReferenceData;
using gpuxtb::detail::cuda::Gfn2D4DeviceWorkspace;
using gpuxtb::detail::cuda::Gfn2PeriodicEmbeddingDeviceBatch;
using gpuxtb::detail::cuda::Gfn2PeriodicEmbeddingDeviceError;
using gpuxtb::detail::cuda::Gfn2PeriodicEmbeddingDeviceWorkspace;
using gpuxtb::detail::cuda::Gfn2SccIterationDeviceActivity;

constexpr std::uint64_t kPlanToken = 0x93d4a11ce5cc2026ULL;
constexpr std::uint64_t kGeometryGeneration = 17u;
constexpr double kSentinel = -741.25;
constexpr double kReferenceC6 = 2.0;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  cudaError_t allocate(std::size_t elements) {
    elements_ = std::max<std::size_t>(elements, 1u);
    return cudaMalloc(reinterpret_cast<void**>(&pointer_), elements_ * sizeof(T));
  }

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream) {
    if (values.size() > elements_) {
      return cudaErrorInvalidValue;
    }
    return values.empty() ? cudaSuccess
                          : cudaMemcpyAsync(pointer_, values.data(), values.size() * sizeof(T),
                                            cudaMemcpyHostToDevice, stream);
  }

  cudaError_t download(std::vector<T>& values, std::size_t elements, cudaStream_t stream) const {
    if (elements > elements_) {
      return cudaErrorInvalidValue;
    }
    values.resize(elements);
    return elements == 0u ? cudaSuccess
                          : cudaMemcpyAsync(values.data(), pointer_, elements * sizeof(T),
                                            cudaMemcpyDeviceToHost, stream);
  }

  T* get() { return pointer_; }
  const T* get() const { return pointer_; }

 private:
  T* pointer_ = nullptr;
  std::size_t elements_ = 0u;
};

bool near(double actual, double expected, double tolerance = 2.0e-12) {
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

double d4_charge_scale(double charge) {
  constexpr double effective_charge = 1.0;
  constexpr double qref = 1.0;
  const double qmod = charge + effective_charge;
  return std::exp(3.0 * (1.0 - std::exp(2.0 * (1.0 - qref / qmod))));
}

double d4_charge_scale_derivative(double charge) {
  constexpr double qref = 1.0;
  const double qmod = charge + 1.0;
  const double inner = std::exp(2.0 * (1.0 - qref / qmod));
  return -6.0 * inner * d4_charge_scale(charge) * qref / (qmod * qmod);
}

struct ActivityBuffers {
  DeviceBuffer<std::uint8_t> mask;
  DeviceBuffer<std::uint32_t> sequence;

  int initialize(const std::vector<std::uint8_t>& host_mask, cudaStream_t stream) {
    const std::vector<std::uint32_t> open{1u};
    CUDA_CHECK(mask.allocate(host_mask.size()));
    CUDA_CHECK(sequence.allocate(1u));
    CUDA_CHECK(mask.upload(host_mask, stream));
    CUDA_CHECK(sequence.upload(open, stream));
    return 0;
  }

  Gfn2SccIterationDeviceActivity view(std::int64_t batch_size) const {
    return {mask.get(), sequence.get(), batch_size, 1, kPlanToken};
  }
};

struct D4Fixture {
  std::size_t batch_size = 0u;
  std::vector<std::int64_t> host_atom_offsets;
  std::vector<std::int64_t> host_pair_offsets;
  std::vector<std::int32_t> host_atomic_numbers;
  std::vector<double> host_pair_data;
  std::vector<double> host_coordination;
  std::vector<double> host_mixed;
  std::vector<double> host_raw;
  std::vector<std::uint8_t> host_active;

  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> pair_offsets;
  DeviceBuffer<std::int32_t> atomic_numbers;
  DeviceBuffer<Gfn2D4DeviceElementData> elements;
  DeviceBuffer<Gfn2D4DeviceReferenceData> references;
  DeviceBuffer<double> reference_c6;
  DeviceBuffer<double> pair_data;
  DeviceBuffer<double> coordination;
  DeviceBuffer<double> mixed;
  DeviceBuffer<double> raw;
  DeviceBuffer<double> potentials;
  DeviceBuffer<double> energies;
  DeviceBuffer<double> weights;
  DeviceBuffer<double> charge_derivatives;
  DeviceBuffer<double> atom_scratch;
  DeviceBuffer<double> batch_scratch;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;
  ActivityBuffers activity;

  Gfn2D4DeviceBatch batch{};
  Gfn2D4DeviceParameters parameters{};
  Gfn2D4DeviceCache cache{};

  int initialize(std::size_t systems, cudaStream_t stream) {
    batch_size = systems;
    const std::size_t atom_count = systems * 2u;
    host_atom_offsets.resize(systems + 1u);
    host_pair_offsets.resize(systems + 1u);
    host_atomic_numbers.resize(atom_count);
    host_pair_data.resize(systems * gpuxtb::detail::cuda::kGfn2D4PairDataElements);
    host_coordination.assign(atom_count, 0.0);
    host_mixed.resize(atom_count);
    host_raw.resize(atom_count);
    host_active.resize(systems);
    for (std::size_t system = 0u; system <= systems; ++system) {
      host_atom_offsets[system] = static_cast<std::int64_t>(system * 2u);
      host_pair_offsets[system] = static_cast<std::int64_t>(system);
    }
    for (std::size_t system = 0u; system < systems; ++system) {
      const std::size_t atom = system * 2u;
      /* Two numerically identical element records make an in-range reorder a
       * pure plan-binding trap: physics stays unchanged, but the D4 topology
       * fingerprint must still reject the reordered device array. */
      host_atomic_numbers[atom] = 1;
      host_atomic_numbers[atom + 1u] = 2;
      host_mixed[atom] = 0.08 + 0.002 * static_cast<double>(system % 7u);
      host_mixed[atom + 1u] = -0.17 + 0.003 * static_cast<double>(system % 5u);
      host_raw[atom] = -0.24 + 0.004 * static_cast<double>(system % 11u);
      host_raw[atom + 1u] = 0.13 - 0.002 * static_cast<double>(system % 3u);
      host_active[system] = system % 2u == 0u ? 1u : 0u;
      const std::size_t pair = system * gpuxtb::detail::cuda::kGfn2D4PairDataElements;
      host_pair_data[pair] = 1.0;
      host_pair_data[pair + 1u] = 0.0;
      host_pair_data[pair + 2u] = 0.0;
      host_pair_data[pair + 3u] = 0.31 + 0.001 * static_cast<double>(system);
      host_pair_data[pair + 4u] = 0.0;
      if (host_active[system] == 0u) {
        /* Inactive numerical slices are poison: any accidental read is caught
         * both semantically and by Compute Sanitizer init/mem checks. */
        host_mixed[atom] = std::numeric_limits<double>::quiet_NaN();
        host_mixed[atom + 1u] = std::numeric_limits<double>::quiet_NaN();
        host_raw[atom] = std::numeric_limits<double>::quiet_NaN();
        host_raw[atom + 1u] = std::numeric_limits<double>::quiet_NaN();
        host_coordination[atom] = std::numeric_limits<double>::quiet_NaN();
        host_coordination[atom + 1u] = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t field = 0u; field < gpuxtb::detail::cuda::kGfn2D4PairDataElements;
             ++field) {
          host_pair_data[pair + field] = std::numeric_limits<double>::quiet_NaN();
        }
      }
    }
    const std::vector<Gfn2D4DeviceElementData> host_elements{{0u, 1u, 1.0, 1.0, 1.0, 1.0, 1.0},
                                                             {0u, 1u, 1.0, 1.0, 1.0, 1.0, 1.0}};
    const std::vector<Gfn2D4DeviceReferenceData> host_references{{0.0, 0.0, 1u}};
    const std::vector<double> host_reference_c6{kReferenceC6};
    const std::vector<double> potential_seed(atom_count, kSentinel);
    const std::vector<double> energy_seed(systems, kSentinel);
    const std::size_t weight_count = atom_count * gpuxtb::detail::cuda::kGfn2D4MaximumReferences;
    CUDA_CHECK(atom_offsets.allocate(host_atom_offsets.size()));
    CUDA_CHECK(pair_offsets.allocate(host_pair_offsets.size()));
    CUDA_CHECK(atomic_numbers.allocate(atom_count));
    CUDA_CHECK(elements.allocate(host_elements.size()));
    CUDA_CHECK(references.allocate(host_references.size()));
    CUDA_CHECK(reference_c6.allocate(host_reference_c6.size()));
    CUDA_CHECK(pair_data.allocate(host_pair_data.size()));
    CUDA_CHECK(coordination.allocate(atom_count));
    CUDA_CHECK(mixed.allocate(atom_count));
    CUDA_CHECK(raw.allocate(atom_count));
    CUDA_CHECK(potentials.allocate(atom_count));
    CUDA_CHECK(energies.allocate(systems));
    CUDA_CHECK(weights.allocate(weight_count));
    CUDA_CHECK(charge_derivatives.allocate(weight_count));
    CUDA_CHECK(atom_scratch.allocate(atom_count));
    CUDA_CHECK(batch_scratch.allocate(systems));
    CUDA_CHECK(system_errors.allocate(systems));
    CUDA_CHECK(device_error.allocate(1u));
    CUDA_CHECK(atom_offsets.upload(host_atom_offsets, stream));
    CUDA_CHECK(pair_offsets.upload(host_pair_offsets, stream));
    CUDA_CHECK(atomic_numbers.upload(host_atomic_numbers, stream));
    CUDA_CHECK(elements.upload(host_elements, stream));
    CUDA_CHECK(references.upload(host_references, stream));
    CUDA_CHECK(reference_c6.upload(host_reference_c6, stream));
    CUDA_CHECK(pair_data.upload(host_pair_data, stream));
    CUDA_CHECK(coordination.upload(host_coordination, stream));
    CUDA_CHECK(mixed.upload(host_mixed, stream));
    CUDA_CHECK(raw.upload(host_raw, stream));
    CUDA_CHECK(potentials.upload(potential_seed, stream));
    CUDA_CHECK(energies.upload(energy_seed, stream));
    CHECK(activity.initialize(host_active, stream) == 0);
    batch = {static_cast<std::int64_t>(systems),
             static_cast<std::int64_t>(atom_count),
             static_cast<std::int64_t>(systems),
             kPlanToken,
             gpuxtb::detail::cuda::gfn2_d4_atomic_number_hash(
                 host_atomic_numbers.data(), static_cast<std::int64_t>(atom_count)),
             atom_offsets.get(),
             pair_offsets.get(),
             atomic_numbers.get()};
    parameters = {elements.get(), 2, references.get(), 1, reference_c6.get(), 1};
    cache = {pair_data.get(),     static_cast<std::int64_t>(host_pair_data.size()),
             coordination.get(),  static_cast<std::int64_t>(atom_count),
             kGeometryGeneration, kPlanToken};
    return 0;
  }

  Gfn2D4DeviceWorkspace potential_workspace() {
    Gfn2D4DeviceWorkspace workspace{};
    workspace.weights = weights.get();
    workspace.weight_charge_derivatives = charge_derivatives.get();
    workspace.weight_elements = static_cast<std::int64_t>(
        batch.total_atoms * gpuxtb::detail::cuda::kGfn2D4MaximumReferences);
    workspace.atom_scratch = atom_scratch.get();
    workspace.atom_elements = batch.total_atoms;
    workspace.system_errors = system_errors.get();
    workspace.system_error_elements = batch.batch_size;
    return workspace;
  }

  Gfn2D4DeviceWorkspace energy_workspace() {
    Gfn2D4DeviceWorkspace workspace{};
    workspace.weights = weights.get();
    workspace.weight_elements = static_cast<std::int64_t>(
        batch.total_atoms * gpuxtb::detail::cuda::kGfn2D4MaximumReferences);
    workspace.batch_scratch = batch_scratch.get();
    workspace.batch_elements = batch.batch_size;
    workspace.system_errors = system_errors.get();
    workspace.system_error_elements = batch.batch_size;
    return workspace;
  }
};

int check_d4_outputs(D4Fixture& fixture, cudaStream_t stream) {
  std::vector<double> potentials;
  std::vector<double> energies;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CUDA_CHECK(fixture.potentials.download(potentials, fixture.host_mixed.size(), stream));
  CUDA_CHECK(fixture.energies.download(energies, fixture.batch_size, stream));
  CUDA_CHECK(fixture.system_errors.download(system_errors, fixture.batch_size, stream));
  CUDA_CHECK(fixture.device_error.download(device_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(device_error[0] == 0u);
  for (std::size_t system = 0u; system < fixture.batch_size; ++system) {
    const std::size_t atom = system * 2u;
    if (fixture.host_active[system] == 0u) {
      CHECK(potentials[atom] == kSentinel && potentials[atom + 1u] == kSentinel);
      CHECK(energies[system] == kSentinel);
      CHECK(system_errors[system] == 0u);
      continue;
    }
    const double damping =
        fixture.host_pair_data[system * gpuxtb::detail::cuda::kGfn2D4PairDataElements + 3u];
    const double mixed_first = d4_charge_scale(fixture.host_mixed[atom]);
    const double mixed_second = d4_charge_scale(fixture.host_mixed[atom + 1u]);
    const double expected_first = -d4_charge_scale_derivative(fixture.host_mixed[atom]) *
                                  mixed_second * kReferenceC6 * damping;
    const double expected_second = -mixed_first *
                                   d4_charge_scale_derivative(fixture.host_mixed[atom + 1u]) *
                                   kReferenceC6 * damping;
    const double expected_energy = -d4_charge_scale(fixture.host_raw[atom]) *
                                   d4_charge_scale(fixture.host_raw[atom + 1u]) * kReferenceC6 *
                                   damping;
    CHECK(near(potentials[atom], expected_first));
    CHECK(near(potentials[atom + 1u], expected_second));
    CHECK(near(energies[system], expected_energy));
    CHECK(system_errors[system] == 0u);
  }
  return 0;
}

int run_d4_batch(std::size_t batch_size, cudaStream_t stream) {
  D4Fixture fixture;
  CHECK(fixture.initialize(batch_size, stream) == 0);
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_errors_cuda(
      static_cast<std::int64_t>(batch_size), fixture.system_errors.get(),
      fixture.device_error.get(), stream));
  Gfn2D4DeviceWorkspace potential_workspace = fixture.potential_workspace();
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_potential_cuda(
      fixture.batch, fixture.parameters, fixture.cache, kGeometryGeneration, fixture.mixed.get(),
      fixture.activity.view(static_cast<std::int64_t>(batch_size)), fixture.potentials.get(),
      potential_workspace, fixture.device_error.get(), stream));
  Gfn2D4DeviceWorkspace energy_workspace = fixture.energy_workspace();
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_energy_cuda(
      fixture.batch, fixture.parameters, fixture.cache, kGeometryGeneration, fixture.raw.get(),
      fixture.activity.view(static_cast<std::int64_t>(batch_size)), fixture.energies.get(),
      energy_workspace, fixture.device_error.get(), stream));
  return check_d4_outputs(fixture, stream);
}

int test_d4_all_inactive_and_stale(cudaStream_t stream) {
  D4Fixture fixture;
  CHECK(fixture.initialize(8u, stream) == 0);
  const std::vector<std::uint8_t> inactive(8u, 0u);
  const std::vector<std::int64_t> poison_offsets(9u, std::numeric_limits<std::int64_t>::min());
  const std::vector<std::int32_t> poison_numbers(16u, -1);
  const std::vector<double> poison_pairs(8u * gpuxtb::detail::cuda::kGfn2D4PairDataElements,
                                         std::numeric_limits<double>::quiet_NaN());
  const std::vector<double> poison_atoms(16u, std::numeric_limits<double>::quiet_NaN());
  CUDA_CHECK(fixture.activity.mask.upload(inactive, stream));
  CUDA_CHECK(fixture.atom_offsets.upload(poison_offsets, stream));
  CUDA_CHECK(fixture.pair_offsets.upload(poison_offsets, stream));
  CUDA_CHECK(fixture.atomic_numbers.upload(poison_numbers, stream));
  CUDA_CHECK(fixture.pair_data.upload(poison_pairs, stream));
  CUDA_CHECK(fixture.coordination.upload(poison_atoms, stream));
  CUDA_CHECK(fixture.mixed.upload(poison_atoms, stream));
  CUDA_CHECK(fixture.raw.upload(poison_atoms, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_errors_cuda(
      8, fixture.system_errors.get(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_potential_cuda(
      fixture.batch, fixture.parameters, fixture.cache, kGeometryGeneration + 1u,
      fixture.mixed.get(), fixture.activity.view(8), fixture.potentials.get(),
      fixture.potential_workspace(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_energy_cuda(
      fixture.batch, fixture.parameters, fixture.cache, kGeometryGeneration + 1u, fixture.raw.get(),
      fixture.activity.view(8), fixture.energies.get(), fixture.energy_workspace(),
      fixture.device_error.get(), stream));
  std::vector<double> potentials;
  std::vector<double> energies;
  std::vector<std::uint32_t> error;
  CUDA_CHECK(fixture.potentials.download(potentials, 16u, stream));
  CUDA_CHECK(fixture.energies.download(energies, 8u, stream));
  CUDA_CHECK(fixture.device_error.download(error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(error[0] == 0u);
  CHECK(std::all_of(potentials.begin(), potentials.end(),
                    [](double value) { return value == kSentinel; }));
  CHECK(std::all_of(energies.begin(), energies.end(),
                    [](double value) { return value == kSentinel; }));

  /* A closed canonical sequence must not even inspect malformed member bytes. */
  const std::vector<std::uint8_t> malformed_mask(8u, 0xffu);
  const std::vector<std::uint32_t> closed{0u};
  CUDA_CHECK(fixture.activity.mask.upload(malformed_mask, stream));
  CUDA_CHECK(fixture.activity.sequence.upload(closed, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_errors_cuda(
      8, fixture.system_errors.get(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_potential_cuda(
      fixture.batch, fixture.parameters, fixture.cache, kGeometryGeneration + 1u,
      fixture.mixed.get(), fixture.activity.view(8), fixture.potentials.get(),
      fixture.potential_workspace(), fixture.device_error.get(), stream));
  CUDA_CHECK(fixture.device_error.download(error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(error[0] == 0u);
  return 0;
}

int test_d4_active_stale_and_peer_failure(cudaStream_t stream) {
  D4Fixture fixture;
  CHECK(fixture.initialize(8u, stream) == 0);
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_errors_cuda(
      8, fixture.system_errors.get(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_energy_cuda(
      fixture.batch, fixture.parameters, fixture.cache, kGeometryGeneration + 1u, fixture.raw.get(),
      fixture.activity.view(8), fixture.energies.get(), fixture.energy_workspace(),
      fixture.device_error.get(), stream));
  std::vector<double> energies;
  std::vector<std::uint32_t> error;
  CUDA_CHECK(fixture.energies.download(energies, 8u, stream));
  CUDA_CHECK(fixture.device_error.download(error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(error[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kStaleGeometry));
  CHECK(std::all_of(energies.begin(), energies.end(),
                    [](double value) { return value == kSentinel; }));

  fixture.host_raw[0] = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(fixture.raw.upload(fixture.host_raw, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_errors_cuda(
      8, fixture.system_errors.get(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_energy_cuda(
      fixture.batch, fixture.parameters, fixture.cache, kGeometryGeneration, fixture.raw.get(),
      fixture.activity.view(8), fixture.energies.get(), fixture.energy_workspace(),
      fixture.device_error.get(), stream));
  std::vector<std::uint32_t> system_errors;
  CUDA_CHECK(fixture.energies.download(energies, 8u, stream));
  CUDA_CHECK(fixture.system_errors.download(system_errors, 8u, stream));
  CUDA_CHECK(fixture.device_error.download(error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(error[0] == 0u);
  CHECK(energies[0] == kSentinel);
  CHECK(system_errors[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kNonfiniteCharge));
  CHECK(energies[2] != kSentinel && system_errors[2] == 0u);
  return 0;
}

int test_d4_atomic_number_reorder_is_plan_failure(cudaStream_t stream) {
  D4Fixture fixture;
  CHECK(fixture.initialize(1u, stream) == 0);
  std::swap(fixture.host_atomic_numbers[0], fixture.host_atomic_numbers[1]);
  CUDA_CHECK(fixture.atomic_numbers.upload(fixture.host_atomic_numbers, stream));

  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_errors_cuda(
      1, fixture.system_errors.get(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_potential_cuda(
      fixture.batch, fixture.parameters, fixture.cache, kGeometryGeneration, fixture.mixed.get(),
      fixture.activity.view(1), fixture.potentials.get(), fixture.potential_workspace(),
      fixture.device_error.get(), stream));
  std::vector<double> potentials;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CUDA_CHECK(fixture.potentials.download(potentials, 2u, stream));
  CUDA_CHECK(fixture.system_errors.download(system_errors, 1u, stream));
  CUDA_CHECK(fixture.device_error.download(device_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(device_error[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kInvalidAtomicNumber));
  CHECK(system_errors[0] == 0u);
  CHECK(potentials[0] == kSentinel && potentials[1] == kSentinel);

  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_errors_cuda(
      1, fixture.system_errors.get(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_energy_cuda(
      fixture.batch, fixture.parameters, fixture.cache, kGeometryGeneration, fixture.raw.get(),
      fixture.activity.view(1), fixture.energies.get(), fixture.energy_workspace(),
      fixture.device_error.get(), stream));
  std::vector<double> energies;
  CUDA_CHECK(fixture.energies.download(energies, 1u, stream));
  CUDA_CHECK(fixture.system_errors.download(system_errors, 1u, stream));
  CUDA_CHECK(fixture.device_error.download(device_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(device_error[0] == static_cast<std::uint32_t>(Gfn2D4DeviceError::kInvalidAtomicNumber));
  CHECK(system_errors[0] == 0u);
  CHECK(energies[0] == kSentinel);
  return 0;
}

int test_d4_inactive_offsets_are_not_consumed(cudaStream_t stream) {
  D4Fixture fixture;
  CHECK(fixture.initialize(8u, stream) == 0);
  fixture.host_active.assign(8u, 0u);
  fixture.host_active[0] = 1u;
  CUDA_CHECK(fixture.activity.mask.upload(fixture.host_active, stream));

  /* System zero remains a valid active slice. Intermediate boundaries belong
   * only to inactive trailing members and are deliberately unusable. The SCC
   * numerical kernels must not binary-search through them to discover a
   * member after activity has already supplied the system index. */
  std::vector<std::int64_t> atom_offsets = fixture.host_atom_offsets;
  std::vector<std::int64_t> pair_offsets = fixture.host_pair_offsets;
  for (std::size_t offset = 2u; offset < 8u; ++offset) {
    atom_offsets[offset] = std::numeric_limits<std::int64_t>::min();
    pair_offsets[offset] = std::numeric_limits<std::int64_t>::min();
  }
  CUDA_CHECK(fixture.atom_offsets.upload(atom_offsets, stream));
  CUDA_CHECK(fixture.pair_offsets.upload(pair_offsets, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_errors_cuda(
      8, fixture.system_errors.get(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_potential_cuda(
      fixture.batch, fixture.parameters, fixture.cache, kGeometryGeneration, fixture.mixed.get(),
      fixture.activity.view(8), fixture.potentials.get(), fixture.potential_workspace(),
      fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_energy_cuda(
      fixture.batch, fixture.parameters, fixture.cache, kGeometryGeneration, fixture.raw.get(),
      fixture.activity.view(8), fixture.energies.get(), fixture.energy_workspace(),
      fixture.device_error.get(), stream));
  return check_d4_outputs(fixture, stream);
}

struct PeriodicFixture {
  std::size_t batch_size = 0u;
  std::vector<std::int64_t> host_atom_offsets;
  std::vector<std::int64_t> host_matrix_offsets;
  std::vector<double> host_shifts;
  std::vector<double> host_matrices;
  std::vector<double> host_mixed;
  std::vector<double> host_raw;
  std::vector<std::uint8_t> host_active;

  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<double> shifts;
  DeviceBuffer<double> matrices;
  DeviceBuffer<double> mixed;
  DeviceBuffer<double> raw;
  DeviceBuffer<double> potentials;
  DeviceBuffer<double> energies;
  DeviceBuffer<double> potential_scratch;
  DeviceBuffer<double> raw_scratch;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> device_error;
  ActivityBuffers activity;
  Gfn2PeriodicEmbeddingDeviceBatch batch{};

  int initialize(std::size_t systems, cudaStream_t stream) {
    batch_size = systems;
    const std::size_t atom_count = systems * 2u;
    host_atom_offsets.resize(systems + 1u);
    host_matrix_offsets.resize(systems + 1u);
    host_shifts.resize(atom_count);
    host_matrices.resize(systems * 4u);
    host_mixed.resize(atom_count);
    host_raw.resize(atom_count);
    host_active.resize(systems);
    for (std::size_t system = 0u; system <= systems; ++system) {
      host_atom_offsets[system] = static_cast<std::int64_t>(system * 2u);
      host_matrix_offsets[system] = static_cast<std::int64_t>(system * 4u);
    }
    for (std::size_t system = 0u; system < systems; ++system) {
      const std::size_t atom = system * 2u;
      const std::size_t matrix = system * 4u;
      host_shifts[atom] = 0.03 + 0.001 * static_cast<double>(system);
      host_shifts[atom + 1u] = -0.04 + 0.0005 * static_cast<double>(system);
      host_mixed[atom] = 0.21 + 0.002 * static_cast<double>(system % 5u);
      host_mixed[atom + 1u] = -0.16;
      host_raw[atom] = -0.31;
      host_raw[atom + 1u] = 0.12 + 0.001 * static_cast<double>(system % 7u);
      host_matrices[matrix] = 0.7;
      host_matrices[matrix + 1u] = -0.13;
      host_matrices[matrix + 2u] = -0.13;
      host_matrices[matrix + 3u] = 0.9;
      host_active[system] = system % 2u == 0u ? 1u : 0u;
      if (host_active[system] == 0u) {
        host_shifts[atom] = std::numeric_limits<double>::quiet_NaN();
        host_shifts[atom + 1u] = std::numeric_limits<double>::quiet_NaN();
        host_mixed[atom] = std::numeric_limits<double>::quiet_NaN();
        host_mixed[atom + 1u] = std::numeric_limits<double>::quiet_NaN();
        host_raw[atom] = std::numeric_limits<double>::quiet_NaN();
        host_raw[atom + 1u] = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t field = 0u; field < 4u; ++field) {
          host_matrices[matrix + field] = std::numeric_limits<double>::quiet_NaN();
        }
      }
    }
    const std::vector<double> potential_seed(atom_count, kSentinel);
    const std::vector<double> energy_seed(systems, kSentinel);
    CUDA_CHECK(atom_offsets.allocate(host_atom_offsets.size()));
    CUDA_CHECK(matrix_offsets.allocate(host_matrix_offsets.size()));
    CUDA_CHECK(shifts.allocate(atom_count));
    CUDA_CHECK(matrices.allocate(host_matrices.size()));
    CUDA_CHECK(mixed.allocate(atom_count));
    CUDA_CHECK(raw.allocate(atom_count));
    CUDA_CHECK(potentials.allocate(atom_count));
    CUDA_CHECK(energies.allocate(systems));
    CUDA_CHECK(potential_scratch.allocate(atom_count));
    CUDA_CHECK(raw_scratch.allocate(atom_count));
    CUDA_CHECK(sequence_active.allocate(1u));
    CUDA_CHECK(system_errors.allocate(systems));
    CUDA_CHECK(device_error.allocate(1u));
    CUDA_CHECK(atom_offsets.upload(host_atom_offsets, stream));
    CUDA_CHECK(matrix_offsets.upload(host_matrix_offsets, stream));
    CUDA_CHECK(shifts.upload(host_shifts, stream));
    CUDA_CHECK(matrices.upload(host_matrices, stream));
    CUDA_CHECK(mixed.upload(host_mixed, stream));
    CUDA_CHECK(raw.upload(host_raw, stream));
    CUDA_CHECK(potentials.upload(potential_seed, stream));
    CUDA_CHECK(energies.upload(energy_seed, stream));
    CHECK(activity.initialize(host_active, stream) == 0);
    batch = {static_cast<std::int64_t>(systems),
             static_cast<std::int64_t>(atom_count),
             static_cast<std::int64_t>(host_matrices.size()),
             static_cast<std::int64_t>(host_atom_offsets.size()),
             static_cast<std::int64_t>(host_matrix_offsets.size()),
             static_cast<std::int64_t>(host_shifts.size()),
             static_cast<std::int64_t>(host_matrices.size()),
             kPlanToken,
             atom_offsets.get(),
             matrix_offsets.get(),
             shifts.get(),
             matrices.get(),
             kGeometryGeneration};
    return 0;
  }

  Gfn2PeriodicEmbeddingDeviceWorkspace potential_workspace() {
    return {potential_scratch.get(),
            nullptr,
            sequence_active.get(),
            static_cast<std::int64_t>(host_shifts.size()),
            1,
            kPlanToken};
  }

  Gfn2PeriodicEmbeddingDeviceWorkspace energy_workspace() {
    return {nullptr,
            raw_scratch.get(),
            sequence_active.get(),
            static_cast<std::int64_t>(host_shifts.size()),
            1,
            kPlanToken};
  }
};

int check_periodic_outputs(PeriodicFixture& fixture, cudaStream_t stream) {
  std::vector<double> potentials;
  std::vector<double> energies;
  std::vector<std::uint32_t> system_errors;
  std::vector<std::uint32_t> device_error;
  CUDA_CHECK(fixture.potentials.download(potentials, fixture.host_shifts.size(), stream));
  CUDA_CHECK(fixture.energies.download(energies, fixture.batch_size, stream));
  CUDA_CHECK(fixture.system_errors.download(system_errors, fixture.batch_size, stream));
  CUDA_CHECK(fixture.device_error.download(device_error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(device_error[0] == 0u);
  for (std::size_t system = 0u; system < fixture.batch_size; ++system) {
    const std::size_t atom = system * 2u;
    if (fixture.host_active[system] == 0u) {
      CHECK(potentials[atom] == kSentinel && potentials[atom + 1u] == kSentinel);
      CHECK(energies[system] == kSentinel && system_errors[system] == 0u);
      continue;
    }
    const double mixed_response0 =
        0.7 * fixture.host_mixed[atom] - 0.13 * fixture.host_mixed[atom + 1u];
    const double mixed_response1 =
        -0.13 * fixture.host_mixed[atom] + 0.9 * fixture.host_mixed[atom + 1u];
    const double raw_response0 = 0.7 * fixture.host_raw[atom] - 0.13 * fixture.host_raw[atom + 1u];
    const double raw_response1 = -0.13 * fixture.host_raw[atom] + 0.9 * fixture.host_raw[atom + 1u];
    const double linear = fixture.host_raw[atom] * fixture.host_shifts[atom] +
                          fixture.host_raw[atom + 1u] * fixture.host_shifts[atom + 1u];
    const double quadratic =
        fixture.host_raw[atom] * raw_response0 + fixture.host_raw[atom + 1u] * raw_response1;
    CHECK(near(potentials[atom], fixture.host_shifts[atom] + mixed_response0));
    CHECK(near(potentials[atom + 1u], fixture.host_shifts[atom + 1u] + mixed_response1));
    CHECK(near(energies[system], linear + 0.5 * quadratic));
    CHECK(system_errors[system] == 0u);
  }
  return 0;
}

int run_periodic_batch(std::size_t batch_size, cudaStream_t stream) {
  PeriodicFixture fixture;
  CHECK(fixture.initialize(batch_size, stream) == 0);
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_scc_device_errors_cuda(
      static_cast<std::int64_t>(batch_size), fixture.system_errors.get(),
      fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_scc_potential_cuda(
      fixture.batch, kGeometryGeneration, fixture.mixed.get(),
      fixture.activity.view(static_cast<std::int64_t>(batch_size)), fixture.potentials.get(),
      fixture.potential_workspace(), fixture.system_errors.get(), fixture.device_error.get(),
      stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_scc_energy_cuda(
      fixture.batch, kGeometryGeneration, fixture.raw.get(),
      fixture.activity.view(static_cast<std::int64_t>(batch_size)), fixture.energies.get(),
      fixture.energy_workspace(), fixture.system_errors.get(), fixture.device_error.get(), stream));
  return check_periodic_outputs(fixture, stream);
}

int test_periodic_all_inactive_and_stale(cudaStream_t stream) {
  PeriodicFixture fixture;
  CHECK(fixture.initialize(8u, stream) == 0);
  const std::vector<std::uint8_t> inactive(8u, 0u);
  const std::vector<std::int64_t> poison_offsets(9u, std::numeric_limits<std::int64_t>::min());
  const std::vector<double> poison_atoms(16u, std::numeric_limits<double>::quiet_NaN());
  const std::vector<double> poison_matrix(32u, std::numeric_limits<double>::quiet_NaN());
  CUDA_CHECK(fixture.activity.mask.upload(inactive, stream));
  CUDA_CHECK(fixture.atom_offsets.upload(poison_offsets, stream));
  CUDA_CHECK(fixture.matrix_offsets.upload(poison_offsets, stream));
  CUDA_CHECK(fixture.shifts.upload(poison_atoms, stream));
  CUDA_CHECK(fixture.matrices.upload(poison_matrix, stream));
  CUDA_CHECK(fixture.mixed.upload(poison_atoms, stream));
  CUDA_CHECK(fixture.raw.upload(poison_atoms, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_scc_device_errors_cuda(
      8, fixture.system_errors.get(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_scc_potential_cuda(
      fixture.batch, kGeometryGeneration + 1u, fixture.mixed.get(), fixture.activity.view(8),
      fixture.potentials.get(), fixture.potential_workspace(), fixture.system_errors.get(),
      fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_scc_energy_cuda(
      fixture.batch, kGeometryGeneration + 1u, fixture.raw.get(), fixture.activity.view(8),
      fixture.energies.get(), fixture.energy_workspace(), fixture.system_errors.get(),
      fixture.device_error.get(), stream));
  std::vector<double> potentials;
  std::vector<double> energies;
  std::vector<std::uint32_t> error;
  CUDA_CHECK(fixture.potentials.download(potentials, 16u, stream));
  CUDA_CHECK(fixture.energies.download(energies, 8u, stream));
  CUDA_CHECK(fixture.device_error.download(error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(error[0] == 0u);
  CHECK(std::all_of(potentials.begin(), potentials.end(),
                    [](double value) { return value == kSentinel; }));
  CHECK(std::all_of(energies.begin(), energies.end(),
                    [](double value) { return value == kSentinel; }));

  const std::vector<std::uint8_t> malformed_mask(8u, 0xffu);
  const std::vector<std::uint32_t> closed{0u};
  CUDA_CHECK(fixture.activity.mask.upload(malformed_mask, stream));
  CUDA_CHECK(fixture.activity.sequence.upload(closed, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_scc_device_errors_cuda(
      8, fixture.system_errors.get(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_scc_energy_cuda(
      fixture.batch, kGeometryGeneration + 1u, fixture.raw.get(), fixture.activity.view(8),
      fixture.energies.get(), fixture.energy_workspace(), fixture.system_errors.get(),
      fixture.device_error.get(), stream));
  CUDA_CHECK(fixture.device_error.download(error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(error[0] == 0u);
  return 0;
}

int test_periodic_active_stale_and_peer_failure(cudaStream_t stream) {
  PeriodicFixture fixture;
  CHECK(fixture.initialize(8u, stream) == 0);
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_scc_device_errors_cuda(
      8, fixture.system_errors.get(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_scc_potential_cuda(
      fixture.batch, kGeometryGeneration + 1u, fixture.mixed.get(), fixture.activity.view(8),
      fixture.potentials.get(), fixture.potential_workspace(), fixture.system_errors.get(),
      fixture.device_error.get(), stream));
  std::vector<double> potentials;
  std::vector<std::uint32_t> error;
  CUDA_CHECK(fixture.potentials.download(potentials, 16u, stream));
  CUDA_CHECK(fixture.device_error.download(error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(error[0] == static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kStaleGeometry));
  CHECK(std::all_of(potentials.begin(), potentials.end(),
                    [](double value) { return value == kSentinel; }));

  fixture.host_mixed[0] = std::numeric_limits<double>::quiet_NaN();
  CUDA_CHECK(fixture.mixed.upload(fixture.host_mixed, stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_scc_device_errors_cuda(
      8, fixture.system_errors.get(), fixture.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_scc_potential_cuda(
      fixture.batch, kGeometryGeneration, fixture.mixed.get(), fixture.activity.view(8),
      fixture.potentials.get(), fixture.potential_workspace(), fixture.system_errors.get(),
      fixture.device_error.get(), stream));
  std::vector<std::uint32_t> system_errors;
  CUDA_CHECK(fixture.potentials.download(potentials, 16u, stream));
  CUDA_CHECK(fixture.system_errors.download(system_errors, 8u, stream));
  CUDA_CHECK(fixture.device_error.download(error, 1u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(error[0] == 0u);
  CHECK(potentials[0] == kSentinel && potentials[1] == kSentinel);
  CHECK(system_errors[0] ==
        static_cast<std::uint32_t>(Gfn2PeriodicEmbeddingDeviceError::kNonfiniteMixedCharge));
  CHECK(potentials[4] != kSentinel && system_errors[2] == 0u);
  return 0;
}

int test_graph_replay(cudaStream_t stream) {
  D4Fixture d4;
  PeriodicFixture periodic;
  CHECK(d4.initialize(1u, stream) == 0);
  CHECK(periodic.initialize(1u, stream) == 0);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_d4_device_errors_cuda(1, d4.system_errors.get(),
                                                                    d4.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_potential_cuda(
      d4.batch, d4.parameters, d4.cache, kGeometryGeneration, d4.mixed.get(), d4.activity.view(1),
      d4.potentials.get(), d4.potential_workspace(), d4.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_d4_scc_energy_cuda(
      d4.batch, d4.parameters, d4.cache, kGeometryGeneration, d4.raw.get(), d4.activity.view(1),
      d4.energies.get(), d4.energy_workspace(), d4.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::reset_gfn2_periodic_embedding_scc_device_errors_cuda(
      1, periodic.system_errors.get(), periodic.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_scc_potential_cuda(
      periodic.batch, kGeometryGeneration, periodic.mixed.get(), periodic.activity.view(1),
      periodic.potentials.get(), periodic.potential_workspace(), periodic.system_errors.get(),
      periodic.device_error.get(), stream));
  CUDA_CHECK(gpuxtb::detail::cuda::evaluate_gfn2_periodic_embedding_scc_energy_cuda(
      periodic.batch, kGeometryGeneration, periodic.raw.get(), periodic.activity.view(1),
      periodic.energies.get(), periodic.energy_workspace(), periodic.system_errors.get(),
      periodic.device_error.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<double> first_d4;
  std::vector<double> first_d4_potential;
  std::vector<double> first_periodic;
  std::vector<double> first_periodic_potential;
  CUDA_CHECK(d4.energies.download(first_d4, 1u, stream));
  CUDA_CHECK(d4.potentials.download(first_d4_potential, 2u, stream));
  CUDA_CHECK(periodic.energies.download(first_periodic, 1u, stream));
  CUDA_CHECK(periodic.potentials.download(first_periodic_potential, 2u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  d4.host_mixed[0] += 0.19;
  d4.host_raw[1] -= 0.11;
  periodic.host_mixed[0] -= 0.23;
  periodic.host_raw[1] += 0.17;
  CUDA_CHECK(d4.mixed.upload(d4.host_mixed, stream));
  CUDA_CHECK(d4.raw.upload(d4.host_raw, stream));
  CUDA_CHECK(periodic.mixed.upload(periodic.host_mixed, stream));
  CUDA_CHECK(periodic.raw.upload(periodic.host_raw, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  std::vector<double> second_d4;
  std::vector<double> second_d4_potential;
  std::vector<double> second_periodic;
  std::vector<double> second_periodic_potential;
  CUDA_CHECK(d4.energies.download(second_d4, 1u, stream));
  CUDA_CHECK(d4.potentials.download(second_d4_potential, 2u, stream));
  CUDA_CHECK(periodic.energies.download(second_periodic, 1u, stream));
  CUDA_CHECK(periodic.potentials.download(second_periodic_potential, 2u, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(!near(first_d4[0], second_d4[0], 1.0e-14));
  CHECK(!near(first_d4_potential[0], second_d4_potential[0], 1.0e-14));
  CHECK(!near(first_periodic[0], second_periodic[0], 1.0e-14));
  CHECK(!near(first_periodic_potential[0], second_periodic_potential[0], 1.0e-14));
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return 77;
  }
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    CHECK(run_d4_batch(batch_size, stream) == 0);
    CHECK(run_periodic_batch(batch_size, stream) == 0);
  }
  CHECK(test_d4_all_inactive_and_stale(stream) == 0);
  CHECK(test_d4_active_stale_and_peer_failure(stream) == 0);
  CHECK(test_d4_atomic_number_reorder_is_plan_failure(stream) == 0);
  CHECK(test_d4_inactive_offsets_are_not_consumed(stream) == 0);
  CHECK(test_periodic_all_inactive_and_stale(stream) == 0);
  CHECK(test_periodic_active_stale_and_peer_failure(stream) == 0);
  CHECK(test_graph_replay(stream) == 0);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
