#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include "gpuxtb/gpuxtb.h"

#define CHECK(condition)                                                                         \
  do {                                                                                           \
    if (!(condition)) {                                                                          \
      std::fprintf(stderr, "public unrestricted CUDA check failed at %s:%d: %s; %s\n", __FILE__, \
                   __LINE__, #condition, gpuxtb_get_last_error());                               \
      return __LINE__;                                                                           \
    }                                                                                            \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

constexpr std::uint32_t kAllProperties =
    GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES | GPUXTB_COMPUTE_ATOMIC_CHARGES;

template <typename T>
gpuxtb_const_buffer_t host_input(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), GPUXTB_MEMORY_HOST,
          0u};
}

template <typename T>
gpuxtb_buffer_t host_output(std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), GPUXTB_MEMORY_HOST,
          0u};
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream) {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
      data_ = nullptr;
    }
    count_ = values.size();
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T));
    return status == cudaSuccess ? cudaMemcpyAsync(data_, values.data(), count_ * sizeof(T),
                                                   cudaMemcpyHostToDevice, stream)
                                 : status;
  }

  const T* get() const noexcept { return data_; }
  std::size_t size() const noexcept { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

struct ContextOwner {
  gpuxtb_context_t* context = nullptr;
  ~ContextOwner() { gpuxtb_context_destroy(context); }
};

struct BatchOwner {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  gpuxtb_batch_t batch{};

  void bind_host(bool abi_v1 = false) noexcept {
    (void)gpuxtb_batch_init(&batch, sizeof(batch));
    batch.batch_size = static_cast<std::int64_t>(molecular_charges.size());
    batch.total_atoms = static_cast<std::int64_t>(atomic_numbers.size());
    batch.atom_offsets = host_input(atom_offsets);
    batch.atomic_numbers = host_input(atomic_numbers);
    batch.positions = host_input(positions);
    batch.molecular_charges = host_input(molecular_charges);
    batch.unpaired_electrons = host_input(unpaired_electrons);
    batch.spin_channels = host_input(spin_channels);
    if (abi_v1) batch.struct_size = GPUXTB_BATCH_V1_SIZE;
  }

  void bind_device_spin(const DeviceBuffer<std::int32_t>& spin) noexcept {
    bind_host();
    batch.spin_channels = {spin.get(), spin.size() * sizeof(std::int32_t),
                           GPUXTB_MEMORY_CUDA_DEVICE, 0u};
  }
};

struct ResultOwner {
  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> charges;
  std::vector<std::int32_t> iterations;
  std::vector<std::uint8_t> converged;
  std::vector<std::int32_t> statuses;
  gpuxtb_batch_result_t result{};

  void bind(const BatchOwner& batch, std::uint32_t flags) {
    const std::size_t systems = batch.molecular_charges.size();
    const std::size_t atoms = batch.atomic_numbers.size();
    energies.assign((flags & GPUXTB_COMPUTE_ENERGY) != 0u ? systems : 0u,
                    std::numeric_limits<double>::quiet_NaN());
    forces.assign((flags & GPUXTB_COMPUTE_FORCES) != 0u ? atoms * 3u : 0u,
                  std::numeric_limits<double>::quiet_NaN());
    charges.assign((flags & GPUXTB_COMPUTE_ATOMIC_CHARGES) != 0u ? atoms : 0u,
                   std::numeric_limits<double>::quiet_NaN());
    iterations.assign(systems, -1);
    converged.assign(systems, 0u);
    statuses.assign(systems, GPUXTB_STATUS_INTERNAL_ERROR);
    (void)gpuxtb_batch_result_init(&result, sizeof(result));
    result.energies = host_output(energies);
    result.forces = host_output(forces);
    result.atomic_charges = host_output(charges);
    result.scc_iterations = host_output(iterations);
    result.scc_converged = host_output(converged);
    result.per_system_status = host_output(statuses);
  }
};

gpuxtb_compute_options_t options(std::uint32_t flags) noexcept {
  gpuxtb_compute_options_t value{};
  (void)gpuxtb_compute_options_init(&value, sizeof(value));
  value.model = GPUXTB_MODEL_GFN2_XTB;
  value.flags = flags;
  value.max_scc_iterations = 64;
  value.charge_tolerance = 1.0e-8;
  value.energy_tolerance = 1.0e-8;
  value.electronic_temperature = 0.0;
  return value;
}

bool near(double actual, double expected, double tolerance = 2.0e-10) {
  return std::abs(actual - expected) <=
         tolerance * (1.0 + std::max(std::abs(actual), std::abs(expected)));
}

int compute(gpuxtb_context_t* context, BatchOwner& batch, std::uint32_t flags,
            ResultOwner& result) {
  result.bind(batch, flags);
  const gpuxtb_compute_options_t request = options(flags);
  CHECK(gpuxtb_compute(context, &batch.batch, &request, &result.result) == GPUXTB_STATUS_SUCCESS);
  for (std::size_t system = 0; system < result.statuses.size(); ++system) {
    CHECK(result.statuses[system] == GPUXTB_STATUS_SUCCESS);
    CHECK(result.converged[system] == 1u);
    CHECK(result.iterations[system] > 0);
    CHECK(result.iterations[system] <= request.max_scc_iterations);
  }
  CHECK(std::all_of(result.energies.begin(), result.energies.end(),
                    [](double value) { return std::isfinite(value); }));
  CHECK(std::all_of(result.forces.begin(), result.forces.end(),
                    [](double value) { return std::isfinite(value); }));
  CHECK(std::all_of(result.charges.begin(), result.charges.end(),
                    [](double value) { return std::isfinite(value); }));
  return 0;
}

enum class SystemKind : std::uint8_t { kOhRadical, kH2Restricted, kH2UnrestrictedSinglet };

void append_system(BatchOwner& batch, SystemKind kind) {
  if (batch.atom_offsets.empty()) batch.atom_offsets.push_back(0);
  if (kind == SystemKind::kOhRadical) {
    batch.atomic_numbers.insert(batch.atomic_numbers.end(), {8, 1});
    batch.positions.insert(batch.positions.end(), {0.0, 0.0, 0.0, 1.8, 0.0, 0.0});
    batch.unpaired_electrons.push_back(1);
    batch.spin_channels.push_back(2);
  } else {
    batch.atomic_numbers.insert(batch.atomic_numbers.end(), {1, 1});
    batch.positions.insert(batch.positions.end(), {-0.71, 0.0, 0.0, 0.71, 0.0, 0.0});
    batch.unpaired_electrons.push_back(0);
    batch.spin_channels.push_back(kind == SystemKind::kH2Restricted ? 1 : 2);
  }
  batch.molecular_charges.push_back(0.0);
  batch.atom_offsets.push_back(static_cast<std::int64_t>(batch.atomic_numbers.size()));
}

BatchOwner make_batch(std::size_t batch_size) {
  BatchOwner batch;
  constexpr SystemKind kinds[] = {SystemKind::kOhRadical, SystemKind::kH2Restricted,
                                  SystemKind::kH2UnrestrictedSinglet};
  for (std::size_t system = 0; system < batch_size; ++system) {
    append_system(batch, kinds[system % 3u]);
  }
  batch.bind_host();
  return batch;
}

BatchOwner make_single(SystemKind kind) {
  BatchOwner batch;
  append_system(batch, kind);
  batch.bind_host();
  return batch;
}

int compare_system(const ResultOwner& batch, std::size_t system, const ResultOwner& reference) {
  CHECK(reference.energies.size() == 1u);
  CHECK(batch.statuses[system] == reference.statuses[0]);
  CHECK(batch.converged[system] == reference.converged[0]);
  CHECK(batch.iterations[system] == reference.iterations[0]);
  CHECK(near(batch.energies[system], reference.energies[0]));
  const std::size_t atom_begin = 2u * system;
  for (std::size_t local = 0; local < 2u; ++local) {
    CHECK(near(batch.charges[atom_begin + local], reference.charges[local]));
    for (std::size_t axis = 0; axis < 3u; ++axis) {
      CHECK(near(batch.forces[(atom_begin + local) * 3u + axis],
                 reference.forces[local * 3u + axis]));
    }
  }
  return 0;
}

int test_mixed_batch_sizes_and_device_spin_descriptor(gpuxtb_context_t* context,
                                                      cudaStream_t stream) {
  ResultOwner references[3];
  for (std::size_t kind = 0; kind < 3u; ++kind) {
    BatchOwner single = make_single(static_cast<SystemKind>(kind));
    CHECK(compute(context, single, kAllProperties, references[kind]) == 0);
  }

  for (const std::size_t batch_size : {1u, 8u, 32u, 128u}) {
    BatchOwner batch = make_batch(batch_size);
    ResultOwner host_result;
    CHECK(compute(context, batch, kAllProperties, host_result) == 0);
    for (std::size_t system = 0; system < batch_size; ++system) {
      CHECK(compare_system(host_result, system, references[system % 3u]) == 0);
    }

    if (batch_size == 8u) {
      DeviceBuffer<std::int32_t> spin;
      CUDA_CHECK(spin.upload(batch.spin_channels, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      batch.bind_device_spin(spin);
      ResultOwner device_spin_result;
      CHECK(compute(context, batch, kAllProperties, device_spin_result) == 0);
      for (std::size_t system = 0; system < batch_size; ++system) {
        CHECK(compare_system(device_spin_result, system, references[system % 3u]) == 0);
      }
    }
  }
  return 0;
}

int test_charge_publication_without_forces(gpuxtb_context_t* context) {
  BatchOwner batch = make_batch(8u);
  ResultOwner reference;
  CHECK(compute(context, batch, kAllProperties, reference) == 0);

  /* Requests without forces publish the SCC-owned physical charge projection,
   * rather than the stationary force buffers. Exercise both supported masks
   * so mixed restricted/unrestricted offsets stay covered on that path. */
  for (const std::uint32_t flags :
       {static_cast<std::uint32_t>(GPUXTB_COMPUTE_ATOMIC_CHARGES),
        static_cast<std::uint32_t>(GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_ATOMIC_CHARGES)}) {
    ResultOwner actual;
    CHECK(compute(context, batch, flags, actual) == 0);
    CHECK(actual.forces.empty());
    CHECK(actual.iterations == reference.iterations);
    CHECK(actual.converged == reference.converged);
    CHECK(actual.statuses == reference.statuses);
    CHECK(actual.charges.size() == reference.charges.size());
    for (std::size_t index = 0; index < actual.charges.size(); ++index) {
      CHECK(near(actual.charges[index], reference.charges[index]));
    }
    if ((flags & GPUXTB_COMPUTE_ENERGY) != 0u) {
      CHECK(actual.energies.size() == reference.energies.size());
      for (std::size_t system = 0; system < actual.energies.size(); ++system) {
        CHECK(near(actual.energies[system], reference.energies[system]));
      }
    } else {
      CHECK(actual.energies.empty());
    }
  }
  return 0;
}

int test_oh_force_finite_difference(gpuxtb_context_t* context) {
  BatchOwner radical = make_single(SystemKind::kOhRadical);
  ResultOwner analytic;
  CHECK(compute(context, radical, kAllProperties, analytic) == 0);
  constexpr double step = 1.0e-4;
  const double analytic_force = analytic.forces[3];

  radical.positions[3] += step;
  radical.bind_host();
  ResultOwner plus;
  CHECK(compute(context, radical, GPUXTB_COMPUTE_ENERGY, plus) == 0);
  radical.positions[3] -= 2.0 * step;
  radical.bind_host();
  ResultOwner minus;
  CHECK(compute(context, radical, GPUXTB_COMPUTE_ENERGY, minus) == 0);
  const double finite_difference = -(plus.energies[0] - minus.energies[0]) / (2.0 * step);
  CHECK(near(analytic_force, finite_difference, 2.0e-5));
  for (std::size_t axis = 0; axis < 3u; ++axis) {
    CHECK(std::abs(analytic.forces[axis] + analytic.forces[3u + axis]) < 2.0e-10);
  }
  return 0;
}

int test_abi_v1_restricted_fallback(gpuxtb_context_t* context) {
  BatchOwner explicit_v2 = make_single(SystemKind::kH2Restricted);
  ResultOwner expected;
  CHECK(compute(context, explicit_v2, kAllProperties, expected) == 0);

  BatchOwner legacy = make_single(SystemKind::kH2Restricted);
  legacy.bind_host(true);
  /* The excluded suffix is deliberately invalid and must not be inspected. */
  legacy.batch.spin_channels = {reinterpret_cast<const void*>(std::uintptr_t{1}),
                                std::numeric_limits<std::size_t>::max(), GPUXTB_MEMORY_ROCM_DEVICE,
                                UINT32_MAX};
  ResultOwner actual;
  CHECK(compute(context, legacy, kAllProperties, actual) == 0);
  CHECK(actual.iterations == expected.iterations);
  CHECK(actual.converged == expected.converged);
  CHECK(actual.statuses == expected.statuses);
  for (std::size_t index = 0; index < actual.energies.size(); ++index) {
    CHECK(near(actual.energies[index], expected.energies[index]));
  }
  for (std::size_t index = 0; index < actual.forces.size(); ++index) {
    CHECK(near(actual.forces[index], expected.forces[index]));
  }
  for (std::size_t index = 0; index < actual.charges.size(); ++index) {
    CHECK(near(actual.charges[index], expected.charges[index]));
  }
  return 0;
}

}  // namespace

int main() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  gpuxtb_context_options_t context_options{};
  CHECK(gpuxtb_context_options_init(&context_options, sizeof(context_options)) ==
        GPUXTB_STATUS_SUCCESS);
  context_options.backend = GPUXTB_BACKEND_CUDA;
  context_options.device_id = device;
  context_options.stream = reinterpret_cast<void*>(stream);
  ContextOwner owner;
  CHECK(gpuxtb_context_create(&context_options, &owner.context) == GPUXTB_STATUS_SUCCESS);

  if (const int status = test_mixed_batch_sizes_and_device_spin_descriptor(owner.context, stream);
      status != 0) {
    return status;
  }
  if (const int status = test_charge_publication_without_forces(owner.context); status != 0) {
    return status;
  }
  if (const int status = test_oh_force_finite_difference(owner.context); status != 0) {
    return status;
  }
  if (const int status = test_abi_v1_restricted_fallback(owner.context); status != 0) {
    return status;
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
