#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_scc_bridge.cuh"

namespace {

using gpuxtb::detail::Gfn2PairMapKind;
using gpuxtb::detail::Gfn2PlanMemorySpace;
using gpuxtb::detail::Gfn2RaggedTopologyView;
using gpuxtb::detail::cuda::collect_gfn2_scc_shell_scalar_potential_cuda;
using gpuxtb::detail::cuda::Gfn2SccBridgeDeviceBatch;
using gpuxtb::detail::cuda::Gfn2SccBridgeDeviceError;
using gpuxtb::detail::cuda::Gfn2SccBridgeDeviceOutput;
using gpuxtb::detail::cuda::Gfn2SccBridgeDevicePotentialFields;
using gpuxtb::detail::cuda::Gfn2SccBridgeDeviceStageInput;
using gpuxtb::detail::cuda::Gfn2SccBridgeDeviceWorkspace;
using gpuxtb::detail::cuda::reset_gfn2_scc_bridge_stage_cuda;

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

constexpr std::uint64_t kPlanToken = 0x4252494447453835ULL;
constexpr double kSentinel = -0x1.3bcp+19;

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
  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream = nullptr) {
    if (values.size() != elements_) {
      return cudaErrorInvalidValue;
    }
    return elements_ == 0u ? cudaSuccess
                           : cudaMemcpyAsync(pointer_, values.data(), elements_ * sizeof(T),
                                             cudaMemcpyHostToDevice, stream);
  }
  cudaError_t upload_one(T value, std::size_t index = 0u, cudaStream_t stream = nullptr) {
    if (index >= elements_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(pointer_ + index, &value, sizeof(T), cudaMemcpyHostToDevice, stream);
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
      std::fprintf(stderr, "CUDA fixture allocation failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }

  T* pointer_ = nullptr;
  std::size_t elements_ = 0u;
};

struct HostCase {
  std::int64_t batch_size = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> shell_offsets;
  std::vector<std::int64_t> orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> shell_orbital_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::int64_t> orbital_to_shell;
  std::vector<std::int64_t> orbital_to_atom;
  std::vector<std::int64_t> qsh_offsets;
  std::vector<std::int64_t> qat_offsets;
  std::vector<double> shell_field;
  std::vector<double> atom_field;
  std::vector<double> expected;
  std::vector<std::uint8_t> requested_active;
};

/* Every system has two atoms and s/p/s/d shells: 10 spherical AOs total. */
HostCase make_case(std::size_t batch_size) {
  HostCase host;
  host.batch_size = static_cast<std::int64_t>(batch_size);
  host.atom_offsets.reserve(batch_size + 1u);
  host.shell_offsets.reserve(batch_size + 1u);
  host.orbital_offsets.reserve(batch_size + 1u);
  host.matrix_offsets.reserve(batch_size + 1u);
  host.atom_shell_offsets.reserve(batch_size * 2u + 1u);
  host.shell_orbital_offsets.reserve(batch_size * 4u + 1u);
  host.qsh_offsets.reserve(batch_size + 1u);
  host.qat_offsets.reserve(batch_size + 1u);
  host.atom_offsets.push_back(0);
  host.shell_offsets.push_back(0);
  host.orbital_offsets.push_back(0);
  host.matrix_offsets.push_back(0);
  host.atom_shell_offsets.push_back(0);
  host.shell_orbital_offsets.push_back(0);
  constexpr std::array<std::int64_t, 4> shell_aos{{1, 3, 1, 5}};
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::int64_t atom_base = static_cast<std::int64_t>(system * 2u);
    const std::int64_t shell_base = static_cast<std::int64_t>(system * 4u);
    host.atom_offsets.push_back(atom_base + 2);
    host.shell_offsets.push_back(shell_base + 4);
    host.orbital_offsets.push_back(static_cast<std::int64_t>((system + 1u) * 10u));
    host.matrix_offsets.push_back(static_cast<std::int64_t>((system + 1u) * 100u));
    host.atom_shell_offsets.push_back(shell_base + 2);
    host.atom_shell_offsets.push_back(shell_base + 4);
    for (std::size_t local_shell = 0; local_shell < shell_aos.size(); ++local_shell) {
      const std::int64_t atom = atom_base + (local_shell < 2u ? 0 : 1);
      const std::int64_t shell = shell_base + static_cast<std::int64_t>(local_shell);
      host.shell_to_atom.push_back(atom);
      const std::int64_t begin = host.shell_orbital_offsets.back();
      const std::int64_t end = begin + shell_aos[local_shell];
      host.shell_orbital_offsets.push_back(end);
      for (std::int64_t orbital = begin; orbital < end; ++orbital) {
        (void)orbital;
        host.orbital_to_shell.push_back(shell);
        host.orbital_to_atom.push_back(atom);
      }
    }
  }

  /* Nonzero bases prove field and topology offsets are not conflated. */
  constexpr std::int64_t qsh_base = 3;
  constexpr std::int64_t qat_base = 2;
  for (std::size_t system = 0; system <= batch_size; ++system) {
    host.qsh_offsets.push_back(qsh_base + static_cast<std::int64_t>(system * 4u));
    host.qat_offsets.push_back(qat_base + static_cast<std::int64_t>(system * 2u));
  }
  host.shell_field.assign(static_cast<std::size_t>(qsh_base) + batch_size * 4u + 2u,
                          std::numeric_limits<double>::quiet_NaN());
  host.atom_field.assign(static_cast<std::size_t>(qat_base) + batch_size * 2u + 3u,
                         std::numeric_limits<double>::quiet_NaN());
  host.expected.resize(batch_size * 4u);
  host.requested_active.assign(batch_size, 1u);
  for (std::size_t system = 0; system < batch_size; ++system) {
    for (std::size_t local_atom = 0; local_atom < 2u; ++local_atom) {
      host.atom_field[static_cast<std::size_t>(host.qat_offsets[system]) + local_atom] =
          -0.125 * static_cast<double>(1u + system * 2u + local_atom);
    }
    for (std::size_t local_shell = 0; local_shell < 4u; ++local_shell) {
      const std::size_t shell = system * 4u + local_shell;
      const std::int64_t atom = host.shell_to_atom[shell];
      const std::size_t local_atom = static_cast<std::size_t>(atom - host.atom_offsets[system]);
      const double vsh = 0.03125 * static_cast<double>(1u + shell * 3u);
      const double vat =
          host.atom_field[static_cast<std::size_t>(host.qat_offsets[system]) + local_atom];
      host.shell_field[static_cast<std::size_t>(host.qsh_offsets[system]) + local_shell] = vsh;
      host.expected[shell] = vsh + vat;
    }
  }
  return host;
}

struct DeviceCase {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> shell_offsets;
  DeviceBuffer<std::int64_t> orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> shell_orbital_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<std::int64_t> orbital_to_shell;
  DeviceBuffer<std::int64_t> orbital_to_atom;
  DeviceBuffer<std::int64_t> qsh_offsets;
  DeviceBuffer<std::int64_t> qat_offsets;
  DeviceBuffer<double> shell_field;
  DeviceBuffer<double> atom_field;
  DeviceBuffer<std::uint8_t> requested_active;
  DeviceBuffer<std::uint32_t> system_errors;
  DeviceBuffer<std::uint32_t> upstream_error;
  DeviceBuffer<std::uint32_t> upstream_sequence_active;
  DeviceBuffer<double> output_shell;
  DeviceBuffer<std::uint8_t> output_active;
  DeviceBuffer<std::uint32_t> output_plan_error;
  DeviceBuffer<double> shell_scratch;
  DeviceBuffer<std::uint32_t> sequence_active;

  Gfn2SccBridgeDeviceBatch batch;
  Gfn2SccBridgeDevicePotentialFields potential;
  Gfn2SccBridgeDeviceStageInput stage;
  Gfn2SccBridgeDeviceOutput output;
  Gfn2SccBridgeDeviceWorkspace workspace;

  explicit DeviceCase(const HostCase& host)
      : atom_offsets(host.atom_offsets.size()),
        shell_offsets(host.shell_offsets.size()),
        orbital_offsets(host.orbital_offsets.size()),
        matrix_offsets(host.matrix_offsets.size()),
        atom_shell_offsets(host.atom_shell_offsets.size()),
        shell_orbital_offsets(host.shell_orbital_offsets.size()),
        shell_to_atom(host.shell_to_atom.size()),
        orbital_to_shell(host.orbital_to_shell.size()),
        orbital_to_atom(host.orbital_to_atom.size()),
        qsh_offsets(host.qsh_offsets.size()),
        qat_offsets(host.qat_offsets.size()),
        shell_field(host.shell_field.size()),
        atom_field(host.atom_field.size()),
        requested_active(host.requested_active.size()),
        system_errors(host.requested_active.size()),
        upstream_error(1u),
        upstream_sequence_active(1u),
        output_shell(host.expected.size()),
        output_active(host.requested_active.size()),
        output_plan_error(1u),
        shell_scratch(host.expected.size()),
        sequence_active(1u) {
#define UPLOAD(name) require(name.upload(host.name));
    UPLOAD(atom_offsets)
    UPLOAD(shell_offsets)
    UPLOAD(orbital_offsets)
    UPLOAD(matrix_offsets)
    UPLOAD(atom_shell_offsets)
    UPLOAD(shell_orbital_offsets)
    UPLOAD(shell_to_atom)
    UPLOAD(orbital_to_shell)
    UPLOAD(orbital_to_atom)
    UPLOAD(qsh_offsets)
    UPLOAD(qat_offsets)
    UPLOAD(shell_field)
    UPLOAD(atom_field)
    UPLOAD(requested_active)
#undef UPLOAD
    bind(host);
    require(reset_status());
    require(fill_output(kSentinel));
    require(cudaDeviceSynchronize());
  }

  void bind(const HostCase& host) {
    Gfn2RaggedTopologyView topology;
    topology.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    topology.pair_map_kind = Gfn2PairMapKind::kNone;
    topology.plan_token = kPlanToken;
    topology.batch_size = host.batch_size;
    topology.total_atoms = static_cast<std::int64_t>(host.atom_field.size() - 5u);
    topology.total_shells = static_cast<std::int64_t>(host.expected.size());
    topology.total_orbitals = static_cast<std::int64_t>(host.orbital_to_shell.size());
    topology.total_matrix_elements = static_cast<std::int64_t>(host.matrix_offsets.back());
    topology.atom_offset_count = static_cast<std::int64_t>(atom_offsets.size());
    topology.batch_shell_offset_count = static_cast<std::int64_t>(shell_offsets.size());
    topology.batch_orbital_offset_count = static_cast<std::int64_t>(orbital_offsets.size());
    topology.matrix_offset_count = static_cast<std::int64_t>(matrix_offsets.size());
    topology.atom_shell_offset_count = static_cast<std::int64_t>(atom_shell_offsets.size());
    topology.shell_orbital_offset_count = static_cast<std::int64_t>(shell_orbital_offsets.size());
    topology.shell_to_atom_count = static_cast<std::int64_t>(shell_to_atom.size());
    topology.orbital_to_shell_count = static_cast<std::int64_t>(orbital_to_shell.size());
    topology.orbital_to_atom_count = static_cast<std::int64_t>(orbital_to_atom.size());
    topology.atom_offsets = atom_offsets.get();
    topology.batch_shell_offsets = shell_offsets.get();
    topology.batch_orbital_offsets = orbital_offsets.get();
    topology.matrix_offsets = matrix_offsets.get();
    topology.atom_shell_offsets = atom_shell_offsets.get();
    topology.shell_orbital_offsets = shell_orbital_offsets.get();
    topology.shell_to_atom = shell_to_atom.get();
    topology.orbital_to_shell = orbital_to_shell.get();
    topology.orbital_to_atom = orbital_to_atom.get();
    batch = {topology, static_cast<std::int64_t>(qsh_offsets.size()),
             static_cast<std::int64_t>(qat_offsets.size()), qsh_offsets.get(), qat_offsets.get()};
    potential = {shell_field.get(), static_cast<std::int64_t>(shell_field.size()), atom_field.get(),
                 static_cast<std::int64_t>(atom_field.size()), kPlanToken};
    stage = {requested_active.get(),
             host.batch_size,
             system_errors.get(),
             host.batch_size,
             upstream_error.get(),
             1,
             upstream_sequence_active.get(),
             1,
             0u,
             kPlanToken};
    output = {output_shell.get(),
              static_cast<std::int64_t>(output_shell.size()),
              output_active.get(),
              host.batch_size,
              output_plan_error.get(),
              1,
              kPlanToken};
    workspace = {shell_scratch.get(), static_cast<std::int64_t>(shell_scratch.size()),
                 sequence_active.get(), 1, kPlanToken};
  }

  cudaError_t reset_status(cudaStream_t stream = nullptr) {
    const std::vector<std::uint32_t> zeros(system_errors.size(), 0u);
    cudaError_t status = system_errors.upload(zeros, stream);
    if (status == cudaSuccess) {
      status = upstream_error.upload_one(0u, 0u, stream);
    }
    if (status == cudaSuccess) {
      status = upstream_sequence_active.upload_one(1u, 0u, stream);
    }
    return status == cudaSuccess
               ? reset_gfn2_scc_bridge_stage_cuda(static_cast<std::int64_t>(output_active.size()),
                                                  output_active.get(), output_plan_error.get(),
                                                  sequence_active.get(), stream)
               : status;
  }
  cudaError_t fill_output(double value, cudaStream_t stream = nullptr) {
    return output_shell.upload(std::vector<double>(output_shell.size(), value), stream);
  }

 private:
  static void require(cudaError_t status) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA fixture setup failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }
};

int launch(DeviceCase& device, cudaStream_t stream = nullptr) {
  CHECK(collect_gfn2_scc_shell_scalar_potential_cuda(device.batch, device.potential, device.stage,
                                                     device.output, device.workspace,
                                                     stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  return 0;
}

int test_batch_parity_real_spd_and_custom_stream() {
  for (std::size_t batch_size : std::array<std::size_t, 4>{{1u, 8u, 32u, 128u}}) {
    HostCase host = make_case(batch_size);
    DeviceCase device(host);
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(launch(device, stream) == 0);
    std::vector<double> shell;
    std::vector<std::uint8_t> active;
    std::vector<std::uint32_t> errors;
    std::uint32_t plan = 99u;
    std::vector<std::uint32_t> plan_values;
    CHECK(device.output_shell.download(shell, stream) == cudaSuccess);
    CHECK(device.output_active.download(active, stream) == cudaSuccess);
    CHECK(device.system_errors.download(errors, stream) == cudaSuccess);
    CHECK(device.output_plan_error.download(plan_values, stream) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    plan = plan_values[0];
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    CHECK(shell == host.expected);
    CHECK(
        std::all_of(active.begin(), active.end(), [](std::uint8_t value) { return value == 1u; }));
    CHECK(
        std::all_of(errors.begin(), errors.end(), [](std::uint32_t value) { return value == 0u; }));
    CHECK(plan == 0u);
  }
  return 0;
}

int test_inactive_poison_and_upstream_peer_isolation() {
  HostCase host = make_case(8u);
  host.requested_active[2] = 0u;
  for (std::size_t shell = 8u; shell < 12u; ++shell) {
    host.shell_field[static_cast<std::size_t>(host.qsh_offsets[2]) + shell - 8u] =
        std::numeric_limits<double>::quiet_NaN();
  }
  DeviceCase device(host);
  CHECK(device.system_errors.upload_one(37u, 5u) == cudaSuccess);
  CHECK(device.upstream_error.upload_one(37u) == cudaSuccess);
  device.stage.peer_error_mask = std::uint64_t{1} << 37u;
  CHECK(device.shell_field.upload(host.shell_field) == cudaSuccess);
  CHECK(launch(device) == 0);

  std::vector<double> shell;
  std::vector<std::uint8_t> active;
  std::vector<std::uint32_t> errors;
  std::vector<std::uint32_t> plan;
  CHECK(device.output_shell.download(shell) == cudaSuccess);
  CHECK(device.output_active.download(active) == cudaSuccess);
  CHECK(device.system_errors.download(errors) == cudaSuccess);
  CHECK(device.output_plan_error.download(plan) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(active[2] == 0u && active[5] == 0u && errors[2] == 0u && errors[5] == 37u);
  CHECK(plan[0] == 0u);
  for (std::size_t system = 0; system < 8u; ++system) {
    for (std::size_t local_shell = 0; local_shell < 4u; ++local_shell) {
      const std::size_t shell_index = system * 4u + local_shell;
      CHECK(shell[shell_index] ==
            (system == 2u || system == 5u ? kSentinel : host.expected[shell_index]));
    }
  }

  /* A later upstream plan failure must not be hidden by the first numerical
   * diagnostic that remains in upstream_device_error. */
  CHECK(reset_gfn2_scc_bridge_stage_cuda(device.batch.topology.batch_size,
                                         device.output_active.get(), device.output_plan_error.get(),
                                         device.sequence_active.get()) == cudaSuccess);
  CHECK(device.upstream_sequence_active.upload_one(0u) == cudaSuccess);
  CHECK(device.fill_output(kSentinel) == cudaSuccess);
  CHECK(launch(device) == 0);
  CHECK(device.output_active.download(active) == cudaSuccess);
  CHECK(device.output_plan_error.download(plan) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(plan[0] == static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kUpstreamPlanFailure));
  CHECK(std::all_of(active.begin(), active.end(), [](std::uint8_t value) { return value == 0u; }));
  return 0;
}

int test_numerical_failures_and_active_validation() {
  HostCase host = make_case(8u);
  host.shell_field[static_cast<std::size_t>(host.qsh_offsets[1])] =
      std::numeric_limits<double>::quiet_NaN();
  host.atom_field[static_cast<std::size_t>(host.qat_offsets[3])] =
      std::numeric_limits<double>::infinity();
  host.requested_active[6] = 2u;
  const double large = 0.75 * std::numeric_limits<double>::max();
  host.shell_field[static_cast<std::size_t>(host.qsh_offsets[4])] = large;
  host.atom_field[static_cast<std::size_t>(host.qat_offsets[4])] = large;
  DeviceCase device(host);
  CHECK(launch(device) == 0);

  std::vector<double> shell;
  std::vector<std::uint8_t> active;
  std::vector<std::uint32_t> errors;
  std::vector<std::uint32_t> plan;
  CHECK(device.output_shell.download(shell) == cudaSuccess);
  CHECK(device.output_active.download(active) == cudaSuccess);
  CHECK(device.system_errors.download(errors) == cudaSuccess);
  CHECK(device.output_plan_error.download(plan) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(errors[1] ==
        static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kNonfiniteShellPotential));
  CHECK(errors[3] ==
        static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kNonfiniteAtomicPotential));
  CHECK(errors[4] ==
        static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kNonfiniteScalarPotentialArithmetic));
  CHECK(errors[6] == static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kInvalidActiveMask));
  CHECK(plan[0] == 0u);
  for (std::size_t system = 0; system < 8u; ++system) {
    const bool failed = system == 1u || system == 3u || system == 4u || system == 6u;
    CHECK(active[system] == (failed ? 0u : 1u));
    for (std::size_t local_shell = 0; local_shell < 4u; ++local_shell) {
      const std::size_t index = system * 4u + local_shell;
      CHECK(shell[index] == (failed ? kSentinel : host.expected[index]));
    }
  }
  return 0;
}

int test_plan_fail_closed_and_sticky_classification() {
  HostCase host = make_case(8u);
  DeviceCase device(host);
  CHECK(device.upstream_error.upload_one(19u) == cudaSuccess);
  device.stage.peer_error_mask = std::uint64_t{1} << 20u;
  CHECK(launch(device) == 0);
  std::vector<double> shell;
  std::vector<std::uint8_t> active;
  std::vector<std::uint32_t> plan;
  CHECK(device.output_shell.download(shell) == cudaSuccess);
  CHECK(device.output_active.download(active) == cudaSuccess);
  CHECK(device.output_plan_error.download(plan) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(plan[0] == 19u);
  CHECK(std::all_of(active.begin(), active.end(), [](std::uint8_t value) { return value == 0u; }));
  CHECK(std::all_of(shell.begin(), shell.end(), [](double value) { return value == kSentinel; }));

  /* The downstream plan latch is sticky until its explicit async reset. */
  CHECK(device.upstream_error.upload_one(0u) == cudaSuccess);
  CHECK(device.upstream_sequence_active.upload_one(1u) == cudaSuccess);
  CHECK(launch(device) == 0);
  CHECK(device.output_plan_error.download(plan) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(plan[0] == 19u);

  cudaStream_t reset_stream = nullptr;
  CHECK(cudaStreamCreateWithFlags(&reset_stream, cudaStreamNonBlocking) == cudaSuccess);
  CHECK(reset_gfn2_scc_bridge_stage_cuda(device.batch.topology.batch_size,
                                         device.output_active.get(), device.output_plan_error.get(),
                                         device.sequence_active.get(),
                                         reset_stream) == cudaSuccess);
  CHECK(device.fill_output(kSentinel, reset_stream) == cudaSuccess);
  CHECK(launch(device, reset_stream) == 0);
  CHECK(device.output_shell.download(shell, reset_stream) == cudaSuccess);
  CHECK(device.output_active.download(active, reset_stream) == cudaSuccess);
  CHECK(device.output_plan_error.download(plan, reset_stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(reset_stream) == cudaSuccess);
  CHECK(cudaStreamDestroy(reset_stream) == cudaSuccess);
  CHECK(plan[0] == 0u && shell == host.expected);
  CHECK(std::all_of(active.begin(), active.end(), [](std::uint8_t value) { return value == 1u; }));

  /* Unknown large codes are always conservative plan failures. */
  CHECK(reset_gfn2_scc_bridge_stage_cuda(device.batch.topology.batch_size,
                                         device.output_active.get(), device.output_plan_error.get(),
                                         device.sequence_active.get()) == cudaSuccess);
  CHECK(device.upstream_error.upload_one(0xf001u) == cudaSuccess);
  CHECK(launch(device) == 0);
  CHECK(device.output_plan_error.download(plan) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(plan[0] == 0xf001u);

  CHECK(device.reset_status() == cudaSuccess);
  CHECK(device.shell_to_atom.upload_one(99, 0u) == cudaSuccess);
  CHECK(launch(device) == 0);
  CHECK(device.output_active.download(active) == cudaSuccess);
  CHECK(device.output_plan_error.download(plan) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(plan[0] == static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kInvalidShellToAtom));
  CHECK(std::all_of(active.begin(), active.end(), [](std::uint8_t value) { return value == 0u; }));

  CHECK(device.shell_to_atom.upload(host.shell_to_atom) == cudaSuccess);
  CHECK(device.reset_status() == cudaSuccess);
  std::vector<std::int64_t> bad_qsh_offsets = host.qsh_offsets;
  ++bad_qsh_offsets[1];
  CHECK(device.qsh_offsets.upload(bad_qsh_offsets) == cudaSuccess);
  CHECK(launch(device) == 0);
  CHECK(device.output_plan_error.download(plan) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(plan[0] == static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kInvalidFieldOffsets));

  CHECK(device.qsh_offsets.upload(host.qsh_offsets) == cudaSuccess);
  CHECK(device.reset_status() == cudaSuccess);
  std::vector<std::int64_t> bad_atom_offsets = host.atom_offsets;
  bad_atom_offsets[0] = -1;
  CHECK(device.atom_offsets.upload(bad_atom_offsets) == cudaSuccess);
  CHECK(launch(device) == 0);
  CHECK(device.output_plan_error.download(plan) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(plan[0] == static_cast<std::uint32_t>(Gfn2SccBridgeDeviceError::kInvalidTopologyOffsets));
  return 0;
}

int test_graph_replay_changed_inputs_and_status() {
  HostCase host = make_case(32u);
  DeviceCase device(host);
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
  CHECK(collect_gfn2_scc_shell_scalar_potential_cuda(device.batch, device.potential, device.stage,
                                                     device.output, device.workspace,
                                                     stream) == cudaSuccess);
  CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  CHECK(cudaGraphInstantiate(&executable, graph, 0) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

  for (std::size_t system = 0; system < 32u; ++system) {
    for (std::size_t local_shell = 0; local_shell < 4u; ++local_shell) {
      const std::size_t index = static_cast<std::size_t>(host.qsh_offsets[system]) + local_shell;
      host.shell_field[index] += 0.5;
      host.expected[system * 4u + local_shell] += 0.5;
    }
  }
  CHECK(device.shell_field.upload(host.shell_field, stream) == cudaSuccess);
  CHECK(device.fill_output(kSentinel, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  std::vector<double> shell;
  CHECK(device.output_shell.download(shell, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  CHECK(shell == host.expected);

  CHECK(device.system_errors.upload_one(23u, 7u, stream) == cudaSuccess);
  CHECK(device.upstream_error.upload_one(23u, 0u, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  device.stage.peer_error_mask = std::uint64_t{1} << 23u;
  /* Graph nodes own the old by-value mask, so recapture is required to change
   * descriptor metadata; changing pointed-to status/potential data is replayable. */
  CHECK(cudaGraphExecDestroy(executable) == cudaSuccess);
  CHECK(cudaGraphDestroy(graph) == cudaSuccess);
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
  CHECK(collect_gfn2_scc_shell_scalar_potential_cuda(device.batch, device.potential, device.stage,
                                                     device.output, device.workspace,
                                                     stream) == cudaSuccess);
  CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  CHECK(cudaGraphInstantiate(&executable, graph, 0) == cudaSuccess);
  CHECK(device.fill_output(kSentinel, stream) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  std::vector<std::uint8_t> active;
  CHECK(device.output_active.download(active, stream) == cudaSuccess);
  CHECK(device.output_shell.download(shell, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  CHECK(active[7] == 0u);
  for (std::size_t local_shell = 0; local_shell < 4u; ++local_shell) {
    CHECK(shell[7u * 4u + local_shell] == kSentinel);
  }

  /* Plan failure remains sticky across Graph replay, then the async reset
   * reopens the same executable after pointed-to status data is repaired. */
  CHECK(device.upstream_error.upload_one(71u, 0u, stream) == cudaSuccess);
  CHECK(device.upstream_sequence_active.upload_one(0u, 0u, stream) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(device.upstream_error.upload_one(0u, 0u, stream) == cudaSuccess);
  CHECK(device.upstream_sequence_active.upload_one(1u, 0u, stream) == cudaSuccess);
  CHECK(device.fill_output(kSentinel, stream) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  std::vector<std::uint32_t> plan;
  CHECK(device.output_plan_error.download(plan, stream) == cudaSuccess);
  CHECK(device.output_shell.download(shell, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  CHECK(plan[0] == 71u);
  CHECK(std::all_of(shell.begin(), shell.end(), [](double value) { return value == kSentinel; }));

  CHECK(device.system_errors.upload(std::vector<std::uint32_t>(32u, 0u), stream) == cudaSuccess);
  CHECK(reset_gfn2_scc_bridge_stage_cuda(device.batch.topology.batch_size,
                                         device.output_active.get(), device.output_plan_error.get(),
                                         device.sequence_active.get(), stream) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(device.output_plan_error.download(plan, stream) == cudaSuccess);
  CHECK(device.output_shell.download(shell, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  CHECK(plan[0] == 0u && shell == host.expected);
  CHECK(cudaGraphExecDestroy(executable) == cudaSuccess);
  CHECK(cudaGraphDestroy(graph) == cudaSuccess);
  CHECK(cudaStreamDestroy(stream) == cudaSuccess);
  return 0;
}

int test_host_binding_validation() {
  HostCase host = make_case(1u);
  DeviceCase device(host);
  const auto call = [&](const Gfn2SccBridgeDeviceBatch& batch,
                        const Gfn2SccBridgeDevicePotentialFields& potential,
                        const Gfn2SccBridgeDeviceStageInput& stage,
                        const Gfn2SccBridgeDeviceOutput& output,
                        const Gfn2SccBridgeDeviceWorkspace& workspace) {
    return collect_gfn2_scc_shell_scalar_potential_cuda(batch, potential, stage, output, workspace);
  };

  Gfn2SccBridgeDevicePotentialFields bad_potential = device.potential;
  bad_potential.plan_token += 1u;
  CHECK(call(device.batch, bad_potential, device.stage, device.output, device.workspace) ==
        cudaErrorInvalidValue);
  bad_potential = device.potential;
  bad_potential.shell = reinterpret_cast<const double*>(
      reinterpret_cast<const unsigned char*>(device.shell_field.get()) + 1u);
  CHECK(call(device.batch, bad_potential, device.stage, device.output, device.workspace) ==
        cudaErrorInvalidValue);
  bad_potential = device.potential;
  bad_potential.shell = device.output_shell.get();
  CHECK(call(device.batch, bad_potential, device.stage, device.output, device.workspace) ==
        cudaErrorInvalidValue);

  Gfn2SccBridgeDeviceOutput bad_output = device.output;
  bad_output.shell_scalar = device.shell_scratch.get();
  CHECK(call(device.batch, device.potential, device.stage, bad_output, device.workspace) ==
        cudaErrorInvalidValue);
  bad_output = device.output;
  bad_output.downstream_plan_error = device.system_errors.get();
  CHECK(call(device.batch, device.potential, device.stage, bad_output, device.workspace) ==
        cudaErrorInvalidValue);

  Gfn2SccBridgeDeviceStageInput bad_stage = device.stage;
  bad_stage.peer_error_mask = 1u;
  CHECK(call(device.batch, device.potential, bad_stage, device.output, device.workspace) ==
        cudaErrorInvalidValue);
  bad_stage = device.stage;
  bad_stage.upstream_device_error = device.output_plan_error.get();
  CHECK(call(device.batch, device.potential, bad_stage, device.output, device.workspace) ==
        cudaErrorInvalidValue);
  bad_stage = device.stage;
  bad_stage.upstream_sequence_active = device.sequence_active.get();
  CHECK(call(device.batch, device.potential, bad_stage, device.output, device.workspace) ==
        cudaErrorInvalidValue);

  Gfn2SccBridgeDeviceBatch bad_batch = device.batch;
  bad_batch.topology.memory_space = Gfn2PlanMemorySpace::kHost;
  CHECK(call(bad_batch, device.potential, device.stage, device.output, device.workspace) ==
        cudaErrorInvalidValue);
  bad_batch = device.batch;
  bad_batch.topology.batch_size = std::numeric_limits<std::int64_t>::max();
  CHECK(call(bad_batch, device.potential, device.stage, device.output, device.workspace) ==
        cudaErrorInvalidConfiguration);

  bad_potential = device.potential;
  bad_potential.shell = reinterpret_cast<const double*>(std::numeric_limits<std::uintptr_t>::max() -
                                                        (alignof(double) - 1u));
  CHECK(call(device.batch, bad_potential, device.stage, device.output, device.workspace) ==
        cudaErrorInvalidValue);
  CHECK(reset_gfn2_scc_bridge_stage_cuda(0, device.output_active.get(),
                                         device.output_plan_error.get(),
                                         device.sequence_active.get()) == cudaErrorInvalidValue);
  CHECK(reset_gfn2_scc_bridge_stage_cuda(1, device.output_active.get(),
                                         device.output_plan_error.get(),
                                         device.output_plan_error.get()) == cudaErrorInvalidValue);
  return 0;
}

}  // namespace

int main() {
  const std::array<int (*)(), 6> tests{{
      test_batch_parity_real_spd_and_custom_stream,
      test_inactive_poison_and_upstream_peer_isolation,
      test_numerical_failures_and_active_validation,
      test_plan_fail_closed_and_sticky_classification,
      test_graph_replay_changed_inputs_and_status,
      test_host_binding_validation,
  }};
  for (const auto test : tests) {
    const int status = test();
    if (status != 0) {
      return status;
    }
  }
  return 0;
}
