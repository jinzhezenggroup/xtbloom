#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_plan_schema.cuh"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::Gfn2AtomPair;
using gpuxtb::detail::Gfn2GenerationScope;
using gpuxtb::detail::Gfn2GeometryCacheProvenanceView;
using gpuxtb::detail::Gfn2PairMapKind;
using gpuxtb::detail::Gfn2PlanMemorySpace;
using gpuxtb::detail::Gfn2PlanSchemaDiagnostic;
using gpuxtb::detail::Gfn2PlanSchemaError;
using gpuxtb::detail::Gfn2PlanSchemaField;
using gpuxtb::detail::Gfn2RaggedTopologyView;
using gpuxtb::detail::cuda::bind_gfn2_geometry_provenance_cuda;
using gpuxtb::detail::cuda::bind_gfn2_topology_cuda;
using gpuxtb::detail::cuda::validate_gfn2_geometry_provenance_cuda_async;
using gpuxtb::detail::cuda::validate_gfn2_topology_cuda_async;

constexpr std::uint64_t kPlanToken = 0xbb67ae8584caa73bULL;
constexpr std::uint64_t kGeneration = 79u;
constexpr Gfn2PlanSchemaDiagnostic kDiagnosticSentinel{
    Gfn2PlanSchemaError::kStaleGeometry, Gfn2PlanSchemaField::kActiveMask, 0x123456789LL};

bool same_diagnostic(const Gfn2PlanSchemaDiagnostic& first,
                     const Gfn2PlanSchemaDiagnostic& second) {
  return first.error == second.error && first.field == second.field && first.index == second.index;
}

std::int64_t triangle(std::int64_t value) {
  return (value & 1LL) == 0LL ? (value / 2LL) * (value - 1LL) : value * ((value - 1LL) / 2LL);
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0u)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0u);
    }
    return *this;
  }
  ~DeviceBuffer() { release(); }

  cudaError_t copy_from(const std::vector<T>& values, cudaStream_t stream) {
    release();
    count_ = values.size();
    if (count_ == 0u) {
      return cudaSuccess;
    }
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T));
    if (status == cudaSuccess) {
      status =
          cudaMemcpyAsync(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice, stream);
    }
    return status;
  }

  cudaError_t allocate(std::size_t count) {
    release();
    count_ = count;
    return count == 0u ? cudaSuccess
                       : cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t copy_to(std::vector<T>& values, cudaStream_t stream) const {
    values.resize(count_);
    return count_ == 0u ? cudaSuccess
                        : cudaMemcpyAsync(values.data(), data_, count_ * sizeof(T),
                                          cudaMemcpyDeviceToHost, stream);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  void release() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
    data_ = nullptr;
    count_ = 0u;
  }

  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

struct HostCase {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> batch_orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> shell_orbital_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::int64_t> orbital_to_shell;
  std::vector<std::int64_t> orbital_to_atom;
  std::vector<std::int64_t> pair_offsets;
  std::vector<Gfn2AtomPair> atom_pairs;
  std::vector<std::int64_t> bucket_offsets;
  std::vector<std::int32_t> bucket_systems;
  std::vector<std::int32_t> bucket_orbital_counts;
};

HostCase make_case(std::int64_t batch_size) {
  HostCase host;
  host.atom_offsets.push_back(0);
  host.batch_shell_offsets.push_back(0);
  host.batch_orbital_offsets.push_back(0);
  host.matrix_offsets.push_back(0);
  host.atom_shell_offsets.push_back(0);
  host.shell_orbital_offsets.push_back(0);
  host.pair_offsets.push_back(0);
  std::vector<std::int32_t> orbital_counts(static_cast<std::size_t>(batch_size));
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t atom_count = batch_size == 1 ? 4 : system % 6;
    const std::int64_t atom_begin = host.atom_offsets.back();
    for (std::int64_t local_atom = 0; local_atom < atom_count; ++local_atom) {
      const std::int64_t atom = atom_begin + local_atom;
      const std::int64_t shell = static_cast<std::int64_t>(host.shell_to_atom.size());
      host.shell_to_atom.push_back(atom);
      host.atom_shell_offsets.push_back(shell + 1);
      host.orbital_to_shell.push_back(shell);
      host.orbital_to_atom.push_back(atom);
      host.shell_orbital_offsets.push_back(static_cast<std::int64_t>(host.orbital_to_shell.size()));
    }
    host.atom_offsets.push_back(atom_begin + atom_count);
    host.batch_shell_offsets.push_back(static_cast<std::int64_t>(host.shell_to_atom.size()));
    host.batch_orbital_offsets.push_back(static_cast<std::int64_t>(host.orbital_to_shell.size()));
    host.matrix_offsets.push_back(host.matrix_offsets.back() + atom_count * atom_count);
    host.pair_offsets.push_back(host.pair_offsets.back() + triangle(atom_count));
    for (std::int64_t first = atom_begin; first < atom_begin + atom_count; ++first) {
      for (std::int64_t second = first + 1; second < atom_begin + atom_count; ++second) {
        host.atom_pairs.push_back({first, second});
      }
    }
    orbital_counts[static_cast<std::size_t>(system)] = static_cast<std::int32_t>(atom_count);
  }
  std::map<std::int32_t, std::vector<std::int32_t>> buckets;
  for (std::int64_t system = 0; system < batch_size; ++system) {
    buckets[orbital_counts[static_cast<std::size_t>(system)]].push_back(
        static_cast<std::int32_t>(system));
  }
  host.bucket_offsets.push_back(0);
  for (const auto& [orbitals, systems] : buckets) {
    host.bucket_orbital_counts.push_back(orbitals);
    host.bucket_systems.insert(host.bucket_systems.end(), systems.begin(), systems.end());
    host.bucket_offsets.push_back(static_cast<std::int64_t>(host.bucket_systems.size()));
  }
  return host;
}

struct DeviceCase {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> batch_orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> shell_orbital_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<std::int64_t> orbital_to_shell;
  DeviceBuffer<std::int64_t> orbital_to_atom;
  DeviceBuffer<std::int64_t> pair_offsets;
  DeviceBuffer<Gfn2AtomPair> atom_pairs;
  DeviceBuffer<std::int64_t> bucket_offsets;
  DeviceBuffer<std::int32_t> bucket_systems;
  DeviceBuffer<std::int32_t> bucket_orbital_counts;
  DeviceBuffer<Gfn2PlanSchemaDiagnostic> diagnostic;

  cudaError_t upload(const HostCase& host, cudaStream_t stream) {
    cudaError_t status = atom_offsets.copy_from(host.atom_offsets, stream);
#define COPY_FIELD(name)                        \
  if (status == cudaSuccess) {                  \
    status = name.copy_from(host.name, stream); \
  }
    COPY_FIELD(batch_shell_offsets)
    COPY_FIELD(batch_orbital_offsets)
    COPY_FIELD(matrix_offsets)
    COPY_FIELD(atom_shell_offsets)
    COPY_FIELD(shell_orbital_offsets)
    COPY_FIELD(shell_to_atom)
    COPY_FIELD(orbital_to_shell)
    COPY_FIELD(orbital_to_atom)
    COPY_FIELD(pair_offsets)
    COPY_FIELD(atom_pairs)
    COPY_FIELD(bucket_offsets)
    COPY_FIELD(bucket_systems)
    COPY_FIELD(bucket_orbital_counts)
#undef COPY_FIELD
    if (status == cudaSuccess) {
      status = diagnostic.allocate(1u);
    }
    return status;
  }

  Gfn2RaggedTopologyView view(const HostCase& host) const {
    Gfn2RaggedTopologyView result{};
    result.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    result.pair_map_kind = Gfn2PairMapKind::kPackedLowerTriangle;
    result.plan_token = kPlanToken;
    result.batch_size = static_cast<std::int64_t>(host.atom_offsets.size() - 1u);
    result.total_atoms = host.atom_offsets.back();
    result.total_shells = host.batch_shell_offsets.back();
    result.total_orbitals = host.batch_orbital_offsets.back();
    result.total_matrix_elements = host.matrix_offsets.back();
    result.total_pairs = host.pair_offsets.back();
    result.bucket_count = static_cast<std::int64_t>(host.bucket_orbital_counts.size());
#define BIND_FIELD(name, count_name)                          \
  result.count_name = static_cast<std::int64_t>(name.size()); \
  result.name = name.get();
    BIND_FIELD(atom_offsets, atom_offset_count)
    BIND_FIELD(batch_shell_offsets, batch_shell_offset_count)
    BIND_FIELD(batch_orbital_offsets, batch_orbital_offset_count)
    BIND_FIELD(matrix_offsets, matrix_offset_count)
    BIND_FIELD(atom_shell_offsets, atom_shell_offset_count)
    BIND_FIELD(shell_orbital_offsets, shell_orbital_offset_count)
    BIND_FIELD(shell_to_atom, shell_to_atom_count)
    BIND_FIELD(orbital_to_shell, orbital_to_shell_count)
    BIND_FIELD(orbital_to_atom, orbital_to_atom_count)
    BIND_FIELD(pair_offsets, pair_offset_count)
    BIND_FIELD(bucket_offsets, bucket_offset_count)
    BIND_FIELD(bucket_systems, bucket_system_count)
    BIND_FIELD(bucket_orbital_counts, bucket_orbital_count)
#undef BIND_FIELD
    return result;
  }
};

int test_batches_and_graph() {
  for (const std::int64_t batch_size : {1, 8, 32, 128}) {
    const HostCase host = make_case(batch_size);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceCase device;
    CUDA_CHECK(device.upload(host, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const Gfn2RaggedTopologyView candidate = device.view(host);
    Gfn2RaggedTopologyView binding{};
    Gfn2PlanSchemaDiagnostic diagnostic{};
    CUDA_CHECK(
        bind_gfn2_topology_cuda(candidate, binding, device.diagnostic.get(), diagnostic, stream));
    CHECK(diagnostic.error == Gfn2PlanSchemaError::kSuccess);
    CHECK(binding.plan_token == kPlanToken);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CUDA_CHECK(validate_gfn2_topology_cuda_async(candidate, device.diagnostic.get(), stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaMemcpyAsync(&diagnostic, device.diagnostic.get(), sizeof(diagnostic),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(diagnostic.error == Gfn2PlanSchemaError::kSuccess);

    const std::int64_t invalid_offset = -1;
    CUDA_CHECK(cudaMemcpyAsync(device.atom_offsets.get() + 1, &invalid_offset,
                               sizeof(invalid_offset), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaMemcpyAsync(&diagnostic, device.diagnostic.get(), sizeof(diagnostic),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(diagnostic.error == Gfn2PlanSchemaError::kInvalidOffsets);
    CUDA_CHECK(cudaMemcpyAsync(device.atom_offsets.get(), host.atom_offsets.data(),
                               host.atom_offsets.size() * sizeof(std::int64_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaMemcpyAsync(&diagnostic, device.diagnostic.get(), sizeof(diagnostic),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(diagnostic.error == Gfn2PlanSchemaError::kSuccess);
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));

    if (batch_size == 1) {
      Gfn2RaggedTopologyView structurally_invalid = candidate;
      structurally_invalid.plan_token = 0u;
      CUDA_CHECK(cudaMemcpyAsync(device.diagnostic.get(), &kDiagnosticSentinel,
                                 sizeof(kDiagnosticSentinel), cudaMemcpyHostToDevice, stream));
      CHECK(validate_gfn2_topology_cuda_async(structurally_invalid, device.diagnostic.get(),
                                              stream) == cudaErrorInvalidValue);
      CUDA_CHECK(cudaMemcpyAsync(&diagnostic, device.diagnostic.get(), sizeof(diagnostic),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(same_diagnostic(diagnostic, kDiagnosticSentinel));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  return 0;
}

int test_hostile_device_topology() {
  HostCase host = make_case(8);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  DeviceCase device;
  CUDA_CHECK(device.upload(host, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  Gfn2RaggedTopologyView candidate = device.view(host);
  Gfn2RaggedTopologyView binding{};
  Gfn2PlanSchemaDiagnostic diagnostic{};

  const std::int64_t invalid_offset = -1;
  CUDA_CHECK(cudaMemcpyAsync(device.atom_offsets.get() + 2, &invalid_offset, sizeof(invalid_offset),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(
      bind_gfn2_topology_cuda(candidate, binding, device.diagnostic.get(), diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kInvalidOffsets);
  CHECK(binding.plan_token == 0u);
  CUDA_CHECK(cudaMemcpyAsync(device.atom_offsets.get(), host.atom_offsets.data(),
                             host.atom_offsets.size() * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  candidate = device.view(host);
  candidate.pair_map_kind = Gfn2PairMapKind::kExplicit;
  candidate.atom_pair_count = static_cast<std::int64_t>(device.atom_pairs.size());
  candidate.atom_pairs = device.atom_pairs.get();
  CUDA_CHECK(
      bind_gfn2_topology_cuda(candidate, binding, device.diagnostic.get(), diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kSuccess);
  Gfn2AtomPair invalid_pair = host.atom_pairs[0];
  invalid_pair.second = invalid_pair.first;
  CUDA_CHECK(cudaMemcpyAsync(device.atom_pairs.get(), &invalid_pair, sizeof(invalid_pair),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(
      bind_gfn2_topology_cuda(candidate, binding, device.diagnostic.get(), diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kInvalidPairMap);
  CHECK(diagnostic.field == gpuxtb::detail::Gfn2PlanSchemaField::kAtomPairs);
  CUDA_CHECK(cudaMemcpyAsync(device.atom_pairs.get(), host.atom_pairs.data(),
                             host.atom_pairs.size() * sizeof(Gfn2AtomPair), cudaMemcpyHostToDevice,
                             stream));

  candidate = device.view(host);
  const std::int64_t invalid_shell_atom = candidate.total_atoms;
  CUDA_CHECK(cudaMemcpyAsync(device.shell_to_atom.get(), &invalid_shell_atom,
                             sizeof(invalid_shell_atom), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(
      bind_gfn2_topology_cuda(candidate, binding, device.diagnostic.get(), diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kInvalidShellMap);
  CUDA_CHECK(cudaMemcpyAsync(device.shell_to_atom.get(), host.shell_to_atom.data(),
                             host.shell_to_atom.size() * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice, stream));

  const std::int64_t invalid_orbital_shell = candidate.total_shells;
  CUDA_CHECK(cudaMemcpyAsync(device.orbital_to_shell.get(), &invalid_orbital_shell,
                             sizeof(invalid_orbital_shell), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(
      bind_gfn2_topology_cuda(candidate, binding, device.diagnostic.get(), diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kInvalidOrbitalMap);
  CUDA_CHECK(cudaMemcpyAsync(device.orbital_to_shell.get(), host.orbital_to_shell.data(),
                             host.orbital_to_shell.size() * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice, stream));

  const std::int64_t invalid_pair_offset = host.pair_offsets[2] + 1;
  CUDA_CHECK(cudaMemcpyAsync(device.pair_offsets.get() + 2, &invalid_pair_offset,
                             sizeof(invalid_pair_offset), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(
      bind_gfn2_topology_cuda(candidate, binding, device.diagnostic.get(), diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kInvalidPairMap);
  CUDA_CHECK(cudaMemcpyAsync(device.pair_offsets.get(), host.pair_offsets.data(),
                             host.pair_offsets.size() * sizeof(std::int64_t),
                             cudaMemcpyHostToDevice, stream));

  const std::int32_t negative_bucket_orbitals = -1;
  CUDA_CHECK(cudaMemcpyAsync(device.bucket_orbital_counts.get(), &negative_bucket_orbitals,
                             sizeof(negative_bucket_orbitals), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(
      bind_gfn2_topology_cuda(candidate, binding, device.diagnostic.get(), diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kInvalidBucketMap);
  CHECK(diagnostic.field == gpuxtb::detail::Gfn2PlanSchemaField::kBucketOrbitalCounts);
  CUDA_CHECK(cudaMemcpyAsync(device.bucket_orbital_counts.get(), host.bucket_orbital_counts.data(),
                             host.bucket_orbital_counts.size() * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  candidate = device.view(host);
  std::vector<std::int64_t> offsets_before;
  std::vector<std::int64_t> offsets_after;
  CUDA_CHECK(device.atom_offsets.copy_to(offsets_before, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto* const aliased_diagnostic =
      reinterpret_cast<Gfn2PlanSchemaDiagnostic*>(device.atom_offsets.get());
  CUDA_CHECK(bind_gfn2_topology_cuda(candidate, binding, aliased_diagnostic, diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kAliasedRange);
  CHECK(diagnostic.field == gpuxtb::detail::Gfn2PlanSchemaField::kAtomOffsets);
  CHECK(binding.plan_token == 0u);
  CHECK(validate_gfn2_topology_cuda_async(candidate, aliased_diagnostic, stream) ==
        cudaErrorInvalidValue);
  CUDA_CHECK(device.atom_offsets.copy_to(offsets_after, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(offsets_after == offsets_before);

  auto* const partially_aliased_diagnostic =
      reinterpret_cast<Gfn2PlanSchemaDiagnostic*>(device.atom_offsets.get() + 1);
  CUDA_CHECK(bind_gfn2_topology_cuda(candidate, binding, partially_aliased_diagnostic, diagnostic,
                                     stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kAliasedRange);
  CHECK(diagnostic.field == gpuxtb::detail::Gfn2PlanSchemaField::kAtomOffsets);
  CUDA_CHECK(device.atom_offsets.copy_to(offsets_after, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(offsets_after == offsets_before);

  Gfn2RaggedTopologyView malformed = candidate;
  malformed.atom_offset_count = -1;
  CHECK(validate_gfn2_topology_cuda_async(malformed, partially_aliased_diagnostic, stream) ==
        cudaErrorInvalidValue);
  malformed.atom_offset_count = 0;
  CHECK(validate_gfn2_topology_cuda_async(malformed, partially_aliased_diagnostic, stream) ==
        cudaErrorInvalidValue);
  malformed.atom_offset_count = std::numeric_limits<std::int64_t>::max();
  CHECK(validate_gfn2_topology_cuda_async(malformed, partially_aliased_diagnostic, stream) ==
        cudaErrorInvalidValue);
  const std::uintptr_t atom_address = reinterpret_cast<std::uintptr_t>(device.atom_offsets.get());
  const std::uintmax_t address_overflow_count =
      (std::numeric_limits<std::uintptr_t>::max() - atom_address) / sizeof(std::int64_t) + 1u;
  CHECK(address_overflow_count <=
        static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()));
  malformed.atom_offset_count = static_cast<std::int64_t>(address_overflow_count);
  CHECK(validate_gfn2_topology_cuda_async(malformed, partially_aliased_diagnostic, stream) ==
        cudaErrorInvalidValue);
  CUDA_CHECK(device.atom_offsets.copy_to(offsets_after, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(offsets_after == offsets_before);

  malformed = candidate;
  malformed.atom_offset_count = -1;
  CUDA_CHECK(cudaMemcpyAsync(device.diagnostic.get(), &kDiagnosticSentinel,
                             sizeof(kDiagnosticSentinel), cudaMemcpyHostToDevice, stream));
  CHECK(validate_gfn2_topology_cuda_async(malformed, device.diagnostic.get(), stream) ==
        cudaErrorInvalidValue);
  CUDA_CHECK(cudaMemcpyAsync(&diagnostic, device.diagnostic.get(), sizeof(diagnostic),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(same_diagnostic(diagnostic, kDiagnosticSentinel));

  const std::uintptr_t overflowing_address =
      std::numeric_limits<std::uintptr_t>::max() - (alignof(Gfn2PlanSchemaDiagnostic) - 1u);
  auto* const overflowing_diagnostic =
      reinterpret_cast<Gfn2PlanSchemaDiagnostic*>(overflowing_address);
  CHECK(validate_gfn2_topology_cuda_async(candidate, overflowing_diagnostic, stream) ==
        cudaErrorInvalidValue);

  candidate.batch_shell_offsets = candidate.atom_offsets;
  CUDA_CHECK(
      bind_gfn2_topology_cuda(candidate, binding, device.diagnostic.get(), diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kAliasedRange);

  candidate = device.view(host);
  candidate.atom_offsets = host.atom_offsets.data();
  CUDA_CHECK(
      bind_gfn2_topology_cuda(candidate, binding, device.diagnostic.get(), diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kInvalidMemorySpace);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_device_provenance() {
  constexpr std::int64_t kBatchSize = 32;
  const HostCase host = make_case(kBatchSize);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  DeviceCase device;
  CUDA_CHECK(device.upload(host, stream));
  std::vector<std::uint64_t> host_generations(static_cast<std::size_t>(kBatchSize), kGeneration);
  std::vector<std::uint8_t> host_active(static_cast<std::size_t>(kBatchSize), 1u);
  DeviceBuffer<std::uint64_t> generations;
  DeviceBuffer<std::uint8_t> active;
  CUDA_CHECK(generations.copy_from(host_generations, stream));
  CUDA_CHECK(active.copy_from(host_active, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const Gfn2RaggedTopologyView topology = device.view(host);

  Gfn2GeometryCacheProvenanceView candidate{};
  candidate.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  candidate.generation_scope = Gfn2GenerationScope::kPerSystem;
  candidate.plan_token = kPlanToken;
  candidate.batch_size = kBatchSize;
  candidate.system_generation_count = kBatchSize;
  candidate.system_geometry_generations = generations.get();
  Gfn2GeometryCacheProvenanceView binding{};
  Gfn2PlanSchemaDiagnostic diagnostic{};
  CUDA_CHECK(bind_gfn2_geometry_provenance_cuda(topology, candidate, kGeneration, active.get(),
                                                kBatchSize, binding, device.diagnostic.get(),
                                                diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kSuccess);
  CHECK(binding.plan_token == kPlanToken);

  std::vector<std::uint64_t> generations_before;
  std::vector<std::uint64_t> generations_after;
  CUDA_CHECK(generations.copy_to(generations_before, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto* const generation_diagnostic =
      reinterpret_cast<Gfn2PlanSchemaDiagnostic*>(generations.get());
  CUDA_CHECK(bind_gfn2_geometry_provenance_cuda(topology, candidate, kGeneration, active.get(),
                                                kBatchSize, binding, generation_diagnostic,
                                                diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kAliasedRange);
  CHECK(diagnostic.field == gpuxtb::detail::Gfn2PlanSchemaField::kSystemGeometryGenerations);
  CHECK(binding.plan_token == 0u);
  CHECK(validate_gfn2_geometry_provenance_cuda_async(topology, candidate, kGeneration, active.get(),
                                                     kBatchSize, generation_diagnostic,
                                                     stream) == cudaErrorInvalidValue);
  auto* const partial_generation_diagnostic =
      reinterpret_cast<Gfn2PlanSchemaDiagnostic*>(generations.get() + 1);
  CUDA_CHECK(bind_gfn2_geometry_provenance_cuda(topology, candidate, kGeneration, active.get(),
                                                kBatchSize, binding, partial_generation_diagnostic,
                                                diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kAliasedRange);
  CHECK(diagnostic.field == Gfn2PlanSchemaField::kSystemGeometryGenerations);
  CHECK(validate_gfn2_geometry_provenance_cuda_async(topology, candidate, kGeneration, active.get(),
                                                     kBatchSize, partial_generation_diagnostic,
                                                     stream) == cudaErrorInvalidValue);
  CUDA_CHECK(generations.copy_to(generations_after, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(generations_after == generations_before);

  std::vector<std::uint8_t> active_before;
  std::vector<std::uint8_t> active_after;
  CUDA_CHECK(active.copy_to(active_before, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto* const active_diagnostic = reinterpret_cast<Gfn2PlanSchemaDiagnostic*>(active.get());
  CUDA_CHECK(bind_gfn2_geometry_provenance_cuda(topology, candidate, kGeneration, active.get(),
                                                kBatchSize, binding, active_diagnostic, diagnostic,
                                                stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kAliasedRange);
  CHECK(diagnostic.field == gpuxtb::detail::Gfn2PlanSchemaField::kActiveMask);
  CHECK(validate_gfn2_geometry_provenance_cuda_async(topology, candidate, kGeneration, active.get(),
                                                     kBatchSize, active_diagnostic,
                                                     stream) == cudaErrorInvalidValue);
  auto* const partial_active_diagnostic =
      reinterpret_cast<Gfn2PlanSchemaDiagnostic*>(active.get() + 8);
  CUDA_CHECK(bind_gfn2_geometry_provenance_cuda(topology, candidate, kGeneration, active.get(),
                                                kBatchSize, binding, partial_active_diagnostic,
                                                diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kAliasedRange);
  CHECK(diagnostic.field == Gfn2PlanSchemaField::kActiveMask);
  CHECK(validate_gfn2_geometry_provenance_cuda_async(topology, candidate, kGeneration, active.get(),
                                                     kBatchSize, partial_active_diagnostic,
                                                     stream) == cudaErrorInvalidValue);
  CUDA_CHECK(active.copy_to(active_after, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(active_after == active_before);

  Gfn2GeometryCacheProvenanceView malformed_provenance = candidate;
  malformed_provenance.system_generation_count = -1;
  CHECK(validate_gfn2_geometry_provenance_cuda_async(
            topology, malformed_provenance, kGeneration, active.get(), kBatchSize,
            partial_generation_diagnostic, stream) == cudaErrorInvalidValue);
  malformed_provenance.system_generation_count = 0;
  CHECK(validate_gfn2_geometry_provenance_cuda_async(
            topology, malformed_provenance, kGeneration, active.get(), kBatchSize,
            partial_generation_diagnostic, stream) == cudaErrorInvalidValue);
  malformed_provenance.system_generation_count = std::numeric_limits<std::int64_t>::max();
  CHECK(validate_gfn2_geometry_provenance_cuda_async(
            topology, malformed_provenance, kGeneration, active.get(), kBatchSize,
            partial_generation_diagnostic, stream) == cudaErrorInvalidValue);
  const std::uintptr_t generation_address = reinterpret_cast<std::uintptr_t>(generations.get());
  const std::uintmax_t generation_address_overflow_count =
      (std::numeric_limits<std::uintptr_t>::max() - generation_address) / sizeof(std::uint64_t) +
      1u;
  CHECK(generation_address_overflow_count <=
        static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()));
  malformed_provenance.system_generation_count =
      static_cast<std::int64_t>(generation_address_overflow_count);
  CHECK(validate_gfn2_geometry_provenance_cuda_async(
            topology, malformed_provenance, kGeneration, active.get(), kBatchSize,
            partial_generation_diagnostic, stream) == cudaErrorInvalidValue);
  CHECK(validate_gfn2_geometry_provenance_cuda_async(topology, candidate, kGeneration, active.get(),
                                                     0, partial_active_diagnostic,
                                                     stream) == cudaErrorInvalidValue);
  CHECK(validate_gfn2_geometry_provenance_cuda_async(topology, candidate, kGeneration, active.get(),
                                                     -1, partial_active_diagnostic,
                                                     stream) == cudaErrorInvalidValue);
  CUDA_CHECK(generations.copy_to(generations_after, stream));
  CUDA_CHECK(active.copy_to(active_after, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(generations_after == generations_before);
  CHECK(active_after == active_before);

  malformed_provenance = candidate;
  malformed_provenance.system_generation_count = -1;
  CUDA_CHECK(cudaMemcpyAsync(device.diagnostic.get(), &kDiagnosticSentinel,
                             sizeof(kDiagnosticSentinel), cudaMemcpyHostToDevice, stream));
  CHECK(validate_gfn2_geometry_provenance_cuda_async(
            topology, malformed_provenance, kGeneration, active.get(), kBatchSize,
            device.diagnostic.get(), stream) == cudaErrorInvalidValue);
  CUDA_CHECK(cudaMemcpyAsync(&diagnostic, device.diagnostic.get(), sizeof(diagnostic),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(same_diagnostic(diagnostic, kDiagnosticSentinel));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  CUDA_CHECK(validate_gfn2_geometry_provenance_cuda_async(
      topology, candidate, kGeneration, active.get(), kBatchSize, device.diagnostic.get(), stream));
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaMemcpyAsync(&diagnostic, device.diagnostic.get(), sizeof(diagnostic),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kSuccess);

  const std::uint64_t graph_stale = kGeneration - 1u;
  CUDA_CHECK(cudaMemcpyAsync(generations.get() + 3, &graph_stale, sizeof(graph_stale),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaMemcpyAsync(&diagnostic, device.diagnostic.get(), sizeof(diagnostic),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kStaleGeometry);
  CHECK(diagnostic.index == 3);
  const std::uint8_t graph_inactive = 0u;
  CUDA_CHECK(cudaMemcpyAsync(active.get() + 3, &graph_inactive, sizeof(graph_inactive),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaMemcpyAsync(&diagnostic, device.diagnostic.get(), sizeof(diagnostic),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kSuccess);
  const std::uint8_t graph_invalid_active = 2u;
  CUDA_CHECK(cudaMemcpyAsync(active.get() + 2, &graph_invalid_active, sizeof(graph_invalid_active),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CUDA_CHECK(cudaMemcpyAsync(&diagnostic, device.diagnostic.get(), sizeof(diagnostic),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kInvalidActiveMask);
  CHECK(diagnostic.index == 2);
  CUDA_CHECK(generations.copy_from(host_generations, stream));
  CUDA_CHECK(active.copy_from(host_active, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));

  CUDA_CHECK(bind_gfn2_geometry_provenance_cuda(
      topology, candidate, kGeneration, reinterpret_cast<const std::uint8_t*>(generations.get()),
      kBatchSize, binding, device.diagnostic.get(), diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kAliasedRange);

  Gfn2GeometryCacheProvenanceView batch_candidate{};
  batch_candidate.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
  batch_candidate.generation_scope = Gfn2GenerationScope::kBatch;
  batch_candidate.plan_token = kPlanToken;
  batch_candidate.geometry_generation = kGeneration;
  batch_candidate.batch_size = kBatchSize;
  const std::uint8_t invalid_batch_active = 2u;
  CUDA_CHECK(cudaMemcpyAsync(active.get() + 2, &invalid_batch_active, sizeof(invalid_batch_active),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(bind_gfn2_geometry_provenance_cuda(topology, batch_candidate, kGeneration,
                                                active.get(), kBatchSize, binding,
                                                device.diagnostic.get(), diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kInvalidActiveMask);
  const std::uint8_t active_value = 1u;
  CUDA_CHECK(cudaMemcpyAsync(active.get() + 2, &active_value, sizeof(active_value),
                             cudaMemcpyHostToDevice, stream));

  const std::uint64_t stale = kGeneration - 1u;
  CUDA_CHECK(cudaMemcpyAsync(generations.get() + 3, &stale, sizeof(stale), cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(bind_gfn2_geometry_provenance_cuda(topology, candidate, kGeneration, active.get(),
                                                kBatchSize, binding, device.diagnostic.get(),
                                                diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kStaleGeometry);
  CHECK(diagnostic.index == 3);
  const std::uint8_t inactive = 0u;
  CUDA_CHECK(cudaMemcpyAsync(active.get() + 3, &inactive, sizeof(inactive), cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(bind_gfn2_geometry_provenance_cuda(topology, candidate, kGeneration, active.get(),
                                                kBatchSize, binding, device.diagnostic.get(),
                                                diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kSuccess);

  candidate.plan_token += 1u;
  CUDA_CHECK(bind_gfn2_geometry_provenance_cuda(topology, candidate, kGeneration, active.get(),
                                                kBatchSize, binding, device.diagnostic.get(),
                                                diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kCrossPlan);
  candidate.plan_token = kPlanToken;
  const std::uint8_t invalid_active = 2u;
  CUDA_CHECK(cudaMemcpyAsync(active.get() + 2, &invalid_active, sizeof(invalid_active),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(bind_gfn2_geometry_provenance_cuda(topology, candidate, kGeneration, active.get(),
                                                kBatchSize, binding, device.diagnostic.get(),
                                                diagnostic, stream));
  CHECK(diagnostic.error == Gfn2PlanSchemaError::kInvalidActiveMask);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main() {
  static_assert(std::is_trivially_copyable_v<Gfn2RaggedTopologyView>);
  int status = test_batches_and_graph();
  if (status == 0) {
    status = test_hostile_device_topology();
  }
  if (status == 0) {
    status = test_device_provenance();
  }
  return status;
}
