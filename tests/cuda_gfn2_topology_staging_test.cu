#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "gpuxtb/gpuxtb.h"
#include "runtime/gfn2_cuda_topology_staging.hpp"

#define CHECK(condition)                                                                    \
  do {                                                                                      \
    if (!(condition)) {                                                                     \
      std::fprintf(stderr, "CUDA topology-staging check failed at line %d: %s\n", __LINE__, \
                   #condition);                                                             \
      return __LINE__;                                                                      \
    }                                                                                       \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::Gfn2CudaTopologyHostSnapshot;
using gpuxtb::detail::Gfn2CudaTopologyStageDisposition;
using gpuxtb::detail::Gfn2CudaTopologyStaging;
using gpuxtb::detail::Gfn2CudaTopologyStagingField;

template <typename T>
gpuxtb_const_buffer_t host_buffer(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), GPUXTB_MEMORY_HOST,
          0u};
}

template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream) {
    if (values.size() != size_) {
      if (data_ != nullptr) {
        cudaError_t status = cudaFree(data_);
        if (status != cudaSuccess) return status;
        data_ = nullptr;
      }
      size_ = values.size();
      if (size_ != 0u) {
        cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T));
        if (status != cudaSuccess) return status;
      }
    }
    return size_ == 0u ? cudaSuccess
                       : cudaMemcpyAsync(data_, values.data(), size_ * sizeof(T),
                                         cudaMemcpyHostToDevice, stream);
  }

  gpuxtb_const_buffer_t view() const noexcept {
    return {data_, size_ * sizeof(T), GPUXTB_MEMORY_CUDA_DEVICE, 0u};
  }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0u;
};

struct Topology {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int64_t> point_offsets;
  std::vector<std::int64_t> response_offsets;

  static Topology make(std::int64_t batch_size) {
    Topology topology;
    topology.atom_offsets.push_back(0);
    topology.point_offsets.push_back(0);
    topology.response_offsets.push_back(0);
    for (std::int64_t system = 0; system < batch_size; ++system) {
      const std::int64_t atoms = 1 + system % 3;
      const std::int64_t points = system % 3;
      topology.atom_offsets.push_back(topology.atom_offsets.back() + atoms);
      topology.point_offsets.push_back(topology.point_offsets.back() + points);
      topology.response_offsets.push_back(topology.response_offsets.back() + atoms * atoms);
      topology.molecular_charges.push_back(0.25 * static_cast<double>(system % 5 - 2));
      topology.unpaired_electrons.push_back(0);
      for (std::int64_t atom = 0; atom < atoms; ++atom) {
        topology.atomic_numbers.push_back(
            static_cast<std::int32_t>(1 + (system * 7 + atom * 13) % 118));
      }
    }
    return topology;
  }
};

struct DeviceTopology {
  DeviceArray<std::int64_t> atom_offsets;
  DeviceArray<std::int32_t> atomic_numbers;
  DeviceArray<double> molecular_charges;
  DeviceArray<std::int32_t> unpaired_electrons;
  DeviceArray<std::int64_t> point_offsets;
  DeviceArray<std::int64_t> response_offsets;

  int upload(const Topology& topology, cudaStream_t stream) {
    CUDA_CHECK(atom_offsets.upload(topology.atom_offsets, stream));
    CUDA_CHECK(atomic_numbers.upload(topology.atomic_numbers, stream));
    CUDA_CHECK(molecular_charges.upload(topology.molecular_charges, stream));
    CUDA_CHECK(unpaired_electrons.upload(topology.unpaired_electrons, stream));
    CUDA_CHECK(point_offsets.upload(topology.point_offsets, stream));
    CUDA_CHECK(response_offsets.upload(topology.response_offsets, stream));
    return 0;
  }
};

enum class SourceMode { kHost, kDevice, kMixed };

gpuxtb_batch_t make_batch(const Topology& host, const DeviceTopology& device, SourceMode mode,
                          bool include_points = true, bool include_response = true) {
  gpuxtb_batch_t batch{};
  batch.struct_size = sizeof(batch);
  batch.api_version = GPUXTB_API_VERSION;
  batch.batch_size = static_cast<std::int64_t>(host.molecular_charges.size());
  batch.total_atoms = static_cast<std::int64_t>(host.atomic_numbers.size());
  batch.total_point_charges = include_points ? host.point_offsets.back() : 0;
  batch.total_charge_response_elements = include_response ? host.response_offsets.back() : 0;
  const bool atom_device = mode == SourceMode::kDevice || mode == SourceMode::kMixed;
  const bool number_device = mode == SourceMode::kDevice;
  const bool charge_device = mode != SourceMode::kHost;
  const bool spin_device = mode == SourceMode::kDevice;
  const bool point_device = mode != SourceMode::kHost;
  const bool response_device = mode == SourceMode::kDevice;
  batch.atom_offsets = atom_device ? device.atom_offsets.view() : host_buffer(host.atom_offsets);
  batch.atomic_numbers =
      number_device ? device.atomic_numbers.view() : host_buffer(host.atomic_numbers);
  batch.molecular_charges =
      charge_device ? device.molecular_charges.view() : host_buffer(host.molecular_charges);
  batch.unpaired_electrons =
      spin_device ? device.unpaired_electrons.view() : host_buffer(host.unpaired_electrons);
  if (include_points) {
    batch.point_charge_offsets =
        point_device ? device.point_offsets.view() : host_buffer(host.point_offsets);
  }
  if (include_response) {
    batch.charge_response_offsets =
        response_device ? device.response_offsets.view() : host_buffer(host.response_offsets);
  }
  return batch;
}

bool snapshot_equals(const Gfn2CudaTopologyHostSnapshot& snapshot, const Topology& topology,
                     bool periodic_enabled = true) {
  return snapshot.batch_size == static_cast<std::int64_t>(topology.molecular_charges.size()) &&
         snapshot.total_atoms == static_cast<std::int64_t>(topology.atomic_numbers.size()) &&
         snapshot.total_point_charges == topology.point_offsets.back() &&
         snapshot.total_charge_response_elements == topology.response_offsets.back() &&
         snapshot.periodic_enabled == periodic_enabled &&
         snapshot.atom_offsets == topology.atom_offsets &&
         snapshot.atomic_numbers == topology.atomic_numbers &&
         snapshot.molecular_charges == topology.molecular_charges &&
         snapshot.unpaired_electrons == topology.unpaired_electrons &&
         snapshot.point_charge_offsets == topology.point_offsets &&
         snapshot.charge_response_offsets == topology.response_offsets;
}

int exercise_sources_and_transactions(std::int64_t batch_size, int device_id, cudaStream_t stream) {
  Topology topology = Topology::make(batch_size);
  DeviceTopology device;
  CHECK(device.upload(topology, stream) == 0);

  Gfn2CudaTopologyStaging staging(device_id, stream);
  CHECK(staging.valid());
  std::string error;
  gpuxtb_batch_t host_batch = make_batch(topology, device, SourceMode::kHost);
  auto result = staging.stage_and_validate(host_batch, error);
  CHECK(result.success());
  CHECK(result.disposition == Gfn2CudaTopologyStageDisposition::kCandidate);
  CHECK(error.empty());
  CHECK(staging.candidate_snapshot() != nullptr);
  CHECK(snapshot_equals(*staging.candidate_snapshot(), topology));
  CHECK(staging.committed_snapshot() == nullptr);
  CHECK(staging.commit_candidate(error).success());
  CHECK(snapshot_equals(*staging.committed_snapshot(), topology));

  const auto fixed_identity = staging.identity();
  CHECK(fixed_identity.committed_generation == 1u);
  CHECK(fixed_identity.full_metadata_downloads == 1u);
  CHECK(fixed_identity.staging.packed_base != 0u);
  CHECK(fixed_identity.committed.packed_base != 0u);
  CHECK(fixed_identity.staging.packed_base != fixed_identity.committed.packed_base);

  for (SourceMode mode : {SourceMode::kHost, SourceMode::kDevice, SourceMode::kMixed}) {
    gpuxtb_batch_t batch = make_batch(topology, device, mode);
    result = staging.stage_and_validate(batch, error);
    CHECK(result.success());
    CHECK(result.disposition == Gfn2CudaTopologyStageDisposition::kMatchesCommitted);
    const auto matched = staging.identity();
    CHECK(matched.committed_owner == fixed_identity.committed_owner);
    CHECK(matched.staging.packed_base == fixed_identity.staging.packed_base);
    CHECK(matched.committed.packed_base == fixed_identity.committed.packed_base);
    CHECK(matched.pinned_staging == fixed_identity.pinned_staging);
    CHECK(matched.full_metadata_downloads == fixed_identity.full_metadata_downloads);
  }

  Topology changed = topology;
  changed.molecular_charges[static_cast<std::size_t>(batch_size / 2)] += 0.125;
  DeviceTopology changed_device;
  CHECK(changed_device.upload(changed, stream) == 0);
  gpuxtb_batch_t changed_batch = make_batch(changed, changed_device, SourceMode::kMixed);
  result = staging.stage_and_validate(changed_batch, error);
  CHECK(result.success());
  CHECK(result.disposition == Gfn2CudaTopologyStageDisposition::kCandidate);
  CHECK(snapshot_equals(*staging.candidate_snapshot(), changed));
  const auto before_abort = staging.identity();
  CHECK(before_abort.candidate_pending == 1u);
  CHECK(before_abort.committed.packed_base == fixed_identity.committed.packed_base);
  staging.abort_candidate();
  const auto aborted = staging.identity();
  CHECK(aborted.candidate_pending == 0u);
  CHECK(aborted.committed_generation == 1u);
  CHECK(aborted.committed.packed_base == fixed_identity.committed.packed_base);
  CHECK(snapshot_equals(*staging.committed_snapshot(), topology));

  result = staging.stage_and_validate(changed_batch, error);
  CHECK(result.success());
  CHECK(result.disposition == Gfn2CudaTopologyStageDisposition::kCandidate);
  CHECK(staging.commit_candidate(error).success());
  const auto replaced = staging.identity();
  CHECK(replaced.committed_generation == 2u);
  CHECK(replaced.committed.packed_base != fixed_identity.committed.packed_base);
  CHECK(snapshot_equals(*staging.committed_snapshot(), changed));
  const std::uint64_t changed_downloads = replaced.full_metadata_downloads;
  result = staging.stage_and_validate(changed_batch, error);
  CHECK(result.success());
  CHECK(result.disposition == Gfn2CudaTopologyStageDisposition::kMatchesCommitted);
  CHECK(staging.identity().full_metadata_downloads == changed_downloads);
  return 0;
}

int exercise_absent_normalization(int device_id, cudaStream_t stream) {
  Topology topology = Topology::make(8);
  topology.point_offsets.assign(9u, 0);
  DeviceTopology device;
  CHECK(device.upload(topology, stream) == 0);
  gpuxtb_batch_t batch = make_batch(topology, device, SourceMode::kMixed, false, false);
  Gfn2CudaTopologyStaging staging(device_id, stream);
  CHECK(staging.valid());
  std::string error;
  auto result = staging.stage_and_validate(batch, error);
  CHECK(result.success());
  CHECK(result.disposition == Gfn2CudaTopologyStageDisposition::kCandidate);
  const auto* snapshot = staging.candidate_snapshot();
  CHECK(snapshot != nullptr);
  CHECK(snapshot_equals(*snapshot, topology, false));
  CHECK(std::all_of(snapshot->point_charge_offsets.begin(), snapshot->point_charge_offsets.end(),
                    [](std::int64_t value) { return value == 0; }));
  CHECK(staging.commit_candidate(error).success());

  /* An explicit all-zero point partition canonicalizes to the same key. */
  batch.point_charge_offsets = host_buffer(topology.point_offsets);
  result = staging.stage_and_validate(batch, error);
  CHECK(result.success());
  CHECK(result.disposition == Gfn2CudaTopologyStageDisposition::kMatchesCommitted);
  return 0;
}

int exercise_layout_replacement(int device_id, cudaStream_t stream) {
  Topology original = Topology::make(8);
  DeviceTopology original_device;
  CHECK(original_device.upload(original, stream) == 0);
  Gfn2CudaTopologyStaging staging(device_id, stream);
  CHECK(staging.valid());
  std::string error;
  auto result =
      staging.stage_and_validate(make_batch(original, original_device, SourceMode::kDevice), error);
  CHECK(result.success());
  CHECK(staging.commit_candidate(error).success());
  const auto original_identity = staging.identity();

  Topology resized = Topology::make(9);
  DeviceTopology resized_device;
  CHECK(resized_device.upload(resized, stream) == 0);
  gpuxtb_batch_t resized_batch = make_batch(resized, resized_device, SourceMode::kMixed);
  result = staging.stage_and_validate(resized_batch, error);
  CHECK(result.success());
  CHECK(result.disposition == Gfn2CudaTopologyStageDisposition::kCandidate);
  CHECK(snapshot_equals(*staging.candidate_snapshot(), resized));
  staging.abort_candidate();
  CHECK(staging.identity().committed.packed_base == original_identity.committed.packed_base);
  CHECK(snapshot_equals(*staging.committed_snapshot(), original));

  result = staging.stage_and_validate(resized_batch, error);
  CHECK(result.success());
  CHECK(staging.commit_candidate(error).success());
  CHECK(staging.identity().committed.packed_base != original_identity.committed.packed_base);
  CHECK(snapshot_equals(*staging.committed_snapshot(), resized));
  return 0;
}

enum class InvalidCase {
  kAtomFirst,
  kAtomEmpty,
  kAtomEndpoint,
  kAtomicNumberLow,
  kAtomicNumberHigh,
  kChargeNan,
  kChargeInf,
  kSpin,
  kPointFirst,
  kPointNonmonotone,
  kPointEndpoint,
  kResponseFirst,
  kResponseShape,
  kResponseEndpoint,
};

void mutate_invalid(Topology& topology, InvalidCase invalid) {
  switch (invalid) {
    case InvalidCase::kAtomFirst:
      topology.atom_offsets[0] = 1;
      break;
    case InvalidCase::kAtomEmpty:
      topology.atom_offsets[1] = topology.atom_offsets[0];
      break;
    case InvalidCase::kAtomEndpoint:
      --topology.atom_offsets.back();
      break;
    case InvalidCase::kAtomicNumberLow:
      topology.atomic_numbers[0] = 0;
      break;
    case InvalidCase::kAtomicNumberHigh:
      topology.atomic_numbers[0] = 119;
      break;
    case InvalidCase::kChargeNan:
      topology.molecular_charges[0] = std::numeric_limits<double>::quiet_NaN();
      break;
    case InvalidCase::kChargeInf:
      topology.molecular_charges[0] = std::numeric_limits<double>::infinity();
      break;
    case InvalidCase::kSpin:
      topology.unpaired_electrons[0] = 1;
      break;
    case InvalidCase::kPointFirst:
      topology.point_offsets[0] = 1;
      break;
    case InvalidCase::kPointNonmonotone:
      topology.point_offsets[2] = topology.point_offsets[1] - 1;
      break;
    case InvalidCase::kPointEndpoint:
      --topology.point_offsets.back();
      break;
    case InvalidCase::kResponseFirst:
      topology.response_offsets[0] = 1;
      break;
    case InvalidCase::kResponseShape:
      ++topology.response_offsets[1];
      break;
    case InvalidCase::kResponseEndpoint:
      --topology.response_offsets.back();
      break;
  }
}

Gfn2CudaTopologyStagingField expected_field(InvalidCase invalid) {
  switch (invalid) {
    case InvalidCase::kAtomFirst:
    case InvalidCase::kAtomEmpty:
    case InvalidCase::kAtomEndpoint:
      return Gfn2CudaTopologyStagingField::kAtomOffsets;
    case InvalidCase::kAtomicNumberLow:
    case InvalidCase::kAtomicNumberHigh:
      return Gfn2CudaTopologyStagingField::kAtomicNumbers;
    case InvalidCase::kChargeNan:
    case InvalidCase::kChargeInf:
      return Gfn2CudaTopologyStagingField::kMolecularCharges;
    case InvalidCase::kSpin:
      return Gfn2CudaTopologyStagingField::kUnpairedElectrons;
    case InvalidCase::kPointFirst:
    case InvalidCase::kPointNonmonotone:
    case InvalidCase::kPointEndpoint:
      return Gfn2CudaTopologyStagingField::kPointChargeOffsets;
    default:
      return Gfn2CudaTopologyStagingField::kChargeResponseOffsets;
  }
}

int exercise_invalid_matrix(int device_id, cudaStream_t stream) {
  const std::vector<InvalidCase> cases = {
      InvalidCase::kAtomFirst,        InvalidCase::kAtomEmpty,
      InvalidCase::kAtomEndpoint,     InvalidCase::kAtomicNumberLow,
      InvalidCase::kAtomicNumberHigh, InvalidCase::kChargeNan,
      InvalidCase::kChargeInf,        InvalidCase::kSpin,
      InvalidCase::kPointFirst,       InvalidCase::kPointNonmonotone,
      InvalidCase::kPointEndpoint,    InvalidCase::kResponseFirst,
      InvalidCase::kResponseShape,    InvalidCase::kResponseEndpoint,
  };
  for (InvalidCase invalid : cases) {
    for (SourceMode mode : {SourceMode::kHost, SourceMode::kDevice, SourceMode::kMixed}) {
      Topology topology = Topology::make(8);
      mutate_invalid(topology, invalid);
      DeviceTopology device;
      CHECK(device.upload(topology, stream) == 0);
      gpuxtb_batch_t batch = make_batch(topology, device, mode);
      /* Declared totals are ABI inline facts and remain unmodified by bad offset bytes. */
      const Topology canonical = Topology::make(8);
      batch.total_point_charges = canonical.point_offsets.back();
      batch.total_charge_response_elements = canonical.response_offsets.back();
      Gfn2CudaTopologyStaging staging(device_id, stream);
      CHECK(staging.valid());
      std::string error;
      auto result = staging.stage_and_validate(batch, error);
      CHECK(!result.success());
      if (result.field != expected_field(invalid)) {
        std::fprintf(stderr, "invalid case=%u mode=%u returned field=%u status=%d error=%s\n",
                     static_cast<unsigned>(invalid), static_cast<unsigned>(mode),
                     static_cast<unsigned>(result.field), result.status, error.c_str());
      }
      CHECK(result.field == expected_field(invalid));
      CHECK(staging.committed_snapshot() == nullptr);
      CHECK(staging.candidate_snapshot() == nullptr);
      CHECK(staging.identity().full_metadata_downloads == 0u);
      if (invalid == InvalidCase::kSpin) {
        CHECK(result.status == GPUXTB_STATUS_NOT_SUPPORTED);
      } else {
        CHECK(result.status == GPUXTB_STATUS_INVALID_ARGUMENT);
      }
    }
  }
  return 0;
}

int exercise_current_device_and_event(int device_id, cudaStream_t stream) {
  int before = -1;
  CUDA_CHECK(cudaGetDevice(&before));
  Topology topology = Topology::make(1);
  DeviceTopology device;
  CHECK(device.upload(topology, stream) == 0);
  {
    Gfn2CudaTopologyStaging staging(device_id, stream);
    CHECK(staging.valid());
    std::string error;
    auto result =
        staging.stage_and_validate(make_batch(topology, device, SourceMode::kMixed), error);
    CHECK(result.success());
    CHECK(staging.commit_candidate(error).success());
    int after_stage = -1;
    CUDA_CHECK(cudaGetDevice(&after_stage));
    CHECK(after_stage == before);

    CHECK(staging.identity().completion_event != 0u);
    CHECK(staging.identity().completion_event_disable_timing == 1u);
  }
  int after_destructor = -1;
  CUDA_CHECK(cudaGetDevice(&after_destructor));
  CHECK(after_destructor == before);
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  CHECK(device_count > 0);
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  for (std::int64_t batch_size : {1, 8, 32, 128}) {
    CHECK(exercise_sources_and_transactions(batch_size, device_id, stream) == 0);
  }
  CHECK(exercise_absent_normalization(device_id, stream) == 0);
  CHECK(exercise_layout_replacement(device_id, stream) == 0);
  CHECK(exercise_invalid_matrix(device_id, stream) == 0);
  CHECK(exercise_current_device_and_event(device_id, stream) == 0);

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
