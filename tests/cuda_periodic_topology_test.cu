// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "backends/cuda/periodic_topology.cuh"
#include "model/gfn2/lattice.hpp"

#define CHECK(condition)                                                                     \
  do {                                                                                       \
    if (!(condition)) {                                                                      \
      std::fprintf(stderr, "CUDA periodic-topology check failed at line %d: %s\n", __LINE__, \
                   #condition);                                                              \
      return __LINE__;                                                                       \
    }                                                                                        \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::Gfn2CudaPeriodicGraphIdentity;
using xtbloom::detail::cuda::Gfn2CudaPeriodicTopology;
using xtbloom::detail::cuda::Gfn2CudaPeriodicTopologyDeviceDiagnostic;
using xtbloom::detail::cuda::Gfn2CudaPeriodicTopologyError;
using xtbloom::detail::cuda::Gfn2CudaPeriodicTopologyField;
using xtbloom::detail::cuda::Gfn2CudaPeriodicTopologyInput;
using xtbloom::detail::cuda::Gfn2CudaPeriodicTranslation;
using xtbloom::detail::cuda::same_gfn2_cuda_periodic_graph_identity;
using xtbloom::detail::gfn2::Lattice3D;
using xtbloom::detail::gfn2::LatticeOriginPolicy;
using xtbloom::detail::gfn2::LatticeTranslation;
using xtbloom::detail::gfn2::make_lattice_3d;
using xtbloom::detail::gfn2::make_lattice_translations;

constexpr std::uint64_t kPlanToken = 0x5a17c0de5a17c0deULL;
constexpr std::uint64_t kCellGeneration = 17u;
constexpr double kCutoff = 5.0;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }

  cudaError_t allocate(std::size_t count) {
    if (data_ != nullptr) return cudaErrorInvalidValue;
    count_ = count;
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

Gfn2CudaPeriodicTranslation convert_translation(const LatticeTranslation& source) {
  Gfn2CudaPeriodicTranslation result{};
  for (int component = 0; component < 3; ++component) {
    result.index[component] = source.index[component];
    result.cartesian[component] =
        source.cartesian[component] == 0.0 ? 0.0 : source.cartesian[component];
  }
  return result;
}

bool same_translation(const Gfn2CudaPeriodicTranslation& first,
                      const Gfn2CudaPeriodicTranslation& second) {
  for (int component = 0; component < 3; ++component) {
    if (first.index[component] != second.index[component] ||
        first.cartesian[component] != second.cartesian[component]) {
      return false;
    }
  }
  return true;
}

bool read_diagnostic(DeviceBuffer<Gfn2CudaPeriodicTopologyDeviceDiagnostic>& device,
                     Gfn2CudaPeriodicTopologyDeviceDiagnostic& host, cudaStream_t stream) {
  return cudaMemcpyAsync(&host, device.get(), sizeof(host), cudaMemcpyDeviceToHost, stream) ==
             cudaSuccess &&
         cudaStreamSynchronize(stream) == cudaSuccess;
}

int test_canonical_upload_and_device_schema(cudaStream_t stream) {
  const std::vector<std::int64_t> atom_offsets{0, 2, 3};
  const std::vector<std::int32_t> periodic_axes{XTBLOOM_PERIODIC_AXES_XYZ,
                                                XTBLOOM_PERIODIC_AXES_NONE};
  const std::array<double, 18> cells{
      9.0, 0.0, 0.0, 1.0, 10.0, 0.0, 0.5, 0.75, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  };
  const Gfn2CudaPeriodicTopologyInput input{
      2,       3,          atom_offsets.data(), cells.data(), periodic_axes.data(),
      kCutoff, kPlanToken, kCellGeneration};

  Gfn2CudaPeriodicTopology topology;
  const auto created = Gfn2CudaPeriodicTopology::create(input, stream, topology);
  CHECK(created.success());
  CHECK(topology.valid());
  const auto view = topology.device_view();
  CHECK(view.batch_size == 2);
  CHECK(view.total_atoms == 3);
  CHECK(view.atom_offset_count == 3);
  CHECK(view.cell_elements == 18);
  CHECK(view.periodic_axes_elements == 2);
  CHECK(view.translation_offset_count == 3);
  CHECK(view.total_translations > 1);

  Lattice3D lattice;
  std::string error;
  CHECK(make_lattice_3d(cells.data(), lattice, error) == XTBLOOM_STATUS_SUCCESS);
  std::vector<LatticeTranslation> expected_lattice;
  CHECK(make_lattice_translations(lattice, kCutoff, LatticeOriginPolicy::kInclude, expected_lattice,
                                  error) == XTBLOOM_STATUS_SUCCESS);
  std::vector<Gfn2CudaPeriodicTranslation> expected;
  expected.reserve(expected_lattice.size() + 1u);
  for (const LatticeTranslation& translation : expected_lattice) {
    expected.push_back(convert_translation(translation));
  }
  expected.push_back({});

  std::vector<std::int64_t> actual_atom_offsets(atom_offsets.size());
  std::vector<double> actual_cells(cells.size());
  std::vector<std::int32_t> actual_axes(periodic_axes.size());
  std::vector<std::int64_t> actual_translation_offsets(3u);
  std::vector<Gfn2CudaPeriodicTranslation> actual_translations(expected.size());
  CUDA_CHECK(cudaMemcpyAsync(actual_atom_offsets.data(), view.atom_offsets,
                             actual_atom_offsets.size() * sizeof(std::int64_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(actual_cells.data(), view.cell_matrices,
                             actual_cells.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(actual_axes.data(), view.periodic_axes,
                             actual_axes.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(actual_translation_offsets.data(), view.translation_offsets,
                             actual_translation_offsets.size() * sizeof(std::int64_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(actual_translations.data(), view.translations,
                             actual_translations.size() * sizeof(Gfn2CudaPeriodicTranslation),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(actual_atom_offsets == atom_offsets);
  CHECK(actual_cells == std::vector<double>(cells.begin(), cells.end()));
  CHECK(actual_axes == periodic_axes);
  CHECK(actual_translation_offsets[0] == 0);
  CHECK(actual_translation_offsets[1] == static_cast<std::int64_t>(expected_lattice.size()));
  CHECK(actual_translation_offsets[2] == static_cast<std::int64_t>(expected.size()));
  CHECK(actual_translations.size() == expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    CHECK(same_translation(actual_translations[index], expected[index]));
  }

  DeviceBuffer<Gfn2CudaPeriodicTopologyDeviceDiagnostic> diagnostic;
  CUDA_CHECK(diagnostic.allocate(1u));
  auto launch = Gfn2CudaPeriodicTopology::validate(view, diagnostic.get(), stream);
  CHECK(launch.success());
  Gfn2CudaPeriodicTopologyDeviceDiagnostic host_diagnostic{};
  CHECK(read_diagnostic(diagnostic, host_diagnostic, stream));
  CHECK(host_diagnostic.error ==
        static_cast<std::uint32_t>(Gfn2CudaPeriodicTopologyError::kSuccess));
  CHECK(host_diagnostic.field == static_cast<std::uint32_t>(Gfn2CudaPeriodicTopologyField::kNone));
  CHECK(host_diagnostic.index == -1);

  /* The validator is allocation-free and can be captured in a reusable Graph. */
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch = Gfn2CudaPeriodicTopology::validate(view, diagnostic.get(), stream);
  CHECK(launch.success());
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphLaunch(executable, stream));
  CHECK(read_diagnostic(diagnostic, host_diagnostic, stream));
  CHECK(host_diagnostic.error ==
        static_cast<std::uint32_t>(Gfn2CudaPeriodicTopologyError::kSuccess));
  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));

  const Gfn2CudaPeriodicGraphIdentity base_identity =
      topology.graph_identity(23u, XTBLOOM_RESULT_DIPOLE_MOMENTS, true, 0u);
  CHECK(same_gfn2_cuda_periodic_graph_identity(
      base_identity, topology.graph_identity(23u, XTBLOOM_RESULT_DIPOLE_MOMENTS, true, 0u)));
  CHECK(!same_gfn2_cuda_periodic_graph_identity(
      base_identity, topology.graph_identity(24u, XTBLOOM_RESULT_DIPOLE_MOMENTS, true, 0u)));
  CHECK(!same_gfn2_cuda_periodic_graph_identity(
      base_identity, topology.graph_identity(23u, XTBLOOM_RESULT_DIPOLE_MOMENTS, false, 0u)));
  CHECK(!same_gfn2_cuda_periodic_graph_identity(base_identity,
                                                topology.graph_identity(23u, 0u, true, 0u)));
  CHECK(!same_gfn2_cuda_periodic_graph_identity(
      base_identity, topology.graph_identity(23u, XTBLOOM_RESULT_DIPOLE_MOMENTS, true, 1u)));

  /* A failed replacement must not invalidate the committed device topology. */
  const auto committed_view = topology.device_view();
  const auto committed_identity = topology.graph_identity(23u, 0u, false, 0u);
  const std::vector<std::int32_t> partial_axes{XTBLOOM_PERIODIC_AXES_XYZ, XTBLOOM_PERIODIC_AXIS_X};
  Gfn2CudaPeriodicTopologyInput rejected_input = input;
  rejected_input.periodic_axes = partial_axes.data();
  const auto rejected = Gfn2CudaPeriodicTopology::create(rejected_input, stream, topology);
  CHECK(!rejected.success());
  CHECK(rejected.status == XTBLOOM_STATUS_NOT_SUPPORTED);
  CHECK(rejected.error == Gfn2CudaPeriodicTopologyError::kUnsupportedPeriodicity);
  CHECK(topology.device_view().atom_offsets == committed_view.atom_offsets);
  CHECK(topology.device_view().translations == committed_view.translations);
  CHECK(same_gfn2_cuda_periodic_graph_identity(committed_identity,
                                               topology.graph_identity(23u, 0u, false, 0u)));

  /* Device-side validation catches corruption after setup without host reads. */
  const std::int32_t invalid_axis = XTBLOOM_PERIODIC_AXIS_X;
  CUDA_CHECK(cudaMemcpyAsync(const_cast<std::int32_t*>(view.periodic_axes), &invalid_axis,
                             sizeof(invalid_axis), cudaMemcpyHostToDevice, stream));
  launch = Gfn2CudaPeriodicTopology::validate(view, diagnostic.get(), stream);
  CHECK(launch.success());
  CHECK(read_diagnostic(diagnostic, host_diagnostic, stream));
  CHECK(host_diagnostic.error ==
        static_cast<std::uint32_t>(Gfn2CudaPeriodicTopologyError::kUnsupportedPeriodicity));
  CHECK(host_diagnostic.field ==
        static_cast<std::uint32_t>(Gfn2CudaPeriodicTopologyField::kPeriodicAxes));
  CHECK(host_diagnostic.index == 0);
  return 0;
}

int test_distinct_cell_fingerprint(cudaStream_t stream) {
  const std::vector<std::int64_t> atom_offsets{0, 1};
  const std::vector<std::int32_t> axes{XTBLOOM_PERIODIC_AXES_XYZ};
  const std::array<double, 9> first_cell{8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0};
  std::array<double, 9> second_cell = first_cell;
  second_cell[0] = 9.0;
  Gfn2CudaPeriodicTopologyInput first_input{
      1, 1, atom_offsets.data(), first_cell.data(), axes.data(), kCutoff, kPlanToken + 1u, 33u};
  Gfn2CudaPeriodicTopologyInput second_input{
      1, 1, atom_offsets.data(), second_cell.data(), axes.data(), kCutoff, kPlanToken + 2u, 34u};
  Gfn2CudaPeriodicTopology first;
  Gfn2CudaPeriodicTopology second;
  CHECK(Gfn2CudaPeriodicTopology::create(first_input, stream, first).success());
  CHECK(Gfn2CudaPeriodicTopology::create(second_input, stream, second).success());
  CHECK(first.topology_fingerprint() != second.topology_fingerprint());
  CHECK(first.graph_identity(1u, 0u, false, 0u).topology_fingerprint !=
        second.graph_identity(1u, 0u, false, 0u).topology_fingerprint);
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 1;
  const int first = test_canonical_upload_and_device_schema(stream);
  const int second = first == 0 ? test_distinct_cell_fingerprint(stream) : first;
  (void)cudaStreamDestroy(stream);
  return second;
}
