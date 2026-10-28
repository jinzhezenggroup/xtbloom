#include <cuda_runtime.h>

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "backends/cuda/gfn2_scc_diagnostics.cuh"

namespace {

using namespace xtbloom::detail::cuda;

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename Element>
class DeviceArray {
 public:
  explicit DeviceArray(const std::vector<Element>& values) : count_(values.size()) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_), count_ * sizeof(Element)));
    try {
      write(values);
    } catch (...) {
      (void)cudaFree(pointer_);
      throw;
    }
  }
  DeviceArray(std::initializer_list<Element> values) : DeviceArray(std::vector<Element>(values)) {}
  ~DeviceArray() { (void)cudaFree(pointer_); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  Element* get() const { return pointer_; }
  void write(const std::vector<Element>& values) {
    check(values.size() == count_, "test buffer extent changed");
    check_cuda(
        cudaMemcpy(pointer_, values.data(), count_ * sizeof(Element), cudaMemcpyHostToDevice));
  }
  std::vector<Element> read() const {
    std::vector<Element> values(count_);
    check_cuda(
        cudaMemcpy(values.data(), pointer_, count_ * sizeof(Element), cudaMemcpyDeviceToHost));
    return values;
  }

 private:
  Element* pointer_ = nullptr;
  std::size_t count_ = 0u;
};

void exercise() {
  DeviceArray<Gfn2SccDiagnosticHeader> header({{}});
  DeviceArray<Gfn2SccDiagnosticIteration> records(std::vector<Gfn2SccDiagnosticIteration>(2));
  DeviceArray<Gfn2SccDiagnosticBucket> rows(std::vector<Gfn2SccDiagnosticBucket>(4));
  Gfn2EigensolverBucket first{};
  first.orbital_count = 4;
  first.system_count = 1;
  first.solve_count = 1;
  Gfn2EigensolverBucket second{};
  second.orbital_count = 5;
  second.system_count = 1;
  second.system_index_offset = 1;
  second.solve_count = 2;
  DeviceArray<Gfn2EigensolverBucket> buckets({first, second});
  DeviceArray<std::int32_t> systems({0, 1});
  DeviceArray<std::int32_t> spins({1, 2});
  DeviceArray<std::uint8_t> active({1u, 1u});
  DeviceArray<std::uint32_t> sequence({1u});
  DeviceArray<std::uint64_t> failure({0u});
  DeviceArray<std::uint64_t> iterations({0u, 0u});
  DeviceArray<std::uint8_t> converged({0u, 0u});
  DeviceArray<xtbloom_status_t> statuses({XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS});
  DeviceArray<Gfn2EigensolverBucketActivity> activity({{1u, 1u, 1u, 1u}, {1u, 2u, 2u, 1u}});
  const Gfn2SccDiagnosticsDevice diagnostics{
      header.get(), records.get(), rows.get(), buckets.get(), 2, 2u};
  Gfn2SccIterationBinding binding{};
  binding.workspace.ledger.active_mask = active.get();
  binding.workspace.ledger.sequence_active = sequence.get();
  binding.workspace.ledger.plan_failure_record = failure.get();
  binding.workspace.ledger.batch_elements = 2;
  binding.workspace.eigensolver_workspace.bucket_activity = activity.get();
  binding.plan.eigensolver_batch.bucket_systems = systems.get();
  binding.plan.wavefunction_layout.spin_channels = spins.get();
  binding.input.activity_state = {iterations.get(), statuses.get(), converged.get(), 2, 1u};

  check_cuda(reset_gfn2_scc_diagnostics_cuda(diagnostics, nullptr));
  check_cuda(begin_gfn2_scc_diagnostics_cuda(diagnostics, binding, nullptr));
  check_cuda(finish_gfn2_scc_diagnostics_cuda(diagnostics, binding, false, nullptr));
  auto observed = rows.read();
  check(header.read()[0].iteration_count == 1u, "full activity did not produce one row");
  check(observed[0].active_systems == 1u && observed[0].active_channels == 1u,
        "restricted activity count");
  check(observed[1].active_systems == 1u && observed[1].active_channels == 2u,
        "mixed-spin activity must count channels separately");
  check(observed[0].submitted_solver_slots == 1u && observed[1].submitted_solver_slots == 2u,
        "fixed-capacity solver submission");

  active.write({0u, 1u});
  converged.write({1u, 0u});
  iterations.write({1u, 1u});
  check_cuda(begin_gfn2_scc_diagnostics_cuda(diagnostics, binding, nullptr));
  statuses.write({XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_EIGENSOLVER_FAILED});
  check_cuda(finish_gfn2_scc_diagnostics_cuda(diagnostics, binding, true, nullptr));
  observed = rows.read();
  check(observed[2].active_systems == 0u && observed[2].converged_systems == 1u,
        "converged peer must not become active");
  check(observed[3].active_channels == 2u && observed[3].failed_systems == 1u,
        "one system failure must remain one, not two channel failures");
  const auto timing = records.read();
  check(timing[0].start_ns <= timing[0].end_ns && timing[0].end_ns <= timing[1].start_ns &&
            timing[1].start_ns <= timing[1].end_ns,
        "numerical intervals overlap or use incompatible clocks");

  check_cuda(begin_gfn2_scc_diagnostics_cuda(diagnostics, binding, nullptr));
  check_cuda(finish_gfn2_scc_diagnostics_cuda(diagnostics, binding, true, nullptr));
  check(header.read()[0].overflow == 1u && header.read()[0].iteration_count == 2u,
        "overflow must be explicit and must not overwrite rows");

  check_cuda(reset_gfn2_scc_diagnostics_cuda(diagnostics, nullptr));
  active.write({0u, 0u});
  check_cuda(begin_gfn2_scc_diagnostics_cuda(diagnostics, binding, nullptr));
  check_cuda(finish_gfn2_scc_diagnostics_cuda(diagnostics, binding, false, nullptr));
  check(header.read()[0].iteration_count == 0u, "terminal replay must not reuse old rows");

  check_cuda(begin_gfn2_scc_diagnostics_cuda(diagnostics, binding, nullptr, true));
  check_cuda(finish_gfn2_scc_diagnostics_cuda(diagnostics, binding, false, nullptr));
  observed = rows.read();
  check(header.read()[0].iteration_count == 1u && observed[0].active_systems == 0u &&
            observed[1].active_channels == 0u && observed[0].submitted_solver_slots == 1u &&
            observed[1].submitted_solver_slots == 2u,
        "bounded inactive provider submissions must remain visible");
  check_cuda(reset_gfn2_scc_diagnostics_cuda(diagnostics, nullptr));

  active.write({1u, 1u});
  statuses.write({XTBLOOM_STATUS_SCC_NOT_CONVERGED, XTBLOOM_STATUS_SUCCESS});
  converged.write({0u, 0u});
  iterations.write({2u, 2u});
  check_cuda(begin_gfn2_scc_diagnostics_cuda(diagnostics, binding, nullptr));
  failure.write({17u});
  check_cuda(finish_gfn2_scc_diagnostics_cuda(diagnostics, binding, true, nullptr));
  observed = rows.read();
  check(observed[0].exhausted_systems == 1u && observed[1].exhausted_systems == 1u,
        "iteration limit is disjoint from numerical peer failure");
  check(observed[0].submitted_solver_slots == kGfn2SccDiagnosticUnknown &&
            observed[1].submitted_backtransform_slots == kGfn2SccDiagnosticUnknown &&
            records.read()[0].plan_failure_record == 17u,
        "interrupted stage must not publish stale capacity telemetry");

  failure.write({0u});
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  check_cuda(reset_gfn2_scc_diagnostics_cuda(diagnostics, stream));
  check_cuda(begin_gfn2_scc_diagnostics_cuda(diagnostics, binding, stream));
  check_cuda(finish_gfn2_scc_diagnostics_cuda(diagnostics, binding, false, stream));
  check_cuda(cudaStreamEndCapture(stream, &graph));
  check_cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0u));
  for (int replay = 0; replay < 2; ++replay) {
    check_cuda(cudaGraphLaunch(executable, stream));
    check_cuda(cudaStreamSynchronize(stream));
    check(header.read()[0].iteration_count == 1u, "Graph replay must reset diagnostics");
  }
  check_cuda(cudaGraphExecDestroy(executable));
  check_cuda(cudaGraphDestroy(graph));
  check_cuda(cudaStreamDestroy(stream));

  auto empty = diagnostics;
  empty.bucket_count = 0;
  empty.buckets = nullptr;
  empty.rows = nullptr;
  binding.workspace.ledger.batch_elements = 0;
  check_cuda(reset_gfn2_scc_diagnostics_cuda(empty, nullptr));
  check_cuda(begin_gfn2_scc_diagnostics_cuda(empty, binding, nullptr));
  check_cuda(finish_gfn2_scc_diagnostics_cuda(empty, binding, false, nullptr));
  check(header.read()[0].iteration_count == 0u, "empty batch must not touch bucket storage");
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    exercise();
    std::cout << "CUDA SCC diagnostic hand-count and replay checks passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 1;
  }
}
