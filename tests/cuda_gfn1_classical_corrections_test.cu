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

#include "backends/cuda/gfn1_classical_corrections.cuh"

namespace {

using xtbloom::detail::XtbModelFlavor;
using xtbloom::detail::cuda::add_gfn1_classical_corrections_cuda;
using xtbloom::detail::cuda::Gfn1ClassicalCorrectionDeviceError;
using xtbloom::detail::cuda::Gfn1ClassicalCorrectionDevicePlan;
using xtbloom::detail::cuda::Gfn1ClassicalCorrectionDeviceWorkspace;
using xtbloom::detail::cuda::kGfn1D3MaximumReferences;
using xtbloom::detail::cuda::kGfn1D3ReferencePairStride;

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return __LINE__;                                                                     \
    }                                                                                      \
  } while (false)

constexpr std::uint64_t kPlanToken = UINT64_C(0x386439d3);
constexpr double kEnergySeed0 = 0.5;
constexpr double kEnergySeed1 = -0.25;
constexpr double kGradientSeed = 0.125;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t elements) { require(allocate(elements)); }
  ~DeviceBuffer() { cudaFree(pointer_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  cudaError_t allocate(std::size_t elements) {
    cudaFree(pointer_);
    pointer_ = nullptr;
    elements_ = elements;
    return elements == 0u ? cudaSuccess
                          : cudaMalloc(reinterpret_cast<void**>(&pointer_), elements * sizeof(T));
  }

  cudaError_t upload(const std::vector<T>& values, cudaStream_t stream = nullptr) {
    if (values.size() != elements_) return cudaErrorInvalidValue;
    return values.empty() ? cudaSuccess
                          : cudaMemcpyAsync(pointer_, values.data(), values.size() * sizeof(T),
                                            cudaMemcpyHostToDevice, stream);
  }

  cudaError_t download(std::vector<T>& values, cudaStream_t stream = nullptr) const {
    values.resize(elements_);
    return values.empty() ? cudaSuccess
                          : cudaMemcpyAsync(values.data(), pointer_, values.size() * sizeof(T),
                                            cudaMemcpyDeviceToHost, stream);
  }

  [[nodiscard]] T* get() const noexcept { return pointer_; }

 private:
  static void require(cudaError_t status) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA allocation failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }

  T* pointer_ = nullptr;
  std::size_t elements_ = 0u;
};

struct Fixture {
  std::vector<std::int64_t> atom_offsets{0, 2, 4};
  std::vector<std::int64_t> pair_offsets{0, 1, 2};
  std::vector<double> covalent_radii{1.0, 1.0, 1.0, 1.0};
  std::vector<std::uint8_t> reference_counts{1u, 1u, 1u, 1u};
  std::vector<double> reference_cn =
      std::vector<double>(4u * static_cast<std::size_t>(kGfn1D3MaximumReferences), 0.0);
  std::vector<double> reference_c6 =
      std::vector<double>(2u * static_cast<std::size_t>(kGfn1D3ReferencePairStride), 0.0);
  std::vector<double> pair_rrij{1.0, 1.0};
  std::vector<double> pair_damping_radii{1.0, 1.0};
  std::vector<double> halogen_scaled_radii{1.0, 1.0, 1.0, 1.0};
  std::vector<double> halogen_strength{0.0, 0.0, 0.0, 0.0};
  std::vector<std::uint8_t> halogen_donor{0u, 0u, 0u, 0u};
  std::vector<std::uint8_t> halogen_acceptor{0u, 0u, 0u, 0u};
  std::vector<double> positions{0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0};
  std::vector<double> coordination{0.0, 0.0, 0.0, 0.0};
  std::vector<std::uint8_t> active{1u, 1u};
  std::vector<double> energy_seed{kEnergySeed0, kEnergySeed1};
  std::vector<double> gradient_seed = std::vector<double>(12u, kGradientSeed);

  DeviceBuffer<std::int64_t> d_atom_offsets{atom_offsets.size()};
  DeviceBuffer<std::int64_t> d_pair_offsets{pair_offsets.size()};
  DeviceBuffer<double> d_covalent_radii{covalent_radii.size()};
  DeviceBuffer<std::uint8_t> d_reference_counts{reference_counts.size()};
  DeviceBuffer<double> d_reference_cn{reference_cn.size()};
  DeviceBuffer<double> d_reference_c6{reference_c6.size()};
  DeviceBuffer<double> d_pair_rrij{pair_rrij.size()};
  DeviceBuffer<double> d_pair_damping_radii{pair_damping_radii.size()};
  DeviceBuffer<double> d_halogen_scaled_radii{halogen_scaled_radii.size()};
  DeviceBuffer<double> d_halogen_strength{halogen_strength.size()};
  DeviceBuffer<std::uint8_t> d_halogen_donor{halogen_donor.size()};
  DeviceBuffer<std::uint8_t> d_halogen_acceptor{halogen_acceptor.size()};
  DeviceBuffer<double> d_positions{positions.size()};
  DeviceBuffer<double> d_coordination{coordination.size()};
  DeviceBuffer<std::uint8_t> d_active{active.size()};
  DeviceBuffer<double> d_energies{energy_seed.size()};
  DeviceBuffer<double> d_gradients{gradient_seed.size()};
  DeviceBuffer<double> d_weights{reference_cn.size()};
  DeviceBuffer<double> d_weight_derivatives{reference_cn.size()};
  DeviceBuffer<double> d_coordination_adjoints{coordination.size()};
  DeviceBuffer<std::int64_t> d_axis_neighbors{coordination.size()};
  DeviceBuffer<double> d_batch_scratch{energy_seed.size()};
  DeviceBuffer<double> d_gradient_scratch{gradient_seed.size()};
  DeviceBuffer<std::uint32_t> d_system_errors{energy_seed.size()};
  DeviceBuffer<std::uint32_t> d_plan_error{1u};

  Gfn1ClassicalCorrectionDevicePlan plan{};
  Gfn1ClassicalCorrectionDeviceWorkspace workspace{};

  Fixture() {
    reference_c6[0] = 1.0;
    reference_c6[static_cast<std::size_t>(kGfn1D3ReferencePairStride)] = 1.0;
    require(upload_static());
    bind();
    require(reset());
  }

  cudaError_t upload_static(cudaStream_t stream = nullptr) {
    const std::array<cudaError_t, 16> statuses{
        d_atom_offsets.upload(atom_offsets, stream),
        d_pair_offsets.upload(pair_offsets, stream),
        d_covalent_radii.upload(covalent_radii, stream),
        d_reference_counts.upload(reference_counts, stream),
        d_reference_cn.upload(reference_cn, stream),
        d_reference_c6.upload(reference_c6, stream),
        d_pair_rrij.upload(pair_rrij, stream),
        d_pair_damping_radii.upload(pair_damping_radii, stream),
        d_halogen_scaled_radii.upload(halogen_scaled_radii, stream),
        d_halogen_strength.upload(halogen_strength, stream),
        d_halogen_donor.upload(halogen_donor, stream),
        d_halogen_acceptor.upload(halogen_acceptor, stream),
        d_positions.upload(positions, stream),
        d_coordination.upload(coordination, stream),
        d_active.upload(active, stream),
        cudaSuccess};
    for (cudaError_t status : statuses) {
      if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
  }

  void bind() {
    plan = {static_cast<std::int64_t>(atom_offsets.size() - 1u),
            static_cast<std::int64_t>(covalent_radii.size()),
            static_cast<std::int64_t>(pair_rrij.size()),
            kPlanToken,
            d_atom_offsets.get(),
            d_pair_offsets.get(),
            d_covalent_radii.get(),
            d_reference_counts.get(),
            d_reference_cn.get(),
            d_reference_c6.get(),
            d_pair_rrij.get(),
            d_pair_damping_radii.get(),
            d_halogen_scaled_radii.get(),
            d_halogen_strength.get(),
            d_halogen_donor.get(),
            d_halogen_acceptor.get(),
            XtbModelFlavor::kGfn1,
            static_cast<std::int64_t>(atom_offsets.size()),
            static_cast<std::int64_t>(pair_offsets.size()),
            static_cast<std::int64_t>(covalent_radii.size()),
            static_cast<std::int64_t>(reference_counts.size()),
            static_cast<std::int64_t>(reference_cn.size()),
            static_cast<std::int64_t>(reference_c6.size()),
            static_cast<std::int64_t>(pair_rrij.size()),
            static_cast<std::int64_t>(pair_damping_radii.size()),
            static_cast<std::int64_t>(halogen_scaled_radii.size()),
            static_cast<std::int64_t>(halogen_strength.size()),
            static_cast<std::int64_t>(halogen_donor.size()),
            static_cast<std::int64_t>(halogen_acceptor.size())};
    workspace = {d_weights.get(),
                 d_weight_derivatives.get(),
                 d_coordination_adjoints.get(),
                 d_axis_neighbors.get(),
                 d_batch_scratch.get(),
                 d_gradient_scratch.get(),
                 kPlanToken,
                 static_cast<std::int64_t>(reference_cn.size()),
                 static_cast<std::int64_t>(reference_cn.size()),
                 static_cast<std::int64_t>(coordination.size()),
                 static_cast<std::int64_t>(coordination.size()),
                 static_cast<std::int64_t>(energy_seed.size()),
                 static_cast<std::int64_t>(gradient_seed.size())};
  }

  cudaError_t configure_three_atom_halogen_system() {
    atom_offsets = {0, 3, 4};
    pair_offsets = {0, 3, 3};
    pair_rrij.assign(3u, 1.0);
    pair_damping_radii.assign(3u, 1.0);
    reference_c6.assign(3u * static_cast<std::size_t>(kGfn1D3ReferencePairStride), 0.0);
    for (std::size_t pair = 0u; pair < 3u; ++pair) {
      reference_c6[pair * static_cast<std::size_t>(kGfn1D3ReferencePairStride)] = 1.0;
    }
    positions = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 4.0, 4.0, 4.0};
    halogen_donor = {1u, 0u, 0u, 0u};
    halogen_acceptor = {0u, 0u, 1u, 0u};
    halogen_strength.assign(4u, 0.0);

    cudaError_t status = d_reference_c6.allocate(reference_c6.size());
    if (status == cudaSuccess) status = d_pair_rrij.allocate(pair_rrij.size());
    if (status == cudaSuccess) {
      status = d_pair_damping_radii.allocate(pair_damping_radii.size());
    }
    if (status == cudaSuccess) status = upload_static();
    if (status != cudaSuccess) return status;
    bind();
    return reset();
  }

  cudaError_t reset(cudaStream_t stream = nullptr) {
    cudaError_t status = d_energies.upload(energy_seed, stream);
    if (status == cudaSuccess) status = d_gradients.upload(gradient_seed, stream);
    if (status == cudaSuccess) status = d_active.upload(active, stream);
    if (status == cudaSuccess) {
      status = cudaMemsetAsync(d_system_errors.get(), 0, 2u * sizeof(std::uint32_t), stream);
    }
    if (status == cudaSuccess) {
      status = cudaMemsetAsync(d_plan_error.get(), 0, sizeof(std::uint32_t), stream);
    }
    return status;
  }

  cudaError_t launch(double* energies = nullptr, double* gradients = nullptr,
                     cudaStream_t stream = nullptr) {
    return add_gfn1_classical_corrections_cuda(
        plan, d_positions.get(), d_coordination.get(), d_active.get(),
        energies == nullptr ? d_energies.get() : energies,
        gradients == nullptr ? d_gradients.get() : gradients, workspace, d_system_errors.get(),
        d_plan_error.get(), stream);
  }

 private:
  static void require(cudaError_t status) {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "CUDA fixture setup failed: %s\n", cudaGetErrorString(status));
      std::abort();
    }
  }
};

double d3_pair_energy(double distance, double c6 = 1.0) {
  const double r2 = distance * distance;
  const double r4 = r2 * r2;
  return -c6 * (1.0 / (r4 * r2 + 1.0) + 2.4 / (r4 * r4 + 1.0));
}

int test_energy_gradient_and_finite_difference() {
  Fixture fixture;
  CHECK(fixture.launch() == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<double> energies;
  std::vector<double> gradients;
  std::vector<std::uint32_t> errors;
  CHECK(fixture.d_energies.download(energies) == cudaSuccess);
  CHECK(fixture.d_gradients.download(gradients) == cudaSuccess);
  CHECK(fixture.d_system_errors.download(errors) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(errors == std::vector<std::uint32_t>({0u, 0u}));
  CHECK(std::abs(energies[0] - (kEnergySeed0 + d3_pair_energy(2.0))) < 1.0e-13);
  CHECK(std::abs(energies[1] - (kEnergySeed1 + d3_pair_energy(3.0))) < 1.0e-13);
  constexpr double step = 2.0e-5;
  const double finite_difference =
      (d3_pair_energy(2.0 + step) - d3_pair_energy(2.0 - step)) / (2.0 * step);
  CHECK(std::abs((gradients[3] - kGradientSeed) - finite_difference) < 2.0e-9);
  CHECK(std::abs((gradients[0] - kGradientSeed) + finite_difference) < 2.0e-9);
  return 0;
}

int test_peer_failure_is_transactional() {
  Fixture fixture;
  fixture.reference_c6[static_cast<std::size_t>(kGfn1D3ReferencePairStride)] =
      std::numeric_limits<double>::quiet_NaN();
  CHECK(fixture.d_reference_c6.upload(fixture.reference_c6) == cudaSuccess);
  CHECK(fixture.reset() == cudaSuccess);
  CHECK(fixture.launch() == cudaSuccess);
  std::vector<double> energies;
  std::vector<double> gradients;
  std::vector<std::uint32_t> errors;
  CHECK(fixture.d_energies.download(energies) == cudaSuccess);
  CHECK(fixture.d_gradients.download(gradients) == cudaSuccess);
  CHECK(fixture.d_system_errors.download(errors) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(errors[0] == 0u);
  CHECK(errors[1] ==
        static_cast<std::uint32_t>(Gfn1ClassicalCorrectionDeviceError::kNonfiniteInput));
  CHECK(energies[0] != kEnergySeed0 && energies[1] == kEnergySeed1);
  CHECK(std::all_of(gradients.begin() + 6, gradients.end(),
                    [](double value) { return value == kGradientSeed; }));
  return 0;
}

int test_nonfinite_reference_cn_is_transactional() {
  Fixture fixture;
  fixture.reference_cn[2u * static_cast<std::size_t>(kGfn1D3MaximumReferences)] =
      std::numeric_limits<double>::quiet_NaN();
  CHECK(fixture.d_reference_cn.upload(fixture.reference_cn) == cudaSuccess);
  CHECK(fixture.reset() == cudaSuccess);
  CHECK(fixture.launch() == cudaSuccess);
  std::vector<double> energies;
  std::vector<double> gradients;
  std::vector<std::uint32_t> errors;
  CHECK(fixture.d_energies.download(energies) == cudaSuccess);
  CHECK(fixture.d_gradients.download(gradients) == cudaSuccess);
  CHECK(fixture.d_system_errors.download(errors) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(errors[0] == 0u);
  CHECK(errors[1] ==
        static_cast<std::uint32_t>(Gfn1ClassicalCorrectionDeviceError::kNonfiniteInput));
  CHECK(energies[0] != kEnergySeed0 && energies[1] == kEnergySeed1);
  CHECK(std::all_of(gradients.begin() + 6, gradients.end(),
                    [](double value) { return value == kGradientSeed; }));
  return 0;
}

int test_active_mask_and_gradient_overflow() {
  Fixture fixture;
  fixture.active = {0u, 2u};
  CHECK(fixture.reset() == cudaSuccess);
  CHECK(fixture.launch() == cudaSuccess);
  std::vector<double> energies;
  std::vector<double> gradients;
  std::vector<std::uint32_t> errors;
  CHECK(fixture.d_energies.download(energies) == cudaSuccess);
  CHECK(fixture.d_gradients.download(gradients) == cudaSuccess);
  CHECK(fixture.d_system_errors.download(errors) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(energies == fixture.energy_seed && gradients == fixture.gradient_seed);
  CHECK(errors[0] == 0u);
  CHECK(errors[1] ==
        static_cast<std::uint32_t>(Gfn1ClassicalCorrectionDeviceError::kInvalidActiveMask));

  fixture.active = {1u, 0u};
  fixture.reference_c6[0] = 1.0e308;
  fixture.gradient_seed.assign(12u, 0.0);
  fixture.gradient_seed[0] = -std::numeric_limits<double>::max();
  CHECK(fixture.d_reference_c6.upload(fixture.reference_c6) == cudaSuccess);
  CHECK(fixture.reset() == cudaSuccess);
  CHECK(fixture.launch() == cudaSuccess);
  CHECK(fixture.d_energies.download(energies) == cudaSuccess);
  CHECK(fixture.d_gradients.download(gradients) == cudaSuccess);
  CHECK(fixture.d_system_errors.download(errors) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(errors[0] ==
        static_cast<std::uint32_t>(Gfn1ClassicalCorrectionDeviceError::kNonfiniteArithmetic));
  CHECK(energies == fixture.energy_seed && gradients == fixture.gradient_seed);
  return 0;
}

int test_late_halogen_failure_is_transactional() {
  Fixture fixture;
  CHECK(fixture.configure_three_atom_halogen_system() == cudaSuccess);

  /* The zero-strength control proves that D3 has already changed both staged
   * accumulators before the later halogen arithmetic is made hostile. */
  CHECK(fixture.launch() == cudaSuccess);
  std::vector<double> energies;
  std::vector<double> gradients;
  std::vector<std::uint32_t> errors;
  CHECK(fixture.d_energies.download(energies) == cudaSuccess);
  CHECK(fixture.d_gradients.download(gradients) == cudaSuccess);
  CHECK(fixture.d_system_errors.download(errors) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(errors == std::vector<std::uint32_t>({0u, 0u}));
  CHECK(energies[0] != kEnergySeed0);
  CHECK(std::any_of(gradients.begin(), gradients.begin() + 9,
                    [](double value) { return value != kGradientSeed; }));

  /* At this collinear geometry the halogen energy remains finite while its
   * radial gradient overflows, after D3 has updated the unpublished candidate. */
  fixture.halogen_strength[0] = std::numeric_limits<double>::max();
  CHECK(fixture.d_halogen_strength.upload(fixture.halogen_strength) == cudaSuccess);
  CHECK(fixture.reset() == cudaSuccess);
  CHECK(fixture.launch() == cudaSuccess);
  CHECK(fixture.d_energies.download(energies) == cudaSuccess);
  CHECK(fixture.d_gradients.download(gradients) == cudaSuccess);
  CHECK(fixture.d_system_errors.download(errors) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(errors[0] ==
        static_cast<std::uint32_t>(Gfn1ClassicalCorrectionDeviceError::kNonfiniteArithmetic));
  CHECK(errors[1] == 0u);
  CHECK(energies == fixture.energy_seed);
  CHECK(gradients == fixture.gradient_seed);
  return 0;
}

int test_synchronous_rejection_and_graph_replay() {
  Fixture fixture;
  auto wrong_workspace = fixture.workspace;
  ++wrong_workspace.plan_token;
  CHECK(add_gfn1_classical_corrections_cuda(fixture.plan, fixture.d_positions.get(),
                                            fixture.d_coordination.get(), fixture.d_active.get(),
                                            fixture.d_energies.get(), fixture.d_gradients.get(),
                                            wrong_workspace, fixture.d_system_errors.get(),
                                            fixture.d_plan_error.get()) == cudaErrorInvalidValue);
  CHECK(add_gfn1_classical_corrections_cuda(
            fixture.plan, fixture.d_positions.get(), fixture.d_coordination.get(),
            fixture.d_active.get(), fixture.d_energies.get(), fixture.workspace.gradient_scratch,
            fixture.workspace, fixture.d_system_errors.get(),
            fixture.d_plan_error.get()) == cudaErrorInvalidValue);

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  CHECK(fixture.reset(stream) == cudaSuccess);
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess);
  CHECK(fixture.launch(nullptr, nullptr, stream) == cudaSuccess);
  CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  CHECK(cudaGraphInstantiate(&executable, graph, 0) == cudaSuccess);
  CHECK(cudaGraphLaunch(executable, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  std::vector<std::uint32_t> errors;
  CHECK(fixture.d_system_errors.download(errors, stream) == cudaSuccess);
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  CHECK(errors == std::vector<std::uint32_t>({0u, 0u}));
  CHECK(cudaGraphExecDestroy(executable) == cudaSuccess);
  CHECK(cudaGraphDestroy(graph) == cudaSuccess);
  CHECK(cudaStreamDestroy(stream) == cudaSuccess);
  return 0;
}

}  // namespace

int main() {
  const std::array<int (*)(), 6> tests{
      {test_energy_gradient_and_finite_difference, test_peer_failure_is_transactional,
       test_nonfinite_reference_cn_is_transactional, test_active_mask_and_gradient_overflow,
       test_late_halogen_failure_is_transactional, test_synchronous_rejection_and_graph_replay}};
  for (const auto test : tests) {
    const int status = test();
    if (status != 0) return status;
  }
  return 0;
}
