#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_scc_iteration.cuh"
#include "tests/support/gfn2_scc_test_case.hpp"

/*
 * The #89 launch entry is intentionally not named here until its signature is
 * sealed. Defining this macro later enables launcher-specific assertions
 * without making the setup fixture depend on an unstable declaration today.
 */
#ifndef GPUXTB_TEST_HAS_GFN2_SCC_ITERATION_COMPOSER_LAUNCHER
#define GPUXTB_TEST_HAS_GFN2_SCC_ITERATION_COMPOSER_LAUNCHER 0
#endif

namespace {

using namespace gpuxtb::detail;
using namespace gpuxtb::detail::cuda;
using gpuxtb::test::gfn2::HostSccCase;
using gpuxtb::test::gfn2::HostSccCaseOptions;
using gpuxtb::test::gfn2::HostSccCheckpoint;
using gpuxtb::test::gfn2::SmallSystemKind;

constexpr std::uint64_t kPlanToken = 0x9500c0deULL;
constexpr bool kComposerLauncherAvailable =
    GPUXTB_TEST_HAS_GFN2_SCC_ITERATION_COMPOSER_LAUNCHER != 0;

bool cuda_ok(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
  return false;
}

bool solver_ok(cusolverStatus_t status, const char* operation) {
  if (status == CUSOLVER_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << operation << ": cuSOLVER status " << static_cast<int>(status) << '\n';
  return false;
}

bool blas_ok(cublasStatus_t status, const char* operation) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << operation << ": cuBLAS status " << static_cast<int>(status) << '\n';
  return false;
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() noexcept = default;
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] bool allocate(std::size_t count) {
    release();
    if (count == 0u) {
      return true;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return false;
    }
    const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T));
    if (status != cudaSuccess) {
      std::cerr << "cudaMalloc: " << cudaGetErrorString(status) << '\n';
      pointer_ = nullptr;
      return false;
    }
    count_ = count;
    return true;
  }

  [[nodiscard]] bool allocate_and_upload(const T* source, std::size_t count, cudaStream_t stream) {
    return allocate(count) && upload(source, count, stream);
  }

  [[nodiscard]] bool allocate_and_upload(const std::vector<T>& source, cudaStream_t stream) {
    return allocate_and_upload(source.data(), source.size(), stream);
  }

  [[nodiscard]] bool upload(const T* source, std::size_t count, cudaStream_t stream) {
    if (count == 0u) {
      return true;
    }
    return source != nullptr && count <= count_ &&
           cuda_ok(
               cudaMemcpyAsync(pointer_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync host to device");
  }

  [[nodiscard]] bool download(std::vector<T>& destination, cudaStream_t stream) const {
    destination.resize(count_);
    return count_ == 0u || cuda_ok(cudaMemcpyAsync(destination.data(), pointer_, count_ * sizeof(T),
                                                   cudaMemcpyDeviceToHost, stream),
                                   "cudaMemcpyAsync device to host");
  }

  [[nodiscard]] T* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  void release() noexcept {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
      pointer_ = nullptr;
    }
    count_ = 0u;
  }

  T* pointer_ = nullptr;
  std::size_t count_ = 0u;
};

class PinnedBuffer {
 public:
  PinnedBuffer() noexcept = default;
  ~PinnedBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFreeHost(pointer_);
    }
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  [[nodiscard]] bool allocate(std::size_t bytes) {
    if (pointer_ != nullptr || bytes_ != 0u) {
      return false;
    }
    bytes_ = bytes;
    return bytes == 0u || cuda_ok(cudaMallocHost(&pointer_, bytes), "cudaMallocHost");
  }

  [[nodiscard]] void* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

struct ProviderHandles {
  cudaStream_t stream = nullptr;
  cusolverDnHandle_t solver = nullptr;
  cusolverDnParams_t parameters = nullptr;
  cublasHandle_t blas = nullptr;

  ProviderHandles() noexcept = default;
  ProviderHandles(const ProviderHandles&) = delete;
  ProviderHandles& operator=(const ProviderHandles&) = delete;
  ~ProviderHandles() {
    if (blas != nullptr) {
      (void)cublasDestroy(blas);
    }
    if (parameters != nullptr) {
      (void)cusolverDnDestroyParams(parameters);
    }
    if (solver != nullptr) {
      (void)cusolverDnDestroy(solver);
    }
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
  }

  [[nodiscard]] bool create() {
    return cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags") &&
           solver_ok(cusolverDnCreate(&solver), "cusolverDnCreate") &&
           solver_ok(cusolverDnCreateParams(&parameters), "cusolverDnCreateParams") &&
           blas_ok(cublasCreate(&blas), "cublasCreate") &&
           solver_ok(cusolverDnSetStream(solver, stream), "cusolverDnSetStream") &&
           blas_ok(cublasSetStream(blas, stream), "cublasSetStream");
  }
};

bool checkpoints_equal(const HostSccCheckpoint& first, const HostSccCheckpoint& second) {
  return first.wavefunction == second.wavefunction && first.mixer_state == second.mixer_state &&
         first.driver_state == second.driver_state &&
         first.driver_workspace == second.driver_workspace;
}

/* Host metadata used both for full host validation and immutable CUDA upload. */
struct HostTopology {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> batch_orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> shell_orbital_offsets;
  std::vector<std::int64_t> shell_to_atom;
  std::vector<std::int64_t> orbital_to_shell;
  std::vector<std::int64_t> orbital_to_atom;
  std::vector<std::int64_t> bucket_offsets;
  std::vector<std::int32_t> bucket_systems;
  std::vector<std::int32_t> bucket_orbital_counts;
  std::vector<Gfn2EigensolverBucket> buckets;

  [[nodiscard]] bool make(const HostSccCase& fixture) {
    const auto& basis = fixture.basis_plan();
    const auto& integrals = fixture.integral_plan();
    atom_offsets = basis.atom_offsets;
    batch_shell_offsets = basis.batch_shell_offsets;
    batch_orbital_offsets = basis.batch_orbital_offsets;
    matrix_offsets = integrals.matrix_offsets;
    atom_shell_offsets = basis.atom_shell_offsets;
    shell_orbital_offsets = basis.shell_orbital_offsets;
    shell_to_atom = basis.shell_to_atom;
    orbital_to_shell.assign(static_cast<std::size_t>(basis.total_orbitals), -1);
    orbital_to_atom.assign(static_cast<std::size_t>(basis.total_orbitals), -1);
    for (std::int64_t shell = 0; shell < basis.total_shells; ++shell) {
      const std::int64_t atom = shell_to_atom[static_cast<std::size_t>(shell)];
      for (std::int64_t orbital = shell_orbital_offsets[static_cast<std::size_t>(shell)];
           orbital < shell_orbital_offsets[static_cast<std::size_t>(shell + 1)]; ++orbital) {
        orbital_to_shell[static_cast<std::size_t>(orbital)] = shell;
        orbital_to_atom[static_cast<std::size_t>(orbital)] = atom;
      }
    }

    std::vector<std::int32_t> dimensions;
    dimensions.reserve(static_cast<std::size_t>(basis.batch_size));
    for (std::int64_t system = 0; system < basis.batch_size; ++system) {
      const std::int64_t orbitals = batch_orbital_offsets[static_cast<std::size_t>(system + 1)] -
                                    batch_orbital_offsets[static_cast<std::size_t>(system)];
      if (orbitals <= 0 || orbitals > std::numeric_limits<std::int32_t>::max()) {
        return false;
      }
      dimensions.push_back(static_cast<std::int32_t>(orbitals));
    }
    std::vector<std::int32_t> unique_dimensions = dimensions;
    std::sort(unique_dimensions.begin(), unique_dimensions.end());
    unique_dimensions.erase(std::unique(unique_dimensions.begin(), unique_dimensions.end()),
                            unique_dimensions.end());

    bucket_offsets.push_back(0);
    std::int64_t packed_matrix_offset = 0;
    std::int64_t packed_orbital_offset = 0;
    for (const std::int32_t dimension : unique_dimensions) {
      const std::int64_t system_index_offset = static_cast<std::int64_t>(bucket_systems.size());
      for (std::int64_t system = 0; system < basis.batch_size; ++system) {
        if (dimensions[static_cast<std::size_t>(system)] == dimension) {
          bucket_systems.push_back(static_cast<std::int32_t>(system));
        }
      }
      const std::int64_t system_count =
          static_cast<std::int64_t>(bucket_systems.size()) - system_index_offset;
      if (system_count <= 0 || system_count > std::numeric_limits<std::int32_t>::max()) {
        return false;
      }
      buckets.push_back({dimension, static_cast<std::int32_t>(system_count), system_index_offset,
                         packed_matrix_offset, packed_orbital_offset});
      bucket_orbital_counts.push_back(dimension);
      bucket_offsets.push_back(static_cast<std::int64_t>(bucket_systems.size()));
      packed_matrix_offset += static_cast<std::int64_t>(dimension) * dimension * system_count;
      packed_orbital_offset += static_cast<std::int64_t>(dimension) * system_count;
    }
    return packed_matrix_offset == integrals.total_matrix_elements &&
           packed_orbital_offset == basis.total_orbitals;
  }

  [[nodiscard]] Gfn2RaggedTopologyView view(const HostSccCase& fixture,
                                            Gfn2PlanMemorySpace memory_space) const noexcept {
    const auto& basis = fixture.basis_plan();
    const auto& integrals = fixture.integral_plan();
    Gfn2RaggedTopologyView result{};
    result.memory_space = memory_space;
    result.pair_map_kind = Gfn2PairMapKind::kNone;
    result.plan_token = kPlanToken;
    result.batch_size = basis.batch_size;
    result.total_atoms = basis.total_atoms;
    result.total_shells = basis.total_shells;
    result.total_orbitals = basis.total_orbitals;
    result.total_matrix_elements = integrals.total_matrix_elements;
    result.bucket_count = static_cast<std::int64_t>(buckets.size());
    result.atom_offset_count = static_cast<std::int64_t>(atom_offsets.size());
    result.batch_shell_offset_count = static_cast<std::int64_t>(batch_shell_offsets.size());
    result.batch_orbital_offset_count = static_cast<std::int64_t>(batch_orbital_offsets.size());
    result.matrix_offset_count = static_cast<std::int64_t>(matrix_offsets.size());
    result.atom_shell_offset_count = static_cast<std::int64_t>(atom_shell_offsets.size());
    result.shell_orbital_offset_count = static_cast<std::int64_t>(shell_orbital_offsets.size());
    result.shell_to_atom_count = static_cast<std::int64_t>(shell_to_atom.size());
    result.orbital_to_shell_count = static_cast<std::int64_t>(orbital_to_shell.size());
    result.orbital_to_atom_count = static_cast<std::int64_t>(orbital_to_atom.size());
    result.bucket_offset_count = static_cast<std::int64_t>(bucket_offsets.size());
    result.bucket_system_count = static_cast<std::int64_t>(bucket_systems.size());
    result.bucket_orbital_count = static_cast<std::int64_t>(bucket_orbital_counts.size());
    result.atom_offsets = atom_offsets.data();
    result.batch_shell_offsets = batch_shell_offsets.data();
    result.batch_orbital_offsets = batch_orbital_offsets.data();
    result.matrix_offsets = matrix_offsets.data();
    result.atom_shell_offsets = atom_shell_offsets.data();
    result.shell_orbital_offsets = shell_orbital_offsets.data();
    result.shell_to_atom = shell_to_atom.data();
    result.orbital_to_shell = orbital_to_shell.data();
    result.orbital_to_atom = orbital_to_atom.data();
    result.bucket_offsets = bucket_offsets.data();
    result.bucket_systems = bucket_systems.data();
    result.bucket_orbital_counts = bucket_orbital_counts.data();
    return result;
  }
};

struct DeviceTopology {
  DeviceBuffer<std::int64_t> atom_offsets;
  DeviceBuffer<std::int64_t> batch_shell_offsets;
  DeviceBuffer<std::int64_t> batch_orbital_offsets;
  DeviceBuffer<std::int64_t> matrix_offsets;
  DeviceBuffer<std::int64_t> atom_shell_offsets;
  DeviceBuffer<std::int64_t> shell_orbital_offsets;
  DeviceBuffer<std::int64_t> shell_to_atom;
  DeviceBuffer<std::int64_t> orbital_to_shell;
  DeviceBuffer<std::int64_t> orbital_to_atom;
  DeviceBuffer<std::int64_t> bucket_offsets;
  DeviceBuffer<std::int32_t> bucket_systems;
  DeviceBuffer<std::int32_t> bucket_orbital_counts;
  Gfn2RaggedTopologyView descriptor{};

  [[nodiscard]] bool upload(const HostSccCase& fixture, const HostTopology& host,
                            cudaStream_t stream) {
    if (!atom_offsets.allocate_and_upload(host.atom_offsets, stream) ||
        !batch_shell_offsets.allocate_and_upload(host.batch_shell_offsets, stream) ||
        !batch_orbital_offsets.allocate_and_upload(host.batch_orbital_offsets, stream) ||
        !matrix_offsets.allocate_and_upload(host.matrix_offsets, stream) ||
        !atom_shell_offsets.allocate_and_upload(host.atom_shell_offsets, stream) ||
        !shell_orbital_offsets.allocate_and_upload(host.shell_orbital_offsets, stream) ||
        !shell_to_atom.allocate_and_upload(host.shell_to_atom, stream) ||
        !orbital_to_shell.allocate_and_upload(host.orbital_to_shell, stream) ||
        !orbital_to_atom.allocate_and_upload(host.orbital_to_atom, stream) ||
        !bucket_offsets.allocate_and_upload(host.bucket_offsets, stream) ||
        !bucket_systems.allocate_and_upload(host.bucket_systems, stream) ||
        !bucket_orbital_counts.allocate_and_upload(host.bucket_orbital_counts, stream)) {
      return false;
    }
    descriptor = host.view(fixture, Gfn2PlanMemorySpace::kCudaDevice);
    descriptor.atom_offsets = atom_offsets.get();
    descriptor.batch_shell_offsets = batch_shell_offsets.get();
    descriptor.batch_orbital_offsets = batch_orbital_offsets.get();
    descriptor.matrix_offsets = matrix_offsets.get();
    descriptor.atom_shell_offsets = atom_shell_offsets.get();
    descriptor.shell_orbital_offsets = shell_orbital_offsets.get();
    descriptor.shell_to_atom = shell_to_atom.get();
    descriptor.orbital_to_shell = orbital_to_shell.get();
    descriptor.orbital_to_atom = orbital_to_atom.get();
    descriptor.bucket_offsets = bucket_offsets.get();
    descriptor.bucket_systems = bucket_systems.get();
    descriptor.bucket_orbital_counts = bucket_orbital_counts.get();
    return true;
  }
};

struct DeviceCheckpointMirror {
  DeviceBuffer<std::byte> wavefunction;
  DeviceBuffer<std::byte> mixer_state;
  DeviceBuffer<std::byte> driver_state;
  DeviceBuffer<std::byte> driver_workspace;

  [[nodiscard]] bool upload(const HostSccCheckpoint& checkpoint, cudaStream_t stream) {
    return wavefunction.allocate_and_upload(checkpoint.wavefunction, stream) &&
           mixer_state.allocate_and_upload(checkpoint.mixer_state, stream) &&
           driver_state.allocate_and_upload(checkpoint.driver_state, stream) &&
           driver_workspace.allocate_and_upload(checkpoint.driver_workspace, stream);
  }

  [[nodiscard]] bool download(HostSccCheckpoint& checkpoint, cudaStream_t stream) const {
    return wavefunction.download(checkpoint.wavefunction, stream) &&
           mixer_state.download(checkpoint.mixer_state, stream) &&
           driver_state.download(checkpoint.driver_state, stream) &&
           driver_workspace.download(checkpoint.driver_workspace, stream);
  }
};

/* Numerical inputs already available from the production host fixture. */
struct DeviceInputs {
  DeviceTopology topology;
  DeviceBuffer<std::int32_t> atomic_numbers;
  DeviceBuffer<double> positions;
  DeviceBuffer<double> h0;
  DeviceBuffer<double> overlap;
  DeviceBuffer<double> dipole_integrals;
  DeviceBuffer<double> quadrupole_integrals;
  DeviceBuffer<double> es2_cache;
  DeviceBuffer<double> aes2_cache;
  DeviceBuffer<double> d4_pair_data;
  DeviceBuffer<double> d4_coordination;
  DeviceBuffer<std::int64_t> point_charge_offsets;
  DeviceBuffer<double> point_charge_positions;
  DeviceBuffer<double> point_charge_charges;
  DeviceBuffer<double> point_charge_hardnesses;
  DeviceBuffer<double> explicit_point_charge_shell_potential;
  DeviceBuffer<double> periodic_shifts;
  DeviceBuffer<double> periodic_response;
  DeviceBuffer<double> electron_counts;
  DeviceBuffer<double> temperatures;
  DeviceBuffer<std::uint8_t> active;

  [[nodiscard]] bool upload(const HostSccCase& fixture, const HostTopology& host_topology,
                            cudaStream_t stream) {
    const auto& wavefunction = fixture.wavefunction_layout();
    const auto& es2 = fixture.es2_cache();
    const auto& aes2 = fixture.aes2_cache();
    std::vector<double> electrons(2u * static_cast<std::size_t>(fixture.batch_size()));
    std::vector<double> temperature(static_cast<std::size_t>(fixture.batch_size()),
                                    fixture.options().electronic_temperature);
    std::vector<std::uint8_t> active_host(static_cast<std::size_t>(fixture.batch_size()), 1u);
    for (std::int64_t system = 0; system < fixture.batch_size(); ++system) {
      electrons[2u * static_cast<std::size_t>(system)] =
          wavefunction.alpha_electron_counts[static_cast<std::size_t>(system)];
      electrons[2u * static_cast<std::size_t>(system) + 1u] =
          wavefunction.beta_electron_counts[static_cast<std::size_t>(system)];
    }

    if (!topology.upload(fixture, host_topology, stream) ||
        !atomic_numbers.allocate_and_upload(fixture.atomic_numbers(), stream) ||
        !positions.allocate_and_upload(fixture.positions(), stream) ||
        !h0.allocate_and_upload(fixture.h0(), stream) ||
        !overlap.allocate_and_upload(fixture.overlap(), stream) ||
        !dipole_integrals.allocate_and_upload(fixture.dipole_integrals(), stream) ||
        !quadrupole_integrals.allocate_and_upload(fixture.quadrupole_integrals(), stream) ||
        !es2_cache.allocate_and_upload(es2.coulomb_matrix,
                                       static_cast<std::size_t>(es2.matrix_elements), stream) ||
        !aes2_cache.allocate_and_upload(
            aes2.pair_data, static_cast<std::size_t>(aes2.pair_data_elements), stream) ||
        !point_charge_offsets.allocate_and_upload(fixture.point_charge_offsets(), stream) ||
        !point_charge_positions.allocate_and_upload(fixture.point_charge_positions(), stream) ||
        !point_charge_charges.allocate_and_upload(fixture.point_charge_charges(), stream) ||
        !point_charge_hardnesses.allocate_and_upload(fixture.point_charge_hardnesses(), stream) ||
        !explicit_point_charge_shell_potential.allocate_and_upload(
            fixture.explicit_point_charge_shell_potential(), stream) ||
        !periodic_shifts.allocate_and_upload(fixture.periodic_shifts(), stream) ||
        !periodic_response.allocate_and_upload(fixture.periodic_response_matrices(), stream) ||
        !electron_counts.allocate_and_upload(electrons, stream) ||
        !temperatures.allocate_and_upload(temperature, stream) ||
        !active.allocate_and_upload(active_host, stream)) {
      return false;
    }

    const auto* d4 = fixture.d4_cache();
    return d4 != nullptr &&
           d4_pair_data.allocate_and_upload(
               d4->pair_data, static_cast<std::size_t>(d4->pair_data_elements), stream) &&
           d4_coordination.allocate_and_upload(d4->coordination_numbers,
                                               static_cast<std::size_t>(d4->coordination_elements),
                                               stream);
  }
};

struct ProviderWorkspace {
  DeviceBuffer<double> matrix_a;
  DeviceBuffer<double> matrix_b;
  DeviceBuffer<double> eigenvalues;
  DeviceBuffer<double*> factor_pointers;
  DeviceBuffer<double*> matrix_pointers;
  DeviceBuffer<int> info_a;
  DeviceBuffer<int> info_b;
  DeviceBuffer<std::uint8_t> eligible;
  DeviceBuffer<std::uint32_t> sequence_active;
  DeviceBuffer<std::byte> solver_device_workspace;
  PinnedBuffer solver_host_workspace;
  Gfn2EigensolverWorkspaceRequirements requirements{};
  Gfn2EigensolverDeviceWorkspace descriptor{};

  [[nodiscard]] bool create(const HostSccCase& fixture, const HostTopology& topology,
                            const ProviderHandles& handles) {
    const std::size_t matrices =
        static_cast<std::size_t>(fixture.integral_plan().total_matrix_elements);
    const std::size_t orbitals =
        static_cast<std::size_t>(fixture.wavefunction_layout().total_orbitals);
    const std::size_t batch = static_cast<std::size_t>(fixture.batch_size());
    if (!matrix_a.allocate(matrices) || !matrix_b.allocate(matrices) ||
        !eigenvalues.allocate(orbitals) || !factor_pointers.allocate(batch) ||
        !matrix_pointers.allocate(batch) || !info_a.allocate(batch) || !info_b.allocate(batch) ||
        !eligible.allocate(batch) || !sequence_active.allocate(1u)) {
      return false;
    }
    for (const Gfn2EigensolverBucket& bucket : topology.buckets) {
      const Gfn2EigensolverLaunchResult query = query_gfn2_eigensolver_bucket_workspace_cuda(
          handles.solver, handles.parameters, bucket, matrix_b.get(), eigenvalues.get(),
          requirements);
      if (!query.success()) {
        std::cerr << "cuSOLVER workspace query failed: " << static_cast<unsigned int>(query.status)
                  << '\n';
        return false;
      }
    }
    if (!solver_device_workspace.allocate(requirements.solver_device_workspace_bytes) ||
        !solver_host_workspace.allocate(requirements.solver_host_workspace_bytes)) {
      return false;
    }
    descriptor.matrix_scratch_a = matrix_a.get();
    descriptor.matrix_a_elements = static_cast<std::int64_t>(matrix_a.size());
    descriptor.matrix_scratch_b = matrix_b.get();
    descriptor.matrix_b_elements = static_cast<std::int64_t>(matrix_b.size());
    descriptor.eigenvalue_scratch = eigenvalues.get();
    descriptor.eigenvalue_elements = static_cast<std::int64_t>(eigenvalues.size());
    descriptor.factor_pointers = factor_pointers.get();
    descriptor.factor_pointer_elements = static_cast<std::int64_t>(factor_pointers.size());
    descriptor.matrix_pointers = matrix_pointers.get();
    descriptor.matrix_pointer_elements = static_cast<std::int64_t>(matrix_pointers.size());
    descriptor.info_a = info_a.get();
    descriptor.info_a_elements = static_cast<std::int64_t>(info_a.size());
    descriptor.info_b = info_b.get();
    descriptor.info_b_elements = static_cast<std::int64_t>(info_b.size());
    descriptor.eligible = eligible.get();
    descriptor.eligible_elements = static_cast<std::int64_t>(eligible.size());
    descriptor.sequence_active = sequence_active.get();
    descriptor.sequence_active_elements = static_cast<std::int64_t>(sequence_active.size());
    descriptor.solver_device_workspace = solver_device_workspace.get();
    descriptor.solver_device_workspace_bytes = requirements.solver_device_workspace_bytes;
    descriptor.solver_host_workspace = solver_host_workspace.get();
    descriptor.solver_host_workspace_bytes = solver_host_workspace.size();
    descriptor.plan_token = kPlanToken;
    return true;
  }

  [[nodiscard]] Gfn2SccIterationCudaEigensolverProvider provider(
      const HostTopology& topology, const ProviderHandles& handles) const noexcept {
    Gfn2SccIterationCudaEigensolverProvider result{};
    result.buckets = topology.buckets.data();
    result.bucket_count = static_cast<std::int64_t>(topology.buckets.size());
    result.solver = handles.solver;
    result.parameters = handles.parameters;
    result.blas = handles.blas;
    result.device_workspace = solver_device_workspace.get();
    result.device_workspace_bytes = requirements.solver_device_workspace_bytes;
    result.host_workspace = solver_host_workspace.get();
    result.host_workspace_bytes = solver_host_workspace.size();
    result.requirements = requirements;
    result.capture_mode = Gfn2SccIterationProviderCaptureMode::kUncapturedSegmentRequired;
    result.plan_token = kPlanToken;
    return result;
  }
};

struct ComposerFixture {
  HostSccCase host;
  HostSccCheckpoint cpu_before;
  HostSccCheckpoint cpu_after_one_iteration;
  HostTopology host_topology;
  ProviderHandles handles;
  DeviceInputs device_inputs;
  DeviceCheckpointMirror initial_checkpoint;
  ProviderWorkspace provider_workspace;
  Gfn2SccIterationBinding partial_binding{};

  [[nodiscard]] bool create(std::string& error) {
    HostSccCaseOptions options;
    options.systems = {SmallSystemKind::kH2, SmallSystemKind::kHe, SmallSystemKind::kLiH,
                       SmallSystemKind::kCH2};
    options.enable_d4 = true;
    options.enable_periodic_embedding = true;
    options.enable_explicit_point_charges = true;
    if (HostSccCase::create(options, host, error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }

    cpu_before = host.checkpoint();
    if (host.run_one_iteration(error) != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    cpu_after_one_iteration = host.checkpoint();
    if (checkpoints_equal(cpu_before, cpu_after_one_iteration)) {
      error = "CPU SCC oracle did not advance any checkpoint bytes";
      return false;
    }
    if (host.restore(cpu_before, error) != GPUXTB_STATUS_SUCCESS ||
        !checkpoints_equal(host.checkpoint(), cpu_before)) {
      return false;
    }

    if (!host_topology.make(host)) {
      error = "failed to construct SCC composer topology and buckets";
      return false;
    }
    const Gfn2PlanSchemaDiagnostic topology_diagnostic =
        validate_gfn2_topology_host(host_topology.view(host, Gfn2PlanMemorySpace::kHost));
    if (topology_diagnostic.error != Gfn2PlanSchemaError::kSuccess) {
      error = "host SCC composer topology validation failed";
      return false;
    }
    if (!handles.create() || !device_inputs.upload(host, host_topology, handles.stream) ||
        !initial_checkpoint.upload(cpu_before, handles.stream) ||
        !provider_workspace.create(host, host_topology, handles)) {
      error = "failed to construct CUDA SCC composer setup fixture";
      return false;
    }

    /*
     * This is deliberately partial until the launch ABI is sealed. It records
     * the stable topology/provider/workspace leaves without pretending that
     * an incomplete graph is safe to pass to the binding validator.
     */
    partial_binding.plan.abi_version = kGfn2SccIterationAbiVersion;
    partial_binding.plan.plan_token = kPlanToken;
    partial_binding.plan.geometry_generation = host.options().geometry_generation;
    partial_binding.plan.topology = device_inputs.topology.descriptor;
    partial_binding.plan.eigensolver_provider = provider_workspace.provider(host_topology, handles);
    partial_binding.plan.eigensolver_batch.batch_size = host.batch_size();
    partial_binding.plan.eigensolver_batch.total_orbitals =
        host.wavefunction_layout().total_orbitals;
    partial_binding.plan.eigensolver_batch.total_matrix_elements =
        host.integral_plan().total_matrix_elements;
    partial_binding.plan.eigensolver_batch.orbital_offset_count =
        static_cast<std::int64_t>(host_topology.batch_orbital_offsets.size());
    partial_binding.plan.eigensolver_batch.matrix_offset_count =
        static_cast<std::int64_t>(host_topology.matrix_offsets.size());
    partial_binding.plan.eigensolver_batch.bucket_system_count = host.batch_size();
    partial_binding.plan.eigensolver_batch.active_elements = host.batch_size();
    partial_binding.plan.eigensolver_batch.plan_token = kPlanToken;
    partial_binding.plan.eigensolver_batch.orbital_offsets =
        device_inputs.topology.batch_orbital_offsets.get();
    partial_binding.plan.eigensolver_batch.matrix_offsets =
        device_inputs.topology.matrix_offsets.get();
    partial_binding.plan.eigensolver_batch.bucket_systems =
        device_inputs.topology.bucket_systems.get();
    partial_binding.plan.eigensolver_batch.active = device_inputs.active.get();
    partial_binding.workspace.eigensolver_workspace = provider_workspace.descriptor;
    partial_binding.workspace.plan_token = kPlanToken;

    HostSccCheckpoint roundtrip;
    if (!initial_checkpoint.download(roundtrip, handles.stream) ||
        !cuda_ok(cudaStreamSynchronize(handles.stream), "cudaStreamSynchronize") ||
        !checkpoints_equal(roundtrip, cpu_before)) {
      error = "initial CPU checkpoint did not survive the CUDA upload roundtrip";
      return false;
    }
    return true;
  }
};

int test_composer_fixture_setup() {
  ComposerFixture fixture;
  std::string error;
  error.reserve(256u);
  if (!fixture.create(error)) {
    std::cerr << error << '\n';
    return __LINE__;
  }
  if (fixture.host_topology.buckets.empty() ||
      fixture.partial_binding.plan.eigensolver_provider.bucket_count !=
          static_cast<std::int64_t>(fixture.host_topology.buckets.size()) ||
      fixture.partial_binding.workspace.eigensolver_workspace.plan_token != kPlanToken ||
      fixture.partial_binding.plan.topology.memory_space != Gfn2PlanMemorySpace::kCudaDevice) {
    return __LINE__;
  }

  /* The launcher-specific parity call is added behind this setup-time gate. */
  if constexpr (kComposerLauncherAvailable) {
    return 0;
  }
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice || count_status == cudaErrorInsufficientDriver ||
      device_count == 0) {
    (void)cudaGetLastError();
    return 0;
  }
  if (!cuda_ok(count_status, "cudaGetDeviceCount")) {
    return 1;
  }
  return test_composer_fixture_setup();
}
