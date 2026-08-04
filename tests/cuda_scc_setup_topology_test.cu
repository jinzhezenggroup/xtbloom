#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_plan_schema.cuh"
#include "backends/cuda/gfn2_scc_setup_topology.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using gpuxtb::detail::Gfn2PlanMemorySpace;
using gpuxtb::detail::Gfn2PlanSchemaDiagnostic;
using gpuxtb::detail::Gfn2PlanSchemaError;
using gpuxtb::detail::Gfn2RaggedTopologyView;
using gpuxtb::detail::Gfn2WavefunctionLayoutView;
using gpuxtb::detail::cuda::Gfn2EigensolverBucket;
using gpuxtb::detail::cuda::Gfn2SccSetupTopology;
using gpuxtb::detail::cuda::Gfn2SccSetupTopologyError;
using gpuxtb::detail::cuda::Gfn2SccSetupTopologyField;
using gpuxtb::detail::cuda::validate_gfn2_topology_cuda_async;
using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::IntegralPlan;
using gpuxtb::detail::gfn2::WavefunctionLayout;

constexpr std::uint64_t kPlanToken = 0x510e527fade682d1ULL;

struct Plans {
  BasisPlan basis;
  IntegralPlan integrals;
  WavefunctionLayout wavefunction;
};

/* AO dimensions [2, 1, 2] exercise ascending buckets while also proving that
 * equal-dimension systems retain their input order. Every atom owns one shell
 * and every shell owns one AO, keeping the topology intentionally transparent. */
Plans make_plans() {
  Plans plans;
  plans.basis.batch_size = 3;
  plans.basis.total_atoms = 5;
  plans.basis.total_shells = 5;
  plans.basis.total_orbitals = 5;
  plans.basis.atom_offsets = {0, 2, 3, 5};
  plans.basis.batch_shell_offsets = {0, 2, 3, 5};
  plans.basis.batch_orbital_offsets = {0, 2, 3, 5};
  plans.basis.atom_shell_offsets = {0, 1, 2, 3, 4, 5};
  plans.basis.shell_orbital_offsets = {0, 1, 2, 3, 4, 5};
  plans.basis.shell_to_atom = {0, 1, 2, 3, 4};

  plans.integrals.batch_size = 3;
  plans.integrals.total_matrix_elements = 9;
  plans.integrals.matrix_offsets = {0, 4, 5, 9};

  plans.wavefunction.batch_size = 3;
  plans.wavefunction.total_atoms = 5;
  plans.wavefunction.total_shells = 5;
  plans.wavefunction.total_orbitals = 5;
  plans.wavefunction.atom_offsets = plans.basis.atom_offsets;
  plans.wavefunction.batch_shell_offsets = plans.basis.batch_shell_offsets;
  plans.wavefunction.batch_orbital_offsets = plans.basis.batch_orbital_offsets;
  plans.wavefunction.spin_channels = {1, 2, 1};
  const auto assign_field = [](auto& field, std::vector<std::int64_t> offsets) {
    field.system_offsets = std::move(offsets);
    field.element_count = field.system_offsets.back();
    field.size_bytes = static_cast<std::size_t>(field.element_count) * sizeof(double);
  };
  assign_field(plans.wavefunction.coefficients, {0, 4, 6, 10});
  assign_field(plans.wavefunction.eigenvalues, {0, 2, 4, 6});
  assign_field(plans.wavefunction.occupations, {0, 4, 6, 10});
  assign_field(plans.wavefunction.density, {0, 4, 6, 10});
  assign_field(plans.wavefunction.qsh, {0, 2, 4, 6});
  assign_field(plans.wavefunction.qat, {0, 2, 4, 6});
  assign_field(plans.wavefunction.dipole, {0, 6, 12, 18});
  assign_field(plans.wavefunction.quadrupole, {0, 12, 24, 36});
  assign_field(plans.wavefunction.energy_weighted_density, {0, 4, 6, 10});
  return plans;
}

template <typename T>
bool same_device_values(const T* device_values, const std::vector<T>& expected) {
  std::vector<T> actual(expected.size());
  if (!expected.empty() && cudaMemcpy(actual.data(), device_values, expected.size() * sizeof(T),
                                      cudaMemcpyDeviceToHost) != cudaSuccess) {
    return false;
  }
  return actual == expected;
}

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  cudaError_t allocate(std::size_t bytes) { return cudaMalloc(&pointer_, bytes); }

  void* get() const noexcept { return pointer_; }

 private:
  void* pointer_ = nullptr;
};

int test_host_blueprint_and_transactional_create() {
  Plans plans = make_plans();
  Gfn2SccSetupTopology empty;
  Gfn2RaggedTopologyView untouched{};
  untouched.plan_token = 17u;
  auto diagnostic = empty.bind_device_arena_and_upload_async(nullptr, 0u, untouched);
  CHECK(!diagnostic.success());
  CHECK(diagnostic.error == Gfn2SccSetupTopologyError::kInvalidPlan);
  CHECK(diagnostic.field == Gfn2SccSetupTopologyField::kHostTopology);
  CHECK(untouched.plan_token == 17u);

  Gfn2SccSetupTopology topology;
  diagnostic = Gfn2SccSetupTopology::create(plans.basis, plans.integrals, plans.wavefunction,
                                            kPlanToken, topology);
  CHECK(diagnostic.success());
  CHECK(topology.valid());

  const Gfn2RaggedTopologyView& host = topology.host_topology();
  CHECK(host.memory_space == Gfn2PlanMemorySpace::kHost);
  CHECK(host.plan_token == kPlanToken);
  CHECK(host.batch_size == 3);
  CHECK(host.total_atoms == 5);
  CHECK(host.total_shells == 5);
  CHECK(host.total_orbitals == 5);
  CHECK(host.total_matrix_elements == 9);
  CHECK(host.bucket_count == 2);
  const Gfn2WavefunctionLayoutView& host_wavefunction = topology.host_wavefunction_layout();
  CHECK(host_wavefunction.memory_space == Gfn2PlanMemorySpace::kHost);
  CHECK(host_wavefunction.plan_token == kPlanToken);
  CHECK(host_wavefunction.total_spin_channels == 4);
  CHECK(host_wavefunction.total_spin_orbitals == 6);
  CHECK(host_wavefunction.total_spin_matrix_elements == 10);
  CHECK(std::vector<std::int32_t>(
            host_wavefunction.spin_channels,
            host_wavefunction.spin_channels + host_wavefunction.spin_channel_count) ==
        std::vector<std::int32_t>({1, 2, 1}));
  CHECK(std::vector<std::int64_t>(
            host_wavefunction.spin_channel_offsets,
            host_wavefunction.spin_channel_offsets + host_wavefunction.spin_channel_offset_count) ==
        std::vector<std::int64_t>({0, 1, 3, 4}));
  CHECK(std::vector<std::int64_t>(host.orbital_to_shell,
                                  host.orbital_to_shell + host.orbital_to_shell_count) ==
        std::vector<std::int64_t>({0, 1, 2, 3, 4}));
  CHECK(std::vector<std::int64_t>(host.orbital_to_atom,
                                  host.orbital_to_atom + host.orbital_to_atom_count) ==
        std::vector<std::int64_t>({0, 1, 2, 3, 4}));
  CHECK(std::vector<std::int64_t>(host.bucket_offsets,
                                  host.bucket_offsets + host.bucket_offset_count) ==
        std::vector<std::int64_t>({0, 1, 3}));
  CHECK(std::vector<std::int32_t>(host.bucket_systems,
                                  host.bucket_systems + host.bucket_system_count) ==
        std::vector<std::int32_t>({1, 0, 2}));
  CHECK(std::vector<std::int32_t>(host.bucket_orbital_counts,
                                  host.bucket_orbital_counts + host.bucket_orbital_count) ==
        std::vector<std::int32_t>({1, 2}));

  const std::vector<Gfn2EigensolverBucket>& buckets = topology.eigensolver_buckets();
  CHECK(buckets.size() == 2u);
  CHECK(buckets[0].orbital_count == 1);
  CHECK(buckets[0].system_count == 1);
  CHECK(buckets[0].system_index_offset == 0);
  CHECK(buckets[0].matrix_scratch_offset == 0);
  CHECK(buckets[0].orbital_scratch_offset == 0);
  CHECK(buckets[0].solve_count == 2);
  CHECK(buckets[0].solve_index_offset == 0);
  CHECK(buckets[0].spin_matrix_scratch_offset == 0);
  CHECK(buckets[0].spin_orbital_scratch_offset == 0);
  CHECK(buckets[1].orbital_count == 2);
  CHECK(buckets[1].system_count == 2);
  CHECK(buckets[1].system_index_offset == 1);
  CHECK(buckets[1].matrix_scratch_offset == 1);
  CHECK(buckets[1].orbital_scratch_offset == 1);
  CHECK(buckets[1].solve_count == 2);
  CHECK(buckets[1].solve_index_offset == 2);
  CHECK(buckets[1].spin_matrix_scratch_offset == 2);
  CHECK(buckets[1].spin_orbital_scratch_offset == 2);

  const auto requirements = topology.requirements();
  CHECK(requirements.immutable_device_bytes > 0u);
  CHECK(requirements.device_alignment == alignof(std::int64_t));

  /* create() is owner-transactional: a rejected replacement cannot destroy a
   * setup that may still back in-flight work on another stream. */
  plans.integrals.matrix_offsets[2] = 6;
  for (auto* field : {&plans.wavefunction.coefficients, &plans.wavefunction.density,
                      &plans.wavefunction.energy_weighted_density}) {
    field->system_offsets = {0, 4, 8, 11};
    field->element_count = 11;
    field->size_bytes = 11u * sizeof(double);
  }
  diagnostic = Gfn2SccSetupTopology::create(plans.basis, plans.integrals, plans.wavefunction,
                                            kPlanToken + 1u, topology);
  CHECK(!diagnostic.success());
  CHECK(diagnostic.error == Gfn2SccSetupTopologyError::kInvalidPlan);
  CHECK(diagnostic.field == Gfn2SccSetupTopologyField::kBuckets);
  CHECK(topology.valid());
  CHECK(topology.host_topology().plan_token == kPlanToken);

  Plans incompatible = make_plans();
  incompatible.wavefunction.spin_channels[1] = 3;
  Gfn2SccSetupTopology rejected;
  diagnostic = Gfn2SccSetupTopology::create(incompatible.basis, incompatible.integrals,
                                            incompatible.wavefunction, kPlanToken, rejected);
  CHECK(!diagnostic.success());
  CHECK(diagnostic.status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(diagnostic.field == Gfn2SccSetupTopologyField::kWavefunction);
  CHECK(diagnostic.index == 1);
  CHECK(!rejected.valid());

  /* A forged layout must fail before setup copies offsets or calls back(). */
  incompatible = make_plans();
  incompatible.wavefunction.eigenvalues.system_offsets.clear();
  diagnostic = Gfn2SccSetupTopology::create(incompatible.basis, incompatible.integrals,
                                            incompatible.wavefunction, kPlanToken, rejected);
  CHECK(!diagnostic.success());
  CHECK(diagnostic.status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(diagnostic.field == Gfn2SccSetupTopologyField::kWavefunction);
  CHECK(!rejected.valid());

  incompatible = make_plans();
  ++incompatible.wavefunction.density.system_offsets[2];
  diagnostic = Gfn2SccSetupTopology::create(incompatible.basis, incompatible.integrals,
                                            incompatible.wavefunction, kPlanToken, rejected);
  CHECK(!diagnostic.success());
  CHECK(diagnostic.status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(diagnostic.field == Gfn2SccSetupTopologyField::kWavefunction);
  CHECK(!rejected.valid());

  incompatible = make_plans();
  incompatible.basis.total_atoms = std::numeric_limits<std::int64_t>::max();
  diagnostic = Gfn2SccSetupTopology::create(incompatible.basis, incompatible.integrals,
                                            incompatible.wavefunction, kPlanToken, rejected);
  CHECK(!diagnostic.success());
  CHECK(diagnostic.field == Gfn2SccSetupTopologyField::kBasis);

  const std::int64_t* atom_offsets_before_move = topology.host_topology().atom_offsets;
  Gfn2SccSetupTopology moved = std::move(topology);
  CHECK(moved.valid());
  CHECK(!topology.valid());
  CHECK(moved.host_topology().atom_offsets == atom_offsets_before_move);
  return 0;
}

int test_device_upload_and_fail_closed_binding() {
  Plans plans = make_plans();
  Gfn2SccSetupTopology topology;
  auto diagnostic = Gfn2SccSetupTopology::create(plans.basis, plans.integrals, plans.wavefunction,
                                                 kPlanToken, topology);
  CHECK(diagnostic.success());
  const auto requirements = topology.requirements();

  DeviceAllocation allocation;
  CUDA_CHECK(
      allocation.allocate(requirements.immutable_device_bytes + requirements.device_alignment));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  Gfn2RaggedTopologyView sentinel{};
  sentinel.memory_space = Gfn2PlanMemorySpace::kHipDevice;
  sentinel.plan_token = 0xdeadbeefULL;
  sentinel.atom_offsets = reinterpret_cast<const std::int64_t*>(0x1000u);
  Gfn2RaggedTopologyView binding = sentinel;
  Gfn2WavefunctionLayoutView wavefunction_sentinel{};
  wavefunction_sentinel.memory_space = Gfn2PlanMemorySpace::kHipDevice;
  wavefunction_sentinel.plan_token = 0xcafebabeULL;
  wavefunction_sentinel.spin_channels = reinterpret_cast<const std::int32_t*>(0x2000u);
  Gfn2WavefunctionLayoutView wavefunction_binding = wavefunction_sentinel;

  diagnostic = topology.bind_device_arena_and_upload_async(
      nullptr, requirements.immutable_device_bytes, binding, stream);
  CHECK(diagnostic.error == Gfn2SccSetupTopologyError::kNullArena);
  CHECK(diagnostic.required_bytes == requirements.immutable_device_bytes);
  CHECK(binding.plan_token == sentinel.plan_token);
  CHECK(binding.atom_offsets == sentinel.atom_offsets);

  auto* const misaligned = static_cast<std::byte*>(allocation.get()) + 1u;
  diagnostic = topology.bind_device_arena_and_upload_async(
      misaligned, requirements.immutable_device_bytes, binding, stream);
  CHECK(diagnostic.error == Gfn2SccSetupTopologyError::kMisalignedArena);
  CHECK(binding.plan_token == sentinel.plan_token);

  diagnostic = topology.bind_device_arena_and_upload_async(
      allocation.get(), requirements.immutable_device_bytes - 1u, binding, stream);
  CHECK(diagnostic.error == Gfn2SccSetupTopologyError::kInsufficientArena);
  CHECK(binding.plan_token == sentinel.plan_token);

  std::vector<std::max_align_t> host_arena(
      (requirements.immutable_device_bytes + sizeof(std::max_align_t) - 1u) /
      sizeof(std::max_align_t));
  diagnostic = topology.bind_device_arena_and_upload_async(
      host_arena.data(), requirements.immutable_device_bytes, binding, stream);
  CHECK(diagnostic.error == Gfn2SccSetupTopologyError::kInvalidArenaMemory);
  CHECK(diagnostic.status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(binding.plan_token == sentinel.plan_token);

  /* A preceding memset on a nonblocking caller stream would race an upload
   * accidentally issued to the legacy default stream. Correct final values
   * therefore also cover stream routing without adding a hidden synchronize. */
  CUDA_CHECK(cudaMemsetAsync(allocation.get(), 0xa5, requirements.immutable_device_bytes, stream));
  diagnostic = topology.bind_device_arena_and_upload_async(
      allocation.get(), requirements.immutable_device_bytes, binding, wavefunction_binding, stream);
  CHECK(diagnostic.success());
  CHECK(binding.memory_space == Gfn2PlanMemorySpace::kCudaDevice);
  CHECK(binding.plan_token == kPlanToken);
  CHECK(binding.atom_offsets != topology.host_topology().atom_offsets);
  CHECK(wavefunction_binding.memory_space == Gfn2PlanMemorySpace::kCudaDevice);
  CHECK(wavefunction_binding.plan_token == kPlanToken);
  CHECK(wavefunction_binding.spin_channels != topology.host_wavefunction_layout().spin_channels);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const Gfn2RaggedTopologyView& host = topology.host_topology();
  CHECK(same_device_values(binding.atom_offsets, plans.basis.atom_offsets));
  CHECK(same_device_values(binding.batch_shell_offsets, plans.basis.batch_shell_offsets));
  CHECK(same_device_values(binding.batch_orbital_offsets, plans.basis.batch_orbital_offsets));
  CHECK(same_device_values(binding.matrix_offsets, plans.integrals.matrix_offsets));
  CHECK(same_device_values(binding.atom_shell_offsets, plans.basis.atom_shell_offsets));
  CHECK(same_device_values(binding.shell_orbital_offsets, plans.basis.shell_orbital_offsets));
  CHECK(same_device_values(binding.shell_to_atom, plans.basis.shell_to_atom));
  CHECK(same_device_values(
      binding.orbital_to_shell,
      std::vector<std::int64_t>(host.orbital_to_shell,
                                host.orbital_to_shell + host.orbital_to_shell_count)));
  CHECK(same_device_values(
      binding.orbital_to_atom,
      std::vector<std::int64_t>(host.orbital_to_atom,
                                host.orbital_to_atom + host.orbital_to_atom_count)));
  CHECK(
      same_device_values(binding.bucket_offsets,
                         std::vector<std::int64_t>(
                             host.bucket_offsets, host.bucket_offsets + host.bucket_offset_count)));
  CHECK(
      same_device_values(binding.bucket_systems,
                         std::vector<std::int32_t>(
                             host.bucket_systems, host.bucket_systems + host.bucket_system_count)));
  CHECK(same_device_values(
      binding.bucket_orbital_counts,
      std::vector<std::int32_t>(host.bucket_orbital_counts,
                                host.bucket_orbital_counts + host.bucket_orbital_count)));
  const Gfn2WavefunctionLayoutView& host_wavefunction = topology.host_wavefunction_layout();
  CHECK(same_device_values(wavefunction_binding.spin_channels,
                           std::vector<std::int32_t>(host_wavefunction.spin_channels,
                                                     host_wavefunction.spin_channels +
                                                         host_wavefunction.spin_channel_count)));
  CHECK(same_device_values(
      wavefunction_binding.spin_channel_offsets,
      std::vector<std::int64_t>(
          host_wavefunction.spin_channel_offsets,
          host_wavefunction.spin_channel_offsets + host_wavefunction.spin_channel_offset_count)));
  CHECK(same_device_values(
      wavefunction_binding.spin_orbital_offsets,
      std::vector<std::int64_t>(
          host_wavefunction.spin_orbital_offsets,
          host_wavefunction.spin_orbital_offsets + host_wavefunction.spin_orbital_offset_count)));
  CHECK(same_device_values(
      wavefunction_binding.spin_matrix_offsets,
      std::vector<std::int64_t>(
          host_wavefunction.spin_matrix_offsets,
          host_wavefunction.spin_matrix_offsets + host_wavefunction.spin_matrix_offset_count)));

  DeviceAllocation device_diagnostic;
  CUDA_CHECK(device_diagnostic.allocate(sizeof(Gfn2PlanSchemaDiagnostic)));
  CUDA_CHECK(validate_gfn2_topology_cuda_async(
      binding, static_cast<Gfn2PlanSchemaDiagnostic*>(device_diagnostic.get()), stream));
  Gfn2PlanSchemaDiagnostic schema{};
  CUDA_CHECK(cudaMemcpyAsync(&schema, device_diagnostic.get(), sizeof(schema),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(schema.error == Gfn2PlanSchemaError::kSuccess);

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_managed_arena() {
  Plans plans = make_plans();
  Gfn2SccSetupTopology topology;
  auto diagnostic = Gfn2SccSetupTopology::create(plans.basis, plans.integrals, plans.wavefunction,
                                                 kPlanToken, topology);
  CHECK(diagnostic.success());
  const auto requirements = topology.requirements();

  void* managed = nullptr;
  CUDA_CHECK(cudaMallocManaged(&managed, requirements.immutable_device_bytes));
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Gfn2RaggedTopologyView binding{};
  diagnostic = topology.bind_device_arena_and_upload_async(
      managed, requirements.immutable_device_bytes, binding, stream);
  CHECK(diagnostic.success());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(same_device_values(binding.bucket_systems, std::vector<std::int32_t>({1, 0, 2})));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(managed));
  return 0;
}

}  // namespace

int main() {
  int status = test_host_blueprint_and_transactional_create();
  if (status != 0) {
    return status;
  }
  CUDA_CHECK(cudaSetDevice(0));
  status = test_device_upload_and_fail_closed_binding();
  if (status != 0) {
    return status;
  }
  return test_managed_arena();
}
