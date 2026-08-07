#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration.cuh"

namespace {

using namespace gpuxtb::detail;
using namespace gpuxtb::detail::cuda;

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

struct FakeDeviceArena {
  std::uintptr_t cursor = 0x100000000ULL;

  template <typename T>
  T* allocate(std::int64_t elements) {
    if (elements == 0) {
      return nullptr;
    }
    constexpr std::uintptr_t alignment = alignof(T) > 64u ? alignof(T) : 64u;
    cursor = (cursor + alignment - 1u) & ~(alignment - 1u);
    T* result = reinterpret_cast<T*>(cursor);
    cursor += static_cast<std::uintptr_t>(elements) * sizeof(T) + 64u;
    return result;
  }

  void* allocate_bytes(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0u) {
      return nullptr;
    }
    cursor = (cursor + alignment - 1u) & ~(static_cast<std::uintptr_t>(alignment) - 1u);
    void* result = reinterpret_cast<void*>(cursor);
    cursor += bytes + 64u;
    return result;
  }
};

struct Fixture {
  static constexpr std::uint64_t kToken = 0x9600c0deULL;
  static constexpr std::int64_t kBatch = 1;
  static constexpr std::int64_t kAtoms = 1;
  static constexpr std::int64_t kShells = 1;
  static constexpr std::int64_t kOrbitals = 1;
  static constexpr std::int64_t kMatrices = 1;
  static constexpr std::int64_t kDipoles = 3;
  static constexpr std::int64_t kQuadrupoles = 6;
  /* Exercise the expanded unrestricted layout, not only the legacy one-channel shape. */
  static constexpr std::int64_t kSpinChannels = 2;
  static constexpr std::int64_t kSpinAtoms = 2 * kAtoms;
  static constexpr std::int64_t kSpinShells = 2 * kShells;
  static constexpr std::int64_t kSpinOrbitals = 2 * kOrbitals;
  static constexpr std::int64_t kSpinMatrices = 2 * kMatrices;
  static constexpr std::int64_t kSpinDipoles = 3 * kSpinAtoms;
  static constexpr std::int64_t kSpinQuadrupoles = 6 * kSpinAtoms;
  static constexpr std::int64_t kMixerVector = kSpinShells + 9 * kSpinAtoms;
  static constexpr std::int64_t kHistory = 2;

  FakeDeviceArena device;
  std::array<Gfn2EigensolverBucket, 1> buckets{};
  Gfn2SccIterationDevicePlan plan{};
  Gfn2SccIterationDeviceInput input{};
  Gfn2SccIterationDeviceState state{};
  Gfn2SccIterationDeviceWorkspace workspace{};

  template <typename T>
  T* ptr(std::int64_t elements) {
    return device.allocate<T>(elements);
  }

  Gfn2SccDeviceMultipoles multipoles() {
    return {ptr<double>(kSpinShells),
            kSpinShells,
            ptr<double>(kSpinDipoles),
            kSpinDipoles,
            ptr<double>(kSpinQuadrupoles),
            kSpinQuadrupoles,
            kToken};
  }

  Gfn2EigensolverDeviceResults eigenpairs() {
    return {ptr<double>(kSpinOrbitals), kSpinOrbitals, ptr<double>(kSpinMatrices), kSpinMatrices,
            kToken};
  }

  Gfn2OccupationsDeviceResults occupations() {
    return {ptr<double>(2 * kOrbitals),
            2 * kOrbitals,
            ptr<double>(2 * kBatch),
            2 * kBatch,
            ptr<double>(2 * kBatch),
            2 * kBatch,
            ptr<double>(kBatch),
            kBatch,
            kToken};
  }

  Gfn2DensityDeviceResults density() {
    return {ptr<double>(kSpinMatrices),
            kSpinMatrices,
            ptr<double>(kSpinMatrices),
            kSpinMatrices,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            kToken,
            ptr<double>(kSpinChannels),
            kSpinChannels,
            ptr<double>(kSpinChannels),
            kSpinChannels,
            ptr<double>(kSpinChannels),
            kSpinChannels,
            ptr<double>(kSpinChannels),
            kSpinChannels};
  }

  Gfn2MullikenDevicePopulation population() {
    return {ptr<double>(kSpinShells),
            kSpinShells,
            ptr<double>(kSpinAtoms),
            kSpinAtoms,
            ptr<double>(kSpinDipoles),
            kSpinDipoles,
            ptr<double>(kSpinQuadrupoles),
            kSpinQuadrupoles,
            kToken};
  }

  Gfn2SccMixerDeviceState mixer() {
    Gfn2SccMixerDeviceState result{};
    result.current_inputs = ptr<double>(kMixerVector);
    result.previous_inputs = ptr<double>(kMixerVector);
    result.previous_residuals = ptr<double>(kMixerVector);
    result.df_history = ptr<double>(kMixerVector * kHistory);
    result.u_history = ptr<double>(kMixerVector * kHistory);
    result.omega = ptr<double>(kBatch * kHistory);
    result.residual_rms = ptr<double>(kBatch);
    result.residual_maximum = ptr<double>(kBatch);
    result.iterations = ptr<std::uint64_t>(kBatch);
    result.restart_counts = ptr<std::uint64_t>(kBatch);
    result.system_statuses = ptr<gpuxtb_status_t>(kBatch);
    result.initialized = ptr<std::uint8_t>(kBatch);
    result.residual_converged = ptr<std::uint8_t>(kBatch);
    result.total_vector_elements = kMixerVector;
    result.history_elements = kMixerVector * kHistory;
    result.omega_elements = kBatch * kHistory;
    result.batch_elements = kBatch;
    result.plan_token = kToken;
    return result;
  }

  Gfn2SccClassicalEnergyDeviceDiagnostics classical_diagnostics() {
    return {ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            ptr<double>(kBatch),
            kBatch,
            kToken};
  }

  Gfn2SccFreeEnergyDeviceDiagnostics free_diagnostics(
      const Gfn2SccClassicalEnergyDeviceDiagnostics& classical, double* spin_energies) {
    Gfn2SccFreeEnergyDeviceDiagnostics result{};
    result.core = ptr<double>(kBatch);
    result.core_elements = kBatch;
    result.es2 = classical.es2;
    result.es2_elements = kBatch;
    result.es3 = classical.es3;
    result.es3_elements = kBatch;
    result.aes2 = classical.aes2;
    result.aes2_elements = kBatch;
    /* The spin stage and the complete free-energy trace intentionally share storage. */
    result.spin = spin_energies;
    result.spin_elements = kBatch;
    result.d4_two_body = classical.d4_two_body;
    result.d4_two_body_elements = kBatch;
    result.explicit_point_charge = classical.explicit_point_charge;
    result.explicit_point_charge_elements = kBatch;
    result.periodic_embedding = classical.periodic_embedding;
    result.periodic_embedding_elements = kBatch;
    result.entropy = ptr<double>(kBatch);
    result.entropy_elements = kBatch;
    result.internal_energy = ptr<double>(kBatch);
    result.internal_energy_elements = kBatch;
    result.free_energy = ptr<double>(kBatch);
    result.free_energy_elements = kBatch;
    result.plan_token = kToken;
    return result;
  }

  void make_topology() {
    auto& topology = plan.topology;
    topology.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    topology.pair_map_kind = Gfn2PairMapKind::kNone;
    topology.plan_token = kToken;
    topology.batch_size = kBatch;
    topology.total_atoms = kAtoms;
    topology.total_shells = kShells;
    topology.total_orbitals = kOrbitals;
    topology.total_matrix_elements = kMatrices;
    topology.bucket_count = 1;
    topology.atom_offset_count = 2;
    topology.batch_shell_offset_count = 2;
    topology.batch_orbital_offset_count = 2;
    topology.matrix_offset_count = 2;
    topology.atom_shell_offset_count = 2;
    topology.shell_orbital_offset_count = 2;
    topology.shell_to_atom_count = 1;
    topology.orbital_to_shell_count = 1;
    topology.orbital_to_atom_count = 1;
    topology.bucket_offset_count = 2;
    topology.bucket_system_count = 1;
    topology.bucket_orbital_count = 1;
    topology.atom_offsets = ptr<std::int64_t>(2);
    topology.batch_shell_offsets = ptr<std::int64_t>(2);
    topology.batch_orbital_offsets = ptr<std::int64_t>(2);
    topology.matrix_offsets = ptr<std::int64_t>(2);
    topology.atom_shell_offsets = ptr<std::int64_t>(2);
    topology.shell_orbital_offsets = ptr<std::int64_t>(2);
    topology.shell_to_atom = ptr<std::int64_t>(1);
    topology.orbital_to_shell = ptr<std::int64_t>(1);
    topology.orbital_to_atom = ptr<std::int64_t>(1);
    topology.bucket_offsets = ptr<std::int64_t>(2);
    topology.bucket_systems = ptr<std::int32_t>(1);
    topology.bucket_orbital_counts = ptr<std::int32_t>(1);
  }

  void make_projections() {
    auto& topology = plan.topology;
    auto& atom = plan.atom_projection;
    atom.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    atom.plan_token = kToken;
    atom.batch_size = kBatch;
    atom.total_atoms = kAtoms;
    atom.atom_offset_count = topology.atom_offset_count;
    atom.atom_offsets = topology.atom_offsets;

    auto& shell = plan.shell_ownership_projection;
    shell.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    shell.plan_token = kToken;
    shell.batch_size = kBatch;
    shell.total_atoms = kAtoms;
    shell.total_shells = kShells;
    shell.batch_shell_offset_count = topology.batch_shell_offset_count;
    shell.atom_shell_offset_count = topology.atom_shell_offset_count;
    shell.shell_to_atom_count = topology.shell_to_atom_count;
    shell.batch_shell_offsets = topology.batch_shell_offsets;
    shell.atom_shell_offsets = topology.atom_shell_offsets;
    shell.shell_to_atom = topology.shell_to_atom;

    auto& ao = plan.ao_matrix_projection;
    ao.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    ao.plan_token = kToken;
    ao.batch_size = kBatch;
    ao.total_shells = kShells;
    ao.total_orbitals = kOrbitals;
    ao.total_matrix_elements = kMatrices;
    ao.batch_orbital_offset_count = topology.batch_orbital_offset_count;
    ao.matrix_offset_count = topology.matrix_offset_count;
    ao.shell_orbital_offset_count = topology.shell_orbital_offset_count;
    ao.orbital_to_shell_count = topology.orbital_to_shell_count;
    ao.orbital_to_atom_count = topology.orbital_to_atom_count;
    ao.batch_orbital_offsets = topology.batch_orbital_offsets;
    ao.matrix_offsets = topology.matrix_offsets;
    ao.shell_orbital_offsets = topology.shell_orbital_offsets;
    ao.orbital_to_shell = topology.orbital_to_shell;
    ao.orbital_to_atom = topology.orbital_to_atom;

    auto& buckets = plan.ao_bucket_projection;
    buckets.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    buckets.plan_token = kToken;
    buckets.batch_size = kBatch;
    buckets.bucket_count = topology.bucket_count;
    buckets.bucket_offset_count = topology.bucket_offset_count;
    buckets.bucket_system_count = topology.bucket_system_count;
    buckets.bucket_orbital_count = topology.bucket_orbital_count;
    buckets.bucket_offsets = topology.bucket_offsets;
    buckets.bucket_systems = topology.bucket_systems;
    buckets.bucket_orbital_counts = topology.bucket_orbital_counts;

    auto& element = plan.element_identity_projection;
    element.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    element.plan_token = kToken;
    element.total_atoms = kAtoms;
    element.atomic_number_count = kAtoms;
    element.atomic_numbers = ptr<std::int32_t>(1);
    element.element_fingerprint = 0xa1b2c3d4e5f60718ULL;
  }

  void make_plan() {
    make_topology();
    make_projections();
    plan.abi_version = kGfn2SccIterationAbiVersion;
    plan.enabled_components = static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES2) |
                              static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kES3) |
                              static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kAES2);
    plan.plan_token = kToken;
    plan.geometry_generation = 7u;

    auto& layout = plan.wavefunction_layout;
    layout.memory_space = Gfn2PlanMemorySpace::kCudaDevice;
    layout.plan_token = kToken;
    layout.layout_fingerprint = 0x51cc0deULL;
    layout.batch_size = kBatch;
    layout.total_spin_channels = kSpinChannels;
    layout.total_spin_orbitals = kSpinOrbitals;
    layout.total_spin_matrix_elements = kSpinMatrices;
    layout.total_spin_shells = kSpinShells;
    layout.total_spin_atoms = kSpinAtoms;
    layout.spin_channel_count = kBatch;
    layout.spin_channel_offset_count = kBatch + 1;
    layout.spin_orbital_offset_count = kBatch + 1;
    layout.spin_matrix_offset_count = kBatch + 1;
    layout.spin_shell_offset_count = kBatch + 1;
    layout.spin_atom_offset_count = kBatch + 1;
    layout.spin_channels = ptr<std::int32_t>(kBatch);
    layout.spin_channel_offsets = ptr<std::int64_t>(kBatch + 1);
    layout.spin_orbital_offsets = ptr<std::int64_t>(kBatch + 1);
    layout.spin_matrix_offsets = ptr<std::int64_t>(kBatch + 1);
    layout.spin_shell_offsets = ptr<std::int64_t>(kBatch + 1);
    layout.spin_atom_offsets = ptr<std::int64_t>(kBatch + 1);

    plan.activity_policy = {kBatch, 50u, kToken};
    plan.state_policy = {50u, 1.0e-6, 1.0e-8, kToken};
    plan.mixer_policy = {kHistory, 0.4, 1.0e-6, 1.0e-4, kToken};
    plan.provenance.expected_geometry_generation = 7u;
    plan.provenance.plan_token = kToken;

    plan.geometry_batch = {kBatch,
                           kAtoms,
                           0,
                           2,
                           2,
                           kAtoms,
                           3 * kAtoms,
                           kToken,
                           plan.topology.atom_offsets,
                           ptr<std::int64_t>(2),
                           ptr<double>(kAtoms)};
    plan.geometry_cache = {nullptr, 0,     ptr<double>(kAtoms), kAtoms, ptr<std::uint64_t>(kBatch),
                           kBatch,  kToken};
    plan.scc_batch = {kBatch,
                      kShells,
                      kAtoms,
                      2,
                      2,
                      kToken,
                      plan.topology.batch_shell_offsets,
                      plan.topology.atom_offsets};

    auto& spin = plan.spin_batch;
    spin.batch_size = kBatch;
    spin.total_atoms = kAtoms;
    spin.total_shells = kShells;
    spin.shell_population_elements = kSpinShells;
    spin.atom_offset_count = kBatch + 1;
    spin.batch_shell_offset_count = kBatch + 1;
    spin.atom_shell_offset_count = kAtoms + 1;
    spin.shell_population_offset_count = kBatch + 1;
    spin.spin_channel_count = kBatch;
    spin.coupling_offset_count = kAtoms + 1;
    spin.coupling_matrix_count = kShells * kShells;
    spin.plan_token = kToken;
    spin.atom_offsets = plan.topology.atom_offsets;
    spin.batch_shell_offsets = plan.topology.batch_shell_offsets;
    spin.atom_shell_offsets = plan.topology.atom_shell_offsets;
    spin.shell_population_offsets = layout.spin_shell_offsets;
    spin.spin_channels = layout.spin_channels;
    spin.coupling_offsets = ptr<std::int64_t>(kAtoms + 1);
    spin.coupling_matrices = ptr<double>(spin.coupling_matrix_count);

    auto& potential = plan.potential_batch;
    potential.batch_size = kBatch;
    potential.total_atoms = kAtoms;
    potential.total_shells = kShells;
    potential.plan_token = kToken;
    potential.atom_offset_count = 2;
    potential.batch_shell_offset_count = 2;
    potential.qsh_offset_count = 2;
    potential.qat_offset_count = 2;
    potential.dipole_offset_count = 2;
    potential.quadrupole_offset_count = 2;
    potential.shell_to_atom_count = kShells;
    potential.atom_offsets = plan.topology.atom_offsets;
    potential.batch_shell_offsets = plan.topology.batch_shell_offsets;
    potential.qsh_offsets = plan.topology.batch_shell_offsets;
    potential.qat_offsets = plan.topology.atom_offsets;
    potential.dipole_offsets = ptr<std::int64_t>(2);
    potential.quadrupole_offsets = ptr<std::int64_t>(2);
    potential.shell_to_atom = plan.topology.shell_to_atom;

    auto& es2 = plan.es2_batch;
    es2.batch_size = kBatch;
    es2.total_atoms = kAtoms;
    es2.total_shells = kShells;
    es2.total_matrix_elements = 1;
    es2.plan_token = kToken;
    es2.atom_offset_count = 2;
    es2.batch_shell_offset_count = 2;
    es2.atom_shell_offset_count = 2;
    es2.matrix_offset_count = 2;
    es2.shell_to_atom_count = 1;
    es2.shell_hardness_count = 1;
    es2.atom_offsets = plan.topology.atom_offsets;
    es2.batch_shell_offsets = plan.topology.batch_shell_offsets;
    es2.atom_shell_offsets = plan.topology.atom_shell_offsets;
    es2.matrix_offsets = ptr<std::int64_t>(2);
    es2.shell_to_atom = plan.topology.shell_to_atom;
    es2.shell_hardness = ptr<double>(1);
    plan.es2_cache = {ptr<double>(1), 1, 7u, kToken};

    plan.es3_batch = {
        kBatch, kShells, 2, kShells, plan.topology.batch_shell_offsets, ptr<double>(kShells),
        kToken};
    auto& aes2 = plan.aes2_batch;
    aes2.batch_size = kBatch;
    aes2.total_atoms = kAtoms;
    aes2.total_pairs = 0;
    aes2.plan_token = kToken;
    aes2.atom_offset_count = 2;
    aes2.pair_offset_count = 2;
    aes2.dipole_kernel_count = kAtoms;
    aes2.quadrupole_kernel_count = kAtoms;
    aes2.multipole_radius_count = kAtoms;
    aes2.multipole_valence_cn_count = kAtoms;
    aes2.atom_offsets = plan.topology.atom_offsets;
    aes2.pair_offsets = plan.geometry_batch.pair_offsets;
    aes2.dipole_kernel = ptr<double>(kAtoms);
    aes2.quadrupole_kernel = ptr<double>(kAtoms);
    aes2.multipole_radius = ptr<double>(kAtoms);
    aes2.multipole_valence_cn = ptr<double>(kAtoms);
    plan.aes2_cache = {nullptr, 0, 7u, kToken};

    plan.scalar_bridge_batch.topology = plan.topology;
    plan.scalar_bridge_batch.qsh_offset_count = 2;
    plan.scalar_bridge_batch.qat_offset_count = 2;
    plan.scalar_bridge_batch.qsh_offsets = potential.qsh_offsets;
    plan.scalar_bridge_batch.qat_offsets = potential.qat_offsets;

    auto& h = plan.hamiltonian_batch;
    h.batch_size = kBatch;
    h.total_atoms = kAtoms;
    h.total_shells = kShells;
    h.total_orbitals = kOrbitals;
    h.total_matrix_elements = kMatrices;
    h.plan_token = kToken;
    h.atom_offset_count = h.batch_shell_offset_count = h.batch_orbital_offset_count = 2;
    h.matrix_offset_count = h.atom_shell_offset_count = h.shell_orbital_offset_count = 2;
    h.shell_to_atom_count = h.orbital_to_shell_count = h.orbital_to_atom_count = 1;
    h.atom_offsets = plan.topology.atom_offsets;
    h.batch_shell_offsets = plan.topology.batch_shell_offsets;
    h.batch_orbital_offsets = plan.topology.batch_orbital_offsets;
    h.matrix_offsets = plan.topology.matrix_offsets;
    h.atom_shell_offsets = plan.topology.atom_shell_offsets;
    h.shell_orbital_offsets = plan.topology.shell_orbital_offsets;
    h.shell_to_atom = plan.topology.shell_to_atom;
    h.orbital_to_shell = plan.topology.orbital_to_shell;
    h.orbital_to_atom = plan.topology.orbital_to_atom;

    plan.eigensolver_batch = {kBatch,
                              kOrbitals,
                              kMatrices,
                              2,
                              2,
                              kBatch,
                              kBatch,
                              kToken,
                              plan.topology.batch_orbital_offsets,
                              plan.topology.matrix_offsets,
                              plan.topology.bucket_systems,
                              nullptr};
    plan.overlap_cache = {ptr<double>(kMatrices),
                          kMatrices,
                          ptr<std::uint64_t>(kBatch),
                          kBatch,
                          ptr<std::uint32_t>(kBatch),
                          kBatch,
                          kToken};
    plan.eigensolver_provider.buckets = buckets.data();
    plan.eigensolver_provider.bucket_count = 1;
    plan.eigensolver_provider.solver = reinterpret_cast<cusolverDnHandle_t>(0x1000);
    plan.eigensolver_provider.parameters = reinterpret_cast<cusolverDnParams_t>(0x2000);
    plan.eigensolver_provider.blas = reinterpret_cast<cublasHandle_t>(0x3000);
    plan.eigensolver_provider.device_workspace = device.allocate_bytes(256u, alignof(double));
    plan.eigensolver_provider.device_workspace_bytes = 256u;
    plan.eigensolver_provider.host_workspace =
        device.allocate_bytes(128u, alignof(std::max_align_t));
    plan.eigensolver_provider.host_workspace_bytes = 128u;
    plan.eigensolver_provider.requirements = {128u, 64u};
    plan.eigensolver_provider.capture_mode = Gfn2SccIterationProviderCaptureMode::kGraphSupported;
    plan.eigensolver_provider.plan_token = kToken;

    auto& bucket = buckets[0];
    bucket.orbital_count = static_cast<std::int32_t>(kOrbitals);
    bucket.system_count = static_cast<std::int32_t>(kBatch);
    bucket.system_index_offset = 0;
    bucket.matrix_scratch_offset = 0;
    bucket.orbital_scratch_offset = 0;
    bucket.solve_count = static_cast<std::int32_t>(kSpinChannels);
    bucket.solve_index_offset = 0;
    bucket.spin_matrix_scratch_offset = 0;
    bucket.spin_orbital_scratch_offset = 0;

    plan.occupations_batch = {kBatch,
                              kOrbitals,
                              2,
                              2 * kBatch,
                              kBatch,
                              kBatch,
                              kToken,
                              plan.topology.batch_orbital_offsets,
                              ptr<double>(2 * kBatch),
                              ptr<double>(kBatch),
                              nullptr};
    plan.density_batch = {kBatch,
                          kOrbitals,
                          kMatrices,
                          2,
                          2,
                          kToken,
                          plan.topology.batch_orbital_offsets,
                          plan.topology.matrix_offsets};

    auto& mulliken = plan.mulliken_batch;
    mulliken.batch_size = kBatch;
    mulliken.total_atoms = kAtoms;
    mulliken.total_shells = kShells;
    mulliken.total_orbitals = kOrbitals;
    mulliken.total_matrix_elements = kMatrices;
    mulliken.maximum_system_atoms = kAtoms;
    mulliken.maximum_system_shells = kShells;
    mulliken.plan_token = kToken;
    mulliken.atom_offset_count = mulliken.batch_shell_offset_count =
        mulliken.batch_orbital_offset_count = mulliken.matrix_offset_count =
            mulliken.atom_shell_offset_count = mulliken.shell_orbital_offset_count = 2;
    mulliken.shell_to_atom_count = mulliken.reference_occupation_count = 1;
    mulliken.atom_offsets = plan.topology.atom_offsets;
    mulliken.batch_shell_offsets = plan.topology.batch_shell_offsets;
    mulliken.batch_orbital_offsets = plan.topology.batch_orbital_offsets;
    mulliken.matrix_offsets = plan.topology.matrix_offsets;
    mulliken.atom_shell_offsets = plan.topology.atom_shell_offsets;
    mulliken.shell_orbital_offsets = plan.topology.shell_orbital_offsets;
    mulliken.shell_to_atom = plan.topology.shell_to_atom;
    mulliken.reference_shell_occupations = ptr<double>(1);

    plan.electronic_energy_batch = {kBatch, kMatrices, 2, kToken, plan.topology.matrix_offsets};
    plan.classical_energy_batch = {kBatch, plan.enabled_components, kToken};
    plan.free_energy_batch = {kBatch, plan.enabled_components, 0.01, kToken};
    auto& publication = plan.publication_plan;
    publication.batch_size = kBatch;
    publication.total_atoms = kAtoms;
    publication.total_shells = kShells;
    publication.total_orbitals = kOrbitals;
    publication.total_matrix_elements = kMatrices;
    publication.total_mixer_vector_elements = kMixerVector;
    publication.history_size = kHistory;
    publication.atom_offset_count = kBatch + 1;
    publication.shell_offset_count = kBatch + 1;
    publication.orbital_offset_count = kBatch + 1;
    publication.matrix_offset_count = kBatch + 1;
    publication.shell_to_atom_count = kShells;
    publication.atom_offsets = plan.topology.atom_offsets;
    publication.shell_offsets = plan.topology.batch_shell_offsets;
    publication.orbital_offsets = plan.topology.batch_orbital_offsets;
    publication.matrix_offsets = plan.topology.matrix_offsets;
    publication.shell_to_atom = plan.topology.shell_to_atom;
    publication.wavefunction_layout = plan.wavefunction_layout;
    publication.maximum_iterations = plan.activity_policy.maximum_iterations;
    publication.residual_rms_tolerance = plan.mixer_policy.rms_tolerance;
    publication.energy_tolerance = plan.state_policy.energy_tolerance;
    publication.plan_token = kToken;
  }

  void make_state_and_workspace() {
    state.plan_token = kToken;
    state.eigenpairs = eigenpairs();
    state.occupations = occupations();
    state.density = density();
    state.raw_population = population();
    state.spin_energies = ptr<double>(kBatch);
    state.spin_energy_elements = kBatch;
    state.classical_energy = classical_diagnostics();
    state.free_energy = free_diagnostics(state.classical_energy, state.spin_energies);
    state.mixer = mixer();
    state.published = {state.raw_population.qsh,
                       kSpinShells,
                       state.raw_population.dipole,
                       kSpinDipoles,
                       state.raw_population.quadrupole,
                       kSpinQuadrupoles,
                       kToken};
    state.scc.current_inputs = multipoles();
    state.scc.free_energies = ptr<double>(kBatch);
    state.scc.previous_free_energies = ptr<double>(kBatch);
    state.scc.free_energy_changes = ptr<double>(kBatch);
    state.scc.residual_rms = ptr<double>(kBatch);
    state.scc.iterations = ptr<std::uint64_t>(kBatch);
    state.scc.system_statuses = ptr<gpuxtb_status_t>(kBatch);
    state.scc.converged = ptr<std::uint8_t>(kBatch);
    state.scc.batch_elements = kBatch;
    state.scc.plan_token = kToken;
    state.publication.wavefunction = {state.eigenpairs, state.occupations, state.density,
                                      state.raw_population, kToken};
    state.publication.energy.classical = state.classical_energy;
    state.publication.energy.free_energy = state.free_energy;
    state.publication.energy.spin_energies = state.spin_energies;
    state.publication.energy.spin_energy_elements = kBatch;
    state.publication.energy.plan_token = kToken;
    state.publication.mixer = state.mixer;
    state.publication.published = state.published;
    state.publication.scc = state.scc;
    state.publication.plan_token = kToken;

    workspace.plan_token = kToken;
    workspace.ledger = {ptr<std::uint8_t>(kBatch),
                        ptr<gpuxtb_status_t>(kBatch),
                        ptr<std::uint64_t>(kBatch),
                        ptr<std::uint64_t>(1),
                        ptr<std::uint32_t>(1),
                        kBatch,
                        1,
                        kToken};
    workspace.activity = {workspace.ledger.active_mask, workspace.ledger.sequence_active, kBatch, 1,
                          kToken};
    workspace.potential_activity = {workspace.ledger.active_mask, kBatch, kToken};
    workspace.hamiltonian_activity = {workspace.ledger.active_mask, kBatch, kToken};
    workspace.mulliken_activity = {workspace.ledger.active_mask, kBatch, kToken};
    workspace.classical_energy_activity = {workspace.ledger.active_mask, kBatch, kToken};
    workspace.free_energy_activity = {workspace.ledger.active_mask, kBatch, kToken};
    plan.eigensolver_batch.active = workspace.ledger.active_mask;
    plan.occupations_batch.active = workspace.ledger.active_mask;

    workspace.staged_eigenpairs = eigenpairs();
    workspace.staged_occupations = occupations();
    workspace.staged_density = density();
    workspace.staged_raw_population = population();
    workspace.staged_spin_energies = ptr<double>(kBatch);
    workspace.staged_spin_energy_elements = kBatch;
    workspace.staged_classical_energy = classical_diagnostics();
    workspace.staged_free_energy =
        free_diagnostics(workspace.staged_classical_energy, workspace.staged_spin_energies);
    workspace.staged_free_energy.entropy = workspace.staged_occupations.entropies;
    workspace.staged_mixer = mixer();
    workspace.next_mixed = multipoles();
    workspace.staged_publication.wavefunction = {
        workspace.staged_eigenpairs, workspace.staged_occupations, workspace.staged_density,
        workspace.staged_raw_population, kToken};
    workspace.staged_publication.energy.classical = workspace.staged_classical_energy;
    workspace.staged_publication.energy.free_energy = workspace.staged_free_energy;
    workspace.staged_publication.energy.spin_energies = workspace.staged_spin_energies;
    workspace.staged_publication.energy.spin_energy_elements = kBatch;
    workspace.staged_publication.energy.plan_token = kToken;
    workspace.staged_publication.mixer = workspace.staged_mixer;
    workspace.staged_publication.next_mixed = {workspace.next_mixed.shell_charges,
                                               workspace.next_mixed.shell_elements,
                                               workspace.next_mixed.atomic_dipoles,
                                               workspace.next_mixed.dipole_elements,
                                               workspace.next_mixed.atomic_quadrupoles,
                                               workspace.next_mixed.quadrupole_elements,
                                               kToken};
    workspace.staged_publication.plan_token = kToken;

    workspace.mixed_topology = {state.scc.current_inputs.shell_charges,
                                kSpinShells,
                                ptr<double>(kSpinAtoms),
                                kSpinAtoms,
                                state.scc.current_inputs.atomic_dipoles,
                                kSpinDipoles,
                                state.scc.current_inputs.atomic_quadrupoles,
                                kSpinQuadrupoles,
                                kToken};
    workspace.physical_topology = {ptr<double>(kShells),
                                   kShells,
                                   ptr<double>(kAtoms),
                                   kAtoms,
                                   ptr<double>(kDipoles),
                                   kDipoles,
                                   ptr<double>(kQuadrupoles),
                                   kQuadrupoles,
                                   kToken};
    auto& storage = workspace.components;
    storage.es2_shell_potential = ptr<double>(kShells);
    storage.es2_shell_elements = kShells;
    storage.es3_shell_potential = ptr<double>(kShells);
    storage.es3_shell_elements = kShells;
    storage.aes2_atomic_potential = ptr<double>(kAtoms);
    storage.aes2_atomic_elements = kAtoms;
    storage.aes2_dipole_potential = ptr<double>(kDipoles);
    storage.aes2_dipole_elements = kDipoles;
    storage.aes2_quadrupole_potential = ptr<double>(kQuadrupoles);
    storage.aes2_quadrupole_elements = kQuadrupoles;
    storage.es2_energy = ptr<double>(kBatch);
    storage.es2_energy_elements = kBatch;
    storage.es3_energy = ptr<double>(kBatch);
    storage.es3_energy_elements = kBatch;
    storage.aes2_energy = ptr<double>(kBatch);
    storage.aes2_energy_elements = kBatch;
    storage.core_energy = ptr<double>(kBatch);
    storage.core_energy_elements = kBatch;
    storage.electronic_free_energy = ptr<double>(kBatch);
    storage.electronic_free_energy_elements = kBatch;
    storage.plan_token = kToken;
    workspace.potential_components.enabled_components = plan.enabled_components;
    workspace.potential_components.es2_shell = storage.es2_shell_potential;
    workspace.potential_components.es2_shell_elements = kShells;
    workspace.potential_components.es3_shell = storage.es3_shell_potential;
    workspace.potential_components.es3_shell_elements = kShells;
    workspace.potential_components.aes2_atomic = storage.aes2_atomic_potential;
    workspace.potential_components.aes2_atomic_elements = kAtoms;
    workspace.potential_components.aes2_dipole = storage.aes2_dipole_potential;
    workspace.potential_components.aes2_dipole_elements = kDipoles;
    workspace.potential_components.aes2_quadrupole = storage.aes2_quadrupole_potential;
    workspace.potential_components.aes2_quadrupole_elements = kQuadrupoles;
    workspace.potential_components.plan_token = kToken;
    workspace.complete_potentials = {ptr<double>(kSpinShells),
                                     kSpinShells,
                                     ptr<double>(kSpinAtoms),
                                     kSpinAtoms,
                                     ptr<double>(kSpinDipoles),
                                     kSpinDipoles,
                                     ptr<double>(kSpinQuadrupoles),
                                     kSpinQuadrupoles,
                                     kToken};
    /* The stable stage observes complete spin fields but keeps restricted output scratch. */
    workspace.scalar_bridge.fields = {workspace.complete_potentials.shell, kSpinShells,
                                      workspace.complete_potentials.atomic, kSpinAtoms, kToken};
    workspace.scalar_bridge.shell_scalar = ptr<double>(kShells);
    workspace.scalar_bridge.shell_elements = kShells;
    workspace.scalar_bridge.workspace = {ptr<double>(kShells), kShells, ptr<std::uint32_t>(1), 1,
                                         kToken};
    workspace.scalar_bridge.plan_token = kToken;
    workspace.hamiltonian = {ptr<double>(kSpinMatrices), kSpinMatrices, kToken};

    workspace.geometry_workspace = {nullptr,
                                    0,
                                    ptr<double>(kAtoms),
                                    kAtoms,
                                    ptr<double>(kDipoles),
                                    kDipoles,
                                    ptr<std::uint32_t>(1),
                                    1,
                                    kToken};
    workspace.es2_workspace = {ptr<double>(1),      1,      ptr<double>(kShells),  kShells,
                               ptr<double>(kBatch), kBatch, ptr<double>(kDipoles), kDipoles};
    workspace.aes2_workspace = {nullptr,
                                0,
                                ptr<double>(kGfn2AES2PotentialElementsPerAtom * kAtoms),
                                kGfn2AES2PotentialElementsPerAtom * kAtoms,
                                ptr<double>(kBatch),
                                kBatch,
                                ptr<double>(kDipoles),
                                kDipoles,
                                ptr<double>(kAtoms),
                                kAtoms,
                                ptr<std::uint32_t>(1),
                                1};
    workspace.potential_workspace = {ptr<double>(kSpinShells),
                                     kSpinShells,
                                     ptr<double>(kSpinAtoms),
                                     kSpinAtoms,
                                     ptr<double>(kSpinDipoles),
                                     kSpinDipoles,
                                     ptr<double>(kSpinQuadrupoles),
                                     kSpinQuadrupoles,
                                     ptr<std::uint32_t>(1),
                                     1,
                                     kToken};
    workspace.hamiltonian_workspace = {ptr<double>(kSpinMatrices), kSpinMatrices,
                                       ptr<std::uint32_t>(1), 1, kToken};

    auto& eig = workspace.eigensolver_workspace;
    eig.matrix_scratch_a = ptr<double>(kSpinMatrices);
    eig.matrix_a_elements = kSpinMatrices;
    eig.matrix_scratch_b = ptr<double>(kSpinMatrices);
    eig.matrix_b_elements = kSpinMatrices;
    eig.eigenvalue_scratch = ptr<double>(kSpinOrbitals);
    eig.eigenvalue_elements = kSpinOrbitals;
    eig.factor_pointers = ptr<double*>(kSpinChannels);
    eig.factor_pointer_elements = kSpinChannels;
    eig.matrix_pointers = ptr<double*>(kSpinChannels);
    eig.matrix_pointer_elements = kSpinChannels;
    eig.info_a = ptr<int>(kSpinChannels);
    eig.info_a_elements = kSpinChannels;
    eig.info_b = ptr<int>(kSpinChannels);
    eig.info_b_elements = kSpinChannels;
    eig.eligible = ptr<std::uint8_t>(kSpinChannels);
    eig.eligible_elements = kSpinChannels;
    eig.sequence_active = ptr<std::uint32_t>(1);
    eig.sequence_active_elements = 1;
    eig.solver_device_workspace = plan.eigensolver_provider.device_workspace;
    eig.solver_device_workspace_bytes = plan.eigensolver_provider.device_workspace_bytes;
    eig.solver_host_workspace = plan.eigensolver_provider.host_workspace;
    eig.solver_host_workspace_bytes = plan.eigensolver_provider.host_workspace_bytes;
    eig.plan_token = kToken;
    eig.compact_systems = ptr<std::int32_t>(kSpinChannels);
    eig.compact_system_elements = kSpinChannels;
    eig.compact_source_slots = ptr<std::int32_t>(kSpinChannels);
    eig.compact_source_slot_elements = kSpinChannels;
    eig.bucket_activity = ptr<Gfn2EigensolverBucketActivity>(1);
    eig.bucket_activity_elements = 1;

    workspace.occupations_workspace = {ptr<double>(2 * kOrbitals),
                                       2 * kOrbitals,
                                       ptr<double>(2 * kBatch),
                                       2 * kBatch,
                                       ptr<double>(2 * kBatch),
                                       2 * kBatch,
                                       ptr<double>(kBatch),
                                       kBatch,
                                       ptr<std::uint32_t>(1),
                                       1,
                                       kToken};
    workspace.density_workspace = {ptr<double>(kSpinMatrices),
                                   kSpinMatrices,
                                   ptr<double>(kSpinMatrices),
                                   kSpinMatrices,
                                   ptr<double>(kSpinOrbitals),
                                   kSpinOrbitals,
                                   ptr<double>(kSpinOrbitals),
                                   kSpinOrbitals,
                                   ptr<double>(kBatch),
                                   kBatch,
                                   ptr<double>(kBatch),
                                   kBatch,
                                   ptr<double>(kBatch),
                                   kBatch,
                                   ptr<double>(kBatch),
                                   kBatch,
                                   ptr<std::uint32_t>(1),
                                   1,
                                   kToken,
                                   ptr<double>(kSpinChannels),
                                   kSpinChannels,
                                   ptr<double>(kSpinChannels),
                                   kSpinChannels,
                                   ptr<double>(kSpinChannels),
                                   kSpinChannels,
                                   ptr<double>(kSpinChannels),
                                   kSpinChannels};
    workspace.mulliken_workspace = {ptr<double>(kSpinShells),
                                    kSpinShells,
                                    ptr<double>(kSpinAtoms),
                                    kSpinAtoms,
                                    ptr<double>(kSpinDipoles),
                                    kSpinDipoles,
                                    ptr<double>(kSpinQuadrupoles),
                                    kSpinQuadrupoles,
                                    ptr<std::uint32_t>(1),
                                    1,
                                    kToken};
    workspace.spin_output.spin_energies = workspace.staged_spin_energies;
    workspace.spin_output.spin_energy_elements = kBatch;
    workspace.spin_output.shell_potentials = ptr<double>(kSpinShells);
    workspace.spin_output.shell_potential_elements = kSpinShells;
    workspace.spin_output.plan_token = kToken;
    workspace.spin_workspace.energy_scratch = ptr<double>(kBatch);
    workspace.spin_workspace.energy_elements = kBatch;
    workspace.spin_workspace.potential_scratch = ptr<double>(kSpinShells);
    workspace.spin_workspace.potential_elements = kSpinShells;
    workspace.spin_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.spin_workspace.sequence_elements = 1;
    workspace.spin_workspace.plan_token = kToken;
    workspace.electronic_energy_workspace = {
        ptr<double>(kBatch), ptr<double>(kBatch), ptr<std::uint32_t>(1), kBatch, 1, kToken};
    workspace.classical_energy_workspace = {
        ptr<double>(kGfn2SccClassicalDiagnosticComponents * kBatch),
        kGfn2SccClassicalDiagnosticComponents * kBatch, ptr<std::uint32_t>(1), 1, kToken};
    workspace.free_energy_workspace = {ptr<double>(kGfn2SccFreeEnergyDiagnosticComponents * kBatch),
                                       kGfn2SccFreeEnergyDiagnosticComponents * kBatch,
                                       ptr<std::uint32_t>(1), 1, kToken};
    workspace.mixer_workspace.residual = ptr<double>(kMixerVector);
    workspace.mixer_workspace.mixed = ptr<double>(kMixerVector);
    workspace.mixer_workspace.delta_f = ptr<double>(kMixerVector);
    workspace.mixer_workspace.new_u = ptr<double>(kMixerVector);
    workspace.mixer_workspace.beta = ptr<double>(kBatch * kHistory * kHistory);
    workspace.mixer_workspace.coefficients = ptr<double>(kBatch * kHistory);
    workspace.mixer_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.mixer_workspace.vector_elements = kMixerVector;
    workspace.mixer_workspace.beta_elements = kBatch * kHistory * kHistory;
    workspace.mixer_workspace.coefficient_elements = kBatch * kHistory;
    workspace.mixer_workspace.sequence_elements = 1;
    workspace.mixer_workspace.plan_token = kToken;
    workspace.mixer_device_error = ptr<std::uint32_t>(1);
    workspace.mixer_device_error_elements = 1;
    workspace.publication_workspace.system_errors = ptr<std::uint32_t>(kBatch);
    workspace.publication_workspace.system_error_elements = kBatch;
    workspace.publication_workspace.device_error = ptr<std::uint32_t>(1);
    workspace.publication_workspace.device_error_elements = 1;
    workspace.publication_workspace.sequence_active = ptr<std::uint32_t>(1);
    workspace.publication_workspace.sequence_elements = 1;
    workspace.publication_workspace.mixed_atomic_charges = workspace.mixed_topology.atomic_charges;
    workspace.publication_workspace.mixed_atomic_charge_elements = kSpinAtoms;
    workspace.publication_workspace.previous_free_energies = ptr<double>(kBatch);
    workspace.publication_workspace.free_energy_changes = ptr<double>(kBatch);
    workspace.publication_workspace.next_iterations = ptr<std::uint64_t>(kBatch);
    workspace.publication_workspace.next_converged = ptr<std::uint8_t>(kBatch);
    workspace.publication_workspace.next_statuses = ptr<gpuxtb_status_t>(kBatch);
    workspace.publication_workspace.batch_elements = kBatch;
    workspace.publication_workspace.plan_token = kToken;
  }

  void make_input() {
    input.plan_token = kToken;
    input.activity_state = {state.scc.iterations, state.scc.system_statuses, state.scc.converged,
                            kBatch, kToken};
    input.mixed_fields = {state.scc.current_inputs.shell_charges,
                          kSpinShells,
                          state.scc.current_inputs.atomic_dipoles,
                          kSpinDipoles,
                          state.scc.current_inputs.atomic_quadrupoles,
                          kSpinQuadrupoles,
                          kToken};
    input.mixed_spin = {input.mixed_fields.qsh, kSpinShells, kToken};
    input.hamiltonian = {ptr<double>(kMatrices),
                         kMatrices,
                         ptr<double>(kMatrices),
                         kMatrices,
                         ptr<double>(3 * kMatrices),
                         3 * kMatrices,
                         ptr<double>(6 * kMatrices),
                         6 * kMatrices,
                         workspace.complete_potentials.shell,
                         kSpinShells,
                         workspace.complete_potentials.dipole,
                         kSpinDipoles,
                         workspace.complete_potentials.quadrupole,
                         kSpinQuadrupoles,
                         kToken};
    input.eigensolver_hamiltonians = workspace.hamiltonian.matrix;
    input.eigensolver_hamiltonian_elements = kSpinMatrices;
    input.occupation_eigenvalues = workspace.staged_eigenpairs.eigenvalues;
    input.occupation_eigenvalue_elements = kSpinOrbitals;
    input.density = {workspace.staged_eigenpairs.coefficients,
                     kSpinMatrices,
                     workspace.staged_eigenpairs.eigenvalues,
                     kSpinOrbitals,
                     workspace.staged_occupations.occupations,
                     2 * kOrbitals,
                     workspace.ledger.active_mask,
                     kBatch,
                     kToken};
    input.mulliken = {workspace.staged_density.density,
                      kSpinMatrices,
                      input.hamiltonian.overlap,
                      kMatrices,
                      input.hamiltonian.dipole_integrals,
                      3 * kMatrices,
                      input.hamiltonian.quadrupole_integrals,
                      6 * kMatrices,
                      kToken};
    input.electronic_energy = {workspace.staged_density.density,
                               kSpinMatrices,
                               input.hamiltonian.h0,
                               kMatrices,
                               workspace.staged_occupations.entropies,
                               kBatch,
                               kToken};
    input.classical_energy = {workspace.components.es2_energy,
                              kBatch,
                              workspace.components.es3_energy,
                              kBatch,
                              workspace.components.aes2_energy,
                              kBatch,
                              nullptr,
                              0,
                              nullptr,
                              0,
                              nullptr,
                              0,
                              kToken};
    input.free_energy = {workspace.components.core_energy,
                         kBatch,
                         workspace.staged_occupations.entropies,
                         kBatch,
                         workspace.components.es2_energy,
                         kBatch,
                         workspace.components.es3_energy,
                         kBatch,
                         workspace.components.aes2_energy,
                         kBatch,
                         workspace.staged_spin_energies,
                         kBatch,
                         nullptr,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         0,
                         kToken};
    input.raw_multipoles = {workspace.staged_raw_population.qsh,
                            kSpinShells,
                            workspace.staged_raw_population.dipole,
                            kSpinDipoles,
                            workspace.staged_raw_population.quadrupole,
                            kSpinQuadrupoles,
                            kToken};
    input.raw_spin = {workspace.staged_raw_population.qsh, kSpinShells, kToken};
    input.complete_free_energies = workspace.staged_free_energy.free_energy;
    input.complete_free_energy_elements = kBatch;
  }

  void make_reports() {
    constexpr std::array<Gfn2SccStageId, kGfn2SccIterationBaseStageReportCount> stages{{
        Gfn2SccStageId::kMixedGather,      Gfn2SccStageId::kES2Potential,
        Gfn2SccStageId::kES3Potential,     Gfn2SccStageId::kAES2Potential,
        Gfn2SccStageId::kSpinPotential,    Gfn2SccStageId::kPotentialCompose,
        Gfn2SccStageId::kScalarBridge,     Gfn2SccStageId::kHamiltonian,
        Gfn2SccStageId::kEigensolver,      Gfn2SccStageId::kOccupations,
        Gfn2SccStageId::kDensity,          Gfn2SccStageId::kMulliken,
        Gfn2SccStageId::kSpinRawEnergy,    Gfn2SccStageId::kES2RawEnergy,
        Gfn2SccStageId::kES3RawEnergy,     Gfn2SccStageId::kAES2RawEnergy,
        Gfn2SccStageId::kClassicalEnergy,  Gfn2SccStageId::kElectronicEnergy,
        Gfn2SccStageId::kFreeEnergy,       Gfn2SccStageId::kMixer,
        Gfn2SccStageId::kStatePublication,
    }};
    plan.report_count = static_cast<std::int64_t>(stages.size());
    for (std::int64_t index = 0; index < plan.report_count; ++index) {
      auto& report = plan.reports[index];
      report.stage = stages[static_cast<std::size_t>(index)];
      report.system_code_format = Gfn2SccStageCodeFormat::kUint32Error;
      report.system_codes = ptr<std::uint32_t>(kBatch);
      report.system_code_elements = kBatch;
      report.device_error = ptr<std::uint32_t>(1);
      report.device_error_elements = 1;
      report.stage_sequence_active = ptr<std::uint32_t>(1);
      report.stage_sequence_elements = 1;
      report.plan_token = kToken;
      switch (report.stage) {
        case Gfn2SccStageId::kMixedGather:
          report.peer_error_mask = 0xfcu;
          report.stage_sequence_active = workspace.potential_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kES2Potential:
          report.device_code_role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
          report.peer_error_mask = 0x380u;
          break;
        case Gfn2SccStageId::kES3Potential:
          report.device_code_role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
          report.peer_error_mask = 0x1cu;
          break;
        case Gfn2SccStageId::kAES2Potential:
          report.device_code_role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
          report.peer_error_mask = 0x706u;
          break;
        case Gfn2SccStageId::kSpinPotential:
        case Gfn2SccStageId::kSpinRawEnergy:
          report.peer_error_mask = kGfn2SpinDevicePeerErrorMask;
          report.stage_sequence_active = workspace.spin_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kPotentialCompose:
          report.peer_error_mask = 0xff0cu;
          report.stage_sequence_active = workspace.potential_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kScalarBridge:
          report.device_code_role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
          report.peer_error_mask = 0xf0u;
          report.stage_sequence_active = workspace.scalar_bridge.workspace.sequence_active;
          break;
        case Gfn2SccStageId::kHamiltonian:
          report.peer_error_mask = 0x1feu;
          report.stage_sequence_active = workspace.hamiltonian_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kEigensolver:
          report.peer_error_mask = 0x3e06u;
          report.peer_failure_status = GPUXTB_STATUS_EIGENSOLVER_FAILED;
          report.stage_sequence_active = workspace.eigensolver_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kOccupations:
          report.peer_error_mask = 0x3feu;
          report.peer_failure_status = GPUXTB_STATUS_EIGENSOLVER_FAILED;
          report.stage_sequence_active = workspace.occupations_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kDensity:
          report.peer_error_mask = 0x7feu;
          report.peer_failure_status = GPUXTB_STATUS_EIGENSOLVER_FAILED;
          report.stage_sequence_active = workspace.density_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kMulliken:
          report.peer_error_mask = 0x1feu;
          report.stage_sequence_active = workspace.mulliken_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kES2RawEnergy:
          report.device_code_role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
          report.peer_error_mask = 0x980u;
          break;
        case Gfn2SccStageId::kES3RawEnergy:
          report.device_code_role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
          report.peer_error_mask = 0x4cu;
          break;
        case Gfn2SccStageId::kAES2RawEnergy:
          report.device_code_role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
          report.peer_error_mask = 0x1b06u;
          break;
        case Gfn2SccStageId::kClassicalEnergy:
          report.peer_error_mask = 0x1feu;
          report.stage_sequence_active = workspace.classical_energy_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kElectronicEnergy:
          report.peer_error_mask = 0xfcu;
          report.stage_sequence_active = workspace.electronic_energy_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kFreeEnergy:
          report.peer_error_mask = 0x1ffeu;
          report.stage_sequence_active = workspace.free_energy_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kMixer:
          report.peer_error_mask = 0x40u;
          report.stage_sequence_active = workspace.mixer_workspace.sequence_active;
          break;
        case Gfn2SccStageId::kStatePublication:
          report.device_code_role = Gfn2SccStageDeviceCodeRole::kPlanOnly;
          report.peer_error_mask = 0x1f8u;
          break;
        default:
          break;
      }
    }
    auto* mixer_report = static_cast<Gfn2SccStageDeviceReport*>(nullptr);
    auto* publication = static_cast<Gfn2SccStageDeviceReport*>(nullptr);
    for (std::int64_t index = 0; index < plan.report_count; ++index) {
      if (plan.reports[index].stage == Gfn2SccStageId::kMixer) mixer_report = &plan.reports[index];
      if (plan.reports[index].stage == Gfn2SccStageId::kStatePublication) {
        publication = &plan.reports[index];
      }
    }
    if (mixer_report == nullptr || publication == nullptr) return;
    mixer_report->system_code_format = Gfn2SccStageCodeFormat::kGpuxtbStatus;
    mixer_report->system_codes = workspace.staged_mixer.system_statuses;
    mixer_report->device_error = nullptr;
    mixer_report->device_error_elements = 0;
    publication->system_code_format = Gfn2SccStageCodeFormat::kUint32Error;
    publication->system_codes = workspace.publication_workspace.system_errors;
    publication->device_error = workspace.publication_workspace.device_error;
    publication->stage_sequence_active = workspace.publication_workspace.sequence_active;
  }

  Fixture() {
    make_plan();
    make_state_and_workspace();
    make_input();
    make_reports();
  }
};

int test_valid_binding_and_fail_closed_copy() {
  Fixture fixture;
  const auto valid = validate_gfn2_scc_iteration_binding_cuda(fixture.plan, fixture.input,
                                                              fixture.state, fixture.workspace);
  if (valid.error != Gfn2SccIterationBindingError::kSuccess) {
    std::fprintf(stderr, "valid fixture rejected: error=%u field=%u index=%lld\n",
                 static_cast<unsigned>(valid.error), static_cast<unsigned>(valid.field),
                 static_cast<long long>(valid.index));
  }
  CHECK(valid.error == Gfn2SccIterationBindingError::kSuccess);

  Gfn2SccIterationBinding binding{};
  auto diagnostic = bind_gfn2_scc_iteration_cuda(fixture.plan, fixture.input, fixture.state,
                                                 fixture.workspace, binding);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kSuccess);
  CHECK(binding.plan.plan_token == Fixture::kToken);
  CHECK(binding.input.complete_free_energies == fixture.input.complete_free_energies);

  fixture.input.plan_token ^= 1u;
  diagnostic = bind_gfn2_scc_iteration_cuda(fixture.plan, fixture.input, fixture.state,
                                            fixture.workspace, binding);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kCrossPlan);
  CHECK(binding.plan.plan_token == 0u);
  CHECK(binding.input.complete_free_energies == nullptr);
  return 0;
}

int test_capacity_alignment_alias_and_bucket_rejection() {
  {
    Fixture fixture;
    fixture.workspace.eigensolver_workspace.matrix_a_elements = 0;
    const auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(
        fixture.plan, fixture.input, fixture.state, fixture.workspace);
    CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInsufficientCapacity);
    CHECK(diagnostic.field == Gfn2SccIterationBindingField::kEigensolver);
  }
  {
    Fixture fixture;
    fixture.input.hamiltonian.h0 = reinterpret_cast<const double*>(
        reinterpret_cast<std::uintptr_t>(fixture.input.hamiltonian.h0) + 1u);
    fixture.input.electronic_energy.h0 = fixture.input.hamiltonian.h0;
    const auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(
        fixture.plan, fixture.input, fixture.state, fixture.workspace);
    CHECK(diagnostic.error == Gfn2SccIterationBindingError::kMisalignedPointer);
  }
  {
    Fixture fixture;
    fixture.workspace.hamiltonian_workspace.matrix_scratch = fixture.state.eigenpairs.coefficients;
    const auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(
        fixture.plan, fixture.input, fixture.state, fixture.workspace);
    CHECK(diagnostic.error == Gfn2SccIterationBindingError::kForbiddenAlias);
  }
  {
    Fixture fixture;
    fixture.buckets[0].matrix_scratch_offset = 1;
    const auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(
        fixture.plan, fixture.input, fixture.state, fixture.workspace);
    CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidBucket);
  }
  {
    Fixture fixture;
    fixture.plan.eigensolver_provider.device_workspace_bytes = 64u;
    fixture.workspace.eigensolver_workspace.solver_device_workspace_bytes = 64u;
    const auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(
        fixture.plan, fixture.input, fixture.state, fixture.workspace);
    CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInsufficientCapacity);
  }
  return 0;
}

int test_unrestricted_layout_and_spin_projection_rejection() {
  {
    Fixture fixture;
    fixture.plan.wavefunction_layout.memory_space = Gfn2PlanMemorySpace::kHost;
    fixture.plan.publication_plan.wavefunction_layout.memory_space = Gfn2PlanMemorySpace::kHost;
    const auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(
        fixture.plan, fixture.input, fixture.state, fixture.workspace);
    CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidTopology);
    CHECK(diagnostic.field == Gfn2SccIterationBindingField::kSpin);
  }
  {
    Fixture fixture;
    fixture.workspace.staged_density.channel_band_energy_elements = 0;
    fixture.workspace.staged_publication.wavefunction.density.channel_band_energy_elements = 0;
    const auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(
        fixture.plan, fixture.input, fixture.state, fixture.workspace);
    CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidCount);
    CHECK(diagnostic.field == Gfn2SccIterationBindingField::kDensity);
  }
  {
    Fixture fixture;
    /* Publication must carry the exact setup-time seal, not merely the same
     * device pointers and aggregate extents. */
    fixture.plan.publication_plan.wavefunction_layout.layout_fingerprint ^= 1u;
    const auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(
        fixture.plan, fixture.input, fixture.state, fixture.workspace);
    CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
    CHECK(diagnostic.field == Gfn2SccIterationBindingField::kStatePublication);
  }
  {
    Fixture fixture;
    fixture.input.raw_spin.shell_populations = fixture.input.mixed_spin.shell_populations;
    const auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(
        fixture.plan, fixture.input, fixture.state, fixture.workspace);
    CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
    CHECK(diagnostic.field == Gfn2SccIterationBindingField::kMulliken);
  }
  {
    Fixture fixture;
    for (std::int64_t index = 0; index < fixture.plan.report_count; ++index) {
      auto& report = fixture.plan.reports[index];
      if (report.stage == Gfn2SccStageId::kSpinPotential) {
        report.stage_sequence_active = fixture.workspace.potential_workspace.sequence_active;
        break;
      }
    }
    const auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(
        fixture.plan, fixture.input, fixture.state, fixture.workspace);
    CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidStageReport);
    CHECK(diagnostic.field == Gfn2SccIterationBindingField::kStageReports);
  }
  return 0;
}

int test_optional_canonical_null_and_report_extension_capacity() {
  Fixture fixture;
  CHECK(validate_gfn2_scc_iteration_binding_cuda(fixture.plan, fixture.input, fixture.state,
                                                 fixture.workspace)
            .error == Gfn2SccIterationBindingError::kSuccess);

  for (std::int64_t index = 0; index < fixture.plan.report_count; ++index) {
    if (fixture.plan.reports[index].stage == Gfn2SccStageId::kES2Potential) {
      fixture.plan.reports[index].device_code_role = Gfn2SccStageDeviceCodeRole::kMixedFirstError;
      break;
    }
  }
  auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(fixture.plan, fixture.input,
                                                             fixture.state, fixture.workspace);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidStageReport);

  fixture.make_reports();

  fixture.plan.d4_batch.plan_token = Fixture::kToken;
  diagnostic = validate_gfn2_scc_iteration_binding_cuda(fixture.plan, fixture.input, fixture.state,
                                                        fixture.workspace);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidCount);
  CHECK(diagnostic.field == Gfn2SccIterationBindingField::kD4);

  fixture.plan.d4_batch.plan_token = 0u;
  fixture.plan.report_count = kGfn2SccIterationStageReportCapacity + 1;
  diagnostic = validate_gfn2_scc_iteration_binding_cuda(fixture.plan, fixture.input, fixture.state,
                                                        fixture.workspace);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidStageReport);
  return 0;
}

int test_convergence_policy_has_one_rms_authority() {
  Fixture fixture;
  fixture.plan.state_policy.residual_rms_tolerance = fixture.plan.mixer_policy.rms_tolerance * 2.0;
  const auto diagnostic = validate_gfn2_scc_iteration_binding_cuda(
      fixture.plan, fixture.input, fixture.state, fixture.workspace);
  CHECK(diagnostic.error == Gfn2SccIterationBindingError::kInvalidCount);
  CHECK(diagnostic.field == Gfn2SccIterationBindingField::kStatePublication);
  return 0;
}

int test_launch_result_preserves_provider_domains() {
  Gfn2SccIterationLaunchResult result{};
  CHECK(result.success());
  result.status = Gfn2SccIterationLaunchStatus::kCusolverError;
  result.stage = Gfn2SccStageId::kEigensolver;
  result.cusolver_status = CUSOLVER_STATUS_INTERNAL_ERROR;
  CHECK(!result.success());
  CHECK(result.cuda_status == cudaSuccess);
  CHECK(result.cublas_status == CUBLAS_STATUS_SUCCESS);
  CHECK(result.cusolver_status == CUSOLVER_STATUS_INTERNAL_ERROR);
  return 0;
}

int test_projection_authority_rejection() {
  /* The sealed common projections are the plan's only topology authority: a
   * cross-plan token, a count/pointer substitution, or a forged element seal
   * must fail the binding exactly like a wrong leaf. */
  Fixture fixture;
  const auto validate = [&fixture]() {
    return validate_gfn2_scc_iteration_binding_cuda(fixture.plan, fixture.input, fixture.state,
                                                    fixture.workspace);
  };
  CHECK(validate().error == Gfn2SccIterationBindingError::kSuccess);

  /* Cross-plan token on the atom projection. */
  fixture.plan.atom_projection.plan_token = Fixture::kToken + 1u;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidTopology);
  fixture.plan.atom_projection.plan_token = Fixture::kToken;

  /* The setup-owned element seal is still part of this plan and atom domain. */
  fixture.plan.element_identity_projection.plan_token = Fixture::kToken + 1u;
  CHECK(validate().error == Gfn2SccIterationBindingError::kCrossPlan);
  fixture.plan.element_identity_projection.plan_token = Fixture::kToken;
  fixture.plan.element_identity_projection.total_atoms = Fixture::kAtoms + 1;
  fixture.plan.element_identity_projection.atomic_number_count = Fixture::kAtoms + 1;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidCount);
  fixture.plan.element_identity_projection.total_atoms = Fixture::kAtoms;
  fixture.plan.element_identity_projection.atomic_number_count = Fixture::kAtoms;

  /* Pointer substitution: leaf arrays must name the projection arrays, not a
   * different-but-equal allocation.  Substitute a foreign offset domain. */
  fixture.plan.ao_matrix_projection.matrix_offsets = fixture.plan.topology.atom_offsets;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidTopology);
  fixture.plan.ao_matrix_projection.matrix_offsets = fixture.plan.topology.matrix_offsets;

  /* Forged element identity seal: a CUDA descriptor proves identity through
   * the nonzero setup-time seal, so a zero seal must be rejected. */
  fixture.plan.element_identity_projection.element_fingerprint = 0u;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidTopology);
  fixture.plan.element_identity_projection.element_fingerprint = 0xa1b2c3d4e5f60718ULL;

  /* Presence mismatch: a packed-all-pair projection on a kNone plan. */
  fixture.plan.packed_all_pair_projection.plan_token = Fixture::kToken;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidTopology);
  fixture.plan.packed_all_pair_projection = {};
  CHECK(validate().error == Gfn2SccIterationBindingError::kSuccess);

  /* Disabled optional projections are canonical-empty, not merely tokenless.
   * A stale count or borrowed pointer must not survive setup and become a
   * future consumer's accidental topology authority. */
  fixture.plan.packed_all_pair_projection.total_pairs = 1;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidTopology);
  fixture.plan.packed_all_pair_projection = {};
  /* The fixture's topology intentionally has one live AO bucket; its disabled
   * form is exercised by the same plan gate in production setup when no bucket
   * domain is present. */

  /* Persistent SCC and potential leaves cannot substitute physical offsets. */
  fixture.plan.scc_batch.shell_offsets = fixture.plan.topology.atom_offsets;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.scc_batch.shell_offsets = fixture.plan.topology.batch_shell_offsets;
  fixture.plan.potential_batch.qsh_offsets = fixture.plan.topology.atom_offsets;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.potential_batch.qsh_offsets = fixture.plan.topology.batch_shell_offsets;
  fixture.plan.potential_batch.shell_to_atom = fixture.plan.topology.orbital_to_atom;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.potential_batch.shell_to_atom = fixture.plan.topology.shell_to_atom;

  /* Spin's physical maps and nspin-expanded fields have separate authorities. */
  fixture.plan.spin_batch.atom_shell_offsets = fixture.plan.topology.shell_orbital_offsets;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.spin_batch.atom_shell_offsets = fixture.plan.topology.atom_shell_offsets;
  fixture.plan.spin_batch.spin_channels = fixture.plan.element_identity_projection.atomic_numbers;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.spin_batch.spin_channels = fixture.plan.wavefunction_layout.spin_channels;

  /* ES2/ES3 and optional point-charge topology all borrow shell ownership. */
  fixture.plan.es2_batch.shell_to_atom = fixture.plan.topology.orbital_to_atom;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.es2_batch.shell_to_atom = fixture.plan.topology.shell_to_atom;
  fixture.plan.es3_batch.batch_shell_offsets = fixture.plan.topology.atom_offsets;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.es3_batch.batch_shell_offsets = fixture.plan.topology.batch_shell_offsets;

  fixture.plan.enabled_components |=
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kExplicitPointCharge);
  fixture.plan.explicit_point_charge_batch.atom_offsets = fixture.plan.topology.atom_offsets;
  fixture.plan.explicit_point_charge_batch.batch_shell_offsets =
      fixture.plan.topology.batch_shell_offsets;
  fixture.plan.explicit_point_charge_batch.shell_to_atom = fixture.plan.topology.orbital_to_atom;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.enabled_components &=
      ~static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kExplicitPointCharge);
  fixture.plan.explicit_point_charge_batch = {};

  /* D4's atom order and the common dense pair partition are setup authorities. */
  fixture.plan.enabled_components |=
      static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kD4TwoBody);
  fixture.plan.d4_batch.atom_offsets = fixture.plan.topology.atom_offsets;
  fixture.plan.d4_batch.pair_offsets = fixture.plan.geometry_batch.pair_offsets;
  fixture.plan.d4_batch.atomic_numbers = fixture.plan.wavefunction_layout.spin_channels;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.enabled_components &=
      ~static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kD4TwoBody);
  fixture.plan.d4_batch = {};

  fixture.plan.aes2_batch.pair_offsets = fixture.plan.topology.atom_offsets;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.aes2_batch.pair_offsets = fixture.plan.geometry_batch.pair_offsets;

  /* The bridge carries the complete master topology, not a partial facsimile. */
  fixture.plan.scalar_bridge_batch.topology.orbital_to_atom = fixture.plan.topology.shell_to_atom;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.scalar_bridge_batch.topology.orbital_to_atom = fixture.plan.topology.orbital_to_atom;

  fixture.plan.publication_plan.matrix_offsets = fixture.plan.topology.atom_offsets;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.publication_plan.matrix_offsets = fixture.plan.topology.matrix_offsets;

  /* Leaf-vs-projection identity: a topology-only consumer (density reducers)
   * must name the sealed projection arrays, not a different allocation. */
  fixture.plan.density_batch.orbital_offsets = fixture.plan.topology.atom_offsets;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.density_batch.orbital_offsets = fixture.plan.topology.batch_orbital_offsets;

  /* Mulliken must borrow shell ownership and AO/matrix projections. */
  fixture.plan.mulliken_batch.shell_to_atom = fixture.plan.topology.atom_offsets;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.mulliken_batch.shell_to_atom = fixture.plan.topology.shell_to_atom;

  /* Eigensolver must borrow the AO bucket projection. */
  fixture.plan.eigensolver_batch.bucket_systems = nullptr;
  CHECK(validate().error == Gfn2SccIterationBindingError::kInvalidZeroCopyView);
  fixture.plan.eigensolver_batch.bucket_systems = fixture.plan.topology.bucket_systems;
  CHECK(validate().error == Gfn2SccIterationBindingError::kSuccess);
  return 0;
}

}  // namespace

int main() {
  const std::array<int (*)(), 7> tests{
      {test_valid_binding_and_fail_closed_copy, test_capacity_alignment_alias_and_bucket_rejection,
       test_unrestricted_layout_and_spin_projection_rejection,
       test_optional_canonical_null_and_report_extension_capacity,
       test_convergence_policy_has_one_rms_authority, test_launch_result_preserves_provider_domains,
       test_projection_authority_rejection}};
  for (const auto test : tests) {
    if (const int line = test(); line != 0) {
      std::fprintf(stderr, "CUDA SCC iteration binding test failed at line %d\n", line);
      return 1;
    }
  }
  return 0;
}
