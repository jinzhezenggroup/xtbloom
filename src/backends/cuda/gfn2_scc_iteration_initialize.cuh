#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_INITIALIZE_CUH
#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_INITIALIZE_CUH

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "backends/cuda/gfn2_scc_iteration_arena.cuh"

namespace gpuxtb::detail::cuda {

inline constexpr std::uint32_t kGfn2SccIterationInitializationAbiVersion = 1u;

enum class Gfn2SccIterationInitializationMode : std::uint32_t {
  kFresh = 1u,
  kWarm = 2u,
};

enum class Gfn2SccIterationInitializationError : std::uint32_t {
  kSuccess = 0u,
  kInvalidPlan = 1u,
  kStaleArenaRequirements = 2u,
  kInvalidArena = 3u,
  kInvalidExtent = 4u,
  kNullPointer = 5u,
  kNonfiniteValue = 6u,
  kCrossPlan = 7u,
  kInvalidFreshState = 8u,
  kInvalidWarmState = 9u,
  kCountOverflow = 10u,
  kAllocationFailed = 11u,
  kInvalidArenaMemory = 12u,
  kCudaError = 13u,
};

enum class Gfn2SccIterationInitializationField : std::uint32_t {
  kNone = 0u,
  kPlan = 1u,
  kArena = 2u,
  kTopology = 3u,
  kWavefunction = 4u,
  kPopulation = 5u,
  kEnergy = 6u,
  kMixer = 7u,
  kSccTrace = 8u,
  kWorkspace = 9u,
  kReportStorage = 10u,
  kReadyPublication = 11u,
};

struct Gfn2SccIterationInitializationDiagnostic {
  gpuxtb_status_t status = GPUXTB_STATUS_SUCCESS;
  Gfn2SccIterationInitializationError error = Gfn2SccIterationInitializationError::kSuccess;
  Gfn2SccIterationInitializationField field = Gfn2SccIterationInitializationField::kNone;
  std::int64_t index = -1;
  std::size_t required_bytes = 0u;
  std::size_t provided_bytes = 0u;
  cudaError_t cuda_status = cudaSuccess;

  [[nodiscard]] bool success() const noexcept {
    return status == GPUXTB_STATUS_SUCCESS &&
           error == Gfn2SccIterationInitializationError::kSuccess;
  }
};

/* Pointer/count host projection used only during setup-image construction. */
template <typename T>
struct Gfn2SccIterationHostArrayView {
  const T* data = nullptr;
  std::int64_t elements = 0;
};

struct Gfn2SccIterationHostTopologyView {
  Gfn2SccIterationHostArrayView<std::int64_t> atom_offsets{};
  Gfn2SccIterationHostArrayView<std::int64_t> shell_offsets{};
  std::uint64_t plan_token = 0u;
};

struct Gfn2SccIterationHostPopulationView {
  Gfn2SccIterationHostArrayView<double> shell_charges{};
  Gfn2SccIterationHostArrayView<double> atomic_charges{};
  Gfn2SccIterationHostArrayView<double> atomic_dipoles{};
  Gfn2SccIterationHostArrayView<double> atomic_quadrupoles{};
  std::uint64_t plan_token = 0u;
};

/*
 * Complete restricted public wavefunction checkpoint. Fresh initialization
 * requires only population and leaves every other field canonically empty;
 * warm initialization requires every field at its exact packed extent.
 */
struct Gfn2SccIterationHostWavefunctionView {
  Gfn2SccIterationHostArrayView<double> eigenvalues{};
  Gfn2SccIterationHostArrayView<double> coefficients{};
  Gfn2SccIterationHostArrayView<double> occupations{};
  Gfn2SccIterationHostArrayView<double> chemical_potentials{};
  Gfn2SccIterationHostArrayView<double> electron_sums{};
  Gfn2SccIterationHostArrayView<double> occupation_entropies{};
  Gfn2SccIterationHostArrayView<double> density{};
  Gfn2SccIterationHostArrayView<double> energy_weighted_density{};
  Gfn2SccIterationHostArrayView<double> band_energies{};
  Gfn2SccIterationHostArrayView<double> occupation_sums{};
  Gfn2SccIterationHostArrayView<double> density_traces{};
  Gfn2SccIterationHostArrayView<double> weighted_density_traces{};
  Gfn2SccIterationHostPopulationView population{};
  std::uint64_t plan_token = 0u;
};

/* Complete CPU-driver-compatible component and free-energy trace. */
struct Gfn2SccIterationHostEnergyView {
  Gfn2SccIterationHostArrayView<double> core{};
  Gfn2SccIterationHostArrayView<double> es2{};
  Gfn2SccIterationHostArrayView<double> es3{};
  Gfn2SccIterationHostArrayView<double> aes2{};
  Gfn2SccIterationHostArrayView<double> d4_two_body{};
  Gfn2SccIterationHostArrayView<double> explicit_point_charge{};
  Gfn2SccIterationHostArrayView<double> periodic_embedding{};
  Gfn2SccIterationHostArrayView<double> entropy{};
  Gfn2SccIterationHostArrayView<double> internal_energy{};
  Gfn2SccIterationHostArrayView<double> free_energy{};
  Gfn2SccIterationHostArrayView<double> classical_total{};
  std::uint64_t plan_token = 0u;
};

/* Exact persistent modified-Broyden checkpoint used only for warm starts. */
struct Gfn2SccIterationHostMixerView {
  Gfn2SccIterationHostArrayView<double> current_inputs{};
  Gfn2SccIterationHostArrayView<double> previous_inputs{};
  Gfn2SccIterationHostArrayView<double> previous_residuals{};
  Gfn2SccIterationHostArrayView<double> df_history{};
  Gfn2SccIterationHostArrayView<double> u_history{};
  Gfn2SccIterationHostArrayView<double> omega{};
  Gfn2SccIterationHostArrayView<double> residual_rms{};
  Gfn2SccIterationHostArrayView<double> residual_maximum{};
  Gfn2SccIterationHostArrayView<std::uint64_t> iterations{};
  Gfn2SccIterationHostArrayView<std::uint64_t> restart_counts{};
  Gfn2SccIterationHostArrayView<gpuxtb_status_t> system_statuses{};
  Gfn2SccIterationHostArrayView<std::uint8_t> initialized{};
  Gfn2SccIterationHostArrayView<std::uint8_t> residual_converged{};
  std::uint64_t plan_token = 0u;
};

/* Driver-visible mixed inputs and independently restartable SCC trace.
 * free_energy_changes is the signed quantity free - previous. At a terminal
 * maximum-iteration checkpoint the mixer transition may remain SUCCESS while
 * the driver-visible SCC status is SCC_NOT_CONVERGED, matching publication. */
struct Gfn2SccIterationHostTraceView {
  Gfn2SccIterationHostArrayView<double> current_shell_charges{};
  Gfn2SccIterationHostArrayView<double> current_atomic_dipoles{};
  Gfn2SccIterationHostArrayView<double> current_atomic_quadrupoles{};
  Gfn2SccIterationHostArrayView<double> free_energies{};
  Gfn2SccIterationHostArrayView<double> previous_free_energies{};
  Gfn2SccIterationHostArrayView<double> free_energy_changes{};
  Gfn2SccIterationHostArrayView<double> residual_rms{};
  Gfn2SccIterationHostArrayView<std::uint64_t> iterations{};
  Gfn2SccIterationHostArrayView<gpuxtb_status_t> system_statuses{};
  Gfn2SccIterationHostArrayView<std::uint8_t> converged{};
  std::uint64_t plan_token = 0u;
};

struct Gfn2SccIterationHostInitialization {
  std::uint32_t abi_version = kGfn2SccIterationInitializationAbiVersion;
  Gfn2SccIterationInitializationMode mode = Gfn2SccIterationInitializationMode::kFresh;
  std::uint64_t plan_token = 0u;
  std::uint64_t initialization_generation = 0u;
  Gfn2SccIterationHostTopologyView topology{};
  Gfn2SccIterationHostWavefunctionView wavefunction{};
  Gfn2SccIterationHostEnergyView energy{};
  Gfn2SccIterationHostMixerView mixer{};
  Gfn2SccIterationHostTraceView scc{};
};

/*
 * Published only after the one packed H2D initialization copy is accepted.
 * ready_on_stream means consumers on stream are ordered after initialization;
 * consumers on another stream must wait on a caller-recorded event.
 */
struct Gfn2SccIterationInitializationReady {
  std::uint32_t abi_version = kGfn2SccIterationInitializationAbiVersion;
  Gfn2SccIterationInitializationMode mode = Gfn2SccIterationInitializationMode::kFresh;
  std::uint64_t plan_token = 0u;
  std::uint64_t initialization_generation = 0u;
  void* device_arena = nullptr;
  std::size_t arena_bytes = 0u;
  std::uint8_t ready_on_stream = 0u;
  std::uint8_t reserved[7]{};
};

static_assert(std::is_trivially_copyable_v<Gfn2SccIterationInitializationDiagnostic>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationInitializationDiagnostic>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationHostArrayView<double>>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationHostArrayView<double>>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationHostInitialization>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationHostInitialization>);
static_assert(std::is_trivially_copyable_v<Gfn2SccIterationInitializationReady>);
static_assert(std::is_standard_layout_v<Gfn2SccIterationInitializationReady>);

/*
 * Move-only owner of one immutable pinned initialization image. create()
 * validates every host extent/value and every arena projection before
 * allocating or publishing output. The owner, arena, and source image must
 * outlive queued uploads; moving or destroying an owner with a copy in flight
 * is unsupported.
 */
class Gfn2SccIterationInitializer {
 public:
  Gfn2SccIterationInitializer() noexcept;
  ~Gfn2SccIterationInitializer();
  Gfn2SccIterationInitializer(Gfn2SccIterationInitializer&&) noexcept;
  Gfn2SccIterationInitializer& operator=(Gfn2SccIterationInitializer&&) noexcept;
  Gfn2SccIterationInitializer(const Gfn2SccIterationInitializer&) = delete;
  Gfn2SccIterationInitializer& operator=(const Gfn2SccIterationInitializer&) = delete;

  [[nodiscard]] static Gfn2SccIterationInitializationDiagnostic create(
      const Gfn2SccIterationDevicePlan& plan,
      const Gfn2SccIterationArenaRequirements& arena_requirements, void* device_arena,
      std::size_t device_arena_bytes, const Gfn2SccIterationDeviceState& state,
      const Gfn2SccIterationDeviceWorkspace& workspace,
      const Gfn2SccIterationReportStorage& report_storage,
      const Gfn2SccIterationHostInitialization& host, Gfn2SccIterationInitializer& output) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t image_bytes() const noexcept;
  [[nodiscard]] std::uint64_t plan_token() const noexcept;
  [[nodiscard]] std::uint64_t initialization_generation() const noexcept;
  [[nodiscard]] Gfn2SccIterationInitializationMode mode() const noexcept;

  /*
   * Allocation-free setup submission. ready is cleared before validation and
   * published only after the single packed transfer is accepted by CUDA.
   */
  [[nodiscard]] Gfn2SccIterationInitializationDiagnostic upload_async(
      void* device_arena, std::size_t device_arena_bytes,
      Gfn2SccIterationInitializationReady& ready, cudaStream_t stream = nullptr) const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_ITERATION_INITIALIZE_CUH
