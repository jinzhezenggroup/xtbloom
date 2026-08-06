#ifndef GPUXTB_BACKENDS_CUDA_GFN2_SCC_SETUP_INPUTS_CUH
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_BACKENDS_CUDA_GFN2_SCC_SETUP_INPUTS_CUH

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "backends/cuda/gfn2_scc_iteration.cuh"
#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/es3.hpp"
#include "model/gfn2/external_point_charges.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/periodic_embedding.hpp"
#include "model/gfn2/scc_driver.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"

namespace gpuxtb::detail::cuda {

/* Exact host array view used at the host/CUDA setup boundary. The owner copies
 * every nonempty view into its pinned packed image during create(), so caller
 * arrays need not remain alive after create returns. */
template <typename T>
struct Gfn2SccSetupHostArray {
  const T* data = nullptr;
  std::int64_t elements = 0;
};

struct Gfn2SccSetupGeometryCacheSource {
  Gfn2SccSetupHostArray<double> pair_data{};
  Gfn2SccSetupHostArray<double> coordination_numbers{};
  Gfn2SccSetupHostArray<std::uint64_t> system_generations{};
};

struct Gfn2SccSetupES2CacheSource {
  Gfn2SccSetupHostArray<double> coulomb_matrix{};
};

struct Gfn2SccSetupAES2CacheSource {
  Gfn2SccSetupHostArray<double> pair_data{};
};

struct Gfn2SccSetupD4Source {
  const gfn2::D4Plan* plan = nullptr;
  Gfn2SccSetupHostArray<Gfn2D4DeviceElementData> elements{};
  Gfn2SccSetupHostArray<Gfn2D4DeviceReferenceData> references{};
  Gfn2SccSetupHostArray<double> reference_c6{};
  Gfn2SccSetupHostArray<double> pair_data{};
  Gfn2SccSetupHostArray<double> coordination_numbers{};
};

struct Gfn2SccSetupPointChargeSource {
  const gfn2::ExternalPointChargePlan* plan = nullptr;
  Gfn2SccSetupHostArray<double> positions{};
  Gfn2SccSetupHostArray<double> charges{};
  Gfn2SccSetupHostArray<double> hardnesses{};
  Gfn2SccSetupHostArray<double> shell_potential_cache{};
};

struct Gfn2SccSetupPeriodicSource {
  const gfn2::PeriodicEmbeddingPlan* plan = nullptr;
  Gfn2SccSetupHostArray<double> shifts{};
  Gfn2SccSetupHostArray<double> response_matrices{};
};

/*
 * Production host sources for the immutable restricted GFN2 SCC plan. Required
 * component plans are accepted by identity and cross-validated against the
 * already-bound common topology. Numerical views contain evaluated H0,
 * overlap/multipole operators, geometry, and cache seeds for one generation.
 *
 * D4, explicit point charges, periodic embedding, and warm-start generations
 * are enabled solely by a non-null optional plan/source. Disabled components
 * are emitted in canonical all-zero form and consume no device-arena bytes.
 */
struct Gfn2SccSetupInputSources {
  const gfn2::BasisPlan* basis = nullptr;
  const gfn2::IntegralPlan* integrals = nullptr;
  const gfn2::H0Plan* h0_plan = nullptr;
  const gfn2::WavefunctionLayout* wavefunction = nullptr;
  const gfn2::ES2Plan* es2 = nullptr;
  const gfn2::ES3Plan* es3 = nullptr;
  const gfn2::AES2Plan* aes2 = nullptr;
  const gfn2::MullikenPlan* mulliken = nullptr;
  const gfn2::SccMixerPlan* mixer = nullptr;
  const gfn2::SccDriverPlan* driver = nullptr;

  std::uint64_t geometry_generation = 0u;
  std::uint64_t warm_start_generation = 0u;
  Gfn2SccSetupHostArray<std::uint64_t> warm_start_generations{};

  Gfn2SccSetupHostArray<std::int32_t> atomic_numbers{};
  Gfn2SccSetupHostArray<double> positions{};
  Gfn2SccSetupHostArray<double> covalent_radii{};
  Gfn2SccSetupHostArray<double> h0{};
  Gfn2SccSetupHostArray<double> overlap{};
  Gfn2SccSetupHostArray<double> dipole_integrals{};
  Gfn2SccSetupHostArray<double> quadrupole_integrals{};

  Gfn2SccSetupGeometryCacheSource geometry_cache{};
  Gfn2SccSetupES2CacheSource es2_cache{};
  Gfn2SccSetupAES2CacheSource aes2_cache{};
  Gfn2SccSetupD4Source d4{};
  Gfn2SccSetupPointChargeSource point_charges{};
  Gfn2SccSetupPeriodicSource periodic{};

  Gfn2EigensolverOptions eigensolver_options{};
};

enum class Gfn2SccSetupInputsError : std::uint32_t {
  kSuccess = 0u,
  kInvalidSource = 1u,
  kCrossPlan = 2u,
  kCountOverflow = 3u,
  kAllocationFailed = 4u,
  kNullArena = 5u,
  kMisalignedArena = 6u,
  kInsufficientArena = 7u,
  kInvalidArenaMemory = 8u,
  kCudaError = 9u,
};

enum class Gfn2SccSetupInputsField : std::uint32_t {
  kNone = 0u,
  kPlanToken = 1u,
  kTopology = 2u,
  kRequiredPlans = 3u,
  kGeometry = 4u,
  kHamiltonian = 5u,
  kES2 = 6u,
  kAES2 = 7u,
  kD4 = 8u,
  kPointCharges = 9u,
  kPeriodic = 10u,
  kEigensolver = 11u,
  kArena = 12u,
};

struct Gfn2SccSetupInputsDiagnostic {
  gpuxtb_status_t status = GPUXTB_STATUS_SUCCESS;
  Gfn2SccSetupInputsError error = Gfn2SccSetupInputsError::kSuccess;
  Gfn2SccSetupInputsField field = Gfn2SccSetupInputsField::kNone;
  std::int64_t index = -1;
  std::size_t required_bytes = 0u;
  cudaError_t cuda_status = cudaSuccess;

  [[nodiscard]] bool success() const noexcept {
    return status == GPUXTB_STATUS_SUCCESS && error == Gfn2SccSetupInputsError::kSuccess;
  }
};

struct Gfn2SccSetupInputsRequirements {
  std::size_t device_bytes = 0u;
  std::size_t device_alignment = 256u;
};

/*
 * Move-only setup owner for production SCC plan/input leaves. create() copies
 * all host numerical data and plan arrays, including the GFN2 atom-local spin
 * couplings, into one pinned packed image. bind() projects the supplied device
 * topology and wavefunction descriptors into a complete non-provider plan
 * seed, enqueues one asynchronous image upload on the caller stream, and
 * publishes plan/input only after CUDA accepted the transfer. The eigensolver
 * provider, overlap cache, and active ledger pointers remain canonical zero;
 * the dedicated setup-eigensolver owner is their single production authority.
 *
 * The owner and caller-owned device arena must outlive queued work. Replacing
 * or destroying an owner while an upload is in flight is forbidden. A consumer
 * on another stream must wait on an event recorded after the upload.
 */
class Gfn2SccSetupInputs {
 public:
  Gfn2SccSetupInputs() noexcept;
  ~Gfn2SccSetupInputs();
  Gfn2SccSetupInputs(Gfn2SccSetupInputs&&) noexcept;
  Gfn2SccSetupInputs& operator=(Gfn2SccSetupInputs&&) noexcept;
  Gfn2SccSetupInputs(const Gfn2SccSetupInputs&) = delete;
  Gfn2SccSetupInputs& operator=(const Gfn2SccSetupInputs&) = delete;

  [[nodiscard]] static Gfn2SccSetupInputsDiagnostic create(
      const Gfn2SccSetupInputSources& sources, const Gfn2RaggedTopologyView& host_topology,
      std::uint64_t plan_token, Gfn2SccSetupInputs& output) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] Gfn2SccSetupInputsRequirements requirements() const noexcept;
  /* Pinned upload storage retained by this immutable input owner. */
  [[nodiscard]] std::size_t retained_host_bytes() const noexcept;

  [[nodiscard]] Gfn2SccSetupInputsDiagnostic bind_device_arena_and_upload_async(
      const Gfn2RaggedTopologyView& device_topology,
      const Gfn2WavefunctionLayoutView& device_wavefunction, void* device_arena,
      std::size_t device_arena_bytes, Gfn2SccIterationDevicePlan& plan_seed,
      Gfn2SccIterationDeviceInput& input_seed, cudaStream_t stream = nullptr) const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpuxtb::detail::cuda

#endif  // GPUXTB_BACKENDS_CUDA_GFN2_SCC_SETUP_INPUTS_CUH
