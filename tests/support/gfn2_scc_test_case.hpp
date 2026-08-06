#ifndef GPUXTB_TESTS_SUPPORT_GFN2_SCC_TEST_CASE_HPP
#define GPUXTB_TESTS_SUPPORT_GFN2_SCC_TEST_CASE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpuxtb/gpuxtb.h"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/external_point_charges.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/scc_driver.hpp"

namespace gpuxtb::test::gfn2 {

/* Small system geometries supported by the reusable SCC test fixture. */
enum class SmallSystemKind {
  kH2,
  kHe,
  kLiH,
  kCH2,
  /* H2 at an 8 bohr bond. The sigma/sigma* frontier gap is then below the
   * default 300 K electronic temperature, producing genuinely fractional and
   * near-degenerate occupations through the composed SCC iteration. */
  kH2Stretched,
  /* A 64-atom carbon 8x8 slab; the largest single system in the fixture.  It
   * crosses the sparse pair-list dense-fallback crossover so the production
   * runtime path exercises the bucketed consistency gate, not only the
   * all-pairs reference. */
  kCarbonSlab,
};

/*
 * Construction policy for a host-resident GFN2 SCC case. systems is a ragged
 * batch in the supplied order and must not be empty. All geometries use bohr.
 * geometry_generation is the nonzero epoch sealed into every geometry cache
 * and the overlap factorization.
 */
struct HostSccCaseOptions {
  std::vector<SmallSystemKind> systems{SmallSystemKind::kH2};
  /* Optional per-system electronic structure. Empty vectors select the
   * historical neutral, paired, restricted defaults. A nonempty vector must
   * have exactly systems.size() entries so the CPU oracle and uploaded CUDA
   * plans are constructed from one canonical electronic specification. */
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  bool enable_d4 = false;
  bool enable_periodic_embedding = false;
  bool enable_explicit_point_charges = false;
  std::uint64_t maximum_iterations = 5u;
  std::int64_t mixer_history = 3;
  double mixer_damping = 0.4;
  double residual_tolerance = 1.0e-10;
  double energy_tolerance = 1.0e-8;
  double electronic_temperature = 0.0;
  std::uint64_t geometry_generation = 1u;
};

/*
 * Byte-exact numerical checkpoint for replay and CPU/CUDA parity tests. The
 * descriptor objects are intentionally excluded: they remain bound to the
 * HostSccCase-owned allocations. driver_workspace is included so a replay
 * starts from identical unpublished scratch as well as identical persistent
 * wavefunction, mixer, and driver state.
 */
struct HostSccCheckpoint {
  std::vector<std::byte> wavefunction;
  std::vector<std::byte> mixer_state;
  std::vector<std::byte> driver_state;
  std::vector<std::byte> driver_workspace;
};

/*
 * Move-only owner of a complete host GFN2 SCC setup. Plans, caches, views, and
 * all numerical backing allocations live until the case is destroyed or move
 * assigned. Descriptor getters expose the real production bindings so CUDA
 * composer tests can upload or compare individual stages without rebuilding
 * an independent oracle.
 */
class HostSccCase {
 public:
  HostSccCase() noexcept;
  ~HostSccCase();
  HostSccCase(HostSccCase&&) noexcept;
  HostSccCase& operator=(HostSccCase&&) noexcept;
  HostSccCase(const HostSccCase&) = delete;
  HostSccCase& operator=(const HostSccCase&) = delete;

  /*
   * Transactionally construct a case. output is unchanged on validation,
   * allocation, plan-construction, cache-update, or binding failure.
   */
  static gpuxtb_status_t create(const HostSccCaseOptions& options, HostSccCase& output,
                                std::string& error);

  [[nodiscard]] bool valid() const noexcept;

  /* Advance the ragged batch by exactly one CPU SCC driver call. */
  gpuxtb_status_t run_one_iteration(std::string& error);

  [[nodiscard]] HostSccCheckpoint checkpoint() const;

  /* Validate every checkpoint extent before modifying any live allocation. */
  gpuxtb_status_t restore(const HostSccCheckpoint& checkpoint, std::string& error);

  /* All accessors below require valid() == true. */
  [[nodiscard]] const HostSccCaseOptions& options() const noexcept;
  [[nodiscard]] std::int64_t batch_size() const noexcept;
  [[nodiscard]] std::int64_t total_atoms() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& atom_offsets() const noexcept;
  [[nodiscard]] const std::vector<std::int32_t>& atomic_numbers() const noexcept;
  [[nodiscard]] const std::vector<double>& positions() const noexcept;
  [[nodiscard]] const std::vector<double>& molecular_charges() const noexcept;
  [[nodiscard]] const std::vector<std::int32_t>& unpaired_electrons() const noexcept;
  [[nodiscard]] const std::vector<std::int32_t>& spin_channels() const noexcept;
  [[nodiscard]] const std::vector<double>& coordination_numbers() const noexcept;

  [[nodiscard]] const std::vector<std::int64_t>& point_charge_offsets() const noexcept;
  [[nodiscard]] const std::vector<double>& point_charge_positions() const noexcept;
  [[nodiscard]] const std::vector<double>& point_charge_charges() const noexcept;
  [[nodiscard]] const std::vector<double>& point_charge_hardnesses() const noexcept;
  [[nodiscard]] const std::vector<double>& explicit_point_charge_shell_potential() const noexcept;

  /* Mutable numerical vectors may be edited in place but must not be resized,
   * because geometry descriptors retain their data pointers. */
  [[nodiscard]] std::vector<double>& explicit_point_charge_shell_potential() noexcept;

  [[nodiscard]] const std::vector<double>& periodic_shifts() const noexcept;
  [[nodiscard]] std::vector<double>& periodic_shifts() noexcept;
  [[nodiscard]] const std::vector<double>& periodic_response_matrices() const noexcept;
  [[nodiscard]] std::vector<double>& periodic_response_matrices() noexcept;

  [[nodiscard]] const std::vector<double>& overlap() const noexcept;
  [[nodiscard]] std::vector<double>& overlap() noexcept;
  [[nodiscard]] const std::vector<double>& dipole_integrals() const noexcept;
  [[nodiscard]] std::vector<double>& dipole_integrals() noexcept;
  [[nodiscard]] const std::vector<double>& quadrupole_integrals() const noexcept;
  [[nodiscard]] std::vector<double>& quadrupole_integrals() noexcept;
  [[nodiscard]] const std::vector<double>& h0() const noexcept;
  [[nodiscard]] std::vector<double>& h0() noexcept;

  [[nodiscard]] const detail::gfn2::BasisPlan& basis_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::IntegralPlan& integral_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::H0Plan& h0_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::WavefunctionLayout& wavefunction_layout() const noexcept;
  [[nodiscard]] const detail::gfn2::ES2Plan& es2_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::ES3Plan& es3_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::AES2Plan& aes2_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::MullikenPlan& mulliken_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::EigensolverPlan& eigensolver_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::SccMixerPlan& mixer_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::D4Plan* d4_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::PeriodicEmbeddingPlan* periodic_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::ExternalPointChargePlan* point_charge_plan() const noexcept;
  [[nodiscard]] const detail::gfn2::SccDriverPlan& driver_plan() const noexcept;

  [[nodiscard]] const detail::gfn2::ES2GeometryCache& es2_cache() const noexcept;
  [[nodiscard]] detail::gfn2::ES2GeometryCache& es2_cache() noexcept;
  [[nodiscard]] const detail::gfn2::AES2GeometryCache& aes2_cache() const noexcept;
  [[nodiscard]] detail::gfn2::AES2GeometryCache& aes2_cache() noexcept;
  [[nodiscard]] const detail::gfn2::D4GeometryCache* d4_cache() const noexcept;
  [[nodiscard]] detail::gfn2::D4GeometryCache* d4_cache() noexcept;
  [[nodiscard]] const detail::gfn2::EigensolverOverlapCache& overlap_cache() const noexcept;

  [[nodiscard]] const detail::gfn2::WavefunctionView& wavefunction() const noexcept;
  [[nodiscard]] detail::gfn2::WavefunctionView& wavefunction() noexcept;
  [[nodiscard]] const detail::gfn2::SccMixerState& mixer_state() const noexcept;
  [[nodiscard]] detail::gfn2::SccMixerState& mixer_state() noexcept;
  [[nodiscard]] const detail::gfn2::SccDriverState& driver_state() const noexcept;
  [[nodiscard]] detail::gfn2::SccDriverState& driver_state() noexcept;
  [[nodiscard]] const detail::gfn2::SccDriverWorkspace& driver_workspace() const noexcept;
  [[nodiscard]] detail::gfn2::SccDriverWorkspace& driver_workspace() noexcept;
  [[nodiscard]] const detail::gfn2::SccDriverGeometryView& geometry() const noexcept;
  [[nodiscard]] detail::gfn2::SccDriverGeometryView& geometry() noexcept;
  [[nodiscard]] const detail::gfn2::CpuLinearAlgebraBackend& cpu_backend() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpuxtb::test::gfn2

#endif  // GPUXTB_TESTS_SUPPORT_GFN2_SCC_TEST_CASE_HPP
