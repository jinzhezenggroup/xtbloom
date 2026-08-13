#include "model/gfn1/wavefunction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "model/gfn2/eigensolver.hpp"

#if defined(XTBLOOM_TEST_SCIPY_PREFIXED_BLAS)
#define LAPACKE_dpotrf_work scipy_LAPACKE_dpotrf_work
#define LAPACKE_dpocon_work scipy_LAPACKE_dpocon_work
#define LAPACKE_dsyevd_work scipy_LAPACKE_dsyevd_work
#define cblas_dtrsm scipy_cblas_dtrsm
#define cblas_dgemm scipy_cblas_dgemm
#endif

extern "C" {
std::int32_t LAPACKE_dpotrf_work(std::int32_t, char, std::int32_t, double*, std::int32_t);
std::int32_t LAPACKE_dpocon_work(std::int32_t, char, std::int32_t, const double*, std::int32_t,
                                 double, double*, double*, std::int32_t*);
std::int32_t LAPACKE_dsyevd_work(std::int32_t, char, char, std::int32_t, double*, std::int32_t,
                                 double*, double*, std::int32_t, std::int32_t*, std::int32_t);
void cblas_dtrsm(int, int, int, int, int, std::int32_t, std::int32_t, double, const double*,
                 std::int32_t, double*, std::int32_t);
void cblas_dgemm(int, int, int, std::int32_t, std::int32_t, std::int32_t, double, const double*,
                 std::int32_t, const double*, std::int32_t, double, double*, std::int32_t);
}

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using xtbloom::detail::gfn1::BasisPlan;
using xtbloom::detail::gfn1::ConstWavefunctionSystemView;
using xtbloom::detail::gfn1::ConstWavefunctionView;
using xtbloom::detail::gfn1::WavefunctionFieldLayout;
using xtbloom::detail::gfn1::WavefunctionLayout;
using xtbloom::detail::gfn1::WavefunctionSystemView;
using xtbloom::detail::gfn1::WavefunctionView;
using xtbloom::detail::gfn1::WavefunctionWarmStartIdentity;

struct AlignedWorkspace {
  std::vector<std::byte> storage;
  void* data = nullptr;

  explicit AlignedWorkspace(std::size_t size)
      : storage(size + xtbloom::detail::gfn1::kWavefunctionWorkspaceAlignment - 1u) {
    void* candidate = storage.data();
    std::size_t space = storage.size();
    data = std::align(xtbloom::detail::gfn1::kWavefunctionWorkspaceAlignment, size, candidate,
                      space);
  }
};

std::array<const WavefunctionFieldLayout*, 7> fields(const WavefunctionLayout& layout) {
  return {{&layout.coefficients, &layout.eigenvalues, &layout.occupations, &layout.density,
           &layout.qsh, &layout.qat, &layout.energy_weighted_density}};
}

bool make_layout(const std::vector<std::int64_t>& atom_offsets,
                 const std::vector<std::int32_t>& atomic_numbers,
                 const std::vector<double>& charges,
                 const std::vector<std::int32_t>& unpaired,
                 const std::vector<std::int32_t>& spins, BasisPlan& basis,
                 WavefunctionLayout& layout, std::string& error) {
  return xtbloom::detail::gfn1::make_basis_plan(
             static_cast<std::int64_t>(atom_offsets.size() - 1u),
             static_cast<std::int64_t>(atomic_numbers.size()), atom_offsets.data(),
             atomic_numbers.data(), basis, error) == XTBLOOM_STATUS_SUCCESS &&
         xtbloom::detail::gfn1::make_wavefunction_layout(
             basis, atomic_numbers.data(), charges.data(), unpaired.data(), spins.data(), layout,
             error) == XTBLOOM_STATUS_SUCCESS;
}

xtbloom::detail::gfn2::EigensolverWavefunctionLayout project(
    const WavefunctionLayout& layout) {
  xtbloom::detail::gfn2::EigensolverWavefunctionLayout result;
  result.batch_size = layout.batch_size;
  result.workspace_size_bytes = layout.workspace_size_bytes;
  result.orbital_offsets = layout.batch_orbital_offsets.data();
  result.orbital_offset_count = layout.batch_orbital_offsets.size();
  result.spin_channels = layout.spin_channels.data();
  result.spin_channel_count = layout.spin_channels.size();
  result.alpha_electron_counts = layout.alpha_electron_counts.data();
  result.beta_electron_counts = layout.beta_electron_counts.data();
  result.electron_count_count = layout.alpha_electron_counts.size();
  const std::array<const WavefunctionFieldLayout*, 5> electronic{
      {&layout.coefficients, &layout.eigenvalues, &layout.occupations, &layout.density,
       &layout.energy_weighted_density}};
  for (std::size_t field = 0; field < electronic.size(); ++field) {
    result.fields[field] = {electronic[field]->offset_bytes, electronic[field]->element_count,
                            electronic[field]->system_offsets.data(),
                            electronic[field]->system_offsets.size()};
  }
  return result;
}

xtbloom::detail::gfn2::EigensolverWavefunctionView project(const WavefunctionView& view) {
  return {view.workspace_base,         view.workspace_size_bytes, view.coefficients,
          view.eigenvalues,            view.occupations,           view.density,
          view.energy_weighted_density};
}

int test_hydrogen_repeated_shell_and_sad() {
  BasisPlan basis;
  WavefunctionLayout layout;
  std::string error;
  CHECK(make_layout({0, 1}, {1}, {0.0}, {1}, {2}, basis, layout, error));
  CHECK(layout.reference_atom_occupations == std::vector<double>({1.0}));
  CHECK(layout.reference_shell_occupations == std::vector<double>({1.0, 0.0}));
  CHECK(layout.electron_counts == std::vector<double>({1.0}));
  CHECK(layout.alpha_electron_counts == std::vector<double>({1.0}));
  CHECK(layout.beta_electron_counts == std::vector<double>({0.0}));
  CHECK(fields(layout).size() == 7u);
  CHECK(layout.energy_weighted_density.offset_bytes >=
        layout.qat.offset_bytes + layout.qat.size_bytes);
  return 0;
}

int test_ragged_layout_views_and_sad() {
  const std::vector<std::int64_t> atom_offsets{0, 1, 3};
  const std::vector<std::int32_t> atomic_numbers{1, 8, 1};
  const std::vector<double> charges{1.0, 0.0};
  const std::vector<std::int32_t> unpaired{0, 1};
  const std::vector<std::int32_t> spins{1, 2};
  BasisPlan basis;
  WavefunctionLayout layout;
  std::string error;
  CHECK(make_layout(atom_offsets, atomic_numbers, charges, unpaired, spins, basis, layout, error));
  CHECK(layout.electron_counts == std::vector<double>({0.0, 7.0}));
  CHECK(layout.alpha_electron_counts == std::vector<double>({0.0, 4.0}));
  CHECK(layout.beta_electron_counts == std::vector<double>({0.0, 3.0}));

  std::size_t previous_end = 0u;
  for (const auto* field : fields(layout)) {
    CHECK(field->offset_bytes % xtbloom::detail::gfn1::kWavefunctionWorkspaceAlignment == 0u);
    CHECK(field->offset_bytes >= previous_end);
    CHECK(field->system_offsets.size() == 3u);
    previous_end = field->offset_bytes + field->size_bytes;
  }
  CHECK(layout.energy_weighted_density.offset_bytes + layout.energy_weighted_density.size_bytes <=
        layout.workspace_size_bytes);

  AlignedWorkspace workspace(layout.workspace_size_bytes);
  CHECK(workspace.data != nullptr);
  WavefunctionView view;
  CHECK(xtbloom::detail::gfn1::bind_wavefunction_view(
            layout, workspace.data, layout.workspace_size_bytes, view, error) ==
        XTBLOOM_STATUS_SUCCESS);
  std::fill_n(view.qsh, static_cast<std::size_t>(layout.qsh.element_count), 99.0);
  std::fill_n(view.qat, static_cast<std::size_t>(layout.qat.element_count), 99.0);
  CHECK(xtbloom::detail::gfn1::initialize_sad_multipole_state(layout, view, error) ==
        XTBLOOM_STATUS_SUCCESS);

  CHECK(view.qsh[0] == 1.0);
  CHECK(view.qsh[1] == 0.0);  // H 2s has zero reference occupation.
  CHECK(view.qat[0] == 1.0);
  const std::size_t second_qat = static_cast<std::size_t>(layout.qat.system_offsets[1]);
  const std::size_t second_qsh = static_cast<std::size_t>(layout.qsh.system_offsets[1]);
  const std::size_t second_atoms =
      static_cast<std::size_t>(atom_offsets[2] - atom_offsets[1]);
  const std::size_t second_shells = static_cast<std::size_t>(
      layout.batch_shell_offsets[2] - layout.batch_shell_offsets[1]);
  double atom_sum = 0.0;
  double shell_sum = 0.0;
  for (std::size_t atom = 0; atom < second_atoms; ++atom) {
    atom_sum += view.qat[second_qat + atom];
    CHECK(view.qat[second_qat + second_atoms + atom] == 0.0);
  }
  for (std::size_t shell = 0; shell < second_shells; ++shell) {
    shell_sum += view.qsh[second_qsh + shell];
    CHECK(view.qsh[second_qsh + second_shells + shell] == 0.0);
  }
  CHECK(std::abs(atom_sum) < 1.0e-15);
  CHECK(std::abs(shell_sum) < 1.0e-15);

  WavefunctionSystemView system;
  CHECK(xtbloom::detail::gfn1::make_wavefunction_system_view(layout, view, 1, system, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(system.atom_count == 2 && system.spin_channels == 2);
  CHECK(system.qsh == view.qsh + layout.qsh.system_offsets[1]);
  ConstWavefunctionView const_view;
  CHECK(xtbloom::detail::gfn1::bind_wavefunction_view(
            layout, static_cast<const void*>(workspace.data), layout.workspace_size_bytes,
            const_view, error) == XTBLOOM_STATUS_SUCCESS);
  ConstWavefunctionSystemView const_system;
  CHECK(xtbloom::detail::gfn1::make_wavefunction_system_view(layout, const_view, 1, const_system,
                                                             error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(const_system.energy_weighted_density ==
        const_view.energy_weighted_density + layout.energy_weighted_density.system_offsets[1]);
  return 0;
}

int test_strong_failure_and_warm_identity() {
  BasisPlan basis;
  WavefunctionLayout layout;
  std::string error;
  CHECK(make_layout({0, 2}, {8, 1}, {0.0}, {1}, {2}, basis, layout, error));

  WavefunctionLayout sentinel;
  sentinel.batch_size = 17;
  const std::array<double, 1> neutral{0.0};
  const std::array<std::int32_t, 1> singlet{0};
  const std::array<std::int32_t, 1> restricted{1};
  const std::array<std::int32_t, 2> invalid_atoms{87, 1};
  CHECK(xtbloom::detail::gfn1::make_wavefunction_layout(
            basis, invalid_atoms.data(), neutral.data(), singlet.data(), restricted.data(),
            sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(sentinel.batch_size == 17);
  const std::array<double, 1> nonfinite{std::numeric_limits<double>::infinity()};
  CHECK(xtbloom::detail::gfn1::make_wavefunction_layout(
            basis, layout.atomic_numbers.data(), nonfinite.data(), singlet.data(), restricted.data(),
            sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  const std::array<double, 1> parity_charge{2.0};
  CHECK(xtbloom::detail::gfn1::make_wavefunction_layout(
            basis, layout.atomic_numbers.data(), parity_charge.data(), singlet.data(),
            restricted.data(), sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  const std::array<std::int32_t, 1> invalid_spin{3};
  CHECK(xtbloom::detail::gfn1::make_wavefunction_layout(
            basis, layout.atomic_numbers.data(), neutral.data(), singlet.data(), invalid_spin.data(),
            sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  AlignedWorkspace workspace(layout.workspace_size_bytes);
  WavefunctionView view;
  view.coefficients = reinterpret_cast<double*>(1);
  CHECK(xtbloom::detail::gfn1::bind_wavefunction_view(
            layout, static_cast<std::byte*>(workspace.data) + 1, layout.workspace_size_bytes - 1u,
            view, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(view.coefficients == reinterpret_cast<double*>(1));

  WavefunctionView valid_view;
  CHECK(xtbloom::detail::gfn1::bind_wavefunction_view(
            layout, workspace.data, layout.workspace_size_bytes, valid_view, error) ==
        XTBLOOM_STATUS_SUCCESS);
  std::fill_n(valid_view.qsh, static_cast<std::size_t>(layout.qsh.element_count), 23.0);
  WavefunctionView forged = valid_view;
  ++forged.qsh;
  CHECK(xtbloom::detail::gfn1::initialize_sad_multipole_state(layout, forged, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::all_of(valid_view.qsh, valid_view.qsh + layout.qsh.element_count,
                    [](double value) { return value == 23.0; }));

  WavefunctionWarmStartIdentity identity;
  CHECK(xtbloom::detail::gfn1::make_wavefunction_warm_start_identity(layout, 9u, identity, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(identity.model == XTBLOOM_MODEL_GFN1_XTB);
  CHECK(xtbloom::detail::gfn1::wavefunction_warm_start_matches(identity, identity));
  for (int mutation = 0; mutation < 5; ++mutation) {
    WavefunctionWarmStartIdentity changed = identity;
    if (mutation == 0) changed.model = XTBLOOM_MODEL_GFN2_XTB;
    if (mutation == 1) ++changed.geometry_cache_generation;
    if (mutation == 2) changed.atomic_numbers[0] = 7;
    if (mutation == 3) changed.molecular_charges[0] = 0.25;
    if (mutation == 4) changed.spin_channels[0] = 1;
    CHECK(!xtbloom::detail::gfn1::wavefunction_warm_start_matches(identity, changed));
    CHECK(xtbloom::detail::gfn1::validate_wavefunction_warm_start(identity, changed, error) ==
          XTBLOOM_STATUS_INVALID_ARGUMENT);
  }
  return 0;
}

int test_model_neutral_eigensolver_projection() {
  BasisPlan basis;
  WavefunctionLayout layout;
  std::string error;
  CHECK(make_layout({0, 1}, {1}, {0.0}, {1}, {2}, basis, layout, error));
  auto projection = project(layout);
  xtbloom::detail::gfn2::EigensolverPlan plan;
  CHECK(xtbloom::detail::gfn2::make_eigensolver_plan(projection, plan, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(plan.batch_size() == 1);
  CHECK(plan.orbital_offsets() == layout.batch_orbital_offsets);
  CHECK(plan.spin_channels() == layout.spin_channels);

  AlignedWorkspace workspace(layout.workspace_size_bytes);
  WavefunctionView view;
  CHECK(xtbloom::detail::gfn1::bind_wavefunction_view(
            layout, workspace.data, layout.workspace_size_bytes, view, error) ==
        XTBLOOM_STATUS_SUCCESS);
  const auto eigensolver_view = project(view);
  CHECK(eigensolver_view.coefficients == view.coefficients);
  CHECK(eigensolver_view.energy_weighted_density == view.energy_weighted_density);

  xtbloom::detail::gfn2::CpuLinearAlgebraBackend backend;
  CHECK(xtbloom::detail::gfn2::make_internal_test_lp64_backend(
            &LAPACKE_dpotrf_work, &LAPACKE_dpocon_work, &LAPACKE_dsyevd_work, &cblas_dtrsm,
            &cblas_dgemm, nullptr, backend, error) == XTBLOOM_STATUS_SUCCESS);
  AlignedWorkspace cache_storage(plan.overlap_cache_size_bytes());
  AlignedWorkspace scratch_storage(plan.workspace_size_bytes());
  xtbloom::detail::gfn2::EigensolverOverlapCache cache;
  xtbloom::detail::gfn2::EigensolverWorkspace scratch;
  CHECK(xtbloom::detail::gfn2::bind_eigensolver_overlap_cache(
            plan, cache_storage.data, plan.overlap_cache_size_bytes(), cache, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::bind_eigensolver_workspace(
            plan, scratch_storage.data, plan.workspace_size_bytes(), scratch, error) ==
        XTBLOOM_STATUS_SUCCESS);
  const std::array<double, 4> overlap{1.0, 0.0, 0.0, 1.0};
  CHECK(xtbloom::detail::gfn2::factor_overlap_cpu(plan, overlap.data(), 7u, backend, scratch, cache,
                                                  error) == XTBLOOM_STATUS_SUCCESS);
  const std::array<double, 8> hamiltonian{-0.5, 0.0, 0.0, 0.5,
                                         -0.4, 0.0, 0.0, 0.6};
  std::array<xtbloom_status_t, 1> statuses{};
  std::array<double, 2> chemical_potentials{};
  std::array<double, 1> entropies{};
  std::array<double, 1> band_energies{};
  std::array<double, 1> free_energies{};
  xtbloom::detail::gfn2::EigensolverThermodynamicsView thermodynamics{
      statuses.data(),          statuses.size(),          chemical_potentials.data(),
      chemical_potentials.size(), entropies.data(),       entropies.size(),
      band_energies.data(),     band_energies.size(),     free_energies.data(),
      free_energies.size()};
  CHECK(xtbloom::detail::gfn2::solve_eigensystems_cpu(
            plan, cache, 7u, hamiltonian.data(), 0.0, backend, scratch, eigensolver_view,
            thermodynamics, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(statuses[0] == XTBLOOM_STATUS_SUCCESS);
  CHECK(view.occupations[0] == 1.0 && view.occupations[1] == 0.0);
  CHECK(view.occupations[2] == 0.0 && view.occupations[3] == 0.0);
  CHECK(std::abs(view.density[0] - 1.0) < 1.0e-14);
  CHECK(std::abs(view.energy_weighted_density[0] + 0.5) < 1.0e-14);

  auto forged = projection;
  forged.fields[4].system_offset_count = 1u;
  xtbloom::detail::gfn2::EigensolverPlan sentinel;
  CHECK(xtbloom::detail::gfn2::make_eigensolver_plan(forged, sentinel, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(!sentinel.sealed());
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_hydrogen_repeated_shell_and_sad(); status != 0) return status;
  if (const int status = test_ragged_layout_views_and_sad(); status != 0) return status;
  if (const int status = test_strong_failure_and_warm_identity(); status != 0) return status;
  return test_model_neutral_eigensolver_projection();
}
