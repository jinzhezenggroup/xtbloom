#include "model/gfn2/wavefunction.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "data/parameters/gfn2.hpp"
#include "model/gfn2/basis.hpp"

namespace allocation_test {
std::atomic<std::size_t> count{0};
std::atomic<bool> enabled{false};
}  // namespace allocation_test

void* operator new(std::size_t size) {
  if (allocation_test::enabled.load(std::memory_order_relaxed)) {
    allocation_test::count.fetch_add(1u, std::memory_order_relaxed);
  }
  if (void* pointer = std::malloc(size == 0u ? 1u : size); pointer != nullptr) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }

#if defined(__GNUC__) && !defined(__clang__)
#define XTBLOOM_TEST_NOINLINE __attribute__((noinline))
#else
#define XTBLOOM_TEST_NOINLINE
#endif

/* GCC 11 diagnoses the intentional malloc/free implementation as a mismatched
 * pair only after inlining this test-only allocation counter shim. */
XTBLOOM_TEST_NOINLINE void operator delete(void* pointer) noexcept { std::free(pointer); }

void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }

void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }

void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete[](pointer); }

#undef XTBLOOM_TEST_NOINLINE

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using xtbloom::detail::gfn2::BasisPlan;
using xtbloom::detail::gfn2::ConstWavefunctionSystemView;
using xtbloom::detail::gfn2::ConstWavefunctionView;
using xtbloom::detail::gfn2::WavefunctionFieldLayout;
using xtbloom::detail::gfn2::WavefunctionLayout;
using xtbloom::detail::gfn2::WavefunctionSystemView;
using xtbloom::detail::gfn2::WavefunctionView;
using xtbloom::detail::gfn2::WavefunctionWarmStartIdentity;

bool near(double actual, double expected, double tolerance = 1.0e-14) {
  return std::abs(actual - expected) <= tolerance;
}

bool make_basis(const std::vector<std::int64_t>& atom_offsets,
                const std::vector<std::int32_t>& atomic_numbers, BasisPlan& basis,
                std::string& error) {
  return xtbloom::detail::gfn2::make_basis_plan(static_cast<std::int64_t>(atom_offsets.size() - 1u),
                                                static_cast<std::int64_t>(atomic_numbers.size()),
                                                atom_offsets.data(), atomic_numbers.data(), basis,
                                                error) == XTBLOOM_STATUS_SUCCESS;
}

double reference_electrons(std::int32_t atomic_number) {
  const auto* element =
      xtbloom::parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_number));
  double result = 0.0;
  for (std::size_t local = 0; local < element->shell_count; ++local) {
    result +=
        xtbloom::parameters::gfn2::kShells[element->shell_offset + local].reference_occupation;
  }
  return result;
}

std::array<const WavefunctionFieldLayout*, 9> fields(const WavefunctionLayout& layout) {
  return {{&layout.coefficients, &layout.eigenvalues, &layout.occupations, &layout.density,
           &layout.qsh, &layout.qat, &layout.dipole, &layout.quadrupole,
           &layout.energy_weighted_density}};
}

std::array<WavefunctionFieldLayout*, 9> mutable_fields(WavefunctionLayout& layout) {
  return {{&layout.coefficients, &layout.eigenvalues, &layout.occupations, &layout.density,
           &layout.qsh, &layout.qat, &layout.dipole, &layout.quadrupole,
           &layout.energy_weighted_density}};
}

bool same_view(const WavefunctionView& lhs, const WavefunctionView& rhs) {
  return lhs.workspace_base == rhs.workspace_base &&
         lhs.workspace_size_bytes == rhs.workspace_size_bytes &&
         lhs.coefficients == rhs.coefficients && lhs.eigenvalues == rhs.eigenvalues &&
         lhs.occupations == rhs.occupations && lhs.density == rhs.density && lhs.qsh == rhs.qsh &&
         lhs.qat == rhs.qat && lhs.dipole == rhs.dipole && lhs.quadrupole == rhs.quadrupole &&
         lhs.energy_weighted_density == rhs.energy_weighted_density;
}

bool same_system_view(const WavefunctionSystemView& lhs, const WavefunctionSystemView& rhs) {
  return lhs.atom_count == rhs.atom_count && lhs.shell_count == rhs.shell_count &&
         lhs.orbital_count == rhs.orbital_count && lhs.spin_channels == rhs.spin_channels &&
         lhs.electron_count == rhs.electron_count &&
         lhs.alpha_electron_count == rhs.alpha_electron_count &&
         lhs.beta_electron_count == rhs.beta_electron_count &&
         lhs.coefficients == rhs.coefficients && lhs.eigenvalues == rhs.eigenvalues &&
         lhs.occupations == rhs.occupations && lhs.density == rhs.density && lhs.qsh == rhs.qsh &&
         lhs.qat == rhs.qat && lhs.dipole == rhs.dipole && lhs.quadrupole == rhs.quadrupole &&
         lhs.energy_weighted_density == rhs.energy_weighted_density;
}

bool same_identity(const WavefunctionWarmStartIdentity& lhs,
                   const WavefunctionWarmStartIdentity& rhs) {
  return lhs.model == rhs.model && lhs.geometry_cache_generation == rhs.geometry_cache_generation &&
         lhs.batch_size == rhs.batch_size && lhs.total_atoms == rhs.total_atoms &&
         lhs.total_shells == rhs.total_shells && lhs.total_orbitals == rhs.total_orbitals &&
         lhs.atom_offsets == rhs.atom_offsets &&
         lhs.batch_shell_offsets == rhs.batch_shell_offsets &&
         lhs.batch_orbital_offsets == rhs.batch_orbital_offsets &&
         lhs.atomic_numbers == rhs.atomic_numbers &&
         lhs.molecular_charges == rhs.molecular_charges &&
         lhs.unpaired_electrons == rhs.unpaired_electrons && lhs.spin_channels == rhs.spin_channels;
}

struct AlignedWorkspace {
  std::vector<std::byte> storage;
  void* data = nullptr;

  explicit AlignedWorkspace(std::size_t size)
      : storage(size + xtbloom::detail::gfn2::kWavefunctionWorkspaceAlignment - 1u) {
    void* candidate = storage.data();
    std::size_t space = storage.size();
    data =
        std::align(xtbloom::detail::gfn2::kWavefunctionWorkspaceAlignment, size, candidate, space);
  }
};

int test_all_elements_reference_occupations() {
  constexpr std::int64_t element_count =
      static_cast<std::int64_t>(xtbloom::parameters::gfn2::kElementCount);
  std::vector<std::int64_t> atom_offsets(static_cast<std::size_t>(element_count) + 1u);
  std::vector<std::int32_t> atomic_numbers(static_cast<std::size_t>(element_count));
  std::vector<double> charges(static_cast<std::size_t>(element_count), 0.0);
  std::vector<std::int32_t> unpaired(static_cast<std::size_t>(element_count));
  std::vector<std::int32_t> spin_channels(static_cast<std::size_t>(element_count), 1);
  for (std::int64_t index = 0; index < element_count; ++index) {
    atom_offsets[static_cast<std::size_t>(index)] = index;
    atomic_numbers[static_cast<std::size_t>(index)] = static_cast<std::int32_t>(index + 1);
    unpaired[static_cast<std::size_t>(index)] = static_cast<std::int32_t>(
        std::llround(reference_electrons(static_cast<std::int32_t>(index + 1))) & 1LL);
  }
  atom_offsets.back() = element_count;

  BasisPlan basis;
  std::string error;
  CHECK(make_basis(atom_offsets, atomic_numbers, basis, error));
  WavefunctionLayout layout;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(
            basis, atomic_numbers.data(), charges.data(), unpaired.data(), spin_channels.data(),
            layout, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());
  CHECK(layout.batch_size == element_count);
  CHECK(layout.reference_atom_occupations.size() == atomic_numbers.size());

  for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
    const auto* element =
        xtbloom::parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_numbers[atom]));
    CHECK(near(layout.reference_atom_occupations[atom], reference_electrons(atomic_numbers[atom]),
               0.0));
    const std::int64_t shell_begin = basis.atom_shell_offsets[atom];
    const std::int64_t shell_end = basis.atom_shell_offsets[atom + 1u];
    CHECK(shell_end - shell_begin == element->shell_count);
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
      const std::size_t local = static_cast<std::size_t>(shell - shell_begin);
      CHECK(layout.reference_shell_occupations[static_cast<std::size_t>(shell)] ==
            xtbloom::parameters::gfn2::kShells[element->shell_offset + local].reference_occupation);
    }
    CHECK(layout.electron_counts[atom] == layout.reference_atom_occupations[atom]);
    CHECK(layout.spin_channels[atom] == 1);
  }
  return 0;
}

int test_ions_radical_fractional_and_layout() {
  /* Cover restricted and unrestricted singlets and radicals in one ragged batch. */
  const std::vector<std::int64_t> atom_offsets{0, 1, 2, 4, 5, 7};
  const std::vector<std::int32_t> atomic_numbers{1, 1, 8, 1, 8, 8, 1};
  const std::vector<double> charges{1.0, -1.0, 0.0, 0.5, 0.0};
  const std::vector<std::int32_t> unpaired{0, 0, 1, 0, 1};
  const std::vector<std::int32_t> spin_channels{1, 2, 1, 2, 2};
  BasisPlan basis;
  std::string error;
  CHECK(make_basis(atom_offsets, atomic_numbers, basis, error));
  WavefunctionLayout layout;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(
            basis, atomic_numbers.data(), charges.data(), unpaired.data(), spin_channels.data(),
            layout, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(layout.electron_counts == std::vector<double>({0.0, 2.0, 7.0, 5.5, 7.0}));
  CHECK(layout.alpha_electron_counts == std::vector<double>({0.0, 1.0, 4.0, 2.75, 4.0}));
  CHECK(layout.beta_electron_counts == std::vector<double>({0.0, 1.0, 3.0, 2.75, 3.0}));
  CHECK(layout.spin_channels == spin_channels);
  CHECK(layout.workspace_size_bytes % xtbloom::detail::gfn2::kWavefunctionWorkspaceAlignment == 0u);

  std::size_t previous_end = 0u;
  for (const WavefunctionFieldLayout* field : fields(layout)) {
    CHECK(field->offset_bytes % xtbloom::detail::gfn2::kWavefunctionWorkspaceAlignment == 0u);
    CHECK(field->offset_bytes >= previous_end);
    CHECK(field->size_bytes == static_cast<std::size_t>(field->element_count) * sizeof(double));
    CHECK(field->system_offsets.size() == atom_offsets.size());
    CHECK(field->system_offsets.front() == 0);
    CHECK(field->system_offsets.back() == field->element_count);
    previous_end = field->offset_bytes + field->size_bytes;
  }
  CHECK(previous_end <= layout.workspace_size_bytes);

  for (std::size_t system = 0; system < charges.size(); ++system) {
    const std::int64_t nao =
        basis.batch_orbital_offsets[system + 1u] - basis.batch_orbital_offsets[system];
    const std::int64_t nat = atom_offsets[system + 1u] - atom_offsets[system];
    const std::int64_t nsh =
        basis.batch_shell_offsets[system + 1u] - basis.batch_shell_offsets[system];
    const std::int64_t nspin = layout.spin_channels[system];
    CHECK(layout.coefficients.system_offsets[system + 1u] -
              layout.coefficients.system_offsets[system] ==
          nao * nao * nspin);
    CHECK(layout.eigenvalues.system_offsets[system + 1u] -
              layout.eigenvalues.system_offsets[system] ==
          nao * nspin);
    CHECK(layout.occupations.system_offsets[system + 1u] -
              layout.occupations.system_offsets[system] ==
          nao * 2);
    CHECK(layout.qsh.system_offsets[system + 1u] - layout.qsh.system_offsets[system] ==
          nsh * nspin);
    CHECK(layout.qat.system_offsets[system + 1u] - layout.qat.system_offsets[system] ==
          nat * nspin);
    CHECK(layout.dipole.system_offsets[system + 1u] - layout.dipole.system_offsets[system] ==
          nat * nspin * 3);
    CHECK(layout.quadrupole.system_offsets[system + 1u] -
              layout.quadrupole.system_offsets[system] ==
          nat * nspin * xtbloom::detail::gfn2::kWavefunctionQuadrupoleComponents);
  }

  AlignedWorkspace workspace(layout.workspace_size_bytes);
  CHECK(workspace.data != nullptr);
  WavefunctionView view;
  CHECK(xtbloom::detail::gfn2::bind_wavefunction_view(layout, workspace.data,
                                                      layout.workspace_size_bytes, view,
                                                      error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(reinterpret_cast<std::byte*>(view.coefficients) ==
        static_cast<std::byte*>(workspace.data) + layout.coefficients.offset_bytes);
  CHECK(view.workspace_base == workspace.data);
  CHECK(view.workspace_size_bytes == layout.workspace_size_bytes);
  CHECK(reinterpret_cast<std::byte*>(view.energy_weighted_density) ==
        static_cast<std::byte*>(workspace.data) + layout.energy_weighted_density.offset_bytes);

  for (std::int64_t system = 0; system < layout.batch_size; ++system) {
    WavefunctionSystemView system_view;
    CHECK(xtbloom::detail::gfn2::make_wavefunction_system_view(layout, view, system, system_view,
                                                               error) == XTBLOOM_STATUS_SUCCESS);
    const std::size_t index = static_cast<std::size_t>(system);
    CHECK(system_view.coefficients ==
          view.coefficients + layout.coefficients.system_offsets[index]);
    CHECK(system_view.occupations == view.occupations + layout.occupations.system_offsets[index]);
    CHECK(system_view.energy_weighted_density ==
          view.energy_weighted_density + layout.energy_weighted_density.system_offsets[index]);
    system_view.coefficients[0] = 100.0 + static_cast<double>(system);
    CHECK(view.coefficients[layout.coefficients.system_offsets[index]] ==
          100.0 + static_cast<double>(system));
  }

  ConstWavefunctionView const_view;
  CHECK(xtbloom::detail::gfn2::bind_wavefunction_view(
            layout, static_cast<const void*>(workspace.data), layout.workspace_size_bytes,
            const_view, error) == XTBLOOM_STATUS_SUCCESS);
  ConstWavefunctionSystemView const_system;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_system_view(layout, const_view, 2, const_system,
                                                             error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(const_system.spin_channels == 1);
  CHECK(const_system.alpha_electron_count == 4.0);
  CHECK(const_system.beta_electron_count == 3.0);

  WavefunctionView sentinel;
  sentinel.coefficients = reinterpret_cast<double*>(1);
  CHECK(xtbloom::detail::gfn2::bind_wavefunction_view(
            layout, static_cast<std::byte*>(workspace.data) + 1, layout.workspace_size_bytes - 1u,
            sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(sentinel.coefficients == reinterpret_cast<double*>(1));
  CHECK(xtbloom::detail::gfn2::bind_wavefunction_view(layout, workspace.data,
                                                      layout.workspace_size_bytes - 1u, sentinel,
                                                      error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  WavefunctionSystemView invalid_system_view;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_system_view(layout, view, layout.batch_size,
                                                             invalid_system_view, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  return 0;
}

int test_invalid_electronic_states_and_overflow() {
  const std::vector<std::int64_t> oh_offsets{0, 2};
  const std::vector<std::int32_t> oh_atoms{8, 1};
  BasisPlan oh_basis;
  std::string error;
  CHECK(make_basis(oh_offsets, oh_atoms, oh_basis, error));
  WavefunctionLayout sentinel;
  sentinel.batch_size = 17;
  const std::array<double, 1> neutral{0.0};
  const std::array<std::int32_t, 1> singlet{0};
  const std::array<std::int32_t, 1> restricted{1};
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(oh_basis, oh_atoms.data(), neutral.data(),
                                                        singlet.data(), restricted.data(), sentinel,
                                                        error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(sentinel.batch_size == 17);

  const std::array<double, 1> fractional_charge{0.5};
  const std::array<std::int32_t, 1> wrong_fractional_parity{0};
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(
            oh_basis, oh_atoms.data(), fractional_charge.data(), wrong_fractional_parity.data(),
            restricted.data(), sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  const std::vector<std::int64_t> h_offsets{0, 1};
  const std::vector<std::int32_t> h_atom{1};
  BasisPlan h_basis;
  CHECK(make_basis(h_offsets, h_atom, h_basis, error));
  const std::array<double, 1> negative_electrons{2.0};
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(
            h_basis, h_atom.data(), negative_electrons.data(), singlet.data(), restricted.data(),
            sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  const std::array<double, 1> too_many_electrons{-2.0};
  const std::array<std::int32_t, 1> doublet{1};
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(
            h_basis, h_atom.data(), too_many_electrons.data(), doublet.data(), restricted.data(),
            sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  const std::array<double, 1> nonfinite_charge{std::numeric_limits<double>::infinity()};
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(
            h_basis, h_atom.data(), nonfinite_charge.data(), singlet.data(), restricted.data(),
            sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  const std::array<std::int32_t, 1> negative_unpaired{-1};
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(
            h_basis, h_atom.data(), neutral.data(), negative_unpaired.data(), restricted.data(),
            sentinel, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(h_basis, nullptr, neutral.data(),
                                                        singlet.data(), restricted.data(), sentinel,
                                                        error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  const std::array<std::int32_t, 1> unsupported{87};
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(h_basis, unsupported.data(), neutral.data(),
                                                        singlet.data(), restricted.data(), sentinel,
                                                        error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  const std::array<std::int32_t, 1> invalid_spin{3};
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(
            h_basis, h_atom.data(), neutral.data(), doublet.data(), invalid_spin.data(), sentinel,
            error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(h_basis, h_atom.data(), neutral.data(),
                                                        doublet.data(), nullptr, sentinel,
                                                        error) == XTBLOOM_STATUS_INVALID_ARGUMENT);

  BasisPlan overflow = h_basis;
  overflow.total_orbitals = std::numeric_limits<std::int64_t>::max();
  overflow.batch_orbital_offsets.back() = overflow.total_orbitals;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(overflow, h_atom.data(), neutral.data(),
                                                        singlet.data(), restricted.data(), sentinel,
                                                        error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(sentinel.batch_size == 17);
  return 0;
}

int test_tampered_layout_rejected_atomically() {
  const std::vector<std::int64_t> atom_offsets{0, 2, 4};
  const std::vector<std::int32_t> atomic_numbers{1, 1, 8, 1};
  const std::array<double, 2> charges{0.0, 0.0};
  const std::array<std::int32_t, 2> unpaired{0, 1};
  const std::array<std::int32_t, 2> spin_channels{1, 2};
  BasisPlan basis;
  std::string error;
  CHECK(make_basis(atom_offsets, atomic_numbers, basis, error));
  WavefunctionLayout layout;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(
            basis, atomic_numbers.data(), charges.data(), unpaired.data(), spin_channels.data(),
            layout, error) == XTBLOOM_STATUS_SUCCESS);

  AlignedWorkspace workspace(layout.workspace_size_bytes);
  WavefunctionView valid_view;
  CHECK(xtbloom::detail::gfn2::bind_wavefunction_view(layout, workspace.data,
                                                      layout.workspace_size_bytes, valid_view,
                                                      error) == XTBLOOM_STATUS_SUCCESS);

  WavefunctionView view_sentinel;
  view_sentinel.workspace_base = reinterpret_cast<void*>(64);
  view_sentinel.workspace_size_bytes = 63u;
  view_sentinel.coefficients = reinterpret_cast<double*>(1);
  view_sentinel.eigenvalues = reinterpret_cast<double*>(2);
  view_sentinel.occupations = reinterpret_cast<double*>(3);
  view_sentinel.density = reinterpret_cast<double*>(4);
  view_sentinel.qsh = reinterpret_cast<double*>(5);
  view_sentinel.qat = reinterpret_cast<double*>(6);
  view_sentinel.dipole = reinterpret_cast<double*>(7);
  view_sentinel.quadrupole = reinterpret_cast<double*>(8);
  view_sentinel.energy_weighted_density = reinterpret_cast<double*>(9);

  WavefunctionSystemView system_sentinel;
  system_sentinel.atom_count = 10;
  system_sentinel.shell_count = 11;
  system_sentinel.orbital_count = 12;
  system_sentinel.spin_channels = 2;
  system_sentinel.electron_count = 13.0;
  system_sentinel.alpha_electron_count = 14.0;
  system_sentinel.beta_electron_count = 15.0;
  system_sentinel.coefficients = reinterpret_cast<double*>(11);
  system_sentinel.eigenvalues = reinterpret_cast<double*>(12);
  system_sentinel.occupations = reinterpret_cast<double*>(13);
  system_sentinel.density = reinterpret_cast<double*>(14);
  system_sentinel.qsh = reinterpret_cast<double*>(15);
  system_sentinel.qat = reinterpret_cast<double*>(16);
  system_sentinel.dipole = reinterpret_cast<double*>(17);
  system_sentinel.quadrupole = reinterpret_cast<double*>(18);
  system_sentinel.energy_weighted_density = reinterpret_cast<double*>(19);

  WavefunctionWarmStartIdentity identity_sentinel;
  identity_sentinel.model = XTBLOOM_MODEL_GFN1_XTB;
  identity_sentinel.geometry_cache_generation = 91u;
  identity_sentinel.batch_size = 92;
  identity_sentinel.atom_offsets = {93, 94};

  const auto rejects = [&](const WavefunctionLayout& bad) {
    WavefunctionView output_view = view_sentinel;
    if (xtbloom::detail::gfn2::bind_wavefunction_view(bad, workspace.data,
                                                      layout.workspace_size_bytes, output_view,
                                                      error) != XTBLOOM_STATUS_INVALID_ARGUMENT ||
        !same_view(output_view, view_sentinel)) {
      return false;
    }
    WavefunctionSystemView output_system = system_sentinel;
    if (xtbloom::detail::gfn2::make_wavefunction_system_view(
            bad, valid_view, 0, output_system, error) != XTBLOOM_STATUS_INVALID_ARGUMENT ||
        !same_system_view(output_system, system_sentinel)) {
      return false;
    }
    WavefunctionWarmStartIdentity output_identity = identity_sentinel;
    if (xtbloom::detail::gfn2::make_wavefunction_warm_start_identity(
            bad, 7u, output_identity, error) != XTBLOOM_STATUS_INVALID_ARGUMENT ||
        !same_identity(output_identity, identity_sentinel)) {
      return false;
    }
    return true;
  };

  WavefunctionLayout bad = layout;
  bad.atom_offsets.clear();
  CHECK(rejects(bad));
  bad = layout;
  bad.atom_offsets[0] = 1;
  CHECK(rejects(bad));
  bad = layout;
  bad.batch_shell_offsets[1] = bad.batch_shell_offsets[0];
  CHECK(rejects(bad));
  bad = layout;
  --bad.batch_orbital_offsets.back();
  CHECK(rejects(bad));
  bad = layout;
  bad.spin_channels[0] = 3;
  CHECK(rejects(bad));
  bad = layout;
  bad.electron_counts[0] += 1.0;
  CHECK(rejects(bad));
  bad = layout;
  bad.reference_atom_occupations[0] += 1.0;
  CHECK(rejects(bad));

  for (std::size_t field_index = 0; field_index < fields(layout).size(); ++field_index) {
    bad = layout;
    auto bad_fields = mutable_fields(bad);
    bad_fields[field_index]->system_offsets[1] = bad_fields[field_index]->element_count;
    CHECK(rejects(bad));
  }
  bad = layout;
  ++bad.coefficients.element_count;
  CHECK(rejects(bad));
  bad = layout;
  bad.eigenvalues.size_bytes += sizeof(double);
  CHECK(rejects(bad));
  bad = layout;
  bad.density.offset_bytes += xtbloom::detail::gfn2::kWavefunctionWorkspaceAlignment;
  CHECK(rejects(bad));
  bad = layout;
  bad.workspace_size_bytes += xtbloom::detail::gfn2::kWavefunctionWorkspaceAlignment;
  CHECK(rejects(bad));

  WavefunctionSystemView output_system = system_sentinel;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_system_view(layout, valid_view, layout.batch_size,
                                                             output_system, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_system_view(output_system, system_sentinel));
  WavefunctionView null_batch_view = valid_view;
  null_batch_view.qsh = nullptr;
  output_system = system_sentinel;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_system_view(
            layout, null_batch_view, 0, output_system, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_system_view(output_system, system_sentinel));

  const auto rejects_view = [&](const WavefunctionView& bad_view) {
    WavefunctionSystemView candidate = system_sentinel;
    return xtbloom::detail::gfn2::make_wavefunction_system_view(
               layout, bad_view, 0, candidate, error) == XTBLOOM_STATUS_INVALID_ARGUMENT &&
           same_system_view(candidate, system_sentinel);
  };
  WavefunctionView bad_view = valid_view;
  bad_view.qsh = bad_view.qat;
  CHECK(rejects_view(bad_view));
  bad_view = valid_view;
  bad_view.qsh = bad_view.qat + 1;
  CHECK(rejects_view(bad_view));
  bad_view = valid_view;
  bad_view.qsh = reinterpret_cast<double*>(reinterpret_cast<std::byte*>(bad_view.qsh) + 1);
  CHECK(rejects_view(bad_view));
  bad_view = valid_view;
  bad_view.workspace_base = static_cast<std::byte*>(workspace.data) +
                            xtbloom::detail::gfn2::kWavefunctionWorkspaceAlignment;
  bad_view.workspace_size_bytes -= xtbloom::detail::gfn2::kWavefunctionWorkspaceAlignment;
  CHECK(rejects_view(bad_view));
  bad_view = valid_view;
  --bad_view.workspace_size_bytes;
  CHECK(rejects_view(bad_view));

  ConstWavefunctionView valid_const_view;
  CHECK(xtbloom::detail::gfn2::bind_wavefunction_view(
            layout, static_cast<const void*>(workspace.data), layout.workspace_size_bytes,
            valid_const_view, error) == XTBLOOM_STATUS_SUCCESS);
  ConstWavefunctionView bad_const_view = valid_const_view;
  bad_const_view.coefficients = bad_const_view.coefficients + 1;
  ConstWavefunctionSystemView const_candidate;
  const_candidate.atom_count = 23;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_system_view(
            layout, bad_const_view, 0, const_candidate, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(const_candidate.atom_count == 23);
  WavefunctionView output_view = view_sentinel;
  CHECK(xtbloom::detail::gfn2::bind_wavefunction_view(
            layout, static_cast<std::byte*>(workspace.data) + 1, layout.workspace_size_bytes - 1u,
            output_view, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_view(output_view, view_sentinel));
  return 0;
}

int test_sad_multipole_initialization() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{8, 1};
  const std::array<double, 1> charge{1.0};
  const std::array<std::int32_t, 1> unpaired{0};
  const std::array<std::int32_t, 1> spin_channels{2};
  BasisPlan basis;
  std::string error;
  CHECK(make_basis(atom_offsets, atomic_numbers, basis, error));
  WavefunctionLayout layout;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(basis, atomic_numbers.data(), charge.data(),
                                                        unpaired.data(), spin_channels.data(),
                                                        layout, error) == XTBLOOM_STATUS_SUCCESS);
  AlignedWorkspace workspace(layout.workspace_size_bytes);
  CHECK(workspace.data != nullptr);
  std::fill_n(static_cast<double*>(workspace.data), layout.workspace_size_bytes / sizeof(double),
              9.0);
  WavefunctionView view;
  CHECK(xtbloom::detail::gfn2::bind_wavefunction_view(layout, workspace.data,
                                                      layout.workspace_size_bytes, view,
                                                      error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::initialize_sad_multipole_state(layout, view, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(view.coefficients[0] == 9.0);

  const std::int64_t nat = layout.total_atoms;
  const std::int64_t nsh = layout.total_shells;
  CHECK(view.qat[0] == 0.5);
  CHECK(view.qat[1] == 0.5);
  CHECK(view.qat[nat] == 0.0);
  CHECK(view.qat[nat + 1] == 0.0);

  std::int64_t shell = 0;
  for (std::size_t atom = 0; atom < atomic_numbers.size(); ++atom) {
    const auto* element =
        xtbloom::parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_numbers[atom]));
    for (std::size_t local_shell = 0; local_shell < element->shell_count; ++local_shell, ++shell) {
      const double expected = 0.5 *
                              layout.reference_shell_occupations[static_cast<std::size_t>(shell)] /
                              layout.reference_atom_occupations[atom];
      CHECK(near(view.qsh[shell], expected));
      CHECK(view.qsh[nsh + shell] == 0.0);
    }
  }
  CHECK(shell == nsh);
  for (std::int64_t index = 0; index < layout.dipole.element_count; ++index) {
    CHECK(view.dipole[index] == 0.0);
  }
  for (std::int64_t index = 0; index < layout.quadrupole.element_count; ++index) {
    CHECK(view.quadrupole[index] == 0.0);
  }

  std::fill_n(view.qsh, static_cast<std::size_t>(layout.qsh.element_count), 13.0);
  std::fill_n(view.qat, static_cast<std::size_t>(layout.qat.element_count), 13.0);
  std::fill_n(view.dipole, static_cast<std::size_t>(layout.dipole.element_count), 13.0);
  std::fill_n(view.quadrupole, static_cast<std::size_t>(layout.quadrupole.element_count), 13.0);
  const auto multipoles_are = [&](double value) {
    return std::all_of(view.qsh, view.qsh + layout.qsh.element_count,
                       [=](double entry) { return entry == value; }) &&
           std::all_of(view.qat, view.qat + layout.qat.element_count,
                       [=](double entry) { return entry == value; }) &&
           std::all_of(view.dipole, view.dipole + layout.dipole.element_count,
                       [=](double entry) { return entry == value; }) &&
           std::all_of(view.quadrupole, view.quadrupole + layout.quadrupole.element_count,
                       [=](double entry) { return entry == value; });
  };
  WavefunctionLayout bad = layout;
  bad.electron_counts[0] += 1.0;
  CHECK(xtbloom::detail::gfn2::initialize_sad_multipole_state(bad, view, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(multipoles_are(13.0));
  WavefunctionView null_view = view;
  null_view.qsh = nullptr;
  CHECK(xtbloom::detail::gfn2::initialize_sad_multipole_state(layout, null_view, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(multipoles_are(13.0));

  const auto sad_rejects = [&](const WavefunctionView& bad_view) {
    return xtbloom::detail::gfn2::initialize_sad_multipole_state(layout, bad_view, error) ==
               XTBLOOM_STATUS_INVALID_ARGUMENT &&
           multipoles_are(13.0);
  };
  WavefunctionView bad_view = view;
  bad_view.qsh = bad_view.qat;
  CHECK(sad_rejects(bad_view));
  bad_view = view;
  bad_view.qsh = bad_view.qat + 1;
  CHECK(sad_rejects(bad_view));
  bad_view = view;
  bad_view.qsh = reinterpret_cast<double*>(reinterpret_cast<std::byte*>(bad_view.qsh) + 1);
  CHECK(sad_rejects(bad_view));
  bad_view = view;
  bad_view.workspace_base = static_cast<std::byte*>(workspace.data) +
                            xtbloom::detail::gfn2::kWavefunctionWorkspaceAlignment;
  bad_view.workspace_size_bytes -= xtbloom::detail::gfn2::kWavefunctionWorkspaceAlignment;
  CHECK(sad_rejects(bad_view));
  bad_view = view;
  --bad_view.workspace_size_bytes;
  CHECK(sad_rejects(bad_view));

  error.reserve(256u);
  const std::size_t before = allocation_test::count.load(std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t sad_status =
      xtbloom::detail::gfn2::initialize_sad_multipole_state(layout, view, error);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  const std::size_t after = allocation_test::count.load(std::memory_order_relaxed);
  CHECK(sad_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(after == before);
  return 0;
}

int test_warm_start_identity_and_zero_allocation_views() {
  const std::vector<std::int64_t> atom_offsets{0, 2};
  const std::vector<std::int32_t> atomic_numbers{8, 1};
  const std::array<double, 1> charge{0.0};
  const std::array<std::int32_t, 1> unpaired{1};
  const std::array<std::int32_t, 1> spin_channels{2};
  BasisPlan basis;
  std::string error;
  CHECK(make_basis(atom_offsets, atomic_numbers, basis, error));
  WavefunctionLayout layout;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_layout(basis, atomic_numbers.data(), charge.data(),
                                                        unpaired.data(), spin_channels.data(),
                                                        layout, error) == XTBLOOM_STATUS_SUCCESS);

  WavefunctionWarmStartIdentity expected;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_warm_start_identity(
            layout, 41u, expected, error) == XTBLOOM_STATUS_SUCCESS);
  WavefunctionWarmStartIdentity candidate = expected;
  CHECK(xtbloom::detail::gfn2::wavefunction_warm_start_matches(expected, candidate));
  CHECK(xtbloom::detail::gfn2::validate_wavefunction_warm_start(expected, candidate, error) ==
        XTBLOOM_STATUS_SUCCESS);

  candidate.model = XTBLOOM_MODEL_GFN1_XTB;
  CHECK(!xtbloom::detail::gfn2::wavefunction_warm_start_matches(expected, candidate));
  candidate = expected;
  ++candidate.geometry_cache_generation;
  CHECK(!xtbloom::detail::gfn2::wavefunction_warm_start_matches(expected, candidate));
  candidate = expected;
  candidate.atomic_numbers[0] = 7;
  CHECK(!xtbloom::detail::gfn2::wavefunction_warm_start_matches(expected, candidate));
  candidate = expected;
  candidate.atom_offsets[1] = 1;
  CHECK(!xtbloom::detail::gfn2::wavefunction_warm_start_matches(expected, candidate));
  candidate = expected;
  candidate.molecular_charges[0] = 1.0;
  CHECK(!xtbloom::detail::gfn2::wavefunction_warm_start_matches(expected, candidate));
  candidate = expected;
  candidate.unpaired_electrons[0] = 3;
  CHECK(!xtbloom::detail::gfn2::wavefunction_warm_start_matches(expected, candidate));
  candidate = expected;
  candidate.spin_channels[0] = 1;
  CHECK(!xtbloom::detail::gfn2::wavefunction_warm_start_matches(expected, candidate));
  CHECK(xtbloom::detail::gfn2::validate_wavefunction_warm_start(expected, candidate, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);

  AlignedWorkspace workspace(layout.workspace_size_bytes);
  WavefunctionView view;
  WavefunctionSystemView system_view;
  CHECK(xtbloom::detail::gfn2::bind_wavefunction_view(layout, workspace.data,
                                                      layout.workspace_size_bytes, view,
                                                      error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::make_wavefunction_system_view(layout, view, 0, system_view, error) ==
        XTBLOOM_STATUS_SUCCESS);
  candidate = expected;

  /* Warm successful paths so std::string retains any implementation capacity. */
  error.reserve(256u);
  const std::size_t before = allocation_test::count.load(std::memory_order_relaxed);
  allocation_test::enabled.store(true, std::memory_order_relaxed);
  const xtbloom_status_t bind_status = xtbloom::detail::gfn2::bind_wavefunction_view(
      layout, workspace.data, layout.workspace_size_bytes, view, error);
  const xtbloom_status_t slice_status =
      xtbloom::detail::gfn2::make_wavefunction_system_view(layout, view, 0, system_view, error);
  const bool identity_matches =
      xtbloom::detail::gfn2::wavefunction_warm_start_matches(expected, candidate);
  allocation_test::enabled.store(false, std::memory_order_relaxed);
  const std::size_t after = allocation_test::count.load(std::memory_order_relaxed);
  CHECK(bind_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(slice_status == XTBLOOM_STATUS_SUCCESS);
  CHECK(identity_matches);
  CHECK(after == before);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_all_elements_reference_occupations(); status != 0) {
    return status;
  }
  if (const int status = test_ions_radical_fractional_and_layout(); status != 0) {
    return status;
  }
  if (const int status = test_invalid_electronic_states_and_overflow(); status != 0) {
    return status;
  }
  if (const int status = test_tampered_layout_rejected_atomically(); status != 0) {
    return status;
  }
  if (const int status = test_sad_multipole_initialization(); status != 0) {
    return status;
  }
  return test_warm_start_identity_and_zero_allocation_views();
}
