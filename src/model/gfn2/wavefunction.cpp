#include "model/gfn2/wavefunction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#include "data/parameters/gfn2.hpp"

namespace gpuxtb::detail::gfn2 {
namespace {

constexpr std::size_t kDoubleBytes = sizeof(double);

bool count_fits_vector(std::int64_t count, std::size_t element_size, bool add_sentinel = false) {
  if (count < 0) {
    return false;
  }
  const auto value = static_cast<std::uint64_t>(count);
  const auto extra = add_sentinel ? std::uint64_t{1} : std::uint64_t{0};
  if (value > std::numeric_limits<std::uint64_t>::max() - extra) {
    return false;
  }
  const std::uint64_t length = value + extra;
  return length <=
             static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / element_size &&
         length <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

bool checked_add(std::int64_t increment, std::int64_t& total) {
  if (increment < 0 || total > std::numeric_limits<std::int64_t>::max() - increment) {
    return false;
  }
  total += increment;
  return true;
}

bool checked_product(std::initializer_list<std::int64_t> factors, std::int64_t& result) {
  result = 1;
  for (const std::int64_t factor : factors) {
    if (factor < 0 || (factor != 0 && result > std::numeric_limits<std::int64_t>::max() / factor)) {
      return false;
    }
    result *= factor;
  }
  return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t& result) {
  const std::size_t remainder = value % alignment;
  const std::size_t padding = remainder == 0u ? 0u : alignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - padding) {
    return false;
  }
  result = value + padding;
  return true;
}

const parameters::gfn2::ShellParameters* element_shells(
    const parameters::gfn2::ElementParameters& element) {
  const std::size_t begin = element.shell_offset;
  const std::size_t count = element.shell_count;
  if (begin > parameters::gfn2::kShells.size() ||
      count > parameters::gfn2::kShells.size() - begin) {
    return nullptr;
  }
  return parameters::gfn2::kShells.data() + begin;
}

gpuxtb_status_t validate_basis_shape(const BasisPlan& basis, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      basis.total_orbitals <= 0 ||
      !count_fits_vector(basis.batch_size, sizeof(std::int64_t), true) ||
      !count_fits_vector(basis.total_atoms, sizeof(std::int64_t), true) ||
      !count_fits_vector(basis.total_shells, sizeof(std::int64_t), true) ||
      !count_fits_vector(basis.total_orbitals, sizeof(double))) {
    error = "wavefunction layout requires representable positive basis dimensions";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch_count = static_cast<std::size_t>(basis.batch_size);
  const std::size_t atom_count = static_cast<std::size_t>(basis.total_atoms);
  const std::size_t shell_count = static_cast<std::size_t>(basis.total_shells);
  if (basis.atom_offsets.size() != batch_count + 1u ||
      basis.batch_shell_offsets.size() != batch_count + 1u ||
      basis.batch_orbital_offsets.size() != batch_count + 1u ||
      basis.atom_shell_offsets.size() != atom_count + 1u ||
      basis.atom_orbital_offsets.size() != atom_count + 1u ||
      basis.shell_orbital_offsets.size() != shell_count + 1u ||
      basis.shell_to_atom.size() != shell_count ||
      basis.principal_quantum_numbers.size() != shell_count ||
      basis.angular_momenta.size() != shell_count || basis.slater_exponents.size() != shell_count ||
      basis.atom_offsets.front() != 0 || basis.atom_offsets.back() != basis.total_atoms ||
      basis.batch_shell_offsets.front() != 0 ||
      basis.batch_shell_offsets.back() != basis.total_shells ||
      basis.batch_orbital_offsets.front() != 0 ||
      basis.batch_orbital_offsets.back() != basis.total_orbitals ||
      basis.atom_shell_offsets.front() != 0 ||
      basis.atom_shell_offsets.back() != basis.total_shells ||
      basis.atom_orbital_offsets.front() != 0 ||
      basis.atom_orbital_offsets.back() != basis.total_orbitals ||
      basis.shell_orbital_offsets.front() != 0 ||
      basis.shell_orbital_offsets.back() != basis.total_orbitals) {
    error = "wavefunction layout received an incomplete or inconsistent basis plan";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  for (std::size_t system = 0; system < batch_count; ++system) {
    const std::int64_t atom_begin = basis.atom_offsets[system];
    const std::int64_t atom_end = basis.atom_offsets[system + 1u];
    const std::int64_t shell_begin = basis.batch_shell_offsets[system];
    const std::int64_t shell_end = basis.batch_shell_offsets[system + 1u];
    const std::int64_t orbital_begin = basis.batch_orbital_offsets[system];
    const std::int64_t orbital_end = basis.batch_orbital_offsets[system + 1u];
    if (atom_begin < 0 || atom_begin > atom_end || atom_end > basis.total_atoms ||
        shell_begin < 0 || shell_begin > shell_end || shell_end > basis.total_shells ||
        orbital_begin < 0 || orbital_begin > orbital_end || orbital_end > basis.total_orbitals ||
        shell_begin != basis.atom_shell_offsets[static_cast<std::size_t>(atom_begin)] ||
        shell_end != basis.atom_shell_offsets[static_cast<std::size_t>(atom_end)] ||
        orbital_begin != basis.atom_orbital_offsets[static_cast<std::size_t>(atom_begin)] ||
        orbital_end != basis.atom_orbital_offsets[static_cast<std::size_t>(atom_end)]) {
      error = "wavefunction basis offsets are not valid ragged partitions";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

bool append_field_count(WavefunctionFieldLayout& field, std::size_t system, std::int64_t count) {
  field.system_offsets[system] = field.element_count;
  return checked_add(count, field.element_count);
}

bool finish_field(WavefunctionFieldLayout& field, std::size_t batch_count, std::size_t& cursor) {
  field.system_offsets[batch_count] = field.element_count;
  if (!align_up(cursor, kWavefunctionWorkspaceAlignment, field.offset_bytes) ||
      static_cast<std::uint64_t>(field.element_count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / kDoubleBytes) {
    return false;
  }
  field.size_bytes = static_cast<std::size_t>(field.element_count) * kDoubleBytes;
  if (field.offset_bytes > std::numeric_limits<std::size_t>::max() - field.size_bytes) {
    return false;
  }
  cursor = field.offset_bytes + field.size_bytes;
  return true;
}

const std::array<const WavefunctionFieldLayout*, 9> field_layouts(
    const WavefunctionLayout& layout) {
  return {{&layout.coefficients, &layout.eigenvalues, &layout.occupations, &layout.density,
           &layout.qsh, &layout.qat, &layout.dipole, &layout.quadrupole,
           &layout.energy_weighted_density}};
}

template <typename View>
std::array<const void*, 9> view_field_pointers(const View& view) {
  return {{view.coefficients, view.eigenvalues, view.occupations, view.density, view.qsh, view.qat,
           view.dipole, view.quadrupole, view.energy_weighted_density}};
}

bool address_range_fits(const void* base, std::size_t size_bytes) {
  if (base == nullptr) {
    return false;
  }
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(base);
  return size_bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool valid_nonempty_partition(const std::vector<std::int64_t>& offsets, std::size_t batch_count,
                              std::int64_t total) {
  if (offsets.size() != batch_count + 1u || offsets.front() != 0 || offsets.back() != total) {
    return false;
  }
  for (std::size_t system = 0; system < batch_count; ++system) {
    if (offsets[system] < 0 || offsets[system] >= offsets[system + 1u] ||
        offsets[system + 1u] > total) {
      return false;
    }
  }
  return true;
}

gpuxtb_status_t validate_layout_metadata(const WavefunctionLayout& layout, std::string& error) {
  if (layout.batch_size <= 0 || layout.total_atoms <= 0 || layout.total_shells <= 0 ||
      layout.total_orbitals <= 0 ||
      !count_fits_vector(layout.batch_size, sizeof(std::int64_t), true) ||
      !count_fits_vector(layout.total_atoms, sizeof(double)) ||
      !count_fits_vector(layout.total_shells, sizeof(double)) ||
      !count_fits_vector(layout.total_orbitals, sizeof(double))) {
    error = "wavefunction layout metadata is incomplete";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t batch_count = static_cast<std::size_t>(layout.batch_size);
  const std::size_t atom_count = static_cast<std::size_t>(layout.total_atoms);
  const std::size_t shell_count = static_cast<std::size_t>(layout.total_shells);
  if (layout.atom_offsets.size() != batch_count + 1u ||
      layout.batch_shell_offsets.size() != batch_count + 1u ||
      layout.batch_orbital_offsets.size() != batch_count + 1u ||
      layout.atomic_numbers.size() != atom_count ||
      layout.molecular_charges.size() != batch_count ||
      layout.unpaired_electrons.size() != batch_count ||
      layout.spin_channels.size() != batch_count ||
      layout.reference_atom_occupations.size() != atom_count ||
      layout.reference_shell_occupations.size() != shell_count ||
      layout.electron_counts.size() != batch_count ||
      layout.alpha_electron_counts.size() != batch_count ||
      layout.beta_electron_counts.size() != batch_count || layout.workspace_size_bytes == 0u ||
      layout.workspace_size_bytes % kWavefunctionWorkspaceAlignment != 0u) {
    error = "wavefunction layout metadata sizes are inconsistent";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  if (!valid_nonempty_partition(layout.atom_offsets, batch_count, layout.total_atoms) ||
      !valid_nonempty_partition(layout.batch_shell_offsets, batch_count, layout.total_shells) ||
      !valid_nonempty_partition(layout.batch_orbital_offsets, batch_count, layout.total_orbitals)) {
    error = "wavefunction batch offsets must be nonempty monotone ragged partitions";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  std::size_t previous_end = 0u;
  for (const WavefunctionFieldLayout* field : field_layouts(layout)) {
    std::size_t expected_offset = 0u;
    if (!align_up(previous_end, kWavefunctionWorkspaceAlignment, expected_offset)) {
      error = "wavefunction field alignment overflows size_t";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    if (field->element_count < 0 || field->system_offsets.size() != batch_count + 1u ||
        field->system_offsets.front() != 0 ||
        field->system_offsets.back() != field->element_count ||
        field->offset_bytes != expected_offset ||
        static_cast<std::uint64_t>(field->element_count) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / kDoubleBytes ||
        field->size_bytes != static_cast<std::size_t>(field->element_count) * kDoubleBytes ||
        field->offset_bytes > layout.workspace_size_bytes ||
        field->size_bytes > layout.workspace_size_bytes - field->offset_bytes) {
      error = "wavefunction field layout is invalid or overlapping";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t system = 0; system < batch_count; ++system) {
      if (field->system_offsets[system] < 0 ||
          field->system_offsets[system] > field->system_offsets[system + 1u]) {
        error = "wavefunction field system offsets are not monotone";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
    previous_end = field->offset_bytes + field->size_bytes;
  }

  std::size_t expected_workspace_size = 0u;
  if (!align_up(previous_end, kWavefunctionWorkspaceAlignment, expected_workspace_size) ||
      expected_workspace_size != layout.workspace_size_bytes) {
    error = "wavefunction workspace size is not the canonical packed size";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const auto fields = field_layouts(layout);
  for (std::size_t system = 0; system < batch_count; ++system) {
    const std::int64_t atom_begin = layout.atom_offsets[system];
    const std::int64_t atom_end = layout.atom_offsets[system + 1u];
    const std::int64_t shell_begin = layout.batch_shell_offsets[system];
    const std::int64_t shell_end = layout.batch_shell_offsets[system + 1u];
    const std::int64_t orbital_begin = layout.batch_orbital_offsets[system];
    const std::int64_t orbital_end = layout.batch_orbital_offsets[system + 1u];
    const std::int64_t atoms = atom_end - atom_begin;
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t orbitals = orbital_end - orbital_begin;
    const std::int32_t unpaired = layout.unpaired_electrons[system];
    const std::int32_t nspin = layout.spin_channels[system];
    const double charge = layout.molecular_charges[system];
    if (!std::isfinite(charge) || unpaired < 0 || (nspin != 1 && nspin != 2)) {
      error = "wavefunction charge, unpaired-electron, or spin-channel metadata is invalid";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    std::int64_t expected_shell = shell_begin;
    std::int64_t expected_orbitals = 0;
    double reference_electrons = 0.0;
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::size_t atom_index = static_cast<std::size_t>(atom);
      const std::int32_t atomic_number = layout.atomic_numbers[atom_index];
      const auto* element =
          parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number) {
        error = "wavefunction layout contains an unsupported atomic number";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      const auto* element_shell_data = element_shells(*element);
      if (element_shell_data == nullptr || element->shell_count == 0u ||
          static_cast<std::int64_t>(element->shell_count) > shell_end - expected_shell) {
        error = "wavefunction shell partition does not match its element list";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }

      double atom_reference = 0.0;
      for (std::size_t local_shell = 0; local_shell < element->shell_count; ++local_shell) {
        const auto& shell_parameters = element_shell_data[local_shell];
        const std::size_t shell_index = static_cast<std::size_t>(expected_shell);
        if (!(shell_parameters.reference_occupation >= 0.0) ||
            !std::isfinite(shell_parameters.reference_occupation) ||
            layout.reference_shell_occupations[shell_index] !=
                shell_parameters.reference_occupation) {
          error = "wavefunction reference shell occupations do not match GFN2 parameters";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        atom_reference += shell_parameters.reference_occupation;
        const std::int64_t shell_orbitals =
            2 * static_cast<std::int64_t>(shell_parameters.angular_momentum) + 1;
        if (!checked_add(shell_orbitals, expected_orbitals)) {
          error = "wavefunction orbital metadata overflows the supported index range";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        ++expected_shell;
      }
      if (!(atom_reference > 0.0) || !std::isfinite(atom_reference) ||
          layout.reference_atom_occupations[atom_index] != atom_reference) {
        error = "wavefunction reference atom occupations do not match GFN2 parameters";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      reference_electrons += atom_reference;
    }
    if (expected_shell != shell_end || expected_orbitals != orbitals ||
        !std::isfinite(reference_electrons)) {
      error = "wavefunction shell or orbital counts do not match its element list";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    const double electrons = reference_electrons - charge;
    if (!std::isfinite(electrons) || electrons < 0.0 ||
        electrons > static_cast<double>(std::numeric_limits<std::int64_t>::max() - 2048)) {
      error = "wavefunction electron-count metadata is invalid";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    const auto integral_electrons = static_cast<std::int64_t>(std::round(electrons));
    const double alpha = 0.5 * (electrons + static_cast<double>(unpaired));
    const double beta = 0.5 * (electrons - static_cast<double>(unpaired));
    if ((integral_electrons & 1) != (static_cast<std::int64_t>(unpaired) & 1) ||
        !std::isfinite(alpha) || !std::isfinite(beta) || beta < 0.0 ||
        alpha > static_cast<double>(orbitals) || beta > static_cast<double>(orbitals) ||
        layout.electron_counts[system] != electrons ||
        layout.alpha_electron_counts[system] != alpha ||
        layout.beta_electron_counts[system] != beta) {
      error = "wavefunction spin-resolved electron metadata is inconsistent";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    std::int64_t matrix_elements = 0;
    std::int64_t spin_matrix_elements = 0;
    std::int64_t spin_orbitals = 0;
    std::int64_t occupation_elements = 0;
    std::int64_t spin_shells = 0;
    std::int64_t spin_atoms = 0;
    std::int64_t dipole_elements = 0;
    std::int64_t quadrupole_elements = 0;
    if (!checked_product({orbitals, orbitals}, matrix_elements) ||
        !checked_product({matrix_elements, nspin}, spin_matrix_elements) ||
        !checked_product({orbitals, nspin}, spin_orbitals) ||
        !checked_product({orbitals, 2}, occupation_elements) ||
        !checked_product({shells, nspin}, spin_shells) ||
        !checked_product({atoms, nspin}, spin_atoms) ||
        !checked_product({spin_atoms, 3}, dipole_elements) ||
        !checked_product({spin_atoms, kWavefunctionQuadrupoleComponents}, quadrupole_elements)) {
      error = "wavefunction field dimensions overflow the supported index range";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    const std::array<std::int64_t, 9> expected_counts{
        {spin_matrix_elements, spin_orbitals, occupation_elements, spin_matrix_elements,
         spin_shells, spin_atoms, dipole_elements, quadrupole_elements, spin_matrix_elements}};
    for (std::size_t field_index = 0; field_index < fields.size(); ++field_index) {
      const auto* field = fields[field_index];
      if (field->system_offsets[system + 1u] - field->system_offsets[system] !=
          expected_counts[field_index]) {
        error = "wavefunction field system offsets do not match its ragged dimensions";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

template <typename View>
gpuxtb_status_t validate_bound_view(const WavefunctionLayout& layout, const View& view,
                                    std::string& error) {
  if (view.workspace_base == nullptr ||
      reinterpret_cast<std::uintptr_t>(view.workspace_base) % kWavefunctionWorkspaceAlignment !=
          0u ||
      view.workspace_size_bytes < layout.workspace_size_bytes ||
      !address_range_fits(view.workspace_base, view.workspace_size_bytes)) {
    error = "wavefunction view workspace binding is NULL, truncated, misaligned, or overflowing";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(view.workspace_base);
  const auto layouts = field_layouts(layout);
  const auto pointers = view_field_pointers(view);
  for (std::size_t field = 0; field < layouts.size(); ++field) {
    if (pointers[field] == nullptr ||
        reinterpret_cast<std::uintptr_t>(pointers[field]) != base + layouts[field]->offset_bytes) {
      error = "wavefunction view fields do not match the canonical workspace binding";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

template <typename Byte, typename View>
gpuxtb_status_t bind_view_impl(const WavefunctionLayout& layout, Byte* workspace,
                               std::size_t workspace_size, View& view, std::string& error) {
  gpuxtb_status_t status = validate_layout_metadata(layout, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (workspace == nullptr || workspace_size < layout.workspace_size_bytes ||
      reinterpret_cast<std::uintptr_t>(workspace) % kWavefunctionWorkspaceAlignment != 0u ||
      !address_range_fits(workspace, workspace_size)) {
    error = "wavefunction workspace is NULL, too small, misaligned, or has an overflowing range";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(workspace);
  View created;
  created.workspace_base = workspace;
  created.workspace_size_bytes = workspace_size;
  created.coefficients =
      reinterpret_cast<decltype(created.coefficients)>(base + layout.coefficients.offset_bytes);
  created.eigenvalues =
      reinterpret_cast<decltype(created.eigenvalues)>(base + layout.eigenvalues.offset_bytes);
  created.occupations =
      reinterpret_cast<decltype(created.occupations)>(base + layout.occupations.offset_bytes);
  created.density = reinterpret_cast<decltype(created.density)>(base + layout.density.offset_bytes);
  created.qsh = reinterpret_cast<decltype(created.qsh)>(base + layout.qsh.offset_bytes);
  created.qat = reinterpret_cast<decltype(created.qat)>(base + layout.qat.offset_bytes);
  created.dipole = reinterpret_cast<decltype(created.dipole)>(base + layout.dipole.offset_bytes);
  created.quadrupole =
      reinterpret_cast<decltype(created.quadrupole)>(base + layout.quadrupole.offset_bytes);
  created.energy_weighted_density = reinterpret_cast<decltype(created.energy_weighted_density)>(
      base + layout.energy_weighted_density.offset_bytes);
  view = created;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

template <typename BatchView, typename SystemView>
gpuxtb_status_t make_system_view_impl(const WavefunctionLayout& layout, const BatchView& batch_view,
                                      std::int64_t system, SystemView& system_view,
                                      std::string& error) {
  const gpuxtb_status_t layout_status = validate_layout_metadata(layout, error);
  if (layout_status != GPUXTB_STATUS_SUCCESS) {
    return layout_status;
  }
  if (system < 0 || system >= layout.batch_size) {
    error = "wavefunction system view requires a valid system index";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  const gpuxtb_status_t view_status = validate_bound_view(layout, batch_view, error);
  if (view_status != GPUXTB_STATUS_SUCCESS) {
    return view_status;
  }
  const std::size_t index = static_cast<std::size_t>(system);
  SystemView created;
  created.atom_count = layout.atom_offsets[index + 1u] - layout.atom_offsets[index];
  created.shell_count = layout.batch_shell_offsets[index + 1u] - layout.batch_shell_offsets[index];
  created.orbital_count =
      layout.batch_orbital_offsets[index + 1u] - layout.batch_orbital_offsets[index];
  created.spin_channels = layout.spin_channels[index];
  created.electron_count = layout.electron_counts[index];
  created.alpha_electron_count = layout.alpha_electron_counts[index];
  created.beta_electron_count = layout.beta_electron_counts[index];
  created.coefficients = batch_view.coefficients + layout.coefficients.system_offsets[index];
  created.eigenvalues = batch_view.eigenvalues + layout.eigenvalues.system_offsets[index];
  created.occupations = batch_view.occupations + layout.occupations.system_offsets[index];
  created.density = batch_view.density + layout.density.system_offsets[index];
  created.qsh = batch_view.qsh + layout.qsh.system_offsets[index];
  created.qat = batch_view.qat + layout.qat.system_offsets[index];
  created.dipole = batch_view.dipole + layout.dipole.system_offsets[index];
  created.quadrupole = batch_view.quadrupole + layout.quadrupole.system_offsets[index];
  created.energy_weighted_density =
      batch_view.energy_weighted_density + layout.energy_weighted_density.system_offsets[index];
  system_view = created;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace

gpuxtb_status_t make_wavefunction_layout(const BasisPlan& basis, const std::int32_t* atomic_numbers,
                                         const double* molecular_charges,
                                         const std::int32_t* unpaired_electrons,
                                         const std::int32_t* spin_channels,
                                         WavefunctionLayout& layout, std::string& error) {
  gpuxtb_status_t status = validate_basis_shape(basis, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (atomic_numbers == nullptr || molecular_charges == nullptr || unpaired_electrons == nullptr ||
      spin_channels == nullptr) {
    error = "wavefunction layout electronic inputs must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch_count = static_cast<std::size_t>(basis.batch_size);
  const std::size_t atom_count = static_cast<std::size_t>(basis.total_atoms);
  const std::size_t shell_count = static_cast<std::size_t>(basis.total_shells);
  try {
    WavefunctionLayout created;
    created.batch_size = basis.batch_size;
    created.total_atoms = basis.total_atoms;
    created.total_shells = basis.total_shells;
    created.total_orbitals = basis.total_orbitals;
    created.atom_offsets = basis.atom_offsets;
    created.batch_shell_offsets = basis.batch_shell_offsets;
    created.batch_orbital_offsets = basis.batch_orbital_offsets;
    created.atomic_numbers.resize(atom_count);
    created.molecular_charges.resize(batch_count);
    created.unpaired_electrons.resize(batch_count);
    created.spin_channels.resize(batch_count);
    created.reference_atom_occupations.assign(atom_count, 0.0);
    created.reference_shell_occupations.assign(shell_count, 0.0);
    created.electron_counts.resize(batch_count);
    created.alpha_electron_counts.resize(batch_count);
    created.beta_electron_counts.resize(batch_count);

    for (WavefunctionFieldLayout* field :
         {&created.coefficients, &created.eigenvalues, &created.occupations, &created.density,
          &created.qsh, &created.qat, &created.dipole, &created.quadrupole,
          &created.energy_weighted_density}) {
      field->system_offsets.resize(batch_count + 1u);
    }

    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::size_t atom_index = static_cast<std::size_t>(atom);
      const std::int32_t atomic_number = atomic_numbers[atom_index];
      const auto* element =
          parameters::gfn2::find_element(static_cast<std::uint32_t>(atomic_number));
      if (element == nullptr || element->atomic_number != atomic_number) {
        error = "wavefunction layout contains an unsupported atomic number";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      const auto* shells = element_shells(*element);
      const std::int64_t shell_begin = basis.atom_shell_offsets[atom_index];
      const std::int64_t shell_end = basis.atom_shell_offsets[atom_index + 1u];
      if (shells == nullptr || shell_begin < 0 || shell_begin > shell_end ||
          shell_end > basis.total_shells || shell_end - shell_begin != element->shell_count) {
        error = "wavefunction basis shell layout does not match the element list";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      const std::int64_t atom_orbital_begin = basis.atom_orbital_offsets[atom_index];
      const std::int64_t atom_orbital_end = basis.atom_orbital_offsets[atom_index + 1u];
      if (atom_orbital_begin < 0 || atom_orbital_begin > atom_orbital_end ||
          atom_orbital_end > basis.total_orbitals ||
          atom_orbital_begin !=
              basis.shell_orbital_offsets[static_cast<std::size_t>(shell_begin)] ||
          atom_orbital_end != basis.shell_orbital_offsets[static_cast<std::size_t>(shell_end)]) {
        error = "wavefunction atom orbital offsets do not match its shell range";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }

      double atom_reference = 0.0;
      for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
        const std::size_t shell_index = static_cast<std::size_t>(shell);
        const std::size_t local_shell = static_cast<std::size_t>(shell - shell_begin);
        const auto& shell_parameters = shells[local_shell];
        const std::int64_t orbital_count = basis.shell_orbital_offsets[shell_index + 1u] -
                                           basis.shell_orbital_offsets[shell_index];
        const std::int64_t expected_orbitals =
            2 * static_cast<std::int64_t>(shell_parameters.angular_momentum) + 1;
        if (basis.shell_to_atom[shell_index] != atom ||
            basis.principal_quantum_numbers[shell_index] !=
                shell_parameters.principal_quantum_number ||
            basis.angular_momenta[shell_index] != shell_parameters.angular_momentum ||
            basis.slater_exponents[shell_index] != shell_parameters.slater ||
            orbital_count != expected_orbitals || !(shell_parameters.reference_occupation >= 0.0) ||
            !std::isfinite(shell_parameters.reference_occupation)) {
          error = "wavefunction basis shell metadata does not match GFN2 parameters";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        created.reference_shell_occupations[shell_index] = shell_parameters.reference_occupation;
        atom_reference += shell_parameters.reference_occupation;
      }
      if (!std::isfinite(atom_reference)) {
        error = "wavefunction reference atom occupation is not finite";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
      created.atomic_numbers[atom_index] = atomic_number;
      created.reference_atom_occupations[atom_index] = atom_reference;
    }

    for (std::size_t system = 0; system < batch_count; ++system) {
      const std::int64_t atom_begin = basis.atom_offsets[system];
      const std::int64_t atom_end = basis.atom_offsets[system + 1u];
      const std::int64_t shell_begin = basis.batch_shell_offsets[system];
      const std::int64_t shell_end = basis.batch_shell_offsets[system + 1u];
      const std::int64_t orbital_begin = basis.batch_orbital_offsets[system];
      const std::int64_t orbital_end = basis.batch_orbital_offsets[system + 1u];
      const std::int64_t atoms = atom_end - atom_begin;
      const std::int64_t shells = shell_end - shell_begin;
      const std::int64_t orbitals = orbital_end - orbital_begin;
      const double charge = molecular_charges[system];
      const std::int32_t unpaired = unpaired_electrons[system];
      const std::int32_t nspin = spin_channels[system];
      if (!std::isfinite(charge) || unpaired < 0 || (nspin != 1 && nspin != 2)) {
        error =
            "molecular charges must be finite, unpaired-electron counts nonnegative, and spin "
            "channels one or two";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }

      double reference_electrons = 0.0;
      for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
        reference_electrons += created.reference_atom_occupations[static_cast<std::size_t>(atom)];
      }
      const double electrons = reference_electrons - charge;
      if (!std::isfinite(electrons) || electrons < 0.0 ||
          electrons > static_cast<double>(std::numeric_limits<std::int64_t>::max() - 2048)) {
        error = "molecular charge produces an invalid or unrepresentable electron count";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      const double rounded_electrons = std::round(electrons);
      const auto integral_electrons = static_cast<std::int64_t>(rounded_electrons);
      if ((integral_electrons & 1) != (static_cast<std::int64_t>(unpaired) & 1)) {
        error = "total electron count and unpaired-electron count have incompatible parity";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      const double alpha = 0.5 * (electrons + static_cast<double>(unpaired));
      const double beta = 0.5 * (electrons - static_cast<double>(unpaired));
      if (!std::isfinite(alpha) || !std::isfinite(beta) || beta < 0.0 ||
          alpha > static_cast<double>(orbitals) || beta > static_cast<double>(orbitals)) {
        error = "alpha or beta electron count cannot be represented by the system orbital space";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }

      std::int64_t matrix_elements = 0;
      std::int64_t spin_matrix_elements = 0;
      std::int64_t spin_orbitals = 0;
      std::int64_t occupation_elements = 0;
      std::int64_t spin_shells = 0;
      std::int64_t spin_atoms = 0;
      std::int64_t dipole_elements = 0;
      std::int64_t quadrupole_elements = 0;
      if (!checked_product({orbitals, orbitals}, matrix_elements) ||
          !checked_product({matrix_elements, nspin}, spin_matrix_elements) ||
          !checked_product({orbitals, nspin}, spin_orbitals) ||
          !checked_product({orbitals, 2}, occupation_elements) ||
          !checked_product({shells, nspin}, spin_shells) ||
          !checked_product({atoms, nspin}, spin_atoms) ||
          !checked_product({spin_atoms, 3}, dipole_elements) ||
          !checked_product({spin_atoms, kWavefunctionQuadrupoleComponents}, quadrupole_elements) ||
          !append_field_count(created.coefficients, system, spin_matrix_elements) ||
          !append_field_count(created.eigenvalues, system, spin_orbitals) ||
          !append_field_count(created.occupations, system, occupation_elements) ||
          !append_field_count(created.density, system, spin_matrix_elements) ||
          !append_field_count(created.qsh, system, spin_shells) ||
          !append_field_count(created.qat, system, spin_atoms) ||
          !append_field_count(created.dipole, system, dipole_elements) ||
          !append_field_count(created.quadrupole, system, quadrupole_elements) ||
          !append_field_count(created.energy_weighted_density, system, spin_matrix_elements)) {
        error = "wavefunction field dimensions overflow the supported index range";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }

      created.molecular_charges[system] = charge == 0.0 ? 0.0 : charge;
      created.unpaired_electrons[system] = unpaired;
      created.spin_channels[system] = nspin;
      created.electron_counts[system] = electrons;
      created.alpha_electron_counts[system] = alpha;
      created.beta_electron_counts[system] = beta;
    }

    std::size_t cursor = 0u;
    if (!finish_field(created.coefficients, batch_count, cursor) ||
        !finish_field(created.eigenvalues, batch_count, cursor) ||
        !finish_field(created.occupations, batch_count, cursor) ||
        !finish_field(created.density, batch_count, cursor) ||
        !finish_field(created.qsh, batch_count, cursor) ||
        !finish_field(created.qat, batch_count, cursor) ||
        !finish_field(created.dipole, batch_count, cursor) ||
        !finish_field(created.quadrupole, batch_count, cursor) ||
        !finish_field(created.energy_weighted_density, batch_count, cursor) ||
        !align_up(cursor, kWavefunctionWorkspaceAlignment, created.workspace_size_bytes)) {
      error = "wavefunction workspace byte size overflows size_t";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }

    status = validate_layout_metadata(created, error);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return status;
    }
    layout = std::move(created);
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate GFN2 wavefunction layout metadata";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

gpuxtb_status_t bind_wavefunction_view(const WavefunctionLayout& layout, void* workspace,
                                       std::size_t workspace_size, WavefunctionView& view,
                                       std::string& error) {
  return bind_view_impl(layout, static_cast<std::byte*>(workspace), workspace_size, view, error);
}

gpuxtb_status_t bind_wavefunction_view(const WavefunctionLayout& layout, const void* workspace,
                                       std::size_t workspace_size, ConstWavefunctionView& view,
                                       std::string& error) {
  return bind_view_impl(layout, static_cast<const std::byte*>(workspace), workspace_size, view,
                        error);
}

gpuxtb_status_t make_wavefunction_system_view(const WavefunctionLayout& layout,
                                              const WavefunctionView& batch_view,
                                              std::int64_t system,
                                              WavefunctionSystemView& system_view,
                                              std::string& error) {
  return make_system_view_impl(layout, batch_view, system, system_view, error);
}

gpuxtb_status_t make_wavefunction_system_view(const WavefunctionLayout& layout,
                                              const ConstWavefunctionView& batch_view,
                                              std::int64_t system,
                                              ConstWavefunctionSystemView& system_view,
                                              std::string& error) {
  return make_system_view_impl(layout, batch_view, system, system_view, error);
}

gpuxtb_status_t initialize_sad_multipole_state(const WavefunctionLayout& layout,
                                               const WavefunctionView& view, std::string& error) {
  const gpuxtb_status_t status = validate_layout_metadata(layout, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  const gpuxtb_status_t view_status = validate_bound_view(layout, view, error);
  if (view_status != GPUXTB_STATUS_SUCCESS) {
    return view_status;
  }

  std::fill_n(view.qsh, static_cast<std::size_t>(layout.qsh.element_count), 0.0);
  std::fill_n(view.qat, static_cast<std::size_t>(layout.qat.element_count), 0.0);
  std::fill_n(view.dipole, static_cast<std::size_t>(layout.dipole.element_count), 0.0);
  std::fill_n(view.quadrupole, static_cast<std::size_t>(layout.quadrupole.element_count), 0.0);

  const std::size_t batch_count = static_cast<std::size_t>(layout.batch_size);
  for (std::size_t system = 0; system < batch_count; ++system) {
    const std::int64_t atom_begin = layout.atom_offsets[system];
    const std::int64_t atom_end = layout.atom_offsets[system + 1u];
    const std::int64_t shell_begin = layout.batch_shell_offsets[system];
    const std::int64_t atoms = atom_end - atom_begin;
    const double atom_charge = layout.molecular_charges[system] / static_cast<double>(atoms);
    const std::int64_t qat_base = layout.qat.system_offsets[system];
    const std::int64_t qsh_base = layout.qsh.system_offsets[system];
    std::int64_t shell = shell_begin;
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::size_t atom_index = static_cast<std::size_t>(atom);
      const std::int64_t local_atom = atom - atom_begin;
      view.qat[qat_base + local_atom] = atom_charge;

      const auto* element = parameters::gfn2::find_element(
          static_cast<std::uint32_t>(layout.atomic_numbers[atom_index]));
      const double atom_reference = layout.reference_atom_occupations[atom_index];
      for (std::size_t local_shell = 0; local_shell < element->shell_count;
           ++local_shell, ++shell) {
        view.qsh[qsh_base + shell - shell_begin] =
            (layout.reference_shell_occupations[static_cast<std::size_t>(shell)] / atom_reference) *
            atom_charge;
      }
    }
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t make_wavefunction_warm_start_identity(const WavefunctionLayout& layout,
                                                      std::uint64_t geometry_cache_generation,
                                                      WavefunctionWarmStartIdentity& identity,
                                                      std::string& error) {
  gpuxtb_status_t status = validate_layout_metadata(layout, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  try {
    WavefunctionWarmStartIdentity created;
    created.model = GPUXTB_MODEL_GFN2_XTB;
    created.geometry_cache_generation = geometry_cache_generation;
    created.batch_size = layout.batch_size;
    created.total_atoms = layout.total_atoms;
    created.total_shells = layout.total_shells;
    created.total_orbitals = layout.total_orbitals;
    created.atom_offsets = layout.atom_offsets;
    created.batch_shell_offsets = layout.batch_shell_offsets;
    created.batch_orbital_offsets = layout.batch_orbital_offsets;
    created.atomic_numbers = layout.atomic_numbers;
    created.molecular_charges = layout.molecular_charges;
    created.unpaired_electrons = layout.unpaired_electrons;
    created.spin_channels = layout.spin_channels;
    identity = std::move(created);
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate wavefunction warm-start identity metadata";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

bool wavefunction_warm_start_matches(const WavefunctionWarmStartIdentity& expected,
                                     const WavefunctionWarmStartIdentity& candidate) noexcept {
  return expected.model == candidate.model &&
         expected.geometry_cache_generation == candidate.geometry_cache_generation &&
         expected.batch_size == candidate.batch_size &&
         expected.total_atoms == candidate.total_atoms &&
         expected.total_shells == candidate.total_shells &&
         expected.total_orbitals == candidate.total_orbitals &&
         expected.atom_offsets == candidate.atom_offsets &&
         expected.batch_shell_offsets == candidate.batch_shell_offsets &&
         expected.batch_orbital_offsets == candidate.batch_orbital_offsets &&
         expected.atomic_numbers == candidate.atomic_numbers &&
         expected.molecular_charges == candidate.molecular_charges &&
         expected.unpaired_electrons == candidate.unpaired_electrons &&
         expected.spin_channels == candidate.spin_channels;
}

gpuxtb_status_t validate_wavefunction_warm_start(const WavefunctionWarmStartIdentity& expected,
                                                 const WavefunctionWarmStartIdentity& candidate,
                                                 std::string& error) {
  if (!wavefunction_warm_start_matches(expected, candidate)) {
    error =
        "wavefunction warm start does not match model, topology, charge, spin, or geometry "
        "generation";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail::gfn2
