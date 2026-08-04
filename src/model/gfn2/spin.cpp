#include "model/gfn2/spin.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace gpuxtb::detail::gfn2 {
namespace {

/*
 * tblite's element spin constants (ss, sp, pp, sd, pd, dd), pinned from
 * src/tblite/data/spin.f90 at the parameter provenance revision recorded in
 * data/parameters/spin_manifest.json. Values are in Hartree and are symmetric in
 * the two angular-momentum indices.
 */
constexpr std::array<std::array<double, 6>, 86> kSpinConstants{{
    {{-0.0716250, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0865500, -0.0386630, -0.0674250, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0178000, -0.0139500, -0.0180500, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0229750, -0.0186250, -0.0175750, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0272500, -0.0219370, -0.0195750, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0305000, -0.0250250, -0.0226750, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0330750, -0.0275000, -0.0254500, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0350750, -0.0295380, -0.0278500, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0369000, -0.0311870, -0.0299250, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0383000, -0.0326250, -0.0317250, -0.0141250, -0.0152500, -0.0413500}},
    {{-0.0150750, -0.0133370, -0.0229250, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0165000, -0.0130750, -0.0175000, -0.0093750, -0.0179630, -0.0223500}},
    {{-0.0182500, -0.0138380, -0.0139750, -0.0082250, -0.0117000, -0.0129000}},
    {{-0.0195250, -0.0150000, -0.0143750, -0.0084500, -0.0116120, -0.0140000}},
    {{-0.0205750, -0.0161250, -0.0149000, -0.0093000, -0.0119870, -0.0148250}},
    {{-0.0213250, -0.0170130, -0.0155000, -0.0100370, -0.0121750, -0.0149500}},
    {{-0.0217500, -0.0177130, -0.0160500, -0.0109750, -0.0126620, -0.0150750}},
    {{-0.0221500, -0.0183630, -0.0165500, -0.0118870, -0.0131130, -0.0153000}},
    {{-0.0106500, -0.0109000, -0.0164750, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0118000, -0.0104500, -0.0134500, -0.0055000, -0.0035130, -0.0101750}},
    {{-0.0127250, -0.0108620, -0.0138500, -0.0047880, -0.0024130, -0.0125250}},
    {{-0.0134250, -0.0112380, -0.0146500, -0.0043380, -0.0019750, -0.0138750}},
    {{-0.0140750, -0.0114630, -0.0152750, -0.0040500, -0.0017250, -0.0149250}},
    {{-0.0144750, -0.0116120, -0.0160000, -0.0037250, -0.0014630, -0.0157750}},
    {{-0.0149000, -0.0118000, -0.0167250, -0.0034870, -0.0013120, -0.0165000}},
    {{-0.0154000, -0.0120250, -0.0177500, -0.0032880, -0.0011630, -0.0171250}},
    {{-0.0157000, -0.0120250, -0.0187000, -0.0031500, -0.0010250, -0.0177500}},
    {{-0.0161500, -0.0122000, -0.0197000, -0.0030370, -0.0009130, -0.0183000}},
    {{-0.0166500, -0.0123500, -0.0203000, -0.0028250, -0.0008620, -0.0188250}},
    {{-0.0168500, -0.0123250, -0.0214500, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0172250, -0.0128120, -0.0134000, -0.0085250, -0.0130000, -0.0157750}},
    {{-0.0174500, -0.0133500, -0.0135500, -0.0081120, -0.0128130, -0.0175250}},
    {{-0.0178750, -0.0137630, -0.0135750, -0.0080500, -0.0123250, -0.0176500}},
    {{-0.0180000, -0.0141250, -0.0136250, -0.0081500, -0.0120130, -0.0172000}},
    {{-0.0181000, -0.0143750, -0.0136750, -0.0082750, -0.0117500, -0.0167750}},
    {{-0.0181250, -0.0145500, -0.0137000, -0.0086880, -0.0118000, -0.0164250}},
    {{-0.0095500, -0.0096000, -0.0167250, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0105750, -0.0092870, -0.0125500, -0.0074000, -0.0059000, -0.0079500}},
    {{-0.0115000, -0.0098250, -0.0136000, -0.0072880, -0.0046250, -0.0090000}},
    {{-0.0121500, -0.0099880, -0.0162250, -0.0066250, -0.0036250, -0.0098750}},
    {{-0.0125750, -0.0102620, -0.0191750, -0.0060630, -0.0029250, -0.0104750}},
    {{-0.0129000, -0.0105000, -0.0222250, -0.0055750, -0.0024250, -0.0109000}},
    {{-0.0131250, -0.0106250, -0.0247250, -0.0051250, -0.0020120, -0.0113000}},
    {{-0.0133500, -0.0107620, -0.0276000, -0.0047370, -0.0016750, -0.0116000}},
    {{-0.0135500, -0.0108380, -0.0320500, -0.0043750, -0.0014250, -0.0118750}},
    {{-0.0136500, -0.0109440, -0.0287000, -0.0041250, -0.0012880, -0.0121250}},
    {{-0.0139250, -0.0110500, -0.0241750, -0.0038870, -0.0009630, -0.0124000}},
    {{-0.0138500, -0.0105000, -0.0196500, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0142250, -0.0105500, -0.0115750, -0.0050370, -0.0093750, -0.0100000}},
    {{-0.0143000, -0.0108750, -0.0116750, -0.0046880, -0.0090750, -0.0118750}},
    {{-0.0145250, -0.0111250, -0.0116250, -0.0043750, -0.0087130, -0.0124250}},
    {{-0.0145250, -0.0112500, -0.0115750, -0.0041870, -0.0081750, -0.0121750}},
    {{-0.0146500, -0.0113870, -0.0114750, -0.0044620, -0.0083620, -0.0128250}},
    {{-0.0146500, -0.0114250, -0.0114500, -0.0048750, -0.0085750, -0.0132000}},
    {{-0.0082000, -0.0085880, -0.0153000, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0092500, -0.0083000, -0.0113750, -0.0063870, -0.0042250, -0.0079250}},
    {{-0.0099000, -0.0084250, -0.0114000, -0.0059370, -0.0033750, -0.0090250}},
    {{-0.0881750, -0.0066380, -0.0019250, -0.0017000, -0.0017000, -0.0234250}},
    {{-0.0890750, -0.0065000, -0.0009500, -0.0015370, -0.0017370, -0.0237000}},
    {{-0.0901000, -0.0063750, -0.0000750, -0.0014630, -0.0016250, -0.0230250}},
    {{-0.0908000, -0.0064880, 0.0004500, -0.0013000, -0.0016500, -0.0226250}},
    {{-0.0918250, -0.0065380, 0.0014000, -0.0012750, -0.0017250, -0.0222250}},
    {{-0.0922250, -0.0065380, 0.0017250, -0.0012000, -0.0018000, -0.0218250}},
    {{-0.0928812, -0.0065798, 0.0024101, -0.0011021, -0.0016846, -0.0209135}},
    {{-0.0936096, -0.0066189, 0.0030779, -0.0010125, -0.0016808, -0.0201625}},
    {{-0.0943380, -0.0066581, 0.0037457, -0.0009229, -0.0016769, -0.0194115}},
    {{-0.0951750, -0.0067500, 0.0042250, -0.0008380, -0.0016250, -0.0190000}},
    {{-0.0956500, -0.0067250, 0.0040000, -0.0007370, -0.0007630, -0.0176000}},
    {{-0.0963500, -0.0067370, 0.0044000, -0.0006750, -0.0007065, -0.0160000}},
    {{-0.0958500, -0.0066500, 0.0024000, -0.0007500, -0.0008000, -0.0175500}},
    {{-0.1086250, -0.0079000, 0.0063250, -0.0047000, -0.0007120, -0.0269000}},
    {{-0.0121750, -0.0096750, -0.0126250, -0.0076250, -0.0041130, -0.0104250}},
    {{-0.0123000, -0.0095750, -0.0134000, -0.0071380, -0.0034630, -0.0109250}},
    {{-0.0125000, -0.0094620, -0.0144500, -0.0066880, -0.0029130, -0.0112500}},
    {{-0.0126000, -0.0093310, -0.0148000, -0.0063000, -0.0026130, -0.0114500}},
    {{-0.0127000, -0.0092000, -0.0205750, -0.0059380, -0.0021120, -0.0115500}},
    {{-0.0127500, -0.0092750, -0.0209250, -0.0056880, -0.0019120, -0.0116000}},
    {{-0.0127500, -0.0092250, -0.0222500, -0.0054370, -0.0017870, -0.0117000}},
    {{-0.0129000, -0.0089380, -0.0257625, -0.0052500, -0.0015000, -0.0117750}},
    {{-0.0129250, -0.0091870, -0.0292750, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0133500, -0.0091120, -0.0107250, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0135750, -0.0094250, -0.0110000, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0136750, -0.0095380, -0.0109500, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0137500, -0.0096380, -0.0108500, 0.0000000, 0.0000000, 0.0000000}},
    {{-0.0137750, -0.0096750, -0.0107250, -0.0026000, -0.0073630, -0.0119000}},
    {{-0.0139000, -0.0097380, -0.0106500, -0.0028750, -0.0078120, -0.0130000}},
}};

bool representable_as_size(std::int64_t value) {
  return value >= 0 && static_cast<std::uint64_t>(value) <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
}

bool count_bytes(std::int64_t count, std::size_t element_size, std::size_t& bytes) {
  if (count < 0 || static_cast<std::uint64_t>(count) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      static_cast<std::uint64_t>(count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                    std::size_t second_bytes) {
  if (first_bytes == 0u || second_bytes == 0u) {
    return false;
  }
  const std::uintptr_t first_begin = reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  return first_begin < second_begin + second_bytes && second_begin < first_begin + first_bytes;
}

std::size_t coupling_index(std::uint8_t first, std::uint8_t second) {
  if (first > second) {
    std::swap(first, second);
  }
  if (first == 0u) {
    return second == 0u ? 0u : (second == 1u ? 1u : 3u);
  }
  if (first == 1u) {
    return second == 1u ? 2u : 4u;
  }
  return 5u;
}

gpuxtb_status_t validate_view(SpinPolarizationView view, std::string& error) {
  if (view.batch_size <= 0 || view.total_atoms <= 0 || view.total_shells <= 0 ||
      view.shell_population_elements <= 0 || !representable_as_size(view.batch_size) ||
      !representable_as_size(view.total_atoms) || !representable_as_size(view.total_shells) ||
      !representable_as_size(view.shell_population_elements) ||
      view.batch_size == std::numeric_limits<std::int64_t>::max() ||
      view.total_atoms == std::numeric_limits<std::int64_t>::max() ||
      view.atom_offset_count != view.batch_size + 1 ||
      view.batch_shell_offset_count != view.batch_size + 1 ||
      view.atom_shell_offset_count != view.total_atoms + 1 ||
      view.shell_population_offset_count != view.batch_size + 1 ||
      view.spin_channel_count != view.batch_size ||
      view.coupling_offset_count != view.total_atoms + 1 || view.coupling_matrix_count <= 0 ||
      view.atom_offsets == nullptr || view.batch_shell_offsets == nullptr ||
      view.atom_shell_offsets == nullptr || view.shell_population_offsets == nullptr ||
      view.spin_channels == nullptr || view.coupling_offsets == nullptr ||
      view.coupling_matrices == nullptr) {
    error = "spin-polarization view is incomplete or has unrepresentable dimensions";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (view.atom_offsets[0] != 0 || view.atom_offsets[view.batch_size] != view.total_atoms ||
      view.batch_shell_offsets[0] != 0 ||
      view.batch_shell_offsets[view.batch_size] != view.total_shells ||
      view.atom_shell_offsets[0] != 0 ||
      view.atom_shell_offsets[view.total_atoms] != view.total_shells ||
      view.shell_population_offsets[0] != 0 ||
      view.shell_population_offsets[view.batch_size] != view.shell_population_elements ||
      view.coupling_offsets[0] != 0 ||
      view.coupling_offsets[view.total_atoms] != view.coupling_matrix_count) {
    error = "spin-polarization view offsets do not span their packed fields";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    const std::int64_t atom_begin = view.atom_offsets[system];
    const std::int64_t atom_end = view.atom_offsets[system + 1];
    const std::int64_t shell_begin = view.batch_shell_offsets[system];
    const std::int64_t shell_end = view.batch_shell_offsets[system + 1];
    const std::int64_t population_begin = view.shell_population_offsets[system];
    const std::int64_t population_end = view.shell_population_offsets[system + 1];
    const std::int32_t channels = view.spin_channels[system];
    if (atom_begin < 0 || atom_begin >= atom_end || atom_end > view.total_atoms ||
        shell_begin < 0 || shell_begin >= shell_end || shell_end > view.total_shells ||
        population_begin < 0 || population_begin >= population_end ||
        population_end > view.shell_population_elements || (channels != 1 && channels != 2) ||
        population_end - population_begin != (shell_end - shell_begin) * channels ||
        view.atom_shell_offsets[atom_begin] != shell_begin ||
        view.atom_shell_offsets[atom_end] != shell_end) {
      error = "spin-polarization view has an invalid ragged system partition";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  for (std::int64_t atom = 0; atom < view.total_atoms; ++atom) {
    const std::int64_t shell_begin = view.atom_shell_offsets[atom];
    const std::int64_t shell_end = view.atom_shell_offsets[atom + 1];
    const std::int64_t matrix_begin = view.coupling_offsets[atom];
    const std::int64_t matrix_end = view.coupling_offsets[atom + 1];
    const std::int64_t shells = shell_end - shell_begin;
    if (shell_begin < 0 || shell_begin >= shell_end || shell_end > view.total_shells ||
        matrix_begin < 0 || matrix_begin > matrix_end || matrix_end > view.coupling_matrix_count ||
        shells > 3 || matrix_end - matrix_begin != shells * shells) {
      error = "spin-polarization view has an invalid atom-local coupling partition";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
    for (std::int64_t matrix = matrix_begin; matrix < matrix_end; ++matrix) {
      if (!std::isfinite(view.coupling_matrices[matrix])) {
        error = "spin-polarization view contains a non-finite coupling";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  return GPUXTB_STATUS_SUCCESS;
}

bool evaluate_unrestricted_system(SpinPolarizationView view, std::int64_t system,
                                  const double* shell_populations, double* potentials,
                                  double& energy) {
  const std::int64_t atom_begin = view.atom_offsets[system];
  const std::int64_t atom_end = view.atom_offsets[system + 1];
  const std::int64_t system_shell_begin = view.batch_shell_offsets[system];
  const std::int64_t system_shell_end = view.batch_shell_offsets[system + 1];
  const std::int64_t system_shells = system_shell_end - system_shell_begin;
  const std::int64_t magnetization_base = view.shell_population_offsets[system] + system_shells;
  energy = 0.0;
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    const std::int64_t shell_begin = view.atom_shell_offsets[atom];
    const std::int64_t shell_end = view.atom_shell_offsets[atom + 1];
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t matrix_begin = view.coupling_offsets[atom];
    for (std::int64_t row = 0; row < shells; ++row) {
      double potential = 0.0;
      for (std::int64_t column = 0; column < shells; ++column) {
        const std::int64_t population =
            magnetization_base + shell_begin - system_shell_begin + column;
        potential = std::fma(view.coupling_matrices[matrix_begin + row * shells + column],
                             shell_populations[population], potential);
      }
      if (!std::isfinite(potential)) {
        return false;
      }
      const std::int64_t population = magnetization_base + shell_begin - system_shell_begin + row;
      energy = std::fma(0.5 * shell_populations[population], potential, energy);
      if (!std::isfinite(energy)) {
        return false;
      }
      if (potentials != nullptr) {
        potentials[population] = potential;
      }
    }
  }
  return true;
}

}  // namespace

gpuxtb_status_t make_spin_polarization_plan(const BasisPlan& basis,
                                            const WavefunctionLayout& wavefunction,
                                            SpinPolarizationPlan& plan, std::string& error) {
  if (basis.batch_size <= 0 || basis.total_atoms <= 0 || basis.total_shells <= 0 ||
      wavefunction.batch_size != basis.batch_size ||
      wavefunction.total_atoms != basis.total_atoms ||
      wavefunction.total_shells != basis.total_shells ||
      basis.atom_offsets != wavefunction.atom_offsets ||
      basis.batch_shell_offsets != wavefunction.batch_shell_offsets ||
      wavefunction.atomic_numbers.size() != static_cast<std::size_t>(basis.total_atoms) ||
      basis.atom_shell_offsets.size() != static_cast<std::size_t>(basis.total_atoms) + 1u ||
      basis.angular_momenta.size() != static_cast<std::size_t>(basis.total_shells) ||
      wavefunction.qsh.system_offsets.size() != static_cast<std::size_t>(basis.batch_size) + 1u ||
      wavefunction.spin_channels.size() != static_cast<std::size_t>(basis.batch_size)) {
    error = "spin-polarization plan requires one complete matching basis and wavefunction layout";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  try {
    SpinPolarizationPlan created;
    created.batch_size = basis.batch_size;
    created.total_atoms = basis.total_atoms;
    created.total_shells = basis.total_shells;
    created.shell_population_elements = wavefunction.qsh.element_count;
    created.atom_offsets = basis.atom_offsets;
    created.batch_shell_offsets = basis.batch_shell_offsets;
    created.atom_shell_offsets = basis.atom_shell_offsets;
    created.shell_population_offsets = wavefunction.qsh.system_offsets;
    created.spin_channels = wavefunction.spin_channels;
    created.coupling_offsets.resize(static_cast<std::size_t>(basis.total_atoms) + 1u, 0);

    std::int64_t coupling_count = 0;
    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::int64_t shell_begin = basis.atom_shell_offsets[static_cast<std::size_t>(atom)];
      const std::int64_t shell_end = basis.atom_shell_offsets[static_cast<std::size_t>(atom) + 1u];
      const std::int64_t shells = shell_end - shell_begin;
      if (shell_begin < 0 || shell_begin >= shell_end || shell_end > basis.total_shells ||
          shells > 3 || coupling_count > std::numeric_limits<std::int64_t>::max() - shells * shells) {
        error = "spin-polarization basis has an unsupported atom-local shell partition";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      coupling_count += shells * shells;
      created.coupling_offsets[static_cast<std::size_t>(atom) + 1u] = coupling_count;
    }
    if (!representable_as_size(coupling_count)) {
      error = "spin-polarization coupling dimensions exceed host container limits";
      return GPUXTB_STATUS_ALLOCATION_FAILED;
    }
    created.coupling_matrices.resize(static_cast<std::size_t>(coupling_count));

    for (std::int64_t atom = 0; atom < basis.total_atoms; ++atom) {
      const std::int32_t atomic_number = wavefunction.atomic_numbers[static_cast<std::size_t>(atom)];
      if (atomic_number <= 0 || static_cast<std::size_t>(atomic_number) > kSpinConstants.size()) {
        error = "spin-polarization plan contains an unsupported element";
        return GPUXTB_STATUS_INVALID_ARGUMENT;
      }
      const std::int64_t shell_begin = basis.atom_shell_offsets[static_cast<std::size_t>(atom)];
      const std::int64_t shell_end = basis.atom_shell_offsets[static_cast<std::size_t>(atom) + 1u];
      const std::int64_t shells = shell_end - shell_begin;
      const std::int64_t matrix_begin = created.coupling_offsets[static_cast<std::size_t>(atom)];
      for (std::int64_t row = 0; row < shells; ++row) {
        const std::uint8_t row_l = basis.angular_momenta[static_cast<std::size_t>(shell_begin + row)];
        if (row_l > 2u) {
          error = "spin-polarization plan supports only GFN2 s, p, and d shells";
          return GPUXTB_STATUS_INVALID_ARGUMENT;
        }
        for (std::int64_t column = 0; column < shells; ++column) {
          const std::uint8_t column_l =
              basis.angular_momenta[static_cast<std::size_t>(shell_begin + column)];
          if (column_l > 2u) {
            error = "spin-polarization plan supports only GFN2 s, p, and d shells";
            return GPUXTB_STATUS_INVALID_ARGUMENT;
          }
          created.coupling_matrices[static_cast<std::size_t>(
              matrix_begin + row * shells + column)] =
              kSpinConstants[static_cast<std::size_t>(atomic_number - 1)]
                            [coupling_index(row_l, column_l)];
        }
      }
    }

    const SpinPolarizationView view = make_spin_polarization_view(created);
    if (validate_view(view, error) != GPUXTB_STATUS_SUCCESS) {
      return GPUXTB_STATUS_INTERNAL_ERROR;
    }
    plan = std::move(created);
    error.clear();
    return GPUXTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate the GFN2 spin-polarization plan";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  } catch (const std::length_error&) {
    error = "GFN2 spin-polarization plan dimensions exceed host container limits";
    return GPUXTB_STATUS_ALLOCATION_FAILED;
  }
}

SpinPolarizationView make_spin_polarization_view(const SpinPolarizationPlan& plan) noexcept {
  const auto count = [](std::size_t value) noexcept {
    return value <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())
               ? static_cast<std::int64_t>(value)
               : std::int64_t{-1};
  };
  return SpinPolarizationView{
      plan.batch_size,
      plan.total_atoms,
      plan.total_shells,
      plan.shell_population_elements,
      count(plan.atom_offsets.size()),
      count(plan.batch_shell_offsets.size()),
      count(plan.atom_shell_offsets.size()),
      count(plan.shell_population_offsets.size()),
      count(plan.spin_channels.size()),
      count(plan.coupling_offsets.size()),
      count(plan.coupling_matrices.size()),
      plan.atom_offsets.data(),
      plan.batch_shell_offsets.data(),
      plan.atom_shell_offsets.data(),
      plan.shell_population_offsets.data(),
      plan.spin_channels.data(),
      plan.coupling_offsets.data(),
      plan.coupling_matrices.data(),
  };
}

gpuxtb_status_t evaluate_spin_polarization_cpu(SpinPolarizationView view,
                                                const double* shell_populations,
                                                double* spin_energies,
                                                double* shell_potentials,
                                                std::string& error) {
  gpuxtb_status_t status = validate_view(view, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (shell_populations == nullptr || spin_energies == nullptr || shell_potentials == nullptr) {
    error = "spin-polarization populations and outputs must not be NULL";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }

  std::size_t population_bytes = 0u;
  std::size_t energy_bytes = 0u;
  if (!count_bytes(view.shell_population_elements, sizeof(double), population_bytes) ||
      !count_bytes(view.batch_size, sizeof(double), energy_bytes) ||
      ranges_overlap(shell_populations, population_bytes, spin_energies, energy_bytes) ||
      ranges_overlap(shell_populations, population_bytes, shell_potentials, population_bytes) ||
      ranges_overlap(spin_energies, energy_bytes, shell_potentials, population_bytes)) {
    error = "spin-polarization outputs must be mutually disjoint from their inputs";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  for (std::int64_t element = 0; element < view.shell_population_elements; ++element) {
    if (!std::isfinite(shell_populations[element])) {
      error = "spin-polarization shell populations contain NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }

  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    if (view.spin_channels[system] == 2) {
      double energy = 0.0;
      if (!evaluate_unrestricted_system(view, system, shell_populations, nullptr, energy)) {
        error = "spin-polarization energy or potential exceeded floating-point range";
        return GPUXTB_STATUS_INTERNAL_ERROR;
      }
    }
  }

  std::fill_n(spin_energies, static_cast<std::size_t>(view.batch_size), 0.0);
  std::fill_n(shell_potentials, static_cast<std::size_t>(view.shell_population_elements), 0.0);
  for (std::int64_t system = 0; system < view.batch_size; ++system) {
    if (view.spin_channels[system] == 2) {
      (void)evaluate_unrestricted_system(view, system, shell_populations, shell_potentials,
                                         spin_energies[system]);
    }
  }
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t add_spin_polarization_energy_system_cpu(
    SpinPolarizationView view, std::int64_t system, const double* shell_populations,
    double& accumulated_energy, std::string& error) {
  gpuxtb_status_t status = validate_view(view, error);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return status;
  }
  if (system < 0 || system >= view.batch_size || shell_populations == nullptr) {
    error = "spin-polarization energy system index or populations are invalid";
    return GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  if (!std::isfinite(accumulated_energy)) {
    error = "spin-polarization accumulated energy is not finite";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  const std::int64_t population_begin = view.shell_population_offsets[system];
  const std::int64_t population_end = view.shell_population_offsets[system + 1];
  for (std::int64_t element = population_begin; element < population_end; ++element) {
    if (!std::isfinite(shell_populations[element])) {
      error = "spin-polarization target populations contain NaN or infinity";
      return GPUXTB_STATUS_INVALID_ARGUMENT;
    }
  }
  double energy = 0.0;
  if (view.spin_channels[system] == 2 &&
      !evaluate_unrestricted_system(view, system, shell_populations, nullptr, energy)) {
    error = "spin-polarization target energy exceeded floating-point range";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  const double updated = accumulated_energy + energy;
  if (!std::isfinite(updated)) {
    error = "spin-polarization accumulated energy exceeded floating-point range";
    return GPUXTB_STATUS_INTERNAL_ERROR;
  }
  accumulated_energy = updated;
  error.clear();
  return GPUXTB_STATUS_SUCCESS;
}

}  // namespace gpuxtb::detail::gfn2
