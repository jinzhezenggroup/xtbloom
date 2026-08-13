// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "runtime/gfn1_cpu_execution.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
xtbloom_const_buffer_t input_buffer(const std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T),
          XTBLOOM_MEMORY_HOST, 0u};
}

template <typename T>
xtbloom_buffer_t output_buffer(std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T),
          XTBLOOM_MEMORY_HOST, 0u};
}

template <typename T>
bool read_values(std::istream& input, std::size_t count, std::vector<T>& values) {
  const std::size_t begin = values.size();
  values.resize(begin + count);
  for (std::size_t index = begin; index < values.size(); ++index) {
    if (!(input >> values[index])) return false;
  }
  return true;
}

template <typename T>
void print_array(const std::vector<T>& values) {
  std::cout << '[';
  for (std::size_t index = 0u; index < values.size(); ++index) {
    if (index != 0u) std::cout << ',';
    if constexpr (std::is_same_v<T, std::uint8_t>) {
      std::cout << static_cast<unsigned int>(values[index]);
    } else {
      std::cout << values[index];
    }
  }
  std::cout << ']';
}

}  // namespace

int main() {
  std::string magic;
  std::int64_t request_count = 0;
  if (!(std::cin >> magic >> request_count) || magic != "XTBLOOM_GFN1_PROBE_V2" ||
      request_count <= 0 || request_count > 100000) {
    std::cerr << "invalid GFN1 probe header\n";
    return 2;
  }

  xtbloom::detail::Gfn1CpuExecutionCache cache;
  std::cout << std::setprecision(17) << '[';
  for (std::int64_t request_index = 0; request_index < request_count; ++request_index) {
    std::int64_t batch_size = 0;
    if (!(std::cin >> batch_size) || batch_size <= 0 || batch_size > 1024) {
      std::cerr << "invalid GFN1 probe batch size\n";
      return 2;
    }

    std::vector<std::int64_t> atom_offsets{0};
    std::vector<std::int32_t> atomic_numbers;
    std::vector<double> positions;
    std::vector<double> molecular_charges;
    std::vector<std::int32_t> unpaired_electrons;
    std::vector<std::int32_t> spin_channels;
    std::vector<std::int64_t> point_offsets{0};
    std::vector<double> point_positions;
    std::vector<double> point_charges;
    std::vector<double> point_gammas;
    std::vector<double> periodic_shifts;
    std::vector<std::int64_t> response_offsets{0};
    std::vector<double> response_matrices;
    bool periodic_enabled = false;

    for (std::int64_t system = 0; system < batch_size; ++system) {
      std::int64_t atoms = 0;
      std::int64_t points = 0;
      double charge = 0.0;
      std::int32_t unpaired = 0;
      std::int32_t channels = 0;
      std::int32_t has_periodic = 0;
      if (!(std::cin >> atoms >> points >> charge >> unpaired >> channels >> has_periodic) ||
          atoms <= 0 || atoms > 100000 || points < 0 || points > 100000 ||
          (has_periodic != 0 && has_periodic != 1)) {
        std::cerr << "invalid GFN1 probe system header\n";
        return 2;
      }
      if (!read_values(std::cin, static_cast<std::size_t>(atoms), atomic_numbers) ||
          !read_values(std::cin, 3u * static_cast<std::size_t>(atoms), positions) ||
          !read_values(std::cin, 3u * static_cast<std::size_t>(points), point_positions) ||
          !read_values(std::cin, static_cast<std::size_t>(points), point_charges) ||
          !read_values(std::cin, static_cast<std::size_t>(points), point_gammas)) {
        std::cerr << "truncated GFN1 probe system payload\n";
        return 2;
      }
      const std::size_t atom_count = static_cast<std::size_t>(atoms);
      if (has_periodic != 0) {
        periodic_enabled = true;
        if (!read_values(std::cin, atom_count, periodic_shifts) ||
            !read_values(std::cin, atom_count * atom_count, response_matrices)) {
          std::cerr << "truncated GFN1 periodic probe payload\n";
          return 2;
        }
      } else {
        periodic_shifts.insert(periodic_shifts.end(), atom_count, 0.0);
        response_matrices.insert(response_matrices.end(), atom_count * atom_count, 0.0);
      }
      atom_offsets.push_back(atom_offsets.back() + atoms);
      point_offsets.push_back(point_offsets.back() + points);
      response_offsets.push_back(response_offsets.back() + atoms * atoms);
      molecular_charges.push_back(charge);
      unpaired_electrons.push_back(unpaired);
      spin_channels.push_back(channels);
    }

    xtbloom_batch_t batch{};
    batch.struct_size = XTBLOOM_BATCH_V3_SIZE;
    batch.api_version = XTBLOOM_API_VERSION;
    batch.batch_size = batch_size;
    batch.total_atoms = atom_offsets.back();
    batch.total_point_charges = point_offsets.back();
    batch.total_charge_response_elements =
        periodic_enabled ? static_cast<std::int64_t>(response_matrices.size()) : 0;
    batch.atom_offsets = input_buffer(atom_offsets);
    batch.atomic_numbers = input_buffer(atomic_numbers);
    batch.positions = input_buffer(positions);
    batch.molecular_charges = input_buffer(molecular_charges);
    batch.unpaired_electrons = input_buffer(unpaired_electrons);
    batch.spin_channels = input_buffer(spin_channels);
    batch.point_charge_offsets = input_buffer(point_offsets);
    batch.point_charge_positions = input_buffer(point_positions);
    batch.point_charge_values = input_buffer(point_charges);
    batch.point_charge_gammas = input_buffer(point_gammas);
    if (periodic_enabled) {
      batch.atomic_potential_shifts = input_buffer(periodic_shifts);
      batch.charge_response_offsets = input_buffer(response_offsets);
      batch.charge_response_matrix = input_buffer(response_matrices);
    }

    constexpr std::uint32_t flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                                    XTBLOOM_COMPUTE_ATOMIC_CHARGES |
                                    XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
    xtbloom_compute_options_t options{};
    options.struct_size = XTBLOOM_COMPUTE_OPTIONS_V3_SIZE;
    options.api_version = XTBLOOM_API_VERSION;
    options.model = XTBLOOM_MODEL_GFN1_XTB;
    options.flags = flags;
    options.max_scc_iterations = 250;
    /* Match the reviewed GFN1 conformance accuracy contract. The tighter
     * energy gate removes residual iteration-history noise without silently
     * redefining the public charge convergence policy. */
    options.charge_tolerance = 1.0e-7;
    options.energy_tolerance = 1.0e-9;
    options.electronic_temperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;
    options.scc_start_mode = XTBLOOM_SCC_START_FRESH;
    options.scc_mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
    options.scc_mixer_history = 8;
    options.scc_mixer_damping = 0.4;
    options.determinism = XTBLOOM_DETERMINISM_REPRODUCIBLE;

    std::vector<double> energies(static_cast<std::size_t>(batch_size));
    std::vector<double> forces(3u * static_cast<std::size_t>(batch.total_atoms));
    std::vector<double> charges(static_cast<std::size_t>(batch.total_atoms));
    std::vector<double> point_forces(3u * static_cast<std::size_t>(batch.total_point_charges));
    std::vector<std::int32_t> iterations(static_cast<std::size_t>(batch_size));
    std::vector<std::uint8_t> converged(static_cast<std::size_t>(batch_size));
    std::vector<std::int32_t> statuses(static_cast<std::size_t>(batch_size));
    xtbloom_batch_result_t result{};
    result.struct_size = XTBLOOM_BATCH_RESULT_V2_SIZE;
    result.api_version = XTBLOOM_API_VERSION;
    result.energies = output_buffer(energies);
    result.forces = output_buffer(forces);
    result.atomic_charges = output_buffer(charges);
    result.point_charge_forces = output_buffer(point_forces);
    result.scc_iterations = output_buffer(iterations);
    result.scc_converged = output_buffer(converged);
    result.per_system_status = output_buffer(statuses);

    std::string error;
    const xtbloom_status_t status =
        xtbloom::detail::execute_gfn1_cpu(cache, batch, options, result, error);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      std::cerr << "hidden GFN1 CPU execution failed with status " << status << ": " << error
                << '\n';
      return 1;
    }

    if (request_index != 0) std::cout << ',';
    std::cout << '{';
    std::cout << "\"flags\":" << result.flags << ",\"energies\":";
    print_array(energies);
    std::cout << ",\"forces\":";
    print_array(forces);
    std::cout << ",\"atomic_charges\":";
    print_array(charges);
    std::cout << ",\"point_charge_forces\":";
    print_array(point_forces);
    std::cout << ",\"iterations\":";
    print_array(iterations);
    std::cout << ",\"converged\":";
    print_array(converged);
    std::cout << ",\"statuses\":";
    print_array(statuses);
    std::cout << '}';
  }
  std::cout << "]\n";
  return 0;
}
