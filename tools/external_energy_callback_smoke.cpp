// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "xtbloom/xtbloom.h"
#include "xtbloom/xtbloom_external_energy.h"

namespace {

struct CallbackState {
  int calls = 0;
  int potential_calls = 0;
  int energy_calls = 0;
  int force_calls = 0;
  int max_spin_channels = 0;
  int failure_mode = 0;
};

xtbloom_status_t callback(
    void* opaque, int64_t, xtbloom_external_energy_phase_t phase, int32_t spin_channels,
    int64_t atom_count, int64_t nao, const int32_t* atomic_numbers, int64_t atomic_number_elements,
    const double* positions, int64_t position_elements, const int64_t* orbital_to_atom,
    int64_t orbital_to_atom_elements, int64_t atom_index_begin, int64_t shell_count,
    const int64_t* shell_orbital_offsets, int64_t shell_orbital_offset_elements,
    const int64_t* shell_to_atom, int64_t shell_to_atom_elements,
    const uint8_t* principal_quantum_numbers, int64_t principal_quantum_number_elements,
    const uint8_t* angular_momenta, int64_t angular_momentum_elements, int64_t shell_index_begin,
    int64_t shell_orbital_index_begin, double molecular_charge, int32_t unpaired_electrons,
    const double* density, int64_t density_elements, const double* overlap,
    int64_t overlap_elements, const double* overlap_gradient, int64_t overlap_gradient_elements,
    int64_t projection_orbitals, int64_t projection_shell_count,
    const int64_t* projection_shell_orbital_offsets,
    int64_t projection_shell_orbital_offset_elements, const int64_t* projection_shell_to_atom,
    int64_t projection_shell_to_atom_elements, const uint8_t* projection_angular_momenta,
    int64_t projection_angular_momentum_elements, const double* projection_overlap,
    int64_t projection_overlap_elements, const double* projection_overlap_gradient,
    int64_t projection_overlap_gradient_elements, double* hamiltonian, int64_t hamiltonian_elements,
    double* force, int64_t force_elements, double* energy) {
  auto& state = *static_cast<CallbackState*>(opaque);
  ++state.calls;
  state.max_spin_channels = std::max(state.max_spin_channels, static_cast<int>(spin_channels));
  if (energy == nullptr || (spin_channels != 1 && spin_channels != 2) || atom_count <= 0 ||
      nao <= 0 || atomic_numbers == nullptr || atomic_number_elements != atom_count ||
      positions == nullptr || position_elements != 3 * atom_count || orbital_to_atom == nullptr ||
      orbital_to_atom_elements != nao || atom_index_begin < 0 || !std::isfinite(molecular_charge) ||
      shell_count <= 0 || shell_orbital_offsets == nullptr ||
      shell_orbital_offset_elements != shell_count + 1 || shell_to_atom == nullptr ||
      shell_to_atom_elements != shell_count || principal_quantum_numbers == nullptr ||
      principal_quantum_number_elements != shell_count || angular_momenta == nullptr ||
      angular_momentum_elements != shell_count || shell_index_begin < 0 ||
      shell_orbital_index_begin < 0 || unpaired_electrons < 0 || density == nullptr ||
      density_elements != spin_channels * nao * nao || overlap == nullptr ||
      overlap_elements != nao * nao || overlap_gradient == nullptr ||
      overlap_gradient_elements != 3 * atom_count * nao * nao ||
      projection_orbitals != 108 * atom_count || projection_shell_count != 36 * atom_count ||
      projection_shell_orbital_offsets == nullptr ||
      projection_shell_orbital_offset_elements != projection_shell_count + 1 ||
      projection_shell_to_atom == nullptr ||
      projection_shell_to_atom_elements != projection_shell_count ||
      projection_angular_momenta == nullptr ||
      projection_angular_momentum_elements != projection_shell_count ||
      projection_overlap == nullptr || projection_overlap_elements != nao * projection_orbitals ||
      projection_overlap_gradient == nullptr ||
      projection_overlap_gradient_elements != 3 * atom_count * nao * projection_orbitals) {
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (int64_t atom = 0; atom < atom_count; ++atom) {
    if (atomic_numbers[atom] != 1 || !std::isfinite(positions[3 * atom]) ||
        !std::isfinite(positions[3 * atom + 1]) || !std::isfinite(positions[3 * atom + 2])) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (int64_t orbital = 0; orbital < nao; ++orbital) {
    if (orbital_to_atom[orbital] < atom_index_begin ||
        orbital_to_atom[orbital] >= atom_index_begin + atom_count) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  if (shell_orbital_offsets[0] != shell_orbital_index_begin ||
      shell_orbital_offsets[shell_count] != shell_orbital_index_begin + nao) {
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (projection_shell_orbital_offsets[0] != 0 ||
      projection_shell_orbital_offsets[projection_shell_count] != projection_orbitals) {
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (int64_t shell = 0; shell < projection_shell_count; ++shell) {
    if (projection_shell_orbital_offsets[shell] >= projection_shell_orbital_offsets[shell + 1] ||
        projection_shell_to_atom[shell] < 0 || projection_shell_to_atom[shell] >= atom_count ||
        projection_angular_momenta[shell] > 2) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  for (int64_t shell = 0; shell < shell_count; ++shell) {
    if (shell_orbital_offsets[shell] >= shell_orbital_offsets[shell + 1] ||
        shell_to_atom[shell] < atom_index_begin ||
        shell_to_atom[shell] >= atom_index_begin + atom_count ||
        principal_quantum_numbers[shell] < 1 || angular_momenta[shell] > 4) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  if (std::abs(molecular_charge) > 1.0e-12 || unpaired_electrons != 2) {
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (phase == XTBLOOM_EXTERNAL_ENERGY_PHASE_POTENTIAL) {
    ++state.potential_calls;
    if (hamiltonian == nullptr || hamiltonian_elements != spin_channels * nao * nao) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (int32_t channel = 0; channel < spin_channels; ++channel) {
      hamiltonian[channel * nao * nao] += 1.0e-4;
    }
    if (force != nullptr || force_elements != 0) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (state.failure_mode == 1) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  } else if (phase == XTBLOOM_EXTERNAL_ENERGY_PHASE_ENERGY) {
    ++state.energy_calls;
    if (hamiltonian != nullptr || hamiltonian_elements != 0 || force != nullptr ||
        force_elements != 0) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    if (state.failure_mode == 2) {
      *energy = std::numeric_limits<double>::infinity();
      return XTBLOOM_STATUS_SUCCESS;
    }
  } else if (phase == XTBLOOM_EXTERNAL_ENERGY_PHASE_FORCE) {
    ++state.force_calls;
    if (hamiltonian != nullptr || hamiltonian_elements != 0 || force == nullptr ||
        force_elements != 3 * atom_count) {
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (int64_t element = 0; element < force_elements; ++element) {
      force[element] = 0.0;
    }
  } else {
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  *energy = 1.0e-3;
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_const_buffer_t input_view(const void* data, std::size_t bytes) {
  return {data, bytes, XTBLOOM_MEMORY_HOST, 0};
}

xtbloom_buffer_t output_view(void* data, std::size_t bytes) {
  return {data, bytes, XTBLOOM_MEMORY_HOST, 0};
}

}  // namespace

int main(int argc, char** argv) {
  const bool cuda_host_staged = argc > 1 && std::strcmp(argv[1], "--cuda-host-staged") == 0;
  if (xtbloom_context_set_external_energy_callback(nullptr, callback, nullptr) !=
          XTBLOOM_STATUS_INVALID_ARGUMENT ||
      xtbloom_context_set_external_energy_device_model(nullptr, nullptr) !=
          XTBLOOM_STATUS_INVALID_ARGUMENT ||
      xtbloom_context_copy_external_energy_device_gradients(nullptr, nullptr, 0) !=
          XTBLOOM_STATUS_INVALID_ARGUMENT) {
    return 2;
  }
  xtbloom_context_options_t context_options;
  xtbloom_batch_t batch;
  xtbloom_compute_options_t options;
  xtbloom_batch_result_t result;
  if (xtbloom_context_options_init(&context_options, sizeof(context_options)) !=
          XTBLOOM_STATUS_SUCCESS ||
      xtbloom_batch_init(&batch, sizeof(batch)) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom_compute_options_init(&options, sizeof(options)) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom_batch_result_init(&result, sizeof(result)) != XTBLOOM_STATUS_SUCCESS) {
    return 2;
  }
  /* The CUDA argument exercises the source-tree compatibility path: installing
   * an external energy callback on a CUDA context deliberately routes synchronous
   * SCC through the context-owned CPU cache until a device evaluator exists. */
  context_options.backend = cuda_host_staged ? XTBLOOM_BACKEND_CUDA : XTBLOOM_BACKEND_CPU;
  context_options.cpu_threads = 1;
  xtbloom_context_t* context = nullptr;
  if (xtbloom_context_create(&context_options, &context) != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "context: %s\n", xtbloom_get_last_error());
    return 3;
  }
  if (!cuda_host_staged && (xtbloom_context_set_external_energy_device_model(context, nullptr) !=
                                XTBLOOM_STATUS_NOT_SUPPORTED ||
                            xtbloom_context_copy_external_energy_device_gradients(
                                context, nullptr, 0) != XTBLOOM_STATUS_NOT_SUPPORTED)) {
    std::fprintf(stderr, "CPU context accepted a CUDA-only external energy operation\n");
    xtbloom_context_destroy(context);
    return 4;
  }

  CallbackState callback_state;
  if (xtbloom_context_set_external_energy_callback(context, callback, &callback_state) !=
      XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "callback setup: %s\n", xtbloom_get_last_error());
    xtbloom_context_destroy(context);
    return 4;
  }

  const int64_t atom_offsets[] = {0, 2};
  const int32_t atomic_numbers[] = {1, 1};
  const double positions[] = {0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  const double charges[] = {0.0};
  const int32_t unpaired[] = {2};
  const int32_t spin_channels[] = {2};
  double energy = 0.0;
  double forces[6] = {};
  int32_t iterations = 0;
  uint8_t converged = 0;
  int32_t status = 0;
  batch.batch_size = 1;
  batch.total_atoms = 2;
  batch.atom_offsets = input_view(atom_offsets, sizeof(atom_offsets));
  batch.atomic_numbers = input_view(atomic_numbers, sizeof(atomic_numbers));
  batch.positions = input_view(positions, sizeof(positions));
  batch.molecular_charges = input_view(charges, sizeof(charges));
  batch.unpaired_electrons = input_view(unpaired, sizeof(unpaired));
  batch.spin_channels = input_view(spin_channels, sizeof(spin_channels));
  options.flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES;
  result.energies = output_view(&energy, sizeof(energy));
  result.forces = output_view(forces, sizeof(forces));
  result.scc_iterations = output_view(&iterations, sizeof(iterations));
  result.scc_converged = output_view(&converged, sizeof(converged));
  result.per_system_status = output_view(&status, sizeof(status));

  const xtbloom_status_t compute_status = xtbloom_compute(context, &batch, &options, &result);
  if (compute_status != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "compute: %s\n", xtbloom_get_last_error());
  }
  std::printf(
      "backend=%s status=%s calls=%d potential=%d energy=%d force=%d max_spin=%d E=%.8f "
      "converged=%u\n",
      cuda_host_staged ? "cuda-host-staged" : "cpu", xtbloom_status_string(compute_status),
      callback_state.calls, callback_state.potential_calls, callback_state.energy_calls,
      callback_state.force_calls, callback_state.max_spin_channels, energy,
      static_cast<unsigned>(converged));
  callback_state.failure_mode = 1;
  status = XTBLOOM_STATUS_SUCCESS;
  const xtbloom_status_t callback_failure_status =
      xtbloom_compute(context, &batch, &options, &result);
  const bool callback_failure_propagated =
      callback_failure_status != XTBLOOM_STATUS_SUCCESS || status != XTBLOOM_STATUS_SUCCESS;
  callback_state.failure_mode = 2;
  status = XTBLOOM_STATUS_SUCCESS;
  const xtbloom_status_t nonfinite_energy_status =
      xtbloom_compute(context, &batch, &options, &result);
  const bool nonfinite_energy_rejected =
      nonfinite_energy_status != XTBLOOM_STATUS_SUCCESS || status != XTBLOOM_STATUS_SUCCESS;
  callback_state.failure_mode = 0;
  /* Clear and reinstall after the first solve so the cached-system update path
   * is covered in addition to the pre-plan callback installation path. */
  const bool callback_reinstall_ok =
      xtbloom_context_set_external_energy_callback(context, nullptr, nullptr) ==
          XTBLOOM_STATUS_SUCCESS &&
      xtbloom_context_set_external_energy_callback(context, callback, &callback_state) ==
          XTBLOOM_STATUS_SUCCESS;
  xtbloom_context_destroy(context);
  return callback_failure_propagated && nonfinite_energy_rejected && callback_reinstall_ok &&
                 compute_status == XTBLOOM_STATUS_SUCCESS && callback_state.potential_calls > 0 &&
                 callback_state.energy_calls > 0 && callback_state.force_calls > 0 &&
                 callback_state.max_spin_channels == 2
             ? 0
             : 5;
}
