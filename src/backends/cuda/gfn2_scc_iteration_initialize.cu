#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include "backends/cuda/gfn2_scc_iteration_initialize.cuh"

namespace gpuxtb::detail::cuda {
namespace {

using Diagnostic = Gfn2SccIterationInitializationDiagnostic;
using Error = Gfn2SccIterationInitializationError;
using Field = Gfn2SccIterationInitializationField;
using Mode = Gfn2SccIterationInitializationMode;

constexpr std::uint32_t component_bit(Gfn2SccPotentialComponent component) noexcept {
  return static_cast<std::uint32_t>(component);
}

constexpr std::uint32_t kMandatoryComponents = component_bit(Gfn2SccPotentialComponent::kES2) |
                                               component_bit(Gfn2SccPotentialComponent::kES3) |
                                               component_bit(Gfn2SccPotentialComponent::kAES2);

Diagnostic fail(Error error, Field field, std::int64_t index = -1, std::size_t required = 0u,
                std::size_t provided = 0u, cudaError_t cuda_status = cudaSuccess) noexcept {
  Diagnostic diagnostic{};
  diagnostic.error = error;
  diagnostic.field = field;
  diagnostic.index = index;
  diagnostic.required_bytes = required;
  diagnostic.provided_bytes = provided;
  diagnostic.cuda_status = cuda_status;
  if (error == Error::kAllocationFailed) {
    diagnostic.status = GPUXTB_STATUS_ALLOCATION_FAILED;
  } else if (error == Error::kCudaError) {
    diagnostic.status = GPUXTB_STATUS_INTERNAL_ERROR;
  } else {
    diagnostic.status = GPUXTB_STATUS_INVALID_ARGUMENT;
  }
  return diagnostic;
}

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t& result) noexcept {
  if (first < 0 || second < 0 ||
      (first != 0 && second > std::numeric_limits<std::int64_t>::max() / first)) {
    return false;
  }
  result = first * second;
  return true;
}

bool checked_add(std::int64_t first, std::int64_t second, std::int64_t& result) noexcept {
  if (first < 0 || second < 0 || first > std::numeric_limits<std::int64_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

struct Shape {
  std::int64_t batch = 0;
  std::int64_t atoms = 0;
  std::int64_t shells = 0;
  std::int64_t orbitals = 0;
  std::int64_t matrices = 0;
  std::int64_t dipoles = 0;
  std::int64_t quadrupoles = 0;
  std::int64_t two_batch = 0;
  std::int64_t two_orbitals = 0;
  std::int64_t mixer_vector = 0;
  std::int64_t history = 0;
  std::int64_t history_elements = 0;
  std::int64_t omega_elements = 0;
  std::uint32_t components = 0u;
  std::uint64_t token = 0u;
};

bool derive_shape(const Gfn2SccIterationDevicePlan& plan, Shape& shape) noexcept {
  shape = {};
  if (plan.abi_version != kGfn2SccIterationAbiVersion || plan.plan_token == 0u ||
      plan.topology.plan_token != plan.plan_token || plan.topology.batch_size <= 0 ||
      plan.topology.total_atoms <= 0 || plan.topology.total_shells <= 0 ||
      plan.topology.total_orbitals <= 0 || plan.topology.total_matrix_elements <= 0 ||
      plan.mixer_policy.history_size <= 0 ||
      (plan.enabled_components & kMandatoryComponents) != kMandatoryComponents ||
      (plan.enabled_components & ~kGfn2SccPotentialAllComponents) != 0u) {
    return false;
  }
  shape.batch = plan.topology.batch_size;
  shape.atoms = plan.topology.total_atoms;
  shape.shells = plan.topology.total_shells;
  shape.orbitals = plan.topology.total_orbitals;
  shape.matrices = plan.topology.total_matrix_elements;
  shape.history = plan.mixer_policy.history_size;
  shape.components = plan.enabled_components;
  shape.token = plan.plan_token;
  return checked_multiply(shape.atoms, 3, shape.dipoles) &&
         checked_multiply(shape.atoms, 6, shape.quadrupoles) &&
         checked_multiply(shape.batch, 2, shape.two_batch) &&
         checked_multiply(shape.orbitals, 2, shape.two_orbitals) &&
         checked_multiply(shape.atoms, 9, shape.mixer_vector) &&
         checked_add(shape.shells, shape.mixer_vector, shape.mixer_vector) &&
         checked_multiply(shape.mixer_vector, shape.history, shape.history_elements) &&
         checked_multiply(shape.batch, shape.history, shape.omega_elements);
}

bool same_requirements(const Gfn2SccIterationArenaRequirements& first,
                       const Gfn2SccIterationArenaRequirements& second) noexcept {
  return first.abi_version == second.abi_version && first.reserved == second.reserved &&
         first.alignment == second.alignment &&
         first.persistent_offset == second.persistent_offset &&
         first.persistent_bytes == second.persistent_bytes &&
         first.workspace_offset == second.workspace_offset &&
         first.workspace_bytes == second.workspace_bytes &&
         first.provider_device_offset == second.provider_device_offset &&
         first.provider_device_bytes == second.provider_device_bytes &&
         first.total_bytes == second.total_bytes && first.plan_token == second.plan_token &&
         first.layout_fingerprint == second.layout_fingerprint;
}

template <typename T>
bool empty(const Gfn2SccIterationHostArrayView<T>& view) noexcept {
  return view.data == nullptr && view.elements == 0;
}

template <typename T>
bool exact_host(const Gfn2SccIterationHostArrayView<T>& view, std::int64_t expected) noexcept {
  return expected > 0 && view.elements == expected && view.data != nullptr;
}

bool finite_array(const Gfn2SccIterationHostArrayView<double>& view) noexcept {
  for (std::int64_t index = 0; index < view.elements; ++index) {
    if (!std::isfinite(view.data[index])) return false;
  }
  return true;
}

bool valid_status(gpuxtb_status_t status) noexcept {
  return status >= GPUXTB_STATUS_SUCCESS && status <= GPUXTB_STATUS_EIGENSOLVER_FAILED;
}

bool enabled(const Shape& shape, Gfn2SccPotentialComponent component) noexcept {
  return (shape.components & component_bit(component)) != 0u;
}

bool valid_offsets(const Gfn2SccIterationHostArrayView<std::int64_t>& offsets,
                   std::int64_t partitions, std::int64_t endpoint) noexcept {
  if (partitions <= 0 || endpoint <= 0 || partitions == std::numeric_limits<std::int64_t>::max() ||
      !exact_host(offsets, partitions + 1) || offsets.data[0] != 0 ||
      offsets.data[partitions] != endpoint) {
    return false;
  }
  for (std::int64_t index = 0; index < partitions; ++index) {
    if (offsets.data[index] < 0 || offsets.data[index] > offsets.data[index + 1] ||
        offsets.data[index + 1] > endpoint) {
      return false;
    }
  }
  return true;
}

bool warm_only_wavefunction_empty(const Gfn2SccIterationHostWavefunctionView& value) noexcept {
  return empty(value.eigenvalues) && empty(value.coefficients) && empty(value.occupations) &&
         empty(value.chemical_potentials) && empty(value.electron_sums) &&
         empty(value.occupation_entropies) && empty(value.density) &&
         empty(value.energy_weighted_density) && empty(value.band_energies) &&
         empty(value.occupation_sums) && empty(value.density_traces) &&
         empty(value.weighted_density_traces);
}

bool energy_empty(const Gfn2SccIterationHostEnergyView& value) noexcept {
  return empty(value.core) && empty(value.es2) && empty(value.es3) && empty(value.aes2) &&
         empty(value.d4_two_body) && empty(value.explicit_point_charge) &&
         empty(value.periodic_embedding) && empty(value.entropy) && empty(value.internal_energy) &&
         empty(value.free_energy) && empty(value.classical_total);
}

bool mixer_empty(const Gfn2SccIterationHostMixerView& value) noexcept {
  return empty(value.current_inputs) && empty(value.previous_inputs) &&
         empty(value.previous_residuals) && empty(value.df_history) && empty(value.u_history) &&
         empty(value.omega) && empty(value.residual_rms) && empty(value.residual_maximum) &&
         empty(value.iterations) && empty(value.restart_counts) && empty(value.system_statuses) &&
         empty(value.initialized) && empty(value.residual_converged);
}

bool trace_empty(const Gfn2SccIterationHostTraceView& value) noexcept {
  return empty(value.current_shell_charges) && empty(value.current_atomic_dipoles) &&
         empty(value.current_atomic_quadrupoles) && empty(value.free_energies) &&
         empty(value.previous_free_energies) && empty(value.free_energy_changes) &&
         empty(value.residual_rms) && empty(value.iterations) && empty(value.system_statuses) &&
         empty(value.converged);
}

Diagnostic validate_common_host(const Shape& shape,
                                const Gfn2SccIterationHostInitialization& host) noexcept {
  if (host.abi_version != kGfn2SccIterationInitializationAbiVersion ||
      (host.mode != Mode::kFresh && host.mode != Mode::kWarm) ||
      host.initialization_generation == 0u) {
    return fail(Error::kInvalidPlan, Field::kPlan);
  }
  if (host.plan_token != shape.token || host.topology.plan_token != shape.token ||
      host.wavefunction.plan_token != shape.token ||
      host.wavefunction.population.plan_token != shape.token ||
      (host.mode == Mode::kWarm &&
       (host.energy.plan_token != shape.token || host.mixer.plan_token != shape.token ||
        host.scc.plan_token != shape.token))) {
    return fail(Error::kCrossPlan, Field::kPlan);
  }
  if (!valid_offsets(host.topology.atom_offsets, shape.batch, shape.atoms) ||
      !valid_offsets(host.topology.shell_offsets, shape.batch, shape.shells)) {
    return fail(Error::kInvalidExtent, Field::kTopology);
  }
  const auto& population = host.wavefunction.population;
  if (!exact_host(population.shell_charges, shape.shells) ||
      !exact_host(population.atomic_charges, shape.atoms) ||
      !exact_host(population.atomic_dipoles, shape.dipoles) ||
      !exact_host(population.atomic_quadrupoles, shape.quadrupoles)) {
    return fail(Error::kInvalidExtent, Field::kPopulation);
  }
  if (!finite_array(population.shell_charges) || !finite_array(population.atomic_charges) ||
      !finite_array(population.atomic_dipoles) || !finite_array(population.atomic_quadrupoles)) {
    return fail(Error::kNonfiniteValue, Field::kPopulation);
  }
  return {};
}

Diagnostic validate_fresh(const Shape& shape,
                          const Gfn2SccIterationHostInitialization& host) noexcept {
  (void)shape;
  if (!warm_only_wavefunction_empty(host.wavefunction) || !energy_empty(host.energy) ||
      !mixer_empty(host.mixer) || !trace_empty(host.scc) || host.energy.plan_token != 0u ||
      host.mixer.plan_token != 0u || host.scc.plan_token != 0u) {
    return fail(Error::kInvalidFreshState, Field::kWavefunction);
  }
  return {};
}

Diagnostic validate_warm_extents(const Shape& shape,
                                 const Gfn2SccIterationHostInitialization& host) noexcept {
  const auto& wave = host.wavefunction;
  if (!exact_host(wave.eigenvalues, shape.orbitals) ||
      !exact_host(wave.coefficients, shape.matrices) ||
      !exact_host(wave.occupations, shape.two_orbitals) ||
      !exact_host(wave.chemical_potentials, shape.two_batch) ||
      !exact_host(wave.electron_sums, shape.two_batch) ||
      !exact_host(wave.occupation_entropies, shape.batch) ||
      !exact_host(wave.density, shape.matrices) ||
      !exact_host(wave.energy_weighted_density, shape.matrices) ||
      !exact_host(wave.band_energies, shape.batch) ||
      !exact_host(wave.occupation_sums, shape.batch) ||
      !exact_host(wave.density_traces, shape.batch) ||
      !exact_host(wave.weighted_density_traces, shape.batch)) {
    return fail(Error::kInvalidExtent, Field::kWavefunction);
  }
  const auto finite_wave =
      finite_array(wave.eigenvalues) && finite_array(wave.coefficients) &&
      finite_array(wave.occupations) && finite_array(wave.chemical_potentials) &&
      finite_array(wave.electron_sums) && finite_array(wave.occupation_entropies) &&
      finite_array(wave.density) && finite_array(wave.energy_weighted_density) &&
      finite_array(wave.band_energies) && finite_array(wave.occupation_sums) &&
      finite_array(wave.density_traces) && finite_array(wave.weighted_density_traces);
  if (!finite_wave) return fail(Error::kNonfiniteValue, Field::kWavefunction);

  const auto& energy = host.energy;
  const auto optional = [&](const Gfn2SccIterationHostArrayView<double>& values,
                            Gfn2SccPotentialComponent component) {
    return enabled(shape, component) ? exact_host(values, shape.batch) : empty(values);
  };
  if (!exact_host(energy.core, shape.batch) || !exact_host(energy.es2, shape.batch) ||
      !exact_host(energy.es3, shape.batch) || !exact_host(energy.aes2, shape.batch) ||
      !optional(energy.d4_two_body, Gfn2SccPotentialComponent::kD4TwoBody) ||
      !optional(energy.explicit_point_charge, Gfn2SccPotentialComponent::kExplicitPointCharge) ||
      !optional(energy.periodic_embedding, Gfn2SccPotentialComponent::kPeriodicEmbedding) ||
      !exact_host(energy.entropy, shape.batch) ||
      !exact_host(energy.internal_energy, shape.batch) ||
      !exact_host(energy.free_energy, shape.batch) ||
      !exact_host(energy.classical_total, shape.batch)) {
    return fail(Error::kInvalidExtent, Field::kEnergy);
  }
  if (!finite_array(energy.core) || !finite_array(energy.es2) || !finite_array(energy.es3) ||
      !finite_array(energy.aes2) ||
      (enabled(shape, Gfn2SccPotentialComponent::kD4TwoBody) &&
       !finite_array(energy.d4_two_body)) ||
      (enabled(shape, Gfn2SccPotentialComponent::kExplicitPointCharge) &&
       !finite_array(energy.explicit_point_charge)) ||
      (enabled(shape, Gfn2SccPotentialComponent::kPeriodicEmbedding) &&
       !finite_array(energy.periodic_embedding)) ||
      !finite_array(energy.entropy) || !finite_array(energy.internal_energy) ||
      !finite_array(energy.free_energy) || !finite_array(energy.classical_total)) {
    return fail(Error::kNonfiniteValue, Field::kEnergy);
  }

  const auto& mixer = host.mixer;
  if (!exact_host(mixer.current_inputs, shape.mixer_vector) ||
      !exact_host(mixer.previous_inputs, shape.mixer_vector) ||
      !exact_host(mixer.previous_residuals, shape.mixer_vector) ||
      !exact_host(mixer.df_history, shape.history_elements) ||
      !exact_host(mixer.u_history, shape.history_elements) ||
      !exact_host(mixer.omega, shape.omega_elements) ||
      !exact_host(mixer.residual_rms, shape.batch) ||
      !exact_host(mixer.residual_maximum, shape.batch) ||
      !exact_host(mixer.iterations, shape.batch) ||
      !exact_host(mixer.restart_counts, shape.batch) ||
      !exact_host(mixer.system_statuses, shape.batch) ||
      !exact_host(mixer.initialized, shape.batch) ||
      !exact_host(mixer.residual_converged, shape.batch)) {
    return fail(Error::kInvalidExtent, Field::kMixer);
  }
  if (!finite_array(mixer.current_inputs) || !finite_array(mixer.previous_inputs) ||
      !finite_array(mixer.previous_residuals) || !finite_array(mixer.df_history) ||
      !finite_array(mixer.u_history) || !finite_array(mixer.omega) ||
      !finite_array(mixer.residual_rms) || !finite_array(mixer.residual_maximum)) {
    return fail(Error::kNonfiniteValue, Field::kMixer);
  }

  const auto& scc = host.scc;
  if (!exact_host(scc.current_shell_charges, shape.shells) ||
      !exact_host(scc.current_atomic_dipoles, shape.dipoles) ||
      !exact_host(scc.current_atomic_quadrupoles, shape.quadrupoles) ||
      !exact_host(scc.free_energies, shape.batch) ||
      !exact_host(scc.previous_free_energies, shape.batch) ||
      !exact_host(scc.free_energy_changes, shape.batch) ||
      !exact_host(scc.residual_rms, shape.batch) || !exact_host(scc.iterations, shape.batch) ||
      !exact_host(scc.system_statuses, shape.batch) || !exact_host(scc.converged, shape.batch)) {
    return fail(Error::kInvalidExtent, Field::kSccTrace);
  }
  if (!finite_array(scc.current_shell_charges) || !finite_array(scc.current_atomic_dipoles) ||
      !finite_array(scc.current_atomic_quadrupoles) || !finite_array(scc.free_energies) ||
      !finite_array(scc.previous_free_energies) || !finite_array(scc.free_energy_changes) ||
      !finite_array(scc.residual_rms)) {
    return fail(Error::kNonfiniteValue, Field::kSccTrace);
  }
  return {};
}

Diagnostic validate_warm_invariants(const Shape& shape,
                                    const Gfn2SccIterationHostInitialization& host,
                                    const Gfn2SccIterationDevicePlan& plan) noexcept {
  const auto& wave = host.wavefunction;
  const auto& mixer = host.mixer;
  const auto& scc = host.scc;
  for (std::int64_t system = 0; system < shape.batch; ++system) {
    const gpuxtb_status_t mixer_status = mixer.system_statuses.data[system];
    const gpuxtb_status_t scc_status = scc.system_statuses.data[system];
    const std::uint64_t iteration = scc.iterations.data[system];
    const bool terminal_nonconvergence =
        scc.converged.data[system] == 0u && iteration == plan.state_policy.maximum_iterations;
    const bool statuses_match_publication =
        terminal_nonconvergence
            ? mixer_status == GPUXTB_STATUS_SUCCESS && scc_status == GPUXTB_STATUS_SCC_NOT_CONVERGED
            : mixer_status == scc_status && scc_status != GPUXTB_STATUS_SCC_NOT_CONVERGED;
    if (mixer.initialized.data[system] != 1u || mixer.residual_converged.data[system] > 1u ||
        scc.converged.data[system] > 1u || !valid_status(mixer_status) ||
        !valid_status(scc_status) || !statuses_match_publication ||
        mixer.iterations.data[system] != scc.iterations.data[system] ||
        mixer.iterations.data[system] > plan.state_policy.maximum_iterations ||
        mixer.residual_rms.data[system] != scc.residual_rms.data[system] ||
        scc.free_energies.data[system] != host.energy.free_energy.data[system] ||
        scc.free_energy_changes.data[system] !=
            scc.free_energies.data[system] - scc.previous_free_energies.data[system] ||
        host.energy.entropy.data[system] != wave.occupation_entropies.data[system] ||
        (scc.converged.data[system] == 1u &&
         scc.system_statuses.data[system] != GPUXTB_STATUS_SUCCESS)) {
      return fail(Error::kInvalidWarmState, Field::kSccTrace, system);
    }

    const std::int64_t atom_begin = host.topology.atom_offsets.data[system];
    const std::int64_t atom_end = host.topology.atom_offsets.data[system + 1];
    const std::int64_t shell_begin = host.topology.shell_offsets.data[system];
    const std::int64_t shell_end = host.topology.shell_offsets.data[system + 1];
    const std::int64_t atoms = atom_end - atom_begin;
    const std::int64_t shells = shell_end - shell_begin;
    const std::int64_t vector_begin = shell_begin + 9 * atom_begin;
    const double* const packed = mixer.current_inputs.data + vector_begin;
    for (std::int64_t shell = 0; shell < shells; ++shell) {
      const double current = scc.current_shell_charges.data[shell_begin + shell];
      if (packed[shell] != current ||
          (scc.converged.data[system] == 0u &&
           wave.population.shell_charges.data[shell_begin + shell] != current)) {
        return fail(Error::kInvalidWarmState, Field::kMixer, system);
      }
    }
    const std::int64_t dipole_base = shells;
    const std::int64_t quadrupole_base = shells + 3 * atoms;
    for (std::int64_t atom = 0; atom < atoms; ++atom) {
      for (std::int64_t component = 0; component < 3; ++component) {
        const std::int64_t global = 3 * (atom_begin + atom) + component;
        const double current = scc.current_atomic_dipoles.data[global];
        if (packed[dipole_base + 3 * atom + component] != current ||
            (scc.converged.data[system] == 0u &&
             wave.population.atomic_dipoles.data[global] != current)) {
          return fail(Error::kInvalidWarmState, Field::kMixer, system);
        }
      }
      for (std::int64_t component = 0; component < 6; ++component) {
        const std::int64_t global = 6 * (atom_begin + atom) + component;
        const double current = scc.current_atomic_quadrupoles.data[global];
        if (packed[quadrupole_base + 6 * atom + component] != current ||
            (scc.converged.data[system] == 0u &&
             wave.population.atomic_quadrupoles.data[global] != current)) {
          return fail(Error::kInvalidWarmState, Field::kMixer, system);
        }
      }
    }
  }
  return {};
}

class ImageBuilder {
 public:
  ImageBuilder(void* device_arena, std::size_t arena_bytes, void* image) noexcept
      : arena_(reinterpret_cast<std::uintptr_t>(device_arena)),
        arena_bytes_(arena_bytes),
        image_(static_cast<std::byte*>(image)) {}

  [[nodiscard]] bool writes_image() const noexcept { return image_ != nullptr; }

  template <typename T>
  T* target(T* pointer, std::int64_t actual, std::int64_t expected, Field field, std::int64_t index,
            Diagnostic& diagnostic) const noexcept {
    if (actual != expected || expected <= 0 || pointer == nullptr) {
      diagnostic = fail(Error::kInvalidExtent, field, index);
      return nullptr;
    }
    const auto elements = static_cast<std::uint64_t>(expected);
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      diagnostic = fail(Error::kCountOverflow, field, index);
      return nullptr;
    }
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(T);
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (address % alignof(T) != 0u || address < arena_ || address - arena_ > arena_bytes_ ||
        bytes > arena_bytes_ - (address - arena_)) {
      diagnostic = fail(Error::kInvalidArena, field, index, bytes, arena_bytes_);
      return nullptr;
    }
    if (image_ == nullptr) return pointer;
    return reinterpret_cast<T*>(image_ + (address - arena_));
  }

  template <typename T>
  bool copy(T* destination, std::int64_t actual, const Gfn2SccIterationHostArrayView<T>& source,
            std::int64_t expected, Field field, std::int64_t index,
            Diagnostic& diagnostic) const noexcept {
    T* const output = target(destination, actual, expected, field, index, diagnostic);
    if (output == nullptr || !exact_host(source, expected)) {
      if (diagnostic.success()) diagnostic = fail(Error::kInvalidExtent, field, index);
      return false;
    }
    if (image_ != nullptr) {
      std::memcpy(output, source.data, static_cast<std::size_t>(expected) * sizeof(T));
    }
    return true;
  }

  template <typename T>
  bool fill(T* destination, std::int64_t actual, std::int64_t expected, T value, Field field,
            std::int64_t index, Diagnostic& diagnostic) const noexcept {
    T* const output = target(destination, actual, expected, field, index, diagnostic);
    if (output == nullptr) return false;
    if (image_ != nullptr) {
      for (std::int64_t element = 0; element < expected; ++element) output[element] = value;
    }
    return true;
  }

 private:
  std::uintptr_t arena_ = 0u;
  std::size_t arena_bytes_ = 0u;
  std::byte* image_ = nullptr;
};

bool valid_state_tokens(const Gfn2SccIterationDeviceState& state, std::uint64_t token) noexcept {
  return state.plan_token == token && state.eigenpairs.plan_token == token &&
         state.occupations.plan_token == token && state.density.plan_token == token &&
         state.raw_population.plan_token == token && state.classical_energy.plan_token == token &&
         state.free_energy.plan_token == token && state.mixer.plan_token == token &&
         state.published.plan_token == token && state.scc.plan_token == token &&
         state.publication.plan_token == token;
}

Diagnostic project_state_image(const Shape& shape, const Gfn2SccIterationDevicePlan& plan,
                               const Gfn2SccIterationDeviceState& state,
                               const Gfn2SccIterationDeviceWorkspace& workspace,
                               const Gfn2SccIterationReportStorage& reports,
                               const Gfn2SccIterationHostInitialization& host,
                               const ImageBuilder& builder) noexcept {
  if (!valid_state_tokens(state, shape.token)) return fail(Error::kCrossPlan, Field::kWavefunction);
  if (workspace.plan_token != shape.token || workspace.ledger.plan_token != shape.token) {
    return fail(Error::kCrossPlan, Field::kWorkspace);
  }
  if (reports.plan_token != shape.token) return fail(Error::kCrossPlan, Field::kReportStorage);

  Gfn2SccIterationReportStorageRequirements report_requirements{};
  const auto report_diagnostic = query_gfn2_scc_iteration_report_storage_cuda(
      shape.components, shape.batch, report_requirements);
  if (report_diagnostic.error != Gfn2SccIterationBindingError::kSuccess ||
      reports.system_error_elements != report_requirements.system_error_elements ||
      reports.device_error_elements != report_requirements.device_error_elements ||
      reports.sequence_latch_elements != report_requirements.sequence_latch_elements) {
    return fail(Error::kInvalidExtent, Field::kReportStorage);
  }

  Diagnostic diagnostic{};
  const auto& wave = host.wavefunction;
  const auto& population = wave.population;
  if (!builder.target(state.eigenpairs.eigenvalues, state.eigenpairs.eigenvalue_elements,
                      shape.orbitals, Field::kWavefunction, 0, diagnostic) ||
      !builder.target(state.eigenpairs.coefficients, state.eigenpairs.coefficient_elements,
                      shape.matrices, Field::kWavefunction, 1, diagnostic) ||
      !builder.target(state.occupations.occupations, state.occupations.occupation_elements,
                      shape.two_orbitals, Field::kWavefunction, 2, diagnostic) ||
      !builder.target(state.occupations.chemical_potentials,
                      state.occupations.chemical_potential_elements, shape.two_batch,
                      Field::kWavefunction, 3, diagnostic) ||
      !builder.target(state.occupations.electron_sums, state.occupations.electron_sum_elements,
                      shape.two_batch, Field::kWavefunction, 4, diagnostic) ||
      !builder.target(state.occupations.entropies, state.occupations.entropy_elements, shape.batch,
                      Field::kWavefunction, 5, diagnostic) ||
      !builder.target(state.density.density, state.density.density_elements, shape.matrices,
                      Field::kWavefunction, 6, diagnostic) ||
      !builder.target(state.density.energy_weighted_density,
                      state.density.weighted_density_elements, shape.matrices, Field::kWavefunction,
                      7, diagnostic) ||
      !builder.target(state.density.band_energies, state.density.band_energy_elements, shape.batch,
                      Field::kWavefunction, 8, diagnostic) ||
      !builder.target(state.density.occupation_sums, state.density.occupation_sum_elements,
                      shape.batch, Field::kWavefunction, 9, diagnostic) ||
      !builder.target(state.density.density_traces, state.density.density_trace_elements,
                      shape.batch, Field::kWavefunction, 10, diagnostic) ||
      !builder.target(state.density.weighted_density_traces,
                      state.density.weighted_density_trace_elements, shape.batch,
                      Field::kWavefunction, 11, diagnostic) ||
      !builder.copy(state.raw_population.qsh, state.raw_population.qsh_elements,
                    population.shell_charges, shape.shells, Field::kPopulation, 0, diagnostic) ||
      !builder.copy(state.raw_population.qat, state.raw_population.qat_elements,
                    population.atomic_charges, shape.atoms, Field::kPopulation, 1, diagnostic) ||
      !builder.copy(state.raw_population.dipole, state.raw_population.dipole_elements,
                    population.atomic_dipoles, shape.dipoles, Field::kPopulation, 2, diagnostic) ||
      !builder.copy(state.raw_population.quadrupole, state.raw_population.quadrupole_elements,
                    population.atomic_quadrupoles, shape.quadrupoles, Field::kPopulation, 3,
                    diagnostic)) {
    return diagnostic;
  }

  if (host.mode == Mode::kWarm) {
    if (!builder.copy(state.eigenpairs.eigenvalues, state.eigenpairs.eigenvalue_elements,
                      wave.eigenvalues, shape.orbitals, Field::kWavefunction, 0, diagnostic) ||
        !builder.copy(state.eigenpairs.coefficients, state.eigenpairs.coefficient_elements,
                      wave.coefficients, shape.matrices, Field::kWavefunction, 1, diagnostic) ||
        !builder.copy(state.occupations.occupations, state.occupations.occupation_elements,
                      wave.occupations, shape.two_orbitals, Field::kWavefunction, 2, diagnostic) ||
        !builder.copy(state.occupations.chemical_potentials,
                      state.occupations.chemical_potential_elements, wave.chemical_potentials,
                      shape.two_batch, Field::kWavefunction, 3, diagnostic) ||
        !builder.copy(state.occupations.electron_sums, state.occupations.electron_sum_elements,
                      wave.electron_sums, shape.two_batch, Field::kWavefunction, 4, diagnostic) ||
        !builder.copy(state.occupations.entropies, state.occupations.entropy_elements,
                      wave.occupation_entropies, shape.batch, Field::kWavefunction, 5,
                      diagnostic) ||
        !builder.copy(state.density.density, state.density.density_elements, wave.density,
                      shape.matrices, Field::kWavefunction, 6, diagnostic) ||
        !builder.copy(state.density.energy_weighted_density,
                      state.density.weighted_density_elements, wave.energy_weighted_density,
                      shape.matrices, Field::kWavefunction, 7, diagnostic) ||
        !builder.copy(state.density.band_energies, state.density.band_energy_elements,
                      wave.band_energies, shape.batch, Field::kWavefunction, 8, diagnostic) ||
        !builder.copy(state.density.occupation_sums, state.density.occupation_sum_elements,
                      wave.occupation_sums, shape.batch, Field::kWavefunction, 9, diagnostic) ||
        !builder.copy(state.density.density_traces, state.density.density_trace_elements,
                      wave.density_traces, shape.batch, Field::kWavefunction, 10, diagnostic) ||
        !builder.copy(state.density.weighted_density_traces,
                      state.density.weighted_density_trace_elements, wave.weighted_density_traces,
                      shape.batch, Field::kWavefunction, 11, diagnostic)) {
      return diagnostic;
    }
  } else {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!builder.fill(state.occupations.chemical_potentials,
                      state.occupations.chemical_potential_elements, shape.two_batch, nan,
                      Field::kWavefunction, 3, diagnostic) ||
        !builder.fill(state.occupations.electron_sums, state.occupations.electron_sum_elements,
                      shape.two_batch, nan, Field::kWavefunction, 4, diagnostic) ||
        !builder.fill(state.occupations.entropies, state.occupations.entropy_elements, shape.batch,
                      nan, Field::kWavefunction, 5, diagnostic) ||
        !builder.fill(state.density.band_energies, state.density.band_energy_elements, shape.batch,
                      nan, Field::kWavefunction, 8, diagnostic) ||
        !builder.fill(state.density.occupation_sums, state.density.occupation_sum_elements,
                      shape.batch, nan, Field::kWavefunction, 9, diagnostic) ||
        !builder.fill(state.density.density_traces, state.density.density_trace_elements,
                      shape.batch, nan, Field::kWavefunction, 10, diagnostic) ||
        !builder.fill(state.density.weighted_density_traces,
                      state.density.weighted_density_trace_elements, shape.batch, nan,
                      Field::kWavefunction, 11, diagnostic)) {
      return diagnostic;
    }
  }

  const auto copy_energy = [&](double* destination, std::int64_t actual,
                               const Gfn2SccIterationHostArrayView<double>& source,
                               std::int64_t index) {
    return builder.copy(destination, actual, source, shape.batch, Field::kEnergy, index,
                        diagnostic);
  };
  const auto fill_energy = [&](double* destination, std::int64_t actual, double value,
                               std::int64_t index) {
    return builder.fill(destination, actual, shape.batch, value, Field::kEnergy, index, diagnostic);
  };
  const auto& energy = host.energy;
  if (host.mode == Mode::kWarm) {
    if (!copy_energy(state.free_energy.core, state.free_energy.core_elements, energy.core, 0) ||
        !copy_energy(state.free_energy.es2, state.free_energy.es2_elements, energy.es2, 1) ||
        !copy_energy(state.free_energy.es3, state.free_energy.es3_elements, energy.es3, 2) ||
        !copy_energy(state.free_energy.aes2, state.free_energy.aes2_elements, energy.aes2, 3) ||
        (enabled(shape, Gfn2SccPotentialComponent::kD4TwoBody) &&
         !copy_energy(state.free_energy.d4_two_body, state.free_energy.d4_two_body_elements,
                      energy.d4_two_body, 4)) ||
        (enabled(shape, Gfn2SccPotentialComponent::kExplicitPointCharge) &&
         !copy_energy(state.free_energy.explicit_point_charge,
                      state.free_energy.explicit_point_charge_elements,
                      energy.explicit_point_charge, 5)) ||
        (enabled(shape, Gfn2SccPotentialComponent::kPeriodicEmbedding) &&
         !copy_energy(state.free_energy.periodic_embedding,
                      state.free_energy.periodic_embedding_elements, energy.periodic_embedding,
                      6)) ||
        !copy_energy(state.free_energy.entropy, state.free_energy.entropy_elements, energy.entropy,
                     7) ||
        !copy_energy(state.free_energy.internal_energy, state.free_energy.internal_energy_elements,
                     energy.internal_energy, 8) ||
        !copy_energy(state.free_energy.free_energy, state.free_energy.free_energy_elements,
                     energy.free_energy, 9) ||
        !copy_energy(state.classical_energy.classical_total,
                     state.classical_energy.classical_total_elements, energy.classical_total, 10)) {
      return diagnostic;
    }
    if (!enabled(shape, Gfn2SccPotentialComponent::kD4TwoBody) &&
        !fill_energy(state.free_energy.d4_two_body, state.free_energy.d4_two_body_elements, 0.0,
                     4)) {
      return diagnostic;
    }
    if (!enabled(shape, Gfn2SccPotentialComponent::kExplicitPointCharge) &&
        !fill_energy(state.free_energy.explicit_point_charge,
                     state.free_energy.explicit_point_charge_elements, 0.0, 5)) {
      return diagnostic;
    }
    if (!enabled(shape, Gfn2SccPotentialComponent::kPeriodicEmbedding) &&
        !fill_energy(state.free_energy.periodic_embedding,
                     state.free_energy.periodic_embedding_elements, 0.0, 6)) {
      return diagnostic;
    }
  } else {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!fill_energy(state.free_energy.core, state.free_energy.core_elements, nan, 0) ||
        !fill_energy(state.free_energy.es2, state.free_energy.es2_elements, nan, 1) ||
        !fill_energy(state.free_energy.es3, state.free_energy.es3_elements, nan, 2) ||
        !fill_energy(state.free_energy.aes2, state.free_energy.aes2_elements, nan, 3) ||
        !fill_energy(state.free_energy.d4_two_body, state.free_energy.d4_two_body_elements,
                     enabled(shape, Gfn2SccPotentialComponent::kD4TwoBody) ? nan : 0.0, 4) ||
        !fill_energy(state.free_energy.explicit_point_charge,
                     state.free_energy.explicit_point_charge_elements,
                     enabled(shape, Gfn2SccPotentialComponent::kExplicitPointCharge) ? nan : 0.0,
                     5) ||
        !fill_energy(
            state.free_energy.periodic_embedding, state.free_energy.periodic_embedding_elements,
            enabled(shape, Gfn2SccPotentialComponent::kPeriodicEmbedding) ? nan : 0.0, 6) ||
        !fill_energy(state.free_energy.entropy, state.free_energy.entropy_elements, nan, 7) ||
        !fill_energy(state.free_energy.internal_energy, state.free_energy.internal_energy_elements,
                     nan, 8) ||
        !fill_energy(state.free_energy.free_energy, state.free_energy.free_energy_elements, nan,
                     9) ||
        !fill_energy(state.classical_energy.classical_total,
                     state.classical_energy.classical_total_elements, nan, 10)) {
      return diagnostic;
    }
  }

  const auto& mixer = host.mixer;
  if (!builder.target(state.mixer.current_inputs, state.mixer.total_vector_elements,
                      shape.mixer_vector, Field::kMixer, 0, diagnostic) ||
      !builder.target(state.mixer.previous_inputs, state.mixer.total_vector_elements,
                      shape.mixer_vector, Field::kMixer, 1, diagnostic) ||
      !builder.target(state.mixer.previous_residuals, state.mixer.total_vector_elements,
                      shape.mixer_vector, Field::kMixer, 2, diagnostic) ||
      !builder.target(state.mixer.df_history, state.mixer.history_elements, shape.history_elements,
                      Field::kMixer, 3, diagnostic) ||
      !builder.target(state.mixer.u_history, state.mixer.history_elements, shape.history_elements,
                      Field::kMixer, 4, diagnostic) ||
      !builder.target(state.mixer.omega, state.mixer.omega_elements, shape.omega_elements,
                      Field::kMixer, 5, diagnostic) ||
      !builder.target(state.mixer.residual_rms, state.mixer.batch_elements, shape.batch,
                      Field::kMixer, 6, diagnostic) ||
      !builder.target(state.mixer.residual_maximum, state.mixer.batch_elements, shape.batch,
                      Field::kMixer, 7, diagnostic) ||
      !builder.target(state.mixer.iterations, state.mixer.batch_elements, shape.batch,
                      Field::kMixer, 8, diagnostic) ||
      !builder.target(state.mixer.restart_counts, state.mixer.batch_elements, shape.batch,
                      Field::kMixer, 9, diagnostic) ||
      !builder.target(state.mixer.system_statuses, state.mixer.batch_elements, shape.batch,
                      Field::kMixer, 10, diagnostic) ||
      !builder.target(state.mixer.initialized, state.mixer.batch_elements, shape.batch,
                      Field::kMixer, 11, diagnostic) ||
      !builder.target(state.mixer.residual_converged, state.mixer.batch_elements, shape.batch,
                      Field::kMixer, 12, diagnostic)) {
    return diagnostic;
  }

  if (host.mode == Mode::kWarm) {
    if (!builder.copy(state.mixer.current_inputs, state.mixer.total_vector_elements,
                      mixer.current_inputs, shape.mixer_vector, Field::kMixer, 0, diagnostic) ||
        !builder.copy(state.mixer.previous_inputs, state.mixer.total_vector_elements,
                      mixer.previous_inputs, shape.mixer_vector, Field::kMixer, 1, diagnostic) ||
        !builder.copy(state.mixer.previous_residuals, state.mixer.total_vector_elements,
                      mixer.previous_residuals, shape.mixer_vector, Field::kMixer, 2, diagnostic) ||
        !builder.copy(state.mixer.df_history, state.mixer.history_elements, mixer.df_history,
                      shape.history_elements, Field::kMixer, 3, diagnostic) ||
        !builder.copy(state.mixer.u_history, state.mixer.history_elements, mixer.u_history,
                      shape.history_elements, Field::kMixer, 4, diagnostic) ||
        !builder.copy(state.mixer.omega, state.mixer.omega_elements, mixer.omega,
                      shape.omega_elements, Field::kMixer, 5, diagnostic) ||
        !builder.copy(state.mixer.residual_rms, state.mixer.batch_elements, mixer.residual_rms,
                      shape.batch, Field::kMixer, 6, diagnostic) ||
        !builder.copy(state.mixer.residual_maximum, state.mixer.batch_elements,
                      mixer.residual_maximum, shape.batch, Field::kMixer, 7, diagnostic) ||
        !builder.copy(state.mixer.iterations, state.mixer.batch_elements, mixer.iterations,
                      shape.batch, Field::kMixer, 8, diagnostic) ||
        !builder.copy(state.mixer.restart_counts, state.mixer.batch_elements, mixer.restart_counts,
                      shape.batch, Field::kMixer, 9, diagnostic) ||
        !builder.copy(state.mixer.system_statuses, state.mixer.batch_elements,
                      mixer.system_statuses, shape.batch, Field::kMixer, 10, diagnostic) ||
        !builder.copy(state.mixer.initialized, state.mixer.batch_elements, mixer.initialized,
                      shape.batch, Field::kMixer, 11, diagnostic) ||
        !builder.copy(state.mixer.residual_converged, state.mixer.batch_elements,
                      mixer.residual_converged, shape.batch, Field::kMixer, 12, diagnostic)) {
      return diagnostic;
    }
  } else {
    auto* const packed =
        builder.target(state.mixer.current_inputs, state.mixer.total_vector_elements,
                       shape.mixer_vector, Field::kMixer, 0, diagnostic);
    if (packed == nullptr ||
        !builder.fill(state.mixer.initialized, state.mixer.batch_elements, shape.batch,
                      static_cast<std::uint8_t>(1u), Field::kMixer, 11, diagnostic)) {
      return diagnostic;
    }
    if (builder.writes_image()) {
      for (std::int64_t system = 0; system < shape.batch; ++system) {
        const std::int64_t atom_begin = host.topology.atom_offsets.data[system];
        const std::int64_t atom_end = host.topology.atom_offsets.data[system + 1];
        const std::int64_t shell_begin = host.topology.shell_offsets.data[system];
        const std::int64_t shell_end = host.topology.shell_offsets.data[system + 1];
        const std::int64_t atoms = atom_end - atom_begin;
        const std::int64_t shells = shell_end - shell_begin;
        const std::int64_t vector_begin = shell_begin + 9 * atom_begin;
        std::memcpy(packed + vector_begin, population.shell_charges.data + shell_begin,
                    static_cast<std::size_t>(shells) * sizeof(double));
        std::memcpy(packed + vector_begin + shells, population.atomic_dipoles.data + 3 * atom_begin,
                    static_cast<std::size_t>(3 * atoms) * sizeof(double));
        std::memcpy(packed + vector_begin + shells + 3 * atoms,
                    population.atomic_quadrupoles.data + 6 * atom_begin,
                    static_cast<std::size_t>(6 * atoms) * sizeof(double));
      }
    }
  }

  const auto& scc = host.scc;
  if (!builder.target(state.scc.current_inputs.shell_charges,
                      state.scc.current_inputs.shell_elements, shape.shells, Field::kSccTrace, 0,
                      diagnostic) ||
      !builder.target(state.scc.current_inputs.atomic_dipoles,
                      state.scc.current_inputs.dipole_elements, shape.dipoles, Field::kSccTrace, 1,
                      diagnostic) ||
      !builder.target(state.scc.current_inputs.atomic_quadrupoles,
                      state.scc.current_inputs.quadrupole_elements, shape.quadrupoles,
                      Field::kSccTrace, 2, diagnostic) ||
      !builder.target(state.scc.free_energies, state.scc.batch_elements, shape.batch,
                      Field::kSccTrace, 3, diagnostic) ||
      !builder.target(state.scc.previous_free_energies, state.scc.batch_elements, shape.batch,
                      Field::kSccTrace, 4, diagnostic) ||
      !builder.target(state.scc.free_energy_changes, state.scc.batch_elements, shape.batch,
                      Field::kSccTrace, 5, diagnostic) ||
      !builder.target(state.scc.residual_rms, state.scc.batch_elements, shape.batch,
                      Field::kSccTrace, 6, diagnostic) ||
      !builder.target(state.scc.iterations, state.scc.batch_elements, shape.batch, Field::kSccTrace,
                      7, diagnostic) ||
      !builder.target(state.scc.system_statuses, state.scc.batch_elements, shape.batch,
                      Field::kSccTrace, 8, diagnostic) ||
      !builder.target(state.scc.converged, state.scc.batch_elements, shape.batch, Field::kSccTrace,
                      9, diagnostic)) {
    return diagnostic;
  }
  if (host.mode == Mode::kWarm) {
    if (!builder.copy(state.scc.current_inputs.shell_charges,
                      state.scc.current_inputs.shell_elements, scc.current_shell_charges,
                      shape.shells, Field::kSccTrace, 0, diagnostic) ||
        !builder.copy(state.scc.current_inputs.atomic_dipoles,
                      state.scc.current_inputs.dipole_elements, scc.current_atomic_dipoles,
                      shape.dipoles, Field::kSccTrace, 1, diagnostic) ||
        !builder.copy(state.scc.current_inputs.atomic_quadrupoles,
                      state.scc.current_inputs.quadrupole_elements, scc.current_atomic_quadrupoles,
                      shape.quadrupoles, Field::kSccTrace, 2, diagnostic) ||
        !builder.copy(state.scc.free_energies, state.scc.batch_elements, scc.free_energies,
                      shape.batch, Field::kSccTrace, 3, diagnostic) ||
        !builder.copy(state.scc.previous_free_energies, state.scc.batch_elements,
                      scc.previous_free_energies, shape.batch, Field::kSccTrace, 4, diagnostic) ||
        !builder.copy(state.scc.free_energy_changes, state.scc.batch_elements,
                      scc.free_energy_changes, shape.batch, Field::kSccTrace, 5, diagnostic) ||
        !builder.copy(state.scc.residual_rms, state.scc.batch_elements, scc.residual_rms,
                      shape.batch, Field::kSccTrace, 6, diagnostic) ||
        !builder.copy(state.scc.iterations, state.scc.batch_elements, scc.iterations, shape.batch,
                      Field::kSccTrace, 7, diagnostic) ||
        !builder.copy(state.scc.system_statuses, state.scc.batch_elements, scc.system_statuses,
                      shape.batch, Field::kSccTrace, 8, diagnostic) ||
        !builder.copy(state.scc.converged, state.scc.batch_elements, scc.converged, shape.batch,
                      Field::kSccTrace, 9, diagnostic)) {
      return diagnostic;
    }
  } else {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!builder.copy(state.scc.current_inputs.shell_charges,
                      state.scc.current_inputs.shell_elements, population.shell_charges,
                      shape.shells, Field::kSccTrace, 0, diagnostic) ||
        !builder.copy(state.scc.current_inputs.atomic_dipoles,
                      state.scc.current_inputs.dipole_elements, population.atomic_dipoles,
                      shape.dipoles, Field::kSccTrace, 1, diagnostic) ||
        !builder.copy(state.scc.current_inputs.atomic_quadrupoles,
                      state.scc.current_inputs.quadrupole_elements, population.atomic_quadrupoles,
                      shape.quadrupoles, Field::kSccTrace, 2, diagnostic) ||
        !builder.fill(state.scc.free_energies, state.scc.batch_elements, shape.batch, nan,
                      Field::kSccTrace, 3, diagnostic) ||
        !builder.fill(state.scc.previous_free_energies, state.scc.batch_elements, shape.batch, nan,
                      Field::kSccTrace, 4, diagnostic) ||
        !builder.fill(state.scc.free_energy_changes, state.scc.batch_elements, shape.batch, nan,
                      Field::kSccTrace, 5, diagnostic)) {
      return diagnostic;
    }
  }

  if (workspace.ledger.batch_elements != shape.batch || workspace.ledger.scalar_elements != 1 ||
      !builder.target(workspace.ledger.active_mask, workspace.ledger.batch_elements, shape.batch,
                      Field::kWorkspace, 0, diagnostic) ||
      !builder.target(workspace.ledger.pending_statuses, workspace.ledger.batch_elements,
                      shape.batch, Field::kWorkspace, 1, diagnostic) ||
      !builder.target(workspace.ledger.system_failure_records, workspace.ledger.batch_elements,
                      shape.batch, Field::kWorkspace, 2, diagnostic) ||
      !builder.target(workspace.ledger.plan_failure_record, workspace.ledger.scalar_elements, 1,
                      Field::kWorkspace, 3, diagnostic) ||
      !builder.target(workspace.ledger.sequence_active, workspace.ledger.scalar_elements, 1,
                      Field::kWorkspace, 4, diagnostic) ||
      !builder.target(reports.system_errors, reports.system_error_elements,
                      reports.system_error_elements, Field::kReportStorage, 0, diagnostic) ||
      !builder.target(reports.device_errors, reports.device_error_elements,
                      reports.device_error_elements, Field::kReportStorage, 1, diagnostic) ||
      !builder.target(reports.sequence_latches, reports.sequence_latch_elements,
                      reports.sequence_latch_elements, Field::kReportStorage, 2, diagnostic)) {
    return diagnostic.success() ? fail(Error::kInvalidExtent, Field::kWorkspace) : diagnostic;
  }

  (void)plan;
  return {};
}

}  // namespace

struct Gfn2SccIterationInitializer::Impl {
  void* device_checkpoint = nullptr;
  void* device_arena = nullptr;
  cudaEvent_t restore_complete = nullptr;
  std::size_t image_bytes = 0u;
  std::uint64_t plan_token = 0u;
  std::uint64_t initialization_generation = 0u;
  Mode mode = Mode::kFresh;
  int device_ordinal = -1;
  bool has_tracked_restore = false;

  ~Impl() {
    int previous_device = -1;
    const bool restore_device =
        device_ordinal >= 0 && cudaGetDevice(&previous_device) == cudaSuccess &&
        previous_device != device_ordinal && cudaSetDevice(device_ordinal) == cudaSuccess;
    /* Every ordinary restore is chained through this event. Waiting for the
     * last record therefore drains source use on every submission stream
     * without escalating teardown to a device-wide synchronization. */
    if (restore_complete != nullptr) {
      if (has_tracked_restore) (void)cudaEventSynchronize(restore_complete);
      (void)cudaEventDestroy(restore_complete);
    }
    if (device_checkpoint != nullptr) (void)cudaFree(device_checkpoint);
    if (restore_device) (void)cudaSetDevice(previous_device);
  }
};

Gfn2SccIterationInitializer::Gfn2SccIterationInitializer() noexcept = default;
Gfn2SccIterationInitializer::~Gfn2SccIterationInitializer() = default;
Gfn2SccIterationInitializer::Gfn2SccIterationInitializer(Gfn2SccIterationInitializer&&) noexcept =
    default;
Gfn2SccIterationInitializer& Gfn2SccIterationInitializer::operator=(
    Gfn2SccIterationInitializer&&) noexcept = default;

Gfn2SccIterationInitializationDiagnostic Gfn2SccIterationInitializer::create(
    const Gfn2SccIterationDevicePlan& plan,
    const Gfn2SccIterationArenaRequirements& arena_requirements, void* device_arena,
    std::size_t device_arena_bytes, const Gfn2SccIterationDeviceState& state,
    const Gfn2SccIterationDeviceWorkspace& workspace,
    const Gfn2SccIterationReportStorage& report_storage,
    const Gfn2SccIterationHostInitialization& host, Gfn2SccIterationInitializer& output) noexcept {
  Shape shape{};
  if (!derive_shape(plan, shape)) return fail(Error::kInvalidPlan, Field::kPlan);

  Gfn2SccIterationArenaRequirements current{};
  const auto arena_query = query_gfn2_scc_iteration_arena_requirements_cuda(
      plan, plan.eigensolver_provider.requirements, current);
  if (!arena_query.success()) return fail(Error::kInvalidPlan, Field::kPlan);
  if (!same_requirements(arena_requirements, current)) {
    return fail(Error::kStaleArenaRequirements, Field::kArena, -1, current.total_bytes,
                arena_requirements.total_bytes);
  }
  if (device_arena == nullptr ||
      reinterpret_cast<std::uintptr_t>(device_arena) % current.alignment != 0u ||
      device_arena_bytes < current.total_bytes) {
    return fail(Error::kInvalidArena, Field::kArena, -1, current.total_bytes, device_arena_bytes);
  }
  const auto arena_address = reinterpret_cast<std::uintptr_t>(device_arena);
  if (current.total_bytes != 0u &&
      current.total_bytes - 1u > std::numeric_limits<std::uintptr_t>::max() - arena_address) {
    return fail(Error::kCountOverflow, Field::kArena, -1, current.total_bytes, device_arena_bytes);
  }

  Diagnostic diagnostic = validate_common_host(shape, host);
  if (!diagnostic.success()) return diagnostic;
  diagnostic =
      host.mode == Mode::kFresh ? validate_fresh(shape, host) : validate_warm_extents(shape, host);
  if (!diagnostic.success()) return diagnostic;
  if (host.mode == Mode::kWarm) {
    diagnostic = validate_warm_invariants(shape, host, plan);
    if (!diagnostic.success()) return diagnostic;
  }

  ImageBuilder validation(device_arena, current.total_bytes, nullptr);
  diagnostic = project_state_image(shape, plan, state, workspace, report_storage, host, validation);
  if (!diagnostic.success()) return diagnostic;

  try {
    std::unique_ptr<Impl> candidate(new (std::nothrow) Impl());
    if (candidate == nullptr) return fail(Error::kAllocationFailed, Field::kArena);
    const cudaError_t device_query_status = cudaGetDevice(&candidate->device_ordinal);
    if (device_query_status != cudaSuccess) {
      candidate->device_ordinal = -1;
      return fail(Error::kCudaError, Field::kArena, -1, current.total_bytes, 0u,
                  device_query_status);
    }
    void* host_image = nullptr;
    const cudaError_t host_allocation_status = cudaMallocHost(&host_image, current.total_bytes);
    if (host_allocation_status != cudaSuccess) {
      return fail(host_allocation_status == cudaErrorMemoryAllocation ? Error::kAllocationFailed
                                                                      : Error::kCudaError,
                  Field::kArena, -1, current.total_bytes, 0u, host_allocation_status);
    }
    std::unique_ptr<void, decltype(&cudaFreeHost)> packed_image(host_image, &cudaFreeHost);
    std::memset(host_image, 0, current.total_bytes);
    ImageBuilder packer(device_arena, current.total_bytes, host_image);
    diagnostic = project_state_image(shape, plan, state, workspace, report_storage, host, packer);
    if (!diagnostic.success()) return diagnostic;

    const cudaError_t device_allocation_status =
        cudaMalloc(&candidate->device_checkpoint, current.total_bytes);
    if (device_allocation_status != cudaSuccess) {
      candidate->device_checkpoint = nullptr;
      return fail(device_allocation_status == cudaErrorMemoryAllocation ? Error::kAllocationFailed
                                                                        : Error::kCudaError,
                  Field::kArena, -1, current.total_bytes, 0u, device_allocation_status);
    }
    const cudaError_t upload_status = cudaMemcpy(candidate->device_checkpoint, host_image,
                                                 current.total_bytes, cudaMemcpyHostToDevice);
    if (upload_status != cudaSuccess) {
      return fail(Error::kCudaError, Field::kArena, -1, current.total_bytes, 0u, upload_status);
    }
    const cudaError_t event_status =
        cudaEventCreateWithFlags(&candidate->restore_complete, cudaEventDisableTiming);
    if (event_status != cudaSuccess) {
      candidate->restore_complete = nullptr;
      return fail(
          event_status == cudaErrorMemoryAllocation ? Error::kAllocationFailed : Error::kCudaError,
          Field::kArena, -1, current.total_bytes, 0u, event_status);
    }

    candidate->device_arena = device_arena;
    candidate->image_bytes = current.total_bytes;
    candidate->plan_token = shape.token;
    candidate->initialization_generation = host.initialization_generation;
    candidate->mode = host.mode;
    Gfn2SccIterationInitializer replacement;
    replacement.impl_ = std::move(candidate);
    output = std::move(replacement);
    return {};
  } catch (const std::bad_alloc&) {
    return fail(Error::kAllocationFailed, Field::kArena, -1, current.total_bytes);
  } catch (...) {
    return fail(Error::kInvalidPlan, Field::kPlan);
  }
}

bool Gfn2SccIterationInitializer::valid() const noexcept { return impl_ != nullptr; }

std::size_t Gfn2SccIterationInitializer::image_bytes() const noexcept {
  return impl_ == nullptr ? 0u : impl_->image_bytes;
}

std::uint64_t Gfn2SccIterationInitializer::plan_token() const noexcept {
  return impl_ == nullptr ? 0u : impl_->plan_token;
}

std::uint64_t Gfn2SccIterationInitializer::initialization_generation() const noexcept {
  return impl_ == nullptr ? 0u : impl_->initialization_generation;
}

Gfn2SccIterationInitializationMode Gfn2SccIterationInitializer::mode() const noexcept {
  return impl_ == nullptr ? Mode::kFresh : impl_->mode;
}

const void* Gfn2SccIterationInitializer::device_checkpoint() const noexcept {
  return impl_ == nullptr ? nullptr : impl_->device_checkpoint;
}

Gfn2SccIterationInitializationDiagnostic Gfn2SccIterationInitializer::upload_async(
    void* device_arena, std::size_t device_arena_bytes, Gfn2SccIterationInitializationReady& ready,
    cudaStream_t stream) const noexcept {
  ready = {};
  if (impl_ == nullptr) return fail(Error::kInvalidPlan, Field::kPlan);
  if (device_arena != impl_->device_arena || device_arena_bytes < impl_->image_bytes) {
    return fail(Error::kInvalidArena, Field::kArena, -1, impl_->image_bytes, device_arena_bytes);
  }

  /* Keep the stale-allocation diagnostic ahead of cudaMemcpyAsync. Some CUDA
   * drivers do not safely reject a previously freed UVA destination inside
   * the D2D submission path itself. This metadata query neither transfers
   * data nor synchronizes queued stream work. */
  cudaPointerAttributes attributes{};
  const cudaError_t attribute_status = cudaPointerGetAttributes(&attributes, device_arena);
  if (attribute_status != cudaSuccess) {
    (void)cudaGetLastError();
    return fail(Error::kInvalidArenaMemory, Field::kArena, -1, impl_->image_bytes,
                device_arena_bytes, attribute_status);
  }
  if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) {
    return fail(Error::kInvalidArenaMemory, Field::kArena, -1, impl_->image_bytes,
                device_arena_bytes);
  }

  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  const cudaError_t capture_query_status = cudaStreamIsCapturing(stream, &capture_status);
  if (capture_query_status != cudaSuccess) {
    return fail(Error::kCudaError, Field::kArena, -1, impl_->image_bytes, device_arena_bytes,
                capture_query_status);
  }
  const bool track_restore = capture_status == cudaStreamCaptureStatusNone;
  if (track_restore && impl_->has_tracked_restore) {
    const cudaError_t wait_status = cudaStreamWaitEvent(stream, impl_->restore_complete, 0u);
    if (wait_status != cudaSuccess) {
      return fail(Error::kCudaError, Field::kArena, -1, impl_->image_bytes, device_arena_bytes,
                  wait_status);
    }
  }

  const cudaError_t status = cudaMemcpyAsync(device_arena, impl_->device_checkpoint,
                                             impl_->image_bytes, cudaMemcpyDeviceToDevice, stream);
  if (status != cudaSuccess) {
    const Error error = status == cudaErrorInvalidValue || status == cudaErrorInvalidDevicePointer
                            ? Error::kInvalidArenaMemory
                            : Error::kCudaError;
    return fail(error, Field::kArena, -1, impl_->image_bytes, device_arena_bytes, status);
  }
  if (track_restore) {
    const cudaError_t record_status = cudaEventRecord(impl_->restore_complete, stream);
    if (record_status != cudaSuccess) {
      /* The D2D submission was already accepted. Drain this exceptional path
       * so a later owner destruction cannot release its source prematurely. */
      (void)cudaStreamSynchronize(stream);
      return fail(Error::kCudaError, Field::kArena, -1, impl_->image_bytes, device_arena_bytes,
                  record_status);
    }
    impl_->has_tracked_restore = true;
  }

  Gfn2SccIterationInitializationReady candidate{};
  candidate.mode = impl_->mode;
  candidate.plan_token = impl_->plan_token;
  candidate.initialization_generation = impl_->initialization_generation;
  candidate.device_arena = device_arena;
  candidate.arena_bytes = impl_->image_bytes;
  candidate.ready_on_stream = 1u;
  ready = candidate;
  return {};
}

}  // namespace gpuxtb::detail::cuda
