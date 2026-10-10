#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "runtime/gfn2_cpu_scc_snapshot.hpp"
#include "xtbloom/xtbloom.h"

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      std::cerr << "CHECK failed at line " << __LINE__ << ": " << #condition \
                << "; last error: " << xtbloom_get_last_error() << '\n';     \
      return 1;                                                              \
    }                                                                        \
  } while (false)

namespace {

using xtbloom::detail::Gfn2CpuSccSnapshot;
using xtbloom::detail::Gfn2CpuSccSystemSnapshot;
using xtbloom::detail::snapshot_public_gfn2_cpu_scc;

template <typename Value>
bool byte_equal(const std::vector<Value>& left, const std::vector<Value>& right) {
  return left.size() == right.size() &&
         (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(Value)) == 0);
}

auto floating_fields(const Gfn2CpuSccSystemSnapshot& system) {
  return std::array{&system.public_forces,
                    &system.public_atomic_charges,
                    &system.overlap,
                    &system.core_hamiltonian,
                    &system.coefficients,
                    &system.eigenvalues,
                    &system.occupations,
                    &system.density,
                    &system.energy_weighted_density,
                    &system.qsh,
                    &system.qat,
                    &system.dipoles,
                    &system.quadrupoles,
                    &system.mixer_current_inputs,
                    &system.mixer_previous_inputs,
                    &system.mixer_previous_residuals,
                    &system.layout.reference_atom_occupations,
                    &system.layout.reference_shell_occupations,
                    &system.layout.electron_counts,
                    &system.layout.alpha_electron_counts,
                    &system.layout.beta_electron_counts,
                    &system.layout.molecular_charges};
}

auto floating_scalars(const Gfn2CpuSccSystemSnapshot& system) {
  return std::array{system.public_energy, system.mixer_residual_rms, system.mixer_residual_maximum,
                    system.scc_free_energy, system.scc_free_energy_change};
}

bool same_layout(const xtbloom::detail::gfn2::WavefunctionLayout& left,
                 const xtbloom::detail::gfn2::WavefunctionLayout& right) {
  if (left.batch_size != right.batch_size || left.total_atoms != right.total_atoms ||
      left.total_shells != right.total_shells || left.total_orbitals != right.total_orbitals ||
      left.workspace_size_bytes != right.workspace_size_bytes ||
      left.atom_offsets != right.atom_offsets ||
      left.batch_shell_offsets != right.batch_shell_offsets ||
      left.batch_orbital_offsets != right.batch_orbital_offsets ||
      left.atomic_numbers != right.atomic_numbers ||
      left.unpaired_electrons != right.unpaired_electrons ||
      left.spin_channels != right.spin_channels)
    return false;
  const auto field_layouts = [](const auto& layout) {
    return std::array{&layout.coefficients, &layout.eigenvalues, &layout.occupations,
                      &layout.density,      &layout.qsh,         &layout.qat,
                      &layout.dipole,       &layout.quadrupole,  &layout.energy_weighted_density};
  };
  const auto left_fields = field_layouts(left);
  const auto right_fields = field_layouts(right);
  for (std::size_t index = 0u; index < left_fields.size(); ++index) {
    const auto& left_field = *left_fields[index];
    const auto& right_field = *right_fields[index];
    if (left_field.offset_bytes != right_field.offset_bytes ||
        left_field.size_bytes != right_field.size_bytes ||
        left_field.element_count != right_field.element_count ||
        left_field.system_offsets != right_field.system_offsets)
      return false;
  }
  return true;
}

bool same_snapshot(const Gfn2CpuSccSnapshot& left, const Gfn2CpuSccSnapshot& right) {
  if (left.cpu_isa != right.cpu_isa ||
      left.public_compute_sequence != right.public_compute_sequence ||
      left.systems.size() != right.systems.size())
    return false;
  for (std::size_t index = 0u; index < left.systems.size(); ++index) {
    const auto& left_system = left.systems[index];
    const auto& right_system = right.systems[index];
    if (!same_layout(left_system.layout, right_system.layout) ||
        left_system.atom_shell_offsets != right_system.atom_shell_offsets ||
        left_system.shell_orbital_offsets != right_system.shell_orbital_offsets ||
        left_system.geometry_generation != right_system.geometry_generation ||
        left_system.status != right_system.status ||
        left_system.iterations != right_system.iterations ||
        left_system.mixer_iterations != right_system.mixer_iterations)
      return false;
    const auto left_fields = floating_fields(left_system);
    const auto right_fields = floating_fields(right_system);
    for (std::size_t field = 0u; field < left_fields.size(); ++field) {
      if (!byte_equal(*left_fields[field], *right_fields[field])) return false;
    }
    const auto left_scalars = floating_scalars(left_system);
    const auto right_scalars = floating_scalars(right_system);
    if (std::memcmp(left_scalars.data(), right_scalars.data(),
                    left_scalars.size() * sizeof(double)) != 0)
      return false;
  }
  return true;
}

bool finite_snapshot(const Gfn2CpuSccSnapshot& snapshot) {
  for (const auto& system : snapshot.systems) {
    for (const auto* field : floating_fields(system)) {
      if (!std::all_of(field->begin(), field->end(),
                       [](double value) { return std::isfinite(value); }))
        return false;
    }
    const auto scalars = floating_scalars(system);
    if (!std::all_of(scalars.begin(), scalars.end(),
                     [](double value) { return std::isfinite(value); }))
      return false;
  }
  return true;
}

struct ContextDeleter {
  void operator()(xtbloom_context_t* context) const noexcept { xtbloom_context_destroy(context); }
};
using ContextHandle = std::unique_ptr<xtbloom_context_t, ContextDeleter>;

template <typename Value>
xtbloom_const_buffer_t input_buffer(const std::vector<Value>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(Value),
          XTBLOOM_MEMORY_HOST, 0};
}

template <typename Value>
xtbloom_buffer_t output_buffer(std::vector<Value>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(Value),
          XTBLOOM_MEMORY_HOST, 0};
}

struct PublicBatch {
  std::vector<std::int64_t> offsets{0, 2, 5};
  std::vector<std::int32_t> numbers{1, 1, 1, 1, 1};
  std::vector<double> positions{0, 0, 0, 1.4, 0, 0, 0, 0, 0, 1.4, 0, 0, 3.1, 0.2, 0};
  std::vector<double> charges{0, 0};
  std::vector<std::int32_t> unpaired{0, 1};
  std::vector<std::int32_t> spins{1, 2};
  std::vector<double> energies;
  std::vector<double> forces;
  std::vector<double> atomic_charges;
  std::vector<std::int32_t> iterations;
  std::vector<std::int32_t> statuses;
  std::vector<std::uint8_t> converged;
  xtbloom_batch_t batch{};
  xtbloom_compute_options_t options{};
  xtbloom_batch_result_t result{};

  void bind() {
    xtbloom_batch_init(&batch, sizeof(batch));
    xtbloom_compute_options_init(&options, sizeof(options));
    xtbloom_batch_result_init(&result, sizeof(result));
    options.flags =
        XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES | XTBLOOM_COMPUTE_ATOMIC_CHARGES;
    options.max_scc_iterations = 500;
    options.energy_tolerance = 1e-10;
    options.charge_tolerance = 1e-8;
    options.electronic_temperature = 300.0 * 3.166808578545117e-6;
    options.scc_mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
    options.scc_mixer_history = 8;
    options.scc_mixer_damping = 0.4;
    options.determinism = XTBLOOM_DETERMINISM_DEFAULT;
    batch.batch_size = static_cast<std::int64_t>(offsets.size() - 1u);
    batch.total_atoms = static_cast<std::int64_t>(numbers.size());
    batch.atom_offsets = input_buffer(offsets);
    batch.atomic_numbers = input_buffer(numbers);
    batch.positions = input_buffer(positions);
    batch.molecular_charges = input_buffer(charges);
    batch.unpaired_electrons = input_buffer(unpaired);
    batch.spin_channels = input_buffer(spins);
    const std::size_t systems = charges.size();
    energies.assign(systems, -71.0);
    forces.assign(3u * numbers.size(), -72.0);
    atomic_charges.assign(numbers.size(), -73.0);
    iterations.assign(systems, -74);
    statuses.assign(systems, -75);
    converged.assign(systems, 76u);
    result.energies = output_buffer(energies);
    result.forces = output_buffer(forces);
    result.atomic_charges = output_buffer(atomic_charges);
    result.scc_iterations = output_buffer(iterations);
    result.scc_converged = output_buffer(converged);
    result.per_system_status = output_buffer(statuses);
  }

  bool same_outputs(const PublicBatch& other) const {
    return byte_equal(energies, other.energies) && byte_equal(forces, other.forces) &&
           byte_equal(atomic_charges, other.atomic_charges) && iterations == other.iterations &&
           statuses == other.statuses && converged == other.converged &&
           result.flags == other.result.flags;
  }
};

int make_context(ContextHandle& handle) {
  xtbloom_context_options_t options{};
  CHECK(xtbloom_context_options_init(&options, sizeof(options)) == XTBLOOM_STATUS_SUCCESS);
  options.backend = XTBLOOM_BACKEND_CPU;
  options.cpu_threads = 1;
  xtbloom_context_t* context = nullptr;
  CHECK(xtbloom_context_create(&options, &context) == XTBLOOM_STATUS_SUCCESS);
  handle.reset(context);
  return 0;
}

int check_snapshot(const Gfn2CpuSccSnapshot& snapshot, const PublicBatch& public_batch) {
  CHECK(snapshot.systems.size() == public_batch.charges.size());
  for (std::size_t index = 0u; index < snapshot.systems.size(); ++index) {
    const auto& system = snapshot.systems[index];
    const auto& layout = system.layout;
    CHECK(layout.batch_size == 1);
    CHECK(layout.spin_channels == std::vector<std::int32_t>{public_batch.spins[index]});
    CHECK(system.status == public_batch.statuses[index]);
    CHECK(system.iterations == public_batch.iterations[index]);
    CHECK(system.public_energy == public_batch.energies[index]);
    const std::size_t atoms = static_cast<std::size_t>(layout.total_atoms);
    const std::size_t orbitals = static_cast<std::size_t>(layout.total_orbitals);
    const std::size_t begin = static_cast<std::size_t>(public_batch.offsets[index]);
    CHECK(atoms == static_cast<std::size_t>(public_batch.offsets[index + 1u]) - begin);
    CHECK(system.public_atomic_charges ==
          std::vector<double>(public_batch.atomic_charges.begin() + begin,
                              public_batch.atomic_charges.begin() + begin + atoms));
    CHECK(system.public_forces ==
          std::vector<double>(public_batch.forces.begin() + 3u * begin,
                              public_batch.forces.begin() + 3u * (begin + atoms)));
    CHECK(system.coefficients.size() ==
          static_cast<std::size_t>(layout.coefficients.element_count));
    CHECK(system.eigenvalues.size() == static_cast<std::size_t>(layout.eigenvalues.element_count));
    CHECK(system.density.size() == static_cast<std::size_t>(layout.density.element_count));
    CHECK(system.energy_weighted_density.size() ==
          static_cast<std::size_t>(layout.energy_weighted_density.element_count));
    CHECK(system.occupations.size() == 2u * orbitals);
    CHECK(system.qsh.size() == static_cast<std::size_t>(layout.qsh.element_count));
    CHECK(system.qat.size() == static_cast<std::size_t>(layout.qat.element_count));
    CHECK(system.dipoles.size() == static_cast<std::size_t>(layout.dipole.element_count));
    CHECK(system.quadrupoles.size() == static_cast<std::size_t>(layout.quadrupole.element_count));
    CHECK(system.overlap.size() == orbitals * orbitals);
    CHECK(system.core_hamiltonian.size() == orbitals * orbitals);
    CHECK(system.mixer_current_inputs.size() ==
          system.qsh.size() + system.dipoles.size() + system.quadrupoles.size());
    CHECK(system.mixer_previous_inputs.size() == system.mixer_current_inputs.size());
    CHECK(system.mixer_previous_residuals.size() == system.mixer_current_inputs.size());
    CHECK(std::isfinite(system.mixer_residual_rms));
    CHECK(std::isfinite(system.scc_free_energy));
    CHECK(system.atom_shell_offsets.size() == atoms + 1u);
    CHECK(system.shell_orbital_offsets.size() ==
          static_cast<std::size_t>(layout.total_shells) + 1u);
  }
  return 0;
}

int run_tests() {
  ContextHandle observed;
  ContextHandle control;
  CHECK(make_context(observed) == 0);
  CHECK(make_context(control) == 0);
  std::string error;
  Gfn2CpuSccSnapshot snapshot;
  snapshot.public_compute_sequence = 999u;
  CHECK(snapshot_public_gfn2_cpu_scc(nullptr, snapshot, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(snapshot_public_gfn2_cpu_scc(observed.get(), snapshot, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(snapshot.public_compute_sequence == 999u && snapshot.systems.empty());

  PublicBatch actual;
  PublicBatch expected;
  actual.bind();
  expected.bind();
  CHECK(xtbloom_compute(observed.get(), &actual.batch, &actual.options, &actual.result) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom_compute(control.get(), &expected.batch, &expected.options, &expected.result) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(actual.same_outputs(expected));
  CHECK(snapshot_public_gfn2_cpu_scc(observed.get(), snapshot, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());
  CHECK(snapshot.public_compute_sequence == 1u);
  CHECK(check_snapshot(snapshot, actual) == 0);
  const auto retained = snapshot;
  const auto retained_original = retained;
  const auto retained_density = retained.systems[1].density;
  const auto retained_overlap = retained.systems[0].overlap;
  CHECK(snapshot_public_gfn2_cpu_scc(observed.get(), snapshot, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(snapshot.systems[1].density == retained.systems[1].density);

  /* Capturing must not replace next inputs, consume the WARM checkpoint, or
   * alter the mixer. Compare the same public sequence in an unobserved context. */
  actual.options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  expected.options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  for (std::size_t restart = 0u; restart < 2u; ++restart) {
    CHECK(xtbloom_compute(observed.get(), &actual.batch, &actual.options, &actual.result) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(xtbloom_compute(control.get(), &expected.batch, &expected.options, &expected.result) ==
          XTBLOOM_STATUS_SUCCESS);
    CHECK(actual.same_outputs(expected));
    CHECK(snapshot_public_gfn2_cpu_scc(observed.get(), snapshot, error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(snapshot.public_compute_sequence == restart + 2u);
    CHECK(check_snapshot(snapshot, actual) == 0);
  }
  CHECK(retained.systems[1].density == retained_density);

  const auto before_rejection = actual.energies;
  const auto before_invalid_snapshot = snapshot;
  actual.positions[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom_compute(observed.get(), &actual.batch, &actual.options, &actual.result) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(actual.energies == before_rejection);
  CHECK(snapshot_public_gfn2_cpu_scc(observed.get(), snapshot, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_snapshot(snapshot, before_invalid_snapshot));

  actual.positions[0] = 0.05;
  actual.options.scc_start_mode = XTBLOOM_SCC_START_FRESH;
  CHECK(xtbloom_compute(observed.get(), &actual.batch, &actual.options, &actual.result) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(snapshot_public_gfn2_cpu_scc(observed.get(), snapshot, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(check_snapshot(snapshot, actual) == 0);
  CHECK(snapshot.systems[0].geometry_generation > retained.systems[0].geometry_generation);
  CHECK(retained.systems[0].overlap == retained_overlap);
  CHECK(retained.systems[1].density == retained_density);

  /* A successful call can contain data-level nonconvergence. Never label such
   * a cache as a complete converged terminal capture merely from call status. */
  actual.options.max_scc_iterations = 1;
  CHECK(xtbloom_compute(observed.get(), &actual.batch, &actual.options, &actual.result) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(std::any_of(actual.statuses.begin(), actual.statuses.end(),
                    [](std::int32_t status) { return status != XTBLOOM_STATUS_SUCCESS; }));
  const auto before_failed_snapshot = snapshot;
  CHECK(snapshot_public_gfn2_cpu_scc(observed.get(), snapshot, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_snapshot(snapshot, before_failed_snapshot));

  actual.offsets = {0, 2};
  actual.numbers.resize(2u);
  actual.positions = {0, 0, 0, 1.5, 0, 0};
  actual.charges = {0};
  actual.unpaired = {0};
  actual.spins = {1};
  actual.bind();
  CHECK(xtbloom_compute(observed.get(), &actual.batch, &actual.options, &actual.result) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(snapshot_public_gfn2_cpu_scc(observed.get(), snapshot, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(check_snapshot(snapshot, actual) == 0);
  CHECK(snapshot.systems.size() == 1u);
  CHECK(retained.systems.size() == 2u);
  CHECK(retained.systems[1].density == retained_density);

  actual.options.model = XTBLOOM_MODEL_GFN1_XTB;
  CHECK(xtbloom_compute(observed.get(), &actual.batch, &actual.options, &actual.result) ==
        XTBLOOM_STATUS_SUCCESS);
  const auto before_other_model = snapshot;
  CHECK(snapshot_public_gfn2_cpu_scc(observed.get(), snapshot, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_snapshot(snapshot, before_other_model));

  actual.offsets = {0};
  actual.numbers.clear();
  actual.positions.clear();
  actual.charges.clear();
  actual.unpaired.clear();
  actual.spins.clear();
  actual.bind();
  CHECK(xtbloom_compute(observed.get(), &actual.batch, &actual.options, &actual.result) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(snapshot_public_gfn2_cpu_scc(observed.get(), snapshot, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(same_snapshot(snapshot, before_other_model));
  observed.reset();
  control.reset();
  CHECK(same_snapshot(retained, retained_original));
  return 0;
}

template <typename Value>
void write_array(const char* name, const std::vector<Value>& values, bool last = false) {
  std::cout << '"' << name << "\":[";
  for (std::size_t index = 0u; index < values.size(); ++index) {
    if (index != 0u) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']' << (last ? "" : ",");
}

void write_snapshot(const Gfn2CpuSccSnapshot& snapshot) {
  const auto& system = snapshot.systems.at(0);
  const auto& layout = system.layout;
  std::cout << "{\"public_compute_sequence\":" << snapshot.public_compute_sequence
            << ",\"cpu_isa\":\"" << xtbloom::detail::cpu_isa_name(snapshot.cpu_isa)
            << "\",\"status\":" << system.status << ",\"iterations\":" << system.iterations
            << ",\"geometry_generation\":" << system.geometry_generation
            << ",\"atoms\":" << layout.total_atoms << ",\"shells\":" << layout.total_shells
            << ",\"orbitals\":" << layout.total_orbitals
            << ",\"spin_channels\":" << layout.spin_channels.at(0)
            << ",\"public_energy\":" << system.public_energy
            << ",\"mixer_residual_rms\":" << system.mixer_residual_rms
            << ",\"mixer_residual_maximum\":" << system.mixer_residual_maximum
            << ",\"mixer_iterations\":" << system.mixer_iterations
            << ",\"scc_free_energy\":" << system.scc_free_energy
            << ",\"scc_free_energy_change\":" << system.scc_free_energy_change << ',';
  write_array("atom_shell_offsets", system.atom_shell_offsets);
  write_array("shell_orbital_offsets", system.shell_orbital_offsets);
  write_array("reference_shell_occupations", layout.reference_shell_occupations);
  write_array("alpha_electron_counts", layout.alpha_electron_counts);
  write_array("beta_electron_counts", layout.beta_electron_counts);
  write_array("public_forces", system.public_forces);
  write_array("public_atomic_charges", system.public_atomic_charges);
  write_array("overlap", system.overlap);
  write_array("core_hamiltonian", system.core_hamiltonian);
  write_array("coefficients", system.coefficients);
  write_array("eigenvalues", system.eigenvalues);
  write_array("occupations", system.occupations);
  write_array("density", system.density);
  write_array("energy_weighted_density", system.energy_weighted_density);
  write_array("qsh", system.qsh);
  write_array("qat", system.qat);
  write_array("dipoles", system.dipoles);
  write_array("quadrupoles", system.quadrupoles);
  write_array("mixer_current_inputs", system.mixer_current_inputs);
  write_array("mixer_previous_inputs", system.mixer_previous_inputs);
  write_array("mixer_previous_residuals", system.mixer_previous_residuals, true);
  std::cout << '}';
}

int capture_input(const std::string& path) {
  std::ifstream input(path);
  std::int64_t atoms = 0;
  double charge = 0.0;
  std::int32_t unpaired = 0;
  std::int32_t spins = 0;
  CHECK(input >> atoms >> charge >> unpaired >> spins);
  CHECK(atoms > 0 && atoms <= 10000);
  PublicBatch public_batch;
  public_batch.offsets = {0, atoms};
  public_batch.charges = {charge};
  public_batch.unpaired = {unpaired};
  public_batch.spins = {spins};
  public_batch.numbers.resize(static_cast<std::size_t>(atoms));
  public_batch.positions.resize(3u * static_cast<std::size_t>(atoms));
  for (std::size_t atom = 0u; atom < static_cast<std::size_t>(atoms); ++atom) {
    CHECK(input >> public_batch.numbers[atom] >> public_batch.positions[3u * atom] >>
          public_batch.positions[3u * atom + 1u] >> public_batch.positions[3u * atom + 2u]);
  }
  input >> std::ws;
  CHECK(input.eof());
  public_batch.bind();
  ContextHandle context;
  CHECK(make_context(context) == 0);
  std::vector<Gfn2CpuSccSnapshot> captures;
  for (std::size_t attempt = 0u; attempt < 3u; ++attempt) {
    CHECK(xtbloom_compute(context.get(), &public_batch.batch, &public_batch.options,
                          &public_batch.result) == XTBLOOM_STATUS_SUCCESS);
    Gfn2CpuSccSnapshot snapshot;
    std::string error;
    CHECK(snapshot_public_gfn2_cpu_scc(context.get(), snapshot, error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(check_snapshot(snapshot, public_batch) == 0);
    CHECK(finite_snapshot(snapshot));
    captures.push_back(std::move(snapshot));
    public_batch.options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  }
  /* Serialize only after the complete sequence succeeds, keeping diagnostic
   * transfers/copies outside SCC and preserving round-trip binary64 values. */
  std::cout << std::showpoint << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "{\"schema\":\"xtbloom.cpu.scc_snapshot.v1\","
               "\"producer\":\"standalone-public-api-diagnostic\","
               "\"units\":\"atomic\",\"mixer_vector_fields\":[\"qsh\",\"dipoles\",\"quadrupoles\"],"
               "\"options\":{\"cpu_threads\":1,\"model\":"
            << public_batch.options.model << ",\"flags\":" << public_batch.options.flags
            << ",\"max_scc_iterations\":" << public_batch.options.max_scc_iterations
            << ",\"electronic_temperature\":" << public_batch.options.electronic_temperature
            << ",\"energy_tolerance\":" << public_batch.options.energy_tolerance
            << ",\"charge_tolerance\":" << public_batch.options.charge_tolerance
            << ",\"scc_mixer\":" << public_batch.options.scc_mixer
            << ",\"scc_mixer_history\":" << public_batch.options.scc_mixer_history
            << ",\"scc_mixer_damping\":" << public_batch.options.scc_mixer_damping
            << ",\"determinism\":" << public_batch.options.determinism << "},\"captures\":[";
  for (std::size_t index = 0u; index < captures.size(); ++index) {
    if (index != 0u) std::cout << ',';
    write_snapshot(captures[index]);
  }
  std::cout << "]}\n";
  std::cout.flush();
  CHECK(std::cout.good());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) return run_tests();
  if (argc == 3 && std::string(argv[1]) == "--capture") return capture_input(argv[2]);
  std::cerr << "usage: xtbloom_cpu_scc_snapshot_test [--capture INPUT]\n";
  return 2;
}
