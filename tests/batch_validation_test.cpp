#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime/validation.hpp"
#include "xtbloom/xtbloom.h"

namespace {

using xtbloom::detail::DescriptorValidationResult;
using xtbloom::detail::kAtomicNumbersNeedStaging;
using xtbloom::detail::kAtomOffsetsNeedStaging;
using xtbloom::detail::kChargeResponseOffsetsNeedStaging;
using xtbloom::detail::kChargeResponseShapeNeedsStaging;
using xtbloom::detail::kMolecularChargesNeedStaging;
using xtbloom::detail::kPointChargeOffsetsNeedStaging;
using xtbloom::detail::kSpinChannelsNeedStaging;
using xtbloom::detail::kTopologyMetadataStagingMask;
using xtbloom::detail::kUnpairedElectronsNeedStaging;
using xtbloom::detail::validate_compute_descriptor_structure;
using xtbloom::detail::validate_compute_descriptors;
using xtbloom::detail::validate_host_topology_semantics;

#define CHECK(condition)                                                                 \
  do {                                                                                   \
    if (!(condition)) {                                                                  \
      std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: " #condition << '\n'; \
      return false;                                                                      \
    }                                                                                    \
  } while (false)

template <typename T>
xtbloom_const_buffer_t input_buffer(const std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0};
}

template <typename T>
xtbloom_buffer_t output_buffer(std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0};
}

struct Fixture {
  std::vector<std::int64_t> atom_offsets{0, 2, 3};
  std::vector<std::int32_t> atomic_numbers{1, 8, 6};
  std::vector<double> positions{0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 2.0, 0.0};
  std::vector<double> molecular_charges{0.0, 0.0};
  std::vector<std::int32_t> unpaired_electrons{0, 0};
  std::vector<std::int32_t> spin_channels;

  std::vector<std::int64_t> point_offsets;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> potential_shifts;
  std::vector<std::int64_t> response_offsets;
  std::vector<double> response_matrix;
  std::vector<xtbloom_interaction_t> interactions;
  std::vector<std::uint8_t> interaction_payload;
  alignas(double) std::array<std::uint8_t, 36> misaligned_interaction_payload{};

  std::vector<double> energies{0.0, 0.0};
  std::vector<double> forces = std::vector<double>(9);
  std::vector<double> atomic_charges = std::vector<double>(3);
  std::vector<double> point_forces;
  std::vector<std::int32_t> scc_iterations = std::vector<std::int32_t>(2);
  std::vector<std::uint8_t> scc_converged = std::vector<std::uint8_t>(2);
  std::vector<std::int32_t> per_system_status = std::vector<std::int32_t>(2);

  xtbloom_batch_t batch{};
  xtbloom_compute_options_t options{};
  xtbloom_batch_result_t result{};

  Fixture() {
    batch.struct_size = sizeof(batch);
    batch.api_version = XTBLOOM_API_VERSION;
    batch.batch_size = 2;
    batch.total_atoms = 3;
    batch.atom_offsets = input_buffer(atom_offsets);
    batch.atomic_numbers = input_buffer(atomic_numbers);
    batch.positions = input_buffer(positions);
    batch.molecular_charges = input_buffer(molecular_charges);
    batch.unpaired_electrons = input_buffer(unpaired_electrons);

    options.struct_size = sizeof(options);
    options.api_version = XTBLOOM_API_VERSION;
    options.model = XTBLOOM_MODEL_GFN2_XTB;
    options.flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES;
    options.max_scc_iterations = 250;
    options.charge_tolerance = 1.0e-6;
    options.energy_tolerance = 1.0e-8;
    options.electronic_temperature = XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE;
    options.scc_start_mode = XTBLOOM_SCC_START_FRESH;
    options.scc_mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
    options.scc_mixer_history = 8;
    options.scc_mixer_damping = 0.4;
    options.determinism = XTBLOOM_DETERMINISM_DEFAULT;

    result.struct_size = sizeof(result);
    result.api_version = XTBLOOM_API_VERSION;
    result.energies = output_buffer(energies);
    result.forces = output_buffer(forces);
    result.atomic_charges = output_buffer(atomic_charges);
    result.scc_iterations = output_buffer(scc_iterations);
    result.scc_converged = output_buffer(scc_converged);
    result.per_system_status = output_buffer(per_system_status);
  }

  void enable_point_charges() {
    point_offsets = {0, 1, 1};
    point_positions = {3.0, 0.0, 0.0};
    point_values = {-0.5};
    point_gammas = {0.2};
    point_forces.assign(3, 0.0);
    batch.total_point_charges = 1;
    batch.point_charge_offsets = input_buffer(point_offsets);
    batch.point_charge_positions = input_buffer(point_positions);
    batch.point_charge_values = input_buffer(point_values);
    batch.point_charge_gammas = input_buffer(point_gammas);
    result.point_charge_forces = output_buffer(point_forces);
  }

  void enable_spin_channels(std::vector<std::int32_t> values = {1, 2}) {
    spin_channels = std::move(values);
    batch.struct_size = XTBLOOM_BATCH_V2_SIZE;
    batch.spin_channels = input_buffer(spin_channels);
  }

  void enable_response() {
    /* The two systems have 2 and 1 atoms, hence 2^2 + 1^2 = 5 elements. */
    response_offsets = {0, 4, 5};
    response_matrix.assign(5, 0.0);
    batch.total_charge_response_elements = 5;
    batch.charge_response_offsets = input_buffer(response_offsets);
    batch.charge_response_matrix = input_buffer(response_matrix);
  }

  void enable_interaction(const xtbloom_interaction_t& interaction,
                          const std::vector<std::uint8_t>& payload) {
    batch.struct_size = sizeof(batch);
    interactions.assign(1, interaction);
    interaction_payload = payload;
    batch.total_interactions = 1;
    batch.interaction_descriptors = input_buffer(interactions);
    batch.interaction_payload = input_buffer(interaction_payload);
  }
};

using Mutation = std::function<void(Fixture&)>;

struct InvalidCase {
  const char* name;
  Mutation mutate;
  const char* error_fragment;
  xtbloom_status_t status = XTBLOOM_STATUS_INVALID_ARGUMENT;
  xtbloom_backend_t backend = XTBLOOM_BACKEND_CPU;
};

bool expect_invalid(const InvalidCase& test) {
  Fixture fixture;
  test.mutate(fixture);
  const DescriptorValidationResult result =
      validate_compute_descriptors(test.backend, &fixture.batch, &fixture.options, &fixture.result);
  if (result.status != test.status || result.error.find(test.error_fragment) == std::string::npos) {
    std::cerr << "case '" << test.name << "' returned status " << result.status << ", error '"
              << result.error << "'\n";
    return false;
  }
  return true;
}

bool test_valid_requests_and_staging_contract() {
  Fixture fixture;
  DescriptorValidationResult checked = validate_compute_descriptors(
      XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options, &fixture.result);
  CHECK(checked.ok());
  CHECK(!checked.requires_backend_staging_validation());

  /* Zero point charges and a requested zero-length point-force output need no dummy pointers. */
  fixture.options.flags |= XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
  checked = validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options,
                                         &fixture.result);
  CHECK(checked.ok());
  CHECK(fixture.batch.point_charge_offsets.data == nullptr);
  CHECK(fixture.result.point_charge_forces.data == nullptr);

  /* CUDA accepts useful hybrid calls: metadata on the host and bulk arrays on device. */
  fixture.batch.positions.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  fixture.result.forces.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(XTBLOOM_BACKEND_CUDA, &fixture.batch, &fixture.options,
                                         &fixture.result);
  CHECK(checked.ok());
  CHECK(!checked.requires_backend_staging_validation());

  fixture.batch.atom_offsets.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(XTBLOOM_BACKEND_CUDA, &fixture.batch, &fixture.options,
                                         &fixture.result);
  CHECK(checked.ok());
  CHECK(checked.requires_backend_staging_validation());
  CHECK((checked.pending_offset_checks & kAtomOffsetsNeedStaging) != 0);

  Fixture point_fixture;
  point_fixture.enable_point_charges();
  point_fixture.batch.point_charge_offsets.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(XTBLOOM_BACKEND_CUDA, &point_fixture.batch,
                                         &point_fixture.options, &point_fixture.result);
  CHECK(checked.ok());
  CHECK((checked.pending_offset_checks & kPointChargeOffsetsNeedStaging) != 0);

  Fixture response_fixture;
  response_fixture.enable_response();
  response_fixture.batch.charge_response_offsets.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(XTBLOOM_BACKEND_CUDA, &response_fixture.batch,
                                         &response_fixture.options, &response_fixture.result);
  CHECK(checked.ok());
  CHECK((checked.pending_offset_checks & kChargeResponseOffsetsNeedStaging) != 0);
  CHECK((checked.pending_offset_checks & kChargeResponseShapeNeedsStaging) != 0);

  response_fixture.batch.atom_offsets.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  response_fixture.batch.charge_response_offsets.memory_space = XTBLOOM_MEMORY_HOST;
  checked = validate_compute_descriptors(XTBLOOM_BACKEND_CUDA, &response_fixture.batch,
                                         &response_fixture.options, &response_fixture.result);
  CHECK(checked.ok());
  CHECK((checked.pending_offset_checks & kAtomOffsetsNeedStaging) != 0);
  CHECK((checked.pending_offset_checks & kChargeResponseShapeNeedsStaging) != 0);
  return true;
}

bool test_structural_layer_never_dereferences_buffer_storage() {
  Fixture fixture;

  /*
   * An opaque address stands in for a CUDA allocation whose caller supplied an
   * incorrect HOST tag. The structural layer must validate only its descriptor
   * and address range. Do not pass this intentionally unreadable address to the
   * host semantic layer: the future CUDA bridge first verifies pointer type and
   * ownership, then rejects the incorrect tag without touching the allocation.
   */
  fixture.batch.atom_offsets.data = reinterpret_cast<const void*>(std::uintptr_t{0x10000u});
  DescriptorValidationResult checked = validate_compute_descriptor_structure(
      XTBLOOM_BACKEND_CUDA, &fixture.batch, &fixture.options, &fixture.result);
  CHECK(checked.ok());
  CHECK(!checked.requires_backend_staging_validation());

  /* Device tags remain opaque through both common validation layers. */
  fixture.batch.atom_offsets.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  fixture.batch.atomic_numbers.data = reinterpret_cast<const void*>(std::uintptr_t{0x20000u});
  fixture.batch.atomic_numbers.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  fixture.batch.molecular_charges.data = reinterpret_cast<const void*>(std::uintptr_t{0x30000u});
  fixture.batch.molecular_charges.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  fixture.batch.unpaired_electrons.data = reinterpret_cast<const void*>(std::uintptr_t{0x40000u});
  fixture.batch.unpaired_electrons.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(XTBLOOM_BACKEND_CUDA, &fixture.batch, &fixture.options,
                                         &fixture.result);
  CHECK(checked.ok());
  CHECK(checked.pending_offset_checks ==
        (kAtomOffsetsNeedStaging | kAtomicNumbersNeedStaging | kMolecularChargesNeedStaging |
         kUnpairedElectronsNeedStaging));
  return true;
}

bool test_device_and_mixed_topology_pending_sets() {
  Fixture device_fixture;
  device_fixture.enable_point_charges();
  device_fixture.enable_response();
  device_fixture.enable_spin_channels();

  /* Distinct opaque addresses exercise every topology staging category. */
  device_fixture.batch.atom_offsets = {reinterpret_cast<const void*>(std::uintptr_t{0x10000u}),
                                       device_fixture.batch.atom_offsets.size_bytes,
                                       XTBLOOM_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.atomic_numbers = {reinterpret_cast<const void*>(std::uintptr_t{0x20000u}),
                                         device_fixture.batch.atomic_numbers.size_bytes,
                                         XTBLOOM_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.molecular_charges = {reinterpret_cast<const void*>(std::uintptr_t{0x30000u}),
                                            device_fixture.batch.molecular_charges.size_bytes,
                                            XTBLOOM_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.unpaired_electrons = {
      reinterpret_cast<const void*>(std::uintptr_t{0x40000u}),
      device_fixture.batch.unpaired_electrons.size_bytes, XTBLOOM_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.spin_channels = {reinterpret_cast<const void*>(std::uintptr_t{0x48000u}),
                                        device_fixture.batch.spin_channels.size_bytes,
                                        XTBLOOM_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.point_charge_offsets = {
      reinterpret_cast<const void*>(std::uintptr_t{0x50000u}),
      device_fixture.batch.point_charge_offsets.size_bytes, XTBLOOM_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.charge_response_offsets = {
      reinterpret_cast<const void*>(std::uintptr_t{0x60000u}),
      device_fixture.batch.charge_response_offsets.size_bytes, XTBLOOM_MEMORY_CUDA_DEVICE, 0};

  DescriptorValidationResult checked = validate_compute_descriptors(
      XTBLOOM_BACKEND_CUDA, &device_fixture.batch, &device_fixture.options, &device_fixture.result);
  CHECK(checked.ok());
  CHECK((checked.pending_offset_checks & kTopologyMetadataStagingMask) ==
        kTopologyMetadataStagingMask);
  CHECK(checked.pending_offset_checks ==
        (kTopologyMetadataStagingMask | kChargeResponseShapeNeedsStaging));

  Fixture mixed_fixture;
  mixed_fixture.enable_point_charges();
  mixed_fixture.enable_response();
  mixed_fixture.batch.atomic_numbers.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  mixed_fixture.batch.unpaired_electrons.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  mixed_fixture.batch.point_charge_offsets.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(XTBLOOM_BACKEND_CUDA, &mixed_fixture.batch,
                                         &mixed_fixture.options, &mixed_fixture.result);
  CHECK(checked.ok());
  CHECK(
      checked.pending_offset_checks ==
      (kAtomicNumbersNeedStaging | kUnpairedElectronsNeedStaging | kPointChargeOffsetsNeedStaging));

  /* HOST atom offsets can prove the total shape while response offsets stage. */
  mixed_fixture.batch.point_charge_offsets.memory_space = XTBLOOM_MEMORY_HOST;
  mixed_fixture.batch.charge_response_offsets.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(XTBLOOM_BACKEND_CUDA, &mixed_fixture.batch,
                                         &mixed_fixture.options, &mixed_fixture.result);
  CHECK(checked.ok());
  CHECK(checked.pending_offset_checks ==
        (kAtomicNumbersNeedStaging | kUnpairedElectronsNeedStaging |
         kChargeResponseOffsetsNeedStaging | kChargeResponseShapeNeedsStaging));
  return true;
}

bool test_host_semantics_remain_separate_and_unchanged() {
  Fixture fixture;
  fixture.atom_offsets[1] = 0;

  DescriptorValidationResult structure = validate_compute_descriptor_structure(
      XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options, &fixture.result);
  CHECK(structure.ok());
  CHECK(structure.pending_offset_checks == 0u);

  DescriptorValidationResult semantics = validate_host_topology_semantics(fixture.batch);
  CHECK(semantics.status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(semantics.error == "atom_offsets must be strictly increasing");

  /*
   * The complete structural entry still detects aliases, but the composed CPU
   * sequence preserves the historical topology-before-alias error priority.
   */
  fixture.result.energies.data = fixture.molecular_charges.data();
  fixture.result.energies.size_bytes = fixture.molecular_charges.size() * sizeof(double);
  structure = validate_compute_descriptor_structure(XTBLOOM_BACKEND_CPU, &fixture.batch,
                                                    &fixture.options, &fixture.result);
  CHECK(structure.status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(structure.error.find("aliases") != std::string::npos);

  DescriptorValidationResult composed = validate_compute_descriptors(
      XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options, &fixture.result);
  CHECK(composed.status == semantics.status);
  CHECK(composed.error == semantics.error);
  return true;
}

bool test_headers_counts_and_overflow() {
  const std::vector<InvalidCase> cases = {
      {"batch ABI size", [](Fixture& f) { f.batch.struct_size = XTBLOOM_BATCH_V1_SIZE - 1; },
       "batch is"},
      {"options ABI size",
       [](Fixture& f) { f.options.struct_size = XTBLOOM_COMPUTE_OPTIONS_V1_SIZE - 1; },
       "compute options"},
      {"options ABI version", [](Fixture& f) { f.options.api_version += 1; }, "compute options"},
      {"result ABI version", [](Fixture& f) { f.result.api_version += 1; }, "batch result"},
      {"empty batch", [](Fixture& f) { f.batch.batch_size = 0; }, "batch_size"},
      {"too few atoms", [](Fixture& f) { f.batch.total_atoms = 1; }, "at least one atom"},
      {"negative point count", [](Fixture& f) { f.batch.total_point_charges = -1; },
       "total_point_charges"},
      {"negative response count", [](Fixture& f) { f.batch.total_charge_response_elements = -1; },
       "total_charge_response_elements"},
      {"batch byte overflow",
       [](Fixture& f) {
         f.batch.batch_size = INT64_MAX;
         f.batch.total_atoms = INT64_MAX;
       },
       "overflows"},
      {"atom byte overflow",
       [](Fixture& f) {
         f.batch.batch_size = 1;
         f.batch.total_atoms = INT64_MAX;
       },
       "overflows"},
      {"point byte overflow", [](Fixture& f) { f.batch.total_point_charges = INT64_MAX; },
       "overflows"},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }

  Fixture fixture;
  CHECK(
      validate_compute_descriptors(XTBLOOM_BACKEND_CPU, nullptr, &fixture.options, &fixture.result)
          .status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, nullptr, &fixture.result)
            .status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options, nullptr)
            .status == XTBLOOM_STATUS_INVALID_ARGUMENT);
  return true;
}

bool test_compute_options() {
  const std::vector<InvalidCase> cases = {
      {"unknown model", [](Fixture& f) { f.options.model = static_cast<xtbloom_model_t>(99); },
       "model"},
      {"zero flags", [](Fixture& f) { f.options.flags = 0; }, "compute flags"},
      {"unknown flag", [](Fixture& f) { f.options.flags |= 1u << 31; }, "compute flags"},
      {"zero SCC iterations", [](Fixture& f) { f.options.max_scc_iterations = 0; },
       "max_scc_iterations"},
      {"negative SCC iterations", [](Fixture& f) { f.options.max_scc_iterations = -2; },
       "max_scc_iterations"},
      {"zero charge tolerance", [](Fixture& f) { f.options.charge_tolerance = 0.0; },
       "charge_tolerance"},
      {"NaN charge tolerance",
       [](Fixture& f) { f.options.charge_tolerance = std::numeric_limits<double>::quiet_NaN(); },
       "charge_tolerance"},
      {"infinite energy tolerance",
       [](Fixture& f) { f.options.energy_tolerance = std::numeric_limits<double>::infinity(); },
       "energy_tolerance"},
      {"negative electronic temperature",
       [](Fixture& f) { f.options.electronic_temperature = -1.0; }, "electronic_temperature"},
      {"NaN electronic temperature",
       [](Fixture& f) {
         f.options.electronic_temperature = std::numeric_limits<double>::quiet_NaN();
       },
       "electronic_temperature"},
      {"options reserved", [](Fixture& f) { f.options.reserved = 1; }, "reserved"},
      {"zero SCC start mode", [](Fixture& f) { f.options.scc_start_mode = 0; }, "scc_start_mode"},
      {"negative SCC start mode", [](Fixture& f) { f.options.scc_start_mode = -1; },
       "scc_start_mode"},
      {"unknown SCC start mode", [](Fixture& f) { f.options.scc_start_mode = 3; },
       "scc_start_mode"},
      {"maximum SCC start mode",
       [](Fixture& f) { f.options.scc_start_mode = std::numeric_limits<std::int32_t>::max(); },
       "scc_start_mode"},
      {"options ABI-v2 reserved", [](Fixture& f) { f.options.reserved_v2 = 1; }, "reserved_v2"},
      {"zero SCC mixer", [](Fixture& f) { f.options.scc_mixer = 0; }, "scc_mixer"},
      {"unknown SCC mixer", [](Fixture& f) { f.options.scc_mixer = 2; }, "scc_mixer"},
      {"zero SCC mixer history", [](Fixture& f) { f.options.scc_mixer_history = 0; },
       "scc_mixer_history"},
      {"negative SCC mixer history", [](Fixture& f) { f.options.scc_mixer_history = -1; },
       "scc_mixer_history"},
      {"oversized SCC mixer history", [](Fixture& f) { f.options.scc_mixer_history = 65; },
       "scc_mixer_history"},
      {"zero SCC mixer damping", [](Fixture& f) { f.options.scc_mixer_damping = 0.0; },
       "scc_mixer_damping"},
      {"negative SCC mixer damping", [](Fixture& f) { f.options.scc_mixer_damping = -0.1; },
       "scc_mixer_damping"},
      {"oversized SCC mixer damping", [](Fixture& f) { f.options.scc_mixer_damping = 1.01; },
       "scc_mixer_damping"},
      {"infinite SCC mixer damping",
       [](Fixture& f) { f.options.scc_mixer_damping = std::numeric_limits<double>::infinity(); },
       "scc_mixer_damping"},
      {"NaN SCC mixer damping",
       [](Fixture& f) { f.options.scc_mixer_damping = std::numeric_limits<double>::quiet_NaN(); },
       "scc_mixer_damping"},
      {"negative determinism", [](Fixture& f) { f.options.determinism = -1; }, "determinism"},
      {"unknown determinism", [](Fixture& f) { f.options.determinism = 2; }, "determinism"},
      {"options ABI-v3 reserved", [](Fixture& f) { f.options.reserved_v3 = 1; }, "reserved_v3"},
      {"result reserved", [](Fixture& f) { f.result.reserved = 1; }, "reserved"},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }

  Fixture fixture;
  fixture.options.model = XTBLOOM_MODEL_GFN1_XTB;  // Known ABI model; dispatch decides support.
  fixture.options.electronic_temperature = 0.0;
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());

  fixture.options.model = XTBLOOM_MODEL_GFN2_XTB;
  fixture.options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  fixture.options.scc_mixer_history = 64;
  fixture.options.scc_mixer_damping = 1.0;
  fixture.options.determinism = XTBLOOM_DETERMINISM_REPRODUCIBLE;
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CUDA, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());
  return true;
}

bool test_compute_options_short_prefixes() {
  Fixture fixture;
  constexpr std::size_t kCanaryBytes = 16;
  for (const std::size_t caller_size :
       {static_cast<std::size_t>(XTBLOOM_COMPUTE_OPTIONS_V1_SIZE),
        static_cast<std::size_t>(XTBLOOM_COMPUTE_OPTIONS_V2_SIZE - 1),
        static_cast<std::size_t>(XTBLOOM_COMPUTE_OPTIONS_V2_SIZE),
        static_cast<std::size_t>(XTBLOOM_COMPUTE_OPTIONS_V2_SIZE + 1), std::size_t{63},
        std::size_t{64}, std::size_t{71}, std::size_t{72},
        static_cast<std::size_t>(XTBLOOM_COMPUTE_OPTIONS_V3_SIZE - 1)}) {
    std::unique_ptr<unsigned char, decltype(&std::free)> storage(
        static_cast<unsigned char*>(std::malloc(caller_size + kCanaryBytes)), &std::free);
    CHECK(storage != nullptr);
    std::memset(storage.get(), 0xa5, caller_size + kCanaryBytes);
    std::memcpy(storage.get(), &fixture.options, caller_size);
    /* Make every byte in an incomplete suffix hostile. Validation must gate
     * the suffix by its complete size rather than by individual field offsets. */
    const std::size_t complete_prefix = caller_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE
                                            ? XTBLOOM_COMPUTE_OPTIONS_V2_SIZE
                                            : XTBLOOM_COMPUTE_OPTIONS_V1_SIZE;
    std::memset(storage.get() + complete_prefix, 0xa5, caller_size - complete_prefix);
    const std::uint32_t encoded_size = static_cast<std::uint32_t>(caller_size);
    std::memcpy(storage.get(), &encoded_size, sizeof(encoded_size));

    const auto* short_options = reinterpret_cast<const xtbloom_compute_options_t*>(storage.get());
    const DescriptorValidationResult checked = validate_compute_descriptors(
        XTBLOOM_BACKEND_CPU, &fixture.batch, short_options, &fixture.result);
    CHECK(checked.ok());
    for (std::size_t index = caller_size; index < caller_size + kCanaryBytes; ++index) {
      CHECK(storage.get()[index] == 0xa5);
    }
  }
  return true;
}

bool test_buffer_descriptors_and_sizes() {
  const std::vector<InvalidCase> cases = {
      {"atomic number undersize", [](Fixture& f) { --f.batch.atomic_numbers.size_bytes; },
       "atomic_numbers"},
      {"position undersize", [](Fixture& f) { f.batch.positions.size_bytes -= sizeof(double); },
       "positions"},
      {"NULL with nonzero size", [](Fixture& f) { f.batch.positions.data = nullptr; }, "positions"},
      {"molecular charge undersize",
       [](Fixture& f) { f.batch.molecular_charges.size_bytes = sizeof(double); },
       "molecular_charges"},
      {"unpaired electron undersize",
       [](Fixture& f) { f.batch.unpaired_electrons.size_bytes = sizeof(std::int32_t); },
       "unpaired_electrons"},
      {"input reserved", [](Fixture& f) { f.batch.positions.reserved = 1; }, "reserved"},
      {"output reserved", [](Fixture& f) { f.result.forces.reserved = 1; }, "reserved"},
      {"unknown memory space",
       [](Fixture& f) { f.batch.positions.memory_space = static_cast<xtbloom_memory_space_t>(99); },
       "memory_space"},
      {"CPU rejects CUDA pointer",
       [](Fixture& f) { f.batch.positions.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE; }, "CPU"},
      {"CUDA rejects ROCm pointer",
       [](Fixture& f) { f.batch.positions.memory_space = XTBLOOM_MEMORY_ROCM_DEVICE; }, "ROCm",
       XTBLOOM_STATUS_NOT_SUPPORTED, XTBLOOM_BACKEND_CUDA},
      {"unresolved AUTO backend", [](Fixture&) {}, "resolved backend",
       XTBLOOM_STATUS_INVALID_ARGUMENT, XTBLOOM_BACKEND_AUTO},
      {"reserved ROCm backend", [](Fixture&) {}, "ROCm backend", XTBLOOM_STATUS_NOT_SUPPORTED,
       XTBLOOM_BACKEND_ROCM},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }

  Fixture fixture;
  fixture.batch.point_charge_positions.memory_space = XTBLOOM_MEMORY_ROCM_DEVICE;
  /* A NULL zero-sized optional field is not a device pointer and remains legal. */
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());

  /* Oversized capacities and an under-aligned but byte-addressable offset view are valid. */
  fixture.batch.positions.size_bytes += 256;
  std::array<unsigned char, 3 * sizeof(std::int64_t) + 1> unaligned{};
  std::memcpy(unaligned.data() + 1, fixture.atom_offsets.data(), 3 * sizeof(std::int64_t));
  fixture.batch.atom_offsets.data = unaligned.data() + 1;
  fixture.batch.atom_offsets.size_bytes = 3 * sizeof(std::int64_t);
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());
  return true;
}

bool test_spin_channel_abi_v2() {
  Fixture legacy;
  legacy.batch.struct_size = XTBLOOM_BATCH_V1_SIZE;
  legacy.batch.spin_channels = {reinterpret_cast<const void*>(std::uintptr_t{0x10000u}),
                                std::numeric_limits<std::size_t>::max(),
                                static_cast<xtbloom_memory_space_t>(99), 1};
  /* ABI-v1 callers do not expose the suffix, so its out-of-range bytes stay unread. */
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &legacy.batch, &legacy.options,
                                     &legacy.result)
            .ok());

  Fixture valid;
  valid.enable_spin_channels();
  CHECK(valid.batch.struct_size == XTBLOOM_BATCH_V2_SIZE);
  CHECK(
      validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &valid.batch, &valid.options, &valid.result)
          .ok());
  valid.batch.spin_channels.size_bytes += 64;
  CHECK(
      validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &valid.batch, &valid.options, &valid.result)
          .ok());
  valid.batch.spin_channels.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
  const DescriptorValidationResult staged_spin = validate_host_topology_semantics(valid.batch);
  CHECK(staged_spin.ok());
  CHECK(staged_spin.pending_offset_checks == kSpinChannelsNeedStaging);
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CUDA, &valid.batch, &valid.options,
                                     &valid.result)
            .ok());

  const std::vector<InvalidCase> cases = {
      {"spin channel undersize",
       [](Fixture& f) {
         f.enable_spin_channels();
         f.batch.spin_channels.size_bytes = sizeof(std::int32_t);
       },
       "spin_channels"},
      {"spin channel reserved",
       [](Fixture& f) {
         f.enable_spin_channels();
         f.batch.spin_channels.reserved = 1;
       },
       "reserved"},
      {"CPU rejects device spin channels",
       [](Fixture& f) {
         f.enable_spin_channels();
         f.batch.spin_channels.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
       },
       "context backend is CPU"},
      {"zero spin channels", [](Fixture& f) { f.enable_spin_channels({1, 0}); }, "one or two"},
      {"three spin channels", [](Fixture& f) { f.enable_spin_channels({3, 1}); }, "one or two"},
      {"spin channels alias output",
       [](Fixture& f) {
         f.enable_spin_channels();
         f.result.scc_iterations.data = f.spin_channels.data();
         f.result.scc_iterations.size_bytes = f.spin_channels.size() * sizeof(std::int32_t);
       },
       "aliases"},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }
  return true;
}

bool test_required_and_optional_outputs() {
  const std::vector<InvalidCase> cases = {
      {"requested energy missing",
       [](Fixture& f) {
         f.result.energies.data = nullptr;
         f.result.energies.size_bytes = 0;
       },
       "energies"},
      {"requested forces undersized",
       [](Fixture& f) { f.result.forces.size_bytes -= sizeof(double); }, "forces"},
      {"requested charges missing",
       [](Fixture& f) {
         f.options.flags |= XTBLOOM_COMPUTE_ATOMIC_CHARGES;
         f.result.atomic_charges = {};
       },
       "atomic_charges"},
      {"SCC iterations missing", [](Fixture& f) { f.result.scc_iterations = {}; },
       "scc_iterations"},
      {"SCC convergence missing", [](Fixture& f) { f.result.scc_converged = {}; }, "scc_converged"},
      {"per-system status missing", [](Fixture& f) { f.result.per_system_status = {}; },
       "per_system_status"},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }

  Fixture fixture;
  fixture.options.flags = XTBLOOM_COMPUTE_FORCES;
  fixture.result.energies = {};
  fixture.result.atomic_charges = {};
  fixture.result.point_charge_forces = {};
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());

  /* Unrequested output descriptors are not written and do not need capacity. */
  fixture.result.energies = {fixture.molecular_charges.data(), 1, XTBLOOM_MEMORY_HOST, 0};
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());
  return true;
}

bool test_point_charge_shapes() {
  const std::vector<InvalidCase> cases = {
      {"missing point offsets",
       [](Fixture& f) {
         f.enable_point_charges();
         f.batch.point_charge_offsets = {};
       },
       "point_charge_offsets"},
      {"missing point positions",
       [](Fixture& f) {
         f.enable_point_charges();
         f.batch.point_charge_positions = {};
       },
       "point_charge_positions"},
      {"missing point values",
       [](Fixture& f) {
         f.enable_point_charges();
         f.batch.point_charge_values = {};
       },
       "point_charge_values"},
      {"missing point gammas",
       [](Fixture& f) {
         f.enable_point_charges();
         f.batch.point_charge_gammas = {};
       },
       "point_charge_gammas"},
      {"point offsets not monotonic",
       [](Fixture& f) {
         f.enable_point_charges();
         f.point_offsets[2] = 0;
       },
       "monotonically"},
      {"point offset endpoint",
       [](Fixture& f) {
         f.enable_point_charges();
         f.point_offsets[1] = 0;
         f.point_offsets[2] = 0;
       },
       "endpoint"},
      {"requested point force undersized",
       [](Fixture& f) {
         f.enable_point_charges();
         f.options.flags |= XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
         f.result.point_charge_forces.size_bytes -= sizeof(double);
       },
       "point_charge_forces"},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }

  Fixture fixture;
  fixture.enable_point_charges();
  fixture.options.flags |= XTBLOOM_COMPUTE_POINT_CHARGE_FORCES;
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());

  /* Read-only input aliases are harmless; outputs are checked separately below. */
  fixture.batch.point_charge_values.data = fixture.molecular_charges.data();
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());
  return true;
}

bool test_host_offsets_and_response_packing() {
  const std::vector<InvalidCase> cases = {
      {"atom offsets nonzero start", [](Fixture& f) { f.atom_offsets[0] = 1; }, "begin with zero"},
      {"empty atom segment", [](Fixture& f) { f.atom_offsets[1] = 0; }, "strictly increasing"},
      {"atom offsets decreasing",
       [](Fixture& f) {
         f.atom_offsets[1] = 3;
         f.atom_offsets[2] = 2;
       },
       "strictly increasing"},
      {"atom offset endpoint",
       [](Fixture& f) {
         f.atom_offsets[1] = 1;
         f.atom_offsets[2] = 2;
       },
       "endpoint"},
      {"response missing matrix",
       [](Fixture& f) {
         f.enable_response();
         f.batch.charge_response_matrix = {};
       },
       "supplied together"},
      {"response missing offsets",
       [](Fixture& f) {
         f.enable_response();
         f.batch.charge_response_offsets = {};
       },
       "supplied together"},
      {"response matrix undersized",
       [](Fixture& f) {
         f.enable_response();
         f.batch.charge_response_matrix.size_bytes -= sizeof(double);
       },
       "charge_response_matrix"},
      {"response offset endpoint",
       [](Fixture& f) {
         f.enable_response();
         f.response_offsets[2] = 4;
       },
       "endpoint"},
      {"response per-system shape",
       [](Fixture& f) {
         f.enable_response();
         f.response_offsets[1] = 3;
       },
       "square matrix"},
      {"response total shape with staged offsets",
       [](Fixture& f) {
         f.enable_response();
         f.batch.total_charge_response_elements = 4;
         f.response_matrix.resize(4);
         f.batch.charge_response_matrix = input_buffer(f.response_matrix);
         f.batch.charge_response_offsets.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
       },
       "total_charge_response_elements", XTBLOOM_STATUS_INVALID_ARGUMENT, XTBLOOM_BACKEND_CUDA},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }

  Fixture fixture;
  fixture.enable_response();
  fixture.potential_shifts.assign(3, 0.25);
  fixture.batch.atomic_potential_shifts = input_buffer(fixture.potential_shifts);
  CHECK(validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());
  return true;
}

bool test_aliasing_and_address_ranges() {
  const std::vector<InvalidCase> cases = {
      {"energy aliases input",
       [](Fixture& f) {
         f.result.energies.data = f.molecular_charges.data();
         f.result.energies.size_bytes = 2 * sizeof(double);
       },
       "aliases"},
      {"force aliases positions",
       [](Fixture& f) {
         f.result.forces.data = f.positions.data();
         f.result.forces.size_bytes = f.positions.size() * sizeof(double);
       },
       "aliases"},
      {"outputs alias each other",
       [](Fixture& f) {
         f.result.scc_iterations.data = f.energies.data();
         f.result.scc_iterations.size_bytes = 2 * sizeof(std::int32_t);
       },
       "aliases"},
      {"same pointer incompatible tags",
       [](Fixture& f) {
         f.result.energies.data = f.molecular_charges.data();
         f.result.energies.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
       },
       "incompatible memory-space", XTBLOOM_STATUS_INVALID_ARGUMENT, XTBLOOM_BACKEND_CUDA},
      {"output address overflow",
       [](Fixture& f) {
         f.result.energies.data =
             reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - sizeof(double));
       },
       "overflows uintptr_t"},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }
  return true;
}

/* Build the released electric-field payload block (block_version 1): one
 * int32_t version, one int32_t reserved, three doubles. */
std::vector<std::uint8_t> efield_block(double ex, double ey, double ez) {
  std::vector<std::uint8_t> block(32, 0);
  const std::int32_t version = 1;
  std::memcpy(block.data(), &version, sizeof(version));
  std::memcpy(block.data() + 8, &ex, sizeof(ex));
  std::memcpy(block.data() + 16, &ey, sizeof(ey));
  std::memcpy(block.data() + 24, &ez, sizeof(ez));
  return block;
}

bool test_interaction_abi_v3() {
  /* A v3-struct batch with zero interactions is a plain v1/v2 request. */
  Fixture fixture;
  fixture.batch.struct_size = sizeof(fixture.batch);
  fixture.batch.total_interactions = 0;
  DescriptorValidationResult checked = validate_compute_descriptors(
      XTBLOOM_BACKEND_CPU, &fixture.batch, &fixture.options, &fixture.result);
  CHECK(checked.ok());
  CHECK(!checked.requires_backend_staging_validation());

  xtbloom_interaction_t interaction{};
  interaction.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
  interaction.system_index = 0;
  interaction.payload_offset = 0;
  interaction.payload_size = 32;

  /* A well-formed electric-field attachment passes CPU structural validation;
   * both released backends execute the field after their pointer gates. */
  {
    Fixture field;
    field.enable_interaction(interaction, efield_block(0.0, 0.0, 0.1));
    checked = validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &field.batch, &field.options,
                                           &field.result);
    CHECK(checked.ok());
    CHECK(!checked.requires_backend_staging_validation());
  }

  /* The CUDA structure-only entry point accepts a host-readable released field;
   * the runtime performs pointer provenance and executes the term. */
  {
    Fixture field;
    field.enable_interaction(interaction, efield_block(0.0, 0.0, 0.1));
    checked = validate_compute_descriptor_structure(XTBLOOM_BACKEND_CUDA, &field.batch,
                                                    &field.options, &field.result);
    CHECK(checked.ok());
  }

  const InvalidCase invalid_interactions[] = {
      {"interactions with no descriptor array",
       [](Fixture& f) {
         f.batch.total_interactions = 1;
         f.batch.struct_size = sizeof(f.batch);
       },
       "interaction_descriptors is required"},
      {"interactions with no payload store",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
         entry.system_index = 0;
         entry.payload_size = 32;
         f.interactions = {entry};
         f.batch.struct_size = sizeof(f.batch);
         f.batch.total_interactions = 1;
         f.batch.interaction_descriptors = input_buffer(f.interactions);
       },
       "interaction_payload is required"},
      {"negative interaction count",
       [](Fixture& f) {
         f.batch.struct_size = sizeof(f.batch);
         f.batch.total_interactions = -1;
       },
       "total_interactions must be nonnegative"},
      {"unknown interaction tag",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = 0x7fffff;
         entry.system_index = 0;
         entry.payload_size = 32;
         f.enable_interaction(entry, efield_block(0.0, 0.0, 0.0));
       },
       "unknown or NONE type tag"},
      {"NONE interaction tag",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_NONE;
         entry.system_index = 0;
         entry.payload_size = 32;
         f.enable_interaction(entry, efield_block(0.0, 0.0, 0.0));
       },
       "unknown or NONE type tag"},
      {"flags must be zero",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
         entry.flags = 1u;
         entry.system_index = 0;
         entry.payload_size = 32;
         f.enable_interaction(entry, efield_block(0.0, 0.0, 0.0));
       },
       "flags must be zero"},
      {"system index out of range",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
         entry.system_index = 7;
         entry.payload_size = 32;
         f.enable_interaction(entry, efield_block(0.0, 0.0, 0.0));
       },
       "system_index lies outside"},
      {"payload block past the payload view",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
         entry.system_index = 0;
         entry.payload_offset = std::numeric_limits<std::uint64_t>::max() - 16;
         entry.payload_size = 32;
         f.enable_interaction(entry, efield_block(0.0, 0.0, 0.0));
       },
       "extends past interaction_payload"},
      {"electric-field payload undersized",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
         entry.system_index = 0;
         entry.payload_size = 16;
         f.enable_interaction(entry, efield_block(0.0, 0.0, 0.0));
       },
       "undersized or misaligned"},
      {"electric-field payload oversized",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
         entry.system_index = 0;
         entry.payload_size = 40;
         f.enable_interaction(entry, std::vector<std::uint8_t>(64, 0));
       },
       "exceeds the released contract"},
      {"electric-field payload misaligned",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
         entry.system_index = 0;
         entry.payload_offset = 1;
         entry.payload_size = 32;
         f.enable_interaction(entry, std::vector<std::uint8_t>(64, 0));
       },
       "undersized or misaligned"},
      {"electric-field payload view base misaligned",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
         entry.system_index = 0;
         entry.payload_size = 32;
         const std::vector<std::uint8_t> payload = efield_block(0.0, 0.0, 0.0);
         std::memcpy(f.misaligned_interaction_payload.data() + 4u, payload.data(), payload.size());
         f.interactions = {entry};
         f.batch.struct_size = sizeof(f.batch);
         f.batch.total_interactions = 1;
         f.batch.interaction_descriptors = input_buffer(f.interactions);
         f.batch.interaction_payload = {f.misaligned_interaction_payload.data() + 4u, 32u,
                                        XTBLOOM_MEMORY_HOST, 0u};
       },
       "undersized or misaligned"},
      {"electric-field payload version unsupported",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
         entry.system_index = 0;
         entry.payload_size = 32;
         std::vector<std::uint8_t> payload = efield_block(0.0, 0.0, 0.0);
         const std::int32_t version = 2;
         std::memcpy(payload.data(), &version, sizeof(version));
         f.enable_interaction(entry, payload);
       },
       "unsupported block_version"},
      {"electric-field payload reserved field nonzero",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
         entry.system_index = 0;
         entry.payload_size = 32;
         std::vector<std::uint8_t> payload = efield_block(0.0, 0.0, 0.0);
         const std::int32_t reserved = 1;
         std::memcpy(payload.data() + sizeof(std::int32_t), &reserved, sizeof(reserved));
         f.enable_interaction(entry, payload);
       },
       "reserved payload field must be zero"},
      {"electric-field payload non-finite",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
         entry.system_index = 0;
         entry.payload_size = 32;
         f.enable_interaction(entry,
                              efield_block(std::numeric_limits<double>::infinity(), 0.0, 0.0));
       },
       "contains NaN or infinity"},
      {"reserved interaction lacks block-version header",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ALPB_SOLVATION;
         entry.system_index = 0;
         entry.payload_size = 3;
         f.enable_interaction(entry, std::vector<std::uint8_t>(3, 0));
       },
       "aligned block_version"},
      {"reserved interaction block-version header misaligned",
       [](Fixture& f) {
         xtbloom_interaction_t entry{};
         entry.type = XTBLOOM_INTERACTION_ALPB_SOLVATION;
         entry.system_index = 0;
         entry.payload_offset = 2;
         entry.payload_size = 4;
         f.enable_interaction(entry, std::vector<std::uint8_t>(8, 0));
       },
       "aligned block_version"},
      {"duplicate interaction on one system",
       [](Fixture& f) {
         xtbloom_interaction_t entries[2] = {};
         for (xtbloom_interaction_t& entry : entries) {
           entry.type = XTBLOOM_INTERACTION_ELECTRIC_FIELD;
           entry.system_index = 0;
           entry.payload_size = 32;
         }
         entries[1].payload_offset = 32;
         f.interactions.assign(std::begin(entries), std::end(entries));
         f.interaction_payload = efield_block(0.0, 0.0, 0.0);
         const std::vector<std::uint8_t> second_payload = efield_block(0.0, 0.0, 0.0);
         f.interaction_payload.insert(f.interaction_payload.end(), second_payload.begin(),
                                      second_payload.end());
         f.batch.struct_size = sizeof(f.batch);
         f.batch.total_interactions = 2;
         f.batch.interaction_descriptors = input_buffer(f.interactions);
         f.batch.interaction_payload = input_buffer(f.interaction_payload);
       },
       "two attachments of the same interaction type"},
  };
  for (const InvalidCase& test : invalid_interactions) {
    CHECK(expect_invalid(test));
  }

  /* A reserved-but-unimplemented tag with valid structure is refused as
   * NOT_IMPLEMENTED after structural validation instead of INVALID_ARGUMENT. */
  {
    Fixture reserved;
    xtbloom_interaction_t entry{};
    entry.type = XTBLOOM_INTERACTION_ALPB_SOLVATION;
    entry.system_index = 0;
    entry.payload_size = 32;
    reserved.enable_interaction(entry, efield_block(0.0, 0.0, 0.0));
    checked = validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &reserved.batch, &reserved.options,
                                           &reserved.result);
    CHECK(checked.status == XTBLOOM_STATUS_NOT_IMPLEMENTED);
  }

  /* Dipole-moment publication is released on both GFN2 backends; requesting
   * it with a correctly sized outlet is accepted structurally. */
  {
    Fixture dipole;
    dipole.options.flags |= XTBLOOM_COMPUTE_DIPOLE_MOMENTS;
    std::vector<double> dipole_output(6, 0.0);
    dipole.result.struct_size = sizeof(dipole.result);
    dipole.result.dipole_moments = output_buffer(dipole_output);
    checked = validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &dipole.batch, &dipole.options,
                                           &dipole.result);
    CHECK(checked.ok());

    Fixture cuda_dipole;
    cuda_dipole.options.flags |= XTBLOOM_COMPUTE_DIPOLE_MOMENTS;
    std::vector<double> cuda_output(6, 0.0);
    cuda_dipole.result.struct_size = sizeof(cuda_dipole.result);
    cuda_dipole.result.dipole_moments = output_buffer(cuda_output);
    checked = validate_compute_descriptor_structure(XTBLOOM_BACKEND_CUDA, &cuda_dipole.batch,
                                                    &cuda_dipole.options, &cuda_dipole.result);
    CHECK(checked.ok());
  }

  /* Validate the released outlet shape before backend execution. */
  {
    Fixture missing_dipole;
    missing_dipole.options.flags |= XTBLOOM_COMPUTE_DIPOLE_MOMENTS;
    checked = validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &missing_dipole.batch,
                                           &missing_dipole.options, &missing_dipole.result);
    CHECK(checked.status == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(checked.error.find("dipole_moments is required") != std::string::npos);

    Fixture short_dipole;
    short_dipole.options.flags |= XTBLOOM_COMPUTE_DIPOLE_MOMENTS;
    alignas(xtbloom_batch_result_t) std::array<unsigned char, XTBLOOM_BATCH_RESULT_V1_SIZE>
        short_result_storage{};
    std::memcpy(short_result_storage.data(), &short_dipole.result, short_result_storage.size());
    auto* short_result = reinterpret_cast<xtbloom_batch_result_t*>(short_result_storage.data());
    short_result->struct_size = XTBLOOM_BATCH_RESULT_V1_SIZE;
    checked = validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &short_dipole.batch,
                                           &short_dipole.options, short_result);
    CHECK(checked.status == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(checked.error.find("dipole_moments is required") != std::string::npos);

    Fixture undersized_dipole;
    undersized_dipole.options.flags |= XTBLOOM_COMPUTE_DIPOLE_MOMENTS;
    std::vector<double> output(5, 0.0);
    undersized_dipole.result.struct_size = sizeof(undersized_dipole.result);
    undersized_dipole.result.dipole_moments = output_buffer(output);
    checked = validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &undersized_dipole.batch,
                                           &undersized_dipole.options, &undersized_dipole.result);
    CHECK(checked.status == XTBLOOM_STATUS_INVALID_ARGUMENT);
    CHECK(checked.error.find("dipole_moments is smaller") != std::string::npos);
  }

  /* A reserved result outlet with no released shape contract is refused. */
  {
    Fixture quadrupole;
    std::vector<double> quadrupole_output(12, 0.0);
    quadrupole.result.struct_size = sizeof(quadrupole.result);
    quadrupole.result.quadrupole_moments = output_buffer(quadrupole_output);
    checked = validate_compute_descriptors(XTBLOOM_BACKEND_CPU, &quadrupole.batch,
                                           &quadrupole.options, &quadrupole.result);
    CHECK(checked.status == XTBLOOM_STATUS_NOT_SUPPORTED);
    CHECK(checked.error.find("reserved batch-result outlet") != std::string::npos);
  }

  /* Device-resident descriptors defer content and reserved-tag checks through
   * the CUDA runtime's stream-ordered interaction gate. */
  {
    Fixture device_field;
    device_field.enable_interaction(interaction, efield_block(0.0, 0.0, 0.1));
    device_field.batch.interaction_descriptors.memory_space = XTBLOOM_MEMORY_CUDA_DEVICE;
    checked = validate_compute_descriptors(XTBLOOM_BACKEND_CUDA, &device_field.batch,
                                           &device_field.options, &device_field.result);
    CHECK(checked.ok());
    CHECK((checked.pending_offset_checks & xtbloom::detail::kInteractionDescriptorsNeedStaging) !=
          0u);
  }
  return true;
}

}  // namespace

int main() {
  const struct Test {
    const char* name;
    bool (*run)();
  } tests[] = {
      {"valid requests and staging contract", test_valid_requests_and_staging_contract},
      {"structural validation does not dereference storage",
       test_structural_layer_never_dereferences_buffer_storage},
      {"device and mixed topology pending sets", test_device_and_mixed_topology_pending_sets},
      {"host topology semantic split", test_host_semantics_remain_separate_and_unchanged},
      {"headers, counts, and overflow", test_headers_counts_and_overflow},
      {"compute options", test_compute_options},
      {"compute-options short prefixes", test_compute_options_short_prefixes},
      {"buffer descriptors and sizes", test_buffer_descriptors_and_sizes},
      {"ABI-v2 spin channels", test_spin_channel_abi_v2},
      {"required and optional outputs", test_required_and_optional_outputs},
      {"point-charge shapes", test_point_charge_shapes},
      {"host offsets and response packing", test_host_offsets_and_response_packing},
      {"aliasing and address ranges", test_aliasing_and_address_ranges},
      {"ABI-v3 interactions", test_interaction_abi_v3},
  };
  for (const Test& test : tests) {
    if (!test.run()) {
      std::cerr << "failed test group: " << test.name << '\n';
      return 1;
    }
  }
  return 0;
}
