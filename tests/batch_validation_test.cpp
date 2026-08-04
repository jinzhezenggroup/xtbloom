#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gpuxtb/gpuxtb.h"
#include "runtime/validation.hpp"

namespace {

using gpuxtb::detail::DescriptorValidationResult;
using gpuxtb::detail::kAtomicNumbersNeedStaging;
using gpuxtb::detail::kAtomOffsetsNeedStaging;
using gpuxtb::detail::kChargeResponseOffsetsNeedStaging;
using gpuxtb::detail::kChargeResponseShapeNeedsStaging;
using gpuxtb::detail::kMolecularChargesNeedStaging;
using gpuxtb::detail::kPointChargeOffsetsNeedStaging;
using gpuxtb::detail::kSpinChannelsNeedStaging;
using gpuxtb::detail::kTopologyMetadataStagingMask;
using gpuxtb::detail::kUnpairedElectronsNeedStaging;
using gpuxtb::detail::validate_compute_descriptor_structure;
using gpuxtb::detail::validate_compute_descriptors;
using gpuxtb::detail::validate_host_topology_semantics;

#define CHECK(condition)                                                                 \
  do {                                                                                   \
    if (!(condition)) {                                                                  \
      std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: " #condition << '\n'; \
      return false;                                                                      \
    }                                                                                    \
  } while (false)

template <typename T>
gpuxtb_const_buffer_t input_buffer(const std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), GPUXTB_MEMORY_HOST,
          0};
}

template <typename T>
gpuxtb_buffer_t output_buffer(std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), GPUXTB_MEMORY_HOST,
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

  std::vector<double> energies{0.0, 0.0};
  std::vector<double> forces = std::vector<double>(9);
  std::vector<double> atomic_charges = std::vector<double>(3);
  std::vector<double> point_forces;
  std::vector<std::int32_t> scc_iterations = std::vector<std::int32_t>(2);
  std::vector<std::uint8_t> scc_converged = std::vector<std::uint8_t>(2);
  std::vector<std::int32_t> per_system_status = std::vector<std::int32_t>(2);

  gpuxtb_batch_t batch{};
  gpuxtb_compute_options_t options{};
  gpuxtb_batch_result_t result{};

  Fixture() {
    batch.struct_size = sizeof(batch);
    batch.api_version = GPUXTB_API_VERSION;
    batch.batch_size = 2;
    batch.total_atoms = 3;
    batch.atom_offsets = input_buffer(atom_offsets);
    batch.atomic_numbers = input_buffer(atomic_numbers);
    batch.positions = input_buffer(positions);
    batch.molecular_charges = input_buffer(molecular_charges);
    batch.unpaired_electrons = input_buffer(unpaired_electrons);

    options.struct_size = sizeof(options);
    options.api_version = GPUXTB_API_VERSION;
    options.model = GPUXTB_MODEL_GFN2_XTB;
    options.flags = GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES;
    options.max_scc_iterations = 250;
    options.charge_tolerance = 1.0e-6;
    options.energy_tolerance = 1.0e-8;
    options.electronic_temperature = GPUXTB_DEFAULT_ELECTRONIC_TEMPERATURE;
    options.scc_start_mode = GPUXTB_SCC_START_FRESH;

    result.struct_size = sizeof(result);
    result.api_version = GPUXTB_API_VERSION;
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
    batch.struct_size = GPUXTB_BATCH_V2_SIZE;
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
};

using Mutation = std::function<void(Fixture&)>;

struct InvalidCase {
  const char* name;
  Mutation mutate;
  const char* error_fragment;
  gpuxtb_status_t status = GPUXTB_STATUS_INVALID_ARGUMENT;
  gpuxtb_backend_t backend = GPUXTB_BACKEND_CPU;
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
      GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options, &fixture.result);
  CHECK(checked.ok());
  CHECK(!checked.requires_backend_staging_validation());

  /* Zero point charges and a requested zero-length point-force output need no dummy pointers. */
  fixture.options.flags |= GPUXTB_COMPUTE_POINT_CHARGE_FORCES;
  checked = validate_compute_descriptors(GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options,
                                         &fixture.result);
  CHECK(checked.ok());
  CHECK(fixture.batch.point_charge_offsets.data == nullptr);
  CHECK(fixture.result.point_charge_forces.data == nullptr);

  /* CUDA accepts useful hybrid calls: metadata on the host and bulk arrays on device. */
  fixture.batch.positions.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  fixture.result.forces.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(GPUXTB_BACKEND_CUDA, &fixture.batch, &fixture.options,
                                         &fixture.result);
  CHECK(checked.ok());
  CHECK(!checked.requires_backend_staging_validation());

  fixture.batch.atom_offsets.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(GPUXTB_BACKEND_CUDA, &fixture.batch, &fixture.options,
                                         &fixture.result);
  CHECK(checked.ok());
  CHECK(checked.requires_backend_staging_validation());
  CHECK((checked.pending_offset_checks & kAtomOffsetsNeedStaging) != 0);

  Fixture point_fixture;
  point_fixture.enable_point_charges();
  point_fixture.batch.point_charge_offsets.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(GPUXTB_BACKEND_CUDA, &point_fixture.batch,
                                         &point_fixture.options, &point_fixture.result);
  CHECK(checked.ok());
  CHECK((checked.pending_offset_checks & kPointChargeOffsetsNeedStaging) != 0);

  Fixture response_fixture;
  response_fixture.enable_response();
  response_fixture.batch.charge_response_offsets.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(GPUXTB_BACKEND_CUDA, &response_fixture.batch,
                                         &response_fixture.options, &response_fixture.result);
  CHECK(checked.ok());
  CHECK((checked.pending_offset_checks & kChargeResponseOffsetsNeedStaging) != 0);
  CHECK((checked.pending_offset_checks & kChargeResponseShapeNeedsStaging) != 0);

  response_fixture.batch.atom_offsets.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  response_fixture.batch.charge_response_offsets.memory_space = GPUXTB_MEMORY_HOST;
  checked = validate_compute_descriptors(GPUXTB_BACKEND_CUDA, &response_fixture.batch,
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
      GPUXTB_BACKEND_CUDA, &fixture.batch, &fixture.options, &fixture.result);
  CHECK(checked.ok());
  CHECK(!checked.requires_backend_staging_validation());

  /* Device tags remain opaque through both common validation layers. */
  fixture.batch.atom_offsets.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  fixture.batch.atomic_numbers.data = reinterpret_cast<const void*>(std::uintptr_t{0x20000u});
  fixture.batch.atomic_numbers.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  fixture.batch.molecular_charges.data = reinterpret_cast<const void*>(std::uintptr_t{0x30000u});
  fixture.batch.molecular_charges.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  fixture.batch.unpaired_electrons.data = reinterpret_cast<const void*>(std::uintptr_t{0x40000u});
  fixture.batch.unpaired_electrons.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(GPUXTB_BACKEND_CUDA, &fixture.batch, &fixture.options,
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
                                       GPUXTB_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.atomic_numbers = {reinterpret_cast<const void*>(std::uintptr_t{0x20000u}),
                                         device_fixture.batch.atomic_numbers.size_bytes,
                                         GPUXTB_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.molecular_charges = {reinterpret_cast<const void*>(std::uintptr_t{0x30000u}),
                                            device_fixture.batch.molecular_charges.size_bytes,
                                            GPUXTB_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.unpaired_electrons = {
      reinterpret_cast<const void*>(std::uintptr_t{0x40000u}),
      device_fixture.batch.unpaired_electrons.size_bytes, GPUXTB_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.spin_channels = {reinterpret_cast<const void*>(std::uintptr_t{0x48000u}),
                                        device_fixture.batch.spin_channels.size_bytes,
                                        GPUXTB_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.point_charge_offsets = {
      reinterpret_cast<const void*>(std::uintptr_t{0x50000u}),
      device_fixture.batch.point_charge_offsets.size_bytes, GPUXTB_MEMORY_CUDA_DEVICE, 0};
  device_fixture.batch.charge_response_offsets = {
      reinterpret_cast<const void*>(std::uintptr_t{0x60000u}),
      device_fixture.batch.charge_response_offsets.size_bytes, GPUXTB_MEMORY_CUDA_DEVICE, 0};

  DescriptorValidationResult checked = validate_compute_descriptors(
      GPUXTB_BACKEND_CUDA, &device_fixture.batch, &device_fixture.options, &device_fixture.result);
  CHECK(checked.ok());
  CHECK((checked.pending_offset_checks & kTopologyMetadataStagingMask) ==
        kTopologyMetadataStagingMask);
  CHECK(checked.pending_offset_checks ==
        (kTopologyMetadataStagingMask | kChargeResponseShapeNeedsStaging));

  Fixture mixed_fixture;
  mixed_fixture.enable_point_charges();
  mixed_fixture.enable_response();
  mixed_fixture.batch.atomic_numbers.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  mixed_fixture.batch.unpaired_electrons.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  mixed_fixture.batch.point_charge_offsets.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(GPUXTB_BACKEND_CUDA, &mixed_fixture.batch,
                                         &mixed_fixture.options, &mixed_fixture.result);
  CHECK(checked.ok());
  CHECK(
      checked.pending_offset_checks ==
      (kAtomicNumbersNeedStaging | kUnpairedElectronsNeedStaging | kPointChargeOffsetsNeedStaging));

  /* HOST atom offsets can prove the total shape while response offsets stage. */
  mixed_fixture.batch.point_charge_offsets.memory_space = GPUXTB_MEMORY_HOST;
  mixed_fixture.batch.charge_response_offsets.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  checked = validate_compute_descriptors(GPUXTB_BACKEND_CUDA, &mixed_fixture.batch,
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
      GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options, &fixture.result);
  CHECK(structure.ok());
  CHECK(structure.pending_offset_checks == 0u);

  DescriptorValidationResult semantics = validate_host_topology_semantics(fixture.batch);
  CHECK(semantics.status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(semantics.error == "atom_offsets must be strictly increasing");

  /*
   * The complete structural entry still detects aliases, but the composed CPU
   * sequence preserves the historical topology-before-alias error priority.
   */
  fixture.result.energies.data = fixture.molecular_charges.data();
  fixture.result.energies.size_bytes = fixture.molecular_charges.size() * sizeof(double);
  structure = validate_compute_descriptor_structure(GPUXTB_BACKEND_CPU, &fixture.batch,
                                                    &fixture.options, &fixture.result);
  CHECK(structure.status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(structure.error.find("aliases") != std::string::npos);

  DescriptorValidationResult composed = validate_compute_descriptors(
      GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options, &fixture.result);
  CHECK(composed.status == semantics.status);
  CHECK(composed.error == semantics.error);
  return true;
}

bool test_headers_counts_and_overflow() {
  const std::vector<InvalidCase> cases = {
      {"batch ABI size", [](Fixture& f) { f.batch.struct_size = GPUXTB_BATCH_V1_SIZE - 1; },
       "batch is"},
      {"options ABI size",
       [](Fixture& f) { f.options.struct_size = GPUXTB_COMPUTE_OPTIONS_V1_SIZE - 1; },
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
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, nullptr, &fixture.options, &fixture.result)
            .status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, &fixture.batch, nullptr, &fixture.result)
            .status == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options, nullptr)
            .status == GPUXTB_STATUS_INVALID_ARGUMENT);
  return true;
}

bool test_compute_options() {
  const std::vector<InvalidCase> cases = {
      {"unknown model", [](Fixture& f) { f.options.model = static_cast<gpuxtb_model_t>(99); },
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
      {"CPU WARM SCC start", [](Fixture& f) { f.options.scc_start_mode = GPUXTB_SCC_START_WARM; },
       "CPU backend", GPUXTB_STATUS_NOT_SUPPORTED},
      {"result reserved", [](Fixture& f) { f.result.reserved = 1; }, "reserved"},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }

  Fixture fixture;
  fixture.options.model = GPUXTB_MODEL_GFN1_XTB;  // Known ABI model; dispatch decides support.
  fixture.options.electronic_temperature = 0.0;
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());

  fixture.options.model = GPUXTB_MODEL_GFN2_XTB;
  fixture.options.scc_start_mode = GPUXTB_SCC_START_WARM;
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CUDA, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());
  return true;
}

bool test_compute_options_short_prefixes() {
  Fixture fixture;
  constexpr std::size_t kCanaryBytes = 16;
  for (const std::size_t caller_size :
       {static_cast<std::size_t>(GPUXTB_COMPUTE_OPTIONS_V1_SIZE),
        static_cast<std::size_t>(GPUXTB_COMPUTE_OPTIONS_V2_SIZE - 1)}) {
    std::unique_ptr<unsigned char, decltype(&std::free)> storage(
        static_cast<unsigned char*>(std::malloc(caller_size + kCanaryBytes)), &std::free);
    CHECK(storage != nullptr);
    std::memset(storage.get(), 0xa5, caller_size + kCanaryBytes);
    std::memcpy(storage.get(), &fixture.options, caller_size);
    const std::uint32_t encoded_size = static_cast<std::uint32_t>(caller_size);
    std::memcpy(storage.get(), &encoded_size, sizeof(encoded_size));

    const auto* short_options = reinterpret_cast<const gpuxtb_compute_options_t*>(storage.get());
    const DescriptorValidationResult checked = validate_compute_descriptors(
        GPUXTB_BACKEND_CPU, &fixture.batch, short_options, &fixture.result);
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
       [](Fixture& f) { f.batch.positions.memory_space = static_cast<gpuxtb_memory_space_t>(99); },
       "memory_space"},
      {"CPU rejects CUDA pointer",
       [](Fixture& f) { f.batch.positions.memory_space = GPUXTB_MEMORY_CUDA_DEVICE; }, "CPU"},
      {"CUDA rejects ROCm pointer",
       [](Fixture& f) { f.batch.positions.memory_space = GPUXTB_MEMORY_ROCM_DEVICE; }, "ROCm",
       GPUXTB_STATUS_NOT_SUPPORTED, GPUXTB_BACKEND_CUDA},
      {"unresolved AUTO backend", [](Fixture&) {}, "resolved backend",
       GPUXTB_STATUS_INVALID_ARGUMENT, GPUXTB_BACKEND_AUTO},
      {"reserved ROCm backend", [](Fixture&) {}, "ROCm backend", GPUXTB_STATUS_NOT_SUPPORTED,
       GPUXTB_BACKEND_ROCM},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }

  Fixture fixture;
  fixture.batch.point_charge_positions.memory_space = GPUXTB_MEMORY_ROCM_DEVICE;
  /* A NULL zero-sized optional field is not a device pointer and remains legal. */
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());

  /* Oversized capacities and an under-aligned but byte-addressable offset view are valid. */
  fixture.batch.positions.size_bytes += 256;
  std::array<unsigned char, 3 * sizeof(std::int64_t) + 1> unaligned{};
  std::memcpy(unaligned.data() + 1, fixture.atom_offsets.data(), 3 * sizeof(std::int64_t));
  fixture.batch.atom_offsets.data = unaligned.data() + 1;
  fixture.batch.atom_offsets.size_bytes = 3 * sizeof(std::int64_t);
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());
  return true;
}

bool test_spin_channel_abi_v2() {
  Fixture legacy;
  legacy.batch.struct_size = GPUXTB_BATCH_V1_SIZE;
  legacy.batch.spin_channels = {reinterpret_cast<const void*>(std::uintptr_t{0x10000u}),
                                std::numeric_limits<std::size_t>::max(),
                                static_cast<gpuxtb_memory_space_t>(99), 1};
  /* ABI-v1 callers do not expose the suffix, so its out-of-range bytes stay unread. */
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, &legacy.batch, &legacy.options,
                                     &legacy.result)
            .ok());

  Fixture valid;
  valid.enable_spin_channels();
  CHECK(valid.batch.struct_size == GPUXTB_BATCH_V2_SIZE);
  CHECK(
      validate_compute_descriptors(GPUXTB_BACKEND_CPU, &valid.batch, &valid.options, &valid.result)
          .ok());
  valid.batch.spin_channels.size_bytes += 64;
  CHECK(
      validate_compute_descriptors(GPUXTB_BACKEND_CPU, &valid.batch, &valid.options, &valid.result)
          .ok());
  valid.batch.spin_channels.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
  const DescriptorValidationResult staged_spin = validate_host_topology_semantics(valid.batch);
  CHECK(staged_spin.ok());
  CHECK(staged_spin.pending_offset_checks == kSpinChannelsNeedStaging);
  CHECK(
      validate_compute_descriptors(GPUXTB_BACKEND_CUDA, &valid.batch, &valid.options, &valid.result)
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
         f.batch.spin_channels.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
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
         f.options.flags |= GPUXTB_COMPUTE_ATOMIC_CHARGES;
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
  fixture.options.flags = GPUXTB_COMPUTE_FORCES;
  fixture.result.energies = {};
  fixture.result.atomic_charges = {};
  fixture.result.point_charge_forces = {};
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());

  /* Unrequested output descriptors are not written and do not need capacity. */
  fixture.result.energies = {fixture.molecular_charges.data(), 1, GPUXTB_MEMORY_HOST, 0};
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options,
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
         f.options.flags |= GPUXTB_COMPUTE_POINT_CHARGE_FORCES;
         f.result.point_charge_forces.size_bytes -= sizeof(double);
       },
       "point_charge_forces"},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }

  Fixture fixture;
  fixture.enable_point_charges();
  fixture.options.flags |= GPUXTB_COMPUTE_POINT_CHARGE_FORCES;
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options,
                                     &fixture.result)
            .ok());

  /* Read-only input aliases are harmless; outputs are checked separately below. */
  fixture.batch.point_charge_values.data = fixture.molecular_charges.data();
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options,
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
         f.batch.charge_response_offsets.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
       },
       "total_charge_response_elements", GPUXTB_STATUS_INVALID_ARGUMENT, GPUXTB_BACKEND_CUDA},
  };
  for (const InvalidCase& test : cases) {
    CHECK(expect_invalid(test));
  }

  Fixture fixture;
  fixture.enable_response();
  fixture.potential_shifts.assign(3, 0.25);
  fixture.batch.atomic_potential_shifts = input_buffer(fixture.potential_shifts);
  CHECK(validate_compute_descriptors(GPUXTB_BACKEND_CPU, &fixture.batch, &fixture.options,
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
         f.result.energies.memory_space = GPUXTB_MEMORY_CUDA_DEVICE;
       },
       "incompatible memory-space", GPUXTB_STATUS_INVALID_ARGUMENT, GPUXTB_BACKEND_CUDA},
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
  };
  for (const Test& test : tests) {
    if (!test.run()) {
      std::cerr << "failed test group: " << test.name << '\n';
      return 1;
    }
  }
  return 0;
}
