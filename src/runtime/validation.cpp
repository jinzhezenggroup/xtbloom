#include "runtime/validation.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace gpuxtb::detail {
namespace {

struct BufferView {
  const void* data;
  std::size_t size_bytes;
  std::uint32_t memory_space;
  std::uint32_t reserved;
};

struct ActiveRange {
  const char* name;
  const void* data;
  std::size_t size_bytes;
  std::uint32_t memory_space;
  bool output;
};

/*
 * Byte extents proven by the descriptor prefix. Keeping this POD state on the
 * stack lets the complete structural validator and the CPU-compatible composed
 * validator share one prefix pass without allocation or pointed-to reads.
 */
struct DescriptorExtentState {
  std::size_t batch_i32_bytes = 0;
  std::size_t batch_u8_bytes = 0;
  std::size_t batch_f64_bytes = 0;
  std::size_t atom_i32_bytes = 0;
  std::size_t atom_f64_bytes = 0;
  std::size_t position_bytes = 0;
  std::size_t point_f64_bytes = 0;
  std::size_t point_position_bytes = 0;
  std::size_t atom_offset_bytes = 0;
  std::size_t response_f64_bytes = 0;
  bool response_enabled = false;
};

struct RequiredInput {
  const char* name;
  BufferView buffer;
  std::size_t bytes;
};

struct RequiredOutput {
  const char* name;
  BufferView buffer;
  std::size_t bytes;
  bool requested;
};

DescriptorValidationResult invalid(std::string error) {
  return {GPUXTB_STATUS_INVALID_ARGUMENT, kNoOffsetValidationPending, std::move(error)};
}

DescriptorValidationResult unsupported(std::string error) {
  return {GPUXTB_STATUS_NOT_SUPPORTED, kNoOffsetValidationPending, std::move(error)};
}

DescriptorValidationResult internal_error(std::string error) {
  return {GPUXTB_STATUS_INTERNAL_ERROR, kNoOffsetValidationPending, std::move(error)};
}

template <typename Structure>
bool has_v1_header(const Structure* value, std::size_t minimum_size) {
  return value != nullptr && value->struct_size >= minimum_size &&
         value->api_version == GPUXTB_API_VERSION;
}

template <typename Enum>
std::uint32_t raw_enum(const Enum& value) {
  static_assert(sizeof(Enum) == sizeof(std::uint32_t),
                "the ABI enums are required to occupy 32 bits");
  std::uint32_t raw = 0;
  /*
   * C callers can place any 32-bit value in an enum field. Copying the object
   * representation avoids a C++ lvalue-to-rvalue conversion of an invalid enum,
   * which is itself undefined behavior before validation could reject it.
   */
  std::memcpy(&raw, &value, sizeof(raw));
  return raw;
}

BufferView view(const gpuxtb_const_buffer_t& buffer) {
  return {buffer.data, buffer.size_bytes, raw_enum(buffer.memory_space), buffer.reserved};
}

BufferView view(const gpuxtb_buffer_t& buffer) {
  return {buffer.data, buffer.size_bytes, raw_enum(buffer.memory_space), buffer.reserved};
}

bool active(const BufferView& buffer) { return buffer.data != nullptr || buffer.size_bytes != 0; }

bool has_spin_channel_suffix(const gpuxtb_batch_t& batch) {
  return batch.struct_size >= GPUXTB_BATCH_V2_SIZE;
}

bool has_scc_start_mode_suffix(const gpuxtb_compute_options_t& options) {
  return options.struct_size >= GPUXTB_COMPUTE_OPTIONS_V2_SIZE;
}

BufferView spin_channel_view(const gpuxtb_batch_t& batch) {
  /* Do not read beyond an ABI-v1 caller's allocation. */
  return has_spin_channel_suffix(batch) ? view(batch.spin_channels)
                                        : BufferView{nullptr, 0u, GPUXTB_MEMORY_HOST, 0u};
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t& result) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool checked_multiply(std::size_t lhs, std::size_t rhs, std::size_t& result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool count_bytes(std::int64_t count, std::size_t values_per_item, std::size_t value_size,
                 std::size_t& result) {
  if (count < 0 || static_cast<std::uint64_t>(count) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  std::size_t values = 0;
  return checked_multiply(static_cast<std::size_t>(count), values_per_item, values) &&
         checked_multiply(values, value_size, result);
}

bool offset_bytes(std::int64_t batch_size, std::size_t& result) {
  if (batch_size < 0 || static_cast<std::uint64_t>(batch_size) >
                            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  std::size_t entries = 0;
  return checked_add(static_cast<std::size_t>(batch_size), 1, entries) &&
         checked_multiply(entries, sizeof(std::int64_t), result);
}

DescriptorValidationResult validate_buffer_descriptor(const char* name, const BufferView& buffer,
                                                      gpuxtb_backend_t backend) {
  const std::uint32_t backend_value = raw_enum(backend);
  if (buffer.reserved != 0) {
    return invalid(std::string(name) + ".reserved must be zero");
  }
  if (buffer.memory_space != GPUXTB_MEMORY_HOST &&
      buffer.memory_space != GPUXTB_MEMORY_CUDA_DEVICE &&
      buffer.memory_space != GPUXTB_MEMORY_ROCM_DEVICE) {
    return invalid(std::string(name) + " has an unknown memory_space value");
  }
  if (buffer.data == nullptr && buffer.size_bytes != 0) {
    return invalid(std::string(name) + " has nonzero size_bytes but a NULL data pointer");
  }

  /* An empty optional descriptor has no pointer for a backend to dereference. */
  if (!active(buffer)) {
    return {};
  }
  if (buffer.memory_space == GPUXTB_MEMORY_ROCM_DEVICE) {
    return unsupported(std::string(name) + " uses reserved ROCm device memory");
  }
  if (backend_value == GPUXTB_BACKEND_CPU && buffer.memory_space != GPUXTB_MEMORY_HOST) {
    return invalid(std::string(name) + " is device memory but the context backend is CPU");
  }
  if (backend_value == GPUXTB_BACKEND_CUDA && buffer.memory_space != GPUXTB_MEMORY_HOST &&
      buffer.memory_space != GPUXTB_MEMORY_CUDA_DEVICE) {
    return invalid(std::string(name) + " is incompatible with the CUDA backend");
  }
  return {};
}

DescriptorValidationResult require_bytes(const char* name, const BufferView& buffer,
                                         std::size_t required_bytes) {
  /* Zero-length logical arrays intentionally permit {NULL, 0}. */
  if (required_bytes == 0) {
    return {};
  }
  if (buffer.data == nullptr) {
    return invalid(std::string(name) + " is required but its data pointer is NULL");
  }
  if (buffer.size_bytes < required_bytes) {
    return invalid(std::string(name) + " is smaller than the required byte size");
  }
  return {};
}

std::int64_t load_offset(const BufferView& buffer, std::size_t index) {
  std::int64_t value = 0;
  /* memcpy avoids undefined behavior for a valid but under-aligned C buffer. */
  std::memcpy(&value, static_cast<const unsigned char*>(buffer.data) + index * sizeof(value),
              sizeof(value));
  return value;
}

DescriptorValidationResult validate_host_offsets(const char* name, const BufferView& buffer,
                                                 std::int64_t batch_size,
                                                 std::int64_t expected_endpoint,
                                                 bool require_nonempty_segments) {
  const std::int64_t first = load_offset(buffer, 0);
  if (first != 0) {
    return invalid(std::string(name) + " must begin with zero");
  }

  std::int64_t previous = first;
  for (std::int64_t system = 0; system < batch_size; ++system) {
    const std::int64_t next = load_offset(buffer, static_cast<std::size_t>(system + 1));
    if (require_nonempty_segments ? next <= previous : next < previous) {
      return invalid(std::string(name) + (require_nonempty_segments
                                              ? " must be strictly increasing"
                                              : " must be monotonically nondecreasing"));
    }
    previous = next;
  }
  if (previous != expected_endpoint) {
    return invalid(std::string(name) + " endpoint does not match its declared total");
  }
  return {};
}

template <std::size_t N>
DescriptorValidationResult add_active_range(std::array<ActiveRange, N>& ranges,
                                            std::size_t& range_count, const char* name,
                                            const BufferView& buffer, std::size_t logical_bytes,
                                            bool output) {
  if (logical_bytes == 0 || buffer.data == nullptr) {
    return {};
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(buffer.data);
  if (logical_bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return invalid(std::string(name) + " address range overflows uintptr_t");
  }
  if (range_count == ranges.size()) {
    return internal_error("ABI-v1 active-range validation capacity was exceeded");
  }
  ranges[range_count++] = {name, buffer.data, logical_bytes, buffer.memory_space, output};
  return {};
}

bool ranges_overlap(const ActiveRange& lhs, const ActiveRange& rhs) {
  const std::uintptr_t lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
  const std::uintptr_t rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
  const std::uintptr_t lhs_end = lhs_begin + lhs.size_bytes;
  const std::uintptr_t rhs_end = rhs_begin + rhs.size_bytes;
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

template <std::size_t N>
DescriptorValidationResult validate_aliases(const std::array<ActiveRange, N>& ranges,
                                            std::size_t range_count) {
  for (std::size_t lhs_index = 0; lhs_index < range_count; ++lhs_index) {
    for (std::size_t rhs_index = lhs_index + 1; rhs_index < range_count; ++rhs_index) {
      const ActiveRange& lhs = ranges[lhs_index];
      const ActiveRange& rhs = ranges[rhs_index];
      if (!lhs.output && !rhs.output) {
        continue;  // Read-only inputs may intentionally share storage.
      }

      if (ranges_overlap(lhs, rhs)) {
        if (lhs.memory_space != rhs.memory_space) {
          return invalid(std::string(lhs.name) + " and " + rhs.name +
                         " overlap with incompatible memory-space tags");
        }
        return invalid(std::string(lhs.name) + " aliases " + rhs.name);
      }
    }
  }
  return {};
}

}  // namespace

namespace {

DescriptorValidationResult validate_compute_descriptor_prefix(
    gpuxtb_backend_t backend, const gpuxtb_batch_t* batch, const gpuxtb_compute_options_t* options,
    const gpuxtb_batch_result_t* result, DescriptorExtentState& extents) {
  const std::uint32_t backend_value = raw_enum(backend);
  if (backend_value == GPUXTB_BACKEND_AUTO) {
    return invalid("descriptor validation requires a resolved backend, not AUTO");
  }
  if (backend_value == GPUXTB_BACKEND_ROCM) {
    return unsupported("the ROCm backend is reserved but not implemented");
  }
  if (backend_value != GPUXTB_BACKEND_CPU && backend_value != GPUXTB_BACKEND_CUDA) {
    return invalid("descriptor validation received an unknown backend value");
  }
  if (!has_v1_header(batch, GPUXTB_BATCH_V1_SIZE)) {
    return invalid("batch is NULL, too small, or uses an unsupported API version");
  }
  if (!has_v1_header(options, GPUXTB_COMPUTE_OPTIONS_V1_SIZE)) {
    return invalid("compute options are NULL, too small, or use an unsupported API version");
  }
  if (!has_v1_header(result, GPUXTB_BATCH_RESULT_V1_SIZE)) {
    return invalid("batch result is NULL, too small, or uses an unsupported API version");
  }

  if (batch->batch_size <= 0) {
    return invalid("batch_size must be positive");
  }
  if (batch->total_atoms < batch->batch_size) {
    return invalid("total_atoms must permit at least one atom in every batch item");
  }
  if (batch->total_point_charges < 0) {
    return invalid("total_point_charges must be nonnegative");
  }
  if (batch->total_charge_response_elements < 0) {
    return invalid("total_charge_response_elements must be nonnegative");
  }

  constexpr std::uint32_t kKnownComputeFlags = GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES |
                                               GPUXTB_COMPUTE_ATOMIC_CHARGES |
                                               GPUXTB_COMPUTE_POINT_CHARGE_FORCES;
  const std::uint32_t model_value = raw_enum(options->model);
  if (model_value != GPUXTB_MODEL_GFN1_XTB && model_value != GPUXTB_MODEL_GFN2_XTB) {
    return invalid("compute options contain an unknown model value");
  }
  if (options->flags == 0 || (options->flags & ~kKnownComputeFlags) != 0) {
    return invalid("compute flags must contain at least one known output flag and no unknown bits");
  }
  if (options->max_scc_iterations <= 0) {
    return invalid("max_scc_iterations must be positive");
  }
  if (!std::isfinite(options->charge_tolerance) || options->charge_tolerance <= 0.0) {
    return invalid("charge_tolerance must be finite and positive");
  }
  if (!std::isfinite(options->energy_tolerance) || options->energy_tolerance <= 0.0) {
    return invalid("energy_tolerance must be finite and positive");
  }
  if (!std::isfinite(options->electronic_temperature) || options->electronic_temperature < 0.0) {
    return invalid("electronic_temperature must be finite and nonnegative");
  }
  if (options->reserved != 0) {
    return invalid("compute options reserved field must be zero");
  }
  if (has_scc_start_mode_suffix(*options)) {
    const std::uint32_t start_mode = raw_enum(options->scc_start_mode);
    if (start_mode != GPUXTB_SCC_START_FRESH && start_mode != GPUXTB_SCC_START_WARM) {
      return invalid("scc_start_mode must be GPUXTB_SCC_START_FRESH or GPUXTB_SCC_START_WARM");
    }
    if (options->reserved_v2 != 0u) {
      return invalid("compute options reserved_v2 field must be zero");
    }
  }
  if (result->reserved != 0) {
    return invalid("batch result reserved field must be zero");
  }

  struct NamedBuffer {
    const char* name;
    BufferView buffer;
  };
  const NamedBuffer all_buffers[] = {
      {"atom_offsets", view(batch->atom_offsets)},
      {"atomic_numbers", view(batch->atomic_numbers)},
      {"positions", view(batch->positions)},
      {"molecular_charges", view(batch->molecular_charges)},
      {"unpaired_electrons", view(batch->unpaired_electrons)},
      {"point_charge_offsets", view(batch->point_charge_offsets)},
      {"point_charge_positions", view(batch->point_charge_positions)},
      {"point_charge_values", view(batch->point_charge_values)},
      {"point_charge_gammas", view(batch->point_charge_gammas)},
      {"atomic_potential_shifts", view(batch->atomic_potential_shifts)},
      {"charge_response_offsets", view(batch->charge_response_offsets)},
      {"charge_response_matrix", view(batch->charge_response_matrix)},
      {"energies", view(result->energies)},
      {"forces", view(result->forces)},
      {"atomic_charges", view(result->atomic_charges)},
      {"point_charge_forces", view(result->point_charge_forces)},
      {"scc_iterations", view(result->scc_iterations)},
      {"scc_converged", view(result->scc_converged)},
      {"per_system_status", view(result->per_system_status)},
  };
  for (const NamedBuffer& named : all_buffers) {
    DescriptorValidationResult checked =
        validate_buffer_descriptor(named.name, named.buffer, backend);
    if (!checked.ok()) {
      return checked;
    }
  }
  const BufferView spin_channels = spin_channel_view(*batch);
  if (active(spin_channels)) {
    DescriptorValidationResult checked =
        validate_buffer_descriptor("spin_channels", spin_channels, backend);
    if (!checked.ok()) {
      return checked;
    }
  }

  std::size_t batch_i32_bytes = 0;
  std::size_t batch_u8_bytes = 0;
  std::size_t batch_f64_bytes = 0;
  std::size_t atom_i32_bytes = 0;
  std::size_t atom_f64_bytes = 0;
  std::size_t position_bytes = 0;
  std::size_t point_f64_bytes = 0;
  std::size_t point_position_bytes = 0;
  std::size_t atom_offset_bytes = 0;
  std::size_t response_f64_bytes = 0;
  if (!count_bytes(batch->batch_size, 1, sizeof(std::int32_t), batch_i32_bytes) ||
      !count_bytes(batch->batch_size, 1, sizeof(std::uint8_t), batch_u8_bytes) ||
      !count_bytes(batch->batch_size, 1, sizeof(double), batch_f64_bytes) ||
      !count_bytes(batch->total_atoms, 1, sizeof(std::int32_t), atom_i32_bytes) ||
      !count_bytes(batch->total_atoms, 1, sizeof(double), atom_f64_bytes) ||
      !count_bytes(batch->total_atoms, 3, sizeof(double), position_bytes) ||
      !count_bytes(batch->total_point_charges, 1, sizeof(double), point_f64_bytes) ||
      !count_bytes(batch->total_point_charges, 3, sizeof(double), point_position_bytes) ||
      !offset_bytes(batch->batch_size, atom_offset_bytes) ||
      !count_bytes(batch->total_charge_response_elements, 1, sizeof(double), response_f64_bytes)) {
    return invalid("a declared count overflows the addressable byte size");
  }

  const RequiredInput required_inputs[] = {
      {"atom_offsets", view(batch->atom_offsets), atom_offset_bytes},
      {"atomic_numbers", view(batch->atomic_numbers), atom_i32_bytes},
      {"positions", view(batch->positions), position_bytes},
      {"molecular_charges", view(batch->molecular_charges), batch_f64_bytes},
      {"unpaired_electrons", view(batch->unpaired_electrons), batch_i32_bytes},
  };
  for (const RequiredInput& input : required_inputs) {
    DescriptorValidationResult checked = require_bytes(input.name, input.buffer, input.bytes);
    if (!checked.ok()) {
      return checked;
    }
  }
  if (active(spin_channels)) {
    DescriptorValidationResult checked =
        require_bytes("spin_channels", spin_channels, batch_i32_bytes);
    if (!checked.ok()) {
      return checked;
    }
  }

  const BufferView point_offsets = view(batch->point_charge_offsets);
  const BufferView point_positions = view(batch->point_charge_positions);
  const BufferView point_values = view(batch->point_charge_values);
  const BufferView point_gammas = view(batch->point_charge_gammas);
  if (batch->total_point_charges != 0) {
    const RequiredInput point_inputs[] = {
        {"point_charge_offsets", point_offsets, atom_offset_bytes},
        {"point_charge_positions", point_positions, point_position_bytes},
        {"point_charge_values", point_values, point_f64_bytes},
        {"point_charge_gammas", point_gammas, point_f64_bytes},
    };
    for (const RequiredInput& input : point_inputs) {
      DescriptorValidationResult checked = require_bytes(input.name, input.buffer, input.bytes);
      if (!checked.ok()) {
        return checked;
      }
    }
  } else if (active(point_offsets)) {
    /* Supplying zero-count offsets is legal, but if present they still describe the batch. */
    DescriptorValidationResult checked =
        require_bytes("point_charge_offsets", point_offsets, atom_offset_bytes);
    if (!checked.ok()) {
      return checked;
    }
  }

  const BufferView potential_shifts = view(batch->atomic_potential_shifts);
  if (active(potential_shifts)) {
    DescriptorValidationResult checked =
        require_bytes("atomic_potential_shifts", potential_shifts, atom_f64_bytes);
    if (!checked.ok()) {
      return checked;
    }
  }

  const BufferView response_offsets = view(batch->charge_response_offsets);
  const BufferView response_matrix = view(batch->charge_response_matrix);
  const bool response_enabled = batch->total_charge_response_elements != 0 ||
                                active(response_offsets) || active(response_matrix);
  if (response_enabled) {
    if (batch->total_charge_response_elements == 0 || !active(response_offsets) ||
        !active(response_matrix)) {
      return invalid("charge response count, offsets, and matrix must be supplied together");
    }
    DescriptorValidationResult checked =
        require_bytes("charge_response_offsets", response_offsets, atom_offset_bytes);
    if (!checked.ok()) {
      return checked;
    }
    checked = require_bytes("charge_response_matrix", response_matrix, response_f64_bytes);
    if (!checked.ok()) {
      return checked;
    }
  }

  const RequiredOutput outputs[] = {
      {"energies", view(result->energies), batch_f64_bytes,
       (options->flags & GPUXTB_COMPUTE_ENERGY) != 0},
      {"forces", view(result->forces), position_bytes,
       (options->flags & GPUXTB_COMPUTE_FORCES) != 0},
      {"atomic_charges", view(result->atomic_charges), atom_f64_bytes,
       (options->flags & GPUXTB_COMPUTE_ATOMIC_CHARGES) != 0},
      {"point_charge_forces", view(result->point_charge_forces), point_position_bytes,
       (options->flags & GPUXTB_COMPUTE_POINT_CHARGE_FORCES) != 0},
      {"scc_iterations", view(result->scc_iterations), batch_i32_bytes, true},
      {"scc_converged", view(result->scc_converged), batch_u8_bytes, true},
      {"per_system_status", view(result->per_system_status), batch_i32_bytes, true},
  };
  for (const RequiredOutput& output : outputs) {
    if (!output.requested) {
      continue;
    }
    DescriptorValidationResult checked = require_bytes(output.name, output.buffer, output.bytes);
    if (!checked.ok()) {
      return checked;
    }
  }

  extents.batch_i32_bytes = batch_i32_bytes;
  extents.batch_u8_bytes = batch_u8_bytes;
  extents.batch_f64_bytes = batch_f64_bytes;
  extents.atom_i32_bytes = atom_i32_bytes;
  extents.atom_f64_bytes = atom_f64_bytes;
  extents.position_bytes = position_bytes;
  extents.point_f64_bytes = point_f64_bytes;
  extents.point_position_bytes = point_position_bytes;
  extents.atom_offset_bytes = atom_offset_bytes;
  extents.response_f64_bytes = response_f64_bytes;
  extents.response_enabled = response_enabled;
  return {};
}

DescriptorValidationResult validate_compute_descriptor_aliases(
    const gpuxtb_batch_t& batch, const gpuxtb_compute_options_t& options,
    const gpuxtb_batch_result_t& result, const DescriptorExtentState& extents) {
  /* One fixed entry per ABI-v1 buffer keeps successful validation allocation-free. */
  std::array<ActiveRange, 20> ranges{};
  std::size_t range_count = 0;
  const RequiredInput alias_inputs[] = {
      {"atom_offsets", view(batch.atom_offsets), extents.atom_offset_bytes},
      {"atomic_numbers", view(batch.atomic_numbers), extents.atom_i32_bytes},
      {"positions", view(batch.positions), extents.position_bytes},
      {"molecular_charges", view(batch.molecular_charges), extents.batch_f64_bytes},
      {"unpaired_electrons", view(batch.unpaired_electrons), extents.batch_i32_bytes},
  };
  for (const RequiredInput& input : alias_inputs) {
    DescriptorValidationResult checked =
        add_active_range(ranges, range_count, input.name, input.buffer, input.bytes, false);
    if (!checked.ok()) {
      return checked;
    }
  }
  const BufferView spin_channels = spin_channel_view(batch);
  if (active(spin_channels)) {
    DescriptorValidationResult checked = add_active_range(
        ranges, range_count, "spin_channels", spin_channels, extents.batch_i32_bytes, false);
    if (!checked.ok()) {
      return checked;
    }
  }
  if (batch.total_point_charges != 0) {
    const RequiredInput point_alias_inputs[] = {
        {"point_charge_offsets", view(batch.point_charge_offsets), extents.atom_offset_bytes},
        {"point_charge_positions", view(batch.point_charge_positions),
         extents.point_position_bytes},
        {"point_charge_values", view(batch.point_charge_values), extents.point_f64_bytes},
        {"point_charge_gammas", view(batch.point_charge_gammas), extents.point_f64_bytes},
    };
    for (const RequiredInput& input : point_alias_inputs) {
      DescriptorValidationResult checked =
          add_active_range(ranges, range_count, input.name, input.buffer, input.bytes, false);
      if (!checked.ok()) {
        return checked;
      }
    }
  }
  const BufferView potential_shifts = view(batch.atomic_potential_shifts);
  if (active(potential_shifts)) {
    DescriptorValidationResult checked =
        add_active_range(ranges, range_count, "atomic_potential_shifts", potential_shifts,
                         extents.atom_f64_bytes, false);
    if (!checked.ok()) {
      return checked;
    }
  }
  if (extents.response_enabled) {
    DescriptorValidationResult checked =
        add_active_range(ranges, range_count, "charge_response_offsets",
                         view(batch.charge_response_offsets), extents.atom_offset_bytes, false);
    if (!checked.ok()) {
      return checked;
    }
    checked =
        add_active_range(ranges, range_count, "charge_response_matrix",
                         view(batch.charge_response_matrix), extents.response_f64_bytes, false);
    if (!checked.ok()) {
      return checked;
    }
  }
  const RequiredOutput outputs[] = {
      {"energies", view(result.energies), extents.batch_f64_bytes,
       (options.flags & GPUXTB_COMPUTE_ENERGY) != 0},
      {"forces", view(result.forces), extents.position_bytes,
       (options.flags & GPUXTB_COMPUTE_FORCES) != 0},
      {"atomic_charges", view(result.atomic_charges), extents.atom_f64_bytes,
       (options.flags & GPUXTB_COMPUTE_ATOMIC_CHARGES) != 0},
      {"point_charge_forces", view(result.point_charge_forces), extents.point_position_bytes,
       (options.flags & GPUXTB_COMPUTE_POINT_CHARGE_FORCES) != 0},
      {"scc_iterations", view(result.scc_iterations), extents.batch_i32_bytes, true},
      {"scc_converged", view(result.scc_converged), extents.batch_u8_bytes, true},
      {"per_system_status", view(result.per_system_status), extents.batch_i32_bytes, true},
  };
  for (const RequiredOutput& output : outputs) {
    if (!output.requested) {
      continue;
    }
    DescriptorValidationResult checked =
        add_active_range(ranges, range_count, output.name, output.buffer, output.bytes, true);
    if (!checked.ok()) {
      return checked;
    }
  }

  DescriptorValidationResult alias_result = validate_aliases(ranges, range_count);
  if (!alias_result.ok()) {
    return alias_result;
  }
  return {};
}

}  // namespace

DescriptorValidationResult validate_compute_descriptor_structure(
    gpuxtb_backend_t backend, const gpuxtb_batch_t* batch, const gpuxtb_compute_options_t* options,
    const gpuxtb_batch_result_t* result) {
  DescriptorExtentState extents;
  DescriptorValidationResult prefix =
      validate_compute_descriptor_prefix(backend, batch, options, result, extents);
  if (!prefix.ok()) {
    return prefix;
  }
  return validate_compute_descriptor_aliases(*batch, *options, *result, extents);
}

DescriptorValidationResult validate_host_topology_semantics(const gpuxtb_batch_t& batch) {
  DescriptorValidationResult validation;
  const BufferView atom_offsets = view(batch.atom_offsets);
  if (atom_offsets.memory_space == GPUXTB_MEMORY_HOST) {
    validation = validate_host_offsets("atom_offsets", atom_offsets, batch.batch_size,
                                       batch.total_atoms, true);
    if (!validation.ok()) {
      return validation;
    }
  } else {
    validation.pending_offset_checks |= kAtomOffsetsNeedStaging;
  }

  const BufferView atomic_numbers = view(batch.atomic_numbers);
  if (atomic_numbers.memory_space != GPUXTB_MEMORY_HOST) {
    validation.pending_offset_checks |= kAtomicNumbersNeedStaging;
  }
  const BufferView molecular_charges = view(batch.molecular_charges);
  if (molecular_charges.memory_space != GPUXTB_MEMORY_HOST) {
    validation.pending_offset_checks |= kMolecularChargesNeedStaging;
  }
  const BufferView unpaired_electrons = view(batch.unpaired_electrons);
  if (unpaired_electrons.memory_space != GPUXTB_MEMORY_HOST) {
    validation.pending_offset_checks |= kUnpairedElectronsNeedStaging;
  }
  const BufferView spin_channels = spin_channel_view(batch);
  if (active(spin_channels)) {
    if (spin_channels.memory_space != GPUXTB_MEMORY_HOST) {
      validation.pending_offset_checks |= kSpinChannelsNeedStaging;
    } else {
      for (std::int64_t system = 0; system < batch.batch_size; ++system) {
        std::int32_t channels = 0;
        std::memcpy(&channels,
                    static_cast<const unsigned char*>(spin_channels.data) +
                        static_cast<std::size_t>(system) * sizeof(channels),
                    sizeof(channels));
        if (channels != 1 && channels != 2) {
          return invalid("spin_channels values must be one or two");
        }
      }
    }
  }

  const BufferView point_offsets = view(batch.point_charge_offsets);
  if (batch.total_point_charges != 0 || active(point_offsets)) {
    if (point_offsets.memory_space == GPUXTB_MEMORY_HOST) {
      DescriptorValidationResult checked =
          validate_host_offsets("point_charge_offsets", point_offsets, batch.batch_size,
                                batch.total_point_charges, false);
      if (!checked.ok()) {
        return checked;
      }
    } else {
      validation.pending_offset_checks |= kPointChargeOffsetsNeedStaging;
    }
  }

  const BufferView response_offsets = view(batch.charge_response_offsets);
  const BufferView response_matrix = view(batch.charge_response_matrix);
  const bool response_enabled = batch.total_charge_response_elements != 0 ||
                                active(response_offsets) || active(response_matrix);
  if (response_enabled) {
    if (response_offsets.memory_space == GPUXTB_MEMORY_HOST) {
      DescriptorValidationResult checked =
          validate_host_offsets("charge_response_offsets", response_offsets, batch.batch_size,
                                batch.total_charge_response_elements, false);
      if (!checked.ok()) {
        return checked;
      }
    } else {
      validation.pending_offset_checks |= kChargeResponseOffsetsNeedStaging;
    }

    const bool atom_offsets_on_host = atom_offsets.memory_space == GPUXTB_MEMORY_HOST;
    const bool response_offsets_on_host = response_offsets.memory_space == GPUXTB_MEMORY_HOST;
    if (atom_offsets_on_host) {
      std::uint64_t expected_total = 0;
      for (std::int64_t system = 0; system < batch.batch_size; ++system) {
        const std::int64_t atom_begin = load_offset(atom_offsets, static_cast<std::size_t>(system));
        const std::int64_t atom_end =
            load_offset(atom_offsets, static_cast<std::size_t>(system + 1));
        const std::uint64_t atoms = static_cast<std::uint64_t>(atom_end - atom_begin);
        if (atoms != 0 && atoms > std::numeric_limits<std::uint64_t>::max() / atoms) {
          return invalid("a per-system charge response dimension overflows uint64_t");
        }
        const std::uint64_t elements = atoms * atoms;
        if (elements > std::numeric_limits<std::uint64_t>::max() - expected_total) {
          return invalid("the packed charge response element count overflows uint64_t");
        }
        expected_total += elements;

        if (response_offsets_on_host) {
          const std::int64_t response_begin =
              load_offset(response_offsets, static_cast<std::size_t>(system));
          const std::int64_t response_end =
              load_offset(response_offsets, static_cast<std::size_t>(system + 1));
          if (static_cast<std::uint64_t>(response_end - response_begin) != elements) {
            return invalid("charge_response_offsets do not pack one square matrix per system");
          }
        }
      }
      if (expected_total > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
          static_cast<std::int64_t>(expected_total) != batch.total_charge_response_elements) {
        return invalid("total_charge_response_elements does not match the packed square matrices");
      }
      if (!response_offsets_on_host) {
        validation.pending_offset_checks |= kChargeResponseShapeNeedsStaging;
      }
    } else {
      validation.pending_offset_checks |= kChargeResponseShapeNeedsStaging;
    }
  }
  return validation;
}

DescriptorValidationResult validate_compute_descriptors(gpuxtb_backend_t backend,
                                                        const gpuxtb_batch_t* batch,
                                                        const gpuxtb_compute_options_t* options,
                                                        const gpuxtb_batch_result_t* result) {
  DescriptorExtentState extents;
  DescriptorValidationResult prefix =
      validate_compute_descriptor_prefix(backend, batch, options, result, extents);
  if (!prefix.ok()) {
    return prefix;
  }
  DescriptorValidationResult semantics = validate_host_topology_semantics(*batch);
  if (!semantics.ok()) {
    return semantics;
  }
  DescriptorValidationResult aliases =
      validate_compute_descriptor_aliases(*batch, *options, *result, extents);
  if (!aliases.ok()) {
    return aliases;
  }
  return semantics;
}

}  // namespace gpuxtb::detail
