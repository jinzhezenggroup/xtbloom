#include "runtime/validation.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace xtbloom::detail {
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
  std::size_t interaction_descriptor_bytes = 0;
  std::size_t dipole_f64_bytes = 0;
  bool response_enabled = false;
  bool interactions_enabled = false;
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
  return {XTBLOOM_STATUS_INVALID_ARGUMENT, kNoOffsetValidationPending, std::move(error)};
}

DescriptorValidationResult unsupported(std::string error) {
  return {XTBLOOM_STATUS_NOT_SUPPORTED, kNoOffsetValidationPending, std::move(error)};
}

DescriptorValidationResult internal_error(std::string error) {
  return {XTBLOOM_STATUS_INTERNAL_ERROR, kNoOffsetValidationPending, std::move(error)};
}

template <typename Structure>
bool has_v1_header(const Structure* value, std::size_t minimum_size) {
  return value != nullptr && value->struct_size >= minimum_size &&
         value->api_version == XTBLOOM_API_VERSION;
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

BufferView view(const xtbloom_const_buffer_t& buffer) {
  return {buffer.data, buffer.size_bytes, raw_enum(buffer.memory_space), buffer.reserved};
}

BufferView view(const xtbloom_buffer_t& buffer) {
  return {buffer.data, buffer.size_bytes, raw_enum(buffer.memory_space), buffer.reserved};
}

bool active(const BufferView& buffer) { return buffer.data != nullptr || buffer.size_bytes != 0; }

bool has_spin_channel_suffix(const xtbloom_batch_t& batch) {
  return batch.struct_size >= XTBLOOM_BATCH_V2_SIZE;
}

bool has_scc_start_mode_suffix(const xtbloom_compute_options_t& options) {
  return options.struct_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE;
}

BufferView spin_channel_view(const xtbloom_batch_t& batch) {
  /* Do not read beyond an ABI-v1 caller's allocation. */
  return has_spin_channel_suffix(batch) ? view(batch.spin_channels)
                                        : BufferView{nullptr, 0u, XTBLOOM_MEMORY_HOST, 0u};
}

bool has_interaction_suffix(const xtbloom_batch_t& batch) {
  return batch.struct_size >= XTBLOOM_BATCH_V3_SIZE;
}

bool has_result_v2_suffix(const xtbloom_batch_result_t& result) {
  return result.struct_size >= XTBLOOM_BATCH_RESULT_V2_SIZE;
}

/* Do not read beyond an ABI-v2 caller's allocation. */
BufferView interaction_descriptor_view(const xtbloom_batch_t& batch) {
  return has_interaction_suffix(batch) ? view(batch.interaction_descriptors)
                                       : BufferView{nullptr, 0u, XTBLOOM_MEMORY_HOST, 0u};
}

BufferView interaction_payload_view(const xtbloom_batch_t& batch) {
  return has_interaction_suffix(batch) ? view(batch.interaction_payload)
                                       : BufferView{nullptr, 0u, XTBLOOM_MEMORY_HOST, 0u};
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
                                                      xtbloom_backend_t backend) {
  const std::uint32_t backend_value = raw_enum(backend);
  if (buffer.reserved != 0) {
    return invalid(std::string(name) + ".reserved must be zero");
  }
  if (buffer.memory_space != XTBLOOM_MEMORY_HOST &&
      buffer.memory_space != XTBLOOM_MEMORY_CUDA_DEVICE &&
      buffer.memory_space != XTBLOOM_MEMORY_ROCM_DEVICE) {
    return invalid(std::string(name) + " has an unknown memory_space value");
  }
  if (buffer.data == nullptr && buffer.size_bytes != 0) {
    return invalid(std::string(name) + " has nonzero size_bytes but a NULL data pointer");
  }

  /* An empty optional descriptor has no pointer for a backend to dereference. */
  if (!active(buffer)) {
    return {};
  }
  if (buffer.memory_space == XTBLOOM_MEMORY_ROCM_DEVICE) {
    return unsupported(std::string(name) + " uses reserved ROCm device memory");
  }
  if (backend_value == XTBLOOM_BACKEND_CPU && buffer.memory_space != XTBLOOM_MEMORY_HOST) {
    return invalid(std::string(name) + " is device memory but the context backend is CPU");
  }
  if (backend_value == XTBLOOM_BACKEND_CUDA && buffer.memory_space != XTBLOOM_MEMORY_HOST &&
      buffer.memory_space != XTBLOOM_MEMORY_CUDA_DEVICE) {
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

/*
 * Byte-exact load of one xtbloom_interaction_t entry.  The layout is fixed by
 * the public ABI and reproduced here field by field (rather than copying the
 * struct by value) so an under-aligned or otherwise hostile caller buffer
 * remains well-defined.
 */
struct InteractionView {
  std::int32_t type;
  std::uint32_t flags;
  std::int64_t system_index;
  std::uint64_t payload_offset;
  std::uint64_t payload_size;
};

InteractionView load_interaction(const BufferView& descriptors, std::size_t index) {
  const unsigned char* base =
      static_cast<const unsigned char*>(descriptors.data) + index * sizeof(xtbloom_interaction_t);
  InteractionView out{};
  std::memcpy(&out.type, base + offsetof(xtbloom_interaction_t, type), sizeof(out.type));
  std::memcpy(&out.flags, base + offsetof(xtbloom_interaction_t, flags), sizeof(out.flags));
  std::memcpy(&out.system_index, base + offsetof(xtbloom_interaction_t, system_index),
              sizeof(out.system_index));
  std::memcpy(&out.payload_offset, base + offsetof(xtbloom_interaction_t, payload_offset),
              sizeof(out.payload_offset));
  std::memcpy(&out.payload_size, base + offsetof(xtbloom_interaction_t, payload_size),
              sizeof(out.payload_size));
  return out;
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
    xtbloom_backend_t backend, const xtbloom_batch_t* batch,
    const xtbloom_compute_options_t* options, const xtbloom_batch_result_t* result,
    DescriptorExtentState& extents) {
  const std::uint32_t backend_value = raw_enum(backend);
  if (backend_value == XTBLOOM_BACKEND_AUTO) {
    return invalid("descriptor validation requires a resolved backend, not AUTO");
  }
  if (backend_value == XTBLOOM_BACKEND_ROCM) {
    return unsupported("the ROCm backend is reserved but not implemented");
  }
  if (backend_value != XTBLOOM_BACKEND_CPU && backend_value != XTBLOOM_BACKEND_CUDA) {
    return invalid("descriptor validation received an unknown backend value");
  }
  if (!has_v1_header(batch, XTBLOOM_BATCH_V1_SIZE)) {
    return invalid("batch is NULL, too small, or uses an unsupported API version");
  }
  if (!has_v1_header(options, XTBLOOM_COMPUTE_OPTIONS_V1_SIZE)) {
    return invalid("compute options are NULL, too small, or use an unsupported API version");
  }
  if (result != nullptr && !has_v1_header(result, XTBLOOM_BATCH_RESULT_V1_SIZE)) {
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

  constexpr std::uint32_t kKnownComputeFlags =
      XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES | XTBLOOM_COMPUTE_ATOMIC_CHARGES |
      XTBLOOM_COMPUTE_POINT_CHARGE_FORCES | XTBLOOM_COMPUTE_DIPOLE_MOMENTS;
  const std::uint32_t model_value = raw_enum(options->model);
  if (model_value != XTBLOOM_MODEL_GFN1_XTB && model_value != XTBLOOM_MODEL_GFN2_XTB) {
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
    if (start_mode != XTBLOOM_SCC_START_FRESH && start_mode != XTBLOOM_SCC_START_WARM) {
      return invalid("scc_start_mode must be XTBLOOM_SCC_START_FRESH or XTBLOOM_SCC_START_WARM");
    }
    if (options->reserved_v2 != 0u) {
      return invalid("compute options reserved_v2 field must be zero");
    }
  }
  if (result != nullptr && result->reserved != 0) {
    return invalid("batch result reserved field must be zero");
  }

  struct NamedBuffer {
    const char* name;
    BufferView buffer;
  };
  const xtbloom_buffer_t empty_buffer{};
  const bool result_v2 = result != nullptr && has_result_v2_suffix(*result);
  /* Keep the size condition outside each member expression. Passing a suffix
   * field to a helper by reference would evaluate the out-of-prefix lvalue
   * before the helper could check struct_size. */
  const BufferView dipole_output = result_v2 ? view(result->dipole_moments) : view(empty_buffer);
  const BufferView quadrupole_output =
      result_v2 ? view(result->quadrupole_moments) : view(empty_buffer);
  const BufferView wiberg_output = result_v2 ? view(result->wiberg_orders) : view(empty_buffer);
  const BufferView spin_output = result_v2 ? view(result->spin_populations) : view(empty_buffer);
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
      {"interaction_descriptors", interaction_descriptor_view(*batch)},
      {"interaction_payload", interaction_payload_view(*batch)},
      {"energies", result == nullptr ? view(empty_buffer) : view(result->energies)},
      {"forces", result == nullptr ? view(empty_buffer) : view(result->forces)},
      {"atomic_charges", result == nullptr ? view(empty_buffer) : view(result->atomic_charges)},
      {"point_charge_forces",
       result == nullptr ? view(empty_buffer) : view(result->point_charge_forces)},
      {"scc_iterations", result == nullptr ? view(empty_buffer) : view(result->scc_iterations)},
      {"scc_converged", result == nullptr ? view(empty_buffer) : view(result->scc_converged)},
      {"per_system_status",
       result == nullptr ? view(empty_buffer) : view(result->per_system_status)},
      {"dipole_moments", dipole_output},
      {"quadrupole_moments", quadrupole_output},
      {"wiberg_orders", wiberg_output},
      {"spin_populations", spin_output},
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

  if (result != nullptr && has_result_v2_suffix(*result)) {
    /* The ABI-v2 result suffix reserves outlets whose shape contract is not
     * published yet. A caller that supplies bytes there has an unfillable
     * request; refuse it instead of silently ignoring their buffer. */
    const BufferView reserved_outputs[] = {view(result->quadrupole_moments),
                                           view(result->wiberg_orders),
                                           view(result->spin_populations)};
    for (const BufferView& reserved : reserved_outputs) {
      if (active(reserved)) {
        return unsupported(
            "a reserved batch-result outlet is supplied but has no released "
            "shape contract yet");
      }
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
  std::size_t interaction_descriptor_bytes = 0;
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
  if (has_interaction_suffix(*batch)) {
    if (batch->total_interactions < 0) {
      return invalid("total_interactions must be nonnegative");
    }
    if (!count_bytes(batch->total_interactions, 1, sizeof(xtbloom_interaction_t),
                     interaction_descriptor_bytes)) {
      return invalid("total_interactions overflows the addressable byte size");
    }
  }
  /* batch_size * 3 doubles for per-system dipole moments; independent of the
   * interaction suffix because the ABI-v2 result outlet is unconditional. */
  std::size_t dipole_f64_bytes = 0;
  if (!checked_multiply(batch_f64_bytes, 3u, dipole_f64_bytes)) {
    return invalid("the dipole-moment extent overflows the addressable byte size");
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

  const BufferView interaction_descriptors = interaction_descriptor_view(*batch);
  const BufferView interaction_payload = interaction_payload_view(*batch);
  if (has_interaction_suffix(*batch) && batch->total_interactions != 0) {
    /* A nonzero attachment count requires the descriptor array and a payload
     * block store. The payload is a byte blob with no declared size of its
     * own; per-tag block extents are proven from the descriptor entries in the
     * host-semantics pass or after CUDA staging. */
    DescriptorValidationResult checked = require_bytes(
        "interaction_descriptors", interaction_descriptors, interaction_descriptor_bytes);
    if (!checked.ok()) {
      return checked;
    }
    if (!active(interaction_payload)) {
      return invalid("interaction_payload is required when interactions are present");
    }
  } else if (active(interaction_descriptors)) {
    /* A zero-count descriptor array is legal, but if present it must describe
     * at least the empty batch and remain free of declared bytes it cannot
     * actually reference. */
    DescriptorValidationResult checked =
        require_bytes("interaction_descriptors", interaction_descriptors, 0u);
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

  if (result != nullptr) {
    const RequiredOutput outputs[] = {
        {"energies", view(result->energies), batch_f64_bytes,
         (options->flags & XTBLOOM_COMPUTE_ENERGY) != 0},
        {"forces", view(result->forces), position_bytes,
         (options->flags & XTBLOOM_COMPUTE_FORCES) != 0},
        {"atomic_charges", view(result->atomic_charges), atom_f64_bytes,
         (options->flags & XTBLOOM_COMPUTE_ATOMIC_CHARGES) != 0},
        {"point_charge_forces", view(result->point_charge_forces), point_position_bytes,
         (options->flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0},
        {"scc_iterations", view(result->scc_iterations), batch_i32_bytes, true},
        {"scc_converged", view(result->scc_converged), batch_u8_bytes, true},
        {"per_system_status", view(result->per_system_status), batch_i32_bytes, true},
        {"dipole_moments", dipole_output, dipole_f64_bytes,
         (options->flags & XTBLOOM_COMPUTE_DIPOLE_MOMENTS) != 0},
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
  extents.interaction_descriptor_bytes = interaction_descriptor_bytes;
  extents.dipole_f64_bytes = dipole_f64_bytes;
  extents.response_enabled = response_enabled;
  extents.interactions_enabled = has_interaction_suffix(*batch) && batch->total_interactions != 0;
  return {};
}

DescriptorValidationResult validate_compute_descriptor_aliases(
    const xtbloom_batch_t& batch, const xtbloom_compute_options_t& options,
    const xtbloom_batch_result_t* result, const DescriptorExtentState& extents) {
  /* One fixed entry per known buffer keeps successful validation allocation-free. */
  std::array<ActiveRange, 28> ranges{};
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
  if (extents.interactions_enabled) {
    DescriptorValidationResult checked = add_active_range(
        ranges, range_count, "interaction_descriptors", interaction_descriptor_view(batch),
        extents.interaction_descriptor_bytes, false);
    if (!checked.ok()) {
      return checked;
    }
    const BufferView payload = interaction_payload_view(batch);
    checked = add_active_range(ranges, range_count, "interaction_payload", payload,
                               payload.size_bytes, false);
    if (!checked.ok()) {
      return checked;
    }
  }
  if (result != nullptr) {
    const RequiredOutput outputs[] = {
        {"energies", view(result->energies), extents.batch_f64_bytes,
         (options.flags & XTBLOOM_COMPUTE_ENERGY) != 0},
        {"forces", view(result->forces), extents.position_bytes,
         (options.flags & XTBLOOM_COMPUTE_FORCES) != 0},
        {"atomic_charges", view(result->atomic_charges), extents.atom_f64_bytes,
         (options.flags & XTBLOOM_COMPUTE_ATOMIC_CHARGES) != 0},
        {"point_charge_forces", view(result->point_charge_forces), extents.point_position_bytes,
         (options.flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES) != 0},
        {"scc_iterations", view(result->scc_iterations), extents.batch_i32_bytes, true},
        {"scc_converged", view(result->scc_converged), extents.batch_u8_bytes, true},
        {"per_system_status", view(result->per_system_status), extents.batch_i32_bytes, true},
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
    if (has_result_v2_suffix(*result) && (options.flags & XTBLOOM_COMPUTE_DIPOLE_MOMENTS) != 0) {
      DescriptorValidationResult checked =
          add_active_range(ranges, range_count, "dipole_moments", view(result->dipole_moments),
                           extents.dipole_f64_bytes, true);
      if (!checked.ok()) {
        return checked;
      }
    }
  }

  DescriptorValidationResult alias_result = validate_aliases(ranges, range_count);
  if (!alias_result.ok()) {
    return alias_result;
  }
  return {};
}

bool is_known_interaction_type(std::int32_t type) {
  switch (type) {
    case XTBLOOM_INTERACTION_ELECTRIC_FIELD:
    case XTBLOOM_INTERACTION_ELECTRIC_FIELD_GRADIENT:
    case XTBLOOM_INTERACTION_POINT_CHARGES_MULTIPOLE:
    case XTBLOOM_INTERACTION_ATOMIC_POTENTIAL_GRID:
    case XTBLOOM_INTERACTION_ALPB_SOLVATION:
    case XTBLOOM_INTERACTION_GBSA_SOLVATION:
    case XTBLOOM_INTERACTION_GB_SOLVATION:
    case XTBLOOM_INTERACTION_GBE_SOLVATION:
    case XTBLOOM_INTERACTION_DDX_SOLVATION:
    case XTBLOOM_INTERACTION_D3_DISPERSION:
    case XTBLOOM_INTERACTION_D4_VARIANT_DISPERSION:
    case XTBLOOM_INTERACTION_HALOGEN_BOND:
      return true;
    default:
      return false;
  }
}

/*
 * Validate the attachment contract from host-resident descriptor storage.
 *
 * This pass dereferences only HOST-tagged interaction_descriptors bytes. A
 * CUDA-device-resident descriptor array is opaque here and reported through
 * kInteractionDescriptorsNeedStaging for the backend to validate after
 * staging. A device-resident payload is reported independently through
 * kInteractionPayloadNeedsStaging; host payload headers and released numeric
 * contracts are validated here with byte-exact loads.
 *
 * Every tag currently passes structural validation and is then refused by
 * validate_interaction_execution_availability because no backend executes
 * interactions yet; this function exists so malformed attachments produce a
 * precise diagnostic before that refusal.
 */
DescriptorValidationResult validate_host_interaction_semantics(const xtbloom_batch_t& batch) {
  if (!has_interaction_suffix(batch) || batch.total_interactions == 0) {
    return {};
  }
  const BufferView descriptors = interaction_descriptor_view(batch);
  const BufferView payload = interaction_payload_view(batch);
  DescriptorValidationResult validation;
  if (payload.memory_space != XTBLOOM_MEMORY_HOST) {
    validation.pending_offset_checks |= kInteractionPayloadNeedsStaging;
  }
  if (descriptors.memory_space != XTBLOOM_MEMORY_HOST) {
    validation.pending_offset_checks |= kInteractionDescriptorsNeedStaging;
    return validation;
  }
  const std::size_t count = static_cast<std::size_t>(batch.total_interactions);
  for (std::size_t index = 0; index < count; ++index) {
    const InteractionView interaction = load_interaction(descriptors, index);
    if (interaction.flags != 0u) {
      return invalid("interaction descriptor flags must be zero");
    }
    if (interaction.type == XTBLOOM_INTERACTION_NONE ||
        !is_known_interaction_type(interaction.type)) {
      return invalid("an interaction descriptor uses an unknown or NONE type tag");
    }
    if (interaction.system_index < 0 || static_cast<std::uint64_t>(interaction.system_index) >=
                                            static_cast<std::uint64_t>(batch.batch_size)) {
      return invalid("an interaction system_index lies outside the batch");
    }
    /* Reject a payload block that extends past the declared payload view
     * without constructing an end address. */
    if (interaction.payload_offset > payload.size_bytes ||
        interaction.payload_size >
            static_cast<std::uint64_t>(payload.size_bytes) - interaction.payload_offset) {
      return invalid("an interaction payload block extends past interaction_payload");
    }
    if (interaction.type != XTBLOOM_INTERACTION_ELECTRIC_FIELD &&
        (interaction.payload_size < sizeof(std::int32_t) ||
         (interaction.payload_offset % alignof(std::int32_t)) != 0u)) {
      return invalid("an interaction payload block must contain an aligned block_version");
    }
    if (interaction.type == XTBLOOM_INTERACTION_ELECTRIC_FIELD) {
      /* Released block contract for block_version 1: 32 bytes (one int32_t
       * version, one int32_t reserved, three doubles) with 8-byte alignment. */
      if (interaction.payload_size < 32u || (interaction.payload_offset % 8u) != 0u) {
        return invalid("an electric-field interaction payload block is undersized or misaligned");
      }
      if (interaction.payload_size > 32u) {
        /* The released block contract fixes the headline length; a larger
         * block belongs to a later block_version that this library cannot
         * interpret yet. */
        return invalid("an electric-field interaction payload block exceeds the released contract");
      }
      if (payload.memory_space == XTBLOOM_MEMORY_HOST) {
        const unsigned char* block = static_cast<const unsigned char*>(payload.data) +
                                     static_cast<std::size_t>(interaction.payload_offset);
        std::int32_t block_version = 0;
        std::int32_t reserved = 0;
        std::array<double, 3> field{};
        std::memcpy(&block_version, block, sizeof(block_version));
        std::memcpy(&reserved, block + sizeof(block_version), sizeof(reserved));
        std::memcpy(field.data(), block + 2u * sizeof(std::int32_t), sizeof(field));
        if (block_version != 1) {
          return invalid("an electric-field interaction uses an unsupported block_version");
        }
        if (reserved != 0) {
          return invalid("an electric-field interaction reserved payload field must be zero");
        }
        if (!std::all_of(field.begin(), field.end(),
                         [](double component) { return std::isfinite(component); })) {
          return invalid("an electric-field interaction contains NaN or infinity");
        }
      }
    }
  }
  /* At most one attachment per (system, type): a second attachment would make
   * the interaction ambiguous rather than stronger. Reloading keeps this
   * successful path allocation-free. */
  for (std::size_t lhs = 0; lhs < count; ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < count; ++rhs) {
      const InteractionView left = load_interaction(descriptors, lhs);
      const InteractionView right = load_interaction(descriptors, rhs);
      if (left.system_index == right.system_index && left.type == right.type) {
        return invalid("the same system has two attachments of the same interaction type");
      }
    }
  }
  return validation;
}

DescriptorValidationResult validate_output_execution_availability(
    const xtbloom_compute_options_t& options, xtbloom_backend_t backend) {
  if ((options.flags & XTBLOOM_COMPUTE_DIPOLE_MOMENTS) != 0u && backend == XTBLOOM_BACKEND_CUDA) {
    /* The CUDA backend has not released dipole-moment publication yet (see
     * #237 P3). This runs after output shape and alias validation so
     * malformed requests still receive precise errors. */
    return {XTBLOOM_STATUS_NOT_IMPLEMENTED, kNoOffsetValidationPending,
            "dipole-moment output is implemented by the CPU backend but is not "
            "released on the CUDA backend yet"};
  }
  return {};
}

/*
 * Dispatch interactions to the backend execution availability.
 *
 * P1 (#237) refused every well-formed attachment after structural validation so
 * a caller could never believe a reserved interaction contributed to the
 * result. The released uniform electric field is now implemented by the CPU
 * backend; every other reserved tag remains refused here so a caller can never
 * believe a reserved interaction contributed to the result.
 *
 * This entry point is shared by the CPU and CUDA structural validators. On the
 * CUDA path the descriptor bytes may be device-resident and must never be
 * dereferenced on the host: the CUDA backend refuses every interaction before
 * staged content validation (see #237 P3), so only host-resident descriptors
 * are read to distinguish the released electric field from reserved tags.
 */
DescriptorValidationResult validate_interaction_execution_availability(const xtbloom_batch_t& batch,
                                                                       xtbloom_backend_t backend) {
  if (!has_interaction_suffix(batch) || batch.total_interactions == 0) {
    return {};
  }
  const BufferView descriptors = interaction_descriptor_view(batch);
  if (backend == XTBLOOM_BACKEND_CUDA) {
    return {XTBLOOM_STATUS_NOT_IMPLEMENTED, kNoOffsetValidationPending,
            "interaction execution is implemented by the CPU backend but is "
            "not released on the CUDA backend yet"};
  }
  if (descriptors.memory_space != XTBLOOM_MEMORY_HOST) {
    /* The CPU backend requires host-resident descriptors (validated earlier),
     * so this branch is defensive and unreachable; refuse without reading. */
    return {XTBLOOM_STATUS_NOT_IMPLEMENTED, kNoOffsetValidationPending,
            "interaction execution requires host-resident descriptor storage"};
  }
  const std::size_t count = static_cast<std::size_t>(batch.total_interactions);
  for (std::size_t index = 0; index < count; ++index) {
    const InteractionView interaction =
        load_interaction(descriptors, static_cast<std::size_t>(index));
    if (interaction.type != XTBLOOM_INTERACTION_ELECTRIC_FIELD) {
      return {XTBLOOM_STATUS_NOT_IMPLEMENTED, kNoOffsetValidationPending,
              "an interaction attachment uses a tag whose backend execution is "
              "reserved but not implemented yet"};
    }
  }
  return {};
}

}  // namespace

DescriptorValidationResult validate_compute_descriptor_structure(
    xtbloom_backend_t backend, const xtbloom_batch_t* batch,
    const xtbloom_compute_options_t* options, const xtbloom_batch_result_t* result) {
  if (result == nullptr) {
    return invalid("batch result is NULL");
  }
  DescriptorExtentState extents;
  DescriptorValidationResult prefix =
      validate_compute_descriptor_prefix(backend, batch, options, result, extents);
  if (!prefix.ok()) {
    return prefix;
  }
  DescriptorValidationResult aliases =
      validate_compute_descriptor_aliases(*batch, *options, result, extents);
  if (!aliases.ok()) {
    return aliases;
  }
  DescriptorValidationResult output_availability =
      validate_output_execution_availability(*options, backend);
  if (!output_availability.ok()) {
    return output_availability;
  }
  return validate_interaction_execution_availability(*batch, backend);
}

DescriptorValidationResult validate_plan_descriptor_structure(
    xtbloom_backend_t backend, const xtbloom_batch_t* batch,
    const xtbloom_compute_options_t* options) {
  /* Plan creation has no result descriptor yet; the batch plus the compute
   * policy that sizes the plan workspace are validated without requiring
   * caller-owned output buffers. Returned checks that must be deferred to a
   * later backend staging step are reported through the same pending mask. */
  DescriptorExtentState extents;
  DescriptorValidationResult prefix =
      validate_compute_descriptor_prefix(backend, batch, options, nullptr, extents);
  if (!prefix.ok()) {
    return prefix;
  }
  DescriptorValidationResult semantics;
  if (backend == XTBLOOM_BACKEND_CPU) {
    semantics = validate_host_topology_semantics(*batch);
    if (!semantics.ok()) {
      return semantics;
    }
    DescriptorValidationResult interaction_semantics = validate_host_interaction_semantics(*batch);
    if (!interaction_semantics.ok()) {
      return interaction_semantics;
    }
    semantics.pending_offset_checks |= interaction_semantics.pending_offset_checks;
  }
  DescriptorValidationResult aliases =
      validate_compute_descriptor_aliases(*batch, *options, nullptr, extents);
  if (!aliases.ok()) {
    return aliases;
  }
  DescriptorValidationResult output_availability =
      validate_output_execution_availability(*options, backend);
  if (!output_availability.ok()) {
    return output_availability;
  }
  return validate_interaction_execution_availability(*batch, backend);
}

DescriptorValidationResult validate_host_topology_semantics(const xtbloom_batch_t& batch) {
  DescriptorValidationResult validation;
  const BufferView atom_offsets = view(batch.atom_offsets);
  if (atom_offsets.memory_space == XTBLOOM_MEMORY_HOST) {
    validation = validate_host_offsets("atom_offsets", atom_offsets, batch.batch_size,
                                       batch.total_atoms, true);
    if (!validation.ok()) {
      return validation;
    }
  } else {
    validation.pending_offset_checks |= kAtomOffsetsNeedStaging;
  }

  const BufferView atomic_numbers = view(batch.atomic_numbers);
  if (atomic_numbers.memory_space != XTBLOOM_MEMORY_HOST) {
    validation.pending_offset_checks |= kAtomicNumbersNeedStaging;
  }
  const BufferView molecular_charges = view(batch.molecular_charges);
  if (molecular_charges.memory_space != XTBLOOM_MEMORY_HOST) {
    validation.pending_offset_checks |= kMolecularChargesNeedStaging;
  }
  const BufferView unpaired_electrons = view(batch.unpaired_electrons);
  if (unpaired_electrons.memory_space != XTBLOOM_MEMORY_HOST) {
    validation.pending_offset_checks |= kUnpairedElectronsNeedStaging;
  }
  const BufferView spin_channels = spin_channel_view(batch);
  if (active(spin_channels)) {
    if (spin_channels.memory_space != XTBLOOM_MEMORY_HOST) {
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
    if (point_offsets.memory_space == XTBLOOM_MEMORY_HOST) {
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
    if (response_offsets.memory_space == XTBLOOM_MEMORY_HOST) {
      DescriptorValidationResult checked =
          validate_host_offsets("charge_response_offsets", response_offsets, batch.batch_size,
                                batch.total_charge_response_elements, false);
      if (!checked.ok()) {
        return checked;
      }
    } else {
      validation.pending_offset_checks |= kChargeResponseOffsetsNeedStaging;
    }

    const bool atom_offsets_on_host = atom_offsets.memory_space == XTBLOOM_MEMORY_HOST;
    const bool response_offsets_on_host = response_offsets.memory_space == XTBLOOM_MEMORY_HOST;
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

DescriptorValidationResult validate_compute_descriptors(xtbloom_backend_t backend,
                                                        const xtbloom_batch_t* batch,
                                                        const xtbloom_compute_options_t* options,
                                                        const xtbloom_batch_result_t* result) {
  if (result == nullptr) {
    return invalid("batch result is NULL");
  }
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
  DescriptorValidationResult interaction_semantics = validate_host_interaction_semantics(*batch);
  if (!interaction_semantics.ok()) {
    return interaction_semantics;
  }
  semantics.pending_offset_checks |= interaction_semantics.pending_offset_checks;
  DescriptorValidationResult aliases =
      validate_compute_descriptor_aliases(*batch, *options, result, extents);
  if (!aliases.ok()) {
    return aliases;
  }
  DescriptorValidationResult output_availability =
      validate_output_execution_availability(*options, backend);
  if (!output_availability.ok()) {
    return output_availability;
  }
  DescriptorValidationResult availability =
      validate_interaction_execution_availability(*batch, backend);
  if (!availability.ok()) {
    return availability;
  }
  return semantics;
}

}  // namespace xtbloom::detail
