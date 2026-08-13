#ifndef XTBLOOM_XTBLOOM_H
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_XTBLOOM_H

#include <stddef.h>
#include <stdint.h>

#include "xtbloom/version.h"

#if defined(_WIN32)
#if defined(XTBLOOM_BUILDING_LIBRARY)
#define XTBLOOM_API __declspec(dllexport)
#else
#define XTBLOOM_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define XTBLOOM_API __attribute__((visibility("default")))
#else
#define XTBLOOM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Increment this value only when an ABI-incompatible C API change is made. */
#define XTBLOOM_API_VERSION 1u

/*
 * Electronic temperatures are k_B*T energy scales in Hartree, not kelvin.
 * This conversion matches the pinned xTB/tblite convention used by xtbloom.
 */
#define XTBLOOM_KELVIN_TO_HARTREE 3.166808578545117e-6
#define XTBLOOM_DEFAULT_ELECTRONIC_TEMPERATURE (300.0 * XTBLOOM_KELVIN_TO_HARTREE)

typedef struct xtbloom_context xtbloom_context_t;

/* Opaque fixed-topology plan handle. See xtbloom_plan_create. */
typedef struct xtbloom_plan xtbloom_plan_t;

/*
 * Opaque reusable owner of one native asynchronous submission. See
 * xtbloom_request_create. A request is bound to the context that created it
 * and must be destroyed before that context.
 */
typedef struct xtbloom_request xtbloom_request_t;

/*
 * Opaque owner of one xtbloom-allocated result arena. See
 * xtbloom_result_owner_create. A result owner is a ref-counted host or CUDA
 * device allocation that outlives the compute context used to fill it, so a
 * DLPack producer can hand the finished bytes to an importing framework
 * without a host round trip and without keeping the context alive.
 */
typedef struct xtbloom_result_owner xtbloom_result_owner_t;

/*
 * ABI tags are explicitly int32_t rather than enum-typed fields. This keeps
 * their object representation and function calling convention identical in C
 * and C++, including C99 builds compiled with options such as -fshort-enums.
 * The named enums below only provide debugger-friendly symbolic constants;
 * callers may pass any int32_t bit pattern and the library validates it.
 */
typedef int32_t xtbloom_status_t;
enum xtbloom_status_value {
  XTBLOOM_STATUS_SUCCESS = 0,
  XTBLOOM_STATUS_INVALID_ARGUMENT = 1,
  XTBLOOM_STATUS_BACKEND_UNAVAILABLE = 2,
  XTBLOOM_STATUS_NOT_SUPPORTED = 3,
  XTBLOOM_STATUS_ALLOCATION_FAILED = 4,
  XTBLOOM_STATUS_NOT_IMPLEMENTED = 5,
  XTBLOOM_STATUS_INTERNAL_ERROR = 6,
  /* Per-system SCC reached max_scc_iterations without satisfying both tolerances. */
  XTBLOOM_STATUS_SCC_NOT_CONVERGED = 7,
  /* Per-system generalized eigensolver failed or produced an unusable eigensystem. */
  XTBLOOM_STATUS_EIGENSOLVER_FAILED = 8
};

typedef int32_t xtbloom_request_state_t;
enum xtbloom_request_state_value {
  /* No submission has ever been accepted by this request. */
  XTBLOOM_REQUEST_IDLE = 0,
  /* Native work has been accepted and may still access caller-owned buffers. */
  XTBLOOM_REQUEST_PENDING = 1,
  /* Native work and result publication have finished, successfully or not. */
  XTBLOOM_REQUEST_COMPLETE = 2
};

typedef int32_t xtbloom_backend_t;
enum xtbloom_backend_value {
  /* Prefer CUDA when it is compiled in and a compatible device is present. */
  XTBLOOM_BACKEND_AUTO = 0,
  XTBLOOM_BACKEND_CPU = 1,
  XTBLOOM_BACKEND_CUDA = 2,
  /* Reserved now so adding HIP kernels does not require redesigning the ABI. */
  XTBLOOM_BACKEND_ROCM = 3
};

typedef int32_t xtbloom_memory_space_t;
enum xtbloom_memory_space_value {
  XTBLOOM_MEMORY_HOST = 0,
  XTBLOOM_MEMORY_CUDA_DEVICE = 1,
  XTBLOOM_MEMORY_ROCM_DEVICE = 2
};

typedef int32_t xtbloom_model_t;
enum xtbloom_model_value { XTBLOOM_MODEL_GFN1_XTB = 1, XTBLOOM_MODEL_GFN2_XTB = 2 };

typedef int32_t xtbloom_scc_start_mode_t;
enum xtbloom_scc_start_mode_value {
  /* Restore the immutable initial electronic state before SCC. */
  XTBLOOM_SCC_START_FRESH = 1,
  /* Strictly consume a checkpoint from the latest fully converged compatible batch call. */
  XTBLOOM_SCC_START_WARM = 2
};

typedef int32_t xtbloom_scc_mixer_t;
enum xtbloom_scc_mixer_value {
  /* Johnson modified-Broyden mixing used by the GFN2 CPU and CUDA backends. */
  XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN = 1
};

typedef int32_t xtbloom_determinism_t;
enum xtbloom_determinism_value {
  /* Use the production execution policy selected by the backend. */
  XTBLOOM_DETERMINISM_DEFAULT = 0,
  /*
   * Request exact replay for an unchanged build, backend, numerical provider
   * or CUDA toolkit, device architecture, complete descriptors/options,
   * launch/bucket geometry, and FRESH/WARM sequence. This is not a bitwise
   * CPU/CUDA, cross-provider, cross-toolkit, or cross-architecture promise.
   */
  XTBLOOM_DETERMINISM_REPRODUCIBLE = 1
};

typedef int32_t xtbloom_compute_flag_t;
enum xtbloom_compute_flag_value {
  XTBLOOM_COMPUTE_ENERGY = 1 << 0,
  XTBLOOM_COMPUTE_FORCES = 1 << 1,
  XTBLOOM_COMPUTE_ATOMIC_CHARGES = 1 << 2,
  XTBLOOM_COMPUTE_POINT_CHARGE_FORCES = 1 << 3,
  /* Reports per-system dipole moments through batch_result.dipole_moments. */
  XTBLOOM_COMPUTE_DIPOLE_MOMENTS = 1 << 4
  /* Bits 16-31 are reserved for future outputs and must be zero on input. */
};

typedef int32_t xtbloom_result_flag_t;
enum xtbloom_result_flag_value {
  /*
   * Set when atomic_potential_shifts or charge_response_matrix was supplied.
   * Forces then exclude coordinate derivatives of those caller-owned fields.
   */
  XTBLOOM_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES = 1 << 0,
  /* Set when the requested per-system dipole moments were published. */
  XTBLOOM_RESULT_DIPOLE_MOMENTS = 1 << 4
  /* Bits 16-31 are reserved; xtbloom-produced result flags are zero there. */
};

/*
 * Tag set for the generic interaction attachment slot (ABI-v3 batch suffix).
 * Tag values are intentionally spread over family ranges so future additions
 * never renumber an existing value. The GFN2 backends currently implement
 * none of these interactions: validating any present interaction returns
 * XTBLOOM_STATUS_NOT_IMPLEMENTED until the matching backend term lands, and
 * the enum values are reserved so later features do not churn the batch
 * layout. XTBLOOM_INTERACTION_NONE is not a valid attachment.
 */
typedef int32_t xtbloom_interaction_type_t;
enum xtbloom_interaction_type_value {
  XTBLOOM_INTERACTION_NONE = 0,
  /* External potentials (0x01xx). */
  XTBLOOM_INTERACTION_ELECTRIC_FIELD = 0x0101,
  XTBLOOM_INTERACTION_ELECTRIC_FIELD_GRADIENT = 0x0102,
  XTBLOOM_INTERACTION_POINT_CHARGES_MULTIPOLE = 0x0103,
  XTBLOOM_INTERACTION_ATOMIC_POTENTIAL_GRID = 0x0104,
  /* Self-consistent solvation models (0x02xx). */
  XTBLOOM_INTERACTION_ALPB_SOLVATION = 0x0201,
  XTBLOOM_INTERACTION_GBSA_SOLVATION = 0x0202,
  XTBLOOM_INTERACTION_GB_SOLVATION = 0x0203,
  XTBLOOM_INTERACTION_GBE_SOLVATION = 0x0204,
  XTBLOOM_INTERACTION_DDX_SOLVATION = 0x0205,
  /* Dispersion models (0x03xx). */
  XTBLOOM_INTERACTION_D3_DISPERSION = 0x0301,
  XTBLOOM_INTERACTION_D4_VARIANT_DISPERSION = 0x0302,
  /* Structure-correction models (0x04xx). */
  XTBLOOM_INTERACTION_HALOGEN_BOND = 0x0401
};

/*
 * Periodic-axis mask for the ABI-v4 native-lattice batch suffix.
 *
 * The individual x/y/z bits are reserved so later releases can describe
 * lower-dimensional boundary conditions without changing the field width.
 * This release accepts NONE for a molecular batch item and XYZ for a native
 * three-dimensional periodic item. Partial masks are not implemented.
 */
typedef int32_t xtbloom_periodic_axes_t;
enum xtbloom_periodic_axes_value {
  XTBLOOM_PERIODIC_AXES_NONE = 0,
  XTBLOOM_PERIODIC_AXIS_X = 1 << 0,
  XTBLOOM_PERIODIC_AXIS_Y = 1 << 1,
  XTBLOOM_PERIODIC_AXIS_Z = 1 << 2,
  XTBLOOM_PERIODIC_AXES_XYZ =
      XTBLOOM_PERIODIC_AXIS_X | XTBLOOM_PERIODIC_AXIS_Y | XTBLOOM_PERIODIC_AXIS_Z
};

/*
 * One attachment of an external interaction to one batch item.
 *
 * type selects the interaction; flags is reserved and must be zero;
 * system_index addresses one batch item in [0, batch_size); payload_offset
 * and payload_size locate the caller-owned payload block inside
 * interaction_payload. Every payload block starts with an int32_t
 * block_version so the byte layout of one tag can evolve independently of
 * this descriptor. The electric-field block (block_version 1) is
 * 32 bytes: int32_t version, int32_t reserved (zero), three finite doubles
 * holding the field vector in Hartree per elementary charge per bohr.
 */
typedef struct xtbloom_interaction {
  xtbloom_interaction_type_t type;
  uint32_t flags;
  int64_t system_index;
  uint64_t payload_offset;
  uint64_t payload_size;
} xtbloom_interaction_t;

#define XTBLOOM_INTERACTION_V1_SIZE \
  (offsetof(xtbloom_interaction_t, payload_size) + sizeof(uint64_t))

/*
 * Keep all public ABI tag and flag aliases at their specified width.
 */
#if defined(__cplusplus)
static_assert(sizeof(xtbloom_status_t) == sizeof(int32_t), "xtbloom_status_t must be 32-bit");
static_assert(sizeof(xtbloom_request_state_t) == sizeof(int32_t),
              "xtbloom_request_state_t must be 32-bit");
static_assert(sizeof(xtbloom_backend_t) == sizeof(int32_t), "xtbloom_backend_t must be 32-bit");
static_assert(sizeof(xtbloom_memory_space_t) == sizeof(int32_t),
              "xtbloom_memory_space_t must be 32-bit");
static_assert(sizeof(xtbloom_model_t) == sizeof(int32_t), "xtbloom_model_t must be 32-bit");
static_assert(sizeof(xtbloom_scc_start_mode_t) == sizeof(int32_t),
              "xtbloom_scc_start_mode_t must be 32-bit");
static_assert(sizeof(xtbloom_scc_mixer_t) == sizeof(int32_t), "xtbloom_scc_mixer_t must be 32-bit");
static_assert(sizeof(xtbloom_determinism_t) == sizeof(int32_t),
              "xtbloom_determinism_t must be 32-bit");
static_assert(sizeof(xtbloom_compute_flag_t) == sizeof(int32_t),
              "xtbloom_compute_flag_t must be 32-bit");
static_assert(sizeof(xtbloom_result_flag_t) == sizeof(int32_t),
              "xtbloom_result_flag_t must be 32-bit");
static_assert(sizeof(xtbloom_interaction_type_t) == sizeof(int32_t),
              "xtbloom_interaction_type_t must be 32-bit");
static_assert(sizeof(xtbloom_periodic_axes_t) == sizeof(int32_t),
              "xtbloom_periodic_axes_t must be 32-bit");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(xtbloom_status_t) == sizeof(int32_t), "xtbloom_status_t must be 32-bit");
_Static_assert(sizeof(xtbloom_request_state_t) == sizeof(int32_t),
               "xtbloom_request_state_t must be 32-bit");
_Static_assert(sizeof(xtbloom_backend_t) == sizeof(int32_t), "xtbloom_backend_t must be 32-bit");
_Static_assert(sizeof(xtbloom_memory_space_t) == sizeof(int32_t),
               "xtbloom_memory_space_t must be 32-bit");
_Static_assert(sizeof(xtbloom_model_t) == sizeof(int32_t), "xtbloom_model_t must be 32-bit");
_Static_assert(sizeof(xtbloom_scc_start_mode_t) == sizeof(int32_t),
               "xtbloom_scc_start_mode_t must be 32-bit");
_Static_assert(sizeof(xtbloom_scc_mixer_t) == sizeof(int32_t),
               "xtbloom_scc_mixer_t must be 32-bit");
_Static_assert(sizeof(xtbloom_determinism_t) == sizeof(int32_t),
               "xtbloom_determinism_t must be 32-bit");
_Static_assert(sizeof(xtbloom_compute_flag_t) == sizeof(int32_t),
               "xtbloom_compute_flag_t must be 32-bit");
_Static_assert(sizeof(xtbloom_result_flag_t) == sizeof(int32_t),
               "xtbloom_result_flag_t must be 32-bit");
_Static_assert(sizeof(xtbloom_interaction_type_t) == sizeof(int32_t),
               "xtbloom_interaction_type_t must be 32-bit");
_Static_assert(sizeof(xtbloom_periodic_axes_t) == sizeof(int32_t),
               "xtbloom_periodic_axes_t must be 32-bit");
#endif

/*
 * Every extensible structure starts with struct_size and api_version. Pass the
 * caller's sizeof(struct) to its initializer before overriding fields. This
 * lets newer and older libraries initialize only the structure prefix known to
 * both sides and reject accidental ABI mismatches.
 */
typedef struct xtbloom_context_options {
  uint32_t struct_size;
  uint32_t api_version;
  xtbloom_backend_t backend;
  int32_t device_id;
  /* CPU batch parallelism: zero selects automatic, one disables parallelism. */
  int32_t cpu_threads;
  uint32_t reserved;
  /* Native cudaStream_t or hipStream_t cast to void*. NULL selects the default. */
  void* stream;
} xtbloom_context_options_t;

#define XTBLOOM_CONTEXT_OPTIONS_V1_SIZE \
  (offsetof(xtbloom_context_options_t, stream) + sizeof(void*))

/* A byte-sized view of caller-owned input memory. xtbloom never takes ownership. */
typedef struct xtbloom_const_buffer {
  const void* data;
  size_t size_bytes;
  xtbloom_memory_space_t memory_space;
  uint32_t reserved;
} xtbloom_const_buffer_t;

/* A byte-sized view of caller-owned output memory. xtbloom never takes ownership. */
typedef struct xtbloom_buffer {
  void* data;
  size_t size_bytes;
  xtbloom_memory_space_t memory_space;
  uint32_t reserved;
} xtbloom_buffer_t;

/*
 * Pointer-bearing ABI images are architecture-local. A wasm32/native ILP32
 * caller and library use 32-bit pointers and size_t, while wasm64/native LP64
 * uses 64-bit pointers and size_t. Fixed-width counts and offsets below remain
 * 64-bit on both targets. Keep exact assertions for both supported widths so a
 * compiler or packing change fails at build time instead of corrupting views.
 */
#if UINTPTR_MAX == UINT64_MAX
#define XTBLOOM_DETAIL_EXPECTED_CONTEXT_OPTIONS_V1_SIZE 32u
#define XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE 24u
#define XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE_BYTES_OFFSET 8u
#define XTBLOOM_DETAIL_EXPECTED_BUFFER_MEMORY_SPACE_OFFSET 16u
#define XTBLOOM_DETAIL_EXPECTED_BUFFER_RESERVED_OFFSET 20u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_V1_SIZE 328u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_V2_SIZE 352u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET 352u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_INTERACTION_DESCRIPTORS_OFFSET 360u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_INTERACTION_PAYLOAD_OFFSET 384u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_V3_SIZE 408u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_CELL_MATRICES_OFFSET 408u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_PERIODIC_AXES_OFFSET 432u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_V4_SIZE 456u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_V1_SIZE 184u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_DIPOLE_OFFSET 184u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_QUADRUPOLE_OFFSET 208u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_WIBERG_OFFSET 232u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_SPIN_OFFSET 256u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_V2_SIZE 280u
#define XTBLOOM_DETAIL_EXPECTED_DLPACK_SHAPE_OFFSET 40u
#define XTBLOOM_DETAIL_EXPECTED_DLPACK_VIEW_V1_SIZE 48u
#elif UINTPTR_MAX == UINT32_MAX
#define XTBLOOM_DETAIL_EXPECTED_CONTEXT_OPTIONS_V1_SIZE 28u
#define XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE 16u
#define XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE_BYTES_OFFSET 4u
#define XTBLOOM_DETAIL_EXPECTED_BUFFER_MEMORY_SPACE_OFFSET 8u
#define XTBLOOM_DETAIL_EXPECTED_BUFFER_RESERVED_OFFSET 12u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_V1_SIZE 232u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_V2_SIZE 248u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET 248u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_INTERACTION_DESCRIPTORS_OFFSET 256u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_INTERACTION_PAYLOAD_OFFSET 272u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_V3_SIZE 288u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_CELL_MATRICES_OFFSET 288u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_PERIODIC_AXES_OFFSET 304u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_V4_SIZE 320u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_V1_SIZE 128u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_DIPOLE_OFFSET 128u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_QUADRUPOLE_OFFSET 144u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_WIBERG_OFFSET 160u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_SPIN_OFFSET 176u
#define XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_V2_SIZE 192u
#define XTBLOOM_DETAIL_EXPECTED_DLPACK_SHAPE_OFFSET 36u
#define XTBLOOM_DETAIL_EXPECTED_DLPACK_VIEW_V1_SIZE 40u
#endif

#if defined(__cplusplus)
#define XTBLOOM_DETAIL_ABI_ASSERT(condition, message) static_assert((condition), message)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define XTBLOOM_DETAIL_ABI_ASSERT(condition, message) _Static_assert((condition), message)
#endif

#if defined(XTBLOOM_DETAIL_ABI_ASSERT) && defined(XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE)
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_context_options_t, stream) == 24u,
                          "xtbloom_context_options_t stream must start at byte 24");
XTBLOOM_DETAIL_ABI_ASSERT(XTBLOOM_CONTEXT_OPTIONS_V1_SIZE ==
                              XTBLOOM_DETAIL_EXPECTED_CONTEXT_OPTIONS_V1_SIZE,
                          "xtbloom_context_options_t image must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(sizeof(xtbloom_context_options_t) == XTBLOOM_CONTEXT_OPTIONS_V1_SIZE,
                          "xtbloom_context_options_t must not add trailing ABI padding");
XTBLOOM_DETAIL_ABI_ASSERT(sizeof(xtbloom_const_buffer_t) == XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE,
                          "xtbloom_const_buffer_t image must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(sizeof(xtbloom_buffer_t) == XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE,
                          "xtbloom_buffer_t image must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_const_buffer_t, size_bytes) ==
                              XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE_BYTES_OFFSET,
                          "xtbloom_const_buffer_t size must follow its target-width pointer");
XTBLOOM_DETAIL_ABI_ASSERT(
    offsetof(xtbloom_const_buffer_t, memory_space) ==
        XTBLOOM_DETAIL_EXPECTED_BUFFER_MEMORY_SPACE_OFFSET,
    "xtbloom_const_buffer_t memory tag offset must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(
    offsetof(xtbloom_const_buffer_t, reserved) == XTBLOOM_DETAIL_EXPECTED_BUFFER_RESERVED_OFFSET,
    "xtbloom_const_buffer_t reserved offset must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_buffer_t, size_bytes) ==
                              XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE_BYTES_OFFSET,
                          "xtbloom_buffer_t size must follow its target-width pointer");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_buffer_t, memory_space) ==
                              XTBLOOM_DETAIL_EXPECTED_BUFFER_MEMORY_SPACE_OFFSET,
                          "xtbloom_buffer_t memory tag offset must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_buffer_t, reserved) ==
                              XTBLOOM_DETAIL_EXPECTED_BUFFER_RESERVED_OFFSET,
                          "xtbloom_buffer_t reserved offset must match the target pointer width");
#endif

/*
 * Ragged molecular batch. All real-valued inputs use IEEE binary64 and atomic
 * units: bohr for positions and elementary-charge units for charges.
 *
 * atom_offsets contains batch_size + 1 int64_t values. atomic_numbers contains
 * total_atoms int32_t values. positions contains total_atoms * 3 doubles.
 * molecular_charges contains batch_size doubles and unpaired_electrons contains
 * batch_size int32_t values. The ABI-v2 spin_channels field, when present,
 * contains batch_size int32_t values equal to one (restricted) or two
 * (unrestricted). A missing or NULL spin_channels buffer preserves the ABI-v1
 * restricted default.
 *
 * External point charges participate in every SCC iteration. When
 * total_point_charges is nonzero, point_charge_offsets has batch_size + 1
 * int64_t values, point_charge_positions has total_point_charges * 3 doubles,
 * point_charge_values has total_point_charges doubles, and point_charge_gammas
 * has total_point_charges doubles. Gamma is the explicit point-site screening
 * parameter used in the softened Coulomb interaction; it is a model parameter,
 * not an optimizable point-charge degree of freedom.
 *
 * atomic_potential_shifts and the response matrix are optional advanced inputs
 * for periodic QM/MM coupling. For molecule i they define a per-atom SCC shift
 * b_i + A_i q_i and variational energy q_i^T b_i + 0.5 q_i^T A_i q_i.
 * atomic_potential_shifts contains total_atoms doubles. charge_response_offsets
 * contains batch_size + 1 int64_t values, and each row-major symmetric A_i is
 * packed consecutively in charge_response_matrix. Derivatives of b and A with
 * respect to coordinates are outside xtbloom and are not included in forces.
 */
typedef struct xtbloom_batch {
  uint32_t struct_size;
  uint32_t api_version;
  int64_t batch_size;
  int64_t total_atoms;
  int64_t total_point_charges;
  int64_t total_charge_response_elements;
  xtbloom_const_buffer_t atom_offsets;
  xtbloom_const_buffer_t atomic_numbers;
  xtbloom_const_buffer_t positions;
  xtbloom_const_buffer_t molecular_charges;
  xtbloom_const_buffer_t unpaired_electrons;
  xtbloom_const_buffer_t point_charge_offsets;
  xtbloom_const_buffer_t point_charge_positions;
  xtbloom_const_buffer_t point_charge_values;
  xtbloom_const_buffer_t point_charge_gammas;
  xtbloom_const_buffer_t atomic_potential_shifts;
  xtbloom_const_buffer_t charge_response_offsets;
  xtbloom_const_buffer_t charge_response_matrix;
  /* ABI v2 optional suffix; NULL selects one restricted channel per system. */
  xtbloom_const_buffer_t spin_channels;
  /* ABI v3 optional suffix: generic external-interaction attachments.
   *
   * total_interactions counts xtbloom_interaction_t entries in
   * interaction_descriptors, each attaching one caller-owned payload block in
   * interaction_payload to one batch item. The suffix may carry any mix of
   * host and CUDA-device storage and may be present with zero interactions,
   * preserving ABI-v1/v2 behavior for callers that never use it. See
   * xtbloom_interaction_type_t for the reserved tag set and payload contract. */
  int64_t total_interactions;
  xtbloom_const_buffer_t interaction_descriptors;
  xtbloom_const_buffer_t interaction_payload;
  /* ABI v4 optional suffix: native lattice/PBC descriptors.
   *
   * When either buffer is active, both are required. cell_matrices contains
   * batch_size row-major 3x3 direct-cell matrices in bohr. The three rows are
   * the a, b, and c lattice vectors, so a fractional row vector u maps to the
   * Cartesian vector u[0]*a + u[1]*b + u[2]*c. periodic_axes contains
   * batch_size xtbloom_periodic_axes_t values. NONE requires the corresponding
   * nine cell entries to be exactly zero; XYZ requires a finite, right-handed,
   * nonsingular cell. Partial-axis masks are reserved but not implemented.
   *
   * Native PBC changes the complete GFN2 topology and is distinct from the
   * caller-supplied b + A*q charge-response operator above. A V4 batch whose
   * masks are all NONE remains a molecular request. If any item uses XYZ,
   * this ABI release validates the complete descriptor set but returns
   * NOT_IMPLEMENTED before execution until every periodic GFN2 energy and
   * derivative term is connected. */
  xtbloom_const_buffer_t cell_matrices;
  xtbloom_const_buffer_t periodic_axes;
} xtbloom_batch_t;

#define XTBLOOM_BATCH_V1_SIZE \
  (offsetof(xtbloom_batch_t, charge_response_matrix) + sizeof(xtbloom_const_buffer_t))
#define XTBLOOM_BATCH_V2_SIZE \
  (offsetof(xtbloom_batch_t, spin_channels) + sizeof(xtbloom_const_buffer_t))
#define XTBLOOM_BATCH_V3_SIZE \
  (offsetof(xtbloom_batch_t, interaction_payload) + sizeof(xtbloom_const_buffer_t))
#define XTBLOOM_BATCH_V4_SIZE \
  (offsetof(xtbloom_batch_t, periodic_axes) + sizeof(xtbloom_const_buffer_t))

/*
 * electronic_temperature is k_B*T in Hartree. Bindings that accept kelvin
 * should multiply by XTBLOOM_KELVIN_TO_HARTREE before populating this struct.
 *
 * The ABI-v2 scc_start_mode suffix is a strict per-call policy. FRESH restores
 * the immutable initial electronic state. WARM consumes the checkpoint from
 * the latest fully converged compatible public batch call; it never falls back
 * to FRESH. A V1/short prefix means FRESH.
 *
 * A compatible identity is a batch whose topology and compute policy
 * (requested-property flags, molecular charges, unpaired electrons, spin
 * channels, point-charge and periodic structure, SCC tolerances, maximum
 * iterations, electronic temperature, mixer algorithm/history/damping, and
 * determinism policy) exactly match the previous fully converged call on the
 * same context. This is the same compute-options identity used by CPU and
 * CUDA. Geometry is not part of the identity: a WARM
 * call reuses the previous converged electronic state as the initial SCC guess
 * for the new coordinates and reconverges. A WARM request with no such
 * compatible fully converged predecessor (first call, changed topology or
 * policy, or a preceding non-converged batch) is rejected with
 * XTBLOOM_STATUS_INVALID_ARGUMENT before any caller output is modified.
 *
 * An accepted FRESH attempt consumes the preceding compatible checkpoint
 * before execution starts. If that attempt later fails, including a CUDA
 * failure discovered in stream order after enqueue, no stale checkpoint from
 * an older call survives; a subsequent strict WARM request is rejected.
 */
typedef struct xtbloom_compute_options {
  uint32_t struct_size;
  uint32_t api_version;
  xtbloom_model_t model;
  uint32_t flags;
  int32_t max_scc_iterations;
  uint32_t reserved;
  double charge_tolerance;
  double energy_tolerance;
  double electronic_temperature;
  /* ABI v2 optional suffix; absent suffix preserves strict FRESH semantics. */
  xtbloom_scc_start_mode_t scc_start_mode;
  uint32_t reserved_v2;
  /*
   * ABI v3 optional suffix. A caller must provide the complete suffix or the
   * library uses modified-Broyden history 8, damping 0.4, and default
   * execution. Partial v3 suffixes are ignored as a unit.
   */
  /* Currently only XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN is accepted. */
  xtbloom_scc_mixer_t scc_mixer;
  /* Modified-Broyden history depth in [1, 64]. */
  int32_t scc_mixer_history;
  /* Linear damping factor, finite and in (0, 1]. */
  double scc_mixer_damping;
  /* XTBLOOM_DETERMINISM_DEFAULT or XTBLOOM_DETERMINISM_REPRODUCIBLE. */
  xtbloom_determinism_t determinism;
  uint32_t reserved_v3;
} xtbloom_compute_options_t;

#define XTBLOOM_COMPUTE_OPTIONS_V1_SIZE \
  (offsetof(xtbloom_compute_options_t, electronic_temperature) + sizeof(double))
#define XTBLOOM_COMPUTE_OPTIONS_V2_SIZE \
  (offsetof(xtbloom_compute_options_t, reserved_v2) + sizeof(uint32_t))
#define XTBLOOM_COMPUTE_OPTIONS_V3_SIZE \
  (offsetof(xtbloom_compute_options_t, reserved_v3) + sizeof(uint32_t))

#if defined(__cplusplus)
static_assert(offsetof(xtbloom_compute_options_t, scc_start_mode) == 48u,
              "xtbloom_compute_options_t ABI-v2 suffix must start at byte 48");
static_assert(XTBLOOM_COMPUTE_OPTIONS_V1_SIZE == 48u,
              "xtbloom_compute_options_t ABI-v1 prefix must remain 48 bytes");
static_assert(XTBLOOM_COMPUTE_OPTIONS_V2_SIZE == 56u,
              "xtbloom_compute_options_t ABI-v2 image must remain 56 bytes");
static_assert(offsetof(xtbloom_compute_options_t, scc_mixer) == 56u,
              "xtbloom_compute_options_t ABI-v3 mixer must start at byte 56");
static_assert(offsetof(xtbloom_compute_options_t, scc_mixer_history) == 60u,
              "xtbloom_compute_options_t ABI-v3 history must start at byte 60");
static_assert(offsetof(xtbloom_compute_options_t, scc_mixer_damping) == 64u,
              "xtbloom_compute_options_t ABI-v3 damping must start at byte 64");
static_assert(offsetof(xtbloom_compute_options_t, determinism) == 72u,
              "xtbloom_compute_options_t ABI-v3 determinism must start at byte 72");
static_assert(offsetof(xtbloom_compute_options_t, reserved_v3) == 76u,
              "xtbloom_compute_options_t ABI-v3 reserved field must start at byte 76");
static_assert(XTBLOOM_COMPUTE_OPTIONS_V3_SIZE == 80u,
              "xtbloom_compute_options_t ABI-v3 image must remain 80 bytes");
static_assert(sizeof(xtbloom_compute_options_t) == XTBLOOM_COMPUTE_OPTIONS_V3_SIZE,
              "xtbloom_compute_options_t must not add trailing ABI padding");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(offsetof(xtbloom_compute_options_t, scc_start_mode) == 48u,
               "xtbloom_compute_options_t ABI-v2 suffix must start at byte 48");
_Static_assert(XTBLOOM_COMPUTE_OPTIONS_V1_SIZE == 48u,
               "xtbloom_compute_options_t ABI-v1 prefix must remain 48 bytes");
_Static_assert(XTBLOOM_COMPUTE_OPTIONS_V2_SIZE == 56u,
               "xtbloom_compute_options_t ABI-v2 image must remain 56 bytes");
_Static_assert(offsetof(xtbloom_compute_options_t, scc_mixer) == 56u,
               "xtbloom_compute_options_t ABI-v3 mixer must start at byte 56");
_Static_assert(offsetof(xtbloom_compute_options_t, scc_mixer_history) == 60u,
               "xtbloom_compute_options_t ABI-v3 history must start at byte 60");
_Static_assert(offsetof(xtbloom_compute_options_t, scc_mixer_damping) == 64u,
               "xtbloom_compute_options_t ABI-v3 damping must start at byte 64");
_Static_assert(offsetof(xtbloom_compute_options_t, determinism) == 72u,
               "xtbloom_compute_options_t ABI-v3 determinism must start at byte 72");
_Static_assert(offsetof(xtbloom_compute_options_t, reserved_v3) == 76u,
               "xtbloom_compute_options_t ABI-v3 reserved field must start at byte 76");
_Static_assert(XTBLOOM_COMPUTE_OPTIONS_V3_SIZE == 80u,
               "xtbloom_compute_options_t ABI-v3 image must remain 80 bytes");
_Static_assert(sizeof(xtbloom_compute_options_t) == XTBLOOM_COMPUTE_OPTIONS_V3_SIZE,
               "xtbloom_compute_options_t must not add trailing ABI padding");
#endif

#if defined(XTBLOOM_DETAIL_ABI_ASSERT) && defined(XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE)
XTBLOOM_DETAIL_ABI_ASSERT(XTBLOOM_BATCH_V1_SIZE == XTBLOOM_DETAIL_EXPECTED_BATCH_V1_SIZE,
                          "xtbloom_batch_t ABI-v1 prefix must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(XTBLOOM_BATCH_V2_SIZE == XTBLOOM_DETAIL_EXPECTED_BATCH_V2_SIZE,
                          "xtbloom_batch_t ABI-v2 image must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_batch_t, total_interactions) ==
                              XTBLOOM_DETAIL_EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET,
                          "xtbloom_batch_t ABI-v3 suffix must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_batch_t, interaction_descriptors) ==
                              XTBLOOM_DETAIL_EXPECTED_BATCH_INTERACTION_DESCRIPTORS_OFFSET,
                          "xtbloom_batch_t descriptors must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_batch_t, interaction_payload) ==
                              XTBLOOM_DETAIL_EXPECTED_BATCH_INTERACTION_PAYLOAD_OFFSET,
                          "xtbloom_batch_t payload must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(XTBLOOM_BATCH_V3_SIZE == XTBLOOM_DETAIL_EXPECTED_BATCH_V3_SIZE,
                          "xtbloom_batch_t ABI-v3 image must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_batch_t, cell_matrices) ==
                              XTBLOOM_DETAIL_EXPECTED_BATCH_CELL_MATRICES_OFFSET,
                          "xtbloom_batch_t cell matrices must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_batch_t, periodic_axes) ==
                              XTBLOOM_DETAIL_EXPECTED_BATCH_PERIODIC_AXES_OFFSET,
                          "xtbloom_batch_t periodic axes must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(XTBLOOM_BATCH_V4_SIZE == XTBLOOM_DETAIL_EXPECTED_BATCH_V4_SIZE,
                          "xtbloom_batch_t ABI-v4 image must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(sizeof(xtbloom_batch_t) == XTBLOOM_BATCH_V4_SIZE,
                          "xtbloom_batch_t must not add trailing ABI padding");
XTBLOOM_DETAIL_ABI_ASSERT(XTBLOOM_INTERACTION_V1_SIZE == 32u,
                          "xtbloom_interaction_t image must remain 32 bytes");
XTBLOOM_DETAIL_ABI_ASSERT(sizeof(xtbloom_interaction_t) == XTBLOOM_INTERACTION_V1_SIZE,
                          "xtbloom_interaction_t must not add trailing ABI padding");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_interaction_t, flags) == 4u,
                          "xtbloom_interaction_t flags must start at byte 4");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_interaction_t, system_index) == 8u,
                          "xtbloom_interaction_t system index must start at byte 8");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_interaction_t, payload_offset) == 16u,
                          "xtbloom_interaction_t payload offset must start at byte 16");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_interaction_t, payload_size) == 24u,
                          "xtbloom_interaction_t payload size must start at byte 24");
#endif

/*
 * Caller-allocated result buffers. At finite electronic temperature, energies
 * are the total electronic Helmholtz free energy E_internal - T*S_electronic
 * in Hartree, matching the variational xTB/tblite quantity. Forces are its
 * negative coordinate derivative in Hartree/bohr. At zero temperature this
 * quantity equals the internal energy.
 *
 * A NULL data pointer is valid for an output not requested by the compute
 * flags. The three SCC diagnostic buffers are always required for a nonempty
 * batch, independent of the requested property flags. scc_iterations and
 * per_system_status store batch_size int32_t values, while scc_converged stores
 * batch_size uint8_t values.
 *
 * A XTBLOOM_STATUS_SUCCESS return means every diagnostic entry was committed,
 * not necessarily that every system converged. per_system_status is SUCCESS,
 * SCC_NOT_CONVERGED, or EIGENSOLVER_FAILED; scc_converged is exactly one only
 * for SUCCESS. A failed system's requested floating-point property slices are
 * filled with quiet NaNs and never contain a partially evaluated result.
 */
typedef struct xtbloom_batch_result {
  uint32_t struct_size;
  uint32_t api_version;
  uint32_t flags;
  uint32_t reserved;
  xtbloom_buffer_t energies;
  xtbloom_buffer_t forces;
  xtbloom_buffer_t atomic_charges;
  xtbloom_buffer_t point_charge_forces;
  xtbloom_buffer_t scc_iterations;
  xtbloom_buffer_t scc_converged;
  xtbloom_buffer_t per_system_status;
  /* ABI v2 optional suffix; absent suffix preserves ABI-v1 behavior.
   *
   * dipole_moments holds batch_size * 3 doubles (atomic units) and is filled
   * when XTBLOOM_COMPUTE_DIPOLE_MOMENTS is requested. The remaining outputs
   * are ABI-reserved: their shape contract is unpublished, their buffers must
   * be NULL until the matching output is released, and requesting them is
   * rejected before execution. */
  xtbloom_buffer_t dipole_moments;
  xtbloom_buffer_t quadrupole_moments;
  xtbloom_buffer_t wiberg_orders;
  xtbloom_buffer_t spin_populations;
} xtbloom_batch_result_t;

#define XTBLOOM_BATCH_RESULT_V1_SIZE \
  (offsetof(xtbloom_batch_result_t, per_system_status) + sizeof(xtbloom_buffer_t))
#define XTBLOOM_BATCH_RESULT_V2_SIZE \
  (offsetof(xtbloom_batch_result_t, spin_populations) + sizeof(xtbloom_buffer_t))

#if defined(XTBLOOM_DETAIL_ABI_ASSERT) && defined(XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE)
XTBLOOM_DETAIL_ABI_ASSERT(
    XTBLOOM_BATCH_RESULT_V1_SIZE == XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_V1_SIZE,
    "xtbloom_batch_result_t ABI-v1 prefix must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(
    offsetof(xtbloom_batch_result_t, dipole_moments) ==
        XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_DIPOLE_OFFSET,
    "xtbloom_batch_result_t ABI-v2 suffix must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(
    offsetof(xtbloom_batch_result_t, quadrupole_moments) ==
        XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_QUADRUPOLE_OFFSET,
    "xtbloom_batch_result_t quadrupole outlet must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(
    offsetof(xtbloom_batch_result_t, wiberg_orders) ==
        XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_WIBERG_OFFSET,
    "xtbloom_batch_result_t Wiberg outlet must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_batch_result_t, spin_populations) ==
                              XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_SPIN_OFFSET,
                          "xtbloom_batch_result_t spin outlet must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(
    XTBLOOM_BATCH_RESULT_V2_SIZE == XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_V2_SIZE,
    "xtbloom_batch_result_t ABI-v2 image must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(sizeof(xtbloom_batch_result_t) == XTBLOOM_BATCH_RESULT_V2_SIZE,
                          "xtbloom_batch_result_t must not add trailing ABI padding");
#endif

/*
 * Caller-friendly workspace sizing for one fixed-topology plan.
 *
 * compute_flags is an input carrying the properties the caller plans to
 * request on the fixed topology and policy. It must equal the flags supplied
 * to xtbloom_plan_create. host_required_bytes / host_required_alignment
 * and device_required_bytes / device_required_alignment are outputs describing
 * the reusable plan-owned workspace xtbloom_plan_compute reserves in host and
 * device memory. A backend that uses no device workspace reports zero device
 * bytes with alignment one. Sizes can differ between backends and between
 * property sets; the returned values cover one steady-state plan compute.
 *
 * This is an accounting query for the plan's retained scratch. CPU values are
 * derived from the immutable topology and policy, while CUDA values are
 * measured from the complete prepared runtime workspace, including numerical,
 * result, validation, setup-owner, model-plan, and mixed-memory topology
 * staging storage. Opaque CUDA provider and Graph bookkeeping is not caller
 * workspace and is intentionally outside these byte counts.
 */
typedef struct xtbloom_workspace_query {
  uint32_t struct_size;
  uint32_t api_version;
  uint32_t compute_flags;
  uint32_t reserved;
  uint64_t host_required_bytes;
  uint32_t host_required_alignment;
  uint64_t device_required_bytes;
  uint32_t device_required_alignment;
  uint32_t reserved_v2;
} xtbloom_workspace_query_t;

#define XTBLOOM_WORKSPACE_QUERY_V1_SIZE \
  (offsetof(xtbloom_workspace_query_t, reserved_v2) + sizeof(uint32_t))

#if defined(__cplusplus)
static_assert(offsetof(xtbloom_workspace_query_t, host_required_bytes) == 16u,
              "xtbloom_workspace_query_t host byte count must start at byte 16");
static_assert(offsetof(xtbloom_workspace_query_t, device_required_bytes) == 32u,
              "xtbloom_workspace_query_t device byte count must start at byte 32");
static_assert(XTBLOOM_WORKSPACE_QUERY_V1_SIZE == 48u,
              "xtbloom_workspace_query_t ABI-v1 image must remain 48 bytes");
static_assert(sizeof(xtbloom_workspace_query_t) == XTBLOOM_WORKSPACE_QUERY_V1_SIZE,
              "xtbloom_workspace_query_t must not add trailing ABI padding");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(offsetof(xtbloom_workspace_query_t, host_required_bytes) == 16u,
               "xtbloom_workspace_query_t host byte count must start at byte 16");
_Static_assert(offsetof(xtbloom_workspace_query_t, device_required_bytes) == 32u,
               "xtbloom_workspace_query_t device byte count must start at byte 32");
_Static_assert(XTBLOOM_WORKSPACE_QUERY_V1_SIZE == 48u,
               "xtbloom_workspace_query_t ABI-v1 image must remain 48 bytes");
_Static_assert(sizeof(xtbloom_workspace_query_t) == XTBLOOM_WORKSPACE_QUERY_V1_SIZE,
               "xtbloom_workspace_query_t must not add trailing ABI padding");
#endif

/*
 * Snapshot of one reusable asynchronous request.
 *
 * state is IDLE immediately after request creation, PENDING after a submission
 * has been accepted, and COMPLETE after all native work and caller-output
 * publication have finished. completion_status is meaningful in COMPLETE and
 * reports the submitted computation's final status; query/wait themselves
 * return a separate status describing whether the snapshot operation worked.
 * result_flags is the asynchronous counterpart of xtbloom_batch_result_t.flags:
 * enqueue functions take a const result descriptor and never modify that
 * descriptor object.
 */
typedef struct xtbloom_request_info {
  uint32_t struct_size;
  uint32_t api_version;
  xtbloom_request_state_t state;
  xtbloom_status_t completion_status;
  uint32_t result_flags;
  uint32_t reserved;
} xtbloom_request_info_t;

#define XTBLOOM_REQUEST_INFO_V1_SIZE (offsetof(xtbloom_request_info_t, reserved) + sizeof(uint32_t))

#if defined(__cplusplus)
static_assert(offsetof(xtbloom_request_info_t, state) == 8u,
              "xtbloom_request_info_t state must start at byte 8");
static_assert(offsetof(xtbloom_request_info_t, completion_status) == 12u,
              "xtbloom_request_info_t completion status must start at byte 12");
static_assert(offsetof(xtbloom_request_info_t, result_flags) == 16u,
              "xtbloom_request_info_t result flags must start at byte 16");
static_assert(XTBLOOM_REQUEST_INFO_V1_SIZE == 24u,
              "xtbloom_request_info_t ABI-v1 image must remain 24 bytes");
static_assert(sizeof(xtbloom_request_info_t) == XTBLOOM_REQUEST_INFO_V1_SIZE,
              "xtbloom_request_info_t must not add trailing ABI padding");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(offsetof(xtbloom_request_info_t, state) == 8u,
               "xtbloom_request_info_t state must start at byte 8");
_Static_assert(offsetof(xtbloom_request_info_t, completion_status) == 12u,
               "xtbloom_request_info_t completion status must start at byte 12");
_Static_assert(offsetof(xtbloom_request_info_t, result_flags) == 16u,
               "xtbloom_request_info_t result flags must start at byte 16");
_Static_assert(XTBLOOM_REQUEST_INFO_V1_SIZE == 24u,
               "xtbloom_request_info_t ABI-v1 image must remain 24 bytes");
_Static_assert(sizeof(xtbloom_request_info_t) == XTBLOOM_REQUEST_INFO_V1_SIZE,
               "xtbloom_request_info_t must not add trailing ABI padding");
#endif

XTBLOOM_API const char* xtbloom_version_string(void);
XTBLOOM_API const char* xtbloom_status_string(xtbloom_status_t status);

/* Returns a thread-local diagnostic for the most recent failing API call. */
XTBLOOM_API const char* xtbloom_get_last_error(void);

XTBLOOM_API xtbloom_status_t xtbloom_context_options_init(xtbloom_context_options_t* options,
                                                          size_t struct_size);
XTBLOOM_API xtbloom_status_t xtbloom_batch_init(xtbloom_batch_t* batch, size_t struct_size);
XTBLOOM_API xtbloom_status_t xtbloom_compute_options_init(xtbloom_compute_options_t* options,
                                                          size_t struct_size);
XTBLOOM_API xtbloom_status_t xtbloom_batch_result_init(xtbloom_batch_result_t* result,
                                                       size_t struct_size);
XTBLOOM_API xtbloom_status_t xtbloom_workspace_query_init(xtbloom_workspace_query_t* query,
                                                          size_t struct_size);
XTBLOOM_API xtbloom_status_t xtbloom_request_info_init(xtbloom_request_info_t* info,
                                                       size_t struct_size);

XTBLOOM_API xtbloom_status_t xtbloom_context_create(const xtbloom_context_options_t* options,
                                                    xtbloom_context_t** context);
XTBLOOM_API void xtbloom_context_destroy(xtbloom_context_t* context);
XTBLOOM_API xtbloom_backend_t xtbloom_context_get_backend(const xtbloom_context_t* context);
XTBLOOM_API int32_t xtbloom_context_get_device_id(const xtbloom_context_t* context);

/*
 * Create a backend-neutral reusable request bound to context. Creating a
 * request is supported for CPU and CUDA contexts, but asynchronous enqueue is
 * currently CUDA-only. The request must be destroyed before its context.
 */
XTBLOOM_API xtbloom_status_t xtbloom_request_create(xtbloom_context_t* context,
                                                    xtbloom_request_t** request);

/*
 * Query without blocking, or wait until the current submission finishes.
 * Query takes a mutable request because observing a ready native event may
 * finalize deferred host-output publication and transition it to COMPLETE.
 * Both calls write info only after validating its complete ABI-v1 header and
 * reserved field. An IDLE request returns immediately. The API return value
 * describes the query/wait operation; inspect info.completion_status only
 * when info.state is COMPLETE for the submitted compute status.
 */
XTBLOOM_API xtbloom_status_t xtbloom_request_query(xtbloom_request_t* request,
                                                   xtbloom_request_info_t* info);
XTBLOOM_API xtbloom_status_t xtbloom_request_wait(xtbloom_request_t* request,
                                                  xtbloom_request_info_t* info);

/*
 * Return the request-owned diagnostic for its completed submission. The
 * returned pointer remains valid until the next accepted enqueue or request
 * destruction. It is empty for IDLE, PENDING, and successful completion;
 * invalid handles return NULL and set xtbloom_get_last_error().
 * Do not concurrently reuse, query, wait on, or destroy the same request while
 * retaining this pointer.
 */
XTBLOOM_API const char* xtbloom_request_get_error(const xtbloom_request_t* request);

/* A NULL request is a harmless no-op. Destroying PENDING waits for completion. */
XTBLOOM_API void xtbloom_request_destroy(xtbloom_request_t* request);

/*
 * Performs a synchronous batched inference. Host buffers are accepted by both
 * CPU and CUDA backends; CUDA device buffers avoid staging copies on CUDA.
 *
 * The complete request is validated before execution. Any failure detected
 * before the final caller-output commit begins leaves result flags and all
 * result buffers unchanged. Once a CUDA caller-output commit has begun, a
 * later catastrophic failure returns XTBLOOM_STATUS_INTERNAL_ERROR and results
 * may already have been modified. CUDA attempts to restore the caller's current
 * device on every exit; restoration failure also returns INTERNAL_ERROR and
 * may leave that device selection changed, independently of whether output
 * commit began. xtbloom_get_last_error identifies the failed boundary.
 * Per-system SCC or eigensolver failures are data-level results: the function
 * returns SUCCESS and records them in per_system_status so one bad batch item
 * does not discard successful peers.
 */
XTBLOOM_API xtbloom_status_t xtbloom_compute(xtbloom_context_t* context,
                                             const xtbloom_batch_t* batch,
                                             const xtbloom_compute_options_t* options,
                                             xtbloom_batch_result_t* result);

/*
 * Submit one CUDA computation on the context stream without waiting for
 * inference or result publication.
 * The request must be IDLE or COMPLETE and bound to context. An accepted
 * submission resets its previous completion status/error and becomes PENDING,
 * or COMPLETE when the backend must settle it inline. Reusing a PENDING
 * request is rejected. Descriptor structs are copied, and every HOST input is
 * copied or fully consumed before a successful enqueue returns; those host
 * descriptors and bytes may then be reused or released.
 * Every CUDA_DEVICE input and every HOST or CUDA_DEVICE output remains a
 * caller-owned borrowed buffer and must stay valid until COMPLETE.
 *
 * The result descriptor is copied during submission and is never retained or
 * modified. In particular, result->flags is not an asynchronous publication
 * channel; completed flags are returned in xtbloom_request_info_t.result_flags.
 * CPU contexts return XTBLOOM_STATUS_NOT_SUPPORTED before descriptor validation
 * and leave all result bytes and the request state unchanged. CUDA context
 * enqueue prepares or reuses the context-owned topology cache. A new or changed
 * topology may perform bounded setup and topology-validation waits before the
 * request is accepted, but accepted numerical inference and result publication
 * remain stream-asynchronous. Once a device-resident topology has established a
 * prepared shape/policy, later same-shape device submissions compare its exact
 * immutable bytes in stream order; a mismatch completes the request with
 * INVALID_ARGUMENT rather than rebuilding it. Use a changed shape/policy or a
 * synchronous convenience call to establish a new device topology. Use
 * xtbloom_plan_compute_enqueue when topology is fixed and allocation-free
 * admission is required. ABI-v2 strict WARM consumes the latest compatible
 * fully converged checkpoint on the same context cache and never falls back to
 * FRESH. Missing or host-visible incompatible state is rejected before
 * admission; a stream-ordered device-topology mismatch completes the accepted
 * request with INVALID_ARGUMENT and invalidates the consumed checkpoint.
 */
XTBLOOM_API xtbloom_status_t xtbloom_compute_enqueue(xtbloom_context_t* context,
                                                     const xtbloom_batch_t* batch,
                                                     const xtbloom_compute_options_t* options,
                                                     const xtbloom_batch_result_t* result,
                                                     xtbloom_request_t* request);

/*
 * Create a fixed-topology plan from one already-validated-shaped batch
 * descriptor and a compute policy. The plan binds the immutable topology (atom
 * offsets, element numbers, spin channels, point-charge and response structure)
 * and the numerical policy (model, requested properties, SCC tolerances,
 * iteration limit, electronic temperature, mixer algorithm/history/damping,
 * and determinism) to the context backend and reserves its reusable
 * host/device workspace. Geometry (positions and
 * point-charge positions/values) is intentionally not part of the plan and
 * may change per xtbloom_plan_compute call.
 *
 * The plan is a setup-with-allocation-permitted path: creating it performs
 * validation and workspace reservation that xtbloom_compute would otherwise
 * repeat on every call. Calling xtbloom_plan_compute repeatedly for the same
 * fixed topology must not allocate steady-state workspace on either backend.
 *
 * A plan is bound to the creating context; it must be destroyed with
 * xtbloom_plan_destroy before the context. Passing a plan whose topology does
 * not match the batch on xtbloom_plan_compute, or a plan created for a
 * different context, fails before any caller output is modified.
 */
XTBLOOM_API xtbloom_status_t xtbloom_plan_create(xtbloom_context_t* context,
                                                 const xtbloom_batch_t* batch,
                                                 const xtbloom_compute_options_t* options,
                                                 xtbloom_plan_t** plan);
XTBLOOM_API void xtbloom_plan_destroy(xtbloom_plan_t* plan);

/*
 * Query the reusable plan-owned workspace xtbloom_plan_compute reserves on the
 * plan's backend for its requested properties. On return query.compute_flags
 * is preserved and must match the plan policy; the four sizing fields are populated as documented
 * on xtbloom_workspace_query_t. Callers that want device sizing must use a CUDA plan; CPU plans
 * always report zero device bytes.
 */
XTBLOOM_API xtbloom_status_t xtbloom_plan_query_workspace(const xtbloom_plan_t* plan,
                                                          xtbloom_workspace_query_t* query);

/*
 * Execute one synchronous batched inference on a fixed-topology plan.
 *
 * geometry-only descriptors: positions and point-charge positions/values may
 * change between calls, but the immutable topology and creation-time compute
 * policy must match the plan exactly or the call fails with
 * XTBLOOM_STATUS_INVALID_ARGUMENT before any caller output is modified.
 * Otherwise semantics match xtbloom_compute: complete
 * validation before execution, per-system SCC/eigensolver failures recorded in
 * per_system_status, and failed systems' floating-point slices filled with
 * quiet NaNs.
 */
XTBLOOM_API xtbloom_status_t xtbloom_plan_compute(xtbloom_plan_t* plan,
                                                  const xtbloom_batch_t* batch,
                                                  const xtbloom_compute_options_t* options,
                                                  xtbloom_batch_result_t* result);

/* Fixed-topology counterpart of xtbloom_compute_enqueue with identical request,
 * result-descriptor, buffer-lifetime, and CPU NOT_SUPPORTED semantics. Host
 * topology is compared before return; CUDA-device topology is compared in
 * stream order, and a mismatch completes with INVALID_ARGUMENT without
 * modifying caller outputs. Accepted inference and publication remain
 * stream-asynchronous. ABI-v2 strict WARM consumes the latest compatible fully
 * converged checkpoint and never falls back to FRESH. Once enqueue is accepted,
 * either FRESH or WARM consumes any preceding plan checkpoint even if
 * completion later reports a deferred topology or execution failure. The
 * request retains the plan's execution cache and the plan handle may be
 * destroyed before completion (the creating context must still outlive the
 * request). */
XTBLOOM_API xtbloom_status_t xtbloom_plan_compute_enqueue(xtbloom_plan_t* plan,
                                                          const xtbloom_batch_t* batch,
                                                          const xtbloom_compute_options_t* options,
                                                          const xtbloom_batch_result_t* result,
                                                          xtbloom_request_t* request);

/*
 * xtbloom-owned result arenas and their DLPack export.
 *
 * A xtbloom_result_owner_t is a ref-counted allocation (host memory or CUDA
 * device memory) that xtbloom itself allocates, fills through a normal compute
 * call, and can hand to an importing framework through the DLPack producer
 * protocol without copying data.
 *
 * Lifetime model:
 * - xtbloom_result_owner_create produces one arena with an initial reference.
 *   xtbloom_result_owner_buffer exposes that arena as a caller-owned
 *   xtbloom_buffer_t view so the caller can bind output slices and run compute.
 * - xtbloom_result_owner_retain / xtbloom_result_owner_release manage the
 *   reference count. The arena allocation is freed exactly once, when the
 *   last reference is released. release(NULL) is a no-op; otherwise every
 *   release must correspond to exactly one prior create or retain.
 * - xtbloom_result_owner_export_dltensor retains the arena for one exported
 *   managed tensor. When the importing framework releases that tensor it
 *   invokes a native deleter that frees the managed tensor (and its shape
 *   storage) and releases the arena reference. The owner therefore remains
 *   valid as long as any exported tensor or explicit reference is alive,
 *   independent of the compute context that filled the arena.
 *
 * The producer object itself (the Python side) wraps the returned managed
 * tensor in a PyCapsule named "dltensor" (legacy) or "dltensor_versioned"
 * (DLPack 1.0) with "used_"-renaming applied by the consumer as usual. The
 * native deleter is immune to the importing framework outliving the Python
 * wrapper: it never calls back into Python.
 *
 * xtbloom's public CUDA compute is synchronous: when xtbloom_compute returns,
 * the requested result bytes are fully committed on the context stream and a
 * producer export needs no additional device-wide synchronization or hidden
 * host polling.
 */

/* Options for allocating one result arena. */
typedef struct xtbloom_result_owner_options {
  uint32_t struct_size;
  uint32_t api_version;
  /* XTBLOOM_MEMORY_HOST or XTBLOOM_MEMORY_CUDA_DEVICE. */
  xtbloom_memory_space_t memory_space;
  /* CUDA device ordinal for CUDA arenas; -1 for host arenas. */
  int32_t device_id;
  /* Byte extent of the arena. Must be nonzero. */
  uint64_t size_bytes;
  uint32_t reserved;
} xtbloom_result_owner_options_t;

#define XTBLOOM_RESULT_OWNER_OPTIONS_V1_SIZE \
  (offsetof(xtbloom_result_owner_options_t, reserved) + sizeof(uint32_t))

/*
 * Describes one compact C-contiguous slice of a result arena to export as a
 * DLPack managed tensor. dtype mirrors DLPack's DLDataType triple
 * (code/bits/lanes); shape holds ndim int64 extents and is caller-owned only
 * for the duration of the export call (xtbloom copies it into the managed
 * tensor's storage). strides are always implicit compact row-major (NULL by
 * DLPack convention). Accepted dtypes: int8/int16/int32/int64, uint8,
 * float32/float64 with lanes == 1. No DLPack headers are required from the
 * caller: this structure is the constructor input, and the managed-tensor
 * struct layout mirrors the pinned DLPack 1.0 specification (see
 * src/runtime/dlpack_layout.hpp for the byte-exact mirrors and provenance).
 */
typedef struct xtbloom_dlpack_view {
  uint32_t struct_size;
  uint32_t api_version;
  /* Byte offset of the slice inside the arena. Must keep dtype alignment. */
  uint64_t byte_offset;
  int32_t dtype_code; /* DLDataTypeCode: 0 int, 1 uint, 2 float, 4 bfloat, 6 bool. */
  int32_t dtype_bits;
  int32_t dtype_lanes;
  int32_t ndim; /* 0..8 */
  uint32_t reserved;
  const int64_t* shape; /* ndim int64 values; copied by xtbloom */
} xtbloom_dlpack_view_t;

#define XTBLOOM_DLPACK_MAX_NDIM 8
#define XTBLOOM_DLPACK_VIEW_V1_SIZE \
  (offsetof(xtbloom_dlpack_view_t, shape) + sizeof(const int64_t*))

#if defined(XTBLOOM_DETAIL_ABI_ASSERT) && defined(XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE)
XTBLOOM_DETAIL_ABI_ASSERT(offsetof(xtbloom_dlpack_view_t, shape) ==
                              XTBLOOM_DETAIL_EXPECTED_DLPACK_SHAPE_OFFSET,
                          "xtbloom_dlpack_view_t shape must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(XTBLOOM_DLPACK_VIEW_V1_SIZE ==
                              XTBLOOM_DETAIL_EXPECTED_DLPACK_VIEW_V1_SIZE,
                          "xtbloom_dlpack_view_t image must match the target pointer width");
XTBLOOM_DETAIL_ABI_ASSERT(sizeof(xtbloom_dlpack_view_t) == XTBLOOM_DLPACK_VIEW_V1_SIZE,
                          "xtbloom_dlpack_view_t must not add trailing ABI padding");
#endif

#undef XTBLOOM_DETAIL_ABI_ASSERT
#undef XTBLOOM_DETAIL_EXPECTED_CONTEXT_OPTIONS_V1_SIZE
#undef XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE
#undef XTBLOOM_DETAIL_EXPECTED_BUFFER_SIZE_BYTES_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BUFFER_MEMORY_SPACE_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BUFFER_RESERVED_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_V1_SIZE
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_V2_SIZE
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_INTERACTION_DESCRIPTORS_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_INTERACTION_PAYLOAD_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_V3_SIZE
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_CELL_MATRICES_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_PERIODIC_AXES_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_V4_SIZE
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_V1_SIZE
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_DIPOLE_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_QUADRUPOLE_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_WIBERG_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_SPIN_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_BATCH_RESULT_V2_SIZE
#undef XTBLOOM_DETAIL_EXPECTED_DLPACK_SHAPE_OFFSET
#undef XTBLOOM_DETAIL_EXPECTED_DLPACK_VIEW_V1_SIZE

XTBLOOM_API xtbloom_status_t
xtbloom_result_owner_options_init(xtbloom_result_owner_options_t* options, size_t struct_size);
XTBLOOM_API xtbloom_status_t xtbloom_result_owner_create(
    const xtbloom_result_owner_options_t* options, xtbloom_result_owner_t** owner);
/* Copies the whole arena into buffer as a caller-owned borrowed view. */
XTBLOOM_API xtbloom_status_t xtbloom_result_owner_buffer(const xtbloom_result_owner_t* owner,
                                                         xtbloom_buffer_t* buffer);
XTBLOOM_API void xtbloom_result_owner_retain(xtbloom_result_owner_t* owner);
XTBLOOM_API void xtbloom_result_owner_release(xtbloom_result_owner_t* owner);
/*
 * Export one arena slice as a heap-allocated DLManagedTensorVersioned (when
 * version != 0) or legacy DLManagedTensor (when version == 0). On success
 * *out_managed receives the pointer and xtbloom owns its lifetime: the import
 * consumer (or capsule destructor) must call the stored deleter exactly once.
 * On failure *out_managed is set to NULL, no arena reference is taken, and
 * the arena reference counting is untouched.
 */
XTBLOOM_API xtbloom_status_t xtbloom_result_owner_export_dltensor(
    const xtbloom_result_owner_t* owner, const xtbloom_dlpack_view_t* view, int version,
    void** out_managed);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XTBLOOM_XTBLOOM_H */
