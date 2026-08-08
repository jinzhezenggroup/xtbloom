#ifndef GPUXTB_GPUXTB_H
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define GPUXTB_GPUXTB_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(GPUXTB_BUILDING_LIBRARY)
#define GPUXTB_API __declspec(dllexport)
#else
#define GPUXTB_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define GPUXTB_API __attribute__((visibility("default")))
#else
#define GPUXTB_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GPUXTB_VERSION_MAJOR 0
#define GPUXTB_VERSION_MINOR 1
#define GPUXTB_VERSION_PATCH 0

/* Increment this value only when an ABI-incompatible C API change is made. */
#define GPUXTB_API_VERSION 1u

/*
 * Electronic temperatures are k_B*T energy scales in Hartree, not kelvin.
 * This conversion matches the pinned xTB/tblite convention used by gpuxtb.
 */
#define GPUXTB_KELVIN_TO_HARTREE 3.166808578545117e-6
#define GPUXTB_DEFAULT_ELECTRONIC_TEMPERATURE (300.0 * GPUXTB_KELVIN_TO_HARTREE)

typedef struct gpuxtb_context gpuxtb_context_t;

/* Opaque fixed-topology plan handle. See gpuxtb_plan_create. */
typedef struct gpuxtb_plan gpuxtb_plan_t;

/*
 * Opaque owner of one gpuxtb-allocated result arena. See
 * gpuxtb_result_owner_create. A result owner is a ref-counted host or CUDA
 * device allocation that outlives the compute context used to fill it, so a
 * DLPack producer can hand the finished bytes to an importing framework
 * without a host round trip and without keeping the context alive.
 */
typedef struct gpuxtb_result_owner gpuxtb_result_owner_t;

/*
 * ABI tags are explicitly int32_t rather than enum-typed fields. This keeps
 * their object representation and function calling convention identical in C
 * and C++, including C99 builds compiled with options such as -fshort-enums.
 * The named enums below only provide debugger-friendly symbolic constants;
 * callers may pass any int32_t bit pattern and the library validates it.
 */
typedef int32_t gpuxtb_status_t;
enum gpuxtb_status_value {
  GPUXTB_STATUS_SUCCESS = 0,
  GPUXTB_STATUS_INVALID_ARGUMENT = 1,
  GPUXTB_STATUS_BACKEND_UNAVAILABLE = 2,
  GPUXTB_STATUS_NOT_SUPPORTED = 3,
  GPUXTB_STATUS_ALLOCATION_FAILED = 4,
  GPUXTB_STATUS_NOT_IMPLEMENTED = 5,
  GPUXTB_STATUS_INTERNAL_ERROR = 6,
  /* Per-system SCC reached max_scc_iterations without satisfying both tolerances. */
  GPUXTB_STATUS_SCC_NOT_CONVERGED = 7,
  /* Per-system generalized eigensolver failed or produced an unusable eigensystem. */
  GPUXTB_STATUS_EIGENSOLVER_FAILED = 8
};

typedef int32_t gpuxtb_backend_t;
enum gpuxtb_backend_value {
  /* Prefer CUDA when it is compiled in and a compatible device is present. */
  GPUXTB_BACKEND_AUTO = 0,
  GPUXTB_BACKEND_CPU = 1,
  GPUXTB_BACKEND_CUDA = 2,
  /* Reserved now so adding HIP kernels does not require redesigning the ABI. */
  GPUXTB_BACKEND_ROCM = 3
};

typedef int32_t gpuxtb_memory_space_t;
enum gpuxtb_memory_space_value {
  GPUXTB_MEMORY_HOST = 0,
  GPUXTB_MEMORY_CUDA_DEVICE = 1,
  GPUXTB_MEMORY_ROCM_DEVICE = 2
};

typedef int32_t gpuxtb_model_t;
enum gpuxtb_model_value { GPUXTB_MODEL_GFN1_XTB = 1, GPUXTB_MODEL_GFN2_XTB = 2 };

typedef int32_t gpuxtb_scc_start_mode_t;
enum gpuxtb_scc_start_mode_value {
  /* Restore the immutable initial electronic state before SCC. */
  GPUXTB_SCC_START_FRESH = 1,
  /* Strictly consume a checkpoint from the latest fully converged compatible batch call. */
  GPUXTB_SCC_START_WARM = 2
};

typedef int32_t gpuxtb_compute_flag_t;
enum gpuxtb_compute_flag_value {
  GPUXTB_COMPUTE_ENERGY = 1 << 0,
  GPUXTB_COMPUTE_FORCES = 1 << 1,
  GPUXTB_COMPUTE_ATOMIC_CHARGES = 1 << 2,
  GPUXTB_COMPUTE_POINT_CHARGE_FORCES = 1 << 3,
  /* Reports per-system dipole moments through batch_result.dipole_moments. */
  GPUXTB_COMPUTE_DIPOLE_MOMENTS = 1 << 4
  /* Bits 16-31 are reserved for future outputs and must be zero on input. */
};

typedef int32_t gpuxtb_result_flag_t;
enum gpuxtb_result_flag_value {
  /*
   * Set when atomic_potential_shifts or charge_response_matrix was supplied.
   * Forces then exclude coordinate derivatives of those caller-owned fields.
   */
  GPUXTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES = 1 << 0
  /* Bits 16-31 are reserved for future annotations and must be zero on input. */
};

/*
 * Tag set for the generic interaction attachment slot (ABI-v3 batch suffix).
 * Tag values are intentionally spread over family ranges so future additions
 * never renumber an existing value. The GFN2 backends currently implement
 * none of these interactions: validating any present interaction returns
 * GPUXTB_STATUS_NOT_IMPLEMENTED until the matching backend term lands, and
 * the enum values are reserved so later features do not churn the batch
 * layout. GPUXTB_INTERACTION_NONE is not a valid attachment.
 */
typedef int32_t gpuxtb_interaction_type_t;
enum gpuxtb_interaction_type_value {
  GPUXTB_INTERACTION_NONE = 0,
  /* External potentials (0x01xx). */
  GPUXTB_INTERACTION_ELECTRIC_FIELD = 0x0101,
  GPUXTB_INTERACTION_ELECTRIC_FIELD_GRADIENT = 0x0102,
  GPUXTB_INTERACTION_POINT_CHARGES_MULTIPOLE = 0x0103,
  GPUXTB_INTERACTION_ATOMIC_POTENTIAL_GRID = 0x0104,
  /* Self-consistent solvation models (0x02xx). */
  GPUXTB_INTERACTION_ALPB_SOLVATION = 0x0201,
  GPUXTB_INTERACTION_GBSA_SOLVATION = 0x0202,
  GPUXTB_INTERACTION_GB_SOLVATION = 0x0203,
  GPUXTB_INTERACTION_GBE_SOLVATION = 0x0204,
  GPUXTB_INTERACTION_DDX_SOLVATION = 0x0205,
  /* Dispersion models (0x03xx). */
  GPUXTB_INTERACTION_D3_DISPERSION = 0x0301,
  GPUXTB_INTERACTION_D4_VARIANT_DISPERSION = 0x0302,
  /* Structure-correction models (0x04xx). */
  GPUXTB_INTERACTION_HALOGEN_BOND = 0x0401
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
 * 32 bytes: int32_t version, int32_t reserved, three doubles holding the
 * field vector in Hartree per elementary charge per bohr.
 */
typedef struct gpuxtb_interaction {
  gpuxtb_interaction_type_t type;
  uint32_t flags;
  int64_t system_index;
  uint64_t payload_offset;
  uint64_t payload_size;
} gpuxtb_interaction_t;

#define GPUXTB_INTERACTION_V1_SIZE (offsetof(gpuxtb_interaction_t, payload_size) + sizeof(uint64_t))

/*
 * Keep all public ABI tag and flag aliases at their specified width.
 */
#if defined(__cplusplus)
static_assert(sizeof(gpuxtb_status_t) == sizeof(int32_t), "gpuxtb_status_t must be 32-bit");
static_assert(sizeof(gpuxtb_backend_t) == sizeof(int32_t), "gpuxtb_backend_t must be 32-bit");
static_assert(sizeof(gpuxtb_memory_space_t) == sizeof(int32_t),
              "gpuxtb_memory_space_t must be 32-bit");
static_assert(sizeof(gpuxtb_model_t) == sizeof(int32_t), "gpuxtb_model_t must be 32-bit");
static_assert(sizeof(gpuxtb_scc_start_mode_t) == sizeof(int32_t),
              "gpuxtb_scc_start_mode_t must be 32-bit");
static_assert(sizeof(gpuxtb_compute_flag_t) == sizeof(int32_t),
              "gpuxtb_compute_flag_t must be 32-bit");
static_assert(sizeof(gpuxtb_result_flag_t) == sizeof(int32_t),
              "gpuxtb_result_flag_t must be 32-bit");
static_assert(sizeof(gpuxtb_interaction_type_t) == sizeof(int32_t),
              "gpuxtb_interaction_type_t must be 32-bit");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(gpuxtb_status_t) == sizeof(int32_t), "gpuxtb_status_t must be 32-bit");
_Static_assert(sizeof(gpuxtb_backend_t) == sizeof(int32_t), "gpuxtb_backend_t must be 32-bit");
_Static_assert(sizeof(gpuxtb_memory_space_t) == sizeof(int32_t),
               "gpuxtb_memory_space_t must be 32-bit");
_Static_assert(sizeof(gpuxtb_model_t) == sizeof(int32_t), "gpuxtb_model_t must be 32-bit");
_Static_assert(sizeof(gpuxtb_scc_start_mode_t) == sizeof(int32_t),
               "gpuxtb_scc_start_mode_t must be 32-bit");
_Static_assert(sizeof(gpuxtb_compute_flag_t) == sizeof(int32_t),
               "gpuxtb_compute_flag_t must be 32-bit");
_Static_assert(sizeof(gpuxtb_result_flag_t) == sizeof(int32_t),
               "gpuxtb_result_flag_t must be 32-bit");
_Static_assert(sizeof(gpuxtb_interaction_type_t) == sizeof(int32_t),
               "gpuxtb_interaction_type_t must be 32-bit");
#endif

/*
 * Every extensible structure starts with struct_size and api_version. Pass the
 * caller's sizeof(struct) to its initializer before overriding fields. This
 * lets newer and older libraries initialize only the structure prefix known to
 * both sides and reject accidental ABI mismatches.
 */
typedef struct gpuxtb_context_options {
  uint32_t struct_size;
  uint32_t api_version;
  gpuxtb_backend_t backend;
  int32_t device_id;
  /* CPU batch parallelism: zero selects automatic, one disables parallelism. */
  int32_t cpu_threads;
  uint32_t reserved;
  /* Native cudaStream_t or hipStream_t cast to void*. NULL selects the default. */
  void* stream;
} gpuxtb_context_options_t;

#define GPUXTB_CONTEXT_OPTIONS_V1_SIZE (offsetof(gpuxtb_context_options_t, stream) + sizeof(void*))

/* A byte-sized view of caller-owned input memory. gpuxtb never takes ownership. */
typedef struct gpuxtb_const_buffer {
  const void* data;
  size_t size_bytes;
  gpuxtb_memory_space_t memory_space;
  uint32_t reserved;
} gpuxtb_const_buffer_t;

/* A byte-sized view of caller-owned output memory. gpuxtb never takes ownership. */
typedef struct gpuxtb_buffer {
  void* data;
  size_t size_bytes;
  gpuxtb_memory_space_t memory_space;
  uint32_t reserved;
} gpuxtb_buffer_t;

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
 * respect to coordinates are outside gpuxtb and are not included in forces.
 */
typedef struct gpuxtb_batch {
  uint32_t struct_size;
  uint32_t api_version;
  int64_t batch_size;
  int64_t total_atoms;
  int64_t total_point_charges;
  int64_t total_charge_response_elements;
  gpuxtb_const_buffer_t atom_offsets;
  gpuxtb_const_buffer_t atomic_numbers;
  gpuxtb_const_buffer_t positions;
  gpuxtb_const_buffer_t molecular_charges;
  gpuxtb_const_buffer_t unpaired_electrons;
  gpuxtb_const_buffer_t point_charge_offsets;
  gpuxtb_const_buffer_t point_charge_positions;
  gpuxtb_const_buffer_t point_charge_values;
  gpuxtb_const_buffer_t point_charge_gammas;
  gpuxtb_const_buffer_t atomic_potential_shifts;
  gpuxtb_const_buffer_t charge_response_offsets;
  gpuxtb_const_buffer_t charge_response_matrix;
  /* ABI v2 optional suffix; NULL selects one restricted channel per system. */
  gpuxtb_const_buffer_t spin_channels;
  /* ABI v3 optional suffix: generic external-interaction attachments.
   *
   * total_interactions counts gpuxtb_interaction_t entries in
   * interaction_descriptors, each attaching one caller-owned payload block in
   * interaction_payload to one batch item. The suffix may carry any mix of
   * host and CUDA-device storage and may be present with zero interactions,
   * preserving ABI-v1/v2 behavior for callers that never use it. See
   * gpuxtb_interaction_type_t for the reserved tag set and payload contract. */
  int64_t total_interactions;
  gpuxtb_const_buffer_t interaction_descriptors;
  gpuxtb_const_buffer_t interaction_payload;
} gpuxtb_batch_t;

#define GPUXTB_BATCH_V1_SIZE \
  (offsetof(gpuxtb_batch_t, charge_response_matrix) + sizeof(gpuxtb_const_buffer_t))
#define GPUXTB_BATCH_V2_SIZE \
  (offsetof(gpuxtb_batch_t, spin_channels) + sizeof(gpuxtb_const_buffer_t))
#define GPUXTB_BATCH_V3_SIZE \
  (offsetof(gpuxtb_batch_t, interaction_payload) + sizeof(gpuxtb_const_buffer_t))

/*
 * electronic_temperature is k_B*T in Hartree. Bindings that accept kelvin
 * should multiply by GPUXTB_KELVIN_TO_HARTREE before populating this struct.
 *
 * The ABI-v2 scc_start_mode suffix is a strict per-call policy. FRESH restores
 * the immutable initial electronic state. WARM consumes the checkpoint from
 * the latest fully converged compatible public batch call; it never falls back
 * to FRESH. A V1/short prefix means FRESH.
 *
 * A compatible identity is a batch whose topology and compute policy
 * (requested-property flags, molecular charges, unpaired electrons, spin
 * channels, point-charge and periodic structure, SCC tolerances, maximum
 * iterations, and electronic temperature) exactly match the previous fully
 * converged call on the same context. This is the same compute-options
 * identity used by CPU and CUDA. Geometry is not part of the identity: a WARM
 * call reuses the previous converged electronic state as the initial SCC guess
 * for the new coordinates and reconverges. A WARM request with no such
 * compatible fully converged predecessor (first call, changed topology or
 * policy, or a preceding non-converged batch) is rejected with
 * GPUXTB_STATUS_INVALID_ARGUMENT before any caller output is modified.
 */
typedef struct gpuxtb_compute_options {
  uint32_t struct_size;
  uint32_t api_version;
  gpuxtb_model_t model;
  uint32_t flags;
  int32_t max_scc_iterations;
  uint32_t reserved;
  double charge_tolerance;
  double energy_tolerance;
  double electronic_temperature;
  /* ABI v2 optional suffix; absent suffix preserves strict FRESH semantics. */
  gpuxtb_scc_start_mode_t scc_start_mode;
  uint32_t reserved_v2;
} gpuxtb_compute_options_t;

#define GPUXTB_COMPUTE_OPTIONS_V1_SIZE \
  (offsetof(gpuxtb_compute_options_t, electronic_temperature) + sizeof(double))
#define GPUXTB_COMPUTE_OPTIONS_V2_SIZE \
  (offsetof(gpuxtb_compute_options_t, reserved_v2) + sizeof(uint32_t))

#if defined(__cplusplus)
static_assert(offsetof(gpuxtb_compute_options_t, scc_start_mode) == 48u,
              "gpuxtb_compute_options_t ABI-v2 suffix must start at byte 48");
static_assert(GPUXTB_COMPUTE_OPTIONS_V1_SIZE == 48u,
              "gpuxtb_compute_options_t ABI-v1 prefix must remain 48 bytes");
static_assert(GPUXTB_COMPUTE_OPTIONS_V2_SIZE == 56u,
              "gpuxtb_compute_options_t ABI-v2 image must remain 56 bytes");
static_assert(sizeof(gpuxtb_compute_options_t) == GPUXTB_COMPUTE_OPTIONS_V2_SIZE,
              "gpuxtb_compute_options_t must not add trailing ABI padding");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(offsetof(gpuxtb_compute_options_t, scc_start_mode) == 48u,
               "gpuxtb_compute_options_t ABI-v2 suffix must start at byte 48");
_Static_assert(GPUXTB_COMPUTE_OPTIONS_V1_SIZE == 48u,
               "gpuxtb_compute_options_t ABI-v1 prefix must remain 48 bytes");
_Static_assert(GPUXTB_COMPUTE_OPTIONS_V2_SIZE == 56u,
               "gpuxtb_compute_options_t ABI-v2 image must remain 56 bytes");
_Static_assert(sizeof(gpuxtb_compute_options_t) == GPUXTB_COMPUTE_OPTIONS_V2_SIZE,
               "gpuxtb_compute_options_t must not add trailing ABI padding");
#endif

#if defined(__cplusplus)
static_assert(offsetof(gpuxtb_batch_t, total_interactions) == 352u,
              "gpuxtb_batch_t ABI-v3 suffix must start at byte 352");
static_assert(offsetof(gpuxtb_batch_t, interaction_payload) == 384u,
              "gpuxtb_batch_t ABI-v3 payload must start at byte 384");
static_assert(GPUXTB_BATCH_V3_SIZE == 408u, "gpuxtb_batch_t ABI-v3 image must remain 408 bytes");
static_assert(sizeof(gpuxtb_batch_t) == GPUXTB_BATCH_V3_SIZE,
              "gpuxtb_batch_t must not add trailing ABI padding");
static_assert(GPUXTB_INTERACTION_V1_SIZE == 32u, "gpuxtb_interaction_t image must remain 32 bytes");
static_assert(sizeof(gpuxtb_interaction_t) == GPUXTB_INTERACTION_V1_SIZE,
              "gpuxtb_interaction_t must not add trailing ABI padding");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(offsetof(gpuxtb_batch_t, total_interactions) == 352u,
               "gpuxtb_batch_t ABI-v3 suffix must start at byte 352");
_Static_assert(offsetof(gpuxtb_batch_t, interaction_payload) == 384u,
               "gpuxtb_batch_t ABI-v3 payload must start at byte 384");
_Static_assert(GPUXTB_BATCH_V3_SIZE == 408u, "gpuxtb_batch_t ABI-v3 image must remain 408 bytes");
_Static_assert(sizeof(gpuxtb_batch_t) == GPUXTB_BATCH_V3_SIZE,
               "gpuxtb_batch_t must not add trailing ABI padding");
_Static_assert(GPUXTB_INTERACTION_V1_SIZE == 32u,
               "gpuxtb_interaction_t image must remain 32 bytes");
_Static_assert(sizeof(gpuxtb_interaction_t) == GPUXTB_INTERACTION_V1_SIZE,
               "gpuxtb_interaction_t must not add trailing ABI padding");
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
 * A GPUXTB_STATUS_SUCCESS return means every diagnostic entry was committed,
 * not necessarily that every system converged. per_system_status is SUCCESS,
 * SCC_NOT_CONVERGED, or EIGENSOLVER_FAILED; scc_converged is exactly one only
 * for SUCCESS. A failed system's requested floating-point property slices are
 * filled with quiet NaNs and never contain a partially evaluated result.
 */
typedef struct gpuxtb_batch_result {
  uint32_t struct_size;
  uint32_t api_version;
  uint32_t flags;
  uint32_t reserved;
  gpuxtb_buffer_t energies;
  gpuxtb_buffer_t forces;
  gpuxtb_buffer_t atomic_charges;
  gpuxtb_buffer_t point_charge_forces;
  gpuxtb_buffer_t scc_iterations;
  gpuxtb_buffer_t scc_converged;
  gpuxtb_buffer_t per_system_status;
  /* ABI v2 optional suffix; absent suffix preserves ABI-v1 behavior.
   *
   * dipole_moments holds batch_size * 3 doubles (atomic units) and is filled
   * when GPUXTB_COMPUTE_DIPOLE_MOMENTS is requested. The remaining outputs
   * are ABI-reserved: their shape contract is unpublished, their buffers must
   * be NULL until the matching output is released, and requesting them is
   * rejected before execution. */
  gpuxtb_buffer_t dipole_moments;
  gpuxtb_buffer_t quadrupole_moments;
  gpuxtb_buffer_t wiberg_orders;
  gpuxtb_buffer_t spin_populations;
} gpuxtb_batch_result_t;

#define GPUXTB_BATCH_RESULT_V1_SIZE \
  (offsetof(gpuxtb_batch_result_t, per_system_status) + sizeof(gpuxtb_buffer_t))
#define GPUXTB_BATCH_RESULT_V2_SIZE \
  (offsetof(gpuxtb_batch_result_t, spin_populations) + sizeof(gpuxtb_buffer_t))

#if defined(__cplusplus)
static_assert(offsetof(gpuxtb_batch_result_t, dipole_moments) == 184u,
              "gpuxtb_batch_result_t ABI-v2 suffix must start at byte 184");
static_assert(GPUXTB_BATCH_RESULT_V2_SIZE == 280u,
              "gpuxtb_batch_result_t ABI-v2 image must remain 280 bytes");
static_assert(sizeof(gpuxtb_batch_result_t) == GPUXTB_BATCH_RESULT_V2_SIZE,
              "gpuxtb_batch_result_t must not add trailing ABI padding");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(offsetof(gpuxtb_batch_result_t, dipole_moments) == 184u,
               "gpuxtb_batch_result_t ABI-v2 suffix must start at byte 184");
_Static_assert(GPUXTB_BATCH_RESULT_V2_SIZE == 280u,
               "gpuxtb_batch_result_t ABI-v2 image must remain 280 bytes");
_Static_assert(sizeof(gpuxtb_batch_result_t) == GPUXTB_BATCH_RESULT_V2_SIZE,
               "gpuxtb_batch_result_t must not add trailing ABI padding");
#endif

/*
 * Caller-friendly workspace sizing for one fixed-topology plan.
 *
 * compute_flags is an input carrying the properties the caller plans to
 * request on the fixed topology and policy. It must equal the flags supplied
 * to gpuxtb_plan_create. host_required_bytes / host_required_alignment
 * and device_required_bytes / device_required_alignment are outputs describing
 * the reusable plan-owned workspace gpuxtb_plan_compute reserves in host and
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
typedef struct gpuxtb_workspace_query {
  uint32_t struct_size;
  uint32_t api_version;
  uint32_t compute_flags;
  uint32_t reserved;
  uint64_t host_required_bytes;
  uint32_t host_required_alignment;
  uint64_t device_required_bytes;
  uint32_t device_required_alignment;
  uint32_t reserved_v2;
} gpuxtb_workspace_query_t;

#define GPUXTB_WORKSPACE_QUERY_V1_SIZE \
  (offsetof(gpuxtb_workspace_query_t, reserved_v2) + sizeof(uint32_t))

#if defined(__cplusplus)
static_assert(offsetof(gpuxtb_workspace_query_t, host_required_bytes) == 16u,
              "gpuxtb_workspace_query_t host byte count must start at byte 16");
static_assert(offsetof(gpuxtb_workspace_query_t, device_required_bytes) == 32u,
              "gpuxtb_workspace_query_t device byte count must start at byte 32");
static_assert(GPUXTB_WORKSPACE_QUERY_V1_SIZE == 48u,
              "gpuxtb_workspace_query_t ABI-v1 image must remain 48 bytes");
static_assert(sizeof(gpuxtb_workspace_query_t) == GPUXTB_WORKSPACE_QUERY_V1_SIZE,
              "gpuxtb_workspace_query_t must not add trailing ABI padding");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(offsetof(gpuxtb_workspace_query_t, host_required_bytes) == 16u,
               "gpuxtb_workspace_query_t host byte count must start at byte 16");
_Static_assert(offsetof(gpuxtb_workspace_query_t, device_required_bytes) == 32u,
               "gpuxtb_workspace_query_t device byte count must start at byte 32");
_Static_assert(GPUXTB_WORKSPACE_QUERY_V1_SIZE == 48u,
               "gpuxtb_workspace_query_t ABI-v1 image must remain 48 bytes");
_Static_assert(sizeof(gpuxtb_workspace_query_t) == GPUXTB_WORKSPACE_QUERY_V1_SIZE,
               "gpuxtb_workspace_query_t must not add trailing ABI padding");
#endif

GPUXTB_API const char* gpuxtb_version_string(void);
GPUXTB_API const char* gpuxtb_status_string(gpuxtb_status_t status);

/* Returns a thread-local diagnostic for the most recent failing API call. */
GPUXTB_API const char* gpuxtb_get_last_error(void);

GPUXTB_API gpuxtb_status_t gpuxtb_context_options_init(gpuxtb_context_options_t* options,
                                                       size_t struct_size);
GPUXTB_API gpuxtb_status_t gpuxtb_batch_init(gpuxtb_batch_t* batch, size_t struct_size);
GPUXTB_API gpuxtb_status_t gpuxtb_compute_options_init(gpuxtb_compute_options_t* options,
                                                       size_t struct_size);
GPUXTB_API gpuxtb_status_t gpuxtb_batch_result_init(gpuxtb_batch_result_t* result,
                                                    size_t struct_size);
GPUXTB_API gpuxtb_status_t gpuxtb_workspace_query_init(gpuxtb_workspace_query_t* query,
                                                       size_t struct_size);

GPUXTB_API gpuxtb_status_t gpuxtb_context_create(const gpuxtb_context_options_t* options,
                                                 gpuxtb_context_t** context);
GPUXTB_API void gpuxtb_context_destroy(gpuxtb_context_t* context);
GPUXTB_API gpuxtb_backend_t gpuxtb_context_get_backend(const gpuxtb_context_t* context);
GPUXTB_API int32_t gpuxtb_context_get_device_id(const gpuxtb_context_t* context);

/*
 * Performs a synchronous batched inference. Host buffers are accepted by both
 * CPU and CUDA backends; CUDA device buffers avoid staging copies on CUDA.
 *
 * The complete request is validated before execution. Any failure detected
 * before the final caller-output commit begins leaves result flags and all
 * result buffers unchanged. Once a CUDA caller-output commit has begun, a
 * later catastrophic failure returns GPUXTB_STATUS_INTERNAL_ERROR and results
 * may already have been modified. CUDA attempts to restore the caller's current
 * device on every exit; restoration failure also returns INTERNAL_ERROR and
 * may leave that device selection changed, independently of whether output
 * commit began. gpuxtb_get_last_error identifies the failed boundary.
 * Per-system SCC or eigensolver failures are data-level results: the function
 * returns SUCCESS and records them in per_system_status so one bad batch item
 * does not discard successful peers.
 */
GPUXTB_API gpuxtb_status_t gpuxtb_compute(gpuxtb_context_t* context, const gpuxtb_batch_t* batch,
                                          const gpuxtb_compute_options_t* options,
                                          gpuxtb_batch_result_t* result);

/*
 * Create a fixed-topology plan from one already-validated-shaped batch
 * descriptor and a compute policy. The plan binds the immutable topology (atom
 * offsets, element numbers, spin channels, point-charge and response structure)
 * and the numerical policy (model, requested properties, SCC tolerances,
 * iteration limit, and electronic temperature) to the context backend and
 * reserves its reusable host/device workspace. Geometry (positions and
 * point-charge positions/values) is intentionally not part of the plan and
 * may change per gpuxtb_plan_compute call.
 *
 * The plan is a setup-with-allocation-permitted path: creating it performs
 * validation and workspace reservation that gpuxtb_compute would otherwise
 * repeat on every call. Calling gpuxtb_plan_compute repeatedly for the same
 * fixed topology must not allocate steady-state workspace on either backend.
 *
 * A plan is bound to the creating context; it must be destroyed with
 * gpuxtb_plan_destroy before the context. Passing a plan whose topology does
 * not match the batch on gpuxtb_plan_compute, or a plan created for a
 * different context, fails before any caller output is modified.
 */
GPUXTB_API gpuxtb_status_t gpuxtb_plan_create(gpuxtb_context_t* context,
                                              const gpuxtb_batch_t* batch,
                                              const gpuxtb_compute_options_t* options,
                                              gpuxtb_plan_t** plan);
GPUXTB_API void gpuxtb_plan_destroy(gpuxtb_plan_t* plan);

/*
 * Query the reusable plan-owned workspace gpuxtb_plan_compute reserves on the
 * plan's backend for its requested properties. On return query.compute_flags
 * is preserved and must match the plan policy; the four sizing fields are populated as documented
 * on gpuxtb_workspace_query_t. Callers that want device sizing must use a CUDA plan; CPU plans
 * always report zero device bytes.
 */
GPUXTB_API gpuxtb_status_t gpuxtb_plan_query_workspace(const gpuxtb_plan_t* plan,
                                                       gpuxtb_workspace_query_t* query);

/*
 * Execute one synchronous batched inference on a fixed-topology plan.
 *
 * geometry-only descriptors: positions and point-charge positions/values may
 * change between calls, but the immutable topology and creation-time compute
 * policy must match the plan exactly or the call fails with
 * GPUXTB_STATUS_INVALID_ARGUMENT before any caller output is modified.
 * Otherwise semantics match gpuxtb_compute: complete
 * validation before execution, per-system SCC/eigensolver failures recorded in
 * per_system_status, and failed systems' floating-point slices filled with
 * quiet NaNs.
 */
GPUXTB_API gpuxtb_status_t gpuxtb_plan_compute(gpuxtb_plan_t* plan, const gpuxtb_batch_t* batch,
                                               const gpuxtb_compute_options_t* options,
                                               gpuxtb_batch_result_t* result);

/*
 * gpuxtb-owned result arenas and their DLPack export.
 *
 * A gpuxtb_result_owner_t is a ref-counted allocation (host memory or CUDA
 * device memory) that gpuxtb itself allocates, fills through a normal compute
 * call, and can hand to an importing framework through the DLPack producer
 * protocol without copying data.
 *
 * Lifetime model:
 * - gpuxtb_result_owner_create produces one arena with an initial reference.
 *   gpuxtb_result_owner_buffer exposes that arena as a caller-owned
 *   gpuxtb_buffer_t view so the caller can bind output slices and run compute.
 * - gpuxtb_result_owner_retain / gpuxtb_result_owner_release manage the
 *   reference count. The arena allocation is freed exactly once, when the
 *   last reference is released. release(NULL) is a no-op; otherwise every
 *   release must correspond to exactly one prior create or retain.
 * - gpuxtb_result_owner_export_dltensor retains the arena for one exported
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
 * gpuxtb's public CUDA compute is synchronous: when gpuxtb_compute returns,
 * the requested result bytes are fully committed on the context stream and a
 * producer export needs no additional device-wide synchronization or hidden
 * host polling.
 */

/* Options for allocating one result arena. */
typedef struct gpuxtb_result_owner_options {
  uint32_t struct_size;
  uint32_t api_version;
  /* GPUXTB_MEMORY_HOST or GPUXTB_MEMORY_CUDA_DEVICE. */
  gpuxtb_memory_space_t memory_space;
  /* CUDA device ordinal for CUDA arenas; -1 for host arenas. */
  int32_t device_id;
  /* Byte extent of the arena. Must be nonzero. */
  uint64_t size_bytes;
  uint32_t reserved;
} gpuxtb_result_owner_options_t;

#define GPUXTB_RESULT_OWNER_OPTIONS_V1_SIZE \
  (offsetof(gpuxtb_result_owner_options_t, reserved) + sizeof(uint32_t))

/*
 * Describes one compact C-contiguous slice of a result arena to export as a
 * DLPack managed tensor. dtype mirrors DLPack's DLDataType triple
 * (code/bits/lanes); shape holds ndim int64 extents and is caller-owned only
 * for the duration of the export call (gpuxtb copies it into the managed
 * tensor's storage). strides are always implicit compact row-major (NULL by
 * DLPack convention). Accepted dtypes: int8/int16/int32/int64, uint8,
 * float32/float64 with lanes == 1. No DLPack headers are required from the
 * caller: this structure is the constructor input, and the managed-tensor
 * struct layout mirrors the pinned DLPack 1.0 specification (see
 * src/runtime/dlpack_layout.hpp for the byte-exact mirrors and provenance).
 */
typedef struct gpuxtb_dlpack_view {
  uint32_t struct_size;
  uint32_t api_version;
  /* Byte offset of the slice inside the arena. Must keep dtype alignment. */
  uint64_t byte_offset;
  int32_t dtype_code; /* DLDataTypeCode: 0 int, 1 uint, 2 float, 4 bfloat, 6 bool. */
  int32_t dtype_bits;
  int32_t dtype_lanes;
  int32_t ndim; /* 0..8 */
  uint32_t reserved;
  const int64_t* shape; /* ndim int64 values; copied by gpuxtb */
} gpuxtb_dlpack_view_t;

#define GPUXTB_DLPACK_MAX_NDIM 8
#define GPUXTB_DLPACK_VIEW_V1_SIZE (offsetof(gpuxtb_dlpack_view_t, shape) + sizeof(const int64_t*))

GPUXTB_API gpuxtb_status_t gpuxtb_result_owner_options_init(gpuxtb_result_owner_options_t* options,
                                                            size_t struct_size);
GPUXTB_API gpuxtb_status_t gpuxtb_result_owner_create(const gpuxtb_result_owner_options_t* options,
                                                      gpuxtb_result_owner_t** owner);
/* Copies the whole arena into buffer as a caller-owned borrowed view. */
GPUXTB_API gpuxtb_status_t gpuxtb_result_owner_buffer(const gpuxtb_result_owner_t* owner,
                                                      gpuxtb_buffer_t* buffer);
GPUXTB_API void gpuxtb_result_owner_retain(gpuxtb_result_owner_t* owner);
GPUXTB_API void gpuxtb_result_owner_release(gpuxtb_result_owner_t* owner);
/*
 * Export one arena slice as a heap-allocated DLManagedTensorVersioned (when
 * version != 0) or legacy DLManagedTensor (when version == 0). On success
 * *out_managed receives the pointer and gpuxtb owns its lifetime: the import
 * consumer (or capsule destructor) must call the stored deleter exactly once.
 * On failure *out_managed is set to NULL, no arena reference is taken, and
 * the arena reference counting is untouched.
 */
GPUXTB_API gpuxtb_status_t gpuxtb_result_owner_export_dltensor(const gpuxtb_result_owner_t* owner,
                                                               const gpuxtb_dlpack_view_t* view,
                                                               int version, void** out_managed);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GPUXTB_GPUXTB_H */
