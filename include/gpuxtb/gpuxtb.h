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
  GPUXTB_COMPUTE_POINT_CHARGE_FORCES = 1 << 3
};

typedef int32_t gpuxtb_result_flag_t;
enum gpuxtb_result_flag_value {
  /*
   * Set when atomic_potential_shifts or charge_response_matrix was supplied.
   * Forces then exclude coordinate derivatives of those caller-owned fields.
   */
  GPUXTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES = 1 << 0
};

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
} gpuxtb_batch_t;

#define GPUXTB_BATCH_V1_SIZE \
  (offsetof(gpuxtb_batch_t, charge_response_matrix) + sizeof(gpuxtb_const_buffer_t))
#define GPUXTB_BATCH_V2_SIZE \
  (offsetof(gpuxtb_batch_t, spin_channels) + sizeof(gpuxtb_const_buffer_t))

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
} gpuxtb_batch_result_t;

#define GPUXTB_BATCH_RESULT_V1_SIZE \
  (offsetof(gpuxtb_batch_result_t, per_system_status) + sizeof(gpuxtb_buffer_t))

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
 * measured from the complete prepared runtime, including numerical, result,
 * validation, and mixed-memory topology staging.
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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GPUXTB_GPUXTB_H */
