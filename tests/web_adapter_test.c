/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Regression coverage for the single-system browser adapter. The production
 * source is included directly so its private JSON and L-BFGS invariants can be
 * exercised against a deterministic fake of the stable xtbloom C ABI. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define main xtbloom_web_embedded_main
#include "../web/xtbloom_web.c"
#undef main

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return 1;                                                                       \
    }                                                                                 \
  } while (0)

enum MockComputeMode {
  MOCK_COMPUTE_SUCCESS,
  MOCK_COMPUTE_SYSTEM_FAILURE,
};

/* Per-call log of the SCC start mode/struct_size requested through the mock so
 * tests can prove which native policy the adapter asked for on each compute. */
#define MOCK_CALL_LOG_CAP 128
static int32_t mock_call_modes[MOCK_CALL_LOG_CAP];
static int32_t mock_call_models[MOCK_CALL_LOG_CAP];
static uint32_t mock_call_struct_sizes[MOCK_CALL_LOG_CAP];
static int mock_call_count = 0;

/* Native strict-WARM gate simulation: a WARM request is accepted only when the
 * latest call fully converged with compatible options, mirroring the real CPU
 * gate. An accepted FRESH/WARM attempt consumes the checkpoint before running;
 * only full convergence replenishes one. Also lets tests force an incompatible
 * identity by rejecting WARM outright. */
static int mock_checkpoint_ready = 0;
static int mock_force_warm_reject = 0;
static int mock_warm_solves = 0;
static int mock_fresh_solves = 0;
static int mock_warm_rejections = 0;
/* 0-based index of the compute call that must report SCC non-convergence, or
 * -1 to never fail. Lets a test make one optimization step fail mid-run. */
static int mock_fail_at_call = -1;

static enum MockComputeMode mock_compute_mode = MOCK_COMPUTE_SUCCESS;
static int mock_saw_force_buffer = 0;
static uint32_t mock_compute_flags = 0;
static const char* mock_last_error = "";

static void reset_adapter(void) {
  g_context = (xtbloom_context_t*)(uintptr_t)1;
  g_warm_ready = 0;
  mock_compute_mode = MOCK_COMPUTE_SUCCESS;
  mock_saw_force_buffer = 0;
  mock_compute_flags = 0;
  mock_last_error = "";
  mock_call_count = 0;
  mock_checkpoint_ready = 0;
  mock_force_warm_reject = 0;
  mock_warm_solves = 0;
  mock_fresh_solves = 0;
  mock_warm_rejections = 0;
  mock_fail_at_call = -1;
}

const char* xtbloom_version_string(void) { return "test"; }

const char* xtbloom_status_string(xtbloom_status_t status) {
  switch (status) {
    case XTBLOOM_STATUS_SUCCESS:
      return "success";
    case XTBLOOM_STATUS_SCC_NOT_CONVERGED:
      return "SCC not converged";
    case XTBLOOM_STATUS_EIGENSOLVER_FAILED:
      return "eigensolver failed";
    default:
      return "mock failure";
  }
}

const char* xtbloom_get_last_error(void) { return mock_last_error; }

xtbloom_status_t xtbloom_context_options_init(xtbloom_context_options_t* options,
                                              size_t struct_size) {
  memset(options, 0, struct_size);
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t xtbloom_batch_init(xtbloom_batch_t* batch, size_t struct_size) {
  memset(batch, 0, struct_size);
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t xtbloom_compute_options_init(xtbloom_compute_options_t* options,
                                              size_t struct_size) {
  memset(options, 0, struct_size);
  /* The real initializer stores the full platform size so the native caller
   * sees the ABI-v2 suffix; the adapter then selects FRESH/WARM explicitly. */
  options->struct_size = (uint32_t)struct_size;
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t xtbloom_batch_result_init(xtbloom_batch_result_t* result, size_t struct_size) {
  memset(result, 0, struct_size);
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t xtbloom_context_create(const xtbloom_context_options_t* options,
                                        xtbloom_context_t** context) {
  (void)options;
  *context = (xtbloom_context_t*)(uintptr_t)1;
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t xtbloom_compute(xtbloom_context_t* context, const xtbloom_batch_t* batch,
                                 const xtbloom_compute_options_t* options,
                                 xtbloom_batch_result_t* result) {
  (void)context;
  mock_compute_flags = options->flags;
  mock_saw_force_buffer = result->forces.data != NULL || result->forces.size_bytes != 0;

  /* An absent V2 suffix means FRESH (the public short-structure contract). */
  const int32_t start_mode = options->struct_size >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE
                                 ? options->scc_start_mode
                                 : XTBLOOM_SCC_START_FRESH;
  if (mock_call_count < MOCK_CALL_LOG_CAP) {
    mock_call_modes[mock_call_count] = start_mode;
    mock_call_models[mock_call_count] = options->model;
    mock_call_struct_sizes[mock_call_count] = options->struct_size;
  }
  ++mock_call_count;

  const int warm_requested = start_mode == XTBLOOM_SCC_START_WARM;
  if (warm_requested && (mock_force_warm_reject != 0 || !mock_checkpoint_ready)) {
    /* Strict native WARM gate: refuse before any caller output is touched. */
    mock_last_error =
        "CPU WARM SCC start requires a previous fully converged call with identical "
        "topology and compute options";
    ++mock_warm_rejections;
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (warm_requested) {
    ++mock_warm_solves;
  } else {
    ++mock_fresh_solves;
  }
  /* An accepted attempt consumes the preceding checkpoint (native semantics);
   * full convergence replenishes one usable by the next WARM request. */
  mock_checkpoint_ready = 0;

  int32_t* statuses = (int32_t*)result->per_system_status.data;
  int32_t* iterations = (int32_t*)result->scc_iterations.data;
  uint8_t* converged = (uint8_t*)result->scc_converged.data;
  double* energies = (double*)result->energies.data;
  double* charges = (double*)result->atomic_charges.data;
  double* forces = (double*)result->forces.data;

  iterations[0] = 3;
  const int fail_here =
      mock_compute_mode == MOCK_COMPUTE_SYSTEM_FAILURE || mock_fail_at_call == mock_call_count - 1;
  if (fail_here) {
    statuses[0] = XTBLOOM_STATUS_SCC_NOT_CONVERGED;
    converged[0] = 0;
    energies[0] = NAN;
    for (int64_t i = 0; i < batch->total_atoms; ++i) {
      charges[i] = NAN;
    }
    if (forces != NULL) {
      for (int64_t i = 0; i < 3 * batch->total_atoms; ++i) {
        forces[i] = NAN;
      }
    }
    return XTBLOOM_STATUS_SUCCESS;
  }

  /* A quadratic single-atom model, E = 0.5*(x-10)^2 in bohr, so the L-BFGS
   * adapter takes several well-defined descent steps and every accepted step
   * leaves a fully converged checkpoint. Forces are -dE/dx in Eh/bohr. */
  const double* pos = (const double*)batch->positions.data;
  const double dx = pos[0] - 10.0;
  statuses[0] = XTBLOOM_STATUS_SUCCESS;
  converged[0] = 1;
  energies[0] = 0.5 * dx * dx;
  for (int64_t i = 0; i < batch->total_atoms; ++i) {
    charges[i] = 0.0;
  }
  if (forces != NULL) {
    forces[0] = -dx;
    forces[1] = 0.0;
    forces[2] = 0.0;
  }
  mock_checkpoint_ready = 1;
  return XTBLOOM_STATUS_SUCCESS;
}

static int test_error_json_escapes_diagnostics(void) {
  const char* actual = error_json("err_ctx", "bad \"path\"\\line\nnext");
  CHECK(strcmp(actual,
               "{\"ok\":0,\"error_code\":\"err_ctx\",\"error\":"
               "\"bad \\\"path\\\"\\\\line\\nnext\"}") == 0);
  return 0;
}

static int test_system_failure_is_not_success(void) {
  reset_adapter();
  mock_compute_mode = MOCK_COMPUTE_SYSTEM_FAILURE;
  const char* actual =
      xtbloom_web_compute("H 0 0 0", XTBLOOM_MODEL_GFN2_XTB, 0.0, 0, 0.0, 1e-8, 1e-5, 1, 1);
  CHECK(strstr(actual, "\"ok\":0") != NULL);
  CHECK(strstr(actual, "\"error_code\":\"err_compute\"") != NULL);
  CHECK(strstr(actual, "SCC not converged") != NULL);
  CHECK(strstr(actual, "\"energy_Eh\"") == NULL);
  return 0;
}

static int test_disabled_forces_are_omitted(void) {
  reset_adapter();
  const char* actual =
      xtbloom_web_compute("H 0 0 0", XTBLOOM_MODEL_GFN2_XTB, 0.0, 0, 0.0, 1e-8, 1e-5, 20, 0);
  CHECK(strstr(actual, "\"ok\":1") != NULL);
  CHECK(strstr(actual, "\"forces\"") == NULL);
  CHECK(mock_saw_force_buffer == 0);
  CHECK((mock_compute_flags & XTBLOOM_COMPUTE_FORCES) == 0);
  return 0;
}

static int test_initial_optimizer_failure_is_defined(void) {
  reset_adapter();
  mock_compute_mode = MOCK_COMPUTE_SYSTEM_FAILURE;
  const char* actual = xtbloom_web_optimize("H 0 0 0", XTBLOOM_MODEL_GFN2_XTB, 0.0, 0, 0.0, 1e-8,
                                            1e-5, 1, 2, 4.5e-4, 0.4);
  CHECK(strstr(actual, "\"ok\":0") != NULL);
  CHECK(strstr(actual, "\"error_code\":\"err_initial_calc\"") != NULL);
  CHECK(strstr(actual, "SCC not converged") != NULL);
  return 0;
}

static int test_lbfgs_invariants(void) {
  const double g[] = {1.0, -2.0};
  double p[] = {1.0, -2.0};
  w_ensure_descent(g, p, 2);
  CHECK(p[0] == -1.0 && p[1] == 2.0);
  CHECK(w_dot(g, p, 2) == -5.0);

  const double s[] = {2.0, 1.0};
  const double y[] = {3.0, 4.0};
  CHECK(fabs(w_reciprocal_curvature(s, y, 2) - 0.1) < 1e-15);
  const double negative_y[] = {-3.0, -4.0};
  CHECK(w_reciprocal_curvature(s, negative_y, 2) == 0.0);
  return 0;
}

/* The quadratic mock model lets the optimizer take several steps. Helpers
 * share the "run a multi-step optimization starting at x=0" setup. */
static const char* run_optimize_model(int model, int opt_max_iterations) {
  return xtbloom_web_optimize("H 0 0 0", model, 0.0, 0, 0.0, 1e-8, 1e-5, 20, opt_max_iterations,
                              1e-12, 0.4);
}

static const char* run_optimize(int opt_max_iterations) {
  return run_optimize_model(XTBLOOM_MODEL_GFN2_XTB, opt_max_iterations);
}

/* First evaluation of an optimization run must be FRESH; every later
 * evaluation with the same topology/charge/spin/options must be WARM, so the
 * browser optimizer reuses the previous converged electronic state. */
static int test_warm_start_first_fresh_then_warm(void) {
  reset_adapter();
  const char* actual = run_optimize(5);
  CHECK(strstr(actual, "\"ok\":1") != NULL);
  CHECK(mock_call_count >= 2);
  CHECK(mock_call_modes[0] == XTBLOOM_SCC_START_FRESH);
  CHECK(mock_call_struct_sizes[0] >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE);
  for (int i = 1; i < mock_call_count; ++i) {
    CHECK(mock_call_modes[i] == XTBLOOM_SCC_START_WARM);
    CHECK(mock_call_struct_sizes[i] >= XTBLOOM_COMPUTE_OPTIONS_V2_SIZE);
  }
  /* exactly one fresh solve, the rest warm, no rejected-warm fallback */
  CHECK(mock_fresh_solves == 1);
  CHECK(mock_warm_solves == mock_call_count - 1);
  CHECK(mock_warm_rejections == 0);
  char want[64];
  snprintf(want, sizeof(want), "\"scc_warm_solves\":%d", mock_warm_solves);
  CHECK(strstr(actual, want) != NULL);
  CHECK(strstr(actual, "\"scc_warm_fallbacks\":0") != NULL);
  return 0;
}

/* A new optimization run must reset the previous run's warm state: its first
 * evaluation is FRESH even when the context still holds a converged checkpoint
 * from an earlier, geometry-identical run. */
static int test_new_optimization_resets_warm_state(void) {
  reset_adapter();
  const char* first = run_optimize(3);
  CHECK(strstr(first, "\"ok\":1") != NULL);
  const int run1_calls = mock_call_count;
  CHECK(run1_calls >= 2);
  CHECK(mock_call_modes[run1_calls - 1] == XTBLOOM_SCC_START_WARM);

  const int before_second_run = mock_call_count;
  const char* second = run_optimize(3);
  CHECK(strstr(second, "\"ok\":1") != NULL);
  CHECK(mock_call_count > before_second_run);
  CHECK(mock_call_modes[before_second_run] == XTBLOOM_SCC_START_FRESH);
  CHECK(mock_call_modes[before_second_run + 1] == XTBLOOM_SCC_START_WARM);
  return 0;
}

/* Standalone single-point evaluations must stay fresh and independent even
 * when the shared context holds a converged checkpoint from an optimization. */
static int test_single_point_stays_fresh(void) {
  reset_adapter();
  const char* opt = run_optimize(3);
  CHECK(strstr(opt, "\"ok\":1") != NULL);
  CHECK(mock_call_modes[mock_call_count - 1] == XTBLOOM_SCC_START_WARM);
  const int before_single = mock_call_count;
  const char* single =
      xtbloom_web_compute("H 0 0 0", XTBLOOM_MODEL_GFN2_XTB, 0.0, 0, 0.0, 1e-8, 1e-5, 20, 1);
  CHECK(strstr(single, "\"ok\":1") != NULL);
  CHECK(mock_call_count == before_single + 1);
  CHECK(mock_call_modes[before_single] == XTBLOOM_SCC_START_FRESH);
  return 0;
}

static int test_model_tags_are_explicit_and_published(void) {
  reset_adapter();
  const char* gfn1 =
      xtbloom_web_compute("H 0 0 0", XTBLOOM_MODEL_GFN1_XTB, 0.0, 0, 0.0, 1e-8, 1e-5, 20, 1);
  CHECK(strstr(gfn1, "\"model\":1") != NULL);
  CHECK(strstr(gfn1, "\"method\":\"GFN1-xTB\"") != NULL);
  CHECK(mock_call_models[0] == XTBLOOM_MODEL_GFN1_XTB);

  const char* gfn2 =
      xtbloom_web_compute("H 0 0 0", XTBLOOM_MODEL_GFN2_XTB, 0.0, 0, 0.0, 1e-8, 1e-5, 20, 1);
  CHECK(strstr(gfn2, "\"model\":2") != NULL);
  CHECK(strstr(gfn2, "\"method\":\"GFN2-xTB\"") != NULL);
  CHECK(mock_call_models[1] == XTBLOOM_MODEL_GFN2_XTB);

  const int calls_before_invalid = mock_call_count;
  const char* invalid = xtbloom_web_compute("H 0 0 0", 99, 0.0, 0, 0.0, 1e-8, 1e-5, 20, 1);
  CHECK(strcmp(invalid, "{\"ok\":0,\"error_code\":\"err_model\"}") == 0);
  CHECK(mock_call_count == calls_before_invalid);
  return 0;
}

static int test_gfn1_optimization_starts_fresh_and_stays_gfn1(void) {
  reset_adapter();
  const char* actual = run_optimize_model(XTBLOOM_MODEL_GFN1_XTB, 3);
  CHECK(strstr(actual, "\"ok\":1") != NULL);
  CHECK(strstr(actual, "\"model\":1") != NULL);
  CHECK(strstr(actual, "\"method\":\"GFN1-xTB\"") != NULL);
  CHECK(mock_call_modes[0] == XTBLOOM_SCC_START_FRESH);
  for (int i = 0; i < mock_call_count; ++i) {
    CHECK(mock_call_models[i] == XTBLOOM_MODEL_GFN1_XTB);
  }
  return 0;
}

/* When the strict native gate refuses a WARM request (incompatible identity or
 * no fully converged predecessor), the adapter must transparently retry FRESH
 * and keep the optimization going. */
static int test_warm_rejection_falls_back_to_fresh(void) {
  reset_adapter();
  mock_force_warm_reject = 1;
  const char* actual = run_optimize(4);
  CHECK(strstr(actual, "\"ok\":1") != NULL);
  CHECK(mock_warm_rejections >= 1);
  CHECK(mock_fresh_solves >= 2);
  CHECK(mock_warm_solves == 0);
  /* Rejected WARM requests are fallbacks, not warm SCC solves. Reconcile the
   * published diagnostics with the mock's accepted native solve counters. */
  char want[64];
  snprintf(want, sizeof(want), "\"scc_warm_solves\":%d", mock_warm_solves);
  CHECK(strstr(actual, want) != NULL);
  snprintf(want, sizeof(want), "\"scc_fresh_solves\":%d", mock_fresh_solves);
  CHECK(strstr(actual, want) != NULL);
  snprintf(want, sizeof(want), "\"scc_warm_fallbacks\":%d", mock_warm_rejections);
  CHECK(strstr(actual, want) != NULL);
  return 0;
}

/* A step that fails to converge must not poison the next calculation: the next
 * optimization starts FRESH again (never consumes the failed run's state). */
static int test_failed_step_does_not_poison_next_calculation(void) {
  reset_adapter();
  mock_fail_at_call = 1; /* first line-search evaluation of the initial run */
  const char* failed = run_optimize(4);
  CHECK(strstr(failed, "\"ok\":0") != NULL);
  CHECK(strstr(failed, "\"error_code\":\"err_step_sp\"") != NULL);
  CHECK(g_warm_ready == 0);

  const int before_second_run = mock_call_count;
  const char* second = run_optimize(3);
  CHECK(strstr(second, "\"ok\":1") != NULL);
  CHECK(mock_call_modes[before_second_run] == XTBLOOM_SCC_START_FRESH);
  return 0;
}

int main(void) {
  CHECK(test_error_json_escapes_diagnostics() == 0);
  CHECK(test_system_failure_is_not_success() == 0);
  CHECK(test_disabled_forces_are_omitted() == 0);
  CHECK(test_initial_optimizer_failure_is_defined() == 0);
  CHECK(test_lbfgs_invariants() == 0);
  CHECK(test_warm_start_first_fresh_then_warm() == 0);
  CHECK(test_new_optimization_resets_warm_state() == 0);
  CHECK(test_single_point_stays_fresh() == 0);
  CHECK(test_warm_rejection_falls_back_to_fresh() == 0);
  CHECK(test_failed_step_does_not_poison_next_calculation() == 0);
  CHECK(test_model_tags_are_explicit_and_published() == 0);
  CHECK(test_gfn1_optimization_starts_fresh_and_stays_gfn1() == 0);
  free(g_result.data);
  return 0;
}
