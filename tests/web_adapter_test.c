/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Regression coverage for the single-system browser adapter. The production
 * source is included directly so its private JSON and L-BFGS invariants can be
 * exercised against a deterministic fake of the stable gpuxtb C ABI. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define main gpuxtb_web_embedded_main
#include "../web/gpuxtb_web.c"
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

static enum MockComputeMode mock_compute_mode = MOCK_COMPUTE_SUCCESS;
static int mock_saw_force_buffer = 0;
static uint32_t mock_compute_flags = 0;
static const char* mock_last_error = "";

static void reset_adapter(void) {
  g_context = (gpuxtb_context_t*)(uintptr_t)1;
  mock_compute_mode = MOCK_COMPUTE_SUCCESS;
  mock_saw_force_buffer = 0;
  mock_compute_flags = 0;
  mock_last_error = "";
}

const char* gpuxtb_version_string(void) { return "test"; }

const char* gpuxtb_status_string(gpuxtb_status_t status) {
  switch (status) {
    case GPUXTB_STATUS_SUCCESS:
      return "success";
    case GPUXTB_STATUS_SCC_NOT_CONVERGED:
      return "SCC not converged";
    case GPUXTB_STATUS_EIGENSOLVER_FAILED:
      return "eigensolver failed";
    default:
      return "mock failure";
  }
}

const char* gpuxtb_get_last_error(void) { return mock_last_error; }

gpuxtb_status_t gpuxtb_context_options_init(gpuxtb_context_options_t* options, size_t struct_size) {
  memset(options, 0, struct_size);
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t gpuxtb_batch_init(gpuxtb_batch_t* batch, size_t struct_size) {
  memset(batch, 0, struct_size);
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t gpuxtb_compute_options_init(gpuxtb_compute_options_t* options, size_t struct_size) {
  memset(options, 0, struct_size);
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t gpuxtb_batch_result_init(gpuxtb_batch_result_t* result, size_t struct_size) {
  memset(result, 0, struct_size);
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t gpuxtb_context_create(const gpuxtb_context_options_t* options,
                                      gpuxtb_context_t** context) {
  (void)options;
  *context = (gpuxtb_context_t*)(uintptr_t)1;
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t gpuxtb_compute(gpuxtb_context_t* context, const gpuxtb_batch_t* batch,
                               const gpuxtb_compute_options_t* options,
                               gpuxtb_batch_result_t* result) {
  (void)context;
  mock_compute_flags = options->flags;
  mock_saw_force_buffer = result->forces.data != NULL || result->forces.size_bytes != 0;

  int32_t* statuses = (int32_t*)result->per_system_status.data;
  int32_t* iterations = (int32_t*)result->scc_iterations.data;
  uint8_t* converged = (uint8_t*)result->scc_converged.data;
  double* energies = (double*)result->energies.data;
  double* charges = (double*)result->atomic_charges.data;
  double* forces = (double*)result->forces.data;

  iterations[0] = 3;
  if (mock_compute_mode == MOCK_COMPUTE_SYSTEM_FAILURE) {
    statuses[0] = GPUXTB_STATUS_SCC_NOT_CONVERGED;
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
    return GPUXTB_STATUS_SUCCESS;
  }

  statuses[0] = GPUXTB_STATUS_SUCCESS;
  converged[0] = 1;
  energies[0] = -1.25;
  for (int64_t i = 0; i < batch->total_atoms; ++i) {
    charges[i] = 0.0;
  }
  if (forces != NULL) {
    for (int64_t i = 0; i < 3 * batch->total_atoms; ++i) {
      forces[i] = 0.0;
    }
  }
  return GPUXTB_STATUS_SUCCESS;
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
  const char* actual = gpuxtb_web_compute("H 0 0 0", 0.0, 0, 0.0, 1e-8, 1e-5, 1, 1);
  CHECK(strstr(actual, "\"ok\":0") != NULL);
  CHECK(strstr(actual, "\"error_code\":\"err_compute\"") != NULL);
  CHECK(strstr(actual, "SCC not converged") != NULL);
  CHECK(strstr(actual, "\"energy_Eh\"") == NULL);
  return 0;
}

static int test_disabled_forces_are_omitted(void) {
  reset_adapter();
  const char* actual = gpuxtb_web_compute("H 0 0 0", 0.0, 0, 0.0, 1e-8, 1e-5, 20, 0);
  CHECK(strstr(actual, "\"ok\":1") != NULL);
  CHECK(strstr(actual, "\"forces\"") == NULL);
  CHECK(mock_saw_force_buffer == 0);
  CHECK((mock_compute_flags & GPUXTB_COMPUTE_FORCES) == 0);
  return 0;
}

static int test_initial_optimizer_failure_is_defined(void) {
  reset_adapter();
  mock_compute_mode = MOCK_COMPUTE_SYSTEM_FAILURE;
  const char* actual = gpuxtb_web_optimize("H 0 0 0", 0.0, 0, 0.0, 1e-8, 1e-5, 1, 2, 4.5e-4, 0.4);
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

int main(void) {
  CHECK(test_error_json_escapes_diagnostics() == 0);
  CHECK(test_system_failure_is_not_success() == 0);
  CHECK(test_disabled_forces_are_omitted() == 0);
  CHECK(test_initial_optimizer_failure_is_defined() == 0);
  CHECK(test_lbfgs_invariants() == 0);
  free(g_result.data);
  return 0;
}
