#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <xtbloom/xtbloom.h>

static xtbloom_const_buffer_t host_input(const void* data, size_t size_bytes) {
  const xtbloom_const_buffer_t view = {data, size_bytes, XTBLOOM_MEMORY_HOST, 0};
  return view;
}

static xtbloom_buffer_t host_output(void* data, size_t size_bytes) {
  const xtbloom_buffer_t view = {data, size_bytes, XTBLOOM_MEMORY_HOST, 0};
  return view;
}

static int select_backend(int argc, char** argv, xtbloom_backend_t* backend) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s cpu|cuda\n", argv[0]);
    return 0;
  }
  if (strcmp(argv[1], "cpu") == 0) {
    *backend = XTBLOOM_BACKEND_CPU;
    return 1;
  }
  if (strcmp(argv[1], "cuda") == 0) {
    *backend = XTBLOOM_BACKEND_CUDA;
    return 1;
  }
  fprintf(stderr, "unknown backend %s; expected cpu or cuda\n", argv[1]);
  return 0;
}

static int check_system(const char* phase, xtbloom_status_t call_status,
                        xtbloom_status_t system_status, uint8_t converged, int32_t iterations,
                        double energy) {
  if (call_status != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "%s call failed: %s (%s)\n", phase, xtbloom_status_string(call_status),
            xtbloom_get_last_error());
    return 0;
  }
  /* API success means diagnostics were published; it does not imply that this
   * batch item converged. Failed property slices are quiet NaNs. */
  if (system_status != XTBLOOM_STATUS_SUCCESS || converged != 1 || iterations <= 0 ||
      !isfinite(energy)) {
    fprintf(stderr, "%s system failed: status=%s converged=%u iterations=%d energy=%.17g\n", phase,
            xtbloom_status_string(system_status), (unsigned)converged, (int)iterations, energy);
    return 0;
  }
  return 1;
}

int main(int argc, char** argv) {
  xtbloom_backend_t requested_backend = XTBLOOM_BACKEND_CPU;
  if (!select_backend(argc, argv, &requested_backend)) {
    return 2;
  }

  xtbloom_context_options_t context_options;
  xtbloom_batch_t batch;
  xtbloom_compute_options_t compute_options;
  xtbloom_batch_result_t result;
  if (xtbloom_context_options_init(&context_options, sizeof(context_options)) !=
          XTBLOOM_STATUS_SUCCESS ||
      xtbloom_batch_init(&batch, sizeof(batch)) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom_compute_options_init(&compute_options, sizeof(compute_options)) !=
          XTBLOOM_STATUS_SUCCESS ||
      xtbloom_batch_result_init(&result, sizeof(result)) != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "descriptor initialization failed: %s\n", xtbloom_get_last_error());
    return 3;
  }

  /* Require the requested backend. AUTO would be appropriate only if CUDA to
   * CPU fallback were an explicit application policy. A CUDA context accepts
   * these host views and stages them internally; no CUDA compiler is needed by
   * this external C consumer. Active CUDA stream capture is rejected. */
  context_options.backend = requested_backend;

  const int64_t atom_offsets[] = {0, 2};
  const int32_t atomic_numbers[] = {1, 1};
  double positions[] = {-0.70, 0.0, 0.0, 0.70, 0.0, 0.0};
  const double molecular_charges[] = {0.0};
  const int32_t unpaired_electrons[] = {0};

  batch.batch_size = 1;
  batch.total_atoms = 2;
  batch.atom_offsets = host_input(atom_offsets, sizeof(atom_offsets));
  batch.atomic_numbers = host_input(atomic_numbers, sizeof(atomic_numbers));
  batch.positions = host_input(positions, sizeof(positions));
  batch.molecular_charges = host_input(molecular_charges, sizeof(molecular_charges));
  batch.unpaired_electrons = host_input(unpaired_electrons, sizeof(unpaired_electrons));

  compute_options.model = XTBLOOM_MODEL_GFN2_XTB;
  compute_options.flags =
      XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES | XTBLOOM_COMPUTE_ATOMIC_CHARGES;
  compute_options.scc_start_mode = XTBLOOM_SCC_START_FRESH;

  double energy = NAN;
  double forces[6] = {NAN, NAN, NAN, NAN, NAN, NAN};
  double charges[2] = {NAN, NAN};
  int32_t iterations = -1;
  uint8_t converged = 0;
  xtbloom_status_t system_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  result.energies = host_output(&energy, sizeof(energy));
  result.forces = host_output(forces, sizeof(forces));
  result.atomic_charges = host_output(charges, sizeof(charges));
  /* These diagnostics are required for every nonempty batch. */
  result.scc_iterations = host_output(&iterations, sizeof(iterations));
  result.scc_converged = host_output(&converged, sizeof(converged));
  result.per_system_status = host_output(&system_status, sizeof(system_status));

  xtbloom_context_t* context = NULL;
  xtbloom_status_t status = xtbloom_context_create(&context_options, &context);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    fprintf(stderr, "context creation failed: %s (%s)\n", xtbloom_status_string(status),
            xtbloom_get_last_error());
    return 4;
  }
  if (xtbloom_context_get_backend(context) != requested_backend) {
    fprintf(stderr, "resolved backend does not match the required backend\n");
    xtbloom_context_destroy(context);
    return 5;
  }

  /* All descriptors are borrowed and must remain alive through this
   * synchronous call. */
  status = xtbloom_compute(context, &batch, &compute_options, &result);
  if (!check_system("fresh", status, system_status, converged, iterations, energy)) {
    xtbloom_context_destroy(context);
    return 6;
  }

  /* WARM is strict. It succeeds here only because the previous call fully
   * converged and topology plus compute policy are unchanged. It never falls
   * back to FRESH after a first call, policy change, or failed predecessor. */
  compute_options.scc_start_mode = XTBLOOM_SCC_START_WARM;
  energy = NAN;
  iterations = -1;
  converged = 0;
  system_status = XTBLOOM_STATUS_INTERNAL_ERROR;
  status = xtbloom_compute(context, &batch, &compute_options, &result);
  if (!check_system("warm", status, system_status, converged, iterations, energy)) {
    xtbloom_context_destroy(context);
    return 7;
  }

  printf("xTBloom %s backend=%s energy=%.16g Eh iterations=%d\n", xtbloom_version_string(), argv[1],
         energy, (int)iterations);
  xtbloom_context_destroy(context);
  return 0;
}
