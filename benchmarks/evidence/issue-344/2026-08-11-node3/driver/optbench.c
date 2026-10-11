#define main xtbloom_web_embedded_main
#include "web/xtbloom_web.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
  if (argc != 9) {
    fprintf(stderr, "usage: %s xyz charge unpaired scc_max opt_max grad_tol warmups reps\n",
            argv[0]);
    return 2;
  }
  FILE* f = fopen(argv[1], "r");
  if (!f) return 2;
  char xyz[65536];
  size_t n = fread(xyz, 1, sizeof(xyz) - 1, f);
  fclose(f);
  xyz[n] = '\0';

  const double charge = atof(argv[2]);
  const int unpaired = atoi(argv[3]);
  const int scc_max = atoi(argv[4]);
  const int opt_max = atoi(argv[5]);
  const double grad_tol = atof(argv[6]);
  const int warmups = atoi(argv[7]);
  const int reps = atoi(argv[8]);

  for (int w = 0; w < warmups; ++w) {
    const char* s = xtbloom_web_optimize(xyz, charge, unpaired, 0.0, 1e-8, 1e-5, scc_max, opt_max,
                                         grad_tol, 0.4);
    if (strstr(s, "\"ok\":1") == NULL) {
      fprintf(stderr, "warmup failed: %s\n", s);
      return 1;
    }
  }
  double* samples = (double*)malloc((size_t)reps * sizeof(double));
  long long scc_total = 0;
  int warm_solves = -1, fresh_solves = -1, fallbacks = -1;
  double energy_final = 0.0;
  int iterations = 0, converged = 0;
  for (int r = 0; r < reps; ++r) {
    const double t0 = now_ms();
    const char* s = xtbloom_web_optimize(xyz, charge, unpaired, 0.0, 1e-8, 1e-5, scc_max, opt_max,
                                         grad_tol, 0.4);
    samples[r] = now_ms() - t0;
    /* A publishable sample set requires every timed repetition to succeed;
     * fail immediately so a later success cannot conceal an earlier error. */
    if (strstr(s, "\"ok\":1") == NULL) {
      fprintf(stderr, "timed repetition %d failed: %s\n", r, s);
      free(samples);
      return 1;
    }
#define FIELD(name)                                                                       \
  do {                                                                                    \
    const char* p = strstr(s, "\"" name "\":");                                           \
    if (p) p += strlen("\"" name "\":");                                                  \
    if (p) {                                                                              \
      char buf[64];                                                                       \
      const char* e = p;                                                                  \
      while (*e && *e != ',' && *e != '}') ++e;                                           \
      size_t len = (size_t)(e - p) < sizeof(buf) - 1 ? (size_t)(e - p) : sizeof(buf) - 1; \
      memcpy(buf, p, len);                                                                \
      buf[len] = '\0';                                                                    \
      if (strcmp(name, "scc_iterations_total") == 0)                                      \
        scc_total = atoll(buf);                                                           \
      else if (strcmp(name, "scc_warm_solves") == 0)                                      \
        warm_solves = atoi(buf);                                                          \
      else if (strcmp(name, "scc_fresh_solves") == 0)                                     \
        fresh_solves = atoi(buf);                                                         \
      else if (strcmp(name, "scc_warm_fallbacks") == 0)                                   \
        fallbacks = atoi(buf);                                                            \
      else if (strcmp(name, "energy_final_Eh") == 0)                                      \
        energy_final = atof(buf);                                                         \
      else if (strcmp(name, "iterations") == 0)                                           \
        iterations = atoi(buf);                                                           \
      else if (strcmp(name, "converged") == 0)                                            \
        converged = atoi(buf);                                                            \
    }                                                                                     \
  } while (0)
    FIELD("scc_iterations_total");
    FIELD("scc_warm_solves");
    FIELD("scc_fresh_solves");
    FIELD("scc_warm_fallbacks");
    FIELD("energy_final_Eh");
    FIELD("iterations");
    FIELD("converged");
#undef FIELD
  }
  printf(
      "{\"ok\":1,\"converged\":%d,\"iterations\":%d,\"scc_iterations_total\":%lld,"
      "\"scc_warm_solves\":%d,\"scc_fresh_solves\":%d,\"scc_warm_fallbacks\":%d,"
      "\"energy_final_Eh\":%.12g,\"samples_ms\":[",
      converged, iterations, scc_total, warm_solves, fresh_solves, fallbacks, energy_final);
  for (int r = 0; r < reps; ++r) {
    if (r) printf(",");
    printf("%.6f", samples[r]);
  }
  printf("]}\n");
  free(samples);
  return 0;
}
