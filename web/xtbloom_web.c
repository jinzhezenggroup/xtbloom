/*
 * xtbloom_web.c — browser-facing adapter compiled into the WebAssembly main module.
 *
 * It wraps the public xtbloom C ABI behind two tiny C entry points so the web
 * front end never has to marshal the ABI structs by hand:
 *
 *   const char* xtbloom_web_version(void);
 *   const char* xtbloom_web_compute(xyz, model, charge, unpaired, etemp_eh,
 *                                   etol, qtol, max_iter, forces);
 *
 * Input is plain XYZ ("Symbol x y z" or "Z x y z", one atom per line, values
 * in angstrom -- converted to bohr internally, since xtbloom positions are in
 * bohr) plus a molecular charge and unpaired-electron count.
 *
 * Output is a JSON document (static, reused buffer); callers must copy before
 * the next call. Units: energy in Hartree, forces in Hartree/bohr, charges in
 * elementary-charge units (xtbloom's public units). The front end converts to
 * eV / kcal/mol / eV per Angstrom for display.
 *
 * This adapter is deliberately small and single-molecule; it does not expose
 * plans, ragged batches, point charges, QM/MM, or result arenas.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xtbloom/xtbloom.h"

/* ------------------------------------------------------------------ */
/* JSON string buffer                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
  char* data;
  size_t len;
  size_t cap;
} StrBuf;

static void sb_grow(StrBuf* b, size_t need) {
  if (b->len + need + 1 <= b->cap) {
    return;
  }
  size_t ncap = b->cap ? b->cap : 256;
  while (ncap < b->len + need + 1) {
    ncap *= 2;
  }
  b->data = (char*)realloc(b->data, ncap);
  b->cap = ncap;
}

static void sb_putc(StrBuf* b, char c) {
  sb_grow(b, 1);
  b->data[b->len++] = c;
}

static void sb_puts(StrBuf* b, const char* s) {
  const size_t n = strlen(s);
  sb_grow(b, n);
  memcpy(b->data + b->len, s, n);
  b->len += n;
}

static void sb_puti(StrBuf* b, long long v) {
  char tmp[32];
  snprintf(tmp, sizeof(tmp), "%lld", v);
  sb_puts(b, tmp);
}

static void sb_putd(StrBuf* b, double v) {
  if (isnan(v)) {
    sb_puts(b, "null");
    return;
  }
  char tmp[64];
  snprintf(tmp, sizeof(tmp), "%.9g", v);
  sb_puts(b, tmp);
}

/* Append one JSON string, including quotes. Diagnostics can contain paths,
 * quotes, or newlines, so escaping only quote/backslash is not sufficient. */
static void sb_put_json_string(StrBuf* b, const char* value) {
  static const char hex[] = "0123456789abcdef";
  sb_putc(b, '"');
  for (const unsigned char* p = (const unsigned char*)(value != NULL ? value : ""); *p; ++p) {
    switch (*p) {
      case '"':
        sb_puts(b, "\\\"");
        break;
      case '\\':
        sb_puts(b, "\\\\");
        break;
      case '\b':
        sb_puts(b, "\\b");
        break;
      case '\f':
        sb_puts(b, "\\f");
        break;
      case '\n':
        sb_puts(b, "\\n");
        break;
      case '\r':
        sb_puts(b, "\\r");
        break;
      case '\t':
        sb_puts(b, "\\t");
        break;
      default:
        if (*p < 0x20) {
          sb_puts(b, "\\u00");
          sb_putc(b, hex[*p >> 4]);
          sb_putc(b, hex[*p & 0x0f]);
        } else {
          sb_putc(b, (char)*p);
        }
        break;
    }
  }
  sb_putc(b, '"');
}

static char* sb_finish(StrBuf* b) {
  sb_grow(b, 1);
  b->data[b->len] = '\0';
  return b->data;
}

/* ------------------------------------------------------------------ */
/* Element symbol -> atomic number                                     */
/* ------------------------------------------------------------------ */

#define MAX_ELEMENT 103

static const char* const kElementSymbols[MAX_ELEMENT + 1] = {
    "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
    "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};

static int symbol_to_z(const char* sym) {
  /* Case-insensitive. Element symbols are title case ("Cl", "Na"); the second
   * letter must be lowered, not uppercased, or two-letter symbols never match
   * the reference table. Elements 1..103 are all one or two letters. */
  if (sym == NULL || sym[0] == '\0') {
    return 0;
  }
  char a = sym[0];
  char b = sym[1];
  if (a >= 'a' && a <= 'z') {
    a = (char)(a - 'a' + 'A');
  }
  if (b == '\0') {
    b = '\0';
  } else if (b >= 'A' && b <= 'Z') {
    b = (char)(b - 'A' + 'a');
  }
  /* Ignore anything beyond the first two characters. */
  char s[3];
  s[0] = a;
  s[1] = b;
  s[2] = '\0';
  for (int z = 1; z <= MAX_ELEMENT; ++z) {
    if (strcmp(kElementSymbols[z], s) == 0) {
      return z;
    }
  }
  return 0;
}

/* One atom of a parsed XYZ block. */
typedef struct {
  int32_t z;
  double px, py, pz;
} Atom;

/* ------------------------------------------------------------------ */
/* XYZ parsing                                                         */
/* ------------------------------------------------------------------ */

/* Returns the number of atoms parsed, or a negative error code:
 * -1 no atoms, -2 malformed line, -3 unknown element, -4 coordinate parse failure */
static int parse_xyz(const char* xyz, Atom* atoms, int max_atoms) {
  int count = 0;
  const char* p = xyz;
  while (*p) {
    while (*p == '\n' || *p == '\r') {
      ++p;
    }
    if (*p == '\0') {
      break;
    }
    /* element token */
    char sym[8];
    int i = 0;
    while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
      if (i < 7) {
        sym[i++] = *p;
      }
      ++p;
    }
    sym[i] = '\0';
    const char* nums[3] = {NULL, NULL, NULL};
    for (int c = 0; c < 3; ++c) {
      while (*p == ' ' || *p == '\t') {
        ++p;
      }
      if (*p == '\n' || *p == '\r' || *p == '\0') {
        return -2;
      }
      nums[c] = p;
      while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
        ++p;
      }
    }
    while (*p && *p != '\n' && *p != '\r') {
      ++p;
    }
    int32_t z = 0;
    {
      char* end = NULL;
      long v = strtol(sym, &end, 10);
      if (end != sym && *end == '\0' && v >= 1 && v <= MAX_ELEMENT) {
        z = (int32_t)v;
      } else {
        z = symbol_to_z(sym);
        if (z == 0) {
          return -3;
        }
      }
    }
    if (count >= max_atoms) {
      return -1;
    }
    char tmp[64];
    for (int c = 0; c < 3; ++c) {
      /* copy coordinate token then parse */
      size_t len = 0;
      const char* q = nums[c];
      while (*q && *q != ' ' && *q != '\t' && *q != '\n' && *q != '\r' && len < 63) {
        tmp[len++] = *q++;
      }
      tmp[len] = '\0';
      char* end2 = NULL;
      double val = strtod(tmp, &end2);
      if (end2 == tmp) {
        return -4;
      }
      if (c == 0) {
        atoms[count].px = val;
      } else if (c == 1) {
        atoms[count].py = val;
      } else {
        atoms[count].pz = val;
      }
    }
    atoms[count].z = z;
    ++count;
  }
  return count;
}

/* ------------------------------------------------------------------ */
/* Compute + JSON result                                               */
/* ------------------------------------------------------------------ */

static StrBuf g_result;
static xtbloom_context_t* g_context = NULL;

/* The Web protocol carries the stable public model tag explicitly so the
 * adapter never relies on an initializer default or silently substitutes one
 * GFN-xTB method for another. */
static int valid_model_tag(int model) {
  return model == XTBLOOM_MODEL_GFN1_XTB || model == XTBLOOM_MODEL_GFN2_XTB;
}

static const char* model_name(int model) {
  return model == XTBLOOM_MODEL_GFN1_XTB ? "GFN1-xTB" : "GFN2-xTB";
}

/* Whether the adapter's most recent compute left a fully converged
 * compatible state on the shared context that a strict WARM SCC request may
 * consume. Mirrors the native gate so the browser optimizer can reuse the
 * previous converged electronic state as the next step's SCC guess. Geometry
 * is not part of the native warm identity, which is exactly what makes
 * successive geometry-optimization steps ideal warm-start candidates.
 *
 * Standalone single-point computes always run FRESH and consume the prior
 * checkpoint (native semantics), so a user calculation can never inherit an
 * unrelated optimization's electronic state. The first step of every new
 * optimization run is also forced FRESH, resetting the previous run's state. */
static int g_warm_ready = 0;

/* Run one native compute with the adapter's automatic warm-start policy.
 *
 * allow_warm selects whether this call may consume the retained checkpoint.
 * When a WARM request is refused (no compatible fully converged predecessor:
 * first call, superseded or consumed checkpoint, or changed charge/spin/
 * topology/compute policy), the strict native gate rejects it with
 * INVALID_ARGUMENT before modifying any caller output, so one independent
 * FRESH retry is safe and transparent.
 *
 * When stats is non-NULL the helper records, per SCC solve, the requested
 * start mode and the reported SCC iteration count so the browser optimizer can
 * report warm-start effectiveness (see WebSccStats below). */
typedef struct {
  int fresh_solves;     /* SCC solves started from the immutable fresh state */
  int warm_solves;      /* SCC solves that consumed a compatible checkpoint */
  int warm_fallbacks;   /* WARM requests rejected, retried transparently FRESH */
  long long iterations; /* total SCC iterations across every solve in the run */
} WebSccStats;

static xtbloom_status_t compute_adaptive(xtbloom_context_t* context, const xtbloom_batch_t* batch,
                                         xtbloom_compute_options_t* options,
                                         xtbloom_batch_result_t* result, int allow_warm,
                                         WebSccStats* stats) {
  const int want_warm = allow_warm != 0 && g_warm_ready != 0;
  options->scc_start_mode = want_warm ? XTBLOOM_SCC_START_WARM : XTBLOOM_SCC_START_FRESH;
  xtbloom_status_t status = xtbloom_compute(context, batch, options, result);
  int used_warm = want_warm;
  if (status == XTBLOOM_STATUS_INVALID_ARGUMENT && want_warm) {
    used_warm = 0;
    options->scc_start_mode = XTBLOOM_SCC_START_FRESH;
    status = xtbloom_compute(context, batch, options, result);
  }
  if (stats != NULL) {
    if (used_warm) {
      ++stats->warm_solves;
    } else {
      ++stats->fresh_solves;
      if (want_warm) {
        ++stats->warm_fallbacks;
      }
    }
    const int32_t* iterations = (const int32_t*)result->scc_iterations.data;
    if (iterations != NULL) {
      stats->iterations += (long long)iterations[0];
    }
  }
  return status;
}

/* Front-end-localizable error: stable ASCII code (+ optional raw diagnostic). */
static const char* error_json(const char* code, const char* raw) {
  g_result.len = 0;
  sb_puts(&g_result, "{\"ok\":0,\"error_code\":");
  sb_put_json_string(&g_result, code != NULL ? code : "err_unknown");
  if (raw != NULL && *raw) {
    sb_puts(&g_result, ",\"error\":");
    sb_put_json_string(&g_result, raw);
  }
  sb_putc(&g_result, '}');
  return sb_finish(&g_result);
}

const char* xtbloom_web_version(void) { return xtbloom_version_string(); }

/* Parse/compute and return a JSON document in a static reused buffer. */
const char* xtbloom_web_compute(const char* xyz, int model, double charge, int unpaired,
                                double electronic_temperature_eh, double energy_tolerance,
                                double charge_tolerance, int max_iterations, int compute_forces) {
  if (!valid_model_tag(model)) {
    return error_json("err_model", NULL);
  }
  /* --- parse geometry --- */
  Atom atoms[512];
  const int n_atoms = parse_xyz(xyz, atoms, 512);
  if (n_atoms <= 0) {
    return error_json("err_xyz_parse", NULL);
  }
  if (n_atoms > 512) {
    return error_json("err_xyz_too_many", NULL);
  }

  /* --- ensure context --- */
  if (g_context == NULL) {
    xtbloom_context_options_t ctx_opts;
    if (xtbloom_context_options_init(&ctx_opts, sizeof(ctx_opts)) != XTBLOOM_STATUS_SUCCESS) {
      return error_json("err_init", NULL);
    }
    ctx_opts.backend = XTBLOOM_BACKEND_CPU;
    /* Single-threaded wasm build: run on the calling thread. */
    ctx_opts.cpu_threads = 1;
    const xtbloom_status_t st = xtbloom_context_create(&ctx_opts, &g_context);
    if (st != XTBLOOM_STATUS_SUCCESS || g_context == NULL) {
      const char* e = xtbloom_get_last_error();
      return error_json("err_ctx", e != NULL && *e ? e : "context create failed");
    }
  }

  /* --- stage arrays --- */
  int64_t* atom_offsets = (int64_t*)malloc(2 * sizeof(int64_t));
  int32_t* atomic_numbers = (int32_t*)malloc((size_t)n_atoms * sizeof(int32_t));
  double* positions = (double*)malloc((size_t)n_atoms * 3 * sizeof(double));
  double* molecular_charges = (double*)malloc(1 * sizeof(double));
  int32_t* unpaired_electrons = (int32_t*)malloc(1 * sizeof(int32_t));
  double* energies = (double*)malloc(1 * sizeof(double));
  double* atomic_charges = (double*)malloc((size_t)n_atoms * sizeof(double));
  double* forces =
      compute_forces != 0 ? (double*)malloc((size_t)n_atoms * 3 * sizeof(double)) : NULL;
  int32_t* scc_iterations = (int32_t*)malloc(sizeof(int32_t));
  uint8_t* scc_converged = (uint8_t*)malloc(sizeof(uint8_t));
  int32_t* per_system_status = (int32_t*)malloc(sizeof(int32_t));
  if (atom_offsets == NULL || atomic_numbers == NULL || positions == NULL ||
      molecular_charges == NULL || unpaired_electrons == NULL || energies == NULL ||
      atomic_charges == NULL || (compute_forces != 0 && forces == NULL) || scc_iterations == NULL ||
      scc_converged == NULL || per_system_status == NULL) {
    free(atom_offsets);
    free(atomic_numbers);
    free(positions);
    free(molecular_charges);
    free(unpaired_electrons);
    free(energies);
    free(atomic_charges);
    free(forces);
    free(scc_iterations);
    free(scc_converged);
    free(per_system_status);
    return error_json("err_alloc", NULL);
  }
  atom_offsets[0] = 0;
  atom_offsets[1] = n_atoms;
  for (int i = 0; i < n_atoms; ++i) {
    atomic_numbers[i] = atoms[i].z;
    positions[i * 3 + 0] = atoms[i].px * 1.8897261254578281; /* angstrom -> bohr */
    positions[i * 3 + 1] = atoms[i].py * 1.8897261254578281;
    positions[i * 3 + 2] = atoms[i].pz * 1.8897261254578281;
  }
  molecular_charges[0] = charge;
  unpaired_electrons[0] = unpaired;

  /* --- batch --- */
  xtbloom_batch_t batch;
  if (xtbloom_batch_init(&batch, sizeof(batch)) != XTBLOOM_STATUS_SUCCESS) {
    return error_json("err_init", NULL);
  }
  batch.batch_size = 1;
  batch.total_atoms = n_atoms;
  batch.atom_offsets.data = atom_offsets;
  batch.atom_offsets.size_bytes = 2 * sizeof(int64_t);
  batch.atomic_numbers.data = atomic_numbers;
  batch.atomic_numbers.size_bytes = (size_t)n_atoms * sizeof(int32_t);
  batch.positions.data = positions;
  batch.positions.size_bytes = (size_t)n_atoms * 3 * sizeof(double);
  batch.molecular_charges.data = molecular_charges;
  batch.molecular_charges.size_bytes = sizeof(double);
  batch.unpaired_electrons.data = unpaired_electrons;
  batch.unpaired_electrons.size_bytes = sizeof(int32_t);

  /* --- options --- */
  xtbloom_compute_options_t options;
  if (xtbloom_compute_options_init(&options, sizeof(options)) != XTBLOOM_STATUS_SUCCESS) {
    return error_json("err_init", NULL);
  }
  options.model = (xtbloom_model_t)model;
  options.flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_ATOMIC_CHARGES;
  if (compute_forces != 0) {
    options.flags |= XTBLOOM_COMPUTE_FORCES;
  }
  if (max_iterations > 0) {
    options.max_scc_iterations = max_iterations;
  }
  if (electronic_temperature_eh > 0.0) {
    options.electronic_temperature = electronic_temperature_eh;
  }
  if (energy_tolerance > 0.0) {
    options.energy_tolerance = energy_tolerance;
  }
  if (charge_tolerance > 0.0) {
    options.charge_tolerance = charge_tolerance;
  }

  /* --- result --- */
  xtbloom_batch_result_t result;
  if (xtbloom_batch_result_init(&result, sizeof(result)) != XTBLOOM_STATUS_SUCCESS) {
    return error_json("err_init", NULL);
  }
  memset(energies, 0, sizeof(double));
  memset(atomic_charges, 0, (size_t)n_atoms * sizeof(double));
  if (compute_forces != 0) {
    memset(forces, 0, (size_t)n_atoms * 3 * sizeof(double));
  }
  result.energies.data = energies;
  result.energies.size_bytes = sizeof(double);
  result.atomic_charges.data = atomic_charges;
  result.atomic_charges.size_bytes = (size_t)n_atoms * sizeof(double);
  if (compute_forces != 0) {
    result.forces.data = forces;
    result.forces.size_bytes = (size_t)n_atoms * 3 * sizeof(double);
  }
  result.scc_iterations.data = scc_iterations;
  result.scc_iterations.size_bytes = sizeof(int32_t);
  result.scc_converged.data = scc_converged;
  result.scc_converged.size_bytes = sizeof(uint8_t);
  result.per_system_status.data = per_system_status;
  result.per_system_status.size_bytes = sizeof(int32_t);

  /* Standalone single-point evaluations always start SCC fresh. The accepted
   * FRESH attempt consumes any preceding checkpoint, so ordinary browser
   * calculations never reuse electronic state from an unrelated request. */
  const xtbloom_status_t status = compute_adaptive(g_context, &batch, &options, &result, 0, NULL);
  g_warm_ready = (status == XTBLOOM_STATUS_SUCCESS &&
                  per_system_status[0] == XTBLOOM_STATUS_SUCCESS && scc_converged[0] == 1u);

  /* --- serialize --- */
  g_result.len = 0;
  if (status != XTBLOOM_STATUS_SUCCESS || per_system_status[0] != XTBLOOM_STATUS_SUCCESS) {
    const char* e = status != XTBLOOM_STATUS_SUCCESS ? xtbloom_get_last_error()
                                                     : xtbloom_status_string(per_system_status[0]);
    error_json("err_compute", e != NULL && *e ? e : NULL);
    free(atom_offsets);
    free(atomic_numbers);
    free(positions);
    free(molecular_charges);
    free(unpaired_electrons);
    free(energies);
    free(atomic_charges);
    free(forces);
    free(scc_iterations);
    free(scc_converged);
    free(per_system_status);
    return sb_finish(&g_result);
  }

  sb_puts(&g_result, "{\"ok\":1,\"model\":");
  sb_puti(&g_result, model);
  sb_puts(&g_result, ",\"method\":");
  sb_put_json_string(&g_result, model_name(model));
  sb_puts(&g_result, ",\"energy_Eh\":");
  sb_putd(&g_result, energies[0]);
  sb_puts(&g_result, ",\"scc_iterations\":");
  sb_puti(&g_result, scc_iterations[0]);
  sb_puts(&g_result, ",\"scc_converged\":");
  sb_puti(&g_result, (long long)scc_converged[0]);
  sb_puts(&g_result, ",\"per_system_status\":");
  sb_puti(&g_result, per_system_status[0]);
  sb_puts(&g_result, ",\"charges\":[");
  for (int i = 0; i < n_atoms; ++i) {
    if (i) {
      sb_putc(&g_result, ',');
    }
    sb_puts(&g_result, "{\"element\":");
    sb_puti(&g_result, atoms[i].z);
    sb_puts(&g_result, ",\"q\":");
    sb_putd(&g_result, atomic_charges[i]);
    sb_putc(&g_result, '}');
  }
  sb_putc(&g_result, ']');
  if (compute_forces != 0) {
    sb_puts(&g_result, ",\"forces\":[");
    for (int i = 0; i < n_atoms; ++i) {
      if (i) {
        sb_putc(&g_result, ',');
      }
      sb_puts(&g_result, "{\"element\":");
      sb_puti(&g_result, atoms[i].z);
      sb_puts(&g_result, ",\"fx_eh_bohr\":");
      sb_putd(&g_result, forces[i * 3 + 0]);
      sb_puts(&g_result, ",\"fy_eh_bohr\":");
      sb_putd(&g_result, forces[i * 3 + 1]);
      sb_puts(&g_result, ",\"fz_eh_bohr\":");
      sb_putd(&g_result, forces[i * 3 + 2]);
      sb_putc(&g_result, '}');
    }
    sb_putc(&g_result, ']');
  }
  sb_putc(&g_result, '}');

  free(atom_offsets);
  free(atomic_numbers);
  free(positions);
  free(molecular_charges);
  free(unpaired_electrons);
  free(energies);
  free(atomic_charges);
  free(forces);
  free(scc_iterations);
  free(scc_converged);
  free(per_system_status);
  return sb_finish(&g_result);
}

/* ------------------------------------------------------------------ */
/* Minimal L-BFGS geometry optimizer (demo quality)                    */
/*                                                                     */
/* Minimizes the selected GFN-xTB energy by moving coordinates (bohr) */
/* along an L-BFGS search direction with an Armijo backtracking line   */
/* search. Each trial reads the analytic energy and force from         */
/* xtbloom_compute on a cached context, so convergence is driven by the */
/* same physics as the single-point path. It is intentionally simple:  */
/* no Hessian, no constraints, no symmetry detection.                  */
/* ------------------------------------------------------------------ */

#define BOHR_PER_ANGSTROM 1.8897261254578281
#define LBFGS_M 8

static double w_dot(const double* a, const double* b, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

static double w_maxabs(const double* a, int n) {
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    const double v = fabs(a[i]);
    if (v > m) {
      m = v;
    }
  }
  return m;
}

static void w_copy(double* dst, const double* src, int n) {
  memcpy(dst, src, (size_t)n * sizeof(double));
}

static void w_axpy(double* y, double a, const double* x, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] += a * x[i];
  }
}

/* Restore the mathematical invariant required by Armijo: p must be a
 * downhill direction. A rejected quasi-Newton direction falls back to -g. */
static void w_ensure_descent(const double* g, double* p, int n) {
  if (w_dot(g, p, n) >= 0.0) {
    for (int i = 0; i < n; ++i) {
      p[i] = -g[i];
    }
  }
}

/* L-BFGS two-loop coefficients use reciprocal curvature. A non-positive
 * curvature pair is ignored instead of fabricating a positive coefficient. */
static double w_reciprocal_curvature(const double* s, const double* y, int n) {
  const double sy = w_dot(s, y, n);
  return sy > 0.0 ? 1.0 / sy : 0.0;
}

/* Optional per-iteration callback that lets the front end animate the
 * geometry during optimization. Called on the worker thread after every
 * accepted L-BFGS step, with the running iteration count (1-based), the atom
 * count, coordinates in angstrom, the energy (Eh), and the largest per-atom
 * force (Eh/bohr). Reset to NULL to disable. */
typedef void (*XTBloomOptimizeStepFn)(int iteration, int atom_count, const double* coords_angstrom,
                                      double energy_eh, double force_max_eh_per_bohr);
static XTBloomOptimizeStepFn g_optimize_step_fn = NULL;

void xtbloom_web_set_optimize_step_cb(XTBloomOptimizeStepFn fn) { g_optimize_step_fn = fn; }

const char* xtbloom_web_optimize(const char* xyz, int model, double charge, int unpaired,
                                 double electronic_temperature_eh, double energy_tolerance,
                                 double charge_tolerance, int scc_max_iterations,
                                 int opt_max_iterations, double grad_tol, double max_move) {
  if (!valid_model_tag(model)) {
    return error_json("err_model", NULL);
  }
  Atom atoms[512];
  const int n_atoms = parse_xyz(xyz, atoms, 512);
  if (n_atoms <= 0) {
    return error_json("err_xyz_parse", NULL);
  }
  if (n_atoms > 512) {
    return error_json("err_xyz_too_many", NULL);
  }
  if (opt_max_iterations <= 0) {
    opt_max_iterations = 200;
  }
  if (grad_tol <= 0.0) {
    grad_tol = 4.5e-4; /* Eh/bohr, per-atom max force */
  }
  if (max_move <= 0.0) {
    max_move = 0.4; /* bohr, per-step displacement clamp */
  }

  /* cached context (shared with xtbloom_web_compute) */
  if (g_context == NULL) {
    xtbloom_context_options_t ctx_opts;
    if (xtbloom_context_options_init(&ctx_opts, sizeof(ctx_opts)) != XTBLOOM_STATUS_SUCCESS) {
      return error_json("err_init", NULL);
    }
    ctx_opts.backend = XTBLOOM_BACKEND_CPU;
    ctx_opts.cpu_threads = 1;
    const xtbloom_status_t st = xtbloom_context_create(&ctx_opts, &g_context);
    if (st != XTBLOOM_STATUS_SUCCESS || g_context == NULL) {
      const char* e = xtbloom_get_last_error();
      return error_json("err_ctx", e != NULL && *e ? e : "context create failed");
    }
  }

  const int dim = 3 * n_atoms;
  const int traj_cap = opt_max_iterations + 1;

  /* --- caller-owned staging (reused across L-BFGS iterations) --- */
  int64_t* atom_offsets = (int64_t*)malloc(2 * sizeof(int64_t));
  int32_t* atomic_numbers = (int32_t*)malloc((size_t)n_atoms * sizeof(int32_t));
  double* positions = (double*)malloc((size_t)dim * sizeof(double));
  double* molecular_charges = (double*)malloc(1 * sizeof(double));
  int32_t* unpaired_electrons = (int32_t*)malloc(1 * sizeof(int32_t));
  double* energy = (double*)malloc(1 * sizeof(double));
  double* charges_q = (double*)malloc((size_t)n_atoms * sizeof(double));
  double* forces = (double*)malloc((size_t)dim * sizeof(double));
  int32_t* scc_iterations = (int32_t*)malloc(sizeof(int32_t));
  uint8_t* scc_converged = (uint8_t*)malloc(sizeof(uint8_t));
  int32_t* per_system_status = (int32_t*)malloc(sizeof(int32_t));
  double* x = (double*)malloc((size_t)dim * sizeof(double));
  double* g = (double*)malloc((size_t)dim * sizeof(double));
  double* g2 = (double*)malloc((size_t)dim * sizeof(double));
  double* x2 = (double*)malloc((size_t)dim * sizeof(double));
  double* p = (double*)malloc((size_t)dim * sizeof(double));
  double* q = (double*)malloc((size_t)dim * sizeof(double));
  double* r = (double*)malloc((size_t)dim * sizeof(double));
  double* s_hist = (double*)malloc(LBFGS_M * (size_t)dim * sizeof(double));
  double* y_hist = (double*)malloc(LBFGS_M * (size_t)dim * sizeof(double));
  double* rho = (double*)malloc(LBFGS_M * sizeof(double));
  double* alpha = (double*)malloc(LBFGS_M * sizeof(double));
  double* trajectory = (double*)malloc((size_t)(traj_cap) * sizeof(double));
  double* step_buf = (double*)malloc((size_t)dim * sizeof(double));
  int32_t* scc_per_step = (int32_t*)malloc((size_t)(traj_cap) * sizeof(int32_t));
  if (atom_offsets == NULL || atomic_numbers == NULL || positions == NULL ||
      molecular_charges == NULL || unpaired_electrons == NULL || energy == NULL ||
      charges_q == NULL || forces == NULL || scc_iterations == NULL || scc_converged == NULL ||
      per_system_status == NULL || x == NULL || g == NULL || g2 == NULL || x2 == NULL ||
      p == NULL || q == NULL || r == NULL || s_hist == NULL || y_hist == NULL || rho == NULL ||
      alpha == NULL || trajectory == NULL || step_buf == NULL || scc_per_step == NULL) {
    free(atom_offsets);
    free(atomic_numbers);
    free(positions);
    free(molecular_charges);
    free(unpaired_electrons);
    free(energy);
    free(charges_q);
    free(forces);
    free(scc_iterations);
    free(scc_converged);
    free(per_system_status);
    free(x);
    free(g);
    free(g2);
    free(x2);
    free(p);
    free(q);
    free(r);
    free(s_hist);
    free(y_hist);
    free(rho);
    free(alpha);
    free(trajectory);
    free(step_buf);
    free(scc_per_step);
    return error_json("err_alloc", NULL);
  }

  atom_offsets[0] = 0;
  atom_offsets[1] = n_atoms;
  for (int i = 0; i < n_atoms; ++i) {
    atomic_numbers[i] = atoms[i].z;
    x[i * 3 + 0] = atoms[i].px * BOHR_PER_ANGSTROM;
    x[i * 3 + 1] = atoms[i].py * BOHR_PER_ANGSTROM;
    x[i * 3 + 2] = atoms[i].pz * BOHR_PER_ANGSTROM;
  }
  molecular_charges[0] = charge;
  unpaired_electrons[0] = unpaired;

  /* batch + result bound once to the staging buffers */
  xtbloom_batch_t batch;
  xtbloom_batch_init(&batch, sizeof(batch));
  batch.batch_size = 1;
  batch.total_atoms = n_atoms;
  batch.atom_offsets.data = atom_offsets;
  batch.atom_offsets.size_bytes = 2 * sizeof(int64_t);
  batch.atomic_numbers.data = atomic_numbers;
  batch.atomic_numbers.size_bytes = (size_t)n_atoms * sizeof(int32_t);
  batch.positions.data = positions;
  batch.positions.size_bytes = (size_t)dim * sizeof(double);
  batch.molecular_charges.data = molecular_charges;
  batch.molecular_charges.size_bytes = sizeof(double);
  batch.unpaired_electrons.data = unpaired_electrons;
  batch.unpaired_electrons.size_bytes = sizeof(int32_t);

  xtbloom_compute_options_t options;
  xtbloom_compute_options_init(&options, sizeof(options));
  options.model = (xtbloom_model_t)model;
  options.flags = XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_ATOMIC_CHARGES | XTBLOOM_COMPUTE_FORCES;
  if (scc_max_iterations > 0) {
    options.max_scc_iterations = scc_max_iterations;
  }
  if (electronic_temperature_eh > 0.0) {
    options.electronic_temperature = electronic_temperature_eh;
  }
  if (energy_tolerance > 0.0) {
    options.energy_tolerance = energy_tolerance;
  }
  if (charge_tolerance > 0.0) {
    options.charge_tolerance = charge_tolerance;
  }

  xtbloom_batch_result_t result;
  xtbloom_batch_result_init(&result, sizeof(result));
  result.energies.data = energy;
  result.energies.size_bytes = sizeof(double);
  result.forces.data = forces;
  result.forces.size_bytes = (size_t)dim * sizeof(double);
  result.atomic_charges.data = charges_q;
  result.atomic_charges.size_bytes = (size_t)n_atoms * sizeof(double);
  result.scc_iterations.data = scc_iterations;
  result.scc_iterations.size_bytes = sizeof(int32_t);
  result.scc_converged.data = scc_converged;
  result.scc_converged.size_bytes = sizeof(uint8_t);
  result.per_system_status.data = per_system_status;
  result.per_system_status.size_bytes = sizeof(int32_t);

  double f = NAN;
  int converged = 0;
  int steps = 0;
  const char* fail_code = NULL;
  const char* fail_reason = NULL;
  WebSccStats scc_stats;
  memset(&scc_stats, 0, sizeof(scc_stats));

  /* The first evaluation of an optimization run always starts SCC fresh.
   * The accepted FRESH attempt consumes any checkpoint retained by a previous
   * run, so a new optimization never inherits an older run's electronic state
   * and the first step cannot reuse a stale warm guess. */
  g_warm_ready = 0;
  w_copy(positions, x, dim);
  const xtbloom_status_t initial_status =
      compute_adaptive(g_context, &batch, &options, &result, 0, &scc_stats);
  if (initial_status != XTBLOOM_STATUS_SUCCESS || *per_system_status != XTBLOOM_STATUS_SUCCESS) {
    g_warm_ready = 0;
    fail_code = "err_initial_calc";
    fail_reason = initial_status != XTBLOOM_STATUS_SUCCESS
                      ? xtbloom_get_last_error()
                      : xtbloom_status_string(*per_system_status);
    goto done;
  }
  g_warm_ready = (initial_status == XTBLOOM_STATUS_SUCCESS &&
                  *per_system_status == XTBLOOM_STATUS_SUCCESS && *scc_converged == 1u);
  f = energy[0];
  for (int i = 0; i < dim; ++i) {
    g[i] = -forces[i]; /* gradient = -force, Eh/bohr */
  }
  if (!isfinite(f)) {
    fail_code = "err_nan_initial";
    goto done;
  }
  trajectory[0] = f;
  scc_per_step[0] = *scc_iterations;

  for (steps = 0; steps < opt_max_iterations; ++steps) {
    if (w_maxabs(g, dim) < grad_tol) {
      converged = 1;
      break;
    }

    /* --- L-BFGS two-loop recursion: p = -H*g --- */
    const int mem = steps < LBFGS_M ? steps : LBFGS_M;
    w_copy(q, g, dim);
    /* first loop: newest to oldest pair; alpha[d] kept per-pair for the
     * second, oldest-to-newest pass (standard two-loop recursion). */
    for (int d = 0; d < mem; ++d) {
      const int k = (steps - 1 - d + 2 * LBFGS_M) % LBFGS_M;
      alpha[d] = rho[k] * w_dot(s_hist + (size_t)k * dim, q, dim);
      w_axpy(q, -alpha[d], y_hist + (size_t)k * dim, dim);
    }
    {
      double gamma = 1.0;
      if (steps > 0) {
        const int k = (steps - 1 + LBFGS_M) % LBFGS_M;
        const double sy = w_dot(s_hist + (size_t)k * dim, y_hist + (size_t)k * dim, dim);
        const double yy = w_dot(y_hist + (size_t)k * dim, y_hist + (size_t)k * dim, dim);
        if (yy > 0.0 && sy > 0.0) {
          gamma = sy / yy;
        }
      }
      for (int j = 0; j < dim; ++j) {
        r[j] = gamma * q[j];
      }
    }
    for (int d = mem - 1; d >= 0; --d) {
      const int k = (steps - 1 - d + 2 * LBFGS_M) % LBFGS_M;
      const double beta = rho[k] * w_dot(y_hist + (size_t)k * dim, r, dim);
      w_axpy(r, alpha[d] - beta, s_hist + (size_t)k * dim, dim);
    }
    for (int j = 0; j < dim; ++j) {
      p[j] = -r[j];
    }

    /* safety: must descend; clamp per-step displacement */
    w_ensure_descent(g, p, dim);
    const double pmax = w_maxabs(p, dim);
    if (pmax > max_move) {
      const double scl = max_move / pmax;
      for (int j = 0; j < dim; ++j) {
        p[j] *= scl;
      }
    }
    const double gpg = w_dot(g, p, dim);
    const double c1 = 1e-4;

    /* --- Armijo backtracking line search --- */
    double step = 1.0;
    int accepted = 0;
    for (int tries = 0; tries < 30; ++tries) {
      for (int j = 0; j < dim; ++j) {
        x2[j] = x[j] + step * p[j];
        positions[j] = x2[j];
      }
      const xtbloom_status_t step_status =
          compute_adaptive(g_context, &batch, &options, &result, 1, &scc_stats);
      if (step_status != XTBLOOM_STATUS_SUCCESS || *per_system_status != XTBLOOM_STATUS_SUCCESS) {
        g_warm_ready = 0;
        fail_code = "err_step_sp";
        fail_reason = step_status != XTBLOOM_STATUS_SUCCESS
                          ? xtbloom_get_last_error()
                          : xtbloom_status_string(*per_system_status);
        if (fail_reason == NULL || *fail_reason == '\0') {
          fail_reason = "err_step_sp";
        }
        goto done;
      }
      g_warm_ready = (step_status == XTBLOOM_STATUS_SUCCESS &&
                      *per_system_status == XTBLOOM_STATUS_SUCCESS && *scc_converged == 1u);
      const double f2 = energy[0];
      if (isfinite(f2) && f2 <= f + c1 * step * gpg) {
        for (int j = 0; j < dim; ++j) {
          g2[j] = -forces[j];
        }
        accepted = 1;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) {
      fail_code = "err_linesearch";
      goto done;
    }

    /* --- accept step, store L-BFGS pair (circular) --- */
    {
      const int k = steps % LBFGS_M;
      for (int j = 0; j < dim; ++j) {
        s_hist[(size_t)k * dim + j] = x2[j] - x[j];
        y_hist[(size_t)k * dim + j] = g2[j] - g[j];
      }
      rho[k] = w_reciprocal_curvature(s_hist + (size_t)k * dim, y_hist + (size_t)k * dim, dim);
    }
    w_copy(x, x2, dim);
    w_copy(g, g2, dim);
    f = energy[0];
    if (!isfinite(f)) {
      fail_code = "err_nan_step";
      goto done;
    }
    trajectory[steps + 1] = f;
    /* SCC iteration count for the accepted point (the current solve
     * published energy/scc_iterations, which is the accepted line-search
     * trial's solve). Reported alongside the trajectory to make warm-start
     * effectiveness observable. */
    scc_per_step[steps + 1] = *scc_iterations;
    if (g_optimize_step_fn != NULL) {
      for (int j = 0; j < dim; ++j) {
        step_buf[j] = x[j] / BOHR_PER_ANGSTROM;
      }
      g_optimize_step_fn(steps + 1, n_atoms, step_buf, f, w_maxabs(g, dim));
    }
  }
  if (steps >= opt_max_iterations && w_maxabs(g, dim) < grad_tol) {
    converged = 1;
  }

done:
  /* --- serialize --- */
  if (fail_code != NULL) {
    /* A failed run must not leave a consumable warm state behind: the next
     * calculation (fresh single point or a new optimization) never inherits
     * the interrupted run's electronic state. */
    g_warm_ready = 0;
    error_json(fail_code, fail_reason != NULL && *fail_reason ? fail_reason : NULL);
  } else {
    g_result.len = 0;
    sb_puts(&g_result, "{\"ok\":1,\"model\":");
    sb_puti(&g_result, model);
    sb_puts(&g_result, ",\"method\":");
    sb_put_json_string(&g_result, model_name(model));
    sb_puts(&g_result, ",\"converged\":");
    sb_puti(&g_result, converged);
    sb_puts(&g_result, ",\"iterations\":");
    sb_puti(&g_result, steps);
    sb_puts(&g_result, ",\"energy_initial_Eh\":");
    sb_putd(&g_result, trajectory[0]);
    sb_puts(&g_result, ",\"energy_final_Eh\":");
    sb_putd(&g_result, f);
    sb_puts(&g_result, ",\"force_max_final_Eh_bohr\":");
    sb_putd(&g_result, w_maxabs(g, dim));
    sb_puts(&g_result, ",\"trajectory\":[");
    for (int i = 0; i <= steps; ++i) {
      if (i) {
        sb_putc(&g_result, ',');
      }
      sb_putd(&g_result, trajectory[i]);
    }
    sb_puts(&g_result, "],\"scc_iterations\":[");
    for (int i = 0; i <= steps; ++i) {
      if (i) {
        sb_putc(&g_result, ',');
      }
      sb_puti(&g_result, scc_per_step[i]);
    }
    sb_puts(&g_result, "],\"scc_iterations_total\":");
    sb_puti(&g_result, scc_stats.iterations);
    sb_puts(&g_result, ",\"scc_fresh_solves\":");
    sb_puti(&g_result, scc_stats.fresh_solves);
    sb_puts(&g_result, ",\"scc_warm_solves\":");
    sb_puti(&g_result, scc_stats.warm_solves);
    sb_puts(&g_result, ",\"scc_warm_fallbacks\":");
    sb_puti(&g_result, scc_stats.warm_fallbacks);
    sb_puts(&g_result, ",\"geometry\":\"");
    for (int i = 0; i < n_atoms; ++i) {
      if (i) {
        sb_puts(&g_result, "\\n");
      }
      const char* sym = kElementSymbols[atoms[i].z];
      sb_puts(&g_result, sym != NULL ? sym : "?");
      sb_putc(&g_result, ' ');
      sb_putd(&g_result, x[i * 3 + 0] / BOHR_PER_ANGSTROM);
      sb_putc(&g_result, ' ');
      sb_putd(&g_result, x[i * 3 + 1] / BOHR_PER_ANGSTROM);
      sb_putc(&g_result, ' ');
      sb_putd(&g_result, x[i * 3 + 2] / BOHR_PER_ANGSTROM);
    }
    sb_puts(&g_result, "\",\"charges\":[");
    for (int i = 0; i < n_atoms; ++i) {
      if (i) {
        sb_putc(&g_result, ',');
      }
      sb_puts(&g_result, "{\"element\":");
      sb_puti(&g_result, atoms[i].z);
      sb_puts(&g_result, ",\"q\":");
      sb_putd(&g_result, charges_q[i]);
      sb_putc(&g_result, '}');
    }
    sb_puts(&g_result, "],\"forces\":[");
    for (int i = 0; i < n_atoms; ++i) {
      if (i) {
        sb_putc(&g_result, ',');
      }
      sb_puts(&g_result, "{\"element\":");
      sb_puti(&g_result, atoms[i].z);
      sb_puts(&g_result, ",\"fx_eh_bohr\":");
      sb_putd(&g_result, forces[i * 3 + 0]);
      sb_puts(&g_result, ",\"fy_eh_bohr\":");
      sb_putd(&g_result, forces[i * 3 + 1]);
      sb_puts(&g_result, ",\"fz_eh_bohr\":");
      sb_putd(&g_result, forces[i * 3 + 2]);
      sb_putc(&g_result, '}');
    }
    sb_puts(&g_result, "]}");
  }

  free(atom_offsets);
  free(atomic_numbers);
  free(positions);
  free(molecular_charges);
  free(unpaired_electrons);
  free(energy);
  free(charges_q);
  free(forces);
  free(scc_iterations);
  free(scc_converged);
  free(per_system_status);
  free(x);
  free(g);
  free(g2);
  free(x2);
  free(p);
  free(q);
  free(r);
  free(s_hist);
  free(y_hist);
  free(rho);
  free(alpha);
  free(trajectory);
  free(step_buf);
  free(scc_per_step);
  return sb_finish(&g_result);
}

int main(void) { return 0; }
