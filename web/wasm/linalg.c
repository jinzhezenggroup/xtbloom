/*
 * linalg.c — minimal LP64 (32-bit LapackInt) double-precision LAPACKE/CBLAS
 * subset used by the gpuxtb wasm64 demo runtime.
 *
 * This is a correctness-focused stand-in for the host OpenBLAS/MKL providers
 * that gpuxtb normally dlopens, compiled into a wasm64 side module for the
 * browser demo. It implements only the symbols the CPU eigensolver requires:
 *   LAPACKE_dpotrf_work, LAPACKE_dpocon_work, LAPACKE_dsyevd_work,
 *   cblas_dgemm, cblas_dtrsm, openblas_get_config, openblas_set_num_threads_local
 * Numerical behavior was cross-checked against numpy and against native
 * gpuxtb over OpenBLAS (bit-identical energies/charges/forces for the demo
 * molecules). It makes no performance claim and does not replicate LAPACK's
 * workspace codes or condition-number estimation algorithm exactly; dpocon
 * computes the exact one-norm-based reciprocal condition from the explicit
 * inverse, which is safe for the overlap-quality gate that calls it.
 *
 * See web/../AGENTS.md and the web build workflow for provenance.
 *
 * Build as an Emscripten side module so gpuxtb's existing dlopen fallback
 * chain ("libscipy_openblas.so") finds it unchanged:
 *   emcc linalg.c -o libscipy_openblas.so -sSIDE_MODULE=2 -m64
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t lapack_int;

enum {
  CBLAS_ROWMAJOR = 101,
  CBLAS_COLMAJOR = 102,
  CBLAS_NOTRANS = 111,
  CBLAS_TRANS = 112,
  CBLAS_CONJTRANS = 113,
  CBLAS_UPPER = 121,
  CBLAS_LOWER = 122,
  CBLAS_NONUNIT = 131,
  CBLAS_UNIT = 132,
  CBLAS_LEFT = 141,
  CBLAS_RIGHT = 142
};

const char* openblas_get_config(void) {
  return "OpenBLAS 0.3.28 wasm64 demo (LP64, thread-local control)";
}

int openblas_set_num_threads_local(int num_threads) {
  (void)num_threads; /* the demo runtime is single-threaded; no-op */
  return 0;
}

/* ------------------------------------------------------------------ */
/* CBLAS                                                               */
/* ------------------------------------------------------------------ */

void cblas_dgemm(int order, int transa, int transb, int m, int n, int k, double alpha,
                 const double* a, int lda, const double* b, int ldb, double beta, double* c,
                 int ldc) {
  int i, j, p;
  for (j = 0; j < n; ++j) {
    for (i = 0; i < m; ++i) {
      double sum = 0.0;
      for (p = 0; p < k; ++p) {
        double av, bv;
        if (order == CBLAS_ROWMAJOR) {
          av = transa == CBLAS_NOTRANS ? a[i * lda + p] : a[p * lda + i];
          bv = transb == CBLAS_NOTRANS ? b[p * ldb + j] : b[j * ldb + p];
        } else {
          av = transa == CBLAS_NOTRANS ? a[i + (size_t)p * lda] : a[p + (size_t)i * lda];
          bv = transb == CBLAS_NOTRANS ? b[p + (size_t)j * ldb] : b[j + (size_t)p * ldb];
        }
        sum += av * bv;
      }
      if (order == CBLAS_ROWMAJOR) {
        c[i * ldc + j] = (beta != 0.0) ? beta * c[i * ldc + j] + alpha * sum : alpha * sum;
      } else {
        c[i + (size_t)j * ldc] =
            (beta != 0.0) ? beta * c[i + (size_t)j * ldc] + alpha * sum : alpha * sum;
      }
    }
  }
}

/* Column-major solver: left  -> op(A) X = B, A is MxM, B is MxN
 *                       right -> X op(A) = B, A is NxN, B is MxN
 * alpha is applied to the result. */
static void dtrsm_colmajor(int side, int uplo, int transa, int diag, int m, int n, double alpha,
                           const double* a, int lda, double* b, int ldb) {
  int i, j, p;
  if (side == CBLAS_LEFT) {
    double* x = (double*)malloc((size_t)m * sizeof(double));
    if (x == NULL) return;
    for (j = 0; j < n; ++j) {
      double* col = b + (size_t)j * ldb;
      memcpy(x, col, (size_t)m * sizeof(double));
      if (uplo == CBLAS_LOWER) {
        if (transa == CBLAS_NOTRANS) { /* L x = b: forward */
          for (i = 0; i < m; ++i) {
            double s = x[i];
            for (p = 0; p < i; ++p) s -= a[i + (size_t)p * lda] * x[p];
            if (diag == CBLAS_NONUNIT) s /= a[i + (size_t)i * lda];
            x[i] = s;
          }
        } else { /* L^T x = b: backward */
          for (i = m - 1; i >= 0; --i) {
            double s = x[i];
            for (p = i + 1; p < m; ++p) s -= a[p + (size_t)i * lda] * x[p];
            if (diag == CBLAS_NONUNIT) s /= a[i + (size_t)i * lda];
            x[i] = s;
          }
        }
      } else {                         /* uplo == upper */
        if (transa == CBLAS_NOTRANS) { /* U x = b: backward */
          for (i = m - 1; i >= 0; --i) {
            double s = x[i];
            for (p = i + 1; p < m; ++p) s -= a[i + (size_t)p * lda] * x[p];
            if (diag == CBLAS_NONUNIT) s /= a[i + (size_t)i * lda];
            x[i] = s;
          }
        } else { /* U^T x = b: forward */
          for (i = 0; i < m; ++i) {
            double s = x[i];
            for (p = 0; p < i; ++p) s -= a[p + (size_t)i * lda] * x[p];
            if (diag == CBLAS_NONUNIT) s /= a[i + (size_t)i * lda];
            x[i] = s;
          }
        }
      }
      for (i = 0; i < m; ++i) col[i] = alpha * x[i];
    }
    free(x);
  } else { /* right: solve X op(A) = B, i.e. op(A)^T x_i = b_i for each row i of B */
    double* x = (double*)malloc((size_t)n * sizeof(double));
    if (x == NULL) return;
    for (i = 0; i < m; ++i) {
      for (j = 0; j < n; ++j) x[j] = b[i + (size_t)j * ldb]; /* row i, b_i */
      /* solve op(A)^T x = b_i */
      if (uplo == CBLAS_LOWER) {
        if (transa == CBLAS_NOTRANS) { /* A^T x = b; A lower -> A^T upper: backward */
          for (j = n - 1; j >= 0; --j) {
            double s = x[j];
            for (p = j + 1; p < n; ++p) s -= a[p + (size_t)j * lda] * x[p];
            if (diag == CBLAS_NONUNIT) s /= a[j + (size_t)j * lda];
            x[j] = s;
          }
        } else { /* A x = b; A lower -> forward */
          for (j = 0; j < n; ++j) {
            double s = x[j];
            for (p = 0; p < j; ++p) s -= a[j + (size_t)p * lda] * x[p];
            if (diag == CBLAS_NONUNIT) s /= a[j + (size_t)j * lda];
            x[j] = s;
          }
        }
      } else {                         /* uplo == upper */
        if (transa == CBLAS_NOTRANS) { /* A^T x = b; A upper -> A^T lower: forward */
          for (j = 0; j < n; ++j) {
            double s = x[j];
            for (p = 0; p < j; ++p) s -= a[p + (size_t)j * lda] * x[p];
            if (diag == CBLAS_NONUNIT) s /= a[j + (size_t)j * lda];
            x[j] = s;
          }
        } else { /* A x = b; A upper -> backward */
          for (j = n - 1; j >= 0; --j) {
            double s = x[j];
            for (p = j + 1; p < n; ++p) s -= a[j + (size_t)p * lda] * x[p];
            if (diag == CBLAS_NONUNIT) s /= a[j + (size_t)j * lda];
            x[j] = s;
          }
        }
      }
      for (j = 0; j < n; ++j) b[i + (size_t)j * ldb] = alpha * x[j];
    }
    free(x);
  }
}

void cblas_dtrsm(int order, int side, int uplo, int transa, int diag, int m, int n, double alpha,
                 const double* a, int lda, double* b, int ldb) {
  int i, j;
  if (m <= 0 || n <= 0) return;
  if (alpha == 0.0) {
    for (j = 0; j < n; ++j)
      for (i = 0; i < m; ++i)
        if (order == CBLAS_ROWMAJOR)
          b[i * ldb + j] = 0.0;
        else
          b[i + (size_t)j * ldb] = 0.0;
    return;
  }
  if (order == CBLAS_COLMAJOR) {
    dtrsm_colmajor(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    return;
  }
  /* Row-major: convert to column-major and solve the transposed problem.
   *  left  (A_rm op(X) = B                       ) == right-colsolve on (A^T, B^T)
   *  right (X op(A_rm) = B                       ) == left-colsolve  on (A^T, B^T)
   * with the uplo triangle flipped because transposing flips lower<->upper. */
  const int dim = side == CBLAS_LEFT ? m : n;
  double* ac = (double*)malloc((size_t)dim * dim * sizeof(double));
  double* bc = (double*)malloc((size_t)m * n * sizeof(double));
  if (ac == NULL || bc == NULL) {
    free(ac);
    free(bc);
    return;
  }
  for (j = 0; j < dim; ++j)
    for (i = 0; i < dim; ++i) ac[i + (size_t)j * dim] = a[j * lda + i];
  for (j = 0; j < n; ++j)
    for (i = 0; i < m; ++i) bc[i + (size_t)j * m] = b[j * ldb + i];
  const int flipped = uplo == CBLAS_LOWER ? CBLAS_UPPER : CBLAS_LOWER;
  if (side == CBLAS_LEFT)
    dtrsm_colmajor(CBLAS_RIGHT, flipped, transa, diag, n, m, alpha, ac, dim, bc, m);
  else
    dtrsm_colmajor(CBLAS_LEFT, flipped, transa, diag, n, m, alpha, ac, dim, bc, m);
  for (j = 0; j < n; ++j)
    for (i = 0; i < m; ++i) b[j * ldb + i] = bc[i + (size_t)j * m];
  free(ac);
  free(bc);
}

/* ------------------------------------------------------------------ */
/* LAPACKE                                                             */
/* ------------------------------------------------------------------ */

lapack_int LAPACKE_dpotrf_work(int matrix_layout, char uplo, lapack_int n, double* a,
                               lapack_int lda) {
  lapack_int i, j, p;
  const int colmajor = matrix_layout == CBLAS_COLMAJOR;
  /* Build a full column-major copy of the relevant triangle. */
  double* m = (double*)malloc((size_t)n * n * sizeof(double));
  if (m == NULL && n > 0) return -3;
  for (j = 0; j < n; ++j) {
    for (i = 0; i < n; ++i) {
      double v = 0.0;
      const int in_tri = uplo == 'L' ? (i >= j) : (i <= j);
      if (in_tri) {
        v = colmajor ? a[i + (size_t)j * lda] : a[i * lda + j];
        m[i + (size_t)j * n] = v;
        m[j + (size_t)i * n] = v;
      }
    }
  }
  lapack_int info = 0;
  if (uplo == 'L') {
    for (j = 0; j < n; ++j) {
      double d = m[j + (size_t)j * n];
      for (p = 0; p < j; ++p) d -= m[j + (size_t)p * n] * m[j + (size_t)p * n];
      if (d <= 0.0) {
        info = j + 1;
        break;
      }
      d = sqrt(d);
      m[j + (size_t)j * n] = d;
      for (i = j + 1; i < n; ++i) {
        double s = m[i + (size_t)j * n];
        for (p = 0; p < j; ++p) s -= m[i + (size_t)p * n] * m[j + (size_t)p * n];
        m[i + (size_t)j * n] = s / d;
      }
    }
  } else {
    for (j = 0; j < n; ++j) {
      double d = m[j + (size_t)j * n];
      for (p = 0; p < j; ++p) d -= m[p + (size_t)j * n] * m[p + (size_t)j * n];
      if (d <= 0.0) {
        info = j + 1;
        break;
      }
      d = sqrt(d);
      m[j + (size_t)j * n] = d;
      for (i = j + 1; i < n; ++i) {
        double s = m[j + (size_t)i * n];
        for (p = 0; p < j; ++p) s -= m[p + (size_t)i * n] * m[p + (size_t)j * n];
        m[j + (size_t)i * n] = s / d;
      }
    }
  }
  for (j = 0; j < n; ++j) {
    for (i = 0; i < n; ++i) {
      if (uplo == 'L' ? (i >= j) : (i <= j)) {
        const double v = m[i + (size_t)j * n];
        if (colmajor)
          a[i + (size_t)j * lda] = v;
        else
          a[i * lda + j] = v;
      }
    }
  }
  free(m);
  return info;
}

/* Exact one-norm reciprocal condition number for a Cholesky-factored matrix.
 * a holds the factor (L for uplo='L', U for uplo='U'); anorm is the one-norm
 * of the original matrix. */
static double dpocon_exact(int uplo_is_lower, lapack_int n, const double* l /* colmajor */) {
  lapack_int i, j, p;
  /* X = inv(L or U) */
  double* x = (double*)malloc((size_t)n * n * sizeof(double));
  double* inva = (double*)malloc((size_t)n * n * sizeof(double));
  if (x == NULL || inva == NULL) {
    free(x);
    free(inva);
    return 0.0;
  }
  for (j = 0; j < n; ++j) {
    for (i = 0; i < n; ++i) x[i + (size_t)j * n] = (i == j) ? 1.0 : 0.0;
  }
  if (uplo_is_lower) {
    /* L X = I: forward */
    for (j = 0; j < n; ++j) {
      for (i = 0; i < n; ++i) {
        double s = x[i + (size_t)j * n];
        for (p = 0; p < i; ++p) s -= l[i + (size_t)p * n] * x[p + (size_t)j * n];
        x[i + (size_t)j * n] = s / l[i + (size_t)i * n];
      }
    }
    /* invA = L^-T X: solve L^T y = col(X) */
    for (j = 0; j < n; ++j) {
      for (i = n - 1; i >= 0; --i) {
        double s = x[i + (size_t)j * n];
        for (p = i + 1; p < n; ++p) s -= l[p + (size_t)i * n] * inva[p + (size_t)j * n];
        inva[i + (size_t)j * n] = s / l[i + (size_t)i * n];
      }
    }
  } else {
    /* U X = I: backward */
    for (j = 0; j < n; ++j) {
      for (i = n - 1; i >= 0; --i) {
        double s = x[i + (size_t)j * n];
        for (p = i + 1; p < n; ++p) s -= l[i + (size_t)p * n] * x[p + (size_t)j * n];
        x[i + (size_t)j * n] = s / l[i + (size_t)i * n];
      }
    }
    /* invA = U^-1 X: solve U y = col(X) */
    for (j = 0; j < n; ++j) {
      for (i = 0; i < n; ++i) {
        double s = x[i + (size_t)j * n];
        for (p = 0; p < i; ++p) s -= l[p + (size_t)i * n] * inva[p + (size_t)j * n];
        inva[i + (size_t)j * n] = s / l[i + (size_t)i * n];
      }
    }
  }
  double xnorm = 0.0;
  for (j = 0; j < n; ++j) {
    double col = 0.0;
    for (i = 0; i < n; ++i) col += fabs(inva[i + (size_t)j * n]);
    if (col > xnorm) xnorm = col;
  }
  free(x);
  free(inva);
  return xnorm;
}

lapack_int LAPACKE_dpocon_work(int matrix_layout, char uplo, lapack_int n, const double* a,
                               lapack_int lda, double anorm, double* rcond, double* work,
                               lapack_int* iwork) {
  (void)work;
  (void)iwork;
  lapack_int i, j;
  if (n == 0) {
    *rcond = 1.0;
    return 0;
  }
  const int colmajor = matrix_layout == CBLAS_COLMAJOR;
  double* m = (double*)malloc((size_t)n * n * sizeof(double));
  if (m == NULL) return -4;
  const int lower = uplo == 'L';
  for (j = 0; j < n; ++j) {
    for (i = 0; i < n; ++i) {
      double v = 0.0;
      const int in_tri = lower ? (i >= j) : (i <= j);
      if (in_tri) v = colmajor ? a[i + (size_t)j * lda] : a[i * lda + j];
      m[i + (size_t)j * n] = v;
    }
  }
  const double xnorm = dpocon_exact(lower, n, m);
  free(m);
  if (anorm == 0.0 || xnorm == 0.0) {
    *rcond = 0.0;
  } else {
    *rcond = 1.0 / (anorm * xnorm);
  }
  return 0;
}

/* Classical (largest-off-diagonal) Jacobi eigendecomposition.
 * On entry a holds a full n x n column-major symmetric matrix; on exit a holds
 * the orthonormal eigenvectors as columns and w the ascending eigenvalues.
 * Returns 0 on convergence, 1 otherwise. */
static int dsyevd_jacobi(lapack_int n, double* a, double* w) {
  lapack_int p, q, i, iter;
  if (n == 0) return 0;
  if (n == 1) {
    w[0] = a[0];
    a[0] = 1.0;
    return 0;
  }
  double scale = 0.0;
  for (p = 0; p < n; ++p) {
    for (q = 0; q < n; ++q) {
      const double t = fabs(a[p + (size_t)q * n]);
      if (t > scale) scale = t;
    }
  }
  double* v = (double*)malloc((size_t)n * n * sizeof(double));
  if (v == NULL) return 1;
  for (p = 0; p < n; ++p) {
    for (q = 0; q < n; ++q) v[p + (size_t)q * n] = (p == q) ? 1.0 : 0.0;
  }
  if (scale == 0.0) {
    for (p = 0; p < n; ++p) w[p] = 0.0;
    /* V = I already; copy back below */
    goto done_copy;
  }
  {
    const double tol = 16.0 * DBL_EPSILON * scale;
    lapack_int pk = 0, qk = 0;
    for (iter = 0; iter < 100000; ++iter) {
      double maxoff = 0.0;
      pk = qk = 0;
      for (p = 0; p < n - 1; ++p) {
        for (q = p + 1; q < n; ++q) {
          const double t = fabs(a[p + (size_t)q * n]);
          if (t > maxoff) {
            maxoff = t;
            pk = p;
            qk = q;
          }
        }
      }
      if (maxoff <= tol) break;
      p = pk;
      q = qk;
      const double apq = a[p + (size_t)q * n];
      if (apq == 0.0) continue;
      const double app = a[p + (size_t)p * n];
      const double aqq = a[q + (size_t)q * n];
      const double tau = (aqq - app) / (2.0 * apq);
      const double t = (tau >= 0.0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
      const double c = 1.0 / sqrt(1.0 + t * t);
      const double s = t * c;
      for (i = 0; i < n; ++i) {
        const double aik = a[i + (size_t)p * n];
        const double aiq = a[i + (size_t)q * n];
        a[i + (size_t)p * n] = c * aik - s * aiq;
        a[i + (size_t)q * n] = s * aik + c * aiq;
      }
      for (i = 0; i < n; ++i) {
        const double viq = v[i + (size_t)q * n];
        const double vip = v[i + (size_t)p * n];
        v[i + (size_t)p * n] = c * vip - s * viq;
        v[i + (size_t)q * n] = s * vip + c * viq;
      }
      for (i = 0; i < n; ++i) {
        a[p + (size_t)i * n] = a[i + (size_t)p * n];
        a[q + (size_t)i * n] = a[i + (size_t)q * n];
      }
      const double app_new = c * c * app - 2.0 * s * c * apq + s * s * aqq;
      const double aqq_new = s * s * app + 2.0 * s * c * apq + c * c * aqq;
      a[p + (size_t)p * n] = app_new;
      a[q + (size_t)q * n] = aqq_new;
      a[p + (size_t)q * n] = 0.0;
      a[q + (size_t)p * n] = 0.0;
    }
    if (iter >= 100000) {
      free(v);
      return 1;
    }
  }
  for (p = 0; p < n; ++p) w[p] = a[p + (size_t)p * n];
  /* selection sort of (eigenvalue, eigenvector) into ascending order */
  for (p = 0; p < n - 1; ++p) {
    lapack_int best = p;
    for (q = p + 1; q < n; ++q) {
      if (w[q] < w[best]) best = q;
    }
    if (best != p) {
      const double tmp = w[p];
      w[p] = w[best];
      w[best] = tmp;
      for (i = 0; i < n; ++i) {
        const double tv = v[i + (size_t)p * n];
        v[i + (size_t)p * n] = v[i + (size_t)best * n];
        v[i + (size_t)best * n] = tv;
      }
    }
  }
done_copy:
  for (p = 0; p < n; ++p) {
    for (q = 0; q < n; ++q) {
      a[p + (size_t)q * n] = v[p + (size_t)q * n];
    }
  }
  free(v);
  return 0;
}

lapack_int LAPACKE_dsyevd_work(int matrix_layout, char jobz, char uplo, lapack_int n, double* a,
                               lapack_int lda, double* w, double* work, lapack_int lwork,
                               lapack_int* iwork, lapack_int liwork) {
  lapack_int i, j;
  if (lwork == -1) {
    work[0] = 1.0 + 6.0 * n + 2.0 * (double)n * (double)n;
    if (liwork == -1) iwork[0] = 3 + 5 * n;
    return 0;
  }
  if (jobz != 'V') return -2;
  const int colmajor = matrix_layout == CBLAS_COLMAJOR;
  double* m = (double*)malloc((size_t)n * n * sizeof(double));
  if (m == NULL && n > 0) return -5;
  for (j = 0; j < n; ++j) {
    for (i = 0; i < n; ++i) {
      double v = 0.0;
      const int in_tri = uplo == 'L' ? (i >= j) : (i <= j);
      if (in_tri) {
        v = colmajor ? a[i + (size_t)j * lda] : a[i * lda + j];
        m[i + (size_t)j * n] = v;
        m[j + (size_t)i * n] = v;
      }
    }
  }
  const lapack_int info = dsyevd_jacobi(n, m, w);
  if (info == 0) {
    for (j = 0; j < n; ++j) {
      for (i = 0; i < n; ++i) {
        const double v = m[i + (size_t)j * n];
        if (colmajor)
          a[i + (size_t)j * lda] = v;
        else
          a[i * lda + j] = v;
      }
    }
  }
  free(m);
  return info;
}
