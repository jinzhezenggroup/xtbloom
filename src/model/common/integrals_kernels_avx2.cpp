// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

/*
 * Instantiate the same reviewed shell-pair implementation with AVX2/FMA
 * enabled. Keeping the variant in its own non-LTO object prevents specialized
 * instructions from leaking into portable callers or baseline dispatch code.
 */
#define XTBLOOM_INTEGRALS_AVX2_FMA_VARIANT 1
#include "model/common/integrals.cpp"
