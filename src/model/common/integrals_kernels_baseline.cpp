// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

/*
 * Instantiate the reviewed common shell-pair implementation in a translation
 * unit whose build flags are guaranteed to remain x86-64 baseline safe. The
 * implementation source suppresses its normal public entry points in this
 * mode, so the only emitted external symbol is the baseline kernel table.
 */
#define XTBLOOM_INTEGRALS_BASELINE_VARIANT 1
#include "model/common/integrals.cpp"
