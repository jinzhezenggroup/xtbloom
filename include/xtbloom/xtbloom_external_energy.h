#ifndef XTBLOOM_EXTERNAL_ENERGY_H
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_EXTERNAL_ENERGY_H

/*
 * Unreleased local research API for an external energy model. This header is
 * intentionally separate from xtbloom.h so it cannot be mistaken for a stable
 * xTBloom ABI.
 */

#include <stdint.h>

#include "xtbloom/xtbloom.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum xtbloom_external_energy_phase {
  XTBLOOM_EXTERNAL_ENERGY_PHASE_POTENTIAL = 0,
  XTBLOOM_EXTERNAL_ENERGY_PHASE_ENERGY = 1,
  XTBLOOM_EXTERNAL_ENERGY_PHASE_FORCE = 2,
} xtbloom_external_energy_phase_t;

/*
 * Callback data are one ragged-batch member in row-major AO order. Density and
 * Hamiltonian are channel-major: [alpha, nao, nao] for unrestricted systems
 * and [nao, nao] for restricted systems. atomic_numbers and positions are
 * local per-system slices. orbital_to_atom has one global atom index per AO;
 * subtract atom_index_begin to obtain a local atom index. Shell metadata is
 * also a global-plan slice: shell_orbital_offsets has shell_count + 1 entries
 * and stores global AO offsets, while shell_to_atom, principal_quantum_numbers,
 * and angular_momenta each have shell_count entries. Subtract
 * shell_orbital_index_begin from the shell offsets and shell_index_begin from
 * shell identifiers to obtain local indices. In the potential phase the
 * callback adds V_alpha/V_beta in place to hamiltonian. In the energy phase
 * hamiltonian is NULL and the callback writes the variational external energy.
 * During the force phase both hamiltonian and hamiltonian_elements are zero,
 * and the callback writes an additive Cartesian force correction into force.
 * overlap_gradient, when present, is laid out as [atom, xyz, nao, nao] and
 * contains the explicit derivative of the AO overlap used by the projection.
 * projection_overlap is the native-AO-to-auxiliary-basis cross overlap in
 * [nao, projection_orbitals] order. projection_overlap_gradient is laid out as
 * [atom, xyz, nao, projection_orbitals]. The auxiliary basis shell metadata is
 * local and zero based.
 * Returning a non-success status aborts the current SCC call.
 */
typedef xtbloom_status_t (*xtbloom_external_energy_callback_t)(
    void* opaque, int64_t system, xtbloom_external_energy_phase_t phase, int32_t spin_channels,
    int64_t atom_count, int64_t nao, const int32_t* atomic_numbers, int64_t atomic_number_elements,
    const double* positions, int64_t position_elements, const int64_t* orbital_to_atom,
    int64_t orbital_to_atom_elements, int64_t atom_index_begin, int64_t shell_count,
    const int64_t* shell_orbital_offsets, int64_t shell_orbital_offset_elements,
    const int64_t* shell_to_atom, int64_t shell_to_atom_elements,
    const uint8_t* principal_quantum_numbers, int64_t principal_quantum_number_elements,
    const uint8_t* angular_momenta, int64_t angular_momentum_elements, int64_t shell_index_begin,
    int64_t shell_orbital_index_begin, double molecular_charge, int32_t unpaired_electrons,
    const double* density, int64_t density_elements, const double* overlap,
    int64_t overlap_elements, const double* overlap_gradient, int64_t overlap_gradient_elements,
    int64_t projection_orbitals, int64_t projection_shell_count,
    const int64_t* projection_shell_orbital_offsets,
    int64_t projection_shell_orbital_offset_elements, const int64_t* projection_shell_to_atom,
    int64_t projection_shell_to_atom_elements, const uint8_t* projection_angular_momenta,
    int64_t projection_angular_momentum_elements, const double* projection_overlap,
    int64_t projection_overlap_elements, const double* projection_overlap_gradient,
    int64_t projection_overlap_gradient_elements, double* hamiltonian, int64_t hamiltonian_elements,
    double* force, int64_t force_elements, double* energy);

/* Flat native CUDA evaluator checkpoint. ``parameters`` is host memory for
 * upload and follows the evaluator's documented flat parameter layout. The
 * evaluator runs inside the device-resident SCC iteration. */
typedef struct xtbloom_external_energy_device_model {
  uint32_t struct_size;
  uint32_t api_version;
  int64_t geometry_dim;
  int64_t electronic_dim;
  int64_t hidden_dim;
  int64_t max_atomic_number;
  int64_t projection_width;
  int64_t radial_count;
  double output_scale;
  xtbloom_const_buffer_t parameters;
  uint32_t flags;
  uint32_t reserved;
} xtbloom_external_energy_device_model_t;

enum {
  XTBLOOM_EXTERNAL_ENERGY_DEVICE_NATIVE_SHELL = 1u << 0,
  XTBLOOM_EXTERNAL_ENERGY_DEVICE_DIAGONAL_PROJECTION = 1u << 1,
  /* Accumulate dE_external/d(parameter) in the context-owned CUDA buffer so a
   * differentiable host binding can attach the native value to PyTorch. */
  XTBLOOM_EXTERNAL_ENERGY_DEVICE_TRAINING_GRADIENT = 1u << 2,
};

/*
 * Install or clear the callback on a context. On CPU contexts the callback is
 * invoked directly by native SCC worker threads. On CUDA contexts it selects
 * the synchronous host-staged compatibility path; if a native device model is
 * installed, that model takes precedence for CUDA execution. Implementations
 * must keep opaque storage alive until the callback is cleared or the context
 * is destroyed.
 */
XTBLOOM_API xtbloom_status_t xtbloom_context_set_external_energy_callback(
    xtbloom_context_t* context, xtbloom_external_energy_callback_t callback, void* opaque);

/* Install or clear the native CUDA evaluator. It executes inside the
 * device-resident SCC graph and never enters a Python/ctypes callback. */
XTBLOOM_API xtbloom_status_t xtbloom_context_set_external_energy_device_model(
    xtbloom_context_t* context, const xtbloom_external_energy_device_model_t* model);

/* Copy the last energy-phase dE_external/d(parameter) vector to host memory. The
 * vector follows the same flat checkpoint order as the installed model and
 * is available only when XTBLOOM_EXTERNAL_ENERGY_DEVICE_TRAINING_GRADIENT is set. */
XTBLOOM_API xtbloom_status_t xtbloom_context_copy_external_energy_device_gradients(
    xtbloom_context_t* context, double* destination, int64_t elements);

#ifdef __cplusplus
}
#endif

#endif  // XTBLOOM_EXTERNAL_ENERGY_H
