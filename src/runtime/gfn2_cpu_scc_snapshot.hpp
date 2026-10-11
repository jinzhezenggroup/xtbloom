#ifndef XTBLOOM_RUNTIME_GFN2_CPU_SCC_SNAPSHOT_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_GFN2_CPU_SCC_SNAPSHOT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "cpu_dispatch/features.hpp"
#include "model/gfn2/wavefunction.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail {

/* Value-owned copies of the actual post-call CPU buffers. Orbital fields retain
 * WavefunctionLayout's alpha/beta convention; populations retain its distinct
 * charge/magnetization convention. Scratch and mixer fields are named after
 * their storage, not certified as a terminal Fock or a fresh fixed-point map. */
struct Gfn2CpuSccSystemSnapshot {
  gfn2::WavefunctionLayout layout;
  std::vector<std::int64_t> atom_shell_offsets;
  std::vector<std::int64_t> shell_orbital_offsets;
  std::uint64_t geometry_generation = 0u;
  xtbloom_status_t status = XTBLOOM_STATUS_INTERNAL_ERROR;
  std::int32_t iterations = 0;
  double public_energy = 0.0;
  std::vector<double> public_forces;
  std::vector<double> public_atomic_charges;
  std::vector<double> overlap;
  std::vector<double> core_hamiltonian;
  std::vector<double> coefficients;
  std::vector<double> eigenvalues;
  std::vector<double> occupations;
  std::vector<double> density;
  std::vector<double> energy_weighted_density;
  std::vector<double> qsh;
  std::vector<double> qat;
  std::vector<double> dipoles;
  std::vector<double> quadrupoles;
  std::vector<double> mixer_current_inputs;
  std::vector<double> mixer_previous_inputs;
  std::vector<double> mixer_previous_residuals;
  double mixer_residual_rms = 0.0;
  double mixer_residual_maximum = 0.0;
  std::uint64_t mixer_iterations = 0u;
  double scc_free_energy = 0.0;
  double scc_free_energy_change = 0.0;
};

struct Gfn2CpuSccSnapshot {
  CpuIsa cpu_isa = CpuIsa::kBaseline;
  std::uint64_t public_compute_sequence = 0u;
  std::vector<Gfn2CpuSccSystemSnapshot> systems;
};

/* Defined only in the standalone diagnostic build, never in libxtbloom. Call
 * immediately after synchronous xtbloom_compute with all E/F/q requested on a
 * CPU GFN2 context and with every peer converged. Enqueue,
 * plan, callback, and concurrent context operations are outside this seam's
 * contract. A rejected direct compute invalidates the receipt; failure leaves
 * the destination unchanged. This is observability, not a stability proof. */
#if defined(XTBLOOM_PUBLIC_SCC_SNAPSHOT_TESTING)
xtbloom_status_t snapshot_public_gfn2_cpu_scc(xtbloom_context_t* context,
                                              Gfn2CpuSccSnapshot& snapshot, std::string& error);
#endif

}  // namespace xtbloom::detail

#endif
