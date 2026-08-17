#ifndef XTBLOOM_MODEL_GFN2_MULLIKEN_KERNELS_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_MULLIKEN_KERNELS_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "cpu_dispatch/features.hpp"

namespace xtbloom::detail::gfn2 {

/* Non-copyable one-shot states passed by pointer to the selected leaf
 * callbacks. The generic Mulliken layer owns all validation, staging,
 * publication, and diagnostics; ISA-specific translation units receive only
 * already-validated flat ranges. orbital_to_atom is non-decreasing because it
 * is derived from validated atom-owned shell ranges, and chunk_count is
 * positive whenever a callback is entered. */
struct MullikenPopulationTask {
  const std::int64_t* orbital_to_shell = nullptr;
  const std::int64_t* orbital_to_atom = nullptr;
  const double* overlap = nullptr;
  const double* dipole_integrals = nullptr;
  const double* quadrupole_integrals = nullptr;
  const double* density = nullptr;
  double* qsh_scratch = nullptr;
  double* dipole_scratch = nullptr;
  double* quadrupole_scratch = nullptr;
  std::int64_t matrix_elements = 0;
  std::int64_t orbital_begin = 0;
  std::int64_t shell_begin = 0;
  std::int64_t atom_begin = 0;
  std::int64_t atoms = 0;
  std::int64_t shells = 0;
  std::int64_t orbitals = 0;
  std::int64_t matrix_base = 0;
  std::int64_t density_base = 0;
  std::int64_t qsh_base = 0;
  std::int64_t dipole_base = 0;
  std::int64_t quadrupole_base = 0;
  std::int64_t chunk_count = 1;
  std::int32_t nspin = 1;
  std::atomic<std::uint64_t> failure{0};
};

struct MullikenHamiltonianTask {
  const std::int64_t* orbital_to_shell = nullptr;
  const std::int64_t* orbital_to_atom = nullptr;
  const double* overlap = nullptr;
  const double* dipole_integrals = nullptr;
  const double* quadrupole_integrals = nullptr;
  const double* vat_scratch = nullptr;
  const double* vsh_scratch = nullptr;
  const double* dipole_scratch = nullptr;
  const double* quadrupole_scratch = nullptr;
  double* hamiltonian_scratch = nullptr;
  std::int64_t matrix_elements = 0;
  std::int64_t orbital_begin = 0;
  std::int64_t shell_begin = 0;
  std::int64_t atom_begin = 0;
  std::int64_t orbitals = 0;
  std::int64_t atoms = 0;
  std::int64_t shells = 0;
  std::int64_t matrix_base = 0;
  std::int64_t hamiltonian_base = 0;
  std::int64_t vat_base = 0;
  std::int64_t vsh_base = 0;
  std::int64_t dipole_base = 0;
  std::int64_t quadrupole_base = 0;
  std::int64_t chunk_count = 1;
  std::int32_t nspin = 1;
  std::atomic<std::uint64_t> failure{0};
};

using MullikenChunkKernel = void (*)(void*, std::size_t) noexcept;

struct MullikenKernelTable {
  MullikenChunkKernel population = nullptr;
  MullikenChunkKernel hamiltonian = nullptr;
  CpuIsa isa = CpuIsa::kBaseline;
};

void mulliken_population_chunk_baseline(void* opaque, std::size_t chunk) noexcept;
void mulliken_hamiltonian_chunk_baseline(void* opaque, std::size_t chunk) noexcept;
void mulliken_population_chunk_avx2_fma(void* opaque, std::size_t chunk) noexcept;
void mulliken_hamiltonian_chunk_avx2_fma(void* opaque, std::size_t chunk) noexcept;

[[nodiscard]] const MullikenKernelTable& mulliken_baseline_kernels() noexcept;
[[nodiscard]] const MullikenKernelTable& mulliken_avx2_fma_kernels() noexcept;
[[nodiscard]] const MullikenKernelTable& mulliken_kernels_for_cpu_isa(CpuIsa isa) noexcept;

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_MULLIKEN_KERNELS_HPP
