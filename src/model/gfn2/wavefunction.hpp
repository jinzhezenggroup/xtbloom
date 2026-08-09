#ifndef XTBLOOM_MODEL_GFN2_WAVEFUNCTION_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_MODEL_GFN2_WAVEFUNCTION_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/gfn2/basis.hpp"
#include "xtbloom/xtbloom.h"

namespace xtbloom::detail::gfn2 {

inline constexpr std::size_t kWavefunctionWorkspaceAlignment = 64u;
inline constexpr std::int32_t kWavefunctionQuadrupoleComponents = 6;

/*
 * One typed segment of the caller-owned wavefunction workspace. Byte offsets
 * are aligned to kWavefunctionWorkspaceAlignment. system_offsets are element
 * offsets relative to this field's base and contain batch_size + 1 entries.
 */
struct WavefunctionFieldLayout {
  std::size_t offset_bytes = 0;
  std::size_t size_bytes = 0;
  std::int64_t element_count = 0;
  std::vector<std::int64_t> system_offsets;
};

/*
 * Exact metadata carried with a reusable wavefunction warm start.
 *
 * Geometry values are intentionally not copied. geometry_cache_generation is
 * the caller's monotonically changing identity for the cached geometry and
 * every geometry-derived operator. Exact topology, charge, and spin metadata
 * avoids accepting a stale state based only on a collision-prone hash.
 */
struct WavefunctionWarmStartIdentity {
  xtbloom_model_t model = XTBLOOM_MODEL_GFN2_XTB;
  std::uint64_t geometry_cache_generation = 0;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> batch_orbital_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
};

/*
 * Geometry-independent, backend-neutral layout for a ragged GFN2 batch.
 *
 * For system i, coefficients, density, and W are spin-major dense row-major
 * [nspin, nao, nao] arrays. Eigenvalues are [nspin, nao]. Occupations always
 * store separate alpha and beta rows [2, nao], matching tblite's restricted
 * convention even when nspin is one. Density-like orbital fields use
 * alpha/beta channels when nspin is two. The SCC qsh/qat/dipole/quadrupole
 * fields use tblite's charge/magnetization representation, with shapes
 * [nspin, nsh/nat], [nspin, nat, 3], and [nspin, nat, 6], respectively.
 *
 * Construction may allocate metadata. The numerical state itself lives only
 * in one aligned caller workspace described by the field layouts below.
 */
struct WavefunctionLayout {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_orbitals = 0;
  std::size_t workspace_size_bytes = 0;

  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> batch_shell_offsets;
  std::vector<std::int64_t> batch_orbital_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;

  /* tblite get_occupation-compatible immutable reference populations. */
  std::vector<double> reference_atom_occupations;
  std::vector<double> reference_shell_occupations;
  std::vector<double> electron_counts;
  std::vector<double> alpha_electron_counts;
  std::vector<double> beta_electron_counts;

  WavefunctionFieldLayout coefficients;
  WavefunctionFieldLayout eigenvalues;
  WavefunctionFieldLayout occupations;
  WavefunctionFieldLayout density;
  WavefunctionFieldLayout qsh;
  WavefunctionFieldLayout qat;
  WavefunctionFieldLayout dipole;
  WavefunctionFieldLayout quadrupole;
  WavefunctionFieldLayout energy_weighted_density;
};

/*
 * Non-owning bases for all arrays in one mutable or immutable workspace.
 *
 * workspace_base and workspace_size_bytes retain the binding metadata produced
 * by bind_wavefunction_view. Consumers validate every field against these
 * values and the canonical layout before performing pointer arithmetic or
 * accessing numerical storage, so aggregate copies remain safe to pass while
 * forged, truncated, or internally aliased views are rejected atomically.
 */
struct WavefunctionView {
  void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0;
  double* coefficients = nullptr;
  double* eigenvalues = nullptr;
  double* occupations = nullptr;
  double* density = nullptr;
  double* qsh = nullptr;
  double* qat = nullptr;
  double* dipole = nullptr;
  double* quadrupole = nullptr;
  double* energy_weighted_density = nullptr;
};

struct ConstWavefunctionView {
  const void* workspace_base = nullptr;
  std::size_t workspace_size_bytes = 0;
  const double* coefficients = nullptr;
  const double* eigenvalues = nullptr;
  const double* occupations = nullptr;
  const double* density = nullptr;
  const double* qsh = nullptr;
  const double* qat = nullptr;
  const double* dipole = nullptr;
  const double* quadrupole = nullptr;
  const double* energy_weighted_density = nullptr;
};

/* A non-owning slice for one member of the ragged batch. */
struct WavefunctionSystemView {
  std::int64_t atom_count = 0;
  std::int64_t shell_count = 0;
  std::int64_t orbital_count = 0;
  std::int32_t spin_channels = 0;
  double electron_count = 0.0;
  double alpha_electron_count = 0.0;
  double beta_electron_count = 0.0;
  double* coefficients = nullptr;
  double* eigenvalues = nullptr;
  double* occupations = nullptr;
  double* density = nullptr;
  double* qsh = nullptr;
  double* qat = nullptr;
  double* dipole = nullptr;
  double* quadrupole = nullptr;
  double* energy_weighted_density = nullptr;
};

struct ConstWavefunctionSystemView {
  std::int64_t atom_count = 0;
  std::int64_t shell_count = 0;
  std::int64_t orbital_count = 0;
  std::int32_t spin_channels = 0;
  double electron_count = 0.0;
  double alpha_electron_count = 0.0;
  double beta_electron_count = 0.0;
  const double* coefficients = nullptr;
  const double* eigenvalues = nullptr;
  const double* occupations = nullptr;
  const double* density = nullptr;
  const double* qsh = nullptr;
  const double* qat = nullptr;
  const double* dipole = nullptr;
  const double* quadrupole = nullptr;
  const double* energy_weighted_density = nullptr;
};

/*
 * Build reference populations, validate charge/spin realizability, and pack
 * all numerical fields. Molecular charges may be fractional. Parity follows
 * tblite's nint(total_electrons) convention; unlike tblite's CLI fallback,
 * xtbloom rejects a supplied incompatible unpaired-electron count. The number
 * of spin channels is independent of the number of unpaired electrons, as in
 * tblite: one channel shares orbitals between alpha and beta occupations,
 * while two channels use separate alpha and beta orbitals.
 */
xtbloom_status_t make_wavefunction_layout(const BasisPlan& basis,
                                          const std::int32_t* atomic_numbers,
                                          const double* molecular_charges,
                                          const std::int32_t* unpaired_electrons,
                                          const std::int32_t* spin_channels,
                                          WavefunctionLayout& layout, std::string& error);

/* Bind an aligned caller allocation without allocating or initializing it. */
xtbloom_status_t bind_wavefunction_view(const WavefunctionLayout& layout, void* workspace,
                                        std::size_t workspace_size, WavefunctionView& view,
                                        std::string& error);
xtbloom_status_t bind_wavefunction_view(const WavefunctionLayout& layout, const void* workspace,
                                        std::size_t workspace_size, ConstWavefunctionView& view,
                                        std::string& error);

/*
 * Produce one system slice without allocating. batch_view must be an
 * unmodified view returned by bind_wavefunction_view for the same layout.
 * This provenance is validated before any pointer arithmetic.
 */
xtbloom_status_t make_wavefunction_system_view(const WavefunctionLayout& layout,
                                               const WavefunctionView& batch_view,
                                               std::int64_t system,
                                               WavefunctionSystemView& system_view,
                                               std::string& error);
xtbloom_status_t make_wavefunction_system_view(const WavefunctionLayout& layout,
                                               const ConstWavefunctionView& batch_view,
                                               std::int64_t system,
                                               ConstWavefunctionSystemView& system_view,
                                               std::string& error);

/*
 * Initialize tblite-compatible superposition-of-atomic-densities multipole
 * state without allocating. Charge is distributed uniformly over atoms in
 * channel zero and then partitioned over shells according to n0sh/n0at.
 * Magnetization channels and all atomic dipoles and quadrupoles are zeroed.
 * view must be an unmodified result of bind_wavefunction_view for layout; the
 * binding is validated before any caller-owned numerical storage is modified.
 */
xtbloom_status_t initialize_sad_multipole_state(const WavefunctionLayout& layout,
                                                const WavefunctionView& view, std::string& error);

/* Create and compare exact warm-start compatibility metadata. */
xtbloom_status_t make_wavefunction_warm_start_identity(const WavefunctionLayout& layout,
                                                       std::uint64_t geometry_cache_generation,
                                                       WavefunctionWarmStartIdentity& identity,
                                                       std::string& error);
bool wavefunction_warm_start_matches(const WavefunctionWarmStartIdentity& expected,
                                     const WavefunctionWarmStartIdentity& candidate) noexcept;
xtbloom_status_t validate_wavefunction_warm_start(const WavefunctionWarmStartIdentity& expected,
                                                  const WavefunctionWarmStartIdentity& candidate,
                                                  std::string& error);

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_WAVEFUNCTION_HPP
