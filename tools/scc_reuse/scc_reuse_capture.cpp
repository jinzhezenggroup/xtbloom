// xtbloom SCC subspace-reuse capture executable (issue #343, Phase 1).
//
// Experimental diagnostic tooling, NOT part of the public C ABI and NOT part
// of any conformance/acceptance gate. It drives the production CPU GFN2 SCC
// driver through one case.spec (and optionally a second geometry for a
// warm-start trajectory) and streams, per completed SCC iteration:
//
//   * the effective Hamiltonian H_k and the core H0 plus overlap S;
//   * the orbital coefficients C_k (generalized eigenpairs with
//     C_k^T S C_k = I), eigenvalues, alpha/beta occupations, and density P_k;
//   * the wall time of the whole driver step;
//   * an isolated eigensolve-only wall time obtained by running the
//     production solve_eigensystem_cpu on the same H_k after the step.
//
// scc_reuse_analyze.py reads the stream and computes the Phase-1 reuse
// metrics: relative ||dH||/||dP||, S-metric principal angles between
// successive occupied subspaces, and the generalized residual of the previous
// eigenspace against the new Hamiltonian. The versioned xtbloom-scc-trace-v1
// contract is deliberately untouched; this tool emits its own diagnostic
// layout described below.
//
// The SCC numerical policy matches the pinned restricted corpus so the
// captured trajectories are the same science as the conformance goldens:
// zero-charge perturbative seed, self-consistent D4 two-body potential,
// tblite single-point convergence tolerances (energy 1e-6, RMS 2e-5).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/eigensolver.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/es3.hpp"
#include "model/gfn2/external_point_charges.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/scc_driver.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"
#include "xtbloom/xtbloom.h"

namespace {

using namespace xtbloom::detail::gfn2;

constexpr double kKelvinToHartree = 3.166808578545117e-6;
constexpr double kTbliteEnergyTolerance = 1.0e-6;
constexpr double kTbliteRmsTolerance = 2.0e-5;

struct AlignedBuffer {
  void* data = nullptr;
  std::size_t size = 0u;
  explicit AlignedBuffer(std::size_t requested) {
    size = (std::max<std::size_t>(requested, 1u) + 63u) & ~std::size_t{63u};
    data = std::aligned_alloc(64u, size);
    if (data != nullptr) {
      std::memset(data, 0, size);
    }
  }
  ~AlignedBuffer() { std::free(data); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

// One corpus-style input; the same layout as data/conformance/scc-traces/specs.
// Positions are in bohr. A trajectory second geometry must be the same molecule.
struct CaseSpec {
  std::string name;
  std::int64_t nat = 0;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;  // bohr, atom-major x y z
  double molecular_charge = 0.0;
  std::int32_t unpaired_electrons = 0;
  double temperature_kelvin = 300.0;
  std::int64_t mixer_memory = 2;
  double mixer_damping = 0.4;
  std::int64_t maximum_iterations = 100;
  std::vector<double> pc_rows;  // x y z q gamma per point charge
};

bool load_spec(const std::string& path, CaseSpec& spec, std::string& err) {
  std::ifstream file(path);
  if (!file) {
    err = "cannot open spec " + path;
    return false;
  }
  if (!(file >> spec.nat) || spec.nat <= 0) {
    err = "missing or invalid atom count in " + path;
    return false;
  }
  spec.atomic_numbers.resize(static_cast<std::size_t>(spec.nat));
  for (auto& number : spec.atomic_numbers) {
    if (!(file >> number)) {
      err = "missing atomic number in " + path;
      return false;
    }
  }
  spec.positions.resize(static_cast<std::size_t>(spec.nat * 3));
  for (auto& value : spec.positions) {
    if (!(file >> value)) {
      err = "missing position in " + path;
      return false;
    }
  }
  if (!(file >> spec.molecular_charge) || !(file >> spec.unpaired_electrons) ||
      !(file >> spec.temperature_kelvin) || !(file >> spec.mixer_memory) ||
      !(file >> spec.mixer_damping) || !(file >> spec.maximum_iterations)) {
    err = "missing numeric policy fields in " + path;
    return false;
  }
  std::int64_t npc = 0;
  if (!(file >> npc) || npc < 0) {
    err = "missing point-charge count in " + path;
    return false;
  }
  spec.pc_rows.resize(static_cast<std::size_t>(npc * 5));
  for (auto& value : spec.pc_rows) {
    if (!(file >> value)) {
      err = "missing point-charge value in " + path;
      return false;
    }
  }
  return true;
}

std::string base_name(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  const std::string leaf = slash == std::string::npos ? path : path.substr(slash + 1u);
  const std::size_t dot = leaf.find_last_of('.');
  return dot == std::string::npos ? leaf : leaf.substr(0u, dot);
}

// Temperature-independent geometry, caches, and wavefunction workspace. One
// driver instance; a trajectory advance reuses this object with a new
// geometry and keeps the converged wavefunction as the SCC warm start.
struct Geometry {
  std::int64_t batch_size = 1;
  std::vector<std::int64_t> atom_offsets{0, 0};
  std::vector<std::int64_t> pc_offsets{0, 0};
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions, pc_positions, pc_charges, pc_hardnesses;
  std::vector<double> molecular_charges{0.0};
  std::vector<std::int32_t> unpaired{0};
  std::vector<std::int32_t> spins{1};
  std::uint64_t geometry_generation = 0u;
  std::int64_t total_point_charges = 0;

  BasisPlan basis;
  IntegralPlan integrals;
  H0Plan h0_plan;
  WavefunctionLayout layout;
  ES2Plan es2_plan;
  ES3Plan es3_plan;
  AES2Plan aes2_plan;
  MullikenPlan mulliken_plan;
  EigensolverPlan eig_plan;
  CoordinationPlan coordination_plan;
  ExternalPointChargePlan pc_plan;
  D4Plan d4_plan;

  std::vector<double> overlap, dint, qint, h0, cn;
  std::vector<double> pc_shell_potential;
  std::vector<double> d4_pair_data, d4_coordination;
  AlignedBuffer iscratch{1u << 20}, e2s{1u << 20}, e2scratch{1u << 20}, a2s{1u << 20},
      a2scratch{1u << 20}, d4scratch{1u << 20};
  ES2GeometryCache es2_cache;
  AES2GeometryCache aes2_cache;
  ES2Workspace es2_ws;
  AES2Workspace aes2_ws;
  D4Workspace d4_ws;
  D4GeometryCache d4_cache;

  AlignedBuffer wfn_s{1u << 21};
  WavefunctionView wfn;
  AlignedBuffer oc_s{1u << 20};
  EigensolverOverlapCache ocache;
  AlignedBuffer eig_s{1u << 20};
  EigensolverWorkspace eig_ws;
  CpuLinearAlgebraBackend backend;

  SccDriverGeometryView geom;

  xtbloom_status_t load(const CaseSpec& spec, std::string& err);
  // Advance to a new geometry without disturbing the wavefunction (warm start).
  xtbloom_status_t advance_geometry(const CaseSpec& spec, std::string& err);
};

// Driver stage: mixer + driver plans, state, workspace, and a scratch
// wavefunction used only to isolate eigensolve timing.
struct Stage {
  SccMixerPlan mixer_plan;
  SccDriverPlan driver_plan;
  AlignedBuffer mixer_s{1u << 20};
  SccMixerState mixer_state;
  AlignedBuffer drv_s{1u << 21};
  SccDriverState driver_state;
  AlignedBuffer drvws_s{1u << 21};
  SccDriverWorkspace drv_ws;
  AlignedBuffer timing_wfn_s{1u << 21};
  WavefunctionView timing_wfn;

  xtbloom_status_t build(Geometry& g, const CaseSpec& spec, std::string& err);
  // Re-seed driver/mixer state from the current wavefunction (warm start).
  void reset_driver_state(Geometry& g);
};

// Per-iteration diagnostic snapshot.
struct Snapshot {
  std::uint64_t iteration = 0u;
  std::vector<double> hamiltonian;   // nao*nao row-major (symmetric)
  std::vector<double> coefficients;  // nspin * nao*nao, spin-major row-major
  std::vector<double> eigenvalues;   // nspin * nao
  std::vector<double> occupations;   // 2 * nao (alpha then beta)
  std::vector<double> density;       // nao*nao row-major
  std::uint64_t step_micros = 0u;
  std::uint64_t eigensolve_micros = 0u;
};

xtbloom_status_t Geometry::load(const CaseSpec& spec, std::string& err) {
  err.clear();
  const std::int64_t nat = spec.nat;
  const std::int64_t npc = static_cast<std::int64_t>(spec.pc_rows.size() / 5u);
  total_point_charges = npc;
  atom_offsets = {0, nat};
  pc_offsets = {0, npc};
  atomic_numbers = spec.atomic_numbers;
  positions = spec.positions;
  molecular_charges = {spec.molecular_charge};
  unpaired = {spec.unpaired_electrons};

  xtbloom_status_t s = make_basis_plan(batch_size, nat, atom_offsets.data(),
                                       atomic_numbers.data(), basis, err);
  if (s) return s;
  s = make_integral_plan(basis, integrals, err);
  if (s) return s;
  s = make_h0_plan(basis, integrals, atomic_numbers.data(), h0_plan, err);
  if (s) return s;
  s = make_wavefunction_layout(basis, atomic_numbers.data(), molecular_charges.data(),
                               unpaired.data(), spins.data(), layout, err);
  if (s) return s;
  s = make_es2_plan(basis, atomic_numbers.data(), es2_plan, err);
  if (s) return s;
  s = make_es3_plan(basis, atomic_numbers.data(), es3_plan, err);
  if (s) return s;
  s = make_aes2_plan(basis, atomic_numbers.data(), aes2_plan, err);
  if (s) return s;
  s = make_mulliken_plan(basis, integrals, layout, mulliken_plan, err);
  if (s) return s;
  s = make_eigensolver_plan(layout, eig_plan, err);
  if (s) return s;
  s = make_coordination_plan(batch_size, nat, atom_offsets.data(), atomic_numbers.data(),
                             coordination_plan, err);
  if (s) return s;
  if (npc > 0) {
    s = make_external_point_charge_plan(basis, atomic_numbers.data(), npc, pc_offsets.data(),
                                        pc_plan, err);
    if (s) return s;
    pc_positions.reserve(3u * static_cast<std::size_t>(npc));
    for (std::int64_t ip = 0; ip < npc; ++ip) {
      const double* row = &spec.pc_rows[static_cast<std::size_t>(ip * 5)];
      pc_positions.insert(pc_positions.end(), {row[0], row[1], row[2]});
      pc_charges.push_back(row[3]);
      pc_hardnesses.push_back(row[4]);
    }
  }
  s = make_d4_plan(batch_size, nat, atom_offsets.data(), atomic_numbers.data(), d4_plan, err);
  if (s) return s;

  const std::size_t matrix = static_cast<std::size_t>(integrals.total_matrix_elements);
  overlap.assign(matrix, 0.0);
  dint.assign(3 * matrix, 0.0);
  qint.assign(6 * matrix, 0.0);
  h0.assign(matrix, 0.0);
  cn.assign(static_cast<std::size_t>(nat), 0.0);
  if (iscratch.size < integrals.workspace_size_bytes) return XTBLOOM_STATUS_ALLOCATION_FAILED;
  s = evaluate_coordination_cpu(coordination_plan, positions.data(), cn.data(), err);
  if (s) return s;
  s = evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(), iscratch.data,
                           iscratch.size, err);
  if (s) return s;
  s = evaluate_multipole_cpu(basis, integrals, positions.data(), dint.data(), qint.data(),
                             iscratch.data, iscratch.size, err);
  if (s) return s;
  s = evaluate_h0_cpu(basis, integrals, h0_plan, positions.data(), cn.data(), overlap.data(),
                      h0.data(), err);
  if (s) return s;

  const std::size_t e2n = static_cast<std::size_t>(es2_plan.total_matrix_elements());
  if (e2s.size < e2n * sizeof(double)) return XTBLOOM_STATUS_ALLOCATION_FAILED;
  es2_ws.matrix_scratch = static_cast<double*>(e2scratch.data);
  es2_ws.matrix_elements = es2_plan.total_matrix_elements();
  s = update_es2_geometry_cache_cpu(es2_plan, positions.data(), 1u, static_cast<double*>(e2s.data),
                                    e2n, es2_ws, es2_cache, err);
  if (s) return s;

  const std::size_t a2n = static_cast<std::size_t>(aes2_plan.pair_data_elements());
  if (a2s.size < a2n * sizeof(double)) return XTBLOOM_STATUS_ALLOCATION_FAILED;
  aes2_ws.pair_scratch = static_cast<double*>(a2scratch.data);
  aes2_ws.pair_elements = aes2_plan.pair_data_elements();
  s = update_aes2_geometry_cache_cpu(aes2_plan, positions.data(), cn.data(), 1u,
                                     static_cast<double*>(a2s.data), a2n, aes2_ws, aes2_cache, err);
  if (s) return s;

  if (d4scratch.size < d4_plan.workspace_size_bytes()) return XTBLOOM_STATUS_ALLOCATION_FAILED;
  s = bind_d4_workspace(d4_plan, d4scratch.data, d4scratch.size, d4_ws, err);
  if (s) return s;
  d4_pair_data.assign(static_cast<std::size_t>(d4_plan.total_pairs()) * kD4PairDataElements, 0.0);
  d4_coordination.assign(static_cast<std::size_t>(d4_plan.total_atoms()), 0.0);
  s = update_d4_geometry_cache_cpu(d4_plan, positions.data(), 1u, d4_pair_data.data(),
                                   d4_pair_data.size(), d4_coordination.data(),
                                   d4_coordination.size(), d4_ws, d4_cache, err);
  if (s) return s;

  if (npc > 0) {
    pc_shell_potential.assign(static_cast<std::size_t>(basis.total_shells), 0.0);
    s = evaluate_external_point_charge_potential_cpu(pc_plan, positions.data(), pc_positions.data(),
                                                     pc_charges.data(), pc_hardnesses.data(),
                                                     pc_shell_potential.data(), err);
    if (s) return s;
  }

  geometry_generation = 1u;
  geom.h0 = h0.data();
  geom.h0_elements = static_cast<std::int64_t>(h0.size());
  geom.integrals = {overlap.data(), dint.data(), qint.data(), integrals.total_matrix_elements,
                    mulliken_plan.identity()};
  geom.es2_cache = es2_cache;
  geom.aes2_cache = aes2_cache;
  geom.d4_cache = d4_cache;
  geom.geometry_generation = geometry_generation;
  if (npc > 0) {
    geom.explicit_point_charge_shell_potential = pc_shell_potential.data();
    geom.explicit_point_charge_shell_elements = static_cast<std::int64_t>(pc_shell_potential.size());
  }

  s = make_mkl_rt_lp64_backend(backend, err);
  if (s) return s;
  if (wfn_s.size < layout.workspace_size_bytes) return XTBLOOM_STATUS_ALLOCATION_FAILED;
  s = bind_wavefunction_view(layout, wfn_s.data, wfn_s.size, wfn, err);
  if (s) return s;
  // tblite's new_wavefunction leaves the SCC perturbative q/d/Q at exactly
  // zero; this seed reproduces the pinned corpus trajectories.
  s = initialize_sad_multipole_state(layout, wfn, err);
  if (s) return s;
  std::fill_n(wfn.qsh, static_cast<std::size_t>(layout.qsh.element_count), 0.0);
  std::fill_n(wfn.qat, static_cast<std::size_t>(layout.qat.element_count), 0.0);
  std::fill_n(wfn.dipole, static_cast<std::size_t>(layout.dipole.element_count), 0.0);
  std::fill_n(wfn.quadrupole, static_cast<std::size_t>(layout.quadrupole.element_count), 0.0);

  if (oc_s.size < eig_plan.overlap_cache_size_bytes()) return XTBLOOM_STATUS_ALLOCATION_FAILED;
  s = bind_eigensolver_overlap_cache(eig_plan, oc_s.data, oc_s.size, ocache, err);
  if (s) return s;
  if (eig_s.size < eig_plan.workspace_size_bytes()) return XTBLOOM_STATUS_ALLOCATION_FAILED;
  s = bind_eigensolver_workspace(eig_plan, eig_s.data, eig_s.size, eig_ws, err);
  if (s) return s;
  s = factor_overlap_cpu(eig_plan, overlap.data(), geometry_generation, backend, eig_ws, ocache,
                         err);
  if (s) return s;
  err.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t Geometry::advance_geometry(const CaseSpec& spec, std::string& err) {
  err.clear();
  if (spec.nat != static_cast<std::int64_t>(atomic_numbers.size()) ||
      spec.atomic_numbers != atomic_numbers || spec.molecular_charge != molecular_charges[0] ||
      spec.unpaired_electrons != unpaired[0]) {
    err = "trajectory second geometry must have identical atoms, charge, and spin";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (geometry_generation == 0u) {
    err = "advance_geometry requires a loaded first geometry";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::int64_t npc = static_cast<std::int64_t>(spec.pc_rows.size() / 5u);
  if (npc != total_point_charges) {
    err = "trajectory second geometry must keep the point-charge set";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  ++geometry_generation;
  if (geometry_generation == 0u) {
    geometry_generation = 1u;
  }
  positions = spec.positions;
  pc_positions.clear();
  pc_charges.clear();
  pc_hardnesses.clear();
  for (std::int64_t ip = 0; ip < npc; ++ip) {
    const double* row = &spec.pc_rows[static_cast<std::size_t>(ip * 5)];
    pc_positions.insert(pc_positions.end(), {row[0], row[1], row[2]});
    pc_charges.push_back(row[3]);
    pc_hardnesses.push_back(row[4]);
  }

  xtbloom_status_t s = evaluate_coordination_cpu(coordination_plan, positions.data(), cn.data(),
                                                 err);
  if (s) return s;
  s = evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(), iscratch.data,
                           iscratch.size, err);
  if (s) return s;
  s = evaluate_multipole_cpu(basis, integrals, positions.data(), dint.data(), qint.data(),
                             iscratch.data, iscratch.size, err);
  if (s) return s;
  s = evaluate_h0_cpu(basis, integrals, h0_plan, positions.data(), cn.data(), overlap.data(),
                      h0.data(), err);
  if (s) return s;

  const std::size_t e2n = static_cast<std::size_t>(es2_plan.total_matrix_elements());
  s = update_es2_geometry_cache_cpu(es2_plan, positions.data(), geometry_generation,
                                    static_cast<double*>(e2s.data), e2n, es2_ws, es2_cache, err);
  if (s) return s;
  const std::size_t a2n = static_cast<std::size_t>(aes2_plan.pair_data_elements());
  s = update_aes2_geometry_cache_cpu(aes2_plan, positions.data(), cn.data(), geometry_generation,
                                     static_cast<double*>(a2s.data), a2n, aes2_ws, aes2_cache, err);
  if (s) return s;
  d4_coordination.assign(static_cast<std::size_t>(d4_plan.total_atoms()), 0.0);
  s = update_d4_geometry_cache_cpu(d4_plan, positions.data(), geometry_generation,
                                   d4_pair_data.data(), d4_pair_data.size(),
                                   d4_coordination.data(), d4_coordination.size(), d4_ws, d4_cache,
                                   err);
  if (s) return s;
  if (npc > 0) {
    s = evaluate_external_point_charge_potential_cpu(pc_plan, positions.data(), pc_positions.data(),
                                                     pc_charges.data(), pc_hardnesses.data(),
                                                     pc_shell_potential.data(), err);
    if (s) return s;
  }
  s = factor_overlap_cpu(eig_plan, overlap.data(), geometry_generation, backend, eig_ws, ocache,
                         err);
  if (s) return s;

  geom.h0 = h0.data();
  geom.h0_elements = static_cast<std::int64_t>(h0.size());
  geom.integrals = {overlap.data(), dint.data(), qint.data(), integrals.total_matrix_elements,
                    mulliken_plan.identity()};
  geom.es2_cache = es2_cache;
  geom.aes2_cache = aes2_cache;
  geom.d4_cache = d4_cache;
  geom.geometry_generation = geometry_generation;
  if (npc > 0) {
    geom.explicit_point_charge_shell_potential = pc_shell_potential.data();
    geom.explicit_point_charge_shell_elements = static_cast<std::int64_t>(pc_shell_potential.size());
  }
  err.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t Stage::build(Geometry& g, const CaseSpec& spec, std::string& err) {
  err.clear();
  xtbloom_status_t s =
      make_scc_mixer_plan(g.layout, spec.mixer_memory, spec.mixer_damping, kTbliteRmsTolerance,
                          kTbliteRmsTolerance, mixer_plan, err);
  if (s) return s;
  s = make_scc_driver_plan(g.layout, g.mulliken_plan, g.es2_plan, g.es3_plan, g.aes2_plan,
                           g.eig_plan, mixer_plan, &g.d4_plan, nullptr,
                           static_cast<std::uint64_t>(spec.maximum_iterations),
                           spec.temperature_kelvin * kKelvinToHartree, kTbliteEnergyTolerance,
                           driver_plan, err);
  if (s) return s;
  if (mixer_s.size < mixer_plan.state_size_bytes()) return XTBLOOM_STATUS_ALLOCATION_FAILED;
  s = bind_scc_mixer_state(mixer_plan, mixer_s.data, mixer_s.size, mixer_state, err);
  if (s) return s;
  if (drv_s.size < driver_plan.state_size_bytes()) return XTBLOOM_STATUS_ALLOCATION_FAILED;
  s = bind_scc_driver_state(driver_plan, drv_s.data, drv_s.size, driver_state, err);
  if (s) return s;
  if (drvws_s.size < driver_plan.workspace_size_bytes()) return XTBLOOM_STATUS_ALLOCATION_FAILED;
  s = bind_scc_driver_workspace(driver_plan, drvws_s.data, drvws_s.size, drv_ws, err);
  if (s) return s;
  if (timing_wfn_s.size < g.layout.workspace_size_bytes) return XTBLOOM_STATUS_ALLOCATION_FAILED;
  s = bind_wavefunction_view(g.layout, timing_wfn_s.data, timing_wfn_s.size, timing_wfn, err);
  if (s) return s;
  s = initialize_scc_driver_state_cpu(driver_plan, g.wfn, mixer_state, driver_state, err);
  if (s) return s;
  err.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

void Stage::reset_driver_state(Geometry& g) {
  std::string ignored;
  initialize_scc_driver_state_cpu(driver_plan, g.wfn, mixer_state, driver_state, ignored);
}

void emit_int(std::ostream& out, std::int64_t value) { out << value << "\n"; }
void emit_value(std::ostream& out, double value) {
  out << std::setprecision(17) << std::scientific << value << "\n";
}

void emit_case(std::ostream& out, const CaseSpec& spec) {
  out << "case " << spec.name << "\n";
  out << "nat " << spec.nat << "\n";
  out << "atomic_numbers\n";
  for (std::int32_t z : spec.atomic_numbers) emit_int(out, z);
  out << "positions\n";
  for (double v : spec.positions) emit_value(out, v);
  out << "molecular_charge\n";
  emit_value(out, spec.molecular_charge);
  out << "unpaired_electrons\n";
  emit_int(out, spec.unpaired_electrons);
  out << "temperature_kelvin\n";
  emit_value(out, spec.temperature_kelvin);
  out << "maximum_iterations\n";
  emit_int(out, spec.maximum_iterations);
}

// Run the current geometry to a terminal state, streaming the diagnostic
// document for this geometry to out. Returns the completed iteration count or
// -1 on a structural failure.
std::int64_t run_geometry(Geometry& g, Stage& st, std::ostream& out, std::string& err) {
  const std::int64_t nao =
      g.layout.eigenvalues.system_offsets[1] - g.layout.eigenvalues.system_offsets[0];
  const std::int64_t matrix = nao * nao;
  const std::int64_t nspin = g.spins[0];
  const std::int64_t matrix_begin = g.mulliken_plan.matrix_offsets()[0];
  const std::int64_t occ_begin = g.layout.occupations.system_offsets[0];
  const double etemp = st.driver_plan.electronic_temperature();

  out << "geometry " << g.geometry_generation << "\n";
  out << "nao " << nao << "\n";
  out << "nspin " << nspin << "\n";
  out << "overlap\n";
  for (std::int64_t r = 0; r < nao; ++r)
    for (std::int64_t c = 0; c < nao; ++c)
      emit_value(out, g.overlap[static_cast<std::size_t>(matrix_begin + r * nao + c)]);
  out << "core_hamiltonian\n";
  for (std::int64_t r = 0; r < nao; ++r)
    for (std::int64_t c = 0; c < nao; ++c)
      emit_value(out, g.h0[static_cast<std::size_t>(matrix_begin + c * nao + r)]);

  std::int64_t completed = 0;
  for (std::uint64_t total = 0u; total < st.driver_plan.maximum_iterations(); ++total) {
    const bool active = st.driver_state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS &&
                        st.driver_state.converged[0] == 0u;
    if (!active) break;
    const auto t0 = std::chrono::steady_clock::now();
    const xtbloom_status_t step_status = iterate_scc_driver_batch_cpu(
        st.driver_plan, g.geom, g.backend, g.ocache, g.wfn, st.mixer_state, st.driver_state,
        st.drv_ws, err);
    const auto t1 = std::chrono::steady_clock::now();
    if (step_status != XTBLOOM_STATUS_SUCCESS && step_status != XTBLOOM_STATUS_SCC_NOT_CONVERGED &&
        step_status != XTBLOOM_STATUS_EIGENSOLVER_FAILED &&
        step_status != XTBLOOM_STATUS_INTERNAL_ERROR) {
      err = "driver structural failure: " + err;
      return -1;
    }
    if (st.driver_state.system_statuses[0] != XTBLOOM_STATUS_SUCCESS &&
        st.driver_state.system_statuses[0] != XTBLOOM_STATUS_SCC_NOT_CONVERGED) {
      break;
    }

    Snapshot snap;
    snap.iteration = st.driver_state.iterations[0];
    snap.step_micros = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    snap.hamiltonian.assign(st.drv_ws.hamiltonian + matrix_begin,
                            st.drv_ws.hamiltonian + matrix_begin + matrix);
    const std::int64_t coeff_base = g.layout.coefficients.system_offsets[0];
    const std::int64_t eig_base = g.layout.eigenvalues.system_offsets[0];
    const std::int64_t density_base = g.layout.density.system_offsets[0];
    snap.coefficients.assign(
        st.drv_ws.staged_wavefunction.coefficients + coeff_base,
        st.drv_ws.staged_wavefunction.coefficients + coeff_base + nspin * matrix);
    snap.eigenvalues.assign(st.drv_ws.staged_wavefunction.eigenvalues + eig_base,
                            st.drv_ws.staged_wavefunction.eigenvalues + eig_base + nspin * nao);
    snap.occupations.assign(st.drv_ws.staged_wavefunction.occupations + occ_begin,
                            st.drv_ws.staged_wavefunction.occupations + occ_begin + 2 * nao);
    snap.density.assign(st.drv_ws.staged_wavefunction.density + density_base,
                        st.drv_ws.staged_wavefunction.density + density_base + matrix);

    // Isolated eigensolve-only timing on the same H_k via the production
    // solver. Uses the factored overlap cache and a scratch wavefunction so
    // driver state is never disturbed.
    xtbloom_status_t solve_statuses[1]{XTBLOOM_STATUS_SUCCESS};
    double cps[2]{0.0, 0.0};
    double entropies[1]{0.0}, bands[1]{0.0}, frees[1]{0.0};
    EigensolverThermodynamicsView therm;
    therm.system_statuses = solve_statuses;
    therm.system_status_capacity = 1u;
    therm.chemical_potentials = cps;
    therm.chemical_potential_capacity = 2u;
    therm.entropies = entropies;
    therm.entropy_capacity = 1u;
    therm.band_energies = bands;
    therm.band_energy_capacity = 1u;
    therm.free_energies = frees;
    therm.free_energy_capacity = 1u;
    std::string solve_error;
    const auto s0 = std::chrono::steady_clock::now();
    const xtbloom_status_t solve_status = solve_eigensystem_cpu(
        g.eig_plan, 0, g.ocache, g.geometry_generation, st.drv_ws.hamiltonian + matrix_begin,
        etemp, g.backend, g.eig_ws, st.timing_wfn, therm, solve_error);
    const auto s1 = std::chrono::steady_clock::now();
    snap.eigensolve_micros = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(s1 - s0).count());
    if (solve_status != XTBLOOM_STATUS_SUCCESS) {
      err = "diagnostic eigensolve timing call failed: " + solve_error;
      return -1;
    }

    out << "iteration " << snap.iteration << "\n";
    out << "step_micros\n";
    emit_int(out, static_cast<std::int64_t>(snap.step_micros));
    out << "eigensolve_micros\n";
    emit_int(out, static_cast<std::int64_t>(snap.eigensolve_micros));
    out << "hamiltonian\n";
    for (std::int64_t r = 0; r < nao; ++r)
      for (std::int64_t c = 0; c < nao; ++c)
        emit_value(out, snap.hamiltonian[static_cast<std::size_t>(r * nao + c)]);
    out << "coefficients\n";
    for (std::int64_t ch = 0; ch < nspin; ++ch)
      for (std::int64_t r = 0; r < nao; ++r)
        for (std::int64_t c = 0; c < nao; ++c)
          emit_value(out, snap.coefficients[static_cast<std::size_t>(ch * matrix + r * nao + c)]);
    out << "eigenvalues\n";
    for (double v : snap.eigenvalues) emit_value(out, v);
    out << "occupations\n";
    for (double v : snap.occupations) emit_value(out, v);
    out << "density\n";
    for (std::int64_t r = 0; r < nao; ++r)
      for (std::int64_t c = 0; c < nao; ++c)
        emit_value(out, snap.density[static_cast<std::size_t>(r * nao + c)]);
    ++completed;
  }

  // Terminal eigenpairs as committed in the live wavefunction.
  const std::int64_t coeff_base = g.layout.coefficients.system_offsets[0];
  const std::int64_t eig_base = g.layout.eigenvalues.system_offsets[0];
  const std::int64_t density_base = g.layout.density.system_offsets[0];
  const std::int64_t occ_base = g.layout.occupations.system_offsets[0];
  out << "converged\n";
  out << "coefficients\n";
  for (std::int64_t ch = 0; ch < nspin; ++ch)
    for (std::int64_t r = 0; r < nao; ++r)
      for (std::int64_t c = 0; c < nao; ++c)
        emit_value(out, g.wfn.coefficients[static_cast<std::size_t>(ch * matrix + r * nao + c)]);
  out << "eigenvalues\n";
  for (std::int64_t i = 0; i < nspin * nao; ++i)
    emit_value(out, g.wfn.eigenvalues[static_cast<std::size_t>(eig_base + i)]);
  out << "occupations\n";
  for (std::int64_t i = 0; i < 2 * nao; ++i)
    emit_value(out, g.wfn.occupations[static_cast<std::size_t>(occ_base + i)]);
  out << "density\n";
  for (std::int64_t r = 0; r < nao; ++r)
    for (std::int64_t c = 0; c < nao; ++c)
      emit_value(out, g.wfn.density[static_cast<std::size_t>(density_base + r * nao + c)]);
  out << "converged_state ";
  emit_int(out, st.driver_state.converged[0] != 0u ? 1 : 0);

  err.clear();
  return completed;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " single <case.spec> [out]\n"
              << "       " << argv[0] << " traj <case.spec> <second.spec> [out]\n";
    return 64;
  }
  const std::string mode = argv[1];
  const std::string first_path = argv[2];
  std::string second_path;
  int trailing_index = 0;
  if (mode == "single") {
    if (argc != 3 && argc != 4) return 64;
    trailing_index = 3;
  } else if (mode == "traj") {
    if (argc != 4 && argc != 5) return 64;
    second_path = argv[3];
    trailing_index = 4;
  } else {
    std::cerr << "unknown mode " << mode << "\n";
    return 64;
  }

  CaseSpec first;
  std::string err;
  if (!load_spec(first_path, first, err)) {
    std::cerr << err << "\n";
    return 2;
  }
  first.name = base_name(first_path);
  CaseSpec second;
  if (mode == "traj" && !load_spec(second_path, second, err)) {
    std::cerr << err << "\n";
    return 2;
  }
  if (mode == "traj") {
    second.name = base_name(second_path);
  }

  std::ostream* out = &std::cout;
  std::ofstream file;
  const std::string out_path = trailing_index < argc ? argv[trailing_index] : "";
  if (!out_path.empty()) {
    file.open(out_path);
    if (!file) {
      std::cerr << "cannot open output " << out_path << "\n";
      return 2;
    }
    out = &file;
  }

  Geometry g;
  if (g.load(first, err) != XTBLOOM_STATUS_SUCCESS) {
    std::cerr << "geometry build failed: " << err << "\n";
    return 100;
  }
  Stage st;
  if (st.build(g, first, err) != XTBLOOM_STATUS_SUCCESS) {
    std::cerr << "stage build failed: " << err << "\n";
    return 100;
  }

  (*out) << "diagnostic xtbloom-scc-reuse-v1\n";
  emit_case(*out, first);

  const std::int64_t first_iters = run_geometry(g, st, *out, err);
  if (first_iters < 0) {
    std::cerr << "single-geometry run failed: " << err << "\n";
    return 100;
  }

  if (mode == "traj") {
    if (g.advance_geometry(second, err) != XTBLOOM_STATUS_SUCCESS) {
      std::cerr << "trajectory advance failed: " << err << "\n";
      return 2;
    }
    // Warm start: keep the converged wavefunction (multipoles seed the next
    // SCC), reinitialize driver/mixer state from it.
    st.reset_driver_state(g);
    emit_case(*out, second);
    const std::int64_t second_iters = run_geometry(g, st, *out, err);
    if (second_iters < 0) {
      std::cerr << "second-geometry run failed: " << err << "\n";
      return 100;
    }
    if (second_iters == 0) {
      std::cerr << "warm-start second geometry completed no iterations\n";
      return 100;
    }
  }
  (*out) << "end-of-diagnostics terminal single_iterations=" << first_iters << "\n";
  return 0;
}