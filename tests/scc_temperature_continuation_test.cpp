// Investigation gate for issue #217: temperature-continuation SCC for the
// separated Me4N+ / Cl- ion pair (tmacl, 18 atoms).
//
// This test is the reproducible committed tooling behind issue #217. It
// drives the internal CPU GFN2 SCC driver on the exact tmacl geometry from
// grimme-lab/xtb issue #678 (see data/conformance/inputs/tmacl.xyz) and
// records per-iteration diagnostics so the different failure modes are
// distinguishable:
//
//   * charge sloshing:   |q(Cl)| oscillates between ~0.9 and ~0.2 while the
//                        residual RMS stays near 1e-2 .. 1e-1 over many
//                        iterations (fresh 300 K solve under the default
//                        Johnson modified-Broyden policy, history 8,
//                        damping 0.4);
//   * mixer/limit cycle: the free energy repeats a short periodic pattern
//                        instead of converging (400 K fresh and coarse
//                        continuation jumps);
//   * occupation effect: raising the electronic temperature to 450 K or above
//                        lets the same policy converge to a localized
//                        stationary state.
//
// The key scientific finding pinned here: a reproducible 300 K localized
// stationary state exists (internal SCC free energy -22.271821505 Eh,
// q(Me4N+) = +0.8285, q(Cl) = -0.8285) and is reachable either by bounded
// temperature continuation (funnel) or by bounded deterministic mixer
// policies, while the default policy seeded from the SAD guess does not
// escape charge sloshing at 300 K.
//
// All energies below are the internal SCC electronic free energy from the
// driver trace (band + SCC interactions - T*S), which excludes geometric
// repulsion and the D4 ATM term that the public total energy adds as a fixed
// offset (~+0.25947534 Eh on this geometry).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gpuxtb/gpuxtb.h"
#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/eigensolver.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/es3.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/scc_driver.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"

#ifndef GPUXTB_TMACL_FIXTURE_PATH
#error "GPUXTB_TMACL_FIXTURE_PATH must name the committed tmacl.xyz fixture"
#endif

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      std::cerr << "CHECK failed at line " << __LINE__ << ": " << #condition << "\n"; \
      return __LINE__;                                                                \
    }                                                                                 \
  } while (false)

namespace {

using namespace gpuxtb::detail::gfn2;

constexpr double kAngPerBohr = 0.529177210903;
constexpr double kKelvinToHartree = 3.166808578545117e-6;
constexpr double kDefaultRmsTolerance = 1.0e-6;
constexpr double kDefaultEnergyTolerance = 1.0e-8;

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

// Temperature-independent geometry and wavefunction storage. The same
// wavefunction buffer is reused across continuation stages so a converged
// state can seed the next, lower-temperature driver.
struct Geometry {
  std::int64_t batch_size = 1;
  std::vector<std::int64_t> atom_offsets{0, 0};
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<double> molecular_charges{0.0};
  std::vector<std::int32_t> unpaired{0};
  std::vector<std::int32_t> spins{1};

  BasisPlan basis;
  IntegralPlan integrals;
  H0Plan h0_plan;
  WavefunctionLayout layout;
  ES2Plan es2_plan;
  ES3Plan es3_plan;
  AES2Plan aes2_plan;
  MullikenPlan mulliken_plan;
  EigensolverPlan eig_plan;
  D4Plan d4_plan;
  CoordinationPlan coordination_plan;

  std::vector<double> overlap, dint, qint, h0, cn;
  std::vector<double> d4_pair_data, d4_coordination;
  AlignedBuffer iscratch{1u << 20}, e2s{1u << 20}, e2scratch{1u << 20}, a2s{1u << 20},
      a2scratch{1u << 20}, d4scratch{1u << 20};
  ES2GeometryCache es2_cache;
  AES2GeometryCache aes2_cache;
  ES2Workspace es2_ws;
  AES2Workspace aes2_ws;
  D4Workspace d4_ws;
  D4GeometryCache d4_cache;

  AlignedBuffer wfn_s{1u << 20};
  WavefunctionView wfn;
  AlignedBuffer oc_s{1u << 20};
  EigensolverOverlapCache ocache;
  AlignedBuffer eig_s{1u << 20};
  EigensolverWorkspace eig_ws;
  CpuLinearAlgebraBackend backend;

  SccDriverGeometryView geom;

  gpuxtb_status_t build(std::string& err);
};

struct Stage {
  SccMixerPlan mixer_plan;
  SccDriverPlan driver_plan;
  AlignedBuffer mixer_s{1u << 20};
  SccMixerState mixer_state;
  AlignedBuffer drv_s{1u << 20};
  SccDriverState driver_state;
  AlignedBuffer drvws_s{1u << 20};
  SccDriverWorkspace drv_ws;

  gpuxtb_status_t build(Geometry& g, std::int64_t history, double damping, double rms_tol,
                        double e_tol, std::uint64_t max_iter, double etemp, std::string& err);
};

struct TraceRow {
  std::uint64_t iter;
  double free_energy;
  double d_free;
  double rms;
  double max_res;
  double q_me4n;
  double q_cl;
  std::uint64_t restarts;
  int status;
  int converged;
};

struct ElectronicState {
  std::vector<double> shell_charges;
  std::vector<double> atomic_charges;
  std::vector<double> atomic_dipoles;
  std::vector<double> atomic_quadrupoles;
  std::vector<double> density;
};

struct PolicyRow {
  double kelvin;
  std::int64_t history;
  double damping;
  bool converged;
  TraceRow terminal;
  ElectronicState state;
};

gpuxtb_status_t Geometry::build(std::string& err) {
  err.clear();
  const std::int64_t natoms = static_cast<std::int64_t>(atomic_numbers.size());
  gpuxtb_status_t s =
      make_basis_plan(batch_size, natoms, atom_offsets.data(), atomic_numbers.data(), basis, err);
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
  s = make_coordination_plan(batch_size, natoms, atom_offsets.data(), atomic_numbers.data(),
                             coordination_plan, err);
  if (s) return s;
  s = make_d4_plan(batch_size, natoms, atom_offsets.data(), atomic_numbers.data(), d4_plan, err);
  if (s) return s;

  const std::size_t matrix = static_cast<std::size_t>(integrals.total_matrix_elements);
  overlap.assign(matrix, 0.0);
  dint.assign(3 * matrix, 0.0);
  qint.assign(6 * matrix, 0.0);
  h0.assign(matrix, 0.0);
  cn.assign(static_cast<std::size_t>(natoms), 0.0);
  if (iscratch.size < integrals.workspace_size_bytes) return GPUXTB_STATUS_ALLOCATION_FAILED;
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
  if (e2s.size < e2n * sizeof(double)) return GPUXTB_STATUS_ALLOCATION_FAILED;
  es2_ws.matrix_scratch = static_cast<double*>(e2scratch.data);
  es2_ws.matrix_elements = es2_plan.total_matrix_elements();
  s = update_es2_geometry_cache_cpu(es2_plan, positions.data(), 1u, static_cast<double*>(e2s.data),
                                    e2n, es2_ws, es2_cache, err);
  if (s) return s;

  const std::size_t a2n = static_cast<std::size_t>(aes2_plan.pair_data_elements());
  if (a2s.size < a2n * sizeof(double)) return GPUXTB_STATUS_ALLOCATION_FAILED;
  aes2_ws.pair_scratch = static_cast<double*>(a2scratch.data);
  aes2_ws.pair_elements = aes2_plan.pair_data_elements();
  s = update_aes2_geometry_cache_cpu(aes2_plan, positions.data(), cn.data(), 1u,
                                     static_cast<double*>(a2s.data), a2n, aes2_ws, aes2_cache, err);
  if (s) return s;

  geom.h0 = h0.data();
  geom.h0_elements = static_cast<std::int64_t>(h0.size());
  geom.integrals = {overlap.data(), dint.data(), qint.data(), integrals.total_matrix_elements,
                    mulliken_plan.identity()};
  geom.es2_cache = es2_cache;
  geom.aes2_cache = aes2_cache;
  geom.geometry_generation = 1u;

  s = make_mkl_rt_lp64_backend(backend, err);
  if (s) return s;
  if (wfn_s.size < layout.workspace_size_bytes) return GPUXTB_STATUS_ALLOCATION_FAILED;
  s = bind_wavefunction_view(layout, wfn_s.data, wfn_s.size, wfn, err);
  if (s) return s;
  s = initialize_sad_multipole_state(layout, wfn, err);
  if (s) return s;
  if (oc_s.size < eig_plan.overlap_cache_size_bytes()) return GPUXTB_STATUS_ALLOCATION_FAILED;
  s = bind_eigensolver_overlap_cache(eig_plan, oc_s.data, oc_s.size, ocache, err);
  if (s) return s;
  if (eig_s.size < eig_plan.workspace_size_bytes()) return GPUXTB_STATUS_ALLOCATION_FAILED;
  s = bind_eigensolver_workspace(eig_plan, eig_s.data, eig_s.size, eig_ws, err);
  if (s) return s;
  s = factor_overlap_cpu(eig_plan, overlap.data(), 1u, backend, eig_ws, ocache, err);
  if (s) return s;

  if (d4scratch.size < d4_plan.workspace_size_bytes()) return GPUXTB_STATUS_ALLOCATION_FAILED;
  s = bind_d4_workspace(d4_plan, d4scratch.data, d4scratch.size, d4_ws, err);
  if (s) return s;
  const std::size_t pair_elements =
      static_cast<std::size_t>(d4_plan.total_pairs()) * kD4PairDataElements;
  d4_pair_data.assign(std::max<std::size_t>(pair_elements, 1u), 0.0);
  d4_coordination.assign(static_cast<std::size_t>(natoms), 0.0);
  s = update_d4_geometry_cache_cpu(d4_plan, positions.data(), 1u, d4_pair_data.data(),
                                   d4_pair_data.size(), d4_coordination.data(),
                                   d4_coordination.size(), d4_ws, d4_cache, err);
  if (s) return s;
  geom.d4_cache = d4_cache;
  return GPUXTB_STATUS_SUCCESS;
}

gpuxtb_status_t Stage::build(Geometry& g, std::int64_t history, double damping, double rms_tol,
                             double e_tol, std::uint64_t max_iter, double etemp, std::string& err) {
  err.clear();
  gpuxtb_status_t s =
      make_scc_mixer_plan(g.layout, history, damping, rms_tol, rms_tol, mixer_plan, err);
  if (s) return s;
  s = make_scc_driver_plan(g.layout, g.mulliken_plan, g.es2_plan, g.es3_plan, g.aes2_plan,
                           g.eig_plan, mixer_plan, &g.d4_plan, nullptr, max_iter, etemp, e_tol,
                           driver_plan, err);
  if (s) return s;
  if (mixer_s.size < mixer_plan.state_size_bytes()) return GPUXTB_STATUS_ALLOCATION_FAILED;
  s = bind_scc_mixer_state(mixer_plan, mixer_s.data, mixer_s.size, mixer_state, err);
  if (s) return s;
  if (drv_s.size < driver_plan.state_size_bytes()) return GPUXTB_STATUS_ALLOCATION_FAILED;
  s = bind_scc_driver_state(driver_plan, drv_s.data, drv_s.size, driver_state, err);
  if (s) return s;
  if (drvws_s.size < driver_plan.workspace_size_bytes()) return GPUXTB_STATUS_ALLOCATION_FAILED;
  s = bind_scc_driver_workspace(driver_plan, drvws_s.data, drvws_s.size, drv_ws, err);
  if (s) return s;
  s = initialize_scc_driver_state_cpu(driver_plan, g.wfn, mixer_state, driver_state, err);
  return s;
}

struct FragmentCharges {
  double me4n;  // sum over atoms 0..16
  double cl;    // atom 17
};

FragmentCharges fragment_charges(const WavefunctionView& wfn) {
  FragmentCharges out{0.0, 0.0};
  for (std::int32_t i = 0; i < 17; ++i) {
    out.me4n += wfn.qat[i];
  }
  out.cl = wfn.qat[17];
  return out;
}

std::vector<double> copy_field(const double* values, std::int64_t count) {
  return std::vector<double>(values, values + static_cast<std::size_t>(count));
}

ElectronicState capture_state(const Geometry& geometry) {
  return {
      copy_field(geometry.wfn.qsh, geometry.layout.qsh.element_count),
      copy_field(geometry.wfn.qat, geometry.layout.qat.element_count),
      copy_field(geometry.wfn.dipole, geometry.layout.dipole.element_count),
      copy_field(geometry.wfn.quadrupole, geometry.layout.quadrupole.element_count),
      copy_field(geometry.wfn.density, geometry.layout.density.element_count),
  };
}

double maximum_difference(const std::vector<double>& lhs, const std::vector<double>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double maximum = 0.0;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    // std::max does not propagate NaNs portably; reject them explicitly so a
    // damaged terminal state cannot compare equal to a finite reference.
    if (!std::isfinite(lhs[index]) || !std::isfinite(rhs[index])) {
      return std::numeric_limits<double>::infinity();
    }
    maximum = std::max(maximum, std::abs(lhs[index] - rhs[index]));
  }
  return maximum;
}

double maximum_multipole_difference(const ElectronicState& lhs, const ElectronicState& rhs) {
  return std::max({maximum_difference(lhs.shell_charges, rhs.shell_charges),
                   maximum_difference(lhs.atomic_charges, rhs.atomic_charges),
                   maximum_difference(lhs.atomic_dipoles, rhs.atomic_dipoles),
                   maximum_difference(lhs.atomic_quadrupoles, rhs.atomic_quadrupoles)});
}

void write_trace(std::ostream& output, const std::string& title,
                 const std::vector<TraceRow>& trace) {
  output << "\n==== " << title << " ====\n"
         << "  it        free_energy        d_free          rms          max_res"
            "     q(Me4N+)   q(Cl)  restarts status conv\n";
  output << std::fixed;
  for (const TraceRow& row : trace) {
    output << std::setw(4) << row.iter << "  " << std::setw(17) << std::setprecision(12)
           << row.free_energy << "  " << std::scientific << std::setw(13) << std::setprecision(6)
           << row.d_free << "  " << std::setw(13) << row.rms << "  " << std::setw(13) << row.max_res
           << std::fixed << "  " << std::setw(8) << std::setprecision(5) << row.q_me4n << " "
           << std::setw(8) << row.q_cl << " " << std::setw(6) << row.restarts << " " << std::setw(7)
           << row.status << " " << row.converged << '\n';
  }
}

bool write_trace_file(const std::string& directory, const std::string& filename,
                      const std::string& title, const std::vector<TraceRow>& trace,
                      std::string& err) {
  std::ofstream output(directory + "/" + filename, std::ios::trunc);
  if (!output) {
    err = "cannot open evidence output " + directory + "/" + filename;
    return false;
  }
  write_trace(output, title, trace);
  return static_cast<bool>(output);
}

// Run one stage to convergence or the iteration ceiling. Returns true when
// the driver reported a converged fixed point.
bool run_stage(Geometry& g, Stage& st, std::uint64_t max_iter, std::vector<TraceRow>& trace,
               std::string& err) {
  trace.clear();
  for (std::uint64_t it = 0; it < max_iter; ++it) {
    const gpuxtb_status_t s =
        iterate_scc_driver_batch_cpu(st.driver_plan, g.geom, g.backend, g.ocache, g.wfn,
                                     st.mixer_state, st.driver_state, st.drv_ws, err);
    if (s != GPUXTB_STATUS_SUCCESS && s != GPUXTB_STATUS_SCC_NOT_CONVERGED) {
      const std::string detail = err;
      err = "unexpected SCC driver status " + std::to_string(static_cast<int>(s));
      if (!detail.empty()) {
        err += ": " + detail;
      }
      trace.clear();
      return false;
    }
    TraceRow row{};
    row.iter = st.driver_state.iterations[0];
    row.free_energy = st.driver_state.free_energies[0];
    row.d_free = st.driver_state.free_energy_changes[0];
    row.rms = st.mixer_state.residual_rms[0];
    row.max_res = st.mixer_state.residual_maximum[0];
    const FragmentCharges frag = fragment_charges(g.wfn);
    row.q_me4n = frag.me4n;
    row.q_cl = frag.cl;
    row.restarts = st.mixer_state.restart_counts[0];
    row.status = static_cast<int>(st.driver_state.system_statuses[0]);
    row.converged = static_cast<int>(st.driver_state.converged[0]);
    if (!std::isfinite(row.free_energy) || !std::isfinite(row.d_free) || !std::isfinite(row.rms) ||
        !std::isfinite(row.max_res) || !std::isfinite(row.q_me4n) || !std::isfinite(row.q_cl)) {
      err = "SCC driver produced a non-finite trace diagnostic";
      trace.clear();
      return false;
    }
    trace.push_back(row);
    if (row.status != 0 || row.converged) {
      break;
    }
  }
  if (trace.empty()) {
    return false;
  }
  const TraceRow& last = trace.back();
  return last.converged == 1;
}

bool load_tmacl(std::vector<std::int32_t>& numbers, std::vector<double>& positions,
                std::string& err) {
  std::ifstream file(GPUXTB_TMACL_FIXTURE_PATH);
  if (!file) {
    err = "cannot open tmacl fixture";
    return false;
  }
  std::string line;
  if (!std::getline(file, line)) {
    err = "tmacl fixture is missing its atom count";
    return false;
  }
  std::istringstream count_stream(line);
  std::int64_t declared_atoms = 0;
  std::string trailing;
  if (!(count_stream >> declared_atoms) || declared_atoms != 18 || count_stream >> trailing) {
    err = "tmacl fixture must declare exactly 18 atoms";
    return false;
  }
  if (!std::getline(file, line) || line != "tmacl") {
    err = "tmacl fixture title must be exactly 'tmacl'";
    return false;
  }
  numbers.clear();
  positions.clear();
  for (std::int64_t atom = 0; atom < declared_atoms; ++atom) {
    if (!std::getline(file, line)) {
      err = "tmacl fixture ended before all atoms were read";
      return false;
    }
    std::istringstream in(line);
    std::string symbol;
    double x, y, z;
    if (!(in >> symbol >> x >> y >> z) || in >> trailing || !std::isfinite(x) ||
        !std::isfinite(y) || !std::isfinite(z)) {
      err = "malformed tmacl atom row " + std::to_string(atom + 1);
      return false;
    }
    int number = 0;
    if (symbol == "C") {
      number = 6;
    } else if (symbol == "H") {
      number = 1;
    } else if (symbol == "N") {
      number = 7;
    } else if (symbol == "Cl") {
      number = 17;
    } else {
      err = "unknown symbol " + symbol;
      return false;
    }
    numbers.push_back(number);
    positions.push_back(x / kAngPerBohr);
    positions.push_back(y / kAngPerBohr);
    positions.push_back(z / kAngPerBohr);
  }
  while (std::getline(file, line)) {
    if (line.find_first_not_of(" \t\r") != std::string::npos) {
      err = "tmacl fixture contains trailing nonempty rows";
      return false;
    }
  }
  if (numbers.size() != 18u) {
    err = "tmacl fixture must contain exactly 18 atoms";
    return false;
  }
  return true;
}

double temperature_hartree(double kelvin) { return kelvin * kKelvinToHartree; }

bool near(double lhs, double rhs, double atol) { return std::abs(lhs - rhs) <= atol; }

bool run_policy(Geometry& geometry, double kelvin, std::int64_t history, double damping,
                std::uint64_t maximum_iterations, PolicyRow& result, std::vector<TraceRow>& trace,
                std::string& err) {
  gpuxtb_status_t status = initialize_sad_multipole_state(geometry.layout, geometry.wfn, err);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  Stage stage;
  status = stage.build(geometry, history, damping, kDefaultRmsTolerance, kDefaultEnergyTolerance,
                       maximum_iterations, temperature_hartree(kelvin), err);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  const bool converged = run_stage(geometry, stage, maximum_iterations, trace, err);
  if (trace.empty()) {
    return false;
  }
  result = {kelvin, history, damping, converged, trace.back(), capture_state(geometry)};
  return true;
}

struct SequenceStage {
  double kelvin;
  bool converged;
  std::vector<TraceRow> trace;
};

struct SequenceResult {
  bool all_converged = false;
  std::vector<SequenceStage> stages;
  ElectronicState final_state;
};

bool run_sequence(Geometry& geometry, const std::vector<double>& temperatures,
                  std::uint64_t maximum_iterations, SequenceResult& result, std::string& err) {
  result = {};
  gpuxtb_status_t status = initialize_sad_multipole_state(geometry.layout, geometry.wfn, err);
  if (status != GPUXTB_STATUS_SUCCESS) {
    return false;
  }
  bool all_converged = true;
  for (double kelvin : temperatures) {
    Stage stage;
    status = stage.build(geometry, 8, 0.4, kDefaultRmsTolerance, kDefaultEnergyTolerance,
                         maximum_iterations, temperature_hartree(kelvin), err);
    if (status != GPUXTB_STATUS_SUCCESS) {
      return false;
    }
    std::vector<TraceRow> trace;
    const bool converged = run_stage(geometry, stage, maximum_iterations, trace, err);
    if (trace.empty()) {
      return false;
    }
    result.stages.push_back({kelvin, converged, std::move(trace)});
    all_converged = all_converged && converged;
    if (!converged) {
      break;
    }
  }
  result.all_converged = all_converged && result.stages.size() == temperatures.size();
  result.final_state = capture_state(geometry);
  return true;
}

bool write_sequence_file(const std::string& directory, const std::string& filename,
                         const SequenceResult& result, std::string& err) {
  std::ofstream output(directory + "/" + filename, std::ios::trunc);
  if (!output) {
    err = "cannot open evidence output " + directory + "/" + filename;
    return false;
  }
  for (std::size_t index = 0; index < result.stages.size(); ++index) {
    const SequenceStage& stage = result.stages[index];
    std::ostringstream title;
    title << "continuation stage " << std::fixed << std::setprecision(0) << stage.kelvin << " K ("
          << (index == 0 ? "fresh SAD seed" : "seeded from prior converged state") << ")";
    write_trace(output, title.str(), stage.trace);
  }
  if (!output) {
    err = "failed while writing evidence output " + directory + "/" + filename;
    return false;
  }
  return true;
}

// Expected converged 300 K internal SCC free energies and fragment charges
// pinned by the issue #217 investigation on gpuxtb main 9fd7d4d.
constexpr double kExpected300KEnergy = -22.271821505;
constexpr double kExpected300KQMe4N = 0.8285;
constexpr double kExpected300KQCl = -0.8285;

}  // namespace

int main(int argc, char** argv) {
  std::optional<std::string> evidence_directory;
  if (argc == 3 && std::string(argv[1]) == "--write-evidence") {
    evidence_directory = argv[2];
  } else if (argc != 1) {
    std::cerr << "usage: " << argv[0] << " [--write-evidence DIRECTORY]\n";
    return 2;
  }

  std::string err;
  std::vector<std::int32_t> numbers;
  std::vector<double> positions;
  if (!load_tmacl(numbers, positions, err)) {
    std::cerr << "fixture load failed: " << err << '\n';
    return __LINE__;
  }

  Geometry geo;
  geo.atomic_numbers = numbers;
  geo.positions = positions;
  geo.atom_offsets[1] = 18;
  CHECK(geo.build(err) == GPUXTB_STATUS_SUCCESS);

  // ------------------------------------------------------------------ baseline
  // Exact baseline matrix reproduced from the issue description, using the
  // public default policy: Johnson modified-Broyden history 8, damping 0.4,
  // charge (rms) tolerance 1e-6 and energy tolerance 1e-8. The 300 K row is
  // exercised at both the 250- and 1000-iteration ceilings from the issue.
  struct BaselineRow {
    double kelvin;
    std::uint64_t maximum_iterations;
    bool expect_converged;
    std::uint64_t expected_iterations;
  };
  const BaselineRow baseline[] = {
      {300.0, 250u, false, 250u}, {300.0, 1000u, false, 1000u}, {350.0, 250u, false, 250u},
      {400.0, 250u, false, 250u}, {450.0, 250u, true, 39u},     {500.0, 250u, true, 29u},
      {1000.0, 250u, true, 25u},
  };
  std::vector<std::pair<double, std::vector<TraceRow>>> baseline_traces;
  for (const BaselineRow& row : baseline) {
    // Each fresh baseline cell must start from the SAD multipole guess so a
    // failure genuinely reflects the default policy, not warm continuation
    // from the previous row's terminal state.
    gpuxtb_status_t sad_status = initialize_sad_multipole_state(geo.layout, geo.wfn, err);
    CHECK(sad_status == GPUXTB_STATUS_SUCCESS);
    Stage st;
    CHECK(st.build(geo, 8, 0.4, kDefaultRmsTolerance, kDefaultEnergyTolerance,
                   row.maximum_iterations, temperature_hartree(row.kelvin),
                   err) == GPUXTB_STATUS_SUCCESS);
    std::vector<TraceRow> trace;
    const bool converged = run_stage(geo, st, row.maximum_iterations, trace, err);
    CHECK(!trace.empty());
    CHECK(converged == row.expect_converged);
    // This exact seven-row status/count matrix is part of the reproducible
    // baseline and is stable across the two independently exercised LP64
    // providers.
    CHECK(trace.back().iter == row.expected_iterations);
    CHECK(trace.back().status == (row.expect_converged
                                      ? static_cast<int>(GPUXTB_STATUS_SUCCESS)
                                      : static_cast<int>(GPUXTB_STATUS_SCC_NOT_CONVERGED)));
    std::printf(
        "baseline %.0f K / %llu: converged=%d iterations=%llu E=%.12f q(Me4N+)=%.5f q(Cl)=%.5f\n",
        row.kelvin, (unsigned long long)row.maximum_iterations, converged ? 1 : 0,
        (unsigned long long)trace.back().iter, trace.back().free_energy, trace.back().q_me4n,
        trace.back().q_cl);
    // The archived 300 K diagnostic remains the reviewed 250-iteration run;
    // the 1000-iteration row proves the issue baseline without adding a
    // redundant large trace artifact.
    if (row.maximum_iterations == 250u) {
      baseline_traces.emplace_back(row.kelvin, std::move(trace));
    }
  }

  // ------------------------------------------------------------------ baseline
  // 300 K fresh default policy: the terminal rows must show persistent charge
  // sloshing, i.e. the atom-resolved charge contrast flips sign repeatedly
  // while the residual stays far above the 1e-6 rms gate. Quantify this by
  // demanding repeated Cl-charge sign changes, order-one free-energy swings,
  // no mixer restarts, and residual RMS that never approaches the 1e-6 gate.
  {
    const std::vector<TraceRow>& trace = baseline_traces.front().second;
    double q_cl_min = 0.0;
    double q_cl_max = 0.0;
    double maximum_energy_step = 0.0;
    bool residual_ever_below = false;
    std::uint64_t sign_changes = 0u;
    int previous_sign = 0;
    for (std::size_t index = 0; index < trace.size(); ++index) {
      const TraceRow& row = trace[index];
      q_cl_min = std::min(q_cl_min, row.q_cl);
      q_cl_max = std::max(q_cl_max, row.q_cl);
      if (index != 0u) {
        maximum_energy_step = std::max(maximum_energy_step, std::abs(row.d_free));
      }
      residual_ever_below = residual_ever_below || row.rms < 1.0e-3;
      CHECK(row.restarts == 0u);
      const int sign = row.q_cl > 0.0 ? 1 : -1;
      if (previous_sign != 0 && sign != previous_sign) {
        ++sign_changes;
      }
      previous_sign = sign;
    }
    CHECK(q_cl_min < 0.0 && q_cl_max > 0.0);
    CHECK(sign_changes >= 20u);
    CHECK(maximum_energy_step > 0.5);
    CHECK(!residual_ever_below);
  }

  // ------------------------------------------------------------------ funnel
  // Bounded temperature continuation: converge at an elevated temperature and
  // reuse only the electronic state (wavefunction multipoles) as the next
  // stage's initial guess, running each stage's requested occupations, free
  // energy, convergence tolerances, and ceiling at its own temperature.
  const std::vector<double> funnel{1000.0, 850.0, 700.0, 550.0, 400.0, 300.0};
  SequenceResult funnel_result;
  ElectronicState funnel_state;
  {
    const std::uint64_t kStageCeiling = 250u;
    CHECK(run_sequence(geo, funnel, kStageCeiling, funnel_result, err));
    for (const SequenceStage& stage : funnel_result.stages) {
      const TraceRow& last = stage.trace.back();
      std::printf("funnel %5.0f K: converged=%d iterations=%llu E=%.12f q(Me4N+)=%.5f q(Cl)=%.5f\n",
                  stage.kelvin, stage.converged ? 1 : 0, (unsigned long long)last.iter,
                  last.free_energy, last.q_me4n, last.q_cl);
    }
    // Every bounded continuation stage must converge, including the final
    // requested 300 K stage, whose result must match the pinned localized
    // stationary state.
    CHECK(funnel_result.all_converged);
    CHECK(funnel_result.stages.size() == funnel.size());
    const TraceRow& last = funnel_result.stages.back().trace.back();
    CHECK(near(last.free_energy, kExpected300KEnergy, 1e-6));
    CHECK(near(last.q_me4n, kExpected300KQMe4N, 2e-3));
    CHECK(near(last.q_cl, kExpected300KQCl, 2e-3));
    CHECK(last.rms < kDefaultRmsTolerance);
    CHECK(std::abs(last.d_free) < kDefaultEnergyTolerance);
    funnel_state = funnel_result.final_state;
  }

  // The successful funnel is path-dependent: direct high-to-target jumps do
  // not converge, while a finer bounded sequence reaches the same q/d/Q state.
  SequenceResult fine_result;
  SequenceResult coarse_1000_result;
  SequenceResult coarse_800_result;
  CHECK(run_sequence(geo, {1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0, 300.0}, 250u,
                     fine_result, err));
  CHECK(fine_result.all_converged);
  CHECK(maximum_multipole_difference(funnel_state, fine_result.final_state) < 5.0e-5);
  CHECK(maximum_difference(funnel_state.density, fine_result.final_state.density) < 5.0e-5);
  CHECK(run_sequence(geo, {1000.0, 300.0}, 250u, coarse_1000_result, err));
  CHECK(!coarse_1000_result.all_converged);
  CHECK(coarse_1000_result.stages.size() == 2u);
  CHECK(!coarse_1000_result.stages.back().converged);
  CHECK(run_sequence(geo, {800.0, 300.0}, 250u, coarse_800_result, err));
  CHECK(!coarse_800_result.all_converged);
  CHECK(coarse_800_result.stages.size() == 2u);
  CHECK(!coarse_800_result.stages.back().converged);

  // ------------------------------------------------------------------ policy
  // The same 300 K localized state is reachable from the SAD guess under
  // bounded deterministic mixer policies without any temperature change.
  // Execute the complete bounded grid and require every successful 300 K cell
  // to reach the funnel's atom-resolved q/d/Q and density, not merely the same
  // total energy and two fragment-charge sums.
  std::vector<PolicyRow> policy_rows;
  {
    const std::int64_t histories[] = {2, 4, 8, 16};
    const double dampings[] = {0.2, 0.4, 0.6};
    for (double kelvin : {300.0, 450.0}) {
      for (std::size_t history_index = 0; history_index < 4u; ++history_index) {
        for (std::size_t damping_index = 0; damping_index < 3u; ++damping_index) {
          PolicyRow result{};
          std::vector<TraceRow> trace;
          CHECK(run_policy(geo, kelvin, histories[history_index], dampings[damping_index], 250u,
                           result, trace, err));
          std::printf(
              "policy %.0f K / history %lld / damping %.1f: converged=%d "
              "iterations=%llu E=%.12f\n",
              kelvin, static_cast<long long>(result.history), result.damping,
              result.converged ? 1 : 0, static_cast<unsigned long long>(result.terminal.iter),
              result.terminal.free_energy);
          // Near the finite iteration ceiling, tiny BLAS reduction-order
          // differences can move a marginal mixer cell across the converged
          // boundary on otherwise compatible LP64 providers and CPU
          // microarchitectures. Execute and archive every cell, but gate only
          // the reviewed policy rows that establish the scientific claim.
          const bool reviewed_300_policy =
              kelvin == 300.0 && ((history_index == 0u && damping_index == 0u) ||
                                  (history_index == 3u && damping_index == 1u));
          if (reviewed_300_policy) {
            CHECK(result.converged);
          }
          if (kelvin == 300.0 && result.converged) {
            CHECK(near(result.terminal.free_energy, kExpected300KEnergy, 1e-6));
            CHECK(maximum_multipole_difference(funnel_state, result.state) < 5.0e-5);
            CHECK(maximum_difference(funnel_state.density, result.state.density) < 5.0e-5);
          }
          policy_rows.push_back(std::move(result));
        }
      }
    }
  }

  if (evidence_directory.has_value()) {
    for (const auto& [kelvin, trace] : baseline_traces) {
      if (kelvin != 300.0 && kelvin != 400.0 && kelvin != 450.0 && kelvin != 500.0 &&
          kelvin != 1000.0) {
        continue;
      }
      std::ostringstream filename;
      filename << "tmacl_trace_" << std::fixed << std::setprecision(0) << kelvin << "K.txt";
      std::ostringstream title;
      title << "T = " << std::fixed << std::setprecision(0) << kelvin
            << " K (history 8, damping 0.40, max_iter 250)";
      CHECK(write_trace_file(*evidence_directory, filename.str(), title.str(), trace, err));
    }
    CHECK(write_sequence_file(*evidence_directory, "tmacl_continuation_funnel.txt", funnel_result,
                              err));
    std::ofstream sweep(*evidence_directory + "/tmacl_mixer_sweep.txt", std::ios::trunc);
    CHECK(static_cast<bool>(sweep));
    for (const PolicyRow& row : policy_rows) {
      sweep << "T=" << std::setw(6) << std::fixed << std::setprecision(0) << row.kelvin
            << " h=" << std::setw(2) << row.history << " d=" << std::setprecision(1) << row.damping
            << " : " << (row.converged ? "CONVERGED" : "NOT_CONVERGED") << " in "
            << row.terminal.iter << " iters, E=" << std::setprecision(12)
            << row.terminal.free_energy << "  q(M4N+)=" << std::setw(8) << std::setprecision(5)
            << row.terminal.q_me4n << " q(Cl)=" << std::setw(8) << row.terminal.q_cl
            << " rms= " << std::scientific << std::setprecision(3) << row.terminal.rms << '\n';
    }
    CHECK(static_cast<bool>(sweep));
    std::ofstream paths(*evidence_directory + "/tmacl_path_sensitivity.txt", std::ios::trunc);
    CHECK(static_cast<bool>(paths));
    const auto write_path = [&paths](const std::string& name, const SequenceResult& result) {
      paths << name << ": " << (result.all_converged ? "CONVERGED" : "NOT_CONVERGED");
      for (const SequenceStage& stage : result.stages) {
        const TraceRow& last = stage.trace.back();
        paths << " | " << std::fixed << std::setprecision(0) << stage.kelvin
              << "K=" << (stage.converged ? "SUCCESS" : "SCC_NOT_CONVERGED") << "/" << last.iter;
      }
      paths << '\n';
    };
    write_path("bounded-150K", funnel_result);
    write_path("bounded-100K", fine_result);
    write_path("coarse-1000-to-300K", coarse_1000_result);
    write_path("coarse-800-to-300K", coarse_800_result);
    CHECK(static_cast<bool>(paths));
  }

  std::printf("issue #217 temperature-continuation investigation: all gates PASS\n");
  return 0;
}
