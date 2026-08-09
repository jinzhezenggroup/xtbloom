// Native ragged-batch SCC trace harness test (issues #49/#50).
//
// Drives the same production CPU GFN2 SCC driver used by the sequential trace
// capture through one heterogeneous ragged batch and verifies the structural
// properties that the Python golden comparison cannot observe directly:
//
//   * per-lane terminal status, iteration counters, and convergence flags;
//   * early-converged systems freeze while slower peers keep advancing
//     (Broyden history ownership and per-lane skip semantics);
//   * a controlled per-system preparation failure is data-level: the failing
//     lane registers INTERNAL_ERROR with zero completed iterations while every
//     healthy peer still converges;
//   * per-system multisite state slices are placed in disjoint lanes.
//
// This test runs in every CPU configuration, including sanitizer builds, so
// ragged execution is covered for out-of-bounds access and cross-system
// workspace aliasing without needing a golden JSON parser in C++.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "support/scc_trace_harness.hpp"

#ifndef GPUXTB_SCC_TRACE_SPEC_DIR
#error "GPUXTB_SCC_TRACE_SPEC_DIR must point at the pinned corpus specs"
#endif

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      std::cerr << "CHECK failed at line " << __LINE__ << ": " << #condition << "\n"; \
      return 1;                                                                       \
    }                                                                                 \
  } while (false)

namespace {

using namespace gpuxtb_trace_harness;

std::string spec_path(const char* name) {
  return std::string(GPUXTB_SCC_TRACE_SPEC_DIR) + "/" + name + ".spec";
}

CaseSpec corpus_spec(const char* name, std::string& err) {
  CaseSpec spec;
  if (!load_spec(spec_path(name), spec, err)) {
    throw std::runtime_error(err);
  }
  return spec;
}

bool load(std::vector<CaseSpec>& specs, const std::vector<std::string>& names, std::string& err) {
  for (const std::string& name : names) {
    CaseSpec spec;
    if (!load_spec(spec_path(name.c_str()), spec, err)) {
      return false;
    }
    specs.push_back(spec);
  }
  return true;
}

bool state_equal(const std::vector<std::vector<double>>& first,
                 const std::vector<std::vector<double>>& second) {
  if (first.size() != second.size()) {
    return false;
  }
  for (std::size_t channel = 0; channel < first.size(); ++channel) {
    if (first[channel].size() != second[channel].size()) {
      return false;
    }
    for (std::size_t i = 0; i < first[channel].size(); ++i) {
      if (first[channel][i] != second[channel][i]) {
        return false;
      }
    }
  }
  return true;
}

int test_early_convergence_freezes_while_peers_advance() {
  std::string err;
  std::vector<CaseSpec> specs;
  if (!load(specs, {"h3_plus", "nenacl", "water_dimer_6pc_hardness"}, err)) {
    std::cerr << err << "\n";
    return 1;
  }
  TraceBatch batch;
  for (const CaseSpec& spec : specs) {
    batch.add_case(spec);
  }
  if (gpuxtb_status_t s = batch.build(err); s != GPUXTB_STATUS_SUCCESS) {
    std::cerr << "build failed: " << err << "\n";
    return 1;
  }

  // Drive the batch by hand so we can observe lane 0 freezing mid-run while
  // the slower lanes (nenacl and the water dimer) continue advancing.
  std::vector<std::vector<double>> frozen_lane0;
  const std::uint64_t lane0_converged_at = 3u;
  for (std::uint64_t step_count = 0u; step_count < kMaximumHarnessIterations; ++step_count) {
    bool any_active = false;
    for (std::int64_t system = 0; system < batch.system_count(); ++system) {
      const bool active = batch.system_status(system) == GPUXTB_STATUS_SUCCESS &&
                          !batch.system_converged(system) && batch.system_iterations(system) < 256u;
      any_active = any_active || active;
    }
    if (!any_active) {
      break;
    }
    const gpuxtb_status_t s = batch.step_once(err);
    if (s != GPUXTB_STATUS_SUCCESS && s != GPUXTB_STATUS_SCC_NOT_CONVERGED) {
      std::cerr << "step failed: " << err << "\n";
      return 1;
    }
    if (batch.system_iterations(0) == lane0_converged_at && frozen_lane0.empty()) {
      // Lane 0 just converged (h3_plus closes at 3 iterations).  Its peers must
      // keep advancing, and lane 0 must stop mutating.
      frozen_lane0 = batch.live_state(0);
      CHECK(batch.system_converged(0));
      // Slowest peer (nenacl) must still be active after lane 0 froze.
      CHECK(!batch.system_converged(1));
      CHECK(batch.system_iterations(1) < 14u);
    }
  }

  CHECK(batch.system_converged(0));
  CHECK(batch.system_converged(1));
  CHECK(batch.system_converged(2));
  CHECK(batch.system_iterations(0) == 3u);
  CHECK(batch.system_iterations(1) == 14u);
  CHECK(batch.system_iterations(2) == 9u);
  CHECK(!frozen_lane0.empty());
  // The frozen snapshot must still be the exact converged lane-0 state after
  // the slower peers fully converged.
  CHECK(state_equal(frozen_lane0, batch.live_state(0)));

  // Per-lane output placement: the converged water-dimer lane and the h3_plus
  // lane must hold different, finite state (no cross-lane aliasing).
  const std::vector<std::vector<double>> lane0 = batch.live_state(0);
  const std::vector<std::vector<double>> lane2 = batch.live_state(2);
  CHECK(!state_equal(lane0, lane2));
  for (const auto& channel : lane2) {
    for (double value : channel) {
      CHECK(std::isfinite(value));
    }
  }
  std::cout << "early-convergence freeze and batch parity: PASS\n";
  return 0;
}

int test_case_spec_iteration_cap_is_honored() {
  std::string err;
  CaseSpec capped = corpus_spec("h3_plus", err);
  capped.maximum_iterations = 2;
  TraceBatch batch;
  batch.add_case(capped);
  if (gpuxtb_status_t s = batch.build(err); s != GPUXTB_STATUS_SUCCESS) {
    std::cerr << "cap build failed: " << err << "\n";
    return 1;
  }
  if (gpuxtb_status_t s = batch.run(err);
      s != GPUXTB_STATUS_SUCCESS && s != GPUXTB_STATUS_SCC_NOT_CONVERGED) {
    std::cerr << "cap run failed: " << err << "\n";
    return 1;
  }
  // A two-iteration cap must terminate h3_plus as not-converged at exactly two
  // iterations instead of continuing to convergence.
  CHECK(batch.system_iterations(0) == 2u);
  CHECK(batch.system_status(0) == GPUXTB_STATUS_SCC_NOT_CONVERGED);
  CHECK(!batch.system_converged(0));
  std::cout << "case-spec iteration cap honored: PASS\n";
  return 0;
}

int test_nonhomogeneous_batch_policy_is_rejected() {
  std::string err;
  CaseSpec hot = corpus_spec("h3_plus", err);
  hot.temperature_kelvin = 30000.0;
  TraceBatch batch;
  batch.add_case(corpus_spec("h3_plus", err));
  batch.add_case(hot);
  gpuxtb_status_t s = batch.build(err);
  CHECK(s != GPUXTB_STATUS_SUCCESS);
  CHECK(err.find("numerical policy") != std::string::npos);
  std::cout << "nonhomogeneous batch policy rejected: PASS\n";
  return 0;
}

int test_failure_lane_is_isolated_from_peers() {
  std::string err;
  std::vector<CaseSpec> specs;
  if (!load(specs, {"h3_plus", "nenacl", "water_dimer_6pc_hardness", "h3_plus"}, err)) {
    std::cerr << err << "\n";
    return 1;
  }
  TraceBatch batch;
  for (const CaseSpec& spec : specs) {
    batch.add_case(spec);
  }
  if (gpuxtb_status_t s = batch.build(err); s != GPUXTB_STATUS_SUCCESS) {
    std::cerr << "build failed: " << err << "\n";
    return 1;
  }
  // Lane 3 is the controlled per-system preparation failure (NaN H0).
  batch.poison_h0(3);
  if (gpuxtb_status_t s = batch.run(err);
      s != GPUXTB_STATUS_SUCCESS && s != GPUXTB_STATUS_INTERNAL_ERROR) {
    std::cerr << "run failed: " << err << "\n";
    return 1;
  }

  CHECK(batch.system_converged(0));
  CHECK(batch.system_converged(1));
  CHECK(batch.system_converged(2));
  CHECK(batch.system_iterations(0) == 3u);
  CHECK(batch.system_iterations(1) == 14u);
  CHECK(batch.system_iterations(2) == 9u);
  CHECK(batch.system_status(0) == GPUXTB_STATUS_SUCCESS);
  CHECK(batch.system_status(1) == GPUXTB_STATUS_SUCCESS);
  CHECK(batch.system_status(2) == GPUXTB_STATUS_SUCCESS);
  // The failing lane advanced no iteration and reported a data-level failure.
  CHECK(batch.system_status(3) == GPUXTB_STATUS_INTERNAL_ERROR);
  CHECK(batch.system_iterations(3) == 0u);
  CHECK(!batch.system_converged(3));
  std::cout << "failure-lane isolation from ragged peers: PASS\n";
  return 0;
}

}  // namespace

int main() {
  if (test_early_convergence_freezes_while_peers_advance() != 0) {
    return 1;
  }
  if (test_failure_lane_is_isolated_from_peers() != 0) {
    return 1;
  }
  if (test_case_spec_iteration_cap_is_honored() != 0) {
    return 1;
  }
  if (test_nonhomogeneous_batch_policy_is_rejected() != 0) {
    return 1;
  }
  return 0;
}
