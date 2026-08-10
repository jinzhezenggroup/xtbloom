// xtbloom single-iteration SCC replay executable (issue #49).
//
// Injects the golden mixed q/d/Q state of one pinned trace iteration into the
// production CPU GFN2 SCC driver, runs exactly one iteration, and streams the
// resulting iteration snapshot in the recorder raw layout.  This diagnoses a
// divergent primitive at the exact iteration where it first appears, without
// inheriting drift from earlier Broyden iterations.
//
// Usage:
//   xtbloom_scc_trace_replay <case.spec> <mixed-state-file> <logical-index> \
//       <previous-energy>
//
// The mixed-state file has the layout documented in
// TraceBatch::inject_mixed_state: qsh/qat/dipoles/quadrupoles sections with
// explicit counts.  previous-energy is the golden trace's iteration
// (logical-index-1) "energy" field, or 0.0 for logical index 1.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "support/scc_trace_harness.hpp"

int main(int argc, char** argv) {
  using namespace xtbloom_trace_harness;
  if (argc != 5) {
    std::cerr << "usage: xtbloom_scc_trace_replay <case.spec> <mixed-state-file> "
                 "<logical-index> <previous-energy>\n";
    return 64;
  }
  const std::uint64_t logical_index = std::strtoull(argv[3], nullptr, 10);
  const double previous_energy = std::strtod(argv[4], nullptr);

  TraceBatch batch;
  CaseSpec spec;
  std::string err;
  if (!load_spec(argv[1], spec, err)) {
    std::cerr << err << "\n";
    return 2;
  }
  if (logical_index == 0u || spec.maximum_iterations <= 0 ||
      logical_index > static_cast<std::uint64_t>(spec.maximum_iterations)) {
    std::cerr << "logical index must be within the case iteration limit\n";
    return 64;
  }
  // The driver counter is seeded to logical_index-1 below.  Capping this
  // replay plan at logical_index makes the one executed attempt a real driver
  // terminal (SCC_NOT_CONVERGED unless it converges), so emit() never invents
  // max-iteration metadata for an otherwise still-active lane.
  spec.maximum_iterations = static_cast<std::int64_t>(logical_index);
  batch.add_case(spec);
  if (xtbloom_status_t s = batch.build(err); s != XTBLOOM_STATUS_SUCCESS) {
    std::cerr << "geometry build failed: " << err << "\n";
    return static_cast<int>(s);
  }
  if (xtbloom_status_t s = batch.inject_mixed_state(0, argv[2], err); s != XTBLOOM_STATUS_SUCCESS) {
    std::cerr << "state injection failed: " << err << "\n";
    return static_cast<int>(s);
  }
  batch.set_replay_context(0, logical_index, previous_energy);
  if (xtbloom_status_t s = batch.restart_mixer_system(0, err); s != XTBLOOM_STATUS_SUCCESS) {
    std::cerr << "mixer restart failed: " << err << "\n";
    return static_cast<int>(s);
  }

  // Exactly one driver iteration from the injected state.  The single injected
  // lane is always active on the first call, so step_once records the one
  // replayed iteration and the raw stream carries niterations=1.
  if (xtbloom_status_t s = batch.step_once(err); s != XTBLOOM_STATUS_SUCCESS) {
    std::cerr << "replay iteration failed: " << err << "\n";
    return static_cast<int>(s);
  }
  if (batch.system_status(0) == XTBLOOM_STATUS_SUCCESS && !batch.system_converged(0)) {
    std::cerr << "replay iteration remained active after its one-step limit\n";
    return static_cast<int>(XTBLOOM_STATUS_INTERNAL_ERROR);
  }
  batch.emit(std::cout, 0);
  return 0;
}
