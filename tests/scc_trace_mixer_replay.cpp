// gpuxtb independent golden-residual Broyden mixer replay (issue #49).
//
// Drives gpuxtb's production Broyden mixer alone through the PINNED golden
// residual sequence (raw minus mixed per iteration), bypassing the driver and
// the eigensolver entirely, and checks that each state transition reproduces
// the golden next mixed state.  This isolates the mixer (flattening order,
// damping, history, and state transitions) from the physical trajectory: a
// self-consistent flatten/unflatten defect inside the driver could reproduce a
// trajectory while a mixer-only replay of the golden residuals would not.
//
// Usage:
//   gpuxtb_scc_trace_mixer <case.spec> <sequence-file>
//
// The sequence file is line-oriented:
//   nat <nat> nsh <nsh> steps <K>
//   step <k>
//   mixed <nsh + 3*nat + 6*nat values in residual order qsh, dpat, qpat>
//   raw   <same layout>
// for each logical step.  On stdout the executable emits one line per
// completed transition:
//   predicted <k+1> <flattened next mixed state in residual order>
// which gpuxtb_scc_cpu_trace.py compares to the pinned golden's mixed state
// for logical iteration k+1 with the cpu_replay_v1 tolerance profile.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "support/scc_trace_harness.hpp"

namespace {

using namespace gpuxtb_trace_harness;

struct Sequence {
  std::int64_t nat = 0;
  std::int64_t nsh = 0;
  std::int64_t steps = 0;
  std::vector<std::vector<double>> mixed;  // per logical step, residual order
  std::vector<std::vector<double>> raw;    // per logical step, residual order
};

bool parse_sequence(const std::string& path, Sequence& sequence, std::string& err) {
  std::ifstream file(path);
  if (!file) {
    err = "cannot open sequence file " + path;
    return false;
  }
  std::string token;
  if (!(file >> token) || token != "nat" || !(file >> sequence.nat)) {
    err = "sequence file missing nat";
    return false;
  }
  if (!(file >> token) || token != "nsh" || !(file >> sequence.nsh)) {
    err = "sequence file missing nsh";
    return false;
  }
  if (!(file >> token) || token != "steps" || !(file >> sequence.steps)) {
    err = "sequence file missing steps";
    return false;
  }
  if (sequence.nat <= 0 || sequence.nsh <= 0 || sequence.steps <= 0) {
    err = "sequence counts must be positive";
    return false;
  }
  const std::size_t dimension = static_cast<std::size_t>(sequence.nsh + 9 * sequence.nat);
  sequence.mixed.assign(static_cast<std::size_t>(sequence.steps), {});
  sequence.raw.assign(static_cast<std::size_t>(sequence.steps), {});
  for (std::int64_t k = 1; k <= sequence.steps; ++k) {
    if (!(file >> token) || token != "step" || !(file >> token) || token != std::to_string(k)) {
      err = "sequence file step header malformed at " + std::to_string(k);
      return false;
    }
    for (const char* section : {"mixed", "raw"}) {
      if (!(file >> token) || token != section) {
        err = std::string("sequence file missing ") + section + " at step " + std::to_string(k);
        return false;
      }
      std::vector<double>& values = section[0] == 'm'
                                        ? sequence.mixed[static_cast<std::size_t>(k - 1)]
                                        : sequence.raw[static_cast<std::size_t>(k - 1)];
      values.reserve(dimension);
      double value = 0.0;
      while (values.size() < dimension && (file >> value)) {
        values.push_back(value);
      }
      if (values.size() != dimension) {
        err = std::string("sequence file ") + section + " at step " + std::to_string(k) +
              " has wrong length";
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace gpuxtb_trace_harness;
  if (argc != 3) {
    std::cerr << "usage: gpuxtb_scc_trace_mixer <case.spec> <sequence-file>\n";
    return 64;
  }
  Sequence sequence;
  std::string err;
  if (!parse_sequence(argv[2], sequence, err)) {
    std::cerr << err << "\n";
    return 2;
  }

  TraceBatch batch;
  CaseSpec spec;
  if (!load_spec(argv[1], spec, err)) {
    std::cerr << err << "\n";
    return 2;
  }
  batch.add_case(spec);
  if (gpuxtb_status_t s = batch.build(err); s != GPUXTB_STATUS_SUCCESS) {
    std::cerr << "geometry build failed: " << err << "\n";
    return static_cast<int>(s);
  }
  const std::int64_t dimension = sequence.nsh + 9 * sequence.nat;

  auto split = [&](const std::vector<double>& flattened, std::vector<double>& qsh,
                   std::vector<double>& dpat, std::vector<double>& qpat) {
    qsh.assign(flattened.begin(), flattened.begin() + sequence.nsh);
    dpat.assign(flattened.begin() + sequence.nsh,
                flattened.begin() + sequence.nsh + 3 * sequence.nat);
    qpat.assign(flattened.begin() + sequence.nsh + 3 * sequence.nat,
                flattened.begin() + sequence.nsh + 9 * sequence.nat);
  };

  // Seed the mixer's current input from the golden mixed state of step 1 and
  // clear its history, then replay every golden residual through the
  // production Broyden transition.  After each mix the wavefunction holds the
  // next mixed input, which must equal the golden mixed state of the next step.
  std::vector<double> qsh, dpat, qpat;
  split(sequence.mixed[0], qsh, dpat, qpat);
  if (gpuxtb_status_t s = batch.write_multipoles(0, qsh, dpat, qpat, err);
      s != GPUXTB_STATUS_SUCCESS) {
    std::cerr << "seed failed: " << err << "\n";
    return static_cast<int>(s);
  }
  if (gpuxtb_status_t s = batch.restart_mixer_system(0, err); s != GPUXTB_STATUS_SUCCESS) {
    std::cerr << "mixer restart failed: " << err << "\n";
    return static_cast<int>(s);
  }

  std::cout << "mixer_replay case nat " << sequence.nat << " nsh " << sequence.nsh << " steps "
            << sequence.steps << " dimension " << dimension << "\n";
  for (std::int64_t k = 1; k <= sequence.steps; ++k) {
    split(sequence.raw[static_cast<std::size_t>(k - 1)], qsh, dpat, qpat);
    if (gpuxtb_status_t s = batch.write_multipoles(0, qsh, dpat, qpat, err);
        s != GPUXTB_STATUS_SUCCESS) {
      std::cerr << "raw write failed at step " << k << ": " << err << "\n";
      return static_cast<int>(s);
    }
    if (gpuxtb_status_t s = batch.mixer_mix(0, err); s != GPUXTB_STATUS_SUCCESS) {
      std::cerr << "mixer transition failed at step " << k << ": " << err << "\n";
      return static_cast<int>(s);
    }
    if (k < sequence.steps) {
      std::cout << "predicted " << (k + 1);
      for (double value : batch.next_mixed_flattened(0)) {
        std::cout << " " << std::setprecision(17) << std::scientific << value;
      }
      std::cout << "\n";
    }
  }
  return 0;
}
