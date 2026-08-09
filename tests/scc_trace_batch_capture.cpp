// gpuxtb ragged-batch SCC trace capture executable (issue #50).
//
// Loads several corpus case.spec files and runs them as ONE heterogeneous
// ragged GFN2 SCC driver batch.  The healthy lanes must follow their pinned
// per-system trajectories exactly as in sequential execution, systems which
// converge early must stop mutating while peers continue, and a controlled
// per-system failure lane must not corrupt or suppress successful members.
//
// Output: for every lane, a "batch_system <index>" marker followed by the
// complete recorder-raw stream for that lane, so gpuxtb_scc_cpu_trace.py can
// canonicalize and compare each lane independently against its golden.
//
// Usage:
//   gpuxtb_scc_trace_batch_capture <spec1> [<spec2> ...] [--poison <index>]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "support/scc_trace_harness.hpp"

int main(int argc, char** argv) {
  using namespace gpuxtb_trace_harness;
  if (argc < 2) {
    std::cerr << "usage: gpuxtb_scc_trace_batch_capture <spec1> [<spec2> ...] "
                 "[--poison <index>]\n";
    return 64;
  }
  std::vector<std::string> spec_paths;
  std::int64_t poison_index = -1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--poison") == 0 && i + 1 < argc) {
      poison_index = std::strtoll(argv[++i], nullptr, 10);
    } else {
      spec_paths.emplace_back(argv[i]);
    }
  }

  TraceBatch batch;
  std::string err;
  for (const std::string& path : spec_paths) {
    CaseSpec spec;
    if (!load_spec(path, spec, err)) {
      std::cerr << err << "\n";
      return 2;
    }
    batch.add_case(spec);
  }
  if (gpuxtb_status_t s = batch.build(err); s != GPUXTB_STATUS_SUCCESS) {
    std::cerr << "geometry build failed: " << err << "\n";
    return static_cast<int>(s);
  }
  if (poison_index >= 0 && poison_index < batch.system_count()) {
    batch.poison_h0(poison_index);
  }
  if (gpuxtb_status_t s = batch.run(err); s != GPUXTB_STATUS_SUCCESS) {
    std::cerr << "SCC run failed: " << err << "\n";
    return static_cast<int>(s);
  }
  for (std::int64_t system = 0; system < batch.system_count(); ++system) {
    std::cout << "batch_system " << system << "\n";
    std::cout << "status " << static_cast<std::int64_t>(batch.system_status(system))
              << " iterations " << batch.system_iterations(system) << "\n";
    batch.emit(std::cout, system);
  }
  return 0;
}
