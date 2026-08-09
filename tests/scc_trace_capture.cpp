// xtbloom CPU SCC trace capture executable for one pinned corpus case
// (issues #42/#49/#50).
//
// Drives the production CPU GFN2 SCC driver through one case.spec and streams
// the captured per-iteration state in the tblite recorder raw layout.
// xtbloom_scc_cpu_trace.py canonicalizes the stream and compares it to the
// pinned golden with the cpu_closed_loop_v1 closed-loop profile.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "support/scc_trace_harness.hpp"

int main(int argc, char** argv) {
  using namespace xtbloom_trace_harness;
  if (argc != 2) {
    std::cerr << "usage: xtbloom_scc_trace_capture <case.spec>\n";
    return 64;
  }
  TraceBatch batch;
  CaseSpec spec;
  std::string err;
  if (!load_spec(argv[1], spec, err)) {
    std::cerr << err << "\n";
    return 2;
  }
  batch.add_case(spec);
  if (xtbloom_status_t s = batch.build(err); s != XTBLOOM_STATUS_SUCCESS) {
    std::cerr << "geometry build failed: " << err << "\n";
    return static_cast<int>(s);
  }
  if (xtbloom_status_t s = batch.run(err); s != XTBLOOM_STATUS_SUCCESS) {
    std::cerr << "SCC run failed: " << err << "\n";
    return static_cast<int>(s);
  }
  batch.emit(std::cout, 0);
  return 0;
}
