// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <string>
#include <thread>

#include "model/gfn2/eigensolver.hpp"

int main() {
  int result = 0;
  std::thread worker([&result] {
    xtbloom::detail::gfn2::CpuLinearAlgebraBackend backend;
    std::string error;
    const xtbloom_status_t status = xtbloom::detail::gfn2::make_mkl_rt_lp64_backend(backend, error);
    if (status != XTBLOOM_STATUS_BACKEND_UNAVAILABLE || backend.ready() ||
        error.find("host-isolated MKL pthread bridge/provider shim") == std::string::npos) {
      result = 1;
    }
  });

  /* The adjacent fake provider intentionally lacks LAPACKE/CBLAS symbols, so
   * verification fails after its constructor registers a base-registry TSS
   * key. Returning from this worker invokes the provider-owned key destructor;
   * a premature dlclose turns this join into a deterministic process crash. */
  worker.join();
  return result;
}
