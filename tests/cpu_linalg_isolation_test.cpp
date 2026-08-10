// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

// Host-isolation tests for the CPU LP64 eigensolver provider (issue #30).
//
// This binary deliberately links only xtbloom_gfn2_cpu (plus dl), never a BLAS or
// MKL runtime encoded in DT_NEEDED. All provider libraries are loaded lazily at
// runtime through the factory, so this process is a clean host in which we can
// prove:
//   1. xtbloom never exposes MKL or LAPACK symbols into the global namespace
//      (RTLD_DEFAULT); the provider lives in a separate glibc link-map.
//   2. xtbloom never loads libmkl_rt at all when the private MKL shim is used.
//   3. A real LP64 generalized eigensolve through the production backend stays
//      correct while the host drives its own MKL instance into and out of ILP64,
//      both before and after xtbloom backend creation.
//   4. The host's MKL state is left unchanged: the host's own libmkl_rt handle
//      continues to behave as ILP64 after xtbloom has created its backend.

#include "model/gfn2/eigensolver.hpp"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cerrno>

using xtbloom::detail::gfn2::CpuLinearAlgebraBackend;
using xtbloom::detail::gfn2::EigensolverOverlapCache;
using xtbloom::detail::gfn2::EigensolverPlan;
using xtbloom::detail::gfn2::EigensolverThermodynamicsView;
using xtbloom::detail::gfn2::EigensolverWorkspace;
using xtbloom::detail::gfn2::WavefunctionLayout;
using xtbloom::detail::gfn2::WavefunctionSystemView;
using xtbloom::detail::gfn2::WavefunctionView;

namespace {

constexpr double kTolerance = 2.0e-11;

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

struct AlignedBuffer {
  void* data = nullptr;
  std::size_t size = 0u;

  explicit AlignedBuffer(std::size_t requested) : size(requested) {
    data = std::aligned_alloc(xtbloom::detail::gfn2::kEigensolverWorkspaceAlignment, requested);
  }

  ~AlignedBuffer() { std::free(data); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

struct Evaluation {
  WavefunctionLayout layout;
  EigensolverPlan plan;
  std::unique_ptr<AlignedBuffer> wavefunction_storage;
  std::unique_ptr<AlignedBuffer> cache_storage;
  std::unique_ptr<AlignedBuffer> scratch_storage;
  WavefunctionView wavefunction;
  EigensolverOverlapCache cache;
  EigensolverWorkspace scratch;
  std::vector<xtbloom_status_t> statuses;
  std::vector<double> chemical_potentials;
  std::vector<double> entropies;
  std::vector<double> band_energies;
  std::vector<double> free_energies;

  EigensolverThermodynamicsView thermodynamics() {
    return {statuses.data(),
            statuses.size(),
            chemical_potentials.data(),
            chemical_potentials.size(),
            entropies.data(),
            entropies.size(),
            band_energies.data(),
            band_energies.size(),
            free_energies.data(),
            free_energies.size()};
  }
};

bool initialize_evaluation(const std::vector<std::int64_t>& atom_offsets,
                           const std::vector<std::int32_t>& atomic_numbers,
                           const std::vector<double>& charges,
                           const std::vector<std::int32_t>& unpaired,
                           const std::vector<std::int32_t>& spins, Evaluation& evaluation,
                           std::string& error) {
  xtbloom::detail::gfn2::BasisPlan basis;
  if (xtbloom::detail::gfn2::make_basis_plan(static_cast<std::int64_t>(atom_offsets.size() - 1u),
                                             static_cast<std::int64_t>(atomic_numbers.size()),
                                             atom_offsets.data(), atomic_numbers.data(), basis,
                                             error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_wavefunction_layout(
          basis, atomic_numbers.data(), charges.data(), unpaired.data(), spins.data(),
          evaluation.layout, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::make_eigensolver_plan(evaluation.layout, evaluation.plan, error,
                                                   1.0e-12) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }

  evaluation.wavefunction_storage =
      std::make_unique<AlignedBuffer>(evaluation.layout.workspace_size_bytes);
  evaluation.cache_storage =
      std::make_unique<AlignedBuffer>(evaluation.plan.overlap_cache_size_bytes());
  evaluation.scratch_storage =
      std::make_unique<AlignedBuffer>(evaluation.plan.workspace_size_bytes());
  if (evaluation.wavefunction_storage->data == nullptr ||
      evaluation.cache_storage->data == nullptr || evaluation.scratch_storage->data == nullptr ||
      xtbloom::detail::gfn2::bind_wavefunction_view(
          evaluation.layout, evaluation.wavefunction_storage->data,
          evaluation.wavefunction_storage->size, evaluation.wavefunction,
          error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::bind_eigensolver_overlap_cache(
          evaluation.plan, evaluation.cache_storage->data, evaluation.cache_storage->size,
          evaluation.cache, error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::gfn2::bind_eigensolver_workspace(
          evaluation.plan, evaluation.scratch_storage->data, evaluation.scratch_storage->size,
          evaluation.scratch, error) != XTBLOOM_STATUS_SUCCESS) {
    return false;
  }
  const std::size_t batch = static_cast<std::size_t>(evaluation.plan.batch_size());
  evaluation.statuses.resize(batch);
  evaluation.chemical_potentials.resize(2u * batch);
  evaluation.entropies.resize(batch);
  evaluation.band_energies.resize(batch);
  evaluation.free_energies.resize(batch);
  return true;
}

bool near(double actual, double expected, double tolerance = kTolerance) {
  return std::abs(actual - expected) <= tolerance;
}

/* Run the unresticted 2x2 literal generalized eigenproblem from the eigensolver
 * test through the given production backend and verify the pinned eigenvalues,
 * densities, and band energies. Returns the CHECK line on failure, 0 on success. */
int run_literal_generalized_eigenproblem(const CpuLinearAlgebraBackend& backend,
                                         std::string& error) {
  Evaluation evaluation;
  CHECK(initialize_evaluation({0, 2}, {1, 1}, {0.0}, {0}, {2}, evaluation, error));
  const std::vector<double> overlap{1.2, 0.15, 0.15, 0.9};
  const std::vector<double> hamiltonian{-0.8, 0.13, 0.13, 0.25, -0.55, -0.08, -0.08, 0.42};
  CHECK(xtbloom::detail::gfn2::factor_overlap_cpu(evaluation.plan, overlap.data(), 73u, backend,
                                                  evaluation.scratch, evaluation.cache,
                                                  error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn2::solve_eigensystems_cpu(
            evaluation.plan, evaluation.cache, 73u, hamiltonian.data(), 0.0, backend,
            evaluation.scratch, evaluation.wavefunction, evaluation.thermodynamics(),
            error) == XTBLOOM_STATUS_SUCCESS);

  CHECK(evaluation.statuses[0] == XTBLOOM_STATUS_SUCCESS);
  WavefunctionSystemView view;
  CHECK(xtbloom::detail::gfn2::make_wavefunction_system_view(
            evaluation.layout, evaluation.wavefunction, 0, view, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(view.eigenvalues[0], -0.7192210550444913, 3.0e-13));
  CHECK(near(view.eigenvalues[1], 0.28517850185300192, 3.0e-13));
  CHECK(near(view.eigenvalues[2], -0.45845957914509017, 3.0e-13));
  CHECK(near(view.eigenvalues[3], 0.48966525290395535, 3.0e-13));
  CHECK(near(evaluation.band_energies[0], -1.1776806341895814, 4.0e-13));
  return 0;
}

using SetInterfaceLayer = int (*)(int layer);
using DpotrfWork64 = std::int32_t (*)(std::int32_t, char, std::int64_t, double*, std::int64_t);

/* Open the host's own libmkl_rt as an embedding application would. Returns a
 * valid handle and the MKL_Set_Interface_Layer / LAPACKE_dpotrf_work_64 entry
 * points, or nulls if no MKL is present on the system. */
void host_mkl(void** handle_out, SetInterfaceLayer* set_interface_out, DpotrfWork64* dpotrf64_out) {
  *handle_out = nullptr;
  *set_interface_out = nullptr;
  *dpotrf64_out = nullptr;
#if !defined(_WIN32)
  static const char* const kMklRtSonames[] = {"libmkl_rt.so.4", "libmkl_rt.so.3", "libmkl_rt.so.2",
                                              "libmkl_rt.so", nullptr};
  for (const char* name : kMklRtSonames) {
    void* handle = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
    if (handle != nullptr) {
      dlerror();
      void* symbol = dlsym(handle, "MKL_Set_Interface_Layer");
      if (symbol == nullptr || dlerror() != nullptr) {
        static_cast<void>(dlclose(handle));
        continue;
      }
      void* dpotrf64 = nullptr;
      dlerror();
      dpotrf64 = dlsym(handle, "LAPACKE_dpotrf_work_64");
      *handle_out = handle;
      *set_interface_out = reinterpret_cast<SetInterfaceLayer>(symbol);
      *dpotrf64_out = reinterpret_cast<DpotrfWork64>(dpotrf64);
      return;
    }
  }
#endif
}

int run_correctness_with_backend() {
  std::string error;
  CpuLinearAlgebraBackend backend;
  const xtbloom_status_t status = xtbloom::detail::gfn2::make_mkl_rt_lp64_backend(backend, error);
  CHECK(status == XTBLOOM_STATUS_SUCCESS);
  CHECK(backend.ready());
#if defined(XTBLOOM_TEST_HAS_WHEEL_OPENBLAS)
  CHECK(backend.production_openblas_isolated());
#else
  /* This test is registered only when the configure-time MKL shim exists. */
  CHECK(backend.production_mkl_isolated());
#endif
  const int solve_status = run_literal_generalized_eigenproblem(backend, error);
  CHECK(solve_status == 0);
  return 0;
}

/* After the host accepts ILP64, its explicit 64-bit LAPACK entry must remain
 * usable before and after xtbloom creates its private provider namespace. */
int host_ilp64_works(void* host_handle, DpotrfWork64 host_dpotrf64) {
  CHECK(host_handle != nullptr);
  CHECK(host_dpotrf64 != nullptr);
  double matrix[4] = {4.0, 2.0, 2.0, 4.0};
  const std::int64_t n = 2;
  const std::int32_t rc = host_dpotrf64(102, 'L', n, matrix, n);
  CHECK(rc == 0);
  return 0;
}

/* Host present with an ILP64 interface layer can coexist with xtbloom: after the
 * host switches its own MKL instance to ILP64, xtbloom's LP64 calls must remain
 * correct and the host's MKL state must not be mutated by xtbloom. This
 * coexistence contract is only claimed for the host-isolated shim provider. */
int run_coexistence_xtbloom_after_host_ilp64() {
#if !defined(XTBLOOM_TEST_HAS_MKL_SHIM)
  return 0; /* No isolated shim was built; skip (see xtbloom.cpu.linalg_isolation). */
#else
  std::string error;
  void* host_handle = nullptr;
  SetInterfaceLayer host_set_interface = nullptr;
  DpotrfWork64 host_dpotrf64 = nullptr;
  host_mkl(&host_handle, &host_set_interface, &host_dpotrf64);
  CHECK(host_handle != nullptr);
  CHECK(dlsym(RTLD_DEFAULT, "MKL_Set_Interface_Layer") != nullptr);

  /* 1 = MKL_INTERFACE_LAYER_ILP64. The host owns this handle and its state. */
  const int ilp64_acceptance = host_set_interface(1);
  CHECK(ilp64_acceptance == 1);
  CHECK(host_ilp64_works(host_handle, host_dpotrf64) == 0);

  const int correct = run_correctness_with_backend();
  CHECK(correct == 0);

  CHECK(host_ilp64_works(host_handle, host_dpotrf64) == 0);

  static_cast<void>(dlclose(host_handle));
  return 0;
#endif
}

/* xtbloom created its backend first (LP64); the host later switches its own MKL
 * to ILP64. xtbloom's already-verified LP64 provider must keep producing correct
 * results, because it is resolved inside its own link-map namespace. */
int run_coexistence_host_ilp64_after_xtbloom() {
#if !defined(XTBLOOM_TEST_HAS_MKL_SHIM)
  return 0; /* No isolated shim was built; skip (see xtbloom.cpu.linalg_isolation). */
#else
  std::string error;
  CpuLinearAlgebraBackend backend;
  if (xtbloom::detail::gfn2::make_mkl_rt_lp64_backend(backend, error) == XTBLOOM_STATUS_SUCCESS &&
      backend.ready() && !backend.production_mkl()) {
    return 0; /* OpenBLAS: no MKL interface-layer state to isolate. */
  }
  CHECK(backend.ready());
  CHECK(backend.production_mkl_isolated());

  int solve_status = run_literal_generalized_eigenproblem(backend, error);
  CHECK(solve_status == 0);

  void* host_handle = nullptr;
  SetInterfaceLayer host_set_interface = nullptr;
  DpotrfWork64 host_dpotrf64 = nullptr;
  host_mkl(&host_handle, &host_set_interface, &host_dpotrf64);
  CHECK(host_handle != nullptr);
  CHECK(dlsym(RTLD_DEFAULT, "MKL_Set_Interface_Layer") != nullptr);
  const int ilp64_acceptance = host_set_interface(1); /* Host switches to ILP64 after xtbloom. */
  CHECK(ilp64_acceptance == 1);
  CHECK(host_ilp64_works(host_handle, host_dpotrf64) == 0);

  solve_status = run_literal_generalized_eigenproblem(backend, error);
  CHECK(solve_status == 0);

  CHECK(host_ilp64_works(host_handle, host_dpotrf64) == 0);
  static_cast<void>(dlclose(host_handle));
  return 0;
#endif
}

/* After xtbloom creates its backend, MKL/LAPACK symbols must not appear in the
 * process-global namespace: the provider is loaded into a new link-map
 * namespace, and the xtbloom path never loads libmkl_rt at all. */
int run_no_global_scope_exposure() {
#if !defined(_WIN32)
#if defined(XTBLOOM_TEST_HAS_WHEEL_OPENBLAS)
  const char* decoy_path = std::getenv("XTBLOOM_TEST_OPENBLAS_DECOY");
  CHECK(decoy_path != nullptr);
  void* decoy = dlopen(decoy_path, RTLD_NOW | RTLD_GLOBAL);
  CHECK(decoy != nullptr);
#endif
  void* mkl_global = dlsym(RTLD_DEFAULT, "MKL_Set_Interface_Layer");
  void* lapacke_global = dlsym(RTLD_DEFAULT, "LAPACKE_dpotrf_work");
  void* scipy_lapacke_global = dlsym(RTLD_DEFAULT, "scipy_LAPACKE_dpotrf_work");
  CHECK(mkl_global == nullptr);
  CHECK(lapacke_global == nullptr);
  CHECK(scipy_lapacke_global == nullptr);
#endif
  const int correct = run_correctness_with_backend();
  CHECK(correct == 0);
#if !defined(_WIN32)
  mkl_global = dlsym(RTLD_DEFAULT, "MKL_Set_Interface_Layer");
  lapacke_global = dlsym(RTLD_DEFAULT, "LAPACKE_dpotrf_work");
  scipy_lapacke_global = dlsym(RTLD_DEFAULT, "scipy_LAPACKE_dpotrf_work");
  CHECK(mkl_global == nullptr);
  CHECK(lapacke_global == nullptr);
  CHECK(scipy_lapacke_global == nullptr);
#endif
  return 0;
}

int wait_and_check(const pid_t child) {
  int child_status = 0;
  pid_t waited = -1;
  do {
    waited = waitpid(child, &child_status, 0);
  } while (waited < 0 && errno == EINTR);
  CHECK(waited == child);
  CHECK(WIFEXITED(child_status));
  CHECK(WEXITSTATUS(child_status) == 0);
  return 0;
}

int run_expect_missing_shim() {
  std::string error;
  CpuLinearAlgebraBackend backend;
  const xtbloom_status_t status = xtbloom::detail::gfn2::make_mkl_rt_lp64_backend(backend, error);
  CHECK(status == XTBLOOM_STATUS_BACKEND_UNAVAILABLE);
  CHECK(!backend.ready());
#if defined(XTBLOOM_TEST_HAS_WHEEL_OPENBLAS)
  CHECK(error.find("private wheel OpenBLAS provider") != std::string::npos);
#else
  CHECK(error.find("host-isolated MKL provider shim") != std::string::npos);
#endif
  return 0;
}

/* Execute a byte-for-byte copy of this standalone test without its adjacent
 * shim. This proves the factory neither follows a baked build-tree path nor
 * falls back to a process-global libmkl_rt when the private artifact is gone. */
int run_missing_shim_failure() {
  char temporary_template[] = "/tmp/xtbloom-linalg-isolation-XXXXXX";
  char* temporary_directory = mkdtemp(temporary_template);
  CHECK(temporary_directory != nullptr);

  char source_path[4096]{};
  const ssize_t source_size = readlink("/proc/self/exe", source_path, sizeof(source_path) - 1u);
  CHECK(source_size > 0);
  source_path[static_cast<std::size_t>(source_size)] = '\0';
  const std::string copied_path = std::string(temporary_directory) + "/isolation-test";

  {
    std::ifstream source(source_path, std::ios::binary);
    std::ofstream destination(copied_path, std::ios::binary | std::ios::trunc);
    CHECK(source.good());
    CHECK(destination.good());
    destination << source.rdbuf();
    CHECK(destination.good());
  }
  CHECK(chmod(copied_path.c_str(), 0700) == 0);

  const pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    execl(copied_path.c_str(), copied_path.c_str(), "--expect-missing-shim",
          static_cast<char*>(nullptr));
    _exit(121);
  }
  const int result = wait_and_check(child);
  CHECK(unlink(copied_path.c_str()) == 0);
  CHECK(rmdir(temporary_directory) == 0);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  const char* mode = argc >= 2 ? argv[1] : "--all";
  if (std::strcmp(mode, "--no-global-scope-exposure") == 0) {
    return run_no_global_scope_exposure();
  }
  if (std::strcmp(mode, "--coexistence-xtbloom-after-host-ilp64") == 0) {
    return run_coexistence_xtbloom_after_host_ilp64();
  }
  if (std::strcmp(mode, "--coexistence-host-ilp64-after-xtbloom") == 0) {
    return run_coexistence_host_ilp64_after_xtbloom();
  }
  if (std::strcmp(mode, "--correctness") == 0) {
    return run_correctness_with_backend();
  }
  if (std::strcmp(mode, "--expect-missing-shim") == 0) {
    return run_expect_missing_shim();
  }

  int result = 0;
  char executable_path[4096]{};
  const ssize_t executable_path_size =
      readlink("/proc/self/exe", executable_path, sizeof(executable_path) - 1u);
  CHECK(executable_path_size > 0);
  executable_path[static_cast<std::size_t>(executable_path_size)] = '\0';

  const pid_t child_global = fork();
  if (child_global == 0) {
    /* dladdr resolves the provider relative to this executable in the static
     * test binary; use its real path rather than the /proc/self symlink. */
    execl(executable_path, executable_path, "--no-global-scope-exposure",
          static_cast<char*>(nullptr));
    _exit(121);
  }
  result = wait_and_check(child_global);
  if (result != 0) {
    return result;
  }

  result = run_missing_shim_failure();
  if (result != 0) {
    return result;
  }

  const pid_t child_xtbloom_after = fork();
  if (child_xtbloom_after == 0) {
    execl(executable_path, executable_path, "--coexistence-xtbloom-after-host-ilp64",
          static_cast<char*>(nullptr));
    _exit(121);
  }
  result = wait_and_check(child_xtbloom_after);
  if (result != 0) {
    return result;
  }

  const pid_t child_host_after = fork();
  if (child_host_after == 0) {
    execl(executable_path, executable_path, "--coexistence-host-ilp64-after-xtbloom",
          static_cast<char*>(nullptr));
    _exit(121);
  }
  result = wait_and_check(child_host_after);
  if (result != 0) {
    return result;
  }

  return result;
}
