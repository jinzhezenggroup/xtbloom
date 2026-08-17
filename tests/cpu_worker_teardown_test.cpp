#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#endif

#include "runtime/backend.hpp"
#include "runtime/gfn2_cpu_execution.hpp"
#include "xtbloom/xtbloom.h"

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      std::cerr << "CHECK failed at line " << __LINE__ << ": " << #condition \
                << "; last error: " << xtbloom_get_last_error() << '\n';     \
      return __LINE__;                                                       \
    }                                                                        \
  } while (false)

namespace {

template <typename T>
xtbloom_const_buffer_t input_buffer(const std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0};
}

template <typename T>
xtbloom_buffer_t output_buffer(std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T), XTBLOOM_MEMORY_HOST,
          0};
}

struct ContextDeleter {
  void operator()(xtbloom_context_t* context) const noexcept { xtbloom_context_destroy(context); }
};
using ContextHandle = std::unique_ptr<xtbloom_context_t, ContextDeleter>;

ContextHandle make_cpu_context(std::int32_t cpu_threads) {
  xtbloom_context_options_t options{};
  if (xtbloom_context_options_init(&options, sizeof(options)) != XTBLOOM_STATUS_SUCCESS) {
    return {};
  }
  options.backend = XTBLOOM_BACKEND_CPU;
  options.cpu_threads = cpu_threads;
  xtbloom_context_t* raw_context = nullptr;
  if (xtbloom_context_create(&options, &raw_context) != XTBLOOM_STATUS_SUCCESS) {
    return {};
  }
  return ContextHandle(raw_context);
}

int test_context_model_caches_are_lazy_and_independent() {
  xtbloom_context_options_t options{};
  CHECK(xtbloom_context_options_init(&options, sizeof(options)) == XTBLOOM_STATUS_SUCCESS);
  options.backend = XTBLOOM_BACKEND_CPU;
  options.cpu_threads = 4;

  xtbloom::detail::Context* gfn2_context = nullptr;
  std::string error;
  CHECK(xtbloom::detail::create_context(options, gfn2_context, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(gfn2_context != nullptr);
  CHECK(gfn2_context->gfn1_cpu_execution_cache == nullptr);
  CHECK(gfn2_context->gfn2_cpu_execution_cache == nullptr);
  CHECK(xtbloom::detail::ensure_gfn2_cpu_execution_cache(*gfn2_context, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(gfn2_context->gfn2_cpu_execution_cache != nullptr);
  CHECK(gfn2_context->gfn1_cpu_execution_cache == nullptr);
  delete gfn2_context;

  xtbloom::detail::Context* gfn1_context = nullptr;
  CHECK(xtbloom::detail::create_context(options, gfn1_context, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(gfn1_context != nullptr);
  CHECK(gfn1_context->gfn1_cpu_execution_cache == nullptr);
  CHECK(gfn1_context->gfn2_cpu_execution_cache == nullptr);
  CHECK(xtbloom::detail::ensure_gfn1_cpu_execution_cache(*gfn1_context, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(gfn1_context->gfn1_cpu_execution_cache != nullptr);
  CHECK(gfn1_context->gfn2_cpu_execution_cache == nullptr);
  delete gfn1_context;
  return 0;
}

struct TwoSystemBatch {
  std::vector<std::int64_t> atom_offsets{0, 2, 4};
  std::vector<std::int32_t> atomic_numbers{1, 1, 1, 1};
  std::vector<double> positions{-0.70, 0.0, 0.0, 0.70, 0.0, 0.0, -0.72, 0.1, 0.0, 0.72, 0.1, 0.0};
  std::vector<double> charges{0.0, 0.0};
  std::vector<std::int32_t> unpaired{0, 0};
  std::vector<double> energies{-1.0, -1.0};
  std::vector<std::int32_t> iterations{-1, -1};
  std::vector<std::uint8_t> converged{0u, 0u};
  std::vector<std::int32_t> statuses{-1, -1};
  xtbloom_batch_t batch{};
  xtbloom_compute_options_t options{};
  xtbloom_batch_result_t result{};

  TwoSystemBatch() {
    xtbloom_batch_init(&batch, sizeof(batch));
    xtbloom_compute_options_init(&options, sizeof(options));
    xtbloom_batch_result_init(&result, sizeof(result));
    batch.batch_size = 2;
    batch.total_atoms = 4;
    batch.atom_offsets = input_buffer(atom_offsets);
    batch.atomic_numbers = input_buffer(atomic_numbers);
    batch.positions = input_buffer(positions);
    batch.molecular_charges = input_buffer(charges);
    batch.unpaired_electrons = input_buffer(unpaired);
    options.flags = XTBLOOM_COMPUTE_ENERGY;
    result.energies = output_buffer(energies);
    result.scc_iterations = output_buffer(iterations);
    result.scc_converged = output_buffer(converged);
    result.per_system_status = output_buffer(statuses);
  }
};

int run_context(std::int32_t cpu_threads, bool expect_background_worker) {
  xtbloom::detail::reset_gfn2_cpu_worker_teardown_test_counters();
  TwoSystemBatch request;
  ContextHandle context = make_cpu_context(cpu_threads);
  CHECK(context != nullptr);
  CHECK(xtbloom_compute(context.get(), &request.batch, &request.options, &request.result) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(request.converged == std::vector<std::uint8_t>({1u, 1u}));
  CHECK(request.statuses ==
        std::vector<std::int32_t>({XTBLOOM_STATUS_SUCCESS, XTBLOOM_STATUS_SUCCESS}));

  const bool provider_needs_cleanup =
      xtbloom::detail::gfn2_cpu_test_provider_requires_thread_cleanup();
  const std::size_t background_runs = xtbloom::detail::gfn2_cpu_test_background_eigensolver_runs();
  if (expect_background_worker) {
    /* The testing worker pool hands one system to a persistent worker before
     * the caller may drain work, so this is a deterministic production-path
     * assertion rather than a scheduler-dependent observation. */
    CHECK(background_runs >= 1u);
  } else {
    CHECK(background_runs == 0u);
  }

  context.reset();
  const std::size_t cleanup_calls = xtbloom::detail::gfn2_cpu_test_background_thread_cleanups();
  CHECK(cleanup_calls == (provider_needs_cleanup && expect_background_worker ? 1u : 0u));
  return 0;
}

}  // namespace

int main() {
#if defined(__linux__)
  /* glibc bug nptl/24776: a second pthread implementation loaded with
   * dlmopen can allocate the same key number as the host while both namespaces
   * write the same THREAD_SELF specific-data slots. Reserve a host TSS key
   * before MKL initialization and prove xTBloom never overwrites it. This is a
   * native analogue of CPython's _Py_tss_tstate / gilstate keys from #381. */
  pthread_key_t host_tss_key{};
  int host_tss_sentinel = 0;
  CHECK(pthread_key_create(&host_tss_key, nullptr) == 0);
  CHECK(pthread_setspecific(host_tss_key, &host_tss_sentinel) == 0);
#endif

  /* Exercise the context-owner thread before creating any background worker.
   * This is the path that can corrupt a CPython caller even when cpu_threads=1. */
  if (const int line = run_context(1, false); line != 0) {
    return line;
  }
#if defined(__linux__)
  CHECK(pthread_getspecific(host_tss_key) == &host_tss_sentinel);
#endif

  if (const int line = test_context_model_caches_are_lazy_and_independent(); line != 0) {
    return line;
  }
  for (int repetition = 0; repetition < 8; ++repetition) {
    if (const int line = run_context(2, true); line != 0) {
      return line;
    }
#if defined(__linux__)
    CHECK(pthread_getspecific(host_tss_key) == &host_tss_sentinel);
#endif
  }

#if defined(__linux__)
  CHECK(pthread_key_delete(host_tss_key) == 0);
#endif
  return 0;
}
