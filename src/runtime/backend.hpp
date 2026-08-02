#ifndef GPUXTB_RUNTIME_BACKEND_HPP
#define GPUXTB_RUNTIME_BACKEND_HPP

#include <cstdint>
#include <string>

#include "gpuxtb/gpuxtb.h"

namespace gpuxtb::detail {

/* Runtime state is opaque at the ABI boundary so backend internals can evolve. */
struct Context {
  gpuxtb_backend_t backend = GPUXTB_BACKEND_CPU;
  std::int32_t device_id = -1;
  std::int32_t cpu_threads = 0;
  void* stream = nullptr;
};

gpuxtb_status_t create_context(const gpuxtb_context_options_t& options, Context*& context,
                               std::string& error);

#if defined(GPUXTB_HAS_CUDA)
bool resolve_cuda_device(std::int32_t requested_device, std::int32_t& resolved_device,
                         std::string& error);
#endif

}  // namespace gpuxtb::detail

#endif  // GPUXTB_RUNTIME_BACKEND_HPP
