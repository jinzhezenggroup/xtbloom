#ifndef XTBLOOM_BINDINGS_TORCH_REQUEST_POOL_SHUTDOWN_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BINDINGS_TORCH_REQUEST_POOL_SHUTDOWN_HPP

#include <atomic>
#include <cstdint>

namespace xtbloom::torch_detail {

// Return whether cleanup may inspect this process's published holder. A false
// result must not touch atomic shared-pointer operations or inherited locks.
// The atomic store type is generic so the same decision can be exercised with
// a barrier-controlled store in a CPU-only test, without runtime test hooks.
template <typename ShutdownPid>
static bool close_request_pool_admission(const std::atomic<std::uint64_t>& owner_pid,
                                         ShutdownPid& shutdown_pid,
                                         std::uint64_t process_id) noexcept {
  const auto observed_owner_pid = owner_pid.load();
  if (observed_owner_pid != 0 && observed_owner_pid != process_id) return false;
  shutdown_pid.store(process_id);
  // First use can claim ownership, publish the holder and admit work between
  // the initial load and closing admission. Only a fresh post-close load can
  // decide that there is no local owner to drain. Later claimants see the
  // closed admission gate before they can submit work.
  return owner_pid.load() == process_id;
}

}  // namespace xtbloom::torch_detail

#endif
