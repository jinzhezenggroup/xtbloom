#ifndef XTBLOOM_RUNTIME_REQUEST_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_REQUEST_HPP

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail {

struct Context;

struct RequestCompletionResult {
  bool complete = false;
  xtbloom_status_t completion_status = XTBLOOM_STATUS_SUCCESS;
  std::uint32_t result_flags = 0u;
  std::string completion_error;
};

/*
 * Type-erased owner of one backend completion primitive. probe(false) must be
 * nonblocking; probe(true) waits for this exact submission. The API return
 * value describes access to that primitive, while result carries the compute
 * status only when complete is true. Destruction must safely settle or release
 * the owned primitive even when a preceding query operation failed. A derived
 * destructor or settle_noexcept() must settle any still-active submission;
 * the base destructor alone has no backend resource knowledge.
 */
class RequestCompletion {
 public:
  virtual ~RequestCompletion() = default;
  [[nodiscard]] virtual xtbloom_status_t probe(bool wait,
                                               RequestCompletionResult& result) noexcept = 0;
  virtual void settle_noexcept() noexcept = 0;
};

struct RequestSubmission {
  std::shared_ptr<RequestCompletion> pending;
  bool completed_inline = false;
  xtbloom_status_t completion_status = XTBLOOM_STATUS_SUCCESS;
  std::uint32_t result_flags = 0u;
  std::string completion_error;
};

/*
 * Backend-neutral state owner for one reusable asynchronous submission.
 *
 * CUDA execution attaches a type-erased exact-event owner after reserving the
 * handle. Keeping the public state machine independent of CUDA types lets
 * CPU-only libraries expose the same additive ABI and lets CPU callers
 * create/query/destroy an IDLE request even though enqueue itself is
 * deliberately unsupported there.
 */
class Request {
 public:
  explicit Request(Context& context) noexcept;
  ~Request();

  Request(const Request&) = delete;
  Request& operator=(const Request&) = delete;

  [[nodiscard]] Context* context() const noexcept { return context_; }
  [[nodiscard]] xtbloom_backend_t backend() const noexcept;

  /*
   * Reserve an IDLE/COMPLETE handle before backend staging. SUBMITTING is an
   * internal state: public query continues to expose the preceding stable
   * snapshot until publish_submission() succeeds. A synchronous
   * pre-acceptance failure must call rollback_submission(), preserving an
   * earlier IDLE or COMPLETE snapshot and its diagnostic exactly.
   */
  [[nodiscard]] xtbloom_status_t reserve_submission(Context& context, std::string& error);
  void rollback_submission() noexcept;
  [[nodiscard]] xtbloom_status_t publish_submission(RequestSubmission submission,
                                                    std::string& error);

  /* Copy a coherent state snapshot without changing the request. */
  [[nodiscard]] xtbloom_status_t query(bool wait, xtbloom_request_info_t& info, std::string& error);

  [[nodiscard]] const char* error() const noexcept;

 private:
  void fill_info_locked(xtbloom_request_info_t& info) const noexcept;
  [[nodiscard]] xtbloom_status_t probe_locked(bool wait, std::string& error);

  enum class Lifecycle : std::uint8_t { kStable, kSubmitting, kPending };

  Context* context_;
  mutable std::mutex mutex_;
  Lifecycle lifecycle_ = Lifecycle::kStable;
  xtbloom_request_state_t state_ = XTBLOOM_REQUEST_IDLE;
  xtbloom_status_t completion_status_ = XTBLOOM_STATUS_SUCCESS;
  std::uint32_t result_flags_ = 0u;
  std::string error_;
  std::shared_ptr<RequestCompletion> completion_;
};

}  // namespace xtbloom::detail

#endif  // XTBLOOM_RUNTIME_REQUEST_HPP
