#include "runtime/request.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <utility>

#include "runtime/backend.hpp"

namespace xtbloom::detail {

namespace {

constexpr char kEmptyRequestError[] = "";

}  // namespace

Request::Request(Context& context) noexcept : context_(&context) {}

Request::~Request() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (lifecycle_ == Lifecycle::kPending && completion_ != nullptr) {
    /* Destruction is a synchronization boundary for this request only. Ignore
     * the API-access diagnostic here: RequestCompletion's destructor remains
     * responsible for safe ownership cleanup even when the explicit wait
     * operation itself cannot report success. */
    std::string ignored_error;
    (void)probe_locked(true, ignored_error);
    if (completion_ != nullptr) {
      completion_->settle_noexcept();
    }
  }
  completion_.reset();
}

xtbloom_backend_t Request::backend() const noexcept { return context_->backend; }

xtbloom_status_t Request::reserve_submission(Context& context, std::string& error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (&context != context_) {
    error = "request was created by a different context";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (lifecycle_ != Lifecycle::kStable) {
    error = "request already has a pending or submitting operation";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  /* SUBMITTING is internal. Preserve the prior public snapshot until native
   * work has actually been accepted, so a staging failure can roll back with
   * no observable request mutation. */
  lifecycle_ = Lifecycle::kSubmitting;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

void Request::rollback_submission() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (lifecycle_ == Lifecycle::kSubmitting) {
    lifecycle_ = Lifecycle::kStable;
  }
}

xtbloom_status_t Request::publish_submission(RequestSubmission submission, std::string& error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (lifecycle_ != Lifecycle::kSubmitting) {
    error = "request has no reserved submission to publish";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (submission.completed_inline == (submission.pending != nullptr)) {
    error = "request submission must be either pending or completed inline";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (submission.completed_inline) {
    lifecycle_ = Lifecycle::kStable;
    state_ = XTBLOOM_REQUEST_COMPLETE;
    completion_status_ = submission.completion_status;
    result_flags_ = submission.result_flags;
    error_ = std::move(submission.completion_error);
  } else {
    completion_ = std::move(submission.pending);
    lifecycle_ = Lifecycle::kPending;
    state_ = XTBLOOM_REQUEST_PENDING;
    completion_status_ = XTBLOOM_STATUS_SUCCESS;
    result_flags_ = 0u;
    error_.clear();
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

void Request::fill_info_locked(xtbloom_request_info_t& info) const noexcept {
  info.state = state_;
  info.completion_status = completion_status_;
  info.result_flags = result_flags_;
}

xtbloom_status_t Request::probe_locked(bool wait, std::string& error) {
  if (lifecycle_ != Lifecycle::kPending) {
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  }
  if (completion_ == nullptr) {
    error = "pending request has no completion owner";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  RequestCompletionResult result;
  const xtbloom_status_t status = completion_->probe(wait, result);
  if (status == XTBLOOM_STATUS_SUCCESS && wait && !result.complete) {
    error = "blocking request completion probe returned an incomplete state";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  if (status != XTBLOOM_STATUS_SUCCESS || !result.complete) {
    if (status != XTBLOOM_STATUS_SUCCESS) {
      error = result.completion_error.empty() ? "failed to access request completion"
                                              : result.completion_error;
    } else {
      error.clear();
    }
    return status;
  }
  state_ = XTBLOOM_REQUEST_COMPLETE;
  completion_status_ = result.completion_status;
  result_flags_ = result.result_flags;
  error_ = std::move(result.completion_error);
  lifecycle_ = Lifecycle::kStable;
  completion_->settle_noexcept();
  completion_.reset();
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t Request::query(bool wait, xtbloom_request_info_t& info, std::string& error) {
  std::lock_guard<std::mutex> lock(mutex_);
  const xtbloom_status_t status = probe_locked(wait, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  fill_info_locked(info);
  return XTBLOOM_STATUS_SUCCESS;
}

const char* Request::error() const noexcept {
  /* The public contract keeps this pointer valid until reuse or destruction.
   * Concurrent reuse with error inspection is therefore outside the contract,
   * just like retaining the pointer past the next enqueue. IDLE/PENDING and
   * successful completion return process-lifetime storage so installing a
   * later deferred diagnostic cannot invalidate an earlier empty pointer. */
  std::lock_guard<std::mutex> lock(mutex_);
  return error_.empty() ? kEmptyRequestError : error_.c_str();
}

}  // namespace xtbloom::detail
