#include <cstdint>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cstdio>
#include <memory>
#include <string>

#include "runtime/backend.hpp"
#include "runtime/request.hpp"

namespace {

int failures = 0;

void check(bool condition, const char* expression, int line) {
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "request_test:%d: CHECK(%s) failed\n", line, expression);
  }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

struct ProbeCounts {
  int nonblocking = 0;
  int blocking = 0;
  int settled = 0;
};

class FakeCompletion final : public xtbloom::detail::RequestCompletion {
 public:
  explicit FakeCompletion(std::shared_ptr<ProbeCounts> counts) : counts_(std::move(counts)) {}

  xtbloom_status_t probe(bool wait,
                         xtbloom::detail::RequestCompletionResult& result) noexcept override {
    if (!wait) {
      ++counts_->nonblocking;
      return XTBLOOM_STATUS_SUCCESS;
    }
    ++counts_->blocking;
    result.complete = true;
    result.completion_status = XTBLOOM_STATUS_INTERNAL_ERROR;
    result.result_flags = XTBLOOM_RESULT_DIPOLE_MOMENTS;
    result.completion_error = "deferred test failure";
    return XTBLOOM_STATUS_SUCCESS;
  }

  void settle_noexcept() noexcept override { ++counts_->settled; }

 private:
  std::shared_ptr<ProbeCounts> counts_;
};

class FailingWaitCompletion final : public xtbloom::detail::RequestCompletion {
 public:
  explicit FailingWaitCompletion(std::shared_ptr<ProbeCounts> counts)
      : counts_(std::move(counts)) {}

  xtbloom_status_t probe(bool wait,
                         xtbloom::detail::RequestCompletionResult& result) noexcept override {
    if (!wait) {
      ++counts_->nonblocking;
      return XTBLOOM_STATUS_SUCCESS;
    }
    ++counts_->blocking;
    result.completion_error = "exact wait failed";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }

  void settle_noexcept() noexcept override { ++counts_->settled; }

 private:
  std::shared_ptr<ProbeCounts> counts_;
};

xtbloom_request_info_t initialized_info() {
  xtbloom_request_info_t info{};
  info.struct_size = sizeof(info);
  info.api_version = XTBLOOM_API_VERSION;
  return info;
}

}  // namespace

void test_request_reserve_complete_and_reuse() {
  xtbloom::detail::Context context;
  context.backend = XTBLOOM_BACKEND_CPU;
  xtbloom::detail::Context other_context;
  other_context.backend = XTBLOOM_BACKEND_CPU;
  xtbloom::detail::Request request(context);
  std::string error;
  xtbloom_request_info_t info = initialized_info();

  CHECK(request.query(false, info, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.state == XTBLOOM_REQUEST_IDLE);
  CHECK(request.reserve_submission(other_context, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(request.reserve_submission(context, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(request.query(false, info, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.state == XTBLOOM_REQUEST_IDLE);  // SUBMITTING is intentionally internal.
  CHECK(request.reserve_submission(context, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  request.rollback_submission();
  CHECK(request.query(false, info, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.state == XTBLOOM_REQUEST_IDLE);

  CHECK(request.reserve_submission(context, error) == XTBLOOM_STATUS_SUCCESS);
  xtbloom::detail::RequestSubmission inline_submission;
  inline_submission.completed_inline = true;
  inline_submission.completion_status = XTBLOOM_STATUS_ALLOCATION_FAILED;
  inline_submission.result_flags = XTBLOOM_RESULT_DIPOLE_MOMENTS;
  inline_submission.completion_error = "inline test failure";
  CHECK(request.publish_submission(std::move(inline_submission), error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(request.query(false, info, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.state == XTBLOOM_REQUEST_COMPLETE);
  CHECK(info.completion_status == XTBLOOM_STATUS_ALLOCATION_FAILED);
  CHECK(info.result_flags == XTBLOOM_RESULT_DIPOLE_MOMENTS);
  CHECK(std::string(request.error()) == "inline test failure");

  /* SUBMITTING is invisible until native work is accepted, so a failed reuse
   * can roll back to the preceding terminal snapshot exactly. */
  CHECK(request.reserve_submission(context, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(request.query(false, info, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.state == XTBLOOM_REQUEST_COMPLETE);
  CHECK(info.completion_status == XTBLOOM_STATUS_ALLOCATION_FAILED);
  CHECK(std::string(request.error()) == "inline test failure");
  request.rollback_submission();
  CHECK(request.query(false, info, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(info.state == XTBLOOM_REQUEST_COMPLETE);
  CHECK(info.completion_status == XTBLOOM_STATUS_ALLOCATION_FAILED);
  CHECK(info.result_flags == XTBLOOM_RESULT_DIPOLE_MOMENTS);
  CHECK(std::string(request.error()) == "inline test failure");
}

void test_request_probe_and_destroy() {
  xtbloom::detail::Context context;
  context.backend = XTBLOOM_BACKEND_CUDA;
  auto counts = std::make_shared<ProbeCounts>();
  {
    xtbloom::detail::Request request(context);
    std::string error;
    CHECK(request.reserve_submission(context, error) == XTBLOOM_STATUS_SUCCESS);
    xtbloom::detail::RequestSubmission submission;
    submission.pending = std::make_shared<FakeCompletion>(counts);
    CHECK(request.publish_submission(std::move(submission), error) == XTBLOOM_STATUS_SUCCESS);

    const char* pending_error = request.error();
    CHECK(pending_error != nullptr);
    CHECK(std::string(pending_error).empty());

    xtbloom_request_info_t info = initialized_info();
    CHECK(request.query(false, info, error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(info.state == XTBLOOM_REQUEST_PENDING);
    CHECK(counts->nonblocking == 1);
    CHECK(request.query(true, info, error) == XTBLOOM_STATUS_SUCCESS);
    CHECK(info.state == XTBLOOM_REQUEST_COMPLETE);
    CHECK(info.completion_status == XTBLOOM_STATUS_INTERNAL_ERROR);
    CHECK(info.result_flags == XTBLOOM_RESULT_DIPOLE_MOMENTS);
    CHECK(std::string(request.error()) == "deferred test failure");
    /* The public pointer lifetime extends through query/wait. Completing with
     * a longer diagnostic must not invalidate the earlier PENDING empty view. */
    CHECK(std::string(pending_error).empty());
    CHECK(counts->blocking == 1);
    CHECK(counts->settled == 1);
  }

  auto destroy_counts = std::make_shared<ProbeCounts>();
  {
    xtbloom::detail::Request request(context);
    std::string error;
    CHECK(request.reserve_submission(context, error) == XTBLOOM_STATUS_SUCCESS);
    xtbloom::detail::RequestSubmission submission;
    submission.pending = std::make_shared<FakeCompletion>(destroy_counts);
    CHECK(request.publish_submission(std::move(submission), error) == XTBLOOM_STATUS_SUCCESS);
  }
  CHECK(destroy_counts->blocking == 1);
  CHECK(destroy_counts->settled == 1);

  auto failed_wait_counts = std::make_shared<ProbeCounts>();
  {
    xtbloom::detail::Request request(context);
    std::string error;
    CHECK(request.reserve_submission(context, error) == XTBLOOM_STATUS_SUCCESS);
    xtbloom::detail::RequestSubmission submission;
    submission.pending = std::make_shared<FailingWaitCompletion>(failed_wait_counts);
    CHECK(request.publish_submission(std::move(submission), error) == XTBLOOM_STATUS_SUCCESS);
  }
  CHECK(failed_wait_counts->blocking == 1);
  CHECK(failed_wait_counts->settled == 1);
}

int main() {
  test_request_reserve_complete_and_reuse();
  test_request_probe_and_destroy();
  return failures == 0 ? 0 : 1;
}
