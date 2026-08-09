#include <cstdint>
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

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

class FakeCompletion final : public gpuxtb::detail::RequestCompletion {
 public:
  explicit FakeCompletion(std::shared_ptr<ProbeCounts> counts) : counts_(std::move(counts)) {}

  gpuxtb_status_t probe(bool wait,
                        gpuxtb::detail::RequestCompletionResult& result) noexcept override {
    if (!wait) {
      ++counts_->nonblocking;
      return GPUXTB_STATUS_SUCCESS;
    }
    ++counts_->blocking;
    result.complete = true;
    result.completion_status = GPUXTB_STATUS_INTERNAL_ERROR;
    result.result_flags = GPUXTB_RESULT_DIPOLE_MOMENTS;
    result.completion_error = "deferred test failure";
    return GPUXTB_STATUS_SUCCESS;
  }

  void settle_noexcept() noexcept override { ++counts_->settled; }

 private:
  std::shared_ptr<ProbeCounts> counts_;
};

gpuxtb_request_info_t initialized_info() {
  gpuxtb_request_info_t info{};
  info.struct_size = sizeof(info);
  info.api_version = GPUXTB_API_VERSION;
  return info;
}

}  // namespace

void test_request_reserve_complete_and_reuse() {
  gpuxtb::detail::Context context;
  context.backend = GPUXTB_BACKEND_CPU;
  gpuxtb::detail::Context other_context;
  other_context.backend = GPUXTB_BACKEND_CPU;
  gpuxtb::detail::Request request(context);
  std::string error;
  gpuxtb_request_info_t info = initialized_info();

  CHECK(request.query(false, info, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(info.state == GPUXTB_REQUEST_IDLE);
  CHECK(request.reserve_submission(other_context, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  CHECK(request.reserve_submission(context, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(request.query(false, info, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(info.state == GPUXTB_REQUEST_IDLE);  // SUBMITTING is intentionally internal.
  CHECK(request.reserve_submission(context, error) == GPUXTB_STATUS_INVALID_ARGUMENT);
  request.rollback_submission();
  CHECK(request.query(false, info, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(info.state == GPUXTB_REQUEST_IDLE);

  CHECK(request.reserve_submission(context, error) == GPUXTB_STATUS_SUCCESS);
  gpuxtb::detail::RequestSubmission inline_submission;
  inline_submission.completed_inline = true;
  inline_submission.completion_status = GPUXTB_STATUS_ALLOCATION_FAILED;
  inline_submission.result_flags = GPUXTB_RESULT_DIPOLE_MOMENTS;
  inline_submission.completion_error = "inline test failure";
  CHECK(request.publish_submission(std::move(inline_submission), error) == GPUXTB_STATUS_SUCCESS);
  CHECK(request.query(false, info, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(info.state == GPUXTB_REQUEST_COMPLETE);
  CHECK(info.completion_status == GPUXTB_STATUS_ALLOCATION_FAILED);
  CHECK(info.result_flags == GPUXTB_RESULT_DIPOLE_MOMENTS);
  CHECK(std::string(request.error()) == "inline test failure");

  /* SUBMITTING is invisible until native work is accepted, so a failed reuse
   * can roll back to the preceding terminal snapshot exactly. */
  CHECK(request.reserve_submission(context, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(request.query(false, info, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(info.state == GPUXTB_REQUEST_COMPLETE);
  CHECK(info.completion_status == GPUXTB_STATUS_ALLOCATION_FAILED);
  CHECK(std::string(request.error()) == "inline test failure");
  request.rollback_submission();
  CHECK(request.query(false, info, error) == GPUXTB_STATUS_SUCCESS);
  CHECK(info.state == GPUXTB_REQUEST_COMPLETE);
}

void test_request_probe_and_destroy() {
  gpuxtb::detail::Context context;
  context.backend = GPUXTB_BACKEND_CUDA;
  auto counts = std::make_shared<ProbeCounts>();
  {
    gpuxtb::detail::Request request(context);
    std::string error;
    CHECK(request.reserve_submission(context, error) == GPUXTB_STATUS_SUCCESS);
    gpuxtb::detail::RequestSubmission submission;
    submission.pending = std::make_unique<FakeCompletion>(counts);
    CHECK(request.publish_submission(std::move(submission), error) == GPUXTB_STATUS_SUCCESS);

    gpuxtb_request_info_t info = initialized_info();
    CHECK(request.query(false, info, error) == GPUXTB_STATUS_SUCCESS);
    CHECK(info.state == GPUXTB_REQUEST_PENDING);
    CHECK(counts->nonblocking == 1);
    CHECK(request.query(true, info, error) == GPUXTB_STATUS_SUCCESS);
    CHECK(info.state == GPUXTB_REQUEST_COMPLETE);
    CHECK(info.completion_status == GPUXTB_STATUS_INTERNAL_ERROR);
    CHECK(info.result_flags == GPUXTB_RESULT_DIPOLE_MOMENTS);
    CHECK(std::string(request.error()) == "deferred test failure");
    CHECK(counts->blocking == 1);
    CHECK(counts->settled == 1);
  }

  auto destroy_counts = std::make_shared<ProbeCounts>();
  {
    gpuxtb::detail::Request request(context);
    std::string error;
    CHECK(request.reserve_submission(context, error) == GPUXTB_STATUS_SUCCESS);
    gpuxtb::detail::RequestSubmission submission;
    submission.pending = std::make_unique<FakeCompletion>(destroy_counts);
    CHECK(request.publish_submission(std::move(submission), error) == GPUXTB_STATUS_SUCCESS);
  }
  CHECK(destroy_counts->blocking == 1);
  CHECK(destroy_counts->settled == 1);
}

int main() {
  test_request_reserve_complete_and_reuse();
  test_request_probe_and_destroy();
  return failures == 0 ? 0 : 1;
}
