#include "bindings/torch/request_pool_shutdown.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>

namespace {

constexpr std::uint64_t kProcessId = 101;
constexpr std::uint64_t kForeignProcessId = 202;

void require(bool condition, const char* message) {
  if (condition) return;
  std::fprintf(stderr, "FAIL: %s\n", message);
  // A failed assertion may leave a test worker at a barrier. Exit instead of
  // attempting an unbounded join or destroying a still-joinable thread.
  std::exit(EXIT_FAILURE);
}

class Event {
 public:
  void signal() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      signaled_ = true;
    }
    changed_.notify_all();
  }

  void wait(const char* message) {
    std::unique_lock<std::mutex> lock(mutex_);
    require(changed_.wait_for(lock, std::chrono::seconds(5), [&] { return signaled_; }), message);
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool signaled_ = false;
};

// Only the test instantiation substitutes the store. Production uses the
// existing std::atomic directly, with no callback, branch or extra allocation.
struct PausedClose {
  std::atomic<std::uint64_t> value{0};
  Event before_store;
  Event allow_store;

  void store(std::uint64_t process_id) {
    before_store.signal();
    allow_store.wait("cleanup did not receive permission to close admission");
    value.store(process_id);
  }
};

struct CountedClose {
  std::atomic<std::uint64_t> value{0};
  unsigned stores = 0;

  void store(std::uint64_t process_id) {
    ++stores;
    value.store(process_id);
  }
};

// A minimal published holder makes the decision's consequence observable:
// cleanup must wait for already-admitted work. Native pool/reaper behavior is
// deliberately outside this CPU-only gate regression.
struct PendingWork {
  Event drain_entered;
  Event work_finished;
  std::atomic<bool> drained{false};

  void shutdown() {
    drain_entered.signal();
    work_finished.wait("admitted work did not finish");
    drained.store(true);
  }
};

void test_unowned() {
  std::atomic<std::uint64_t> owner{0};
  std::atomic<std::uint64_t> closed{0};
  require(!xtbloom::torch_detail::close_request_pool_admission(owner, closed, kProcessId),
          "an unused pool must not require a holder lookup");
  require(owner.load() == 0, "cleanup must not claim or initialize an unused pool");
  require(closed.load() == kProcessId, "cleanup must close an unused pool's admission");
}

void test_foreign_owner() {
  std::atomic<std::uint64_t> owner{kForeignProcessId};
  CountedClose closed;
  closed.value.store(kForeignProcessId);
  require(!xtbloom::torch_detail::close_request_pool_admission(owner, closed, kProcessId),
          "an inherited foreign pool must not permit a holder lookup");
  require(closed.stores == 0, "the foreign-owner check must precede closing admission");
  require(closed.value.load() == kForeignProcessId,
          "foreign cleanup must leave the inherited gate untouched");
}

void test_close_before_publication() {
  std::atomic<std::uint64_t> owner{kProcessId};
  std::atomic<std::uint64_t> closed{0};
  std::shared_ptr<PendingWork> published;
  require(xtbloom::torch_detail::close_request_pool_admission(owner, closed, kProcessId),
          "a claimed local owner must permit checking for a published holder");
  require(std::atomic_load(&published) == nullptr,
          "closing admission must not initialize an unpublished pool");
  std::atomic_store(&published, std::make_shared<PendingWork>());
  require(closed.load() == kProcessId,
          "a holder published after cleanup must observe closed admission");
}

void test_admitted_before_close() {
  std::atomic<std::uint64_t> owner{kProcessId};
  std::atomic<std::uint64_t> closed{0};
  auto work = std::make_shared<PendingWork>();
  std::shared_ptr<PendingWork> published = work;
  require(xtbloom::torch_detail::close_request_pool_admission(owner, closed, kProcessId),
          "an existing local owner with admitted work must be drained");
  require(closed.load() == kProcessId, "admission must close before the holder lookup");
  work->work_finished.signal();
  std::atomic_load(&published)->shutdown();
  require(work->drained.load(), "already-admitted work must be drained");
}

void test_first_admission_races_close() {
  std::atomic<std::uint64_t> owner{0};
  PausedClose closed;
  std::shared_ptr<PendingWork> published;
  auto work = std::make_shared<PendingWork>();
  Event decision_ready;
  Event cleanup_finished;
  std::atomic<bool> should_drain{false};
  std::atomic<bool> returned{false};

  std::thread cleanup([&] {
    // This is the same gate decision used by xtbloom_torch_shutdown, not a
    // second implementation of its owner-load/close/owner-reload sequence.
    const bool inspect_holder =
        xtbloom::torch_detail::close_request_pool_admission(owner, closed, kProcessId);
    should_drain.store(inspect_holder);
    decision_ready.signal();
    if (inspect_holder) {
      if (auto holder = std::atomic_load(&published)) holder->shutdown();
    }
    returned.store(true);
    cleanup_finished.signal();
  });

  // Reaching the store proves cleanup has already observed owner == 0. Keep
  // it paused while first use claims ownership, publishes and admits work.
  closed.before_store.wait("cleanup did not reach the pre-close barrier");
  require(owner.load() == 0, "the regression must begin with an unowned pool");
  std::uint64_t unclaimed = 0;
  require(owner.compare_exchange_strong(unclaimed, kProcessId),
          "first use must claim the initially unowned pool");
  std::atomic_store(&published, work);
  require(closed.value.load() == 0, "first work must be admitted before cleanup closes");

  closed.allow_store.signal();
  decision_ready.wait("cleanup did not finish its gate decision");
  require(should_drain.load(),
          "cleanup reused the pre-close unowned observation and skipped admitted work");
  require(closed.value.load() == kProcessId, "cleanup must close admission before draining");
  work->drain_entered.wait("cleanup did not drain the published holder");
  require(!returned.load(), "cleanup returned while admitted work was still pending");
  work->work_finished.signal();
  cleanup_finished.wait("cleanup did not finish after admitted work settled");
  cleanup.join();
  require(work->drained.load(), "cleanup returned without draining admitted work");
}

}  // namespace

int main() {
  test_unowned();
  test_foreign_owner();
  test_close_before_publication();
  test_admitted_before_close();
  test_first_admission_races_close();
  std::puts("PASS: 5 Torch pool shutdown gate tests");
  return EXIT_SUCCESS;
}
