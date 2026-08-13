#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "model/common/scc_mixer.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

using xtbloom::detail::common::SccMixerPlan;
using xtbloom::detail::common::SccMixerState;
using xtbloom::detail::common::SccMixerVectorLayoutView;
using xtbloom::detail::common::SccMixerVectorView;
using xtbloom::detail::common::SccMixerWorkspace;

class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t bytes) {
    constexpr std::size_t alignment = xtbloom::detail::common::kSccMixerWorkspaceAlignment;
    bytes_ = (std::max<std::size_t>(bytes, 1u) + alignment - 1u) & ~(alignment - 1u);
    data_ = std::aligned_alloc(xtbloom::detail::common::kSccMixerWorkspaceAlignment, bytes_);
    if (data_ != nullptr) {
      std::memset(data_, 0, bytes_);
    }
  }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  ~AlignedBuffer() { std::free(data_); }

  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0u;
};

struct Fixture {
  static constexpr std::size_t kVectorBytes = 128u;
  static constexpr std::size_t kQshOffset = 64u;
  static constexpr std::size_t kQshElements = 6u;
  /* Keep byte-address arithmetic explicit: this is the first guard byte after
   * the double-valued qsh field, not an element offset on a typed pointer. */
  static constexpr std::size_t kQshEndOffset = kQshOffset + kQshElements * sizeof(double);
  static constexpr std::size_t kSuffixBytes = kVectorBytes - kQshEndOffset;

  std::int64_t qsh_offsets[3]{0, 2, 6};
  SccMixerVectorLayoutView layout;
  SccMixerPlan plan;
  AlignedBuffer vector_storage{kVectorBytes};
  std::unique_ptr<AlignedBuffer> state_storage;
  std::unique_ptr<AlignedBuffer> scratch_storage;
  SccMixerVectorView vector;
  SccMixerState state;
  SccMixerWorkspace scratch;

  explicit Fixture(std::string& error) {
    make_plan(error);
    if (!plan.sealed()) {
      return;
    }
    state_storage = std::make_unique<AlignedBuffer>(plan.state_size_bytes());
    scratch_storage = std::make_unique<AlignedBuffer>(plan.workspace_size_bytes());
    if (vector_storage.data() == nullptr || state_storage->data() == nullptr ||
        scratch_storage->data() == nullptr) {
      return;
    }
    vector.workspace_base = vector_storage.data();
    vector.workspace_size_bytes = vector_storage.size();
    vector.fields[0] =
        reinterpret_cast<double*>(static_cast<std::byte*>(vector_storage.data()) + kQshOffset);
    vector.field_count = 1u;
    if (xtbloom::detail::common::bind_scc_mixer_state(plan, state_storage->data(),
                                                      state_storage->size(), state,
                                                      error) != XTBLOOM_STATUS_SUCCESS ||
        xtbloom::detail::common::bind_scc_mixer_workspace(plan, scratch_storage->data(),
                                                          scratch_storage->size(), scratch,
                                                          error) != XTBLOOM_STATUS_SUCCESS) {
      plan = {};
    }
  }

 private:
  void make_plan(std::string& error) {
    layout.batch_size = 2;
    layout.workspace_size_bytes = kVectorBytes;
    layout.workspace_alignment = xtbloom::detail::common::kSccMixerWorkspaceAlignment;
    layout.field_count = 1u;
    layout.fields[0] = {kQshOffset, kQshElements * sizeof(double),
                        static_cast<std::int64_t>(kQshElements), qsh_offsets, 3u};
    if (xtbloom::detail::common::make_scc_mixer_plan(layout, 3, 0.4, 1.0e-8, 2.0e-8, plan, error) !=
        XTBLOOM_STATUS_SUCCESS) {
      return;
    }
  }
};

bool equal(const double* first, const double* second, std::size_t count) {
  return std::equal(first, first + count, second);
}

int test_qsh_only_ragged_mix_restart_and_transaction() {
  std::string error;
  Fixture fixture(error);
  CHECK(fixture.plan.sealed());
  CHECK(fixture.plan.vector_offsets() == std::vector<std::int64_t>({0, 2, 6}));
  CHECK(fixture.plan.total_vector_elements() == 6);
  CHECK(fixture.plan.maximum_vector_elements() == 4);
  CHECK(fixture.plan.matches_vector_layout(fixture.layout));

  /* Guard bytes prove a qsh-only plan never assumes adjacent D/Q fields. */
  std::memset(fixture.vector_storage.data(), 0x5a, Fixture::kQshOffset);
  std::memset(static_cast<std::byte*>(fixture.vector_storage.data()) + Fixture::kQshEndOffset, 0x5a,
              Fixture::kSuffixBytes);
  const double initial[6]{0.10, -0.20, 0.30, -0.40, 0.05, -0.06};
  std::copy(std::begin(initial), std::end(initial), fixture.vector.fields[0]);
  CHECK(xtbloom::detail::common::initialize_scc_mixer_state_cpu(
            fixture.plan, fixture.vector, fixture.state, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(equal(initial, fixture.state.current_inputs, 6u));
  CHECK(fixture.state.initialized[0] == 1u && fixture.state.initialized[1] == 1u);

  const std::vector<std::byte> prefix_before(
      static_cast<const std::byte*>(fixture.vector_storage.data()),
      static_cast<const std::byte*>(fixture.vector_storage.data()) + Fixture::kQshOffset);
  const std::vector<std::byte> suffix_before(
      static_cast<const std::byte*>(fixture.vector_storage.data()) + Fixture::kQshEndOffset,
      static_cast<const std::byte*>(fixture.vector_storage.data()) + Fixture::kVectorBytes);

  fixture.vector.fields[0][0] = initial[0] + 0.02;
  fixture.vector.fields[0][1] = initial[1] - 0.01;
  CHECK(xtbloom::detail::common::mix_scc_broyden_system_cpu(fixture.plan, 0, fixture.vector,
                                                            fixture.state, fixture.scratch,
                                                            error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::abs(fixture.vector.fields[0][0] - (initial[0] + 0.4 * 0.02)) < 1.0e-15);
  CHECK(std::abs(fixture.vector.fields[0][1] - (initial[1] - 0.4 * 0.01)) < 1.0e-15);
  CHECK(fixture.state.iterations[0] == 1u && fixture.state.iterations[1] == 0u);
  CHECK(std::equal(prefix_before.begin(), prefix_before.end(),
                   static_cast<const std::byte*>(fixture.vector_storage.data())));
  CHECK(std::equal(
      suffix_before.begin(), suffix_before.end(),
      static_cast<const std::byte*>(fixture.vector_storage.data()) + Fixture::kQshEndOffset));

  AlignedBuffer staged_storage(fixture.plan.state_size_bytes());
  SccMixerState staged;
  CHECK(xtbloom::detail::common::bind_scc_mixer_state(fixture.plan, staged_storage.data(),
                                                      staged_storage.size(), staged,
                                                      error) == XTBLOOM_STATUS_SUCCESS);
  const std::vector<std::byte> state_before(
      static_cast<const std::byte*>(fixture.state.workspace_base),
      static_cast<const std::byte*>(fixture.state.workspace_base) +
          fixture.state.workspace_size_bytes);
  CHECK(xtbloom::detail::common::commit_scc_mixer_system_transaction_cpu(
            fixture.plan, 1, staged, fixture.state, error) == XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(std::equal(state_before.begin(), state_before.end(),
                   static_cast<const std::byte*>(fixture.state.workspace_base)));
  std::memset(staged_storage.data(), 0x3c, staged_storage.size());
  CHECK(xtbloom::detail::common::prepare_scc_mixer_system_transaction_cpu(
            fixture.plan, 1, fixture.state, staged, error) == XTBLOOM_STATUS_SUCCESS);
  const double first_system_sentinel = staged.current_inputs[0];
  CHECK(xtbloom::detail::common::commit_scc_mixer_system_transaction_cpu(
            fixture.plan, 1, staged, fixture.state, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.state.current_inputs[0] != first_system_sentinel);

  const double restarted[4]{-0.1, 0.2, -0.3, 0.4};
  std::copy(std::begin(restarted), std::end(restarted), fixture.vector.fields[0] + 2);
  CHECK(xtbloom::detail::common::restart_scc_mixer_system_cpu(
            fixture.plan, 1, fixture.vector, fixture.state, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(fixture.state.restart_counts[1] == 1u);
  CHECK(fixture.state.iterations[1] == 0u);
  CHECK(equal(restarted, fixture.state.current_inputs + 2, 4u));
  CHECK(fixture.state.iterations[0] == 1u);

  const double current_before[4]{fixture.state.current_inputs[2], fixture.state.current_inputs[3],
                                 fixture.state.current_inputs[4], fixture.state.current_inputs[5]};
  fixture.vector.fields[0][2] = std::numeric_limits<double>::quiet_NaN();
  CHECK(xtbloom::detail::common::mix_scc_broyden_system_cpu(
            fixture.plan, 1, fixture.vector, fixture.state, fixture.scratch, error) ==
        XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(equal(current_before, fixture.state.current_inputs + 2, 4u));
  CHECK(fixture.state.iterations[1] == 0u);
  CHECK(fixture.state.system_statuses[1] == XTBLOOM_STATUS_INTERNAL_ERROR);
  CHECK(fixture.state.system_statuses[0] == XTBLOOM_STATUS_SUCCESS);
  return 0;
}

int test_layout_rejects_fake_or_overlapping_fields() {
  std::string error;
  std::int64_t offsets[2]{0, 2};
  SccMixerVectorLayoutView layout;
  layout.batch_size = 1;
  layout.workspace_size_bytes = 64u;
  layout.workspace_alignment = 64u;
  layout.field_count = 2u;
  layout.fields[0] = {0u, 2u * sizeof(double), 2, offsets, 2u};
  layout.fields[1] = layout.fields[0];
  SccMixerPlan plan;
  CHECK(xtbloom::detail::common::make_scc_mixer_plan(layout, 3, 0.4, 1.0e-8, 1.0e-8, plan, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(!plan.sealed());
  layout.field_count = 1u;
  offsets[1] = 0;
  CHECK(xtbloom::detail::common::make_scc_mixer_plan(layout, 3, 0.4, 1.0e-8, 1.0e-8, plan, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  return 0;
}

}  // namespace

int main() {
  if (const int status = test_qsh_only_ragged_mix_restart_and_transaction(); status != 0) {
    return status;
  }
  return test_layout_rejects_fake_or_overlapping_fields();
}
