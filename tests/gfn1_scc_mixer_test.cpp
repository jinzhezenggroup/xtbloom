#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>

#include "model/gfn1/scc_mixer.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

namespace {

class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t bytes) {
    constexpr std::size_t alignment = xtbloom::detail::gfn1::kSccMixerWorkspaceAlignment;
    bytes_ = (std::max<std::size_t>(bytes, 1u) + alignment - 1u) & ~(alignment - 1u);
    data_ = std::aligned_alloc(xtbloom::detail::gfn1::kSccMixerWorkspaceAlignment, bytes_);
    if (data_ != nullptr) {
      std::memset(data_, 0, bytes_);
    }
  }
  ~AlignedBuffer() { std::free(data_); }
  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0u;
};

int test_qsh_only_wrapper() {
  xtbloom::detail::gfn1::WavefunctionLayout layout;
  layout.batch_size = 2;
  layout.workspace_size_bytes = 128u;
  layout.qsh.offset_bytes = 64u;
  layout.qsh.size_bytes = 6u * sizeof(double);
  layout.qsh.element_count = 6;
  layout.qsh.system_offsets = {0, 2, 6};

  std::string error;
  xtbloom::detail::gfn1::SccMixerPlan plan;
  CHECK(xtbloom::detail::gfn1::make_scc_mixer_plan(layout, 3, 0.4, 1.0e-8, 2.0e-8, plan, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(plan.vector_offsets() == layout.qsh.system_offsets);
  CHECK(plan.total_vector_elements() == layout.qsh.element_count);
  CHECK(plan.matches_wavefunction_layout(layout));

  AlignedBuffer wavefunction_storage(layout.workspace_size_bytes);
  AlignedBuffer state_storage(plan.state_size_bytes());
  AlignedBuffer scratch_storage(plan.workspace_size_bytes());
  CHECK(wavefunction_storage.data() != nullptr && state_storage.data() != nullptr &&
        scratch_storage.data() != nullptr);

  xtbloom::detail::gfn1::WavefunctionView wavefunction;
  wavefunction.workspace_base = wavefunction_storage.data();
  wavefunction.workspace_size_bytes = wavefunction_storage.size();
  wavefunction.qsh = reinterpret_cast<double*>(
      static_cast<std::byte*>(wavefunction_storage.data()) + layout.qsh.offset_bytes);
  const double initial[6]{0.1, -0.2, 0.3, -0.4, 0.05, -0.06};
  std::copy(std::begin(initial), std::end(initial), wavefunction.qsh);

  xtbloom::detail::gfn1::SccMixerState state;
  xtbloom::detail::gfn1::SccMixerWorkspace scratch;
  CHECK(xtbloom::detail::gfn1::bind_scc_mixer_state(plan, state_storage.data(),
                                                    state_storage.size(), state,
                                                    error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn1::bind_scc_mixer_workspace(plan, scratch_storage.data(),
                                                        scratch_storage.size(), scratch,
                                                        error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn1::initialize_scc_mixer_state_cpu(plan, wavefunction, state, error) ==
        XTBLOOM_STATUS_SUCCESS);

  wavefunction.qsh[2] = initial[2] + 0.02;
  wavefunction.qsh[3] = initial[3] - 0.01;
  wavefunction.qsh[4] = initial[4] + 0.04;
  wavefunction.qsh[5] = initial[5] - 0.03;
  CHECK(xtbloom::detail::gfn1::mix_scc_broyden_system_cpu(plan, 1, wavefunction, state, scratch,
                                                          error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(std::abs(wavefunction.qsh[2] - (initial[2] + 0.4 * 0.02)) < 1.0e-15);
  CHECK(std::abs(wavefunction.qsh[5] - (initial[5] - 0.4 * 0.03)) < 1.0e-15);
  CHECK(state.iterations[0] == 0u && state.iterations[1] == 1u);

  AlignedBuffer staged_storage(plan.state_size_bytes());
  xtbloom::detail::gfn1::SccMixerState staged;
  CHECK(xtbloom::detail::gfn1::bind_scc_mixer_state(plan, staged_storage.data(),
                                                    staged_storage.size(), staged,
                                                    error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn1::prepare_scc_mixer_system_transaction_cpu(
            plan, 1, state, staged, error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(xtbloom::detail::gfn1::commit_scc_mixer_system_transaction_cpu(
            plan, 1, staged, state, error) == XTBLOOM_STATUS_SUCCESS);

  const double restart[4]{-0.1, 0.2, -0.3, 0.4};
  std::copy(std::begin(restart), std::end(restart), wavefunction.qsh + 2);
  CHECK(xtbloom::detail::gfn1::restart_scc_mixer_system_cpu(plan, 1, wavefunction, state, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(state.restart_counts[1] == 1u && state.iterations[1] == 0u);
  CHECK(std::equal(std::begin(restart), std::end(restart), state.current_inputs + 2));
  return 0;
}

}  // namespace

int main() { return test_qsh_only_wrapper(); }
