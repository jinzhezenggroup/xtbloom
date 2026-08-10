#include <cuda_runtime_api.h>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "data/parameters/gfn2.hpp"
#include "runtime/backend.hpp"

namespace xtbloom::detail::cuda {

constexpr std::size_t kPairOverrideStorage = parameters::gfn2::kPairScaleOverrides.size() == 0u
                                                 ? 1u
                                                 : parameters::gfn2::kPairScaleOverrides.size();

/* Initializers make these external-linkage definitions visible to nvlink. */
extern __device__ __constant__ parameters::gfn2::GlobalParameters g_gfn2_global{};
extern __device__ __constant__
    parameters::gfn2::ElementParameters g_gfn2_elements[parameters::gfn2::kElementCount]{};
extern __device__ __constant__
    parameters::gfn2::ShellParameters g_gfn2_shells[parameters::gfn2::kShellCount]{};
extern __device__ __constant__
    parameters::gfn2::PairScaleOverride g_gfn2_pair_overrides[kPairOverrideStorage]{};

namespace {

constexpr std::size_t kConstantBytes = sizeof(g_gfn2_global) + sizeof(g_gfn2_elements) +
                                       sizeof(g_gfn2_shells) + sizeof(g_gfn2_pair_overrides);
constexpr std::size_t kPairOverrideBytes =
    parameters::gfn2::kPairScaleOverrides.size() * sizeof(parameters::gfn2::PairScaleOverride);
static_assert(kConstantBytes <= 64u * 1024u,
              "GFN2 parameter tables exceed CUDA constant-memory capacity");

struct UploadState {
  std::mutex mutex;
  std::vector<std::uint8_t> ready;
  std::vector<std::uint64_t> upload_count;
};

UploadState& upload_state() {
  static UploadState state;
  return state;
}

bool cuda_success(cudaError_t status, const char* operation, std::string& error) {
  if (status == cudaSuccess) {
    return true;
  }
  error = std::string(operation) + ": " + cudaGetErrorString(status);
  return false;
}

bool restore_device(int previous_device, int selected_device, std::string& error) {
  if (previous_device == selected_device) {
    return true;
  }
  return cuda_success(cudaSetDevice(previous_device), "failed to restore the CUDA device", error);
}

template <typename T, std::size_t Size>
bool table_bytes_match(const std::array<T, Size>& actual, const std::array<T, Size>& expected) {
  if constexpr (Size == 0u) {
    return true;
  } else {
    return std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0;
  }
}

}  // namespace

}  // namespace xtbloom::detail::cuda

namespace xtbloom::detail {

bool ensure_cuda_gfn2_parameters(std::int32_t device_id, std::string& error) {
  int device_count = 0;
  if (!cuda::cuda_success(cudaGetDeviceCount(&device_count),
                          "failed to query CUDA devices before parameter upload", error)) {
    return false;
  }
  if (device_id < 0 || device_id >= device_count) {
    error = "cannot upload GFN2 parameters to an invalid CUDA device";
    return false;
  }

  cuda::UploadState& state = cuda::upload_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  try {
    state.ready.resize(static_cast<std::size_t>(device_count), 0u);
    state.upload_count.resize(static_cast<std::size_t>(device_count), 0u);
  } catch (const std::bad_alloc&) {
    error = "failed to allocate CUDA parameter-upload bookkeeping";
    return false;
  }

  const std::size_t index = static_cast<std::size_t>(device_id);
  if (state.ready[index] != 0u) {
    error.clear();
    return true;
  }

  int previous_device = -1;
  if (!cuda::cuda_success(cudaGetDevice(&previous_device),
                          "failed to query the current CUDA device", error)) {
    return false;
  }
  if (previous_device != device_id &&
      !cuda::cuda_success(cudaSetDevice(device_id), "failed to select the CUDA parameter device",
                          error)) {
    return false;
  }

  bool copied =
      cuda::cuda_success(cudaMemcpyToSymbol(cuda::g_gfn2_global, &parameters::gfn2::kGlobal,
                                            sizeof(parameters::gfn2::kGlobal)),
                         "failed to upload global GFN2 parameters", error);
  copied = copied && cuda::cuda_success(cudaMemcpyToSymbol(cuda::g_gfn2_elements,
                                                           parameters::gfn2::kElements.data(),
                                                           sizeof(parameters::gfn2::kElements)),
                                        "failed to upload element GFN2 parameters", error);
  copied = copied && cuda::cuda_success(
                         cudaMemcpyToSymbol(cuda::g_gfn2_shells, parameters::gfn2::kShells.data(),
                                            sizeof(parameters::gfn2::kShells)),
                         "failed to upload shell GFN2 parameters", error);
  if (copied && !parameters::gfn2::kPairScaleOverrides.empty()) {
    copied = cuda::cuda_success(
        cudaMemcpyToSymbol(cuda::g_gfn2_pair_overrides,
                           parameters::gfn2::kPairScaleOverrides.data(), cuda::kPairOverrideBytes),
        "failed to upload pair-scale GFN2 parameters", error);
  }

  std::string restore_error;
  const bool restored = cuda::restore_device(previous_device, device_id, restore_error);
  if (!restored) {
    if (copied) {
      error = std::move(restore_error);
    } else {
      error += "; " + restore_error;
    }
    return false;
  }
  if (!copied) {
    return false;
  }

  state.ready[index] = 1u;
  ++state.upload_count[index];
  error.clear();
  return true;
}

std::uint64_t cuda_gfn2_parameter_upload_count(std::int32_t device_id) {
  cuda::UploadState& state = cuda::upload_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (device_id < 0 || static_cast<std::size_t>(device_id) >= state.upload_count.size()) {
    return 0u;
  }
  return state.upload_count[static_cast<std::size_t>(device_id)];
}

bool cuda_gfn2_parameters_match_host(std::int32_t device_id, std::string& error) {
  if (!ensure_cuda_gfn2_parameters(device_id, error)) {
    return false;
  }

  int previous_device = -1;
  if (!cuda::cuda_success(cudaGetDevice(&previous_device),
                          "failed to query the current CUDA device", error)) {
    return false;
  }
  if (previous_device != device_id &&
      !cuda::cuda_success(cudaSetDevice(device_id), "failed to select the CUDA parameter device",
                          error)) {
    return false;
  }

  parameters::gfn2::GlobalParameters global{};
  std::array<parameters::gfn2::ElementParameters, parameters::gfn2::kElementCount> elements{};
  std::array<parameters::gfn2::ShellParameters, parameters::gfn2::kShellCount> shells{};
  std::array<parameters::gfn2::PairScaleOverride, parameters::gfn2::kPairScaleOverrides.size()>
      pair_overrides{};
  bool copied =
      cuda::cuda_success(cudaMemcpyFromSymbol(&global, cuda::g_gfn2_global, sizeof(global)),
                         "failed to read back global GFN2 parameters", error);
  copied = copied && cuda::cuda_success(cudaMemcpyFromSymbol(elements.data(), cuda::g_gfn2_elements,
                                                             sizeof(elements)),
                                        "failed to read back element GFN2 parameters", error);
  copied = copied && cuda::cuda_success(
                         cudaMemcpyFromSymbol(shells.data(), cuda::g_gfn2_shells, sizeof(shells)),
                         "failed to read back shell GFN2 parameters", error);
  if (copied && !parameters::gfn2::kPairScaleOverrides.empty()) {
    copied =
        cuda::cuda_success(cudaMemcpyFromSymbol(pair_overrides.data(), cuda::g_gfn2_pair_overrides,
                                                cuda::kPairOverrideBytes),
                           "failed to read back pair-scale GFN2 parameters", error);
  }

  std::string restore_error;
  const bool restored = cuda::restore_device(previous_device, device_id, restore_error);
  if (!restored) {
    if (copied) {
      error = std::move(restore_error);
    } else {
      error += "; " + restore_error;
    }
    return false;
  }
  if (!copied) {
    return false;
  }
  if (std::memcmp(&global, &parameters::gfn2::kGlobal, sizeof(global)) != 0 ||
      std::memcmp(elements.data(), parameters::gfn2::kElements.data(), sizeof(elements)) != 0 ||
      std::memcmp(shells.data(), parameters::gfn2::kShells.data(), sizeof(shells)) != 0 ||
      !cuda::table_bytes_match(pair_overrides, parameters::gfn2::kPairScaleOverrides)) {
    error = "CUDA GFN2 parameter tables differ from the generated host tables";
    return false;
  }
  error.clear();
  return true;
}

}  // namespace xtbloom::detail
