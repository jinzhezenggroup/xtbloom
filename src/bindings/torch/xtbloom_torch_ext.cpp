// xtbloom's PyTorch integration as a compiled extension built against the
// LibTorch Stable ABI (torch 2.10+).
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.
//
// TORCH_TARGET_VERSION floors the extension at the torch 2.10 stable symbol
// set: building with newer headers only emits symbols that already existed in
// 2.10, so the same binary loads on any torch >= 2.10. Without it, the build
// tracks the compile-time torch version and a 2.13-built extension would
// reference 2.13-only symbols and fail to load on 2.10-2.12.
#define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 10) << 48))
//
// Why a compiled extension at all (the "vesin way"): the torch interface used
// to marshal every tensor through ctypes + DLPack from Python.  This target
// instead keeps the same zero-copy data plane native: it takes torch tensors
// directly, binds their data pointers to the public xtbloom C ABI descriptors,
// and writes the results into caller-owned output tensors. CPU and host-output
// calls retain the synchronous xtbloom_compute path. CUDA device outputs use the
// additive request ABI and a bounded persistent context/plan/request pool, so
// the operator returns stream-ordered tensors without exposing requests,
// futures, or raw stream parameters to Python users. The Python layer only
// keeps the thin autograd Function and the torch.compile graph-break shim.
//
// Why the LibTorch Stable ABI instead of the libtorch C++ API: the extension
// uses only the ABI-stable pieces (stable C shims, torch/csrc/stable,
// torch/headeronly), registers one boxed operator with STABLE_TORCH_LIBRARY,
// and links only the platform Torch CPU runtime for its aoti_torch_* symbols.
// A single platform binary therefore works across torch 2.10+ releases without
// the per-minor-version rebuilds that a torch/extension.h C++ extension would
// require. The build uses a manifest-pinned vendored header closure and a
// generated build-only stub with the real runtime identity (ELF SONAME,
// Mach-O install name, or PE DLL/import-library name), so it does not need a
// Torch install; the shipped extension resolves the real stable symbols from
// the Torch runtime that the Python layer loads first (see pytorch
// notes/libtorch_stable_abi).
//
// Failure semantics remain those of the public C ABI: validation before
// publication, per-system SCC failures as data-level per_system_status results
// with quiet-NaN floating slices, and call-level errors surfaced immediately
// on synchronous paths or at the first internal CUDA settlement point.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>
#include <xtbloom/xtbloom.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// ---------------------------------------------------------------------------
// Native library loading. XTBLOOM_LIBRARY is authoritative when set so the
// torch data plane cannot silently use a different native implementation from
// the rest of the Python package. Otherwise libxtbloom is found next to this
// extension (both live in the wheel's xtbloom/lib directory), with a plain
// system-name fallback. It is dlopen'ed rather than linked so the extension
// never hard-codes a xtbloom SONAME.
// ---------------------------------------------------------------------------

struct XTBloomApi {
  int32_t (*context_options_init)(xtbloom_context_options_t*, size_t);
  int32_t (*batch_init)(xtbloom_batch_t*, size_t);
  int32_t (*compute_options_init)(xtbloom_compute_options_t*, size_t);
  int32_t (*batch_result_init)(xtbloom_batch_result_t*, size_t);
  int32_t (*context_create)(const xtbloom_context_options_t*, xtbloom_context_t**);
  void (*context_destroy)(xtbloom_context_t*);
  int32_t (*context_get_backend)(const xtbloom_context_t*);
  int32_t (*context_get_device_id)(const xtbloom_context_t*);
  int32_t (*compute)(xtbloom_context_t*, const xtbloom_batch_t*, const xtbloom_compute_options_t*,
                     xtbloom_batch_result_t*);
  int32_t (*plan_create)(xtbloom_context_t*, const xtbloom_batch_t*,
                         const xtbloom_compute_options_t*, xtbloom_plan_t**);
  void (*plan_destroy)(xtbloom_plan_t*);
  int32_t (*request_info_init)(xtbloom_request_info_t*, size_t);
  int32_t (*request_create)(xtbloom_context_t*, xtbloom_request_t**);
  int32_t (*compute_enqueue)(xtbloom_context_t*, const xtbloom_batch_t*,
                             const xtbloom_compute_options_t*, const xtbloom_batch_result_t*,
                             xtbloom_request_t*);
  int32_t (*plan_compute_enqueue)(xtbloom_plan_t*, const xtbloom_batch_t*,
                                  const xtbloom_compute_options_t*, const xtbloom_batch_result_t*,
                                  xtbloom_request_t*);
  int32_t (*request_query)(xtbloom_request_t*, xtbloom_request_info_t*);
  int32_t (*request_wait)(xtbloom_request_t*, xtbloom_request_info_t*);
  const char* (*request_get_error)(const xtbloom_request_t*);
  void (*request_destroy)(xtbloom_request_t*);
  const char* (*get_last_error)(void);
  const char* (*status_string)(int32_t);
  bool request_api_available = false;
  bool request_api_incomplete = false;
  bool compute_options_v3_available = false;
};

// dladdr anchor: must keep default visibility so dladdr() can resolve this
// object's own path even inside a hidden-visibility build.
extern "C" void xtbloom_torch_ext_anchor() {}

#if defined(_WIN32)
std::string wide_to_utf8(const std::wstring& value) {
  if (value.empty()) return {};
  int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                                  static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
  if (bytes <= 0) return {};
  std::string result(static_cast<size_t>(bytes), '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), result.data(), bytes, nullptr,
                          nullptr) != bytes) {
    return {};
  }
  return result;
}

std::wstring utf8_to_wide(const std::string& value) {
  if (value.empty()) return {};
  int units = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                  static_cast<int>(value.size()), nullptr, 0);
  if (units <= 0) return {};
  std::wstring result(static_cast<size_t>(units), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), units,
                          result.data(), units) != units) {
    return {};
  }
  return result;
}

std::string own_directory() {
  HMODULE module = nullptr;
  if (!GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<const char*>(&xtbloom_torch_ext_anchor), &module)) {
    return {};
  }
  std::vector<wchar_t> buffer(512);
  for (;;) {
    SetLastError(ERROR_SUCCESS);
    DWORD length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0) return {};
    if (length < buffer.size()) {
      std::wstring path(buffer.data(), length);
      size_t separator = path.find_last_of(L"\\/");
      return separator == std::wstring::npos ? std::string()
                                             : wide_to_utf8(path.substr(0, separator));
    }
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || buffer.size() >= 32768) return {};
    buffer.resize(buffer.size() * 2);
  }
}

void* load_shared(const std::string& path) {
  std::wstring wide = utf8_to_wide(path);
  return wide.empty() ? nullptr : static_cast<void*>(LoadLibraryW(wide.c_str()));
}
void* resolve_symbol(void* handle, const char* symbol) {
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), symbol));
}
void* open_candidate(const std::string& directory, const std::string& name) {
  if (directory.empty()) {
    return nullptr;
  }
  return load_shared(directory + "\\" + name);
}
#else
std::string own_directory() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&xtbloom_torch_ext_anchor), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  std::string path(info.dli_fname);
  size_t separator = path.find_last_of('/');
  return separator == std::string::npos ? std::string() : path.substr(0, separator);
}

void* load_shared(const std::string& path) { return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL); }
void* resolve_symbol(void* handle, const char* symbol) { return dlsym(handle, symbol); }
void* open_candidate(const std::string& directory, const std::string& name) {
  if (directory.empty()) {
    return nullptr;
  }
  std::string path = directory + "/" + name;
  struct stat buffer;
  if (stat(path.c_str(), &buffer) != 0) {
    return nullptr;
  }
  return load_shared(path);
}
#endif

std::string default_search_name() {
#if defined(_WIN32)
  return "xtbloom.dll";
#elif defined(__APPLE__)
  return "libxtbloom.dylib";
#else
  return "libxtbloom.so";
#endif
}

const XTBloomApi& xtbloom_api() {
  // The API table is initialized once per process.  xtbloom itself is
  // thread-safe for separate per-call contexts, and reads of the immutable
  // table after the once-flag needs no further synchronization.
  static const XTBloomApi* table = []() -> const XTBloomApi* {
    void* handle = nullptr;
#if defined(_WIN32)
    const wchar_t* override_path = _wgetenv(L"XTBLOOM_LIBRARY");
    const bool has_override = override_path != nullptr && override_path[0] != L'\0';
    if (has_override) {
      handle = static_cast<void*>(LoadLibraryW(override_path));
    }
#else
    const char* override_path = std::getenv("XTBLOOM_LIBRARY");
    const bool has_override = override_path != nullptr && override_path[0] != '\0';
    if (has_override) {
      handle = load_shared(override_path);
    }
#endif
    if (!has_override) {
      std::string directory = own_directory();
      for (const char* name :
           {"libxtbloom.so", "libxtbloom.so.0", "libxtbloom.dylib", "xtbloom.dll"}) {
        handle = open_candidate(directory, name);
        if (handle != nullptr) {
          break;
        }
      }
      if (handle == nullptr) {
        // System-lookup fallback matching ctypes' find_library("xtbloom").
        handle = load_shared(default_search_name());
      }
    }
    if (handle == nullptr) {
      return nullptr;
    }
    auto api = new XTBloomApi();
    api->context_options_init = reinterpret_cast<int32_t (*)(xtbloom_context_options_t*, size_t)>(
        resolve_symbol(handle, "xtbloom_context_options_init"));
    api->batch_init = reinterpret_cast<int32_t (*)(xtbloom_batch_t*, size_t)>(
        resolve_symbol(handle, "xtbloom_batch_init"));
    api->compute_options_init = reinterpret_cast<int32_t (*)(xtbloom_compute_options_t*, size_t)>(
        resolve_symbol(handle, "xtbloom_compute_options_init"));
    api->batch_result_init = reinterpret_cast<int32_t (*)(xtbloom_batch_result_t*, size_t)>(
        resolve_symbol(handle, "xtbloom_batch_result_init"));
    api->context_create =
        reinterpret_cast<int32_t (*)(const xtbloom_context_options_t*, xtbloom_context_t**)>(
            resolve_symbol(handle, "xtbloom_context_create"));
    api->context_destroy = reinterpret_cast<void (*)(xtbloom_context_t*)>(
        resolve_symbol(handle, "xtbloom_context_destroy"));
    api->context_get_backend = reinterpret_cast<int32_t (*)(const xtbloom_context_t*)>(
        resolve_symbol(handle, "xtbloom_context_get_backend"));
    api->context_get_device_id = reinterpret_cast<int32_t (*)(const xtbloom_context_t*)>(
        resolve_symbol(handle, "xtbloom_context_get_device_id"));
    api->compute =
        reinterpret_cast<int32_t (*)(xtbloom_context_t*, const xtbloom_batch_t*,
                                     const xtbloom_compute_options_t*, xtbloom_batch_result_t*)>(
            resolve_symbol(handle, "xtbloom_compute"));
    api->plan_create =
        reinterpret_cast<int32_t (*)(xtbloom_context_t*, const xtbloom_batch_t*,
                                     const xtbloom_compute_options_t*, xtbloom_plan_t**)>(
            resolve_symbol(handle, "xtbloom_plan_create"));
    api->plan_destroy =
        reinterpret_cast<void (*)(xtbloom_plan_t*)>(resolve_symbol(handle, "xtbloom_plan_destroy"));
    api->request_info_init = reinterpret_cast<int32_t (*)(xtbloom_request_info_t*, size_t)>(
        resolve_symbol(handle, "xtbloom_request_info_init"));
    api->request_create = reinterpret_cast<int32_t (*)(xtbloom_context_t*, xtbloom_request_t**)>(
        resolve_symbol(handle, "xtbloom_request_create"));
    api->compute_enqueue = reinterpret_cast<int32_t (*)(
        xtbloom_context_t*, const xtbloom_batch_t*, const xtbloom_compute_options_t*,
        const xtbloom_batch_result_t*, xtbloom_request_t*)>(
        resolve_symbol(handle, "xtbloom_compute_enqueue"));
    api->plan_compute_enqueue = reinterpret_cast<int32_t (*)(
        xtbloom_plan_t*, const xtbloom_batch_t*, const xtbloom_compute_options_t*,
        const xtbloom_batch_result_t*, xtbloom_request_t*)>(
        resolve_symbol(handle, "xtbloom_plan_compute_enqueue"));
    api->request_query = reinterpret_cast<int32_t (*)(xtbloom_request_t*, xtbloom_request_info_t*)>(
        resolve_symbol(handle, "xtbloom_request_query"));
    api->request_wait = reinterpret_cast<int32_t (*)(xtbloom_request_t*, xtbloom_request_info_t*)>(
        resolve_symbol(handle, "xtbloom_request_wait"));
    api->request_get_error = reinterpret_cast<const char* (*)(const xtbloom_request_t*)>(
        resolve_symbol(handle, "xtbloom_request_get_error"));
    api->request_destroy = reinterpret_cast<void (*)(xtbloom_request_t*)>(
        resolve_symbol(handle, "xtbloom_request_destroy"));
    api->get_last_error =
        reinterpret_cast<const char* (*)(void)>(resolve_symbol(handle, "xtbloom_get_last_error"));
    api->status_string =
        reinterpret_cast<const char* (*)(int32_t)>(resolve_symbol(handle, "xtbloom_status_string"));
    const std::array request_symbols = {
        reinterpret_cast<void*>(api->request_info_init),
        reinterpret_cast<void*>(api->request_create),
        reinterpret_cast<void*>(api->compute_enqueue),
        reinterpret_cast<void*>(api->plan_compute_enqueue),
        reinterpret_cast<void*>(api->request_query),
        reinterpret_cast<void*>(api->request_wait),
        reinterpret_cast<void*>(api->request_get_error),
        reinterpret_cast<void*>(api->request_destroy),
    };
    size_t request_symbol_count = 0;
    for (void* symbol : request_symbols) {
      if (symbol != nullptr) ++request_symbol_count;
    }
    api->request_api_available = request_symbol_count == request_symbols.size();
    api->request_api_incomplete = request_symbol_count != 0 && !api->request_api_available;
    if (api->request_api_available &&
        (api->plan_create == nullptr || api->plan_destroy == nullptr)) {
      api->request_api_available = false;
      api->request_api_incomplete = true;
    }

    if (api->context_options_init == nullptr || api->batch_init == nullptr ||
        api->compute_options_init == nullptr || api->batch_result_init == nullptr ||
        api->context_create == nullptr || api->context_destroy == nullptr ||
        api->context_get_backend == nullptr || api->context_get_device_id == nullptr ||
        api->compute == nullptr || api->get_last_error == nullptr ||
        api->status_string == nullptr) {
      return nullptr;
    }
    xtbloom_compute_options_t probe;
    std::memset(&probe, 0xA5, sizeof(probe));
    if (api->compute_options_init(&probe, sizeof(probe)) != XTBLOOM_STATUS_SUCCESS) {
      return nullptr;
    }
    api->compute_options_v3_available =
        probe.scc_mixer == XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN && probe.scc_mixer_history == 8 &&
        probe.scc_mixer_damping == 0.4 && probe.determinism == XTBLOOM_DETERMINISM_DEFAULT &&
        probe.reserved_v3 == 0u;
    return api;
  }();
  if (table == nullptr) {
    STD_TORCH_CHECK(false,
                    "xtbloom_torch: cannot load libxtbloom; set XTBLOOM_LIBRARY or "
                    "install xtbloom with its native library bundled against the "
                    "torch extension");
  }
  STD_TORCH_CHECK(!table->request_api_incomplete,
                  "xtbloom_torch: the loaded libxtbloom exports only part of the "
                  "additive request ABI; install a coherent library build or "
                  "select a fully synchronous older library");
  return *table;
}

// ---------------------------------------------------------------------------
// Tensor contract helpers
// ---------------------------------------------------------------------------

xtbloom_memory_space_t memory_space_of(const Tensor& tensor) {
  return tensor.is_cuda() ? XTBLOOM_MEMORY_CUDA_DEVICE : XTBLOOM_MEMORY_HOST;
}

void require_contiguous_1d(const Tensor& tensor, ScalarType dtype, const char* name,
                           int64_t expected_size = -1) {
  STD_TORCH_CHECK(tensor.defined(), "xtbloom_torch: ", name, " is undefined");
  STD_TORCH_CHECK(tensor.is_contiguous(), "xtbloom_torch: ", name,
                  " must be C-contiguous (the op packs non-contiguous inputs)");
  STD_TORCH_CHECK(tensor.dim() == 1, "xtbloom_torch: ", name, " must be one-dimensional");
  STD_TORCH_CHECK(tensor.scalar_type() == dtype, "xtbloom_torch: ", name, " has the wrong dtype");
  if (expected_size >= 0) {
    STD_TORCH_CHECK(tensor.size(0) == expected_size, "xtbloom_torch: ", name,
                    " has the wrong length");
  }
}

// RAII guard that always destroys a created xtbloom context, including on throw.
class ContextGuard {
 public:
  explicit ContextGuard(xtbloom_context_t* handle) : handle_(handle) {}
  ~ContextGuard() {
    if (handle_ != nullptr) {
      xtbloom_api().context_destroy(handle_);
    }
  }
  ContextGuard(const ContextGuard&) = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;

 private:
  xtbloom_context_t* handle_;
};

void check_status(int32_t status, const char* what) {
  if (status == XTBLOOM_STATUS_SUCCESS) {
    return;
  }
  const char* detail = xtbloom_api().get_last_error();
  const char* label = xtbloom_api().status_string(status);
  STD_TORCH_CHECK(false, "xtbloom_torch: ", what, " failed with ",
                  label != nullptr ? label : "unknown status", " (", status, ")",
                  detail != nullptr && detail[0] != '\0' ? ": " : "",
                  detail != nullptr ? detail : "");
}

std::string status_diagnostic(const XTBloomApi& api, int32_t status, const char* what,
                              const char* detail) {
  const char* label = api.status_string(status);
  std::string message = what;
  message += " failed with ";
  message += label != nullptr ? label : "unknown status";
  message += " (" + std::to_string(status) + ")";
  if (detail != nullptr && detail[0] != '\0') {
    message += ": ";
    message += detail;
  }
  return message;
}

std::uint64_t double_bits(double value) noexcept {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t hash_host_bytes(const void* data, size_t bytes) noexcept {
  constexpr std::uint64_t kOffset = 1469598103934665603ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  std::uint64_t hash = kOffset;
  const auto* cursor = static_cast<const unsigned char*>(data);
  for (size_t index = 0; index < bytes; ++index) {
    hash ^= cursor[index];
    hash *= kPrime;
  }
  return hash;
}

struct BufferIdentity {
  std::uintptr_t device_pointer = 0;
  std::uint64_t host_hash = 0;
  std::int64_t tensor_version = 0;
  size_t bytes = 0;
  xtbloom_memory_space_t memory_space = XTBLOOM_MEMORY_HOST;

  bool operator==(const BufferIdentity& other) const noexcept {
    return device_pointer == other.device_pointer && host_hash == other.host_hash &&
           tensor_version == other.tensor_version && bytes == other.bytes &&
           memory_space == other.memory_space;
  }
};

BufferIdentity topology_identity(const Tensor& tensor, std::int64_t tensor_version) {
  BufferIdentity identity;
  identity.tensor_version = tensor_version;
  identity.bytes = static_cast<size_t>(tensor.numel()) * tensor.element_size();
  identity.memory_space = memory_space_of(tensor);
  if (tensor.is_cuda()) {
    identity.device_pointer = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
  } else if (identity.bytes != 0u) {
    identity.host_hash = hash_host_bytes(tensor.data_ptr(), identity.bytes);
  }
  return identity;
}

struct ContextKey {
  int64_t backend = XTBLOOM_BACKEND_AUTO;
  int64_t device = -1;
  int64_t cpu_threads = 1;
  int64_t stream = 0;

  bool operator==(const ContextKey& other) const noexcept {
    return backend == other.backend && device == other.device && cpu_threads == other.cpu_threads &&
           stream == other.stream;
  }
};

struct StreamKey {
  int64_t device = -1;
  int64_t stream = 0;

  bool operator==(const StreamKey& other) const noexcept {
    return device == other.device && stream == other.stream;
  }
};

StreamKey stream_key_of(const ContextKey& key) noexcept { return {key.device, key.stream}; }

struct PlanKey {
  int64_t nsystems = 0;
  int64_t natoms = 0;
  int64_t model = XTBLOOM_MODEL_GFN2_XTB;
  std::array<BufferIdentity, 5> topology{};
  int64_t max_scc_iterations = 0;
  std::uint64_t charge_tolerance = 0;
  std::uint64_t energy_tolerance = 0;
  std::uint64_t electronic_temperature = 0;
  int64_t scc_mixer = XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN;
  int64_t scc_mixer_history = 8;
  std::uint64_t scc_mixer_damping = 0;
  int64_t determinism = XTBLOOM_DETERMINISM_DEFAULT;

  bool operator==(const PlanKey& other) const noexcept {
    return nsystems == other.nsystems && natoms == other.natoms && model == other.model &&
           topology == other.topology && max_scc_iterations == other.max_scc_iterations &&
           charge_tolerance == other.charge_tolerance && energy_tolerance == other.energy_tolerance &&
           electronic_temperature == other.electronic_temperature && scc_mixer == other.scc_mixer &&
           scc_mixer_history == other.scc_mixer_history &&
           scc_mixer_damping == other.scc_mixer_damping && determinism == other.determinism;
  }
};

enum class SlotState : std::uint8_t { kIdle, kReserved, kPending, kBroken };

struct PlanGroup;

// The first eight tensors directly back public descriptors/results. The final
// five own auxiliary DLPack imports whose strided data may have been copied
// into those compact descriptor tensors on the caller's CUDA stream.
using RetainedTensors = std::array<std::optional<Tensor>, 13>;

std::uint64_t current_process_id() noexcept {
#if defined(_WIN32)
  return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
  return static_cast<std::uint64_t>(getpid());
#endif
}

struct DeferredError {
  StreamKey stream{};
  std::uint64_t submission_id = 0;
  std::string message;
  /* Stream-level reporting and forward-token settlement are independent.
   * A later call may report this failure, but backward must still find the
   * originating submission and refuse to consume its invalid force tensor. */
  bool stream_reported = false;
};

struct ContextState {
  const XTBloomApi* api = nullptr;
  ContextKey key{};
  xtbloom_context_t* context = nullptr;
  bool stopping = false;
  size_t exact_waiters = 0;
  std::uint64_t last_used = 0;
  std::vector<std::shared_ptr<PlanGroup>> groups;
  /* Native enqueue order is CUDA stream order. Serialize the short admission
   * boundary so monotonically assigned submission ids describe that same
   * order even when Python threads call one stream concurrently. */
  std::mutex admission_mutex;
  std::thread reaper;

  ~ContextState() {
    groups.clear();
    if (context != nullptr) api->context_destroy(context);
  }
};

struct PlanSlot {
  const XTBloomApi* api = nullptr;
  ContextState* context = nullptr;
  PlanGroup* group = nullptr;
  xtbloom_plan_t* plan = nullptr;
  xtbloom_request_t* request = nullptr;
  SlotState state = SlotState::kIdle;
  bool completion_in_progress = false;
  std::uint64_t submission_id = 0;
  RetainedTensors retained;
  std::vector<double> atomic_charges;
  std::vector<int32_t> scc_iterations;
  std::vector<std::uint8_t> scc_converged;
  std::vector<int32_t> per_system_status;

  PlanSlot(const XTBloomApi& selected_api, ContextState& selected_context,
           PlanGroup& selected_group, int64_t nsystems, int64_t natoms)
      : api(&selected_api),
        context(&selected_context),
        group(&selected_group),
        atomic_charges(static_cast<size_t>(natoms)),
        scc_iterations(static_cast<size_t>(nsystems)),
        scc_converged(static_cast<size_t>(nsystems)),
        per_system_status(static_cast<size_t>(nsystems)) {}

  ~PlanSlot() {
    if (request != nullptr) api->request_destroy(request);
    if (plan != nullptr) api->plan_destroy(plan);
  }

  void move_retained_to(RetainedTensors& destination) noexcept { destination.swap(retained); }
};

struct PlanGroup {
  ContextState* context = nullptr;
  PlanKey key{};
  size_t creating = 0;
  bool retired = false;
  std::uint64_t last_used = 0;
  std::vector<std::shared_ptr<PlanSlot>> slots;
};

struct CompletionOutcome {
  bool reusable = true;
  bool retire_group = false;
  std::string error;
};

class TorchRequestPool {
 public:
  explicit TorchRequestPool(const XTBloomApi& api)
      : api_(api), owner_process_id_(current_process_id()) {
    contexts_.reserve(kMaximumContexts);
    deferred_errors_.reserve(kMaximumDeferredErrors);
  }

  ~TorchRequestPool() { shutdown(); }

  void shutdown() noexcept {
    std::vector<std::shared_ptr<ContextState>> contexts;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) return;
      stopping_ = true;
      for (const auto& context : contexts_) context->stopping = true;
      contexts.swap(contexts_);
    }
    changed_.notify_all();
    for (const auto& context : contexts) {
      if (context->reaper.joinable()) context->reaper.join();
      std::unique_lock<std::mutex> lock(mutex_);
      changed_.wait(lock, [&] { return context->exact_waiters == 0u; });
    }
    report_deferred_errors("process shutdown");
    contexts.clear();
  }

  TorchRequestPool(const TorchRequestPool&) = delete;
  TorchRequestPool& operator=(const TorchRequestPool&) = delete;

  std::int64_t submit(const ContextKey& context_key, const PlanKey& plan_key,
                      const xtbloom_batch_t& batch, const xtbloom_compute_options_t& options,
                      const std::array<Tensor, 13>& tensors) {
    check_process();
    const StreamKey stream_key = stream_key_of(context_key);
    if (auto deferred = take_deferred_error(stream_key)) {
      STD_TORCH_CHECK(
          false, "xtbloom_torch: a preceding CUDA submission failed on this stream: ", *deferred);
    }
    std::shared_ptr<ContextState> context = context_for(context_key);
    std::unique_lock<std::mutex> admission(context->admission_mutex);

    for (int attempt = 0; attempt != 2; ++attempt) {
      std::shared_ptr<PlanGroup> group = group_for(*context, plan_key);
      std::shared_ptr<PlanSlot> slot = acquire_slot(group, batch, options);
      ReservationGuard reservation(*this, slot);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t index = 0; index < tensors.size(); ++index) {
          slot->retained[index].emplace(tensors[index]);
        }
      }

      xtbloom_batch_result_t result;
      int32_t status = api_.batch_result_init(&result, sizeof(result));
      check_status(status, "xtbloom_batch_result_init");
      auto bind_output = [](xtbloom_buffer_t* buffer, void* data, size_t bytes,
                            xtbloom_memory_space_t space) {
        buffer->data = data;
        buffer->size_bytes = bytes;
        buffer->memory_space = space;
        buffer->reserved = 0;
      };
      const Tensor& out_energies = tensors[6];
      const Tensor& out_forces = tensors[7];
      bind_output(&result.energies, out_energies.mutable_data_ptr(),
                  static_cast<size_t>(plan_key.nsystems) * sizeof(double),
                  XTBLOOM_MEMORY_CUDA_DEVICE);
      bind_output(&result.forces, out_forces.mutable_data_ptr(),
                  static_cast<size_t>(plan_key.natoms * 3) * sizeof(double),
                  XTBLOOM_MEMORY_CUDA_DEVICE);
      bind_output(&result.atomic_charges, slot->atomic_charges.data(),
                  slot->atomic_charges.size() * sizeof(double), XTBLOOM_MEMORY_HOST);
      bind_output(&result.scc_iterations, slot->scc_iterations.data(),
                  slot->scc_iterations.size() * sizeof(int32_t), XTBLOOM_MEMORY_HOST);
      bind_output(&result.scc_converged, slot->scc_converged.data(),
                  slot->scc_converged.size() * sizeof(std::uint8_t), XTBLOOM_MEMORY_HOST);
      bind_output(&result.per_system_status, slot->per_system_status.data(),
                  slot->per_system_status.size() * sizeof(int32_t), XTBLOOM_MEMORY_HOST);

      const std::uint64_t submission_id = reserve_submission_id(slot);
      status = api_.plan_compute_enqueue(slot->plan, &batch, &options, &result, slot->request);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        const char* detail = api_.get_last_error();
        const std::string detail_copy = detail != nullptr ? detail : "";
        const std::string message =
            status_diagnostic(api_, status, "xtbloom_plan_compute_enqueue", detail_copy.c_str());
        const bool topology_mismatch =
            attempt == 0 && is_fixed_topology_mismatch(status, detail_copy);
        if (topology_mismatch) {
          retire_group(group);
          continue;
        }
        STD_TORCH_CHECK(false, "xtbloom_torch: ", message);
      }

      // The native request is now accepted and owns the borrowed descriptors.
      // From this point an exception must settle the accepted request before
      // releasing tensors. The guard switches from rollback to settlement
      // until PENDING publication is complete.
      reservation.mark_accepted(submission_id);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        slot->state = SlotState::kPending;
        slot->completion_in_progress = false;
        group->last_used = next_use_locked();
        context->last_used = group->last_used;
      }
      reservation.disarm();
      changed_.notify_all();
      return static_cast<std::int64_t>(submission_id);
    }

    STD_TORCH_CHECK(false,
                    "xtbloom_torch: fixed CUDA plan topology changed again while rebuilding it");
  }

  void wait_for_submission(std::int64_t signed_submission_id) {
    if (signed_submission_id == 0) return;
    check_process();
    STD_TORCH_CHECK(signed_submission_id > 0,
                    "xtbloom_torch: internal CUDA submission token is invalid");
    const auto submission_id = static_cast<std::uint64_t>(signed_submission_id);

    for (;;) {
      std::shared_ptr<PlanSlot> slot;
      ContextState* waiter_context = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (auto error = take_submission_error_locked(submission_id)) {
          lock.unlock();
          STD_TORCH_CHECK(false, "xtbloom_torch: CUDA submission failed: ", *error);
        }
        slot = find_pending_submission_locked(submission_id);
        if (slot == nullptr) {
          STD_TORCH_CHECK(
              submission_id > completion_history_floor_,
              "xtbloom_torch: this CUDA submission is older than the bounded completion "
              "history retained for backward; its force tensor is not safe to consume");
          return;
        }
        if (slot->completion_in_progress) {
          changed_.wait(lock);
          continue;
        }
        slot->completion_in_progress = true;
        waiter_context = slot->context;
        ++waiter_context->exact_waiters;
      }

      CompletionOutcome outcome = wait_native(*slot);
      finalize_completion(slot, submission_id, outcome, true);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        --waiter_context->exact_waiters;
      }
      changed_.notify_all();
      if (!outcome.error.empty()) {
        STD_TORCH_CHECK(false, "xtbloom_torch: CUDA submission failed: ", outcome.error);
      }
      return;
    }
  }

 private:
  static constexpr size_t kMaximumContexts = 4;
  static constexpr size_t kMaximumPlanGroupsPerContext = 2;
  static constexpr size_t kMaximumSlotsPerPlan = 2;
  static constexpr size_t kMaximumDeferredErrors = 1024;

  class ReservationGuard {
   public:
    ReservationGuard(TorchRequestPool& pool, std::shared_ptr<PlanSlot> slot)
        : pool_(&pool), slot_(std::move(slot)) {}
    ~ReservationGuard() {
      if (!armed_) return;
      if (accepted_) {
        pool_->settle_accepted_reservation(slot_, submission_id_);
      } else {
        pool_->release_reserved(slot_);
      }
    }
    ReservationGuard(const ReservationGuard&) = delete;
    ReservationGuard& operator=(const ReservationGuard&) = delete;

    void mark_accepted(std::uint64_t submission_id) noexcept {
      accepted_ = true;
      submission_id_ = submission_id;
    }
    void disarm() noexcept { armed_ = false; }

   private:
    TorchRequestPool* pool_ = nullptr;
    std::shared_ptr<PlanSlot> slot_;
    bool armed_ = true;
    bool accepted_ = false;
    std::uint64_t submission_id_ = 0;
  };

  void check_process() const {
    STD_TORCH_CHECK(
        current_process_id() == owner_process_id_,
        "xtbloom_torch: CUDA use after fork is unsupported because the child inherited native "
        "contexts, mutexes, and worker threads; create the child process before its first "
        "xtbloom CUDA call or use a spawn-based multiprocessing start method");
  }

  std::uint64_t next_use_locked() noexcept { return ++use_clock_; }

  std::uint64_t reserve_submission_id(const std::shared_ptr<PlanSlot>& slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::uint64_t submission_id = next_submission_id_++;
    if (submission_id == 0u ||
        submission_id > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      /* More than 2^63 submissions in one process is not practically
       * reachable. Preserve zero as the synchronous-path sentinel. */
      submission_id = 1u;
      next_submission_id_ = 2u;
    }
    slot->submission_id = submission_id;
    return submission_id;
  }

  static bool is_fixed_topology_mismatch(int32_t status, std::string_view detail) noexcept {
    return status == XTBLOOM_STATUS_INVALID_ARGUMENT &&
           detail.find("does not match the fixed CUDA plan topology") != std::string_view::npos;
  }

  std::optional<std::string> take_deferred_error_locked(StreamKey stream) {
    for (size_t index = 0; index < lost_deferred_stream_count_; ++index) {
      if (!(lost_deferred_streams_[index] == stream)) continue;
      lost_deferred_streams_[index] = lost_deferred_streams_[--lost_deferred_stream_count_];
      return std::string(
          "the bounded deferred-error history lost an exact stream diagnostic; a prior CUDA "
          "submission failed and its outputs must not be consumed");
    }
    if (lost_deferred_stream_overflow_pending_) {
      /* More unique unsurfaced failing streams than the complete error-history
       * bound is pathological. Preserve safety with one conservative global
       * report only after the stream-keyed fixed storage itself is exhausted. */
      lost_deferred_stream_overflow_pending_ = false;
      return std::string(
          "the bounded deferred-error history exhausted its stream-keyed loss records; a prior "
          "CUDA submission failed and its outputs must not be consumed");
    }
    auto selected = deferred_errors_.end();
    for (auto iterator = deferred_errors_.begin(); iterator != deferred_errors_.end(); ++iterator) {
      if (iterator->stream_reported || !(iterator->stream == stream)) continue;
      if (selected == deferred_errors_.end() || iterator->submission_id < selected->submission_id) {
        selected = iterator;
      }
    }
    if (selected == deferred_errors_.end()) return std::nullopt;
    std::string message = selected->message;
    selected->stream_reported = true;
    return message;
  }

  void remember_lost_deferred_stream_locked(StreamKey stream) noexcept {
    for (size_t index = 0; index < lost_deferred_stream_count_; ++index) {
      if (lost_deferred_streams_[index] == stream) return;
    }
    if (lost_deferred_stream_count_ < lost_deferred_streams_.size()) {
      lost_deferred_streams_[lost_deferred_stream_count_++] = stream;
    } else {
      lost_deferred_stream_overflow_pending_ = true;
    }
  }

  std::optional<std::string> take_deferred_error(StreamKey stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    return take_deferred_error_locked(stream);
  }

  std::optional<std::string> take_submission_error_locked(std::uint64_t submission_id) {
    for (auto iterator = deferred_errors_.begin(); iterator != deferred_errors_.end(); ++iterator) {
      if (iterator->submission_id != submission_id) continue;
      std::string message = iterator->message;
      iterator->stream_reported = true;
      return message;
    }
    return std::nullopt;
  }

  std::shared_ptr<ContextState> create_context(const ContextKey& key) {
    auto candidate = std::make_shared<ContextState>();
    candidate->api = &api_;
    candidate->key = key;
    candidate->groups.reserve(kMaximumPlanGroupsPerContext);

    xtbloom_context_options_t context_options;
    check_status(api_.context_options_init(&context_options, sizeof(context_options)),
                 "xtbloom_context_options_init");
    context_options.backend = static_cast<xtbloom_backend_t>(key.backend);
    context_options.device_id = static_cast<int32_t>(key.device);
    context_options.cpu_threads = static_cast<int32_t>(key.cpu_threads);
    context_options.stream =
        key.stream != 0 ? reinterpret_cast<void*>(static_cast<uintptr_t>(key.stream)) : nullptr;

    check_status(api_.context_create(&context_options, &candidate->context),
                 "xtbloom_context_create");
    const int32_t resolved_backend = api_.context_get_backend(candidate->context);
    const int32_t resolved_device = api_.context_get_device_id(candidate->context);
    STD_TORCH_CHECK(resolved_backend == XTBLOOM_BACKEND_CUDA && resolved_device == key.device,
                    "xtbloom_torch: persistent CUDA context resolved to backend ", resolved_backend,
                    " device ", resolved_device, " instead of CUDA device ", key.device);
    return candidate;
  }

  bool group_idle_locked(const PlanGroup& group) const noexcept {
    if (group.creating != 0u) return false;
    for (const auto& slot : group.slots) {
      /* A waiter or recovery guard may retain a slot without retaining its raw
       * ContextState pointer. Keep the context alive until that external slot
       * owner has destroyed its request and plan in the required order. */
      if (slot.use_count() != 1u) return false;
      if (slot->state == SlotState::kReserved || slot->state == SlotState::kPending ||
          slot->completion_in_progress) {
        return false;
      }
    }
    return true;
  }

  bool context_idle_locked(const std::shared_ptr<ContextState>& context) const noexcept {
    if (context.use_count() != 1u || context->exact_waiters != 0u) return false;
    for (const auto& group : context->groups) {
      if (group.use_count() != 1u || !group_idle_locked(*group)) return false;
    }
    return true;
  }

  void report_deferred_errors(const char* reason) const noexcept {
    for (const DeferredError& deferred : deferred_errors_) {
      if (deferred.stream_reported) continue;
      std::fprintf(stderr, "xtbloom_torch: deferred CUDA error was never surfaced before %s: %s\n",
                   reason, deferred.message.c_str());
    }
    for (size_t index = 0; index < lost_deferred_stream_count_; ++index) {
      std::fprintf(stderr,
                   "xtbloom_torch: a lost deferred CUDA error for device %lld stream %lld was "
                   "never surfaced before %s\n",
                   static_cast<long long>(lost_deferred_streams_[index].device),
                   static_cast<long long>(lost_deferred_streams_[index].stream), reason);
    }
    if (lost_deferred_stream_overflow_pending_) {
      std::fprintf(stderr,
                   "xtbloom_torch: deferred CUDA error stream-loss records overflowed before %s\n",
                   reason);
    }
  }

  void shutdown_context(const std::shared_ptr<ContextState>& context) noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      context->stopping = true;
    }
    changed_.notify_all();
    if (context->reaper.joinable()) context->reaper.join();
  }

  std::shared_ptr<ContextState> context_for(const ContextKey& key) {
    for (;;) {
      std::shared_ptr<ContextState> victim;
      bool create = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        STD_TORCH_CHECK(!stopping_, "xtbloom_torch: CUDA request pool is shutting down");
        for (const auto& context : contexts_) {
          if (!context->stopping && context->key == key) {
            context->last_used = next_use_locked();
            return context;
          }
        }
        if (contexts_.size() + creating_contexts_ < kMaximumContexts) {
          ++creating_contexts_;
          create = true;
        } else {
          auto selected = contexts_.end();
          for (auto iterator = contexts_.begin(); iterator != contexts_.end(); ++iterator) {
            if (!context_idle_locked(*iterator)) continue;
            if (selected == contexts_.end() || (*iterator)->last_used < (*selected)->last_used) {
              selected = iterator;
            }
          }
          if (selected == contexts_.end()) {
            changed_.wait(lock);
            continue;
          }
          victim = *selected;
          victim->stopping = true;
          contexts_.erase(selected);
        }
      }

      if (victim != nullptr) {
        shutdown_context(victim);
        victim.reset();
        changed_.notify_all();
        continue;
      }

      if (create) {
        std::shared_ptr<ContextState> candidate;
        try {
          candidate = create_context(key);
          candidate->reaper = std::thread([this, state = candidate.get()] { reap(*state); });
        } catch (...) {
          {
            std::lock_guard<std::mutex> lock(mutex_);
            --creating_contexts_;
          }
          changed_.notify_all();
          throw;
        }

        std::shared_ptr<ContextState> selected;
        bool keep_candidate = false;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          --creating_contexts_;
          for (const auto& context : contexts_) {
            if (!context->stopping && context->key == key) {
              selected = context;
              break;
            }
          }
          if (selected == nullptr && !stopping_) {
            candidate->last_used = next_use_locked();
            contexts_.push_back(candidate);
            selected = candidate;
            keep_candidate = true;
          } else {
            candidate->stopping = true;
          }
        }
        changed_.notify_all();
        if (!keep_candidate) {
          shutdown_context(candidate);
          candidate.reset();
        }
        STD_TORCH_CHECK(selected != nullptr,
                        "xtbloom_torch: CUDA request pool stopped during context creation");
        return selected;
      }
    }
  }

  std::shared_ptr<PlanGroup> group_for(ContextState& context, const PlanKey& key) {
    for (;;) {
      std::shared_ptr<PlanGroup> victim;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        STD_TORCH_CHECK(!context.stopping, "xtbloom_torch: CUDA stream context is shutting down");
        for (const auto& group : context.groups) {
          if (!group->retired && group->key == key) {
            group->last_used = next_use_locked();
            context.last_used = group->last_used;
            return group;
          }
        }
        if (context.groups.size() < kMaximumPlanGroupsPerContext) {
          auto group = std::make_shared<PlanGroup>();
          group->context = &context;
          group->key = key;
          group->last_used = next_use_locked();
          group->slots.reserve(kMaximumSlotsPerPlan);
          context.last_used = group->last_used;
          context.groups.push_back(group);
          return group;
        }

        auto selected = context.groups.end();
        for (auto iterator = context.groups.begin(); iterator != context.groups.end(); ++iterator) {
          if (iterator->use_count() != 1u || !group_idle_locked(**iterator)) continue;
          if (selected == context.groups.end() || ((*iterator)->retired && !(*selected)->retired) ||
              ((*iterator)->retired == (*selected)->retired &&
               (*iterator)->last_used < (*selected)->last_used)) {
            selected = iterator;
          }
        }
        if (selected == context.groups.end()) {
          changed_.wait(lock);
          continue;
        }
        victim = *selected;
        context.groups.erase(selected);
      }
      victim.reset();
      changed_.notify_all();
    }
  }

  void retire_group(const std::shared_ptr<PlanGroup>& group) noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      group->retired = true;
      group->last_used = next_use_locked();
    }
    changed_.notify_all();
  }

  std::shared_ptr<PlanSlot> make_slot(PlanGroup& group, const xtbloom_batch_t& batch,
                                      const xtbloom_compute_options_t& options) {
    auto slot = std::make_shared<PlanSlot>(api_, *group.context, group, group.key.nsystems,
                                           group.key.natoms);
    int32_t status = api_.plan_create(group.context->context, &batch, &options, &slot->plan);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      const char* detail = api_.get_last_error();
      throw std::runtime_error(status_diagnostic(api_, status, "xtbloom_plan_create", detail));
    }
    status = api_.request_create(group.context->context, &slot->request);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      const char* detail = api_.get_last_error();
      throw std::runtime_error(status_diagnostic(api_, status, "xtbloom_request_create", detail));
    }
    return slot;
  }

  std::shared_ptr<PlanSlot> acquire_slot(const std::shared_ptr<PlanGroup>& group,
                                         const xtbloom_batch_t& batch,
                                         const xtbloom_compute_options_t& options) {
    for (;;) {
      std::shared_ptr<PlanSlot> wait_slot;
      std::uint64_t wait_id = 0;
      bool create = false;
      std::optional<std::string> deferred;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        deferred = take_deferred_error_locked(stream_key_of(group->context->key));
        if (!deferred) {
          /* Healthy slots remain persistent after a burst. Recreating the
           * second plan/request on every later two-deep burst would reintroduce
           * steady-state allocation. Only broken slots are pruned here; the
           * bounded group/context LRU owns normal idle reclamation. */
          for (auto iterator = group->slots.begin(); iterator != group->slots.end();) {
            if ((*iterator)->state == SlotState::kBroken && iterator->use_count() == 1u) {
              iterator = group->slots.erase(iterator);
            } else {
              ++iterator;
            }
          }
        }
        if (!deferred) {
          for (const auto& slot : group->slots) {
            if (slot->state == SlotState::kIdle) {
              slot->state = SlotState::kReserved;
              group->last_used = next_use_locked();
              return slot;
            }
          }
          if (group->slots.size() + group->creating < kMaximumSlotsPerPlan) {
            ++group->creating;
            create = true;
          } else {
            /* Backpressure settles the oldest submission on the whole CUDA
             * stream context, not merely this topology group. This preserves
             * FIFO error observation when several cached plans share a
             * stream. */
            wait_slot = next_pending_locked(*group->context, wait_id);
            if (wait_slot == nullptr) {
              changed_.wait(lock);
              continue;
            }
          }
        }
      }

      if (deferred) {
        STD_TORCH_CHECK(
            false, "xtbloom_torch: a preceding CUDA submission failed on this stream: ", *deferred);
      }
      if (create) {
        try {
          std::shared_ptr<PlanSlot> slot = make_slot(*group, batch, options);
          {
            std::lock_guard<std::mutex> lock(mutex_);
            --group->creating;
            slot->state = SlotState::kReserved;
            group->slots.push_back(slot);
          }
          changed_.notify_all();
          return slot;
        } catch (...) {
          {
            std::lock_guard<std::mutex> lock(mutex_);
            --group->creating;
          }
          changed_.notify_all();
          throw;
        }
      }

      CompletionOutcome outcome = wait_native(*wait_slot);
      finalize_completion(wait_slot, wait_id, outcome, true);
      if (!outcome.error.empty()) {
        STD_TORCH_CHECK(false,
                        "xtbloom_torch: a preceding CUDA submission failed: ", outcome.error);
      }
    }
  }

  void erase_broken_slot_locked(const std::shared_ptr<PlanSlot>& slot) noexcept {
    auto& slots = slot->group->slots;
    /* The group must remain the sole owner until destruction. Otherwise a
     * detached waiter would retain only raw context/group pointers. */
    if (slot->state != SlotState::kBroken || slot.use_count() != 1u) return;
    const auto iterator = std::find(slots.begin(), slots.end(), slot);
    if (iterator != slots.end()) slots.erase(iterator);
  }

  void release_reserved(const std::shared_ptr<PlanSlot>& slot) noexcept {
    RetainedTensors released;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      slot->move_retained_to(released);
      slot->state = SlotState::kIdle;
      slot->submission_id = 0u;
      slot->completion_in_progress = false;
      erase_broken_slot_locked(slot);
    }
    released = {};
    changed_.notify_all();
  }

  void recreate_request(PlanSlot& slot, CompletionOutcome& outcome) noexcept {
    if (slot.request != nullptr) api_.request_destroy(slot.request);
    slot.request = nullptr;
    const int32_t status = api_.request_create(slot.context->context, &slot.request);
    if (status != XTBLOOM_STATUS_SUCCESS) {
      outcome.reusable = false;
      try {
        const char* detail = api_.get_last_error();
        if (!outcome.error.empty()) outcome.error += "; ";
        outcome.error += status_diagnostic(api_, status, "xtbloom_request_create", detail);
      } catch (...) {
        /* Preserve the original access failure if formatting the secondary
         * request-recreation diagnostic itself cannot allocate. */
      }
    }
  }

  CompletionOutcome wait_native(PlanSlot& slot) noexcept {
    CompletionOutcome outcome;
    try {
      xtbloom_request_info_t info{};
      int32_t status = api_.request_info_init(&info, sizeof(info));
      if (status == XTBLOOM_STATUS_SUCCESS) status = api_.request_wait(slot.request, &info);
      if (status != XTBLOOM_STATUS_SUCCESS) {
        const char* detail = api_.get_last_error();
        outcome.error = status_diagnostic(api_, status, "xtbloom_request_wait", detail);
        recreate_request(slot, outcome);
        return outcome;
      }
      if (info.state != XTBLOOM_REQUEST_COMPLETE) {
        outcome.error = "blocking request wait returned without a COMPLETE state";
        recreate_request(slot, outcome);
        return outcome;
      }
      if (info.completion_status != XTBLOOM_STATUS_SUCCESS) {
        const char* detail = api_.request_get_error(slot.request);
        outcome.retire_group = is_fixed_topology_mismatch(
            info.completion_status,
            detail != nullptr ? std::string_view(detail) : std::string_view());
        outcome.error =
            status_diagnostic(api_, info.completion_status, "asynchronous xtbloom compute", detail);
      }
      return outcome;
    } catch (const std::exception& exception) {
      outcome.error = exception.what();
      recreate_request(slot, outcome);
      return outcome;
    } catch (...) {
      outcome.error = "unknown exception while settling a CUDA request";
      recreate_request(slot, outcome);
      return outcome;
    }
  }

  void settle_accepted_reservation(const std::shared_ptr<PlanSlot>& slot,
                                   std::uint64_t submission_id) noexcept {
    CompletionOutcome outcome = wait_native(*slot);
    RetainedTensors released;
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (slot->submission_id == submission_id && slot->state == SlotState::kReserved) {
        slot->move_retained_to(released);
        slot->completion_in_progress = false;
        slot->state = SlotState::kBroken;
        slot->submission_id = 0u;
        if (outcome.retire_group) slot->group->retired = true;
        erase_broken_slot_locked(slot);
      }
    } catch (...) {
      std::fprintf(stderr,
                   "xtbloom_torch: failed to release an accepted request after publication "
                   "bookkeeping failed\n");
    }
    released = {};
    if (!outcome.error.empty()) {
      std::fprintf(stderr,
                   "xtbloom_torch: accepted CUDA request also failed while recovering from "
                   "publication bookkeeping: %s\n",
                   outcome.error.c_str());
    }
    changed_.notify_all();
  }

  void finalize_completion(const std::shared_ptr<PlanSlot>& slot, std::uint64_t submission_id,
                           CompletionOutcome& outcome, bool surface_now) noexcept {
    RetainedTensors released;
    bool lost_deferred_error = false;
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (slot->submission_id != submission_id || slot->state != SlotState::kPending) return;
      slot->move_retained_to(released);
      slot->completion_in_progress = false;
      slot->state = outcome.reusable ? SlotState::kIdle : SlotState::kBroken;
      slot->submission_id = 0u;
      if (outcome.retire_group) {
        slot->group->retired = true;
        slot->group->last_used = next_use_locked();
      }
      if (!outcome.error.empty()) {
        if (deferred_errors_.size() >= kMaximumDeferredErrors) {
          auto oldest = std::min_element(deferred_errors_.begin(), deferred_errors_.end(),
                                         [](const DeferredError& left, const DeferredError& right) {
                                           return left.submission_id < right.submission_id;
                                         });
          if (oldest != deferred_errors_.end()) {
            completion_history_floor_ = std::max(completion_history_floor_, oldest->submission_id);
            if (!oldest->stream_reported) remember_lost_deferred_stream_locked(oldest->stream);
            deferred_errors_.erase(oldest);
          }
        }
        try {
          deferred_errors_.push_back(DeferredError{stream_key_of(slot->context->key), submission_id,
                                                   outcome.error, surface_now});
        } catch (...) {
          completion_history_floor_ = std::max(completion_history_floor_, submission_id);
          remember_lost_deferred_stream_locked(stream_key_of(slot->context->key));
          lost_deferred_error = true;
        }
      }
      erase_broken_slot_locked(slot);
    } catch (...) {
      lost_deferred_error = !outcome.error.empty();
    }
    released = {};
    if (lost_deferred_error) {
      std::fprintf(stderr, "xtbloom_torch: could not retain deferred CUDA error: %s\n",
                   outcome.error.c_str());
    }
    changed_.notify_all();
  }

  std::shared_ptr<PlanSlot> find_pending_submission_locked(
      std::uint64_t submission_id) const noexcept {
    for (const auto& context : contexts_) {
      for (const auto& group : context->groups) {
        for (const auto& slot : group->slots) {
          if (slot->state == SlotState::kPending && slot->submission_id == submission_id)
            return slot;
        }
      }
    }
    return nullptr;
  }

  std::shared_ptr<PlanSlot> next_pending_locked(ContextState& context,
                                                std::uint64_t& submission_id) {
    std::shared_ptr<PlanSlot> selected;
    for (const auto& group : context.groups) {
      for (const auto& slot : group->slots) {
        if (slot->state != SlotState::kPending) continue;
        if (selected == nullptr || slot->submission_id < selected->submission_id) selected = slot;
      }
    }
    if (selected != nullptr) {
      if (selected->completion_in_progress) return nullptr;
      selected->completion_in_progress = true;
      submission_id = selected->submission_id;
    }
    return selected;
  }

  bool has_pending_locked(const ContextState& context) const noexcept {
    for (const auto& group : context.groups) {
      for (const auto& slot : group->slots) {
        if (slot->state == SlotState::kPending) return true;
      }
    }
    return false;
  }

  void reap(ContextState& context) noexcept {
    for (;;) {
      std::shared_ptr<PlanSlot> slot;
      std::uint64_t submission_id = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return context.stopping || has_pending_locked(context); });
        slot = next_pending_locked(context, submission_id);
        if (slot == nullptr) {
          if (context.stopping && !has_pending_locked(context)) return;
          changed_.wait(lock);
          continue;
        }
      }
      CompletionOutcome outcome = wait_native(*slot);
      finalize_completion(slot, submission_id, outcome, false);
    }
  }

  const XTBloomApi& api_;
  const std::uint64_t owner_process_id_;
  std::mutex mutex_;
  std::condition_variable changed_;
  bool stopping_ = false;
  size_t creating_contexts_ = 0;
  std::uint64_t use_clock_ = 0;
  std::uint64_t next_submission_id_ = 1u;
  std::vector<std::shared_ptr<ContextState>> contexts_;
  std::vector<DeferredError> deferred_errors_;
  /* Tokens at or below the floor no longer have exact retained history. They
   * fail conservatively, while newer unrelated successful tokens remain
   * usable instead of inheriting a permanent global poison bit. */
  std::uint64_t completion_history_floor_ = 0u;
  /* Loss markers are fixed-capacity and keyed by the originating stream. A
   * later successful call on an unrelated stream must not report someone
   * else's dropped diagnostic merely because exact backward history is
   * bounded. */
  std::array<StreamKey, kMaximumDeferredErrors> lost_deferred_streams_{};
  size_t lost_deferred_stream_count_ = 0u;
  bool lost_deferred_stream_overflow_pending_ = false;
};

class TorchRequestPoolHolder {
 public:
  explicit TorchRequestPoolHolder(const XTBloomApi& api)
      : owner_process_id_(current_process_id()), pool_(new TorchRequestPool(api)) {}
  ~TorchRequestPoolHolder() {
    /* A forked child cannot safely destroy inherited mutex/thread/CUDA state.
     * Intentionally leak that copied state; process teardown will reclaim it. */
    if (current_process_id() == owner_process_id_) delete pool_;
  }

  TorchRequestPool& get() const noexcept { return *pool_; }

 private:
  std::uint64_t owner_process_id_ = 0;
  TorchRequestPool* pool_ = nullptr;
};

TorchRequestPool& torch_request_pool(const XTBloomApi& api) {
  static TorchRequestPoolHolder holder(api);
  return holder.get();
}

}  // namespace

// ---------------------------------------------------------------------------
// The registered operator.
//
// dtype/device rules mirror the historical Python op exactly:
//  - positions (natoms, 3) float64, required be C-contiguous (Python already
//    packs strided inputs before calling, this is defense in depth);
//  - atomic_numbers (natoms,) int32, atom_offsets (nsystems + 1,) int64,
//    molecular_charges (nsystems,) float64, unpaired_electrons (nsystems,)
//    int32, spin_channels (nsystems,) int32;
//  - host and CUDA descriptors may be mixed exactly as the public ABI permits;
//    every CUDA-resident tensor must use the context's resolved device;
//  - out_energies (nsystems,) float64 and out_forces (natoms, 3) float64 are
//    written in place and returned by the op. A third private integer lets the
//    Python autograd implementation settle exactly its own forward failure;
//    xtbloom_torch still exposes only the ordinary (energies, forces) pair.
//  - electronic_temperature is in kelvin, exactly like xtbloom.xtbloom_torch;
//    it is converted to the native k_B*T Hartree scale inside the op.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, std::int64_t> xtbloom_torch_forward(
    Tensor positions, Tensor atomic_numbers, Tensor atom_offsets, Tensor molecular_charges,
    Tensor unpaired_electrons, Tensor spin_channels, Tensor atomic_numbers_owner,
    Tensor atom_offsets_owner, Tensor molecular_charges_owner, Tensor unpaired_electrons_owner,
    Tensor spin_channels_owner, int64_t atomic_numbers_version, int64_t atom_offsets_version,
    int64_t molecular_charges_version, int64_t unpaired_electrons_version,
    int64_t spin_channels_version, Tensor out_energies, Tensor out_forces, int64_t model,
    int64_t backend, int64_t device_id, int64_t cpu_threads, int64_t stream,
    int64_t max_scc_iterations, double charge_tolerance, double energy_tolerance,
    double electronic_temperature, int64_t scc_mixer, int64_t scc_mixer_history,
    double scc_mixer_damping, int64_t determinism) {
  const XTBloomApi& api = xtbloom_api();
  // Fail fast when the torch extension was built against an incompatible
  // ABI level, before any tensor contract is assumed.
  STD_TORCH_CHECK(TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0,
                  "xtbloom_torch requires the torch stable ABI (torch >= 2.10)");
  // The dispatcher exposes int64 scalars. Validate before narrowing them into
  // the fixed-width public ABI so direct private-op callers cannot wrap a
  // hostile value into an otherwise valid tag or history length.
  STD_TORCH_CHECK(model == XTBLOOM_MODEL_GFN1_XTB || model == XTBLOOM_MODEL_GFN2_XTB,
                  "xtbloom_torch: model has an unknown value");
  STD_TORCH_CHECK(scc_mixer == XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN,
                  "xtbloom_torch: scc_mixer must select modified Broyden");
  STD_TORCH_CHECK(scc_mixer_history >= 1 && scc_mixer_history <= 64,
                  "xtbloom_torch: scc_mixer_history must be between 1 and 64");
  STD_TORCH_CHECK(
      std::isfinite(scc_mixer_damping) && scc_mixer_damping > 0.0 && scc_mixer_damping <= 1.0,
      "xtbloom_torch: scc_mixer_damping must be finite and in (0, 1]");
  STD_TORCH_CHECK(
      determinism == XTBLOOM_DETERMINISM_DEFAULT || determinism == XTBLOOM_DETERMINISM_REPRODUCIBLE,
      "xtbloom_torch: determinism has an unknown value");
  const bool default_v3_policy = scc_mixer == XTBLOOM_SCC_MIXER_MODIFIED_BROYDEN &&
                                 scc_mixer_history == 8 && scc_mixer_damping == 0.4 &&
                                 determinism == XTBLOOM_DETERMINISM_DEFAULT;
  STD_TORCH_CHECK(api.compute_options_v3_available || default_v3_policy,
                  "xtbloom_torch: the loaded xTBloom core does not support "
                  "compute-options ABI v3; nondefault SCC policy would be ignored");

  // --- validate shapes/dtypes (mirror of the Python preflight) --------------
  STD_TORCH_CHECK(positions.is_contiguous(), "xtbloom_torch: positions must be C-contiguous");
  STD_TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
                  "xtbloom_torch: positions must have shape (natoms, 3)");
  STD_TORCH_CHECK(positions.scalar_type() == ScalarType::Double,
                  "xtbloom_torch: positions must be float64");
  const int64_t natoms = positions.size(0);
  require_contiguous_1d(atomic_numbers, ScalarType::Int, "atomic_numbers", natoms);
  require_contiguous_1d(molecular_charges, ScalarType::Double, "molecular_charges");
  const int64_t nsystems = molecular_charges.size(0);
  STD_TORCH_CHECK(nsystems >= 1, "xtbloom_torch: a batch needs at least one system");
  require_contiguous_1d(atom_offsets, ScalarType::Long, "atom_offsets", nsystems + 1);
  require_contiguous_1d(unpaired_electrons, ScalarType::Int, "unpaired_electrons", nsystems);
  require_contiguous_1d(spin_channels, ScalarType::Int, "spin_channels", nsystems);
  require_contiguous_1d(out_energies, ScalarType::Double, "out_energies", nsystems);
  STD_TORCH_CHECK(out_forces.is_contiguous(), "xtbloom_torch: out_forces must be C-contiguous");
  STD_TORCH_CHECK(out_forces.dim() == 2 && out_forces.size(1) == 3,
                  "xtbloom_torch: out_forces must have shape (natoms, 3)");
  STD_TORCH_CHECK(out_forces.size(0) == natoms, "xtbloom_torch: out_forces atom count mismatch");
  STD_TORCH_CHECK(out_forces.scalar_type() == ScalarType::Double,
                  "xtbloom_torch: out_forces must be float64");

  // The public CUDA ABI accepts mixed host/device descriptors. Ignore host
  // tensors here, but require every CUDA tensor to use one device. Reject
  // other accelerator types before memory_space_of could mislabel them HOST.
  bool any_cuda = false;
  int64_t cuda_device_index = -1;
  for (const Tensor& tensor : {positions, atomic_numbers, atom_offsets, molecular_charges,
                               unpaired_electrons, spin_channels, out_energies, out_forces}) {
    STD_TORCH_CHECK(tensor.is_cpu() || tensor.is_cuda(),
                    "xtbloom_torch: only CPU and CUDA tensors are supported");
    if (!tensor.is_cuda()) {
      continue;
    }
    const int64_t tensor_device = tensor.get_device_index();
    if (!any_cuda) {
      any_cuda = true;
      cuda_device_index = tensor_device;
      continue;
    }
    STD_TORCH_CHECK(tensor_device == cuda_device_index,
                    "xtbloom_torch: all CUDA tensors must be on device ", cuda_device_index);
  }
  const bool use_request_path = api.request_api_available && backend != XTBLOOM_BACKEND_CPU &&
                                out_energies.is_cuda() && out_forces.is_cuda();

  // --- batch -----------------------------------------------------------------
  xtbloom_batch_t batch;
  check_status(api.batch_init(&batch, sizeof(batch)), "xtbloom_batch_init");
  batch.batch_size = nsystems;
  batch.total_atoms = natoms;
  batch.total_point_charges = 0;
  batch.total_charge_response_elements = 0;

  auto bind_input = [&](xtbloom_const_buffer_t* buffer, const Tensor& tensor) {
    buffer->data = tensor.data_ptr();
    buffer->size_bytes = static_cast<size_t>(tensor.numel()) * tensor.element_size();
    buffer->memory_space = memory_space_of(tensor);
    buffer->reserved = 0;
  };
  bind_input(&batch.atom_offsets, atom_offsets);
  bind_input(&batch.atomic_numbers, atomic_numbers);
  bind_input(&batch.positions, positions);
  bind_input(&batch.molecular_charges, molecular_charges);
  bind_input(&batch.unpaired_electrons, unpaired_electrons);
  bind_input(&batch.spin_channels, spin_channels);

  // --- compute options ---------------------------------------------------------
  xtbloom_compute_options_t options;
  check_status(api.compute_options_init(&options, sizeof(options)), "xtbloom_compute_options_init");
  options.model = static_cast<xtbloom_model_t>(model);
  options.flags = static_cast<uint32_t>(XTBLOOM_COMPUTE_ENERGY | XTBLOOM_COMPUTE_FORCES |
                                        XTBLOOM_COMPUTE_ATOMIC_CHARGES);
  options.max_scc_iterations = static_cast<int32_t>(max_scc_iterations);
  options.charge_tolerance = charge_tolerance;
  options.energy_tolerance = energy_tolerance;
  options.electronic_temperature = electronic_temperature * XTBLOOM_KELVIN_TO_HARTREE;
  options.scc_mixer = static_cast<xtbloom_scc_mixer_t>(scc_mixer);
  options.scc_mixer_history = static_cast<int32_t>(scc_mixer_history);
  options.scc_mixer_damping = scc_mixer_damping;
  options.determinism = static_cast<xtbloom_determinism_t>(determinism);

  if (use_request_path) {
    STD_TORCH_CHECK(any_cuda && cuda_device_index >= 0,
                    "xtbloom_torch: CUDA request execution requires a CUDA tensor device");
    STD_TORCH_CHECK(device_id < 0 || device_id == cuda_device_index,
                    "xtbloom_torch: CUDA tensor on device ", cuda_device_index,
                    " does not match requested context device ", device_id);
    // AUTO with CUDA outputs and explicit CUDA resolve to the same native CUDA
    // execution semantics. Canonicalizing the otherwise irrelevant backend and
    // CPU thread fields gives each (device, stream) exactly one completion
    // owner, so deferred failures cannot disappear across configuration-only
    // changes on that stream.
    const ContextKey context_key{XTBLOOM_BACKEND_CUDA, cuda_device_index, 1, stream};
    const PlanKey plan_key{
        nsystems,
        natoms,
        model,
        {topology_identity(atom_offsets, atom_offsets_version),
         topology_identity(atomic_numbers, atomic_numbers_version),
         topology_identity(molecular_charges, molecular_charges_version),
         topology_identity(unpaired_electrons, unpaired_electrons_version),
         topology_identity(spin_channels, spin_channels_version)},
        max_scc_iterations,
        double_bits(charge_tolerance),
        double_bits(energy_tolerance),
        double_bits(options.electronic_temperature),
        scc_mixer,
        scc_mixer_history,
        double_bits(scc_mixer_damping),
        determinism,
    };
    const std::array<Tensor, 13> retained = {
        positions,
        atomic_numbers,
        atom_offsets,
        molecular_charges,
        unpaired_electrons,
        spin_channels,
        out_energies,
        out_forces,
        atomic_numbers_owner,
        atom_offsets_owner,
        molecular_charges_owner,
        unpaired_electrons_owner,
        spin_channels_owner,
    };
    const std::int64_t submission_id =
        torch_request_pool(api).submit(context_key, plan_key, batch, options, retained);
    return {out_energies, out_forces, submission_id};
  }

  // --- synchronous context ---------------------------------------------------
  // CPU, host-output CUDA, and native libraries predating the additive request
  // ABI retain the established immediate-error path.
  xtbloom_context_options_t context_options;
  check_status(api.context_options_init(&context_options, sizeof(context_options)),
               "xtbloom_context_options_init");
  context_options.backend = static_cast<xtbloom_backend_t>(backend);
  context_options.device_id = static_cast<int32_t>(device_id);
  context_options.cpu_threads = static_cast<int32_t>(cpu_threads);
  context_options.stream =
      stream > 0 ? reinterpret_cast<void*>(static_cast<uintptr_t>(stream)) : nullptr;

  xtbloom_context_t* context = nullptr;
  int32_t context_status = api.context_create(&context_options, &context);
  if (context_status != XTBLOOM_STATUS_SUCCESS && backend == XTBLOOM_BACKEND_AUTO && !any_cuda &&
      device_id < 0 && context_options.stream != nullptr) {
    // AUTO with all-host tensors may legitimately resolve to CPU in a CPU-only
    // xtbloom build. Python cannot know that resolution before the context is
    // created, so it supplies Torch's current CUDA stream as a candidate. Retry
    // without that candidate to preserve AUTO's established CPU fallback.
    if (context != nullptr) {
      api.context_destroy(context);
      context = nullptr;
    }
    context_options.stream = nullptr;
    context_status = api.context_create(&context_options, &context);
  }
  check_status(context_status, "xtbloom_context_create");
  ContextGuard context_guard(context);

  // Mirror the Python layer's device-consistency gate: CUDA inputs must match
  // the device the context actually resolved.
  if (any_cuda) {
    int32_t resolved_backend = api.context_get_backend(context);
    STD_TORCH_CHECK(resolved_backend == XTBLOOM_BACKEND_CUDA,
                    "xtbloom_torch: CUDA tensors require the CUDA backend");
    int32_t resolved_device = api.context_get_device_id(context);
    STD_TORCH_CHECK(resolved_device == cuda_device_index, "xtbloom_torch: CUDA tensor on device ",
                    cuda_device_index, " does not match the context device ", resolved_device);
  }

  // --- result ------------------------------------------------------------------
  // energies/forces go straight into the caller's out tensors (zero copy).
  // Diagnostics and the (unreported) atomic charges use small host buffers
  // that the C ABI always requires; a CUDA compute still accepts host result
  // buffers, matching the historical per-call host diagnostics.
  std::vector<int32_t> scc_iterations(static_cast<size_t>(nsystems));
  std::vector<uint8_t> scc_converged(static_cast<size_t>(nsystems));
  std::vector<int32_t> per_system_status(static_cast<size_t>(nsystems));
  std::vector<double> atomic_charges(static_cast<size_t>(natoms));

  auto bind_output = [](xtbloom_buffer_t* buffer, void* data, size_t bytes,
                        xtbloom_memory_space_t space) {
    buffer->data = data;
    buffer->size_bytes = bytes;
    buffer->memory_space = space;
    buffer->reserved = 0;
  };

  xtbloom_batch_result_t result;
  check_status(api.batch_result_init(&result, sizeof(result)), "xtbloom_batch_result_init");
  bind_output(&result.energies, out_energies.mutable_data_ptr(),
              static_cast<size_t>(nsystems) * sizeof(double), memory_space_of(out_energies));
  bind_output(&result.forces, out_forces.mutable_data_ptr(),
              static_cast<size_t>(natoms * 3) * sizeof(double), memory_space_of(out_forces));
  bind_output(&result.atomic_charges, atomic_charges.data(), atomic_charges.size() * sizeof(double),
              XTBLOOM_MEMORY_HOST);
  bind_output(&result.scc_iterations, scc_iterations.data(),
              scc_iterations.size() * sizeof(int32_t), XTBLOOM_MEMORY_HOST);
  bind_output(&result.scc_converged, scc_converged.data(),
              scc_converged.size() * sizeof(uint8_t), XTBLOOM_MEMORY_HOST);
  bind_output(&result.per_system_status, per_system_status.data(),
              per_system_status.size() * sizeof(int32_t), XTBLOOM_MEMORY_HOST);

  check_status(api.compute(context, &batch, &options, &result), "xtbloom_compute");

  return {out_energies, out_forces, 0};
}

void xtbloom_torch_wait(std::int64_t submission_id) {
  if (submission_id == 0) return;
  torch_request_pool(xtbloom_api()).wait_for_submission(submission_id);
}

STABLE_TORCH_LIBRARY(xtbloom, m) {
  m.def(
      "_xtbloom_torch_forward(Tensor positions, Tensor atomic_numbers, Tensor atom_offsets, "
      "Tensor molecular_charges, Tensor unpaired_electrons, Tensor spin_channels, "
      "Tensor atomic_numbers_owner, Tensor atom_offsets_owner, Tensor molecular_charges_owner, "
      "Tensor unpaired_electrons_owner, Tensor spin_channels_owner, "
      "int atomic_numbers_version, int atom_offsets_version, int molecular_charges_version, "
      "int unpaired_electrons_version, int spin_channels_version, "
      "Tensor(a!) out_energies, Tensor(b!) out_forces, int model, int backend, int device_id, "
      "int cpu_threads, "
      "int stream, int max_scc_iterations, float charge_tolerance, float energy_tolerance, "
      "float electronic_temperature, int scc_mixer, int scc_mixer_history, "
      "float scc_mixer_damping, int determinism) -> (Tensor(a!), Tensor(b!), int)");
  m.def("_xtbloom_torch_wait(int submission_id) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(xtbloom, CompositeExplicitAutograd, m) {
  m.impl("_xtbloom_torch_forward", TORCH_BOX(&xtbloom_torch_forward));
  m.impl("_xtbloom_torch_wait", TORCH_BOX(&xtbloom_torch_wait));
}
