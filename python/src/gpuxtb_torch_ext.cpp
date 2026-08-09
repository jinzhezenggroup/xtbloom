// gpuxtb's PyTorch integration as a compiled extension built against the
// LibTorch Stable ABI (torch 2.10+).
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
// directly, binds their data pointers to the public gpuxtb C ABI descriptors,
// runs one synchronous gpuxtb_compute call, and writes the results into
// caller-owned output tensors. The Python layer only keeps the thin autograd
// Function and the torch.compile graph-break shim.
//
// Why the LibTorch Stable ABI instead of the libtorch C++ API: the extension
// uses only the ABI-stable pieces (stable C shims, torch/csrc/stable,
// torch/headeronly), registers one boxed operator with STABLE_TORCH_LIBRARY,
// and links nothing but libtorch_cpu.so for its aoti_torch_* symbols.  A
// single binary therefore works across torch 2.10+ releases without the
// per-minor-version rebuilds that a torch/extension.h C++ extension would
// require. The build uses a manifest-pinned vendored header closure and a
// generated libtorch_cpu.so SONAME stub, so it does not need a Torch install;
// the shipped extension resolves the real stable symbols from the Torch
// runtime that the Python layer loads first (see pytorch
// notes/libtorch_stable_abi).
//
// The public gpuxtb C ABI is host-synchronous; this op is a direct wrapper
// around gpuxtb_compute, so failure semantics are inherited unchanged (per
// descriptor docs): validation before publication, per-system SCC failures as
// data-level per_system_status results with quiet-NaN floating slices, and
// call-level errors surfaced as exceptions with gpuxtb_get_last_error text.

#include <gpuxtb/gpuxtb.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#include <sys/stat.h>
#endif

namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// ---------------------------------------------------------------------------
// Native library loading. GPUXTB_LIBRARY is authoritative when set so the
// torch data plane cannot silently use a different native implementation from
// the rest of the Python package. Otherwise libgpuxtb is found next to this
// extension (both live in the wheel's gpuxtb/lib directory), with a plain
// system-name fallback. It is dlopen'ed rather than linked so the extension
// never hard-codes a gpuxtb SONAME.
// ---------------------------------------------------------------------------

struct GpuxtbApi {
  int32_t (*context_options_init)(gpuxtb_context_options_t*, size_t);
  int32_t (*batch_init)(gpuxtb_batch_t*, size_t);
  int32_t (*compute_options_init)(gpuxtb_compute_options_t*, size_t);
  int32_t (*batch_result_init)(gpuxtb_batch_result_t*, size_t);
  int32_t (*context_create)(const gpuxtb_context_options_t*, gpuxtb_context_t**);
  void (*context_destroy)(gpuxtb_context_t*);
  int32_t (*context_get_backend)(const gpuxtb_context_t*);
  int32_t (*context_get_device_id)(const gpuxtb_context_t*);
  int32_t (*compute)(gpuxtb_context_t*, const gpuxtb_batch_t*, const gpuxtb_compute_options_t*,
                     gpuxtb_batch_result_t*);
  const char* (*get_last_error)(void);
  const char* (*status_string)(int32_t);
};

// dladdr anchor: must keep default visibility so dladdr() can resolve this
// object's own path even inside a hidden-visibility build.
extern "C" void gpuxtb_torch_ext_anchor() {}

#if defined(_WIN32)
std::string own_directory() {
  HMODULE module = nullptr;
  if (!GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<const char*>(&gpuxtb_torch_ext_anchor), &module)) {
    return {};
  }
  char buffer[MAX_PATH] = {0};
  DWORD length = GetModuleFileNameA(module, buffer, MAX_PATH);
  if (length == 0) {
    return {};
  }
  std::string path(buffer, length);
  size_t separator = path.find_last_of("\\/");
  return separator == std::string::npos ? std::string() : path.substr(0, separator);
}

void* load_shared(const std::string& path) {
  return static_cast<void*>(LoadLibraryA(path.c_str()));
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
  if (dladdr(reinterpret_cast<void*>(&gpuxtb_torch_ext_anchor), &info) == 0 ||
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
  return "gpuxtb.dll";
#elif defined(__APPLE__)
  return "libgpuxtb.dylib";
#else
  return "libgpuxtb.so";
#endif
}

const GpuxtbApi& gpuxtb_api() {
  // The API table is initialized once per process.  gpuxtb itself is
  // thread-safe for separate per-call contexts, and reads of the immutable
  // table after the once-flag needs no further synchronization.
  static const GpuxtbApi* table = []() -> const GpuxtbApi* {
    void* handle = nullptr;
    const char* override_path = std::getenv("GPUXTB_LIBRARY");
    const bool has_override = override_path != nullptr && override_path[0] != '\0';
    if (has_override) {
      handle = load_shared(override_path);
    } else {
      std::string directory = own_directory();
      for (const char* name : {"libgpuxtb.so", "libgpuxtb.so.0", "libgpuxtb.dylib", "gpuxtb.dll"}) {
        handle = open_candidate(directory, name);
        if (handle != nullptr) {
          break;
        }
      }
      if (handle == nullptr) {
        // System-lookup fallback matching ctypes' find_library("gpuxtb").
        handle = load_shared(default_search_name());
      }
    }
    if (handle == nullptr) {
      return nullptr;
    }
    auto api = new GpuxtbApi();
    api->context_options_init = reinterpret_cast<int32_t (*)(gpuxtb_context_options_t*, size_t)>(
        resolve_symbol(handle, "gpuxtb_context_options_init"));
    api->batch_init = reinterpret_cast<int32_t (*)(gpuxtb_batch_t*, size_t)>(
        resolve_symbol(handle, "gpuxtb_batch_init"));
    api->compute_options_init = reinterpret_cast<int32_t (*)(gpuxtb_compute_options_t*, size_t)>(
        resolve_symbol(handle, "gpuxtb_compute_options_init"));
    api->batch_result_init = reinterpret_cast<int32_t (*)(gpuxtb_batch_result_t*, size_t)>(
        resolve_symbol(handle, "gpuxtb_batch_result_init"));
    api->context_create =
        reinterpret_cast<int32_t (*)(const gpuxtb_context_options_t*, gpuxtb_context_t**)>(
            resolve_symbol(handle, "gpuxtb_context_create"));
    api->context_destroy = reinterpret_cast<void (*)(gpuxtb_context_t*)>(
        resolve_symbol(handle, "gpuxtb_context_destroy"));
    api->context_get_backend = reinterpret_cast<int32_t (*)(const gpuxtb_context_t*)>(
        resolve_symbol(handle, "gpuxtb_context_get_backend"));
    api->context_get_device_id = reinterpret_cast<int32_t (*)(const gpuxtb_context_t*)>(
        resolve_symbol(handle, "gpuxtb_context_get_device_id"));
    api->compute =
        reinterpret_cast<int32_t (*)(gpuxtb_context_t*, const gpuxtb_batch_t*,
                                     const gpuxtb_compute_options_t*, gpuxtb_batch_result_t*)>(
            resolve_symbol(handle, "gpuxtb_compute"));
    api->get_last_error =
        reinterpret_cast<const char* (*)(void)>(resolve_symbol(handle, "gpuxtb_get_last_error"));
    api->status_string =
        reinterpret_cast<const char* (*)(int32_t)>(resolve_symbol(handle, "gpuxtb_status_string"));
    if (api->context_options_init == nullptr || api->batch_init == nullptr ||
        api->compute_options_init == nullptr || api->batch_result_init == nullptr ||
        api->context_create == nullptr || api->context_destroy == nullptr ||
        api->context_get_backend == nullptr || api->context_get_device_id == nullptr ||
        api->compute == nullptr || api->get_last_error == nullptr ||
        api->status_string == nullptr) {
      return nullptr;
    }
    return api;
  }();
  if (table == nullptr) {
    STD_TORCH_CHECK(false,
                    "gpuxtb_torch: cannot load libgpuxtb; set GPUXTB_LIBRARY or "
                    "install gpuxtb with its native library bundled against the "
                    "torch extension");
  }
  return *table;
}

// ---------------------------------------------------------------------------
// Tensor contract helpers
// ---------------------------------------------------------------------------

gpuxtb_memory_space_t memory_space_of(const Tensor& tensor) {
  return tensor.is_cuda() ? GPUXTB_MEMORY_CUDA_DEVICE : GPUXTB_MEMORY_HOST;
}

void require_contiguous_1d(const Tensor& tensor, ScalarType dtype, const char* name,
                           int64_t expected_size = -1) {
  STD_TORCH_CHECK(tensor.defined(), "gpuxtb_torch: ", name, " is undefined");
  STD_TORCH_CHECK(tensor.is_contiguous(), "gpuxtb_torch: ", name,
                  " must be C-contiguous (the op packs non-contiguous inputs)");
  STD_TORCH_CHECK(tensor.dim() == 1, "gpuxtb_torch: ", name, " must be one-dimensional");
  STD_TORCH_CHECK(tensor.scalar_type() == dtype, "gpuxtb_torch: ", name, " has the wrong dtype");
  if (expected_size >= 0) {
    STD_TORCH_CHECK(tensor.size(0) == expected_size, "gpuxtb_torch: ", name,
                    " has the wrong length");
  }
}

// RAII guard that always destroys a created gpuxtb context, including on throw.
class ContextGuard {
 public:
  explicit ContextGuard(gpuxtb_context_t* handle) : handle_(handle) {}
  ~ContextGuard() {
    if (handle_ != nullptr) {
      gpuxtb_api().context_destroy(handle_);
    }
  }
  ContextGuard(const ContextGuard&) = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;

 private:
  gpuxtb_context_t* handle_;
};

void check_status(int32_t status, const char* what) {
  if (status == GPUXTB_STATUS_SUCCESS) {
    return;
  }
  const char* detail = gpuxtb_api().get_last_error();
  const char* label = gpuxtb_api().status_string(status);
  STD_TORCH_CHECK(false, "gpuxtb_torch: ", what, " failed with ",
                  label != nullptr ? label : "unknown status", " (", status, ")",
                  detail != nullptr && detail[0] != '\0' ? ": " : "",
                  detail != nullptr ? detail : "");
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
//    written in place and returned by the op.
//  - electronic_temperature is in kelvin, exactly like gpuxtb.gpuxtb_torch;
//    it is converted to the native k_B*T Hartree scale inside the op.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> gpuxtb_torch_forward(
    Tensor positions, Tensor atomic_numbers, Tensor atom_offsets, Tensor molecular_charges,
    Tensor unpaired_electrons, Tensor spin_channels, Tensor out_energies, Tensor out_forces,
    int64_t backend, int64_t device_id, int64_t cpu_threads, int64_t stream,
    int64_t max_scc_iterations, double charge_tolerance, double energy_tolerance,
    double electronic_temperature) {
  const GpuxtbApi& api = gpuxtb_api();
  // Fail fast when the torch extension was built against an incompatible
  // ABI level, before any tensor contract is assumed.
  STD_TORCH_CHECK(TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0,
                  "gpuxtb_torch requires the torch stable ABI (torch >= 2.10)");

  // --- validate shapes/dtypes (mirror of the Python preflight) --------------
  STD_TORCH_CHECK(positions.is_contiguous(), "gpuxtb_torch: positions must be C-contiguous");
  STD_TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
                  "gpuxtb_torch: positions must have shape (natoms, 3)");
  STD_TORCH_CHECK(positions.scalar_type() == ScalarType::Double,
                  "gpuxtb_torch: positions must be float64");
  const int64_t natoms = positions.size(0);
  require_contiguous_1d(atomic_numbers, ScalarType::Int, "atomic_numbers", natoms);
  require_contiguous_1d(molecular_charges, ScalarType::Double, "molecular_charges");
  const int64_t nsystems = molecular_charges.size(0);
  STD_TORCH_CHECK(nsystems >= 1, "gpuxtb_torch: a batch needs at least one system");
  require_contiguous_1d(atom_offsets, ScalarType::Long, "atom_offsets", nsystems + 1);
  require_contiguous_1d(unpaired_electrons, ScalarType::Int, "unpaired_electrons", nsystems);
  require_contiguous_1d(spin_channels, ScalarType::Int, "spin_channels", nsystems);
  require_contiguous_1d(out_energies, ScalarType::Double, "out_energies", nsystems);
  STD_TORCH_CHECK(out_forces.is_contiguous(), "gpuxtb_torch: out_forces must be C-contiguous");
  STD_TORCH_CHECK(out_forces.dim() == 2 && out_forces.size(1) == 3,
                  "gpuxtb_torch: out_forces must have shape (natoms, 3)");
  STD_TORCH_CHECK(out_forces.size(0) == natoms, "gpuxtb_torch: out_forces atom count mismatch");
  STD_TORCH_CHECK(out_forces.scalar_type() == ScalarType::Double,
                  "gpuxtb_torch: out_forces must be float64");

  // The public CUDA ABI accepts mixed host/device descriptors. Ignore host
  // tensors here, but require every CUDA tensor to use one device. Reject
  // other accelerator types before memory_space_of could mislabel them HOST.
  bool any_cuda = false;
  int64_t cuda_device_index = -1;
  for (const Tensor& tensor : {positions, atomic_numbers, atom_offsets, molecular_charges,
                               unpaired_electrons, spin_channels, out_energies, out_forces}) {
    STD_TORCH_CHECK(tensor.is_cpu() || tensor.is_cuda(),
                    "gpuxtb_torch: only CPU and CUDA tensors are supported");
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
                    "gpuxtb_torch: all CUDA tensors must be on device ", cuda_device_index);
  }

  // --- context ---------------------------------------------------------------
  gpuxtb_context_options_t context_options;
  check_status(api.context_options_init(&context_options, sizeof(context_options)),
               "gpuxtb_context_options_init");
  context_options.backend = static_cast<gpuxtb_backend_t>(backend);
  context_options.device_id = static_cast<int32_t>(device_id);
  context_options.cpu_threads = static_cast<int32_t>(cpu_threads);
  context_options.stream =
      stream > 0 ? reinterpret_cast<void*>(static_cast<uintptr_t>(stream)) : nullptr;

  gpuxtb_context_t* context = nullptr;
  int32_t context_status = api.context_create(&context_options, &context);
  if (context_status != GPUXTB_STATUS_SUCCESS && backend == GPUXTB_BACKEND_AUTO && !any_cuda &&
      device_id < 0 && context_options.stream != nullptr) {
    // AUTO with all-host tensors may legitimately resolve to CPU in a CPU-only
    // gpuxtb build. Python cannot know that resolution before the context is
    // created, so it supplies Torch's current CUDA stream as a candidate. Retry
    // without that candidate to preserve AUTO's established CPU fallback.
    if (context != nullptr) {
      api.context_destroy(context);
      context = nullptr;
    }
    context_options.stream = nullptr;
    context_status = api.context_create(&context_options, &context);
  }
  check_status(context_status, "gpuxtb_context_create");
  ContextGuard context_guard(context);

  // Mirror the Python layer's device-consistency gate: CUDA inputs must match
  // the device the context actually resolved.
  if (any_cuda) {
    int32_t resolved_backend = api.context_get_backend(context);
    STD_TORCH_CHECK(resolved_backend == GPUXTB_BACKEND_CUDA,
                    "gpuxtb_torch: CUDA tensors require the CUDA backend");
    int32_t resolved_device = api.context_get_device_id(context);
    STD_TORCH_CHECK(resolved_device == cuda_device_index, "gpuxtb_torch: CUDA tensor on device ",
                    cuda_device_index, " does not match the context device ", resolved_device);
  }

  // --- batch -----------------------------------------------------------------
  gpuxtb_batch_t batch;
  check_status(api.batch_init(&batch, sizeof(batch)), "gpuxtb_batch_init");
  batch.batch_size = nsystems;
  batch.total_atoms = natoms;
  batch.total_point_charges = 0;
  batch.total_charge_response_elements = 0;

  auto bind_input = [&](gpuxtb_const_buffer_t* buffer, const Tensor& tensor) {
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
  gpuxtb_compute_options_t options;
  check_status(api.compute_options_init(&options, sizeof(options)), "gpuxtb_compute_options_init");
  options.model = GPUXTB_MODEL_GFN2_XTB;
  options.flags = static_cast<uint32_t>(GPUXTB_COMPUTE_ENERGY | GPUXTB_COMPUTE_FORCES |
                                        GPUXTB_COMPUTE_ATOMIC_CHARGES);
  options.max_scc_iterations = static_cast<int32_t>(max_scc_iterations);
  options.charge_tolerance = charge_tolerance;
  options.energy_tolerance = energy_tolerance;
  options.electronic_temperature = electronic_temperature * GPUXTB_KELVIN_TO_HARTREE;

  // --- result ------------------------------------------------------------------
  // energies/forces go straight into the caller's out tensors (zero copy).
  // Diagnostics and the (unreported) atomic charges use small host buffers
  // that the C ABI always requires; a CUDA compute still accepts host result
  // buffers, matching the historical per-call host diagnostics.
  std::vector<int32_t> scc_iterations(static_cast<size_t>(nsystems));
  std::vector<uint8_t> scc_converged(static_cast<size_t>(nsystems));
  std::vector<int32_t> per_system_status(static_cast<size_t>(nsystems));
  std::vector<double> atomic_charges(static_cast<size_t>(natoms));

  auto bind_output = [](gpuxtb_buffer_t* buffer, void* data, size_t bytes,
                        gpuxtb_memory_space_t space) {
    buffer->data = data;
    buffer->size_bytes = bytes;
    buffer->memory_space = space;
    buffer->reserved = 0;
  };

  gpuxtb_batch_result_t result;
  check_status(api.batch_result_init(&result, sizeof(result)), "gpuxtb_batch_result_init");
  bind_output(&result.energies, out_energies.mutable_data_ptr(),
              static_cast<size_t>(nsystems) * sizeof(double), memory_space_of(out_energies));
  bind_output(&result.forces, out_forces.mutable_data_ptr(),
              static_cast<size_t>(natoms * 3) * sizeof(double), memory_space_of(out_forces));
  bind_output(&result.atomic_charges, atomic_charges.data(), atomic_charges.size() * sizeof(double),
              GPUXTB_MEMORY_HOST);
  bind_output(&result.scc_iterations, scc_iterations.data(),
              scc_iterations.size() * sizeof(int32_t), GPUXTB_MEMORY_HOST);
  bind_output(&result.scc_converged, scc_converged.data(), scc_converged.size() * sizeof(uint8_t),
              GPUXTB_MEMORY_HOST);
  bind_output(&result.per_system_status, per_system_status.data(),
              per_system_status.size() * sizeof(int32_t), GPUXTB_MEMORY_HOST);

  check_status(api.compute(context, &batch, &options, &result), "gpuxtb_compute");

  return {out_energies, out_forces};
}

STABLE_TORCH_LIBRARY(gpuxtb, m) {
  m.def(
      "gpuxtb_torch_forward(Tensor positions, Tensor atomic_numbers, Tensor atom_offsets, "
      "Tensor molecular_charges, Tensor unpaired_electrons, Tensor spin_channels, "
      "Tensor(a!) out_energies, Tensor(b!) out_forces, int backend, int device_id, "
      "int cpu_threads, "
      "int stream, int max_scc_iterations, float charge_tolerance, float energy_tolerance, "
      "float electronic_temperature) -> (Tensor(a!), Tensor(b!))");
}
STABLE_TORCH_LIBRARY_IMPL(gpuxtb, CompositeExplicitAutograd, m) {
  m.impl("gpuxtb_torch_forward", TORCH_BOX(&gpuxtb_torch_forward));
}
