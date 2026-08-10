#include "model/gfn2/eigensolver.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/gfn2/occupation_binary64_policy.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if (defined(XTBLOOM_CONFIGURED_CPU_LINALG_SHIM) || defined(XTBLOOM_CONFIGURED_WHEEL_OPENBLAS)) && \
    defined(__linux__)
#include <link.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace xtbloom::detail::gfn2 {

namespace {

constexpr std::size_t kEigensolverFieldCount = 5u;

struct EigensolverFieldData {
  std::size_t offset_bytes = 0u;
  std::int64_t element_count = 0;
  std::vector<std::int64_t> system_offsets;
};

}  // namespace

struct EigensolverPlanData final {
  std::int64_t batch_size = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t maximum_orbitals = 0;
  double minimum_overlap_rcond = 1.0e-12;
  std::vector<std::int64_t> matrix_offsets;
  std::vector<std::int64_t> orbital_offsets;
  std::vector<std::int32_t> spin_channels;
  std::vector<double> alpha_electron_counts;
  std::vector<double> beta_electron_counts;

  std::size_t wavefunction_workspace_size_bytes = 0u;
  std::array<EigensolverFieldData, kEigensolverFieldCount> wavefunction_fields;

  std::size_t overlap_cache_size_bytes = 0u;
  std::size_t overlap_factor_offset_bytes = 0u;
  std::size_t overlap_generation_offset_bytes = 0u;
  std::size_t overlap_status_offset_bytes = 0u;

  std::size_t worker_workspace_size_bytes = 0u;
  std::size_t workspace_size_bytes = 0u;
  std::size_t coefficient_scratch_offset_bytes = 0u;
  std::size_t density_scratch_offset_bytes = 0u;
  std::size_t energy_weighted_density_scratch_offset_bytes = 0u;
  std::size_t eigenvalue_scratch_offset_bytes = 0u;
  std::size_t occupation_scratch_offset_bytes = 0u;
  std::size_t lapack_work_offset_bytes = 0u;
  std::size_t lapack_integer_work_offset_bytes = 0u;
  std::size_t factor_staging_offset_bytes = 0u;
  std::size_t factor_generation_staging_offset_bytes = 0u;
  std::size_t factor_status_staging_offset_bytes = 0u;
  std::size_t batch_coefficient_staging_offset_bytes = 0u;
  std::size_t batch_density_staging_offset_bytes = 0u;
  std::size_t batch_energy_weighted_density_staging_offset_bytes = 0u;
  std::size_t batch_eigenvalue_staging_offset_bytes = 0u;
  std::size_t batch_occupation_staging_offset_bytes = 0u;
  std::size_t batch_system_status_staging_offset_bytes = 0u;
  std::size_t batch_chemical_potential_staging_offset_bytes = 0u;
  std::size_t batch_entropy_staging_offset_bytes = 0u;
  std::size_t batch_band_energy_staging_offset_bytes = 0u;
  std::size_t batch_free_energy_staging_offset_bytes = 0u;
  LapackInt lapack_work_count = 0;
  LapackInt lapack_integer_work_count = 0;
};

struct CpuLinearAlgebraAccess {
  static CpuLinearAlgebraBackend make(CpuLinearAlgebraBackend::Origin origin,
                                      LapackDpotrfWork dpotrf_work, LapackDpoconWork dpocon_work,
                                      LapackDsyevdWork dsyevd_work, CblasDtrsm dtrsm,
                                      CblasDgemm dgemm,
                                      BlasSetNumThreadsLocal set_num_threads_local) noexcept {
    CpuLinearAlgebraBackend backend;
    backend.origin_ = origin;
    backend.dpotrf_work_ = dpotrf_work;
    backend.dpocon_work_ = dpocon_work;
    backend.dsyevd_work_ = dsyevd_work;
    backend.dtrsm_ = dtrsm;
    backend.dgemm_ = dgemm;
    backend.set_num_threads_local_ = set_num_threads_local;
    return backend;
  }

  static LapackDpotrfWork dpotrf(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.dpotrf_work_;
  }
  static LapackDpoconWork dpocon(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.dpocon_work_;
  }
  static LapackDsyevdWork dsyevd(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.dsyevd_work_;
  }
  static CblasDtrsm dtrsm(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.dtrsm_;
  }
  static CblasDgemm dgemm(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.dgemm_;
  }
  static BlasSetNumThreadsLocal set_threads(const CpuLinearAlgebraBackend& backend) noexcept {
    return backend.set_num_threads_local_;
  }
};

namespace {

static_assert(sizeof(LapackInt) == 4u, "the CPU eigensolver requires an LP64 LAPACK ABI");
static_assert(sizeof(xtbloom_status_t) == 4u, "cache statuses require the public 32-bit ABI");
static_assert(std::is_trivially_copyable_v<EigensolverOverlapCache>);
static_assert(std::is_standard_layout_v<EigensolverOverlapCache>);
static_assert(std::is_trivially_copyable_v<EigensolverWorkspace>);
static_assert(std::is_standard_layout_v<EigensolverWorkspace>);

constexpr int kCblasColMajor = 102;
constexpr int kCblasNoTrans = 111;
constexpr int kCblasTrans = 112;
constexpr int kCblasLower = 122;
constexpr int kCblasNonUnit = 131;
constexpr int kCblasLeft = 141;
constexpr int kCblasRight = 142;
constexpr char kLower = 'L';
constexpr char kEigenvectors = 'V';

enum class NumericalResult { kSuccess, kDataFailure, kBackendFailure };

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

struct MemoryRange {
  const void* data = nullptr;
  std::size_t size_bytes = 0u;
};

bool checked_add(std::size_t left, std::size_t right, std::size_t& result) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

bool checked_multiply(std::size_t left, std::size_t right, std::size_t& result) {
  if (right != 0u && left > std::numeric_limits<std::size_t>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

bool align_up(std::size_t value, std::size_t& result) {
  const std::size_t remainder = value % kEigensolverWorkspaceAlignment;
  const std::size_t padding = remainder == 0u ? 0u : kEigensolverWorkspaceAlignment - remainder;
  return checked_add(value, padding, result);
}

bool append_segment(std::size_t byte_count, std::size_t& cursor, std::size_t& offset) {
  return align_up(cursor, offset) && checked_add(offset, byte_count, cursor);
}

bool make_range(const void* pointer, std::size_t byte_count, AddressRange& range) {
  if (pointer == nullptr || byte_count == 0u) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - byte_count) {
    return false;
  }
  range = {begin, begin + byte_count};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.begin < second.end && second.begin < first.end;
}

bool is_aligned(const void* pointer, std::size_t alignment) {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

template <typename T>
T* offset_pointer(void* base, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

template <typename T>
const T* offset_pointer(const void* base, std::size_t offset) {
  return reinterpret_cast<const T*>(static_cast<const std::byte*>(base) + offset);
}

std::array<const WavefunctionFieldLayout*, kEigensolverFieldCount> eigensolver_layout_fields(
    const WavefunctionLayout& layout) {
  return {{&layout.coefficients, &layout.eigenvalues, &layout.occupations, &layout.density,
           &layout.energy_weighted_density}};
}

std::array<double*, kEigensolverFieldCount> eigensolver_view_fields(const WavefunctionView& view) {
  return {{view.coefficients, view.eigenvalues, view.occupations, view.density,
           view.energy_weighted_density}};
}

xtbloom_status_t validate_plan(const EigensolverPlan& plan, std::string& error) {
  if (!plan.sealed()) {
    error = "eigensolver plan is default-constructed or moved-from";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_backend(const CpuLinearAlgebraBackend& backend, std::string& error) {
  if (!backend.ready()) {
    error = "CPU eigensolver requires a verified LP64 backend";
    return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

template <typename Function>
bool load_symbol(void* handle, const char* name, Function& function) {
#if defined(_WIN32)
  static_assert(sizeof(Function) == sizeof(FARPROC));
  const FARPROC symbol = GetProcAddress(static_cast<HMODULE>(handle), name);
  if (symbol == nullptr) {
    return false;
  }
#else
  static_assert(sizeof(Function) == sizeof(void*));
  dlerror();
  void* symbol = dlsym(handle, name);
  if (symbol == nullptr || dlerror() != nullptr) {
    return false;
  }
#endif
  std::memcpy(&function, &symbol, sizeof(function));
  return true;
}

bool load_lapacke_cblas_symbols(void* handle, bool scipy_prefix, LapackDpotrfWork& dpotrf_work,
                                LapackDpoconWork& dpocon_work, LapackDsyevdWork& dsyevd_work,
                                CblasDtrsm& dtrsm, CblasDgemm& dgemm) {
  /* Load one coherent ABI. scipy-openblas32 prefixes every public symbol to
   * coexist safely with another BLAS; mixing standard and prefixed functions
   * from the same handle could combine incompatible providers accidentally. */
  dpotrf_work = nullptr;
  dpocon_work = nullptr;
  dsyevd_work = nullptr;
  dtrsm = nullptr;
  dgemm = nullptr;
  if (scipy_prefix) {
    return load_symbol(handle, "scipy_LAPACKE_dpotrf_work", dpotrf_work) &&
           load_symbol(handle, "scipy_LAPACKE_dpocon_work", dpocon_work) &&
           load_symbol(handle, "scipy_LAPACKE_dsyevd_work", dsyevd_work) &&
           load_symbol(handle, "scipy_cblas_dtrsm", dtrsm) &&
           load_symbol(handle, "scipy_cblas_dgemm", dgemm);
  }
  return load_symbol(handle, "LAPACKE_dpotrf_work", dpotrf_work) &&
         load_symbol(handle, "LAPACKE_dpocon_work", dpocon_work) &&
         load_symbol(handle, "LAPACKE_dsyevd_work", dsyevd_work) &&
         load_symbol(handle, "cblas_dtrsm", dtrsm) && load_symbol(handle, "cblas_dgemm", dgemm);
}

bool backend_self_test(const CpuLinearAlgebraBackend& backend) {
  double factor[1]{1.0};
  double reciprocal_condition = 0.0;
  double work[9]{};
  LapackInt integer_work[8]{};
  if (CpuLinearAlgebraAccess::dpotrf(backend)(kCblasColMajor, kLower, 1, factor, 1) != 0 ||
      CpuLinearAlgebraAccess::dpocon(backend)(kCblasColMajor, kLower, 1, factor, 1, 1.0,
                                              &reciprocal_condition, work, integer_work) != 0 ||
      !(reciprocal_condition > 0.0) || !std::isfinite(reciprocal_condition)) {
    return false;
  }
  double eigenvectors[1]{2.0};
  double eigenvalues[1]{};
  if (CpuLinearAlgebraAccess::dsyevd(backend)(kCblasColMajor, kEigenvectors, kLower, 1,
                                              eigenvectors, 1, eigenvalues, work, 9, integer_work,
                                              8) != 0 ||
      eigenvalues[0] != 2.0 || eigenvectors[0] != 1.0) {
    return false;
  }
  double rhs[1]{4.0};
  const double triangular[1]{2.0};
  CpuLinearAlgebraAccess::dtrsm(backend)(kCblasColMajor, kCblasLeft, kCblasLower, kCblasNoTrans,
                                         kCblasNonUnit, 1, 1, 1.0, triangular, 1, rhs, 1);
  double product[1]{};
  CpuLinearAlgebraAccess::dgemm(backend)(kCblasColMajor, kCblasNoTrans, kCblasTrans, 1, 1, 1, 1.0,
                                         rhs, 1, rhs, 1, 0.0, product, 1);
  return rhs[0] == 2.0 && product[0] == 4.0;
}

#if defined(XTBLOOM_CONFIGURED_WHEEL_OPENBLAS) && defined(_WIN32)
void* open_private_bundled_sibling(const char* filename) {
  /* Resolve relative to xtbloom.dll itself, never the process working
   * directory or PATH. LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR keeps any provider
   * dependencies in the same private wheel directory, while the default
   * directories retain Windows system-runtime resolution. */
  static const unsigned char kModuleAnchor = 0u;
  HMODULE module = nullptr;
  const DWORD flags =
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
  if (GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&kModuleAnchor), &module) == 0) {
    return nullptr;
  }

  std::array<wchar_t, 32768> module_path{};
  const DWORD length =
      GetModuleFileNameW(module, module_path.data(), static_cast<DWORD>(module_path.size()));
  if (length == 0u || length >= module_path.size()) {
    return nullptr;
  }
  std::wstring path(module_path.data(), length);
  const std::size_t separator = path.find_last_of(L"\\/");
  if (separator == std::wstring::npos) {
    return nullptr;
  }
  path.resize(separator + 1u);
  for (const char* character = filename; *character != '\0'; ++character) {
    path.push_back(static_cast<wchar_t>(static_cast<unsigned char>(*character)));
  }
  return static_cast<void*>(LoadLibraryExW(
      path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
}

void close_dynamic_library(void* handle) {
  if (handle != nullptr) {
    static_cast<void>(FreeLibrary(static_cast<HMODULE>(handle)));
  }
}
#elif defined(XTBLOOM_CONFIGURED_WHEEL_OPENBLAS) && defined(__APPLE__)
std::string canonical_path(const char* path) {
  char* resolved = realpath(path, nullptr);
  if (resolved == nullptr) {
    return {};
  }
  std::string result(resolved);
  std::free(resolved);
  return result;
}

void* open_private_bundled_sibling(const char* filename, std::string& expected_path) {
  /* The provider has an xTBloom-private LC_ID and is opened by the absolute
   * path beside libxtbloom. This prevents name-based discovery and host SciPy
   * reuse without claiming Linux dlmopen-style namespace isolation. */
  static const unsigned char kModuleAnchor = 0u;
  Dl_info module{};
  if (dladdr(&kModuleAnchor, &module) == 0 || module.dli_fname == nullptr) {
    return nullptr;
  }
  std::string path(module.dli_fname);
  const std::size_t separator = path.find_last_of('/');
  if (separator == std::string::npos) {
    return nullptr;
  }
  path.resize(separator + 1u);
  path += filename;
  expected_path = canonical_path(path.c_str());
  if (expected_path.empty()) {
    return nullptr;
  }
  return dlopen(expected_path.c_str(), RTLD_NOW | RTLD_LOCAL);
}

void close_dynamic_library(void* handle) {
  if (handle != nullptr) {
    static_cast<void>(dlclose(handle));
  }
}
#elif defined(XTBLOOM_CONFIGURED_CPU_LINALG_SHIM) || defined(XTBLOOM_CONFIGURED_WHEEL_OPENBLAS)
void* open_host_isolated_sibling(const char* soname) {
  /* A LOCAL handle still resolves relocations against already-global objects.
   * A new link-map namespace is required to keep a host BLAS implementation
   * from interposing on xTBloom's private LP64 provider cohort. */
  static const unsigned char kModuleAnchor = 0u;
  Dl_info module{};
  if (dladdr(&kModuleAnchor, &module) == 0 || module.dli_fname == nullptr) {
    return nullptr;
  }
  std::string path(module.dli_fname);
  const std::size_t separator = path.find_last_of('/');
  if (separator == std::string::npos) {
    return nullptr;
  }
  path.resize(separator + 1u);
  path += soname;

  void* handle = dlmopen(LM_ID_NEWLM, path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    return nullptr;
  }
  Lmid_t namespace_id = LM_ID_BASE;
  if (dlinfo(handle, RTLD_DI_LMID, &namespace_id) != 0 || namespace_id == LM_ID_BASE) {
    static_cast<void>(dlclose(handle));
    return nullptr;
  }
  return handle;
}
#endif

#if defined(XTBLOOM_CONFIGURED_WHEEL_OPENBLAS) && defined(_WIN32)
template <typename Function>
HMODULE dynamic_symbol_module(Function function) {
  static_assert(sizeof(Function) == sizeof(void*));
  void* address = nullptr;
  std::memcpy(&address, &function, sizeof(address));
  HMODULE module = nullptr;
  const DWORD flags =
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
  if (GetModuleHandleExW(flags, static_cast<LPCWSTR>(address), &module) == 0) {
    return nullptr;
  }
  return module;
}

template <typename First, typename... Rest>
bool symbols_belong_to_private_provider(void* opened_handle, First first, Rest... rest) {
  const HMODULE module = static_cast<HMODULE>(opened_handle);
  if (module == nullptr || dynamic_symbol_module(first) != module ||
      ((dynamic_symbol_module(rest) != module) || ...)) {
    return false;
  }
  return true;
}
#elif defined(XTBLOOM_CONFIGURED_WHEEL_OPENBLAS) && defined(__APPLE__)
template <typename Function>
bool dynamic_symbol_info(Function function, Dl_info& info) {
  static_assert(sizeof(Function) == sizeof(void*));
  void* address = nullptr;
  std::memcpy(&address, &function, sizeof(address));
  return dladdr(address, &info) != 0 && info.dli_fbase != nullptr && info.dli_fname != nullptr;
}

template <typename First, typename... Rest>
bool symbols_belong_to_private_provider(const std::string& expected_path, First first,
                                        Rest... rest) {
  Dl_info first_info{};
  if (!dynamic_symbol_info(first, first_info)) {
    return false;
  }
  bool same_image = true;
  const auto check = [&](auto function) {
    Dl_info info{};
    same_image =
        same_image && dynamic_symbol_info(function, info) && info.dli_fbase == first_info.dli_fbase;
  };
  (check(rest), ...);
  return same_image && canonical_path(first_info.dli_fname) == expected_path;
}
#endif

class ScopedSequentialBlas final {
 public:
  explicit ScopedSequentialBlas(const CpuLinearAlgebraBackend& backend)
      : setter_(CpuLinearAlgebraAccess::set_threads(backend)) {
    if (setter_ != nullptr) {
      previous_ = setter_(1);
    }
  }

  ~ScopedSequentialBlas() {
    if (setter_ != nullptr) {
      static_cast<void>(setter_(previous_));
    }
  }

  ScopedSequentialBlas(const ScopedSequentialBlas&) = delete;
  ScopedSequentialBlas& operator=(const ScopedSequentialBlas&) = delete;

 private:
  BlasSetNumThreadsLocal setter_ = nullptr;
  int previous_ = 0;
};

std::array<MemoryRange, 11> plan_storage_ranges(const EigensolverPlan& plan) {
  const EigensolverPlanData& data = *plan.identity();
  std::array<MemoryRange, 11> ranges{};
  ranges[0] = {&data, sizeof(data)};
  ranges[1] = {data.matrix_offsets.data(), data.matrix_offsets.capacity() * sizeof(std::int64_t)};
  ranges[2] = {data.orbital_offsets.data(), data.orbital_offsets.capacity() * sizeof(std::int64_t)};
  ranges[3] = {data.spin_channels.data(), data.spin_channels.capacity() * sizeof(std::int32_t)};
  ranges[4] = {data.alpha_electron_counts.data(),
               data.alpha_electron_counts.capacity() * sizeof(double)};
  ranges[5] = {data.beta_electron_counts.data(),
               data.beta_electron_counts.capacity() * sizeof(double)};
  for (std::size_t field = 0u; field < kEigensolverFieldCount; ++field) {
    ranges[6u + field] = {
        data.wavefunction_fields[field].system_offsets.data(),
        data.wavefunction_fields[field].system_offsets.capacity() * sizeof(std::int64_t)};
  }
  return ranges;
}

bool overlaps_plan_storage(const EigensolverPlan& plan, const AddressRange& active) {
  for (const MemoryRange& range : plan_storage_ranges(plan)) {
    if (range.size_bytes == 0u) {
      continue;
    }
    AddressRange stored;
    if (!make_range(range.data, range.size_bytes, stored) || ranges_overlap(active, stored)) {
      return true;
    }
  }
  return false;
}

template <std::size_t N>
bool pairwise_disjoint(const std::array<AddressRange, N>& ranges) {
  for (std::size_t first = 0u; first < N; ++first) {
    for (std::size_t second = first + 1u; second < N; ++second) {
      if (ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

template <std::size_t ActiveCount, std::size_t ControlCount>
bool disjoint_from_control(const EigensolverPlan& plan,
                           const std::array<AddressRange, ActiveCount>& active,
                           const std::array<AddressRange, ControlCount>& controls) {
  for (const AddressRange& range : active) {
    if (overlaps_plan_storage(plan, range)) {
      return false;
    }
    for (const AddressRange& control : controls) {
      if (ranges_overlap(range, control)) {
        return false;
      }
    }
  }
  return true;
}

xtbloom_status_t validate_cache(const EigensolverPlan& plan, const EigensolverOverlapCache& cache,
                                std::string& error) {
  const EigensolverPlanData& data = *plan.identity();
  if (cache.workspace_base == nullptr ||
      cache.workspace_size_bytes < data.overlap_cache_size_bytes ||
      !is_aligned(cache.workspace_base, kEigensolverWorkspaceAlignment) ||
      cache.plan_identity != &data ||
      cache.cholesky_factors !=
          offset_pointer<double>(cache.workspace_base, data.overlap_factor_offset_bytes) ||
      cache.geometry_generations !=
          offset_pointer<std::uint64_t>(cache.workspace_base,
                                        data.overlap_generation_offset_bytes) ||
      cache.system_statuses != offset_pointer<xtbloom_status_t>(cache.workspace_base,
                                                                data.overlap_status_offset_bytes)) {
    error = "eigensolver overlap cache is not a canonical binding for this plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  AddressRange range;
  if (!make_range(cache.workspace_base, data.overlap_cache_size_bytes, range)) {
    error = "eigensolver overlap cache address range is not representable";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_worker_workspace(const EigensolverPlan& plan,
                                           const EigensolverWorkspace& workspace,
                                           std::string& error) {
  const EigensolverPlanData& data = *plan.identity();
  if (workspace.workspace_base == nullptr ||
      workspace.workspace_size_bytes < data.worker_workspace_size_bytes ||
      !is_aligned(workspace.workspace_base, kEigensolverWorkspaceAlignment) ||
      workspace.plan_identity != &data ||
      workspace.coefficients !=
          offset_pointer<double>(workspace.workspace_base, data.coefficient_scratch_offset_bytes) ||
      workspace.densities !=
          offset_pointer<double>(workspace.workspace_base, data.density_scratch_offset_bytes) ||
      workspace.energy_weighted_densities !=
          offset_pointer<double>(workspace.workspace_base,
                                 data.energy_weighted_density_scratch_offset_bytes) ||
      workspace.eigenvalues !=
          offset_pointer<double>(workspace.workspace_base, data.eigenvalue_scratch_offset_bytes) ||
      workspace.occupations !=
          offset_pointer<double>(workspace.workspace_base, data.occupation_scratch_offset_bytes) ||
      workspace.lapack_work !=
          offset_pointer<double>(workspace.workspace_base, data.lapack_work_offset_bytes) ||
      workspace.lapack_integer_work !=
          offset_pointer<LapackInt>(workspace.workspace_base,
                                    data.lapack_integer_work_offset_bytes)) {
    error = "eigensolver worker scratch is not a canonical binding for this plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  AddressRange range;
  if (!make_range(workspace.workspace_base, data.worker_workspace_size_bytes, range)) {
    error = "eigensolver worker scratch address range is not representable";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_workspace(const EigensolverPlan& plan,
                                    const EigensolverWorkspace& workspace, std::string& error) {
  xtbloom_status_t status = validate_worker_workspace(plan, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  const EigensolverPlanData& data = *plan.identity();
  if (workspace.workspace_size_bytes < data.workspace_size_bytes ||
      workspace.factor_staging !=
          offset_pointer<double>(workspace.workspace_base, data.factor_staging_offset_bytes) ||
      workspace.factor_generation_staging !=
          offset_pointer<std::uint64_t>(workspace.workspace_base,
                                        data.factor_generation_staging_offset_bytes) ||
      workspace.factor_status_staging !=
          offset_pointer<xtbloom_status_t>(workspace.workspace_base,
                                           data.factor_status_staging_offset_bytes) ||
      workspace.batch_coefficients !=
          offset_pointer<double>(workspace.workspace_base,
                                 data.batch_coefficient_staging_offset_bytes) ||
      workspace.batch_densities !=
          offset_pointer<double>(workspace.workspace_base,
                                 data.batch_density_staging_offset_bytes) ||
      workspace.batch_energy_weighted_densities !=
          offset_pointer<double>(workspace.workspace_base,
                                 data.batch_energy_weighted_density_staging_offset_bytes) ||
      workspace.batch_eigenvalues !=
          offset_pointer<double>(workspace.workspace_base,
                                 data.batch_eigenvalue_staging_offset_bytes) ||
      workspace.batch_occupations !=
          offset_pointer<double>(workspace.workspace_base,
                                 data.batch_occupation_staging_offset_bytes) ||
      workspace.batch_system_statuses !=
          offset_pointer<xtbloom_status_t>(workspace.workspace_base,
                                           data.batch_system_status_staging_offset_bytes) ||
      workspace.batch_chemical_potentials !=
          offset_pointer<double>(workspace.workspace_base,
                                 data.batch_chemical_potential_staging_offset_bytes) ||
      workspace.batch_entropies !=
          offset_pointer<double>(workspace.workspace_base,
                                 data.batch_entropy_staging_offset_bytes) ||
      workspace.batch_band_energies !=
          offset_pointer<double>(workspace.workspace_base,
                                 data.batch_band_energy_staging_offset_bytes) ||
      workspace.batch_free_energies !=
          offset_pointer<double>(workspace.workspace_base,
                                 data.batch_free_energy_staging_offset_bytes)) {
    error = "eigensolver full-batch staging is not a canonical binding for this plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  AddressRange range;
  if (!make_range(workspace.workspace_base, data.workspace_size_bytes, range)) {
    error = "eigensolver full-batch workspace address range is not representable";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t validate_wavefunction(const EigensolverPlan& plan,
                                       const WavefunctionView& wavefunction, std::string& error) {
  const EigensolverPlanData& data = *plan.identity();
  if (wavefunction.workspace_base == nullptr ||
      wavefunction.workspace_size_bytes < data.wavefunction_workspace_size_bytes ||
      !is_aligned(wavefunction.workspace_base, kWavefunctionWorkspaceAlignment)) {
    error = "eigensolver wavefunction is not a canonical binding for this plan";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  AddressRange range;
  if (!make_range(wavefunction.workspace_base, data.wavefunction_workspace_size_bytes, range)) {
    error = "eigensolver wavefunction address range is not representable";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const auto pointers = eigensolver_view_fields(wavefunction);
  for (std::size_t field = 0u; field < pointers.size(); ++field) {
    if (pointers[field] != offset_pointer<double>(wavefunction.workspace_base,
                                                  data.wavefunction_fields[field].offset_bytes)) {
      error = "eigensolver wavefunction is not a canonical binding for this plan";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  return XTBLOOM_STATUS_SUCCESS;
}

bool validate_result_view(const EigensolverPlan& plan, const EigensolverThermodynamicsView& results,
                          std::array<AddressRange, 5>& ranges, std::string& error) {
  const std::size_t batch = static_cast<std::size_t>(plan.batch_size());
  std::size_t chemical_count = 0u;
  if (!checked_multiply(batch, 2u, chemical_count) || results.system_status_capacity < batch ||
      results.chemical_potential_capacity < chemical_count || results.entropy_capacity < batch ||
      results.band_energy_capacity < batch || results.free_energy_capacity < batch ||
      !is_aligned(results.system_statuses, alignof(xtbloom_status_t)) ||
      !is_aligned(results.chemical_potentials, alignof(double)) ||
      !is_aligned(results.entropies, alignof(double)) ||
      !is_aligned(results.band_energies, alignof(double)) ||
      !is_aligned(results.free_energies, alignof(double)) ||
      !make_range(results.system_statuses, batch * sizeof(xtbloom_status_t), ranges[0]) ||
      !make_range(results.chemical_potentials, chemical_count * sizeof(double), ranges[1]) ||
      !make_range(results.entropies, batch * sizeof(double), ranges[2]) ||
      !make_range(results.band_energies, batch * sizeof(double), ranges[3]) ||
      !make_range(results.free_energies, batch * sizeof(double), ranges[4]) ||
      !pairwise_disjoint(ranges)) {
    error = "eigensolver thermodynamic outputs are invalid or overlap";
    return false;
  }
  return true;
}

bool symmetric_finite_row_major(const double* matrix, std::size_t n) {
  constexpr double multiplier = 64.0 * std::numeric_limits<double>::epsilon();
  for (std::size_t row = 0u; row < n; ++row) {
    for (std::size_t column = 0u; column < n; ++column) {
      const double value = matrix[row * n + column];
      if (!std::isfinite(value)) {
        return false;
      }
      if (column < row) {
        const double transpose = matrix[column * n + row];
        const double scale = std::max({1.0, std::abs(value), std::abs(transpose)});
        if (std::abs(value - transpose) > multiplier * scale) {
          return false;
        }
      }
    }
  }
  return true;
}

bool symmetric_finite_column_major(const double* matrix, std::size_t n) {
  constexpr double multiplier = 256.0 * std::numeric_limits<double>::epsilon();
  for (std::size_t column = 0u; column < n; ++column) {
    for (std::size_t row = 0u; row < n; ++row) {
      const double value = matrix[row + column * n];
      const double transpose = matrix[column + row * n];
      const double scale = std::max({1.0, std::abs(value), std::abs(transpose)});
      if (!std::isfinite(value) || std::abs(value - transpose) > multiplier * scale) {
        return false;
      }
    }
  }
  return true;
}

void copy_symmetric_row_to_column(const double* input, std::size_t n, double* output) {
  for (std::size_t row = 0u; row < n; ++row) {
    output[row + row * n] = input[row * n + row];
    for (std::size_t column = 0u; column < row; ++column) {
      const double value = 0.5 * (input[row * n + column] + input[column * n + row]);
      output[row + column * n] = value;
      output[column + row * n] = value;
    }
  }
}

void copy_column_to_row(const double* input, std::size_t rows, std::size_t columns,
                        double* output) {
  for (std::size_t row = 0u; row < rows; ++row) {
    for (std::size_t column = 0u; column < columns; ++column) {
      output[row * columns + column] = input[row + column * rows];
    }
  }
}

double matrix_one_norm_column_major(const double* matrix, std::size_t n) {
  double maximum = 0.0;
  for (std::size_t column = 0u; column < n; ++column) {
    double sum = 0.0;
    for (std::size_t row = 0u; row < n; ++row) {
      sum += std::abs(matrix[row + column * n]);
    }
    maximum = std::max(maximum, sum);
  }
  return maximum;
}

bool finite_array(const double* values, std::size_t count) {
  for (std::size_t index = 0u; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

long double fermi_value(long double shifted_energy, long double shifted_chemical_potential,
                        long double temperature) {
  const long double argument = (shifted_energy - shifted_chemical_potential) / temperature;
  if (argument >= 0.0L) {
    const long double exponential = std::exp(-argument);
    return exponential / (1.0L + exponential);
  }
  return 1.0L / (std::exp(argument) + 1.0L);
}

long double fermi_hole_value(long double shifted_energy, long double shifted_chemical_potential,
                             long double temperature) {
  const long double argument = (shifted_energy - shifted_chemical_potential) / temperature;
  if (argument >= 0.0L) {
    return 1.0L / (std::exp(-argument) + 1.0L);
  }
  const long double exponential = std::exp(argument);
  return exponential / (1.0L + exponential);
}

long double fermi_quantity(const double* eigenvalues, std::size_t count,
                           long double energy_reference, long double shifted_chemical_potential,
                           long double temperature, bool solve_holes) {
  long double result = 0.0L;
  for (std::size_t orbital = 0u; orbital < count; ++orbital) {
    const long double shifted_energy =
        static_cast<long double>(eigenvalues[orbital]) - energy_reference;
    result += solve_holes
                  ? fermi_hole_value(shifted_energy, shifted_chemical_potential, temperature)
                  : fermi_value(shifted_energy, shifted_chemical_potential, temperature);
  }
  return result;
}

bool compute_occupations(const double* eigenvalues, std::size_t count, double electron_count,
                         double temperature, double* occupations, double& chemical_potential,
                         double& entropy) {
  chemical_potential = 0.0;
  entropy = 0.0;
  if (temperature == 0.0) {
    if (occupations != nullptr) {
      std::fill_n(occupations, count, 0.0);
      const std::size_t full =
          std::min(static_cast<std::size_t>(std::floor(electron_count)), count);
      std::fill_n(occupations, full, 1.0);
      if (full < count) {
        occupations[full] = electron_count - static_cast<double>(full);
      }
    }
    return true;
  }
  if (electron_count == 0.0) {
    if (occupations != nullptr) {
      std::fill_n(occupations, count, 0.0);
    }
    /* tblite leaves e_fermi at zero when nel is zero. */
    return true;
  }

  const long double target = static_cast<long double>(electron_count);
  const long double capacity = static_cast<long double>(count);
  const long double thermal = static_cast<long double>(temperature);
  if (target == capacity) {
    if (occupations != nullptr) {
      std::fill_n(occupations, count, 1.0);
    }
    const long double mu = static_cast<long double>(eigenvalues[count - 1u]) + 50.0L * thermal;
    chemical_potential = static_cast<double>(
        std::clamp(mu, -static_cast<long double>(std::numeric_limits<double>::max()),
                   static_cast<long double>(std::numeric_limits<double>::max())));
    return std::isfinite(chemical_potential);
  }

  const bool solve_holes = target > 0.5L * capacity;
  const long double quantity_target = solve_holes ? capacity - target : target;
  const long double fraction = quantity_target / capacity;
  const long double thermal_steps = std::max(64.0L, -std::log(fraction) + 8.0L);
  /*
   * Solve in a translated energy frame. Occupations depend only on E-mu,
   * while the floating-point spacing near a large common offset can exceed a
   * small kBT and otherwise prevent bisection from resolving the root.
   */
  const long double energy_reference =
      static_cast<long double>(solve_holes ? eigenvalues[count - 1u] : eigenvalues[0]);
  const long double shifted_minimum = static_cast<long double>(eigenvalues[0]) - energy_reference;
  const long double shifted_maximum =
      static_cast<long double>(eigenvalues[count - 1u]) - energy_reference;
  const long double energy_span = shifted_maximum - shifted_minimum;
  const long double energy_scale = std::max(1.0L, std::abs(energy_span));
  const long double representation_margin =
      64.0L * std::numeric_limits<long double>::epsilon() * energy_scale;
  const long double margin = thermal_steps * thermal + representation_margin;
  long double lower = shifted_minimum - margin;
  long double upper = shifted_maximum + margin;
  const long double lower_quantity =
      fermi_quantity(eigenvalues, count, energy_reference, lower, thermal, solve_holes);
  const long double upper_quantity =
      fermi_quantity(eigenvalues, count, energy_reference, upper, thermal, solve_holes);
  const bool bracketed =
      solve_holes ? lower_quantity >= quantity_target && upper_quantity <= quantity_target
                  : lower_quantity <= quantity_target && upper_quantity >= quantity_target;
  if (!std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper) || !bracketed) {
    return false;
  }

  /*
   * Solve more tightly than tblite's Newton stopping threshold.  The final
   * occupations are published as doubles, so a long-double root leaves only
   * the unavoidable rounding error from that conversion and keeps the
   * electron sum stable enough for repeated SCC iterations.
   */
  const long double tolerance =
      64.0L * static_cast<long double>(std::numeric_limits<double>::epsilon()) * quantity_target;
  for (int iteration = 0; iteration < 4096; ++iteration) {
    const long double middle = lower + 0.5L * (upper - lower);
    const long double quantity =
        fermi_quantity(eigenvalues, count, energy_reference, middle, thermal, solve_holes);
    if (std::abs(quantity - quantity_target) <= tolerance) {
      lower = middle;
      upper = middle;
      break;
    }
    if (middle == lower || middle == upper) {
      break;
    }
    if ((!solve_holes && quantity < quantity_target) ||
        (solve_holes && quantity > quantity_target)) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  const long double mu = lower + 0.5L * (upper - lower);
  const long double absolute_mu = energy_reference + mu;
  chemical_potential = static_cast<double>(
      std::clamp(absolute_mu, -static_cast<long double>(std::numeric_limits<double>::max()),
                 static_cast<long double>(std::numeric_limits<double>::max())));
  long double ideal_quantity = 0.0L;
  long double published_quantity = 0.0L;
  for (std::size_t orbital = 0u; orbital < count; ++orbital) {
    const long double shifted_energy =
        static_cast<long double>(eigenvalues[orbital]) - energy_reference;
    const long double occupation = fermi_value(shifted_energy, mu, thermal);
    ideal_quantity += solve_holes ? fermi_hole_value(shifted_energy, mu, thermal) : occupation;
    const double published = static_cast<double>(occupation);
    published_quantity += solve_holes ? 1.0L - static_cast<long double>(published)
                                      : static_cast<long double>(published);
    if (occupations != nullptr) {
      occupations[orbital] = published;
    }
  }

  const long double publication_tolerance =
      64.0L * static_cast<long double>(std::numeric_limits<double>::epsilon()) * quantity_target;
  const bool ideal_acceptable = std::abs(ideal_quantity - quantity_target) <= tolerance;
  const bool publication_acceptable =
      std::abs(published_quantity - quantity_target) <= publication_tolerance;

  if (ideal_acceptable && publication_acceptable) {
    /* Keep the established long-double path byte-identical when the directly
     * rounded publication already satisfies strict conservation. */
    long double entropy_value = 0.0L;
    for (std::size_t orbital = 0u; orbital < count; ++orbital) {
      const long double occupation = static_cast<long double>(static_cast<double>(fermi_value(
          static_cast<long double>(eigenvalues[orbital]) - energy_reference, mu, thermal)));
      if (occupation > 0.0L && occupation < 1.0L) {
        entropy_value -=
            occupation * std::log(occupation) + (1.0L - occupation) * std::log(1.0L - occupation);
      }
    }
    entropy = static_cast<double>(entropy_value);
    return std::isfinite(chemical_potential) && std::isfinite(entropy);
  }

  namespace policy = binary64_policy;
  const std::int64_t binary64_count = static_cast<std::int64_t>(count);
  const std::int64_t largest_degenerate_block =
      policy::largest_degenerate_block(eigenvalues, binary64_count);
  if (!ideal_acceptable && largest_degenerate_block <= 1) {
    /* A wider-precision root failure without a real degenerate frontier is not
     * a publication representability case and must remain a data failure. */
    return false;
  }

  const double binary64_capacity = static_cast<double>(count);
  const double binary64_target = solve_holes ? binary64_capacity - electron_count : electron_count;
  policy::Root root{};
  if (!(binary64_target > 0.0) || !std::isfinite(binary64_target) ||
      !policy::solve_root(eigenvalues, binary64_count, binary64_target, temperature, solve_holes,
                          root)) {
    return false;
  }
  policy::Publication publication{};
  if (!policy::select_publication(eigenvalues, binary64_count, binary64_target, temperature,
                                  solve_holes, root, publication)) {
    return false;
  }
  const double root_tolerance = policy::root_acceptance_tolerance(binary64_target, root);
  if (!std::isfinite(root.quantity) ||
      policy::absolute(root.quantity - binary64_target) > root_tolerance) {
    return false;
  }
  chemical_potential = policy::saturated_affine(root.energy_reference, root.scaled_mu, temperature);
  if (largest_degenerate_block == binary64_count && binary64_target / binary64_capacity == 0.0) {
    /* A valid subnormal total can underflow when divided across a fully
     * degenerate block. The analytic equal-level logit is the only deliberate
     * chemical-potential override; all other rare cases use the shared root. */
    const double log_fraction =
        policy::logarithm(binary64_target) - policy::logarithm(binary64_capacity);
    const double log_complement = policy::logarithm_one_plus(-policy::exponential(log_fraction));
    const double canonical_scaled_mu =
        solve_holes ? log_complement - log_fraction : log_fraction - log_complement;
    chemical_potential =
        policy::saturated_affine(root.energy_reference, canonical_scaled_mu, temperature);
  }
  entropy =
      policy::publication_entropy(eigenvalues, binary64_count, temperature, root, publication);
  if (!std::isfinite(chemical_potential) || !std::isfinite(entropy)) {
    return false;
  }
  if (occupations != nullptr) {
    for (std::int64_t orbital = 0; orbital < binary64_count; ++orbital) {
      occupations[orbital] =
          policy::published_occupation(eigenvalues, orbital, temperature, root, publication);
    }
  }
  return true;
}

NumericalResult solve_one_spin(const CpuLinearAlgebraBackend& backend,
                               const EigensolverPlanData& data, LapackInt n, const double* factor,
                               const double* hamiltonian, double* coefficients, double* eigenvalues,
                               const EigensolverWorkspace& workspace) {
  const std::size_t dimension = static_cast<std::size_t>(n);
  const std::size_t matrix_count = dimension * dimension;
  copy_symmetric_row_to_column(hamiltonian, dimension, coefficients);
  CpuLinearAlgebraAccess::dtrsm(backend)(kCblasColMajor, kCblasLeft, kCblasLower, kCblasNoTrans,
                                         kCblasNonUnit, n, n, 1.0, factor, n, coefficients, n);
  CpuLinearAlgebraAccess::dtrsm(backend)(kCblasColMajor, kCblasRight, kCblasLower, kCblasTrans,
                                         kCblasNonUnit, n, n, 1.0, factor, n, coefficients, n);
  for (std::size_t column = 0u; column < dimension; ++column) {
    for (std::size_t row = column + 1u; row < dimension; ++row) {
      const double average =
          0.5 * (coefficients[row + column * dimension] + coefficients[column + row * dimension]);
      coefficients[row + column * dimension] = average;
      coefficients[column + row * dimension] = average;
    }
  }
  const LapackInt info = CpuLinearAlgebraAccess::dsyevd(backend)(
      kCblasColMajor, kEigenvectors, kLower, n, coefficients, n, eigenvalues, workspace.lapack_work,
      data.lapack_work_count, workspace.lapack_integer_work, data.lapack_integer_work_count);
  if (info < 0) {
    return NumericalResult::kBackendFailure;
  }
  if (info > 0) {
    return NumericalResult::kDataFailure;
  }
  CpuLinearAlgebraAccess::dtrsm(backend)(kCblasColMajor, kCblasLeft, kCblasLower, kCblasTrans,
                                         kCblasNonUnit, n, n, 1.0, factor, n, coefficients, n);
  return finite_array(coefficients, matrix_count) && finite_array(eigenvalues, dimension)
             ? NumericalResult::kSuccess
             : NumericalResult::kDataFailure;
}

void form_density_column_major(const CpuLinearAlgebraBackend& backend, LapackInt n,
                               const double* coefficients, const double* weights,
                               double* weighted_coefficients, double* density) {
  const std::size_t dimension = static_cast<std::size_t>(n);
  for (std::size_t orbital = 0u; orbital < dimension; ++orbital) {
    for (std::size_t row = 0u; row < dimension; ++row) {
      const std::size_t index = row + orbital * dimension;
      weighted_coefficients[index] = coefficients[index] * weights[orbital];
    }
  }
  CpuLinearAlgebraAccess::dgemm(backend)(kCblasColMajor, kCblasNoTrans, kCblasTrans, n, n, n, 1.0,
                                         weighted_coefficients, n, coefficients, n, 0.0, density,
                                         n);
}

NumericalResult solve_system_unchecked(const EigensolverPlanData& data, std::size_t system,
                                       const EigensolverOverlapCache& overlap_cache,
                                       std::uint64_t geometry_generation,
                                       const double* system_hamiltonians, double temperature,
                                       const CpuLinearAlgebraBackend& backend,
                                       const EigensolverWorkspace& workspace,
                                       const WavefunctionView& wavefunction,
                                       const EigensolverThermodynamicsView& thermodynamics) {
  if (overlap_cache.geometry_generations[system] != geometry_generation ||
      overlap_cache.system_statuses[system] != XTBLOOM_STATUS_SUCCESS) {
    thermodynamics.system_statuses[system] = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
    return NumericalResult::kDataFailure;
  }

  const LapackInt n =
      static_cast<LapackInt>(data.orbital_offsets[system + 1u] - data.orbital_offsets[system]);
  const std::size_t orbital_count = static_cast<std::size_t>(n);
  const std::size_t matrix_count = orbital_count * orbital_count;
  const std::size_t matrix_offset = static_cast<std::size_t>(data.matrix_offsets[system]);
  const std::int32_t nspin = data.spin_channels[system];
  const double* factor = overlap_cache.cholesky_factors + matrix_offset;
  for (std::int32_t spin = 0; spin < nspin; ++spin) {
    const std::size_t spin_index = static_cast<std::size_t>(spin);
    const NumericalResult result =
        solve_one_spin(backend, data, n, factor, system_hamiltonians + spin_index * matrix_count,
                       workspace.coefficients + spin_index * matrix_count,
                       workspace.eigenvalues + spin_index * orbital_count, workspace);
    if (result != NumericalResult::kSuccess) {
      if (result == NumericalResult::kDataFailure) {
        thermodynamics.system_statuses[system] = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
      }
      return result;
    }
  }

  std::fill_n(workspace.occupations, 2u * orbital_count, 0.0);
  double chemical_potentials[2]{0.0, 0.0};
  double spin_entropies[2]{0.0, 0.0};
  for (std::int32_t spin = 0; spin < 2; ++spin) {
    const std::size_t spin_index = static_cast<std::size_t>(spin);
    const std::size_t eigenvalue_spin = nspin == 1 ? 0u : spin_index;
    const double electron_count =
        spin == 0 ? data.alpha_electron_counts[system] : data.beta_electron_counts[system];
    if (!compute_occupations(workspace.eigenvalues + eigenvalue_spin * orbital_count, orbital_count,
                             electron_count, temperature,
                             workspace.occupations + spin_index * orbital_count,
                             chemical_potentials[spin_index], spin_entropies[spin_index])) {
      thermodynamics.system_statuses[system] = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
      return NumericalResult::kDataFailure;
    }
  }

  double band_energy = 0.0;
  double* weighted_coefficients = workspace.lapack_work;
  double* weights = workspace.lapack_work + matrix_count;
  if (nspin == 1) {
    for (std::size_t orbital = 0u; orbital < orbital_count; ++orbital) {
      weights[orbital] =
          workspace.occupations[orbital] + workspace.occupations[orbital_count + orbital];
      band_energy += weights[orbital] * workspace.eigenvalues[orbital];
    }
    form_density_column_major(backend, n, workspace.coefficients, weights, weighted_coefficients,
                              workspace.densities);
    for (std::size_t orbital = 0u; orbital < orbital_count; ++orbital) {
      weights[orbital] *= workspace.eigenvalues[orbital];
    }
    form_density_column_major(backend, n, workspace.coefficients, weights, weighted_coefficients,
                              workspace.energy_weighted_densities);
  } else {
    for (std::int32_t spin = 0; spin < 2; ++spin) {
      const std::size_t spin_index = static_cast<std::size_t>(spin);
      const std::size_t orbital_offset = spin_index * orbital_count;
      const std::size_t spin_matrix_offset = spin_index * matrix_count;
      const double* spin_occupations = workspace.occupations + orbital_offset;
      const double* spin_eigenvalues = workspace.eigenvalues + orbital_offset;
      for (std::size_t orbital = 0u; orbital < orbital_count; ++orbital) {
        band_energy += spin_occupations[orbital] * spin_eigenvalues[orbital];
      }
      form_density_column_major(backend, n, workspace.coefficients + spin_matrix_offset,
                                spin_occupations, weighted_coefficients,
                                workspace.densities + spin_matrix_offset);
      for (std::size_t orbital = 0u; orbital < orbital_count; ++orbital) {
        weights[orbital] = spin_occupations[orbital] * spin_eigenvalues[orbital];
      }
      form_density_column_major(backend, n, workspace.coefficients + spin_matrix_offset, weights,
                                weighted_coefficients,
                                workspace.energy_weighted_densities + spin_matrix_offset);
    }
  }

  const std::size_t spin_matrix_count = static_cast<std::size_t>(nspin) * matrix_count;
  const double entropy = spin_entropies[0] + spin_entropies[1];
  const double free_energy = band_energy - temperature * entropy;
  for (std::int32_t spin = 0; spin < nspin; ++spin) {
    const std::size_t spin_matrix_offset = static_cast<std::size_t>(spin) * matrix_count;
    if (!symmetric_finite_column_major(workspace.densities + spin_matrix_offset, orbital_count) ||
        !symmetric_finite_column_major(workspace.energy_weighted_densities + spin_matrix_offset,
                                       orbital_count)) {
      thermodynamics.system_statuses[system] = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
      return NumericalResult::kDataFailure;
    }
  }
  if (!finite_array(workspace.densities, spin_matrix_count) ||
      !finite_array(workspace.energy_weighted_densities, spin_matrix_count) ||
      !std::isfinite(chemical_potentials[0]) || !std::isfinite(chemical_potentials[1]) ||
      !std::isfinite(entropy) || !std::isfinite(band_energy) || !std::isfinite(free_energy)) {
    thermodynamics.system_statuses[system] = XTBLOOM_STATUS_EIGENSOLVER_FAILED;
    return NumericalResult::kDataFailure;
  }

  const auto& coefficient_field = data.wavefunction_fields[0];
  const auto& eigenvalue_field = data.wavefunction_fields[1];
  const auto& occupation_field = data.wavefunction_fields[2];
  const auto& density_field = data.wavefunction_fields[3];
  const auto& weighted_density_field = data.wavefunction_fields[4];
  double* coefficient_output = wavefunction.coefficients + coefficient_field.system_offsets[system];
  double* density_output = wavefunction.density + density_field.system_offsets[system];
  double* weighted_density_output =
      wavefunction.energy_weighted_density + weighted_density_field.system_offsets[system];
  for (std::int32_t spin = 0; spin < nspin; ++spin) {
    const std::size_t spin_matrix_offset = static_cast<std::size_t>(spin) * matrix_count;
    copy_column_to_row(workspace.coefficients + spin_matrix_offset, orbital_count, orbital_count,
                       coefficient_output + spin_matrix_offset);
    copy_column_to_row(workspace.densities + spin_matrix_offset, orbital_count, orbital_count,
                       density_output + spin_matrix_offset);
    copy_column_to_row(workspace.energy_weighted_densities + spin_matrix_offset, orbital_count,
                       orbital_count, weighted_density_output + spin_matrix_offset);
  }
  std::copy_n(workspace.eigenvalues, static_cast<std::size_t>(nspin) * orbital_count,
              wavefunction.eigenvalues + eigenvalue_field.system_offsets[system]);
  std::copy_n(workspace.occupations, 2u * orbital_count,
              wavefunction.occupations + occupation_field.system_offsets[system]);
  thermodynamics.chemical_potentials[2u * system] = chemical_potentials[0];
  thermodynamics.chemical_potentials[2u * system + 1u] = chemical_potentials[1];
  thermodynamics.entropies[system] = entropy;
  thermodynamics.band_energies[system] = band_energy;
  thermodynamics.free_energies[system] = free_energy;
  thermodynamics.system_statuses[system] = XTBLOOM_STATUS_SUCCESS;
  return NumericalResult::kSuccess;
}

WavefunctionView make_batch_staging_wavefunction(const EigensolverWorkspace& workspace) {
  WavefunctionView staging;
  staging.coefficients = workspace.batch_coefficients;
  staging.eigenvalues = workspace.batch_eigenvalues;
  staging.occupations = workspace.batch_occupations;
  staging.density = workspace.batch_densities;
  staging.energy_weighted_density = workspace.batch_energy_weighted_densities;
  return staging;
}

EigensolverThermodynamicsView make_batch_staging_thermodynamics(
    const EigensolverPlanData& data, const EigensolverWorkspace& workspace) {
  const std::size_t batch = static_cast<std::size_t>(data.batch_size);
  return {workspace.batch_system_statuses, batch, workspace.batch_chemical_potentials, 2u * batch,
          workspace.batch_entropies,       batch, workspace.batch_band_energies,       batch,
          workspace.batch_free_energies,   batch};
}

void commit_batch_solve_results(const EigensolverPlanData& data,
                                const EigensolverWorkspace& workspace,
                                const WavefunctionView& wavefunction,
                                const EigensolverThermodynamicsView& thermodynamics) {
  const std::array<const double*, kEigensolverFieldCount> staged_fields{
      {workspace.batch_coefficients, workspace.batch_eigenvalues, workspace.batch_occupations,
       workspace.batch_densities, workspace.batch_energy_weighted_densities}};
  const auto output_fields = eigensolver_view_fields(wavefunction);
  const std::size_t batch = static_cast<std::size_t>(data.batch_size);
  for (std::size_t system = 0u; system < batch; ++system) {
    const xtbloom_status_t system_status = workspace.batch_system_statuses[system];
    thermodynamics.system_statuses[system] = system_status;
    if (system_status != XTBLOOM_STATUS_SUCCESS) {
      continue;
    }
    for (std::size_t field = 0u; field < kEigensolverFieldCount; ++field) {
      const std::size_t begin =
          static_cast<std::size_t>(data.wavefunction_fields[field].system_offsets[system]);
      const std::size_t end =
          static_cast<std::size_t>(data.wavefunction_fields[field].system_offsets[system + 1u]);
      std::copy_n(staged_fields[field] + begin, end - begin, output_fields[field] + begin);
    }
    thermodynamics.chemical_potentials[2u * system] =
        workspace.batch_chemical_potentials[2u * system];
    thermodynamics.chemical_potentials[2u * system + 1u] =
        workspace.batch_chemical_potentials[2u * system + 1u];
    thermodynamics.entropies[system] = workspace.batch_entropies[system];
    thermodynamics.band_energies[system] = workspace.batch_band_energies[system];
    thermodynamics.free_energies[system] = workspace.batch_free_energies[system];
  }
}

xtbloom_status_t validate_solve_bindings(
    const EigensolverPlan& plan, const EigensolverOverlapCache& overlap_cache,
    const CpuLinearAlgebraBackend& backend, const EigensolverWorkspace& workspace,
    const WavefunctionView& wavefunction, const EigensolverThermodynamicsView& thermodynamics,
    bool require_full_batch_staging, std::array<AddressRange, 5>& result_ranges,
    std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS ||
      (status = validate_backend(backend, error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = validate_cache(plan, overlap_cache, error)) != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  status = require_full_batch_staging ? validate_workspace(plan, workspace, error)
                                      : validate_worker_workspace(plan, workspace, error);
  if (status != XTBLOOM_STATUS_SUCCESS ||
      (status = validate_wavefunction(plan, wavefunction, error)) != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (!validate_result_view(plan, thermodynamics, result_ranges, error)) {
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace

bool CpuLinearAlgebraBackend::ready() const noexcept {
  return origin_ != Origin::kNone && dpotrf_work_ != nullptr && dpocon_work_ != nullptr &&
         dsyevd_work_ != nullptr && dtrsm_ != nullptr && dgemm_ != nullptr;
}

bool CpuLinearAlgebraBackend::production() const noexcept {
  return origin_ == Origin::kMklShimLp64 || origin_ == Origin::kOpenBlasIsolatedLp64 ||
         origin_ == Origin::kBundledOpenBlasLp64 || origin_ == Origin::kOpenBlasLp64;
}

bool CpuLinearAlgebraBackend::production_mkl() const noexcept {
  return origin_ == Origin::kMklShimLp64;
}

bool CpuLinearAlgebraBackend::production_mkl_isolated() const noexcept {
  return origin_ == Origin::kMklShimLp64;
}

bool CpuLinearAlgebraBackend::production_openblas_isolated() const noexcept {
  return origin_ == Origin::kOpenBlasIsolatedLp64;
}

xtbloom_status_t make_internal_test_lp64_backend(
    LapackDpotrfWork dpotrf_work, LapackDpoconWork dpocon_work, LapackDsyevdWork dsyevd_work,
    CblasDtrsm dtrsm, CblasDgemm dgemm, BlasSetNumThreadsLocal set_num_threads_local,
    CpuLinearAlgebraBackend& backend, std::string& error) {
  CpuLinearAlgebraBackend created =
      CpuLinearAlgebraAccess::make(CpuLinearAlgebraBackend::Origin::kInternalTestLp64, dpotrf_work,
                                   dpocon_work, dsyevd_work, dtrsm, dgemm, set_num_threads_local);
  if (!created.ready() || !backend_self_test(created)) {
    error = "internal LP64 test backend failed its column-major preflight";
    return XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
  }
  backend = created;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t make_mkl_rt_lp64_backend(CpuLinearAlgebraBackend& backend, std::string& error) {
  struct LinalgRuntimeState {
    CpuLinearAlgebraBackend backend;
    xtbloom_status_t status = XTBLOOM_STATUS_BACKEND_UNAVAILABLE;
    std::string message;
  };
  static const LinalgRuntimeState runtime = [] {
    LinalgRuntimeState state;
#ifdef XTBLOOM_CONFIGURED_WHEEL_OPENBLAS
    /* Python wheels carry one hash-verified scipy-openblas32 provider cohort
     * as a private sibling. Linux loads an auditwheel-repaired shim in a fresh
     * glibc link-map namespace. macOS/Windows load a renamed provider image by
     * absolute path and verify that every dispatch symbol comes from that
     * image. A configured bundle is an all-or-nothing contract, so a failure
     * never falls back to an unrelated system provider. */
    {
#if defined(_WIN32)
      void* handle = open_private_bundled_sibling(XTBLOOM_CONFIGURED_WHEEL_OPENBLAS_FILENAME);
      constexpr CpuLinearAlgebraBackend::Origin kOrigin =
          CpuLinearAlgebraBackend::Origin::kBundledOpenBlasLp64;
#elif defined(__APPLE__)
      dlerror();
      std::string provider_path;
      void* handle =
          open_private_bundled_sibling(XTBLOOM_CONFIGURED_WHEEL_OPENBLAS_FILENAME, provider_path);
      constexpr CpuLinearAlgebraBackend::Origin kOrigin =
          CpuLinearAlgebraBackend::Origin::kBundledOpenBlasLp64;
#else
      dlerror();
      void* handle = open_host_isolated_sibling("libxtbloom_openblas_lp64_shim.so");
      constexpr CpuLinearAlgebraBackend::Origin kOrigin =
          CpuLinearAlgebraBackend::Origin::kOpenBlasIsolatedLp64;
#endif
      if (handle != nullptr) {
        LapackDpotrfWork dpotrf_work = nullptr;
        LapackDpoconWork dpocon_work = nullptr;
        LapackDsyevdWork dsyevd_work = nullptr;
        CblasDtrsm dtrsm = nullptr;
        CblasDgemm dgemm = nullptr;
        BlasSetNumThreadsLocal set_threads = nullptr;
        using OpenBlasGetConfig = const char* (*)();
        OpenBlasGetConfig get_config = nullptr;
        if (load_lapacke_cblas_symbols(handle, true, dpotrf_work, dpocon_work, dsyevd_work, dtrsm,
                                       dgemm) &&
            load_symbol(handle, "scipy_openblas_get_config", get_config)) {
          const char* config = get_config();
          constexpr const char* kExpectedConfigPrefix =
              XTBLOOM_CONFIGURED_WHEEL_OPENBLAS_CONFIG_PREFIX;
          if (config != nullptr &&
              std::strncmp(config, kExpectedConfigPrefix, std::strlen(kExpectedConfigPrefix)) ==
                  0 &&
              std::strstr(config, "USE64BITINT") == nullptr) {
#if defined(_WIN32) || defined(__APPLE__)
            using OpenBlasSetNumThreadsGlobal = void (*)(int);
            using OpenBlasGetNumThreads = int (*)();
            OpenBlasSetNumThreadsGlobal set_threads_global = nullptr;
            OpenBlasGetNumThreads get_threads = nullptr;
            if (load_symbol(handle, "scipy_openblas_set_num_threads", set_threads_global) &&
                load_symbol(handle, "scipy_openblas_get_num_threads", get_threads)
#if defined(_WIN32)
                && symbols_belong_to_private_provider(handle, dpotrf_work, dpocon_work, dsyevd_work,
                                                      dtrsm, dgemm, get_config, set_threads_global,
                                                      get_threads)
#else
                && symbols_belong_to_private_provider(provider_path, dpotrf_work, dpocon_work,
                                                      dsyevd_work, dtrsm, dgemm, get_config,
                                                      set_threads_global, get_threads)
#endif
            ) {
              /* Desktop providers do not export local thread control. This
               * renamed private image is initialized exactly once by the
               * thread-safe function-static factory, so fixing its global
               * setting cannot alter an unrelated host OpenBLAS instance. */
              set_threads_global(1);
              if (get_threads() == 1) {
                CpuLinearAlgebraBackend created = CpuLinearAlgebraAccess::make(
                    kOrigin, dpotrf_work, dpocon_work, dsyevd_work, dtrsm, dgemm, nullptr);
                if (backend_self_test(created)) {
                  state.backend = created;
                  state.status = XTBLOOM_STATUS_SUCCESS;
                  return state;
                }
              }
            }
#else
            if (!load_symbol(handle, "openblas_set_num_threads_local", set_threads)) {
              static_cast<void>(
                  load_symbol(handle, "scipy_openblas_set_num_threads_local", set_threads));
            }
            if (set_threads != nullptr) {
              CpuLinearAlgebraBackend created = CpuLinearAlgebraAccess::make(
                  kOrigin, dpotrf_work, dpocon_work, dsyevd_work, dtrsm, dgemm, set_threads);
              if (backend_self_test(created)) {
                state.backend = created;
                state.status = XTBLOOM_STATUS_SUCCESS;
                return state;
              }
            }
#endif
          }
        }
#if defined(_WIN32) || defined(__APPLE__)
        close_dynamic_library(handle);
#else
        static_cast<void>(dlclose(handle));
#endif
      }
#if defined(_WIN32) || defined(__APPLE__)
      constexpr const char* kPrivateProviderName =
          XTBLOOM_CONFIGURED_WHEEL_OPENBLAS_FILENAME;
#else
      constexpr const char* kPrivateProviderName =
          "libxtbloom_openblas_lp64_shim.so";
#endif
      state.message = std::string("private wheel OpenBLAS provider is missing or failed ") +
                      "verification (" + kPrivateProviderName + ")";
      return state;
    }
#endif

#if defined(_WIN32)
    /* Native system-provider discovery remains POSIX-only. Windows wheels use
     * the private provider above; a non-wheel Windows build must configure a
     * future explicit LoadLibrary provider path instead of searching PATH. */
    state.message = "CPU linear-algebra runtime is unavailable in this Windows build";
    return state;
#else

#ifdef XTBLOOM_CONFIGURED_CPU_LINALG_SHIM
    /* Preferred isolated MKL provider: a private shim built at CMake time with
     * fixed DT_NEEDED dependencies on libmkl_intel_lp64, libmkl_sequential, and
     * libmkl_core. RTLD_LOCAL alone does not prevent a global host libmkl_rt
     * from interposing on those dependencies, so load the adjacent shim in a
     * new glibc link-map namespace. We never load libmkl_rt, call
     * MKL_Set_Interface_Layer, or read MKL interface-layer state. */
    {
      dlerror();
      void* handle = open_host_isolated_sibling("libxtbloom_mkl_lp64_shim.so");
      if (handle != nullptr) {
        LapackDpotrfWork dpotrf_work = nullptr;
        LapackDpoconWork dpocon_work = nullptr;
        LapackDsyevdWork dsyevd_work = nullptr;
        CblasDtrsm dtrsm = nullptr;
        CblasDgemm dgemm = nullptr;
        BlasSetNumThreadsLocal set_threads = nullptr;
        if (load_lapacke_cblas_symbols(handle, false, dpotrf_work, dpocon_work, dsyevd_work, dtrsm,
                                       dgemm) &&
            load_symbol(handle, "MKL_Set_Num_Threads_Local", set_threads)) {
          CpuLinearAlgebraBackend created = CpuLinearAlgebraAccess::make(
              CpuLinearAlgebraBackend::Origin::kMklShimLp64, dpotrf_work, dpocon_work, dsyevd_work,
              dtrsm, dgemm, set_threads);
          if (backend_self_test(created)) {
            /* Retain one process-lifetime loader reference so all dispatch
             * pointers and the private namespace stay valid. */
            state.backend = created;
            state.status = XTBLOOM_STATUS_SUCCESS;
            return state;
          }
        }
        static_cast<void>(dlclose(handle));
      }
      state.message =
          "host-isolated MKL provider shim is configured but did not verify "
          "(libxtbloom_mkl_lp64_shim)";
      return state;
    }
#endif

#if !defined(XTBLOOM_CONFIGURED_CPU_LINALG_MKL)
#ifdef XTBLOOM_CONFIGURED_CPU_LINALG_RUNTIME
    constexpr const char* kConfiguredRuntime = XTBLOOM_CONFIGURED_CPU_LINALG_RUNTIME;
#else
    constexpr const char* kConfiguredRuntime = nullptr;
#endif

    using OpenBlasGetConfig = const char* (*)();
    /* dlopen candidates in preference order: the absolute path CMake baked in
     * (XTBLOOM_CPU_LINALG_LIBRARY / find_package(BLAS) in CMakeLists.txt) first,
     * then known OpenBLAS sonames that the Python layer may already have
     * preloaded. MKL is never accepted through this base-namespace fallback;
     * its only production path is the isolated component shim above. */
    const char* const runtime_names[] = {
        kConfiguredRuntime, "libscipy_openblas.so", "libscipy_openblas32_.so",
        "libopenblas.so.0", "libopenblas.so",       "libopenblas.so.3",
    };

    for (const char* name : runtime_names) {
      if (name == nullptr || *name == '\0') {
        continue;
      }
      void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (handle == nullptr) {
        continue;
      }
      LapackDpotrfWork dpotrf_work = nullptr;
      LapackDpoconWork dpocon_work = nullptr;
      LapackDsyevdWork dsyevd_work = nullptr;
      CblasDtrsm dtrsm = nullptr;
      CblasDgemm dgemm = nullptr;
      BlasSetNumThreadsLocal set_threads = nullptr;
      bool scipy_prefix = false;
      if (!load_lapacke_cblas_symbols(handle, false, dpotrf_work, dpocon_work, dsyevd_work, dtrsm,
                                      dgemm)) {
        scipy_prefix = true;
        if (!load_lapacke_cblas_symbols(handle, true, dpotrf_work, dpocon_work, dsyevd_work, dtrsm,
                                        dgemm)) {
          static_cast<void>(dlclose(handle));
          continue;
        }
      }
      /* INTERFACE64 OpenBLAS builds may retain unsuffixed function names, so
       * symbol spelling alone cannot prove the 32-bit LapackInt ABI. Reject
       * providers that cannot identify themselves or report USE64BITINT. */
      OpenBlasGetConfig get_config = nullptr;
      if (!load_symbol(handle, scipy_prefix ? "scipy_openblas_get_config" : "openblas_get_config",
                       get_config)) {
        static_cast<void>(dlclose(handle));
        continue;
      }
      const char* config = get_config();
      if (config == nullptr || std::strstr(config, "USE64BITINT") != nullptr) {
        static_cast<void>(dlclose(handle));
        continue;
      }
      if (!load_symbol(handle, "openblas_set_num_threads_local", set_threads)) {
        /* scipy-openblas32 currently retains the unprefixed local-control
         * symbol, while other prefixed builds may follow the public header. */
        static_cast<void>(load_symbol(handle, "scipy_openblas_set_num_threads_local", set_threads));
      }
      if (set_threads == nullptr) {
        static_cast<void>(dlclose(handle));
        continue;
      }
      CpuLinearAlgebraBackend created =
          CpuLinearAlgebraAccess::make(CpuLinearAlgebraBackend::Origin::kOpenBlasLp64, dpotrf_work,
                                       dpocon_work, dsyevd_work, dtrsm, dgemm, set_threads);
      if (!backend_self_test(created)) {
        static_cast<void>(dlclose(handle));
        continue;
      }
      /* Retain one process-lifetime loader reference so all dispatch pointers stay valid. */
      state.backend = created;
      state.status = XTBLOOM_STATUS_SUCCESS;
      return state;
    }
    state.message =
        "failed to load an LP64 OpenBLAS runtime (libopenblas*.so, "
        "libscipy_openblas.so, or the CMake-configured path)";
    return state;
#else
    state.message =
        "host-isolated MKL provider shim is unavailable; configure the adjacent MKL "
        "LP64/sequential components or select an LP64 OpenBLAS runtime";
    return state;
#endif
#endif
  }();

  if (runtime.status != XTBLOOM_STATUS_SUCCESS) {
    error = runtime.message.empty() ? "LP64 CPU linear-algebra backend initialization failed"
                                    : runtime.message;
    return runtime.status;
  }
  backend = runtime.backend;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

namespace {

const std::vector<std::int64_t> kEmptyInt64Vector;
const std::vector<std::int32_t> kEmptyInt32Vector;
const std::vector<double> kEmptyDoubleVector;

}  // namespace

EigensolverPlan::EigensolverPlan(std::shared_ptr<const EigensolverPlanData> data) noexcept
    : data_(std::move(data)) {}

bool EigensolverPlan::sealed() const noexcept { return data_ != nullptr; }
std::int64_t EigensolverPlan::batch_size() const noexcept {
  return data_ == nullptr ? 0 : data_->batch_size;
}
std::int64_t EigensolverPlan::total_matrix_elements() const noexcept {
  return data_ == nullptr ? 0 : data_->total_matrix_elements;
}
std::int64_t EigensolverPlan::maximum_orbitals() const noexcept {
  return data_ == nullptr ? 0 : data_->maximum_orbitals;
}
double EigensolverPlan::minimum_overlap_rcond() const noexcept {
  return data_ == nullptr ? 0.0 : data_->minimum_overlap_rcond;
}
std::size_t EigensolverPlan::overlap_cache_size_bytes() const noexcept {
  return data_ == nullptr ? 0u : data_->overlap_cache_size_bytes;
}
std::size_t EigensolverPlan::worker_workspace_size_bytes() const noexcept {
  return data_ == nullptr ? 0u : data_->worker_workspace_size_bytes;
}
std::size_t EigensolverPlan::workspace_size_bytes() const noexcept {
  return data_ == nullptr ? 0u : data_->workspace_size_bytes;
}
std::size_t EigensolverPlan::resident_bytes() const noexcept {
  if (data_ == nullptr) {
    return 0u;
  }
  std::size_t total = sizeof(*data_) + data_->matrix_offsets.capacity() * sizeof(std::int64_t) +
                      data_->orbital_offsets.capacity() * sizeof(std::int64_t) +
                      data_->spin_channels.capacity() * sizeof(std::int32_t) +
                      data_->alpha_electron_counts.capacity() * sizeof(double) +
                      data_->beta_electron_counts.capacity() * sizeof(double);
  for (const EigensolverFieldData& field : data_->wavefunction_fields) {
    total += field.system_offsets.capacity() * sizeof(std::int64_t);
  }
  return total;
}
const std::vector<std::int64_t>& EigensolverPlan::matrix_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->matrix_offsets;
}
const std::vector<std::int64_t>& EigensolverPlan::orbital_offsets() const noexcept {
  return data_ == nullptr ? kEmptyInt64Vector : data_->orbital_offsets;
}
const std::vector<std::int32_t>& EigensolverPlan::spin_channels() const noexcept {
  return data_ == nullptr ? kEmptyInt32Vector : data_->spin_channels;
}
const std::vector<double>& EigensolverPlan::alpha_electron_counts() const noexcept {
  return data_ == nullptr ? kEmptyDoubleVector : data_->alpha_electron_counts;
}
const std::vector<double>& EigensolverPlan::beta_electron_counts() const noexcept {
  return data_ == nullptr ? kEmptyDoubleVector : data_->beta_electron_counts;
}
bool EigensolverPlan::overlaps_storage(const void* data, std::size_t size_bytes) const noexcept {
  AddressRange range;
  return size_bytes != 0u && (data_ == nullptr || !make_range(data, size_bytes, range) ||
                              overlaps_plan_storage(*this, range));
}
const EigensolverPlanData* EigensolverPlan::identity() const noexcept { return data_.get(); }

xtbloom_status_t make_eigensolver_plan(const WavefunctionLayout& layout, EigensolverPlan& plan,
                                       std::string& error, double minimum_overlap_rcond) {
  WavefunctionWarmStartIdentity validated_layout;
  xtbloom_status_t status =
      make_wavefunction_warm_start_identity(layout, 1u, validated_layout, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  if (!std::isfinite(minimum_overlap_rcond) || minimum_overlap_rcond <= 0.0 ||
      minimum_overlap_rcond >= 1.0) {
    error = "minimum overlap reciprocal condition must be finite and in (0, 1)";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    EigensolverPlanData created;
    created.batch_size = layout.batch_size;
    created.minimum_overlap_rcond = minimum_overlap_rcond;
    created.orbital_offsets = layout.batch_orbital_offsets;
    created.spin_channels = layout.spin_channels;
    created.alpha_electron_counts = layout.alpha_electron_counts;
    created.beta_electron_counts = layout.beta_electron_counts;
    created.wavefunction_workspace_size_bytes = layout.workspace_size_bytes;
    const auto fields = eigensolver_layout_fields(layout);
    for (std::size_t field = 0u; field < fields.size(); ++field) {
      created.wavefunction_fields[field].offset_bytes = fields[field]->offset_bytes;
      created.wavefunction_fields[field].element_count = fields[field]->element_count;
      created.wavefunction_fields[field].system_offsets = fields[field]->system_offsets;
    }

    const std::size_t batch = static_cast<std::size_t>(layout.batch_size);
    created.matrix_offsets.resize(batch + 1u, 0);
    for (std::size_t system = 0u; system < batch; ++system) {
      const std::int64_t orbitals =
          layout.batch_orbital_offsets[system + 1u] - layout.batch_orbital_offsets[system];
      if (orbitals <= 0 || orbitals > std::numeric_limits<LapackInt>::max() ||
          (layout.spin_channels[system] != 1 && layout.spin_channels[system] != 2) ||
          !std::isfinite(layout.alpha_electron_counts[system]) ||
          !std::isfinite(layout.beta_electron_counts[system]) ||
          layout.alpha_electron_counts[system] < 0.0 || layout.beta_electron_counts[system] < 0.0 ||
          layout.alpha_electron_counts[system] > static_cast<double>(orbitals) ||
          layout.beta_electron_counts[system] > static_cast<double>(orbitals) ||
          orbitals > std::numeric_limits<std::int64_t>::max() / orbitals ||
          created.total_matrix_elements >
              std::numeric_limits<std::int64_t>::max() - orbitals * orbitals) {
        error = "wavefunction dimensions or electron counts exceed LP64 eigensolver limits";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
      created.total_matrix_elements += orbitals * orbitals;
      created.matrix_offsets[system + 1u] = created.total_matrix_elements;
      created.maximum_orbitals = std::max(created.maximum_orbitals, orbitals);
    }

    const std::size_t maximum = static_cast<std::size_t>(created.maximum_orbitals);
    std::size_t maximum_matrix = 0u;
    if (!checked_multiply(maximum, maximum, maximum_matrix)) {
      error = "eigensolver maximum matrix size overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::uint64_t work_count = 1u + 6u * static_cast<std::uint64_t>(maximum) +
                                     2u * static_cast<std::uint64_t>(maximum_matrix);
    const std::uint64_t integer_work_count = 3u + 5u * static_cast<std::uint64_t>(maximum);
    if (work_count > static_cast<std::uint64_t>(std::numeric_limits<LapackInt>::max()) ||
        integer_work_count > static_cast<std::uint64_t>(std::numeric_limits<LapackInt>::max())) {
      error = "LAPACK eigensolver work dimensions exceed LP64 integer limits";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    created.lapack_work_count = static_cast<LapackInt>(work_count);
    created.lapack_integer_work_count = static_cast<LapackInt>(integer_work_count);

    std::size_t factor_bytes = 0u;
    std::size_t generation_bytes = 0u;
    std::size_t status_bytes = 0u;
    if (!checked_multiply(static_cast<std::size_t>(created.total_matrix_elements), sizeof(double),
                          factor_bytes) ||
        !checked_multiply(batch, sizeof(std::uint64_t), generation_bytes) ||
        !checked_multiply(batch, sizeof(xtbloom_status_t), status_bytes)) {
      error = "eigensolver overlap cache size overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    std::size_t cursor = 0u;
    if (!append_segment(factor_bytes, cursor, created.overlap_factor_offset_bytes) ||
        !append_segment(generation_bytes, cursor, created.overlap_generation_offset_bytes) ||
        !append_segment(status_bytes, cursor, created.overlap_status_offset_bytes) ||
        !align_up(cursor, created.overlap_cache_size_bytes)) {
      error = "eigensolver overlap cache packing overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    std::size_t two_matrices = 0u;
    std::size_t two_orbitals = 0u;
    if (!checked_multiply(maximum_matrix, 2u, two_matrices) ||
        !checked_multiply(maximum, 2u, two_orbitals)) {
      error = "eigensolver scratch dimensions overflow size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::array<std::size_t, 6> worker_double_counts{
        {two_matrices, two_matrices, two_matrices, two_orbitals, two_orbitals,
         static_cast<std::size_t>(created.lapack_work_count)}};
    std::array<std::size_t*, 6> worker_double_offsets{
        {&created.coefficient_scratch_offset_bytes, &created.density_scratch_offset_bytes,
         &created.energy_weighted_density_scratch_offset_bytes,
         &created.eigenvalue_scratch_offset_bytes, &created.occupation_scratch_offset_bytes,
         &created.lapack_work_offset_bytes}};
    cursor = 0u;
    for (std::size_t field = 0u; field < worker_double_counts.size(); ++field) {
      std::size_t bytes = 0u;
      if (!checked_multiply(worker_double_counts[field], sizeof(double), bytes) ||
          !append_segment(bytes, cursor, *worker_double_offsets[field])) {
        error = "eigensolver worker floating-point scratch packing overflows size_t";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    std::size_t integer_bytes = 0u;
    if (!checked_multiply(static_cast<std::size_t>(created.lapack_integer_work_count),
                          sizeof(LapackInt), integer_bytes) ||
        !append_segment(integer_bytes, cursor, created.lapack_integer_work_offset_bytes) ||
        !align_up(cursor, created.worker_workspace_size_bytes)) {
      error = "eigensolver worker integer scratch packing overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    cursor = created.worker_workspace_size_bytes;
    std::size_t chemical_potential_count = 0u;
    if (!checked_multiply(batch, 2u, chemical_potential_count)) {
      error = "eigensolver thermodynamic staging dimensions overflow size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    const std::array<std::size_t, 10> staging_double_counts{
        {static_cast<std::size_t>(created.total_matrix_elements),
         static_cast<std::size_t>(created.wavefunction_fields[0].element_count),
         static_cast<std::size_t>(created.wavefunction_fields[3].element_count),
         static_cast<std::size_t>(created.wavefunction_fields[4].element_count),
         static_cast<std::size_t>(created.wavefunction_fields[1].element_count),
         static_cast<std::size_t>(created.wavefunction_fields[2].element_count),
         chemical_potential_count, batch, batch, batch}};
    std::array<std::size_t*, 10> staging_double_offsets{
        {&created.factor_staging_offset_bytes, &created.batch_coefficient_staging_offset_bytes,
         &created.batch_density_staging_offset_bytes,
         &created.batch_energy_weighted_density_staging_offset_bytes,
         &created.batch_eigenvalue_staging_offset_bytes,
         &created.batch_occupation_staging_offset_bytes,
         &created.batch_chemical_potential_staging_offset_bytes,
         &created.batch_entropy_staging_offset_bytes,
         &created.batch_band_energy_staging_offset_bytes,
         &created.batch_free_energy_staging_offset_bytes}};
    for (std::size_t field = 0u; field < staging_double_counts.size(); ++field) {
      std::size_t bytes = 0u;
      if (!checked_multiply(staging_double_counts[field], sizeof(double), bytes) ||
          !append_segment(bytes, cursor, *staging_double_offsets[field])) {
        error = "eigensolver full-batch floating-point staging packing overflows size_t";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    if (!append_segment(generation_bytes, cursor, created.factor_generation_staging_offset_bytes) ||
        !append_segment(status_bytes, cursor, created.factor_status_staging_offset_bytes) ||
        !append_segment(status_bytes, cursor, created.batch_system_status_staging_offset_bytes) ||
        !align_up(cursor, created.workspace_size_bytes)) {
      error = "eigensolver full-batch integral staging packing overflows size_t";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }

    auto sealed = std::make_shared<const EigensolverPlanData>(std::move(created));
    plan = EigensolverPlan(std::move(sealed));
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate CPU eigensolver plan metadata";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

xtbloom_status_t bind_eigensolver_overlap_cache(const EigensolverPlan& plan, void* workspace,
                                                std::size_t workspace_size,
                                                EigensolverOverlapCache& cache,
                                                std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  const EigensolverPlanData& data = *plan.identity();
  AddressRange workspace_range;
  AddressRange plan_range;
  AddressRange cache_descriptor;
  AddressRange error_descriptor;
  if (!is_aligned(workspace, kEigensolverWorkspaceAlignment) ||
      workspace_size < data.overlap_cache_size_bytes ||
      !make_range(workspace, data.overlap_cache_size_bytes, workspace_range) ||
      !make_range(&plan, sizeof(plan), plan_range) ||
      !make_range(&cache, sizeof(cache), cache_descriptor) ||
      !make_range(&error, sizeof(error), error_descriptor) ||
      overlaps_plan_storage(plan, workspace_range) || ranges_overlap(workspace_range, plan_range) ||
      ranges_overlap(workspace_range, cache_descriptor) ||
      ranges_overlap(workspace_range, error_descriptor)) {
    error = "eigensolver overlap cache workspace is invalid or overlaps control storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  EigensolverOverlapCache created;
  created.workspace_base = workspace;
  created.workspace_size_bytes = workspace_size;
  created.cholesky_factors = offset_pointer<double>(workspace, data.overlap_factor_offset_bytes);
  created.geometry_generations =
      offset_pointer<std::uint64_t>(workspace, data.overlap_generation_offset_bytes);
  created.system_statuses =
      offset_pointer<xtbloom_status_t>(workspace, data.overlap_status_offset_bytes);
  created.plan_identity = &data;
  std::fill_n(created.cholesky_factors, static_cast<std::size_t>(data.total_matrix_elements), 0.0);
  std::fill_n(created.geometry_generations, static_cast<std::size_t>(data.batch_size), 0u);
  std::fill_n(created.system_statuses, static_cast<std::size_t>(data.batch_size),
              XTBLOOM_STATUS_EIGENSOLVER_FAILED);
  cache = created;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t bind_eigensolver_workspace(const EigensolverPlan& plan, void* workspace,
                                            std::size_t workspace_size, EigensolverWorkspace& view,
                                            std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  const EigensolverPlanData& data = *plan.identity();
  AddressRange workspace_range;
  AddressRange plan_range;
  AddressRange view_descriptor;
  AddressRange error_descriptor;
  if (!is_aligned(workspace, kEigensolverWorkspaceAlignment) ||
      workspace_size < data.workspace_size_bytes ||
      !make_range(workspace, data.workspace_size_bytes, workspace_range) ||
      !make_range(&plan, sizeof(plan), plan_range) ||
      !make_range(&view, sizeof(view), view_descriptor) ||
      !make_range(&error, sizeof(error), error_descriptor) ||
      overlaps_plan_storage(plan, workspace_range) || ranges_overlap(workspace_range, plan_range) ||
      ranges_overlap(workspace_range, view_descriptor) ||
      ranges_overlap(workspace_range, error_descriptor)) {
    error = "eigensolver scratch workspace is invalid or overlaps control storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  EigensolverWorkspace created;
  created.workspace_base = workspace;
  created.workspace_size_bytes = workspace_size;
  created.coefficients = offset_pointer<double>(workspace, data.coefficient_scratch_offset_bytes);
  created.densities = offset_pointer<double>(workspace, data.density_scratch_offset_bytes);
  created.energy_weighted_densities =
      offset_pointer<double>(workspace, data.energy_weighted_density_scratch_offset_bytes);
  created.eigenvalues = offset_pointer<double>(workspace, data.eigenvalue_scratch_offset_bytes);
  created.occupations = offset_pointer<double>(workspace, data.occupation_scratch_offset_bytes);
  created.lapack_work = offset_pointer<double>(workspace, data.lapack_work_offset_bytes);
  created.lapack_integer_work =
      offset_pointer<LapackInt>(workspace, data.lapack_integer_work_offset_bytes);
  created.factor_staging = offset_pointer<double>(workspace, data.factor_staging_offset_bytes);
  created.factor_generation_staging =
      offset_pointer<std::uint64_t>(workspace, data.factor_generation_staging_offset_bytes);
  created.factor_status_staging =
      offset_pointer<xtbloom_status_t>(workspace, data.factor_status_staging_offset_bytes);
  created.batch_coefficients =
      offset_pointer<double>(workspace, data.batch_coefficient_staging_offset_bytes);
  created.batch_densities =
      offset_pointer<double>(workspace, data.batch_density_staging_offset_bytes);
  created.batch_energy_weighted_densities =
      offset_pointer<double>(workspace, data.batch_energy_weighted_density_staging_offset_bytes);
  created.batch_eigenvalues =
      offset_pointer<double>(workspace, data.batch_eigenvalue_staging_offset_bytes);
  created.batch_occupations =
      offset_pointer<double>(workspace, data.batch_occupation_staging_offset_bytes);
  created.batch_system_statuses =
      offset_pointer<xtbloom_status_t>(workspace, data.batch_system_status_staging_offset_bytes);
  created.batch_chemical_potentials =
      offset_pointer<double>(workspace, data.batch_chemical_potential_staging_offset_bytes);
  created.batch_entropies =
      offset_pointer<double>(workspace, data.batch_entropy_staging_offset_bytes);
  created.batch_band_energies =
      offset_pointer<double>(workspace, data.batch_band_energy_staging_offset_bytes);
  created.batch_free_energies =
      offset_pointer<double>(workspace, data.batch_free_energy_staging_offset_bytes);
  created.plan_identity = &data;
  view = created;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t bind_eigensolver_worker_workspace(const EigensolverPlan& plan, void* workspace,
                                                   std::size_t workspace_size,
                                                   EigensolverWorkspace& view, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  const EigensolverPlanData& data = *plan.identity();
  AddressRange workspace_range;
  AddressRange plan_range;
  AddressRange view_descriptor;
  AddressRange error_descriptor;
  if (!is_aligned(workspace, kEigensolverWorkspaceAlignment) ||
      workspace_size < data.worker_workspace_size_bytes ||
      !make_range(workspace, data.worker_workspace_size_bytes, workspace_range) ||
      !make_range(&plan, sizeof(plan), plan_range) ||
      !make_range(&view, sizeof(view), view_descriptor) ||
      !make_range(&error, sizeof(error), error_descriptor) ||
      overlaps_plan_storage(plan, workspace_range) || ranges_overlap(workspace_range, plan_range) ||
      ranges_overlap(workspace_range, view_descriptor) ||
      ranges_overlap(workspace_range, error_descriptor)) {
    error = "eigensolver worker scratch is invalid or overlaps control storage";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  EigensolverWorkspace created;
  created.workspace_base = workspace;
  created.workspace_size_bytes = workspace_size;
  created.coefficients = offset_pointer<double>(workspace, data.coefficient_scratch_offset_bytes);
  created.densities = offset_pointer<double>(workspace, data.density_scratch_offset_bytes);
  created.energy_weighted_densities =
      offset_pointer<double>(workspace, data.energy_weighted_density_scratch_offset_bytes);
  created.eigenvalues = offset_pointer<double>(workspace, data.eigenvalue_scratch_offset_bytes);
  created.occupations = offset_pointer<double>(workspace, data.occupation_scratch_offset_bytes);
  created.lapack_work = offset_pointer<double>(workspace, data.lapack_work_offset_bytes);
  created.lapack_integer_work =
      offset_pointer<LapackInt>(workspace, data.lapack_integer_work_offset_bytes);
  created.plan_identity = &data;
  view = created;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t factor_overlap_cpu(const EigensolverPlan& plan, const double* overlap,
                                    std::uint64_t geometry_generation,
                                    const CpuLinearAlgebraBackend& backend,
                                    const EigensolverWorkspace& workspace,
                                    const EigensolverOverlapCache& cache, std::string& error) {
  xtbloom_status_t status = validate_plan(plan, error);
  if (status != XTBLOOM_STATUS_SUCCESS ||
      (status = validate_backend(backend, error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = validate_workspace(plan, workspace, error)) != XTBLOOM_STATUS_SUCCESS ||
      (status = validate_cache(plan, cache, error)) != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  const EigensolverPlanData& data = *plan.identity();
  if (geometry_generation == 0u || !is_aligned(overlap, alignof(double))) {
    error = "overlap factorization requires a nonzero generation and aligned overlap input";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t overlap_bytes =
      static_cast<std::size_t>(data.total_matrix_elements) * sizeof(double);
  std::array<AddressRange, 3> active{};
  std::array<AddressRange, 5> controls{};
  if (!make_range(overlap, overlap_bytes, active[0]) ||
      !make_range(cache.workspace_base, data.overlap_cache_size_bytes, active[1]) ||
      !make_range(workspace.workspace_base, data.workspace_size_bytes, active[2]) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&backend, sizeof(backend), controls[1]) ||
      !make_range(&workspace, sizeof(workspace), controls[2]) ||
      !make_range(&cache, sizeof(cache), controls[3]) ||
      !make_range(&error, sizeof(error), controls[4]) || !pairwise_disjoint(active) ||
      !disjoint_from_control(plan, active, controls)) {
    error = "overlap input, cache, scratch, plan, and descriptors must not overlap";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t batch = static_cast<std::size_t>(data.batch_size);
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::size_t n =
        static_cast<std::size_t>(data.orbital_offsets[system + 1u] - data.orbital_offsets[system]);
    if (!symmetric_finite_row_major(overlap + static_cast<std::size_t>(data.matrix_offsets[system]),
                                    n)) {
      error = "overlap matrices must be finite and symmetric";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }

  ScopedSequentialBlas sequential_blas(backend);
  for (std::size_t system = 0u; system < batch; ++system) {
    const LapackInt n =
        static_cast<LapackInt>(data.orbital_offsets[system + 1u] - data.orbital_offsets[system]);
    const std::size_t dimension = static_cast<std::size_t>(n);
    const std::size_t matrix_offset = static_cast<std::size_t>(data.matrix_offsets[system]);
    double* candidate = workspace.factor_staging + matrix_offset;
    copy_symmetric_row_to_column(overlap + matrix_offset, dimension, candidate);
    const double one_norm = matrix_one_norm_column_major(candidate, dimension);
    LapackInt info =
        CpuLinearAlgebraAccess::dpotrf(backend)(kCblasColMajor, kLower, n, candidate, n);
    double reciprocal_condition = 0.0;
    if (info == 0) {
      info = CpuLinearAlgebraAccess::dpocon(backend)(
          kCblasColMajor, kLower, n, candidate, n, one_norm, &reciprocal_condition,
          workspace.lapack_work, workspace.lapack_integer_work);
    }
    if (info < 0) {
      error = "LP64 LAPACK rejected an internal overlap-factorization argument";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
    const bool usable = info == 0 && std::isfinite(reciprocal_condition) &&
                        reciprocal_condition >= data.minimum_overlap_rcond;
    if (usable) {
      for (std::size_t column = 0u; column < dimension; ++column) {
        for (std::size_t row = 0u; row < column; ++row) {
          candidate[row + column * dimension] = 0.0;
        }
      }
    }
    workspace.factor_generation_staging[system] = geometry_generation;
    workspace.factor_status_staging[system] =
        usable ? XTBLOOM_STATUS_SUCCESS : XTBLOOM_STATUS_EIGENSOLVER_FAILED;
  }

  /* No backend failure occurred: publish the complete staged batch. */
  for (std::size_t system = 0u; system < batch; ++system) {
    cache.geometry_generations[system] = workspace.factor_generation_staging[system];
    cache.system_statuses[system] = workspace.factor_status_staging[system];
    if (workspace.factor_status_staging[system] == XTBLOOM_STATUS_SUCCESS) {
      const std::size_t begin = static_cast<std::size_t>(data.matrix_offsets[system]);
      const std::size_t end = static_cast<std::size_t>(data.matrix_offsets[system + 1u]);
      std::copy_n(workspace.factor_staging + begin, end - begin, cache.cholesky_factors + begin);
    }
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t fill_occupations_cpu(std::int64_t orbital_count, const double* eigenvalues,
                                      double electron_count, double temperature,
                                      double* occupations, double& chemical_potential,
                                      double& entropy, std::string& error) {
  if (orbital_count <= 0 || orbital_count > std::numeric_limits<LapackInt>::max() ||
      !is_aligned(eigenvalues, alignof(double)) || !is_aligned(occupations, alignof(double)) ||
      !std::isfinite(electron_count) || electron_count < 0.0 ||
      electron_count > static_cast<double>(orbital_count) || !std::isfinite(temperature) ||
      temperature < 0.0) {
    error = "occupation inputs are invalid, unrepresentable, or misaligned";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t count = static_cast<std::size_t>(orbital_count);
  std::array<AddressRange, 5> ranges{};
  if (!make_range(eigenvalues, count * sizeof(double), ranges[0]) ||
      !make_range(occupations, count * sizeof(double), ranges[1]) ||
      !make_range(&chemical_potential, sizeof(chemical_potential), ranges[2]) ||
      !make_range(&entropy, sizeof(entropy), ranges[3]) ||
      !make_range(&error, sizeof(error), ranges[4]) || !pairwise_disjoint(ranges)) {
    error = "occupation inputs and scalar outputs must not overlap";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t orbital = 0u; orbital < count; ++orbital) {
    if (!std::isfinite(eigenvalues[orbital]) ||
        (orbital != 0u && eigenvalues[orbital] < eigenvalues[orbital - 1u])) {
      error = "orbital energies must be finite and nondecreasing";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  double candidate_mu = 0.0;
  double candidate_entropy = 0.0;
  if (!compute_occupations(eigenvalues, count, electron_count, temperature, nullptr, candidate_mu,
                           candidate_entropy)) {
    error = "Fermi filling failed to produce finite electron-conserving results";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!compute_occupations(eigenvalues, count, electron_count, temperature, occupations,
                           candidate_mu, candidate_entropy)) {
    error = "Fermi filling failed during atomic publication";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  chemical_potential = candidate_mu;
  entropy = candidate_entropy;
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t solve_eigensystems_cpu(
    const EigensolverPlan& plan, const EigensolverOverlapCache& overlap_cache,
    std::uint64_t geometry_generation, const double* hamiltonians, double temperature,
    const CpuLinearAlgebraBackend& backend, const EigensolverWorkspace& workspace,
    const WavefunctionView& wavefunction, const EigensolverThermodynamicsView& thermodynamics,
    std::string& error) {
  std::array<AddressRange, 5> result_ranges{};
  xtbloom_status_t status =
      validate_solve_bindings(plan, overlap_cache, backend, workspace, wavefunction, thermodynamics,
                              true, result_ranges, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  const EigensolverPlanData& data = *plan.identity();
  if (geometry_generation == 0u || !is_aligned(hamiltonians, alignof(double)) ||
      !std::isfinite(temperature) || temperature < 0.0) {
    error = "eigensolver requires aligned Hamiltonians, a nonzero generation, and valid kBT";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t hamiltonian_count =
      static_cast<std::size_t>(data.wavefunction_fields[0].element_count);
  std::array<AddressRange, 4> principal{};
  std::array<AddressRange, 7> controls{};
  if (!make_range(hamiltonians, hamiltonian_count * sizeof(double), principal[0]) ||
      !make_range(overlap_cache.workspace_base, data.overlap_cache_size_bytes, principal[1]) ||
      !make_range(workspace.workspace_base, data.workspace_size_bytes, principal[2]) ||
      !make_range(wavefunction.workspace_base, data.wavefunction_workspace_size_bytes,
                  principal[3]) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&overlap_cache, sizeof(overlap_cache), controls[1]) ||
      !make_range(&backend, sizeof(backend), controls[2]) ||
      !make_range(&workspace, sizeof(workspace), controls[3]) ||
      !make_range(&wavefunction, sizeof(wavefunction), controls[4]) ||
      !make_range(&thermodynamics, sizeof(thermodynamics), controls[5]) ||
      !make_range(&error, sizeof(error), controls[6]) || !pairwise_disjoint(principal) ||
      !disjoint_from_control(plan, principal, controls)) {
    error = "eigensolver arrays, plan storage, and descriptors must not overlap";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (const AddressRange& result : result_ranges) {
    if (overlaps_plan_storage(plan, result)) {
      error = "eigensolver scalar outputs must not overlap immutable plan storage";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (const AddressRange& range : principal) {
      if (ranges_overlap(result, range)) {
        error = "eigensolver scalar outputs must not overlap numerical arrays";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    for (const AddressRange& control : controls) {
      if (ranges_overlap(result, control)) {
        error = "eigensolver scalar outputs must not overlap descriptors";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }

  const std::size_t batch = static_cast<std::size_t>(data.batch_size);
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::size_t n =
        static_cast<std::size_t>(data.orbital_offsets[system + 1u] - data.orbital_offsets[system]);
    const std::size_t matrix_count = n * n;
    const std::size_t hamiltonian_offset =
        static_cast<std::size_t>(data.wavefunction_fields[0].system_offsets[system]);
    for (std::int32_t spin = 0; spin < data.spin_channels[system]; ++spin) {
      if (!symmetric_finite_row_major(
              hamiltonians + hamiltonian_offset + static_cast<std::size_t>(spin) * matrix_count,
              n)) {
        error = "Hamiltonian matrices must be finite and symmetric";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }

  const WavefunctionView staging_wavefunction = make_batch_staging_wavefunction(workspace);
  const EigensolverThermodynamicsView staging_thermodynamics =
      make_batch_staging_thermodynamics(data, workspace);
  ScopedSequentialBlas sequential_blas(backend);
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::size_t hamiltonian_offset =
        static_cast<std::size_t>(data.wavefunction_fields[0].system_offsets[system]);
    const NumericalResult result = solve_system_unchecked(
        data, system, overlap_cache, geometry_generation, hamiltonians + hamiltonian_offset,
        temperature, backend, workspace, staging_wavefunction, staging_thermodynamics);
    if (result == NumericalResult::kBackendFailure) {
      error = "LP64 LAPACK rejected an internal eigensolver argument";
      return XTBLOOM_STATUS_INTERNAL_ERROR;
    }
  }
  commit_batch_solve_results(data, workspace, wavefunction, thermodynamics);
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

xtbloom_status_t solve_eigensystem_cpu(
    const EigensolverPlan& plan, std::int64_t system, const EigensolverOverlapCache& overlap_cache,
    std::uint64_t geometry_generation, const double* system_hamiltonians, double temperature,
    const CpuLinearAlgebraBackend& backend, const EigensolverWorkspace& workspace,
    const WavefunctionView& wavefunction, const EigensolverThermodynamicsView& thermodynamics,
    std::string& error) {
  std::array<AddressRange, 5> result_ranges{};
  xtbloom_status_t status =
      validate_solve_bindings(plan, overlap_cache, backend, workspace, wavefunction, thermodynamics,
                              false, result_ranges, error);
  if (status != XTBLOOM_STATUS_SUCCESS) {
    return status;
  }
  const EigensolverPlanData& data = *plan.identity();
  if (system < 0 || system >= data.batch_size || geometry_generation == 0u ||
      !is_aligned(system_hamiltonians, alignof(double)) || !std::isfinite(temperature) ||
      temperature < 0.0) {
    error = "one-system eigensolver inputs are invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t system_index = static_cast<std::size_t>(system);
  const std::size_t n = static_cast<std::size_t>(data.orbital_offsets[system_index + 1u] -
                                                 data.orbital_offsets[system_index]);
  const std::size_t matrix_count = n * n;
  const std::size_t hamiltonian_count =
      static_cast<std::size_t>(data.spin_channels[system_index]) * matrix_count;
  std::array<AddressRange, 4> principal{};
  std::array<AddressRange, 7> controls{};
  if (!make_range(system_hamiltonians, hamiltonian_count * sizeof(double), principal[0]) ||
      !make_range(overlap_cache.workspace_base, data.overlap_cache_size_bytes, principal[1]) ||
      !make_range(workspace.workspace_base, data.worker_workspace_size_bytes, principal[2]) ||
      !make_range(wavefunction.workspace_base, data.wavefunction_workspace_size_bytes,
                  principal[3]) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&overlap_cache, sizeof(overlap_cache), controls[1]) ||
      !make_range(&backend, sizeof(backend), controls[2]) ||
      !make_range(&workspace, sizeof(workspace), controls[3]) ||
      !make_range(&wavefunction, sizeof(wavefunction), controls[4]) ||
      !make_range(&thermodynamics, sizeof(thermodynamics), controls[5]) ||
      !make_range(&error, sizeof(error), controls[6]) || !pairwise_disjoint(principal) ||
      !disjoint_from_control(plan, principal, controls)) {
    error = "one-system eigensolver arrays and control storage must not overlap";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  for (const AddressRange& result : result_ranges) {
    if (overlaps_plan_storage(plan, result)) {
      error = "one-system scalar outputs must not overlap immutable plan storage";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    for (const AddressRange& range : principal) {
      if (ranges_overlap(result, range)) {
        error = "one-system scalar outputs must not overlap numerical arrays";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
    for (const AddressRange& control : controls) {
      if (ranges_overlap(result, control)) {
        error = "one-system scalar outputs must not overlap descriptors";
        return XTBLOOM_STATUS_INVALID_ARGUMENT;
      }
    }
  }
  for (std::int32_t spin = 0; spin < data.spin_channels[system_index]; ++spin) {
    if (!symmetric_finite_row_major(
            system_hamiltonians + static_cast<std::size_t>(spin) * matrix_count, n)) {
      error = "one-system Hamiltonians must be finite and symmetric";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
  }
  ScopedSequentialBlas sequential_blas(backend);
  const NumericalResult result = solve_system_unchecked(
      data, system_index, overlap_cache, geometry_generation, system_hamiltonians, temperature,
      backend, workspace, wavefunction, thermodynamics);
  if (result == NumericalResult::kBackendFailure) {
    error = "LP64 LAPACK rejected an internal one-system eigensolver argument";
    return XTBLOOM_STATUS_INTERNAL_ERROR;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail::gfn2
